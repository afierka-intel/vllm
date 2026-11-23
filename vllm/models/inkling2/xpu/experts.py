# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts on Intel XPU: the `FlatExperts` backends of
`common.moe.flat_experts_class`.

MXFP8 runs the SYCL grouped-GEMM fused MoE of vllm-xpu-kernels
(`XpuFusedMoe`) on bf16 activations (W8A16, as Humming on Hopper). It takes
`[gate | up]` halves where the checkpoint interleaves the `w13` rows, and
applies the routing weight to the down-projection rows in fp32 while
combining them (`combines`), where Hopper scales the SiTU output before the
down projection and rounds it to bf16.

NVFP4 has no XPU kernel elsewhere: it runs Hopper's flow with the Triton
W4A16 grouped GEMM `common.ops.nvfp4_w4a16_moe` in place of Humming, on the
checkpoint layout (interleaved `w13` rows, consumed by `swiglu`).
"""

from collections.abc import Callable
from functools import partial

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.models.inkling2.common.moe import FlatExperts, QuantFormat
from vllm.models.inkling2.common.ops.moe import swiglu
from vllm.models.inkling2.common.ops.nvfp4_w4a16_moe import (
    nvfp4_block_m,
    nvfp4_w4a16_moe,
)

ExpertWeights = dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]


def deinterleave_rows(t: torch.Tensor) -> None:
    """In place: each expert's rows `(g0, u0, g1, u1, ...)` of `t`
    `[E, 2I, ...]` become `[g0 .. g(I-1) | u0 .. u(I-1)]`. One expert is
    copied at a time, so the extra memory is one expert's rows."""
    rows = t.view(torch.uint8) if t.element_size() == 1 else t
    for expert in rows:
        expert.copy_(torch.cat((expert[0::2], expert[1::2])))


def _situ_activation(gate_cap: float, up_cap: float) -> dict:
    """`XpuFusedMoe` activation arguments of the Inkling2 SiTU."""
    if gate_cap > 0:
        return {
            "activation": "situ",
            "activation_situ_beta": gate_cap,
            "activation_situ_linear_beta": up_cap if up_cap > 0 else None,
        }
    if up_cap == 0:
        return {"activation": "silu"}
    raise NotImplementedError(
        "XPU fused MoE has no SiTU with only the up projection softcapped"
    )


class XpuMxfp8Experts:
    """MXFP8 experts on the SYCL fused MoE (`XpuFusedMoe`).

    Args:
        weights: Per projection (`w13`, `w2`), the checkpoint weights
            `[E, N, K]` e4m3 and their e8m0 block scales `[E, N, K / 32]`
            (uint8), one per 32 along K. `XpuFusedMoe` rewrites these
            tensors in place into its kernel layout.
        shapes: `(N, K)` of each projection.
        nvfp4: Must be False.
        topk: Slots per token.
        gate_cap, up_cap: SiTU softcaps.

    """

    # The fused MoE sums the routed rows itself (`FlatExperts`).
    combines = True

    def __init__(
        self,
        weights: ExpertWeights,
        shapes: tuple[tuple[int, int], tuple[int, int]],
        nvfp4: bool,
        *,
        topk: int,
        gate_cap: float,
        up_cap: float,
    ) -> None:
        from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

        assert not nvfp4
        (w13, w13_scale, _), (w2, w2_scale, _) = weights["w13"], weights["w2"]
        for (n, k), w, s in zip(shapes, (w13, w2), (w13_scale, w2_scale)):
            assert w.shape[1:] == (n, k) and s.shape[1:] == (n, k // 32)
            assert w.dtype == torch.float8_e4m3fn and s.dtype == torch.uint8
        deinterleave_rows(w13)
        deinterleave_rows(w13_scale)
        self.moe = XpuFusedMoe(
            w13=w13,
            w13_scales=w13_scale,
            w13_bias=None,
            w2=w2,
            w2_scales=w2_scale,
            w2_bias=None,
            n_experts_per_token=topk,
            num_experts=w13.shape[0],
            **_situ_activation(gate_cap, up_cap),
        )

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
    ) -> torch.Tensor:
        """bf16 `[T, H]`: each token's local rows (`ids`, -1: not local)
        times their fp32 routing `weights`, summed and rounded once."""
        out = torch.empty_like(x)
        if x.shape[0] > 0:
            self.moe.apply(out, x, weights.view(ids.shape), ids)
        return out


class XpuNvfp4Experts:
    """NVFP4 experts on the Triton W4A16 grouped GEMM, with bf16
    activations: `w13` GEMM, `swiglu` (routing weight before the down
    projection, as on Hopper) and `w2` GEMM over the flat rows.

    Args:
        weights: Per projection (`w13`, `w2`), the checkpoint weights
            `[E, N, K / 2]` (uint8, two e2m1 each), e4m3 block scales
            `[E, N, K / 16]` and fp32 global scales `[E]`
            (`amax / outer_scale_denominator`), used as they are.
        shapes: `(N, K)` of each projection.
        nvfp4: Must be True.
        topk, gate_cap, up_cap: Unused (passed per call).

    """

    def __init__(
        self,
        weights: ExpertWeights,
        shapes: tuple[tuple[int, int], tuple[int, int]],
        nvfp4: bool,
        *,
        topk: int,
        gate_cap: float,
        up_cap: float,
    ) -> None:
        assert nvfp4
        self.weights = {}
        for name, (n, k) in zip(("w13", "w2"), shapes):
            weight, scale, global_scale = weights[name]
            assert global_scale is not None
            assert weight.shape[1:] == (n, k // 2) and scale.shape[1:] == (n, k // 16)
            self.weights[name] = (weight, scale, global_scale)
        self.num_experts = weights["w13"][0].shape[0]

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
    ) -> torch.Tensor:
        """bf16 `[T * topk, H]` down-projection rows of the local slots of
        `ids` (-1: not local); the other rows are left unset."""
        tokens, topk = ids.shape
        rows = tokens * topk
        block_m = nvfp4_block_m(rows, num_global_experts)
        alignment = moe_align_block_size(
            ids, block_m, self.num_experts, ignore_invalid_experts=True
        )[:3]
        gate_up = nvfp4_w4a16_moe(
            x, *self.weights["w13"], alignment, rows, topk, block_m
        )
        activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
        del gate_up
        return nvfp4_w4a16_moe(
            activated, *self.weights["w2"], alignment, rows, 1, block_m
        )


def xpu_flat_experts(
    quant_format: QuantFormat, topk: int, gate_cap: float, up_cap: float
) -> Callable[..., FlatExperts]:
    """The XPU `FlatExperts` constructor of `quant_format` experts."""
    if quant_format == "mxfp8":
        cls: type = XpuMxfp8Experts
    else:
        cls = XpuNvfp4Experts
    return partial(cls, topk=topk, gate_cap=gate_cap, up_cap=up_cap)
