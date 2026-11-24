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

NVFP4 runs Hopper's flow with the vendor-neutral NVFP4 emulation MoE Triton
kernel (`fused_moe/experts/nvfp4_emulation_moe.py`) in W4A16 form (bf16
activations) in place of Humming, on the checkpoint layout (interleaved
`w13` rows, consumed by `swiglu`).

Both backends keep the layer's Parameters (`keeps_parameters`): the kernel
layout lives in them, visible as module state.
"""

from collections.abc import Callable
from functools import partial

import torch

from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
    invoke_fused_moe_nvfp4_emulation_kernel,
    nvfp4_emulation_use_split_dots,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.models.inkling2.common.moe import FlatExperts, QuantFormat
from vllm.models.inkling2.common.ops.moe import swiglu
from vllm.triton_utils import tl

ExpertWeights = dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]


def deinterleave_rows(t: torch.Tensor) -> None:
    """In place: each expert's rows `(g0, u0, g1, u1, ...)` of `t`
    `[E, 2I, ...]` become `[g0 .. g(I-1) | u0 .. u(I-1)]`. One expert is
    copied at a time, so this step needs one expert's rows of extra
    memory."""
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

    Load-time memory: the de-interleave copies one expert at a time, but
    `XpuFusedMoe` then builds the relayout (`[E, K, N]` on the xe_2 GEMM)
    of the whole `w13` and `w2` of the local experts before it releases
    the loaded storage. The extra peak per layer is therefore the local
    `w13` plus `w2` (with their scales): about 0.95 GiB for Inkling 2 Mini
    (TP1) and 2.85 GiB for Small (TP4), as for upstream `XPUExperts`.

    Args:
        weights: Per projection (`w13`, `w2`), the checkpoint weights
            `[E, N, K]` e4m3 and their e8m0 block scales `[E, N, K / 32]`
            (uint8), one per 32 along K: the layer's Parameters.
            `deinterleave_rows` and then `XpuFusedMoe` rewrite them in place
            (`.data`) into the kernel layout, so they stay module state and
            the library's rebuild mark (`xpu_fused_moe`) is set on the `w13`
            Parameter itself.
        shapes: `(N, K)` of each projection.
        nvfp4: Must be False.
        topk: Slots per token.
        gate_cap, up_cap: SiTU softcaps.

    """

    # The fused MoE sums the routed rows itself (`FlatExperts`).
    combines = True
    # Built from the Parameters, which hold the kernel layout afterwards.
    keeps_parameters = True

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
        deinterleave_rows(w13.data)
        deinterleave_rows(w13_scale.data)
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
    """NVFP4 experts on the NVFP4 emulation MoE Triton kernel
    (`invoke_fused_moe_nvfp4_emulation_kernel`) with bf16 activations
    (W4A16): `w13` GEMM, `swiglu` (routing weight before the down
    projection, as on Hopper) and `w2` GEMM over the flat rows. Batches of
    up to `SPLIT_DOTS_MAX_TOKENS` tokens take the kernel's split-dots path.

    Args:
        weights: Per projection (`w13`, `w2`), the checkpoint weights
            `[E, N, K / 2]` (uint8, two e2m1 each), e4m3 block scales
            `[E, N, K / 16]` (the layer's Parameters, used as they are) and
            fp32 global scales `[E]` (`amax / outer_scale_denominator`).
        shapes: `(N, K)` of each projection.
        nvfp4: Must be True.
        topk, gate_cap, up_cap: Unused (passed per call).

    """

    # Reads the Parameters in the checkpoint layout (no copy).
    keeps_parameters = True

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
            assert global_scale is not None and global_scale.dtype == torch.float32
            assert weight.shape[1:] == (n, k // 2) and scale.shape[1:] == (n, k // 16)
            assert weight.dtype == torch.uint8 and scale.dtype == torch.float8_e4m3fn
            self.weights[name] = (weight, scale, global_scale)
        self.num_experts = weights["w13"][0].shape[0]

    def _gemm(self, name, a, top_k, alignment, config, split_dots):
        """bf16 `[rows, N]`: row `r` of the aligned rows is
        `a[r // top_k] @ W[e].T` for its expert `e`; the rows the alignment
        does not list are left unset."""
        weight, scale, global_scale = self.weights[name]
        sorted_ids, expert_ids, num_post_padded = alignment
        out = torch.empty(
            (a.shape[0], top_k, weight.shape[1]), dtype=torch.bfloat16, device=a.device
        )
        invoke_fused_moe_nvfp4_emulation_kernel(
            a,
            weight,
            out,
            scale.view(torch.uint8),
            None,
            global_scale,
            None,
            sorted_ids,
            expert_ids,
            num_post_padded,
            False,
            top_k,
            config,
            compute_type=tl.bfloat16,
            split_dots=split_dots,
        )
        return out.view(-1, weight.shape[1])

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
        w13, w2 = self.weights["w13"][0], self.weights["w2"][0]
        if tokens == 0:
            return x.new_empty((0, w2.shape[1]))
        # The config `Nvfp4QuantizationEmulationTritonExperts` uses (bf16).
        config = try_get_optimal_moe_config(w13.size(), w2.size(), topk, None, tokens)
        split_dots = nvfp4_emulation_use_split_dots(tokens)
        alignment = moe_align_block_size(
            ids, config["BLOCK_SIZE_M"], self.num_experts, ignore_invalid_experts=True
        )[:3]
        gate_up = self._gemm("w13", x, topk, alignment, config, split_dots)
        activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
        del gate_up
        return self._gemm("w2", activated, 1, alignment, config, split_dots)


def mxfp8_experts_supported() -> bool:
    """Whether `XpuMxfp8Experts` runs on the current device: where
    `XpuFusedMoe` picks the `xe_2` grouped GEMM (Xe2 / Xe3, e.g. Arc Pro
    B60 / B70), which takes bf16 activations with e4m3 weights. Its `xe_3`
    GEMM picks the weight layout from the activation dtype, so bf16
    activations would read the e4m3 weights as bf16; upstream `XPUExperts`
    quantizes the activations there. That W8A8 path is not implemented
    here."""
    from vllm_xpu_kernels.fused_moe_interface import _uses_xe2_grouped_gemm

    device = torch.device("xpu", torch.xpu.current_device())
    return _uses_xe2_grouped_gemm(torch.empty(0, device=device))


def xpu_flat_experts(
    quant_format: QuantFormat, topk: int, gate_cap: float, up_cap: float
) -> Callable[..., FlatExperts]:
    """The XPU `FlatExperts` constructor of `quant_format` experts. Raises
    `NotImplementedError` for MXFP8 on devices without the `xe_2` grouped
    GEMM (`mxfp8_experts_supported`)."""
    if quant_format == "mxfp8":
        if not mxfp8_experts_supported():
            raise NotImplementedError(
                "Inkling2 MXFP8 routed experts on XPU run the xe_2 grouped "
                "GEMM of vllm-xpu-kernels (Xe2 / Xe3 devices, e.g. Arc Pro "
                "B60 / B70); this device (or FORCE_XE3P_KERNEL) selects the "
                "xe_3 GEMM, which needs quantized activations"
            )
        cls: type = XpuMxfp8Experts
    else:
        cls = XpuNvfp4Experts
    return partial(cls, topk=topk, gate_cap=gate_cap, up_cap=up_cap)
