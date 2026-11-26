# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU-only module (no counterpart at upstream `b7186d137a`): the Inkling2
routed-expert backends (`FlatExperts`) of `moe.Inkling2Experts`, in place of
the Humming expert GEMMs of the CUDA copy.

MXFP8 runs the SYCL grouped-GEMM fused MoE of vllm-xpu-kernels
(`XpuFusedMoe`): bf16 activations (W8A16, as Humming on Hopper) where the
built `XpuFusedMoe` runs its xe_2 grouped GEMM, MXFP8-quantized ones where it
runs xe_3 (`mxfp8_quantizes_activations`). It takes `[gate | up]` halves
where the checkpoint interleaves the `w13` rows, and applies the routing
weight to the down-projection rows in fp32 while combining them
(`combines`), where Hopper scales the SiTU output before the down projection
and rounds it to bf16.

NVFP4 runs Hopper's flow with the NVFP4 W4A16 grouped expert GEMM of
`ops/nvfp4_moe.py` on bf16 activations (the checkpoint's own recipe
quantizes the expert activations to NVFP4 on SM10x / SM12x) in place of
Humming, on the checkpoint layout (interleaved `w13` rows, consumed by
`swiglu`).

Both backends keep the layer's Parameters: the kernel layout lives in them,
visible as module state.
"""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

from .ops.moe import swiglu
from .ops.nvfp4_moe import nvfp4_moe_config, nvfp4_moe_w4a16, use_nvfp4_moe_td

if TYPE_CHECKING:
    from .moe import FlatExperts, QuantFormat

ExpertWeights = dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]


def deinterleave_rows(t: torch.Tensor) -> None:
    """In place: each expert's rows `(g0, u0, g1, u1, ...)` of `t`
    `[E, 2I, ...]` become `[g0 .. g(I-1) | u0 .. u(I-1)]`. One expert is
    copied at a time, so this step needs one expert's rows of extra
    memory."""
    rows = t.view(torch.uint8) if t.element_size() == 1 else t
    for expert in rows:
        expert.copy_(torch.cat((expert[0::2], expert[1::2])))


_SITU_ARGS = ("activation_situ_beta", "activation_situ_linear_beta")


def check_fused_moe_support(fused_moe_interface) -> None:
    """Fail at model init, before any weight is touched, unless the installed
    vllm-xpu-kernels `fused_moe_interface` has what the MXFP8 experts use:
    `XpuFusedMoe._uses_xe2` (from `_uses_xe2_grouped_gemm`, added in 0.1.17),
    which `mxfp8_quantizes_activations` reads, and the SiTU betas of
    `XpuFusedMoe` (since 0.1.13). The private module function stands in for
    the private attribute it sets: a release that renames the function but
    keeps the attribute is rejected here until this check follows it."""
    import inspect

    params = inspect.signature(fused_moe_interface.XpuFusedMoe).parameters
    if hasattr(fused_moe_interface, "_uses_xe2_grouped_gemm") and all(
        name in params for name in _SITU_ARGS
    ):
        return
    from importlib.metadata import PackageNotFoundError, version

    try:
        found = version("vllm-xpu-kernels")
    except PackageNotFoundError:
        found = "unknown"
    raise NotImplementedError(
        "Inkling 2 MXFP8 experts on XPU need vllm-xpu-kernels >= 0.1.17 "
        f"(XpuFusedMoe with SiTU and _uses_xe2); found {found}"
    )


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


def mxfp8_quantizes_activations(moe) -> bool:
    """Whether the MXFP8 experts feed the built `XpuFusedMoe` `moe`
    MXFP8-quantized activations: exactly when its grouped GEMM is the xe_3
    one, which takes block-scaled activations (`not moe._uses_xe2`, the
    library's own kernel choice: devices that are neither Xe2 nor Xe3, or
    `FORCE_XE3P_KERNEL`); the xe_2 one takes bf16 activations (W8A16).
    Upstream `XPUExperts` decides from the device test alone
    (`fused_moe/experts/xpu_moe.py`), which disagrees with the library
    under `FORCE_XE3P_KERNEL`."""
    return not moe._uses_xe2


def quantize_mxfp8_activations(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """MXFP8 activations as the modular kernel's prepare step makes them for
    `XPUExperts` (`prepare_finalize/no_dp_ep.py:_quantize_input` with a
    dynamic MXFP8 quant config): `moe_kernel_quantize_input(...,
    quant_dtype="mxfp8")`, on XPU `xpu_mxfp8_quantize`: e4m3 values and e8m0
    scales `[T, H / 32]`."""
    return moe_kernel_quantize_input(
        A=x, A_scale=None, quant_dtype="mxfp8", per_act_token_quant=False
    )


class XpuMxfp8Experts:
    """MXFP8 experts on the SYCL fused MoE (`XpuFusedMoe`).

    `XpuFusedMoe` (vllm-xpu-kernels' fused MoE interface) is called
    directly, because upstream `XPUExperts` has no SiTU yet
    (vllm-project/vllm#49598). The activation decision
    (`mxfp8_quantizes_activations`, from the built `XpuFusedMoe`) and the
    activation quantization (`quantize_mxfp8_activations`, upstream's
    prepare step) are separate functions; `XPUExperts` is a modular-kernel
    expert class with its own config and prepare / finalize, so a later
    switch to it is not shown to be local to this class.

    Load-time memory: the de-interleave copies one expert at a time, but
    `XpuFusedMoe` then builds the relayout (`[E, K, N]` on the xe_2 GEMM)
    of the whole `w13` and `w2` of the local experts before it releases
    the loaded storage. The extra peak per layer is therefore about the
    local `w13` plus `w2` weights (on the xe_2 GEMM the scales are used as
    loaded): 0.95 GiB for Inkling 2 Mini (TP1) and 2.85 GiB for Small
    (TP4), as for upstream `XPUExperts`.

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
        # Validated before the Parameters are rewritten.
        activation = _situ_activation(gate_cap, up_cap)
        # The checkpoint interleaves the gate / up rows; the SYCL kernel
        # reads `[gate | up]` halves (and would silently compute garbage on
        # the interleaved layout).
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
            **activation,
        )
        self.quantize_activations = mxfp8_quantizes_activations(self.moe)

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
            a, a_scale = x, None
            if self.quantize_activations:
                a, a_scale = quantize_mxfp8_activations(x)
            self.moe.apply(out, a, weights.view(ids.shape), ids, a1q_scale=a_scale)
        return out


class XpuNvfp4Experts:
    """NVFP4 experts on the NVFP4 W4A16 Triton kernel (`nvfp4_moe_w4a16`)
    with bf16 activations: `w13` GEMM, `swiglu` (routing weight before the
    down projection, as on Hopper) and `w2` GEMM over the flat rows. The
    path (split-dots below `split_dots_bound` rows per expert, else the
    default path) and the launch config come from `nvfp4_moe_config`, for
    the weight loads the kernel takes (`use_nvfp4_moe_td`); the two paths
    round differently, and `VLLM_BATCH_INVARIANT` pins the default path,
    one config and the pointer loads.

    Args:
        weights: Per projection (`w13`, `w2`), the checkpoint weights
            `[E, N, K / 2]` (uint8, two e2m1 each), e4m3 block scales
            `[E, N, K / 16]` (the layer's Parameters, used as they are) and
            fp32 global scales `[E]` (`amax / outer_scale_denominator`).
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
        nvfp4_moe_w4a16(
            a,
            weight,
            scale,
            global_scale,
            out,
            sorted_ids,
            expert_ids,
            num_post_padded,
            top_k,
            config,
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
        # The split-dots bound is in rows per expert over all experts (the
        # routing's), as it was measured, for the kernel's weight loads.
        config, split_dots = nvfp4_moe_config(
            tokens,
            topk,
            num_global_experts,
            self.num_experts,
            x.shape[-1],
            use_nvfp4_moe_td(w13),
        )
        alignment = moe_align_block_size(
            ids, config["BLOCK_SIZE_M"], self.num_experts, ignore_invalid_experts=True
        )[:3]
        gate_up = self._gemm("w13", x, topk, alignment, config, split_dots)
        activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
        del gate_up
        return self._gemm("w2", activated, 1, alignment, config, split_dots)


def xpu_flat_experts(
    quant_format: "QuantFormat", topk: int, gate_cap: float, up_cap: float
) -> Callable[..., "FlatExperts"]:
    """The `FlatExperts` constructor of `quant_format` experts with `topk`
    slots per token and the SiTU softcaps of the layer. MXFP8 fails here
    (at model init, before any weight is loaded) on a vllm-xpu-kernels
    without the SiTU fused MoE, or on softcaps it cannot express."""
    cls: type = XpuMxfp8Experts if quant_format == "mxfp8" else XpuNvfp4Experts
    if quant_format == "mxfp8":
        from vllm_xpu_kernels import fused_moe_interface

        check_fused_moe_support(fused_moe_interface)
        _situ_activation(gate_cap, up_cap)
    return partial(cls, topk=topk, gate_cap=gate_cap, up_cap=up_cap)
