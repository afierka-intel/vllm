# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the tensor-descriptor (TD) load path in
fused_moe_kernel_gptq_awq (VLLMZ-1811).

Covers use_int4_w4a16 (packed-nibble unpack via tl.interleave), gated on
VLLM_TRITON_USE_TD. XPU-portable (unlike tests/kernels/moe/test_moe.py::
test_fused_moe_wn16, which is hardcoded to device="cuda").

use_int8_w8a16's TD path is force-disabled at the launch site
(invoke_fused_moe_wna16_triton_kernel) due to a Triton-XPU 3.7.1 codegen
bug: a tensor-descriptor-loaded uint8 B tile, multiplied by a dequant
scale and fed into tl.dot, produces garbage whenever BLOCK_SIZE_N >= 64
and BLOCK_SIZE_M != 32 -- exactly this kernel's default wna16 config for
int8. The kernel has no int8 branch under USE_TD at all (only int4 does);
the TD-specific tests below
(test_fused_moe_wn16_td_bit_exact_vs_pointer,
test_fused_moe_wn16_td_k_tail_bit_exact) are therefore int4-only --
parametrizing them over int8 would just compare the pointer path against
itself. test_fused_moe_wn16_use_td still covers both weight bit widths,
since it validates correctness against the fp32 reference independent of
TD. test_int8_td_forced_off_at_default_config pins the int8 guard
explicitly so a future accidental re-enable of int8 TD is caught loudly.
"""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_topk, override_config
from vllm.model_executor.layers.fused_moe.config import (
    int4_w4a16_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl

DEVICE = "xpu" if current_platform.is_xpu() else "cuda"
_HAS_TL_MAKE_DESC = hasattr(tl, "make_tensor_descriptor")

vllm_config = VllmConfig()

WN16_MNK = [
    (1, 128, 128),
    (32, 2048, 128),
    (222, 2048, 1024),
]
NUM_EXPERTS = [8]
TOP_KS = [2]
GROUP_SIZES = [128]
WEIGHT_BITS = [4, 8]
HAS_ZP = [True, False]

_USE_TD_PARAMS = [
    False,
    pytest.param(
        True,
        marks=pytest.mark.skipif(
            not _HAS_TL_MAKE_DESC,
            reason="Triton < 3.6 lacks tl.make_tensor_descriptor",
        ),
    ),
]


def fused_moe(
    hidden_states,
    w1,
    w2,
    score,
    topk,
    renormalize=False,
    quant_config=None,
    global_num_experts=-1,
    expert_map=None,
):
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score.float(), topk, renormalize
    )
    return fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        quant_config=quant_config,
    )


def torch_moe(a, w1, w2, score, topk):
    """Pure-PyTorch MoE reference for correctness validation.

    Implements fused MoE with SiLU+Mul activation and expert routing.
    Used as reference to validate Triton kernel outputs.
    """
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    m, k = a.shape
    a_rep = a.view(m, -1, k).repeat(1, topk, 1).reshape(-1, k)
    out = torch.zeros(m * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    topk_flat = topk_ids.view(-1)
    act = SiluAndMul()
    for i in range(w1.shape[0]):
        mask = topk_flat == i
        if mask.sum():
            tmp = a_rep[mask] @ w1[i].transpose(0, 1)
            tmp = act(tmp)
            out[mask] = tmp @ w2[i].transpose(0, 1)

    return (
        (out.view(m, -1, w2.shape[1]).to(torch.float32) * topk_weight.view(m, -1, 1))
        .sum(dim=1)
        .to(out.dtype)
    )


def _prepare_quantized_weights(e, n, k, group_size, weight_bits, has_zp, device, dtype):
    """Prepare quantized MoE weights with scales and zero-points.

    Returns: (w1_ref, w2_ref, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp)
    """
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10

    if weight_bits == 4:
        pack_factor = 2
        quant_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    else:
        pack_factor = 1
        quant_type = scalar_types.uint8 if has_zp else scalar_types.uint8b128

    w1_ref = w1.clone()
    w2_ref = w2.clone()
    w1_qw = torch.empty((e, 2 * n, k // pack_factor), device=device, dtype=torch.uint8)
    w2_qw = torch.empty((e, k, n // pack_factor), device=device, dtype=torch.uint8)
    w1_sc = torch.empty((e, 2 * n, k // group_size), device=device, dtype=dtype)
    w2_sc = torch.empty((e, k, n // group_size), device=device, dtype=dtype)

    w1_zp = torch.empty(
        (e, 2 * n // pack_factor, k // group_size), device=device, dtype=torch.uint8
    )
    w2_zp = torch.empty(
        (e, k // pack_factor, n // group_size), device=device, dtype=torch.uint8
    )

    for i in range(e * 2):
        expert_id = i % e
        if i // e == 0:
            w, w_ref_arr, w_qw_arr, w_sc_arr, w_zp_arr = w1, w1_ref, w1_qw, w1_sc, w1_zp
        else:
            w, w_ref_arr, w_qw_arr, w_sc_arr, w_zp_arr = w2, w2_ref, w2_qw, w2_sc, w2_zp

        weight, qweight, scales, qzeros = quantize_weights(
            w[expert_id].T, quant_type, group_size, has_zp, False
        )
        weight = weight.T
        qweight = qweight.T.contiguous().to(torch.uint8)
        scales = scales.T

        if has_zp:
            qzeros = qzeros.T.contiguous().to(torch.uint8)

        if weight_bits == 4:
            qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]
            if has_zp:
                qzeros = qzeros[1::2, :] * 16 + qzeros[::2, :]

        w_ref_arr[expert_id] = weight
        w_qw_arr[expert_id] = qweight
        w_sc_arr[expert_id] = scales
        if has_zp:
            w_zp_arr[expert_id] = qzeros

    return w1_ref, w2_ref, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp


def _build_quant_config(weight_bits, w1_sc, w2_sc, w1_zp, w2_zp, has_zp, group_size):
    kwargs = dict(
        w1_scale=w1_sc,
        w2_scale=w2_sc,
        w1_zp=w1_zp if has_zp else None,
        w2_zp=w2_zp if has_zp else None,
        block_shape=[0, group_size],
    )
    if weight_bits == 4:
        return int4_w4a16_moe_quant_config(**kwargs)
    return int8_w8a16_moe_quant_config(**kwargs)


@pytest.mark.parametrize("m,n,k", WN16_MNK)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("has_zp", HAS_ZP)
@pytest.mark.parametrize("weight_bits", WEIGHT_BITS)
@pytest.mark.parametrize("use_td", _USE_TD_PARAMS)
def test_fused_moe_wn16_use_td(
    m, n, k, e, topk, group_size, has_zp, weight_bits, use_td, monkeypatch
):
    """fused_moe_kernel_gptq_awq correctness vs PyTorch reference, with the
    tensor-descriptor path forced on or off via VLLM_TRITON_USE_TD."""
    monkeypatch.setenv("VLLM_TRITON_USE_TD", "1" if use_td else "0")
    dtype = torch.bfloat16
    torch.manual_seed(7)

    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    score = torch.randn((m, e), device=DEVICE, dtype=dtype)

    w1_ref, w2_ref, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp = (
        _prepare_quantized_weights(
            e, n, k, group_size, weight_bits, has_zp, DEVICE, dtype
        )
    )
    quant_config = _build_quant_config(
        weight_bits, w1_sc, w2_sc, w1_zp, w2_zp, has_zp, group_size
    )

    with set_current_vllm_config(vllm_config):
        triton_output = fused_moe(
            a,
            w1_qw,
            w2_qw,
            score,
            topk,
            renormalize=False,
            global_num_experts=e,
            quant_config=quant_config,
        )
        torch_output = torch_moe(a, w1_ref, w2_ref, score, topk)

    torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)


@pytest.mark.skipif(
    not _HAS_TL_MAKE_DESC, reason="Triton < 3.6 lacks tl.make_tensor_descriptor"
)
@pytest.mark.parametrize("m,n,k", WN16_MNK)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("has_zp", HAS_ZP)
def test_fused_moe_wn16_td_bit_exact_vs_pointer(
    m, n, k, e, topk, group_size, has_zp, monkeypatch
):
    """Direct TD-vs-pointer-path comparison on identical inputs, int4 only.

    Tighter than the fp32-reference tolerance check above (atol=2e-2), which
    is loose enough to miss a subtly wrong nibble interleave -- a swapped
    low/high nibble would often still land within that tolerance for random
    weights. The two Triton paths should agree much more closely than either
    agrees with the fp32 reference.

    int8_w8a16 is deliberately excluded here: its TD path is force-disabled
    at the launch site (invoke_fused_moe_wna16_triton_kernel) due to a
    Triton-XPU codegen bug, so both legs would run the pointer path and the
    comparison would be a no-op. See test_int8_td_forced_off_at_default_config
    below for the dedicated guard-pinning test.
    """
    weight_bits = 4
    dtype = torch.bfloat16
    torch.manual_seed(7)

    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    score = torch.randn((m, e), device=DEVICE, dtype=dtype)

    _, _, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp = _prepare_quantized_weights(
        e, n, k, group_size, weight_bits, has_zp, DEVICE, dtype
    )
    quant_config = _build_quant_config(
        weight_bits, w1_sc, w2_sc, w1_zp, w2_zp, has_zp, group_size
    )

    def run(use_td: bool) -> torch.Tensor:
        monkeypatch.setenv("VLLM_TRITON_USE_TD", "1" if use_td else "0")
        with set_current_vllm_config(vllm_config):
            return fused_moe(
                a,
                w1_qw,
                w2_qw,
                score,
                topk,
                renormalize=False,
                global_num_experts=e,
                quant_config=quant_config,
            )

    pointer_output = run(use_td=False)
    td_output = run(use_td=True)

    torch.testing.assert_close(td_output, pointer_output, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not _HAS_TL_MAKE_DESC, reason="Triton < 3.6 lacks tl.make_tensor_descriptor"
)
@pytest.mark.parametrize("has_zp", HAS_ZP)
def test_fused_moe_wn16_td_k_tail_bit_exact(has_zp, monkeypatch):
    """K-tail (block_k_diviable=False) case, int4 only: forces the
    automatic tensor-descriptor zero-fill, compared bit-exact against the
    pointer path's explicit K-mask.

    Bypasses get_moe_wna16_block_config's auto block-size selection (which
    always keeps K block-aligned for the group_size/BLOCK_SIZE_K combinations
    it picks) via override_config, forcing a BLOCK_SIZE_K that does not
    divide K.

    int8_w8a16 is excluded (see the module docstring and
    test_fused_moe_wn16_td_bit_exact_vs_pointer above): its TD path is
    force-disabled at the launch site, so an int8 case here would be a
    pointer-vs-pointer no-op.
    """
    weight_bits = 4
    m, n, k = 33, 512, 96
    e, topk, group_size = 8, 2, 32
    # BLOCK_SIZE_K=64 does not divide K=96 -> block_k_diviable=False.
    # k % group_size == 0 (96 % 32 == 0) keeps the scale-tensor shape valid.
    forced_config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
    }
    dtype = torch.bfloat16
    torch.manual_seed(7)

    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    score = torch.randn((m, e), device=DEVICE, dtype=dtype)

    _, _, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp = _prepare_quantized_weights(
        e, n, k, group_size, weight_bits, has_zp, DEVICE, dtype
    )
    quant_config = _build_quant_config(
        weight_bits, w1_sc, w2_sc, w1_zp, w2_zp, has_zp, group_size
    )

    def run(use_td: bool) -> torch.Tensor:
        monkeypatch.setenv("VLLM_TRITON_USE_TD", "1" if use_td else "0")
        with set_current_vllm_config(vllm_config), override_config(forced_config):
            return fused_moe(
                a,
                w1_qw,
                w2_qw,
                score,
                topk,
                renormalize=False,
                global_num_experts=e,
                quant_config=quant_config,
            )

    pointer_output = run(use_td=False)
    td_output = run(use_td=True)

    torch.testing.assert_close(td_output, pointer_output, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not _HAS_TL_MAKE_DESC, reason="Triton < 3.6 lacks tl.make_tensor_descriptor"
)
def test_int8_td_forced_off_at_default_config(monkeypatch):
    """Pins the int8_w8a16 TD guard at the launch site.

    At the wna16 kernel's default config (BLOCK_SIZE_N=64), a Triton-XPU
    codegen bug corrupts int8's TD path (descriptor uint8 load * scale ->
    tl.dot). resolve_moe_gptq_awq_use_td() alone would return True here
    (VLLM_TRITON_USE_TD=1 forces it), but invoke_fused_moe_wna16_triton_kernel
    must still force USE_TD=False for use_int8_w8a16 regardless. m=222 forces
    the non-batch-1 default BLOCK_SIZE_N=64 via get_moe_wna16_block_config --
    exactly the failing shape. If the guard is ever removed without the
    underlying bug being fixed, this assertion fails loudly instead of
    silently corrupting output.
    """
    m, n, k, e, topk, group_size = 222, 2048, 1024, 8, 2, 128
    dtype = torch.bfloat16
    torch.manual_seed(7)

    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    score = torch.randn((m, e), device=DEVICE, dtype=dtype)

    w1_ref, w2_ref, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp = (
        _prepare_quantized_weights(e, n, k, group_size, 8, False, DEVICE, dtype)
    )
    quant_config = _build_quant_config(8, w1_sc, w2_sc, w1_zp, w2_zp, False, group_size)

    monkeypatch.setenv("VLLM_TRITON_USE_TD", "1")
    with set_current_vllm_config(vllm_config):
        triton_output = fused_moe(
            a,
            w1_qw,
            w2_qw,
            score,
            topk,
            renormalize=False,
            global_num_experts=e,
            quant_config=quant_config,
        )
        torch_output = torch_moe(a, w1_ref, w2_ref, score, topk)

    torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)
