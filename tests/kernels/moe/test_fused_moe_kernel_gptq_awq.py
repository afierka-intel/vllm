# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the tensor-descriptor (TD) load path in
fused_moe_kernel_gptq_awq.

Covers both weight bit widths the kernel supports -- use_int8_w8a16 (direct
descriptor load) and use_int4_w4a16 (packed-nibble unpack via tl.interleave)
-- gated on VLLM_TRITON_USE_TD. Complements
tests/kernels/moe/test_moe.py::test_fused_moe_wn16, which covers the pointer
path over a wider shape sweep but has no TD coverage.

Every test here forces TD on and is skipped where TD cannot run, so nothing
in this file exercises the pointer path *in isolation* -- the bit-exactness
tests do run it, as the comparison baseline. Standalone pointer-path coverage
lives in tests/kernels/moe/test_moe.py::test_fused_moe_wn16, over a wider
shape sweep.

test_fused_moe_wn16_use_td checks TD against an fp32 reference; the
_matches_pointer, _k_tail and _n_tail tests compare TD against the
pointer path directly, which is tight enough to catch a subtly wrong nibble
interleave that the fp32 tolerance would let through. The K- and N-tail cases
target the two places the paths legitimately read different B values.
test_int8_td_at_default_block_config pins the block shape that used to
miscompile on triton-xpu 3.7.1 (see
intel/intel-xpu-backend-for-triton#7510, fixed in 3.7.2).
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
from vllm.model_executor.layers.fused_moe.utils import (
    TD_MIN_GATHER_ROWS,
    moe_use_td_hw_supported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl

DEVICE = "xpu" if current_platform.is_xpu() else "cuda"
_HAS_TL_MAKE_DESC = hasattr(tl, "make_tensor_descriptor")
_TD_SKIP_REASON = (
    "TD path needs tl.make_tensor_descriptor (Triton >= 3.6) and hardware "
    "whose A-gather compiles: XPU, or NVIDIA Blackwell (sm100+), since "
    "tile::gather4 (tcgen05/TMEM) is rejected by ptxas on Hopper and earlier"
)


def _td_unsupported() -> bool:
    return not _HAS_TL_MAKE_DESC or not moe_use_td_hw_supported()


# One bf16 ULP at the magnitudes this kernel produces (2^-13 for values around
# 0.03), doubled to leave headroom over a single-element rounding disagreement.
_ONE_ULP_ATOL = 2.5e-4


@pytest.fixture(scope="module")
def vllm_config():
    return VllmConfig()


# m=1 is kept on purpose even though the launcher gates TD off for a
# single-row A (see the M == 1 check in invoke_fused_moe_wna16_triton_kernel):
# its second GEMM still runs with M = m * topk, so the TD path is exercised
# there, and the case also guards the gate itself against regressing into
# wrong output rather than merely slower output.
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


def _assert_td_matches_pointer(td_output, pointer_output):
    """Compare the TD and pointer paths at one bf16 ULP.

    Both accumulate in fp32 and round to bf16 at slightly different points, so a
    single-element 1-ULP disagreement (2^-13 at the magnitudes here) is expected
    rather than a defect; observed on Blackwell with TD the value closer to the
    fp32 reference. Anything larger is a real divergence -- fault injection at
    4 ULP fails, as does 1% of elements off by 5%.
    """
    torch.testing.assert_close(td_output, pointer_output, atol=_ONE_ULP_ATOL, rtol=1e-4)


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


@pytest.mark.skipif(_td_unsupported(), reason=_TD_SKIP_REASON)
@pytest.mark.parametrize("m,n,k", WN16_MNK)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("has_zp", HAS_ZP)
@pytest.mark.parametrize("weight_bits", WEIGHT_BITS)
def test_fused_moe_wn16_use_td(
    m, n, k, e, topk, group_size, has_zp, weight_bits, monkeypatch, vllm_config
):
    """TD-path correctness vs the PyTorch reference.

    TD-on only: the TD-off leg would duplicate
    tests/kernels/moe/test_moe.py::test_fused_moe_wn16, which already covers
    the pointer path against the same reference over a wider shape sweep.
    """
    monkeypatch.setenv("VLLM_TRITON_USE_TD", "1")
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


@pytest.mark.skipif(_td_unsupported(), reason=_TD_SKIP_REASON)
@pytest.mark.parametrize("m,n,k", WN16_MNK)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("has_zp", HAS_ZP)
@pytest.mark.parametrize("weight_bits", WEIGHT_BITS)
def test_fused_moe_wn16_td_matches_pointer(
    m, n, k, e, topk, group_size, has_zp, weight_bits, monkeypatch, vllm_config
):
    """Direct TD-vs-pointer-path comparison on identical inputs.

    Tighter than the fp32-reference tolerance check above (atol=2e-2), which
    is loose enough to miss a subtly wrong nibble interleave -- a swapped
    low/high nibble would often still land within that tolerance for random
    weights. The two Triton paths should agree much more closely than either
    agrees with the fp32 reference.
    """
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

    _assert_td_matches_pointer(td_output, pointer_output)


@pytest.mark.skipif(_td_unsupported(), reason=_TD_SKIP_REASON)
@pytest.mark.parametrize("has_zp", HAS_ZP)
@pytest.mark.parametrize("weight_bits", WEIGHT_BITS)
def test_fused_moe_wn16_td_k_tail_matches_pointer(
    has_zp, weight_bits, monkeypatch, vllm_config
):
    """K-tail (block_k_diviable=False) case: forces the automatic
    tensor-descriptor zero-fill, compared bit-exact against the pointer
    path's explicit K-mask.

    Bypasses get_moe_wna16_block_config's auto block-size selection (which
    always keeps K block-aligned for the group_size/BLOCK_SIZE_K combinations
    it picks) via override_config, forcing a BLOCK_SIZE_K that does not
    divide K.

    As with the N-tail test, the kernel's ``K`` is ``A.size(1)``, not this
    ``k``: GEMM1 runs with ``K = k`` and GEMM2 with ``K = n`` (its A is the
    intermediate activation). Here the tail lands in GEMM1 (96 % 64 == 32),
    which is enough to exercise the descriptor's zero-fill; asserted below so
    the premise cannot silently decay.
    """
    m, n, k = 33, 512, 96
    e, topk, group_size = 8, 2, 32
    # k % group_size == 0 (96 % 32 == 0) keeps the scale-tensor shape valid.
    forced_config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
    }
    assert k % forced_config["BLOCK_SIZE_K"] != 0, "GEMM1 (K=k) must have a K tail"
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

    _assert_td_matches_pointer(td_output, pointer_output)


@pytest.mark.skipif(_td_unsupported(), reason=_TD_SKIP_REASON)
@pytest.mark.parametrize("has_zp", HAS_ZP)
@pytest.mark.parametrize("weight_bits", WEIGHT_BITS)
def test_fused_moe_wn16_td_n_tail_matches_pointer(
    has_zp, weight_bits, monkeypatch, vllm_config
):
    """N-tail: the pointer path wraps tail lanes with ``% N`` while TD gets
    zero-fill, reconciled only by the ``offs_cn < N`` store mask.

    The kernel's N is ``B.size(1)``, so GEMM1 runs at ``N = 2n`` and GEMM2 at
    ``N = k``; both are asserted below to keep a tail.
    """
    m, n, k = 33, 48, 48
    e, topk, group_size = 8, 2, 16
    forced_config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
    }
    block_n = forced_config["BLOCK_SIZE_N"]
    assert (2 * n) % block_n != 0, "GEMM1 (N=2n) must have an N tail"
    assert k % block_n != 0, "GEMM2 (N=k) must have an N tail"
    # These shapes also leave a K tail, which off XPU trips the K-alignment
    # bail-out and disables TD for both GEMMs -- the comparison would then be
    # pointer-vs-pointer and pass for the wrong reason. Skip instead.
    if not current_platform.is_xpu() and k % forced_config["BLOCK_SIZE_K"] != 0:
        pytest.skip(
            "TD is disabled off-XPU for unaligned K, so this would compare the "
            "pointer path against itself"
        )
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

    _assert_td_matches_pointer(run(use_td=True), run(use_td=False))


@pytest.mark.skipif(_td_unsupported(), reason=_TD_SKIP_REASON)
@pytest.mark.parametrize("block_size_m", [2, 4])
@pytest.mark.parametrize("weight_bits", WEIGHT_BITS)
def test_td_skipped_below_min_gather_rows(
    block_size_m, weight_bits, monkeypatch, vllm_config
):
    """A BLOCK_SIZE_M below the gather minimum must fall back, not abort.

    tensor_descriptor.gather() asserts at least TD_MIN_GATHER_ROWS rows, and
    get_default_config's use_moe_wna16_cuda branch picks
    min(16, next_power_of_2(M)) -- so M <= 4 yields 1, 2 or 4 and used to kill
    the launch outright ("descriptor gather must have at least 8 rows").
    Reachable via TritonExperts.apply, which calls the launcher without the
    should_moe_wna16_use_cuda check that diverts fused_experts_impl away from
    such configs; reproduced on B200. Every other test here forces a block
    config >= 8, which is why a green suite still shipped the crash.
    """
    assert block_size_m < TD_MIN_GATHER_ROWS, "premise: must be below the minimum"
    m, n, k = 33, 512, 128
    e, topk, group_size = 8, 2, 32
    forced_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
    }
    dtype = torch.bfloat16
    torch.manual_seed(7)

    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    score = torch.randn((m, e), device=DEVICE, dtype=dtype)
    w1_ref, w2_ref, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp = (
        _prepare_quantized_weights(
            e, n, k, group_size, weight_bits, False, DEVICE, dtype
        )
    )
    quant_config = _build_quant_config(
        weight_bits, w1_sc, w2_sc, w1_zp, w2_zp, False, group_size
    )

    monkeypatch.setenv("VLLM_TRITON_USE_TD", "1")
    with set_current_vllm_config(vllm_config), override_config(forced_config):
        out = fused_moe(
            a,
            w1_qw,
            w2_qw,
            score,
            topk,
            renormalize=False,
            global_num_experts=e,
            quant_config=quant_config,
        )
        ref = torch_moe(a, w1_ref, w2_ref, score, topk)

    # Reaching here at all is the regression check; the tolerance mirrors the
    # fp32-reference comparison used elsewhere in this file.
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


@pytest.mark.skipif(_td_unsupported(), reason=_TD_SKIP_REASON)
def test_int8_td_at_default_block_config(monkeypatch, vllm_config):
    """int8_w8a16 TD at the launcher's own default block config.

    Regression pin for intel/intel-xpu-backend-for-triton#7510: a
    descriptor-loaded uint8 B tile multiplied by a dequant scale and fed into
    tl.dot produced garbage on triton-xpu 3.7.1 whenever BLOCK_SIZE_N >= 64
    and BLOCK_SIZE_M != 32 -- exactly the config get_moe_wna16_block_config
    selects for int8 at non-batch-1 sizes, so it would have hit the majority
    of real int8 traffic. Fixed in 3.7.2, which docker/Dockerfile.xpu pins --
    but requirements/xpu.txt does not, which is why the launcher checks the
    version at runtime and falls back to the pointer path on an affected
    build.

    m=222 is the shape that originally reproduced it. The generic
    parametrized tests above use override_config or smaller shapes, so this
    keeps an explicit case pinned at the auto-selected default.
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
