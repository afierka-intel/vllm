# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 XPU MoE: routing / SiTU / combine kernels, the NVFP4 W4A16 grouped
expert GEMM (`xpu/ops/nvfp4_moe.py`) and the routed-expert backends
(`xpu/experts.py`) vs. dequantized torch references, at the expert shapes of
Inkling 2 Mini (hidden 2304, 128 experts) and Small (hidden 4608, 384
experts, one EP rank of 4), top-8."""

import functools
import types
from typing import Any

import pytest
import torch

from tests.models.inkling2.xpu.utils import (
    DEVICE,
    dequant_mxfp8,
    dequant_nvfp4,
    requires_first_config,
    requires_xpu,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.models.inkling2.configs import Inkling2ModelConfig
from vllm.models.inkling2.xpu.experts import (
    XpuMxfp8Experts,
    XpuNvfp4Experts,
    _situ_activation,
    check_fused_moe_support,
    deinterleave_rows,
    quantize_mxfp8_activations,
    xpu_flat_experts,
)
from vllm.models.inkling2.xpu.moe import Inkling2MoE
from vllm.models.inkling2.xpu.ops import moe as moe_ops
from vllm.models.inkling2.xpu.ops import nvfp4_moe
from vllm.models.inkling2.xpu.ops.activation import _div_rn
from vllm.models.inkling2.xpu.ops.linear import linear_k_groups
from vllm.models.inkling2.xpu.ops.moe import combine, select_experts, swiglu
from vllm.models.inkling2.xpu.ops.nvfp4_moe import (
    _block_scales_to_f32,
    nvfp4_moe_w4a16,
    split_dots_bound,
    split_dots_config,
    use_nvfp4_moe_td,
    use_split_dots,
)
from vllm.triton_utils import tl, triton

pytestmark = requires_xpu

GATE_CAP, UP_CAP, TOPK = 4.0, 25.0, 8
# hidden, intermediate, experts, local experts (EP rank 1 of 4 for Small).
SHAPES = {"mini": (2304, 1152, 128, 128), "small": (4608, 2304, 384, 96)}
FORMATS = ["mxfp8", "nvfp4"]
# NVFP4 outer scale denominator of the checkpoints' quantizer.
DENOMINATOR = 2688.0
# NVFP4 takes the split-dots path below 3072 tokens here (Mini with descriptor
# loads below 512, with pointer loads below 256; Small up to 3024).
EXPERT_TOKENS = [1, 64, 300, 3072]


def _torch_routing(x, gate, bias, route_scale):
    logits = linear_k_groups(x, gate)
    _, ids = torch.topk(logits.sigmoid() + bias, TOPK, sorted=True)
    selected = torch.nn.functional.logsigmoid(logits.gather(1, ids))
    return selected.softmax(-1) * route_scale, ids


def _router(hidden, num_experts, ties=False):
    gate = (torch.randn(num_experts, hidden, device=DEVICE) * 0.02).bfloat16()
    bias = torch.randn(num_experts, device=DEVICE) * 0.01
    if ties:
        # Equal rows and biases tie experts 1-3, which every token selects.
        gate[[2, 3]] = gate[1].clone()
        bias[[1, 2, 3]] = 1.0
    return gate, bias


@pytest.mark.parametrize(
    "hidden,num_experts,tokens,ties",
    [
        (2304, 128, 5, True),
        (2304, 128, 300, False),
        (4608, 384, 64, False),
        (4608, 384, 2100, True),
    ],
)
def test_select_experts_matches_torch(hidden, num_experts, tokens, ties):
    """Each (token, slot) gets its expert's index among the local experts
    (-1 if not local, or for a padding token), torch's routing weight (ties
    broken towards the lower expert id) and its own flat row in `src2dst`
    when local."""
    torch.manual_seed(0)
    x = torch.randn(tokens, hidden, device=DEVICE).bfloat16()
    gate, bias = _router(hidden, num_experts, ties)
    lo, hi = 0 if ties else 4, num_experts // 2
    bounds = torch.tensor([lo, hi], dtype=torch.int32, device=DEVICE)
    is_padding = torch.arange(tokens, device=DEVICE) >= tokens - 2
    ids, weights, src2dst = select_experts(x, gate, bias, TOPK, 8.0, bounds, is_padding)

    ref_weights, ref_ids = _torch_routing(x, gate, bias, 8.0)
    if ties:
        # The tied experts in id order; torch's order of ties is unspecified.
        tied = torch.tensor([1, 2, 3], device=DEVICE)
        assert (ids[:-2, :3] == tied).all()
        assert (ref_ids[:, :3].sort(-1).values == tied).all()
        ref_ids[:, :3] = tied
    local = (ref_ids >= lo) & (ref_ids < hi) & ~is_padding[:, None]
    torch.testing.assert_close(ids.long(), torch.where(local, ref_ids - lo, -1))
    real = ~is_padding
    torch.testing.assert_close(weights.view(tokens, -1)[real], ref_weights[real])
    rows = torch.arange(tokens * TOPK, device=DEVICE).view(tokens, -1)
    torch.testing.assert_close(
        src2dst.view(tokens, -1).long(), torch.where(local, rows, -1)
    )


@pytest.mark.parametrize("tokens", [5, 300])
def test_select_kernel_every_tuned_config(tokens):
    """Each autotuned warp count of the router top-k (small and large
    batches) gives torch's experts and routing weights."""
    torch.manual_seed(0)
    hidden, num_experts = 2304, 128
    x = torch.randn(tokens, hidden, device=DEVICE).bfloat16()
    gate, bias = _router(hidden, num_experts)
    bounds = torch.tensor([0, num_experts], dtype=torch.int32, device=DEVICE)
    ref_weights, ref_ids = _torch_routing(x, gate, bias, 8.0)
    partials = moe_ops.k_group_partials(x, gate)
    for config in moe_ops._select_kernel_tuned(tokens <= 256).configs:
        ids = torch.empty(tokens, TOPK, dtype=torch.int32, device=DEVICE)
        weights = torch.empty(tokens * TOPK, device=DEVICE)
        src2dst = torch.empty(tokens * TOPK, dtype=torch.int32, device=DEVICE)
        moe_ops._select_kernel[(tokens,)](
            partials,
            bias,
            bounds,
            None,
            ids,
            weights,
            src2dst,
            tokens,
            8.0,
            E=num_experts,
            E_POW2=num_experts,
            GROUPS=partials.shape[0],
            TOPK=TOPK,
            SORT=tokens <= 128,
            **config.all_kwargs(),
        )
        torch.testing.assert_close(ids.long(), ref_ids)
        torch.testing.assert_close(weights.view(tokens, -1), ref_weights)
        assert torch.equal(src2dst.long(), torch.arange(tokens * TOPK, device=DEVICE))


@pytest.mark.parametrize("hidden", [512, 2304, 4608])
def test_combine_sums_rows_in_slot_order(hidden):
    """Sum routed rows in slot order, including absent rows (-1) and a
    256-wide tail."""
    torch.manual_seed(42)
    down = torch.randn(16, hidden, device=DEVICE).bfloat16()
    rows = torch.tensor([0, 3, -1, 7, 2, 15, 4, 6] * 2, device=DEVICE).int()
    ref = torch.zeros(2, hidden, device=DEVICE)
    for slot in range(8):
        index = rows.view(2, 8)[:, slot].long()
        ref += torch.where(index[:, None] >= 0, down[index.clamp_min(0)].float(), 0)
    torch.testing.assert_close(
        combine(down, rows, 8, torch.bfloat16), ref.bfloat16(), atol=0, rtol=0
    )


@triton.jit
def _div_kernel(X, Out, Ref, CAP: tl.constexpr):
    offs = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    x = tl.load(X + offs)
    tl.store(Out + offs, _div_rn(x, CAP))
    tl.store(Ref + offs, tl.div_rn(x, CAP))


@pytest.mark.parametrize("cap", [GATE_CAP, UP_CAP])
def test_softcap_division_rounds_correctly(cap: float):
    """The softcap's division matches IEEE division bitwise on every bf16
    input (the GEMM outputs), including signed zeros and infinities."""
    x = torch.arange(1 << 16, dtype=torch.int32, device=DEVICE) << 16
    x = x.view(torch.float32)
    out, ref = torch.empty_like(x), torch.empty_like(x)
    _div_kernel[(x.numel() // 1024,)](x, out, ref, cap, enable_fp_fusion=False)
    assert torch.equal(out.view(torch.int32), ref.view(torch.int32))


def _situ(gate_up: torch.Tensor) -> torch.Tensor:
    """fp32 SiTU of the interleaved (gate, up) columns."""
    gate, up = gate_up.float()[:, 0::2], gate_up.float()[:, 1::2]
    gate_capped = GATE_CAP * torch.tanh(gate / GATE_CAP)
    return gate_capped * torch.sigmoid(gate) * (UP_CAP * torch.tanh(up / UP_CAP))


@pytest.mark.parametrize("inter", [1152, 2304])
def test_swiglu_matches_situ_reference(inter):
    """The softcapped SwiGLU of the local rows times their routing weight,
    rounded once to bf16; non-local rows (-1) are skipped."""
    torch.manual_seed(0)
    rows = 64
    gate_up = (torch.randn(rows, 2 * inter, device=DEVICE) * 8).bfloat16()
    gammas = torch.rand(rows, device=DEVICE) + 0.5
    src2dst = torch.arange(rows, device=DEVICE, dtype=torch.int32)
    src2dst[::5] = -1
    out = swiglu(gate_up, gammas, src2dst, GATE_CAP, UP_CAP)
    local = src2dst >= 0
    ref = (_situ(gate_up) * gammas[:, None]).bfloat16()
    torch.testing.assert_close(out[local], ref[local], atol=0, rtol=1e-2)


def random_nvfp4(experts: int, n: int, k: int):
    """Packed e2m1 values `[E, N, K / 2]`, e4m3 block scales `[E, N, K / 16]`
    and per-expert global scales `[E]`, for a GEMM output of about unit size
    on unit-variance activations."""
    values = torch.randint(0, 256, (experts, n, k // 2), device=DEVICE)
    scales = torch.rand(experts, n, k // 16, device=DEVICE) + 0.5
    global_scale = (torch.rand(experts, device=DEVICE) + 0.5) / (3 * k**0.5)
    return values.to(torch.uint8), scales.to(torch.float8_e4m3fn), global_scale


def _launch_nvfp4(
    a,
    values,
    scales,
    global_scale,
    alignment: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    top_k,
    config,
    split,
):
    """The `[M * top_k, N]` rows of the output, NaN where the kernel writes
    none."""
    out = torch.full(
        (a.shape[0], top_k, values.shape[1]),
        float("nan"),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    nvfp4_moe_w4a16(
        a,
        values,
        scales,
        global_scale,
        out,
        *alignment,
        top_k,
        config,
        split_dots=split,
    )
    return out.view(-1, values.shape[1])


def _check_nvfp4_rows(out, a, values, scales, global_scale, ids, a_div, split):
    """Each listed row is its activation row times its expert's dequantized
    weights; the rows of -1 slots are not written."""
    flat = ids.flatten()
    assert out[flat < 0].isnan().all()
    for expert in flat[flat >= 0].unique().tolist():
        (rows,) = (flat == expert).nonzero(as_tuple=True)
        w = dequant_nvfp4(values[expert], scales[expert]) * global_scale[expert]
        ref = a[rows // a_div].float() @ w.T
        if split:
            # Exact bf16 weights: only the fp32 summation order differs.
            torch.testing.assert_close(out[rows].float(), ref, atol=1e-3, rtol=2**-8)
        else:
            # Weights times the global scale, rounded to bf16 (2^-9 relative
            # per weight): the absolute error grows with the column's
            # weights, so the bound scales with each column's max |ref|.
            column = ref.abs().amax(0, keepdim=True).clamp_min(1.0)
            err = (out[rows].float() - ref).abs()
            assert (err <= 1e-2 * column + 2e-2 * ref.abs()).all(), err.max()


def _nvfp4_problem(
    n, k, block_m, a_div, experts=24, tokens=40
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Random weights, activations and the row alignment of a routing with
    several row blocks per expert, a one-row expert, empty experts and -1
    slots."""
    torch.manual_seed(0)
    values, scales, global_scale = random_nvfp4(experts, n, k)
    ids = torch.randint(0, experts - 3, (tokens, TOPK), dtype=torch.int32)
    ids[ids == 2] = -1
    ids[0, 0] = experts - 1
    ids = ids.to(DEVICE)
    a = torch.randn(tokens * TOPK // a_div, k, device=DEVICE).bfloat16()
    sorted_ids, expert_ids, num_post_padded = moe_align_block_size(
        ids, block_m, experts, ignore_invalid_experts=True
    )[:3]
    return (
        a,
        (values, scales, global_scale),
        ids,
        (sorted_ids, expert_ids, num_post_padded),
    )


# (N, K, BLOCK_SIZE_M, BLOCK_SIZE_K, rows of A per routed slot): the w13 (one
# row per token) and w2 (one per slot) GEMMs of Mini and Small; Mini w2 with
# BLOCK_SIZE_K 256 has a K tail.
NVFP4_GEMMS = [
    (2304, 2304, 16, 128, TOPK),
    (2304, 1152, 32, 256, 1),
    (4608, 4608, 16, 128, TOPK),
    (4608, 2304, 64, 64, 1),
]
NVFP4_CONFIG = {"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}


@pytest.mark.parametrize("split", [True, False])
@pytest.mark.parametrize("n,k,block_m,block_k,a_div", NVFP4_GEMMS)
def test_nvfp4_kernel_matches_dequantized_matmul(n, k, block_m, block_k, a_div, split):
    """Both K-loop paths (default and split-dots) of the expert GEMM."""
    a, weights, ids, alignment = _nvfp4_problem(n, k, block_m, a_div)
    config = {**NVFP4_CONFIG, "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_K": block_k}
    out = _launch_nvfp4(a, *weights, alignment, a_div, config, split)
    _check_nvfp4_rows(out, a, *weights, ids, a_div, split)


@pytest.mark.parametrize("split", [True, False])
def test_nvfp4_kernel_every_tile(kernel_env, split):
    """Each tuned weight tile and warp count of a path, launched as the
    fixed config (`VLLM_BATCH_INVARIANT=1`: no autotuning), at Mini's w13
    shape."""
    kernel_env(VLLM_BATCH_INVARIANT="1")
    tiles = (
        nvfp4_moe._SPLIT_DOTS_TUNED_TILES if split else nvfp4_moe._DEFAULT_TUNED_TILES
    )
    a, weights, ids, alignment = _nvfp4_problem(2304, 2304, 16, TOPK)
    for block_n, block_k, warps, stages in tiles:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": 1,
            "num_warps": warps,
            "num_stages": stages,
        }
        out = _launch_nvfp4(a, *weights, alignment, TOPK, config, split)
        _check_nvfp4_rows(out, a, *weights, ids, TOPK, split)


@requires_first_config
@pytest.mark.parametrize("split", [True, False])
@pytest.mark.parametrize("n,k,block_m,block_k,a_div", NVFP4_GEMMS[::2])
def test_nvfp4_descriptor_path_matches_pointer_path_bitwise(
    n, k, block_m, block_k, a_div, split
):
    """The tensor-descriptor weight loads give the pointer loads' bits at
    the same tile."""
    a, weights, _, alignment = _nvfp4_problem(n, k, block_m, a_div)
    assert use_nvfp4_moe_td(weights[0], True)
    config = {**NVFP4_CONFIG, "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_K": block_k}
    outs = []
    for use_td in (True, False):
        out = torch.full(
            (a.shape[0], a_div, n), float("nan"), dtype=torch.bfloat16, device=DEVICE
        )
        nvfp4_moe_w4a16(
            a, *weights, out, *alignment, a_div, config, split, use_td=use_td
        )
        outs.append(out)
    assert torch.equal(outs[0].isnan(), outs[1].isnan())
    assert torch.equal(outs[0].nan_to_num(), outs[1].nan_to_num())


@pytest.mark.parametrize("split", [True, False])
@pytest.mark.parametrize("n,k,block_m", [(2304, 2304, 16), (4608, 4608, 32)])
def test_nvfp4_kernel_every_finite_scale_code(n, k, block_m, split):
    """Block scales of every finite e4m3 code (subnormal, the largest,
    negative and zero), one code per output column."""
    torch.manual_seed(0)
    experts, tokens = 4, 16
    values, _, global_scale = random_nvfp4(experts, n, k)
    codes = torch.arange(256, dtype=torch.int32)
    codes = codes[(codes & 0x7F) != 0x7F]
    column = codes[torch.arange(n) % codes.numel()].to(torch.uint8)
    scales = column[None, :, None].expand(experts, n, k // 16).contiguous()
    scales = scales.to(DEVICE).view(torch.float8_e4m3fn)
    ids = torch.randint(0, experts, (tokens, TOPK), dtype=torch.int32).to(DEVICE)
    a = torch.randn(tokens * TOPK, k, device=DEVICE).bfloat16()
    alignment = moe_align_block_size(ids, block_m, experts)[:3]
    config = {**NVFP4_CONFIG, "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_K": 128}
    out = _launch_nvfp4(a, values, scales, global_scale, alignment, 1, config, split)
    _check_nvfp4_rows(out, a, values, scales, global_scale, ids, 1, split)


@triton.jit
def _decode_scales_kernel(codes_ptr, out_ptr):
    offs = tl.arange(0, 256)
    tl.store(out_ptr + offs, _block_scales_to_f32(tl.load(codes_ptr + offs)))


def test_nvfp4_block_scale_decode_matches_torch():
    """All 256 e4m3fn codes (uint8) decode as torch does: zeros,
    subnormals, normals up to 448, negatives, NaN for 0x7F / 0xFF."""
    codes = torch.arange(256, device=DEVICE).to(torch.uint8)
    out = torch.empty(256, device=DEVICE)
    _decode_scales_kernel[(1,)](codes, out)
    expected = codes.view(torch.float8_e4m3fn).float()
    assert torch.equal(out.isnan(), expected.isnan())
    finite = ~expected.isnan()
    # Bits, so that the sign of zero counts too.
    assert torch.equal(
        out[finite].view(torch.int32), expected[finite].view(torch.int32)
    )


@pytest.mark.parametrize(
    "tokens,top_k,experts,hidden,block_m",
    [(1, 8, 128, 2304, 16), (40, 8, 16, 4608, 32), (31, 8, 4, 4608, 64)],
)
def test_split_dots_config_row_block(tokens, top_k, experts, hidden, block_m):
    """The split-dots config's row block is about one expert's rows, 16 to
    64, with tiles that hold whole scale groups."""
    assert use_split_dots(tokens, top_k, experts, hidden)
    config = split_dots_config(tokens, top_k, experts)
    assert config["BLOCK_SIZE_M"] == block_m
    assert config["BLOCK_SIZE_K"] % 16 == 0 and config["BLOCK_SIZE_K"] >= 32


@pytest.mark.parametrize(
    "tokens,experts,hidden,td,split",
    [
        (240, 128, 2304, False, True),
        (256, 128, 2304, False, False),
        (300, 128, 2304, True, True),
        (512, 128, 2304, True, False),
        (1536, 384, 4608, False, True),
        (3072, 384, 4608, True, False),
    ],
)
def test_split_dots_below_the_bound(kernel_env, tokens, experts, hidden, td, split):
    """The split-dots path serves batches below the measured bound of their
    shape and load path (Mini 16 rows per expert, 32 with descriptors;
    Small 64), and none under `VLLM_BATCH_INVARIANT`."""
    kernel_env(VLLM_BATCH_INVARIANT="0")
    assert split_dots_bound(2304, False) == 16
    assert split_dots_bound(2304, True) == 32
    assert split_dots_bound(4608, False) == split_dots_bound(4608, True) == 64
    assert use_split_dots(tokens, TOPK, experts, hidden, td) == split
    kernel_env(VLLM_BATCH_INVARIANT="1")
    assert not use_split_dots(tokens, TOPK, experts, hidden, td)


def random_weights(fmt: str, num_experts: int, hidden: int, inter: int):
    """Checkpoint-layout `{w13, w2}: (weight, block scales, global scale)`
    with a GEMM output of about unit size; the NVFP4 global scale is
    `amax / DENOMINATOR`."""
    shapes = {"w13": (2 * inter, hidden), "w2": (hidden, inter)}
    weights = {}
    for name, (n, k) in shapes.items():
        if fmt == "mxfp8":
            values = torch.randn(num_experts, n, k, device=DEVICE)
            base = 127 - round(0.5 * torch.log2(torch.tensor(float(k))).item())
            scales = torch.randint(base - 1, base + 2, (num_experts, n, k // 32))
            weights[name] = (
                values.to(torch.float8_e4m3fn),
                scales.to(torch.uint8).to(DEVICE),
                None,
            )
        else:
            values = torch.randint(0, 256, (num_experts, n, k // 2), device=DEVICE)
            scales = torch.rand(num_experts, n, k // 16, device=DEVICE) + 0.5
            amax = torch.rand(num_experts, device=DEVICE) + 0.5
            amax *= DENOMINATOR / (3 * k**0.5)
            weights[name] = (
                values.to(torch.uint8),
                scales.to(torch.float8_e4m3fn),
                amax / DENOMINATOR,
            )
    return weights


def _clone(weights):
    return {
        name: tuple(None if t is None else t.clone() for t in tensors)
        for name, tensors in weights.items()
    }


def _dequant(fmt: str, tensors: tuple, expert: int) -> torch.Tensor:
    weight, scale, global_scale = tensors
    if fmt == "mxfp8":
        return dequant_mxfp8(weight[expert], scale[expert])
    return dequant_nvfp4(weight[expert], scale[expert]) * global_scale[expert]


def reference(fmt, weights, x, ids, gammas):
    """Per-expert fp32 loop over the dequantized checkpoint `weights`: the
    bf16 gate/up GEMM output, SiTU of the interleaved (gate, up) columns and
    a bf16 down projection. The routing weight scales the SiTU output
    (rounded to bf16 with it) for NVFP4, and the fp32 down-projection rows
    for MXFP8 (the fused MoE combines them)."""
    after_w2 = fmt == "mxfp8"
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert in ids.unique().tolist():
        if expert < 0:
            continue
        token, slot = (ids == expert).nonzero(as_tuple=True)
        w13 = _dequant(fmt, weights["w13"], expert)
        act = _situ((x[token].float() @ w13.T).bfloat16())
        gamma = gammas.view(ids.shape)[token, slot][:, None]
        act = (act if after_w2 else act * gamma).bfloat16()
        down = (act.float() @ _dequant(fmt, weights["w2"], expert).T).bfloat16()
        out.index_add_(0, token, down.float() * gamma if after_w2 else down.float())
    return out.bfloat16()


def routing(tokens: int, num_global: int, num_local: int, rank: int):
    """`select_experts`-style local ids `[T, topk]` (-1: not local) with a
    few empty local experts and an all-padding last token, fp32 routing
    weights `[T * topk]` and the identity `src2dst`."""
    start = rank * num_local
    empty = torch.tensor([start, start + 5, start + num_local - 1])
    scores = torch.rand(tokens, num_global)
    # Each token has a local expert (not one of the empty ones).
    scores[torch.arange(tokens), start + 1 + torch.randint(0, 4, (tokens,))] = 2.0
    scores[:, empty] = -1.0
    ids = scores.topk(TOPK, dim=-1).indices
    local = torch.where((ids >= start) & (ids < start + num_local), ids - start, -1)
    if tokens > 1:
        local[-1] = -1
    local = local.to(torch.int32).to(DEVICE)
    gammas = torch.rand(tokens * TOPK, device=DEVICE) + 0.5
    rows = torch.arange(tokens * TOPK, dtype=torch.int32, device=DEVICE)
    return local, gammas, torch.where(local.flatten() >= 0, rows, -1)


def _build_experts(fmt, weights, hidden, inter):
    return xpu_flat_experts(fmt, TOPK, GATE_CAP, UP_CAP)(
        weights, ((2 * inter, hidden), (hidden, inter)), fmt == "nvfp4"
    )


def _run_experts(experts, x, ids, gammas, src2dst, num_global):
    out = experts(x, ids, gammas, src2dst, num_global, GATE_CAP, UP_CAP)
    if getattr(experts, "combines", False):
        return out
    return combine(out, src2dst, TOPK, torch.bfloat16)


def _assert_matches(out, ref, tokens):
    if tokens > 1:
        assert (out[-1] == 0).all()
    scale = ref.abs().max().item()
    assert scale > 0.1
    torch.testing.assert_close(out, ref, atol=1e-2 * scale, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.uint8, torch.bfloat16])
def test_deinterleave_rows_round_trip(dtype: torch.dtype):
    """Interleaved (gate, up) rows become `[gate | up]` halves, and
    interleaving the halves again gives back the checkpoint rows."""
    torch.manual_seed(0)
    gate, up = torch.randn(2, 6, 64, 96, device=DEVICE).to(dtype)
    checkpoint = torch.stack((gate, up), 2).reshape(6, 128, 96)
    rows = checkpoint.clone()
    deinterleave_rows(rows)
    halves = torch.cat((gate, up), 1)
    assert torch.equal(rows.view(torch.uint8), halves.view(torch.uint8))
    again = torch.stack(rows.chunk(2, 1), 2).reshape(rows.shape)
    assert torch.equal(again.view(torch.uint8), checkpoint.view(torch.uint8))


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("model", list(SHAPES))
@pytest.mark.parametrize("tokens", EXPERT_TOKENS)
def test_xpu_experts_match_reference(fmt: str, model: str, tokens: int):
    """The XPU backend on the local experts (one EP rank for Small) vs. the
    dequantized reference, with non-local slots, empty experts and a token
    routed to no local expert (which comes out 0)."""
    torch.manual_seed(0)
    hidden, inter, num_global, num_local = SHAPES[model]
    weights = random_weights(fmt, num_local, hidden, inter)
    checkpoint = _clone(weights)
    experts = _build_experts(fmt, weights, hidden, inter)
    assert isinstance(experts, XpuMxfp8Experts if fmt == "mxfp8" else XpuNvfp4Experts)
    assert getattr(experts, "combines", False) == (fmt == "mxfp8")
    ids, gammas, src2dst = routing(
        tokens, num_global, num_local, num_local < num_global
    )
    x = torch.randn(tokens, hidden, device=DEVICE).bfloat16()
    out = _run_experts(experts, x, ids, gammas, src2dst, num_global)
    _assert_matches(out, reference(fmt, checkpoint, x, ids, gammas), tokens)


def test_expert_tokens_cover_both_nvfp4_paths():
    """`EXPERT_TOKENS` take both the split-dots and the default path at
    each model's shape."""
    for hidden, _, num_global, _ in SHAPES.values():
        paths = {use_split_dots(t, TOPK, num_global, hidden) for t in EXPERT_TOKENS}
        assert paths == {True, False}, hidden


def test_mxfp8_fused_moe_needs_deinterleaved_rows():
    """Negative control of the MXFP8 layout: the SYCL fused MoE built on the
    checkpoint's interleaved `w13` rows misses the reference by far more than
    the tolerance that `XpuMxfp8Experts` (de-interleaved rows) meets on the
    same weights."""
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

    torch.manual_seed(0)
    tokens = 64
    hidden, inter, num_global, _ = SHAPES["mini"]
    weights = random_weights("mxfp8", num_global, hidden, inter)
    checkpoint = _clone(weights)
    interleaved = _clone(weights)
    ids, gammas, src2dst = routing(tokens, num_global, num_global, 0)
    x = torch.randn(tokens, hidden, device=DEVICE).bfloat16()
    ref = reference("mxfp8", checkpoint, x, ids, gammas)
    experts = _build_experts("mxfp8", weights, hidden, inter)
    _assert_matches(_run_experts(experts, x, ids, gammas, src2dst, num_global), ref, 64)
    (w13, w13_scale, _), (w2, w2_scale, _) = interleaved["w13"], interleaved["w2"]
    moe = XpuFusedMoe(
        w13=w13,
        w13_scales=w13_scale,
        w13_bias=None,
        w2=w2,
        w2_scales=w2_scale,
        w2_bias=None,
        n_experts_per_token=TOPK,
        num_experts=num_global,
        **_situ_activation(GATE_CAP, UP_CAP),
    )
    a, a_scale = x, None
    if experts.quantize_activations:
        a, a_scale = quantize_mxfp8_activations(x)
    wrong = torch.empty_like(x)
    moe.apply(wrong, a, gammas.view(ids.shape), ids, a1q_scale=a_scale)
    error = (wrong[:-1].float() - ref[:-1].float()).abs().max().item()
    assert error > 0.5 * ref.abs().max().item()


def mini_layer(fmt: str):
    """`Inkling2MoE` (TP=1) at Mini shapes with random checkpoint weights
    loaded through its weight loaders, before `process_weights_after_loading`;
    and those weights."""
    hidden, inter, num_experts, _ = SHAPES["mini"]
    config = Inkling2ModelConfig(
        hidden_size=hidden,
        intermediate_size=inter,
        n_routed_experts=num_experts,
        num_experts_per_tok=TOPK,
        mlp_gate_softcap=GATE_CAP,
        mlp_up_softcap=UP_CAP,
    )
    with torch.device(DEVICE):
        moe = Inkling2MoE(config, fmt)
    weights = random_weights(fmt, num_experts, hidden, inter)
    gate, bias = _router(hidden, num_experts)
    loaded = {"gate.weight": gate, "gate.bias": bias}
    for name, (weight, scale, global_scale) in weights.items():
        prefix = f"experts.{name}_weight"
        loaded[prefix], loaded[f"{prefix}_scale"] = weight, scale
        if fmt == "nvfp4":
            loaded[f"{prefix}_amax"] = global_scale * DENOMINATOR
            loaded[f"{prefix}_outer_scale_denominator"] = torch.tensor(
                [DENOMINATOR], device=DEVICE
            )
    for name, param in moe.named_parameters():
        param.weight_loader(param, loaded[name])
    return moe, weights, loaded


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tokens", [5, 64])
def test_xpu_moe_layer_matches_reference(dist_init, fmt: str, tokens: int):
    """`Inkling2MoE` (TP=1) at Mini shapes: `select_experts`, the XPU
    backend and the combine, vs. the reference on the layer's own routing."""
    torch.manual_seed(0)
    moe, weights, _ = mini_layer(fmt)
    moe.experts.process_weights_after_loading()
    x = torch.randn(tokens, SHAPES["mini"][0], device=DEVICE).bfloat16()
    out = moe(x)
    gate = moe.gate
    ids, gammas, _ = select_experts(
        x, gate.weight, gate.bias, TOPK, gate.route_scale, moe.experts.expert_bounds
    )
    ref = reference(fmt, weights, x, ids, gammas)
    scale = ref.abs().max().item()
    assert scale > 0.1
    torch.testing.assert_close(out, ref, atol=1e-2 * scale, rtol=2e-2)


@pytest.mark.parametrize("fmt", FORMATS)
def test_xpu_experts_keep_kernel_weights_as_parameters(dist_init, fmt: str):
    """After `process_weights_after_loading` the backend computes on the
    layer's Parameters (MXFP8: rewritten into the kernel layout, with the
    library's rebuild mark on `w13`; NVFP4: the checkpoint layout as
    loaded). A reload is refused by the weight loaders before they write
    anything, and a second processing call fails loudly."""
    torch.manual_seed(0)
    moe, weights, loaded = mini_layer(fmt)
    experts = moe.experts
    experts.process_weights_after_loading()
    params = dict(experts.named_parameters())
    backend = experts.flat_experts
    if fmt == "mxfp8":
        assert isinstance(backend, XpuMxfp8Experts)
        assert backend.moe.w13 is params["w13_weight"]
        assert backend.moe.w2 is params["w2_weight"]
        assert getattr(params["w13_weight"], "xpu_fused_moe", False)
    else:
        assert isinstance(backend, XpuNvfp4Experts)
        for name in ("w13", "w2"):
            weight, scale, _ = backend.weights[name]
            assert weight is params[f"{name}_weight"]
            assert scale is params[f"{name}_weight_scale"]
            assert torch.equal(weight, weights[name][0])
    kernel_bytes = {name: p.detach().clone() for name, p in params.items()}
    for name, param in params.items():
        with pytest.raises(RuntimeError, match="reloading"):
            param.weight_loader(param, loaded[f"experts.{name}"])
    for name, param in experts.named_parameters():
        assert torch.equal(
            param.view(torch.uint8), kernel_bytes[name].view(torch.uint8)
        )
    with pytest.raises(RuntimeError, match="already called"):
        experts.process_weights_after_loading()


def test_mxfp8_build_memory_peak_is_documented():
    """The load-time extra memory of `XpuMxfp8Experts` is what its docstring
    states: about one copy of the local `w13` and `w2` weights."""
    torch.manual_seed(0)
    hidden, inter, _, num_local = SHAPES["mini"]
    weights = random_weights("mxfp8", num_local, hidden, inter)
    layer = sum(t.numel() * t.element_size() for w in weights.values() for t in w[:2])
    w13_bytes = weights["w13"][0].numel()
    torch.xpu.synchronize()
    torch.xpu.reset_peak_memory_stats()
    before = torch.xpu.memory_allocated()
    experts = _build_experts("mxfp8", weights, hidden, inter)
    torch.xpu.synchronize()
    extra = torch.xpu.max_memory_allocated() - before
    assert w13_bytes < extra <= 1.05 * layer + 2**20, (extra, layer)
    del experts


@pytest.mark.parametrize("force_xe3p", [None, "1"])
def test_mxfp8_activation_decision_follows_the_fused_moe_kernel(
    monkeypatch, force_xe3p
):
    """MXFP8 activations are quantized exactly when the built `XpuFusedMoe`
    runs its xe_3 grouped GEMM (`not _uses_xe2`), which `FORCE_XE3P_KERNEL`
    also selects; only the build runs (no GEMM)."""
    if force_xe3p is None:
        monkeypatch.delenv("FORCE_XE3P_KERNEL", raising=False)
    else:
        monkeypatch.setenv("FORCE_XE3P_KERNEL", force_xe3p)
    hidden, inter, _, _ = SHAPES["mini"]
    experts = _build_experts(
        "mxfp8", random_weights("mxfp8", 8, hidden, inter), hidden, inter
    )
    assert isinstance(experts, XpuMxfp8Experts)
    assert experts.quantize_activations == (not experts.moe._uses_xe2)
    if force_xe3p:
        assert experts.quantize_activations
    for fmt, cls in (("mxfp8", XpuMxfp8Experts), ("nvfp4", XpuNvfp4Experts)):
        factory = xpu_flat_experts(fmt, TOPK, GATE_CAP, UP_CAP)
        assert isinstance(factory, functools.partial) and factory.func is cls


@pytest.mark.parametrize("model", list(SHAPES))
def test_nvfp4_experts_batch_invariant(kernel_env, model: str):
    """Under `VLLM_BATCH_INVARIANT` a token's NVFP4 expert rows are bitwise
    the same alone (1 token, the split-dots path otherwise) and in a batch
    of 600 tokens (the default path)."""
    kernel_env(VLLM_BATCH_INVARIANT="1")
    torch.manual_seed(0)
    hidden, inter, num_global, num_local = SHAPES[model]
    experts = _build_experts(
        "nvfp4", random_weights("nvfp4", num_local, hidden, inter), hidden, inter
    )
    ids, gammas, src2dst = routing(600, num_global, num_local, num_local < num_global)
    x = torch.randn(600, hidden, device=DEVICE).bfloat16()
    batch = experts(x, ids, gammas, src2dst, num_global, GATE_CAP, UP_CAP)
    alone = experts(
        x[:1], ids[:1], gammas[:TOPK], src2dst[:TOPK], num_global, GATE_CAP, UP_CAP
    )
    local = ids[0] >= 0
    assert local.any()
    assert torch.equal(alone[:TOPK][local], batch[:TOPK][local])


REQUIREMENT_ERROR = (
    r"vllm-xpu-kernels >= 0\.1\.17 \(XpuFusedMoe with SiTU and _uses_xe2\)"
)


class _FusedMoeWithSitu:
    def __init__(
        self, w13, *, activation_situ_beta=None, activation_situ_linear_beta=None
    ):
        pass


class _FusedMoeWithoutSitu:
    def __init__(self, w13, *, activation=None):
        pass


@pytest.mark.parametrize(
    "fused_moe,has_xe2_query,supported",
    [
        (_FusedMoeWithSitu, True, True),
        # 0.1.13-0.1.16: SiTU, but no `_uses_xe2`.
        (_FusedMoeWithSitu, False, False),
        (_FusedMoeWithoutSitu, True, False),
    ],
)
def test_fused_moe_support_check(fused_moe, has_xe2_query, supported):
    """A vllm-xpu-kernels fused-MoE interface without `_uses_xe2` (from
    `_uses_xe2_grouped_gemm`, added in 0.1.17) or without the SiTU betas
    (since 0.1.13) fails at model init with an error naming the requirement
    (stand-in modules of the external package); a complete one passes, as
    does the installed release."""
    from vllm_xpu_kernels import fused_moe_interface

    check_fused_moe_support(fused_moe_interface)
    module: Any = types.ModuleType("fused_moe_interface")
    module.XpuFusedMoe = fused_moe
    if has_xe2_query:
        module._uses_xe2_grouped_gemm = lambda weight: True
    if supported:
        check_fused_moe_support(module)
        return
    with pytest.raises(NotImplementedError, match=REQUIREMENT_ERROR):
        check_fused_moe_support(module)
