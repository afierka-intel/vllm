# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 XPU routed-expert backends vs. dequantized torch references, at
the expert shapes of Inkling 2 Mini and Small."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tensor_descriptor_compatible, tl, triton

if not current_platform.is_xpu():
    pytest.skip("Inkling2 XPU expert backends", allow_module_level=True)

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.models.inkling2.common.moe import (  # noqa: E402
    Inkling2MoE,
    flat_experts_class,
)
from vllm.models.inkling2.common.ops import moe as moe_ops  # noqa: E402
from vllm.models.inkling2.common.ops.linear import linear_k_groups  # noqa: E402
from vllm.models.inkling2.common.ops.moe import select_experts  # noqa: E402
from vllm.models.inkling2.common.ops.nvfp4_w4a16_moe import (  # noqa: E402
    _e4m3_to_f32,
    nvfp4_w4a16_moe,
)
from vllm.models.inkling2.configs import Inkling2ModelConfig  # noqa: E402
from vllm.models.inkling2.xpu.experts import (  # noqa: E402
    XpuMxfp8Experts,
    XpuNvfp4Experts,
    deinterleave_rows,
)

DEVICE = "xpu"
GATE_CAP, UP_CAP, TOPK = 4.0, 25.0, 8
# hidden, intermediate, experts, local experts (EP rank 1 of 4 for Small).
SHAPES = {"mini": (2304, 1152, 128, 128), "small": (4608, 2304, 384, 96)}
FORMATS = ["mxfp8", "nvfp4"]
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
# NVFP4 outer scale denominator of the checkpoints' quantizer (test_moe.py).
DENOMINATOR = 2688.0


def dequant_nvfp4(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(E2M1, device=values.device)
    v = torch.stack([lut[(values & 0xF).long()], lut[(values >> 4).long()]], -1)
    return v.flatten(-2) * scales.float().repeat_interleave(16, -1)


def random_nvfp4(*shape: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed values, e4m3 block scales and the weight amax per expert, with
    a GEMM output of about unit size."""
    values = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), device=DEVICE)
    scales = torch.rand(*shape[:-1], shape[-1] // 16, device=DEVICE) + 0.5
    amax = torch.rand(shape[0], device=DEVICE) + 0.5
    amax *= DENOMINATOR / (3 * shape[-1] ** 0.5)
    return values.to(torch.uint8), scales.to(torch.float8_e4m3fn), amax


def dequant_mxfp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    exp2 = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    return values.float() * exp2


def random_mxfp8(*shape: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Values and scales with a GEMM output of about unit size."""
    values = torch.randn(shape, device=DEVICE).to(torch.float8_e4m3fn)
    base = 127 - round(0.5 * torch.log2(torch.tensor(float(shape[-1]))).item())
    scales = torch.randint(base - 1, base + 2, (*shape[:-1], shape[-1] // 32))
    return values, scales.to(torch.uint8).to(DEVICE)


def random_weights(fmt: str, num_experts: int, hidden: int, inter: int):
    """Checkpoint-layout `{w13, w2}: (weight, block scales, global scale)`;
    the NVFP4 global scale is `amax / DENOMINATOR`."""
    shapes = {"w13": (2 * inter, hidden), "w2": (hidden, inter)}
    if fmt == "mxfp8":
        return {
            name: (*random_mxfp8(num_experts, *shape), None)
            for name, shape in shapes.items()
        }
    weights = {}
    for name, shape in shapes.items():
        values, scales, amax = random_nvfp4(num_experts, *shape)
        weights[name] = (values, scales, amax / DENOMINATOR)
    return weights


def dequant(fmt: str, tensors: tuple, expert: int) -> torch.Tensor:
    """fp32 weights `[N, K]` of `expert` from `(weight, scales, global)`."""
    weight, scale, global_scale = tensors
    if fmt == "mxfp8":
        return dequant_mxfp8(weight[expert], scale[expert])
    return dequant_nvfp4(weight[expert], scale[expert]) * global_scale[expert]


def reference(fmt, weights, x, ids, gammas, gamma_after_w2: bool):
    """Per-expert fp32 loop over the dequantized checkpoint `weights`: the
    bf16 gate/up GEMM output, SiTU of the interleaved (gate, up) columns and
    a bf16 down projection, with the routing weight before the down
    projection (rounded to bf16 with the SiTU) or after it (in fp32)."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert in ids.unique().tolist():
        if expert < 0:
            continue
        token, slot = (ids == expert).nonzero(as_tuple=True)
        w13 = dequant(fmt, weights["w13"], expert)
        gate_up = (x[token].float() @ w13.T).bfloat16().float()
        gate, up = gate_up[:, 0::2], gate_up[:, 1::2]
        act = (
            GATE_CAP
            * torch.tanh(gate / GATE_CAP)
            * torch.sigmoid(gate)
            * UP_CAP
            * torch.tanh(up / UP_CAP)
        )
        gamma = gammas.view(ids.shape)[token, slot][:, None]
        act = act if gamma_after_w2 else act * gamma
        w2 = dequant(fmt, weights["w2"], expert)
        down = (act.bfloat16().float() @ w2.T).bfloat16().float()
        out.index_add_(0, token, down * gamma if gamma_after_w2 else down)
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
    src2dst = torch.where(local.flatten() >= 0, rows, -1)
    return local, gammas, src2dst


def combined(experts, out: torch.Tensor, src2dst: torch.Tensor) -> torch.Tensor:
    """The combined `[T, H]` output of a `FlatExperts` call."""
    if getattr(experts, "combines", False):
        return out
    from vllm.models.inkling2.common.ops.moe import combine

    return combine(out, src2dst, TOPK, torch.bfloat16)


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
@pytest.mark.parametrize("tokens", [1, 7, 64, 300])
def test_xpu_experts_match_reference(fmt: str, model: str, tokens: int):
    """The XPU backend of `flat_experts_class` on local experts (one EP
    rank for Small) vs. the dequantized reference, with non-local slots,
    empty experts and a token routed to no local expert (which comes out
    0)."""
    torch.manual_seed(0)
    hidden, inter, num_global, num_local = SHAPES[model]
    weights = random_weights(fmt, num_local, hidden, inter)
    checkpoint = {
        name: tuple(None if t is None else t.clone() for t in tensors)
        for name, tensors in weights.items()
    }
    experts = flat_experts_class(fmt, TOPK, GATE_CAP, UP_CAP)(
        weights, ((2 * inter, hidden), (hidden, inter)), fmt == "nvfp4"
    )
    assert isinstance(experts, XpuMxfp8Experts if fmt == "mxfp8" else XpuNvfp4Experts)
    rank = 0 if num_local == num_global else 1
    ids, gammas, src2dst = routing(tokens, num_global, num_local, rank)
    x = torch.randn(tokens, hidden, device=DEVICE).bfloat16()
    out = experts(x, ids, gammas, src2dst, num_global, GATE_CAP, UP_CAP)
    out = combined(experts, out, src2dst)
    gamma_after_w2 = getattr(experts, "combines", False)
    ref = reference(fmt, checkpoint, x, ids, gammas, gamma_after_w2)
    if tokens > 1:
        assert (out[-1] == 0).all()
    scale = ref.abs().max().item()
    assert scale > 0.1
    torch.testing.assert_close(out, ref, atol=1e-2 * scale, rtol=2e-2)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tokens", [5, 64])
def test_xpu_moe_layer_matches_reference(dist_init, fmt: str, tokens: int):
    """`Inkling2MoE` (TP=1) at Mini shapes: `select_experts`, the XPU
    backend and the combine, vs. the reference on the layer's own routing."""
    torch.manual_seed(0)
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
    loaded = {
        "gate.weight": (torch.randn(num_experts, hidden, device=DEVICE) * 0.02),
        "gate.bias": torch.randn(num_experts, device=DEVICE) * 0.01,
    }
    loaded["gate.weight"] = loaded["gate.weight"].bfloat16()
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
    moe.experts.process_weights_after_loading()
    x = torch.randn(tokens, hidden, device=DEVICE).bfloat16()
    out = moe(x)

    gate = moe.gate
    ids, gammas, _ = select_experts(
        x, gate.weight, gate.bias, TOPK, gate.route_scale, moe.experts.expert_bounds
    )
    gamma_after_w2 = getattr(moe.experts.flat_experts, "combines", False)
    ref = reference(fmt, weights, x, ids, gammas, gamma_after_w2)
    scale = ref.abs().max().item()
    assert scale > 0.1
    torch.testing.assert_close(out, ref, atol=1e-2 * scale, rtol=2e-2)


@pytest.mark.parametrize("use_td", [False, True])
@pytest.mark.parametrize("a_div", [1, TOPK])
@pytest.mark.parametrize(
    "n,k,block_m",
    # Mini down projection (K % 256 != 0) and Small gate/up, every row block.
    [(2304, 1152, 16), (2304, 1152, 64), (4608, 4608, 16), (4608, 4608, 32)],
)
def test_nvfp4_w4a16_moe_matches_dequantized_matmul(
    n: int, k: int, block_m: int, a_div: int, use_td: bool
):
    """Each row the alignment lists (not the -1 slots, none of the empty
    experts) is its activation row times its expert's dequantized weights;
    rows of several row blocks per expert and of one-row experts. Weight
    tiles load by pointers and through a tensor descriptor."""
    torch.manual_seed(0)
    num_experts, tokens = 24, 40
    values, scales, amax = random_nvfp4(num_experts, n, k)
    global_scale = amax / DENOMINATOR
    ids = torch.randint(0, num_experts - 3, (tokens, TOPK), dtype=torch.int32)
    ids[ids == 2] = -1
    ids[0, 0] = num_experts - 1  # a one-row expert
    ids = ids.to(DEVICE)
    rows = tokens * TOPK
    a = torch.randn(rows // a_div, k, device=DEVICE).bfloat16()
    alignment = moe_align_block_size(
        ids, block_m, num_experts, ignore_invalid_experts=True
    )[:3]
    # The descriptor path is taken when asked for.
    assert tensor_descriptor_compatible(values)
    out = nvfp4_w4a16_moe(
        a, values, scales, global_scale, alignment, rows, a_div, block_m, use_td
    )
    flat = ids.flatten()
    for expert in range(num_experts):
        (r,) = (flat == expert).nonzero(as_tuple=True)
        w = dequant_nvfp4(values[expert], scales[expert]) * global_scale[expert]
        ref = a[r // a_div].float() @ w.T
        # Only the fp32 summation order differs; one bf16 rounding.
        torch.testing.assert_close(out[r].float(), ref, atol=1e-3, rtol=2**-8)


@triton.jit
def _decode_e4m3_kernel(Codes, Out):
    offs = tl.arange(0, 256)
    tl.store(Out + offs, _e4m3_to_f32(tl.load(Codes + offs)))


def test_nvfp4_e4m3_scale_decode_matches_torch():
    """All 256 e4m3fn codes decode as torch does: zeros, subnormals, normals
    up to 448, negatives, and NaN for 0x7F / 0xFF."""
    codes = torch.arange(256, device=DEVICE).to(torch.uint8)
    out = torch.empty(256, device=DEVICE)
    _decode_e4m3_kernel[(1,)](codes, out)
    expected = codes.view(torch.float8_e4m3fn).float()
    assert torch.equal(out.isnan(), expected.isnan())
    assert expected.isnan().sum() == 2
    finite = ~expected.isnan()
    # Bits, so that the sign of zero counts too.
    assert torch.equal(
        out[finite].view(torch.int32), expected[finite].view(torch.int32)
    )


@pytest.mark.parametrize("use_td", [False, True])
@pytest.mark.parametrize("n,k,block_m", [(2304, 1152, 16), (4608, 4608, 32)])
def test_nvfp4_w4a16_moe_every_finite_scale_code(
    n: int, k: int, block_m: int, use_td: bool
):
    """Block scales of every finite e4m3 code (subnormal, the largest,
    negative and zero), one code per output column, against the dequantized
    matmul (same tolerance as the random-scale test)."""
    torch.manual_seed(0)
    num_experts, tokens = 4, 16
    values, _, amax = random_nvfp4(num_experts, n, k)
    global_scale = amax / DENOMINATOR
    codes = torch.arange(256, dtype=torch.int32)
    codes = codes[(codes & 0x7F) != 0x7F]
    column = codes[torch.arange(n) % codes.numel()].to(torch.uint8)
    scales = column[None, :, None].expand(num_experts, n, k // 16).contiguous()
    scales = scales.to(DEVICE).view(torch.float8_e4m3fn)
    assert not scales.float().isnan().any()
    ids = torch.randint(0, num_experts, (tokens, TOPK), dtype=torch.int32)
    ids = ids.to(DEVICE)
    rows = tokens * TOPK
    a = torch.randn(rows, k, device=DEVICE).bfloat16()
    alignment = moe_align_block_size(ids, block_m, num_experts)[:3]
    out = nvfp4_w4a16_moe(
        a, values, scales, global_scale, alignment, rows, 1, block_m, use_td
    )
    flat = ids.flatten()
    for expert in range(num_experts):
        (r,) = (flat == expert).nonzero(as_tuple=True)
        w = dequant_nvfp4(values[expert], scales[expert]) * global_scale[expert]
        ref = a[r].float() @ w.T
        torch.testing.assert_close(out[r].float(), ref, atol=1e-3, rtol=2**-8)


@pytest.mark.parametrize("tokens", [5, 300])
def test_select_experts_every_tuned_config(tokens: int, monkeypatch):
    """Each autotuned warp count of the router top-k (small and large
    batches) gives torch's experts and routing weights."""
    torch.manual_seed(0)
    hidden, num_experts, route_scale = 2304, 128, 8.0
    x = torch.randn(tokens, hidden, device=DEVICE).bfloat16()
    gate = torch.randn(num_experts, hidden, device=DEVICE).bfloat16() * 0.02
    bias = torch.randn(num_experts, device=DEVICE) * 0.01
    bounds = torch.tensor([4, 68], dtype=torch.int32, device=DEVICE)
    logits = linear_k_groups(x, gate)
    _, ref_ids = torch.topk(logits.sigmoid() + bias, TOPK, sorted=True)
    selected = torch.nn.functional.logsigmoid(logits.gather(1, ref_ids))
    ref_weights = selected.softmax(-1) * route_scale
    local = (ref_ids >= 4) & (ref_ids < 68)
    tuner = moe_ops._select_kernel_tuned(tokens <= 256)
    for config in list(tuner.configs):
        # A one-config tuner launches it without benchmarking.
        monkeypatch.setattr(tuner, "configs", [config])
        ids, weights, src2dst = select_experts(x, gate, bias, TOPK, route_scale, bounds)
        torch.testing.assert_close(ids.long(), torch.where(local, ref_ids - 4, -1))
        torch.testing.assert_close(weights.view(tokens, -1), ref_weights)
        rows = torch.arange(tokens * TOPK, device=DEVICE).view(tokens, -1)
        torch.testing.assert_close(
            src2dst.view(tokens, -1).long(), torch.where(local, rows, -1)
        )
