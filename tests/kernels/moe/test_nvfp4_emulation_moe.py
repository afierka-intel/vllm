# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 emulation MoE kernel (`fused_moe_nvfp4_emulation_kernel`): its
launcher, both K-loop paths against the dequantized matmul, and the W4A16
mode of `Nvfp4QuantizationEmulationTritonExperts`."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts import nvfp4_emulation_moe
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
    Nvfp4QuantizationEmulationTritonExperts,
    _block_scales_to_f32,
    invoke_fused_moe_nvfp4_emulation_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

requires_gpu = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Triton NVFP4 emulation kernel needs a CUDA, ROCm or XPU device",
)
DEVICE = current_platform.device_type
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


class _LaunchSpy:
    """Stands in for `JITFunction.run`: records the launch kwargs, does not
    launch."""

    def __init__(self):
        self.kwargs: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.kwargs.append(kwargs)


def _launch(
    config: dict, tokens: int = 4, top_k: int = 2, use_td: bool | None = False
) -> None:
    """Launch the kernel over random-shaped operands (on the CPU: the spy
    only records the launch)."""
    experts, n, k = 4, 128, 256
    a = torch.empty(tokens, k, dtype=torch.bfloat16)
    b = torch.empty(experts, n, k // 2, dtype=torch.uint8)
    c = torch.empty(tokens, top_k, n, dtype=torch.bfloat16)
    b_scale = torch.empty(experts, n, k // 16, dtype=torch.uint8)
    w_global_scale = torch.ones(experts)
    sorted_ids = torch.zeros(64, dtype=torch.int32)
    expert_ids = torch.zeros(4, dtype=torch.int32)
    num_post_padded = torch.tensor([64], dtype=torch.int32)
    nvfp4_emulation_moe.invoke_fused_moe_nvfp4_emulation_kernel(
        a,
        b,
        c,
        b_scale,
        None,
        w_global_scale,
        None,
        sorted_ids,
        expert_ids,
        num_post_padded,
        False,
        top_k,
        config,
        compute_type=tl.bfloat16,
        use_td=use_td,
    )


TILES = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}


@pytest.mark.parametrize("num_warps,num_stages", [(8, 3), (4, 2), (2, 4)])
def test_launch_uses_config_warps_and_stages(monkeypatch, num_warps, num_stages):
    """The config's `num_warps` / `num_stages` reach the kernel launch, as
    with `invoke_fused_moe_triton_kernel` (`**config`)."""
    spy = _LaunchSpy()
    monkeypatch.setattr(
        nvfp4_emulation_moe.fused_moe_nvfp4_emulation_kernel, "run", spy
    )
    config = {
        **TILES,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    _launch(config)
    (kwargs,) = spy.kwargs
    assert kwargs["num_warps"] == num_warps
    assert kwargs["num_stages"] == num_stages
    for name, value in TILES.items():
        assert kwargs[name] == value


def test_launch_without_warps_keeps_triton_defaults(monkeypatch):
    """A config without launch options (e.g. a tuned JSON file that omits
    them) passes none, so Triton's defaults apply."""
    spy = _LaunchSpy()
    monkeypatch.setattr(
        nvfp4_emulation_moe.fused_moe_nvfp4_emulation_kernel, "run", spy
    )
    _launch({**TILES, "GROUP_SIZE_M": 1})
    (kwargs,) = spy.kwargs
    assert "num_warps" not in kwargs and "num_stages" not in kwargs


@pytest.mark.parametrize("use_td", [None, False, True])
def test_launch_td_on_xpu_only(monkeypatch, use_td):
    """The weight tiles load through a tensor descriptor on XPU unless
    turned off (`use_tensor_descriptor`), and never elsewhere."""
    monkeypatch.delenv("VLLM_TRITON_USE_TD", raising=False)
    spy = _LaunchSpy()
    monkeypatch.setattr(
        nvfp4_emulation_moe.fused_moe_nvfp4_emulation_kernel, "run", spy
    )
    monkeypatch.setattr(nvfp4_emulation_moe, "set_triton_allocator", lambda d: None)
    _launch({**TILES, "GROUP_SIZE_M": 1}, use_td=use_td)
    (kwargs,) = spy.kwargs
    assert kwargs["USE_TD"] == (current_platform.is_xpu() and use_td is not False)


def random_nvfp4(experts: int, n: int, k: int):
    """Packed e2m1 values `[E, N, K / 2]`, e4m3 block scales `[E, N, K / 16]`
    and per-expert global scales `[E]`, for a GEMM output of about unit size
    on unit-variance activations."""
    values = torch.randint(0, 256, (experts, n, k // 2), device=DEVICE)
    scales = torch.rand(experts, n, k // 16, device=DEVICE) + 0.5
    global_scale = (torch.rand(experts, device=DEVICE) + 0.5) / (3 * k**0.5)
    return values.to(torch.uint8), scales.to(torch.float8_e4m3fn), global_scale


def dequant_nvfp4(values, scales, global_scale) -> torch.Tensor:
    """fp32 weights `[N, K]` of one expert."""
    lut = torch.tensor(E2M1, device=values.device)
    pairs = torch.stack([lut[(values & 0xF).long()], lut[(values >> 4).long()]], -1)
    block = scales.float().repeat_interleave(16, -1)
    return pairs.flatten(-2) * block * global_scale


def launch(
    a, values, scales, global_scale, alignment, top_k, config, split_dots, use_td
):
    """The `[M * top_k, N]` rows of `C`, NaN where the kernel writes none."""
    c = torch.full(
        (a.shape[0], top_k, values.shape[1]),
        float("nan"),
        dtype=torch.bfloat16,
        device=a.device,
    )
    sorted_ids, expert_ids, num_post_padded = alignment
    invoke_fused_moe_nvfp4_emulation_kernel(
        a,
        values,
        c,
        scales.view(torch.uint8),
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
        use_td=use_td,
    )
    return c.view(-1, values.shape[1])


def check_rows(out, a, values, scales, global_scale, ids, a_div, split_dots):
    """Each listed row is its activation row times its expert's dequantized
    weights; the rows of -1 slots are not written."""
    flat = ids.flatten()
    assert out[flat < 0].isnan().all()
    for expert in flat[flat >= 0].unique().tolist():
        (rows,) = (flat == expert).nonzero(as_tuple=True)
        w = dequant_nvfp4(values[expert], scales[expert], global_scale[expert])
        ref = a[rows // a_div].float() @ w.T
        if split_dots:
            # Exact bf16 weights: only the fp32 summation order differs.
            torch.testing.assert_close(out[rows].float(), ref, atol=1e-3, rtol=2**-8)
        else:
            # Weights times the global scale, rounded to bf16 (2^-9
            # relative): the error grows with each column's scale.
            column = ref.abs().amax(0, keepdim=True).clamp_min(1.0)
            err = (out[rows].float() - ref).abs()
            assert (err <= 1e-2 * column + 2e-2 * ref.abs()).all(), err.max()


CONFIG = {"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
# Weight tiles by pointers and (XPU only) through a tensor descriptor.
USE_TD = [False, True] if current_platform.is_xpu() else [None]


@requires_gpu
@pytest.mark.parametrize("use_td", USE_TD)
@pytest.mark.parametrize("split_dots", [True, False])
@pytest.mark.parametrize("a_div", [1, 8])
@pytest.mark.parametrize(
    "n,k,block_m,block_k",
    [
        # Inkling 2 Mini down projection (K % 256 != 0, so BLOCK_SIZE_K=256
        # has a K tail) and Small gate/up, small and large row blocks.
        (2304, 1152, 16, 128),
        (2304, 1152, 64, 256),
        (4608, 4608, 16, 128),
        (4608, 4608, 32, 64),
    ],
)
def test_kernel_matches_dequantized_matmul(
    n, k, block_m, block_k, a_div, split_dots, use_td
):
    """Both K-loop paths over rows of several row blocks per expert, of a
    one-row expert, of empty experts and of -1 slots (left unwritten)."""
    torch.manual_seed(0)
    experts, tokens, top_k = 24, 40, 8
    values, scales, global_scale = random_nvfp4(experts, n, k)
    ids = torch.randint(0, experts - 3, (tokens, top_k), dtype=torch.int32)
    ids[ids == 2] = -1
    ids[0, 0] = experts - 1  # a one-row expert
    ids = ids.to(DEVICE)
    a = torch.randn(tokens * top_k // a_div, k, device=DEVICE).bfloat16()
    alignment = moe_align_block_size(
        ids, block_m, experts, ignore_invalid_experts=True
    )[:3]
    config = {**CONFIG, "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_K": block_k}
    out = launch(
        a, values, scales, global_scale, alignment, a_div, config, split_dots, use_td
    )
    check_rows(out, a, values, scales, global_scale, ids, a_div, split_dots)


@triton.jit
def _decode_scales_kernel(codes_ptr, out_ptr):
    offs = tl.arange(0, 256)
    tl.store(out_ptr + offs, _block_scales_to_f32(tl.load(codes_ptr + offs)))


@requires_gpu
def test_block_scale_decode_matches_torch():
    """All 256 e4m3fn codes (uint8) decode as torch does: zeros,
    subnormals, normals up to 448, negatives, NaN for 0x7F / 0xFF."""
    codes = torch.arange(256, device=DEVICE).to(torch.uint8)
    out = torch.empty(256, device=DEVICE)
    _decode_scales_kernel[(1,)](codes, out)
    expected = codes.view(torch.float8_e4m3fn).float()
    assert torch.equal(out.isnan(), expected.isnan())
    assert expected.isnan().sum() == 2
    finite = ~expected.isnan()
    # Bits, so that the sign of zero counts too.
    assert torch.equal(
        out[finite].view(torch.int32), expected[finite].view(torch.int32)
    )


@requires_gpu
@pytest.mark.parametrize("use_td", USE_TD)
@pytest.mark.parametrize("split_dots", [True, False])
@pytest.mark.parametrize("n,k,block_m", [(2304, 1152, 16), (4608, 4608, 32)])
def test_kernel_every_finite_scale_code(n, k, block_m, split_dots, use_td):
    """Block scales of every finite e4m3 code (subnormal, the largest,
    negative and zero), one code per output column."""
    torch.manual_seed(0)
    experts, tokens, top_k = 4, 16, 8
    values, _, global_scale = random_nvfp4(experts, n, k)
    codes = torch.arange(256, dtype=torch.int32)
    codes = codes[(codes & 0x7F) != 0x7F]
    column = codes[torch.arange(n) % codes.numel()].to(torch.uint8)
    scales = column[None, :, None].expand(experts, n, k // 16).contiguous()
    scales = scales.to(DEVICE).view(torch.float8_e4m3fn)
    ids = torch.randint(0, experts, (tokens, top_k), dtype=torch.int32).to(DEVICE)
    a = torch.randn(tokens * top_k, k, device=DEVICE).bfloat16()
    alignment = moe_align_block_size(ids, block_m, experts)[:3]
    config = {**CONFIG, "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_K": 128}
    out = launch(
        a, values, scales, global_scale, alignment, 1, config, split_dots, use_td
    )
    check_rows(out, a, values, scales, global_scale, ids, 1, split_dots)


def _experts(num_experts, hidden, inter, top_k, weights, quantize_activations):
    (w1, s1, g1), (w2, s2, g2) = weights
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden,
        intermediate_size=inter,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=DEVICE,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=512,
    )
    ones = torch.ones((), device=DEVICE)
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=g1,
        g2_alphas=g2,
        a1_gscale=ones,
        a2_gscale=ones,
        w1_scale=s1.view(torch.uint8),
        w2_scale=s2.view(torch.uint8),
    )
    return Nvfp4QuantizationEmulationTritonExperts(
        moe_config, quant_config, quantize_activations=quantize_activations
    )


def _apply(experts, x, w1, w2, topk_weights, topk_ids, num_experts):
    tokens, top_k = topk_ids.shape
    n, hidden = w1.shape[1], x.shape[1]
    workspace13 = torch.zeros(
        tokens * top_k * max(n, hidden), dtype=x.dtype, device=x.device
    )
    workspace2 = torch.zeros_like(workspace13)
    out = torch.zeros_like(x)
    experts.apply(
        output=out,
        hidden_states=x,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    return out


def _random_moe(num_experts, hidden, inter, top_k, tokens):
    weights = (
        random_nvfp4(num_experts, 2 * inter, hidden),
        random_nvfp4(num_experts, hidden, inter),
    )
    x = torch.randn(tokens, hidden, device=DEVICE).bfloat16()
    topk_weights = torch.rand(tokens, top_k, device=DEVICE).softmax(-1)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=DEVICE)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    return weights, x, topk_weights, topk_ids


@requires_gpu
@pytest.mark.parametrize("tokens", [1, 7, 64, 65, 300])
def test_w4a16_experts_match_reference(tokens):
    """`quantize_activations=False` (W4A16) against an fp32 loop over the
    dequantized experts with bf16 activations, on both sides of the
    split-dots batch bound."""
    torch.manual_seed(0)
    num_experts, hidden, inter, top_k = 16, 512, 256, 4
    weights, x, topk_weights, topk_ids = _random_moe(
        num_experts, hidden, inter, top_k, tokens
    )
    experts = _experts(num_experts, hidden, inter, top_k, weights, False)
    (w1, s1, g1), (w2, s2, g2) = weights
    out = _apply(experts, x, w1, w2, topk_weights, topk_ids, num_experts)

    ref = torch.zeros(tokens, hidden, device=DEVICE)
    for expert in topk_ids.unique().tolist():
        token, slot = (topk_ids == expert).nonzero(as_tuple=True)
        w13 = dequant_nvfp4(w1[expert], s1[expert], g1[expert])
        gate, up = (x[token].float() @ w13.T).bfloat16().float().chunk(2, -1)
        act = (torch.nn.functional.silu(gate) * up).bfloat16().float()
        down = act @ dequant_nvfp4(w2[expert], s2[expert], g2[expert]).T
        gamma = topk_weights[token, slot, None]
        ref.index_add_(0, token, down.bfloat16().float() * gamma)
    scale = ref.abs().max().item()
    torch.testing.assert_close(out.float(), ref, atol=2e-2 * scale, rtol=2e-2)


@requires_gpu
@pytest.mark.parametrize("quantize_activations", [True, False])
def test_activation_quantization_follows_mode(monkeypatch, quantize_activations):
    """The default (W4A4) fake-quantizes both GEMM inputs to NVFP4 and keeps
    the default kernel path; W4A16 fake-quantizes neither and takes the
    split-dots path at a decode batch."""
    calls, launches = [], []

    def spy_quantize(A, A_scale, quant_dtype, per_act_token_quant, **kwargs):
        calls.append(quant_dtype)
        return A, None

    def spy_launch(*args, **kwargs):
        launches.append(kwargs["split_dots"])

    monkeypatch.setattr(nvfp4_emulation_moe, "moe_kernel_quantize_input", spy_quantize)
    monkeypatch.setattr(
        nvfp4_emulation_moe, "invoke_fused_moe_nvfp4_emulation_kernel", spy_launch
    )
    torch.manual_seed(0)
    num_experts, hidden, inter, top_k = 8, 256, 128, 2
    weights, x, topk_weights, topk_ids = _random_moe(
        num_experts, hidden, inter, top_k, 3
    )
    experts = _experts(num_experts, hidden, inter, top_k, weights, quantize_activations)
    w1, w2 = weights[0][0], weights[1][0]
    _apply(experts, x, w1, w2, topk_weights, topk_ids, num_experts)
    assert calls == (["nvfp4", "nvfp4"] if quantize_activations else [])
    assert launches == [not quantize_activations] * 2


@pytest.mark.parametrize(
    "tokens,top_k,experts,block_m",
    [(1, 8, 128, 16), (64, 8, 128, 16), (64, 8, 16, 32), (64, 4, 4, 64)],
)
def test_split_dots_config_row_block(tokens, top_k, experts, block_m):
    """The split-dots config's row block is about one expert's rows, 16 to
    64, with tiles that hold whole scale groups."""
    config = nvfp4_emulation_moe.nvfp4_emulation_split_dots_config(
        tokens, top_k, experts
    )
    assert config["BLOCK_SIZE_M"] == block_m
    assert config["BLOCK_SIZE_K"] % 16 == 0
