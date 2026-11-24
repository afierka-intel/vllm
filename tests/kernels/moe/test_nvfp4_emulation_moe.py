# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 emulation MoE kernel (`fused_moe_nvfp4_emulation_kernel`): its
launcher's options, and both K-loop paths (default and split-dots) against
the dequantized matmul, over one row of A per routed slot or one per
`top_k` slots."""

import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import break_fp4_bytes
from vllm.model_executor.layers.fused_moe.experts import nvfp4_emulation_moe
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
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


class _LaunchSpy:
    """Stands in for `JITFunction.run`: records the launch kwargs, does not
    launch."""

    def __init__(self):
        self.kwargs: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.kwargs.append(kwargs)


def _launch(config: dict, **launch) -> None:
    """Launch the kernel over random-shaped operands (on the CPU: the spy
    only records the launch)."""
    # Pointer loads unless a test asks for the TD path (XPU).
    launch.setdefault("use_td", False)
    experts, n, k, tokens, top_k = 4, 128, 256, 4, 2
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
        **launch,
    )


# A tuned-file style entry: tiles plus the launch config of fused_moe_kernel.
CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 8,
    "num_stages": 3,
}
# The launch kwargs of `main`'s launcher besides the operands and strides.
MAIN_KWARGS = {
    "block_k_diviable",
    "MUL_ROUTED_WEIGHT",
    "top_k",
    "compute_type",
    "group_size",
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
}


@pytest.fixture
def spy(monkeypatch):
    spy = _LaunchSpy()
    monkeypatch.setattr(
        nvfp4_emulation_moe.fused_moe_nvfp4_emulation_kernel, "run", spy
    )
    return spy


@pytest.mark.parametrize(
    "launch_options",
    [
        {"num_warps": 8, "num_stages": 3},
        {"num_warps": 2, "num_stages": 4, "num_ctas": 1},
        {"num_warps": 16, "num_stages": 2, "grf_mode": "256"},
    ],
)
def test_caller_launch_options_reach_the_launch(spy, launch_options):
    """`launch_options` reach the kernel launch as they are, whatever the
    config holds (fails before the option existed: unexpected keyword)."""
    _launch({**CONFIG, "num_warps": 4}, launch_options=launch_options)
    (kwargs,) = spy.kwargs
    for name, value in launch_options.items():
        assert kwargs[name] == value
    assert "SPLIT_K" not in kwargs


@pytest.mark.parametrize("launch_options", [None, {}])
def test_default_launch_keeps_triton_defaults(spy, launch_options):
    """Without caller options the launch kwargs are the ones the launcher
    has always passed: the config's `num_warps` / `num_stages` (tuned for
    `fused_moe_kernel`) are not forwarded, so Triton's defaults apply."""
    launch = {} if launch_options is None else {"launch_options": launch_options}
    _launch(CONFIG, **launch)
    (kwargs,) = spy.kwargs
    # `grid` / `warmup` come from `JITFunction.__getitem__`; SPLIT_DOTS and
    # USE_TD are constexprs of this PR's paths, off here (pointer loads, the
    # default path compiles as before).
    assert set(kwargs) - {"grid", "warmup"} == MAIN_KWARGS | {"SPLIT_DOTS", "USE_TD"}
    assert kwargs["SPLIT_DOTS"] is False and kwargs["USE_TD"] is False


@pytest.mark.parametrize("block_k", [16, 24])
def test_split_dots_needs_whole_groups_and_a_16_wide_dot(monkeypatch, block_k):
    """The split-dots path multiplies [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]
    tiles: BLOCK_SIZE_K must hold whole 16-element scale groups and give
    `tl.dot` an inner dimension of at least 16."""
    monkeypatch.setattr(
        nvfp4_emulation_moe.fused_moe_nvfp4_emulation_kernel, "run", _LaunchSpy()
    )
    with pytest.raises(AssertionError):
        _launch({**CONFIG, "BLOCK_SIZE_K": block_k}, split_dots=True)


@pytest.mark.parametrize("use_td", [None, False, True])
def test_launch_td_on_xpu_only(monkeypatch, use_td):
    """The weight tiles load through a tensor descriptor on XPU unless
    turned off (`use_tensor_descriptor`), and never elsewhere (also not
    when forced with `VLLM_TRITON_USE_TD=1`)."""
    monkeypatch.delenv("VLLM_TRITON_USE_TD", raising=False)
    spy = _LaunchSpy()
    monkeypatch.setattr(
        nvfp4_emulation_moe.fused_moe_nvfp4_emulation_kernel, "run", spy
    )
    monkeypatch.setattr(nvfp4_emulation_moe, "set_triton_allocator", lambda d: None)
    _launch(CONFIG, use_td=use_td)
    (kwargs,) = spy.kwargs
    assert kwargs["USE_TD"] == (current_platform.is_xpu() and use_td is not False)


def test_launch_td_off_with_a_k_tail(monkeypatch):
    """No tensor descriptor when K is not a multiple of BLOCK_SIZE_K (the
    known Triton TD + tl.dot miscompilation that fused_moe_kernel avoids
    the same way)."""
    spy = _LaunchSpy()
    monkeypatch.setattr(
        nvfp4_emulation_moe.fused_moe_nvfp4_emulation_kernel, "run", spy
    )
    monkeypatch.setattr(nvfp4_emulation_moe, "set_triton_allocator", lambda d: None)
    # K = 256 in `_launch`.
    _launch({**CONFIG, "BLOCK_SIZE_K": 96}, use_td=True)
    (kwargs,) = spy.kwargs
    assert kwargs["USE_TD"] is False


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
    block = scales.float().repeat_interleave(16, -1)
    return break_fp4_bytes(values, torch.float32) * block * global_scale


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
            # relative per weight): the absolute error grows with the
            # column's weights (its block scales), so the bound scales with
            # each output column's max |ref|.
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


@pytest.mark.skipif(not current_platform.is_xpu(), reason="TD path is XPU-only")
@pytest.mark.parametrize("split_dots", [True, False])
@pytest.mark.parametrize(
    "n,k,block_m,block_k", [(2304, 1152, 16, 128), (4608, 4608, 32, 128)]
)
def test_td_matches_pointer_path_bitwise(n, k, block_m, block_k, split_dots):
    """The tensor-descriptor weight loads give bitwise the same output as
    the pointer loads (same config, no K tail)."""
    torch.manual_seed(0)
    experts, tokens, top_k = 24, 40, 8
    values, scales, global_scale = random_nvfp4(experts, n, k)
    ids = torch.randint(0, experts, (tokens, top_k), dtype=torch.int32).to(DEVICE)
    a = torch.randn(tokens * top_k, k, device=DEVICE).bfloat16()
    alignment = moe_align_block_size(ids, block_m, experts)[:3]
    config = {**CONFIG, "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_K": block_k}
    td0, td1 = (
        launch(
            a, values, scales, global_scale, alignment, 1, config, split_dots, use_td
        )
        for use_td in (False, True)
    )
    assert torch.equal(td0.view(torch.int16), td1.view(torch.int16))
