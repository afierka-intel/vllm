# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 emulation MoE kernel (`fused_moe_nvfp4_emulation_kernel`): its
launcher, both K-loop paths against the dequantized matmul (bf16 and
scaled-down activations), and the split-dots path's selection and config."""

import pytest
import torch

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
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


class _LaunchSpy:
    """Stands in for `JITFunction.run`: records the launch kwargs, does not
    launch."""

    def __init__(self):
        self.kwargs: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.kwargs.append(kwargs)


def _launch(config: dict, tokens: int = 4, top_k: int = 2) -> None:
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


def launch(a, values, scales, global_scale, alignment, top_k, config, split_dots):
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
            # Weights times the global scale, rounded to bf16.
            torch.testing.assert_close(out[rows].float(), ref, atol=1e-2, rtol=2e-2)


CONFIG = {"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}


@requires_gpu
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
def test_kernel_matches_dequantized_matmul(n, k, block_m, block_k, a_div, split_dots):
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
    out = launch(a, values, scales, global_scale, alignment, a_div, config, split_dots)
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
@pytest.mark.parametrize("split_dots", [True, False])
@pytest.mark.parametrize("n,k,block_m", [(2304, 1152, 16), (4608, 4608, 32)])
def test_kernel_every_finite_scale_code(n, k, block_m, split_dots):
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
    out = launch(a, values, scales, global_scale, alignment, 1, config, split_dots)
    check_rows(out, a, values, scales, global_scale, ids, 1, split_dots)


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


@pytest.mark.parametrize(
    "tokens,top_k,experts,split",
    [
        (1, 8, 128, True),
        (300, 8, 128, True),
        (512, 8, 128, False),
        (1024, 8, 384, True),
    ],
)
def test_split_dots_below_32_rows_per_expert(tokens, top_k, experts, split):
    """The split-dots path serves batches of fewer than 32 rows per expert."""
    assert (
        nvfp4_emulation_moe.nvfp4_emulation_use_split_dots(tokens, top_k, experts)
        == split
    )
