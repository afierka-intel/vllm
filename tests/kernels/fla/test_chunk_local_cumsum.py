# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""chunk_local_cumsum on a scalar (3D) gate, pointer and tensor-descriptor paths."""

from typing import Any

import pytest
import torch

import vllm.third_party.flash_linear_attention.ops.cumsum as cumsum_module
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.triton_utils import tl

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="needs a GPU",
)

DEVICE = current_platform.device_type
BT = 64


def _reference(
    g: torch.Tensor, cu_seqlens: list[int], reverse: bool, head_first: bool
) -> torch.Tensor:
    """fp32 cumsum within each BT-token chunk of each sequence."""
    x = g.float()
    if head_first:
        x = x.transpose(1, 2)
    out = torch.empty_like(x)
    for b in range(x.shape[0]):
        for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            for start in range(bos, eos, BT):
                chunk = x[b, start : min(start + BT, eos)]
                if reverse:
                    chunk = chunk.flip(0)
                y = chunk.cumsum(0)
                out[b, start : start + y.shape[0]] = y.flip(0) if reverse else y
    return out.transpose(1, 2) if head_first else out


def _run(
    g: torch.Tensor,
    reverse: bool,
    cu_seqlens: list[int] | None,
    head_first: bool,
    monkeypatch,
) -> tuple[torch.Tensor, dict[str, Any]]:
    launch_kwargs: dict[str, Any] = {}
    kernel = cumsum_module.chunk_local_cumsum_scalar_kernel

    class SpyKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launch_kwargs.update(kwargs)
                return kernel[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(cumsum_module, "chunk_local_cumsum_scalar_kernel", SpyKernel())
    out = chunk_local_cumsum(
        g,
        chunk_size=BT,
        reverse=reverse,
        cu_seqlens=(
            None
            if cu_seqlens is None
            else torch.tensor(cu_seqlens, device=DEVICE, dtype=torch.int32)
        ),
        head_first=head_first,
    )
    return out, launch_kwargs


def _set_td(use_td: bool, monkeypatch) -> None:
    if use_td and not hasattr(tl, "make_tensor_descriptor"):
        pytest.skip("Triton lacks tl.make_tensor_descriptor")
    if use_td and current_platform.is_rocm():
        pytest.skip("the TD path is not validated on ROCm")
    monkeypatch.setenv("VLLM_TRITON_USE_TD", "1" if use_td else "0")


# H=6: 24-byte fp32 rows are not 16-byte aligned, so TD falls back to pointers.
# H=40: three 16-head tiles, the last one half past H.
@pytest.mark.parametrize("H", [4, 6, 32, 40])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("use_td", [False, True])
@torch.inference_mode()
def test_chunk_local_cumsum_scalar(
    H: int, varlen: bool, reverse: bool, use_td: bool, monkeypatch
):
    _set_td(use_td, monkeypatch)
    torch.manual_seed(0)
    if varlen:
        # Sequences shorter than, equal to and crossing chunk boundaries.
        cu_seqlens = [0, 1, 64, 129, 192, 392]
        B, T = 1, cu_seqlens[-1]
    else:
        B, T = 2, 200
        cu_seqlens = [0, T]
    g = -torch.rand(B, T, H, device=DEVICE, dtype=torch.float32)

    out, kwargs = _run(g, reverse, cu_seqlens if varlen else None, False, monkeypatch)

    assert kwargs["USE_TD"] == (use_td and H % 4 == 0)
    assert out.dtype == torch.float32
    torch.testing.assert_close(
        out, _reference(g, cu_seqlens, reverse, False), atol=1e-5, rtol=1e-5
    )


# bf16 in, fp32 out: H=4 gives 8-byte input rows (pointer fallback), H=16 TD.
@pytest.mark.parametrize("H", [4, 16])
@pytest.mark.parametrize("use_td", [False, True])
@torch.inference_mode()
def test_chunk_local_cumsum_scalar_bf16_input(H: int, use_td: bool, monkeypatch):
    _set_td(use_td, monkeypatch)
    torch.manual_seed(0)
    T = 130
    g = -torch.rand(1, T, H, device=DEVICE).to(torch.bfloat16)

    out, kwargs = _run(g, False, None, False, monkeypatch)

    assert kwargs["USE_TD"] == (use_td and H % 8 == 0)
    torch.testing.assert_close(
        out, _reference(g, [0, T], False, False), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("use_td", [False, True])
@torch.inference_mode()
def test_chunk_local_cumsum_scalar_unaligned_view(use_td: bool, monkeypatch):
    """A contiguous view whose base is not 16-byte aligned keeps the pointer
    path."""
    _set_td(use_td, monkeypatch)
    torch.manual_seed(0)
    B, T, H = 1, 100, 8
    flat = -torch.rand(B * T * H + 1, device=DEVICE, dtype=torch.float32)
    g = flat[1:].view(B, T, H)

    out, kwargs = _run(g, False, None, False, monkeypatch)

    assert kwargs["USE_TD"] is False
    torch.testing.assert_close(
        out, _reference(g, [0, T], False, False), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("use_td", [False, True])
@torch.inference_mode()
def test_chunk_local_cumsum_scalar_head_first(reverse: bool, use_td: bool, monkeypatch):
    """head_first=True ([B, H, T]) always uses the pointer path."""
    _set_td(use_td, monkeypatch)
    torch.manual_seed(0)
    B, H, T = 2, 8, 150
    g = -torch.rand(B, H, T, device=DEVICE, dtype=torch.float32)

    out, kwargs = _run(g, reverse, None, True, monkeypatch)

    assert kwargs["USE_TD"] is False
    torch.testing.assert_close(
        out, _reference(g, [0, T], reverse, True), atol=1e-5, rtol=1e-5
    )
