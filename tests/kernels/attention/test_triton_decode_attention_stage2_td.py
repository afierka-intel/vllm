# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-descriptor (USE_TD) path of the split-KV reduce kernel
(_fwd_kernel_stage2). The host gate must only enable it for 16-byte aligned
logits layouts, and when enabled it must match the pointer path exactly."""

import pytest
import torch

import vllm.v1.attention.ops.triton_decode_attention as tda
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


class _Stage2Spy:
    """Wraps _fwd_kernel_stage2 and records the USE_TD of every launch."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.use_td: list[bool] = []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            self.use_td.append(kwargs["USE_TD"])
            return launch(*args, **kwargs)

        return run


def _reduce(monkeypatch, logits, b_seq_len, H, Lv, td_enabled):
    """Run stage 2 alone; return (o, lse, USE_TD values seen by the kernel)."""
    B, _, num_kv_splits, _ = logits.shape
    spy = _Stage2Spy(tda._fwd_kernel_stage2)
    monkeypatch.setattr(tda, "_fwd_kernel_stage2", spy)
    monkeypatch.setattr(tda, "use_tensor_descriptor", lambda: td_enabled)
    # Stage 2 reads only q.shape[:2] and v_buffer.shape[-1].
    q = torch.empty(B, H, 1, device=DEVICE_TYPE)
    v_buffer = torch.empty(1, 1, Lv, device=DEVICE_TYPE)
    o = torch.zeros(B, H, Lv, dtype=torch.bfloat16, device=DEVICE_TYPE)
    lse = torch.zeros(B, H, dtype=torch.float32, device=DEVICE_TYPE)
    tda._decode_softmax_reducev_fwd(
        logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits
    )
    monkeypatch.undo()
    return o, lse, spy.use_td


def _padded_logits(B, H, S, Lv, row):
    """(B, H, S, Lv + 1) fp32 logits view whose split rows are `row` apart."""
    buf = torch.randn(B, H, S, row, dtype=torch.float32, device=DEVICE_TYPE)
    return buf[..., : Lv + 1]


@pytest.mark.parametrize("Lv", [128, 512])
@pytest.mark.parametrize(
    "pad,expect_td",
    [
        (1, False),  # default layout: row = Lv + 1 fp32, not 16-byte aligned
        (4, True),  # row = Lv + 4 fp32, 16-byte aligned
    ],
)
def test_stage2_td_gate_and_equivalence(monkeypatch, Lv, pad, expect_td):
    B, H, S = 3, 4, 4
    logits = _padded_logits(B, H, S, Lv, Lv + pad)
    # seq_len 1 leaves 3 of 4 splits empty; those must be skipped, not read.
    b_seq_len = torch.tensor([37, 1, 64], dtype=torch.int32, device=DEVICE_TYPE)

    o_ref, lse_ref, seen_ref = _reduce(monkeypatch, logits, b_seq_len, H, Lv, False)
    o_td, lse_td, seen_td = _reduce(monkeypatch, logits, b_seq_len, H, Lv, True)

    assert seen_ref == [False]
    assert seen_td == [expect_td]
    assert torch.equal(o_ref, o_td)
    assert torch.equal(lse_ref, lse_td)


def test_stage2_td_int64_batch_offset(monkeypatch):
    """The descriptor base cur_batch * stride_mid_ob must not wrap in int32.
    stride(0) stays below 2**31 (so Triton passes it as i32) while the last
    batch starts past 2**31 elements; the result must match a compact copy of
    the same data. The pointer path is covered by
    test_decode_attention_int64_logits_offset."""
    B, H, S, Lv = 3, 2, 4, 128
    row = Lv + 4
    stride_b = 2**30 + 64
    assert (B - 1) * stride_b >= 2**31 and stride_b < 2**31
    need = ((B - 1) * stride_b + H * S * row) * 4
    total = torch.get_device_module(DEVICE_TYPE).get_device_properties(0).total_memory
    if total < need + (2 << 30):
        pytest.skip(f"needs {need / 2**30:.1f} GiB of device memory")

    buf = torch.empty((B - 1) * stride_b + H * S * row, device=DEVICE_TYPE)
    big = buf.as_strided((B, H, S, Lv + 1), (stride_b, S * row, row, 1))
    compact = _padded_logits(B, H, S, Lv, row)
    big.copy_(compact)
    b_seq_len = torch.tensor([37, 1, 64], dtype=torch.int32, device=DEVICE_TYPE)

    o_c, lse_c, seen_c = _reduce(monkeypatch, compact, b_seq_len, H, Lv, True)
    o_b, lse_b, seen_b = _reduce(monkeypatch, big, b_seq_len, H, Lv, True)

    assert seen_c == seen_b == [True]
    assert torch.equal(o_c, o_b)
    assert torch.equal(lse_c, lse_b)
