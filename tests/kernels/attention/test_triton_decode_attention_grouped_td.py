# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The tensor-descriptor Q load in _fwd_grouped_kernel_stage1 must give the same
bits as the masked pointer load, and must actually be the path that ran."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops import triton_decode_attention as tda

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="the TD path is XPU-only"
)

DEVICE_TYPE = current_platform.device_type


class _LaunchSpy:
    """Records USE_TD and the compiled TTIR of each grouped stage-1 launch."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.launches: list[tuple[bool, str]] = []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            compiled = launch(*args, **kwargs)
            self.launches.append((kwargs["USE_TD"], compiled.asm["ttir"]))
            return compiled

        return run


def _run(monkeypatch, td_env, q, k_buffer, v_buffer, req_to_page, D_V, is_mla):
    """Runs decode attention with VLLM_TRITON_USE_TD=td_env (None = unset).

    Returns (o, lse, attn_logits, ran_td); ran_td is the USE_TD the grouped
    stage-1 kernel was launched with, cross-checked against its TTIR."""
    B, H_Q = q.shape[:2]
    page_size = k_buffer.shape[1]
    num_kv_splits = 8
    o = torch.zeros(B, H_Q, D_V, dtype=q.dtype, device=DEVICE_TYPE)
    lse = torch.zeros(B, H_Q, dtype=q.dtype, device=DEVICE_TYPE)
    attn_logits = torch.full(
        (B, H_Q, num_kv_splits, D_V + 1),
        float("nan"),
        dtype=torch.float32,
        device=DEVICE_TYPE,
    )
    b_seq_len = torch.full((B,), 1027, device=DEVICE_TYPE)
    spy = _LaunchSpy(tda._fwd_grouped_kernel_stage1)
    with monkeypatch.context() as m:
        if td_env is None:
            m.delenv("VLLM_TRITON_USE_TD", raising=False)
        else:
            m.setenv("VLLM_TRITON_USE_TD", td_env)
        m.setattr(tda, "_fwd_grouped_kernel_stage1", spy)
        tda.decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_page,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            1.0 / (q.shape[-1] ** 0.5),
            page_size,
            is_mla=is_mla,
        )
    assert len(spy.launches) == 1
    ran_td, ttir = spy.launches[0]
    assert ("tt.descriptor_load" in ttir) == ran_td
    return o, lse, attn_logits, ran_td


@pytest.mark.parametrize(
    "H_Q,H_KV,D_QK,D_V,is_mla",
    [
        # MLA, BLOCK_DMODEL=512 + BLOCK_DPE=64: tile wider than a 256 box.
        (16, 1, 576, 512, True),
        # MLA with 8 heads (a 16-head model at TP=2): rows past q_head_num.
        (8, 1, 576, 512, True),
        # MLA, BLOCK_DMODEL=256 + BLOCK_DPE=32.
        (16, 1, 288, 256, True),
        # GQA, VALID_BLOCK_H=4 < BLOCK_H=16: neighbour heads inside the tile.
        (32, 8, 128, 128, False),
        # GQA, non-power-of-2 group (VALID_BLOCK_H=5).
        (20, 4, 128, 128, False),
    ],
)
@pytest.mark.parametrize("poison_q", [False, True])
def test_grouped_stage1_td_matches_pointer_path(
    monkeypatch, H_Q, H_KV, D_QK, D_V, is_mla, poison_q
):
    torch.manual_seed(0)
    B, PAGE_SIZE, num_pages = 3, 16, 1024
    q = torch.randn(B, H_Q, D_QK, dtype=torch.bfloat16, device=DEVICE_TYPE)
    if poison_q:
        # Non-finite values in one head must stay in that head, as with the
        # masked pointer load (they are neighbour rows of other programs).
        q[0, 1, 0] = float("nan")
        q[1, H_Q - 1, -1] = float("inf")
    k_buffer = torch.randn(
        num_pages, PAGE_SIZE, H_KV, D_QK, dtype=torch.bfloat16, device=DEVICE_TYPE
    )
    if is_mla:
        v_buffer = k_buffer[..., :D_V]
    else:
        v_buffer = torch.randn(
            num_pages, PAGE_SIZE, H_KV, D_V, dtype=torch.bfloat16, device=DEVICE_TYPE
        )
    req_to_page = torch.randint(
        0, num_pages, (B, cdiv(1027, PAGE_SIZE)), device=DEVICE_TYPE
    )

    args = (q, k_buffer, v_buffer, req_to_page, D_V, is_mla)
    o_ptr, lse_ptr, logits_ptr, ran_td = _run(monkeypatch, "0", *args)
    assert not ran_td
    o_td, lse_td, logits_td, ran_td = _run(monkeypatch, "1", *args)
    assert ran_td

    exact = dict(rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(logits_td, logits_ptr, **exact)
    torch.testing.assert_close(o_td, o_ptr, **exact)
    torch.testing.assert_close(lse_td, lse_ptr, **exact)
    if poison_q:
        clean = torch.ones(B, H_Q, dtype=torch.bool, device=DEVICE_TYPE)
        clean[0, 1] = clean[1, H_Q - 1] = False
        assert torch.isfinite(o_td[clean]).all()


def test_grouped_stage1_td_is_opt_in(monkeypatch):
    """Unset VLLM_TRITON_USE_TD must keep the pointer path on XPU: the Q-only
    descriptor has no measured gain, so it is not part of the XPU default."""
    B, H_Q, D_QK, D_V, PAGE_SIZE, num_pages = 2, 16, 576, 512, 16, 128
    q = torch.randn(B, H_Q, D_QK, dtype=torch.bfloat16, device=DEVICE_TYPE)
    k_buffer = torch.randn(
        num_pages, PAGE_SIZE, 1, D_QK, dtype=torch.bfloat16, device=DEVICE_TYPE
    )
    req_to_page = torch.randint(
        0, num_pages, (B, cdiv(1027, PAGE_SIZE)), device=DEVICE_TYPE
    )
    args = (q, k_buffer, k_buffer[..., :D_V], req_to_page, D_V, True)
    assert not _run(monkeypatch, None, *args)[3]
