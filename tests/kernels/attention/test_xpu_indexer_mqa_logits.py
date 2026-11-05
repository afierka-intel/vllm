# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU indexer prefill path used by the GLM-5.3 kpool sparse indexer.

On XPU the indexer swaps DeepGEMM's compressed ``fp8_fp4_mqa_logits`` for
``torch.ops.vllm.xpu_fp8_mqa_logits``, which returns uncompressed logits over
the whole K buffer, and then ranks each row with ``top_k_per_row_prefill`` over
``[cu_seqlen_ks, cu_seqlen_ke)``. The indexer relies on those top-k indices
being relative to each row's start (pool ids within the request), exactly like
the compressed CUDA path; this test pins that contract.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")


def _ref_mqa_logits(q, k, k_scale, weights, ks, ke):
    """FP32 reference on the dequantized FP8 inputs, -inf outside [ks, ke)."""
    kv = k.float() * k_scale[:, None]
    score = torch.einsum("mhd,nd->mhn", q.float(), kv)
    logits = (score.relu() * weights[:, :, None]).sum(dim=1)
    cols = torch.arange(k.shape[0], device=q.device)[None, :]
    valid = (cols >= ks[:, None]) & (cols < ke[:, None])
    return logits.masked_fill(~valid, float("-inf")), valid


@pytest.mark.parametrize("topk", [512, 2048])
def test_xpu_mqa_logits_topk_is_row_relative(topk: int):
    import vllm._xpu_ops  # noqa: F401  # registers torch.ops.vllm.xpu_*

    torch.manual_seed(0)
    device = "xpu"
    num_heads, head_dim = 32, 128
    # Two prefill requests packed into one K buffer: the second request's rows
    # start at a nonzero K position, which is what row-relative indices cover.
    rows_per_req, kv_per_req = 64, 3000
    m, n = 2 * rows_per_req, 2 * kv_per_req
    row = torch.arange(m, device=device, dtype=torch.int32)
    ks = torch.where(row < rows_per_req, 0, kv_per_req).to(torch.int32)
    ke = ks + kv_per_req - rows_per_req + 1 + row % rows_per_req

    fp8 = current_platform.fp8_dtype()
    q = torch.randn(m, num_heads, head_dim, device=device).to(fp8)
    k_bf16 = torch.randn(n, head_dim, device=device)
    k_scale = (k_bf16.abs().amax(dim=-1) / torch.finfo(fp8).max).float()
    k = (k_bf16 / k_scale[:, None]).to(fp8)
    weights = torch.randn(m, num_heads, device=device, dtype=torch.float32)

    logits = torch.ops.vllm.xpu_fp8_mqa_logits(q, k, k_scale, weights, ks, ke)
    assert logits.shape == (m, n) and logits.dtype == torch.float32

    ref, valid = _ref_mqa_logits(q, k, k_scale, weights, ks, ke)
    got = logits.masked_fill(~valid, 0)
    exp = ref.masked_fill(~valid, 0)
    rel = (got - exp).abs().max() / exp.abs().max()
    assert rel < 1e-2, f"{rel=}"

    out = torch.empty(m, topk, device=device, dtype=torch.int32)
    torch.ops._C.top_k_per_row_prefill(
        logits, ks, ke, out, m, logits.stride(0), logits.stride(1), topk
    )
    row_len = (ke - ks).long()
    n_sel = torch.clamp(row_len, max=topk)
    slot = torch.arange(topk, device=device)[None, :]
    sel = slot < n_sel[:, None]
    # Row-relative: every selected id lies in [0, ke - ks).
    assert ((out >= 0) & (out.long() < row_len[:, None]))[sel].all()
    # Same selection as the reference top-k (compared by score, robust to ties).
    picked = ref.gather(1, (out.long() + ks[:, None].long()).clamp(0, n - 1))
    picked = picked.masked_fill(~sel, 0).sum(dim=1)
    ref_top = ref.topk(topk, dim=1).values.masked_fill(~sel, 0).sum(dim=1)
    torch.testing.assert_close(picked, ref_top, rtol=1e-3, atol=1e-2)
