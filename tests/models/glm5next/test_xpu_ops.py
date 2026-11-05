# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash XPU path (``vllm/models/glm5next/xpu``) against torch references:
the Triton op copies (Hadamard FP8 quant, MTP input norm), the indexer helpers
and the XPU MQA-logits calls as the XPU indexer issues them (prefill rows
re-based to the request start; decode with ``next_n`` folded into the batch)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")

DEVICE = "xpu"
HEAD_DIM = 128
N_HEADS = 32
PAGE = 64  # entries per page taken by the XPU paged MQA-logits kernel


def _hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


@pytest.mark.parametrize("rows", [1, 33, 512])
def test_fwht128_quant_fp8_matches_reference(rows):
    from vllm.models.glm5next.xpu.ops.kpool_compress import fwht128_quant_fp8

    torch.manual_seed(0)
    q = torch.randn(rows, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    q_fp8, scale = fwht128_quant_fp8(q)

    rot = q.float() @ _hadamard(HEAD_DIM).to(DEVICE) / HEAD_DIM**0.5
    absmax = rot.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    ref_scale = torch.exp2(torch.ceil(torch.log2(absmax * (1.0 / 448.0))))
    # Power-of-two (ue8m0) scales; a row whose absmax/448 sits exactly at a
    # power of two may round to the neighbouring exponent.
    ratio = scale / ref_scale
    assert torch.isin(ratio, ratio.new_tensor([0.5, 1.0, 2.0])).all()
    assert (ratio != 1).float().mean() <= 0.01
    assert (q_fp8.float().abs() <= 448).all()
    deq = q_fp8.float() * scale
    torch.testing.assert_close(deq, rot, atol=0.07 * scale.max().item(), rtol=0.07)


@pytest.mark.parametrize("n,h", [(1, 4096), (37, 4096), (5, 1000)])
def test_fused_eh_norm_matches_reference(n, h):
    from vllm.models.glm5next.xpu.ops.fused_eh_norm import fused_eh_norm

    torch.manual_seed(0)
    pos = torch.randint(0, 5, (n,), device=DEVICE)
    pos[0] = 0
    emb = torch.randn(n, h, dtype=torch.bfloat16, device=DEVICE)
    prev = torch.randn(n, h, dtype=torch.bfloat16, device=DEVICE)
    ew = torch.rand(h, dtype=torch.bfloat16, device=DEVICE) + 0.5
    hw = torch.rand(h, dtype=torch.bfloat16, device=DEVICE) + 0.5
    eps = 1e-5
    out = fused_eh_norm(pos, emb, prev, ew, hw, eps)

    def rms(x, w):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()

    e = torch.where((pos == 0)[:, None], 0.0, emb.float())
    ref = torch.cat([rms(e, ew), rms(prev, hw)], -1).to(out.dtype)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_rebase_logits_rows():
    from vllm.models.glm5next.xpu.sparse_indexer import _rebase_logits_rows

    full = torch.arange(3 * 10, dtype=torch.float32, device=DEVICE).view(3, 10)
    starts = torch.tensor([0, 4, 7], device=DEVICE, dtype=torch.int32)
    lens = torch.tensor([3, 6, 2], device=DEVICE, dtype=torch.int32)
    out = _rebase_logits_rows(full, starts, lens, max_len=6)
    assert out.shape == (3, 6)
    for i in range(3):
        s, n = int(starts[i]), int(lens[i])
        torch.testing.assert_close(out[i, :n], full[i, s : s + n])
        assert torch.isinf(out[i, n:]).all()


def test_decode_row_context_lens():
    from vllm.models.glm5next.xpu.sparse_indexer import _decode_row_context_lens

    seq_2d = torch.tensor([[3, 3, 4], [7, 7, 7]], device=DEVICE, dtype=torch.int32)
    out = _decode_row_context_lens(
        seq_2d, None, seq_2d.new_tensor([3, 3]), 6, 2, 3, False, 4
    )
    torch.testing.assert_close(out, seq_2d.flatten())
    # [B, 1] lengths, two decode tokens per request: per-row lengths come from
    # the positions (pool-granular: (pos + 1) // kpool).
    positions = torch.tensor([14, 15, 30, 31], device=DEVICE)
    out = _decode_row_context_lens(
        torch.tensor([[4], [8]], device=DEVICE, dtype=torch.int32),
        positions,
        torch.tensor([2, 2], device=DEVICE, dtype=torch.int32),
        4,
        2,
        2,
        False,
        4,
    )
    torch.testing.assert_close(out, torch.tensor([3, 4, 7, 8], device=DEVICE).int())


def _quantize_rows(k: torch.Tensor):
    """Per-row block-128 ue8m0 FP8 quant (indexer K cache convention)."""
    amax = k.float().abs().amax(-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (k.float() / scale).to(torch.float8_e4m3fn), scale.squeeze(-1)


def _ref_logits(q_fp8, k_fp8, k_scale, weights):
    """DeepGEMM MQA-logits semantics: sum_h w[h] * relu(q[h] . k * k_scale)."""
    s = torch.einsum("rhd,nd->rhn", q_fp8.float(), k_fp8.float())
    s = torch.relu(s * k_scale[None, None, :])
    return (s * weights[:, :, None]).sum(1)


def test_xpu_prefill_logits_rebased_rows():
    """Two requests sharing one gathered K buffer: after re-basing, row i holds
    its own request's logits from column 0, as the DeepGEMM path does."""
    from vllm.models.glm5next.xpu.sparse_indexer import _rebase_logits_rows

    torch.manual_seed(0)
    lens = [300, 517]
    num_k = sum(lens)
    k_fp8, k_scale = _quantize_rows(
        torch.randn(num_k, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    )
    rows = 6  # 3 query rows per request, causal ends
    q_fp8 = torch.randn(rows, N_HEADS, HEAD_DIM, device=DEVICE).to(torch.float8_e4m3fn)
    weights = torch.rand(rows, N_HEADS, device=DEVICE)
    ks = torch.tensor([0, 0, 0, 300, 300, 300], device=DEVICE, dtype=torch.int32)
    ke = torch.tensor([100, 200, 300, 400, 600, 817], device=DEVICE, dtype=torch.int32)

    full = torch.ops.vllm.xpu_fp8_mqa_logits(q_fp8, k_fp8, k_scale, weights, ks, ke)
    out = _rebase_logits_rows(full, ks, ke - ks, max(lens))

    ref = _ref_logits(q_fp8, k_fp8, k_scale, weights)
    for i in range(rows):
        s, e = int(ks[i]), int(ke[i])
        torch.testing.assert_close(out[i, : e - s], ref[i, s:e], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("next_n", [1, 2, 4])
def test_xpu_paged_logits_per_row_context(next_n):
    """Decode with next_n folded into the batch: every row uses its own
    (pool-granular) context length, including rows of one request that see
    the same number of complete pools."""
    from vllm import _custom_ops as ops

    torch.manual_seed(0)
    batch = 3
    ctx = [[130, 130, 131, 131], [64, 65, 65, 65], [500, 501, 502, 502]]
    ctx = torch.tensor([c[:next_n] for c in ctx], device=DEVICE, dtype=torch.int32)
    max_len = int(ctx.max()) + 64
    n_pages = (max_len + PAGE - 1) // PAGE
    num_blocks = batch * n_pages + 3

    kv = torch.zeros(num_blocks, PAGE, HEAD_DIM + 4, dtype=torch.uint8, device=DEVICE)
    block_table = torch.randperm(num_blocks, device=DEVICE)[: batch * n_pages]
    block_table = block_table.view(batch, n_pages).to(torch.int32)
    k_all = []
    for b in range(batch):
        k = torch.randn(max_len, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        pos = torch.arange(max_len, device=DEVICE)
        slots = block_table[b, pos // PAGE].long() * PAGE + pos % PAGE
        ops.indexer_k_quant_and_cache(k, kv, slots, HEAD_DIM, "ue8m0")
        k_all.append(k)

    rows = batch * next_n
    q_fp8 = torch.randn(rows, 1, N_HEADS, HEAD_DIM, device=DEVICE).to(
        torch.float8_e4m3fn
    )
    weights = torch.rand(rows, N_HEADS, device=DEVICE)
    row_ctx = ctx.reshape(rows).contiguous()
    logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
        q_fp8,
        kv.unsqueeze(-2),
        weights,
        row_ctx,
        block_table.repeat_interleave(next_n, dim=0),
        torch.empty(0, dtype=torch.int32, device=DEVICE),
        max_len,
    )
    assert logits.shape[0] == rows

    for r in range(rows):
        b, n = r // next_n, int(row_ctx[r])
        k_fp8, k_scale = _quantize_rows(k_all[b][:n])
        ref = _ref_logits(q_fp8[r], k_fp8, k_scale, weights[r : r + 1])[0]
        torch.testing.assert_close(logits[r, :n], ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("seqs", [[300], [1000, 77, 513]])
def test_kda_chunk_prefill_td_matches_pointer_path(seqs, monkeypatch):
    """The tensor-descriptor (2D block IO) loads in the XPU KDA chunk kernels
    must give the same result as the pointer loads, including partial chunks
    at the end of each sequence."""
    import vllm.models.glm5next.xpu.ops.third_party.kda.kernels as kda_k

    torch.manual_seed(0)
    h, d, t = 8, 128, sum(seqs)

    def r(*shape, dtype=torch.bfloat16, s=1.0):
        return (torch.randn(*shape, device=DEVICE) * s).to(dtype)

    x = dict(
        q=r(1, t, h, d),
        k=r(1, t, h, d),
        v=r(1, t, h, d),
        raw_g=r(1, t, h, d, s=0.5),
        beta=torch.sigmoid(r(1, t, h, dtype=torch.float32)),
        A_log=r(h, dtype=torch.float32, s=0.1),
        g_bias=r(h * d, dtype=torch.float32, s=0.1),
        initial_state=r(len(seqs), h, d, d, dtype=torch.float32, s=0.01),
        cu_seqlens=torch.tensor(
            [0, *torch.tensor(seqs).cumsum(0).tolist()],
            device=DEVICE,
            dtype=torch.int32,
        ),
    )

    outs = []
    for td in (False, True):
        monkeypatch.setattr(kda_k, "use_tensor_descriptor", lambda td=td: td)
        # Fresh inputs per run: the chunk path may update states in place.
        outs.append(
            kda_k.chunk_kda_with_fused_gate(
                **{name: t.clone() for name, t in x.items()},
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                safe_gate=True,
                lower_bound=-5.0,
            )
        )
    (o0, s0), (o1, s1) = outs
    torch.testing.assert_close(o1, o0, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(s1, s0, atol=2e-2, rtol=2e-2)
