# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="XPU sparse MLA backend",
)


def test_xpu_sparse_impl_implements_logical_topk_hook():
    """`MLAAttention.forward` calls `impl.record_logical_topk_ready()`
    unconditionally for sparse layers, so every sparse MLA impl must define it.
    Without it DeepSeek-V3.2 / GLM-5 crash on the first forward on XPU."""
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import XPUMLASparseImpl

    hook = getattr(XPUMLASparseImpl, "record_logical_topk_ready", None)
    assert callable(hook)
    # The XPU impl does not use sparse-MLA index groups: the hook is a no-op.
    impl = object.__new__(XPUMLASparseImpl)
    impl.record_logical_topk_ready()


def _make_impl(head_size: int, num_heads: int, topk_buffer):
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import XPUMLASparseImpl

    return XPUMLASparseImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        topk_indices_buffer=topk_buffer,
        indexer=None,
        kv_lora_rank=512,
    )


@pytest.mark.parametrize(
    "head_size,width",
    [
        (576, 2048),  # DeepSeek-V3.2 / GLM-5.x DSA: 512 latent + 64 RoPE
        (512, 2048),  # rope-free (NoPE) MLA
        (512, 2176),  # GLM-5.3-Flash: index_topk 2048 + kpool tail columns
    ],
)
def test_xpu_sparse_forward_mqa_head_size_and_topk_width(head_size, width):
    """forward_mqa must split the cached row by the model's RoPE dim (64, or 0
    for NoPE) and attend to every column of the shared top-k buffer, including
    tail entries past index_topk."""
    from tests.kernels.attention.test_xpu_mla_sparse import (
        reference_mla_sparse_prefill,
    )
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import XPUMLASparseMetadata

    torch.manual_seed(0)
    device = "xpu"
    num_heads, block_size = 64, 64
    num_blocks, seq_len, num_tokens = 48, 3000, 2

    kv_cache = torch.randn(
        num_blocks, block_size, head_size, dtype=torch.bfloat16, device=device
    )
    # One request; blocks mapped in reverse order to exercise the block table.
    n_req_blocks = (seq_len + block_size - 1) // block_size
    block_table = torch.arange(
        num_blocks - 1, num_blocks - 1 - n_req_blocks, -1, dtype=torch.int32
    ).to(device)[None, :]

    topk = torch.full((num_tokens, width), -1, dtype=torch.int32, device=device)
    topk[:, :1024] = torch.randperm(seq_len - 256)[:1024].to(device)
    # The last columns point at the newest tokens (where a kpool tail lands).
    topk[:, width - 128 :] = torch.arange(
        seq_len - 128, seq_len, dtype=torch.int32, device=device
    )

    impl = _make_impl(head_size, num_heads, topk)
    meta = XPUMLASparseMetadata(
        num_reqs=1,
        max_query_len=num_tokens,
        max_seq_len=seq_len,
        num_actual_tokens=num_tokens,
        query_start_loc=torch.tensor([0, num_tokens], device=device),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64, device=device),
        block_table=block_table,
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32, device=device),
        block_size=block_size,
        topk_tokens=2048,
    )
    q = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16).to(device)
    out, _ = impl.forward_mqa(q, kv_cache, meta, layer=None)

    # Reference: map request-local positions to physical rows by hand.
    local = topk.long()
    phys = block_table[0].long()[local.clamp(min=0) // block_size] * block_size + (
        local.clamp(min=0) % block_size
    )
    phys = torch.where(local >= 0, phys, -1).to(torch.int32)
    rows = kv_cache.view(-1, 1, head_size)
    ref, _, _, _ = reference_mla_sparse_prefill(
        q, rows, phys[:, None, :], head_size**-0.5, 512
    )
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("kv_cache_dtype", ["fp8_e5m2", "fp8_inc"])
def test_xpu_sparse_rejects_unsupported_kv_cache_dtype_at_selection(kv_cache_dtype):
    """A KV cache dtype XPU_MLA_SPARSE cannot store (only fp8 as fp8_ds_mla is
    supported): backend selection must fail with a clear error instead of the
    first forward pass."""
    from vllm.platforms.xpu import XPUPlatform
    from vllm.v1.attention.selector import AttentionSelectorConfig

    cfg = AttentionSelectorConfig(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        block_size=64,
        use_mla=True,
        use_sparse=True,
    )
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        XPUPlatform.get_attn_backend_cls(None, cfg)
    # Supported dtypes still select the backend.
    for supported in ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
        path = XPUPlatform.get_attn_backend_cls(
            None, cfg._replace(kv_cache_dtype=supported)
        )
        assert path.endswith("XPUMLASparseBackend")


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize("pe_dim", [64, 0])
@pytest.mark.parametrize("max_rows", [None, 300])
@torch.inference_mode()
def test_xpu_sparse_fp8_ds_mla_forward(
    pe_dim: int, max_rows: int | None, portable: bool, monkeypatch
):
    """forward_mqa over an fp8_ds_mla cache equals sparse attention over the
    dequantized cache, and stays close to the BF16 KV it was written from.

    The batch mixes decode-like requests (only their top-k rows are
    dequantized) and a prefill-like one (its whole context is gathered with
    cp_gather_and_upconvert_fp8_kv_cache); max_rows=300 splits it into
    several runs. pe_dim=0 is a rope-free model (zero RoPE envelope).
    portable=True writes and gathers the cache with the torch implementations
    instead of the SYCL ops.
    """
    import numpy as np

    from tests.kernels.attention.test_xpu_mla_sparse import (
        reference_mla_sparse_prefill,
    )
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.attention.mla_attention import (
        _canonicalize_sparse_mla_kv_cache_dtype,
    )
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
        XPUMLASparseBackend,
        XPUMLASparseImpl,
        XPUMLASparseMetadata,
        build_fp8_ds_mla_chunks,
    )
    from vllm.v1.attention.ops.xpu_mla_sparse import DS_MLA_ENTRY_BYTES

    # --kv-cache-dtype fp8 is stored as fp8_ds_mla on this backend.
    for alias in ("fp8", "fp8_e4m3"):
        assert (
            _canonicalize_sparse_mla_kv_cache_dtype(XPUMLASparseBackend, alias)
            == "fp8_ds_mla"
        )

    device = torch.device("xpu")
    torch_calls: dict[str, int] = {}
    if portable:
        from vllm.v1.attention.ops import xpu_mla_sparse as xpu_ops

        monkeypatch.setattr(xpu_ops, "_mla_kv_cache_backend", "triton")
        for op in ("concat_and_cache_mla", "cp_gather_and_upconvert_fp8_kv_cache"):
            impl_fn = getattr(xpu_ops, f"{op}_torch")

            def counted(*args, _op=op, _fn=impl_fn, **kwargs):
                torch_calls[_op] = torch_calls.get(_op, 0) + 1
                return _fn(*args, **kwargs)

            monkeypatch.setattr(xpu_ops, f"{op}_torch", counted)
    torch.random.manual_seed(0)
    num_heads, topk, block_size, num_blocks = 64, 128, 64, 32
    dim_qk = 512 + pe_dim
    scale = dim_qk**-0.5
    # (query_len, context_len): decode-like, prefill-like, decode-like.
    reqs = [(2, 900), (200, 200), (1, 450)]
    query_start_loc = np.cumsum([0] + [q for q, _ in reqs]).astype(np.int32)
    seq_lens = np.array([c for _, c in reqs], dtype=np.int32)
    num_tokens = int(query_start_loc[-1])

    topk_buffer = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=device)
    impl_kwargs = dict(
        num_heads=num_heads,
        head_size=dim_qk,
        scale=scale,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        topk_indices_buffer=topk_buffer,
        kv_lora_rank=512,
        qk_rope_head_dim=pe_dim,
    )
    impl = XPUMLASparseImpl(kv_cache_dtype="fp8_ds_mla", **impl_kwargs)

    # Each request owns a disjoint, shuffled set of blocks.
    perm = torch.randperm(num_blocks).to(torch.int32)
    max_blocks = 16
    block_table = torch.zeros(len(reqs), max_blocks, dtype=torch.int32)
    used = 0
    for r, (_, ctx) in enumerate(reqs):
        n = -(-ctx // block_size)
        block_table[r, :n] = perm[used : used + n]
        used += n
    block_table = block_table.to(device)

    kv_c = torch.randn(num_blocks * block_size, 512, dtype=torch.bfloat16) * 2
    k_pe = torch.randn(num_blocks * block_size, pe_dim, dtype=torch.bfloat16)
    kv_c, k_pe = kv_c.to(device), k_pe.to(device)
    cache = torch.zeros(
        num_blocks, block_size, DS_MLA_ENTRY_BYTES, dtype=torch.uint8, device=device
    )
    ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        cache,
        torch.arange(num_blocks * block_size, dtype=torch.int64, device=device),
        kv_cache_dtype="fp8_ds_mla",
        scale=torch.ones(1, dtype=torch.float32, device=device),
    )

    # Causal request-local top-k with -1 padding, and the slot each maps to.
    req_ids, global_rows = [], []
    for r, (q_len, ctx) in enumerate(reqs):
        for i in range(q_len):
            t = int(query_start_loc[r]) + i
            visible = ctx - q_len + i + 1
            sel = torch.randperm(visible)[: min(topk, visible)].to(torch.int32)
            topk_buffer[t, : sel.numel()] = sel.to(device)
            req_ids.append(r)
            row = torch.full((topk,), -1, dtype=torch.int64)
            row[: sel.numel()] = (
                block_table[r, (sel // block_size).long()].cpu().long() * block_size
                + (sel % block_size).long()
            )
            global_rows.append(row)
    req_id_per_token = torch.tensor(req_ids, dtype=torch.int32, device=device)
    global_rows_t = torch.stack(global_rows).to(device)

    chunk_kwargs = {} if max_rows is None else {"max_rows": max_rows}
    chunks, context_starts = build_fp8_ds_mla_chunks(
        query_start_loc, seq_lens, topk, **chunk_kwargs
    )
    assert any(c[4] > 0 for c in chunks) and any(c[4] == 0 for c in chunks)
    if max_rows is not None:
        assert len(chunks) > 2
    metadata = XPUMLASparseMetadata(
        num_reqs=len(reqs),
        max_query_len=max(q for q, _ in reqs),
        max_seq_len=int(seq_lens.max()),
        num_actual_tokens=num_tokens,
        query_start_loc=torch.from_numpy(query_start_loc).to(device),
        slot_mapping=torch.empty(0, dtype=torch.int64, device=device),
        block_table=block_table,
        req_id_per_token=req_id_per_token,
        block_size=block_size,
        topk_tokens=topk,
        fp8_chunks=chunks,
        fp8_context_starts=torch.from_numpy(context_starts).to(device),
    )

    q = torch.randn(num_tokens, num_heads, dim_qk, dtype=torch.bfloat16, device=device)
    out, _ = impl.forward_mqa(q, cache, metadata, None)

    # Reference: sparse attention over the dequantized cache (pe_dim=0
    # entries read back with a zero RoPE tail).
    dequant = torch.empty(
        num_blocks * block_size, 576, dtype=torch.bfloat16, device=device
    )
    ops.cp_gather_and_upconvert_fp8_kv_cache(
        src_cache=cache,
        dst=dequant,
        block_table=torch.arange(num_blocks, dtype=torch.int32, device=device)[None],
        workspace_starts=torch.zeros(1, dtype=torch.int32, device=device),
        batch_size=1,
    )
    indices = global_rows_t.to(torch.int32).view(num_tokens, 1, topk)
    ref, _, _, _ = reference_mla_sparse_prefill(
        q, dequant[:, None, :dim_qk], indices, scale, 512
    )
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    # fp8 KV vs the unquantized BF16 KV.
    bf16_kv = torch.cat([kv_c, k_pe], dim=-1)[:, None, :]
    ref_bf16, _, _, _ = reference_mla_sparse_prefill(q, bf16_kv, indices, scale, 512)
    rel = (out.float() - ref_bf16.float()).norm() / ref_bf16.float().norm()
    assert rel < 0.05, f"fp8 KV vs BF16 KV relative L2 {rel:.4f}"
    if portable:
        assert set(torch_calls) == {
            "concat_and_cache_mla",
            "cp_gather_and_upconvert_fp8_kv_cache",
        }
