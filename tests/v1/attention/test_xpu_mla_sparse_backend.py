# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

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
    import torch

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


@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "fp8_e4m3"])
def test_xpu_sparse_rejects_unsupported_kv_cache_dtype_at_selection(kv_cache_dtype):
    """An FP8 KV cache is not supported by XPU_MLA_SPARSE yet: backend selection
    must fail with a clear error instead of the first forward pass."""
    import torch

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
    path = XPUPlatform.get_attn_backend_cls(None, cfg._replace(kv_cache_dtype="auto"))
    assert path.endswith("XPUMLASparseBackend")
