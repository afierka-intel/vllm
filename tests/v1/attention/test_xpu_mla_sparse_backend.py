# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="XPU sparse MLA backend",
)


def test_xpu_sparse_impl_implements_logical_topk_hook():
    """``MLAAttention.forward`` calls ``impl.record_logical_topk_ready()``
    unconditionally for sparse layers, so every sparse MLA impl must define it.
    Without it DeepSeek-V3.2 / GLM-5 crash on the first forward on XPU."""
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import XPUMLASparseImpl

    hook = getattr(XPUMLASparseImpl, "record_logical_topk_ready", None)
    assert callable(hook)
    # The XPU impl does not use sparse-MLA index groups: the hook is a no-op.
    impl = object.__new__(XPUMLASparseImpl)
    impl.record_logical_topk_ready()


def test_xpu_sparse_supports_nope_head_size():
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import XPUMLASparseBackend

    # DeepSeek V3.2 (512 latent + 64 RoPE) and GLM-5.3-Flash (NoPE, 512).
    assert XPUMLASparseBackend.get_supported_head_sizes() == [576, 512]


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


@pytest.mark.parametrize("head_size,rope_dim", [(576, 64), (512, 0)])
def test_xpu_sparse_impl_rope_dim(head_size, rope_dim):
    impl = _make_impl(head_size, 64, None)
    assert impl.rope_dim == rope_dim


def test_xpu_sparse_forward_mqa_reads_full_topk_width():
    """GLM-5.3-Flash writes 2176-wide top-k rows (2048 + kpool tail): the
    entries past index_topk must reach the attention kernel."""
    import torch

    from tests.kernels.attention.test_xpu_mla_sparse import (
        reference_mla_sparse_prefill,
    )
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import XPUMLASparseMetadata

    torch.manual_seed(0)
    device = "xpu"
    num_heads, head_size, block_size, width = 64, 512, 64, 2176
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
    # Only the tail columns point at the newest tokens.
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
