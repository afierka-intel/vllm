# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8_ds_mla KV cache for the XPU sparse MLA backend: writer layout,
dequantization and sparse attention against torch references."""

import pytest
import torch

from tests.kernels.attention.test_xpu_mla_sparse import reference_mla_sparse_prefill
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")

DEVICE = "xpu"


def _ref_write(kv_c: torch.Tensor, k_pe: torch.Tensor, row_bytes: int):
    """Torch reference of one fp8_ds_mla row per token (CUDA kernel semantics)."""
    n = kv_c.shape[0]
    x = kv_c.float().view(n, 4, 128)
    scale = (x.abs().amax(-1) / 448.0).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(scale)))  # [n, 4]
    q = (x / scale[..., None]).to(torch.float8_e4m3fn).view(n, 512)
    rows = torch.zeros(n, row_bytes, dtype=torch.uint8, device=kv_c.device)
    rows[:, :512] = q.view(torch.uint8)
    rows[:, 512:528] = scale.contiguous().view(torch.uint8).view(n, 16)
    if row_bytes == 656 and k_pe.numel():
        rows[:, 528:656] = k_pe.to(torch.bfloat16).contiguous().view(torch.uint8)
    return rows


def _ref_dequant(rows: torch.Tensor, with_rope: bool) -> torch.Tensor:
    n = rows.shape[0]
    q = rows[:, :512].contiguous().view(torch.float8_e4m3fn).float().view(n, 4, 128)
    scale = rows[:, 512:528].contiguous().view(torch.float32).view(n, 4)
    nope = (q * scale[..., None]).view(n, 512)
    if not with_rope:
        return nope.to(torch.bfloat16)
    pe = rows[:, 528:656].contiguous().view(torch.bfloat16).float()
    return torch.cat([nope, pe], -1).to(torch.bfloat16)


def _make_cache(num_blocks, block_size, row_bytes, padded_stride=None):
    """uint8 cache, optionally strided as when other layers share the pages."""
    stride = padded_stride or block_size * row_bytes
    buf = torch.zeros(num_blocks * stride, dtype=torch.uint8, device=DEVICE)
    return torch.as_strided(
        buf, (num_blocks, block_size, row_bytes), (stride, row_bytes, 1)
    )


@pytest.mark.parametrize("pe_dim,row_bytes", [(0, 528), (0, 656), (64, 656)])
@pytest.mark.parametrize("padded", [False, True])
def test_ds_mla_fp8_write_layout(pe_dim, row_bytes, padded):
    from vllm.v1.attention.ops.xpu_mla_sparse_fp8 import ds_mla_fp8_write

    torch.manual_seed(0)
    num_blocks, block_size, n = 8, 64, 200
    stride = block_size * row_bytes + (4096 if padded else 0)
    cache = _make_cache(num_blocks, block_size, row_bytes, stride)
    kv_c = torch.randn(n, 512, dtype=torch.bfloat16, device=DEVICE) * 3
    kv_c[0].zero_()  # all-zero tile -> minimum scale
    k_pe = torch.randn(n, pe_dim, dtype=torch.bfloat16, device=DEVICE)
    slots = torch.randperm(num_blocks * block_size, device=DEVICE)[:n].long()
    slots[5] = -1  # padded token: must not be written

    ds_mla_fp8_write(kv_c, k_pe, cache, slots)

    ref = _ref_write(kv_c, k_pe, row_bytes)
    got = cache.reshape(-1, row_bytes)[slots.clamp(min=0)]
    keep = slots >= 0
    # Scales and the RoPE tail are bit-exact.
    torch.testing.assert_close(got[keep, 512:], ref[keep, 512:], atol=0, rtol=0)
    # FP8 values: same rounding target; allow 1-ulp differences.
    g = got[keep, :512].view(torch.float8_e4m3fn).float()
    r = ref[keep, :512].view(torch.float8_e4m3fn).float()
    assert ((g - r).abs() <= r.abs() / 8 + 1e-6).all()
    assert (g != r).float().mean() < 1e-3
    # The skipped slot's row stays zero.
    skipped = torch.ones(num_blocks * block_size, dtype=torch.bool, device=DEVICE)
    skipped[slots[keep]] = False
    assert (cache.reshape(-1, row_bytes)[skipped] == 0).all()


@pytest.mark.parametrize("with_rope", [False, True])
def test_ds_mla_fp8_gather_dequant(with_rope):
    from vllm.v1.attention.ops.xpu_mla_sparse_fp8 import (
        ds_mla_fp8_gather_dequant,
        ds_mla_fp8_write,
    )

    torch.manual_seed(0)
    row_bytes = 656 if with_rope else 528
    head = 576 if with_rope else 512
    num_rows = 1024
    cache = _make_cache(num_rows // 64, 64, row_bytes)
    kv_c = torch.randn(num_rows, 512, dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.randn(num_rows, 64 if with_rope else 0, dtype=torch.bfloat16).to(
        DEVICE
    )
    ds_mla_fp8_write(kv_c, k_pe, cache, torch.arange(num_rows, device=DEVICE))
    rows = cache.reshape(num_rows, row_bytes)

    idx = torch.randint(0, num_rows, (300,), dtype=torch.int32, device=DEVICE)
    idx[::7] = -1
    out = torch.full((300, head), float("nan"), dtype=torch.bfloat16, device=DEVICE)
    ds_mla_fp8_gather_dequant(out, rows, idx)

    ref = _ref_dequant(rows[idx.clamp(min=0).long()], with_rope)
    ref[idx < 0] = 0
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    # And close to the unquantized input (fp8 e4m3: ~2^-4 relative).
    src = torch.cat([kv_c, k_pe], -1)[idx.clamp(min=0).long()]
    valid = idx >= 0
    torch.testing.assert_close(
        out[valid].float(), src[valid].float(), atol=0.15, rtol=0.07
    )


@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("width", [2048, 2176])
@pytest.mark.parametrize("force_chunks", [False, True])
def test_fp8_ds_mla_sparse_attention(rope_dim, width, force_chunks, monkeypatch):
    import vllm.v1.attention.ops.xpu_mla_sparse_fp8 as fp8_ops

    if force_chunks:
        # ~2 tokens per chunk -> exercises the chunk loop and index offsets.
        monkeypatch.setattr(fp8_ops, "_WORKSPACE_BYTES", 2 * width * 576 * 2)

    torch.manual_seed(0)
    row_bytes = 656 if rope_dim else 528
    head = 512 + rope_dim
    num_rows, num_tokens, num_heads = 4096, 5, 64
    cache = _make_cache(num_rows // 64, 64, row_bytes)
    kv_c = torch.randn(num_rows, 512, dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.randn(num_rows, rope_dim, dtype=torch.bfloat16, device=DEVICE)
    fp8_ops.ds_mla_fp8_write(kv_c, k_pe, cache, torch.arange(num_rows, device=DEVICE))
    rows = cache.reshape(num_rows, row_bytes)

    topk = torch.full((num_tokens, width), -1, dtype=torch.int32, device=DEVICE)
    for t in range(num_tokens):
        n = 300 * (t + 1)
        topk[t, :n] = torch.randperm(num_rows - 128)[:n].to(DEVICE)
        topk[t, width - 64 :] = torch.arange(num_rows - 64, num_rows).to(DEVICE)
    q = torch.randn(num_tokens, num_heads, head, dtype=torch.bfloat16, device=DEVICE)
    sm_scale = head**-0.5

    out = fp8_ops.fp8_ds_mla_sparse_attention(q, rows, topk, sm_scale, rope_dim)

    kv_ref = _ref_dequant(rows, rope_dim > 0).view(num_rows, 1, head)
    ref, _, _, _ = reference_mla_sparse_prefill(
        q, kv_ref, topk[:, None, :], sm_scale, 512
    )
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("head_size,row_bytes", [(512, 528), (576, 656)])
def test_xpu_sparse_fp8_ds_mla_row_bytes(head_size, row_bytes):
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import XPUMLASparseImpl

    impl = XPUMLASparseImpl(
        num_heads=64,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_ds_mla",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        topk_indices_buffer=None,
        indexer=None,
        kv_lora_rank=512,
    )
    assert impl.get_fp8_ds_mla_row_bytes() == row_bytes


@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3", "fp8_ds_mla"])
def test_xpu_sparse_canonicalizes_fp8_to_ds_mla(dtype):
    from vllm.model_executor.layers.attention.mla_attention import (
        _canonicalize_sparse_mla_kv_cache_dtype,
    )
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import XPUMLASparseBackend

    assert dtype in XPUMLASparseBackend.supported_kv_cache_dtypes
    got = _canonicalize_sparse_mla_kv_cache_dtype(XPUMLASparseBackend, dtype)
    assert got == "fp8_ds_mla"
    assert _canonicalize_sparse_mla_kv_cache_dtype(XPUMLASparseBackend, "auto") == (
        "auto"
    )
