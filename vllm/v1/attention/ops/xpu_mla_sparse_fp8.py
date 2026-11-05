# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``fp8_ds_mla`` KV cache for the XPU sparse MLA backend.

Row layout (same as the CUDA ``concat_and_cache_ds_mla`` kernel):

- bytes ``[0, 512)``: the 512 latent values as ``float8_e4m3``;
- bytes ``[512, 528)``: 4 ``float32`` scales, one per 128 latent values,
  ``scale = 2 ** ceil(log2(max(amax / 448, 1e-4)))``;
- bytes ``[528, 656)``: the 64 RoPE values as ``bfloat16`` (656-byte rows
  only; NoPE models use 528-byte rows without this tail).

Reading dequantizes the selected top-k rows to BF16 and reuses the BF16 sparse
MLA kernel. Queries are processed in token chunks so the BF16 workspace stays
bounded during prefill.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface

KV_LORA_RANK = 512
ROPE_DIM = 64
QUANT_TILE = 128
NUM_TILES = KV_LORA_RANK // QUANT_TILE
SCALE_OFFSET = KV_LORA_RANK  # bytes
ROPE_OFFSET = KV_LORA_RANK + 4 * NUM_TILES  # bytes
ROW_BYTES_NOPE = ROPE_OFFSET  # 528
ROW_BYTES_ROPE = ROPE_OFFSET + 2 * ROPE_DIM  # 656
FP8_MAX = 448.0
MIN_SCALE = 1e-4

# Upper bound of the BF16 workspace used to dequantize top-k rows.
_WORKSPACE_BYTES = 256 * 1024 * 1024


def _flat_bytes(t: torch.Tensor) -> torch.Tensor:
    """1-D uint8 view spanning ``t`` (which may be strided, e.g. a layer's
    pages inside a shared KV buffer), so it can be re-viewed as fp32/bf16."""
    assert t.dtype == torch.uint8 and t.stride(-1) == 1
    span = 1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()))
    span = (span + 3) // 4 * 4
    assert t.storage_offset() % 4 == 0, "fp8_ds_mla cache must be 4-byte aligned"
    return torch.as_strided(t, (span,), (1,))


@triton.jit
def _ds_mla_fp8_write_kernel(
    kv_c_ptr,  # [num_tokens, 512] bf16/fp16
    k_pe_ptr,  # [num_tokens, pe_dim] bf16/fp16
    cache_u8_ptr,  # cache as uint8
    cache_f32_ptr,  # same memory as float32
    cache_b16_ptr,  # same memory as 16-bit
    slot_mapping_ptr,  # [num_tokens] int64
    kv_c_stride,
    k_pe_stride,
    block_stride,  # bytes
    entry_stride,  # bytes
    block_size,
    PE_DIM: tl.constexpr,
    HAS_ROPE_TAIL: tl.constexpr,
    QUANT_TILE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
    ROPE_OFFSET: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MIN_SCALE: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + token)
    if slot < 0:
        return
    block_idx = slot // block_size
    block_off = slot % block_size
    dst = block_idx * block_stride + block_off * entry_stride  # bytes

    tiles = tl.arange(0, NUM_TILES)
    cols = tl.arange(0, QUANT_TILE)
    x = tl.load(
        kv_c_ptr + token * kv_c_stride + tiles[:, None] * QUANT_TILE + cols[None, :]
    ).to(tl.float32)  # [NUM_TILES, QUANT_TILE]
    amax = tl.max(tl.abs(x), axis=1)
    scale = tl.maximum(amax / FP8_MAX, MIN_SCALE)
    scale = tl.exp2(tl.ceil(tl.log2(scale)))

    q = (x / scale[:, None]).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(cache_u8_ptr + dst + tiles[:, None] * QUANT_TILE + cols[None, :], q)
    tl.store(cache_f32_ptr + (dst + SCALE_OFFSET) // 4 + tiles, scale)

    if HAS_ROPE_TAIL:
        r = tl.arange(0, ROPE_DIM)
        rope_dst = cache_b16_ptr + (dst + ROPE_OFFSET) // 2 + r
        if PE_DIM == 0:
            zeros = tl.zeros([ROPE_DIM], dtype=cache_b16_ptr.dtype.element_ty)
            tl.store(rope_dst, zeros)
        else:
            pe = tl.load(k_pe_ptr + token * k_pe_stride + r)
            tl.store(rope_dst, pe)


def ds_mla_fp8_write(
    kv_c: torch.Tensor,  # [num_tokens, 512]
    k_pe: torch.Tensor,  # [num_tokens, pe_dim] (pe_dim 0 or 64)
    kv_cache: torch.Tensor,  # [num_blocks, block_size, row_bytes] uint8
    slot_mapping: torch.Tensor,  # [num_tokens]
) -> None:
    """Write ``kv_c`` / ``k_pe`` into an ``fp8_ds_mla`` cache."""
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0 or kv_cache.numel() == 0:
        return
    assert kv_cache.dtype == torch.uint8
    row_bytes = kv_cache.shape[-1]
    pe_dim = k_pe.shape[-1] if k_pe.numel() > 0 else 0
    assert kv_c.shape[-1] == KV_LORA_RANK, "fp8_ds_mla needs kv_lora_rank 512"
    assert pe_dim in (0, ROPE_DIM)
    assert row_bytes == ROW_BYTES_ROPE or (
        row_bytes == ROW_BYTES_NOPE and pe_dim == 0
    ), f"fp8_ds_mla row of {row_bytes} bytes with pe_dim={pe_dim}"
    assert kv_cache.stride(0) % 4 == 0 and kv_cache.stride(1) % 4 == 0
    assert kv_c.stride(-1) == 1
    if pe_dim:
        assert k_pe.stride(-1) == 1

    b16 = torch.bfloat16 if kv_c.dtype == torch.bfloat16 else torch.float16
    if pe_dim:
        k_pe = k_pe.to(b16)
    flat = _flat_bytes(kv_cache)
    _ds_mla_fp8_write_kernel[(num_tokens,)](
        kv_c,
        k_pe if pe_dim else kv_c,  # unused when PE_DIM == 0
        flat,
        flat.view(torch.float32),
        flat.view(b16),
        slot_mapping,
        kv_c.stride(0),
        k_pe.stride(0) if pe_dim else 0,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.shape[1],
        PE_DIM=pe_dim,
        HAS_ROPE_TAIL=row_bytes == ROW_BYTES_ROPE,
        QUANT_TILE=QUANT_TILE,
        NUM_TILES=NUM_TILES,
        SCALE_OFFSET=SCALE_OFFSET,
        ROPE_OFFSET=ROPE_OFFSET,
        ROPE_DIM=ROPE_DIM,
        FP8_MAX=FP8_MAX,
        MIN_SCALE=MIN_SCALE,
    )


@triton.jit
def _ds_mla_fp8_gather_dequant_kernel(
    out_ptr,  # [num_rows_out, out_dim] bf16
    rows_u8_ptr,  # flat cache rows as uint8
    rows_f32_ptr,  # same memory as float32
    rows_b16_ptr,  # same memory as bf16
    indices_ptr,  # [num_rows_out] int32 row ids, -1 = empty
    row_bytes,
    out_stride,
    num_rows_src,
    QUANT_TILE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
    ROPE_OFFSET: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    OUT_ROPE: tl.constexpr,
):
    i = tl.program_id(0)
    row = tl.load(indices_ptr + i)
    # Empty slots (-1) are masked by the attention kernel; write zeros so the
    # workspace stays finite.
    valid = (row >= 0) & (row < num_rows_src)
    row = tl.where(valid, row, 0)
    tiles = tl.arange(0, NUM_TILES)
    cols = tl.arange(0, QUANT_TILE)
    offs = tiles[:, None] * QUANT_TILE + cols[None, :]
    out = out_ptr + i.to(tl.int64) * out_stride
    base = row.to(tl.int64) * row_bytes
    q = tl.load(rows_u8_ptr + base + offs).to(tl.float8e4nv, bitcast=True)
    scale = tl.load(rows_f32_ptr + (base + SCALE_OFFSET) // 4 + tiles)
    x = tl.where(valid, q.to(tl.float32) * scale[:, None], 0.0)
    tl.store(out + offs, x.to(tl.bfloat16))
    if OUT_ROPE:
        r = tl.arange(0, ROPE_DIM)
        pe = tl.load(rows_b16_ptr + (base + ROPE_OFFSET) // 2 + r).to(tl.float32)
        pe = tl.where(valid, pe, 0.0)
        tl.store(out + NUM_TILES * QUANT_TILE + r, pe.to(tl.bfloat16))


def ds_mla_fp8_gather_dequant(
    out: torch.Tensor,  # [n, 512 or 576] bf16
    rows: torch.Tensor,  # [num_rows, row_bytes] uint8 (flat cache rows)
    indices: torch.Tensor,  # [n] int32, -1 = empty
) -> None:
    n = indices.shape[0]
    if n == 0:
        return
    row_bytes = rows.shape[-1]
    out_rope = out.shape[-1] == KV_LORA_RANK + ROPE_DIM
    assert out.shape[-1] in (KV_LORA_RANK, KV_LORA_RANK + ROPE_DIM)
    assert not out_rope or row_bytes == ROW_BYTES_ROPE
    assert rows.stride(0) == row_bytes and rows.stride(1) == 1
    flat = _flat_bytes(rows)
    _ds_mla_fp8_gather_dequant_kernel[(n,)](
        out,
        flat,
        flat.view(torch.float32),
        flat.view(torch.bfloat16),
        indices,
        row_bytes,
        out.stride(0),
        rows.shape[0],
        QUANT_TILE=QUANT_TILE,
        NUM_TILES=NUM_TILES,
        SCALE_OFFSET=SCALE_OFFSET,
        ROPE_OFFSET=ROPE_OFFSET,
        ROPE_DIM=ROPE_DIM,
        OUT_ROPE=out_rope,
    )


def fp8_ds_mla_sparse_attention(
    q: torch.Tensor,  # [num_tokens, num_heads, head_size] bf16
    rows: torch.Tensor,  # [num_rows, row_bytes] uint8 (flat cache rows)
    topk_global: torch.Tensor,  # [num_tokens, width] int32 row ids, -1 = empty
    sm_scale: float,
    rope_dim: int,
) -> torch.Tensor:
    """Sparse MLA over an ``fp8_ds_mla`` cache: dequantize the selected rows
    to BF16 and run the BF16 kernel, in bounded token chunks."""
    num_tokens, num_heads, head_size = q.shape
    width = topk_global.shape[-1]
    out = torch.empty(
        (num_tokens, num_heads, KV_LORA_RANK), dtype=q.dtype, device=q.device
    )
    if num_tokens == 0:
        return out
    per_token = width * head_size * 2
    chunk = max(1, min(num_tokens, _WORKSPACE_BYTES // per_token))
    workspace = torch.empty(
        (chunk * width, head_size), dtype=torch.bfloat16, device=q.device
    )
    # Workspace row of each (token, column): token-local, -1 stays -1.
    local = torch.arange(width, device=q.device, dtype=torch.int32)
    for start in range(0, num_tokens, chunk):
        end = min(num_tokens, start + chunk)
        n = end - start
        idx = topk_global[start:end]
        ds_mla_fp8_gather_dequant(
            workspace[: n * width], rows, idx.reshape(-1).to(torch.int32)
        )
        ws_idx = torch.arange(n, device=q.device, dtype=torch.int32)[:, None] * width
        ws_idx = torch.where(idx >= 0, ws_idx + local[None, :], -1)
        o, _, _ = triton_bf16_mla_sparse_interface(
            q[start:end].to(torch.bfloat16),
            workspace[: n * width].view(n * width, 1, head_size),
            ws_idx.view(n, 1, width),
            sm_scale=sm_scale,
            block_dpe=rope_dim,
        )
        out[start:end] = o.to(q.dtype)
    return out
