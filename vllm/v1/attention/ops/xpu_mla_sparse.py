# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import LOG2E, LOGE2, tl, triton


@triton.jit
def _bf16_mla_sparse_kernel(
    q_buffer,
    k_buffer,
    v_buffer,
    indices_ptr,
    out_ptr,
    softmax_lse_ptr,
    max_logits_ptr,
    seq_q,
    seq_kv,
    h_q,
    dim_qk,
    dim_v,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_head,
    stride_v_token,
    stride_v_head,
    stride_out_token,
    stride_out_head,
    stride_lse,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    kv_group_num: tl.constexpr,
    index_topk: tl.constexpr,
    BLOCK_H: tl.constexpr,  # block size for num heads
    BLOCK_M: tl.constexpr,  # block size for num tokens
    BLOCK_N: tl.constexpr,  # block size for indices
    BLOCK_DV: tl.constexpr,  # block size for dim_v
    BLOCK_DMODEL: tl.constexpr,  # block size for dim_nope
    BLOCK_DPE: tl.constexpr,  # block size for positional embedding
    LOGE2: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < h_q)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)

    off_q = cur_q * stride_q_token + cur_head[:, None] * stride_q_head + offs_d[None, :]
    mask_dmodel = offs_d < BLOCK_DMODEL
    q = tl.load(
        q_buffer + off_q, mask=(mask_h[:, None]) & (mask_dmodel[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        off_qpe = (
            cur_q * stride_q_token
            + cur_head[:, None] * stride_q_head
            + offs_dpe[None, :]
        )
        # assume dim_qk == BLOCK_DMODEL + BLOCK_DPE
        mask_dpe = offs_dpe < dim_qk
        qpe = tl.load(
            q_buffer + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
        )

    # Use a finite sentinel rather than -inf: if every key in a chunk is
    # masked (e.g. leading -1 padding in `indices`) an -inf running max gives
    # re_scale = exp2(-inf - -inf) = NaN, permanently poisoning acc / e_sum.
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_indice in range(0, index_topk, BLOCK_N):
        offs_indice = start_indice + tl.arange(0, BLOCK_N)
        mask_indice = offs_indice < index_topk
        indices = tl.load(
            indices_ptr
            + (
                cur_q * stride_indices_token
                + cur_kv_head_id * stride_indices_head
                + offs_indice
            ),
            mask=mask_indice,
            other=-1,
        )

        mask_kv = (indices >= 0) & (indices < seq_kv)
        mask_kv_d = mask_dmodel
        offs_k = (
            indices[None, :] * stride_k_token
            + cur_kv_head_id * stride_k_head
            + offs_d[:, None]
        )

        # q_nope @ k_nope
        k = tl.load(
            k_buffer + offs_k, mask=(mask_kv[None, :]) & (mask_kv_d[:, None]), other=0.0
        )
        qk = tl.dot(q, k.to(q.dtype))

        if BLOCK_DPE > 0:
            # q_rope @ k_rope
            offs_kpe = (
                indices[None, :] * stride_k_token
                + cur_kv_head_id * stride_k_head
                + offs_dpe[:, None]
            )
            mask_k_dpe = offs_dpe < dim_qk
            kpe = tl.load(
                k_buffer + offs_kpe,
                mask=(mask_kv[None, :]) & (mask_k_dpe[:, None]),
                other=0.0,
            )
            qk += tl.dot(qpe, kpe.to(q.dtype))

        # apply scaling
        qk *= sm_scale
        qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, -1.0e30)

        # load v
        mask_v_d = offs_dv < dim_v
        offs_v = (
            indices[:, None] * stride_v_token
            + cur_kv_head_id * stride_v_head
            + offs_dv[None, :]
        )
        v = tl.load(
            v_buffer + offs_v, mask=(mask_kv[:, None]) & (mask_v_d[None, :]), other=0.0
        )

        # online softmax
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]

        # score @ v
        acc += tl.dot(p.to(v.dtype), v)

        # update global sum and max
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # rescaling
    acc /= e_sum[:, None]

    max_logits = e_max * LOGE2
    # calculate lse
    lse = max_logits + tl.log2(e_sum) * LOGE2

    # write output
    offs_o = (
        cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :]
    )
    mask_out_d = offs_dv < dim_v
    tl.store(
        out_ptr + offs_o,
        acc.to(tl.bfloat16),
        mask=(mask_h[:, None]) & (mask_out_d[None, :]),
    )

    offs_lse = cur_q * stride_lse + cur_head
    tl.store(softmax_lse_ptr + offs_lse, lse, mask=mask_h)
    tl.store(max_logits_ptr + offs_lse, max_logits, mask=mask_h)


# reference implementation of bf16 sparse prefill kernel
def triton_bf16_mla_sparse_interface(
    q: torch.Tensor,  # [num_tokens, num_heads_q, dim_qk]
    kv: torch.Tensor,  # [num_tokens, num_heads_kv, dim_qk]
    indices: torch.Tensor,  # [num_tokens, num_heads_kv, topk]
    sm_scale: float,
    d_v: int = 512,
    block_dpe: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse MLA attention over a BF16 latent KV cache, using Triton.

    Args:
        q: Query tensor, [num_tokens, num_heads_q, dim_qk].
        kv: Latent KV cache tensor.
        indices: Sparse KV indices selected for each query token.
        sm_scale: Softmax scale applied to the QK product.
        d_v: Size of the value dimension.
        block_dpe: Size of positional embedding portion of dim_qk.
            Set to 0 when q/kv contain only the nope latent (e.g. DSv4
            prefill where RoPE is not split out).

    Returns:
        `out` `[num_tokens, num_heads_q, d_v]`, `max_logits`
        `[num_tokens, num_heads_q]` and the logsumexp `lse`
        `[num_tokens, num_heads_q]`.

    """
    num_tokens, num_heads_q, dim_qk = q.shape
    _, num_heads_kv, _ = kv.shape
    assert dim_qk == kv.shape[2], "q and kv have different head dimensions"

    # for deepseek v3.2, index topk should be 2048
    _, _, index_topk = indices.shape

    BLOCK_H = 16
    BLOCK_DPE = block_dpe
    BLOCK_DMODEL = dim_qk - BLOCK_DPE
    BLOCK_M = 32
    BLOCK_N = 16
    BLOCK_DV = 512
    assert d_v == BLOCK_DV, "only support d_v = 512"

    assert dim_qk == BLOCK_DMODEL + BLOCK_DPE, (
        "dim_qk does not match BLOCK_DMODEL + BLOCK_DPE"
    )
    assert num_heads_kv == 1, "only support kv head = 1 for now"
    assert index_topk % BLOCK_N == 0, "index_topk must be multiple of BLOCK_N"

    sm_scale *= LOG2E

    kv_group_num = num_heads_q // num_heads_kv
    grid = (
        num_tokens,
        triton.cdiv(num_heads_q, min(BLOCK_H, kv_group_num)),
    )

    out = torch.zeros((num_tokens, num_heads_q, d_v), dtype=q.dtype, device=q.device)
    softmax_lse = torch.zeros(
        (num_tokens, num_heads_q), dtype=torch.float32, device=q.device
    )
    max_logits = torch.zeros(
        (num_tokens, num_heads_q), dtype=torch.float32, device=q.device
    )

    k = kv
    v = kv[..., :d_v]

    _bf16_mla_sparse_kernel[grid](
        q_buffer=q,
        k_buffer=k,
        v_buffer=v,
        indices_ptr=indices,
        out_ptr=out,
        softmax_lse_ptr=softmax_lse,
        max_logits_ptr=max_logits,
        seq_q=num_tokens,
        seq_kv=kv.shape[0],
        h_q=num_heads_q,
        dim_qk=dim_qk,
        dim_v=d_v,
        stride_q_token=q.stride(0),
        stride_q_head=q.stride(1),
        stride_k_token=k.stride(0),
        stride_k_head=k.stride(1),
        stride_v_token=v.stride(0),
        stride_v_head=v.stride(1),
        stride_out_token=out.stride(0),
        stride_out_head=out.stride(1),
        stride_lse=softmax_lse.stride(0),
        stride_indices_token=indices.stride(0),
        stride_indices_head=indices.stride(1),
        sm_scale=sm_scale,
        kv_group_num=kv_group_num,
        index_topk=index_topk,
        BLOCK_H=BLOCK_H,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DV=BLOCK_DV,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        LOGE2=LOGE2,
    )

    return out, max_logits, softmax_lse


# fp8_ds_mla entry (656 bytes): 512 fp8-e4m3 NoPE values in 4 tiles of 128,
# 4 fp32 tile scales, then 64 BF16 RoPE values (zeros for rope-free models).
DS_MLA_NOPE_DIM = 512
DS_MLA_PE_DIM = 64
DS_MLA_TILE = 128
DS_MLA_SCALES_OFFSET = DS_MLA_NOPE_DIM
DS_MLA_ROPE_OFFSET = DS_MLA_SCALES_OFFSET + 4 * (DS_MLA_NOPE_DIM // DS_MLA_TILE)
DS_MLA_ENTRY_BYTES = DS_MLA_ROPE_OFFSET + 2 * DS_MLA_PE_DIM


@triton.jit
def _dequant_gather_ds_mla_kernel(
    out_ptr,  # bf16 [num_rows_out, NOPE + PE]
    cache_ptr,  # uint8, rows of ENTRY_BYTES at row_stride
    rows_ptr,  # int32 [num_rows_out], -1 = invalid
    row_stride,
    NOPE: tl.constexpr,
    PE: tl.constexpr,
    TILE: tl.constexpr,
    SCALES_OFFSET: tl.constexpr,
    ROPE_OFFSET: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.load(rows_ptr + pid).to(tl.int64)
    out_row = out_ptr + pid.to(tl.int64) * (NOPE + PE)
    nope_offs = tl.arange(0, NOPE)
    pe_offs = tl.arange(0, PE)
    if row < 0:
        tl.store(out_row + nope_offs, tl.zeros([NOPE], dtype=tl.bfloat16))
        tl.store(out_row + NOPE + pe_offs, tl.zeros([PE], dtype=tl.bfloat16))
        return
    entry = cache_ptr + row * row_stride
    x = tl.load(entry + nope_offs).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    scales = tl.load(
        (entry + SCALES_OFFSET).to(tl.pointer_type(tl.float32)) + nope_offs // TILE
    )
    tl.store(out_row + nope_offs, (x * scales).to(tl.bfloat16))
    rope = tl.load((entry + ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16)) + pe_offs)
    tl.store(out_row + NOPE + pe_offs, rope)


def dequant_gather_ds_mla_rows(
    kv_rows: torch.Tensor,  # uint8 [num_cache_rows, 656] (row view of the cache)
    rows: torch.Tensor,  # int32, any shape; -1 marks an invalid slot
) -> torch.Tensor:
    """BF16 [rows.numel(), 576] copy of the selected fp8_ds_mla cache rows.

    Dequantizes only the rows a sparse attention step reads (the top-k of
    each query), the same math as cp_gather_and_upconvert_fp8_kv_cache.
    Invalid rows become zeros; the attention kernel never reads them.
    """
    assert kv_rows.dtype == torch.uint8 and kv_rows.shape[-1] == DS_MLA_ENTRY_BYTES
    assert kv_rows.stride(-1) == 1
    flat = rows.reshape(-1).to(torch.int32)
    out = torch.empty(
        (flat.numel(), DS_MLA_NOPE_DIM + DS_MLA_PE_DIM),
        dtype=torch.bfloat16,
        device=kv_rows.device,
    )
    if flat.numel():
        _dequant_gather_ds_mla_kernel[(flat.numel(),)](
            out,
            kv_rows,
            flat,
            kv_rows.stride(0),
            NOPE=DS_MLA_NOPE_DIM,
            PE=DS_MLA_PE_DIM,
            TILE=DS_MLA_TILE,
            SCALES_OFFSET=DS_MLA_SCALES_OFFSET,
            ROPE_OFFSET=DS_MLA_ROPE_OFFSET,
        )
    return out


# Portable torch implementations of the MLA KV cache write and read ops, used
# on XPU when the vllm-xpu-kernels ops are unavailable on the device (or
# kernel_config.sparse_indexer_backend="triton"). They follow the math of the
# SYCL ops so that the cache bytes match. Unlike the CUDA writer, the SYCL
# fp8_ds_mla writer does not round tile scales up to a power of two; readers
# use the stored scales, so either cache decodes correctly.
_FP8_E4M3_MAX = 448.0


def _write_cache_rows(
    kv_cache: torch.Tensor,  # [num_blocks, block_size, entry]
    slot_mapping: torch.Tensor,  # [num_tokens], -1 = padded token
    rows: torch.Tensor,  # [num_tokens, entry], kv_cache.dtype
) -> None:
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    block_size = kv_cache.shape[1]
    valid = slot_mapping >= 0
    # Padded tokens must not write. Without a host sync (graph capture), point
    # them at the first valid token, which writes the same row with the same
    # bytes; with no valid token, rewrite one row with its own content.
    first_valid = torch.argmax(valid.to(torch.int32))
    src = torch.where(valid, torch.arange(num_tokens, device=valid.device), first_valid)
    slot = slot_mapping[src].clamp(min=0)
    block, offset = slot // block_size, slot % block_size
    rows = torch.where(valid.any(), rows[src], kv_cache[block, offset])
    kv_cache[block, offset] = rows


def ds_mla_entries_torch(kv_c: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """fp8_ds_mla entries (uint8 [num_tokens, 656]) of kv_c [num_tokens, 512]
    and k_pe [num_tokens, 64 or 0]; rope-free tokens get a zero RoPE tail."""
    num_tokens = kv_c.shape[0]
    tiles = kv_c.float().reshape(num_tokens, -1, DS_MLA_TILE)
    scales = (tiles.abs().amax(dim=-1) / _FP8_E4M3_MAX).clamp(
        min=torch.finfo(torch.float32).tiny
    )
    nope = (
        (tiles * scales.reciprocal().unsqueeze(-1))
        .clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )
    rope = (
        k_pe.to(torch.bfloat16).contiguous()
        if k_pe.shape[-1] == DS_MLA_PE_DIM
        else kv_c.new_zeros((num_tokens, DS_MLA_PE_DIM), dtype=torch.bfloat16)
    )
    return torch.cat(
        [
            nope.view(torch.uint8).view(num_tokens, DS_MLA_NOPE_DIM),
            scales.view(torch.uint8),
            rope.view(torch.uint8),
        ],
        dim=-1,
    )


def ds_mla_dequant_torch(entries: torch.Tensor) -> torch.Tensor:
    """BF16 [num_rows, 576] values of fp8_ds_mla entries (uint8 [num_rows, 656])."""
    num_rows = entries.shape[0]
    nope = entries[:, :DS_MLA_NOPE_DIM].view(torch.float8_e4m3fn).float()
    scales = entries[:, DS_MLA_SCALES_OFFSET:DS_MLA_ROPE_OFFSET].view(torch.float32)
    nope = nope.view(num_rows, -1, DS_MLA_TILE) * scales.unsqueeze(-1)
    rope = entries[:, DS_MLA_ROPE_OFFSET:DS_MLA_ENTRY_BYTES].view(torch.bfloat16)
    return torch.cat(
        [nope.view(num_rows, DS_MLA_NOPE_DIM).to(torch.bfloat16), rope], dim=-1
    )


def concat_and_cache_mla_torch(
    kv_c: torch.Tensor,  # [>= num_tokens, kv_lora_rank]
    k_pe: torch.Tensor,  # [>= num_tokens, pe_dim]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, entry]
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str,
) -> None:
    """Torch concat_and_cache_mla for unquantized ("auto", "bfloat16",
    "float16") and "fp8_ds_mla" caches; slot -1 is skipped."""
    num_tokens = slot_mapping.shape[0]
    kv_c, k_pe = kv_c[:num_tokens], k_pe[:num_tokens]
    if kv_cache_dtype == "fp8_ds_mla":
        assert kv_c.shape[-1] == DS_MLA_NOPE_DIM
        assert k_pe.shape[-1] in (0, DS_MLA_PE_DIM)
        cache = kv_cache.view(torch.uint8)
        assert cache.shape[-1] == DS_MLA_ENTRY_BYTES
        rows = ds_mla_entries_torch(kv_c, k_pe)
    else:
        assert kv_cache.shape[-1] == kv_c.shape[-1] + k_pe.shape[-1]
        cache = kv_cache
        rows = torch.cat([kv_c, k_pe], dim=-1).to(kv_cache.dtype)
    _write_cache_rows(cache, slot_mapping.flatten(), rows)


def cp_gather_and_upconvert_fp8_kv_cache_torch(
    src_cache: torch.Tensor,  # uint8 [num_blocks, block_size, 656]
    dst: torch.Tensor,  # bf16 [total_rows, 576]
    block_table: torch.Tensor,  # int32 [batch_size, max_blocks]
    workspace_starts: torch.Tensor,  # int32 [batch_size], ascending
    batch_size: int,
    seq_starts: torch.Tensor | None = None,  # int32 [batch_size]
) -> None:
    """Torch cp_gather_and_upconvert_fp8_kv_cache: dst row i belongs to the
    last request whose workspace start is <= i and holds that request's
    context token seq_starts[req] + i - workspace_starts[req]."""
    total_rows = dst.shape[0]
    if total_rows == 0 or batch_size == 0:
        return
    starts = workspace_starts[:batch_size]
    out_row = torch.arange(total_rows, dtype=starts.dtype, device=dst.device)
    req = (torch.searchsorted(starts, out_row, right=True) - 1).clamp(min=0)
    token = out_row - starts[req]
    if seq_starts is not None:
        token = token + seq_starts[req]
    block_size = src_cache.shape[1]
    # Rows past a request's context are never read; keep their (unused) block
    # lookup inside the table.
    col = (token // block_size).clamp(max=block_table.shape[1] - 1)
    block = block_table[req, col].long()
    entries = src_cache[block, (token % block_size).long()]
    dst.copy_(ds_mla_dequant_torch(entries))


# kernel_config.sparse_indexer_backend, captured by
# prepare_xpu_mla_kv_cache_ops() while the vLLM config is set.
_mla_kv_cache_backend = "auto"


def _mla_kv_cache_probe_name(op: str, kv_cache_dtype: str, pe_dim: int) -> str | None:
    if op == "cp_gather_and_upconvert_fp8_kv_cache":
        return f"_C.{op}"
    if kv_cache_dtype in ("auto", "bfloat16", "float16"):
        return f"_C.{op}"
    if kv_cache_dtype == "fp8_ds_mla":
        # Rope-free entries are a separate check: some vllm-xpu-kernels
        # releases reject pe_dim == 0.
        return f"_C.{op}[fp8_ds_mla{'' if pe_dim else ',rope_free'}]"
    return None  # no torch implementation; always use the compiled op


def _mla_kv_cache_probe(name: str, reference: bool) -> list[torch.Tensor]:
    """Run the compiled op `name` (or its torch implementation) once."""
    from vllm.platforms import current_platform

    gen = torch.Generator().manual_seed(0)
    device = current_platform.device_type

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(shape, generator=gen) * scale).to(device, torch.bfloat16)

    num_blocks, block_size, num_tokens = 2, 8, 6
    slot_mapping = torch.tensor([5, -1, 0, 9, 15, -1], device=device)
    if name == "_C.cp_gather_and_upconvert_fp8_kv_cache":
        cache = torch.zeros(
            num_blocks, block_size, DS_MLA_ENTRY_BYTES, dtype=torch.uint8, device=device
        )
        concat_and_cache_mla_torch(
            randn(num_tokens, DS_MLA_NOPE_DIM, scale=2.0),
            randn(num_tokens, DS_MLA_PE_DIM),
            cache,
            slot_mapping,
            "fp8_ds_mla",
        )
        # Two requests, the second one starting mid-block.
        args = (
            cache,
            torch.empty(12, DS_MLA_NOPE_DIM + DS_MLA_PE_DIM, dtype=torch.bfloat16).to(
                device
            ),
            torch.tensor([[1, 0], [0, 1]], dtype=torch.int32, device=device),
            torch.tensor([0, 7], dtype=torch.int32, device=device),
            2,
            torch.tensor([0, 3], dtype=torch.int32, device=device),
        )
        if reference:
            cp_gather_and_upconvert_fp8_kv_cache_torch(*args)
        else:
            torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(
                *args, None, None, None
            )
        return [args[1]]

    ds_mla = "fp8_ds_mla" in name
    pe_dim = 0 if "rope_free" in name else DS_MLA_PE_DIM
    kv_c = randn(num_tokens, DS_MLA_NOPE_DIM, scale=2.0)
    k_pe = randn(num_tokens, pe_dim)
    if ds_mla:
        cache = torch.zeros(
            num_blocks, block_size, DS_MLA_ENTRY_BYTES, dtype=torch.uint8, device=device
        )
    else:
        cache = randn(num_blocks, block_size, DS_MLA_NOPE_DIM + pe_dim)
    kv_cache_dtype = "fp8_ds_mla" if ds_mla else "auto"
    if reference:
        concat_and_cache_mla_torch(kv_c, k_pe, cache, slot_mapping, kv_cache_dtype)
    else:
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            cache,
            slot_mapping,
            kv_cache_dtype,
            torch.ones(1, dtype=torch.float32, device=device),
            False,
        )
    if ds_mla:
        return [ds_mla_dequant_torch(cache.view(-1, DS_MLA_ENTRY_BYTES))]
    return [cache]


def xpu_mla_kv_cache_use_native(
    op: str, kv_cache_dtype: str = "fp8_ds_mla", pe_dim: int = DS_MLA_PE_DIM
) -> bool:
    """Whether the XPU MLA KV cache op `op` ("concat_and_cache_mla" or
    "cp_gather_and_upconvert_fp8_kv_cache") should run the vllm-xpu-kernels
    op rather than its torch implementation.

    Follows kernel_config.sparse_indexer_backend: "native" always uses the
    compiled op, "triton" the portable (torch) implementation, and "auto" the
    compiled op if a one-time probe shows it works on this device.
    """
    from vllm.utils.kernel_probe import kernel_usable

    name = _mla_kv_cache_probe_name(op, kv_cache_dtype, pe_dim)
    if name is None:
        return True
    if _mla_kv_cache_backend == "triton":
        return False
    # Unquantized and read results must match exactly. fp8_ds_mla writes may
    # round a value to an adjacent e4m3 code (2^-3 relative) if the tile scale
    # is computed in a different order; a wrong layout is off by far more.
    fp8_write = "fp8_ds_mla" in name
    usable = kernel_usable(
        name,
        lambda: _mla_kv_cache_probe(name, reference=False),
        lambda: _mla_kv_cache_probe(name, reference=True),
        atol=1e-4 if fp8_write else 0.0,
        rtol=0.125 if fp8_write else 0.0,
    )
    if not usable and _mla_kv_cache_backend == "native":
        raise RuntimeError(
            f"kernel_config.sparse_indexer_backend='native' requires {name} "
            "from vllm-xpu-kernels, which is not usable on this device. Use "
            "'auto' or 'triton'."
        )
    return usable


def prepare_xpu_mla_kv_cache_ops(kv_cache_dtype: str, pe_dim: int) -> None:
    """Resolve the MLA KV cache op implementations of an MLA layer.

    Call at layer construction: the config is read here, and the probes must
    not run during torch.compile tracing or graph capture.
    """
    from vllm.config import get_current_vllm_config_or_none

    global _mla_kv_cache_backend
    config = get_current_vllm_config_or_none()
    if config is not None:
        _mla_kv_cache_backend = config.kernel_config.sparse_indexer_backend
    xpu_mla_kv_cache_use_native("concat_and_cache_mla", kv_cache_dtype, pe_dim)
    if kv_cache_dtype == "fp8_ds_mla":
        xpu_mla_kv_cache_use_native("cp_gather_and_upconvert_fp8_kv_cache")
