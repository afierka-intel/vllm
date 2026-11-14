# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton FP8 MQA logits for the DSA sparse indexer.

Platform-neutral kernels shared by the ROCm and XPU indexer paths: the
paged decode/MTP logits over the block-flat FP8 indexer K cache and the
ragged prefill logits over a gathered FP8 K.
"""

import functools

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.workspace import current_workspace_manager


def _decode_logits_workspace(num_rows: int, max_model_len: int) -> torch.Tensor:
    from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE

    # GLM-5-next's kpool indexer runs SparseIndexerTopk on these logits, which
    # takes its top-k workspace from the start of the same buffer.
    _, logits = current_workspace_manager().get_simultaneous(
        ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        ((num_rows, max_model_len), torch.float32),
    )
    return logits


@functools.cache
def _num_compute_units() -> int:
    return current_platform.num_compute_units()


@triton.jit
def _fp8_paged_mqa_logits_decode_kernel(
    q_ptr,  # fp8 [B, NEXT_N, H, D]
    kv_val_ptr,  # fp8, block-flat: [num_blocks, block_size*(D+4)]
    kv_scale_ptr,  # fp32, block-flat: [num_blocks, block_size*(D+4)//4]
    weights_ptr,  # fp32 [B*NEXT_N, H]
    ctx_lens_ptr,  # int32 [B*NEXT_N] if CTX_PER_ROW else [B]
    block_tables_ptr,  # int32 [B, max_blocks]
    logits_ptr,  # fp32 [B*NEXT_N, max_model_len]
    stride_q_b,
    stride_q_n,
    stride_q_h,
    stride_w_row,
    stride_kvblk_fp8,
    stride_kvblk_f32,
    scale_region_off,  # block_size*D // 4 (fp32 offset of scale region within a block)
    stride_bt_b,
    stride_logits_row,
    max_blocks,  # block_tables width; guards the block-table gather
    max_model_len,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # paged-cache page size
    BLOCK_KV: tl.constexpr,  # positions per tile; must divide BLOCK_SIZE
    N_SPLITS: tl.constexpr,  # KV-tile parallelism factor (grid dim 1)
    NEXT_N: tl.constexpr,  # query positions per batch (1 = decode; >1 = MTP verify)
    CTX_PER_ROW: tl.constexpr,  # ctx_lens is already per-(b, n)
):
    # scale_region_off = block_size*D//4 needs D % 4 == 0 (D=128).
    tl.static_assert(HEAD_SIZE % 4 == 0)
    # Grid (B*NEXT_N, N_SPLITS): disjoint KV tiles per program, no host sync.
    row = tl.program_id(0)
    split = tl.program_id(1)
    b = row // NEXT_N
    n = row % NEXT_N
    if CTX_PER_ROW:
        seq_len = tl.load(ctx_lens_ptr + row)
    else:
        seq_len = tl.load(ctx_lens_ptr + b) - NEXT_N + n + 1
    seq_len = tl.minimum(tl.maximum(seq_len, 0), max_model_len)

    h = tl.arange(0, NUM_HEADS)
    d = tl.arange(0, HEAD_SIZE)
    # keep q/kv fp8 -> fp8 MFMA (f32 accum), not the slow f32 path
    q = tl.load(
        q_ptr + b * stride_q_b + n * stride_q_n + h[:, None] * stride_q_h + d[None, :]
    )
    w = tl.load(weights_ptr + row * stride_w_row + h).to(tl.float32)  # [H]

    kv_col = tl.arange(0, BLOCK_KV)
    for kv_start in tl.range(split * BLOCK_KV, seq_len, N_SPLITS * BLOCK_KV):
        pos = kv_start + kv_col
        # Tiles are page-aligned (BLOCK_KV | BLOCK_SIZE), so a tile is one page.
        logical_blk = kv_start // BLOCK_SIZE
        blk_ok = logical_blk < max_blocks  # guard against a wild page -> fault
        mask_pos = (pos < seq_len) & blk_ok
        page = tl.load(
            block_tables_ptr + b * stride_bt_b + logical_blk, mask=blk_ok, other=0
        ).to(tl.int64)
        pos_in_blk = pos - logical_blk * BLOCK_SIZE  # [BLOCK_KV] == pos % BLOCK_SIZE

        # block-flat layout: values region, then scales region.
        val_off = page * stride_kvblk_fp8 + pos_in_blk[:, None] * HEAD_SIZE + d[None, :]
        kv = tl.load(
            kv_val_ptr + val_off, mask=mask_pos[:, None], other=0.0
        )  # [BLOCK_KV, D] fp8
        sc = tl.load(
            kv_scale_ptr + page * stride_kvblk_f32 + scale_region_off + pos_in_blk,
            mask=mask_pos,
            other=0.0,
        )  # [BLOCK_KV]

        # fp8 upcasts exactly; bit-identical on gfx942, ~1e-5 rel on gfx950
        # (accum order), top-k unchanged.
        s = tl.dot(kv, tl.trans(q))  # [BLOCK_KV, H]
        s = tl.maximum(s, 0.0)
        s = s * w[None, :]
        s = tl.sum(s, axis=1)  # [BLOCK_KV]
        s = s * sc
        tl.store(logits_ptr + row * stride_logits_row + pos, s, mask=mask_pos)


def fp8_paged_mqa_logits_triton(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton paged MQA-logits for decode and MTP; matches the torch ref but
    has no host sync, so it is safe to capture under a full CUDA graph.

    Only key positions inside each row's context are written. `out`, if
    given, is the [B * next_n, max_model_len] fp32 output; by default the
    rows come from the shared workspace and the positions past the context
    are left unset."""
    batch_size, next_n, num_heads, head_size = q_fp8.shape
    block_size = kv_cache_fp8.shape[1]
    BLOCK_KV = 64
    assert block_size % BLOCK_KV == 0

    fp8_dtype = current_platform.fp8_dtype()
    num_blocks = kv_cache_fp8.shape[0]
    kv_flat = kv_cache_fp8.reshape(
        num_blocks, -1
    )  # uint8 [num_blocks, block_size*(D+4)]
    kv_val = kv_flat.view(fp8_dtype)  # [num_blocks, block_size*(D+4)] fp8
    kv_scale = kv_flat.view(torch.float32)  # [num_blocks, block_size*(D+4)//4] fp32

    cl = context_lens.reshape(-1)
    ctx_per_row = not (next_n > 1 and cl.numel() == batch_size)

    max_blocks = block_tables.shape[1]
    if out is None:
        out_logits = _decode_logits_workspace(batch_size * next_n, max_model_len)
    else:
        out_logits = out

    # Memory-bound over the KV range: split each row's keys across programs so
    # few-row / long-context launches still fill the GPU. Cap splits at the
    # device CU count (304 on gfx942, 256 on gfx950) rather than a gfx950-sized
    # constant. All terms are static at launch, so the grid stays CUDA-graph-safe.
    rows = batch_size * next_n
    tiles_cap = (max_model_len + BLOCK_KV - 1) // BLOCK_KV
    N_SPLITS = max(1, min(max(1, _num_compute_units()), tiles_cap, 1024 // rows))
    _fp8_paged_mqa_logits_decode_kernel[(rows, N_SPLITS)](
        q_fp8,
        kv_val,
        kv_scale,
        weights,
        cl,
        block_tables,
        out_logits,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        weights.stride(0),
        kv_val.stride(0),
        kv_scale.stride(0),
        (block_size * head_size) // 4,
        block_tables.stride(0),
        out_logits.stride(0),
        max_blocks,
        max_model_len,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        BLOCK_SIZE=block_size,
        BLOCK_KV=BLOCK_KV,
        N_SPLITS=N_SPLITS,
        NEXT_N=next_n,
        CTX_PER_ROW=ctx_per_row,
        num_warps=4,
        num_stages=2,
    )
    return out_logits


@triton.jit
def _fp8_mqa_logits_kernel(
    q_ptr,  # fp8 [M, H, D]
    k_ptr,  # fp8 [N, D]
    k_scale_ptr,  # fp32 [N]
    weights_ptr,  # fp32 [M, H]
    ks_ptr,  # int32 [M]
    ke_ptr,  # int32 [M]
    logits_ptr,  # fp32 [M, N]
    M,
    N,
    stride_q_m,
    stride_q_h,
    stride_k_n,
    stride_w_m,
    stride_logits_m,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N)): one [BLOCK_M, BLOCK_N] tile
    # of logits per program, reduced over the heads.
    n_start = tl.program_id(1) * BLOCK_N
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    d = tl.arange(0, HEAD_SIZE)
    m_ok = offs_m < M
    n_ok = offs_n < N
    ks = tl.load(ks_ptr + offs_m, mask=m_ok, other=0)
    ke = tl.load(ke_ptr + offs_m, mask=m_ok, other=0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    # Skip the head loop for tiles outside every row's [ks, ke) window.
    lo = tl.min(tl.where(m_ok, ks, N))
    hi = tl.max(tl.where(m_ok, ke, 0))
    if (hi > n_start) & (lo < n_start + BLOCK_N):
        k = tl.load(
            k_ptr + offs_n.to(tl.int64)[:, None] * stride_k_n + d[None, :],
            mask=n_ok[:, None],
            other=0.0,
        )  # [BLOCK_N, D] fp8
        k_scale = tl.load(k_scale_ptr + offs_n, mask=n_ok, other=0.0)
        q_rows = q_ptr + offs_m.to(tl.int64)[:, None] * stride_q_m + d[None, :]
        for h in range(NUM_HEADS):
            q = tl.load(
                q_rows + h * stride_q_h, mask=m_ok[:, None], other=0.0
            )  # [BLOCK_M, D] fp8
            w = tl.load(weights_ptr + offs_m * stride_w_m + h, mask=m_ok, other=0.0)
            # fp8 products are exact in fp32; accumulate in fp32.
            s = tl.dot(q, tl.trans(k)) * k_scale[None, :]
            acc += tl.maximum(s, 0.0) * w[:, None]

    in_range = (offs_n[None, :] >= ks[:, None]) & (offs_n[None, :] < ke[:, None])
    acc = tl.where(in_range, acc, float("-inf"))
    tl.store(
        logits_ptr + offs_m.to(tl.int64)[:, None] * stride_logits_m + offs_n[None, :],
        acc,
        mask=m_ok[:, None] & n_ok[None, :],
    )


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Triton FP8 MQA logits over a gathered (non-paged) K.

    Same contract as `fp8_mqa_logits_torch`: q is FP8 [M, H, D], kv is
    `(k_fp8 [N, D], k_scales [N] or [N, 1] fp32)`, weights are fp32 [M, H] and
    each row m keeps key columns [cu_seqlen_ks[m], cu_seqlen_ke[m]). Returns
    fp32 [M, N] logits with -inf outside each row's range.
    """
    k_fp8, k_scale = kv
    num_rows, num_heads, head_size = q.shape
    num_keys = k_fp8.shape[0]
    assert q.stride(-1) == 1 and k_fp8.stride(-1) == 1
    k_scale = k_scale.reshape(-1).contiguous()
    logits = torch.empty((num_rows, num_keys), dtype=torch.float32, device=q.device)
    if num_rows == 0 or num_keys == 0:
        return logits
    BLOCK_M, BLOCK_N = 32, 64
    grid = (cdiv(num_rows, BLOCK_M), cdiv(num_keys, BLOCK_N))
    _fp8_mqa_logits_kernel[grid](
        q,
        k_fp8,
        k_scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        num_rows,
        num_keys,
        q.stride(0),
        q.stride(1),
        k_fp8.stride(0),
        weights.stride(0),
        logits.stride(0),
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return logits
