# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendor-neutral steps of the glm5next kpool sparse indexer.

The kpool indexer pools `index_kpool` consecutive tokens into one compressed
index-K entry, scores pools instead of tokens, and expands the selected pools
back to token indices (plus the request's incomplete tail pool). Everything
except the logits and top-k kernels is the same on every platform and lives
here; a vendor indexer (`nvidia/`, `xpu/`) provides the logits/top-k calls and
composes these steps.
"""

from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.models.glm5next.common.ops import kpool_compress as kpool_ops
from vllm.models.glm5next.common.sparse_indexer import (
    _build_decode_scatter_indices,
    _decode_topk_seq_lens,
    _fill_causal_indices,
    _scatter_decode_tokens_by_request,
)
from vllm.utils.torch_utils import _resolve_layer_name
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata


def kpool_compress_insert(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
    round_scale: bool,
) -> None:
    """Pool `kpool` consecutive tokens into one fp8 K and write at pool slots.

    `slot_mapping` is pool-granular (compress_ratio == kpool on the spec):
    only the *last* token of each complete pool carries a valid (>=0) slot;
    intra-pool tokens are -1. Every position is treated as a pool-completion
    candidate and non-completions are masked off inside the kernel. Compacting
    the valid rows first costs two device syncs on the eager prefill path and
    buys nothing numerically. Assumes pool-aligned chunk starts.
    """
    n = slot_mapping.shape[0]
    # No pool can complete in a batch smaller than one pool; also keeps the
    # clamped gather indices below in bounds.
    if n < kpool:
        return
    pos = torch.arange(n, device=k.device)
    valid = slot_mapping >= 0
    # Drop pools whose start falls before the batch (leading padding); their
    # gate/k data is undefined anyway.
    write_mask = valid & (pos >= kpool - 1)
    offs = torch.arange(kpool, device=k.device)
    idx = (pos - (kpool - 1)).clamp_min(0)[:, None] + offs[None, :]
    kpool_ops.kpool_compress_and_write_cache(
        kv_cache,
        k[idx],  # [n, kpool, head_dim]
        gate_score[idx],
        ape,
        slot_mapping.to(torch.int64),
        pool_size=kpool,
        head_dim=head_dim,
        write_mask=write_mask,
        round_scale=round_scale,
        write_cache=True,
        return_compressed=False,
    )


def reserve_profiling_logits(
    hidden_states: torch.Tensor,
    max_pool_len: int,
    logits_dtype: torch.dtype,
) -> None:
    """Reserve profiler-visible memory for the worst-case indexer logits.

    Decode logits are [B * next_n, max_pool_len]; the profiling run returns
    before invoking the logits kernel itself.
    """
    cfg = get_current_vllm_config_or_none()
    worst_decode_tokens = 0
    if cfg is not None:
        sched = cfg.scheduler_config
        num_spec = (
            cfg.speculative_config.num_speculative_tokens
            if cfg.speculative_config is not None
            else 0
        )
        worst_decode_tokens = min(
            sched.max_num_seqs * (num_spec + 1),
            sched.max_num_batched_tokens,
        )
    # FP32 logits, plus the kernel's output when it emits BF16; uint8
    # sentinel so elems == bytes.
    bytes_per_elem = torch.float32.itemsize
    if logits_dtype != torch.float32:
        bytes_per_elem += logits_dtype.itemsize
    decode_logits_elems = worst_decode_tokens * max_pool_len * bytes_per_elem
    prefill_cap_elems = (
        envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
        * 1024
        * 1024
        // torch.float32.itemsize
        * bytes_per_elem
    )
    max_logits_elems = max(decode_logits_elems, prefill_cap_elems)
    _ = torch.empty(max_logits_elems, dtype=torch.uint8, device=hidden_states.device)


def kpool_prefill_insert(
    attn_metadata: dict[str, Any],
    k: torch.Tensor,
    gate_score: torch.Tensor,
    compress_ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    num_tokens: int,
    index_kpool: int,
    head_dim: int,
    round_scale: bool,
    tail_kv_cache: torch.Tensor | None,
    tail_prefix: str | None,
) -> None:
    """Kpool prefill write: pool kpool consecutive prefill tokens via
    softmax(gate+ape)-weighted sum -> Hadamard -> fp8 -> pool slots, then
    persist each request's incomplete prefill pool in the paged tail cache.

    Decode tokens (the first num_decode_tokens in the batch) cannot be pooled
    here -- their pool's earlier tokens are not in this batch -- so they are
    deferred to the tail-buffer kernel (`kpool_decode_tail_update`).
    compress_ratio == index_kpool makes slot_mapping pool-granular.
    """
    n_prefill = num_tokens - num_decode_tokens
    if n_prefill <= 0:
        return
    # decode tokens are batched first; prefill tokens follow.
    prefill_slice = slice(num_decode_tokens, num_tokens)
    kpool_compress_insert(
        k[prefill_slice],
        gate_score[prefill_slice],
        compress_ape,
        kv_cache,
        slot_mapping[prefill_slice],
        index_kpool,
        head_dim,
        round_scale=round_scale,
    )
    # Persist each request's incomplete prefill pool so decode can finish it,
    # including after PD transfer. Tail slots use ``pos % kpool`` within the
    # request's tail block. Processing only the batch's trailing tokens would
    # miss all but the last request in a multi-request prefill.
    if tail_kv_cache is not None and tail_prefix is not None:
        tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
        if tail_meta is not None:
            assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
            kpool_ops.kpool_seed_tail_cache(
                tail_kv_cache,
                k[prefill_slice],
                gate_score[prefill_slice],
                tail_meta.slot_mapping[prefill_slice],
                index_kpool,
                head_dim,
            )


def fill_short_prefill(
    prefill_metadata: Any,
    positions: torch.Tensor | None,
    num_decode_tokens: int,
    num_tokens: int,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
) -> bool:
    """Short sequences select every pool: fill the prefill rows of the top-k
    buffer with all causal token indices and return True (sparse scoring can
    be skipped). The index-K cache must already be written.
    """
    n_prefill = num_tokens - num_decode_tokens
    # Host-side short-prefill predicate: max_prefill_seq_len is computed in
    # the metadata builder (exact for prefill rows) and equals
    # positions[prefill_slice].max() + 1, so this replaces a
    # positions.max().item() device sync per layer. -1 (unknown metadata)
    # falls back to the device-side check.
    if prefill_metadata.max_prefill_seq_len >= 0:
        short_prefill = (
            n_prefill > 0
            and positions is not None
            and prefill_metadata.max_prefill_seq_len <= topk_tokens
        )
    else:
        short_prefill = (
            n_prefill > 0
            and positions is not None
            and int(positions[num_decode_tokens:num_tokens].max().item()) + 1
            <= topk_tokens
        )
    if short_prefill:
        assert positions is not None
        _pos = positions[num_decode_tokens:num_tokens].to(torch.int32)
        _buf = topk_indices_buffer[num_decode_tokens:num_tokens]
        _fill_causal_indices(_buf, _pos)
    return short_prefill


def expand_prefill_pools(
    pool_topk: torch.Tensor,
    prefill_positions: torch.Tensor | None,
    row_start: int,
    row_end: int,
    topk_tokens: int,
    index_kpool: int,
    topk_indices_buffer: torch.Tensor,
) -> None:
    """Expand a prefill chunk's selected pools to token indices (plus the
    request's tail pool) into topk_indices_buffer[row_start:row_end]."""
    pool_ids = pool_topk.to(torch.int64)
    if prefill_positions is not None:
        # Fused expand-pools + append-tail into one Triton kernel (replaces
        # ~25 elementwise ops). seq_len is token-granular (pos+1); the kernel
        # derives pool_len internally.
        q_seq = prefill_positions[row_start:row_end].to(torch.int32) + 1
        expanded = kpool_ops.expand_pools_and_append_tail(pool_ids, q_seq, index_kpool)
    else:
        valid = pool_ids >= 0
        expanded = kpool_ops.expand_pools_to_tokens(
            pool_ids, valid, topk_tokens, index_kpool
        )
    topk_indices_buffer[row_start:row_end, : expanded.shape[-1]] = expanded


def kpool_decode_tail_update(
    attn_metadata: dict[str, Any],
    decode_metadata: Any,
    num_requests: int,
    num_decode_tokens: int,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    compress_ape: torch.Tensor,
    kv_cache_raw: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    index_kpool: int,
    head_dim: int,
    round_scale: bool,
    tail_kv_cache: torch.Tensor | None,
    tail_prefix: str | None,
    update_fn: Callable[..., None] | None = None,
) -> None:
    """Advance each decode request's tail pool before the logits are read;
    completed pools are compressed into the slot supplied by slot_mapping.

    Spec verification groups tokens by request and preserves position order
    so each token is stashed before the next completes its pool. Positions
    must remain token-granular because the kernel derives the pool phase and
    tail index from `pos % kpool`. `update_fn` replaces the Triton kernel
    (same signature), e.g. a vendor's fused decode stage.
    """
    # Kpool writes must recover the original request grouping after the
    # indexer's flattened decode path. Host metadata avoids a graph sync when
    # choosing the uniform or padded layout.
    per_req_lens = decode_metadata.per_req_decode_lens
    if per_req_lens is not None:
        use_uniform = (
            decode_metadata.decode_is_uniform
            and num_decode_tokens == num_requests * decode_metadata.write_max_decode_len
        )
        group_lens = per_req_lens
        lmax = decode_metadata.write_max_decode_len
    else:
        # Legacy metadata without per-request lens: fall back to the host-side
        # requires_padding flag. Unreached now (per-request lens is always
        # populated for decode), kept defensive.
        use_uniform = not decode_metadata.requires_padding
        group_lens = decode_metadata.decode_lens
        lmax = int(decode_metadata.decode_lens.max().item())
    if not use_uniform:
        # Non-uniform decode_lens (mixed plain-decode + spec-verify, or a
        # variable MTP-verify batch): scatter actual tokens into a padded
        # [B, lmax] layout. int32 tensors can't go through pack_seq_triton
        # (float/uint8 only). The scatter indices are shared by all five
        # scatters below (and the tail slot one).
        scatter_idx = _build_decode_scatter_indices(
            group_lens, num_requests, num_decode_tokens
        )
        dec_k = _scatter_decode_tokens_by_request(
            k[:num_decode_tokens], 0, num_requests, lmax, scatter_idx
        )
        dec_gate = _scatter_decode_tokens_by_request(
            gate_score[:num_decode_tokens], 0, num_requests, lmax, scatter_idx
        )
        dec_slot = _scatter_decode_tokens_by_request(
            slot_mapping[:num_decode_tokens], -1, num_requests, lmax, scatter_idx
        )
        dec_pos = _scatter_decode_tokens_by_request(
            positions[:num_decode_tokens].to(torch.int32),
            -1,
            num_requests,
            lmax,
            scatter_idx,
        )
    else:
        next_n = num_decode_tokens // num_requests
        shape2 = (num_requests, next_n)
        dec_k = k[:num_decode_tokens].view(*shape2, head_dim)
        dec_gate = gate_score[:num_decode_tokens].view(*shape2, head_dim)
        dec_slot = slot_mapping[:num_decode_tokens].view(shape2)
        dec_pos = positions[:num_decode_tokens].view(shape2)
    tail_meta = (
        attn_metadata.get(_resolve_layer_name(tail_prefix))
        if tail_prefix is not None
        else None
    )
    # Group the tail group's token-granular slot_mapping per request,
    # mirroring dec_slot / dec_pos, so the kernel gets each request's
    # current-token tail slot (block * kpool + pos % kpool).
    if tail_meta is not None:
        assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
    if tail_meta is None or tail_kv_cache is None:
        return
    if not use_uniform:
        dec_tail_slot = _scatter_decode_tokens_by_request(
            tail_meta.slot_mapping[:num_decode_tokens],
            -1,
            num_requests,
            lmax,
            scatter_idx,
        )
    else:
        dec_tail_slot = tail_meta.slot_mapping[:num_decode_tokens].view(shape2)
    # Single batched launch over [num_requests, next_n]; the kernel iterates
    # each request's tokens in position order, preserving the pool-completion
    # read-after-stash dependency. The compress kernel writes the raw fp8
    # cache (not the quant view).
    (update_fn or kpool_ops.kpool_decode_update_and_maybe_write_cache_batched)(
        kv_cache_raw,
        tail_kv_cache,
        dec_tail_slot,
        dec_k,
        dec_gate,
        compress_ape,
        dec_slot,
        dec_pos,
        index_kpool,
        head_dim,
        round_scale=round_scale,
    )


def decode_pool_seq_lens(
    positions: torch.Tensor | None,
    decode_metadata: Any,
    decode_lens: torch.Tensor,
    num_decode_tokens: int,
    batch_size: int,
    next_n: int,
    n: int,
) -> torch.Tensor:
    """Token-granular seq_len per decode pool-topk row (padded layout aware).

    Decode seq_lens are pool-granular; recover token lengths from positions
    using the padded [B, next_n] row layout when needed.
    """
    if positions is not None:
        return _decode_topk_seq_lens(
            positions,
            decode_lens,
            num_decode_tokens,
            batch_size,
            next_n,
            decode_metadata.requires_padding,
        )
    dec_seq = decode_metadata.seq_lens[:n]
    if dec_seq.ndim == 2:
        dec_seq = dec_seq[:, -1]
    return dec_seq.to(torch.int32)
