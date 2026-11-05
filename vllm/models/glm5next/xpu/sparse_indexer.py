# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kpool sparse-attention indexer for Intel XPU.

Slimmed copy of ``nvidia/sparse_indexer.py``: the MQA logits come from the XPU
``fp8_mqa_logits`` / ``fp8_paged_mqa_logits`` ops (the ones the DeepSeek-V3.2
indexer uses on XPU) instead of DeepGEMM, and the CUDA-only paths (MXFP4 index
cache, fused MQA top-k, DCP top-k merge) are left out.
"""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.models.glm5next.common.sparse_indexer import (
    _build_decode_scatter_indices,
    _decode_topk_seq_lens,
    _fill_causal_indices,
    _gather_workspace_shapes,
    _scatter_decode_tokens_by_request,
    kv_cache_as_quant_view,
)
from vllm.models.glm5next.xpu.ops import kpool_compress as kpool_ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    LayerNameType,
    _resolve_layer_name,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    dense_logits_rows,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager


def _kpool_compress_insert(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
    round_scale: bool,
) -> None:
    """Pool ``kpool`` consecutive tokens into one fp8 K and write at pool slots.

    ``slot_mapping`` is pool-granular: only the last token of each complete
    pool carries a valid (>=0) slot; non-completions are masked in the kernel.
    """
    n = slot_mapping.shape[0]
    if n < kpool:
        return
    pos = torch.arange(n, device=k.device)
    write_mask = (slot_mapping >= 0) & (pos >= kpool - 1)
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


@eager_break_during_capture
def sparse_attn_indexer_kpool_xpu(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_pool_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
    topk_backend: str = "auto",
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    if not isinstance(attn_metadata, dict):
        # Profiling run: reserve the prefill gather workspace and the
        # worst-case FP32 logits so the KV-cache budget accounts for them.
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, False
        )
        current_workspace_manager().get_simultaneous(values_spec, scales_spec)
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
                sched.max_num_seqs * (num_spec + 1), sched.max_num_batched_tokens
            )
        bytes_per_elem = torch.float32.itemsize
        decode_logits_bytes = worst_decode_tokens * max_pool_len * bytes_per_elem
        prefill_cap_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max(decode_logits_bytes, prefill_cap_bytes),
            dtype=torch.uint8,
            device=hidden_states.device,
        )
        return topk_indices_buffer

    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # k may be padded to the graph batch size; slot_mapping covers real tokens.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        if index_kpool > 1 and gate_score is not None and compress_ape is not None:
            # Prefill tokens (batched after the decode tokens) are pooled here;
            # decode tokens complete their pools in the tail-cache kernel below.
            n_prefill = num_tokens - num_decode_tokens
            if n_prefill > 0:
                prefill_slice = slice(num_decode_tokens, num_tokens)
                _kpool_compress_insert(
                    k[prefill_slice],
                    gate_score[prefill_slice],
                    compress_ape,
                    kv_cache,
                    slot_mapping[prefill_slice],
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
                # Persist each request's incomplete prefill pool for decode.
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
        else:
            assert scale_fmt is not None
            ops.indexer_k_quant_and_cache(
                k, kv_cache, slot_mapping, quant_block_size, scale_fmt
            )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Short sequences select every pool: fill the causal token indices.
        n_prefill_sf = num_tokens - num_decode_tokens
        if prefill_metadata.max_prefill_seq_len >= 0:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and prefill_metadata.max_prefill_seq_len <= topk_tokens
            )
        else:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and int(positions[num_decode_tokens:num_tokens].max().item()) + 1
                <= topk_tokens
            )
        if short_prefill:
            assert positions is not None
            _fill_causal_indices(
                topk_indices_buffer[num_decode_tokens:num_tokens],
                positions[num_decode_tokens:num_tokens].to(torch.int32),
            )
        # Under ShardQ the chunks index the shardq_gather_indices tokens.
        prefill_q, prefill_weights, prefill_positions = q_quant, weights, positions
        if (indices := prefill_metadata.shardq_gather_indices) is not None:
            prefill_q, prefill_weights = q_quant[indices], weights[indices]
            if positions is not None:
                prefill_positions = positions[indices]
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, False
        )
        k_quant_full, k_scale_full = current_workspace_manager().get_simultaneous(
            values_spec, scales_spec
        )
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
        for chunk in prefill_metadata.chunks if not short_prefill else ():
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )
            k_scale = k_scale.view(torch.float32).squeeze(-1)

            row_start, row_end = chunk.token_start, chunk.token_end
            cu_seqlen_ks, cu_seqlen_ke = chunk.cu_seqlen_ks, chunk.cu_seqlen_ke
            q_slice = prefill_q[row_start:row_end]
            num_rows = q_slice.shape[0]
            # kpool: logits are pool-granular, so top-k selects pools that are
            # expanded back to their tokens below.
            if index_kpool > 1:
                pool_topk = torch.empty(
                    (num_rows, select_k), dtype=torch.int32, device=q_slice.device
                )
                topk_dst = pool_topk
            else:
                topk_dst = topk_indices_buffer[row_start:row_end, :topk_tokens]

            if chunk.local_total_seq_lens == 0:
                topk_dst.fill_(-1)
            else:
                # Rows per logits call under the logits budget.
                dense_rows = dense_logits_rows(chunk.max_seqlen_k)
                for sub in range(0, num_rows, dense_rows):
                    rows = slice(sub, min(sub + dense_rows, num_rows))
                    # Uncompressed FP32 logits over the whole gathered K buffer;
                    # row i is valid on [cu_seqlen_ks[i], cu_seqlen_ke[i]).
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice[rows],
                        k_quant,
                        k_scale,
                        prefill_weights[row_start + rows.start : row_start + rows.stop],
                        cu_seqlen_ks[rows],
                        cu_seqlen_ke[rows],
                    )
                    # Indices come back relative to each row's start, i.e.
                    # request-relative pool ids.
                    torch.ops._C.top_k_per_row_prefill(
                        logits,
                        cu_seqlen_ks[rows],
                        cu_seqlen_ke[rows],
                        topk_dst[rows],
                        logits.shape[0],
                        logits.stride(0),
                        logits.stride(1),
                        select_k,
                    )
                    del logits

            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                if prefill_positions is not None:
                    q_seq = prefill_positions[row_start:row_end].to(torch.int32) + 1
                    expanded = kpool_ops.expand_pools_and_append_tail(
                        pool_ids, q_seq, index_kpool
                    )
                else:
                    expanded = kpool_ops.expand_pools_to_tokens(
                        pool_ids, pool_ids >= 0, topk_tokens, index_kpool
                    )
                topk_indices_buffer[row_start:row_end, : expanded.shape[-1]] = expanded

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache_raw = kv_cache  # raw [num_blocks, block_size, head_dim + 4]
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, False)

        # Update the tail before reading logits; completed pools are compressed
        # into the slot supplied by slot_mapping. Tokens are grouped per request
        # in position order so each token is stashed before the next completes
        # its pool.
        if (
            index_kpool > 1
            and gate_score is not None
            and compress_ape is not None
            and positions is not None
            and not skip_k_cache_insert
        ):
            num_requests = attn_metadata_narrowed.num_decodes
            per_req_lens = decode_metadata.per_req_decode_lens
            if per_req_lens is not None:
                use_uniform = (
                    decode_metadata.decode_is_uniform
                    and num_decode_tokens
                    == num_requests * decode_metadata.write_max_decode_len
                )
                group_lens = per_req_lens
                lmax = decode_metadata.write_max_decode_len
            else:
                use_uniform = not decode_metadata.requires_padding
                group_lens = decode_metadata.decode_lens
                lmax = int(decode_metadata.decode_lens.max().item())
            tail_meta = (
                attn_metadata.get(_resolve_layer_name(tail_prefix))
                if tail_prefix is not None
                else None
            )
            if tail_meta is not None:
                assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
            if not use_uniform:
                scatter_idx = _build_decode_scatter_indices(
                    group_lens, num_requests, num_decode_tokens
                )

                def _by_request(t: torch.Tensor, pad) -> torch.Tensor:
                    return _scatter_decode_tokens_by_request(
                        t[:num_decode_tokens], pad, num_requests, lmax, scatter_idx
                    )

                dec_k = _by_request(k, 0)
                dec_gate = _by_request(gate_score, 0)
                dec_slot = _by_request(slot_mapping, -1)
                dec_pos = _by_request(positions.to(torch.int32), -1)
                dec_tail_slot = (
                    _by_request(tail_meta.slot_mapping, -1)
                    if tail_meta is not None and tail_kv_cache is not None
                    else None
                )
            else:
                shape2 = (num_requests, num_decode_tokens // num_requests)
                dec_k = k[:num_decode_tokens].view(*shape2, head_dim)
                dec_gate = gate_score[:num_decode_tokens].view(*shape2, head_dim)
                dec_slot = slot_mapping[:num_decode_tokens].view(shape2)
                dec_pos = positions[:num_decode_tokens].to(torch.int32).view(shape2)
                dec_tail_slot = (
                    tail_meta.slot_mapping[:num_decode_tokens].view(shape2)
                    if tail_meta is not None and tail_kv_cache is not None
                    else None
                )
            if dec_tail_slot is not None:
                assert tail_kv_cache is not None
                kpool_ops.kpool_decode_update_and_maybe_write_cache_batched(
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
                    round_scale=(scale_fmt is not None),
                )

        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # Padding also covers short chunked prefills classified as decode.
            padded_q = pack_seq_triton(q_quant[:num_decode_tokens], decode_lens)
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            ).reshape(-1, *weights.shape[1:])
        else:
            padded_q = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            padded_weights = weights[:num_decode_tokens]
        batch_size, next_n = padded_q.shape[0], padded_q.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # One pool-granular context length per request; FP32 logits
        # [B * next_n, max_pool_len].
        logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
            padded_q,
            kv_cache,
            padded_weights[:num_padded_tokens],
            seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_pool_len,
        )
        num_rows = logits.shape[0]
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
        if index_kpool > 1:
            pool_topk = torch.empty(
                (num_rows, select_k), dtype=torch.int32, device=logits.device
            )
            topk_dst = pool_topk
        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        get_indexer_topk(topk_backend)(
            logits,
            seq_lens,
            next_n,
            topk_dst,
            select_k,
            attn_metadata_narrowed.max_seq_len,
        )

        if index_kpool > 1:
            # Decode seq_lens are pool-granular; recover token lengths from
            # positions using the padded [B, next_n] row layout when needed.
            if positions is not None:
                dec_seq = _decode_topk_seq_lens(
                    positions,
                    decode_lens,
                    num_decode_tokens,
                    batch_size,
                    next_n,
                    decode_metadata.requires_padding,
                )
            else:
                dec_seq = decode_metadata.seq_lens[: pool_topk.shape[0]]
                if dec_seq.ndim == 2:
                    dec_seq = dec_seq[:, -1]
                dec_seq = dec_seq.to(torch.int32)
            out = kpool_ops.expand_pools_and_append_tail(
                pool_topk.to(torch.int64), dec_seq, index_kpool
            )
        else:
            out = topk_dst

        if decode_metadata.requires_padding:
            # Drop padded query rows introduced by the next_n padding above.
            out = unpack_seq_triton(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out

    return topk_indices_buffer


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Kpool sparse-attention indexer layer for XPU (same interface as the
    ``nvidia/`` and ``amd/`` layers)."""

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_pool_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ):
        super().__init__()
        if use_fp4_cache:
            raise NotImplementedError(
                "The MXFP4 indexer cache is not supported on XPU."
            )
        _cfg = get_current_vllm_config_or_none()
        if _cfg is not None:
            if _cfg.parallel_config.decode_context_parallel_size > 1:
                raise NotImplementedError(
                    "SparseAttnIndexerKpool does not support DCP on XPU."
                )
            # The XPU MQA-logits ops produce FP32 logits only.
            logits_dtype = _cfg.attention_config.resolve_indexer_logits_dtype("float32")
            if STR_DTYPE_TO_TORCH_DTYPE[logits_dtype] != torch.float32:
                raise NotImplementedError(
                    f"indexer_logits_dtype={logits_dtype} is not supported on XPU."
                )
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_pool_len = max_pool_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.topk_backend = (
            _cfg.kernel_config.sparse_indexer_topk_backend
            if _cfg is not None
            else "auto"
        )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        return self.forward_xpu(
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        # FP8 path only: the per-token Q scale is folded into ``weights``.
        if isinstance(q_quant, tuple):
            raise NotImplementedError("FP4 indexer queries are not supported on XPU.")
        return sparse_attn_indexer_kpool_xpu(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q_quant,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_pool_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            self.tail_cache.kv_cache if self.tail_cache is not None else None,
            self.tail_cache.prefix if self.tail_cache is not None else None,
            self.topk_backend,
        )
