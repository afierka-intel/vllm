# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

from typing import Any

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
    fused_mqa_topk_chunk,
    fused_mqa_topk_key_scales,
    fused_mqa_topk_prefill,
    get_fused_mqa_topk_workspaces,
    use_fused_mqa_topk,
)
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.model_executor.layers.sparse_attn_indexer import (
    _merge_dcp_topk_global,
    as_fp32_logits,
)
from vllm.models.glm5next.common.kpool_indexer import (
    decode_pool_seq_lens,
    expand_prefill_pools,
    fill_short_prefill,
    kpool_decode_tail_update,
    kpool_prefill_insert,
    reserve_profiling_logits,
)
from vllm.models.glm5next.common.ops import kpool_compress as kpool_ops
from vllm.models.glm5next.common.sparse_indexer import (
    RADIX_TOPK_WORKSPACE_SIZE,
    _fill_short_decode_causal_indices,
    _gather_workspace_shapes,
    kv_cache_as_quant_view,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
    mqa_logits_mx_inputs,
    mqa_logits_use_mx_scales,
)
from vllm.utils.import_utils import has_cutedsl
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

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops

logger = init_logger(__name__)

# The specialized decode's kpool stage (specialized_decode/kpool.py), when
# installed: CuTe kernels for the decode kernels below inside planned steps.
SPECIALIZED_KPOOL: Any = None


@eager_break_during_capture
def sparse_attn_indexer_kpool(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
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
    use_fp4_cache: bool = False,
    # kpool params (Plan-A: gate is consumed at write time and read back at
    # topk time to softmax-weight the pool).
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    # Paged tail cache (in-progress pool's raw K + gate score), replacing the
    # transient _DECODE_TAIL ring. tail_prefix resolves attn_metadata[tail_prefix]
    # for the tail group's token-granular slot_mapping. None on the dummy/profiling
    # path and when the tail cache is disabled.
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
    topk_backend: str = "auto",
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    logits_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    fused_mqa_topk = use_fused_mqa_topk(
        query_shape=q_quant.shape[1:],
        topk=topk_tokens // max(index_kpool, 1),
        fp4_cache=use_fp4_cache,
        full_context=dcp_world_size == 1,
    )

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )
        # Allocate the largest fused_mqa_topk workspaces so they are accounted
        # for (freed again, they overlap the dense logits below).
        if fused_mqa_topk:
            get_fused_mqa_topk_workspaces(q_quant.device, q_quant.shape[0])

        reserve_profiling_logits(hidden_states, max_pool_len, logits_dtype)

        return topk_indices_buffer
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens
    # Pools are the DCP sharding unit of the index K cache.
    pool_cp_interleave = (
        cp_kv_cache_interleave_size // max(index_kpool, 1)
        if dcp_world_size > 1
        else cp_kv_cache_interleave_size
    )

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        if index_kpool > 1 and gate_score is not None and compress_ape is not None:
            kpool_prefill_insert(
                attn_metadata,
                k,
                gate_score,
                compress_ape,
                kv_cache,
                slot_mapping,
                num_decode_tokens,
                num_tokens,
                index_kpool,
                head_dim,
                round_scale=(scale_fmt is not None),
                tail_kv_cache=tail_kv_cache,
                tail_prefix=tail_prefix,
            )
        else:
            # standard: per-token fp8 quant + scatter (all tokens).
            assert scale_fmt is not None
            ops.indexer_k_quant_and_cache(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                "mxfp8" if mqa_logits_use_mx_scales() else scale_fmt,
            )

    kpool_stage = SPECIALIZED_KPOOL
    if kpool_stage is not None and not kpool_stage.active():
        kpool_stage = None
    # Uniform kpool decode rows can leave their expansion to the layer's index
    # remap (the stage's fused kernel), which needs no -1 filled logical rows.
    defer_topk = (
        kpool_stage is not None
        and has_decode
        and not has_prefill
        and index_kpool > 1
        and positions is not None
        and dcp_world_size == 1
        and attn_metadata_narrowed.decode is not None
        and not attn_metadata_narrowed.decode.requires_padding
        and attn_metadata_narrowed.max_seq_len > topk_tokens
        and kpool_stage.defers(
            k_cache_prefix,
            num_decode_tokens,
            topk_tokens,
            index_kpool,
            topk_indices_buffer.shape[1],
        )
    )
    if not defer_topk:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Short sequences select every pool, so skip sparse scoring and fill
        # the top-k buffer with all causal token indices. The index-K cache was
        # already written above.
        short_prefill = fill_short_prefill(
            prefill_metadata,
            positions,
            num_decode_tokens,
            num_tokens,
            topk_tokens,
            topk_indices_buffer,
        )
        # Under ShardQ the chunks index the shardq_gather_indices tokens.
        prefill_q, prefill_q_scale = q_quant, q_scale
        prefill_weights, prefill_positions = weights, positions
        if (indices := prefill_metadata.shardq_gather_indices) is not None:
            prefill_q, prefill_weights = q_quant[indices], weights[indices]
            if q_scale is not None:
                prefill_q_scale = q_scale[indices]
            if positions is not None:
                prefill_positions = positions[indices]
        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks if not short_prefill else ():
            # Under DCP this rank only gathers the pools it owns.
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

            row_start, row_end = chunk.token_start, chunk.token_end
            cu_seqlen_ks, cu_seqlen_ke = chunk.cu_seqlen_ks, chunk.cu_seqlen_ke

            q_slice = prefill_q[row_start:row_end]
            q_scale_slice = (
                prefill_q_scale[row_start:row_end]
                if prefill_q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(
                    torch.int32 if mqa_logits_use_mx_scales() else torch.float32
                ).squeeze(-1)

            num_rows = q_slice.shape[0]
            # kpool: logits are pool-granular (compress_ratio == index_kpool),
            # so topk selects pools. We pick topk_tokens // kpool pools then
            # expand each pool back to its kpool constituent tokens.
            select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
            if index_kpool > 1:
                pool_topk = torch.empty(
                    (num_rows, select_k), dtype=torch.int32, device=q_slice.device
                )
                topk_dst = pool_topk
            else:
                topk_dst = topk_indices_buffer[row_start:row_end, :topk_tokens]

            # Rows per dense logits call under the logits budget.
            dense_rows = dense_logits_rows(chunk.max_seqlen_k)
            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((num_rows, 0), dtype=torch.float32)
                topk_dst.fill_(-1)
            elif fused_mqa_topk and fused_mqa_topk_chunk(
                chunk.max_seqlen_k, k_quant.shape[0], select_k, num_rows, dense_rows
            ):
                fused_mqa_topk_prefill(
                    q_slice,
                    k_quant,
                    fused_mqa_topk_key_scales(k_scale),
                    prefill_weights[row_start:row_end],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_dst,
                )
                logits = q_slice.new_empty((num_rows, 0), dtype=torch.float32)
            else:
                # Chunks planned for fused_mqa_topk are not bounded by the logits
                # budget: split their rows (budget-planned chunks, the only ones
                # under DCP, are one window, so the merge below sees all the
                # rows).
                # One pass also for 0 rows: the merge is collective.
                for sub in range(0, max(1, num_rows), dense_rows):
                    rows = slice(sub, min(sub + dense_rows, num_rows))
                    q_scale_rows, chunk_weights = mqa_logits_mx_inputs(
                        q_slice_cast[rows],
                        q_scale_slice[rows] if q_scale_slice is not None else None,
                        prefill_weights[row_start + rows.start : row_start + rows.stop],
                        logits_dtype,
                    )
                    # Compressed: row i starts at K position cu_seqlen_ks[i].
                    row_ends = cu_seqlen_ke[rows] - cu_seqlen_ks[rows]
                    logits = as_fp32_logits(
                        fp8_fp4_mqa_logits(
                            (q_slice_cast[rows], q_scale_rows),
                            (k_quant_cast, k_scale_cast),
                            chunk_weights,
                            cu_seqlen_ks[rows],
                            cu_seqlen_ke[rows],
                            chunk.max_seqlen_k,
                            logits_dtype=logits_dtype,
                        ),
                        row_ends,
                    )
                    if logits.shape[1] == 0:
                        topk_dst[rows].fill_(-1)
                    else:
                        get_indexer_topk(topk_backend).forward_prefill(
                            logits, None, row_ends, topk_dst[rows], select_k
                        )
            # Local pool ids -> request-relative global pool ids.
            _merge_dcp_topk_global(
                logits,
                topk_dst,
                select_k,
                dcp_rank,
                dcp_world_size,
                pool_cp_interleave,
                row_starts=None,
                logits_dtype=logits_dtype,
                num_positions=max_pool_len,
            )

            if index_kpool > 1:
                expand_prefill_pools(
                    pool_topk,
                    prefill_positions,
                    row_start,
                    row_end,
                    topk_tokens,
                    index_kpool,
                    topk_indices_buffer,
                )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache_raw = kv_cache  # raw [num_blocks, block_size, head_dim+4] for writes
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)

        # Update the tail before reading logits; completed pools are compressed
        # into the slot supplied by slot_mapping.
        if (
            index_kpool > 1
            and gate_score is not None
            and compress_ape is not None
            and positions is not None
            and not skip_k_cache_insert
        ):
            kpool_decode_tail_update(
                attn_metadata,
                decode_metadata,
                attn_metadata_narrowed.num_decodes,
                num_decode_tokens,
                k,
                gate_score,
                compress_ape,
                kv_cache_raw,
                slot_mapping,
                positions,
                index_kpool,
                head_dim,
                round_scale=(scale_fmt is not None),
                tail_kv_cache=tail_kv_cache,
                tail_prefix=tail_prefix,
                update_fn=None if kpool_stage is None else kpool_stage.update,
            )
        if current_platform.is_cuda_alike() and _fill_short_decode_causal_indices(
            topk_indices_buffer,
            positions,
            num_decode_tokens,
            attn_metadata_narrowed.max_seq_len,
            topk_tokens,
        ):
            return topk_indices_buffer
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # Padding also covers short chunked prefills classified as decode.
            # MXFP4 uses zero-byte padding so padded slots dequantize to zero.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            ).reshape(-1, *weights.shape[1:])
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
            padded_weights = weights[:num_decode_tokens]
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        padded_q_scale, decode_weights = mqa_logits_mx_inputs(
            padded_q_quant_cast,
            padded_q_scale,
            padded_weights[:num_padded_tokens],
            logits_dtype,
        )
        # SM100 decode is varlen: one row and context length per token.
        logits = (as_fp32_logits if kpool_stage is None else kpool_stage.upcast)(
            fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                decode_weights,
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_pool_len,
                logits_dtype=logits_dtype,
                indices=decode_metadata.indices,
            ),
            seq_lens.view(-1),
        )
        num_rows = logits.shape[0]
        # kpool: logits are pool-granular -> select topk_tokens//kpool pools,
        # then expand each pool back to its kpool tokens.
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
        if defer_topk:
            assert num_rows == num_decode_tokens
            pool_topk = kpool_stage.pool_ids(num_rows, select_k)
            topk_dst = pool_topk
        elif index_kpool > 1:
            pool_topk = torch.empty(
                (num_rows, select_k), dtype=torch.int32, device=logits.device
            )
            topk_dst = pool_topk
        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        # Shared dispatcher with the DSA sparse indexer, so the kpool select
        # honors kernel_config.sparse_indexer_topk_backend and picks between
        # cooperative/persistent by batch size instead of always taking the
        # same kernel.
        get_indexer_topk(topk_backend)(
            logits,
            seq_lens,
            next_n,
            topk_dst,
            select_k,
            attn_metadata_narrowed.max_seq_len,
        )
        _merge_dcp_topk_global(
            logits,
            topk_dst,
            select_k,
            dcp_rank,
            dcp_world_size,
            pool_cp_interleave,
            logits_dtype=logits_dtype,
            num_positions=max_pool_len,
        )

        # Resolve to token-level indices in the output buffer.
        if index_kpool > 1:
            n = pool_topk.shape[0]
            if positions is not None and not decode_metadata.requires_padding:
                # Uniform rows: the kernel takes seq_len = position + 1 and
                # writes the expansion straight into the output buffer.
                if defer_topk:
                    kpool_stage.defer(pool_topk, positions[:n], index_kpool)
                    return topk_indices_buffer
                (
                    kpool_ops.expand_pools_and_append_tail
                    if kpool_stage is None
                    else kpool_stage.expand
                )(
                    pool_topk,
                    positions[:n],
                    index_kpool,
                    seq_offset=1,
                    out=topk_indices_buffer[:n],
                )
                return topk_indices_buffer
            pool_ids = pool_topk
            dec_seq = decode_pool_seq_lens(
                positions,
                decode_metadata,
                decode_lens,
                num_decode_tokens,
                batch_size,
                next_n,
                n,
            )
            out = kpool_ops.expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)
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
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

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
        self.use_fp4_cache = use_fp4_cache
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )
        _cfg = get_current_vllm_config_or_none()
        self.topk_backend = (
            _cfg.kernel_config.sparse_indexer_topk_backend
            if _cfg is not None
            else "auto"
        )
        self.logits_dtype = (
            STR_DTYPE_TO_TORCH_DTYPE[
                _cfg.attention_config.resolve_indexer_logits_dtype("float32")
            ]
            if _cfg is not None
            else torch.float32
        )
        _parallel = _cfg.parallel_config if _cfg is not None else None
        self._parallel_config = _parallel
        self.dcp_world_size = 1
        self.dcp_rank = 0
        if _parallel is not None and _parallel.decode_context_parallel_size > 1:
            if _parallel.prefill_context_parallel_size > 1:
                raise NotImplementedError(
                    "SparseAttnIndexerKpool does not support PCP+DCP."
                )
            self.dcp_world_size = _parallel.decode_context_parallel_size
            self.dcp_rank = get_dcp_group().rank_in_group
            if current_platform.is_cuda() and has_cutedsl():
                from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (  # noqa: E501
                    _PACK_DCP_TOPK_CANDIDATES_KERNEL,
                    _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL,
                )

                _PACK_DCP_TOPK_CANDIDATES_KERNEL.register_warmup()
                _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL.register_warmup()

    @property
    def cp_kv_cache_interleave_size(self) -> int:
        if self.dcp_world_size == 1:
            return 1
        assert self._parallel_config is not None
        return self._parallel_config.cp_kv_cache_interleave_size

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
        return self.forward_cuda(
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )

    def forward_cuda(
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
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return sparse_attn_indexer_kpool(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q_values,
            q_scale,
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
            self.use_fp4_cache,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            self.tail_cache.kv_cache if self.tail_cache is not None else None,
            self.tail_cache.prefix if self.tail_cache is not None else None,
            self.topk_backend,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
            self.logits_dtype,
        )
