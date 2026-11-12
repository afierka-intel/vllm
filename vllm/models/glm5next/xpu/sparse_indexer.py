# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash kpool sparse indexer on XPU.

The kpool algorithm (pooled index-K writes, tail pool, expansion of selected
pools to tokens) is shared with the other backends (common/kpool_indexer.py);
this module only provides the XPU logits and top-k: the SYCL MQA-logits ops
(torch.ops.vllm.xpu_fp8_mqa_logits / xpu_fp8_paged_mqa_logits) and the
top_k_per_row kernels. The FP4 index cache and decode context parallelism are
not supported on XPU.
"""

import torch

from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
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
    _decode_topk_seq_lens,
    _fill_short_decode_causal_indices,
    _gather_workspace_shapes,
    kv_cache_as_quant_view,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import LayerNameType, _resolve_layer_name
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    dense_logits_rows,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager


def _decode_row_pool_lens(
    seq_lens: torch.Tensor,
    positions: torch.Tensor | None,
    decode_lens: torch.Tensor,
    num_decode_tokens: int,
    batch_size: int,
    next_n: int,
    requires_padding: bool,
    index_kpool: int,
) -> torch.Tensor:
    """Pool-granular context length (complete pools) of every
    [batch_size, next_n] decode row, int32 [batch_size * next_n].

    The XPU paged MQA-logits op derives the lengths of a request's next_n rows
    from one context length, which does not hold for pools (consecutive
    tokens can see the same number of complete pools). Folding next_n into
    the batch with one exact length per row avoids that.
    """
    n = batch_size * next_n
    if seq_lens.numel() == n:
        # (B, next_n) per-row lengths, or (B, 1) with next_n == 1.
        return seq_lens.reshape(n).to(torch.int32)
    # (B, 1) with padded next_n rows: recover each row's length from its
    # position (padded rows get 0).
    assert positions is not None, "per-row kpool decode lengths need positions"
    token_lens = _decode_topk_seq_lens(
        positions, decode_lens, num_decode_tokens, batch_size, next_n, requires_padding
    )
    return (token_lens // index_kpool).to(torch.int32)


@eager_break_during_capture
def sparse_attn_indexer_kpool(
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
    gate_score: torch.Tensor | None,
    compress_ape: torch.Tensor | None,
    index_kpool: int,
    positions: torch.Tensor | None,
    tail_kv_cache: torch.Tensor | None,
    tail_prefix: str | None,
    topk_backend: str,
) -> torch.Tensor:
    # None in the profiling (dummy) run.
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)
    kpool = max(index_kpool, 1)
    pooled = index_kpool > 1 and gate_score is not None and compress_ape is not None

    if not isinstance(attn_metadata, dict):
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, False
        )
        current_workspace_manager().get_simultaneous(
            values_spec, scales_spec, ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8)
        )
        reserve_profiling_logits(hidden_states, max_pool_len, torch.float32)
        return topk_indices_buffer
    meta = attn_metadata[k_cache_prefix]
    assert isinstance(meta, DeepseekV32IndexerMetadata)
    slot_mapping = meta.slot_mapping
    num_decode_tokens = meta.num_decode_tokens
    num_tokens = slot_mapping.shape[0]
    k = k[:num_tokens]

    if not skip_k_cache_insert:
        if pooled:
            assert gate_score is not None and compress_ape is not None
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
                round_scale=scale_fmt is not None,
                tail_kv_cache=tail_kv_cache,
                tail_prefix=tail_prefix,
            )
        else:
            assert scale_fmt is not None
            ops.indexer_k_quant_and_cache(
                k, kv_cache, slot_mapping, quant_block_size, scale_fmt
            )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    select_k = topk_tokens // kpool

    if meta.num_prefills > 0:
        prefill = meta.prefill
        assert prefill is not None
        assert prefill.shardq_gather_indices is None, "ShardQ is CUDA-only"
        short_prefill = fill_short_prefill(
            prefill,
            positions,
            num_decode_tokens,
            num_tokens,
            topk_tokens,
            topk_indices_buffer,
        )
        k_quant_full, k_scale_full = current_workspace_manager().get_simultaneous(
            *_gather_workspace_shapes(total_seq_lens, head_dim, fp8_dtype, False)
        )
        for chunk in prefill.chunks if not short_prefill else ():
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
            num_rows = row_end - row_start
            topk_dst = (
                torch.empty(
                    (num_rows, select_k), dtype=torch.int32, device=q_quant.device
                )
                if index_kpool > 1
                else topk_indices_buffer[row_start:row_end, :topk_tokens]
            )
            if chunk.local_total_seq_lens == 0:
                topk_dst.fill_(-1)
            else:
                k_scale_f32 = k_scale.view(torch.float32).squeeze(-1)
                # The XPU op returns [rows, num_k] over the whole gathered K,
                # not [rows, max_seqlen_k]: budget the rows on num_k.
                dense_rows = dense_logits_rows(k_quant.shape[0])
                for sub in range(0, num_rows, dense_rows):
                    end = min(sub + dense_rows, num_rows)
                    ks = chunk.cu_seqlen_ks[sub:end]
                    ke = chunk.cu_seqlen_ke[sub:end]
                    # Logits span the whole gathered K ([rows, num_k]); the
                    # top-k reads [ks, ke) of each row and returns indices
                    # relative to ks, i.e. request-relative pools.
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_quant[row_start + sub : row_start + end],
                        k_quant,
                        k_scale_f32,
                        weights[row_start + sub : row_start + end],
                        ks,
                        ke,
                    )
                    ops.top_k_per_row_prefill(
                        logits,
                        ks,
                        ke,
                        topk_dst[sub:end],
                        end - sub,
                        logits.stride(0),
                        logits.stride(1),
                        select_k,
                    )
            if index_kpool > 1:
                expand_prefill_pools(
                    topk_dst,
                    positions,
                    row_start,
                    row_end,
                    topk_tokens,
                    index_kpool,
                    topk_indices_buffer,
                )

    if meta.num_decodes > 0:
        decode = meta.decode
        assert decode is not None
        if pooled and positions is not None and not skip_k_cache_insert:
            assert gate_score is not None and compress_ape is not None
            kpool_decode_tail_update(
                attn_metadata,
                decode,
                meta.num_decodes,
                num_decode_tokens,
                k,
                gate_score,
                compress_ape,
                kv_cache,
                slot_mapping,
                positions,
                index_kpool,
                head_dim,
                round_scale=scale_fmt is not None,
                tail_kv_cache=tail_kv_cache,
                tail_prefix=tail_prefix,
            )
        if _fill_short_decode_causal_indices(
            topk_indices_buffer,
            positions,
            num_decode_tokens,
            meta.max_seq_len,
            topk_tokens,
        ):
            return topk_indices_buffer
        decode_lens = decode.decode_lens
        if decode.requires_padding:
            q_dec = pack_seq_triton(q_quant[:num_decode_tokens], decode_lens)
            w_dec = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            ).reshape(-1, *weights.shape[1:])
        else:
            q_dec = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            w_dec = weights[:num_decode_tokens]
        batch_size, next_n = q_dec.shape[:2]
        rows = batch_size * next_n
        row_pools = _decode_row_pool_lens(
            decode.seq_lens[:batch_size],
            positions,
            decode_lens,
            num_decode_tokens,
            batch_size,
            next_n,
            decode.requires_padding,
            kpool,
        )
        # One single-token row per decode token, each with its own length
        # (row_pools); logits columns are pool positions.
        logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
            q_dec.reshape(rows, 1, *q_dec.shape[2:]),
            kv_cache_as_quant_view(kv_cache, head_dim, False),
            w_dec[:rows],
            row_pools,
            decode.block_table[:batch_size].repeat_interleave(next_n, dim=0),
            decode.schedule_metadata,
            max_pool_len,
        )
        pool_topk = (
            torch.empty((rows, select_k), dtype=torch.int32, device=logits.device)
            if index_kpool > 1
            else topk_indices_buffer[:rows, :topk_tokens]
        )
        # Same per-row lengths as the logits, as 1D with next_n=1: each row is
        # its own request after the fold above.
        get_indexer_topk(topk_backend)(
            logits, row_pools, 1, pool_topk, select_k, meta.max_seq_len
        )
        if index_kpool > 1:
            if positions is not None and not decode.requires_padding:
                kpool_ops.expand_pools_and_append_tail(
                    pool_topk,
                    positions[:rows],
                    index_kpool,
                    seq_offset=1,
                    out=topk_indices_buffer[:rows],
                )
                return topk_indices_buffer
            dec_seq = decode_pool_seq_lens(
                positions,
                decode,
                decode_lens,
                num_decode_tokens,
                batch_size,
                next_n,
                rows,
            )
            out = kpool_ops.expand_pools_and_append_tail(
                pool_topk, dec_seq, index_kpool
            )
        else:
            out = pool_topk
        if decode.requires_padding:
            out = unpack_seq_triton(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out

    return topk_indices_buffer


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """kpool sparse attention indexer for XPU (see the module docstring)."""

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
            raise NotImplementedError("The FP4 index cache is not supported on XPU.")
        cfg = get_current_vllm_config_or_none()
        if cfg is not None and cfg.parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError(
                "Decode context parallelism is not supported by the XPU kpool indexer."
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
        self.use_fp4_cache = use_fp4_cache
        self.topk_backend = (
            cfg.kernel_config.sparse_indexer_topk_backend if cfg is not None else "auto"
        )

    @property
    def cp_kv_cache_interleave_size(self) -> int:
        return 1

    def forward_native(self, *args, **kwargs):
        return self.forward_xpu(*args, **kwargs)

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
        assert not isinstance(q_quant, tuple), "FP4 Q is not supported on XPU"
        return sparse_attn_indexer_kpool(
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
