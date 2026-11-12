# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU sparse MLA attention for DeepSeek V4.1.

Subclasses the shared V4.1 `DeepseekV4Attention` and keeps its control flow
(KV/indexer sharing across layers, V4.1 compress ratios 0/1/2, chunked
prefill plan). Only the kernels change: the FlashMLA decode/prefill calls
become the DeepSeek V4 XPU Triton kernels, the fused `_C` qnorm/RoPE/KV insert
becomes its XPU Triton counterpart, and the DeepGEMM wo_a einsum becomes
`xpu_fp8_bmm` on the MXFP8 (1x32 e8m0) weights.

The KV cache keeps the 584-byte `fp8_ds_mla` record (as on every non-SM100
platform); the NVFP4 compressed cache is not supported on XPU.
"""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
    xpu_qnorm_rope_kv_fp8_insert,
)
from vllm.models.deepseek_v4.xpu.xpu_sparse_decode_fp8 import xpu_sparse_decode_fp8
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


def _adaptive_verification_cg_support(
    vllm_config: VllmConfig,
) -> AttentionCGSupport | None:
    spec_config = vllm_config.speculative_config
    if spec_config is not None and spec_config.enable_adaptive_verification:
        return AttentionCGSupport.ALWAYS
    return None


class DeepseekV41XPUSparseMLAMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> AttentionCGSupport:
        return _adaptive_verification_cg_support(
            vllm_config
        ) or super().get_cudagraph_support(vllm_config, kv_cache_spec)


class DeepseekV41XPUSparseSWAMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    """V4.1 SWA metadata (ratios 0/1/2) for XPU.

    The default `DeepseekSparseSWABackend` builder on XPU classifies layers
    with V4 semantics; V4.1 layers must use this one.
    """

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> AttentionCGSupport:
        return _adaptive_verification_cg_support(
            vllm_config
        ) or super().get_cudagraph_support(vllm_config, kv_cache_spec)


class DeepseekV41XPUSparseSWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekV41XPUSparseSWAMetadataBuilder]:
        return DeepseekV41XPUSparseSWAMetadataBuilder


class DeepseekV41XPUSparseBackend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "XPU_V41_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV41XPUSparseMLAMetadataBuilder]:
        return DeepseekV41XPUSparseMLAMetadataBuilder


class DeepseekV41XPUAttention(DeepseekV4Attention):
    """XPU sparse MLA attention layer for DeepSeek V4.1."""

    backend_cls = DeepseekV41XPUSparseBackend
    swa_backend_cls = DeepseekV41XPUSparseSWABackend

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.use_pcp:
            raise NotImplementedError(
                "DeepSeek V4.1 prefill context parallel is not supported on XPU."
            )
        if not self._uses_fp8_ds_mla_layout() or self.kv_mxfp8:
            raise NotImplementedError(
                "DeepSeek V4.1 on XPU supports only the fp8_ds_mla KV cache "
                "(--kv-cache-dtype auto/fp8)."
            )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    # ---- KV / Q preparation ----

    def _kv_rope_insert(
        self,
        kv: torch.Tensor,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> None:
        xpu_qnorm_rope_kv_fp8_insert(
            None,
            kv,
            cache,
            slot_mapping,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.eps,
            block_size,
        )

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        if not isinstance(attn_metadata, dict):
            # Profile run: no kernels; no head padding on XPU.
            return q
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None
        assert positions.dtype == torch.int64
        xpu_qnorm_rope_kv_fp8_insert(
            q,
            kv,
            self.swa_cache_layer.kv_cache,
            swa_metadata.slot_mapping,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
            apply_q_norm=False,  # V4.1 norms qr before wq_b.
        )
        return q

    # ---- output projection ----

    def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bmm_weight = getattr(self.wo_a, "bmm_weight", None)
        bmm_scale = getattr(self.wo_a, "bmm_scale", None)
        if bmm_weight is None or bmm_scale is None:
            raise NotImplementedError(
                "DeepSeek V4.1 on XPU requires an MXFP8 wo_a "
                "(XPUMxFp8LinearKernel bmm weights)."
            )
        o = attn_out[:, : self.n_local_heads, :]
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            quant_group_size=self._o_proj_block_size,
            tma_aligned_scales=False,
        )
        # [T, G, K] x [G, K, N] with 1x32 activation and e8m0 weight scales.
        z = torch.ops.vllm.xpu_fp8_bmm(
            o_fp8.transpose(0, 1),
            bmm_weight,
            torch.bfloat16,
            o_scale.transpose(0, 1).contiguous(),
            bmm_scale,
            None,
        )
        return self._wo_b_proj(z.transpose(0, 1).flatten(1))

    # ---- attention ----

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )
        attn_metadata = get_forward_context().attn_metadata
        swa_only = self.compress_ratio == 0

        if attn_metadata is None:
            # Warmup dummy run: reserve the prefill workspace only.
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            top_k = 0
            if not swa_only:
                assert self.topk_indices_buffer is not None
                top_k = self.topk_indices_buffer.shape[-1]
            combined_topk = round_up(top_k + self.window_size, 128)
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        # Compressed-cache metadata lives on the kv-source layer's prefix.
        compressed_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        compressed_cache = None if swa_only else self._compressed_kv_cache()
        num_decode_tokens = swa_metadata.num_decode_tokens

        if swa_metadata.num_prefill_tokens > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                compressed_k_cache=compressed_cache,
                swa_k_cache=self.swa_cache_layer.kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=compressed_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decode_tokens > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=compressed_cache,
                swa_metadata=swa_metadata,
                attn_metadata=compressed_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            # Local indices filled by the index-source layer's indexer.
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert self.topk_indices_buffer is not None
            global_indices, topk_lens = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                attn_metadata.block_size // self.compress_ratio,
                swa_metadata.is_valid_token[:num_decode_tokens],
            )
            topk_indices = global_indices.view(num_decode_tokens, 1, -1)

        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.decode_swa_lens is not None
        xpu_sparse_decode_fp8(
            q=q,
            kv_cache=kv_cache,
            swa_kv_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            attn_sink=self.attn_sink,
            softmax_scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            out=output,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        # Same chunk plan and gathers as the CUDA FlashMLA layer
        # (nvidia/flashmla.py); only the sparse attention kernel differs.
        swa_only = self.compress_ratio == 0
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert seq_lens is not None and gather_lens is not None
        assert query_start_loc_cpu is not None and query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:]
        topk_indices = topk_indices[:num_prefill_tokens]
        top_k = 0 if swa_only else topk_indices.shape[-1]
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
            has_compressed=not swa_only,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty when num_prefills > 0"
        workspace_manager = current_workspace_manager()
        combined_topk = round_up(top_k + self.window_size, 128)
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            chunk_size = chunk_end - chunk_start
            kv, combined_indices_out, combined_lens_out = (
                workspace_manager.get_simultaneous(
                    ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),
                    ((self.max_num_batched_tokens, combined_topk), torch.int32),
                    ((self.max_num_batched_tokens,), torch.int32),
                )
            )
            if not swa_only:
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=chunk_N,
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            combined_indices, _ = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_M,
                chunk_N,
                out=(
                    combined_indices_out[: query_end - query_start],
                    combined_lens_out[: query_end - query_start],
                ),
            )
            # Unused slots are -1 (combine_topk_swa_indices fills the buffer).
            out, _, _ = triton_bf16_mla_sparse_interface(
                q=q[query_start:query_end],
                kv=kv[:chunk_size].reshape(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                d_v=q.shape[-1],
                block_dpe=0,
                attn_sink=self.attn_sink,
            )
            output[query_start:query_end] = out
