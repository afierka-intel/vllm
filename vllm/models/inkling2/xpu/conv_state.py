# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU copy of `InklingConvState` from `vllm/models/inkling/nvidia/sconv_swa_attn.py`
and `sconv_seq_metadata` from `vllm/models/inkling/nvidia/ops/sconv.py`
(upstream `b7186d137a`).
Adapted: only the pieces Inkling2 uses; the hidden streams are always
replicated over the local KV heads (the v1 `replicate_hidden=True`).

Inkling2 short-conv state managed as a sliding-window KV cache.

Each decoder layer owns one `InklingConvState` (an `AttentionLayerBase`) that
emits a single sliding-window KV cache for the layer's 4 sconv streams (K, V,
attn-output, mlp-output), packed head-major into one block:

    H = num_kv_heads (per-rank), N = block_size = sconv_kernel_size,
    D = head_dim(K) + head_dim(V) + 2 * hidden / H (attn, mlp)

K/V are TP-sharded with the KV heads; the attn-/mlp-output streams are
replicated full-width on every rank, spread over its `H` local heads, so `D`
grows with TP. The conv reads/writes this cache out-of-band via a custom
backend; the (smaller) conv page is padded up to the uniform attention page by
`unify_kv_cache_spec_page_size`.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig


@triton.jit
def _seq_metadata_kernel(
    qsl_ptr,  # [num_reqs + 1] int32 cumulative query start rows
    seq_idx_ptr,  # [T] int32 out: token -> owning request
    query_start_ptr,  # [T] int32 out: first x-row of the token's request
    num_reqs,
    num_actual_tokens,
    num_padded_tokens,
    n_iters,  # ceil(log2(num_reqs)): binary-search depth
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tok = offs.to(tl.int32)
    # Largest j in [0, num_reqs) with qsl[j] <= tok.
    lo = tl.zeros([BLOCK], tl.int32)
    hi = tl.full([BLOCK], num_reqs - 1, tl.int32)
    for _ in range(n_iters):
        mid = (lo + hi + 1) // 2
        below = tl.load(qsl_ptr + mid) <= tok
        lo = tl.where(below, mid, lo)
        hi = tl.where(below, hi, mid - 1)
    actual = offs < num_actual_tokens
    padded = offs < num_padded_tokens
    query_start = tl.load(qsl_ptr + lo)
    tl.store(seq_idx_ptr + offs, tl.where(actual, lo, 0), mask=padded)
    tl.store(
        query_start_ptr + offs,
        tl.where(actual, query_start, 0),
        mask=padded,
    )


def sconv_seq_metadata(
    query_start_loc: torch.Tensor,
    num_reqs: int,
    num_actual_tokens: int,
    seq_idx_out: torch.Tensor,
    query_start_out: torch.Tensor,
    num_padded_tokens: int | None = None,
) -> None:
    """Fill static per-token seq_idx / query_start buffers in one launch.

    Replaces the arange + searchsorted + clamp + gather + 2x copy chain of the
    sconv metadata build with a single kernel writing both persistent buffers.
    Padded rows are filled with zero and must have `slot_mapping == -1`.
    """
    if num_padded_tokens is None:
        num_padded_tokens = num_actual_tokens
    if num_padded_tokens < num_actual_tokens:
        raise ValueError("num_padded_tokens must cover all actual tokens")
    if num_padded_tokens > seq_idx_out.shape[0]:
        raise ValueError("seq_idx_out is too small for the padded token count")
    if num_padded_tokens > query_start_out.shape[0]:
        raise ValueError("query_start_out is too small for the padded token count")

    BLOCK = 256
    n_iters = (num_reqs - 1).bit_length()
    grid = (triton.cdiv(num_padded_tokens, BLOCK),)
    _seq_metadata_kernel[grid](
        query_start_loc,
        seq_idx_out,
        query_start_out,
        num_reqs,
        num_actual_tokens,
        num_padded_tokens,
        n_iters,
        BLOCK=BLOCK,
    )


@dataclass
class InklingSconvMetadata(AttentionMetadata):
    block_table: torch.Tensor  # [num_reqs, max_blocks] physical blocks per req
    slot_mapping: torch.Tensor  # [T] int64 flat slot of each token (-1 => skip)
    seq_idx: torch.Tensor  # [T] int32 token -> batch request
    query_start: torch.Tensor  # [T] int32 first x-row of each token's request


class InklingSconvMetadataBuilder(AttentionMetadataBuilder[InklingSconvMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, AttentionSpec)
        # Persistent per-token buffers for graph capture.
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.seq_idx_buffer = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )
        self.query_start_buffer = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> InklingSconvMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
        num_padded_tokens = common_attn_metadata.slot_mapping.shape[0]
        assert num_padded_tokens >= num_actual_tokens

        # Per-token seq_idx (owning request) and query_start (first x-row of
        # that request; the fused kernel uses it to tell same-forward taps,
        # read from x, from pre-forward taps, read from cache) in one launch.
        sconv_seq_metadata(
            common_attn_metadata.query_start_loc,
            num_reqs,
            num_actual_tokens,
            self.seq_idx_buffer,
            self.query_start_buffer,
            num_padded_tokens,
        )

        return InklingSconvMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:num_padded_tokens],
            seq_idx=self.seq_idx_buffer[:num_padded_tokens],
            query_start=self.query_start_buffer[:num_padded_tokens],
        )


class InklingSconvBackend(AttentionBackend):
    """Custom dummy backend for the sconv sliding-window cache management."""

    @staticmethod
    def get_name() -> str:
        return "INKLING_SCONV_SWA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "InklingSconvBackend has no attention impl; the conv runs out-of-band."
        )

    @staticmethod
    def get_builder_cls() -> type[InklingSconvMetadataBuilder]:
        return InklingSconvMetadataBuilder


class InklingConvState(nn.Module, AttentionLayerBase):
    """Per-decoder-layer owner emitting one sliding-window conv-state spec."""

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        kernel_size: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        # Bound to the manager-allocated paged cache by bind_kv_cache; a
        # placeholder until then. Read out-of-band by Inkling2ShortConv.
        self.kv_cache = torch.tensor([])
        tp_size = get_tensor_model_parallel_world_size()
        # Guardrails for the conv-state layout below; only these are exercised.
        # tp_size <= num_kv_heads keeps >=1 whole KV head per rank (no
        # replication/clamping).
        assert tp_size <= num_kv_heads, (
            f"sconv SWA cache supports tp_size <= num_kv_heads ({num_kv_heads}), "
            f"got {tp_size}"
        )
        # Each rank spreads the full hidden width of the attn-/mlp-output
        # streams over its local heads.
        self.num_kv_heads = num_kv_heads // tp_size
        hidden_per_head = hidden_size // self.num_kv_heads
        # Packed per-head width: K + V + attn-output chunk + mlp-output chunk,
        # padded to a power of two so every layer's conv page is the same size.
        # The page unifier may still scale the logical conv block size up to
        # match a larger attention page; index the bound cache with its runtime
        # token dim (`cache_block_size`), not this kernel window size.
        raw_head_size = 2 * head_dim + 2 * hidden_per_head
        self.head_size = 1 << (raw_head_size - 1).bit_length()
        self.sliding_window = kernel_size
        self.block_size = kernel_size
        # Per-head D-sub-range (offset, width) for each stream (K, V, attn,
        # mlp). Streams share the cache; each writes/reads its own width
        # across all H heads.
        self.stream_ranges: tuple[tuple[int, int], ...] = (
            (0, head_dim),
            (head_dim, head_dim),
            (2 * head_dim, hidden_per_head),
            (2 * head_dim + hidden_per_head, hidden_per_head),
        )
        vllm_config = get_current_vllm_config()
        self._dtype = vllm_config.model_config.dtype
        assert self._dtype == torch.bfloat16, (
            f"sconv SWA cache supports bfloat16 only, got {self._dtype}"
        )
        # Register in the forward context so the runner enumerates this owner as
        # an attention-like layer (get_kv_cache_config / get_attn_backend).
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(self): ...

    @property
    def cache_block_size(self) -> int:
        """Return the block size used by cache metadata and physical indexing."""
        if self.kv_cache.numel() > 0:
            return self.kv_cache.shape[2]
        return self.block_size

    def get_attn_backend(self) -> type[AttentionBackend]:
        return InklingSconvBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        spec = AttentionSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=0,  # all 4 streams packed into head_size
            dtype=self._dtype,
        )
        return SlidingWindowManager.layer_config(
            spec, sliding_window=self.sliding_window
        )
