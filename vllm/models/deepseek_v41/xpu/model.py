# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 text backbone for Intel XPU.

The platform-neutral layer/model/LM bodies live in `common/model_base.py`;
this module binds the XPU pieces to its hooks. The control flow, parameter
layout, checkpoint mapping and the delayed-pre mHC math are shared; the kernels
change:

- Attention is always `DeepseekV41XPUAttention` (`xpu/xpu_sparse.py`).
- mHC runs on the `MHCPreDelayedOp` / `MHCPostOp` CustomOps
  (`vllm/model_executor/layers/mhc.py`) as on ROCm, instead of the TileLang
  entry points; the attn/ffn RMSNorm runs as a separate `RMSNorm` instead of
  being fused into the pre kernel.
- The MoE is the platform-neutral `common/moe.py` `DeepseekV4MoE` (the NVIDIA
  V4 fused-MoE path without MegaMoE).

Dropped CUDA-only features: MegaMoE and the mega gate, TRTLLM deferred MoE
finalize, the FlashMLA/FlashInfer/mega attention classes, the
latency-specialized decode, mHC overlap streams, the fused mHC all-reduce and
the CuTe DSL boundary kernel, sequence parallel and GEMM-RS, the token-sharded
`fused_wqa_wkv`, DBO microbatching, attention aux streams, the Engram batch
(MXFP8 exchange), fabric/peer tables, THP and DP-shared host tables, and the
mHC JIT warmup. Settings that request any of them raise `NotImplementedError`
at construction.
"""

import typing

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.kernel import MEGA_MOE_BACKENDS
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.mhc import MHCPostOp, MHCPreDelayedOp, hc_expand
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.xpu.engram import XPUChunkedEngramEmbedding
from vllm.models.deepseek_v41.xpu.xpu_sparse import DeepseekV41XPUAttention

from ..common.engram import Engram, EngramLayout, ParallelEngramEmbedding
from ..common.model_base import (
    DeepseekV4DecoderLayerBase,
    DeepseekV4ModelBase,
    DeepseekV41LLMForCausalLMBase,
    MHCSeam,
    _linear_scale_param_name,
    _make_deepseek_v4_weights_mapper,
)
from ..common.moe import DeepseekV4MoE

if typing.TYPE_CHECKING:
    from vllm.config.engram import EngramConfig

# Re-exported for the DSpark drafter and the VL wrapper.
__all__ = [
    "DeepseekV41LLMForCausalLM",
    "DeepseekV4DecoderLayer",
    "DeepseekV4Model",
    "DeepseekV4MoE",
    "_linear_scale_param_name",
    "_make_deepseek_v4_weights_mapper",
    "check_xpu_supported",
]

logger = init_logger(__name__)


def check_xpu_supported(vllm_config: VllmConfig) -> None:
    """Reject settings that select a CUDA-only DeepSeek V4.1 feature."""
    extra = vllm_config.additional_config
    decode_backend = (
        extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    )
    if decode_backend != "auto":
        raise NotImplementedError(
            f"DeepSeek V4.1 decode_backend={decode_backend!r} is not supported on "
            "XPU; use the default (auto)."
        )
    moe_backend = vllm_config.kernel_config.moe_backend
    if moe_backend in MEGA_MOE_BACKENDS:
        raise NotImplementedError(
            f"DeepSeek V4.1 MegaMoE (moe_backend={moe_backend!r}) is not "
            "supported on XPU."
        )
    parallel_config = vllm_config.parallel_config
    if parallel_config.prefill_context_parallel_size > 1:
        raise NotImplementedError(
            "DeepSeek V4.1 prefill context parallel is not supported on XPU."
        )
    if parallel_config.use_ubatching:
        raise NotImplementedError(
            "DeepSeek V4.1 dual-batch overlap (microbatching) is not supported on XPU."
        )
    if vllm_config.lora_config is not None:
        # The XPU o-projection multiplies the grouped wo_a weights directly
        # (xpu_sparse.py `_o_proj`), so a LoRA delta on wo_a would be skipped.
        raise NotImplementedError("DeepSeek V4.1 LoRA is not supported on XPU.")


def _check_engram_config(engram_config: "EngramConfig | None") -> None:
    """Keep Engram on the private pinned host tables (or plain device tables)."""
    if engram_config is None:
        return
    if engram_config.peer_group_size:
        raise NotImplementedError(
            "Engram peer_group_size (CUDA FABRIC tables) is not supported on XPU."
        )
    if engram_config.use_thp:
        raise NotImplementedError(
            "Engram use_thp (cudaHostRegister huge pages) is not supported on XPU."
        )
    if not engram_config.cpu_offload and engram_config.embedding_across_dp:
        raise NotImplementedError(
            "Engram device tables sharded across DP (peer reads over symmetric "
            "memory) are not supported on XPU; keep cpu_offload=True."
        )
    if engram_config.dp_shared_memory:
        # On by default for DP > 1; the shared tables are registered with
        # cudaHostRegister. Each replica keeps its own pinned table instead.
        logger.warning_once(
            "Engram dp_shared_memory is not supported on XPU; storing the "
            "offloaded tables per DP replica instead."
        )
        engram_config.dp_shared_memory = False


class XPUEngram(Engram):
    """Engram with its host-table lookups inline on the current stream.

    The CUDA layer prefetches offloaded rows on a side stream; on XPU the
    lookup runs where the rows are staged, like the GPU-resident tables.
    """

    def _create_embedding(
        self, layout: EngramLayout, layer_hash_index: int
    ) -> ParallelEngramEmbedding:
        # `Engram._create_embedding` with pinned-chunk host storage; peer groups
        # and THP are rejected on XPU by `_check_engram_config`.
        engram_config = get_current_vllm_config().engram_config
        assert engram_config is not None
        return XPUChunkedEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            cpu_offload=engram_config.cpu_offload,
            dp_shared_memory=False,
            use_thp=False,
            packed_fp4=layout.packed_fp4,
            peer_group=None,
        )

    def _init_staging(self, max_tokens: int, head_dim: int) -> None:
        # `Engram._init_staging` minus the prefetch stream and event: with
        # both None, `prepare_embeddings` looks rows up inline and
        # `_ready_rows` waits on nothing.
        self.staged_rows = torch.empty(
            max_tokens * self.embed_tokens.dp_size,
            self.embed_tokens.part_n_hash_cols,
            head_dim,
            dtype=torch.bfloat16,
        )
        self._prefetch_stream = None
        self._prefetch_done = None


class DeepseekV4DecoderLayer(DeepseekV4DecoderLayerBase):
    def _make_engram(
        self,
        vllm_config: VllmConfig,
        engram_layout: EngramLayout,
        layer_index: int,
        prefix: str,
    ) -> Engram:
        return XPUEngram(
            vllm_config.model_config.hf_config,
            vllm_config.quant_config,
            engram_layout,
            layer_index,
            use_sequence_parallel=False,
            prefix=prefix,
        )

    def _select_attn_cls(self, vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
        return DeepseekV41XPUAttention

    def _make_ffn(self, vllm_config: VllmConfig, prefix: str) -> nn.Module:
        return DeepseekV4MoE(vllm_config, prefix=prefix)

    def _init_mhc_state(self) -> None:
        self.mhc_pre_delayed = MHCPreDelayedOp()
        self.mhc_post = MHCPostOp()

    def _mhc_pre(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: RMSNorm,
        *,
        pre_mix: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        sublayer_out: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
    ) -> MHCSeam:
        """One mHC seam: the previous sublayer's post (when `sublayer_out`
        is given), then the delayed pre collapsing with the carried `pre_mix`.

        Returns residual, post_mix, res_mix, normed layer input and this
        sublayer's pre-mix (the one the next seam collapses with). The NVIDIA
        kernels fuse `norm` into the collapse; here it runs after it.
        """
        residual, post_mix, res_mix, layer_input, next_pre_mix = self.mhc_pre_delayed(
            residual,
            fn,
            hc_scale,
            hc_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            pre_mix=pre_mix,
            x=x,
            sublayer_out=sublayer_out,
            post_layer_mix=post_mix,
            comb_res_mix=res_mix,
        )
        return residual, post_mix, res_mix, norm(layer_input), next_pre_mix

    def _mhc_first_seam(self, x: torch.Tensor, mhc_stream: None) -> MHCSeam:
        assert self.hc_attn_fn_broadcast is not None
        residual = hc_expand(x, self.hc_mult)
        return self._mhc_pre(
            residual,
            self.hc_attn_fn_broadcast,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.attn_norm,
            x=x,
        )

    def _mhc_attn_pre(
        self, residual: torch.Tensor, pre_mix: torch.Tensor | None, mhc_stream: None
    ) -> MHCSeam:
        return self._mhc_pre(
            residual,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.attn_norm,
            pre_mix=pre_mix,
        )

    def _mhc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.mhc_post(x, residual, post_mix, res_mix)

    def _mhc_attn_seam(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        mhc_stream: None,
        capture_aux: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        # The previous layer's FFN post folded into this attention seam.
        residual, post_mix, res_mix, x, attn_pre = self._mhc_pre(
            residual,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.attn_norm,
            pre_mix=pre_mix,
            sublayer_out=x,
            post_mix=post_mix,
            res_mix=res_mix,
        )
        # The mean of the post-mapped streams (the previous layer's output),
        # as the fused CUDA collapse returns it.
        aux = residual.mean(dim=1) if capture_aux else None
        return residual, post_mix, res_mix, x, attn_pre, aux

    def _mhc_ffn_seam(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        attn_pre: torch.Tensor,
        mhc_stream: None,
    ) -> MHCSeam:
        return self._mhc_pre(
            residual,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.ffn_norm,
            pre_mix=attn_pre,
            sublayer_out=x,
            post_mix=post_mix,
            res_mix=res_mix,
        )


class DeepseekV4Model(DeepseekV4ModelBase):
    decoder_layer_cls = DeepseekV4DecoderLayer

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.mhc_post = MHCPostOp()

    def _init_platform(self, vllm_config: VllmConfig) -> None:
        check_xpu_supported(vllm_config)

    def _prepare_layers(self, vllm_config: VllmConfig) -> dict[str, typing.Any]:
        if self.engram_layout is not None:
            _check_engram_config(vllm_config.engram_config)
        return {}

    def _mhc_final_post(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return self.mhc_post(hidden_states, residual, post_mix, res_mix)

    def _hc_collapse(
        self, hidden_states: torch.Tensor, pre_mix: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.vllm.hc_collapse_triton(hidden_states, pre_mix)


class DeepseekV41LLMForCausalLM(DeepseekV41LLMForCausalLMBase):
    model_cls = DeepseekV4Model
    moe_cls = DeepseekV4MoE

    @staticmethod
    def get_model_state_cls():
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState
