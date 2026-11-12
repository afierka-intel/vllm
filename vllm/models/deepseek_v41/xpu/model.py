# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 text backbone for Intel XPU.

Port of `deepseek_v41/nvidia/model.py` (decoder layer, model, weight loading,
`DeepseekV41LLMForCausalLM`). The control flow, parameter layout, checkpoint
mapping and the delayed-pre mHC math are kept; the kernels change:

- Attention is always `DeepseekV41XPUAttention` (`xpu/xpu_sparse.py`).
- mHC runs on the `MHCPreDelayedOp` / `MHCPostOp` CustomOps
  (`vllm/model_executor/layers/mhc.py`) as on ROCm, instead of the TileLang
  entry points; the attn/ffn RMSNorm runs as a separate `RMSNorm` instead of
  being fused into the pre kernel.
- The MoE is a copy of the NVIDIA V4 `DeepseekV4MoE` without MegaMoE.

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
from collections.abc import Callable, Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.kernel import MEGA_MOE_BACKENDS
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import MHCPostOp, MHCPreDelayedOp, hc_expand
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.models.deepseek_v4.common.spec_decode import needs_mtp_hidden_buffer
from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
from vllm.models.deepseek_v41.xpu.xpu_sparse import DeepseekV41XPUAttention
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv

from ..common.engram import (
    Engram,
    EngramLayout,
    NgramHashState,
    gather_engram_hashes,
)
from ..common.mm_preprocess import IMAGE_SENTINEL_BASE_ID, image_sentinel_mask

if typing.TYPE_CHECKING:
    from vllm.config.engram import EngramConfig
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

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


class DeepseekV4MLP(nn.Module):
    """Shared expert; copy of the NVIDIA V4 MLP (sequence parallel dropped)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Block-FP8 shards in whole 128-blocks; cdiv rounds the per-rank block
        # count up so the linear's even TP split stays block-aligned, with the
        # trailing ranks zero-filled by load_weights.
        block_size = getattr(quant_config, "weight_block_size", None)
        if block_size is not None:
            tp_size = get_tensor_model_parallel_world_size()
            n_local = cdiv(intermediate_size // block_size[0], tp_size)
            intermediate_size = n_local * block_size[0] * tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV4MoE(nn.Module):
    """V4.1 MoE: the NVIDIA V4 fused-MoE path plus the V4.1 DSpark expert
    counts (`nvidia/model.py` `DeepseekV4MoE`), without MegaMoE."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.prefix = prefix

        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        if extract_layer_index(prefix) >= config.num_hidden_layers:
            # DSpark draft layers use their own, smaller expert set.
            self.n_routed_experts = (
                getattr(config, "dspark_n_routed_experts", 0) or self.n_routed_experts
            )
            self.n_activated_experts = (
                getattr(config, "dspark_num_experts_per_tok", 0)
                or self.n_activated_experts
            )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=self.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        # V4.1 has no hash-routed layers (num_hash_layers=0 in the NVIDIA port).
        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        self.gate.bias_vl = None
        has_vision = getattr(config, "vision_n_layers", 0) > 0
        # Image tokens borrow five consecutive reserved in-vocab ids starting
        # at IMAGE_SENTINEL_BASE_ID; 0 disables vision routing (text model).
        self.image_sentinel_lo = IMAGE_SENTINEL_BASE_ID if has_vision else 0
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        if has_vision:
            # Image sentinel tokens route with bias_vl instead of
            # e_score_correction_bias.
            self.gate.bias_vl = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        ep_group = get_ep_group()
        eplb_config = parallel_config.eplb_config
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_shared_experts = config.n_shared_experts or 0
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // ep_group.world_size
        self.n_local_experts = self.n_local_physical_experts
        self.experts_start_idx = ep_group.rank_in_group * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.physical_expert_start = self.experts_start_idx
        self.physical_expert_end = self.experts_end_idx

        self.experts = FusedMoEFactory(
            reduce_results=True,
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=self.n_activated_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            ckpt_names=("w1", "w2", "w3"),
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=None,
            bias_vl=self.gate.bias_vl,
            image_sentinel_lo=self.image_sentinel_lo,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=eplb_config.num_redundant_experts,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.gate.bias_vl is not None and input_ids is None:
            raise ValueError("DeepSeek V4 vision MoE routing requires input_ids.")
        org_shape = hidden_states.shape
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=hidden_states,
            input_ids=input_ids,
        )
        return final_hidden_states.view(org_shape)


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
        engram_layout: EngramLayout | None = None,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size

        self.engram: Engram | None = None
        if engram_layout is not None:
            layer_id = extract_layer_index(prefix)
            if layer_id in engram_layout.layer_ids:
                self.engram = XPUEngram(
                    config,
                    vllm_config.quant_config,
                    engram_layout,
                    engram_layout.layer_ids.index(layer_id),
                    use_sequence_parallel=False,
                    prefix=f"{prefix}.engram",
                )

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = DeepseekV41XPUAttention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
            candidate_block_buffer=candidate_block_buffer,
        )
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_fn_broadcast: torch.Tensor | None = None
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        *,
        capture_previous_aux: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        previous_aux: torch.Tensor | None = None
        # The reference collapses each sublayer's input with the *previous*
        # sublayer's pre-mix: attention uses the pre-mix carried in (identity
        # for the first layer), the FFN uses this layer's attention pre-mix.
        if residual is None:
            if x.dim() == 2:
                # First layer: the stream is the embedding broadcast to hc
                # copies and the identity pre-mix selects copy 0.
                assert self.hc_attn_fn_broadcast is not None
                residual = hc_expand(x, self.hc_mult)
                residual, post_mix, res_mix, x, attn_pre = self._mhc_pre(
                    residual,
                    self.hc_attn_fn_broadcast,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.attn_norm,
                    x=x,
                )
            else:
                # First local layer of a later pipeline stage.
                residual = x
                if self.engram is not None and engram_hashes is not None:
                    residual = self.engram(
                        residual,
                        engram_hashes[:, self.engram.layer_hash_index],
                        engram_mask,
                    )
                residual, post_mix, res_mix, x, attn_pre = self._mhc_pre(
                    residual,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.attn_norm,
                    pre_mix=pre_mix,
                )
        elif self.engram is not None and engram_hashes is not None:
            # Engram injection happens between the previous sublayer's post
            # and this block's pre, on the full hc stream, so the mix
            # coefficients see the injected stream.
            previous_post = self.mhc_post(x, residual, post_mix, res_mix)
            if capture_previous_aux:
                previous_aux = previous_post.mean(dim=1)
            residual = self.engram(
                previous_post,
                engram_hashes[:, self.engram.layer_hash_index],
                engram_mask,
            )
            del previous_post
            residual, post_mix, res_mix, x, attn_pre = self._mhc_pre(
                residual,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.attn_norm,
                pre_mix=pre_mix,
            )
        else:
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
            if capture_previous_aux:
                # The mean of the post-mapped streams (the previous layer's
                # output), as the fused CUDA collapse returns it.
                previous_aux = residual.mean(dim=1)

        x = self.attn(positions, x, None)

        residual, post_mix, res_mix, x, ffn_pre = self._mhc_pre(
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
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix, ffn_pre, previous_aux

    def write_kv(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        """Write the KV `forward` would for the replay layers' inputs, skipping
        the rest of the layer; the replay batch then reruns its rows."""
        # The replay batch rules out Engram here.
        assert self.engram is None
        *_, x, _ = self._mhc_pre(
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
        self.attn.forward_kv(positions, x)


class DeepseekV4Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.parallel_config = vllm_config.parallel_config
        check_xpu_supported(vllm_config)
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # No attention aux streams on XPU: the input GEMMs, compressor and
        # indexer run in order on the current stream.
        aux_stream_list = None

        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        # Two-level candidate filtering: the indexer at
        # candidate_source_layer_id publishes the top candidate blocks of
        # compressed positions here; later ratio-1 indexers mask their scores
        # with it.
        candidate_source_layer = getattr(config, "candidate_source_layer_id", -1)
        candidate_topk_blocks = getattr(config, "candidate_topk_blocks", 0)
        if candidate_source_layer >= 0 and candidate_topk_blocks > 0:
            self.candidate_block_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                candidate_topk_blocks,
                dtype=torch.int32,
            )
        else:
            self.candidate_block_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.engram_layout = EngramLayout.from_config(config)
        self.vllm_config = vllm_config
        if self.engram_layout is not None:
            _check_engram_config(vllm_config.engram_config)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
                candidate_block_buffer=self.candidate_block_buffer,
                engram_layout=self.engram_layout,
            ),
            prefix=f"{prefix}.layers",
        )

        # Decoder-side SWA bounded replay: in eager prefill steps the layers past
        # the last KV source run on each request's trailing window only
        # (decoder_replay_layers.py). The replay batch runs eagerly on XPU.
        self.decoder_replay_layers: DecoderReplayLayers | None = None
        self.decoder_replay_start = self.end_layer
        cut = max(config.kv_source_layer_ids)
        if (
            cut < self.end_layer - 1
            and self._decoder_replay_supported(vllm_config, cut)
            and self.layers[cut].attn.swa_cache_layer.bounded_replay
        ):
            assert self.layers[cut].attn.compress_ratio <= 1
            # The last KV source writes every row's KV, then replays.
            self.decoder_replay_start = cut
            # The attention metadata keys the replay layers read.
            metadata_prefixes = set()
            for layer in islice(self.layers, cut, self.end_layer):
                attn = typing.cast(DeepseekV4DecoderLayer, layer).attn
                metadata_prefixes.add(attn.swa_cache_layer.prefix)
                if attn.compressed_cache_prefix is not None:
                    metadata_prefixes.add(attn.compressed_cache_prefix)
                if attn.indexer is not None:
                    metadata_prefixes.add(attn.indexer.k_cache.prefix)
            self.decoder_replay_layers = DecoderReplayLayers(
                config.sliding_window,
                self._run_replay_layers,
                self.layers[cut].write_kv,
                metadata_prefixes,
                self.layers[cut].attn.swa_cache_layer.prefix,
                lambda: any(i < cut for i in self.aux_hidden_state_layers),
            )
            logger.info_once(
                "Decoder SWA bounded replay: in eager prefill steps, layer %d "
                "writes its KV for every token, and the rest of it and layers "
                "%d-%d run on each request's last %d tokens only.",
                cut,
                cut + 1,
                self.end_layer - 1,
                config.sliding_window,
            )

        # The n-gram hash needs a slot-keyed rolling store of compressed ids
        # (chunked prefill / decode lookback); key it off the first local
        # layer's sliding-window KV cache. Only PP ranks owning an engram
        # layer need it.
        self.engram_hash: NgramHashState | None = None
        self.engram_dp_shared_memory = False
        self.engram_swa_prefix: str | None = None
        if self.engram_layout is not None:
            engram = next(
                (
                    layer.engram
                    for layer in islice(self.layers, self.start_layer, self.end_layer)
                    if isinstance(layer, DeepseekV4DecoderLayer)
                    and layer.engram is not None
                ),
                None,
            )
            if engram is not None:
                # Device tables split over the DP group look up only local
                # tokens (the same rule as the CUDA model).
                self.engram_dp_shared_memory = engram.embed_tokens.dp_size == 1
                first_layer = next(
                    iter(islice(self.layers, self.start_layer, self.end_layer))
                )
                swa_cache_module = first_layer.attn.swa_cache_layer
                self.engram_hash = NgramHashState(
                    vllm_config, self.engram_layout, swa_cache_module
                )
                self.engram_swa_prefix = swa_cache_module.prefix

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.mhc_post = MHCPostOp()

        spec_config = vllm_config.speculative_config
        if get_pp_group().is_last_rank and needs_mtp_hidden_buffer(spec_config):
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # (num_tokens, hc_mult, hidden_size) plus the (num_tokens, hc_mult)
        # pre-mix the next rank's first layer needs for its attention collapse.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "pre_mix": torch.zeros(
                    (batch_size, self.hc_mult),
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        # Engram n-gram hashes for the whole (flattened) batch, computed once
        # and consumed by the engram layers below. Profile runs (KV cache
        # unbound) look up dead rows instead.
        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if (
            self.engram_hash is not None
            and input_ids is not None
            and is_forward_context_available()
        ):
            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, dict) and self.engram_hash.ensure_cache():
                assert self.engram_swa_prefix is not None
                swa_metadata = typing.cast(
                    "DeepseekSparseSWAMetadata", attn_metadata[self.engram_swa_prefix]
                )
                # Image-span tokens are dead: they break n-grams (hash op
                # takes True=dead) and their gate is zeroed (Engram.forward
                # takes True=keep).
                image_mask = image_sentinel_mask(input_ids)
                engram_mask = ~image_mask
                if lookback_token_ids is None:
                    raise NotImplementedError(
                        "engram needs `lookback_token_ids` from the model runner"
                    )
                engram_hashes = self.engram_hash(
                    input_ids,
                    positions,
                    swa_metadata.query_start_loc,
                    image_mask,
                    lookback_token_ids,
                    image_sentinel_mask(lookback_token_ids),
                )
            else:
                # DP-sharded lookups are collective, so a replica skipping the
                # hash still has to reach them; the memory profile also has to
                # see the Engram activations.
                engram_hashes, engram_mask = self.engram_hash.dummy_hashes(input_ids)
            if engram_hashes is not None:
                # Gather all Engram rows before entering the decoder layers.
                gathered_hashes = gather_engram_hashes(
                    engram_hashes, dp_shared_memory=self.engram_dp_shared_memory
                )
                for layer in islice(self.layers, self.start_layer, self.end_layer):
                    engram = getattr(layer, "engram", None)
                    if engram is not None:
                        engram.prepare_embeddings(
                            gathered_hashes[:, engram.layer_hash_index]
                        )

        residual, post_mix, res_mix = None, None, None
        pre_mix: torch.Tensor | None = None
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
        aux_hidden_by_layer: dict[int, torch.Tensor] = {}
        hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(
            range(self.start_layer, self.decoder_replay_start),
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            aux_hidden_by_layer,
            engram_hashes,
            engram_mask,
        )
        late_aux: list[torch.Tensor] = []
        if self.decoder_replay_layers is not None:
            hidden_states, pre_mix, *late_aux = self.decoder_replay_layers(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
            )
        else:
            hidden_states = self._collapse(
                hidden_states, residual, post_mix, res_mix, aux_hidden_by_layer
            )

        aux_hidden_states = [
            aux_hidden_by_layer[layer_id]
            for layer_id in self.aux_hidden_state_layers
            if layer_id in aux_hidden_by_layer
        ] + late_aux

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )

        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        # Collapse the hc copies with the pre-mix from the last layer's FFN
        # mixes -- the mix the reference applies via
        # `last_layer.hc_pre(h, pre_mix)` (v4.1 has no learned hc_head).
        assert pre_mix is not None
        hidden_states = torch.ops.vllm.hc_collapse_triton(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _run_layers(
        self,
        layer_ids: range,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Every layer's post runs inside the next layer's pre seam, so aux
        # hidden states are read back from there instead of recomputed.
        for idx in layer_ids:
            hidden_states, residual, post_mix, res_mix, pre_mix, previous_aux = (
                self.layers[idx](
                    hidden_states,
                    positions,
                    input_ids,
                    pre_mix,
                    post_mix,
                    res_mix,
                    residual,
                    engram_hashes,
                    engram_mask,
                    capture_previous_aux=idx in self.aux_hidden_state_layers,
                )
            )
            if previous_aux is not None:
                # idx is the one-based id of the layer whose post this is.
                aux_hidden_by_layer[idx] = previous_aux
        assert residual is not None and post_mix is not None
        assert res_mix is not None and pre_mix is not None
        return hidden_states, residual, post_mix, res_mix, pre_mix

    def _collapse(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        aux_hidden_by_layer: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        # The last layer has no successor to fold its post into.
        hidden_states = self.mhc_post(hidden_states, residual, post_mix, res_mix)
        if self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_by_layer[self.end_layer] = hidden_states.mean(dim=1)
        return hidden_states

    def _run_replay_layers(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Layers `decoder_replay_start`.. on the given rows; returns their output,
        the last FFN's pre-mix and the aux hidden states they capture."""
        assert self.decoder_replay_layers is not None
        aux_hidden_by_layer: dict[int, torch.Tensor] = {}
        hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(
            range(self.decoder_replay_start, self.end_layer),
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            aux_hidden_by_layer,
        )
        hidden_states = self._collapse(
            hidden_states, residual, post_mix, res_mix, aux_hidden_by_layer
        )
        return (
            hidden_states,
            pre_mix,
            *(
                aux_hidden_by_layer[layer_id]
                for layer_id in self.aux_hidden_state_layers
                if layer_id in aux_hidden_by_layer
            ),
        )

    def _decoder_replay_supported(self, vllm_config: VllmConfig, cut: int) -> bool:
        """Whether this rank may replay from layer `cut` on; warns when not."""
        spec_config = vllm_config.speculative_config
        draft_config = spec_config.draft_model_config if spec_config else None
        draft_hf_config = getattr(draft_config, "hf_config", None)
        draft_window = getattr(draft_hf_config, "sliding_window", None)
        draft_layer_types = getattr(draft_hf_config, "layer_types", None) or ()
        window = self.config.sliding_window
        if self.start_layer > cut or self.end_layer < self.config.num_hidden_layers:
            reason = (
                "the pipeline stage holding the last KV source layer must also "
                "hold every layer after it"
            )
        elif any(i >= cut for i in getattr(self.config, "engram_layer_ids", ())):
            reason = "an Engram layer sits at or after the last KV source layer"
        elif draft_config is not None and (
            draft_window is None
            or draft_window > window
            or any(t != "sliding_attention" for t in draft_layer_types)
        ):
            reason = (
                f"the drafter (sliding window {draft_window}) reads hidden states "
                f"outside the target's {window}-token window"
            )
        else:
            return True
        logger.warning_once("Decoder SWA bounded replay is off: %s.", reason)
        return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        # Block-FP8 shared experts: pad the intermediate up to the TP-uniform
        # block count so the standard loaders below slice it evenly (trailing
        # ranks land on the zero pad).
        pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
        )

        for name, loaded_weight in weights:
            if name.startswith(("vision.", "aligner.", "image_")):
                # Vision weights are loaded by the outer multimodal wrapper.
                logger.warning_once("Skipping non-text weight: %s", name)
                continue
            if pad_shared_expert and ".shared_experts." in name:
                loaded_weight = self._pad_shared_expert_weight(
                    self.quant_config, name, loaded_weight
                )
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                if name not in params_dict:
                    head, _, leaf = name.rpartition(".")
                    suffixed = f"{head}.base_layer.{leaf}"
                    if suffixed in params_dict:
                        name = suffixed
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 -> 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    name_mapped = None
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        # Ask the weight loader for success, since otherwise we
                        # may skip experts with other available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    if name_mapped is not None:
                        loaded_params.add(name_mapped)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Non-LoRA params on a LoRA-wrapped module live at
                    # `<head>.base_layer.<leaf>`; the checkpoint is plain.
                    if name not in params_dict:
                        head, _, leaf = name.rpartition(".")
                        suffixed = f"{head}.base_layer.{leaf}"
                        if suffixed in params_dict:
                            name = suffixed
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    @staticmethod
    def _pad_shared_expert_weight(
        quant_config: QuantizationConfig | None,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-pad a block-FP8 shared-expert weight/scale on its intermediate
        axis so the standard TP loaders split it into even, block-aligned shards
        (trailing ranks get the zero pad). gate (w1)/up (w3) [I, H] pad dim 0;
        down (w2 -> down_proj) [H, I] pads dim 1.
        """
        block_size = getattr(quant_config, "weight_block_size", None)
        assert block_size is not None
        # Round the intermediate axis up to a whole number of TP shards. The axis
        # is in elements for weights (step = block) and in blocks for scales.
        step = (
            1 if name.endswith(("weight_scale_inv", "weight_scale")) else block_size[0]
        )
        dim = 1 if ".down_proj." in name else 0
        mult = get_tensor_model_parallel_world_size() * step
        pad = cdiv(loaded_weight.shape[dim], mult) * mult - loaded_weight.shape[dim]
        if pad == 0:
            return loaded_weight
        pad_shape = list(loaded_weight.shape)
        pad_shape[dim] = pad
        return torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=dim)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        first_layer = next(iter(islice(self.layers, self.start_layer, self.end_layer)))
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=first_layer.ffn.n_redundant_experts,
        )

    def finalize_mhc_broadcast_weights(self) -> None:
        if not get_pp_group().is_first_rank or self.start_layer >= self.end_layer:
            return
        layer = self.layers[self.start_layer]
        if isinstance(layer, DeepseekV4DecoderLayer):
            broadcast = (
                layer.hc_attn_fn.detach()
                .view(-1, layer.hc_mult, layer.hidden_size)
                .sum(dim=1)
            )
            if layer.hc_attn_fn_broadcast is None:
                layer.hc_attn_fn_broadcast = broadcast
            else:
                layer.hc_attn_fn_broadcast.copy_(broadcast)


def _linear_scale_param_name(vllm_config: VllmConfig, expert_dtype: str) -> str:
    """Parameter name the linear quant method registers for weight scales.

    Native MXFP8 checkpoints ([32, 32] blocks with MXFP4 experts) route linear
    layers through ModelOptLinearMethod, which registers `weight_scale`;
    block-FP8 linear layers register `weight_scale_inv`.
    """
    use_mxfp8 = (
        getattr(vllm_config.quant_config, "weight_block_size", None) == [32, 32]
        and expert_dtype == "fp4"
    )
    return "weight_scale" if use_mxfp8 else "weight_scale_inv"


def _make_deepseek_v4_weights_mapper(
    expert_dtype: str, linear_scale_name: str = "weight_scale_inv"
) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # `w{1,2,3}_weight_scale` (no _inv suffix). Linear scales use the
        # parameter name registered by their quantization method.
        scale_regex = {
            # `.base_layer.`-namespace variant (LoRA-wrapped experts).
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale",
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            # The `embed.weight` -> `embed_tokens.weight` suffix rule renames
            # the engram fp8 table but not its scale; route the scale
            # explicitly to the same module.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as `w{13,2}_weight_scale_inv`. Map all `.scale` keys there.
        scale_regex = {
            # `.base_layer.`-namespace variant of the above.
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale_inv",
            # Same engram reroute as the fp4 branch above.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
            "mtp.": None,
            # The v4.1 checkpoint declares the VL arch even for text-only
            # use; the text backbone drops the vision-tower weights.
            "vision.": None,
            "aligner.": None,
            "image_": None,
        },
    )


class DeepseekV4MixtureOfExperts(MixtureOfExperts):
    """EPLB hooks; copy of the NVIDIA V4 mixin."""

    moe_mlp_layers: list[DeepseekV4MoE]

    def extract_moe_parameters(self, example_moe: DeepseekV4MoE | None) -> None:
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            return
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class DeepseekV41LLMForCausalLM(
    nn.Module,
    SupportsPP,
    SupportsEagle3,
    SupportsLoRA,
    DeepseekV4MixtureOfExperts,
):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    # The MTP draft head is not LoRA-adapted.
    lora_skip_prefixes = ["mtp."]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(
            expert_dtype, _linear_scale_param_name(vllm_config, expert_dtype)
        )

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.num_moe_layers = self.config.num_hidden_layers
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV4MoE] = []
        example_moe: DeepseekV4MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, DeepseekV4DecoderLayer):
                continue
            if isinstance(layer.ffn, DeepseekV4MoE):
                example_moe = layer.ffn
                self.moe_mlp_layers.append(layer.ffn)
                self.moe_layers.append(layer.ffn.experts)

        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    @staticmethod
    def get_model_state_cls():
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    @property
    def token_lookback_depth(self) -> int:
        """Tokens before a chunk start the engram hash needs; the model runner
        passes them as `lookback_token_ids`."""
        engram_hash = self.model.engram_hash
        return engram_hash.lookback_depth if engram_hash is not None else 0

    @property
    def decoder_replay_layers(self) -> DecoderReplayLayers | None:
        return self.model.decoder_replay_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
        )

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-collapse residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def process_weights_after_loading(self) -> None:
        self.model.finalize_mhc_broadcast_weights()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
