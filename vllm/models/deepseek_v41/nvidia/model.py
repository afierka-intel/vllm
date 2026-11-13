# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 text backbone for CUDA.

The platform-neutral layer/model/LM bodies live in `common/model_base.py`;
this module binds the CUDA pieces to its hooks: the FlashMLA / FlashInfer /
mega attention selection, the V4 MoE with MegaMoE, the mega gate and the TRTLLM
deferred finalize, the TileLang / Mega-mHC seams with overlap streams, the
fused mHC all-reduce and the CuTe DSL boundary, sequence parallel and GEMM-RS,
the token-sharded `fused_wqa_wkv`, the Engram batch / prefetch stream / THP
tables and the latency-specialized decode.
"""

import typing
from collections.abc import Callable
from functools import partial
from itertools import islice
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.kernel import MEGA_MOE_BACKENDS, NATIVE_MEGA_MOE_BACKENDS
from vllm.distributed import (
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import in_piecewise_cudagraph
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_post_tilelang,
    mhc_pre_delayed_tilelang,
)
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
    TrtLlmMxfp4ExpertsModular,
    TrtLlmMxfp4Mxfp8ExpertsModular,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import _unpack
from vllm.model_executor.model_loader.weight_utils import drop_checkpoint_cache
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MoE as DeepseekV4MoEBase,
)
from vllm.models.deepseek_v4.nvidia.model import (
    MegaGateRoutingMetadata,
    _use_mega_gate,
    make_deepseek_v4_expert_params_mapping,
    prepare_mega_gate_routing_metadata,
)
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
    DeepseekV4MegaAttnAttention,
)
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLAAttention,
    DeepseekV4FlashInferSM120Attention,
)
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from ..common.engram import (
    Engram,
    EngramBatch,
    EngramLayout,
    can_share_engram_tables,
)
from ..common.mm_preprocess import IMAGE_SENTINEL_BASE_ID
from ..common.model_base import (
    DeepseekV4DecoderLayerBase,
    DeepseekV4ModelBase,
    DeepseekV41LLMForCausalLMBase,
    MHCSeam,
    _linear_scale_param_name,
    _make_deepseek_v4_weights_mapper,
)
from .ops.mhc import (
    MHC_OVERLAP_MAX_TOKENS,
    hc_expand,
    init_mhc_all_reduce,
    mhc_boundary_weight,
    mhc_pre_delayed_overlap,
    mhc_shifted_post_pre,
    supports_mhc_all_reduce,
    supports_mhc_overlap,
)
from .specialized_decode import specialized_decode_config

if typing.TYPE_CHECKING:
    from .specialized_decode.runtime.decode import SpecializedDecode

# Re-exported for the DSpark drafter, the VL wrapper and tests.
__all__ = [
    "DeepseekV41LLMForCausalLM",
    "DeepseekV4DecoderLayer",
    "DeepseekV4Model",
    "DeepseekV4MoE",
    "_linear_scale_param_name",
    "_make_deepseek_v4_weights_mapper",
    "_use_mega_gate",
    "_use_sequence_parallel",
    "make_deepseek_v4_expert_params_mapping",
    "maybe_init_gemm_rs",
    "prepare_mega_gate_routing_metadata",
]

logger = init_logger(__name__)


class DeepseekV4MoE(DeepseekV4MoEBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_sequence_parallel: bool = False,
        reduce_results: bool = True,
    ):
        config = vllm_config.model_config.hf_config
        n_routed_experts = config.n_routed_experts
        n_activated_experts = config.num_experts_per_tok
        if extract_layer_index(prefix) >= config.num_hidden_layers:
            n_routed_experts = (
                getattr(config, "dspark_n_routed_experts", 0) or n_routed_experts
            )
            n_activated_experts = (
                getattr(config, "dspark_num_experts_per_tok", 0) or n_activated_experts
            )
        super().__init__(
            vllm_config,
            prefix=prefix,
            use_sequence_parallel=use_sequence_parallel,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            num_hash_layers=0,
            reduce_results=reduce_results,
            image_sentinel_lo=IMAGE_SENTINEL_BASE_ID,
        )

    def defer_finalize(self) -> None:
        """Leave the routed top-k reduction to the next fused all-reduce + mHC.

        Only the modular TRTLLM MXFP4 experts can stop after GEMM2; the fused
        kernel takes as many tokens as the mHC overlap path.
        """
        moe_config = self.experts.moe_config
        experts_cls = getattr(
            self.experts.routed_experts.quant_method, "experts_cls", None
        )
        if (
            experts_cls in (TrtLlmMxfp4ExpertsModular, TrtLlmMxfp4Mxfp8ExpertsModular)
            and self.experts.routed_scaling_factor == 1.0
        ):
            moe_config.defer_moe_finalize(MHC_OVERLAP_MAX_TOKENS)
            if moe_config.use_deferred_moe_finalize:
                logger.info_once(
                    "DSV4.1 mHC: MoE top-k finalize fused into the all-reduce for "
                    "up to %d tokens.",
                    MHC_OVERLAP_MAX_TOKENS,
                )

    def defers_finalize(self, num_tokens: int) -> bool:
        return (
            not self.use_native_mega_moe
            and self.experts.moe_config.should_defer_moe_finalize(num_tokens)
        )

    def forward_unfinalized(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None
    ) -> MoEOutput:
        """`forward` with the routed top-k reduction and all-reduce left open."""
        # The runner's custom op returns tensors only, so run its body directly.
        shared_output, routed = _unpack(
            self.experts._forward_impl(
                hidden_states, hidden_states, hidden_states, input_ids
            )
        )
        assert shared_output is not None
        assert isinstance(routed, UnfinalizedMoEOutput)
        return MoEOutput(routed=routed, shared_output=shared_output)


def _select_dsv4_attn_cls(vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
    """Pick the CUDA sparse-MLA attention class for the configured backend.

    The generic CUDA backend selector does not instantiate DSv4 layers directly,
    so map generic sparse-MLA choices to the DSv4-specialized attention class.
    Without an explicit backend: SM12 takes FlashInfer, SM100 takes mega
    attention where the topology allows it, and everything else keeps the
    FlashMLA path.
    """
    backend = vllm_config.attention_config.backend
    if vllm_config.parallel_config.prefill_context_parallel_size > 1:
        if backend not in (
            None,
            AttentionBackendEnum.FLASHMLA_SPARSE,
            AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
            AttentionBackendEnum.FLASHMLA_SPARSE_DSV41,
        ):
            raise NotImplementedError(
                "DeepSeek-V4.1 PCP requires FlashMLA sparse attention."
            )
        return DeepseekV4FlashMLAAttention
    device_capability = current_platform.get_device_capability()
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
    ):
        raise ValueError(
            f"{backend.name} is not a DeepSeek V4.1 attention backend. "
            "Use FLASHINFER_MLA_SPARSE_DSV41 for DeepSeek V4.1 FlashInfer "
            "sparse MLA."
        )
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41,
    ):
        if device_capability is not None and device_capability.major == 12:
            return DeepseekV4FlashInferSM120Attention
        return DeepseekV4FlashInferMLAAttention
    if backend is AttentionBackendEnum.FLASHMLA_MEGA_ATTN_DSV41:
        # FlashMLA mega attention reads the grouped ``wo_a`` weight directly
        # from its fused quantized output. LoRA wraps that layer after model
        # construction, so use the split-KV path whose grouped projection
        # explicitly applies the adapter delta.
        if vllm_config.lora_config is not None:
            return DeepseekV4FlashMLAAttention
        return DeepseekV4MegaAttnAttention
    if backend in (
        AttentionBackendEnum.FLASHMLA_SPARSE,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV41,
    ):
        return DeepseekV4FlashMLAAttention

    if device_capability is not None and device_capability.major == 12:
        return DeepseekV4FlashInferSM120Attention
    # Mega attention is the SM100 default: it fuses Q RoPE, sparse attention,
    # the output's inverse RoPE and its FP8 cast into one launch, and brings
    # the 288 B NVFP4 compressed record -- the format the reference
    # implementation itself stores. It declines topologies it cannot serve
    # (non-SM100, TP that leaves fewer than WV_GROUP_SIZE heads per wo_a
    # group, a build without the kernel), which then fall through to FlashMLA.
    if vllm_config.lora_config is None and DeepseekV4MegaAttnAttention.is_available_for(
        vllm_config
    ):
        return DeepseekV4MegaAttnAttention
    return DeepseekV4FlashMLAAttention


def _use_sequence_parallel(vllm_config: VllmConfig) -> bool:
    parallel_config = vllm_config.parallel_config
    moe_needs_token_sharded_input = (
        vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS
    )
    return (
        parallel_config.enable_expert_parallel
        and parallel_config.tensor_parallel_size > 1
        and (moe_needs_token_sharded_input or parallel_config.data_parallel_size > 1)
    )


def _shard_fused_wqa_wkv(vllm_config: VllmConfig) -> bool:
    """Whether large batches shard fused_wqa_wkv by token and all-gather the output."""
    ca_comm = getattr(get_tp_group().device_communicator, "ca_comm", None)
    return (
        current_platform.is_device_capability(90)
        and not _use_sequence_parallel(vllm_config)
        and vllm_config.lora_config is None
        and not vllm_config.parallel_config.use_ubatching
        and not envs.VLLM_BATCH_INVARIANT
        and ca_comm is not None
        and not ca_comm.disabled
        and ca_comm.fully_connected
    )


def maybe_init_gemm_rs(vllm_config: VllmConfig, use_sequence_parallel: bool) -> bool:
    """Set up the fused `wo_b` GEMM + reduce-scatter when opted in.

    Sequence parallel is the only topology where `wo_b` ends in a
    reduce-scatter, so the kernel is bound to RS mode. The workspace is a
    process-wide singleton shared with any other model code in this worker
    (the DSpark drafter reuses it), and its NVLink multicast rendezvous is
    collective, so every TP rank must take the same decision here.
    """
    if not (use_sequence_parallel and envs.VLLM_ENABLE_GEMM_RS):
        return False
    if vllm_config.lora_config is not None:
        # The fused kernel reads wo_b's base weight and has no LoRA epilogue.
        logger.warning_once(
            "DeepSeek-V4.1 GEMM-RS is disabled because LoRA is enabled."
        )
        return False

    # The kernel module pulls in cute_dsl, so import it only once opted in.
    from vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar import (
        maybe_init_gemm_rs_ar,
    )

    hidden_size = vllm_config.model_config.hf_config.hidden_size
    if not maybe_init_gemm_rs_ar(vllm_config, N=hidden_size, all_reduce=False):
        return False
    logger.info_once("To disable DeepSeek-V4.1 GEMM-RS, set VLLM_ENABLE_GEMM_RS=0.")
    return True


class DeepseekV4DecoderLayer(DeepseekV4DecoderLayerBase):
    ffn: DeepseekV4MoE

    def _init_platform_options(
        self,
        vllm_config: VllmConfig,
        *,
        engram_prefetch_stream: torch.cuda.Stream | None = None,
        run_gemm_rs: bool = False,
        mhc_stream: torch.cuda.Stream | None = None,
        fuse_mhc_all_reduce: bool = False,
    ) -> None:
        self.mhc_stream = mhc_stream
        self.fuse_mhc_all_reduce = fuse_mhc_all_reduce
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)
        # Read by `_make_engram` and `_configure_attn` below.
        self._engram_prefetch_stream = engram_prefetch_stream
        self._run_gemm_rs = run_gemm_rs

    def _make_engram(
        self,
        vllm_config: VllmConfig,
        engram_layout: EngramLayout,
        layer_index: int,
        prefix: str,
    ) -> Engram:
        return Engram(
            vllm_config.model_config.hf_config,
            vllm_config.quant_config,
            engram_layout,
            layer_index,
            use_sequence_parallel=self.use_sequence_parallel,
            prefix=prefix,
            prefetch_stream=self._engram_prefetch_stream,
        )

    def _select_attn_cls(self, vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
        return _select_dsv4_attn_cls(vllm_config)

    def _configure_attn(self, vllm_config: VllmConfig) -> None:
        if self.use_sequence_parallel or self.fuse_mhc_all_reduce:
            self.attn.wo_b.reduce_results = False
            if self._run_gemm_rs:
                # Binds only when wo_b's kernel and shape qualify; otherwise
                # forward keeps the separate reduce-scatter below.
                self.attn.bind_gemm_rs()
        self.attn.shard_fused_wqa_wkv = _shard_fused_wqa_wkv(vllm_config)

    def _make_ffn(self, vllm_config: VllmConfig, prefix: str) -> nn.Module:
        return DeepseekV4MoE(
            vllm_config,
            prefix=prefix,
            use_sequence_parallel=self.use_sequence_parallel,
            reduce_results=not self.fuse_mhc_all_reduce,
        )

    def _init_mhc_warmup(self, vllm_config: VllmConfig) -> None:
        mhc_stream = self.mhc_stream
        if vllm_config.kernel_config.enable_jit_warmup and current_platform.is_cuda():
            from vllm.model_executor.kernels.mhc.tilelang_kernels import (
                mhc_fused_post_pre_splits,
            )
            from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL

            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            if self.use_sequence_parallel:
                tp_size = vllm_config.parallel_config.tensor_parallel_size
                max_tokens = (max_tokens + tp_size - 1) // tp_size
            # The epilogue compiles per projection width and per pre-mix mode.
            # The first layer projects the broadcast embedding, so it reads one
            # hidden_size-wide row and selects stream zero; every later sublayer
            # projects the full hc stream, and collapses it with the pre-mix the
            # previous sublayer carried in. Those later shapes also arrive
            # through the fused post + pre-norm GEMM, which picks split-k
            # factors the token sweep below never produces.
            broadcast_embedding = {
                "rms_numel": self.hidden_size,
                "use_pre_mix_in": False,
                "extra_splits": (),
            }
            hc_stream = {
                "rms_numel": self.hc_mult * self.hidden_size,
                "extra_splits": mhc_fused_post_pre_splits(
                    self.hidden_size, self.hc_mult
                ),
            }
            variants = [
                broadcast_embedding,
                {**hc_stream, "use_pre_mix_in": False},
                {**hc_stream, "use_pre_mix_in": True},
            ]
            if vllm_config.speculative_config is not None:
                # Draft setups read aux hidden states out of the same collapse,
                # which is its own specialization of the kernel.
                variants.append(
                    {**hc_stream, "use_pre_mix_in": True, "write_aux": True}
                )
            # Shifted mHC splits the epilogue in two: the input collapse on the
            # caller stream and coefficient generation on a side stream.
            split_modes = (
                ("fused", "stats", "input") if mhc_stream is not None else ("fused",)
            )
            for variant in variants:
                for split_mode in split_modes:
                    MHC_PRE_NORM_KERNEL.register_warmup(
                        split_mode=split_mode,
                        max_tokens=(
                            max_tokens
                            if split_mode == "fused"
                            else min(max_tokens, MHC_OVERLAP_MAX_TOKENS)
                        ),
                        hidden_size=self.hidden_size,
                        rms_eps=self.rms_norm_eps,
                        hc_pre_eps=self.hc_eps,
                        hc_sinkhorn_eps=self.hc_eps,
                        hc_post_mult_value=self.hc_post_alpha,
                        sinkhorn_repeat=self.hc_sinkhorn_iters,
                        norm_eps=self.rms_norm_eps,
                        hc_mult=self.hc_mult,
                        **variant,
                    )

    def _init_mhc_state(self) -> None:
        # The mixer weights laid out for the CuTe DSL mHC boundary, filled after
        # loading; None where that kernel cannot run.
        for name in ("hc_attn_boundary_weight", "hc_ffn_boundary_weight"):
            weight = mhc_boundary_weight(
                self.hidden_size, self.hc_mult, self.hc_attn_fn.device
            )
            self.register_buffer(name, weight, persistent=False)

    def process_weights_after_loading(self) -> None:
        """Lay out the mHC mixer weights for the CuTe DSL boundary kernel (see
        ops/mhc_boundary.py), in place so captured graphs see reloads."""
        if self.hc_attn_boundary_weight is None:
            return
        from .ops.mhc_boundary import prepare_boundary_weight

        self.hc_attn_boundary_weight.copy_(prepare_boundary_weight(self.hc_attn_fn))
        self.hc_ffn_boundary_weight.copy_(prepare_boundary_weight(self.hc_ffn_fn))

    def _forward_mhc_stream(self, positions: torch.Tensor) -> torch.cuda.Stream | None:
        mhc_stream = self.mhc_stream
        if mhc_stream is not None and (
            in_piecewise_cudagraph()
            or not 0 < positions.shape[0] <= MHC_OVERLAP_MAX_TOKENS
            or not torch.cuda.is_current_stream_capturing()
        ):
            # Use overlap only in FULL graphs; eager warmup initializes Mega mHC.
            mhc_stream = None
        return mhc_stream

    @staticmethod
    def _mhc_pre(mhc_stream: torch.cuda.Stream | None) -> Callable[..., Any]:
        return (
            partial(mhc_pre_delayed_overlap, stream=mhc_stream)
            if mhc_stream is not None
            else mhc_pre_delayed_tilelang
        )

    def _mhc_first_seam(
        self, x: torch.Tensor, mhc_stream: torch.cuda.Stream | None
    ) -> MHCSeam:
        assert self.hc_attn_fn_broadcast is not None
        residual = hc_expand(x, self.hc_mult)
        post_mix, res_mix, x, attn_pre = self._mhc_pre(mhc_stream)(
            residual,
            self.hc_attn_fn_broadcast,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            x=x,
            norm_weight=self.attn_norm.weight,
            norm_eps=self.attn_norm.variance_epsilon,
        )
        return residual, post_mix, res_mix, x, attn_pre

    def _mhc_attn_pre(
        self,
        residual: torch.Tensor,
        pre_mix: torch.Tensor | None,
        mhc_stream: torch.cuda.Stream | None,
    ) -> MHCSeam:
        post_mix, res_mix, x, attn_pre = self._mhc_pre(mhc_stream)(
            residual,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            pre_mix=pre_mix,
            norm_weight=self.attn_norm.weight,
            norm_eps=self.attn_norm.variance_epsilon,
        )
        return residual, post_mix, res_mix, x, attn_pre

    def _mhc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.fuse_mhc_all_reduce:
            x = tensor_model_parallel_all_reduce(x)
        return mhc_post_tilelang(x, residual, post_mix, res_mix)

    def _mhc_attn_seam(
        self,
        x: torch.Tensor | MoEOutput,
        residual: torch.Tensor,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        mhc_stream: torch.cuda.Stream | None,
        capture_aux: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        # The collapse already reads the post-mapped streams, so the mean
        # aux consumers want comes out of the same kernel.
        return mhc_shifted_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            pre_mix=pre_mix,
            norm_weight=self.attn_norm.weight,
            norm_eps=self.attn_norm.variance_epsilon,
            capture_aux=capture_aux,
            stream=mhc_stream,
            reduce_results=self.fuse_mhc_all_reduce,
            boundary_weight=self.hc_attn_boundary_weight,
        )

    def _mhc_ffn_seam(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        attn_pre: torch.Tensor,
        mhc_stream: torch.cuda.Stream | None,
    ) -> MHCSeam:
        if mhc_stream is not None:
            torch.cuda.current_stream().wait_stream(mhc_stream)
        residual, post_mix, res_mix, x, ffn_pre, _ = mhc_shifted_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            pre_mix=attn_pre,
            norm_weight=self.ffn_norm.weight,
            norm_eps=self.ffn_norm.variance_epsilon,
            stream=mhc_stream,
            reduce_results=self.fuse_mhc_all_reduce,
            boundary_weight=self.hc_ffn_boundary_weight,
        )
        return residual, post_mix, res_mix, x, ffn_pre

    def _ffn_forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor | None,
        mega_gate_metadata: MegaGateRoutingMetadata | None,
        mhc_stream: torch.cuda.Stream | None,
    ) -> torch.Tensor | MoEOutput:
        out: torch.Tensor | MoEOutput
        if self.fuse_mhc_all_reduce and self.ffn.defers_finalize(x.shape[0]):
            # The next layer's first boundary finalizes it with the all-reduce.
            out = self.ffn.forward_unfinalized(x, input_ids)
        else:
            out = self.ffn(x, input_ids, mega_gate_metadata)
        if mhc_stream is not None:
            torch.cuda.current_stream().wait_stream(mhc_stream)
        return out


class DeepseekV4Model(DeepseekV4ModelBase):
    decoder_layer_cls = DeepseekV4DecoderLayer

    def _init_platform(self, vllm_config: VllmConfig) -> None:
        if self.parallel_config.prefill_context_parallel_size > 1 and (
            self.parallel_config.decode_context_parallel_size != 1
            or self.parallel_config.use_ubatching
            or (
                vllm_config.speculative_config is not None
                and (
                    self.parallel_config.pipeline_parallel_size > 1
                    or not vllm_config.speculative_config.use_dspark()
                    or vllm_config.speculative_config.enable_adaptive_verification
                )
            )
            or vllm_config.kv_transfer_config is not None
            or (
                vllm_config.cache_config.enable_prefix_caching
                and not vllm_config.cache_config.swa_bounded_replay
            )
            or vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS
        ):
            raise NotImplementedError(
                "DeepSeek-V4.1 PCP currently requires DCP=1, with no "
                "microbatching, KV transfer, or MegaMoE. "
                "Speculative decoding requires fixed-length DSpark with PP=1. "
                "Prefix caching requires SWA bounded replay."
            )
        self.use_native_mega_moe = (
            vllm_config.kernel_config.moe_backend in NATIVE_MEGA_MOE_BACKENDS
        )
        self.use_mega_gate = _use_mega_gate(vllm_config)
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)
        if (
            self.use_native_mega_moe
            and not vllm_config.parallel_config.enable_expert_parallel
        ):
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently requires expert parallel. "
                "Enable it with --enable-expert-parallel, or pick a different "
                "moe backend."
            )

    def _init_streams(self, vllm_config: VllmConfig) -> dict[str, Any]:
        # Three aux streams: one per non-default input GEMM in
        # DeepseekV4Attention._run_parallel_input_projections
        # (compressor kv_score, indexer.weights_proj). fused_wqa_wkv stays on
        # the default stream.
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        # Keep mHC independent of the streams used inside attention.
        mhc_stream = torch.cuda.Stream() if supports_mhc_overlap(vllm_config) else None
        self.fuse_mhc_all_reduce = mhc_stream is not None and supports_mhc_all_reduce(
            vllm_config
        )
        if self.fuse_mhc_all_reduce:
            init_mhc_all_reduce(vllm_config)
        return {
            "aux_stream_list": aux_stream_list,
            "mhc_stream": mhc_stream,
            "fuse_mhc_all_reduce": self.fuse_mhc_all_reduce,
        }

    def _prepare_layers(self, vllm_config: VllmConfig) -> dict[str, Any]:
        engram_config = vllm_config.engram_config
        engram_prefetch_stream = (
            torch.cuda.Stream()
            if self.engram_layout is not None
            and engram_config is not None
            and engram_config.cpu_offload
            else None
        )
        if (
            self.engram_layout is not None
            and engram_config is not None
            and engram_config.dp_shared_memory
        ):
            engram_config.dp_shared_memory = can_share_engram_tables(self.engram_layout)

        if self.engram_layout is not None and engram_config and engram_config.use_thp:
            # Release old checkpoint cache before allocating the Engram host tables.
            model_config = vllm_config.model_config
            drop_checkpoint_cache(
                model_config.model_weights or model_config.model,
                revision=model_config.revision,
                cache_dir=vllm_config.load_config.download_dir,
            )

        # GEMM-RS uses NCCL symmetric-memory multicast, which requires all TP
        # ranks to belong to one NVLink domain. Collective: run before layers.
        self.run_gemm_rs = maybe_init_gemm_rs(vllm_config, self.use_sequence_parallel)
        return {
            "engram_prefetch_stream": engram_prefetch_stream,
            "run_gemm_rs": self.run_gemm_rs,
        }

    def _finalize_layers(self, vllm_config: VllmConfig) -> None:
        self.engram_batch = EngramBatch.create(
            [
                layer.engram
                for layer in islice(self.layers, self.start_layer, self.end_layer)
                if getattr(layer, "engram", None) is not None
            ],
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        if self.fuse_mhc_all_reduce:
            # A MoE's top-k finalize folds into the next layer's first mHC
            # boundary, which the last local layer lacks and an engram layer
            # replaces with its own all-reduce.
            local_layers = list(islice(self.layers, self.start_layer, self.end_layer))
            for layer, successor in zip(local_layers, local_layers[1:]):
                assert isinstance(layer, DeepseekV4DecoderLayer)
                assert isinstance(successor, DeepseekV4DecoderLayer)
                if successor.engram is None:
                    layer.ffn.defer_finalize()

    def _mega_gate_metadata(
        self, input_ids: torch.Tensor | None
    ) -> MegaGateRoutingMetadata | None:
        if not self.use_mega_gate:
            return None
        assert input_ids is not None
        return prepare_mega_gate_routing_metadata(
            input_ids,
            has_hash_routing=False,
            image_sentinel_base_id=IMAGE_SENTINEL_BASE_ID
            if getattr(self.config, "vision_n_layers", 0) > 0
            else None,
        )

    def _mhc_final_post(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
    ) -> torch.Tensor:
        # The last layer has no successor to fold its post into.
        if self.fuse_mhc_all_reduce:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)

    def _hc_collapse(
        self, hidden_states: torch.Tensor, pre_mix: torch.Tensor
    ) -> torch.Tensor:
        return hc_collapse_triton(hidden_states, pre_mix)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        first_layer = next(iter(islice(self.layers, self.start_layer, self.end_layer)))
        if first_layer.ffn.use_native_mega_moe:
            return make_deepseek_v4_expert_params_mapping(self.config.n_routed_experts)
        return super().get_expert_mapping()


class DeepseekV41LLMForCausalLM(DeepseekV41LLMForCausalLMBase):
    model_cls = DeepseekV4Model
    moe_cls = DeepseekV4MoE
    model: DeepseekV4Model

    def _init_platform(self, vllm_config: VllmConfig) -> None:
        # decode_backend=latency_specialized: a decode executor over self.model, built
        # by the model state once the weights are loaded.
        self.specialized_decode_config = specialized_decode_config(vllm_config)
        self.specialized_decode: SpecializedDecode | None = None

    def build_specialized_decode(self) -> "SpecializedDecode":
        from .specialized_decode.logits.logits import SpecializedDecodeLogitsProcessor
        from .specialized_decode.runtime.decode import SpecializedDecode

        assert self.specialized_decode_config is not None
        self.specialized_decode = SpecializedDecode.for_target(
            self.model, self.specialized_decode_config, self.lm_head
        )
        self.logits_processor = SpecializedDecodeLogitsProcessor(
            self.config.vocab_size, self.specialized_decode.all_gather
        )
        return self.specialized_decode

    def get_model_state_cls(self):
        if self.specialized_decode_config is not None:
            from .specialized_decode.runtime.model_state import (
                DeepseekV41SpecializedDecodeModelState,
            )

            return DeepseekV41SpecializedDecodeModelState
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        decode = self.specialized_decode
        step = None if decode is None else decode.plan(positions, intermediate_tensors)
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
            decode_step=step,
        )
        return hidden_states

    def process_weights_after_loading(self) -> None:
        super().process_weights_after_loading()
        config = self.model.vllm_config
        if config.engram_config and config.engram_config.use_thp:
            # Loading weights refills the file cache; release it before MADV_COLLAPSE.
            model_config = config.model_config
            drop_checkpoint_cache(
                model_config.model_weights or model_config.model,
                revision=model_config.revision,
                cache_dir=config.load_config.download_dir,
            )
            for layer in islice(
                self.model.layers, self.model.start_layer, self.model.end_layer
            ):
                if layer.engram is not None:
                    layer.engram.embed_tokens.collapse_huge_pages()
