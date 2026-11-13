# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-neutral DeepSeek V4.1 MoE pieces.

`DeepseekV4MoE` is the fused-MoE layer of the NVIDIA V4 model without MegaMoE
(used by the XPU port); `DeepseekV4MixtureOfExperts` is the EPLB mixin every
V4.1 platform shares.
"""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import cdiv

from .mm_preprocess import IMAGE_SENTINEL_BASE_ID


class DeepseekV4MLP(nn.Module):
    """Shared expert; the NVIDIA V4 `DeepseekV4MLP` without sequence parallel."""

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
    """Platform-neutral V4.1 MoE: the NVIDIA V4 fused-MoE path plus the V4.1
    DSpark expert counts (`nvidia/model.py` `DeepseekV4MoE`), without MegaMoE,
    the mega gate or the TRTLLM deferred finalize."""

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


class DeepseekV4MixtureOfExperts(MixtureOfExperts):
    """EPLB hooks; the NVIDIA V4 `DeepseekV4MixtureOfExperts`."""

    # Any MoE exposing the `n_*_experts` counts and `experts` runner.
    moe_mlp_layers: list[Any]

    def extract_moe_parameters(self, example_moe: Any) -> None:
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
