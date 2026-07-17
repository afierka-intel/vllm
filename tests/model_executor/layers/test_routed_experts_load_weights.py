# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for RoutedExperts.load_weights() tolerating checkpoint
tensors the layer never registered as a param (e.g. per-expert bias when
the quant method didn't register one, or a GPTQ exporter's g_idx).

Previously, `param = getattr(self, param_name)` had no fallback: an
unmatched checkpoint tensor whose name still matched an entry in
`expert_mapping` (e.g. `experts.0.gate_proj.bias` matching the
`gate_proj` mapping entry meant for `.weight`) crashed with
`AttributeError` instead of being skipped.
"""

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.platforms import current_platform


@pytest.fixture
def cpu_vllm_config(monkeypatch):
    # RoutedExperts construction routes through the unquantized MoE backend
    # oracle, which branches on current_platform.is_cpu()/is_xpu()/is_cuda()
    # directly (independent of DeviceConfig) -- force CPU so this test runs
    # the same on any host, regardless of what GPU (if any) is visible.
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        yield config


def _build_routed_experts() -> RoutedExperts:
    moe_parallel_config = FusedMoEParallelConfig.make_no_parallel()

    expert_map_manager = ExpertMapManager(
        max_num_batched_tokens=512,
        top_k=1,
        global_num_experts=2,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=moe_parallel_config,
        placement_strategy="linear",
        enable_eplb=False,
    )

    moe_config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=8,
        intermediate_size=4,
        num_local_experts=expert_map_manager.local_num_experts,
        num_logical_experts=2,
        moe_parallel_config=moe_parallel_config,
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
    )

    return RoutedExperts(
        layer_name="layer.0.mlp.experts",
        params_dtype=torch.bfloat16,
        moe_config=moe_config,
        quant_config=None,
        expert_map_manager=expert_map_manager,
    )


def test_routed_experts_load_weights_skips_unmatched_bias(cpu_vllm_config):
    routed_experts = _build_routed_experts()

    loaded = list(
        routed_experts.load_weights(
            [("0.gate_proj.bias", torch.zeros(4, dtype=torch.bfloat16))]
        )
    )

    assert loaded == []


def test_routed_experts_load_weights_loads_matched_weight(cpu_vllm_config):
    routed_experts = _build_routed_experts()

    # gate_proj shard: [intermediate_size, hidden_dim] = [4, 8]
    weight = torch.randn(4, 8, dtype=torch.bfloat16)
    loaded = list(routed_experts.load_weights([("0.gate_proj.weight", weight)]))

    assert loaded == ["w13_weight"]
