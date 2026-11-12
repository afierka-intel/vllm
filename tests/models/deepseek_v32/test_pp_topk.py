# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import DeepseekV3Config

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models import utils as model_utils
from vllm.models.deepseek_v32.nvidia import model as deepseek_v32_model
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import set_default_torch_dtype

# GLM-5.3: each group of 4 layers from layer 3 reuses its first layer's top-k.
GLM5_TOPK = dict(index_topk_freq=4, index_skip_topk_offset=3)


def _set_pp_size(monkeypatch, pp_size: int) -> None:
    monkeypatch.setattr(
        parallel_state, "get_pp_group", lambda: SimpleNamespace(world_size=pp_size)
    )


class TestPPSplitsLayerSharing:
    @pytest.mark.parametrize(
        "num_layers,users,consumers,pp_size,expected",
        [
            (8, range(8), {1, 2, 3, 5, 6, 7}, 2, False),
            (8, range(8), {1, 2, 4, 5, 6, 7}, 2, True),
            (8, range(8), {1, 2, 4, 5, 6, 7}, 1, False),
            # Layers 4-5 do not use the state, so stage 1's first user is 6.
            (8, {0, 1, 2, 3, 6, 7}, {1, 2, 3, 7}, 2, False),
            (8, {0, 1, 2, 3, 6, 7}, {1, 2, 3, 6, 7}, 2, True),
            (12, range(12), {1, 2, 3, 5, 6, 7, 9, 10, 11}, 3, False),
            (12, range(12), {1, 2, 3, 4, 6, 7, 9, 10, 11}, 3, True),
        ],
        ids=[
            "split-on-producer",
            "split-on-consumer",
            "no-pp",
            "skip-unrelated-layers",
            "unrelated-then-consumer",
            "pp3-aligned",
            "pp3-middle-split",
        ],
    )
    def test_stage_boundaries(
        self, monkeypatch, num_layers, users, consumers, pp_size, expected
    ):
        _set_pp_size(monkeypatch, pp_size)
        assert (
            model_utils.pp_splits_layer_sharing(num_layers, users, consumers)
            == expected
        )

    @pytest.mark.parametrize(
        "num_layers,expected",
        [(78, True), (76, False)],
        ids=["glm5-default-split", "split-on-group-start"],
    )
    def test_topk_buffers(self, monkeypatch, num_layers, expected):
        _set_pp_size(monkeypatch, 2)
        config = SimpleNamespace(
            num_hidden_layers=num_layers, index_topk=8, **GLM5_TOPK
        )
        buffer = torch.empty(4, 8, dtype=torch.int32)
        buffers = model_utils.pp_topk_buffers(config, buffer, range(num_layers))
        assert buffers == ({"topk_indices": buffer} if expected else {})


def test_empty_intermediate_tensors_follow_buffers():
    buffer = torch.empty(16, 8, dtype=torch.int32)
    make_empty = model_utils.make_empty_intermediate_tensors_factory(
        ["hidden_states", "residual"], 4, {"topk_indices": buffer}
    )
    tensors = make_empty(3, torch.bfloat16, torch.device("cpu"))

    assert tensors["residual"].shape == (3, 4)
    assert tensors["topk_indices"].shape == (3, 8)
    assert tensors["topk_indices"].dtype == torch.int32


class _TopkReusingLayer(nn.Module):
    """Reads the shared top-k like a reusing layer, then indexes like the next."""

    def __init__(self, topk_indices_buffer: torch.Tensor) -> None:
        super().__init__()
        self.topk_indices_buffer = topk_indices_buffer
        self.seen_topk: torch.Tensor | None = None

    def forward(self, positions, hidden_states, residual, attn_in=None):
        num_tokens = hidden_states.shape[0]
        self.seen_topk = self.topk_indices_buffer[:num_tokens].clone()
        self.topk_indices_buffer[:num_tokens] += 10
        return hidden_states, torch.zeros_like(hidden_states)


def test_middle_pp_stage_receives_and_forwards_topk(monkeypatch):
    model = object.__new__(deepseek_v32_model.DeepseekV32Model)
    nn.Module.__init__(model)
    model.topk_indices_buffer = torch.full((8, 2), -1, dtype=torch.int32)
    layer = _TopkReusingLayer(model.topk_indices_buffer)
    model.layers = nn.ModuleList([layer])
    model.start_layer, model.end_layer = 0, 1
    model.use_sequence_parallel = False
    model.shardq_sequence_parallel = False
    model.aux_hidden_state_layers = ()
    model.pp_buffers = {"topk_indices": model.topk_indices_buffer}
    monkeypatch.setattr(
        deepseek_v32_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
    )
    monkeypatch.setattr(
        deepseek_v32_model, "tensor_model_parallel_all_reduce", lambda t: t
    )

    received_topk = torch.arange(6, dtype=torch.int32).view(3, 2)
    output = model(
        None,
        torch.arange(3),
        IntermediateTensors(
            {
                "hidden_states": torch.zeros(3, 4),
                "topk_indices": received_topk,
            }
        ),
    )

    torch.testing.assert_close(layer.seen_topk, received_topk)
    torch.testing.assert_close(output["topk_indices"], received_topk + 10)


def test_generic_middle_pp_stage_receives_and_forwards_topk(monkeypatch):
    """Same contract on the generic path (deepseek_v2.py), which serves
    DeepSeek-V3.2 and GLM-5.x on XPU and CPU."""
    model = object.__new__(deepseek_v2.DeepseekV2Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace()
    model.topk_indices_buffer = torch.full((8, 2), -1, dtype=torch.int32)
    layer = _TopkReusingLayer(model.topk_indices_buffer)
    layer.use_sequence_parallel_moe = False
    model.layers = nn.ModuleList([layer])
    model.start_layer, model.end_layer = 0, 1
    model.aux_hidden_state_layers = ()
    model.pp_buffers = {"topk_indices": model.topk_indices_buffer}
    monkeypatch.setattr(
        deepseek_v2,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
    )

    received_topk = torch.arange(6, dtype=torch.int32).view(3, 2)
    # forward() directly: DeepseekV2Model is wrapped by support_torch_compile.
    output = model.forward(
        None,
        torch.arange(3),
        IntermediateTensors(
            {
                "hidden_states": torch.zeros(3, 4),
                "residual": torch.zeros(3, 4),
                "topk_indices": received_topk,
            }
        ),
    )

    torch.testing.assert_close(layer.seen_topk, received_topk)
    torch.testing.assert_close(output["topk_indices"], received_topk + 10)


# The generic deepseek_v2.py path, which vllm/models/deepseek_v32 selects on
# platforms without a vendor model (XPU, CPU).
HIDDEN = 512
EXPERTS = 64


def _dsa_config(**overrides) -> DeepseekV3Config:
    common = dict(
        hidden_size=HIDDEN,
        intermediate_size=1024,
        moe_intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=EXPERTS,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=64,
        vocab_size=1024,
        max_position_embeddings=4096,
        index_topk=64,
        index_n_heads=4,
        index_head_dim=128,
    )
    common.update(overrides)
    return DeepseekV3Config(**common)


@pytest.mark.parametrize(
    "config_kwargs",
    [
        dict(model_type="glm_moe_dsa"),
        dict(moe_router_dtype="float32"),
    ],
    ids=["glm-moe-dsa", "moe-router-dtype-fp32"],
)
@torch.inference_mode()
def test_fp32_router_logits_are_not_rounded(config_kwargs, dist_init):
    """Configs that ask for an fp32 router must get fp32-accurate router
    logits on every platform, not bf16 logits cast to fp32."""
    config = _dsa_config()
    for key, value in config_kwargs.items():
        setattr(config, key, value)
    device = current_platform.device_type or "cpu"
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = dict()
    # Built under the model dtype, as during model initialization.
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.bfloat16):
        moe = deepseek_v2.DeepseekV2MoE(
            config, vllm_config.parallel_config, prefix="model.layers.0.mlp"
        )
    gate = moe.gate.to(device)
    torch.manual_seed(0)
    # Checkpoint router weights are bf16.
    weight = (torch.randn(EXPERTS, HIDDEN) * 0.05).to(torch.bfloat16)
    gate.weight.data.copy_(weight.to(device))
    x = torch.randn(32, HIDDEN, dtype=torch.bfloat16, device=device)

    logits, _ = gate(x)

    reference = x.float() @ weight.to(device).float().T
    assert logits.dtype == torch.float32
    # bf16 rounding of the logits alone is ~4e-3 relative at |logit| ~ 1.
    torch.testing.assert_close(logits, reference, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="builds a sparse MLA attention layer",
)
@torch.inference_mode()
def test_indexer_types_decide_which_layers_build_an_indexer(tmp_path, dist_init):
    """indexer_types overrides the index_topk_freq formula. Here they
    disagree on layers 1 and 3."""
    config = _dsa_config(
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=["full", "shared", "full", "full"],
    )
    config.architectures = ["DeepseekV32ForCausalLM"]
    config.save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=128,
        dtype="bfloat16",
    )
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.compilation_config.static_forward_context = dict()

    has_indexer = {}
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.bfloat16):
        for layer_id in range(4):
            attn = deepseek_v2.DeepseekV2MLAAttention(
                vllm_config=vllm_config,
                config=model_config.hf_config,
                hidden_size=HIDDEN,
                num_heads=4,
                qk_nope_head_dim=64,
                qk_rope_head_dim=64,
                v_head_dim=64,
                q_lora_rank=128,
                kv_lora_rank=512,
                cache_config=vllm_config.cache_config,
                prefix=f"model.layers.{layer_id}.self_attn",
                topk_indices_buffer=torch.empty(16, 64, dtype=torch.int32),
            )
            has_indexer[layer_id] = attn.indexer is not None

    assert has_indexer == {0: True, 1: False, 2: True, 3: True}
