# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MergedColumnParallelLinear/QKVParallelLinear
.load_weights() handling checkpoint tensors the layer never registered.

``getattr(self, name, self)`` used the layer itself as a "not found"
sentinel, so a tensor with no registered param fell through to
``param.weight_loader(param, ...)`` with ``param`` bound to the whole layer
module, crashing with ``AttributeError: ... has no attribute 'data'`` deep
inside ``weight_loader``.

The pre-existing ``param is None and name == "bias"`` guard already covered a
bare ``bias``: ``ColumnParallelLinear`` calls ``register_parameter("bias",
None)`` when ``bias=False``, so ``getattr`` returns None rather than the
sentinel. The sentinel was reachable only for genuinely absent attributes, and
every one of those is a real mismatch between checkpoint and quantization
config -- so the tests below pin that such a tensor produces a diagnosable
``ValueError`` naming the layer, instead of the opaque ``AttributeError``.
"""

import pytest
import torch

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
)


def test_merged_column_parallel_load_weights_raises_on_unregistered_g_idx(
    dist_init, default_vllm_config
):
    layer = MergedColumnParallelLinear(
        4, [2, 2], bias=False, params_dtype=torch.float16
    )
    assert not hasattr(layer, "g_idx"), "premise: g_idx must be unregistered"

    # An unquantized layer has no use for an activation-order permutation, so
    # receiving one means the config and the checkpoint disagree. That must be
    # reported, not skipped: skipping would run the layer on weights whose row
    # order was never applied.
    with pytest.raises(ValueError, match="does not match any parameter"):
        list(layer.load_weights([("g_idx", torch.zeros(4, dtype=torch.int32))]))


def test_qkv_parallel_load_weights_raises_on_unregistered_g_idx(
    dist_init, default_vllm_config
):
    layer = QKVParallelLinear(4, 2, 2, bias=False, params_dtype=torch.float16)
    assert not hasattr(layer, "g_idx"), "premise: g_idx must be unregistered"

    with pytest.raises(ValueError, match="does not match any parameter"):
        list(layer.load_weights([("g_idx", torch.zeros(4, dtype=torch.int32))]))


def test_merged_column_parallel_load_weights_raises_on_unregistered_qweight(
    dist_init, default_vllm_config
):
    layer = MergedColumnParallelLinear(
        4, [2, 2], bias=False, params_dtype=torch.float16
    )
    assert not hasattr(layer, "qweight"), "premise: qweight must be unregistered"

    with pytest.raises(ValueError, match="does not match any parameter"):
        list(layer.load_weights([("qweight", torch.zeros(2, 2, dtype=torch.int32))]))


def test_merged_column_parallel_load_weights_error_names_layer_and_tensor(
    dist_init, default_vllm_config
):
    layer = MergedColumnParallelLinear(
        4, [2, 2], bias=False, params_dtype=torch.float16, prefix="model.layers.0.mlp"
    )

    with pytest.raises(ValueError) as excinfo:
        list(layer.load_weights([("input_scale", torch.zeros(1))]))

    message = str(excinfo.value)
    assert "input_scale" in message
    assert "MergedColumnParallelLinear" in message
    assert "model.layers.0.mlp" in message


def test_merged_column_parallel_load_weights_skips_submodule_bias_none(
    dist_init, default_vllm_config
):
    layer = MergedColumnParallelLinear(
        4, [2, 2], bias=False, params_dtype=torch.float16
    )
    # A dotted name resolves through the submodule, and `nn.Linear(bias=False)`
    # registers `bias` as None there too. Before the shared helper the skip
    # compared the full dotted name against the literal "bias", so this crashed
    # with `AttributeError: 'NoneType' object has no attribute 'weight_loader'`.
    layer.add_module("sub", torch.nn.Linear(2, 2, bias=False))
    assert layer.sub.bias is None, "premise: submodule bias must be None"

    loaded = list(layer.load_weights([("sub.bias", torch.zeros(2))]))

    assert loaded == []


def test_merged_column_parallel_load_weights_skips_bias_registered_as_none(
    dist_init, default_vllm_config
):
    layer = MergedColumnParallelLinear(
        4, [2, 2], bias=False, params_dtype=torch.float16
    )
    # bias=False registers the param as None rather than omitting it.
    assert layer.bias is None, "premise: bias must be registered as None"

    # Pre-existing behaviour, preserved -- passes on unpatched main too.
    loaded = list(layer.load_weights([("bias", torch.zeros(2))]))

    assert loaded == []


def test_merged_column_parallel_load_weights_loads_matched_weight(
    dist_init, default_vllm_config
):
    layer = MergedColumnParallelLinear(
        4, [2, 2], bias=False, params_dtype=torch.float16
    )
    weight = torch.rand(2, 4, dtype=torch.float16)
    weight.shard_id = 0

    # Pre-existing behaviour, preserved -- passes on unpatched main too.
    loaded = list(layer.load_weights([("weight", weight)]))

    assert loaded == ["weight"]
