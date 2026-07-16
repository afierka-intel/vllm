# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MergedColumnParallelLinear/QKVParallelLinear
.load_weights() tolerating checkpoint tensors the layer never registered
as a param (e.g. bias when bias=False, or a GPTQ exporter's g_idx when
desc_act=False).

Previously, ``getattr(self, name, self)`` used the layer itself as a
"not found" sentinel, so an unmatched ``bias``/``g_idx`` fell through to
``param.weight_loader(param, ...)`` with ``param`` bound to the whole
layer module -- crashing with ``AttributeError: ... has no attribute
'data'`` deep inside ``weight_loader`` instead of being skipped.
"""

import torch

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
)


def test_merged_column_parallel_load_weights_skips_unmatched_bias(
    dist_init, default_vllm_config
):
    layer = MergedColumnParallelLinear(
        4, [2, 2], bias=False, params_dtype=torch.float16
    )

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

    loaded = list(layer.load_weights([("weight", weight)]))

    assert loaded == ["weight"]


def test_qkv_parallel_load_weights_skips_unmatched_g_idx(
    dist_init, default_vllm_config
):
    layer = QKVParallelLinear(4, 2, 2, bias=False, params_dtype=torch.float16)

    loaded = list(layer.load_weights([("g_idx", torch.zeros(4, dtype=torch.int32))]))

    assert loaded == []
