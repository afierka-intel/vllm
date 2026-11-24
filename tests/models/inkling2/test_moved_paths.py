# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Inkling2 module paths that moved from `nvidia/` to `common/` stay
importable: each old path is the moved module object (`nvidia/__init__.py`,
`_MOVED`), and the names that moved out of the partly moved `nvidia/ops`
modules resolve to the moved objects."""

import importlib

import pytest

from vllm.models.inkling2 import nvidia


@pytest.mark.parametrize("old,new", sorted(nvidia._MOVED.items()))
def test_old_module_path_is_the_moved_module(old: str, new: str):
    moved = importlib.import_module(f"vllm.models.inkling2.{new}")
    assert importlib.import_module(f"vllm.models.inkling2.nvidia.{old}") is moved


@pytest.mark.parametrize(
    "old,name,new",
    [
        ("ops.moe", "combine", "ops.moe"),
        ("ops.moe_sm90", "select_experts", "ops.moe"),
        ("ops.attention", "rel_decode_attention", "ops.attention"),
        ("ops.activation", "_div_rn", "ops.activation"),
        ("ops.moe_sm90", "_situ", "ops.activation"),
    ],
)
def test_names_moved_out_resolve_from_the_old_module(old: str, name: str, new: str):
    old_module = importlib.import_module(f"vllm.models.inkling2.nvidia.{old}")
    new_module = importlib.import_module(f"vllm.models.inkling2.common.{new}")
    assert getattr(old_module, name) is getattr(new_module, name)
