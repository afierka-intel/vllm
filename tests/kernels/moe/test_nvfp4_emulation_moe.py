# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launcher of the NVFP4 emulation MoE kernel
(`invoke_fused_moe_nvfp4_emulation_kernel`)."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_emulation_moe
from vllm.triton_utils import tl


class _LaunchSpy:
    """Stands in for `JITFunction.run`: records the launch kwargs, does not
    launch."""

    def __init__(self):
        self.kwargs: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.kwargs.append(kwargs)


def _launch(config: dict, tokens: int = 4, top_k: int = 2) -> None:
    """Launch the kernel over random-shaped operands (on the CPU: the spy
    only records the launch)."""
    experts, n, k = 4, 128, 256
    a = torch.empty(tokens, k, dtype=torch.bfloat16)
    b = torch.empty(experts, n, k // 2, dtype=torch.uint8)
    c = torch.empty(tokens, top_k, n, dtype=torch.bfloat16)
    b_scale = torch.empty(experts, n, k // 16, dtype=torch.uint8)
    w_global_scale = torch.ones(experts)
    sorted_ids = torch.zeros(64, dtype=torch.int32)
    expert_ids = torch.zeros(4, dtype=torch.int32)
    num_post_padded = torch.tensor([64], dtype=torch.int32)
    nvfp4_emulation_moe.invoke_fused_moe_nvfp4_emulation_kernel(
        a,
        b,
        c,
        b_scale,
        None,
        w_global_scale,
        None,
        sorted_ids,
        expert_ids,
        num_post_padded,
        False,
        top_k,
        config,
        compute_type=tl.bfloat16,
    )


TILES = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}


@pytest.mark.parametrize("num_warps,num_stages", [(8, 3), (4, 2), (2, 4)])
def test_launch_uses_config_warps_and_stages(monkeypatch, num_warps, num_stages):
    """The config's `num_warps` / `num_stages` reach the kernel launch, as
    with `invoke_fused_moe_triton_kernel` (`**config`)."""
    spy = _LaunchSpy()
    monkeypatch.setattr(
        nvfp4_emulation_moe.fused_moe_nvfp4_emulation_kernel, "run", spy
    )
    config = {
        **TILES,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    _launch(config)
    (kwargs,) = spy.kwargs
    assert kwargs["num_warps"] == num_warps
    assert kwargs["num_stages"] == num_stages
    for name, value in TILES.items():
        assert kwargs[name] == value


def test_launch_without_warps_keeps_triton_defaults(monkeypatch):
    """A config without launch options (e.g. a tuned JSON file that omits
    them) passes none, so Triton's defaults apply."""
    spy = _LaunchSpy()
    monkeypatch.setattr(
        nvfp4_emulation_moe.fused_moe_nvfp4_emulation_kernel, "run", spy
    )
    _launch({**TILES, "GROUP_SIZE_M": 1})
    (kwargs,) = spy.kwargs
    assert "num_warps" not in kwargs and "num_stages" not in kwargs
