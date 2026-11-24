# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launcher of the NVFP4 emulation MoE kernel
(`invoke_fused_moe_nvfp4_emulation_kernel`): its launch options."""

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


def _launch(config: dict, **launch) -> None:
    """Launch the kernel over random-shaped operands (on the CPU: the spy
    only records the launch)."""
    experts, n, k, tokens, top_k = 4, 128, 256, 4, 2
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
        **launch,
    )


# A tuned-file style entry: tiles plus the launch config of fused_moe_kernel.
CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 8,
    "num_stages": 3,
}
# The launch kwargs of `main`'s launcher besides the operands and strides.
MAIN_KWARGS = {
    "block_k_diviable",
    "MUL_ROUTED_WEIGHT",
    "top_k",
    "compute_type",
    "group_size",
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
}


@pytest.fixture
def spy(monkeypatch):
    spy = _LaunchSpy()
    monkeypatch.setattr(
        nvfp4_emulation_moe.fused_moe_nvfp4_emulation_kernel, "run", spy
    )
    return spy


@pytest.mark.parametrize(
    "launch_options",
    [
        {"num_warps": 8, "num_stages": 3},
        {"num_warps": 2, "num_stages": 4, "num_ctas": 1},
        {"num_warps": 16, "num_stages": 2, "grf_mode": "256"},
    ],
)
def test_caller_launch_options_reach_the_launch(spy, launch_options):
    """`launch_options` reach the kernel launch as they are, whatever the
    config holds (fails before the option existed: unexpected keyword)."""
    _launch({**CONFIG, "num_warps": 4}, launch_options=launch_options)
    (kwargs,) = spy.kwargs
    for name, value in launch_options.items():
        assert kwargs[name] == value
    assert "SPLIT_K" not in kwargs


@pytest.mark.parametrize("launch_options", [None, {}])
def test_default_launch_keeps_triton_defaults(spy, launch_options):
    """Without caller options the launch kwargs are the ones the launcher
    has always passed: the config's `num_warps` / `num_stages` (tuned for
    `fused_moe_kernel`) are not forwarded, so Triton's defaults apply."""
    launch = {} if launch_options is None else {"launch_options": launch_options}
    _launch(CONFIG, **launch)
    (kwargs,) = spy.kwargs
    # `grid` / `warmup` come from `JITFunction.__getitem__`.
    assert set(kwargs) - {"grid", "warmup"} == MAIN_KWARGS
