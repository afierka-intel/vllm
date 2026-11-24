# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform and dequantization helpers shared by the Inkling2 kernel tests
(CUDA and XPU)."""

import torch

from vllm.platforms import current_platform

DEVICE = current_platform.device_type


def device_graph():
    """A device graph and the context manager that captures it."""
    if current_platform.is_xpu():
        graph = torch.xpu.XPUGraph()
        return graph, torch.xpu.graph(graph)
    graph = torch.cuda.CUDAGraph()
    return graph, torch.cuda.graph(graph)


# e2m1 values of the 16 FP4 codes.
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def dequant_nvfp4(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 `[..., K]` of packed e2m1 `values` `[..., K / 2]` (low nibble
    first) times their e4m3 block `scales` `[..., K / 16]`."""
    lut = torch.tensor(E2M1, device=values.device)
    v = torch.stack([lut[(values & 0xF).long()], lut[(values >> 4).long()]], -1)
    return v.flatten(-2) * scales.float().repeat_interleave(16, -1)


def dequant_mxfp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 `[..., K]` of e4m3 `values` times their e8m0 block `scales`
    `[..., K / 32]` (uint8)."""
    exp2 = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    return values.float() * exp2
