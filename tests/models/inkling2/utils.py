# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform helpers shared by the Inkling2 kernel tests (CUDA and XPU)."""

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
