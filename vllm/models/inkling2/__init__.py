# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2: the model and its portable kernels live under `common/`, the
CUDA-only kernels and features (Blackwell / SM12x / Humming kernels, fused
all-reduce, vision tower, DSpark draft, LoRA) under `nvidia/`; configs,
processing, quantization metadata and the DSpark speculator are shared."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .common.loop import Inkling2LoopForConditionalGeneration
    from .common.model import Inkling2ForConditionalGeneration
    from .nvidia.dspark import Inkling2DSparkModel

# The module of each export.
_MODULES = {
    "Inkling2DSparkModel": "nvidia.dspark",
    "Inkling2ForConditionalGeneration": "common.model",
    "Inkling2LoopForConditionalGeneration": "common.loop",
}

__all__ = [
    "Inkling2DSparkModel",
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
]


def __getattr__(name: str):
    if name in _MODULES:
        return getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
