# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2: platform-specific implementations live under `nvidia/` and
`xpu/` (Intel GPUs, text only); configs, processing, quantization metadata
and the DSpark speculator are shared."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nvidia.dspark import Inkling2DSparkModel
    from .nvidia.loop import Inkling2LoopForConditionalGeneration
    from .nvidia.model import Inkling2ForConditionalGeneration

# The module of each export in the platform's package.
_MODULES = {
    "Inkling2DSparkModel": "dspark",
    "Inkling2ForConditionalGeneration": "model",
    "Inkling2LoopForConditionalGeneration": "loop",
}
# XPU has no DSpark draft: `xpu/model.py` names it and rejects it at init.
_XPU_MODULES = {**_MODULES, "Inkling2DSparkModel": "model"}

__all__ = [
    "Inkling2DSparkModel",
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
]


def __getattr__(name: str):
    if name in _MODULES:
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            module = f"{__name__}.xpu.{_XPU_MODULES[name]}"
        else:
            module = f"{__name__}.nvidia.{_MODULES[name]}"
        return getattr(import_module(module), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
