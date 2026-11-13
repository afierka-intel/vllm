# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 vision variant for XPU: the platform-neutral wrapper
(`common/vl_model.py`) around the XPU text backbone (`xpu/model.py`).

Nothing is dropped: the vision tower, aligner and the encoder CUDA-graph mixin
(`common/vl_cudagraph.py`, opt-in through `cudagraph_mm_encoder`) are
vendor-neutral.
"""

from ..common.vl_model import (
    DeepseekV4VLImagePixelInputs,
    DeepseekV41ForCausalLMBase,
    _make_deepseek_v4_vl_weights_mapper,
)
from .model import DeepseekV41LLMForCausalLM

__all__ = [
    "DeepseekV41ForCausalLM",
    "DeepseekV4VLImagePixelInputs",
    "_make_deepseek_v4_vl_weights_mapper",
]


class DeepseekV41ForCausalLM(DeepseekV41ForCausalLMBase):
    __doc__ = DeepseekV41ForCausalLMBase.__doc__

    language_model_cls = DeepseekV41LLMForCausalLM
