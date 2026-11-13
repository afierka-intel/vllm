# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 vision variant for CUDA: the platform-neutral wrapper
(`common/vl_model.py`) around the CUDA text backbone (`nvidia/model.py`)."""

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
