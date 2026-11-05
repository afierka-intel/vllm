# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash. XPU runs the shared ``common/`` code with the ``nvidia/`` Triton
kernels and the XPU MQA-logits ops (see ``nvidia/sparse_indexer.py``)."""

from .common.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration
from .common.mtp import Glm5NextMTP

__all__ = [
    "Glm5NextForCausalLM",
    "Glm5NextForConditionalGeneration",
    "Glm5NextMTP",
]
