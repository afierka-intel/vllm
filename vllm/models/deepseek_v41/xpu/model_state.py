# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 model state for XPU.

The platform-neutral state (`common/model_state.py`): the engram lookback
window, the prefix-cache SWA replay slot padding and the decoder-side replay
batch. No decoder replay CUDA graphs (`DecoderReplayCudaGraphManager`): the
replay batch always runs eagerly, and
`compilation_config.cudagraph_decoder_replay` is ignored.
"""

from ..common.model_state import DeepseekV41ModelStateBase


class DeepseekV41ModelState(DeepseekV41ModelStateBase):
    __doc__ = DeepseekV41ModelStateBase.__doc__
