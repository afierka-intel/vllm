# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 model state for CUDA: the platform-neutral state
(`common/model_state.py`) plus the decoder replay CUDA graphs."""

import torch

from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
from vllm.models.deepseek_v41.nvidia.decoder_replay_cudagraph import (
    DecoderReplayCudaGraphManager,
)

from ..common.model_state import DeepseekV41ModelStateBase


class DeepseekV41ModelState(DeepseekV41ModelStateBase):
    __doc__ = DeepseekV41ModelStateBase.__doc__

    replay_cudagraphs: DecoderReplayCudaGraphManager | None

    def _make_replay_cudagraphs(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        layers: DecoderReplayLayers | None,
    ) -> DecoderReplayCudaGraphManager | None:
        compilation_config = vllm_config.compilation_config
        if (
            layers is not None
            and compilation_config.cudagraph_decoder_replay
            and compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
            and is_breakable_cudagraph_enabled(compilation_config)
            and get_pp_group().is_first_rank
            # The graphs' fixed inputs hold no LoRA mapping and no None input ids.
            and vllm_config.lora_config is None
            and not vllm_config.model_config.enable_prompt_embeds
        ):
            threshold = compilation_config.decoder_replay_trim_threshold
            if threshold <= layers.window:
                raise ValueError(
                    "decoder_replay_trim_threshold must exceed the replay window "
                    f"of {layers.window} tokens: smaller graphs have nothing to trim"
                )
            replay_cudagraphs = DecoderReplayCudaGraphManager(
                vllm_config, device, layers
            )
            layers.trim_threshold = threshold
            return replay_cudagraphs
        return None
