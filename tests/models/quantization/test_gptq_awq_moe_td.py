# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E correctness test for the tensor-descriptor (TD) load path in
fused_moe_kernel_gptq_awq (VLLMZ-1811).

Compares real-model greedy-decode output of a GPTQ-int4 MoE model with
VLLM_TRITON_USE_TD=0 vs =1. Complements the kernel-level unit tests in
tests/kernels/moe/test_fused_moe_kernel_gptq_awq.py, which never load a real
model. XPU-only: resolve_moe_gptq_awq_use_td() always returns False off XPU,
so this comparison would be a vacuous no-op elsewhere.

MOE_MODEL's native shape (hidden_size=2048, moe_intermediate_size=1408,
group_size=128) satisfies check_moe_marlin_supports_layer(), so the default
quant method would dispatch to the native XPUExpertsWNA16/Marlin path, never
reaching fused_moe_kernel_gptq_awq at all. quantization="moe_wna16" is an
explicit, already-wired vLLM override (MoeWNA16Config.override_quantization_
method) that skips that Marlin-eligibility gate entirely and always calls
fused_experts() -> invoke_fused_moe_wna16_triton_kernel() -> this kernel,
regardless of shape -- required here so the test actually exercises the
code this PR changes.
"""

import pytest

from vllm.envs import disable_envs_cache
from vllm.platforms import current_platform

from ..utils import check_logprobs_close

MOE_MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4"
MAX_MODEL_LEN = 1024
MAX_TOKENS = 32
NUM_LOG_PROBS = 5


@pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="TD path for fused_moe_kernel_gptq_awq is XPU-only.",
)
def test_gptq_moe_td_vs_pointer_logprobs_close(
    vllm_runner,
    example_prompts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TD and pointer paths must produce matching output for a real
    GPTQ-int4 MoE model, forced through fused_moe_kernel_gptq_awq via
    quantization="moe_wna16" (see module docstring for why the plain
    default quant method would bypass this kernel entirely).

    No @pytest.mark.flaky here: unlike Marlin's lock-based nondeterminism
    (see test_gptq_marlin.py), the TD path introduces no new source of
    nondeterminism.
    """
    prompts = example_prompts[:4]

    with monkeypatch.context() as m:
        m.setenv("VLLM_TRITON_USE_TD", "0")
        disable_envs_cache()
        with vllm_runner(
            MOE_MODEL,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
            quantization="moe_wna16",
        ) as vllm_model:
            pointer_outputs = vllm_model.generate_greedy_logprobs(
                prompts, MAX_TOKENS, NUM_LOG_PROBS
            )

    with monkeypatch.context() as m:
        m.setenv("VLLM_TRITON_USE_TD", "1")
        disable_envs_cache()
        with vllm_runner(
            MOE_MODEL,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
            quantization="moe_wna16",
        ) as vllm_model:
            td_outputs = vllm_model.generate_greedy_logprobs(
                prompts, MAX_TOKENS, NUM_LOG_PROBS
            )

    check_logprobs_close(
        outputs_0_lst=pointer_outputs,
        outputs_1_lst=td_outputs,
        name_0="pointer",
        name_1="td",
    )
