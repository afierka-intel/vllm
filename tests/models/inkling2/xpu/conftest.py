# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


@pytest.fixture
def kernel_env(monkeypatch):
    """`kernel_env(**variables)` sets environment variables that the host
    switches of the Inkling2 XPU kernels read (`VLLM_BATCH_INVARIANT`,
    `VLLM_TRITON_USE_TD`), live: the envs cache is off and the cached
    `use_kernel_autotune` is recomputed, also after the test."""
    from vllm import envs
    from vllm.models.inkling2.xpu.ops.platform import use_kernel_autotune

    # An engine started earlier in the session may have cached envs.
    envs.disable_envs_cache()

    def apply(**variables: str) -> None:
        for name, value in variables.items():
            monkeypatch.setenv(name, value)
        use_kernel_autotune.cache_clear()

    yield apply
    monkeypatch.undo()
    use_kernel_autotune.cache_clear()
