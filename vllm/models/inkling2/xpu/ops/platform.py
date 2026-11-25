# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side switches of the Inkling2 XPU Triton kernels.

The CUDA predicates of `nvidia/` (PDL, TMA, SM100/SM12x kernels, native
conv/mix, native e4m3) are all false on XPU, so the copies in this package
contain only the paths that run here. What remains is the choice between the
autotuned launches (with their tensor-descriptor paths) and the fixed
pointer launches.
"""

import functools


@functools.cache
def use_kernel_autotune() -> bool:
    """Whether the kernels autotune their tiles: always, unless
    `VLLM_BATCH_INVARIANT` is set. A timing-based pick can differ from run
    to run, so that mode runs the fixed launch configs (the first config of
    every tuned list)."""
    from vllm import envs

    return not envs.VLLM_BATCH_INVARIANT


def use_kernel_td(use_td: bool | None = None) -> bool:
    """Whether the kernels take their tensor-descriptor paths: on the
    autotuned launch path, where `use_tensor_descriptor(use_td)` holds
    (`VLLM_TRITON_USE_TD`; unset = on for XPU). Under `VLLM_BATCH_INVARIANT`
    every kernel runs its fixed pointer launch."""
    from vllm.triton_utils import use_tensor_descriptor

    return use_kernel_autotune() and use_tensor_descriptor(use_td)
