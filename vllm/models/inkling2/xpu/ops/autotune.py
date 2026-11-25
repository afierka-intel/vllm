# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Autotuning of the Inkling2 XPU Triton kernels.

Each tuned launch site benchmarks its configs once per tuning key (shape
class, path and, where the best tile depends on it, a token bucket; Triton
appends the argument dtypes) and then reuses the pick. The first config of
every list is the fixed launch config of that key class, so
`VLLM_TRITON_FORCE_FIRST_CONFIG=1` (first valid config, no timing) runs the
fixed configs, and `VLLM_BATCH_INVARIANT=1` runs them without autotuning and
without tensor descriptors (`platform.use_kernel_autotune`).

First-use cost: a key that is not cached yet compiles and times every config
that survives pruning before the launch returns, from about a second (row
kernels) to tens of seconds per key on a B70; the engine step that meets a
new key blocks for that long. The picks persist on disk because vLLM sets
`TRITON_CACHE_AUTOTUNING=1` (`env_override.py`); clear the Triton cache dir
to re-tune.
"""

from collections.abc import Sequence
from typing import Any

from vllm.triton_utils import triton


def autotune(configs: Sequence[Any], key: Sequence[str], **kwargs: Any):
    """`triton.autotune` over `configs`; `configs[0]` must be the fixed
    launch config of the kernel."""
    return triton.autotune(configs=list(configs), key=list(key), **kwargs)
