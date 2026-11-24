# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Autotuning of the portable Triton kernels (`use_kernel_autotune`: XPU).

Each tuned launch site benchmarks its configs once per tuning key (shape
class, path and, where the best tile depends on it, a token bucket; Triton
appends the argument dtypes) and then reuses the pick. The first config of
every list is the launch config the kernel used before autotuning (its
fixed config for that key class), so `VLLM_TRITON_FORCE_FIRST_CONFIG=1`
(first valid config, no timing) runs the pre-autotuning configs; with
`VLLM_TRITON_USE_TD=0` too, it runs the pre-series pointer launches.

First-use cost: a key that is not cached yet compiles and times every config
that survives pruning before the launch returns, from about a second (row
kernels) to tens of seconds per key on a B70, and longer with
`VLLM_TRITON_USE_TD=0`; the engine step that meets a new key blocks for that
long. The profile and warmup dummy runs tune the keys of their own uniform
batches; real batches then reach new keys.

The picks persist on disk because vLLM sets `TRITON_CACHE_AUTOTUNING=1`
(`env_override.py`; `=0` re-tunes in every process). A pick is the fastest
config of one timing run for the first shape met with that key; clear the
Triton cache dir to re-tune. `VLLM_BATCH_INVARIANT=1` runs the fixed launch
configs, without autotuning and without tensor descriptors.
"""

from collections.abc import Sequence
from typing import Any

from vllm.triton_utils import triton


def autotune(configs: Sequence[Any], key: Sequence[str], **kwargs: Any):
    """`triton.autotune` over `configs`; `configs[0]` must be the launch
    config the kernel used before autotuning."""
    return triton.autotune(configs=list(configs), key=list(key), **kwargs)
