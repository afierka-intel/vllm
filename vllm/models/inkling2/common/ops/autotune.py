# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Autotuning of the portable Triton kernels (`use_kernel_autotune`: XPU).

Each tuned launch site benchmarks its configs once per tuning key (shape
class, path and, where the best tile depends on it, a token bucket; Triton
appends the argument dtypes) and then reuses the pick.

First-use cost: a key that is not cached yet compiles and times every config
that survives pruning before the launch returns, from about a second (row
kernels) to tens of seconds (`_rel_attention_kernel`, `_qr_rel_kernel`) per
key on a B70, and longer with `VLLM_TRITON_USE_TD=0`; the engine step that
meets a new key blocks for that long. The profile and warmup dummy runs tune
only the keys of their own batches (uniform requests; the attention-side
kernels do not run in them), so with an empty cache the first real requests tune the
attention-side keys and the token buckets they reach (Inkling-2-Mini on one
B70: about 4 minutes over the first requests of a 1024-token batch).

The picks are cached on disk (`cache_results=True`, under
`TRITON_CACHE_DIR`), keyed by the Triton version, the backend target, the
kernel source, the cache-invalidating environment, the tuning key and the
configs, so a later process reuses them without timing. A pick is the
fastest config of one timing run; clear the cache dir to re-tune.

`VLLM_TRITON_FORCE_FIRST_CONFIG=1` skips the timing (first valid config:
the fixed config of the kernel, or the smallest valid tile);
`VLLM_BATCH_INVARIANT=1` runs the fixed launch configs without autotuning.
"""

import inspect
from collections.abc import Sequence
from typing import Any

from vllm.triton_utils import triton

_CACHE_RESULTS = (
    "cache_results" in inspect.signature(triton.autotune).parameters
    if callable(getattr(triton, "autotune", None))
    else False
)


def autotune(configs: Sequence[Any], key: Sequence[str], **kwargs: Any):
    """`triton.autotune` with the picks cached on disk where the installed
    Triton supports it (`cache_results`; Triton 3.2+, incl. triton-xpu)."""
    if _CACHE_RESULTS:
        kwargs.setdefault("cache_results", True)
    return triton.autotune(configs=list(configs), key=list(key), **kwargs)
