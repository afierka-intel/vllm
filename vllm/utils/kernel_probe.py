# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-shot availability probes for compiled kernels.

A compiled op can be registered and still be unusable on a given device: the
library may have been built without that kernel family, or the kernel may
reject the device at runtime. `kernel_usable` runs the op once on a small
input, compares the result with a reference implementation and caches the
answer for the process, so callers can fall back to a portable
implementation instead of failing.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_RESULTS: dict[str, bool] = {}


def _flatten(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, Sequence):
        return [t for o in out for t in _flatten(o)]
    raise TypeError(f"probe output must be tensors, got {type(out)}")


def _is_capturing() -> bool:
    if torch.compiler.is_compiling():
        return True
    for mod_name in ("cuda", "xpu"):
        mod = getattr(torch, mod_name, None)
        if mod is None or not mod.is_available():
            continue
        fn = getattr(mod, "is_current_stream_capturing", None)
        if fn is not None and fn():
            return True
    return False


def kernel_usable(
    name: str,
    run_fn: Callable[[], Any],
    ref_fn: Callable[[], Any],
    atol: float = 0.0,
    rtol: float = 0.0,
) -> bool:
    """Return whether the compiled kernel `name` works on this device.

    `run_fn` calls the compiled kernel on a small input and `ref_fn` computes
    the expected output. An exception from `run_fn` or a mismatch beyond
    `atol`/`rtol` marks the kernel unusable. The result is cached per process
    and logged once.

    Probe from model initialization or warm-up. Probing an uncached kernel
    under torch.compile tracing or graph capture raises, because the probe
    may launch a failing kernel or synchronize the device.
    """
    cached = _RESULTS.get(name)
    if cached is not None:
        return cached
    if _is_capturing():
        raise RuntimeError(
            f"Availability of kernel {name!r} was not probed before "
            "torch.compile tracing or graph capture; probe it during "
            "initialization or warm-up."
        )

    reason = ""
    try:
        got = _flatten(run_fn())
        expected = _flatten(ref_fn())
        if len(got) != len(expected):
            reason = f"returned {len(got)} tensors, expected {len(expected)}"
        for g, e in zip(got, expected):
            if reason:
                break
            if g.shape != e.shape:
                reason = f"shape {tuple(g.shape)} != {tuple(e.shape)}"
            elif not torch.allclose(
                g.float(), e.float(), atol=atol, rtol=rtol, equal_nan=True
            ):
                diff = (g.float() - e.float()).abs().max().item()
                reason = f"output mismatch (max abs diff {diff:.3e})"
    except Exception as e:  # noqa: BLE001 - any failure means unusable
        reason = f"{type(e).__name__}: {e}".splitlines()[0]

    usable = not reason
    _RESULTS[name] = usable
    if not usable:
        logger.info_once(
            "Kernel %s is not usable on this device (%s); using the portable "
            "implementation.",
            name,
            reason,
        )
    return usable


def set_kernel_usable(name: str, usable: bool | None) -> None:
    """Override (or with None, clear) the cached probe result. For tests."""
    if usable is None:
        _RESULTS.pop(name, None)
    else:
        _RESULTS[name] = usable
