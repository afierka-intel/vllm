# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.utils import kernel_probe
from vllm.utils.kernel_probe import kernel_usable, set_kernel_usable


@pytest.fixture(autouse=True)
def _clear_cache():
    kernel_probe._RESULTS.clear()
    yield
    kernel_probe._RESULTS.clear()


def test_matching_kernel_is_usable():
    x = torch.arange(8, dtype=torch.float32)
    assert kernel_usable("ok", lambda: x * 2, lambda: x + x)


def test_exception_marks_unusable_and_is_cached():
    calls = []

    def run():
        calls.append(1)
        raise RuntimeError("kernel not supported on this device")

    assert not kernel_usable("raises", run, lambda: torch.zeros(1))
    assert not kernel_usable("raises", run, lambda: torch.zeros(1))
    assert len(calls) == 1


def test_mismatch_marks_unusable():
    x = torch.ones(4)
    assert not kernel_usable("wrong", lambda: x * 3, lambda: x)
    assert kernel_usable("close", lambda: x + 1e-4, lambda: x, atol=1e-3)


def test_shape_and_count_mismatch():
    x = torch.ones(4)
    assert not kernel_usable("shape", lambda: x[:2], lambda: x)
    assert not kernel_usable("count", lambda: (x, x), lambda: x)


def test_override():
    set_kernel_usable("forced", False)
    assert not kernel_usable("forced", lambda: torch.ones(1), lambda: torch.ones(1))
    set_kernel_usable("forced", None)
    assert kernel_usable("forced", lambda: torch.ones(1), lambda: torch.ones(1))


def test_uncached_probe_during_compile_raises(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match="not probed"):
        kernel_usable("late", lambda: torch.ones(1), lambda: torch.ones(1))
    set_kernel_usable("late", False)
    assert not kernel_usable("late", lambda: torch.ones(1), lambda: torch.ones(1))
