# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Autotuning of the portable Inkling2 kernels (XPU).

The CI step runs the other Inkling2 test files with
`VLLM_TRITON_FORCE_FIRST_CONFIG=1` (first config, no timing: the fixed
launch configs) and this file without it, so that the tuned launch path
stays covered: every config of each tuned launch is checked against the
reference (one config at a time), and the tuned launch times every config
of a fresh key (no pick from memory or disk) and checks its pick.
"""

from types import SimpleNamespace

import pytest

import vllm.models.inkling2.common.ops.attention_sm90 as rel_attention_ops
from tests.models.inkling2 import test_attention
from vllm.platforms import current_platform
from vllm.triton_utils import force_first_config


def _tuners():
    """Every autotuned launch of the Inkling2 kernels."""
    return [*rel_attention_ops._rel_attention_kernel_tuned.values()]


# The first config of each tuned launch: the launch config the kernel used
# before autotuning (`VLLM_TRITON_FORCE_FIRST_CONFIG=1` takes it). Literal
# values, (tuner, kwargs, num_warps, num_stages).
FIXED_FIRST = [
    # _rel_attention_kernel: decode (BLOCK_M <= 32), 33-127 rows, prefill.
    (lambda: rel_attention_ops._rel_attention_tuner(16), {"BLOCK_N": 128}, 4, 2),
    (lambda: rel_attention_ops._rel_attention_tuner(32), {"BLOCK_N": 128}, 4, 2),
    (lambda: rel_attention_ops._rel_attention_tuner(64), {"BLOCK_N": 64}, 4, 2),
    (lambda: rel_attention_ops._rel_attention_tuner(128), {"BLOCK_N": 64}, 8, 2),
]


@pytest.mark.parametrize("case", range(len(FIXED_FIRST)))
def test_first_config_is_the_fixed_config(case):
    """`configs[0]` of every tuned launch is its pre-autotuning launch config
    (runs on any platform: the tuners are built at import)."""
    tuner, kwargs, num_warps, num_stages = FIXED_FIRST[case]
    first = tuner().configs[0]
    assert (first.kwargs, first.num_warps, first.num_stages) == (
        kwargs,
        num_warps,
        num_stages,
    )


def test_every_tuner_has_a_fixed_first_case():
    """`FIXED_FIRST` covers every tuned launch."""
    covered = {id(tuner()) for tuner, *_ in FIXED_FIRST}
    assert {id(tuner) for tuner in _tuners()} <= covered


def test_tuned_launches_do_not_force_the_disk_cache():
    """The picks persist only where Triton's autotuning cache is on (vLLM's
    default `TRITON_CACHE_AUTOTUNING=1`), so `=0` re-tunes."""
    for tuner in _tuners():
        assert not getattr(tuner, "cache_results", False), tuner.base_fn.__name__


requires_xpu = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="autotuned on XPU"
)
requires_tuning = pytest.mark.skipif(
    force_first_config.is_installed(),
    reason="VLLM_TRITON_FORCE_FIRST_CONFIG replaces the autotuner",
)


@pytest.fixture
def cold_tuning(monkeypatch):
    """Times every key met in the test: no pick from memory or disk. Yields
    the tuning events `(name, configs timed, cache hit)`."""
    from triton import knobs
    from triton.runtime.autotuner import Autotuner

    events: list[SimpleNamespace] = []

    def listener(fn, key, best_config, configs_timings, duration, cache_hit):
        events.append(
            SimpleNamespace(
                name=fn.__name__, timed=len(configs_timings or {}), hit=cache_hit
            )
        )

    monkeypatch.setattr(knobs.autotuning, "listener", listener)
    monkeypatch.setattr(
        Autotuner,
        "check_disk_cache",
        lambda self, key, configs, bench_fn: bool(bench_fn()),
    )
    seen: set[int] = set()
    run = Autotuner.run

    def cold_run(self, *args, **kwargs):
        if id(self) not in seen:
            seen.add(id(self))
            self.cache.clear()
        return run(self, *args, **kwargs)

    monkeypatch.setattr(Autotuner, "run", cold_run)
    return events


def _assert_tuned(events, *names):
    for name in names:
        assert any(e.name == name and not e.hit and e.timed > 1 for e in events), (
            name,
            events,
        )


def _pin(monkeypatch, tuners, index):
    """Each tuner launches only its `index`-th config (no benchmarking), or
    the test is skipped when a tuner has fewer configs."""
    for tuner in tuners:
        if index >= len(tuner.configs):
            pytest.skip(f"{tuner.base_fn.__name__} has {len(tuner.configs)} configs")
        monkeypatch.setattr(tuner, "configs", [tuner.configs[index]])
        tuner.cache.clear()


# The prefill cases run at page 64: the XPU default page, K/V descriptors
# with TD on.


@requires_xpu
@pytest.mark.parametrize("index", range(len(rel_attention_ops._TUNED_TILES)))
def test_rel_attention_every_config_matches_reference(monkeypatch, index):
    """Each config of the prefill and decode tuners, alone, against the
    reference (with TD, a key tile that does not divide the page is pruned
    and skipped)."""
    prefill = rel_attention_ops._rel_attention_tuner(128)
    decode = rel_attention_ops._rel_attention_tuner(16)
    _pin(monkeypatch, [prefill, decode], index)
    if prefill.configs[0].kwargs["BLOCK_N"] <= 64:
        test_attention.test_inkling2_rel_attention(12, 2, 64, None, "prefill", 1, True)
    test_attention.test_inkling2_rel_decode_attention(
        24, 8, None, [1] * 4, [1, 200, 1025, 3000], 128, 9
    )


@requires_xpu
@requires_tuning
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_rel_attention_tunes_and_matches_reference(cold_tuning, phase):
    """The tuned launch times every config of a fresh key and its pick
    matches the reference."""
    if phase == "prefill":
        test_attention.test_inkling2_rel_attention(12, 2, 64, None, "prefill", 1, True)
    else:
        test_attention.test_inkling2_rel_decode_attention(
            24, 8, None, [1] * 4, [1, 200, 1025, 3000], 128, 9
        )
    _assert_tuned(cold_tuning, "_rel_attention_kernel")
