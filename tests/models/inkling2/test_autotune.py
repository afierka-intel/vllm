# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Autotuning of the portable Inkling2 kernels (XPU).

The CI step runs the other Inkling2 test files with
`VLLM_TRITON_FORCE_FIRST_CONFIG=1` (first valid config, no timing) and this
file without it, so that the tuned launch path stays covered: each test
times every config of a fresh key (no pick from memory or disk) and checks
the result of the picked one.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.models.inkling2.common.ops.linear as linear_ops
import vllm.models.inkling2.common.ops.rel_attention as rel_attention_ops
from tests.models.inkling2 import test_attention, test_core_ops
from tests.models.inkling2.utils import DEVICE
from vllm.platforms import current_platform
from vllm.triton_utils import force_first_config

pytestmark = [
    pytest.mark.skipif(not current_platform.is_xpu(), reason="autotuned on XPU"),
    pytest.mark.skipif(
        force_first_config.is_installed(),
        reason="VLLM_TRITON_FORCE_FIRST_CONFIG replaces the autotuner",
    ),
]


def _tuners():
    """Every autotuned launch of the Inkling2 kernels (factories: each
    variant)."""
    return [
        rel_attention_ops._rel_attention_kernel_tuned,
        *(linear_ops._k_group_kernel_tuned(b) for b, _ in linear_ops._TOKEN_BUCKETS),
    ]


def test_tuned_launches_cache_picks_on_disk():
    """The picks persist across processes (`cache_results`), independently
    of `TRITON_CACHE_AUTOTUNING`."""
    for tuner in _tuners():
        assert tuner.cache_results, tuner.base_fn.__name__


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


def test_k_group_tunes_and_matches_ordered_groups(cold_tuning, monkeypatch):
    test_core_ops.test_k_group_partials_match_ordered_groups(
        37, 384, 4608, torch.float32, monkeypatch
    )
    _assert_tuned(cold_tuning, "_k_group_kernel")


def test_rel_attention_prefill_tunes_and_matches_reference(cold_tuning):
    test_attention.test_inkling2_rel_attention(12, 2, 16, None, "prefill", 1, True)
    _assert_tuned(cold_tuning, "_rel_attention_kernel")


def test_rel_decode_attention_tunes_and_matches_reference(cold_tuning):
    test_attention.test_inkling2_rel_decode_attention(
        24, 8, None, [1] * 4, [1, 200, 1025, 3000], 128, 9
    )
    _assert_tuned(cold_tuning, "_rel_attention_kernel")


def test_k_group_buckets_keep_separate_tuning_keys():
    """Buckets 64 and 1024 prune to the same configs at N <= 32 (Inkling 2's
    dynamic taps); the bucket in the tuning key still keeps their picks
    apart, in memory and in Triton's on-disk cache (keyed by the tuning key
    and the configs)."""
    n, k = 32, 4608
    small = linear_ops._k_group_kernel_tuned(64)
    large = linear_ops._k_group_kernel_tuned(1024)
    args = {"N": n}
    pruned = [
        [str(c) for c in linear_ops._prune_k_group_tiles(t.configs, args)]
        for t in (small, large)
    ]
    assert pruned[0] == pruned[1]
    torch.manual_seed(0)
    w = torch.randn(n, k, device=DEVICE).bfloat16()
    for tokens in (40, 100):
        x = torch.randn(tokens, k, device=DEVICE).bfloat16()
        linear_ops.k_group_partials(x, w)
    keys = []
    for bucket, tuner in ((64, small), (1024, large)):
        assert tuner.keys[-1] == "BUCKET"
        # (N, K, SPLIT, USE_TD, BUCKET, dtypes...) of this weight shape.
        shape_keys = {key for key in tuner.cache if key[:2] == (n, k)}
        assert shape_keys and all(key[4] == bucket for key in shape_keys)
        keys.append(shape_keys)
    assert not keys[0] & keys[1]
