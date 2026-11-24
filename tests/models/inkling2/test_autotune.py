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
import torch

import vllm.models.inkling2.common.ops.attention as attention_ops
import vllm.models.inkling2.common.ops.attention_sm90 as rel_attention_ops
import vllm.models.inkling2.common.ops.linear as linear_ops
import vllm.models.inkling2.common.ops.moe as moe_ops
import vllm.models.inkling2.common.ops.sconv as sconv_ops
from tests.models.inkling2 import test_attention, test_core_ops
from tests.models.inkling2.utils import DEVICE
from vllm.platforms import current_platform
from vllm.triton_utils import force_first_config


def _tuners():
    """Every autotuned launch of the Inkling2 kernels."""
    return [
        *rel_attention_ops._rel_attention_kernel_tuned.values(),
        *(linear_ops._k_group_kernel_tuned(b) for b, _ in linear_ops._TOKEN_BUCKETS),
        *(
            sconv_ops._sconv_kernel_tuned(b, prefill)
            for b, _ in sconv_ops._SCONV_BUCKETS
            for prefill in (False, True)
        ),
        *(moe_ops._select_kernel_tuned(small) for small in (False, True)),
        *(
            attention_ops._rel_decode_combine_kernel_tuned(large, many)
            for large, many in ((False, False), (True, False), (True, True))
        ),
    ]


# The first config of each tuned launch: the launch config the kernel used
# before autotuning (`VLLM_TRITON_FORCE_FIRST_CONFIG=1` takes it). Literal
# values, (tuner, kwargs, num_warps, num_stages).
FIXED_FIRST = [
    # _rel_attention_kernel: decode (BLOCK_M <= 32), 33-127 rows, prefill.
    (lambda: rel_attention_ops._rel_attention_tuner(16), {"BLOCK_N": 128}, 4, 2),
    (lambda: rel_attention_ops._rel_attention_tuner(32), {"BLOCK_N": 128}, 4, 2),
    (lambda: rel_attention_ops._rel_attention_tuner(64), {"BLOCK_N": 64}, 4, 2),
    (lambda: rel_attention_ops._rel_attention_tuner(128), {"BLOCK_N": 64}, 8, 2),
    # _k_group_kernel per token bucket (XPU's fixed tiles: 2 stages).
    (lambda: linear_ops._k_group_kernel_tuned(16), {"BM": 16, "BN": 32}, 4, 2),
    (lambda: linear_ops._k_group_kernel_tuned(128), {"BM": 16, "BN": 64}, 4, 2),
    (lambda: linear_ops._k_group_kernel_tuned(256), {"BM": 64, "BN": 64}, 4, 2),
    (lambda: linear_ops._k_group_kernel_tuned(1 << 62), {"BM": 64, "BN": 128}, 8, 2),
    # _sconv_kernel per token bucket, decode / prefill (the fixed launch:
    # XPU's default 2 stages).
    *(
        (
            lambda p=p: sconv_ops._sconv_kernel_tuned(8, p),
            {"BT": 1, "BLOCK_C": 1024},
            4,
            2,
        )
        for p in (False, True)
    ),
    *(
        (
            lambda b=b: sconv_ops._sconv_kernel_tuned(b, False),
            {"BT": 2, "BLOCK_C": 512},
            4,
            2,
        )
        for b in (256, 1 << 62)
    ),
    *(
        (
            lambda b=b: sconv_ops._sconv_kernel_tuned(b, True),
            {"BT": 4, "BLOCK_C": 256},
            4,
            2,
        )
        for b in (256, 1 << 62)
    ),
    # _select_kernel, up to 256 tokens / more (XPU default 2 stages).
    (lambda: moe_ops._select_kernel_tuned(True), {}, 4, 2),
    (lambda: moe_ops._select_kernel_tuned(False), {}, 1, 2),
    # _rel_decode_combine_kernel: small batches, large with <= 32 / more
    # splits.
    (
        lambda: attention_ops._rel_decode_combine_kernel_tuned(False, False),
        {"BLOCK_D": 32},
        4,
        2,
    ),
    (
        lambda: attention_ops._rel_decode_combine_kernel_tuned(True, False),
        {"BLOCK_D": 128},
        1,
        2,
    ),
    (
        lambda: attention_ops._rel_decode_combine_kernel_tuned(True, True),
        {"BLOCK_D": 128},
        2,
        2,
    ),
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


@requires_xpu
@requires_tuning
def test_k_group_tunes_and_matches_ordered_groups(cold_tuning, monkeypatch):
    """The tuned K-group launch times every config of a fresh key, and its
    pick matches the ordered K-group sums. (Every K-group config gives the
    bits of the fixed one: `test_k_group_tiles_are_bitwise_identical`.)"""
    # 37 tokens at N = 128: a regime that tunes (`_FIXED_ONLY_ABOVE_N`).
    test_core_ops.test_k_group_partials_match_ordered_groups(
        37, 128, 4608, torch.float32, monkeypatch
    )
    _assert_tuned(cold_tuning, "_k_group_kernel")


@pytest.mark.parametrize(
    "bucket,n,tuned",
    [
        (16, 32, False),
        (16, 384, False),
        (128, 128, True),
        (128, 384, False),
        (256, 384, True),
        (1 << 62, 32, False),
        (1 << 62, 128, False),
    ],
)
def test_k_group_tunes_only_where_it_gained(bucket, n, tuned):
    """The K-group tuners time more than one tile only in the regimes where
    tuning measured a gain (any platform: the tuners are built at import)."""
    tuner = linear_ops._k_group_kernel_tuned(bucket)
    kept = linear_ops._prune_k_group_tiles(tuner.configs, {"N": n, "BUCKET": bucket})
    assert (len(kept) > 1) == tuned


@requires_xpu
@requires_tuning
def test_k_group_buckets_keep_separate_tuning_keys():
    """Buckets 128 and 256 prune to the same configs at N <= 32 (Inkling 2's
    dynamic taps); the bucket in the tuning key still keeps their picks
    apart, in memory and in Triton's on-disk cache (keyed by the tuning key
    and the configs)."""
    n, k = 32, 4608
    small = linear_ops._k_group_kernel_tuned(128)
    large = linear_ops._k_group_kernel_tuned(256)
    pruned = [
        [
            str(c)
            for c in linear_ops._prune_k_group_tiles(
                t.configs, {"N": n, "BUCKET": bucket}
            )
        ]
        for t, bucket in ((small, 128), (large, 256))
    ]
    assert pruned[0] == pruned[1]
    torch.manual_seed(0)
    w = torch.randn(n, k, device=DEVICE).bfloat16()
    for tokens in (100, 200):
        x = torch.randn(tokens, k, device=DEVICE).bfloat16()
        linear_ops.k_group_partials(x, w)
    keys = []
    for bucket, tuner in ((128, small), (256, large)):
        assert tuner.keys[-1] == "BUCKET"
        # (N, K, SPLIT, USE_TD, BUCKET, dtypes...) of this weight shape.
        shape_keys = {key for key in tuner.cache if key[:2] == (n, k)}
        assert shape_keys and all(key[4] == bucket for key in shape_keys)
        keys.append(shape_keys)
    assert not keys[0] & keys[1]


@requires_xpu
@requires_tuning
def test_short_conv_tunes_and_matches_full_sequence(cold_tuning):
    """The tuned short conv times every config of a fresh key and its pick
    matches the full-sequence conv."""
    test_core_ops.test_short_conv_chunked_matches_full_sequence(
        False, test_core_ops.LONG_STEPS, 0
    )
    _assert_tuned(cold_tuning, "_sconv_kernel")


@requires_xpu
@pytest.mark.parametrize("index", range(len(sconv_ops._SCONV_TILES)))
def test_short_conv_every_config_matches_full_sequence(monkeypatch, index):
    """Each config of the short-conv tuners, alone, against the
    full-sequence conv."""
    tuners = [
        sconv_ops._sconv_kernel_tuned(b, prefill)
        for b, _ in sconv_ops._SCONV_BUCKETS
        for prefill in (False, True)
    ]
    _pin(monkeypatch, tuners, index)
    test_core_ops.test_short_conv_chunked_matches_full_sequence(
        False, test_core_ops.LONG_STEPS, 0
    )


@pytest.mark.parametrize(
    "bucket,fused,channels,prefill,fixed_only",
    [
        (8, False, 4608, False, True),
        (8, True, 2304, False, True),
        (256, False, 2304, False, False),
        (256, True, 2304, False, True),
        (256, True, 4608, False, False),
        (1 << 62, False, 4608, True, False),
        (1 << 62, True, 4608, True, True),
        (1 << 62, True, 2304, True, False),
    ],
)
def test_short_conv_tunes_only_where_it_gained(
    bucket, fused, channels, prefill, fixed_only
):
    """The short conv runs its fixed tile alone where tuning gave no stable
    gain (any platform)."""
    args = {
        "BUCKET": bucket,
        "HAS_RES": fused,
        "H": 1,
        "WS": channels,
        "SKIP_IDLE_CACHE": prefill,
    }
    assert sconv_ops._sconv_fixed_only(args) == fixed_only
