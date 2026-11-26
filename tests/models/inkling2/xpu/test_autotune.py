# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Autotuning of the Inkling2 XPU kernels.

The CI step runs the other files of this package with
`VLLM_TRITON_FORCE_FIRST_CONFIG=1` (first valid config, no timing: the fixed
launch configs) and this file without it, so that the tuned launch path stays
covered: each tuned launch times its configs on a fresh key (no pick from
memory or disk) and its pick matches the reference. The host-side checks
(first configs, pruning, switches) run either way.
"""

import ast
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
import torch

import vllm.models.inkling2.xpu as xpu_package
from tests.models.inkling2.xpu import test_attention, test_moe, test_ops
from tests.models.inkling2.xpu.utils import LONG_STEPS, REL_EXTENT, requires_xpu
from vllm.models.inkling2.xpu.ops import attention as attention_ops
from vllm.models.inkling2.xpu.ops import linear as linear_ops
from vllm.models.inkling2.xpu.ops import moe as moe_ops
from vllm.models.inkling2.xpu.ops import nvfp4_moe
from vllm.models.inkling2.xpu.ops import rel_attention as rel_ops
from vllm.models.inkling2.xpu.ops import sconv as sconv_ops
from vllm.models.inkling2.xpu.ops.autotune import autotune
from vllm.models.inkling2.xpu.ops.platform import use_kernel_autotune, use_kernel_td
from vllm.triton_utils import force_first_config, tl, triton

pytestmark = requires_xpu
requires_tuning = pytest.mark.skipif(
    force_first_config.is_installed(),
    reason="VLLM_TRITON_FORCE_FIRST_CONFIG=1 replaces the autotuner",
)


def _nvfp4_tuner(split: bool):
    """The NVFP4 expert GEMM tuner of a split-dots (1 token) or default
    (3072 tokens) Mini batch: the batch's config tile first."""
    if split:
        config = nvfp4_moe.split_dots_config(1, 8, 128)
    else:
        config = nvfp4_moe.default_config(3072, 128)
    fixed = (
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["num_warps"],
        config["num_stages"],
    )
    return nvfp4_moe._nvfp4_moe_kernel_tuned(split, fixed), fixed


def _tuners():
    """Every autotuned launch of the Inkling2 XPU kernels with a fixed tile
    of its own (the NVFP4 tuners are built per config tile)."""
    return [
        *rel_ops._rel_attention_kernel_tuned.values(),
        *(linear_ops._k_group_kernel_tuned(b) for b, _ in linear_ops._TOKEN_BUCKETS),
        *(sconv_ops._sconv_kernel_tuned(prefill) for prefill in (False, True)),
        *(moe_ops._select_kernel_tuned(small) for small in (False, True)),
        attention_ops._rel_decode_combine_kernel_tuned(),
    ]


# The first config of each tuned launch: the fixed launch config (the one
# `VLLM_BATCH_INVARIANT=1` runs, and `VLLM_TRITON_FORCE_FIRST_CONFIG=1`
# takes). Literal values, (tuner, kwargs, num_warps, num_stages).
FIXED_FIRST = [
    # _rel_attention_kernel: decode (BLOCK_M <= 32), 33-127 rows, prefill.
    (lambda: rel_ops._rel_attention_tuner(16), {"BLOCK_N": 128}, 4, 2),
    (lambda: rel_ops._rel_attention_tuner(32), {"BLOCK_N": 128}, 4, 2),
    (lambda: rel_ops._rel_attention_tuner(64), {"BLOCK_N": 64}, 4, 2),
    (lambda: rel_ops._rel_attention_tuner(128), {"BLOCK_N": 64}, 8, 2),
    # _k_group_kernel per token bucket.
    (lambda: linear_ops._k_group_kernel_tuned(16), {"BM": 16, "BN": 32}, 4, 2),
    (lambda: linear_ops._k_group_kernel_tuned(128), {"BM": 16, "BN": 64}, 4, 2),
    (lambda: linear_ops._k_group_kernel_tuned(256), {"BM": 64, "BN": 64}, 4, 2),
    (lambda: linear_ops._k_group_kernel_tuned(1 << 62), {"BM": 64, "BN": 128}, 8, 2),
    # _sconv_kernel, 9-256 tokens / prefill.
    (lambda: sconv_ops._sconv_kernel_tuned(False), {"BT": 2, "BLOCK_C": 512}, 4, 2),
    (lambda: sconv_ops._sconv_kernel_tuned(True), {"BT": 4, "BLOCK_C": 256}, 4, 2),
    # _select_kernel, up to 256 tokens / more.
    (lambda: moe_ops._select_kernel_tuned(True), {}, 4, 2),
    (lambda: moe_ops._select_kernel_tuned(False), {}, 1, 2),
    # _rel_decode_combine_kernel of small batches.
    (
        lambda: attention_ops._rel_decode_combine_kernel_tuned(),
        {"BLOCK_D": 32},
        4,
        2,
    ),
    # _nvfp4_moe_kernel: split-dots and default config tiles.
    (lambda: _nvfp4_tuner(True)[0], {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, 8, 2),
    (lambda: _nvfp4_tuner(False)[0], {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, 8, 3),
]


@pytest.mark.parametrize("case", range(len(FIXED_FIRST)))
def test_first_config_is_the_fixed_config(case):
    """`configs[0]` of every tuned launch is its fixed launch config."""
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


@triton.jit
def _noop_kernel(X, N: tl.constexpr):
    pass


def test_tuned_launches_follow_the_disk_cache_knob(monkeypatch):
    """The picks persist only where Triton's autotuning cache knob is on
    (`TRITON_CACHE_AUTOTUNING`, which vLLM's `env_override.py` defaults to
    1), so `=0` re-tunes: no call in the package passes `cache_results` (nor
    `launch_pdl`, which Intel's Triton rejects), and every tuner follows the
    knob, which Triton reads when the tuner is built."""
    root = pathlib.Path(xpu_package.__file__).parent
    forced = [
        f"{path.relative_to(root)}:{node.lineno}"
        for path in sorted(root.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and any(k.arg in ("cache_results", "launch_pdl") for k in node.keywords)
    ]
    assert not forced, forced
    knob = triton.knobs.autotuning.cache
    for tuner in _tuners():
        assert tuner.cache_results == knob, tuner.base_fn.__name__
    monkeypatch.setenv("TRITON_CACHE_AUTOTUNING", "0")
    tuner = autotune([triton.Config({})], key=["N"])(_noop_kernel)
    assert not tuner.cache_results


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
    tuning measured a gain."""
    tuner = linear_ops._k_group_kernel_tuned(bucket)
    kept = linear_ops._prune_k_group_tiles(tuner.configs, {"N": n, "BUCKET": bucket})
    assert (len(kept) > 1) == tuned


@pytest.mark.parametrize(
    "tokens,channels,fused,fixed_only",
    [
        (8, 4608, False, True),
        (8, 2304, True, True),
        (200, 2304, False, False),
        (200, 2304, True, True),
        (200, 4608, True, False),
        (300, 4608, False, False),
        (300, 4608, True, True),
        (300, 2304, True, False),
    ],
)
def test_short_conv_tunes_only_where_it_gained(tokens, channels, fused, fixed_only):
    """The short conv runs its fixed tile alone where tuning gave no stable
    gain."""
    assert sconv_ops._sconv_fixed_only(tokens, channels, fused) == fixed_only


def test_batch_invariant_runs_fixed_pointer_launches(kernel_env):
    """`VLLM_BATCH_INVARIANT=1`: no autotuning and no tensor descriptors,
    even with `VLLM_TRITON_USE_TD=1` (the fixed pointer launches).
    Otherwise the kernels autotune and follow `VLLM_TRITON_USE_TD` (unset:
    on)."""
    kernel_env(VLLM_TRITON_USE_TD="1", VLLM_BATCH_INVARIANT="1")
    assert not use_kernel_autotune()
    assert not use_kernel_td()
    assert not use_kernel_td(True)
    kernel_env(VLLM_BATCH_INVARIANT="0")
    assert use_kernel_autotune()
    assert use_kernel_td()
    assert not use_kernel_td(False)
    kernel_env(VLLM_TRITON_USE_TD="")
    assert use_kernel_td()
    kernel_env(VLLM_TRITON_USE_TD="0")
    assert not use_kernel_td()


def _rel_prefill():
    test_attention.test_rel_attention_matches_reference(12, 2, 64, None, "prefill")


def _rel_decode():
    test_attention.test_rel_decode_attention_matches_reference(
        24, 8, None, [1] * 4, [1, 200, 1025, 3000], 128, 9
    )


def _rel_window_decode():
    test_attention.test_rel_decode_attention_matches_reference(
        24, 8, REL_EXTENT - 1, [1] * 4, [1, 200, 1025, 3000], 64, 64
    )


def _nvfp4_experts(tokens: int):
    test_moe.test_xpu_experts_match_reference("nvfp4", "mini", tokens)


# (tuner, the reference test that launches it on a fresh key).
TUNED_LAUNCHES: dict[str, tuple[Callable[[], Any], Callable[[], None]]] = {
    "rel_attention_prefill": (lambda: rel_ops._rel_attention_tuner(128), _rel_prefill),
    "rel_attention_decode": (lambda: rel_ops._rel_attention_tuner(16), _rel_decode),
    "rel_decode_combine": (
        attention_ops._rel_decode_combine_kernel_tuned,
        _rel_window_decode,
    ),
    "k_group": (
        # 37 tokens at N = 128: a regime that tunes (`_FIXED_ONLY_ABOVE_N`).
        lambda: linear_ops._k_group_kernel_tuned(128),
        lambda: test_ops.test_k_group_partials_match_ordered_groups(
            37, 128, 4608, torch.float32
        ),
    ),
    "short_conv": (
        lambda: sconv_ops._sconv_kernel_tuned(True),
        lambda: test_ops.test_short_conv_chunked_matches_full_sequence(
            False, LONG_STEPS, 0
        ),
    ),
    "select_experts": (
        lambda: moe_ops._select_kernel_tuned(True),
        lambda: test_moe.test_select_experts_matches_torch(4608, 384, 64, False),
    ),
    "nvfp4_split_dots": (lambda: _nvfp4_tuner(True)[0], lambda: _nvfp4_experts(1)),
    "nvfp4_default": (lambda: _nvfp4_tuner(False)[0], lambda: _nvfp4_experts(3072)),
}


@requires_tuning
@pytest.mark.parametrize("launch", list(TUNED_LAUNCHES))
def test_tuned_launch_times_configs_and_matches_reference(
    monkeypatch, tmp_path, launch
):
    """The tuned launch times more than one config of a fresh key (an empty
    Triton cache dir and in-memory cache) and its pick passes the launch's
    reference test."""
    get_tuner, run_reference = TUNED_LAUNCHES[launch]
    tuner = get_tuner()
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    tuner.cache.clear()
    before = getattr(tuner, "configs_timings", None)
    run_reference()
    timings = getattr(tuner, "configs_timings", None)
    assert timings is not None and timings is not before, "no tuning ran"
    assert len(timings) > 1, timings
