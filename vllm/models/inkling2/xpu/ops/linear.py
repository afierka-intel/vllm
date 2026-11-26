# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU copy of `vllm/models/inkling2/nvidia/ops/linear.py` (upstream `b7186d137a`).

Adapted: `linear` is `torch.nn.functional.linear` (no FlashInfer / SM12x
low-M kernels); the K-group kernel runs without PDL, the L2 prefetch, the
MMA rounding barrier and the TMA kernel, autotunes its tile per token bucket
(fixed tile first) and loads / stores through tensor descriptors with
`use_kernel_td()` (RFC vllm-project/vllm#42545).

bf16 linears: `linear`, and the reference `linear_k_groups` with fp32
accumulation in ordered 1024-wide K groups.
"""

import functools

import torch

from vllm.triton_utils import tensor_descriptor_compatible, tl, triton
from vllm.triton_utils.allocation import set_triton_allocator

from .autotune import autotune
from .platform import use_kernel_autotune, use_kernel_td


def linear(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """bf16 `x @ weight.T` with fp32 accumulation (into `out` if given)."""
    if out is not None:
        return torch.mm(x, weight.t(), out=out)
    return torch.nn.functional.linear(x, weight)


@triton.jit
def _load_k_tile_td(desc, row, k):
    """`[rows, 128]` tile at `(row, k)` of a row-major `[rows, K]` operand;
    out-of-range rows and columns read as zero."""
    return desc.load([row, k])


@triton.jit
def _store_y_td(
    Y, total, M, N: tl.constexpr, row, col, BM: tl.constexpr, BN: tl.constexpr
):
    """Store the `[BM, BN]` tile at `(row, col)` of the `[M, N]` output `Y`;
    out-of-range rows and columns are dropped."""
    desc = tl.make_tensor_descriptor(
        Y, shape=[M, N], strides=[N, 1], block_shape=[BM, BN]
    )
    desc.store([row, col], total.to(Y.dtype.element_ty))


@triton.jit
def _k_group_tile(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
    pid_n,
    pid_m,
    pid_k,
    USE_TD: tl.constexpr = False,
):
    # With SPLIT, each program computes one K group's partial; otherwise it
    # sums the groups in order.
    cols = pid_n * BN + tl.arange(0, BN)
    rows = pid_m * BM + tl.arange(0, BM)
    ks = tl.arange(0, 128)
    first = pid_k if SPLIT else 0
    NUM_GROUPS: tl.constexpr = 1 if SPLIT else (K + 1023) // 1024
    if USE_TD:
        x_desc = tl.make_tensor_descriptor(
            X, shape=[M, K], strides=[K, 1], block_shape=[BM, 128]
        )
        w_desc = tl.make_tensor_descriptor(
            W, shape=[N, K], strides=[K, 1], block_shape=[BN, 128]
        )
    total = tl.zeros((BM, BN), tl.float32)
    for g in tl.static_range(NUM_GROUPS):
        group = first + g
        partial = tl.zeros((BM, BN), tl.float32)
        for tile in range(tl.minimum(8, tl.cdiv(K - group * 1024, 128))):
            if USE_TD:
                k0 = group * 1024 + tile * 128
                x = _load_k_tile_td(x_desc, pid_m * BM, k0)
                w = _load_k_tile_td(w_desc, pid_n * BN, k0).T
            else:
                k = group * 1024 + tile * 128 + ks
                x = tl.load(
                    X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K),
                    other=0,
                )
                w = tl.load(
                    W + cols[None, :] * K + k[:, None],
                    (cols[None, :] < N) & (k[:, None] < K),
                    other=0,
                )
            partial = tl.dot(x, w, partial)
        total = partial if g == 0 else total + partial
    if USE_TD:
        _store_y_td(Y + first * M * N, total, M, N, pid_m * BM, pid_n * BN, BM, BN)
    else:
        tl.store(
            Y + (first * M + rows[:, None]) * N + cols[None, :],
            total.to(Y.dtype.element_ty),
            (rows[:, None] < M) & (cols[None, :] < N),
        )


@triton.jit
def _k_group_kernel(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
    USE_TD: tl.constexpr = False,
    # Token bucket of the autotuned launch (unused in the body): an autotune
    # key, so that the buckets never share a tuned tile.
    BUCKET: tl.constexpr = 0,
):
    _k_group_tile(
        X,
        W,
        Y,
        M,
        N,
        K,
        BM,
        BN,
        SPLIT,
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
        USE_TD,
    )


@triton.jit
def _k_group_sum_kernel(Partials, Y, MN, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    total = tl.load(Partials + offs, mask)
    for group in range(1, GROUPS):
        total = total + tl.load(Partials + group * MN + offs, mask)
    tl.store(Y + offs, total, mask)


def k_group_partials(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """fp32 partials [G, T, N] of `x @ weight.T` whose in-order sum over G is
    the reference `linear`. G is 1 (already summed, in `out_dtype`) when
    there are enough rows to fill the GPU without splitting K."""
    assert x.is_contiguous() and weight.is_contiguous()
    tokens, hidden = x.shape
    n = weight.shape[0]
    groups = triton.cdiv(hidden, 1024)
    split = groups > 1 and tokens < 4096
    out = torch.empty(
        (groups if split else 1, tokens, n),
        dtype=torch.float32 if split else out_dtype,
        device=x.device,
    )
    if use_kernel_autotune():
        use_td = use_kernel_td() and tensor_descriptor_compatible(x, weight, out[0])
        if use_td:
            # On every launch: Triton's allocator is a per-thread ContextVar.
            set_triton_allocator(x.device)

        def tuned_grid(meta):
            return (
                triton.cdiv(n, meta["BN"]),
                triton.cdiv(tokens, meta["BM"]),
                groups if split else 1,
            )

        bucket = _token_bucket(tokens)
        _k_group_kernel_tuned(bucket)[tuned_grid](
            x,
            weight,
            out,
            tokens,
            n,
            hidden,
            SPLIT=split,
            USE_TD=use_td,
            BUCKET=bucket,
            enable_fp_fusion=False,
        )
        return out
    block_m, block_n, num_warps, num_stages = _fixed_tile(tokens)
    grid = (
        triton.cdiv(n, block_n),
        triton.cdiv(tokens, block_m),
        groups if split else 1,
    )
    _k_group_kernel[grid](
        x,
        weight,
        out,
        tokens,
        n,
        hidden,
        block_m,
        block_n,
        split,
        num_warps=num_warps,
        num_stages=num_stages,
        enable_fp_fusion=False,
    )
    return out


# Autotuning buckets of `k_group_partials`: (upper token count, row tiles),
# at the boundaries of the fixed tiles (`_fixed_tile`). Each bucket has its
# own tuner, and the bucket is part of the tuning key, so their tuned tiles
# stay apart, also in Triton's on-disk autotuning cache when pruning leaves
# two buckets the same configs.
_TOKEN_BUCKETS = (
    (16, (16,)),
    (128, (16, 32)),
    (256, (16, 32, 64)),
    (1 << 62, (32, 64)),
)

# (BM, BN, num_warps, num_stages) autotuned on XPU: the fixed tiles, then the
# others. Every tile gives the same bits (each output's K-group sums run in
# the same order).
_TUNED_TILES = (
    (16, 32, 4, 2),
    (16, 64, 4, 2),
    (64, 64, 4, 2),
    (64, 128, 8, 2),
    (16, 32, 4, 3),
    (16, 64, 4, 3),
    (32, 32, 4, 3),
    (32, 64, 4, 3),
    (64, 64, 4, 3),
)


def _fixed_tile(tokens: int) -> tuple[int, int, int, int]:
    """The fixed (BM, BN, num_warps, num_stages) of `tokens` tokens: the
    launch without autotuning, and the first config of each bucket's
    tuner."""
    block_m = 16 if tokens <= 128 else 64
    block_n = 32 if tokens <= 16 else 64 if tokens <= 256 else 128
    return block_m, block_n, 4 if block_n <= 64 else 8, 2


def _token_bucket(tokens: int) -> int:
    return next(b for b, _ in _TOKEN_BUCKETS if tokens <= b)


# Where tuning gave no gain (A/A-controlled kernel timings, one B70, Inkling 2
# router N = 128 / 384 and dynamic-tap N = 32 shapes), the fixed tile runs:
# up to 16 tokens (tuned / fixed 0.95-1.04, so that bucket lists the fixed
# tile only); 17-128 tokens with N > 128 (the Small router: the pick was
# 1.05-1.19x slower than the fixed tile); above 256 tokens with N > 32 (both
# routers 1.02). Tuning stays where it gained: 17-128 tokens at N <= 128
# (0.75-0.86) and 129-256 tokens (0.50-0.88); above 256 tokens the N = 32
# taps run the one tile that fits their width (0.61 of the fixed tile).
_FIXED_ONLY_ABOVE_N = {128: 128, 1 << 62: 32}


def _prune_k_group_tiles(configs, named_args, **kwargs):
    """Column tiles up to the output width, or the fixed tile alone where
    tuning gave no gain (`_FIXED_ONLY_ABOVE_N`)."""
    args = {**named_args, **kwargs}
    n = args["N"]
    limit = _FIXED_ONLY_ABOVE_N.get(args["BUCKET"])
    if limit is not None and n > limit:
        return configs[:1]
    width = max(32, triton.next_power_of_2(n))
    return [c for c in configs if c.kwargs["BN"] <= width] or configs[:1]


@functools.cache
def _k_group_kernel_tuned(bucket: int):
    """`_k_group_kernel` with the tile autotuned per weight shape, split and
    path, for the token counts of `bucket`: its fixed tile first."""
    rows = dict(_TOKEN_BUCKETS)[bucket]
    fixed = _fixed_tile(bucket)
    tiles = [fixed, *(t for t in _TUNED_TILES if t[0] in rows and t != fixed)]
    if bucket == 16:
        # Up to 16 tokens tuning gave no gain: the fixed tile only (with TD).
        tiles = [fixed]
    return autotune(
        configs=[
            triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=s)
            for bm, bn, w, s in tiles
        ],
        key=["N", "K", "SPLIT", "USE_TD", "BUCKET"],
        prune_configs_by={"early_config_prune": _prune_k_group_tiles},
    )(_k_group_kernel)


def linear_k_groups(
    x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """`x @ weight.T` for bf16 `x` [T, K] and `weight` [N, K].

    Each 1024-wide K group is accumulated in fp32 and the partials are summed in
    order, as in the reference `linear`.
    """
    partials = k_group_partials(x, weight, out_dtype)
    if partials.shape[0] == 1:
        return partials[0]
    out = torch.empty(partials.shape[1:], dtype=out_dtype, device=x.device)
    _k_group_sum_kernel[(triton.cdiv(out.numel(), 1024),)](
        partials, out, out.numel(), partials.shape[0], 1024
    )
    return out
