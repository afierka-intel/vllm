# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 linears: small batches on FlashInfer's CuTe DSL low-M kernels, and the
reference `linear` with fp32 accumulation in ordered 1024-wide K groups.

On XPU (`use_kernel_autotune`) the K-group kernel's tile is autotuned per
token bucket and weight shape, with the fixed tile of the bucket first,
and with `use_kernel_td()` its x / W tiles load and its output tile stores
through tensor descriptors (2D block IO). CUDA keeps the fixed tiles and
pointer accesses, whatever `VLLM_TRITON_USE_TD` says."""

import functools

import torch

from vllm.models.inkling2.nvidia.ops.ll_gemm_sm120 import (
    sm12x_bf16_ll_gemm,
    use_sm12x_bf16_ll_gemm,
)
from vllm.platforms import current_platform
from vllm.triton_utils import (
    tensor_descriptor_compatible,
    tl,
    triton,
)
from vllm.triton_utils.allocation import set_triton_allocator
from vllm.utils.flashinfer import is_flashinfer_autotuning

from .autotune import autotune
from .intrinsics import keep_separate, prefetch_l2
from .pdl import gdc_launch_dependents, gdc_wait
from .platform import (
    pdl_launch_kwargs,
    use_kernel_autotune,
    use_kernel_td,
    use_pdl,
    use_sm100_kernels,
)

# Up to this many rows, FlashInfer's CuTe DSL low-M kernels run near HBM
# bandwidth (cuBLAS is up to 2x slower); larger batches use cuBLAS.
LOW_M_MAX_TOKENS = 32


def linear(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """bf16 `x @ weight.T` with fp32 accumulation (into `out` if given,
    batches over LOW_M_MAX_TOKENS only)."""
    if out is not None:
        assert x.shape[0] > LOW_M_MAX_TOKENS
        return torch.mm(x, weight.t(), out=out)
    if use_sm12x_bf16_ll_gemm(x, weight):
        return sm12x_bf16_ll_gemm(x, weight)
    # Autotuning would profile the CuTe DSL kernels past their 32-row limit.
    # The FlashInfer query comes last: FlashInfer is a CUDA-only dependency.
    if (
        not use_sm100_kernels()
        or x.shape[0] > LOW_M_MAX_TOKENS
        or is_flashinfer_autotuning()
    ):
        return torch.nn.functional.linear(x, weight)
    from flashinfer import mm_bf16

    return mm_bf16(x, weight.t(), pdl=True, backend="cute-dsl")


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
    # Row tiles of at most 64 keep tl.dot on the same mma.sync instruction
    # sequence as the reference's 16x32x32 tiling. With SPLIT, each program computes one
    # K group's partial; otherwise it sums the groups in order.
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
        if g > 0:
            # Keep a short tail's dot separate from the preceding 1024-wide
            # groups; folding total into its MMA changes the rounding order
            # (`keep_separate` is an identity off CUDA, see `intrinsics`).
            partial = keep_separate(partial)
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
def _prefetch_k_group_tile(
    W, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, pid_n, pid_k
):
    # Pulls one K group's weight tile into L2, one 128-byte line per element.
    cols = pid_n * BN + tl.arange(0, BN)
    k = pid_k * 1024 + tl.arange(0, 16) * 64
    mask = (cols[:, None] < N) & (k[None, :] < K)
    prefetch_l2(W + cols[:, None] * K + k[None, :], mask)


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
    if SPLIT and tl.program_id(1) == 0:
        _prefetch_k_group_tile(W, N, K, BN, tl.program_id(0), tl.program_id(2))
    gdc_wait()
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
    gdc_wait()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    total = tl.load(Partials + offs, mask)
    for group in range(1, GROUPS):
        total = total + tl.load(Partials + group * MN + offs, mask)
    tl.store(Y + offs, total, mask)


@triton.jit
def _k_group_tma_kernel(
    XDesc,
    WDesc,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    LAUNCH_DEPENDENTS: tl.constexpr,
):
    # One output tile over all of K: each group's partial accumulates on the
    # tensor cores and is added to the running sum in order.
    TILES: tl.constexpr = (K + BK - 1) // BK
    GROUP_TILES: tl.constexpr = 1024 // BK
    rows = tl.program_id(1) * BM
    cols = tl.program_id(0) * BN
    gdc_wait()
    if LAUNCH_DEPENDENTS:
        gdc_launch_dependents()
    total = tl.zeros((BM, BN), tl.float32)
    partial = tl.zeros((BM, BN), tl.float32)
    for tile in tl.range(TILES, warp_specialize=True):
        x = XDesc.load([rows, tile * BK])
        w = WDesc.load([cols, tile * BK])
        partial = tl.dot(x, w.T, partial)
        if tile % GROUP_TILES == GROUP_TILES - 1 or tile == TILES - 1:
            total = total + partial
            partial = tl.zeros((BM, BN), tl.float32)
    r = rows + tl.arange(0, BM)[:, None]
    c = cols + tl.arange(0, BN)[None, :]
    tl.store(Y + r * N + c, total.to(Y.dtype.element_ty), (r < M) & (c < N))


# From this many rows, 128-row TMA tiles over all of K fill the GPU.
TMA_MIN_TOKENS = 2048


def k_group_partials(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
    launch_dependents: bool = False,
) -> torch.Tensor:
    """fp32 partials [G, T, N] of `x @ weight.T` whose in-order sum over G is
    the reference `linear`. G is 1 (already summed, in `out_dtype`) when
    there are enough rows to fill the GPU without splitting K. With
    `launch_dependents`, the next kernel may start (and must wait) while
    the TMA kernel runs."""
    assert x.is_contiguous() and weight.is_contiguous()
    tokens, hidden = x.shape
    n = weight.shape[0]
    sm120 = current_platform.is_device_capability_family(120)
    # SM12x, SM86 and SM89 have ~100 KB of shared memory per block; SM80's
    # 164 KB fits the default stage counts. SM8x has no TMA.
    small_smem = sm120 or not (use_pdl() or current_platform.is_device_capability(80))
    if tokens >= TMA_MIN_TOKENS and use_pdl():
        from triton.tools.tensor_descriptor import TensorDescriptor

        out = torch.empty((1, tokens, n), dtype=out_dtype, device=x.device)
        # SM12x allows ~99 KiB of shared memory per block: two 48 KiB stages fit.
        block_n = 64 if tokens < 4096 or sm120 else 128
        _k_group_tma_kernel[(triton.cdiv(n, block_n), triton.cdiv(tokens, 128))](
            TensorDescriptor.from_tensor(x, [128, 128]),
            TensorDescriptor.from_tensor(weight, [block_n, 128]),
            out,
            tokens,
            n,
            hidden,
            128,
            block_n,
            128,
            launch_dependents,
            num_warps=4 if block_n == 64 else 8,
            num_stages=2 if sm120 else 4 if block_n == 64 else 3,
            enable_fp_fusion=False,
            **pdl_launch_kwargs(),
        )
        return out
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
    block_m = 16 if tokens <= 128 else 64
    block_n = 32 if tokens <= 16 else 64 if tokens <= 256 else 128
    num_warps = 4 if block_n <= 64 else 8
    num_stages = 4 if not split else 3 if block_n <= 64 else 2
    if small_smem:
        num_stages = min(num_stages, 2)
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
        **pdl_launch_kwargs(),
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

# (BM, BN, num_warps, num_stages) autotuned on XPU: the fixed tiles as XPU
# ran them (2 stages, `small_smem`), then the others. Every tile gives the
# same bits (each output's K-group sums run in the same order; see
# test_k_group_tiles_are_bitwise_identical).
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
    """The fixed (BM, BN, num_warps, num_stages) of `tokens` tokens on XPU
    (its `small_smem` caps the stages at 2), as `k_group_partials` launches
    without autotuning."""
    block_m = 16 if tokens <= 128 else 64
    block_n = 32 if tokens <= 16 else 64 if tokens <= 256 else 128
    return block_m, block_n, 4 if block_n <= 64 else 8, 2


def _token_bucket(tokens: int) -> int:
    return next(b for b, _ in _TOKEN_BUCKETS if tokens <= b)


def _prune_k_group_tiles(configs, named_args, **kwargs):
    """Column tiles up to the output width."""
    n = {**named_args, **kwargs}["N"]
    width = max(32, triton.next_power_of_2(n))
    return [c for c in configs if c.kwargs["BN"] <= width] or configs[:1]


@functools.cache
def _k_group_kernel_tuned(bucket: int):
    """`_k_group_kernel` with the tile autotuned per weight shape, split and
    path, for the token counts of `bucket`: its fixed tile first."""
    rows = dict(_TOKEN_BUCKETS)[bucket]
    fixed = _fixed_tile(bucket)
    tiles = [fixed, *(t for t in _TUNED_TILES if t[0] in rows and t != fixed)]
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
        partials, out, out.numel(), partials.shape[0], 1024, **pdl_launch_kwargs()
    )
    return out
