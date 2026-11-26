# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU copy of `fused_moe_nvfp4_emulation_kernel` and its launcher from
`vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py`
(upstream `b7186d137a`).

Adapted: W4A16 only (bf16 activations, no activation QDQ), plus a
split-dots path for small batches, a tensor-descriptor load of the packed
weight tile (`USE_TD`, RFC vllm-project/vllm#42545), the config's launch
options, and tiles autotuned per path with the fixed config first. The
shared kernel is unchanged.

The NVFP4 grouped expert GEMM of the Inkling 2 XPU experts (Small: every
MoE layer; Mini: layers 0-19) over the checkpoint layout. A batch takes the
split-dots path below `split_dots_bound` rows per expert (`use_split_dots`)
and the default path above; the two paths round differently.
`VLLM_BATCH_INVARIANT` pins the default path, one config and the pointer
loads.
"""

import functools
from collections.abc import Mapping
from typing import Any

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.fused_moe import write_zeros_to_output
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _e2m1_inline,
)
from vllm.triton_utils import tensor_descriptor_compatible, tl, triton
from vllm.triton_utils.allocation import set_triton_allocator

from .autotune import autotune
from .platform import use_kernel_autotune, use_kernel_td

# NVFP4 block size along K: one e4m3 scale per 16 weights.
GROUP_SIZE = 16

# Batches with fewer rows (token x slot) per expert than `split_dots_bound`
# take the split-dots path (`use_split_dots`). The bound depends on the
# shape and on the weight loads (kernel timings, one Intel Arc Pro B70,
# Inkling 2 expert shapes, uniform top-8 routing, rows per expert 8 / 16 /
# 32 / 64 / 128; default / split time, w13 and w2):
# - Mini (hidden 2304, 128 experts), pointer loads: 2.1x at 8 rows,
#   0.82-0.83 at 16, 0.53-0.70 at 32: split below 16 rows;
# - Mini, tensor-descriptor loads: 8.4-9.8x at 8, 1.9-2.3x at 16,
#   0.85-0.93 at 32: split below 32 rows;
# - Small (hidden 4608, 384 experts), either load path: 1.2x / 3.2-3.4x at
#   8, 2.0-2.3x / 11x at 16, 1.6-1.85x / 4.2-4.3x at 32, 0.75-0.86 /
#   0.35-0.36 at 64: split below 64 rows.
# One bound serves both GEMMs, and it uses the mean rows per expert, so a
# hot expert can get more rows than the bound.
SPLIT_DOTS_SMALL_HIDDEN_BOUNDS = {False: 16, True: 32}
SPLIT_DOTS_LARGE_HIDDEN_BOUND = 64
SPLIT_DOTS_LARGE_HIDDEN = 4608

# Under VLLM_BATCH_INVARIANT every batch takes the default path with this
# config, so that a row's result does not depend on its batch (the path and
# BLOCK_SIZE_K set the rounding): `get_batch_invariant_moe_config`'s default,
# which `try_get_optimal_moe_config` returns there on XPU. No launch options:
# Triton's default warps and stages.
BATCH_INVARIANT_CONFIG: Mapping[str, int] = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
}

# The config entries passed to the fixed launch as launch options.
_LAUNCH_OPTION_KEYS = ("num_warps", "num_stages", "grf_mode")


def _rows_per_expert(num_tokens: int, top_k: int, num_experts: int) -> int:
    return triton.cdiv(num_tokens * top_k, max(num_experts, 1))


def split_dots_bound(hidden: int, td: bool) -> int:
    """Rows per expert below which the split-dots path is faster at hidden
    size `hidden`, with (`td`) or without tensor-descriptor weight loads."""
    if hidden >= SPLIT_DOTS_LARGE_HIDDEN:
        return SPLIT_DOTS_LARGE_HIDDEN_BOUND
    return SPLIT_DOTS_SMALL_HIDDEN_BOUNDS[td]


def use_split_dots(
    num_tokens: int, top_k: int, num_experts: int, hidden: int, td: bool = False
) -> bool:
    """Whether a batch of `num_tokens` tokens routed to `top_k` of
    `num_experts` experts at hidden size `hidden` takes the split-dots
    path: fewer rows per expert than `split_dots_bound(hidden, td)` (decode,
    and short prefills of models with many experts), and never under
    `VLLM_BATCH_INVARIANT`."""
    if envs.VLLM_BATCH_INVARIANT:
        return False
    return _rows_per_expert(num_tokens, top_k, num_experts) < split_dots_bound(
        hidden, td
    )


def split_dots_config(num_tokens: int, top_k: int, num_experts: int) -> dict:
    """Launch config of the split-dots path: a row block of about one
    expert's rows (16 to 64) and 128 x 128 weight tiles over 8 warps, the
    best single tile of a sweep at Inkling 2 expert shapes on an Intel Arc
    Pro B70 for 1-64 tokens."""
    rows_per_expert = _rows_per_expert(num_tokens, top_k, num_experts)
    return {
        "BLOCK_SIZE_M": min(64, max(16, triton.next_power_of_2(rows_per_expert))),
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 2,
    }


def default_config(num_tokens: int, num_local_experts: int) -> dict:
    """Launch config of the default path for a batch of `num_tokens` tokens
    over `num_local_experts` experts: the bf16 tiles of `get_default_config`
    (no tuned file under `fused_moe/configs/` matches the Inkling 2 expert
    shapes), with its warps and stages as launch options (its 8 warps above
    128 tokens are several times faster than Triton's default 4 on a
    B70). `BATCH_INVARIANT_CONFIG` under `VLLM_BATCH_INVARIANT`."""
    if envs.VLLM_BATCH_INVARIANT:
        return dict(BATCH_INVARIANT_CONFIG)
    m = num_tokens
    if m <= 32:
        block_m = 16
    elif m <= 96:
        block_m = 32
    elif m <= 512:
        block_m = 64
    else:
        block_m = 128
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": 64 if m <= 64 else 128,
        "BLOCK_SIZE_K": 128 if m <= 64 else 64,
        "GROUP_SIZE_M": 16 if m // max(num_local_experts, 1) > 128 else 1,
        "num_warps": 4 if m <= 128 else 8,
        "num_stages": 4 if m <= 32 else 3,
    }


def use_nvfp4_moe_td(weight: torch.Tensor, use_td: bool | None = None) -> bool:
    """Whether `nvfp4_moe_w4a16` loads the packed tiles of `weight` through
    a tensor descriptor: `use_kernel_td(use_td)` (the autotuned launches)
    and a per-expert `[N, K // 2]` descriptor of 16-byte aligned rows."""
    return use_kernel_td(use_td) and tensor_descriptor_compatible(weight)


def nvfp4_moe_config(
    num_tokens: int,
    top_k: int,
    num_global_experts: int,
    num_local_experts: int,
    hidden: int,
    td: bool,
) -> tuple[dict, bool]:
    """`(config, split_dots)` of a batch of `num_tokens` tokens routed to
    `top_k` of `num_global_experts` experts (`num_local_experts` here) at
    hidden size `hidden`, with (`td`) or without tensor-descriptor loads:
    the split-dots path and config below `split_dots_bound`, else the
    default path and `default_config`. One config serves both GEMMs;
    `config["BLOCK_SIZE_M"]` is the row block of `moe_align_block_size`."""
    split = use_split_dots(num_tokens, top_k, num_global_experts, hidden, td)
    if split:
        return split_dots_config(num_tokens, top_k, num_global_experts), True
    return default_config(num_tokens, num_local_experts), False


@triton.jit
def _block_scales_to_f32(raw):
    """fp32 values of FP8-E4M3 block scales stored as uint8 (exact for every
    finite code, subnormals included)."""
    return tl.cast(raw, tl.float8e4nv, bitcast=True).to(tl.float32)


@triton.jit
def _load_b_td(b_desc, row, col):
    """Packed weight tile [BLOCK_SIZE_N, BLOCK_SIZE_K // 2] at (row, col) of
    an expert's [N, K // 2] descriptor; out-of-range bytes load as 0."""
    return b_desc.load([row, col])


@triton.jit
def _split_dots_step(
    accumulator,
    a_ptrs,
    raw_bytes,
    s_ptrs,
    token_mask,
    k_left,
    stride_ak,
    block_k_diviable: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    group_size: tl.constexpr,
    compute_type: tl.constexpr,
):
    """One K step of the SPLIT_DOTS path: `accumulator` plus the even and
    the odd K indices of the A rows (`a_ptrs`: the even ones) times the
    weight tile `raw_bytes` [BLOCK_SIZE_N, BLOCK_SIZE_K // 2] (low nibble:
    even K index) with its block scales at `s_ptrs` [BLOCK_SIZE_N,
    BLOCK_SIZE_K // group_size]. `k_left`: the K indices from this step on."""
    GROUPS: tl.constexpr = BLOCK_SIZE_K // group_size
    if block_k_diviable:
        a_mask = token_mask[:, None]
        scale_raw = tl.load(s_ptrs)
    else:
        offs_k_packed = tl.arange(0, BLOCK_SIZE_K // 2)
        # K is even, so an even index below k_left has its odd one too.
        a_mask = token_mask[:, None] & (2 * offs_k_packed[None, :] < k_left)
        offs_g = tl.arange(0, GROUPS)
        scale_raw = tl.load(s_ptrs, mask=offs_g[None, :] * group_size < k_left, other=0)
    a_even = tl.load(a_ptrs, mask=a_mask, other=0.0)
    a_odd = tl.load(a_ptrs + stride_ak, mask=a_mask, other=0.0)
    # The scale of K index pair j (2j, 2j + 1) is scale[:, j // (group / 2)].
    scale = _block_scales_to_f32(scale_raw)
    scale = tl.reshape(
        tl.broadcast_to(scale[:, :, None], (BLOCK_SIZE_N, GROUPS, group_size // 2)),
        (BLOCK_SIZE_N, BLOCK_SIZE_K // 2),
    )
    w_even = (_e2m1_inline(raw_bytes & 0x0F) * scale).to(compute_type)
    w_odd = (_e2m1_inline((raw_bytes >> 4) & 0x0F) * scale).to(compute_type)
    accumulator = tl.dot(a_even, tl.trans(w_even), acc=accumulator)
    return tl.dot(a_odd, tl.trans(w_odd), acc=accumulator)


@triton.jit
def _nvfp4_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    w_global_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    # Strides: A [M, K]
    stride_am,
    stride_ak,
    # Strides: B [E, N, K // 2], passed as (expert, K-packed, N)
    stride_be,
    stride_bk,
    stride_bn,
    # Strides: C [M, topk, N]
    stride_cm,
    stride_cn,
    # Strides: B_scale [E, N, K // group_size], passed as (expert, K-scale, N)
    stride_bse,
    stride_bsk,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    group_size: tl.constexpr,
    SPLIT_DOTS: tl.constexpr = False,
    USE_TD: tl.constexpr = False,
):
    """Fused MoE kernel for NVFP4 weight-only dequantization + GEMM.

    Activations A are bf16 / fp16. Weights B are packed uint8 NVFP4
    [E, N, K // 2], two e2m1 values per byte along K (low nibble: even K
    index). B_scale holds per-block FP8-E4M3 scales [E, N, K // group_size]
    (as uint8); w_global_scale is a per-expert fp32 scalar. Per element:
        w = e2m1_decode(nibble) * (block_scale * global_scale)

    Default path: each packed byte is loaded once as a [BLOCK_SIZE_N,
    BLOCK_SIZE_K // 2] tile (N-major), both nibbles are decoded and scaled
    (block scale times global scale), and tl.interleave produces the
    [BLOCK_SIZE_N, BLOCK_SIZE_K] tile, transposed to [BLOCK_SIZE_K,
    BLOCK_SIZE_N] for tl.dot.

    SPLIT_DOTS (small batches) keeps the two nibbles apart instead: the even
    and odd K indices of the tile are two [BLOCK_SIZE_N, BLOCK_SIZE_K // 2]
    weight tiles, each multiplied (tl.dot) with the matching strided columns
    of A, so there is no interleave or [N, K] transpose. The block scales are
    loaded once per group and broadcast over it, and the global scale is
    applied to the fp32 accumulator: the e2m1 x e4m3 products (at most 6
    significant bits) are then exact in bf16 / fp16.

    USE_TD loads the packed weight tiles through a tensor descriptor over
    the expert's [N, K // 2] bytes (2D block loads); A, the block scales and
    C stay pointer accesses. The loaded values are the same.
    """
    BLOCK_SIZE_K_PACKED: tl.constexpr = BLOCK_SIZE_K // 2
    block_k_diviable: tl.constexpr = K % BLOCK_SIZE_K == 0

    # Map program ids to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Token / expert setup
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    # Pointer setup
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_packed = tl.arange(0, BLOCK_SIZE_K_PACKED)

    if SPLIT_DOTS:
        # A pointers: even K indices [BLOCK_SIZE_M, BLOCK_SIZE_K_PACKED]; the
        # odd ones are one stride_ak further.
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am
            + 2 * offs_k_packed[None, :] * stride_ak
        )
        # Block scales of the tile: one per group of K.
        GROUPS: tl.constexpr = BLOCK_SIZE_K // group_size
        offs_g = tl.arange(0, GROUPS)
        s_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[:, None] * stride_bsn
            + offs_g[None, :] * stride_bsk
        )
    else:
        # A pointers: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )

    # B pointers: [BLOCK_SIZE_N, BLOCK_SIZE_K_PACKED], N-major so that
    # tl.interleave (which operates on the last dim) produces a
    # [BLOCK_SIZE_N, BLOCK_SIZE_K] tile that is transposed for tl.dot.
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + offs_bn[:, None] * stride_bn
        + offs_k_packed[None, :] * stride_bk
    )

    # The two K elements of a packed byte always fall within the same scale
    # group (group_size = 16: 8 packed bytes per group), so packed indices
    # map to scale indices through group_size // 2.
    group_size_packed: tl.constexpr = group_size // 2

    if USE_TD:
        # The expert's packed weights [N, K // 2] as a tensor descriptor.
        b_desc = tl.make_tensor_descriptor(
            base=b_ptr + off_experts * stride_be,
            shape=(N, K // 2),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K_PACKED),
        )

    # Per-expert global scale (scalar).
    w_global_scale = tl.load(w_global_scale_ptr + off_experts).to(tl.float32)

    # K loop with fp32 accumulation
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if SPLIT_DOTS:
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Packed weight tile [BLOCK_SIZE_N, BLOCK_SIZE_K_PACKED].
            if USE_TD:
                raw_bytes = _load_b_td(
                    b_desc, pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K_PACKED
                )
            elif block_k_diviable:
                raw_bytes = tl.load(b_ptrs)
            else:
                kp_mask = offs_k_packed[None, :] < (K // 2) - k * BLOCK_SIZE_K_PACKED
                raw_bytes = tl.load(b_ptrs, mask=kp_mask, other=0)
            accumulator = _split_dots_step(
                accumulator,
                a_ptrs,
                raw_bytes,
                s_ptrs,
                token_mask,
                K - k * BLOCK_SIZE_K,
                stride_ak,
                block_k_diviable,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                group_size,
                compute_type,
            )
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K_PACKED * stride_bk
            s_ptrs += GROUPS * stride_bsk
        accumulator = accumulator * w_global_scale
    else:
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # A tile [BLOCK_SIZE_M, BLOCK_SIZE_K].
            if block_k_diviable:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0,
                )

            # Packed weight tile [BLOCK_SIZE_N, BLOCK_SIZE_K_PACKED].
            if not block_k_diviable:
                kp_mask = offs_k_packed[None, :] < (K // 2) - k * BLOCK_SIZE_K_PACKED
            if USE_TD:
                raw_bytes = _load_b_td(
                    b_desc, pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K_PACKED
                )
            elif block_k_diviable:
                raw_bytes = tl.load(b_ptrs)
            else:
                raw_bytes = tl.load(b_ptrs, mask=kp_mask, other=0)

            # Both nibbles of each byte (each [N, K_packed]).
            low_decoded = _e2m1_inline(raw_bytes & 0x0F)
            high_decoded = _e2m1_inline((raw_bytes >> 4) & 0x0F)

            # Per-block FP8 scales [BLOCK_SIZE_N, BLOCK_SIZE_K_PACKED], one
            # per group_size_packed packed elements.
            b_scale_ptrs = (
                b_scale_ptr
                + off_experts * stride_bse
                + offs_bn[:, None] * stride_bsn
                + (
                    (offs_k_packed[None, :] + BLOCK_SIZE_K_PACKED * k)
                    // group_size_packed
                )
                * stride_bsk
            )
            if block_k_diviable:
                b_scale_raw = tl.load(b_scale_ptrs)
            else:
                b_scale_raw = tl.load(b_scale_ptrs, mask=kp_mask, other=0.0)
            b_scale = _block_scales_to_f32(b_scale_raw) * w_global_scale

            # Interleave along the last dim: [N, K_packed] x2 -> [N, K], then
            # transpose to [K, N] for tl.dot.
            b = tl.trans(
                tl.interleave(low_decoded * b_scale, high_decoded * b_scale)
            ).to(compute_type)

            accumulator = tl.dot(a, b, acc=accumulator)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K_PACKED * stride_bk

    # Router weight multiplication (in fp32)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# (BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages) autotuned after the
# fixed tile of the launch (its config), per path.
_SPLIT_DOTS_TUNED_TILES = (
    (128, 128, 8, 2),
    (64, 128, 4, 2),
    (64, 128, 8, 2),
    (128, 64, 8, 2),
    (128, 128, 8, 3),
    (128, 128, 16, 2),
)
_DEFAULT_TUNED_TILES = (
    (128, 64, 8, 3),
    (128, 64, 8, 2),
    (128, 128, 8, 2),
    (64, 128, 8, 3),
    (64, 64, 4, 3),
    (128, 64, 4, 3),
)


def _prune_tiles(configs, named_args, **kwargs):
    """Column tiles up to the output width, and K tiles of at least 32 (16
    packed bytes) for the split-dots dot width and the descriptor block."""
    args = {**named_args, **kwargs}
    width = max(16, triton.next_power_of_2(args["N"]))
    min_k = 32 if args["SPLIT_DOTS"] or args["USE_TD"] else 16
    kept = [
        c
        for c in configs
        if c.kwargs["BLOCK_SIZE_N"] <= width and c.kwargs["BLOCK_SIZE_K"] >= min_k
    ]
    return kept or configs[:1]


@functools.cache
def _nvfp4_moe_kernel_tuned(split_dots: bool, fixed: tuple[int, int, int, int]):
    """`_nvfp4_moe_kernel` with the weight tile and warps autotuned per
    shape, row block and path: the launch's fixed tile `fixed` (`BLOCK_SIZE_N`,
    `BLOCK_SIZE_K`, warps, stages) first. The row block is not tuned: the
    alignment of the rows (`moe_align_block_size`) follows it."""
    tuned = _SPLIT_DOTS_TUNED_TILES if split_dots else _DEFAULT_TUNED_TILES
    tiles = [fixed, *(t for t in tuned if t != fixed)]
    return autotune(
        configs=[
            triton.Config(
                {"BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk}, num_warps=w, num_stages=s
            )
            for bn, bk, w, s in tiles
        ],
        key=[
            "N",
            "K",
            "BLOCK_SIZE_M",
            "GROUP_SIZE_M",
            "MUL_ROUTED_WEIGHT",
            "top_k",
            "SPLIT_DOTS",
            "USE_TD",
        ],
        prune_configs_by={"early_config_prune": _prune_tiles},
    )(_nvfp4_moe_kernel)


def nvfp4_moe_w4a16(
    a: torch.Tensor,
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    out: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    config: Mapping[str, Any],
    split_dots: bool = False,
    use_td: bool | None = None,
    topk_weights: torch.Tensor | None = None,
):
    """`out[m, j] = a[m] @ W[e].T` (times `topk_weights[m * top_k + j]` if
    given) for each flat row `r = m * top_k + j` that `sorted_token_ids`
    lists in a block of expert `e` of `expert_ids` (-1: the rows are
    zeroed); unlisted rows of `out` are left unset.

    Tensor contract:
        a: `[M, K]` bf16 (or fp16) activations, unquantized (W4A16).
        weight: `[E, N, K // 2]` uint8, two e2m1 values per byte along K
            (low nibble: even K index), the checkpoint layout.
        block_scale: `[E, N, K // 16]` e4m3 block scales, one per 16
            weights along K, as `torch.float8_e4m3fn` or its uint8 view.
        global_scale: `[E]` fp32 per-expert scale; a weight dequantizes to
            `e2m1 * block_scale * global_scale`.
        out: `[M, top_k, N]` contiguous, `a.dtype`.
        sorted_token_ids, expert_ids, num_tokens_post_padded: the
            `moe_align_block_size` output for `config["BLOCK_SIZE_M"]`.
        topk_weights: `[M * top_k]` fp32 routing weights, applied to the
            fp32 accumulator, or None.

    `config` holds `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K` (with
    `split_dots` a multiple of 16 and at least 32), `GROUP_SIZE_M` and
    optionally the launch options `num_warps`, `num_stages`, `grf_mode`
    (see `nvfp4_moe_config`). With `use_kernel_autotune()` the weight tile,
    warps and stages are autotuned per shape and path, starting from the
    config's (the row block is the config's), and the packed weight tiles
    load through a tensor descriptor where `use_nvfp4_moe_td(weight,
    use_td)`. Otherwise the config launches as given, with pointer loads.
    `split_dots` selects the split-dots path; its rounding differs from the
    default path (exact dequantized weights, global scale on the fp32
    accumulator). Returns the launched kernel.
    """
    assert a.dim() == 2 and a.dtype in (torch.bfloat16, torch.float16)
    assert out.dtype == a.dtype and out.is_contiguous()
    m, k = a.shape
    num_experts, n = weight.shape[:2]
    assert weight.dtype == torch.uint8 and weight.shape[2] * 2 == k
    assert out.shape == (m, top_k, n)
    if block_scale.dtype == torch.float8_e4m3fn:
        block_scale = block_scale.view(torch.uint8)
    assert block_scale.dtype == torch.uint8
    assert block_scale.shape == (num_experts, n, k // GROUP_SIZE)
    assert global_scale.dtype == torch.float32 and global_scale.shape == (num_experts,)
    if split_dots:
        # Whole scale groups per K tile, and a tl.dot inner dimension
        # (BLOCK_SIZE_K // 2) of at least 16.
        assert (
            config["BLOCK_SIZE_K"] % GROUP_SIZE == 0 and config["BLOCK_SIZE_K"] >= 32
        ), config
    block_m = config["BLOCK_SIZE_M"]
    em = sorted_token_ids.size(0)
    if m < block_m:
        em = min(em, m * top_k * block_m)

    def grid(meta):
        return (
            triton.cdiv(em, meta["BLOCK_SIZE_M"])
            * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        )

    args = (
        a,
        weight,
        out,
        block_scale,
        global_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        k,
        em,
        m * top_k,
        a.stride(0),
        a.stride(1),
        # weight is [E, N, K // 2]: swap N and K strides so the kernel
        # indexes [K, N].
        weight.stride(0),
        weight.stride(2),
        weight.stride(1),
        out.stride(1),
        out.stride(2),
        # block_scale is [E, N, K // 16]: swap N and K strides likewise.
        block_scale.stride(0),
        block_scale.stride(2),
        block_scale.stride(1),
    )
    meta = dict(
        BLOCK_SIZE_M=block_m,
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=topk_weights is not None,
        top_k=top_k,
        compute_type=tl.bfloat16 if a.dtype == torch.bfloat16 else tl.float16,
        group_size=GROUP_SIZE,
        SPLIT_DOTS=split_dots,
    )
    if not use_kernel_autotune():
        launch_options = {
            key: config[key] for key in _LAUNCH_OPTION_KEYS if key in config
        }
        return _nvfp4_moe_kernel[grid](
            *args,
            **meta,
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            USE_TD=False,
            **launch_options,
        )
    td = use_nvfp4_moe_td(weight, use_td)
    if td:
        # On every launch: Triton's allocator is a per-thread ContextVar.
        set_triton_allocator(a.device)
    fixed = (
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config.get("num_warps", 4),
        config.get("num_stages", 2),
    )
    extra = {"grf_mode": config["grf_mode"]} if "grf_mode" in config else {}
    return _nvfp4_moe_kernel_tuned(split_dots, fixed)[grid](
        *args, **meta, USE_TD=td, **extra
    )


def invoke_fused_moe_nvfp4_emulation_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    act_global_scale: torch.Tensor | None,
    w_global_scale: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Mapping[str, Any],
    compute_type: tl.dtype,
    launch_options: Mapping[str, Any] | None = None,
    split_dots: bool = False,
):
    """`nvfp4_moe_w4a16` behind the argument list of the shared
    `invoke_fused_moe_nvfp4_emulation_kernel`: W4A16 only
    (`act_global_scale` must be None), `launch_options` merged into
    `config`, `compute_type` the dtype of `A`."""
    assert act_global_scale is None, "W4A16: the activations are not quantized"
    assert compute_type == (tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16)
    return nvfp4_moe_w4a16(
        A,
        B,
        B_scale,
        w_global_scale,
        C,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        {**config, **(launch_options or {})},
        split_dots=split_dots,
        topk_weights=topk_weights if mul_routed_weight else None,
    )
