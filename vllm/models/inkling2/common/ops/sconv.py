# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 short convolution over a paged sliding-window conv-state cache.

The cache (`[num_blocks, H, N, D]`, see `InklingConvState` in
`vllm/models/inkling2/common/sconv.py`) stores each stream's conv input at every
absolute position. Per token, one launch writes the current input to its slot
and convolves the `W` taps ending at its position: taps inside this forward
are read from `x`, earlier taps from the paged cache, and taps before
position 0 are zero.

Static convolution (fp32 accumulate, oldest tap first)::

    y = (sum_k w[c, k] * x[t - (W - 1) + k] + x[t]) * (W + 1) ** -0.5

Dynamic convolution (newest tap first, per-token per-head tap weights)::

    y = (sum_l (static_w[l, c] + dyn[t, head(c), l]) * x[t - l] + x[t])
        * (2 * W + 1) ** -0.5

`y` is rounded to the input dtype. The K/V variant then L2-clips K and
RMS-normalizes V per head and writes both to the paged KV cache.

On XPU with `use_tensor_descriptor()`, the row tiles of the current inputs,
the residual / addend / output tiles and the q / rel-logit tiles of the
prologue go through tensor descriptors (2D block IO). The paged conv-state and
KV-cache accesses (one slot per token), the earlier taps and the per-token
scalars stay pointer accesses. CUDA keeps pointer accesses, whatever
`VLLM_TRITON_USE_TD` says. On XPU (`use_kernel_autotune`) the tiles and warps
are autotuned; CUDA keeps the fixed launch configs.
"""

import functools

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import (
    tensor_descriptor_compatible,
    tl,
    triton,
    use_tensor_descriptor,
)
from vllm.triton_utils.allocation import set_triton_allocator

from .intrinsics import rsqrt
from .norm import qr_clip_rel_tokens
from .pdl import gdc_launch_dependents, gdc_wait
from .platform import pdl_launch_kwargs, use_kernel_autotune


@triton.jit
def _load_rows_td(
    ptr, rows, cols, stride, row0, col0, BR: tl.constexpr, BC: tl.constexpr
):
    """`[BR, BC]` tile at `(row0, col0)` of the row-major `[rows, cols]`
    tensor at `ptr` (row stride `stride`); out-of-range elements read 0."""
    desc = tl.make_tensor_descriptor(
        ptr, shape=[rows, cols], strides=[stride, 1], block_shape=[BR, BC]
    )
    return desc.load([row0, col0])


@triton.jit
def _store_rows_td(
    ptr, value, rows, cols, stride, row0, col0, BR: tl.constexpr, BC: tl.constexpr
):
    """Store the `[BR, BC]` tile at `(row0, col0)` of `[rows, cols]`;
    out-of-range elements are dropped."""
    desc = tl.make_tensor_descriptor(
        ptr, shape=[rows, cols], strides=[stride, 1], block_shape=[BR, BC]
    )
    desc.store([row0, col0], value.to(ptr.dtype.element_ty))


@triton.jit
def _cached_taps(
    cache_ptr,
    cache_col,
    stride_c_blk,
    stride_c_n,
    N,
    c_mask,
    block_table_ptr,
    stride_bt,
    max_blocks,
    req,
    src,
    cached,
):
    safe_src = tl.maximum(src, 0)
    lblk = tl.minimum(safe_src // N, max_blocks - 1)
    blk = tl.load(block_table_ptr + req * stride_bt + lblk, mask=cached, other=0)
    return tl.load(
        cache_ptr
        + blk.to(tl.int64)[:, None] * stride_c_blk
        + (safe_src % N)[:, None] * stride_c_n
        + cache_col[None, :],
        mask=c_mask[None, :] & cached[:, None],
        other=0.0,
    ).to(tl.float32)


@triton.jit
def _cached_tap(
    cache_ptr,
    cache_col,
    stride_c_blk,
    stride_c_n,
    N,
    c_mask,
    block_table_ptr,
    stride_bt,
    max_blocks,
    req,
    pos,
    toff,
    qstart,
    valid,
    lag: tl.constexpr,
):
    src = pos - lag
    cached = valid & (src >= 0) & (toff - lag < qstart)
    return _cached_taps(
        cache_ptr,
        cache_col,
        stride_c_blk,
        stride_c_n,
        N,
        c_mask,
        block_table_ptr,
        stride_bt,
        max_blocks,
        req,
        src,
        cached,
    )


@triton.jit
def _tap_sum(
    x_ptr,
    stride_x,
    cache_ptr,
    cache_col,
    stride_c_blk,
    stride_c_n,
    N,
    weight_ptr,
    stride_w0,
    stride_w1,
    block_table_ptr,
    stride_bt,
    max_blocks,
    toff,
    coff,
    c_mask,
    tc_mask,
    pos,
    req,
    qstart,
    valid,
    cur,
    W: tl.constexpr,
    dyn_ptr,
    stride_dyn,
    HEAD_DIM: tl.constexpr,
    preloaded=None,
    READ_CACHE: tl.constexpr = False,
):
    """Weighted sum of the W taps; earlier steps' taps come from
    `preloaded` or, with READ_CACHE, from the cache (else they are zero)."""
    dtype = cache_ptr.dtype.element_ty
    acc = tl.zeros(cur.shape, dtype=tl.float32)
    for k in tl.static_range(W):
        lag = k if dyn_ptr is not None else W - 1 - k
        src = pos - lag
        row = toff - lag
        in_win = valid & (src >= 0)
        intra = in_win & (row >= qstart)
        cached = in_win & (row < qstart)
        xt = (
            tl.load(
                x_ptr + tl.maximum(row, 0)[:, None] * stride_x + coff[None, :],
                mask=c_mask[None, :] & intra[:, None],
                other=0.0,
            )
            .to(dtype)
            .to(tl.float32)
        )
        if preloaded is not None:
            xt += preloaded[k]
        elif READ_CACHE:
            xt += _cached_taps(
                cache_ptr,
                cache_col,
                stride_c_blk,
                stride_c_n,
                N,
                c_mask,
                block_table_ptr,
                stride_bt,
                max_blocks,
                req,
                src,
                cached,
            )
        value = xt
        if dyn_ptr is not None:
            w = tl.load(
                weight_ptr + lag * stride_w0 + coff * stride_w1, mask=c_mask, other=0
            ).to(tl.float32)
            dyn = tl.load(
                dyn_ptr
                + toff[:, None] * stride_dyn
                + (coff // HEAD_DIM)[None, :] * W
                + lag,
                mask=tc_mask,
                other=0,
            ).to(tl.float32)
            w = w[None, :] + dyn
        else:
            w = tl.load(
                weight_ptr + coff * stride_w0 + k * stride_w1, mask=c_mask, other=0
            )
            w = w.to(dtype).to(tl.float32)[None, :]
        acc = value * w if dyn_ptr is not None and k == 0 else tl.fma(value, w, acc)
    return acc


@triton.jit
def _conv(
    x_ptr,  # [T, C] current-token inputs (also the residual)
    stride_x,
    cache_ptr,  # [num_blocks, H, N, D] paged conv-state cache
    cache_col,  # [C] offset of each channel within a cache block row
    stride_c_blk,
    stride_c_n,
    N,
    weight_ptr,  # static: [C, W]; dynamic: [W, C]
    stride_w0,
    stride_w1,
    pos_ptr,  # [T] absolute positions
    seq_idx_ptr,  # [T] request of each token
    slot_ptr,  # [T] flat cache slot (< 0: padding)
    block_table_ptr,  # [num_reqs, max_blocks]
    stride_bt,
    max_blocks,
    qstart_ptr,  # [T] first row of each token's request
    scale,
    toff,
    coff,
    t_mask,
    c_mask,
    W: tl.constexpr,
    dyn_ptr=None,  # dynamic: [T, C // HEAD_DIM, W]
    stride_dyn=0,
    HEAD_DIM: tl.constexpr = 1,
    SKIP_IDLE_CACHE: tl.constexpr = False,
    CACHE_FIRST: tl.constexpr = False,
    tile_reads_cache=True,
    USE_TD: tl.constexpr = False,
    x_rows=0,
    x_cols=0,
    row0=0,
    col0=0,
):
    """Store the current inputs to the cache; return `y` in fp32.

    With `CACHE_FIRST` (static W=4 conv), the taps from earlier steps are
    loaded before waiting on the producer of `x` (PDL). With `USE_TD`, the
    current inputs (the tile at `(row0, col0)` of the `[x_rows, x_cols]`
    `x`) load through a tensor descriptor.
    """
    tc_mask = t_mask[:, None] & c_mask[None, :]
    slot = tl.load(slot_ptr + toff, mask=t_mask, other=-1)
    valid = slot >= 0
    pos = tl.load(pos_ptr + toff, mask=t_mask, other=0)
    req = tl.load(seq_idx_ptr + toff, mask=t_mask, other=0)
    qstart = tl.load(qstart_ptr + toff, mask=t_mask, other=0)
    if CACHE_FIRST:
        tl.static_assert(W == 4 and dyn_ptr is None)
        taps = (
            cache_ptr,
            cache_col,
            stride_c_blk,
            stride_c_n,
            N,
            c_mask,
            block_table_ptr,
            stride_bt,
            max_blocks,
            req,
            pos,
            toff,
            qstart,
            valid,
        )
        cached0 = _cached_tap(*taps, 3)
        cached1 = _cached_tap(*taps, 2)
        cached2 = _cached_tap(*taps, 1)
        cached3 = _cached_tap(*taps, 0)
        gdc_wait()

    dtype = cache_ptr.dtype.element_ty
    if USE_TD:
        cur = _load_rows_td(
            x_ptr,
            x_rows,
            x_cols,
            stride_x,
            row0,
            col0,
            toff.shape[0],
            coff.shape[0],
        )
    else:
        cur = tl.load(x_ptr + toff[:, None] * stride_x + coff[None, :], mask=tc_mask)
    cur = cur.to(dtype)
    safe_slot = tl.maximum(slot, 0)
    dst = (
        cache_ptr
        + (safe_slot // N)[:, None] * stride_c_blk
        + (safe_slot % N)[:, None] * stride_c_n
        + cache_col[None, :]
    )
    tl.store(dst, cur, mask=tc_mask & valid[:, None])

    tap_args = (
        x_ptr,
        stride_x,
        cache_ptr,
        cache_col,
        stride_c_blk,
        stride_c_n,
        N,
        weight_ptr,
        stride_w0,
        stride_w1,
        block_table_ptr,
        stride_bt,
        max_blocks,
        toff,
        coff,
        c_mask,
        tc_mask,
        pos,
        req,
        qstart,
        valid,
        cur,
    )
    # Only tiles with a token among the first W - 1 of a request read taps
    # from the cache; SKIP_IDLE_CACHE branches around the others' loads once.
    if CACHE_FIRST:
        preloaded = (cached0, cached1, cached2, cached3)
        acc = _tap_sum(*tap_args, W, dyn_ptr, stride_dyn, HEAD_DIM, preloaded)
    elif not SKIP_IDLE_CACHE or tile_reads_cache:
        acc = _tap_sum(*tap_args, W, dyn_ptr, stride_dyn, HEAD_DIM, READ_CACHE=True)
    else:
        acc = _tap_sum(*tap_args, W, dyn_ptr, stride_dyn, HEAD_DIM)
    return (acc + cur.to(tl.float32)) * scale


@triton.jit
def _sconv_kernel(
    x_ptr,
    cache_ptr,
    weight_ptr,
    out_ptr,  # [T, C]
    res_ptr,  # [T, C] optional residual base, added before addend
    addend_ptr,  # [T, C] optional
    pos_ptr,
    seq_idx_ptr,
    slot_ptr,
    block_table_ptr,
    qstart_ptr,
    T,
    stride_x,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    stride_w0,
    stride_w1,
    stride_bt,
    max_blocks,
    N,
    scale,
    W: tl.constexpr,
    OFF: tl.constexpr,
    WS: tl.constexpr,
    H: tl.constexpr,
    HAS_RES: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SKIP_IDLE_CACHE: tl.constexpr,
    USE_TD: tl.constexpr = False,
    # Token bucket of the autotuned launch (unused in the body): an autotune
    # key, so that the buckets never share a tuned tile.
    BUCKET: tl.constexpr = 0,
):
    first = tl.program_id(0) * BT
    tile_reads_cache = True
    if SKIP_IDLE_CACHE:
        gdc_wait()
        # Request starts are non-decreasing: no token of the tile is among the
        # first W - 1 of its request unless the last token's request starts
        # after first - (W - 1).
        last = tl.minimum(first + BT, T) - 1
        tile_reads_cache = tl.load(qstart_ptr + last) > first - (W - 1)
    toff = first + tl.arange(0, BT)
    coff = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    C = H * WS
    t_mask = toff < T
    c_mask = coff < C
    y = _conv(
        x_ptr,
        stride_x,
        cache_ptr,
        tl.minimum(coff // WS, H - 1) * stride_c_h + OFF + coff % WS,
        stride_c_blk,
        stride_c_n,
        N,
        weight_ptr,
        stride_w0,
        stride_w1,
        pos_ptr,
        seq_idx_ptr,
        slot_ptr,
        block_table_ptr,
        stride_bt,
        max_blocks,
        qstart_ptr,
        scale,
        toff,
        coff,
        t_mask,
        c_mask,
        W,
        SKIP_IDLE_CACHE=SKIP_IDLE_CACHE,
        CACHE_FIRST=not SKIP_IDLE_CACHE,
        tile_reads_cache=tile_reads_cache,
        USE_TD=USE_TD,
        x_rows=T,
        x_cols=C,
        row0=first,
        col0=tl.program_id(1) * BLOCK_C,
    ).to(out_ptr.dtype.element_ty)
    if USE_TD:
        col0 = tl.program_id(1) * BLOCK_C
        if HAS_ADDEND:
            acc = _load_rows_td(addend_ptr, T, C, C, first, col0, BT, BLOCK_C)
            acc = acc.to(tl.float32)
            if HAS_RES:
                res = _load_rows_td(res_ptr, T, C, C, first, col0, BT, BLOCK_C)
                acc = res.to(tl.float32) + acc
            y = (acc + y.to(tl.float32)).to(out_ptr.dtype.element_ty)
        _store_rows_td(out_ptr, y, T, C, C, first, col0, BT, BLOCK_C)
        return
    offs = toff[:, None] * C + coff[None, :]
    mask = t_mask[:, None] & c_mask[None, :]
    if HAS_ADDEND:
        # Residual update in the reference order: (res + addend) + y in fp32.
        acc = tl.load(addend_ptr + offs, mask, other=0).to(tl.float32)
        if HAS_RES:
            acc = tl.load(res_ptr + offs, mask, other=0).to(tl.float32) + acc
        y = (acc + y.to(tl.float32)).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, y, mask=mask)


def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    addend: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Insert `x` (rounded to the cache dtype) into the conv-state cache and
    return the static convolution with `(C, W)` taps `weight`.

    With `addend`, returns `(residual + addend) + y` accumulated in fp32
    (`residual` may alias `out`).
    """
    num_tokens, channels = x.shape
    if out is None:
        out = torch.empty((num_tokens, channels), dtype=cache.dtype, device=x.device)
    assert x.stride(1) == 1 and cache.stride(3) == 1 and out.is_contiguous()
    assert residual is None or addend is not None
    for t in (addend, residual):
        assert t is None or (t.is_contiguous() and t.shape == out.shape)
    num_heads = cache.shape[1]
    assert channels == num_heads * stream_width
    width = weight.shape[1]
    # Prefill tiles skip the cache reads they never need.
    prefill = num_tokens > 256
    args = (
        x,
        cache,
        weight,
        out,
        residual,
        addend,
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        query_start,
        num_tokens,
        x.stride(0),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        weight.stride(0),
        weight.stride(1),
        block_table.stride(0),
        block_table.shape[1],
        cache.shape[2],
        (width + 1) ** -0.5,
    )
    meta = dict(
        W=width,
        OFF=stream_offset,
        WS=stream_width,
        H=num_heads,
        HAS_RES=residual is not None,
        HAS_ADDEND=addend is not None,
        SKIP_IDLE_CACHE=prefill,
        enable_fp_fusion=False,
    )
    if use_kernel_autotune():
        use_td = (
            current_platform.is_xpu()
            and use_tensor_descriptor()
            and tensor_descriptor_compatible(x, out, addend, residual)
        )
        if use_td:
            # On every launch: Triton's allocator is a per-thread ContextVar.
            set_triton_allocator(x.device)

        def tuned_grid(meta):
            return (
                triton.cdiv(num_tokens, meta["BT"]),
                triton.cdiv(channels, meta["BLOCK_C"]),
            )

        bucket = _sconv_bucket(num_tokens)
        _sconv_kernel_tuned(bucket)[tuned_grid](
            *args, **meta, USE_TD=use_td, BUCKET=bucket
        )
        return out
    if num_tokens <= 8:
        block_t, block_c = 1, 1024
    else:
        block_t, block_c = (4, 256) if prefill else (2, 512)
    block_c = min(triton.next_power_of_2(channels), block_c)
    grid = (triton.cdiv(num_tokens, block_t), triton.cdiv(channels, block_c))
    _sconv_kernel[grid](
        *args,
        **meta,
        BT=block_t,
        BLOCK_C=block_c,
        num_warps=4,
        **pdl_launch_kwargs(),
    )
    return out


# Autotuning buckets of `short_conv`: (upper token count, token tiles) for
# decode, mixed batches and prefill. Each bucket has its own tuner, and the
# bucket is part of the tuning key, so their tuned tiles stay apart, also in
# Triton's on-disk autotuning cache. The first config of every bucket
# (VLLM_TRITON_FORCE_FIRST_CONFIG) is valid at any shape: channel tiles past
# the channels are masked.
_SCONV_BUCKETS = ((8, (1, 2)), (256, (1, 2, 4)), (1 << 62, (4, 8)))

# (BT, BLOCK_C, num_warps) autotuned on XPU; includes the fixed tiles.
_SCONV_TILES = (
    (1, 1024, 4),
    (2, 512, 4),
    (4, 256, 4),
    (2, 1024, 8),
    (4, 512, 8),
    (8, 256, 8),
)


def _sconv_bucket(tokens: int) -> int:
    return next(b for b, _ in _SCONV_BUCKETS if tokens <= b)


def _prune_sconv_tiles(configs, named_args, **kwargs):
    """Channel tiles up to the channels."""
    args = {**named_args, **kwargs}
    channels = triton.next_power_of_2(args["H"] * args["WS"])
    return [c for c in configs if c.kwargs["BLOCK_C"] <= channels] or configs[:1]


@functools.cache
def _sconv_kernel_tuned(bucket: int):
    """`_sconv_kernel` with the tile and warps autotuned per shape and path,
    for token counts up to `bucket`."""
    tiles = dict(_SCONV_BUCKETS)[bucket]
    return triton.autotune(
        configs=[
            triton.Config({"BT": bt, "BLOCK_C": bc}, num_warps=w)
            for bt, bc, w in _SCONV_TILES
            if bt in tiles
        ],
        key=[
            "W",
            "WS",
            "H",
            "HAS_RES",
            "HAS_ADDEND",
            "SKIP_IDLE_CACHE",
            "USE_TD",
            "BUCKET",
        ],
        prune_configs_by={"early_config_prune": _prune_sconv_tiles},
        # The residual may alias the output.
        restore_value=["out_ptr"],
    )(_sconv_kernel)


# Columns of shift_rel_proj beyond the extent, 256 on either side.
_SHIFT_PAD = 512


def shift_rel_proj(rel_proj: torch.Tensor, is_local: bool) -> torch.Tensor:
    """Return the `(8, d_rel + 1, extent + 512)` table `attention_prologue`
    reads the `(d_rel, extent)` relative projection from, so that every
    sheared output column block is an aligned slice of one of 8 shifts.

    `out[s, :d_rel, y]` is `rel_proj[:, d]` for distance `d = s + extent
    + 256 - y`, or zero out of range; row `d_rel` is the padding added to
    the logits: -inf for negative distances and, if `is_local`, those past
    the extent, else zero.
    """
    d_rel, extent = rel_proj.shape
    shift = torch.arange(8, device=rel_proj.device)[:, None]
    y = torch.arange(extent + _SHIFT_PAD, device=rel_proj.device)
    dist = shift + extent + 256 - y
    outside = (dist < 0) | (dist >= extent)
    proj = rel_proj[:, dist.clamp(0, extent - 1)].transpose(0, 1)
    pad = (dist >= extent) if is_local else torch.zeros_like(outside)
    pad = (dist < 0) | pad
    pad = torch.zeros_like(proj[:, :1]).masked_fill(pad[:, None], float("-inf"))
    return torch.cat([proj.masked_fill(outside[:, None], 0), pad], 1).contiguous()


@triton.jit
def _attn_prologue_kernel(
    k_ptr,  # [T, H * D] K projection
    v_ptr,  # [T, H * D] V projection
    wk_ptr,  # static: [H * D, W]; dynamic: [W, H * D]
    wv_ptr,
    dk_ptr,  # dynamic: [T, H, W] per-token K tap deltas
    dv_ptr,
    cache_ptr,
    key_cache_ptr,  # [num_blocks, block_size, H, D]
    value_cache_ptr,
    kv_slot_ptr,  # [T] KV cache slot (< 0: skip)
    pos_ptr,
    seq_idx_ptr,
    slot_ptr,
    block_table_ptr,
    qstart_ptr,
    T,
    stride_k,
    stride_v,
    stride_w0,
    stride_w1,
    stride_dk,
    stride_dv,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    stride_kv_blk,
    stride_kv_n,
    stride_kv_h,
    kv_block_size,
    stride_bt,
    max_blocks,
    N,
    scale,
    clip_threshold,
    clip_eps,
    norm_eps,
    q_ptr,  # [T, HQ, D] query projection (row-strided)
    r_ptr,  # [T, HQ, D_REL] relative-query projection (row-strided)
    q_scale_ptr,
    r_scale_ptr,
    tau_ptr,
    rel_proj_ptr,
    q_out_ptr,
    rel_out_ptr,
    cu_seqlens_q_ptr,
    seq_lens_ptr,
    stride_q,
    stride_r,
    rel_threshold,
    W: tl.constexpr,
    D: tl.constexpr,
    DYNAMIC: tl.constexpr,
    BT: tl.constexpr,
    HQ: tl.constexpr,
    D_REL: tl.constexpr,
    E: tl.constexpr,
    SHIFT_W: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_HQ: tl.constexpr,
    BLOCK_E: tl.constexpr,
    QR_TOKENS: tl.constexpr,
    FUSE_QR: tl.constexpr,
    USE_TD: tl.constexpr = False,
):
    gdc_wait()
    # The decode attention kernel prefetches cached keys meanwhile.
    gdc_launch_dependents()
    # Programs (tile, head, 0 | 1) run the K | V conv of one KV head; programs
    # (tile, head, 2) clip q of QR_TOKENS of the tile's tokens and programs
    # (tile, head, 3 + c) their r and column block c of the relative logits.
    if FUSE_QR and tl.program_id(2) >= 2:
        if tl.program_id(1) * QR_TOKENS < BT:
            qr_clip_rel_tokens(
                tl.program_id(0) * BT + tl.program_id(1) * QR_TOKENS,
                QR_TOKENS,
                tl.program_id(2) - 3,
                T,
                q_ptr,
                r_ptr,
                q_scale_ptr,
                r_scale_ptr,
                tau_ptr,
                rel_proj_ptr,
                q_out_ptr,
                rel_out_ptr,
                seq_idx_ptr,
                cu_seqlens_q_ptr,
                seq_lens_ptr,
                stride_q,
                stride_r,
                clip_threshold,
                rel_threshold,
                clip_eps,
                HQ,
                D,
                D_REL,
                E,
                SHIFT_W,
                HAS_TAU,
                BLOCK_HQ,
                BLOCK_E,
                USE_TD,
            )
    else:
        _kv_conv_tile(
            k_ptr,
            v_ptr,
            wk_ptr,
            wv_ptr,
            dk_ptr,
            dv_ptr,
            cache_ptr,
            key_cache_ptr,
            value_cache_ptr,
            kv_slot_ptr,
            pos_ptr,
            seq_idx_ptr,
            slot_ptr,
            block_table_ptr,
            qstart_ptr,
            T,
            stride_k,
            stride_v,
            stride_w0,
            stride_w1,
            stride_dk,
            stride_dv,
            stride_c_blk,
            stride_c_h,
            stride_c_n,
            stride_kv_blk,
            stride_kv_n,
            stride_kv_h,
            kv_block_size,
            stride_bt,
            max_blocks,
            N,
            scale,
            clip_threshold,
            clip_eps,
            norm_eps,
            W,
            D,
            DYNAMIC,
            BT,
            USE_TD,
        )


@triton.jit
def _qr_rel_kernel(
    q_ptr,
    r_ptr,
    q_scale_ptr,
    r_scale_ptr,
    tau_ptr,
    rel_proj_ptr,
    q_out_ptr,
    rel_out_ptr,
    seq_idx_ptr,
    cu_seqlens_q_ptr,
    seq_lens_ptr,
    T,
    stride_q,
    stride_r,
    clip_threshold,
    rel_threshold,
    clip_eps,
    HQ: tl.constexpr,
    D: tl.constexpr,
    D_REL: tl.constexpr,
    E: tl.constexpr,
    SHIFT_W: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_HQ: tl.constexpr,
    BLOCK_E: tl.constexpr,
    QR_TOKENS: tl.constexpr,
    USE_TD: tl.constexpr = False,
):
    # Runs alongside the K/V conv launch (`_attn_prologue_kernel` without
    # FUSE_QR), which started it after its own grid dependency wait; waiting
    # for that launch before exiting keeps the KV cache written for the
    # attention that follows.
    qr_clip_rel_tokens(
        tl.program_id(0) * QR_TOKENS,
        QR_TOKENS,
        tl.program_id(1) - 1,
        T,
        q_ptr,
        r_ptr,
        q_scale_ptr,
        r_scale_ptr,
        tau_ptr,
        rel_proj_ptr,
        q_out_ptr,
        rel_out_ptr,
        seq_idx_ptr,
        cu_seqlens_q_ptr,
        seq_lens_ptr,
        stride_q,
        stride_r,
        clip_threshold,
        rel_threshold,
        clip_eps,
        HQ,
        D,
        D_REL,
        E,
        SHIFT_W,
        HAS_TAU,
        BLOCK_HQ,
        BLOCK_E,
        USE_TD,
    )
    gdc_wait()


@triton.jit
def _kv_conv_tile(
    k_ptr,
    v_ptr,
    wk_ptr,
    wv_ptr,
    dk_ptr,
    dv_ptr,
    cache_ptr,
    key_cache_ptr,
    value_cache_ptr,
    kv_slot_ptr,
    pos_ptr,
    seq_idx_ptr,
    slot_ptr,
    block_table_ptr,
    qstart_ptr,
    T,
    stride_k,
    stride_v,
    stride_w0,
    stride_w1,
    stride_dk,
    stride_dv,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    stride_kv_blk,
    stride_kv_n,
    stride_kv_h,
    kv_block_size,
    stride_bt,
    max_blocks,
    N,
    scale,
    clip_threshold,
    clip_eps,
    norm_eps,
    W: tl.constexpr,
    D: tl.constexpr,
    DYNAMIC: tl.constexpr,
    BT: tl.constexpr,
    USE_TD: tl.constexpr = False,
):
    toff = tl.program_id(0) * BT + tl.arange(0, BT)
    head = tl.program_id(1)
    is_v = tl.program_id(2) == 1
    d_idx = tl.arange(0, D)
    coff = head * D + d_idx
    t_mask = toff < T
    if is_v:
        x_ptr = v_ptr
        w_ptr = wv_ptr
        dyn_ptr = dv_ptr
        stride_x = stride_v
        stride_dyn = stride_dv
    else:
        x_ptr = k_ptr
        w_ptr = wk_ptr
        dyn_ptr = dk_ptr
        stride_x = stride_k
        stride_dyn = stride_dk
    y = _conv(
        x_ptr,
        stride_x,
        cache_ptr,
        head * stride_c_h + tl.program_id(2) * D + d_idx,
        stride_c_blk,
        stride_c_n,
        N,
        w_ptr,
        stride_w0,
        stride_w1,
        pos_ptr,
        seq_idx_ptr,
        slot_ptr,
        block_table_ptr,
        stride_bt,
        max_blocks,
        qstart_ptr,
        scale,
        toff,
        coff,
        t_mask,
        d_idx < D,
        W,
        dyn_ptr if DYNAMIC else None,
        stride_dyn,
        D,
        USE_TD=USE_TD,
        x_rows=T,
        x_cols=tl.num_programs(1) * D,
        row0=tl.program_id(0) * BT,
        col0=head * D,
    )
    y = y.to(k_ptr.dtype.element_ty).to(tl.float32)
    sum_sq = tl.sum(y * y, axis=1)
    if is_v:
        y = y * rsqrt(tl.div_rn(sum_sq, D) + norm_eps)[:, None]
        cache = value_cache_ptr
    else:
        norm = tl.where(sum_sq > 0, sum_sq * rsqrt(sum_sq), 0.0)
        y = y * tl.minimum(tl.div_rn(clip_threshold, norm + clip_eps), 1.0)[:, None]
        cache = key_cache_ptr
    kv_slot = tl.load(kv_slot_ptr + toff, mask=t_mask, other=-1)
    safe = tl.maximum(kv_slot, 0)
    dst = (
        cache
        + (safe // kv_block_size)[:, None] * stride_kv_blk
        + (safe % kv_block_size)[:, None] * stride_kv_n
        + head * stride_kv_h
        + d_idx[None, :]
    )
    tl.store(dst, y.to(cache.dtype.element_ty), mask=(kv_slot >= 0)[:, None])


# From this many tokens, q/r run in their own launch (8 tokens per program)
# beside the K/V convs (8 per program): the fused launch's one-token q/r
# programs dominate prefill.
_PROLOGUE_SPLIT_MIN_TOKENS = 2048


def _prologue_block_t(num_tokens: int) -> int:
    return 8 if num_tokens >= _PROLOGUE_SPLIT_MIN_TOKENS else 4


def _prologue_qr_tokens(num_tokens: int) -> int:
    """Tokens per program of the separate q/r launch, or 0 to run q/r in the
    K/V conv launch."""
    return 8 if num_tokens >= _PROLOGUE_SPLIT_MIN_TOKENS else 0


def attention_prologue(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    r_scale: torch.Tensor,
    tau: torch.Tensor | None,
    rel_proj: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_dynamic: torch.Tensor | None,
    v_dynamic: torch.Tensor | None,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    q_out: torch.Tensor,
    rel_out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    clip_threshold: float,
    rel_threshold: float,
    norm_eps: float,
    clip_eps: float = 1e-6,
) -> None:
    """Attention input processing in one launch.

    Writes the L2-clipped `q * q_scale` to `q_out` and the relative logits
    of the clipped `r * r_scale` to `rel_out` (with `tau`, clipped values
    are rounded, then scaled by the fp32 per-token `tau`). `rel_proj` is
    the `shift_rel_proj` table; the logits come out sheared as tml_fa4 reads
    them for a varlen launch over all tokens (`(T + 128, HQ, extent +
    256)`) from the per-request `cu_seqlens_q` and `seq_lens`. Runs the
    K/V short convolutions, clips K, RMS-normalizes V and inserts both into
    the KV cache (tokens with `kv_slot_mapping < 0` skip the write).

    `q` `(T, HQ, D)`, `r` `(T, HQ, D_rel)` and `k`/`v`
    `(T, H * D)` may be row-strided; K and V occupy `[0, D)` and
    `[D, 2 * D)` of each conv-state head. Conv weights are static
    `(H * D, W)` taps, or with `(T, H * W)` per-token deltas
    `k/v_dynamic` the `(W, H * D)` static part.
    """
    num_tokens = kv_slot_mapping.shape[0]
    num_heads, head_dim = key_cache.shape[2:]
    num_q_heads, d_rel = r.shape[1:]
    extent = rel_proj.shape[2] - _SHIFT_PAD
    assert rel_proj.shape[:2] == (8, d_rel + 1)
    assert k.stride(1) == v.stride(1) == 1 and cache.stride(3) == 1
    assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
    assert k_weight.stride() == v_weight.stride()
    assert q.shape[1:] == (num_q_heads, head_dim)
    assert q_out.shape == q.shape
    assert rel_out.shape == (q.shape[0] + 128, num_q_heads, extent + 256)
    assert q.stride(2) == 1 and q.stride(1) == head_dim
    assert r.stride(2) == 1 and r.stride(1) == d_rel
    assert q_out.is_contiguous() and rel_out.is_contiguous()
    assert q_scale.is_contiguous() and r_scale.is_contiguous()
    assert rel_proj.is_contiguous() and d_rel >= 16
    block_e = min(extent, 256)
    for size in (head_dim, d_rel, block_e):
        assert size & (size - 1) == 0, "tile sizes must be powers of 2"
    assert extent % block_e == 0
    dynamic = k_dynamic is not None
    if dynamic:
        assert k_dynamic is not None and v_dynamic is not None
        width = k_weight.shape[0]
        assert k_dynamic.stride(1) == v_dynamic.stride(1) == 1
        dyn_strides = (k_dynamic.stride(0), v_dynamic.stride(0))
    else:
        width = k_weight.shape[1]
        dyn_strides = (0, 0)
    block_t = _prologue_block_t(num_tokens)
    qr_tokens = _prologue_qr_tokens(num_tokens)
    qr_blocks = 1 + (extent + 256) // block_e
    grid = (
        triton.cdiv(num_tokens, block_t),
        num_heads,
        2 + (0 if qr_tokens else qr_blocks),
    )
    tuned = use_kernel_autotune()
    use_td = (
        current_platform.is_xpu()
        and use_tensor_descriptor()
        and tensor_descriptor_compatible(
            k,
            v,
            q.flatten(1),
            q_out.flatten(1),
            q_scale,
            rel_out.flatten(1),
            rel_proj.flatten(0, 1),
        )
    )
    if use_td:
        # On every launch: Triton's allocator is a per-thread ContextVar.
        set_triton_allocator(q.device)
    kernel = _attn_prologue_kernel_tuned if tuned else _attn_prologue_kernel
    kernel[grid](
        k,
        v,
        k_weight,
        v_weight,
        # Unused placeholders for static convs keep both branches typed alike.
        k_dynamic if dynamic else k_weight,
        v_dynamic if dynamic else v_weight,
        cache,
        key_cache,
        value_cache,
        kv_slot_mapping,
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        query_start,
        num_tokens,
        k.stride(0),
        v.stride(0),
        k_weight.stride(0),
        k_weight.stride(1),
        *dyn_strides,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.shape[1],
        block_table.stride(0),
        block_table.shape[1],
        cache.shape[2],
        (width * (2 if dynamic else 1) + 1) ** -0.5,
        clip_threshold,
        clip_eps,
        norm_eps,
        q,
        r,
        q_scale,
        r_scale,
        tau,
        rel_proj,
        q_out,
        rel_out,
        cu_seqlens_q,
        seq_lens,
        q.stride(0),
        r.stride(0),
        rel_threshold,
        W=width,
        D=head_dim,
        DYNAMIC=dynamic,
        BT=block_t,
        HQ=num_q_heads,
        D_REL=d_rel,
        E=extent,
        SHIFT_W=rel_proj.shape[2],
        HAS_TAU=tau is not None,
        BLOCK_HQ=max(triton.next_power_of_2(num_q_heads), 16),
        BLOCK_E=block_e,
        QR_TOKENS=triton.cdiv(block_t, num_heads),
        FUSE_QR=not qr_tokens,
        USE_TD=use_td,
        enable_fp_fusion=False,
        **({} if tuned else dict(num_warps=4, **pdl_launch_kwargs())),
    )
    if qr_tokens and tuned:

        def grid_qr(meta):
            return (triton.cdiv(num_tokens, meta["QR_TOKENS"]), qr_blocks)

        _qr_rel_kernel_tuned[grid_qr](
            q,
            r,
            q_scale,
            r_scale,
            tau,
            rel_proj,
            q_out,
            rel_out,
            seq_idx,
            cu_seqlens_q,
            seq_lens,
            num_tokens,
            q.stride(0),
            r.stride(0),
            clip_threshold,
            rel_threshold,
            clip_eps,
            HQ=num_q_heads,
            D=head_dim,
            D_REL=d_rel,
            E=extent,
            SHIFT_W=rel_proj.shape[2],
            HAS_TAU=tau is not None,
            BLOCK_HQ=max(triton.next_power_of_2(num_q_heads), 16),
            BLOCK_E=block_e,
            USE_TD=use_td,
            enable_fp_fusion=False,
        )
    elif qr_tokens:
        _qr_rel_kernel[(triton.cdiv(num_tokens, qr_tokens), qr_blocks)](
            q,
            r,
            q_scale,
            r_scale,
            tau,
            rel_proj,
            q_out,
            rel_out,
            seq_idx,
            cu_seqlens_q,
            seq_lens,
            num_tokens,
            q.stride(0),
            r.stride(0),
            clip_threshold,
            rel_threshold,
            clip_eps,
            HQ=num_q_heads,
            D=head_dim,
            D_REL=d_rel,
            E=extent,
            SHIFT_W=rel_proj.shape[2],
            HAS_TAU=tau is not None,
            BLOCK_HQ=max(triton.next_power_of_2(num_q_heads), 16),
            BLOCK_E=block_e,
            QR_TOKENS=qr_tokens,
            num_warps=4,
            enable_fp_fusion=False,
            **pdl_launch_kwargs(),
        )


# The prologue launches with their warps (and the q/r token count of the
# separate q/r launch) autotuned on XPU; the configs include the fixed ones,
# the prologue's first (VLLM_TRITON_FORCE_FIRST_CONFIG).
_attn_prologue_kernel_tuned = triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in (4, 2, 8)],
    key=["W", "D", "DYNAMIC", "BT", "HQ", "E", "FUSE_QR", "HAS_TAU", "USE_TD"],
)(_attn_prologue_kernel)

_qr_rel_kernel_tuned = triton.autotune(
    configs=[
        triton.Config({"QR_TOKENS": t}, num_warps=w)
        for t, w in ((8, 4), (4, 4), (16, 4), (8, 8), (16, 8))
    ],
    key=["HQ", "D", "D_REL", "E", "HAS_TAU", "USE_TD"],
)(_qr_rel_kernel)
