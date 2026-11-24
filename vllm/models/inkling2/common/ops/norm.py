# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 row-wise kernels: RMSNorm and the q/r L2 clipping (fused into the
attention prologue in `ops/sconv.py`).

All math is fp32; each output is rounded once to its storage dtype.
"""

import torch

from vllm.triton_utils import tl, triton

from .intrinsics import rsqrt


@triton.jit
def _rms_norm_kernel(
    X,
    W,
    Y,
    stride_x,
    N: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask, other=0).to(tl.float32)
    rstd = rsqrt(tl.div_rn(tl.sum(x * x), N) + eps)
    y = x * rstd
    y = y * tl.load(W + cols, mask, other=0).to(tl.float32)
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """RMSNorm over the last dim of a row-major (possibly row-strided) tensor."""
    n = x.shape[-1]
    assert x.stride(-1) == 1
    x2d = x if x.dim() == 2 else x.reshape(-1, n)
    out = torch.empty(x.shape, dtype=out_dtype or x.dtype, device=x.device)
    if x2d.shape[0]:
        block = triton.next_power_of_2(n)
        _rms_norm_kernel[(x2d.shape[0],)](
            x2d,
            weight,
            out,
            x2d.stride(0),
            n,
            eps,
            BLOCK=block,
            num_warps=min(max(block // 512, 1), 16),
        )
    return out


@triton.jit
def _load_heads_td(X, H, BLOCK_H: tl.constexpr, D: tl.constexpr):
    """`[BLOCK_H, D]` rows of the `[H, D]` matrix at `X`; heads past `H`
    read 0."""
    desc = tl.make_tensor_descriptor(
        X, shape=[H, D], strides=[D, 1], block_shape=[BLOCK_H, D]
    )
    return desc.load([0, 0])


@triton.jit
def _store_heads_td(X, value, H, W, col, BLOCK_H: tl.constexpr, BC: tl.constexpr):
    """Store `[BLOCK_H, BC]` at column `col` of the `[H, W]` matrix at `X`;
    heads past `H` are dropped."""
    desc = tl.make_tensor_descriptor(
        X, shape=[H, W], strides=[W, 1], block_shape=[BLOCK_H, BC]
    )
    desc.store([0, col], value.to(X.dtype.element_ty))


@triton.jit
def _load_proj_td(table, rows, W, row, col, D_REL: tl.constexpr, BC: tl.constexpr):
    """`[D_REL, BC]` tile at `(row, col)` of the `[rows, W]` shifted
    relative-projection table."""
    desc = tl.make_tensor_descriptor(
        table, shape=[rows, W], strides=[W, 1], block_shape=[D_REL, BC]
    )
    return desc.load([row.to(tl.int32), col.to(tl.int32)])


@triton.jit
def _l2_clip(
    X,
    Scale,
    heads,
    cols,
    mask,
    threshold,
    eps,
    tau,
    HAS_TAU,
    USE_TD: tl.constexpr = False,
    H=0,
):
    if USE_TD:
        BLOCK_H: tl.constexpr = heads.shape[0]
        D: tl.constexpr = cols.shape[1]
        x = _load_heads_td(X, H, BLOCK_H, D).to(tl.float32)
        x = x * _load_heads_td(Scale, H, BLOCK_H, D).to(tl.float32)
    else:
        x = tl.load(X + heads * cols.shape[1] + cols, mask, other=0).to(tl.float32)
        x = x * tl.load(Scale + heads * cols.shape[1] + cols, mask, other=0).to(
            tl.float32
        )
    sum_sq = tl.sum(x * x, axis=1)
    norm = tl.where(sum_sq > 0, sum_sq * rsqrt(sum_sq), 0.0)
    y = x * tl.minimum(tl.div_rn(threshold, norm + eps), 1.0)[:, None]
    if HAS_TAU:
        y = y.to(X.dtype.element_ty).to(tl.float32) * tau
    return y.to(X.dtype.element_ty)


@triton.jit
def qr_clip_rel_tokens(
    t0,
    num_t: tl.constexpr,
    chunk,
    T,
    Q,
    R,
    QScale,
    RScale,
    Tau,
    RelProj,
    QOut,
    RelOut,
    SeqIdx,
    CuSeqlensQ,
    SeqLens,
    stride_q,
    stride_r,
    q_threshold,
    r_threshold,
    eps,
    H: tl.constexpr,
    D: tl.constexpr,
    D_REL: tl.constexpr,
    E: tl.constexpr,
    W: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
    USE_TD: tl.constexpr = False,
):
    """For tokens `[t0, t0 + num_t)`: per-head L2 clipping of `q * q_scale`
    (`chunk < 0`) or of `r * r_scale` and the `BLOCK_E` columns `chunk`
    of the relative logits `r @ rel_proj` (fp32 accumulate; `rel_proj`
    read from the `shift_rel_proj` table `RelProj`). Clipped values are
    rounded, then scaled by the fp32 per-token `tau`.

    The logits are written sheared, as tml_fa4's `ShearingBias` does for
    128-token query tiles: `E + 256` columns per head, column `c` holding
    key `c + (n_block_max - 10) * 128`, where `n_block_max` bounds the key
    blocks of the token's tile, plus the table's padding (keys right of the
    diagonal, and for window layers those past the extent, are -inf).

    With `USE_TD`, q, its scale, the q output, the `[D_REL, BLOCK_E]` table
    tile and the logits go through tensor descriptors (r rows of `D_REL`
    values are narrower than a 2D block)."""
    E_PAD: tl.constexpr = E + 256
    heads = tl.arange(0, BLOCK_H)[:, None]
    h_mask = heads < H
    q_cols = tl.arange(0, D)[None, :]
    r_cols = tl.arange(0, D_REL)[None, :]
    e_idx = tl.arange(0, BLOCK_E)[None, :]
    for i in tl.static_range(num_t):
        t = (t0 + i).to(tl.int64)
        if t < T and chunk < 0:
            tau = tl.load(Tau + t) if HAS_TAU else 1.0
            q = _l2_clip(
                Q + t * stride_q,
                QScale,
                heads,
                q_cols,
                h_mask,
                q_threshold,
                eps,
                tau,
                HAS_TAU,
                USE_TD,
                H,
            )
            if USE_TD:
                _store_heads_td(QOut + t * H * D, q, H, D, 0, BLOCK_H, D)
            else:
                tl.store(QOut + t * H * D + heads * D + q_cols, q, h_mask)
        elif t < T:
            seq = tl.load(SeqIdx + t)
            q_start = tl.load(CuSeqlensQ + seq)
            q_len = tl.load(CuSeqlensQ + seq + 1) - q_start
            q_idx = t - q_start
            pos = tl.load(SeqLens + seq) - q_len + q_idx
            tile_end = tl.minimum((q_idx // 128 + 1) * 128, q_len)
            # Column of the diagonal key (distance 0), in [E + 1, E + 255] for a
            # real token. CUDA-graph padding rows (seq_idx 0, q_idx past the
            # request) would land outside and read before the shifted table.
            diag = pos + E_PAD - tl.cdiv(pos - q_idx + tile_end, 128) * 128
            diag = tl.minimum(tl.maximum(diag, E + 1), E_PAD - 1)
            tau = tl.load(Tau + t) if HAS_TAU else 1.0
            r = _l2_clip(
                R + t * stride_r,
                RScale,
                heads,
                r_cols,
                h_mask,
                r_threshold,
                eps,
                tau,
                HAS_TAU,
            )
            # Distance diag - c of column c is column (c - diag8 + E + 256) of
            # shifted table diag - diag8: 16-byte aligned loads.
            diag8 = diag // 8 * 8
            table = RelProj + (diag - diag8) * (D_REL + 1) * W + E + 256 - diag8
            col = chunk * BLOCK_E + e_idx
            if USE_TD:
                proj = _load_proj_td(
                    RelProj,
                    8 * (D_REL + 1),
                    W,
                    (diag - diag8) * (D_REL + 1),
                    E + 256 - diag8 + chunk * BLOCK_E,
                    D_REL,
                    BLOCK_E,
                )
            else:
                proj = tl.load(table + tl.trans(r_cols) * W + col)
            pad = tl.load(table + D_REL * W + col)
            rel = tl.dot(r, proj) + pad.to(tl.float32)
            if USE_TD:
                _store_heads_td(
                    RelOut + t * H * E_PAD,
                    rel,
                    H,
                    E_PAD,
                    chunk * BLOCK_E,
                    BLOCK_H,
                    BLOCK_E,
                )
            else:
                tl.store(
                    RelOut + t * H * E_PAD + heads * E_PAD + col,
                    rel.to(RelOut.dtype.element_ty),
                    h_mask,
                )
