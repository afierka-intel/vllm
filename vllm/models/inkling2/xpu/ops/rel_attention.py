# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU copy of `vllm/models/inkling2/nvidia/ops/attention_sm90.py` (upstream
`b7186d137a`).

Adapted for XPU: BF16 KV only (the FP8 KV loads and the DSpark draft mode are
removed), per-page tensor-descriptor K/V loads and autotuned key tiles.

Inkling2 relative-bias paged attention (Triton), for prefill and for the
split-KV decode/verify launches of `ops/attention.py`. It reads the sheared
relative logits of `attention_prologue`: column `c` of token `t` holds key
`c + cdiv(past + tile_end, 128) * 128 - (E + 256)`, where `tile_end` ends
the token's 128-row query tile (see `qr_clip_rel_tokens`).

A program owns `BLOCK_M` rows (tokens x GQA group) of one request and KV
head, and one KV split: split `s` of `S` takes key blocks `s, s + S, ...`
of the rows' visible range, so a split count that depends only on the batch
holds for every replay of a graph.

With `use_kernel_td()`, the K and V tiles load through per-page tensor
descriptors (2D block loads); a key tile then never spans two pages
(`PAGE % BLOCK_N == 0`, asserted in the kernel). Q, the sheared relative
logits and the outputs stay pointer accesses: their rows are tokens x GQA
group, and the group (3 or 6 for Inkling 2) is not a power of two. With
`use_kernel_autotune()` the key tile and warps are autotuned, with the fixed
launch config first.
"""

import torch

from vllm.triton_utils import (
    tensor_descriptor_compatible,
    tl,
    triton,
)
from vllm.triton_utils.allocation import set_triton_allocator

from .autotune import autotune
from .intrinsics import LOG2E, sigmoid
from .platform import use_kernel_autotune, use_kernel_td

# The query tile that shears the relative logits.
_SHEAR_TILE = tl.constexpr(128)

# Output of `_rel_attention_kernel`: fp32 output and natural-log LSE
# (prefill), the gated bf16 output, or fp32 split partials with log2 LSEs
# (merged by `_rel_decode_combine_kernel`).
MODE_OUT_LSE = 0
MODE_GATED = 1
MODE_PARTIAL = 2


def check_kv_cache_dtype(key_cache: torch.Tensor, value_cache: torch.Tensor) -> None:
    """Raise unless the KV cache is BF16, the only KV cache dtype of the
    Inkling2 XPU kernels."""
    for cache in (key_cache, value_cache):
        if cache.dtype != torch.bfloat16:
            raise ValueError(
                f"Inkling2 on XPU supports only a bfloat16 KV cache, got {cache.dtype}"
            )


@triton.jit
def _load_kv_tile_td(
    cache,
    page,
    kv_off,
    offset_in_page,
    stride_kv_blk,
    stride_kv_slot,
    PAGE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """`[BLOCK_N, D]` keys `offset_in_page + [0, BLOCK_N)` of KV-cache page
    `page` through a descriptor over that page's `[PAGE, D]` slots of one
    KV head."""
    desc = tl.make_tensor_descriptor(
        cache + page * stride_kv_blk + kv_off,
        shape=[PAGE, D],
        strides=[stride_kv_slot, 1],
        block_shape=[BLOCK_N, D],
    )
    return desc.load([offset_in_page, 0])


@triton.jit
def _rel_attention_kernel(
    Q,
    K,
    V,
    Rel,
    BlockTable,
    SeqLens,
    CuSeqlensQ,
    Out,
    Lse,
    Gate,
    Sink,
    softmax_scale,
    num_tokens,
    num_splits,
    stride_q_t,
    stride_q_h,
    stride_kv_blk,
    stride_kv_slot,
    stride_kv_h,
    stride_rel_t,
    stride_bt,
    stride_out_t,
    stride_out_h,
    stride_lse_h,
    stride_lse_t,
    stride_gate,
    stride_sink,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    MODE: tl.constexpr,
    USE_TD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if USE_TD:
        # The per-page descriptors load whole key tiles of one page.
        tl.static_assert(PAGE % BLOCK_N == 0)
    TOKENS: tl.constexpr = BLOCK_M // G
    E_PAD: tl.constexpr = E + 256
    m_tile = tl.program_id(0)
    kv_head = tl.program_id(1) // num_splits
    split = tl.program_id(1) % num_splits
    req = tl.program_id(2)
    q_start = tl.load(CuSeqlensQ + req)
    q_len = tl.load(CuSeqlensQ + req + 1) - q_start
    first_tok = m_tile * TOKENS
    if first_tok >= q_len:
        return
    seqlen_k = tl.load(SeqLens + req)
    past = seqlen_k - q_len

    rows = tl.arange(0, BLOCK_M)
    tok = first_tok + rows // G
    row_valid = (rows < TOKENS * G) & (tok < q_len)
    head = kv_head * G + rows % G
    token = (q_start + tok).to(tl.int64)
    pos = past + tok
    pos_lo = past + first_tok
    pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1
    # Visible key blocks of the tile, split `split` taking every
    # `num_splits`-th of them.
    n_lo = 0
    if WINDOW > 0:
        n_lo = tl.maximum(pos_lo - (WINDOW - 1), 0) // BLOCK_N
    n_hi = tl.minimum(pos_hi + 1, seqlen_k)
    n_hi = (n_hi + BLOCK_N - 1) // BLOCK_N
    num_iters = tl.maximum(n_hi - n_lo - split + num_splits - 1, 0) // num_splits

    offs_d = tl.arange(0, D)
    q = tl.load(
        Q + token[:, None] * stride_q_t + head[:, None] * stride_q_h + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )
    tile_end = tl.minimum((tok // _SHEAR_TILE + 1) * _SHEAR_TILE, q_len)
    rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
    rel_row = Rel + token * stride_rel_t + head * E_PAD - rel_base

    qk_scale = softmax_scale * LOG2E
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    block_table = BlockTable + req.to(tl.int64) * stride_bt
    kv_off = kv_head * stride_kv_h
    for i in range(num_iters):
        n = n_lo + split + i * num_splits
        keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
        key_valid = keys < seqlen_k
        if USE_TD:
            # The tile's keys share one page; slots past `seqlen_k` are read
            # as stored, then masked below.
            tile_page = tl.load(block_table + n * BLOCK_N // PAGE).to(tl.int64)
            offset_in_page = n * BLOCK_N % PAGE
            k = _load_kv_tile_td(
                K,
                tile_page,
                kv_off,
                offset_in_page,
                stride_kv_blk,
                stride_kv_slot,
                PAGE,
                BLOCK_N,
                D,
            ).T.to(q.dtype)
        else:
            page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
            kv_ptrs = (
                page.to(tl.int64) * stride_kv_blk
                + (keys % PAGE) * stride_kv_slot
                + kv_off
            )
            k = tl.load(
                K + kv_ptrs[None, :] + offs_d[:, None],
                mask=key_valid[None, :],
                other=0.0,
            ).to(q.dtype)
        s = tl.dot(q, k) * qk_scale
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & key_valid[None, :]
        if WINDOW > 0:
            visible &= dist < WINDOW
        in_extent = visible & (dist < E) & row_valid[:, None]
        bias = tl.load(rel_row[:, None] + keys[None, :], mask=in_extent, other=0.0)
        s += bias.to(tl.float32) * LOG2E
        s = tl.where(visible, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if USE_TD:
            v = _load_kv_tile_td(
                V,
                tile_page,
                kv_off,
                offset_in_page,
                stride_kv_blk,
                stride_kv_slot,
                PAGE,
                BLOCK_N,
                D,
            ).to(q.dtype)
            # Stale slots may hold non-finite values, where p is 0.
            v = tl.where(key_valid[:, None], v, 0.0).to(q.dtype)
        else:
            v = tl.load(
                V + kv_ptrs[:, None] + offs_d[None, :],
                mask=key_valid[:, None],
                other=0.0,
            ).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    empty = l_i == 0.0
    o = acc / tl.where(empty, 1.0, l_i)[:, None]
    lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))
    if MODE == 0:
        tl.store(
            Out
            + token[:, None] * stride_out_t
            + head[:, None] * stride_out_h
            + offs_d[None, :],
            o,
            mask=row_valid[:, None],
        )
        tl.store(
            Lse + head * stride_lse_h + token * stride_lse_t,
            lse2 * 0.6931471805599453,
            mask=row_valid,
        )
    elif MODE == 1:
        cols = head[:, None] * D + offs_d[None, :]
        gate = tl.load(
            Gate + token[:, None] * stride_gate + cols,
            mask=row_valid[:, None],
            other=0.0,
        )
        sink = tl.load(Sink + token * stride_sink + head, mask=row_valid, other=0.0)
        o = o * sigmoid(gate.to(tl.float32))
        o = o * sigmoid(lse2 * 0.6931471805599453 - sink.to(tl.float32))[:, None]
        tl.store(
            Out + token[:, None] * stride_out_t + cols,
            o.to(Out.dtype.element_ty),
            mask=row_valid[:, None],
        )
    else:
        part = (split * num_tokens + token) * H + head
        tl.store(Lse + part, lse2, mask=row_valid)
        tl.store(Out + part[:, None] * D + offs_d[None, :], o, mask=row_valid[:, None])


def _block_m(max_query_len: int, group: int) -> int:
    rows = max_query_len * group
    return 16 if rows <= 16 else 32 if rows <= 32 else 64 if rows <= 64 else 128


def _fixed_config(block_m: int) -> dict[str, int]:
    """The launch config without autotuning (`VLLM_BATCH_INVARIANT`)."""
    return dict(
        BLOCK_N=64 if block_m >= 128 else 128 if block_m <= 32 else 64,
        num_warps=8 if block_m >= 128 else 4,
        num_stages=2,
    )


# (BLOCK_N, num_warps) autotuned, the fixed configs among them.
_TUNED_TILES = ((128, 4), (64, 4), (64, 8), (128, 8))


def _prune_tiles(configs, named_args, **kwargs):
    """With USE_TD, the key tiles that do not span two pages."""
    args = {**named_args, **kwargs}
    if not args["USE_TD"]:
        return configs
    return [c for c in configs if args["PAGE"] % c.kwargs["BLOCK_N"] == 0]


def _tuned_tiles(block_m: int) -> tuple[tuple[int, int], ...]:
    """`_TUNED_TILES` with the fixed config of `block_m` first."""
    fixed = _fixed_config(block_m)
    first = (fixed["BLOCK_N"], fixed["num_warps"])
    return (first, *(tile for tile in _TUNED_TILES if tile != first))


def _tuned_kernel(tiles: tuple[tuple[int, int], ...]):
    # The same kernel with the key tile and warps autotuned per shape and
    # path. A launch only writes its outputs, so the tuning runs leave no
    # trace.
    return autotune(
        configs=[
            triton.Config({"BLOCK_N": n}, num_warps=w, num_stages=2) for n, w in tiles
        ],
        key=["G", "D", "PAGE", "WINDOW", "MODE", "USE_TD", "BLOCK_M"],
        prune_configs_by={"early_config_prune": _prune_tiles},
    )(_rel_attention_kernel)


# One autotuner per fixed config (decode BLOCK_M <= 32, 33-127, prefill 128),
# each listing that fixed config first.
_rel_attention_kernel_tuned = {
    tiles[0]: _tuned_kernel(tiles)
    for tiles in (_tuned_tiles(16), _tuned_tiles(64), _tuned_tiles(128))
}


def _rel_attention_tuner(block_m: int):
    """The autotuner of `block_m`'s launches (its fixed config first)."""
    return _rel_attention_kernel_tuned[_tuned_tiles(block_m)[0]]


def use_kv_descriptors(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_m: int,
    use_td: bool | None = None,
) -> bool:
    """Whether `rel_attention` loads K/V tiles through per-page descriptors:
    on the autotuned path (`use_kernel_td(use_td)`), with pages that hold
    whole key tiles of the smallest tuned tile (tiles that do not divide the
    page are pruned)."""
    if not use_kernel_td(use_td):
        return False
    min_block_n = min(n for n, _ in _TUNED_TILES)
    return key_cache.shape[1] % min_block_n == 0 and tensor_descriptor_compatible(
        key_cache, value_cache
    )


def rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    rel_logits: torch.Tensor,
    gate: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
    mode: int,
    use_td: bool | None = None,
) -> None:
    """Launch `_rel_attention_kernel` over the requests of `cu_seqlens_q`.

    `rel_logits` are the sheared `(T + 128, H, extent + 256)` relative
    logits. `window` (keys up to a query's position, inclusive) is 0 for full
    layers. `mode` selects the output: `MODE_OUT_LSE` writes the fp32 `out`
    `(T, H, D)` and natural-log `lse` `(H, T)`; `MODE_GATED` the bf16
    `(T, H * D)` `out`, gated by `gate` and `sinks` (a single split);
    `MODE_PARTIAL` the fp32 `out` `(S, T, H, D)` and log2 `lse`
    `(S, T, H)` partials. `use_td` overrides `use_tensor_descriptor()`
    (see `use_kv_descriptors`). The KV cache must be bfloat16.
    """
    check_kv_cache_dtype(key_cache, value_cache)
    num_tokens, num_heads, head_dim = q.shape
    page, num_kv_heads = key_cache.shape[1:3]
    group = num_heads // num_kv_heads
    assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
    assert q.stride(2) == 1
    assert mode != MODE_GATED or num_splits == 1
    extent = rel_logits.shape[2] - 256
    assert rel_logits.stride(1) == rel_logits.shape[2]
    assert rel_logits.stride(2) == 1
    if mode == MODE_GATED:
        assert gate is not None and sinks is not None
        assert gate.stride(1) == sinks.stride(1) == 1
    # Unused pointers keep the launches typed alike.
    gate = q if gate is None else gate
    sinks = q if sinks is None else sinks
    if lse is None:
        lse = out
    block_m = _block_m(max_query_len, group)
    tokens_per_tile = block_m // group
    num_reqs = cache_seqlens.shape[0]
    grid = (
        triton.cdiv(max_query_len, tokens_per_tile),
        num_kv_heads * num_splits,
        num_reqs,
    )
    if mode == MODE_OUT_LSE:
        out_strides = (out.stride(0), out.stride(1), lse.stride(0), lse.stride(1))
    elif mode == MODE_GATED:
        assert out.is_contiguous()
        out_strides = (out.stride(0), 0, 0, 0)
    else:
        assert out.is_contiguous() and lse.is_contiguous()
        out_strides = (0, 0, 0, 0)
    tuned = use_kernel_autotune()
    config = {} if tuned else _fixed_config(block_m)
    use_td = use_kv_descriptors(key_cache, value_cache, block_m, use_td)
    if use_td:
        # On every launch: Triton's allocator is a per-thread ContextVar.
        set_triton_allocator(q.device)
    kernel = _rel_attention_tuner(block_m) if tuned else _rel_attention_kernel
    kernel[grid](
        q,
        key_cache,
        value_cache,
        rel_logits,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        out,
        lse,
        gate,
        sinks,
        softmax_scale,
        num_tokens,
        num_splits,
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        rel_logits.stride(0),
        block_table.stride(0),
        *out_strides,
        gate.stride(0),
        sinks.stride(0),
        H=num_heads,
        G=group,
        D=head_dim,
        E=extent,
        PAGE=page,
        WINDOW=window,
        MODE=mode,
        USE_TD=use_td,
        BLOCK_M=block_m,
        **config,
    )
