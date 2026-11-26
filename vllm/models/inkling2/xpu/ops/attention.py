# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU copy of `vllm/models/inkling2/nvidia/ops/attention.py` (upstream
`b7186d137a`): the gate/sink, log-scaling tau and split-KV decode parts.

Adapted for XPU: the decode/verify launch runs `rel_attention`
(`ops/rel_attention.py`) instead of the tcgen05 kernel, the small-batch
partials combine is autotuned, and the FA4 prefill, the DSpark draft block
attention and the PTX math are removed.
"""

import functools

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.platform_utils import num_compute_units

from .autotune import autotune
from .intrinsics import ex2, lg2, sigmoid
from .platform import use_kernel_autotune
from .rel_attention import MODE_GATED, MODE_PARTIAL, rel_attention


@triton.jit
def _gate_sink_kernel(
    X,
    Gate,
    LSE,
    Sink,
    Y,
    stride_x,
    stride_gate,
    stride_lse_t,
    stride_lse_h,
    stride_sink,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H * D
    head = col // D
    x = tl.load(X + row * stride_x + col, mask, other=0)
    gate = tl.load(Gate + row * stride_gate + col, mask, other=0).to(tl.float32)
    lse = tl.load(LSE + row * stride_lse_t + head * stride_lse_h, mask, other=0)
    sink = tl.load(Sink + row * stride_sink + head, mask, other=0).to(tl.float32)
    result = x.to(tl.float32) * sigmoid(gate)
    result = result * sigmoid(lse.to(tl.float32) - sink)
    tl.store(Y + row * H * D + col, result, mask)


def gate_sink(
    output: torch.Tensor,
    gate: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
    result: torch.Tensor,
) -> None:
    """Write bf16 `output * sigmoid(gate) * sigmoid(lse - sink)` per head to
    `result`.

    `gate` and `result` are `(T, H * D)`, `sinks` and `lse` are
    `(T, H)` and `output` is `(T, H, D)`.
    """
    num_tokens, num_heads = sinks.shape
    assert lse.shape == (num_tokens, num_heads)
    assert output.shape[1] == num_heads and output.is_contiguous()
    assert gate.stride(1) == sinks.stride(1) == 1
    assert result.shape == gate.shape and result.is_contiguous()
    width = gate.shape[1]
    _gate_sink_kernel[(num_tokens, triton.cdiv(width, 512))](
        output,
        gate,
        lse,
        sinks,
        result,
        output.stride(0),
        gate.stride(0),
        lse.stride(0),
        lse.stride(1),
        sinks.stride(0),
        num_heads,
        width // num_heads,
        512,
        enable_fp_fusion=False,
    )


@triton.jit
def _log_scaling_tau_kernel(
    Pos,
    Out,
    n,
    FLOOR: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(Pos + i, i < n, other=0)
    ratio = tl.maximum((pos + 1).to(tl.float32) * (1.0 / FLOOR), 1.0)
    tau = 1.0 + ALPHA * tldevice.log(ratio)
    tl.store(Out + i, tau, i < n)


def compute_log_scaling_tau(
    positions: torch.Tensor, n_floor: int, alpha: float
) -> torch.Tensor:
    """Return fp32 `1 + alpha * log(max((positions + 1) / n_floor, 1))`."""
    output = torch.empty(positions.shape, device=positions.device, dtype=torch.float32)
    n = positions.numel()
    _log_scaling_tau_kernel[(triton.cdiv(n, 256),)](
        positions, output, n, n_floor, alpha, 256, enable_fp_fusion=False
    )
    return output


# Key block of the sheared relative-logit layout: the logits carry
# `2 * _REL_BLOCK_N` padding columns, and the local split count is one
# block of the window per split.
_REL_BLOCK_N = 128
# Query rows (tokens x GQA group) of a decode/verify program.
_VERIFY_BLOCK_M = 64


def plan_decode_attention(
    is_local: bool,
    num_reqs: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    rel_extent: int,
) -> int:
    """Return the split count of `rel_decode_attention`, or 0 to run the
    batch as prefill (`rel_attention` with `MODE_OUT_LSE`).

    Each program holds all queries of one request for one KV head. Local
    layers (`rel_extent`: the window) give each split one key block of the
    window, or a few blocks for larger batches (a single split writes the
    gated output without the combine kernel); decode keeps one block per
    split up to twice the work of verify. Full layers give each request and
    KV head `num_cus / 8` splits, fewer once that exceeds 6 programs per
    compute unit, but never less than one program per compute unit. The
    count depends only on the batch, not the KV lengths, so it holds for
    every replay of a graph.
    """
    if max_query_len > _VERIFY_BLOCK_M // (num_heads // num_kv_heads):
        return 0
    work = num_reqs * num_kv_heads
    num_cus = num_compute_units()
    if is_local:
        if work <= (16 if max_query_len <= 2 else 8):
            return triton.cdiv(rel_extent - 1, _REL_BLOCK_N) + 1
        return 5 if work <= 16 else 3 if work <= 32 else 2 if work <= 64 else 1
    return max(num_cus // work, min(num_cus // 8, 6 * num_cus // work))


# `num_tokens` (the batch's rows, padded to a graph size) and
# `num_actual_tokens` take every value serving produces; they only index and
# bound rows, so without specialization the kernel compiles once per split
# plan instead of once per (==1, %16==0, other) class of each.
@triton.jit(do_not_specialize=["num_tokens", "num_actual_tokens"])
def _rel_decode_combine_kernel(
    OPart,
    LsePart,
    Gate,
    Sink,
    Y,
    num_tokens,
    num_actual_tokens,
    stride_gate,
    stride_sink,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Merge the split partials, then apply the output gate and attention
    sink."""
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs_d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    gate = tl.load(Gate + row * stride_gate + head * D + offs_d).to(tl.float32)
    sink = tl.load(Sink + row * stride_sink + head).to(tl.float32)
    y_ptrs = Y + row * H * D + head * D + offs_d
    if row >= num_actual_tokens:
        tl.store(y_ptrs, tl.zeros([BLOCK_D], tl.float32))
        return
    splits = tl.arange(0, BLOCK_S)
    part = (splits * num_tokens + row) * H + head
    lse_s = tl.load(LsePart + part, mask=splits < NUM_SPLITS, other=float("-inf"))
    lse_max = tl.max(lse_s)
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    w = ex2(lse_s - lse_max)
    w_sum = tl.sum(w)
    o_s = tl.load(
        OPart + part[:, None] * D + offs_d[None, :],
        mask=(w > 0.0)[:, None],
        other=0.0,
    )
    # Requests without keys (padding) come out zero.
    o = tl.sum(o_s * (w / tl.where(w_sum == 0.0, 1.0, w_sum))[:, None], 0)
    lse = (lg2(w_sum) + lse_max) * 0.6931471805599453
    o = o * sigmoid(gate)
    o = o * sigmoid(lse - sink)
    tl.store(y_ptrs, o)


def rel_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    rel_logits: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
    is_local: bool,
    num_splits: int,
) -> None:
    """Write the gated bf16 `(T, H * D)` attention output for small batches.

    Split-KV relative attention on `rel_attention` (see
    `plan_decode_attention`) reading the sheared relative logits. With more
    than one split, a second launch merges the fp32 partials and applies the
    output gate and attention sink. Tokens from `num_actual_tokens` on are
    zeroed.
    """
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = key_cache.shape[2]
    group = num_heads // num_kv_heads
    extent_pad = rel_logits.shape[2]
    extent = extent_pad - 2 * _REL_BLOCK_N
    assert rel_logits.shape[0] >= num_tokens and rel_logits.shape[1] == num_heads
    assert rel_logits.stride(1) == extent_pad and rel_logits.stride(2) == 1
    assert extent % _REL_BLOCK_N == 0
    assert gate.stride(1) == sinks.stride(1) == 1
    assert max_query_len <= _VERIFY_BLOCK_M // group
    assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
    assert q.stride(1) == head_dim and q.stride(2) == 1
    assert out.is_contiguous()
    # A single split writes the gated output itself.
    fuse_gate = num_splits == 1
    o_part = q.new_empty(
        (0 if fuse_gate else num_splits, *q.shape), dtype=torch.float32
    )
    lse_part = q.new_empty(
        (0 if fuse_gate else num_splits, num_tokens, num_heads), dtype=torch.float32
    )
    rel_attention(
        q,
        key_cache,
        value_cache,
        out if fuse_gate else o_part,
        None if fuse_gate else lse_part,
        rel_logits,
        gate,
        sinks,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_query_len=max_query_len,
        softmax_scale=softmax_scale,
        window=extent if is_local else 0,
        num_splits=num_splits,
        mode=MODE_GATED if fuse_gate else MODE_PARTIAL,
    )
    if fuse_gate:
        out[num_actual_tokens:].zero_()
        return
    args = (
        o_part,
        lse_part,
        gate,
        sinks,
        out,
        num_tokens,
        num_actual_tokens,
        gate.stride(0),
        sinks.stride(0),
    )
    meta = dict(
        H=num_heads,
        D=head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_S=triton.next_power_of_2(num_splits),
    )
    large = num_tokens * num_heads >= 512
    if use_kernel_autotune() and not large:

        def grid(meta):
            return (num_tokens, num_heads, head_dim // meta["BLOCK_D"])

        _rel_decode_combine_kernel_tuned()[grid](*args, **meta)
        return
    # Small batches spread the merge over more programs; larger ones merge a
    # whole head per (narrow) program. Large batches always run this fixed
    # config: the tuned picks were within 0.96-1.00 of it (A/A-controlled
    # timings on a B70).
    if large:
        block_d, num_warps = head_dim, 2 if num_splits > 32 else 1
    else:
        block_d, num_warps = 32, 4
    _rel_decode_combine_kernel[(num_tokens, num_heads, head_dim // block_d)](
        *args, **meta, BLOCK_D=block_d, num_warps=num_warps
    )


# (BLOCK_D, num_warps) of the small-batch partials combine
# (`num_tokens * num_heads < 512`), the fixed tile first; every BLOCK_D
# divides the head dim 128.
_COMBINE_TILES = ((32, 4), (64, 2), (128, 1))


def _prune_combine_tiles(configs, named_args, **kwargs):
    """The tiles whose BLOCK_D divides the head dim (the grid covers
    `D // BLOCK_D` blocks)."""
    head_dim = {**named_args, **kwargs}["D"]
    return [c for c in configs if head_dim % c.kwargs["BLOCK_D"] == 0] or configs[:1]


@functools.cache
def _rel_decode_combine_kernel_tuned():
    """The combine autotuned for small batches; the fixed (BLOCK_D,
    num_warps) first, with the fixed launch's (XPU default) 2 stages."""
    return autotune(
        configs=[
            triton.Config({"BLOCK_D": b}, num_warps=w, num_stages=2)
            if i == 0
            else triton.Config({"BLOCK_D": b}, num_warps=w)
            for i, (b, w) in enumerate(_COMBINE_TILES)
        ],
        key=["H", "D", "NUM_SPLITS"],
        prune_configs_by={"early_config_prune": _prune_combine_tiles},
    )(_rel_decode_combine_kernel)
