# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU copy of `vllm/models/inkling2/nvidia/ops/conv_mix.py` (upstream `b7186d137a`).

Adapted: the fused `csrc/inkling2` conv/mix ops (and their sequence-parallel
variant) are CUDA only, so the conv and the mix run as `short_conv` +
`attn_res`, the path that the CUDA copy takes on SM12x.

Hidden-stream short conv followed by the AttnRes mix + RMSNorm of the next
sublayer, for large batches of an already reduced input.
"""

import torch

from .attn_res import attn_res
from .sconv import short_conv


def conv_mix(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    conv_out: torch.Tensor,
    residual: torch.Tensor | None,
    addend: torch.Tensor | None,
    stack: torch.Tensor,
    last: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """`short_conv` the reduced bf16 `x` into `conv_out` (with the
    optional `(residual + addend) + y` update, and the conv-state cache
    writes), then `attn_res` over `[*stack, (last +) conv_out]` with each
    query. Returns the `(Q, T, D)` bf16 mix outputs, or without
    `norm_weight` the fp32 mixes before the RMSNorm.
    """
    num_tokens, d = conv_out.shape
    assert num_tokens > 0 and conv_out.is_contiguous()
    assert x.shape == (num_tokens, d) and x.stride(-1) == 1 and x.stride(-2) == d
    assert x.dtype == torch.bfloat16
    for t in (residual, addend, last):
        assert t is None or (t.is_contiguous() and t.shape == conv_out.shape)
    assert residual is None or addend is not None
    assert stack.stride(1) == d and stack.stride(2) == 1
    assert queries.dtype == torch.float32 and queries.is_contiguous()
    if norm_weight is not None:
        assert norm_weight.shape == queries.shape and norm_weight.is_contiguous()
    assert conv_weight.shape == (d, 4) and conv_weight.is_contiguous()
    assert d == cache.shape[1] * stream_width and cache.stride(3) == 1
    assert positions.shape[0] == num_tokens
    assert positions.dtype == slot_mapping.dtype == torch.int64
    short_conv(
        x,
        conv_weight,
        cache,
        positions,
        block_table,
        seq_idx,
        slot_mapping,
        query_start,
        stream_offset,
        stream_width,
        addend,
        residual,
        conv_out,
    )
    return attn_res(
        stack,
        conv_out if last is None else last,
        None if last is None else conv_out,
        queries,
        norm_weight,
        norm_eps,
        True,
        eps,
    )
