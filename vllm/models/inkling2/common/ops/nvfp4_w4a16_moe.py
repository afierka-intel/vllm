# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped expert GEMM with NVFP4 weights and bf16 activations (W4A16).

The weights stay in the checkpoint layout:
- `[E, N, K / 2]` uint8, two e2m1 values per byte, the even K index in the
  low nibble;
- e4m3 block scales `[E, N, K / 16]`, one per 16 along K;
- fp32 per-expert global scales `[E]` (`amax / outer_scale_denominator`).

A program dequantizes a weight tile to bf16 and multiplies it with bf16
activation rows, accumulating in fp32. The rows are the token-major flat
rows of `moe_align_block_size`, grouped by expert in blocks of `BLOCK_M`.
The e2m1 x e4m3 product has at most 6 significant bits, so it is exact in
bf16, and the global scale is applied to the fp32 accumulator. Fp8 types
are not used (XPU Triton has no `float8e4nv`): the codes are decoded with
integer ops.

With `use_tensor_descriptor()` the weight tiles load through a tensor
descriptor over the `[E * N, K / 2]` bytes (2D block loads on Intel GPUs);
the gathered activation rows, the block scales (rows of `K / 16` bytes, not
16-byte aligned at every shape) and the scattered output rows stay pointer
accesses. The tiles are autotuned per weight shape, row block and path.
"""

import torch

from vllm.triton_utils import (
    tensor_descriptor_compatible,
    tl,
    triton,
    use_tensor_descriptor,
)
from vllm.triton_utils.allocation import set_triton_allocator

# (BLOCK_N, BLOCK_K, num_warps, num_stages), tuned on Intel B70 (Xe2).
_TILES = (
    (64, 128, 4, 2),
    (128, 64, 4, 2),
    (128, 128, 4, 2),
    (128, 128, 8, 2),
    (256, 64, 8, 2),
    (256, 128, 8, 2),
    (64, 256, 4, 2),
    (128, 256, 8, 3),
)


def _prune_tiles(configs, named_args, **kwargs):
    """The tiles that divide the weight shape `(N, K)`."""
    args = {**named_args, **kwargs}
    n, k = args["N"], args["K"]
    return [
        c
        for c in configs
        if n % c.kwargs["BLOCK_N"] == 0 and k % c.kwargs["BLOCK_K"] == 0
    ]


@triton.jit
def _e4m3_to_f32(codes):
    """fp32 values of e4m3fn codes (finite only: the block scales)."""
    bits = codes.to(tl.int32)
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).to(tl.float32, bitcast=True)
    value = tl.where(exponent == 0, mantissa.to(tl.float32) * 0.001953125, normal)
    return tl.where((bits & 0x80) != 0, -value, value)


@triton.jit
def _e2m1_times4(codes):
    """4 x the e2m1 values of the 4-bit `codes` (int32), as int32: the
    magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 become 0, 2, 4, 6, 8, 12, 16, 24."""
    exponent = (codes >> 1) & 0x3
    mantissa = codes & 0x1
    value = tl.where(exponent == 0, 2 * mantissa, (2 + mantissa) << exponent)
    return tl.where((codes & 0x8) != 0, -value, value)


@triton.jit
def _dequant(packed, scales, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """bf16 weights of the even and odd K indices `[BLOCK_N, BLOCK_K / 2]`
    of a packed tile with its block scales `[BLOCK_N, BLOCK_K / 16]`."""
    scale = _e4m3_to_f32(scales) * 0.25
    # The block scale of pair j (K indices 2j and 2j + 1) is scales[:, j // 8].
    scale = tl.reshape(
        tl.broadcast_to(scale[:, :, None], (BLOCK_N, BLOCK_K // 16, 8)),
        (BLOCK_N, BLOCK_K // 2),
    )
    packed = packed.to(tl.int32)
    even = (_e2m1_times4(packed & 0xF).to(tl.float32) * scale).to(tl.bfloat16)
    odd = (_e2m1_times4(packed >> 4).to(tl.float32) * scale).to(tl.bfloat16)
    return even, odd


@triton.jit
def _load_weight(w_ptrs):
    """Packed weight tile `[BLOCK_N, BLOCK_K / 2]` by pointers."""
    return tl.load(w_ptrs)


@triton.jit
def _load_weight_td(w_desc, row, col):
    """Packed weight tile `[BLOCK_N, BLOCK_K / 2]` at `(row, col)` of the
    `[E * N, K / 2]` descriptor."""
    return w_desc.load([row, col])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": n, "BLOCK_K": k}, num_warps=w, num_stages=s)
        for n, k, w, s in _TILES
    ],
    key=["N", "K", "BLOCK_M", "USE_TD"],
    prune_configs_by={"early_config_prune": _prune_tiles},
)
@triton.jit
def _nvfp4_w4a16_moe_kernel(
    A,
    Weight,
    Scales,
    GlobalScales,
    Out,
    SortedRows,
    BlockExperts,
    NumRowsPadded,
    num_rows,
    num_experts,
    N: tl.constexpr,
    K: tl.constexpr,
    A_DIV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    USE_TD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """`Out[r] = A[r // A_DIV] @ dequant(W[e]).T` for the rows `r` of
    expert `e` in row block `program_id(0)` (`r == num_rows`: padding)."""
    pid_m = tl.program_id(0)
    if pid_m * BLOCK_M >= tl.load(NumRowsPadded):
        return
    expert = tl.load(BlockExperts + pid_m).to(tl.int64)
    if expert < 0:
        return
    rows = tl.load(SortedRows + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    valid = rows < num_rows
    a_rows = tl.where(valid, rows // A_DIV, 0).to(tl.int64)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = tl.arange(0, BLOCK_K // 2)
    offs_g = tl.arange(0, BLOCK_K // 16)
    a_ptrs = A + a_rows[:, None] * K + 2 * offs_j[None, :]
    w_ptrs = (
        Weight
        + expert * (N * (K // 2))
        + offs_n[:, None].to(tl.int64) * (K // 2)
        + offs_j[None, :]
    )
    s_ptrs = (
        Scales
        + expert * (N * (K // 16))
        + offs_n[:, None].to(tl.int64) * (K // 16)
        + offs_g[None, :]
    )
    if USE_TD:
        w_desc = tl.make_tensor_descriptor(
            Weight,
            shape=[num_experts * N, K // 2],
            strides=[K // 2, 1],
            block_shape=[BLOCK_N, BLOCK_K // 2],
        )
        w_row = expert.to(tl.int32) * N + tl.program_id(1) * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(K // BLOCK_K):
        a_even = tl.load(a_ptrs, mask=valid[:, None], other=0.0)
        a_odd = tl.load(a_ptrs + 1, mask=valid[:, None], other=0.0)
        if USE_TD:
            packed = _load_weight_td(w_desc, w_row, k * (BLOCK_K // 2))
        else:
            packed = _load_weight(w_ptrs)
        w_even, w_odd = _dequant(packed, tl.load(s_ptrs), BLOCK_N, BLOCK_K)
        acc = tl.dot(a_even, tl.trans(w_even), acc)
        acc = tl.dot(a_odd, tl.trans(w_odd), acc)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K // 2
        s_ptrs += BLOCK_K // 16
    acc = acc * tl.load(GlobalScales + expert)
    out_ptrs = Out + rows.to(tl.int64)[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=valid[:, None])


def nvfp4_block_m(rows: int, num_global_experts: int) -> int:
    """Row block of `nvfp4_w4a16_moe` for `rows` flat rows over
    `num_global_experts` experts: about the rows of one expert, 16 to 64."""
    per_expert = -(-rows // num_global_experts)
    return min(64, max(16, triton.next_power_of_2(per_expert)))


def nvfp4_w4a16_moe(
    a: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    global_scales: torch.Tensor,
    alignment: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_rows: int,
    a_div: int,
    block_m: int,
    use_td: bool | None = None,
) -> torch.Tensor:
    """bf16 `[num_rows, N]`: row `r` of the aligned rows is
    `a[r // a_div] @ W[e].T` for its expert `e`, with `W` the NVFP4
    `weight` `[E, N, K / 2]`, `scales` `[E, N, K / 16]` (e4m3) and
    `global_scales` `[E]`. `alignment` is `(sorted_token_ids, expert_ids,
    num_tokens_post_padded)` of `moe_align_block_size(ids, block_m, E)`;
    the rows it does not list are left unset. `use_td` overrides
    `use_tensor_descriptor()`."""
    sorted_rows, block_experts, num_rows_padded = alignment
    _, n, k_half = weight.shape
    k = 2 * k_half
    assert a.shape[1] == k and a.dtype == torch.bfloat16 and a.is_contiguous()
    assert scales.shape == (weight.shape[0], n, k // 16)
    assert scales.dtype == torch.float8_e4m3fn and global_scales.dtype == torch.float32
    assert n % 64 == 0 and k % 64 == 0, (n, k)
    use_td = use_tensor_descriptor(use_td) and tensor_descriptor_compatible(weight)
    if use_td:
        # On every launch: Triton's allocator is a per-thread ContextVar.
        set_triton_allocator(a.device)
    out = torch.empty((num_rows, n), dtype=torch.bfloat16, device=a.device)

    def grid(meta):
        return (triton.cdiv(sorted_rows.numel(), block_m), n // meta["BLOCK_N"])

    _nvfp4_w4a16_moe_kernel[grid](
        a,
        weight,
        scales.view(torch.uint8),
        global_scales,
        out,
        sorted_rows,
        block_experts,
        num_rows_padded,
        num_rows,
        weight.shape[0],
        N=n,
        K=k,
        A_DIV=a_div,
        BLOCK_M=block_m,
        USE_TD=use_td,
    )
    return out
