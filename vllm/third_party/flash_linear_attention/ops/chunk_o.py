# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# ruff: noqa: E501


import torch

from vllm.triton_utils import tl, triton, use_tensor_descriptor
from vllm.triton_utils.allocation import set_triton_allocator

from .index import prepare_chunk_indices
from .op import exp
from .utils import FLA_CHUNK_SIZE, check_shared_mem, is_nvidia_hopper

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]

_TD_ALLOCATOR_DEVICES: set[torch.device] = set()


def _td_ok(t: torch.Tensor, *dims: int) -> bool:
    # Descriptors need unit inner stride and 16-byte aligned base/rows.
    es = t.element_size()
    return (
        t.stride(-1) == 1
        and (t.storage_offset() * es) % 16 == 0
        and all((d * es) % 16 == 0 for d in dims)
    )


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_TD: tl.constexpr = False,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    if USE_TD:
        desc_q = tl.make_tensor_descriptor(
            q, shape=[T, K], strides=[Hg * K, 1], block_shape=[BT, BK]
        )
        # k is consumed as [BK, BT]; describe the T-major buffer and transpose
        # each tile, since the [K, T] view is not contiguous in its last dim.
        desc_k = tl.make_tensor_descriptor(
            k, shape=[T, K], strides=[Hg * K, 1], block_shape=[BT, BK]
        )
        desc_h = tl.make_tensor_descriptor(
            h, shape=[V, K], strides=[K, 1], block_shape=[BV, BK]
        )

    for i_k in range(tl.cdiv(K, BK)):
        if USE_TD:
            # [BT, BK]
            b_q = desc_q.load([i_t * BT, i_k * BK])
            # [BK, BT]
            b_k = tl.trans(desc_k.load([i_t * BT, i_k * BK]))
            # [BV, BK]
            b_h = desc_h.load([i_v * BV, i_k * BK])
        else:
            p_q = tl.make_block_ptr(
                q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            p_k = tl.make_block_ptr(
                k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
            )
            p_h = tl.make_block_ptr(
                h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
            )
            # [BT, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            # [BK, BT]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BV, BK]
            b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, tl.trans(b_h))
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    if USE_TD:
        desc_v = tl.make_tensor_descriptor(
            v, shape=[T, V], strides=[H * V, 1], block_shape=[BT, BV]
        )
        b_v = desc_v.load([i_t * BT, i_v * BV])
    else:
        p_v = tl.make_block_ptr(
            v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    if USE_TD:
        desc_o = tl.make_tensor_descriptor(
            o, shape=[T, V], strides=[H * V, 1], block_shape=[BT, BV]
        )
        desc_o.store([i_t * BT, i_v * BV], b_o.to(o.dtype.element_ty))
    else:
        p_o = tl.make_block_ptr(
            o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,  # cumsum of log decay
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    core_attn_out: torch.Tensor | None = None,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if core_attn_out is not None:
        assert core_attn_out.numel() >= v.numel(), (
            f"core_attn_out too small: {core_attn_out.numel()} < {v.numel()}"
        )
        o = core_attn_out[: v.numel()].view(*v.shape)
    else:
        o = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    # Descriptors cover q/k/v/o/h. g keeps block pointers: its stride is H,
    # so no descriptor can describe it.
    use_td = (
        use_tensor_descriptor()
        and BT & (BT - 1) == 0
        and _td_ok(q, Hg * K, K)
        and _td_ok(k, Hg * K, K)
        and _td_ok(v, H * V, V)
        and _td_ok(o, H * V, V)
        and _td_ok(h, K)
        and h.stride(-2) == K
    )
    if use_td and q.device not in _TD_ALLOCATOR_DEVICES:
        set_triton_allocator(q.device)
        _TD_ALLOCATOR_DEVICES.add(q.device)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_TD=use_td,
    )
    return o
