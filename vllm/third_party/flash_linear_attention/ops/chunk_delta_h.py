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

from .index import prepare_chunk_indices, prepare_chunk_offsets
from .op import exp, exp2
from .utils import FLA_CHUNK_SIZE, use_cuda_graph

NUM_WARPS = [2, 4, 8, 16]
# Triton's AMD backend fails to lower this kernel with num_stages=4.
_CHUNK_DELTA_H_NUM_STAGES = [2, 3] if torch.version.hip else [2, 3, 4]

_TD_ALLOCATOR_DEVICES: set[torch.device] = set()


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in _CHUNK_DELTA_H_NUM_STAGES
        for BV in [32, 64]
    ],
    key=["H", "K", "V", "BT"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_TD: tl.constexpr = False,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, BK]
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * V * K
    if STORE_FINAL_STATE:
        ht = ht + i_nh * V * K

    if USE_TD:
        # k is [T, K]-major from this head's base; the (K, T) tile the non-TD
        # path loads is obtained by transposing a [BT, 64] descriptor tile.
        k_desc = tl.make_tensor_descriptor(
            k, shape=[T, K], strides=[stride_k, 1], block_shape=[BT, 64]
        )
        v_desc = tl.make_tensor_descriptor(
            v, shape=[T, V], strides=[stride_v, 1], block_shape=[BT, BV]
        )
        w_desc = tl.make_tensor_descriptor(
            w, shape=[T, K], strides=[stride_w, 1], block_shape=[BT, 64]
        )
        # h is contiguous, so the chunk-major region is a flat row matrix with
        # row stride K; row (i_t, i) lives at i_t * H * V + i. The host gates
        # this on V % BV == 0 so a tile never crosses into the next head.
        h_desc = tl.make_tensor_descriptor(
            h, shape=[NT * H * V, K], strides=[K, 1], block_shape=[BV, 64]
        )
        if SAVE_NEW_VALUE:
            vn_desc = tl.make_tensor_descriptor(
                v_new, shape=[T, V], strides=[stride_v, 1], block_shape=[BT, BV]
            )

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        if USE_TD:
            o_h = i_t * (H * V) + i_v * BV
            h_desc.store([o_h, 0], b_h1.to(h.dtype.element_ty))
            if K > 64:
                h_desc.store([o_h, 64], b_h2.to(h.dtype.element_ty))
            if K > 128:
                h_desc.store([o_h, 128], b_h3.to(h.dtype.element_ty))
            if K > 192:
                h_desc.store([o_h, 192], b_h4.to(h.dtype.element_ty))
        else:
            p_h1 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 0),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(
                    h + i_t.to(tl.int64) * stride_h,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 64),
                    (BV, 64),
                    (1, 0),
                )
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h3 = tl.make_block_ptr(
                    h + i_t.to(tl.int64) * stride_h,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 128),
                    (BV, 64),
                    (1, 0),
                )
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h4 = tl.make_block_ptr(
                    h + i_t.to(tl.int64) * stride_h,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 192),
                    (BV, 64),
                    (1, 0),
                )
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        if USE_TD:
            b_w = w_desc.load([i_t * BT, 0])
        else:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            if USE_TD:
                b_w = w_desc.load([i_t * BT, 64])
            else:
                p_w = tl.make_block_ptr(
                    w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
                )
                b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            if USE_TD:
                b_w = w_desc.load([i_t * BT, 128])
            else:
                p_w = tl.make_block_ptr(
                    w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
                )
                b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            if USE_TD:
                b_w = w_desc.load([i_t * BT, 192])
            else:
                p_w = tl.make_block_ptr(
                    w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
                )
                b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        if USE_TD:
            b_v = v_desc.load([i_t * BT, i_v * BV]) - b_v
        else:
            p_v = tl.make_block_ptr(
                v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            if USE_TD:
                vn_desc.store([i_t * BT, i_v * BV], b_v.to(v_new.dtype.element_ty))
            else:
                p_v = tl.make_block_ptr(
                    v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
                )
                tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t.to(tl.int64) * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        if USE_TD:
            b_k = tl.trans(k_desc.load([i_t * BT, 0]))
        else:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            if USE_TD:
                b_k = tl.trans(k_desc.load([i_t * BT, 64]))
            else:
                p_k = tl.make_block_ptr(
                    k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            if USE_TD:
                b_k = tl.trans(k_desc.load([i_t * BT, 128]))
            else:
                p_k = tl.make_block_ptr(
                    k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            if USE_TD:
                b_k = tl.trans(k_desc.load([i_t * BT, 192]))
            else:
                p_k = tl.make_block_ptr(
                    k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))
    # epilogue
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = FLA_CHUNK_SIZE,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_exp2: bool = False,
    use_td: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, V, K)
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    # TD operand loads/stores for k, v, w, v_new and h. Gated on last-dim
    # contiguity, power-of-two tiles and 16-byte alignment of every descriptor
    # row stride. V % 64 == 0 keeps an h tile (BV in {32, 64}) inside one head.
    e_k, e_u, e_w = k.element_size(), u.element_size(), w.element_size()
    use_td = (
        use_tensor_descriptor(use_td)
        and k.is_contiguous()
        and u.is_contiguous()
        and w.is_contiguous()
        and V % 64 == 0
        and (BT & (BT - 1)) == 0
        and (K * e_k) % 16 == 0
        and (V * e_u) % 16 == 0
        and (Hg * K * e_k) % 16 == 0
        and (H * K * e_w) % 16 == 0
        and (H * V * e_u) % 16 == 0
    )
    if use_td and k.device not in _TD_ALLOCATOR_DEVICES:
        set_triton_allocator(k.device)
        _TD_ALLOCATOR_DEVICES.add(k.device)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
        USE_TD=use_td,
    )
    return h, v_new, final_state
