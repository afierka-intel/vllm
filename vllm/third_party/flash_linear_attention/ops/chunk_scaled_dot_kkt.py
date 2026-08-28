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

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton, use_tensor_descriptor
from vllm.triton_utils.allocation import set_triton_allocator

from .index import prepare_chunk_indices
from .op import exp
from .utils import FLA_CHUNK_SIZE

# On RDNA (gfx11xx/gfx12xx) WMMA only
# accepts 16-bit/int inputs, so a widened (e.g. fp32) tl.dot is lowered to a
# software matmul (~190x amdgcn-stage blowup). There we cast both operands down
# to k's native storage dtype (bf16/fp16) so fast WMMA is used instead.
_CAST_DOT_TO_K_DTYPE = False
if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx1x

    _CAST_DOT_TO_K_DTYPE = on_gfx1x()

_TD_ALLOCATOR_DEVICES: set[torch.device] = set()


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    CAST_DOT_TO_K_DTYPE: tl.constexpr,
    USE_TD: tl.constexpr = False,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))

    if USE_TD:
        # k rows are [T, K] with unit K stride, so the [BT, BK] tile the loop
        # already walks maps 1:1 onto a descriptor block.
        k_desc = tl.make_tensor_descriptor(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            shape=[T, K],
            strides=[Hg * K, 1],
            block_shape=[BT, BK],
        )

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        if USE_TD:
            b_k = k_desc.load([i_t * BT, i_k * BK])
        else:
            p_k = tl.make_block_ptr(
                k + (bos * Hg + i_h // (H // Hg)) * K,
                (T, K),
                (Hg * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        if CAST_DOT_TO_K_DTYPE:
            # RDNA: force operands to k's native dtype so WMMA is used.
            b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))
        else:
            # Keep the promoted precision of the beta-scaled operand (WGMMA/MFMA).
            b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))

    if USE_G:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * exp(b_g_diff)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    if USE_TD:
        A_desc = tl.make_tensor_descriptor(
            A + (bos * H + i_h) * BT,
            shape=[T, BT],
            strides=[BT * H, 1],
            block_shape=[BT, BT],
        )
        A_desc.store([i_t * BT, 0], b_A.to(A.dtype.element_ty))
    else:
        p_A = tl.make_block_ptr(
            A + (bos * H + i_h) * BT,
            (T, BT),
            (BT * H, 1),
            (i_t * BT, 0),
            (BT, BT),
            (1, 0),
        )
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype = torch.float32,
    use_td: bool | None = None,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_indices (torch.Tensor):
            Pre-computed chunk indices. If None and cu_seqlens is provided,
            computed internally. Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)

    # Descriptors need unit inner stride, power-of-two tiles and 16B-aligned
    # bases/rows. In-kernel bases are multiples of K (k) and BT (A), so the
    # per-row byte checks below cover every program's offset.
    k_es, A_es = k.element_size(), A.element_size()
    use_td = (
        use_tensor_descriptor(use_td)
        and k.stride(-1) == 1
        and (BT & (BT - 1)) == 0
        and k.data_ptr() % 16 == 0
        and (K * k_es) % 16 == 0
        and (Hg * K * k_es) % 16 == 0
        and (BT * A_es) % 16 == 0
        and (BT * H * A_es) % 16 == 0
    )
    if use_td and k.device not in _TD_ALLOCATOR_DEVICES:
        set_triton_allocator(k.device)
        _TD_ALLOCATOR_DEVICES.add(k.device)

    chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        CAST_DOT_TO_K_DTYPE=_CAST_DOT_TO_K_DTYPE,
        USE_TD=use_td,
    )
    return A
