# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface


# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L7
def _merge_two_lse(
    lse0: torch.Tensor, lse1: torch.Tensor | None, s_q: int, h_q: int
) -> torch.Tensor:
    if lse1 is None:
        return lse0
    else:
        return torch.logsumexp(
            torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
            dim=0,
        )


# Adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L19
def reference_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns:
    - o: [s_q, h_q, dv]
    - o_fp32: [s_q, h_q, dv]
    - max_logits: [s_q, h_q]
    - lse: [s_q, h_q]

    """
    s_q, h_q, d_qk = q.shape
    s_kv, _, _ = kv.shape
    _, _, topk = indices.shape

    indices = indices.clone().squeeze(1)
    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0).broadcast_to(
            s_q, topk
        ) >= topk_length.unsqueeze(1)  # [s_q, topk]
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= s_kv)  # [s_q, topk]
    indices[invalid_mask] = 0

    q = q.float()
    gathered_kv = (
        kv.index_select(dim=0, index=indices.flatten()).reshape(s_q, topk, d_qk).float()
    )  # [s_q, topk, d_qk]
    P = q @ gathered_kv.transpose(1, 2)  # [s_q, h_q, topk]
    P *= sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)  # [s_q, h_q]
    max_logits = P.max(dim=-1).values  # [s_q, h_q]

    lse_for_o = _merge_two_lse(orig_lse, attn_sink, s_q, h_q)
    if not torch.is_inference_mode_enabled():
        lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float(
        "+inf"
    )  # So that corresponding O will be 0
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., :d_v]  # [s_q, h_q, dv]

    lonely_q_mask = orig_lse == float("-inf")  # [s_q, h_q]
    orig_lse[lonely_q_mask] = float("+inf")
    return (out.to(kv.dtype), out, max_logits, orig_lse)


@pytest.mark.parametrize("device_str", ["xpu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_bf16_triton_sparse_mla(device_str, dtype):
    device = torch.device(device_str)
    s_q = 1
    s_kv = 256
    h_q = 64  # kernel expects multiple of 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128

    torch.random.manual_seed(1234)

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, h_kv, topk), -1, dtype=torch.int32, device=device)
    for t in range(s_q):
        for h in range(h_kv):
            i_i = torch.randperm(max(1, t))[:topk]
            indices[t, h, : len(i_i)] = i_i

    sm_scale = d_qk**-0.5

    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v
    )
    assert out.shape == (s_q, h_q, d_v)
    assert max_logits.shape == (s_q, h_q)
    assert lse.shape == (s_q, h_q)

    ref_out, ref_out_fp32, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v
    )
    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)
    assert torch.allclose(max_logits, ref_max_logits, atol=1e-3, rtol=1e-3)
    assert torch.allclose(lse, ref_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("device_str", ["xpu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_bf16_triton_sparse_mla_masked_chunks(device_str, dtype):
    """Rows whose leading BLOCK_N index entries are all masked must not NaN.

    Regression test: with an -inf running max, a fully-masked leading chunk
    produced re_scale = exp2(-inf - -inf) = NaN, permanently poisoning the
    accumulator even though valid keys followed in later chunks.
    """
    device = torch.device(device_str)
    s_q = 3
    s_kv = 256
    h_q = 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128  # 8 chunks of BLOCK_N=16

    torch.random.manual_seed(1234)

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, h_kv, topk), -1, dtype=torch.int32, device=device)
    # row 0: valid keys only in chunks 1-2 -> leading AND trailing masked chunks
    indices[0, 0, 16:48] = torch.arange(32, dtype=torch.int32, device=device)
    # row 1: fully valid
    indices[1, 0, :] = torch.arange(topk, dtype=torch.int32, device=device)
    # row 2: no valid key at all

    sm_scale = d_qk**-0.5

    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v
    )
    assert out.isfinite().all()

    ref_out, _, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v
    )
    assert torch.allclose(out[:2], ref_out[:2], atol=1e-2, rtol=1e-2)
    assert torch.allclose(max_logits[:2], ref_max_logits[:2], atol=1e-3, rtol=1e-3)
    assert torch.allclose(lse[:2], ref_lse[:2], atol=1e-3, rtol=1e-3)
    # A row with no valid key yields zeros (the reference's convention); its
    # lse/max_logits are large-negative finite rather than the reference's
    # +inf/-inf placeholders, so only the output is compared here.
    assert torch.allclose(out[2], torch.zeros_like(out[2]))


# DSA sparse indexer FP8 MQA logits: native (vllm-xpu-kernels) vs Triton.
requires_xpu = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")

_HEADS, _DIM, _BLOCK_SIZE = 64, 128, 64


def _native_usable(name: str) -> bool:
    import vllm._xpu_ops as xpu_ops_mod

    return xpu_ops_mod.kernel_usable(name, *xpu_ops_mod._SPARSE_INDEXER_PROBES[name])


@pytest.fixture
def indexer_backend(monkeypatch):
    import vllm._xpu_ops as xpu_ops_mod

    def set_backend(backend: str) -> None:
        monkeypatch.setattr(xpu_ops_mod, "_sparse_indexer_backend", backend)

    return set_backend


def _mqa_inputs(num_rows: int, num_keys: int):
    """Random FP8 q/k with power-of-two K scales and ragged [ks, ke) rows;
    with several rows, the first is empty and the last spans all keys."""
    dev = "xpu"
    fp8 = torch.float8_e4m3fn
    q = torch.randn(num_rows, _HEADS, _DIM, device=dev).to(fp8)
    k = torch.randn(num_keys, _DIM, device=dev).to(fp8)
    k_scale = torch.pow(2.0, -torch.randint(0, 4, (num_keys,), device=dev).float())
    weights = torch.randn(num_rows, _HEADS, device=dev)
    ks = torch.randint(0, num_keys, (num_rows,), device=dev)
    ke = (ks + torch.randint(1, num_keys, (num_rows,), device=dev)).clamp(max=num_keys)
    if num_rows > 1:
        ks[0], ke[0] = 5, 5
        ks[-1], ke[-1] = 0, num_keys
    return q, k, k_scale, weights, ks.int(), ke.int()


def _logits_bound(q, k, k_scale, weights, unit: float) -> torch.Tensor:
    """Elementwise error bound for sum_h relu(q_h . k * s) * w_h.

    FP8 products are exact in fp32, so an fp32 implementation only adds
    rounding over D + H + 2 operations per term (gamma_n, Higham); `unit` adds
    one more relative rounding per head term (e.g. a bf16-rounded q . k).
    q [M, H, D], k [N, D] or per row [M, N, D], k_scale [N] or [M, N],
    weights [M, H].
    """
    eq = "mhd,nd->mhn" if k.dim() == 2 else "mhd,mnd->mhn"
    dot = torch.einsum(eq, q.cpu().double(), k.cpu().double())
    terms = dot.abs() * k_scale.cpu().double()[..., None, :]
    terms = terms * weights.cpu().double().abs()[..., None]
    gamma = (q.shape[-1] + q.shape[1] + 2) * 2.0**-24
    return (gamma + unit) * terms.sum(dim=1)


def _assert_logits_close(got, ref, bound) -> None:
    got, ref = got.cpu(), ref.cpu()
    assert got.dtype == torch.float32 and got.shape == ref.shape
    masked = ref == float("-inf")
    assert torch.equal(got == float("-inf"), masked)
    err = (got.double() - ref.double()).abs()[~masked]
    assert (err <= bound[~masked]).all(), f"max err {err.max().item():.3e}"


def _assert_same_topk(got, ref, k: int = 64) -> None:
    k = min(k, ref.shape[1])
    got_vals, got_idx = got.topk(k, dim=-1)
    ref_vals, ref_idx = ref.topk(k, dim=-1)
    for row in range(ref.shape[0]):
        got_set = set(got_idx[row][got_vals[row].isfinite()].tolist())
        assert got_set == set(ref_idx[row][ref_vals[row].isfinite()].tolist())


@requires_xpu
@pytest.mark.parametrize("num_rows,num_keys", [(1, 64), (40, 300), (96, 1100)])
def test_fp8_mqa_logits_triton_matches_torch_ref(num_rows, num_keys):
    from vllm.v1.attention.ops.dsa_mqa_logits_triton import fp8_mqa_logits_triton
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_mqa_logits_torch

    torch.manual_seed(0)
    q, k, k_scale, weights, ks, ke = _mqa_inputs(num_rows, num_keys)
    got = fp8_mqa_logits_triton(q, (k, k_scale), weights, ks, ke)
    ref = fp8_mqa_logits_torch(q, (k, k_scale), weights, ks, ke)
    # The reference rounds each q . k to bf16 (unit roundoff 2^-8).
    bound = _logits_bound(q, k, k_scale, weights, unit=2.0**-8)
    _assert_logits_close(got, ref, bound)


@requires_xpu
@pytest.mark.parametrize("num_rows,num_keys", [(1, 64), (40, 300), (96, 1100)])
def test_fp8_mqa_logits_native_vs_triton(indexer_backend, num_rows, num_keys):
    from vllm._xpu_ops import _xpu_fp8_mqa_logits_impl

    if not _native_usable("_xpu_C.fp8_mqa_logits"):
        pytest.skip("native fp8_mqa_logits is not usable")
    torch.manual_seed(0)
    inputs = _mqa_inputs(num_rows, num_keys)
    indexer_backend("native")
    native = _xpu_fp8_mqa_logits_impl(*inputs)
    indexer_backend("triton")
    triton_out = _xpu_fp8_mqa_logits_impl(*inputs)
    q, k, k_scale, weights, _, _ = inputs
    _assert_logits_close(triton_out, native, _logits_bound(q, k, k_scale, weights, 0.0))
    _assert_same_topk(triton_out, native)


def _paged_inputs(next_n: int):
    """Block-flat FP8 indexer K cache ([values | fp32 scales] per block) with
    shuffled pages, ragged context lengths and power-of-two K scales."""
    dev = "xpu"
    fp8 = torch.float8_e4m3fn
    context_lens = torch.tensor([next_n, 150, 257], dtype=torch.int32, device=dev)
    batch, max_blocks = context_lens.numel(), 5
    num_blocks = batch * max_blocks
    q = torch.randn(batch, next_n, _HEADS, _DIM, device=dev).to(fp8)
    values = torch.randn(num_blocks, _BLOCK_SIZE * _DIM, device=dev).to(fp8)
    scales = torch.pow(
        2.0, -torch.randint(0, 4, (num_blocks, _BLOCK_SIZE), device=dev).float()
    )
    kv_cache = torch.cat(
        [values.view(torch.uint8), scales.view(torch.uint8)], dim=1
    ).view(num_blocks, _BLOCK_SIZE, 1, _DIM + 4)
    weights = torch.randn(batch * next_n, _HEADS, device=dev)
    block_tables = torch.randperm(num_blocks, device=dev).int().view(batch, -1)
    max_model_len = max_blocks * _BLOCK_SIZE
    paged = (q, kv_cache, weights, context_lens, block_tables, max_model_len)
    # Error bound over each row's dense K: [B * next_n, max_model_len, D].
    pages = block_tables.cpu().repeat_interleave(next_n, dim=0)
    k = values.cpu().float().view(num_blocks, _BLOCK_SIZE, _DIM)[pages]
    bound = _logits_bound(
        q.view(batch * next_n, _HEADS, _DIM),
        k.view(batch * next_n, max_model_len, _DIM),
        scales.cpu()[pages].view(batch * next_n, max_model_len),
        weights,
        unit=0.0,
    )
    return paged, bound


@requires_xpu
@pytest.mark.parametrize("next_n", [1, 2, 4])
def test_fp8_paged_mqa_logits_triton_matches_torch_ref(indexer_backend, next_n):
    from vllm._xpu_ops import _xpu_fp8_paged_mqa_logits_impl
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        fp8_paged_mqa_logits_torch,
    )

    torch.manual_seed(0)
    (q, kv_cache, weights, ctx, block_tables, max_len), bound = _paged_inputs(next_n)
    indexer_backend("triton")
    got = _xpu_fp8_paged_mqa_logits_impl(
        q, kv_cache, weights, ctx, block_tables, None, max_len
    )
    ref = fp8_paged_mqa_logits_torch(q, kv_cache, weights, ctx, block_tables, max_len)
    _assert_logits_close(got, ref, bound)


@requires_xpu
@pytest.mark.parametrize("next_n", [1, 2, 4])
def test_fp8_paged_mqa_logits_native_vs_triton(indexer_backend, next_n):
    from vllm._xpu_ops import _xpu_fp8_paged_mqa_logits_impl

    if not _native_usable("_xpu_C.fp8_paged_mqa_logits"):
        pytest.skip("native fp8_paged_mqa_logits is not usable")
    torch.manual_seed(0)
    (q, kv_cache, weights, ctx, block_tables, max_len), bound = _paged_inputs(next_n)
    args = (q, kv_cache, weights, ctx, block_tables, None, max_len)
    indexer_backend("native")
    native = _xpu_fp8_paged_mqa_logits_impl(*args)
    indexer_backend("triton")
    triton_out = _xpu_fp8_paged_mqa_logits_impl(*args)
    _assert_logits_close(triton_out, native, bound)
    _assert_same_topk(triton_out, native)


@requires_xpu
def test_sparse_indexer_auto_falls_back_to_triton(indexer_backend, monkeypatch):
    """With "auto" and the native kernels marked unusable, the custom ops run
    the Triton kernels."""
    import vllm._xpu_ops  # noqa: F401 registers torch.ops.vllm.xpu_*mqa_logits
    from vllm.utils.kernel_probe import set_kernel_usable
    from vllm.v1.attention.ops import dsa_mqa_logits_triton as mod

    calls: list[str] = []

    def counted(name: str):
        fn = getattr(mod, name)

        def wrapper(*args, **kwargs):
            calls.append(name)
            return fn(*args, **kwargs)

        return wrapper

    for name in ("fp8_mqa_logits_triton", "fp8_paged_mqa_logits_triton"):
        monkeypatch.setattr(mod, name, counted(name))
    indexer_backend("auto")
    torch.manual_seed(0)
    try:
        for name in ("_xpu_C.fp8_mqa_logits", "_xpu_C.fp8_paged_mqa_logits"):
            set_kernel_usable(name, False)
        torch.ops.vllm.xpu_fp8_mqa_logits(*_mqa_inputs(8, 128))
        (q, kv_cache, weights, ctx, block_tables, max_len), _ = _paged_inputs(2)
        torch.ops.vllm.xpu_fp8_paged_mqa_logits(
            q, kv_cache, weights, ctx, block_tables, None, max_len
        )
    finally:
        for name in ("_xpu_C.fp8_mqa_logits", "_xpu_C.fp8_paged_mqa_logits"):
            set_kernel_usable(name, None)
    assert calls == ["fp8_mqa_logits_triton", "fp8_paged_mqa_logits_triton"]


@requires_xpu
def test_sparse_indexer_explicit_backends(indexer_backend):
    from vllm._xpu_ops import _use_native_indexer_op

    indexer_backend("triton")
    assert not _use_native_indexer_op("_xpu_C.fp8_mqa_logits")
    indexer_backend("native")
    with pytest.raises(RuntimeError, match="sparse_indexer_backend='native'"):
        _use_native_indexer_op("_xpu_C.op_missing_from_this_build")


# DSA indexer K cache write/gather: native (vllm-xpu-kernels) vs Triton.
_CACHE_BLOCKS = 8


def _indexer_k_cache_inputs(num_tokens: int = 200):
    """bf16 K rows, shuffled int64 slots with some -1 (skipped) and a cache
    of random bytes (to check that only the written slots change)."""
    dev = "xpu"
    k = 3 * torch.randn(num_tokens, _DIM, device=dev, dtype=torch.bfloat16)
    slots = torch.randperm(_CACHE_BLOCKS * _BLOCK_SIZE, device=dev)[:num_tokens]
    slots[::7] = -1
    kv_cache = torch.randint(
        0, 256, (_CACHE_BLOCKS, _BLOCK_SIZE, _DIM + 4), dtype=torch.uint8, device=dev
    )
    return k, kv_cache, slots


def _cache_regions(kv_cache):
    """[block, token] views of the value bytes and the fp32 scale bytes."""
    flat = kv_cache.view(kv_cache.shape[0], -1)
    values = flat[:, : _BLOCK_SIZE * _DIM].view(-1, _BLOCK_SIZE, _DIM)
    scales = flat[:, _BLOCK_SIZE * _DIM :].view(-1, _BLOCK_SIZE, 4)
    return values, scales


def _indexer_k_cache_ref(k, kv_cache, slots, scale_fmt):
    """Torch reference of the cache write: per-token amax / 448 scale (rounded
    up to a power of two for UE8M0), flat token rows, fp32 scales."""
    out = kv_cache.clone()
    valid = slots >= 0
    k, slots = k[valid].float(), slots[valid]
    scale = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / 448.0
    if scale_fmt == "ue8m0":
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    values, scales = _cache_regions(out)
    blocks, offsets = slots // _BLOCK_SIZE, slots % _BLOCK_SIZE
    values[blocks, offsets] = (k / scale).to(torch.float8_e4m3fn).view(torch.uint8)
    scales[blocks, offsets] = scale.view(torch.uint8)
    return out


@requires_xpu
@pytest.mark.parametrize("scale_fmt", ["ue8m0", "fp32"])
def test_indexer_k_cache_write_triton_matches_torch_ref(indexer_backend, scale_fmt):
    from vllm._xpu_ops import indexer_k_quant_and_cache

    torch.manual_seed(0)
    k, kv_cache, slots = _indexer_k_cache_inputs()
    ref = _indexer_k_cache_ref(k, kv_cache, slots, scale_fmt)
    indexer_backend("triton")
    indexer_k_quant_and_cache(k, kv_cache, slots, _DIM, scale_fmt)
    assert torch.equal(kv_cache, ref)


@requires_xpu
@pytest.mark.parametrize("scale_fmt", ["ue8m0", "fp32"])
def test_indexer_k_cache_write_native_vs_triton(indexer_backend, scale_fmt):
    from vllm._xpu_ops import indexer_k_quant_and_cache

    if not _native_usable("_C_cache_ops.indexer_k_quant_and_cache"):
        pytest.skip("native indexer_k_quant_and_cache is not usable")
    torch.manual_seed(0)
    k, native, slots = _indexer_k_cache_inputs()
    triton_cache = native.clone()
    indexer_backend("native")
    indexer_k_quant_and_cache(k, native, slots, _DIM, scale_fmt)
    indexer_backend("triton")
    indexer_k_quant_and_cache(k, triton_cache, slots, _DIM, scale_fmt)
    assert torch.equal(triton_cache, native)


def _gather_inputs():
    """Ragged sequences over shuffled pages of a random-byte cache."""
    dev = "xpu"
    seq_lens = [1, 69, 130, 64]
    max_blocks = 3
    kv_cache = torch.randint(
        0,
        256,
        (_CACHE_BLOCKS * 2, _BLOCK_SIZE, _DIM + 4),
        dtype=torch.uint8,
        device=dev,
    )
    pages = torch.randperm(kv_cache.shape[0], device=dev)[: len(seq_lens) * max_blocks]
    block_table = pages.int().view(len(seq_lens), max_blocks)
    cu_seq_lens = torch.tensor([0, *seq_lens], device=dev).cumsum(0).int()
    return kv_cache, block_table, cu_seq_lens


def _gather(kv_cache, block_table, cu_seq_lens):
    from vllm._xpu_ops import cp_gather_indexer_k_quant_cache

    num_tokens = int(cu_seq_lens[-1])
    dst_k = torch.empty(
        num_tokens, _DIM, dtype=torch.float8_e4m3fn, device=kv_cache.device
    )
    dst_scale = torch.empty(num_tokens, 4, dtype=torch.uint8, device=kv_cache.device)
    cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    )
    return dst_k.view(torch.uint8), dst_scale


@requires_xpu
def test_indexer_k_cache_gather_native_vs_triton(indexer_backend):
    if not _native_usable("_C_cache_ops.cp_gather_indexer_k_quant_cache"):
        pytest.skip("native cp_gather_indexer_k_quant_cache is not usable")
    torch.manual_seed(0)
    inputs = _gather_inputs()
    indexer_backend("native")
    native_k, native_scale = _gather(*inputs)
    indexer_backend("triton")
    triton_k, triton_scale = _gather(*inputs)
    assert torch.equal(triton_k, native_k)
    assert torch.equal(triton_scale, native_scale)


@requires_xpu
@pytest.mark.parametrize("scale_fmt", ["ue8m0", "fp32"])
def test_indexer_k_cache_triton_round_trip(indexer_backend, scale_fmt):
    """Triton write then Triton gather returns the stored bytes, and they
    dequantize to K within FP8 E4M3 rounding."""
    from vllm._xpu_ops import indexer_k_quant_and_cache

    torch.manual_seed(0)
    indexer_backend("triton")
    kv_cache, block_table, cu_seq_lens = _gather_inputs()
    seq_lens = (cu_seq_lens[1:] - cu_seq_lens[:-1]).tolist()
    # Slot of every token of every sequence, in gather order.
    slots = torch.cat(
        [
            block_table[i, torch.arange(n, device="xpu") // _BLOCK_SIZE] * _BLOCK_SIZE
            + torch.arange(n, device="xpu") % _BLOCK_SIZE
            for i, n in enumerate(seq_lens)
        ]
    )
    k = 3 * torch.randn(slots.numel(), _DIM, device="xpu", dtype=torch.bfloat16)
    indexer_k_quant_and_cache(k, kv_cache, slots, _DIM, scale_fmt)
    k_bytes, scale_bytes = _gather(kv_cache, block_table, cu_seq_lens)

    values, scales = _cache_regions(kv_cache)
    blocks, offsets = slots // _BLOCK_SIZE, slots % _BLOCK_SIZE
    assert torch.equal(k_bytes, values[blocks, offsets])
    assert torch.equal(scale_bytes, scales[blocks, offsets])

    scale = scale_bytes.view(torch.float32)
    k_deq = k_bytes.view(torch.float8_e4m3fn).float() * scale
    # E4M3 rounds to 2^-4 relative, or 2^-10 absolute (in units of the scale)
    # in the subnormal range; 2^-20 covers the fp32 divide and multiply.
    bound = (2.0**-4 + 2.0**-20) * k.float().abs() + 2.0**-10 * scale
    assert ((k_deq - k.float()).abs() <= bound).all()


@requires_xpu
def test_indexer_k_cache_auto_falls_back_to_triton(indexer_backend, monkeypatch):
    """With "auto" and the native cache kernels marked unusable, the
    vllm._custom_ops wrappers run the Triton kernels."""
    from vllm import _custom_ops as ops
    from vllm.utils.kernel_probe import set_kernel_usable
    from vllm.v1.attention.ops import dsa_indexer_cache_triton as mod

    names = (
        "_C_cache_ops.indexer_k_quant_and_cache",
        "_C_cache_ops.cp_gather_indexer_k_quant_cache",
    )
    calls: list[str] = []

    def counted(name: str):
        fn = getattr(mod, name)

        def wrapper(*args, **kwargs):
            calls.append(name)
            return fn(*args, **kwargs)

        return wrapper

    for name in (
        "indexer_k_quant_and_cache_triton",
        "cp_gather_indexer_k_quant_cache_triton",
    ):
        monkeypatch.setattr(mod, name, counted(name))
    indexer_backend("auto")
    torch.manual_seed(0)
    try:
        for name in names:
            set_kernel_usable(name, False)
        k, kv_cache, slots = _indexer_k_cache_inputs()
        ops.indexer_k_quant_and_cache(k, kv_cache, slots, _DIM, "ue8m0")
        kv_cache, block_table, cu_seq_lens = _gather_inputs()
        num_tokens = int(cu_seq_lens[-1])
        ops.cp_gather_indexer_k_quant_cache(
            kv_cache,
            kv_cache.new_empty(num_tokens, _DIM).view(torch.float8_e4m3fn),
            kv_cache.new_empty(num_tokens, 4),
            block_table,
            cu_seq_lens,
        )
    finally:
        for name in names:
            set_kernel_usable(name, None)
    assert calls == [
        "indexer_k_quant_and_cache_triton",
        "cp_gather_indexer_k_quant_cache_triton",
    ]


@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
@torch.inference_mode()
def test_dequant_gather_ds_mla_rows_matches_upconvert():
    """Top-k-only dequantization of fp8_ds_mla rows reads the same bytes with
    the same math as the SYCL cp_gather_and_upconvert_fp8_kv_cache op, and
    -1 slots become zeros."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops.xpu_mla_sparse import (
        DS_MLA_ENTRY_BYTES,
        dequant_gather_ds_mla_rows,
    )

    device = torch.device("xpu")
    torch.random.manual_seed(0)
    num_blocks, block_size = 8, 64
    num_slots = num_blocks * block_size
    kv_c = torch.randn(num_slots, 512, dtype=torch.bfloat16, device=device) * 4
    k_pe = torch.randn(num_slots, 64, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(
        num_blocks, block_size, DS_MLA_ENTRY_BYTES, dtype=torch.uint8, device=device
    )
    ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        cache,
        torch.arange(num_slots, dtype=torch.int64, device=device),
        kv_cache_dtype="fp8_ds_mla",
        scale=torch.ones(1, dtype=torch.float32, device=device),
    )
    full = torch.empty(num_slots, 576, dtype=torch.bfloat16, device=device)
    ops.cp_gather_and_upconvert_fp8_kv_cache(
        src_cache=cache,
        dst=full,
        block_table=torch.arange(num_blocks, dtype=torch.int32, device=device)[None],
        workspace_starts=torch.zeros(1, dtype=torch.int32, device=device),
        batch_size=1,
    )

    rows = torch.randint(0, num_slots, (5, 96), dtype=torch.int32, device=device)
    rows[:, -7:] = -1
    got = dequant_gather_ds_mla_rows(cache.view(num_slots, DS_MLA_ENTRY_BYTES), rows)
    expected = full[rows.clamp(min=0).flatten().long()]
    expected[rows.flatten() < 0] = 0
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    # fp8 storage stays close to the BF16 values it was written from.
    ref = torch.cat([kv_c, k_pe], dim=-1)[rows.clamp(min=0).flatten().long()]
    valid = rows.flatten() >= 0
    rel = (got[valid].float() - ref[valid].float()).norm() / ref[valid].float().norm()
    assert rel < 0.05


def _mla_cache_case(device, kv_cache_dtype, pe_dim, num_blocks=4, block_size=16):
    """Inputs for an MLA KV cache write: kv_c / k_pe have two more (padding)
    rows than slot_mapping, slot_mapping has padded (-1) tokens, and the cache
    starts as 0xFF bytes so that untouched rows are visible."""
    from vllm.v1.attention.ops.xpu_mla_sparse import DS_MLA_ENTRY_BYTES

    num_slots = num_blocks * block_size
    slots = torch.randperm(num_slots)[:20]
    slots[[3, 11, 19]] = -1
    kv_c = torch.randn(22, 512, dtype=torch.bfloat16) * 4
    kv_c[5] = 0  # all-zero tiles
    kv_c[6, :128] *= 1e-3  # one small tile
    k_pe = torch.randn(22, pe_dim, dtype=torch.bfloat16)
    if kv_cache_dtype == "fp8_ds_mla":
        cache = torch.full(
            (num_blocks, block_size, DS_MLA_ENTRY_BYTES), 0xFF, dtype=torch.uint8
        )
    else:
        cache = torch.full((num_blocks, block_size, 512 + pe_dim), float("nan"))
        cache = cache.to(torch.bfloat16)
    return [t.to(device) for t in (kv_c, k_pe, cache, slots)]


@pytest.mark.parametrize("pe_dim", [64, 0])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_ds_mla"])
@torch.inference_mode()
def test_concat_and_cache_mla_torch_reference(kv_cache_dtype, pe_dim):
    """Layout of the torch MLA KV cache writer, on any device: unquantized rows
    are [kv_c, k_pe]; fp8_ds_mla rows hold e4m3 NoPE with fp32 tile scales
    amax / 448 and the BF16 RoPE (zeros when rope-free). Padded tokens and
    padding rows of kv_c write nothing."""
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops.xpu_mla_sparse import (
        concat_and_cache_mla_torch,
        ds_mla_dequant_torch,
    )

    device = current_platform.device_type
    torch.random.manual_seed(0)
    kv_c, k_pe, cache, slots = _mla_cache_case(device, kv_cache_dtype, pe_dim)
    before = cache.clone()
    concat_and_cache_mla_torch(kv_c, k_pe, cache, slots, kv_cache_dtype)

    entry = cache.shape[-1]
    rows = cache.view(-1, entry)
    valid = slots >= 0
    written = slots[valid]
    untouched = torch.ones(rows.shape[0], dtype=torch.bool, device=device)
    untouched[written] = False
    assert torch.equal(
        rows[untouched].view(torch.uint8),
        before.view(-1, entry)[untouched].view(torch.uint8),
    )
    x, pe = kv_c[: slots.shape[0]][valid], k_pe[: slots.shape[0]][valid]
    got = rows[written]
    if kv_cache_dtype == "auto":
        assert torch.equal(got, torch.cat([x, pe], dim=-1))
        return

    tiles = x.float().view(-1, 4, 128)
    expected_scales = (tiles.abs().amax(-1) / 448.0).clamp(
        min=torch.finfo(torch.float32).tiny
    )
    assert torch.equal(got[:, 512:528].view(torch.float32), expected_scales)
    expected_pe = pe if pe_dim else torch.zeros_like(x[:, :64])
    assert torch.equal(got[:, 528:].view(torch.bfloat16), expected_pe)
    # Round-to-nearest e4m3 (3 mantissa bits): at most half an ulp, i.e.
    # 2^-4 relative for normal values and 2^-10 * scale for subnormal ones.
    deq = ds_mla_dequant_torch(got).float()[:, :512].view(-1, 4, 128)
    bound = (tiles.abs() * 2**-4).maximum(expected_scales.unsqueeze(-1) * 2**-10)
    # The dequantized value is rounded to BF16 (2^-8 relative) on top.
    assert ((deq - tiles).abs() <= bound + deq.abs() * 2**-8).all()


@pytest.mark.parametrize("with_seq_starts", [False, True])
@torch.inference_mode()
def test_cp_gather_and_upconvert_fp8_kv_cache_torch_reference(with_seq_starts):
    """The torch fp8_ds_mla gather reads each request's context rows through
    its block table into its workspace range, on any device."""
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops.xpu_mla_sparse import (
        cp_gather_and_upconvert_fp8_kv_cache_torch,
        ds_mla_dequant_torch,
    )

    device = current_platform.device_type
    torch.random.manual_seed(0)
    cache = _full_ds_mla_cache(device)
    num_blocks, block_size = cache.shape[:2]
    block_table, starts, seq_starts, lens = _gather_case(
        num_blocks, block_size, with_seq_starts
    )
    total = int(sum(lens))
    dst = torch.full((total, 576), float("nan"), dtype=torch.bfloat16)
    dst = dst.to(device)
    cp_gather_and_upconvert_fp8_kv_cache_torch(
        cache,
        dst,
        block_table.to(device),
        starts.to(device),
        len(lens),
        None if seq_starts is None else seq_starts.to(device),
    )

    entries = cache.view(-1, cache.shape[-1])
    for r, n in enumerate(lens):
        first = 0 if seq_starts is None else int(seq_starts[r])
        for i in range(n):
            token = first + i
            slot = int(block_table[r, token // block_size]) * block_size
            slot += token % block_size
            expected = ds_mla_dequant_torch(entries[slot : slot + 1])[0]
            assert torch.equal(dst[int(starts[r]) + i], expected)


def _full_ds_mla_cache(device, num_blocks=4, block_size=16):
    """An fp8_ds_mla cache with every slot written by the torch writer."""
    from vllm.v1.attention.ops.xpu_mla_sparse import (
        DS_MLA_ENTRY_BYTES,
        concat_and_cache_mla_torch,
    )

    num_slots = num_blocks * block_size
    kv_c = (torch.randn(num_slots, 512, dtype=torch.bfloat16) * 4).to(device)
    k_pe = torch.randn(num_slots, 64, dtype=torch.bfloat16).to(device)
    cache = torch.zeros(
        num_blocks, block_size, DS_MLA_ENTRY_BYTES, dtype=torch.uint8, device=device
    )
    slots = torch.arange(num_slots, device=device)
    concat_and_cache_mla_torch(kv_c, k_pe, cache, slots, "fp8_ds_mla")
    return cache


def _gather_case(num_blocks, block_size, with_seq_starts):
    """Three requests over shuffled blocks: block_table, workspace_starts,
    seq_starts (or None) and the number of rows gathered per request."""
    block_table = torch.randperm(num_blocks).to(torch.int32).view(2, 2)
    block_table = torch.cat([block_table, block_table.flip(0)[:1]])
    lens = [2 * block_size - 3, 5, block_size + 1]
    seq_starts = None
    if with_seq_starts:
        seq_starts = torch.tensor([0, block_size - 2, 3], dtype=torch.int32)
    starts = torch.tensor([0, lens[0], lens[0] + lens[1]], dtype=torch.int32)
    return block_table, starts, seq_starts, lens


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required")
@pytest.mark.parametrize("pe_dim", [64, 0])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_ds_mla"])
@torch.inference_mode()
def test_concat_and_cache_mla_torch_matches_sycl(kv_cache_dtype, pe_dim):
    """The torch MLA KV cache writer is byte-identical to the SYCL op."""
    from vllm.v1.attention.ops.xpu_mla_sparse import concat_and_cache_mla_torch

    torch.random.manual_seed(0)
    kv_c, k_pe, cache, slots = _mla_cache_case("xpu", kv_cache_dtype, pe_dim)
    expected = cache.clone()
    try:
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            expected,
            slots,
            kv_cache_dtype,
            torch.ones(1, dtype=torch.float32, device="xpu"),
            False,
        )
    except RuntimeError as e:
        if kv_cache_dtype == "fp8_ds_mla" and pe_dim == 0:
            pytest.skip(f"SYCL concat_and_cache_mla rejects pe_dim=0: {e}")
        raise
    concat_and_cache_mla_torch(kv_c, k_pe, cache, slots, kv_cache_dtype)
    assert torch.equal(cache.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required")
@pytest.mark.parametrize("with_seq_starts", [False, True])
@torch.inference_mode()
def test_cp_gather_and_upconvert_fp8_kv_cache_torch_matches_sycl(with_seq_starts):
    """The torch fp8_ds_mla gather is byte-identical to the SYCL op."""
    from vllm.v1.attention.ops.xpu_mla_sparse import (
        cp_gather_and_upconvert_fp8_kv_cache_torch,
    )

    torch.random.manual_seed(0)
    cache = _full_ds_mla_cache("xpu")
    num_blocks, block_size = cache.shape[:2]
    block_table, starts, seq_starts, lens = _gather_case(
        num_blocks, block_size, with_seq_starts
    )
    args = (
        cache,
        block_table.to("xpu"),
        starts.to("xpu"),
        len(lens),
        None if seq_starts is None else seq_starts.to("xpu"),
    )
    expected = torch.empty(sum(lens), 576, dtype=torch.bfloat16, device="xpu")
    got = torch.empty_like(expected)
    torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(
        args[0], expected, *args[1:], None, None, None
    )
    cp_gather_and_upconvert_fp8_kv_cache_torch(args[0], got, *args[1:])
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required")
@pytest.mark.parametrize(
    "op", ["concat_and_cache_mla", "cp_gather_and_upconvert_fp8_kv_cache"]
)
@torch.inference_mode()
def test_xpu_mla_kv_cache_forced_fallback(op, monkeypatch):
    """With the SYCL op marked unusable, the _custom_ops wrapper runs the
    torch implementation."""
    from vllm import _custom_ops as ops
    from vllm.utils.kernel_probe import set_kernel_usable
    from vllm.v1.attention.ops import xpu_mla_sparse

    torch.random.manual_seed(0)
    kv_c, k_pe, cache, slots = _mla_cache_case("xpu", "fp8_ds_mla", 64)
    torch_impl = getattr(xpu_mla_sparse, f"{op}_torch")
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return torch_impl(*args, **kwargs)

    monkeypatch.setattr(xpu_mla_sparse, f"{op}_torch", spy)
    name = (
        "_C.concat_and_cache_mla[fp8_ds_mla]"
        if op == "concat_and_cache_mla"
        else f"_C.{op}"
    )
    set_kernel_usable(name, False)
    try:
        if op == "concat_and_cache_mla":
            expected = cache.clone()
            torch_impl(kv_c, k_pe, expected, slots, "fp8_ds_mla")
            ops.concat_and_cache_mla(
                kv_c, k_pe, cache, slots, "fp8_ds_mla", torch.ones(1, device="xpu")
            )
            got = cache
        else:
            cache = _full_ds_mla_cache("xpu")
            block_table, starts, _, lens = _gather_case(4, 16, False)
            args = (cache, block_table.to("xpu"), starts.to("xpu"), len(lens))
            expected = torch.empty(sum(lens), 576, dtype=torch.bfloat16, device="xpu")
            got = torch.empty_like(expected)
            torch_impl(args[0], expected, *args[1:])
            ops.cp_gather_and_upconvert_fp8_kv_cache(args[0], got, *args[1:])
    finally:
        set_kernel_usable(name, None)
    assert len(calls) == 1
    assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))


def test_xpu_mla_kv_cache_backend_option(monkeypatch):
    """kernel_config.sparse_indexer_backend also selects the XPU MLA KV cache
    ops: "triton" uses torch without probing, "native" fails loudly when the
    compiled op is unusable, and formats without a torch implementation
    always use the compiled op."""
    from types import SimpleNamespace

    import vllm.config
    from vllm.utils import kernel_probe
    from vllm.v1.attention.ops import xpu_mla_sparse

    real_kernel_usable = kernel_probe.kernel_usable

    def no_probe(*args, **kwargs):
        raise AssertionError("probed")

    config = SimpleNamespace(
        kernel_config=SimpleNamespace(sparse_indexer_backend="triton")
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(kernel_probe, "kernel_usable", no_probe)
    monkeypatch.setattr(xpu_mla_sparse, "_mla_kv_cache_backend", "auto")
    xpu_mla_sparse.prepare_xpu_mla_kv_cache_ops("fp8_ds_mla", 0)
    assert xpu_mla_sparse._mla_kv_cache_backend == "triton"
    use_native = xpu_mla_sparse.xpu_mla_kv_cache_use_native
    assert not use_native("concat_and_cache_mla", "auto", 64)
    assert not use_native("concat_and_cache_mla", "fp8_ds_mla", 64)
    assert not use_native("cp_gather_and_upconvert_fp8_kv_cache")
    assert use_native("concat_and_cache_mla", "fp8", 64)

    monkeypatch.setattr(kernel_probe, "kernel_usable", real_kernel_usable)
    monkeypatch.setattr(xpu_mla_sparse, "_mla_kv_cache_backend", "native")
    name = "_C.concat_and_cache_mla[fp8_ds_mla,rope_free]"
    kernel_probe.set_kernel_usable(name, False)
    try:
        with pytest.raises(RuntimeError, match="sparse_indexer_backend='native'"):
            use_native("concat_and_cache_mla", "fp8_ds_mla", 0)
    finally:
        kernel_probe.set_kernel_usable(name, None)
