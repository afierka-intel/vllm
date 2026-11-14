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
