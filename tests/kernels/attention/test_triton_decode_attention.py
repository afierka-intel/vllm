# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops import triton_decode_attention as tda
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

DEVICE_TYPE = current_platform.device_type


@pytest.mark.parametrize("B", [3, 5])
@pytest.mark.parametrize("L", [1027, 1025])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D_QK", [128, 192, 576])
@pytest.mark.parametrize("D_V", [128, 512])
@pytest.mark.parametrize("CACHE_SIZE", [16384])
@pytest.mark.parametrize("PAGE_SIZE", [1, 16])
def test_decode_attention(B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE):
    assert CACHE_SIZE % PAGE_SIZE == 0
    dtype = torch.bfloat16
    seq_len = L  # This represents the number of tokens already in the sequence
    sm_scale = 1.0 / (D_QK**0.5)
    num_kv_splits = 8

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.randint(
        0, CACHE_SIZE // PAGE_SIZE, (B, num_pages_per_batch, 1), device=DEVICE_TYPE
    )
    req_to_token = req_to_page * PAGE_SIZE
    req_to_token = req_to_token.expand(B, num_pages_per_batch, PAGE_SIZE)
    req_to_token = req_to_token + torch.arange(PAGE_SIZE, device=DEVICE_TYPE).view(
        1, 1, -1
    )
    req_to_token = req_to_token.view(B, -1)
    req_to_token = req_to_token[:, :seq_len].contiguous()

    # q represents the new token being generated, one per batch
    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE)

    # k_buffer and v_buffer represent all previous tokens
    # Page size is 1.
    k_buffer = torch.randn(CACHE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE)
    v_buffer = torch.randn(CACHE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE)

    # o will have the same shape as q
    o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)

    lse = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)

    b_seq_len = torch.full((B,), seq_len, device=DEVICE_TYPE)

    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1),
        dtype=torch.float32,
        device=DEVICE_TYPE,
    )

    # Call the original implementation.
    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o,
        lse,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
    )

    # Page size can be larger than 1.
    k_buffer = k_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
    v_buffer = v_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)

    o1 = torch.zeros_like(o)
    lse1 = torch.zeros_like(lse)

    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o1,
        lse1,
        req_to_page,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        PAGE_SIZE,
    )

    assert torch.allclose(o, o1)


def _quantize_to_fp8(tensor: torch.Tensor):
    """Quantize a BF16 tensor to FP8 e4m3fn with per-tensor scale.

    Returns (fp8_tensor, scale) where:
        fp8_tensor ≈ tensor / scale  (stored as float8_e4m3fn)
        tensor ≈ fp8_tensor.to(float32) * scale  (dequantized)
    """
    amax = tensor.abs().amax()
    # float8_e4m3fn max representable value is 448.0
    scale = (amax / 448.0).clamp(min=1e-12).to(torch.float32)
    fp8_tensor = (
        (tensor.to(torch.float32) / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    return fp8_tensor, scale


@pytest.mark.parametrize("B", [3])
@pytest.mark.parametrize("L", [1025])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D_QK", [128, 576])
@pytest.mark.parametrize("D_V", [128, 512])
@pytest.mark.parametrize("CACHE_SIZE", [16384])
@pytest.mark.parametrize("PAGE_SIZE", [1, 16])
def test_decode_attention_fp8(B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE):
    """Test FP8 KV cache path: quantize K/V to FP8, run kernel with scales,
    and compare against BF16 reference output."""
    assert CACHE_SIZE % PAGE_SIZE == 0
    dtype = torch.bfloat16
    seq_len = L
    sm_scale = 1.0 / (D_QK**0.5)
    num_kv_splits = 8

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.randint(
        0, CACHE_SIZE // PAGE_SIZE, (B, num_pages_per_batch, 1), device=DEVICE_TYPE
    )
    req_to_token = req_to_page * PAGE_SIZE
    req_to_token = req_to_token.expand(B, num_pages_per_batch, PAGE_SIZE)
    req_to_token = req_to_token + torch.arange(PAGE_SIZE, device=DEVICE_TYPE).view(
        1, 1, -1
    )
    req_to_token = req_to_token.view(B, -1)
    req_to_token = req_to_token[:, :seq_len].contiguous()

    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE)

    # Create BF16 K/V as reference
    k_bf16 = torch.randn(CACHE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE)
    v_bf16 = torch.randn(CACHE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE)

    # --- BF16 reference ---
    o_ref = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
    lse_ref = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)
    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
    )

    if PAGE_SIZE == 1:
        decode_attention_fwd(
            q,
            k_bf16,
            v_bf16,
            o_ref,
            lse_ref,
            req_to_token,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
        )
    else:
        k_paged = k_bf16.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
        v_paged = v_bf16.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)
        decode_attention_fwd(
            q,
            k_paged,
            v_paged,
            o_ref,
            lse_ref,
            req_to_page,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            page_size=PAGE_SIZE,
        )

    # --- FP8 path ---
    k_fp8, k_scale = _quantize_to_fp8(k_bf16)
    v_fp8, v_scale = _quantize_to_fp8(v_bf16)

    o_fp8 = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
    lse_fp8 = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)
    attn_logits_fp8 = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
    )

    if PAGE_SIZE == 1:
        decode_attention_fwd(
            q,
            k_fp8,
            v_fp8,
            o_fp8,
            lse_fp8,
            req_to_token,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits_fp8,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    else:
        k_fp8_paged = k_fp8.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
        v_fp8_paged = v_fp8.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)
        decode_attention_fwd(
            q,
            k_fp8_paged,
            v_fp8_paged,
            o_fp8,
            lse_fp8,
            req_to_page,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits_fp8,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            page_size=PAGE_SIZE,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    # FP8 tolerances match test_mla_backends.py test_backend_correctness.
    torch.testing.assert_close(o_ref, o_fp8, atol=5e-1, rtol=1e-2)


def _spy_stage1_use_td(monkeypatch):
    """Record the USE_TD value of every _fwd_kernel_stage1 launch."""
    real = tda._fwd_kernel_stage1
    seen: list[bool] = []

    class _Spy:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                seen.append(kwargs["USE_TD"])
                return real[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(tda, "_fwd_kernel_stage1", _Spy())
    return seen


@pytest.mark.skipif(not current_platform.is_xpu(), reason="TD path is XPU-only")
@pytest.mark.parametrize("D_QK", [64, 128, 256])
@pytest.mark.parametrize("D_V", [64, 128, 256])
@pytest.mark.parametrize("H", [1, 8])
@pytest.mark.parametrize("PAGE_SIZE", [1, 16])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_decode_attention_stage1_td(monkeypatch, D_QK, D_V, H, PAGE_SIZE, kv_dtype):
    """Every head-dim combo the stage1 TD gate accepts must take the TD path
    and match the pointer path. Page 0 holds NaN/Inf and is never referenced:
    out-of-range gather lanes alias to it and must not leak into the output."""
    torch.manual_seed(0)
    B, seq_len, num_kv_splits = 3, 1025, 8
    num_pages = 4096 // PAGE_SIZE
    dtype = torch.bfloat16

    req_to_page = torch.randint(
        1, num_pages, (B, cdiv(seq_len, PAGE_SIZE)), device=DEVICE_TYPE
    )
    b_seq_len = torch.full((B,), seq_len, device=DEVICE_TYPE)
    q = torch.randn(B, H, D_QK, dtype=dtype, device=DEVICE_TYPE)
    k = torch.randn(num_pages, PAGE_SIZE, H, D_QK, dtype=dtype, device=DEVICE_TYPE)
    v = torch.randn(num_pages, PAGE_SIZE, H, D_V, dtype=dtype, device=DEVICE_TYPE)
    k[0] = float("inf")
    v[0] = float("nan")
    k, v = k.to(kv_dtype), v.to(kv_dtype)
    if PAGE_SIZE == 1:
        k, v = k.flatten(0, 1), v.flatten(0, 1)

    seen = _spy_stage1_use_td(monkeypatch)

    def run(td):
        monkeypatch.setenv("VLLM_TRITON_USE_TD", td)
        o = torch.zeros(B, H, D_V, dtype=dtype, device=DEVICE_TYPE)
        lse = torch.zeros(B, H, dtype=dtype, device=DEVICE_TYPE)
        attn_logits = torch.empty(
            (B, H, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
        )
        decode_attention_fwd(
            q,
            k,
            v,
            o,
            lse,
            req_to_page,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            1.0 / (D_QK**0.5),
            PAGE_SIZE,
        )
        return o, lse

    o_ref, lse_ref = run("0")
    o_td, lse_td = run("1")

    assert seen == [False, True]
    assert torch.isfinite(o_ref).all()
    torch.testing.assert_close(o_td, o_ref)
    torch.testing.assert_close(lse_td, lse_ref)


def test_stage1_td_gate_rejects_int32_row_overflow(monkeypatch):
    """Gather row indices are int32, so a KV cache with more than 2**31 - 1
    flattened (page, token) rows must keep the pointer path. Meta tensors stand
    in for a cache that large (256 GiB at this shape)."""
    monkeypatch.setattr(tda, "_td_gather_supported", lambda: True)
    page_size, d = 16, 64

    def gate(num_pages):
        q = torch.empty(1, 1, d, dtype=torch.bfloat16, device="meta")
        kv = torch.empty(
            num_pages, page_size, 1, d, dtype=torch.bfloat16, device="meta"
        )
        stride = tda._page_stride(kv, page_size)
        return tda._stage1_use_td(q, kv, kv, page_size, stride, stride, d, d, True)

    assert gate(2**31 // page_size - 1)
    assert not gate(2**31 // page_size)


@pytest.mark.parametrize(
    "H_Q,H_KV,D_QK,D_V,is_mla",
    [
        (16, 1, 576, 512, True),  # MLA path (grouped kernel, v = trans(k))
        (32, 8, 128, 128, False),  # GQA path (grouped kernel)
        (32, 32, 128, 128, False),  # MHA path (normal kernel)
    ],
)
@pytest.mark.parametrize("PAGE_SIZE", [16])
def test_decode_attention_cross_layer_view(H_Q, H_KV, D_QK, D_V, is_mla, PAGE_SIZE):
    """The kernel must honor the cache's page-dim stride, not assume pages are
    packed back-to-back. A per-layer view into a cross-layer (block-major)
    cache has stride(0) inflated by num_layers; outputs must match a
    contiguous cache holding the same data exactly."""
    B = 3
    seq_len = 1027
    CACHE_SIZE = 16384
    NUM_LAYERS = 3
    LAYER_IDX = 1
    dtype = torch.bfloat16
    sm_scale = 1.0 / (D_QK**0.5)
    num_kv_splits = 8
    num_pages = CACHE_SIZE // PAGE_SIZE

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.randint(
        0, num_pages, (B, num_pages_per_batch), device=DEVICE_TYPE
    )

    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE)
    b_seq_len = torch.full((B,), seq_len, device=DEVICE_TYPE)

    # Reference: contiguous paged cache.
    k_ref = torch.randn(
        num_pages, PAGE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE
    )
    if is_mla:
        v_ref = k_ref[..., :D_V]
    else:
        v_ref = torch.randn(
            num_pages, PAGE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE
        )

    # Cross-layer cache: all layers' pages for a block are adjacent. The
    # per-layer view has the same shape as the contiguous cache but
    # stride(0) is NUM_LAYERS x larger. Neighbor layers hold random data so
    # any packed-pages addressing reads garbage rather than zeros.
    k_xl = torch.randn(
        num_pages, NUM_LAYERS, PAGE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE
    )
    k_view = k_xl[:, LAYER_IDX]
    k_view.copy_(k_ref)
    assert k_view.stride(0) == NUM_LAYERS * PAGE_SIZE * H_KV * D_QK
    if is_mla:
        v_view = k_view[..., :D_V]
    else:
        v_xl = torch.randn(
            num_pages, NUM_LAYERS, PAGE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE
        )
        v_view = v_xl[:, LAYER_IDX]
        v_view.copy_(v_ref)

    def run(k_buffer, v_buffer):
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
        lse = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)
        attn_logits = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
        )
        decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_page,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            PAGE_SIZE,
            is_mla=is_mla,
        )
        return o, lse

    o_ref, lse_ref = run(k_ref, v_ref)
    o_xl, lse_xl = run(k_view, v_view)

    # Same data and same compute order; only addressing differs.
    assert torch.equal(o_ref, o_xl)
    assert torch.equal(lse_ref, lse_xl)
