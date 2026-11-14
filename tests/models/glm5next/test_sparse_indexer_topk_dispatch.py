# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM kpool indexer's top-k backend wiring."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform


def _require_deep_gemm() -> None:
    from vllm.utils.deep_gemm import has_deep_gemm

    if not has_deep_gemm():
        pytest.skip("kpool indexer requires DeepGEMM")


def _build(indexer_cls, backend: str):
    cfg = VllmConfig(kernel_config={"sparse_indexer_topk_backend": backend})
    with set_current_vllm_config(cfg):
        return indexer_cls(
            k_cache=None,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=2048,
            head_dim=128,
            max_pool_len=4096,
            max_total_seq_len=8192,
            topk_indices_buffer=torch.empty(8, 2176, dtype=torch.int32, device="cuda"),
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only dispatch")
@pytest.mark.parametrize("backend", ["auto", "persistent", "cooperative", "torch"])
def test_kpool_indexer_dispatches_through_shared_topk_backend(backend: str) -> None:
    """The kpool indexer must read kernel_config.sparse_indexer_topk_backend
    and hand it to the shared SparseIndexerTopk dispatcher, rather than
    hard-coding its own cooperative/persistent/per_row choice."""
    _require_deep_gemm()
    from vllm.models.glm5next.nvidia.sparse_indexer import SparseAttnIndexerKpool

    assert _build(SparseAttnIndexerKpool, backend).topk_backend == backend


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only dispatch")
@pytest.mark.parametrize("backend", ["auto", "aiter", "per_row", "torch"])
def test_kpool_indexer_dispatches_through_shared_topk_backend_rocm(
    backend: str,
) -> None:
    """The AMD kpool indexer must go through the same dispatcher, so the
    AITER decode top-k is reachable from kernel_config."""
    from vllm.models.glm5next.amd.sparse_indexer import SparseAttnIndexerKpool

    assert _build(SparseAttnIndexerKpool, backend).topk_backend == backend


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only dispatch")
@pytest.mark.parametrize(
    "configured,full_cudagraph,expected",
    [
        ("auto", False, "aiter"),
        ("auto", True, "auto"),
        ("aiter", True, "aiter"),
        ("per_row", False, "per_row"),
    ],
)
def test_kpool_decode_topk_backend_never_auto_selects_aiter_under_full_cudagraph(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    full_cudagraph: bool,
    expected: str,
) -> None:
    """A FULL cudagraph bakes in one kernel for every replay, so "auto" cannot
    use the context length to pick AITER and must stay on the in-tree kernel.
    Explicit backends keep working, and outside FULL the long-context heuristic
    still narrows to AITER."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.glm5next.amd import sparse_indexer

    monkeypatch.setattr(
        rocm_aiter_ops, "is_indexer_top_k_enabled", lambda: True, raising=False
    )
    monkeypatch.setattr(
        rocm_aiter_ops,
        "is_indexer_top_k_supported",
        lambda **kwargs: True,
        raising=False,
    )

    assert (
        sparse_indexer._kpool_decode_topk_backend(
            configured,
            num_rows=8,
            max_valid_seq_len=64 * 1024,
            select_k=512,
            index_kpool=4,
            full_cudagraph=full_cudagraph,
        )
        == expected
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only workspace")
def test_kpool_profiling_reserves_decode_logits_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profiling must reserve worst-case decode logits in WorkspaceManager so a
    subsequent decode step of SparseAttnIndexerKpool does not grow a locked
    workspace when decode_rows * max_pool_len * 4 exceeds the prefill gather
    workspace."""
    from types import SimpleNamespace

    from transformers import Glm5NextTextConfig

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import CacheConfig, SchedulerConfig
    from vllm.forward_context import set_forward_context
    from vllm.models.glm5next.amd.sparse_indexer import SparseAttnIndexerKpool
    from vllm.models.glm5next.common.attention import Glm5NextIndexerCache
    from vllm.models.glm5next.common.model import get_sparse_topk_buffer_width
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        lock_workspace,
        reset_workspace_manager,
    )

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)

    device = torch.device("cuda")
    text_cfg = Glm5NextTextConfig()
    head_dim = text_cfg.index_head_dim
    n_heads = text_cfg.index_n_heads
    topk_tokens = text_cfg.index_topk
    index_kpool = text_cfg.index_kpool
    assert head_dim is not None and n_heads is not None
    assert topk_tokens is not None and index_kpool is not None

    max_num_seqs = 16
    num_spec = 5
    decode_rows = max_num_seqs * (num_spec + 1)
    max_num_batched_tokens = 256
    max_model_len = 16384
    max_pool_len = max_model_len // index_kpool
    # Prefill gather workspace is smaller than worst-case decode logits
    # (decode_rows * max_pool_len * 4 B = 1.5 MiB).
    max_total_seq_len = max_pool_len // 2

    cfg = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(block_size=128),
        kernel_config={"sparse_indexer_topk_backend": "torch"},
    )
    cfg.speculative_config = SimpleNamespace(num_speculative_tokens=num_spec)  # type: ignore[assignment]

    prefix = "model.layers.0.self_attn.indexer.k_cache"
    topk_buf = torch.empty(
        (max_num_batched_tokens, get_sparse_topk_buffer_width(text_cfg)),
        dtype=torch.int32,
        device=device,
    )

    reset_workspace_manager()
    init_workspace_manager(device)
    try:
        with set_current_vllm_config(cfg):
            k_cache = Glm5NextIndexerCache(
                head_dim=head_dim + 4,
                dtype=torch.uint8,
                prefix=prefix,
                cache_config=cfg.cache_config,
                index_kpool=index_kpool,
            )
            k_cache.kv_cache = torch.zeros(
                (4, 64, head_dim + 4), dtype=torch.uint8, device=device
            )
            indexer = SparseAttnIndexerKpool(
                k_cache=k_cache,
                quant_block_size=head_dim,
                scale_fmt="ue8m0",
                topk_tokens=topk_tokens,
                head_dim=head_dim,
                max_pool_len=max_pool_len,
                max_total_seq_len=max_total_seq_len,
                topk_indices_buffer=topk_buf,
                skip_k_cache_insert=True,
            )

        hidden_states = torch.zeros(
            (max_num_batched_tokens, text_cfg.hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        q_quant = torch.zeros(
            (max_num_batched_tokens, n_heads, head_dim),
            dtype=current_platform.fp8_dtype(),
            device=device,
        )
        k = torch.zeros(
            (max_num_batched_tokens, head_dim), dtype=torch.bfloat16, device=device
        )
        weights = torch.zeros(
            (max_num_batched_tokens, n_heads), dtype=torch.float32, device=device
        )

        # 1. Profiling dummy run (attn_metadata=None) reserves workspace.
        with set_current_vllm_config(cfg), set_forward_context(None, cfg):
            indexer(hidden_states, q_quant, k, weights, index_kpool=index_kpool)

        lock_workspace()

        # 2. Runtime decode step at worst-case decode_rows must fit the locked
        # workspace. max_seq_len > topk_tokens bypasses short-decode fast path.
        decode_meta = DeepseekV32IndexerMetadata(
            seq_lens=torch.ones(decode_rows, dtype=torch.int32, device=device),
            max_seq_len=topk_tokens + index_kpool,
            slot_mapping=torch.full(
                (decode_rows,), -1, dtype=torch.int64, device=device
            ),
            num_decodes=decode_rows,
            num_decode_tokens=decode_rows,
            num_prefills=0,
            num_prefill_tokens=0,
            decode=DeepSeekV32IndexerDecodeMetadata(
                block_table=torch.zeros(
                    (decode_rows, 4), dtype=torch.int32, device=device
                ),
                seq_lens=torch.ones((decode_rows, 1), dtype=torch.int32, device=device),
                decode_lens=torch.ones(decode_rows, dtype=torch.int32, device=device),
                requires_padding=False,
                schedule_metadata=torch.zeros(
                    (16, 2), dtype=torch.int32, device=device
                ),
            ),
        )
        with (
            set_current_vllm_config(cfg),
            set_forward_context({prefix: decode_meta}, cfg),
        ):
            out = indexer(
                hidden_states[:decode_rows],
                q_quant[:decode_rows],
                k[:decode_rows],
                weights[:decode_rows],
                index_kpool=index_kpool,
            )
        assert out is topk_buf
    finally:
        reset_workspace_manager()


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU-only dispatch")
@pytest.mark.parametrize("backend", ["auto", "per_row", "torch"])
def test_kpool_indexer_dispatches_through_shared_topk_backend_xpu(
    backend: str,
) -> None:
    """The XPU kpool indexer goes through the shared dispatcher too."""
    from vllm.models.glm5next.sparse_indexer import SparseAttnIndexerKpool
    from vllm.models.glm5next.xpu.sparse_indexer import (
        SparseAttnIndexerKpool as XpuSparseAttnIndexerKpool,
    )

    assert SparseAttnIndexerKpool is XpuSparseAttnIndexerKpool
    cfg = VllmConfig(kernel_config={"sparse_indexer_topk_backend": backend})
    with set_current_vllm_config(cfg):
        indexer = XpuSparseAttnIndexerKpool(
            k_cache=None,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=2048,
            head_dim=128,
            max_pool_len=4096,
            max_total_seq_len=8192,
            topk_indices_buffer=torch.empty(8, 2176, dtype=torch.int32, device="xpu"),
        )
    assert indexer.topk_backend == backend


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU import isolation")
def test_glm5next_imports_no_vendor_modules_on_xpu() -> None:
    """On XPU the GLM-5.3-Flash model, MTP and indexer entry points must not
    import the nvidia/ or amd/ implementations."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import vllm.models.glm5next\n"
        "import vllm.models.glm5next.sparse_indexer\n"
        "from vllm.model_executor.models.registry import ModelRegistry\n"
        "for arch in ('Glm5NextForConditionalGeneration', 'Glm5NextMTPModel'):\n"
        "    ModelRegistry._try_load_model_cls(arch)\n"
        "bad = sorted(m for m in sys.modules if m.startswith(\n"
        "    ('vllm.models.glm5next.nvidia', 'vllm.models.glm5next.amd')))\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def _ref_mqa_logits(q, k, k_scale, weights):
    """sum_h w[h] * relu(q[h] . k * k_scale), the MQA-logits definition."""
    s = torch.einsum("rhd,nd->rhn", q.float(), k.float())
    return (torch.relu(s * k_scale[None, None, :]) * weights[:, :, None]).sum(1)


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU kpool indexer")
@pytest.mark.parametrize("next_n", [1, 2])
@torch.inference_mode()
def test_xpu_kpool_indexer_decode_matches_reference(next_n: int, workspace_init):
    """Decode through the XPU kpool indexer with (B, next_n) per-row pool
    lengths: every row's selected tokens must be the top pools of its own
    context (by reference logits), expanded to tokens plus its tail pool.
    With MTP (next_n=2) consecutive rows of a request see different numbers
    of complete pools, and requests have different lengths."""
    from vllm import _custom_ops as ops
    from vllm.config import CacheConfig
    from vllm.forward_context import set_forward_context
    from vllm.models.glm5next.common.attention import Glm5NextIndexerCache
    from vllm.models.glm5next.common.ops import kpool_compress as kpool_ops
    from vllm.models.glm5next.xpu.sparse_indexer import SparseAttnIndexerKpool
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )

    torch.manual_seed(0)
    device = torch.device("xpu")
    kpool, page, head_dim, n_heads, topk_tokens = 4, 64, 128, 32, 64
    select_k = topk_tokens // kpool
    # Token positions of each request's last next_n tokens; pools complete at
    # pos % kpool == kpool - 1, so pool counts differ between rows.
    last_pos = [523, 290, 1001]
    positions = torch.tensor(
        [p - next_n + 1 + j for p in last_pos for j in range(next_n)],
        dtype=torch.int64,
        device=device,
    )
    batch, rows = len(last_pos), len(last_pos) * next_n
    row_pools = ((positions + 1) // kpool).to(torch.int32)

    max_pools = int(row_pools.max())
    n_pages = (max_pools + page - 1) // page
    kv = torch.zeros(batch * n_pages + 2, page, head_dim + 4, dtype=torch.uint8)
    kv = kv.to(device)
    block_table = torch.randperm(kv.shape[0], device=device)[: batch * n_pages]
    block_table = block_table.view(batch, n_pages).to(torch.int32)
    pool_k = []
    for b in range(batch):
        k = torch.randn(n_pages * page, head_dim, dtype=torch.bfloat16, device=device)
        idx = torch.arange(n_pages * page, device=device)
        slots = block_table[b, idx // page].long() * page + idx % page
        ops.indexer_k_quant_and_cache(k, kv, slots, head_dim, "ue8m0")
        pool_k.append(k)

    prefix = "model.layers.3.self_attn.indexer.k_cache"
    cfg = VllmConfig(
        cache_config=CacheConfig(block_size=kpool * page),
        kernel_config={"sparse_indexer_topk_backend": "per_row"},
    )
    width = topk_tokens + kpool - 1
    out_buf = torch.empty((rows, width), dtype=torch.int32, device=device)
    with set_current_vllm_config(cfg):
        k_cache = Glm5NextIndexerCache(
            head_dim=head_dim + 4,
            dtype=torch.uint8,
            prefix=prefix,
            cache_config=cfg.cache_config,
            index_kpool=kpool,
        )
        k_cache.kv_cache = kv
        indexer = SparseAttnIndexerKpool(
            k_cache=k_cache,
            quant_block_size=head_dim,
            scale_fmt="ue8m0",
            topk_tokens=topk_tokens,
            head_dim=head_dim,
            max_pool_len=n_pages * page,
            max_total_seq_len=n_pages * page,
            topk_indices_buffer=out_buf,
            skip_k_cache_insert=True,
        )
    q = torch.randn(rows, n_heads, head_dim, device=device).to(torch.float8_e4m3fn)
    weights = torch.rand(rows, n_heads, device=device)
    meta = DeepseekV32IndexerMetadata(
        seq_lens=row_pools.view(batch, next_n)[:, -1].contiguous(),
        max_seq_len=int(positions.max()) + 1,
        slot_mapping=torch.full((rows,), -1, dtype=torch.int64, device=device),
        num_decodes=batch,
        num_decode_tokens=rows,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=DeepSeekV32IndexerDecodeMetadata(
            block_table=block_table,
            seq_lens=row_pools.view(batch, next_n).contiguous(),
            decode_lens=torch.full((batch,), next_n, dtype=torch.int32, device=device),
            requires_padding=False,
            schedule_metadata=torch.empty(0, dtype=torch.int32, device=device),
        ),
    )
    hidden = torch.zeros(rows, 8, dtype=torch.bfloat16, device=device)
    k_new = torch.zeros(rows, head_dim, dtype=torch.bfloat16, device=device)
    with set_current_vllm_config(cfg), set_forward_context({prefix: meta}, cfg):
        out = indexer(hidden, q, k_new, weights, index_kpool=kpool, positions=positions)

    for r in range(rows):
        b, n = r // next_n, int(row_pools[r])
        k_ref = pool_k[b][:n].float()
        amax = k_ref.abs().amax(-1, keepdim=True).clamp(min=1e-4)
        scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        k_fp8 = (k_ref / scale).to(torch.float8_e4m3fn)
        ref = _ref_mqa_logits(
            q[r : r + 1], k_fp8, scale.squeeze(-1), weights[r : r + 1]
        )
        ref_pools = ref[0].topk(min(select_k, n)).indices.to(torch.int32)
        ref_pools = torch.cat(
            [ref_pools, ref_pools.new_full((select_k - ref_pools.numel(),), -1)]
        )
        expected = kpool_ops.expand_pools_and_append_tail(
            ref_pools[None], positions[r : r + 1], kpool, seq_offset=1
        )[0]
        got = set(out[r].tolist()) - {-1}
        assert got == set(expected.tolist()) - {-1}, f"row {r} (request {b})"
