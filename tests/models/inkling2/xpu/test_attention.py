# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 XPU attention kernels (`rel_attention`, the split-KV decode and
its combine, gate/sink) against the fp32 reference of `utils.ref_attention`,
on BF16 paged KV caches (pages 16, 64 (the XPU default) and 128)."""

import pytest
import torch

from tests.models.inkling2.xpu.utils import (
    DEVICE,
    HEAD_DIM,
    REL_EXTENT,
    device_graph,
    ref_attention,
    requires_first_config,
    requires_xpu,
    shear_rel_logits,
)
from vllm.models.inkling2.xpu.ops import attention as attention_ops
from vllm.models.inkling2.xpu.ops import rel_attention as rel_ops
from vllm.models.inkling2.xpu.ops.attention import (
    compute_log_scaling_tau,
    gate_sink,
    rel_decode_attention,
)
from vllm.models.inkling2.xpu.ops.rel_attention import (
    MODE_OUT_LSE,
    MODE_PARTIAL,
    rel_attention,
)
from vllm.models.inkling2.xpu.ops.sconv import attention_prologue, shift_rel_proj
from vllm.triton_utils import triton

pytestmark = requires_xpu

SCENARIOS = {
    "prefill": [(1500, 1500), (37, 37), (300, 300)],
    "chunked_prefill": [(200, 1700), (64, 64), (513, 2100)],
    "decode": [(1, 1), (1, 900), (1, 3000), (1, 1025)],
}


def _paged_kv(num_blocks, block_size, num_kv_heads):
    """The model's packed `[blocks, page, kv_heads, (K | V)]` BF16 cache,
    split into K and V views; keys L2-normalized as in the model."""
    kv = torch.randn(num_blocks, block_size, num_kv_heads, 2 * HEAD_DIM, device=DEVICE)
    kv[..., :HEAD_DIM] = torch.nn.functional.normalize(kv[..., :HEAD_DIM], dim=-1) * 2
    return kv.bfloat16().split(HEAD_DIM, dim=-1)


def _queries(num_tokens, num_heads):
    q = torch.randn(num_tokens, num_heads, HEAD_DIM, device=DEVICE)
    return (torch.nn.functional.normalize(q, dim=-1) * 3).bfloat16()


def _cu_seqlens(q_lens):
    return torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE
    )


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,block_size,window_left,scenario",
    [
        (24, 8, 64, None, "prefill"),
        (24, 8, 128, REL_EXTENT - 1, "chunked_prefill"),
        (12, 2, 16, None, "prefill"),
        (48, 8, 64, None, "prefill"),
        (12, 2, 128, REL_EXTENT - 1, "prefill"),
        (12, 2, 64, None, "chunked_prefill"),
        (48, 8, 16, REL_EXTENT - 1, "chunked_prefill"),
        (48, 8, 128, None, "chunked_prefill"),
        (12, 2, 64, None, "decode"),
        (48, 8, 128, REL_EXTENT - 1, "decode"),
    ],
)
@torch.inference_mode()
def test_rel_attention_matches_reference(
    num_heads, num_kv_heads, block_size, window_left, scenario
):
    """The prefill launch (`MODE_OUT_LSE`): fp32 output and natural-log LSE
    of full and windowed layers, over random pages of a shared cache."""
    torch.manual_seed(0)
    seqs = SCENARIOS[scenario]
    q_lens = [q_len for q_len, _ in seqs]
    num_tokens = sum(q_lens)
    max_blocks = -(-max(kv_len for _, kv_len in seqs) // block_size)
    num_blocks = len(seqs) * max_blocks + 1
    key_cache, value_cache = _paged_kv(num_blocks, block_size, num_kv_heads)
    block_table = (
        (torch.randperm(num_blocks - 1, device=DEVICE) + 1)[: len(seqs) * max_blocks]
        .view(len(seqs), max_blocks)
        .int()
    )
    q = _queries(num_tokens, num_heads)
    rel_logits = torch.randn(num_tokens, num_heads, REL_EXTENT, device=DEVICE)
    rel_logits = rel_logits.bfloat16()
    seqused_k = torch.tensor([kv for _, kv in seqs], dtype=torch.int32, device=DEVICE)
    scale = HEAD_DIM**-0.5
    out = torch.empty(q.shape, device=DEVICE)
    lse = torch.empty(q.shape[1::-1], device=DEVICE)
    rel_attention(
        q,
        key_cache,
        value_cache,
        out,
        lse,
        shear_rel_logits(rel_logits, seqs, 128, window_left is not None),
        block_table=block_table,
        cache_seqlens=seqused_k,
        cu_seqlens_q=_cu_seqlens(q_lens),
        max_query_len=max(q_lens),
        softmax_scale=scale,
        window=0 if window_left is None else window_left + 1,
        num_splits=1,
        mode=MODE_OUT_LSE,
    )
    ref_out, ref_lse = ref_attention(
        q, key_cache, value_cache, rel_logits, seqs, block_table, scale, window_left
    )
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse.t(), ref_lse, atol=1e-4, rtol=1e-5)


@requires_first_config
@pytest.mark.parametrize(
    "heads,query_len,splits,page",
    [(24, 300, 1, 64), (48, 300, 1, 64), (24, 300, 1, 128), (24, 1, 1, 128)]
    + [(48, 1, 5, 128)],
)
@pytest.mark.parametrize("window", [0, 1024])
@torch.inference_mode()
def test_rel_attention_descriptor_path_matches_pointer_path(
    heads, query_len, splits, page, window
):
    """The per-page tensor-descriptor K/V loads agree with the pointer path
    to fp32 rounding at the same tile (the dot operands come from different
    layouts), including a partial last page of stale NaNs."""
    torch.manual_seed(0)
    kv_heads, dim, kv_len = 8, 128, 1900
    num_pages = -(-kv_len // page) + 2
    q = torch.randn(query_len, heads, dim, device=DEVICE).bfloat16()
    kv = torch.randn(num_pages, page, kv_heads, 2 * dim, device=DEVICE).bfloat16()
    table = torch.randperm(num_pages, device=DEVICE)[: num_pages - 2].view(1, -1)
    # Slots past the sequence end hold stale values.
    kv[table[0, -1], kv_len % page :] = float("nan")
    lengths = torch.tensor([kv_len], device=DEVICE, dtype=torch.int32)
    cu = torch.tensor([0, query_len], device=DEVICE, dtype=torch.int32)
    rel = torch.randn(query_len, heads, 1024, device=DEVICE).bfloat16()
    sheared = shear_rel_logits(rel, [(query_len, kv_len)], 128, bool(window))
    k, v = kv.split(dim, dim=-1)
    block_m = rel_ops._block_m(query_len, heads // kv_heads)
    assert rel_ops.use_kv_descriptors(k, v, block_m, True)
    assert not rel_ops.use_kv_descriptors(k, v, block_m, False)

    def run(use_td):
        if splits == 1:
            out = torch.empty_like(q, dtype=torch.float32)
            lse = torch.empty(heads, query_len, device=DEVICE)
        else:
            out = torch.empty(splits, *q.shape, device=DEVICE)
            lse = torch.empty(splits, query_len, heads, device=DEVICE)
        rel_attention(
            q,
            k,
            v,
            out,
            lse,
            sheared,
            block_table=table.int(),
            cache_seqlens=lengths,
            cu_seqlens_q=cu,
            max_query_len=query_len,
            softmax_scale=dim**-0.5,
            window=window,
            num_splits=splits,
            mode=MODE_OUT_LSE if splits == 1 else MODE_PARTIAL,
            use_td=use_td,
        )
        return out, lse

    for observed, reference in zip(run(True), run(False)):
        assert not observed.isnan().any()
        torch.testing.assert_close(observed, reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("cache_dtype", [torch.float16, torch.float8_e4m3fn])
def test_rel_attention_rejects_non_bf16_kv(cache_dtype):
    """FP16 / FP8 KV caches fail loudly instead of being read as BF16."""
    q = torch.zeros(1, 8, HEAD_DIM, device=DEVICE).bfloat16()
    kv = torch.zeros(2, 64, 2, 2 * HEAD_DIM, device=DEVICE).to(cache_dtype)
    k, v = kv.split(HEAD_DIM, dim=-1)
    one = torch.ones(1, device=DEVICE, dtype=torch.int32)
    with pytest.raises(ValueError, match="only a bfloat16 KV cache"):
        rel_attention(
            q,
            k,
            v,
            torch.empty(q.shape, device=DEVICE),
            torch.empty(8, 1, device=DEVICE),
            q.new_zeros(129, 8, REL_EXTENT + 256),
            block_table=torch.zeros(1, 2, device=DEVICE, dtype=torch.int32),
            cache_seqlens=one,
            cu_seqlens_q=torch.tensor([0, 1], device=DEVICE, dtype=torch.int32),
            max_query_len=1,
            softmax_scale=HEAD_DIM**-0.5,
            window=0,
            num_splits=1,
            mode=MODE_OUT_LSE,
        )


def _decode_case(num_heads, num_kv_heads, window_left, query_lens, kv_lens, page):
    """Inputs of a decode/verify batch (plus one padding token) and the
    reference gated output of its real tokens."""
    torch.manual_seed(0)
    seqs = list(zip(query_lens, kv_lens))
    num_tokens = sum(query_lens)
    max_blocks = -(-max(kv_lens) // page)
    num_blocks = len(seqs) * max_blocks + 1
    key_cache, value_cache = _paged_kv(num_blocks, page, num_kv_heads)
    block_table = (
        (torch.randperm(num_blocks - 1, device=DEVICE) + 1)
        .view(len(seqs), max_blocks)
        .int()
    )
    q = _queries(num_tokens, num_heads)
    rel_logits = torch.randn(num_tokens, num_heads, REL_EXTENT, device=DEVICE)
    rel_logits = rel_logits.bfloat16()
    width = num_heads * HEAD_DIM
    proj = torch.randn(num_tokens + 1, width + num_heads + 4, device=DEVICE).bfloat16()
    gate, sinks = proj[:, :width], proj[:, width : width + num_heads]
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)
    scale = HEAD_DIM**-0.5
    inputs = dict(
        q=torch.cat([q, q[:1]]),
        key_cache=key_cache,
        value_cache=value_cache,
        rel_logits=shear_rel_logits(rel_logits, seqs, 128, window_left is not None),
        gate=gate,
        sinks=sinks,
        block_table=block_table,
        cache_seqlens=seqused_k,
        cu_seqlens_q=_cu_seqlens(query_lens),
        max_query_len=max(query_lens),
        softmax_scale=scale,
    )
    ref_out, ref_lse = ref_attention(
        q, key_cache, value_cache, rel_logits, seqs, block_table, scale, window_left
    )
    gate, sinks = gate[:num_tokens].float(), sinks[:num_tokens].float()
    expected = ref_out.view(num_tokens, -1) * torch.sigmoid(gate)
    expected *= torch.sigmoid(ref_lse - sinks).repeat_interleave(HEAD_DIM, dim=1)
    return inputs, num_tokens, expected


def _check_decode(out, num_tokens, expected):
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out[:num_tokens].float(), expected, atol=1e-2, rtol=1e-2)
    assert (out[num_tokens:] == 0).all()


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,window_left,query_lens,kv_lens,page,num_splits",
    [
        (24, 8, None, [1], [5000], 128, 1),
        (24, 8, None, [1] * 4, [1, 200, 1025, 3000], 128, 9),
        (24, 8, REL_EXTENT - 1, [1] * 4, [1, 200, 1025, 3000], 64, 64),
        (12, 2, None, [2, 1, 2], [2, 1500, 2100], 64, 9),
        (12, 2, REL_EXTENT - 1, [8, 8, 3], [8, 1500, 2100], 128, 1),
        (6, 1, REL_EXTENT - 1, [8, 8, 3], [8, 1500, 2100], 16, 9),
        (48, 8, None, [8, 8, 3], [8, 1500, 2100], 128, 64),
        (48, 8, REL_EXTENT - 1, [2, 1, 2], [2, 1500, 2100], 16, 1),
    ],
)
@torch.inference_mode()
def test_rel_decode_attention_matches_reference(
    num_heads, num_kv_heads, window_left, query_lens, kv_lens, page, num_splits
):
    """Decode and verify batches match the reference, gate and sink
    included, with one split (gated in the kernel) and split-KV (the
    combine); a padding token comes out zero."""
    inputs, num_tokens, expected = _decode_case(
        num_heads, num_kv_heads, window_left, query_lens, kv_lens, page
    )
    out = torch.full_like(inputs["gate"], float("nan"))
    rel_decode_attention(
        **inputs,
        out=out,
        num_actual_tokens=num_tokens,
        is_local=window_left is not None,
        num_splits=num_splits,
    )
    _check_decode(out, num_tokens, expected)


@pytest.mark.parametrize("query_lens", [[8, 8, 3], [1, 1]])
@torch.inference_mode()
def test_rel_decode_combine_every_tuned_config(query_lens):
    """Each config of the autotuned small-batch combine merges the split
    partials of `rel_attention` into the reference gated output."""
    num_heads, num_splits = 24, 9
    kv_lens = [8, 1500, 2100][: len(query_lens)]
    inputs, num_tokens, expected = _decode_case(
        num_heads, 8, None, query_lens, kv_lens, 128
    )
    q, gate, sinks = inputs["q"], inputs["gate"], inputs["sinks"]
    total = q.shape[0]
    assert total * num_heads < 512, "the tuned (small-batch) combine"
    o_part = torch.empty(num_splits, *q.shape, device=DEVICE)
    lse_part = torch.empty(num_splits, total, num_heads, device=DEVICE)
    rel_attention(
        q,
        inputs["key_cache"],
        inputs["value_cache"],
        o_part,
        lse_part,
        inputs["rel_logits"],
        gate,
        sinks,
        block_table=inputs["block_table"],
        cache_seqlens=inputs["cache_seqlens"],
        cu_seqlens_q=inputs["cu_seqlens_q"],
        max_query_len=inputs["max_query_len"],
        softmax_scale=inputs["softmax_scale"],
        window=0,
        num_splits=num_splits,
        mode=MODE_PARTIAL,
    )
    tuner = attention_ops._rel_decode_combine_kernel_tuned()
    assert len(tuner.configs) > 1
    for config in tuner.configs:
        out = torch.full_like(gate, float("nan"))
        block_d = config.kwargs["BLOCK_D"]
        attention_ops._rel_decode_combine_kernel[
            (total, num_heads, HEAD_DIM // block_d)
        ](
            o_part,
            lse_part,
            gate,
            sinks,
            out,
            total,
            num_tokens,
            gate.stride(0),
            sinks.stride(0),
            H=num_heads,
            D=HEAD_DIM,
            NUM_SPLITS=num_splits,
            BLOCK_S=triton.next_power_of_2(num_splits),
            **config.all_kwargs(),
        )
        _check_decode(out, num_tokens, expected)


@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("num_splits", [1, 9])
@torch.inference_mode()
def test_decode_attention_after_prologue_replays(is_local, num_splits):
    """The prologue (K/V into the cache, q/r clipping, sheared logits)
    followed by split-KV decode, captured in an XPU graph, replays to the
    output of the eager, synchronized launches."""
    torch.manual_seed(0)
    heads, kv_heads, d_rel, page = 12, 2, 16, 128
    q_lens, kv_lens = [8, 1], [3000, 300]
    num_reqs, num_tokens = len(q_lens), sum(q_lens)
    max_blocks = -(-max(kv_lens) // page)
    req = torch.arange(num_reqs, device=DEVICE).repeat_interleave(
        torch.tensor(q_lens, device=DEVICE)
    )
    cu_seqlens_q = torch.tensor([0, *q_lens], device=DEVICE).cumsum(0).int()
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)
    pos = seq_lens[req] - cu_seqlens_q[1:][req] + torch.arange(num_tokens).to(DEVICE)
    kv = torch.randn(num_reqs * max_blocks + 1, page, kv_heads, 2 * HEAD_DIM)
    kv = kv.to(DEVICE, torch.bfloat16)
    key_cache, value_cache = kv.split(HEAD_DIM, dim=-1)
    block_table = (torch.randperm(kv.shape[0] - 1, device=DEVICE) + 1).int()
    block_table = block_table.view(num_reqs, max_blocks)
    kv_slots = block_table[req, pos // page].long() * page + pos % page
    # Paged short-conv state: 4 positions per block, distinct blocks per request.
    conv_blocks = max(kv_lens) // 4 + 1
    conv_cache = torch.randn(num_reqs * conv_blocks + 1, kv_heads, 4, 1024)
    conv_cache = conv_cache.to(DEVICE, torch.bfloat16)
    conv_table = torch.arange(1, num_reqs * conv_blocks + 1, device=DEVICE).int()
    conv_table = conv_table.view(num_reqs, conv_blocks)
    conv_slots = conv_table[req, pos // 4].long() * 4 + pos % 4
    width = heads * (2 * HEAD_DIM + d_rel + 1)
    proj = torch.randn(num_tokens, width, device=DEVICE).bfloat16()
    q = proj[:, : heads * HEAD_DIM].view(num_tokens, heads, HEAD_DIM)
    r = proj[:, heads * HEAD_DIM : heads * (HEAD_DIM + d_rel)]
    gate = proj[:, heads * (HEAD_DIM + d_rel) : -heads]
    k, v = gate[:, : kv_heads * HEAD_DIM], gate[:, -kv_heads * HEAD_DIM :]
    sinks = proj[:, -heads:]
    conv_weight = torch.randn(kv_heads * HEAD_DIM, 4, device=DEVICE).bfloat16()
    rel_proj = torch.randn(d_rel, REL_EXTENT, device=DEVICE).bfloat16()
    rel_proj = shift_rel_proj(rel_proj, is_local)
    clipped_q = torch.empty_like(q)
    rel = q.new_empty(num_tokens + 128, heads, REL_EXTENT + 256)
    out = torch.empty(gate.shape, device=DEVICE, dtype=torch.bfloat16)
    kv_init, conv_init = kv.clone(), conv_cache.clone()

    def run(sync: bool):
        attention_prologue(
            q,
            r.view(num_tokens, heads, d_rel),
            k,
            v,
            q.new_ones(heads, HEAD_DIM),
            q.new_ones(heads, d_rel),
            None,
            rel_proj,
            conv_weight,
            conv_weight,
            None,
            None,
            conv_cache,
            pos,
            conv_table,
            req.int(),
            conv_slots,
            cu_seqlens_q[req],
            key_cache,
            value_cache,
            kv_slots,
            clipped_q,
            rel,
            cu_seqlens_q,
            seq_lens,
            clip_threshold=22.6,
            rel_threshold=8.0,
            norm_eps=1e-6,
        )
        if sync:
            torch.accelerator.synchronize()
        rel_decode_attention(
            clipped_q,
            key_cache,
            value_cache,
            rel,
            gate,
            sinks,
            out,
            block_table=block_table,
            cache_seqlens=seq_lens,
            cu_seqlens_q=cu_seqlens_q,
            max_query_len=max(q_lens),
            num_actual_tokens=num_tokens,
            softmax_scale=HEAD_DIM**-0.5,
            is_local=is_local,
            num_splits=num_splits,
        )

    run(sync=True)
    expected = out.clone()
    graph, capture = device_graph()
    with capture:
        run(sync=False)
    for _ in range(3):
        kv.copy_(kv_init)
        conv_cache.copy_(conv_init)
        out.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [1, 300])
def test_gate_sink(num_tokens):
    torch.manual_seed(0)
    num_heads, head_dim = 12, 128
    width = num_heads * head_dim
    output = torch.randn(num_tokens, num_heads, head_dim, device=DEVICE)
    # gate and sinks are strided column slices of a fused projection.
    proj = torch.randn(num_tokens, width + num_heads + 4, device=DEVICE).bfloat16()
    gate, sinks = proj[:, :width], proj[:, width : width + num_heads]
    lse = (torch.randn(num_heads, num_tokens, device=DEVICE) * 5).t()
    actual = torch.empty(gate.shape, device=DEVICE, dtype=torch.bfloat16)
    gate_sink(output, gate, lse, sinks, actual)
    scale = torch.sigmoid(lse - sinks.float()).repeat_interleave(head_dim, dim=1)
    expected = output.view(num_tokens, -1) * torch.sigmoid(gate.float()) * scale
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)


def test_log_scaling_tau():
    """`1 + alpha * log(max((pos + 1) / n_floor, 1))` per token."""
    positions = torch.tensor([0, 1, 4095, 4096, 100_000, 1 << 20], device=DEVICE)
    tau = compute_log_scaling_tau(positions, 4096, 0.1)
    ratio = ((positions + 1).double() / 4096).clamp(min=1.0)
    torch.testing.assert_close(tau, (1 + 0.1 * ratio.log()).float())


@torch.inference_mode()
def test_rel_attention_reuses_jit_across_prefill_lengths():
    """A prefill length off the %16 grid (and an unaligned `cache_seqlens`
    slice) reuses the kernel compiled for an aligned one."""
    heads, kv_heads, page = 24, 8, 64
    key_cache, value_cache = _paged_kv(8, page, kv_heads)
    table = torch.arange(8, device=DEVICE, dtype=torch.int32)[None]
    device = torch.accelerator.current_device_index()
    kernel_cache = rel_ops._rel_attention_kernel.device_caches[device][0]
    kernel_cache.clear()

    added = []
    for num_tokens, offset in ((64, 0), (72, 1)):
        before = len(kernel_cache)
        q = _queries(num_tokens, heads)
        seq_lens = torch.full((2,), num_tokens, device=DEVICE, dtype=torch.int32)
        rel_attention(
            q,
            key_cache,
            value_cache,
            torch.empty(q.shape, device=DEVICE),
            torch.empty(heads, num_tokens, device=DEVICE),
            q.new_zeros(num_tokens + 128, heads, REL_EXTENT + 256),
            block_table=table,
            cache_seqlens=seq_lens[offset : offset + 1],
            cu_seqlens_q=_cu_seqlens([num_tokens]),
            max_query_len=num_tokens,
            softmax_scale=HEAD_DIM**-0.5,
            window=0,
            num_splits=1,
            mode=MODE_OUT_LSE,
        )
        added.append(len(kernel_cache) - before)

    assert added[0] > 0 and added[1] == 0, added


@torch.inference_mode()
def test_rel_decode_combine_reuses_jit_across_batches():
    """Decode batches whose row counts fall in different specialization
    classes (%16, other, ==1) reuse the combine compiled for the first."""
    heads, num_splits = 24, 9
    device = torch.accelerator.current_device_index()
    kernel_cache = attention_ops._rel_decode_combine_kernel.device_caches[device][0]
    kernel_cache.clear()

    added = []
    for num_tokens, num_actual_tokens in ((32, 32), (24, 17), (1, 1)):
        o_part = torch.randn(num_splits, num_tokens, heads, HEAD_DIM, device=DEVICE)
        lse_part = torch.randn(num_splits, num_tokens, heads, device=DEVICE)
        gate = torch.randn(num_tokens, heads * HEAD_DIM, device=DEVICE).bfloat16()
        sinks = torch.randn(num_tokens, heads, device=DEVICE).bfloat16()
        out = torch.empty(gate.shape, device=DEVICE, dtype=torch.bfloat16)
        before = len(kernel_cache)
        attention_ops._rel_decode_combine_kernel[(num_tokens, heads, HEAD_DIM // 32)](
            o_part,
            lse_part,
            gate,
            sinks,
            out,
            num_tokens,
            num_actual_tokens,
            gate.stride(0),
            sinks.stride(0),
            H=heads,
            D=HEAD_DIM,
            NUM_SPLITS=num_splits,
            BLOCK_S=triton.next_power_of_2(num_splits),
            BLOCK_D=32,
            num_warps=4,
        )
        added.append(len(kernel_cache) - before)
        # The partial LSEs are base 2; the merged one (vs the sink) is natural.
        lse_nat = lse_part * 0.6931471805599453
        w, lse = torch.softmax(lse_nat, 0), torch.logsumexp(lse_nat, 0)
        expected = (w[..., None] * o_part).sum(0).view(num_tokens, -1)
        expected *= torch.sigmoid(gate.float())
        expected *= torch.sigmoid(lse - sinks.float()).repeat_interleave(HEAD_DIM, 1)
        expected[num_actual_tokens:] = 0
        torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)

    assert added == [1, 0, 0], added
