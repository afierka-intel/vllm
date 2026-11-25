# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 XPU core kernels (`vllm/models/inkling2/xpu/ops/`) against plain
torch references of the same math, at Inkling 2 shapes (hidden 2304 / 4608,
1024-wide relative extent)."""

import pytest
import torch

from tests.models.inkling2.xpu.utils import (
    DEVICE,
    LONG_STEPS,
    PADDED_STEPS,
    STEPS,
    chunked_steps,
    clip,
    conv_cache_and_table,
    dense_conv,
    device_graph,
    pad_rows,
    requires_first_config,
    requires_xpu,
    shear_rel_logits,
)
from vllm.models.inkling2.xpu.ops import linear as linear_ops
from vllm.models.inkling2.xpu.ops.attn_res import attn_res
from vllm.models.inkling2.xpu.ops.conv_mix import conv_mix
from vllm.models.inkling2.xpu.ops.loop import pair_rms_norm
from vllm.models.inkling2.xpu.ops.norm import rms_norm
from vllm.models.inkling2.xpu.ops.sconv import (
    attention_prologue,
    shift_rel_proj,
    short_conv,
)
from vllm.triton_utils import tensor_descriptor_compatible, triton
from vllm.triton_utils.allocation import set_triton_allocator

pytestmark = requires_xpu


def _ordered_groups(x, w):
    return [
        x[:, g : g + 1024].float() @ w[:, g : g + 1024].float().T
        for g in range(0, x.shape[1], 1024)
    ]


@pytest.mark.parametrize(
    "tokens,n,k",
    [(1, 128, 2304), (37, 384, 4608), (300, 32, 4608), (2048, 128, 2304)],
)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_k_group_partials_match_ordered_groups(tokens, n, k, out_dtype):
    """`k_group_partials` keeps fp32 1024-wide K-group partials (or their
    in-order sum), as the routers' reference `linear`."""
    torch.manual_seed(0)
    x = torch.randn(tokens, k, device=DEVICE).bfloat16()
    w = torch.randn(n, k, device=DEVICE).bfloat16()
    groups = _ordered_groups(x, w)
    out = linear_ops.k_group_partials(x, w, out_dtype)
    if out.shape[0] == len(groups):
        expected = torch.stack(groups)
    else:
        total = groups[0]
        for partial in groups[1:]:
            total = total + partial
        expected = total[None].to(out_dtype)
    torch.testing.assert_close(out.float(), expected.float(), atol=2e-3, rtol=1e-2)


@pytest.mark.parametrize(
    "tokens,n,k",
    # Router / dynamic taps (split K groups) and an unsplit in-order sum.
    [(8, 32, 4608), (40, 384, 4608), (300, 128, 2304), (4096, 128, 2304)],
)
def test_k_group_tiles_and_paths_are_bitwise_identical(tokens, n, k):
    """Every autotuned K-group tile, with and without tensor descriptors,
    gives the bits of the fixed pointer launch, so neither the tuned pick
    (per process, per TP rank) nor the load path can change a router's
    top-k."""
    torch.manual_seed(0)
    x = torch.randn(tokens, k, device=DEVICE).bfloat16()
    w = torch.randn(n, k, device=DEVICE).bfloat16()
    groups = triton.cdiv(k, 1024)
    split = groups > 1 and tokens < 4096
    set_triton_allocator(x.device)
    results = []
    for use_td in (False, True):
        for bm, bn, warps, stages in linear_ops._TUNED_TILES:
            out = torch.empty(groups if split else 1, tokens, n, device=DEVICE)
            if use_td:
                assert tensor_descriptor_compatible(x, w, out[0])
            grid = (triton.cdiv(n, bn), triton.cdiv(tokens, bm), out.shape[0])
            linear_ops._k_group_kernel[grid](
                x,
                w,
                out,
                tokens,
                n,
                k,
                bm,
                bn,
                split,
                USE_TD=use_td,
                num_warps=warps,
                num_stages=stages,
                enable_fp_fusion=False,
            )
            results.append(((use_td, bm, bn, warps, stages), out))
    first = results[0][1]
    for tile, out in results[1:]:
        assert torch.equal(out.view(torch.int32), first.view(torch.int32)), tile


@pytest.mark.parametrize("tokens,hidden", [(1, 4608), (33, 1152), (4000, 4608)])
def test_linear_k_groups_fp32(tokens, hidden):
    """The split-K partials summed by `_k_group_sum_kernel` (few tokens) and
    the in-program group sum both give the fp32 product."""
    torch.manual_seed(0)
    x = torch.randn(tokens, hidden, device=DEVICE).bfloat16()
    w = (torch.randn(384, hidden, device=DEVICE) * 0.02).bfloat16()
    out = linear_ops.linear_k_groups(x, w)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, x.double() @ w.double().T, check_dtype=False)


@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [4608, 128])
def test_rms_norm_rounds_once_to_bf16(in_dtype, width):
    x = torch.randn(37, width, device=DEVICE).to(in_dtype) * 3
    weight = torch.randn(width, device=DEVICE).bfloat16()
    ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    ref = ref * weight.float()
    out = rms_norm(x, weight, 1e-6, torch.bfloat16)
    torch.testing.assert_close(out, ref.bfloat16(), atol=0, rtol=1e-2)


@pytest.mark.parametrize("tokens", [0, 1, 37, 8192])
@pytest.mark.parametrize("broadcast", [False, True])
def test_pair_rms_norm_matches_stacked_reference_and_graph(tokens, broadcast):
    """The loop controllers' pair norm equals `rms_norm` of the stacked
    `[dest, message]`, also on graph replay with new inputs."""
    torch.manual_seed(42)
    streams, width = 3, 2304
    source = torch.randn(1 if broadcast else streams, tokens, width, device=DEVICE)
    dest = source.expand(streams, -1, -1)
    message = torch.randn(streams, tokens, width, device=DEVICE)
    weight = torch.ones(width, device=DEVICE)

    def reference():
        pair = torch.stack((dest, message), dim=2)
        return rms_norm(pair, weight, 1e-6, torch.bfloat16).flatten(2)

    out = pair_rms_norm(dest, message)
    torch.testing.assert_close(out, reference(), atol=0, rtol=0)
    if tokens:
        graph, capture = device_graph()
        with capture:
            out = pair_rms_norm(dest, message)
        for _ in range(3):
            source.copy_(torch.randn_like(source))
            message.copy_(torch.randn_like(message))
            graph.replay()
            torch.testing.assert_close(out, reference(), atol=0, rtol=0)


def _prologue_qr(
    q, r, q_scale, r_scale, tau, rel_proj, kv_heads, seqs=None, is_local=False
):
    """Run the attention prologue for q/r only: zero K/V, no KV-cache writes.

    Returns the sheared relative logits of requests `seqs` (`(q_len,
    kv_len)` each). By default the tokens form one request after 2048 cached
    ones, so that every distance is present, and the logits come unsheared.
    """
    tokens, heads, dim = q.shape
    extent = rel_proj.shape[1]
    out_q = q.new_empty(tokens, heads, dim)
    rel = q.new_empty(tokens + 128, heads, extent + 256)
    past = 2048
    q_lens = [tokens] if seqs is None else [q_len for q_len, _ in seqs]
    kv_lens = [tokens + past] if seqs is None else [kv_len for _, kv_len in seqs]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE
    )
    seq_idx = torch.repeat_interleave(
        torch.arange(len(q_lens), device=DEVICE), torch.tensor(q_lens, device=DEVICE)
    ).int()
    kv = q.new_zeros(tokens, kv_heads * dim)
    kv_cache = q.new_zeros(1, 16, kv_heads, 2 * dim)
    cache, table = conv_cache_and_table(kv_heads, 4, 1024, tokens // 4 + 1)
    idx = torch.arange(tokens, device=DEVICE)
    no_slot = torch.full_like(idx, -1)
    key_cache, value_cache = kv_cache.split(dim, dim=-1)
    attention_prologue(
        q,
        r,
        kv,
        kv,
        q_scale,
        r_scale,
        tau,
        shift_rel_proj(rel_proj, is_local),
        q.new_zeros(kv_heads * dim, 4),
        q.new_zeros(kv_heads * dim, 4),
        None,
        None,
        cache,
        idx,
        table[:1],
        seq_idx,
        no_slot,
        torch.zeros_like(idx).int(),
        key_cache,
        value_cache,
        no_slot,
        out_q,
        rel,
        cu_seqlens_q,
        torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE),
        clip_threshold=22.6,
        rel_threshold=8.0,
        norm_eps=1e-6,
    )
    if seqs is not None:
        return out_q, rel
    tile_end = ((idx // 128 + 1) * 128).clamp(max=tokens)
    diag = past + idx - ((past + tile_end + 127) // 128 - 10) * 128
    cols = diag[:, None] - torch.arange(extent, device=DEVICE)
    rel = rel[:tokens].gather(2, cols[:, None].expand(-1, heads, -1))
    return out_q, rel


@pytest.mark.parametrize("heads,kv_heads", [(12, 2), (6, 1), (48, 8)])
@pytest.mark.parametrize("with_tau", [True, False])
def test_qr_clip_rel_rounds_before_tau(heads, kv_heads, with_tau):
    torch.manual_seed(0)
    tokens, dim, d_rel, extent = 33, 128, 16, 1024
    # q and r are strided column slices of the fused projection.
    proj = torch.randn(tokens, heads * (dim + d_rel) + 64, device=DEVICE) * 4
    proj = proj.bfloat16()
    q = proj[:, : heads * dim].view(tokens, heads, dim)
    r = proj[:, heads * dim : heads * (dim + d_rel)].view(tokens, heads, d_rel)
    q_scale = torch.rand(heads, dim, device=DEVICE).bfloat16() * 2
    r_scale = torch.rand(heads, d_rel, device=DEVICE).bfloat16() * 2
    rel_proj = torch.randn(d_rel, extent, device=DEVICE).bfloat16()
    tau = 1 + torch.rand(tokens, device=DEVICE) if with_tau else None

    def ref_clip(x, scale, threshold):
        y = clip(x.float() * scale.float(), threshold).bfloat16()
        if tau is None:
            return y
        return (y.float() * tau[:, None, None]).bfloat16()

    ref_q = ref_clip(q, q_scale, 22.6)
    ref_r = ref_clip(r, r_scale, 8.0)
    ref_rel = torch.einsum("thd,de->the", ref_r, rel_proj)
    out_q, rel = _prologue_qr(q, r, q_scale, r_scale, tau, rel_proj, kv_heads)
    torch.testing.assert_close(out_q, ref_q, atol=0, rtol=1e-2)
    # A 1-ulp difference in a clipped r element shifts its 1024 logits by
    # about ulp(r) * |rel_proj|.
    torch.testing.assert_close(rel, ref_rel, atol=0.25, rtol=1e-2)


@pytest.mark.parametrize(
    "seqs",
    [
        # Decode, verify, chunked and fresh prefill requests.
        [(1, 3000), (8, 1500), (300, 300), (200, 1700), (130, 1029)],
        # 2048 tokens or more: q/r run in their own launch.
        [(1, 3000), (8, 1500), (2100, 2100), (130, 1029)],
    ],
    ids=["fused", "split"],
)
@pytest.mark.parametrize("is_local", [False, True])
def test_prologue_shears_rel_logits(is_local, seqs):
    """The relative logits land in the sheared layout `rel_attention` reads,
    padding included, from the fused and the separate q/r launch."""
    torch.manual_seed(0)
    heads, kv_heads, dim, d_rel, extent = 12, 2, 128, 16, 1024
    tokens = sum(q_len for q_len, _ in seqs)
    q = torch.randn(tokens, heads, dim, device=DEVICE).bfloat16()
    r = (torch.randn(tokens, heads, d_rel, device=DEVICE) * 4).bfloat16()
    q_scale = q.new_ones(heads, dim)
    r_scale = q.new_ones(heads, d_rel)
    rel_proj = torch.randn(d_rel, extent, device=DEVICE).bfloat16()
    args = (q, r, q_scale, r_scale, None, rel_proj, kv_heads)
    _, rel = _prologue_qr(*args, seqs=seqs, is_local=is_local)
    _, compact = _prologue_qr(*args)
    ref = shear_rel_logits(compact, seqs, 128, is_local)
    torch.testing.assert_close(rel[:tokens], ref[:tokens], atol=0, rtol=0)


def _attn_res_reference(stack, last, addend, queries, normalize):
    values = torch.cat([stack, last[None]]).float()
    if addend is not None:
        values[-1] += addend.float()
    inv_rms = torch.rsqrt(values.square().mean(-1) + 1e-6)
    probs = torch.einsum("ntd,qd,nt->qtn", values, queries, inv_rms).softmax(-1)
    if normalize:
        values = values * inv_rms[..., None]
    return torch.einsum("qtn,ntd->qtd", probs, values)


@pytest.mark.parametrize("num_stack,num_q,with_addend", [(1, 1, False), (5, 3, True)])
@pytest.mark.parametrize("normalize", [True, False])
# Small batches keep rows in registers; large ones loop over D chunks.
@pytest.mark.parametrize("tokens", [29, 300])
def test_attn_res_matches_softmax_mix_then_norm(
    num_stack, num_q, with_addend, normalize, tokens
):
    dim = 4608
    stack = torch.randn(num_stack, tokens, dim, device=DEVICE).bfloat16()
    last = torch.randn(tokens, dim, device=DEVICE).bfloat16()
    addend = torch.randn(tokens, dim, device=DEVICE).bfloat16() if with_addend else None
    queries = torch.randn(num_q, dim, device=DEVICE) * dim**-0.5
    weight = torch.randn(num_q, dim, device=DEVICE).bfloat16()
    mix = _attn_res_reference(stack, last, addend, queries, normalize)
    ref = mix * torch.rsqrt(mix.square().mean(-1, keepdim=True) + 1e-6)
    ref = (ref * weight.float()[:, None]).bfloat16()
    out = attn_res(stack, last, addend, queries, weight, 1e-6, normalize)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-2)


@pytest.mark.parametrize("num_stack,with_addend", [(1, False), (3, True)])
@pytest.mark.parametrize("tokens", [29, 300])
def test_attn_res_without_norm_returns_fp32_mix(num_stack, with_addend, tokens):
    """Without a norm weight, `attn_res` returns the fp32 softmax mix (the
    loop's sublayer inputs before their controllers, and its message
    reads)."""
    dim = 2304
    stack = torch.randn(num_stack, tokens, dim, device=DEVICE).bfloat16()
    last = torch.randn(tokens, dim, device=DEVICE).bfloat16()
    addend = torch.randn(tokens, dim, device=DEVICE).bfloat16() if with_addend else None
    queries = torch.randn(4, dim, device=DEVICE) * dim**-0.5
    ref = _attn_res_reference(stack, last, addend, queries, True)
    out = attn_res(stack, last, addend, queries, None, 0.0, True)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def _run_short_conv(steps, fused_residual, pad, seed=0):
    """`short_conv` over the chunked `steps` of 4 requests: per request the
    concatenated outputs and its inputs, and the conv-state cache."""
    torch.manual_seed(seed)
    block, heads, stream_width, offset, row = 4, 2, 128, 128, 1024
    channels = heads * stream_width
    totals = [sum(s[i] for s in steps) for i in range(4)]
    xs = [torch.randn(n, channels, device=DEVICE).bfloat16() for n in totals]
    adds = [torch.randn(n, channels, device=DEVICE).bfloat16() for n in totals]
    ress = [torch.randn(n, channels, device=DEVICE).bfloat16() for n in totals]
    weight = torch.randn(channels, 4, device=DEVICE).bfloat16()
    cache, block_table = conv_cache_and_table(heads, block, row, 64)
    outs: list[list[torch.Tensor]] = [[] for _ in totals]
    for reqs, rows, starts, meta in chunked_steps(steps, block_table, block):
        x = torch.cat([xs[i][r] for i, r in zip(reqs, rows)])
        kwargs: dict[str, torch.Tensor] = {}
        if fused_residual:
            res = torch.cat([ress[i][r] for i, r in zip(reqs, rows)])
            add = torch.cat([adds[i][r] for i, r in zip(reqs, rows)])
            kwargs = dict(addend=add, residual=res, out=res)
        if pad:
            for name, t in kwargs.items():
                kwargs[name] = torch.cat([t, t.new_zeros(pad, channels)])
            if fused_residual:
                kwargs["out"] = kwargs["residual"]
            x, meta = pad_rows(x, meta, pad)
        out = short_conv(x, weight, cache, *meta, offset, stream_width, **kwargs)
        for j, i in enumerate(reqs):
            outs[i].append(out[starts[j] : starts[j + 1]])
    outs_cat = [torch.cat(o) for o in outs]
    return outs_cat, (xs, adds, ress, weight), cache


@pytest.mark.parametrize("pad", [0, 3])
@pytest.mark.parametrize("fused_residual", [False, True])
@pytest.mark.parametrize("steps", [STEPS, LONG_STEPS, PADDED_STEPS])
def test_short_conv_chunked_matches_full_sequence(fused_residual, steps, pad):
    """Chunked prefill + decode through the paged conv state equals one pass,
    with or without padding rows; the residual update `(res + addend) + y` is
    fused in fp32."""
    outs, (xs, adds, ress, weight), _ = _run_short_conv(steps, fused_residual, pad)
    for i in range(4):
        ref = dense_conv(xs[i], weight, None)
        if fused_residual:
            ref = (ress[i].float() + adds[i].float() + ref.float()).bfloat16()
        torch.testing.assert_close(outs[i], ref, atol=0, rtol=1e-2)


@requires_first_config
@pytest.mark.parametrize("fused_residual", [False, True])
def test_short_conv_descriptor_path_matches_pointer_path(kernel_env, fused_residual):
    """The tensor-descriptor row tiles of `short_conv` give the pointer
    path's outputs and conv-state cache bitwise, at the same tile."""
    results = []
    for use_td in ("0", "1"):
        kernel_env(VLLM_TRITON_USE_TD=use_td)
        outs, _, cache = _run_short_conv(LONG_STEPS, fused_residual, 0)
        results.append((outs, cache))
    (td_outs, td_cache), (outs, cache) = results[1], results[0]
    assert torch.equal(td_cache, cache)
    for td_out, out in zip(td_outs, outs):
        assert torch.equal(td_out, out)


@pytest.mark.parametrize(
    "num_stack,num_q,with_last,with_residual,with_norm",
    [
        (1, 1, True, False, True),
        (2, 1, False, False, False),
        (3, 1, False, True, True),
        (8, 3, False, True, False),
    ],
)
@pytest.mark.parametrize("dim", [4608, 2304])
def test_conv_mix_matches_conv_then_attn_res(
    num_stack, num_q, with_last, with_residual, with_norm, dim
):
    """The prefill conv + mix equals short_conv then attn_res over chunked
    steps (bitwise conv and cache; the mix up to its fp32 summation order),
    with the RMSNorm or returning the fp32 mixes."""
    block, heads, offset = 4, 2, 256
    stream_width = dim // heads
    row = offset + stream_width
    weight = torch.randn(dim, 4, device=DEVICE).bfloat16()
    queries = torch.randn(num_q, dim, device=DEVICE) * dim**-0.5
    norm_w = torch.randn(num_q, dim, device=DEVICE).bfloat16() if with_norm else None
    eps = 1e-6 if with_norm else 0.0
    cache, block_table = conv_cache_and_table(heads, block, row, 64)
    ref_cache = cache.clone()
    for _, _, _, meta in chunked_steps(LONG_STEPS, block_table, block):
        tokens = meta[0].shape[0]
        x = torch.randn(tokens, dim, device=DEVICE).bfloat16()
        stack = torch.randn(num_stack, tokens, dim, device=DEVICE).bfloat16()
        last = torch.randn(tokens, dim, device=DEVICE).bfloat16()
        addend = torch.randn(tokens, dim, device=DEVICE).bfloat16()
        residual = torch.randn(tokens, dim, device=DEVICE).bfloat16()
        ref_conv = residual.clone() if with_residual else None
        ref_conv = short_conv(
            x,
            weight,
            ref_cache,
            *meta,
            offset,
            stream_width,
            addend if with_residual else None,
            ref_conv,
            ref_conv,
        )
        if with_last:
            ref = attn_res(stack, last, ref_conv, queries, norm_w, eps, True)
        else:
            ref = attn_res(stack, ref_conv, None, queries, norm_w, eps, True)
        conv = residual.clone() if with_residual else torch.empty_like(x)
        out = conv_mix(
            x,
            weight,
            cache,
            *meta,
            offset,
            stream_width,
            conv,
            conv if with_residual else None,
            addend if with_residual else None,
            stack,
            last if with_last else None,
            queries,
            norm_w,
            eps,
        )
        torch.testing.assert_close(conv, ref_conv, atol=0, rtol=0)
        torch.testing.assert_close(cache, ref_cache, atol=0, rtol=0)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# A step of 2048 tokens or more runs q/r in their own launch.
SPLIT_STEPS = [[3, 2050, 2, 9], [2, 6, 0, 0], [1, 1, 1, 1]]


@pytest.mark.parametrize("steps", [STEPS, SPLIT_STEPS], ids=["fused", "split"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_attention_prologue_writes_clipped_k_and_normed_v(dynamic, steps):
    """K/V convs over chunked steps, then K L2 clipping and V RMSNorm, land in
    the paged KV cache at each token's slot; padded tokens (slot -1, as in
    graph padding) write nothing."""
    torch.manual_seed(0)
    width, block, heads, dim, kv_block = 4, 4, 2, 128, 16
    channels = heads * dim
    totals = [sum(s[i] for s in steps) for i in range(4)]
    ks, vs = (
        [torch.randn(n, channels, device=DEVICE).bfloat16() * 3 for n in totals]
        for _ in range(2)
    )
    dks, dvs = (
        [torch.randn(n, heads * width, device=DEVICE).bfloat16() for n in totals]
        for _ in range(2)
    )
    shape = (width, channels) if dynamic else (channels, width)
    wk, wv = (torch.randn(shape, device=DEVICE).bfloat16() for _ in range(2))
    cache, block_table = conv_cache_and_table(heads, block, 1024, max(totals) // 4 + 1)
    num_kv_blocks = triton.cdiv(2 * sum(totals), kv_block)
    kv = torch.zeros(num_kv_blocks, heads, kv_block, 2 * dim, device=DEVICE)
    kv = kv.bfloat16()
    key_cache, value_cache = kv.transpose(1, 2).split(dim, dim=-1)
    perm = torch.randperm(num_kv_blocks * kv_block, device=DEVICE)[: sum(totals)]
    kv_slots = list(perm.split(totals))

    for reqs, rows, starts, meta in chunked_steps(steps, block_table, block):

        def cat(ts, reqs=reqs, rows=rows, pad=None):
            x = torch.cat([ts[i][r] for i, r in zip(reqs, rows)])
            pad = torch.full_like(x[:2], -1) if pad is None else pad(x[:2])
            return torch.cat([x, pad])

        positions, table, seq_idx, slots, qstart = meta
        num_tokens = positions.shape[0] + 2
        q = torch.zeros(num_tokens, 6, dim, device=DEVICE).bfloat16()
        r = q.new_zeros(num_tokens, 6, 16)
        attention_prologue(
            q,
            r,
            cat(ks, pad=torch.randn_like),
            cat(vs, pad=torch.randn_like),
            q.new_ones(6, dim),
            q.new_ones(6, 16),
            None,
            shift_rel_proj(q.new_zeros(16, 1024), True),
            wk,
            wv,
            cat(dks, pad=torch.randn_like) if dynamic else None,
            cat(dvs, pad=torch.randn_like) if dynamic else None,
            cache,
            torch.cat([positions, positions[:2]]),
            table,
            torch.cat([seq_idx, seq_idx[:2]]),
            torch.cat([slots, torch.full_like(slots[:2], -1)]),
            torch.cat([qstart, qstart[:2]]),
            key_cache,
            value_cache,
            cat(kv_slots),
            torch.empty_like(q),
            q.new_empty(num_tokens + 128, 6, 1024 + 256),
            starts.int().to(DEVICE),
            starts.diff().int().to(DEVICE) + 4096,
            clip_threshold=22.6,
            rel_threshold=8.0,
            norm_eps=1e-6,
        )
    for i in range(4):
        dyn = (
            (dks[i].view(-1, heads, width), dvs[i].view(-1, heads, width))
            if dynamic
            else (None, None)
        )
        k = dense_conv(ks[i], wk, dyn[0]).float().view(-1, heads, dim)
        v = dense_conv(vs[i], wv, dyn[1]).float().view(-1, heads, dim)
        ref_k = clip(k, 22.6).bfloat16()
        ref_v = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        slot = kv_slots[i]
        got_k = key_cache[slot // kv_block, slot % kv_block]
        got_v = value_cache[slot // kv_block, slot % kv_block]
        torch.testing.assert_close(got_k, ref_k, atol=0, rtol=1e-2)
        torch.testing.assert_close(got_v, ref_v, atol=0, rtol=1e-2)
    num_tokens = sum(totals)
    assert (kv != 0).any(-1).sum() == num_tokens * heads
    assert (cache[..., : 2 * dim] != 0).any(-1).sum() == num_tokens * heads


@pytest.mark.parametrize("cache_dtype", [torch.float16, torch.float8_e4m3fn])
def test_attention_prologue_rejects_low_precision_kv(cache_dtype):
    """A KV cache other than bfloat16 fails loudly before any launch (the
    XPU kernels store and read BF16 K/V only), and leaves it untouched."""
    heads, dim, tokens = 2, 128, 3
    q = torch.zeros(tokens, 6, dim, device=DEVICE).bfloat16()
    kv_input = torch.full((tokens, heads * dim), 2.0, device=DEVICE).bfloat16()
    weights = torch.zeros(heads * dim, 4, device=DEVICE).bfloat16()
    conv, table = conv_cache_and_table(heads, 4, 1024)
    kv = torch.full((1, 16, heads, 2 * dim), -3.0, device=DEVICE).to(cache_dtype)
    keys, values = kv.split(dim, dim=-1)
    indices = torch.arange(tokens, device=DEVICE)
    with pytest.raises(ValueError, match="only a bfloat16 KV cache"):
        attention_prologue(
            q,
            q.new_zeros(tokens, 6, 16),
            kv_input,
            kv_input,
            q.new_ones(6, dim),
            q.new_ones(6, 16),
            None,
            shift_rel_proj(q.new_zeros(16, 1024), True),
            weights,
            weights,
            None,
            None,
            conv,
            indices,
            table[:1],
            torch.zeros_like(indices).int(),
            indices,
            torch.zeros_like(indices).int(),
            keys,
            values,
            torch.tensor([2, 7, -1], device=DEVICE),
            torch.empty_like(q),
            q.new_empty(tokens + 128, 6, 1280),
            torch.tensor([0, tokens], device=DEVICE, dtype=torch.int32),
            torch.tensor([tokens], device=DEVICE, dtype=torch.int32),
            clip_threshold=32.0,
            rel_threshold=8.0,
            norm_eps=1e-6,
        )
    assert (kv.float() == -3.0).all()
