# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch references, launch gates and tiny configs of the Inkling2 XPU tests.

The references are kept here rather than imported from the CUDA tests in
`tests/models/inkling2/`, whose modules skip themselves (and so any module
importing them) off CUDA.
"""

import json
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import force_first_config

DEVICE = "xpu"
HEAD_DIM = 128
REL_EXTENT = 1024

requires_xpu = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="Inkling2 XPU kernels"
)
# The descriptor and pointer paths of a tuned launch run the same tile only
# when the tuned launches take their first config; tuning may pick a
# different tile for each path.
requires_first_config = pytest.mark.skipif(
    not force_first_config.is_installed(),
    reason="compares two paths at one tile: needs VLLM_TRITON_FORCE_FIRST_CONFIG=1",
)


def device_graph():
    """An XPU graph and the context manager that captures it."""
    graph = torch.xpu.XPUGraph()
    return graph, torch.xpu.graph(graph)


def shear_rel_logits(rel_logits, seqs, tile, is_local):
    """Torch reference of the sheared layout `attention_prologue` writes.

    Token `i` of a request with `past` cached keys sits in a tile of `tile`
    query tokens whose key blocks end at `n_max`; column `c` holds key
    `c + (n_max - 10) * 128`, padded with -inf right of the diagonal and
    -inf (window) or 0 past the extent (also for keys before the sequence,
    which no kernel reads).
    """
    tokens, heads, extent = rel_logits.shape
    out = rel_logits.new_full((tokens + 128, heads, extent + 256), float("nan"))
    cols = torch.arange(extent + 256, device=rel_logits.device)
    start = 0
    for q_len, kv_len in seqs:
        idx = torch.arange(q_len, device=rel_logits.device)
        pos = kv_len - q_len + idx
        tile_end = ((idx // tile + 1) * tile).clamp(max=q_len)
        n_max = (kv_len - q_len + tile_end + 127) // 128
        dist = pos[:, None] - (cols[None] + (n_max[:, None] - 10) * 128)
        valid = (dist >= 0) & (dist < extent)
        rel = rel_logits[start : start + q_len]
        vals = rel.gather(2, dist.clamp(0, extent - 1)[:, None].expand(-1, heads, -1))
        pad = torch.where((dist < 0) | is_local, float("-inf"), 0.0)
        pad = pad.to(rel_logits.dtype)
        out[start : start + q_len] = torch.where(valid[:, None], vals, pad[:, None])
        start += q_len
    return out


def ref_attention(
    q, key_cache, value_cache, rel_logits, seqs, block_table, scale, window_left
):
    """fp32 output and natural-log LSE `(T, H)` of the scale-after-add
    relative-bias attention::

        score(i, j, h) = (q[i, h] . k[j, h] + rel_bias(i, j, h) / scale) * scale
        rel_bias(i, j, h) = rel_logits[i, h, i - j] if 0 <= i - j < extent else 0
    """
    num_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    start = 0
    for i, (q_len, kv_len) in enumerate(seqs):
        blocks = block_table[i].long()
        k, v = (
            cache[blocks]
            .flatten(0, 1)[:kv_len]
            .float()
            .repeat_interleave(num_heads // num_kv_heads, dim=1)
            for cache in (key_cache, value_cache)
        )
        qpos = torch.arange(kv_len - q_len, kv_len, device=q.device)
        dist = qpos[:, None] - torch.arange(kv_len, device=q.device)[None]
        rel = rel_logits[start : start + q_len].float().transpose(0, 1)
        idx = dist.clamp(0, REL_EXTENT - 1).expand(num_heads, -1, -1)
        bias = rel.gather(2, idx).masked_fill(~((dist >= 0) & (dist < REL_EXTENT)), 0)
        qk = torch.einsum("qhd,khd->hqk", q[start : start + q_len].float(), k)
        scores = (qk + bias / scale) * scale
        mask = dist < 0
        if window_left is not None:
            mask |= dist > window_left
        scores.masked_fill_(mask, float("-inf"))
        lse[start : start + q_len] = scores.logsumexp(-1).T
        out[start : start + q_len] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
        start += q_len
    return out, lse


def clip(x, threshold, eps=1e-6):
    """L2 clipping of the last dim to `threshold`."""
    norm = x.norm(dim=-1, keepdim=True)
    return x * (threshold / (norm + eps)).clamp(max=1.0)


def dense_conv(x, weight, dyn):
    """Torch reference of the short conv over whole sequences `x` `(L, C)`:
    static `(C, W)` taps, or with `dyn` `(L, H, W)` the `(W, C)` static
    part plus per-token per-head deltas."""
    width = weight.shape[0] if dyn is not None else weight.shape[1]
    padded = torch.cat([x.new_zeros(width - 1, x.shape[1]), x]).float()
    acc = torch.zeros_like(x, dtype=torch.float32)
    for lag in range(width):
        value = padded[width - 1 - lag : width - 1 - lag + x.shape[0]]
        if dyn is None:
            w = weight[:, width - 1 - lag].float()
        else:
            head_dim = x.shape[1] // dyn.shape[1]
            w = weight[lag].float() + dyn[:, :, lag].float().repeat_interleave(
                head_dim, dim=1
            )
        acc += value * w
    scale = (width * (2 if dyn is not None else 1) + 1) ** -0.5
    return ((acc + x.float()) * scale).to(x.dtype)


ConvMeta = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# Tokens per request (4 requests) of each engine step: chunked prefill, then
# decode.
STEPS = [[3, 7, 2, 9], [2, 6, 0, 0], [1, 1, 1, 1]]
# Steps of over 256 tokens take the prefill tiling.
LONG_STEPS = [[3, 150, 2, 140], [200, 6, 90, 0], [1, 1, 1, 1]]
# A long step ending in a one-token request that continues earlier context:
# with padding, its tile ends in padding rows (request start 0).
PADDED_STEPS = [[120, 100, 3, 2], [120, 0, 138, 1]]


def chunked_steps(
    lengths_steps, block_table, block
) -> Iterator[tuple[Any, Any, Any, ConvMeta]]:
    """Per-step conv metadata `(positions, block table, seq_idx, slots,
    query start)` for chunked prefill + decode, with the requests and rows
    of the step."""
    done = [0] * len(lengths_steps[0])
    for step in lengths_steps:
        reqs = [i for i in range(len(step)) if step[i]]
        rows = [torch.arange(done[i], done[i] + step[i], device=DEVICE) for i in reqs]
        starts = torch.tensor([0, *[step[i] for i in reqs]]).cumsum(0)
        seq_idx = torch.repeat_interleave(
            torch.arange(len(reqs)), torch.tensor([step[i] for i in reqs])
        )
        slots = torch.cat(
            [
                block_table[i][r // block].long() * block + r % block
                for i, r in zip(reqs, rows)
            ]
        )
        meta = (
            torch.cat(rows),
            block_table[reqs],
            seq_idx.int().to(DEVICE),
            slots,
            starts[seq_idx].int().to(DEVICE),
        )
        yield reqs, rows, starts, meta
        for i in reqs:
            done[i] += step[i]


def pad_rows(x: torch.Tensor, meta: ConvMeta, rows: int) -> tuple[Any, ConvMeta]:
    """Append `rows` padding rows: position 0, request 0, no slot, start 0."""
    pos, table, seq_idx, slots, qstart = meta
    zeros = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
    meta = (
        torch.cat([pos, zeros.to(pos.dtype)]),
        table,
        torch.cat([seq_idx, zeros]),
        torch.cat([slots, torch.full_like(zeros, -1).to(slots.dtype)]),
        torch.cat([qstart, zeros]),
    )
    return torch.cat([x, x.new_zeros(rows, x.shape[1])]), meta


def conv_cache_and_table(heads, block, row, blocks_per_req=8):
    """A zero paged conv-state cache and a random block table of 4 requests
    (block 0 unused)."""
    num_blocks = 4 * blocks_per_req
    cache = torch.zeros(2 * num_blocks, heads, block, row, device=DEVICE).bfloat16()
    perm = torch.randperm(2 * num_blocks - 1, device=DEVICE)[:num_blocks] + 1
    return cache, perm.view(4, blocks_per_req).int()


# e2m1 values of the 16 FP4 codes.
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def dequant_nvfp4(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 `[..., K]` of packed e2m1 `values` `[..., K / 2]` (low nibble
    first) times their e4m3 block `scales` `[..., K / 16]`."""
    lut = torch.tensor(E2M1, device=values.device)
    v = torch.stack([lut[(values & 0xF).long()], lut[(values >> 4).long()]], -1)
    return v.flatten(-2) * scales.float().repeat_interleave(16, -1)


def dequant_mxfp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 `[..., K]` of e4m3 `values` times their e8m0 block `scales`
    `[..., K / 32]` (uint8)."""
    exp2 = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    return values.float() * exp2


# A tiny Inkling2 text model (16 routed experts, no quantization config, so
# NVFP4 experts); the configs of `tests/models/inkling2/test_model.py`.
TINY_CONFIG: dict[str, Any] = {
    "architectures": ["Inkling2ForConditionalGeneration"],
    "model_type": "inkling2",
    "eos_token_id": 1,
    "text_config": {
        "torch_dtype": "bfloat16",
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "local_layer_ids": [0],
        "sliding_window_size": 128,
        "vocab_size": 1024,
        "unpadded_vocab_size": 1000,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "n_routed_experts": 16,
        "num_experts_per_tok": 8,
        "intermediate_size": 256,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
    },
    "mtp_config": {
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1e6,
        "max_position_embeddings": 4096,
        "local_layer_ids": [0],
        "sliding_window_size": 128,
        "block_size": 8,
        "markov_rank": 32,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [1, 2],
    },
}

# A looped variant: layer 0 is the prelude, layers 1-2 (one local, one full)
# run three passes, layer 3 is the coda; one-layer AttnRes blocks make every
# layer a block, so later passes read a two-block message bank. Hidden size
# 2304 (Inkling 2 Mini's).
TINY_LOOP_CONFIG: dict[str, Any] = {
    **TINY_CONFIG,
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "text_config": {
        **TINY_CONFIG["text_config"],
        "hidden_size": 2304,
        "num_hidden_layers": 4,
        "local_layer_ids": [0, 1, 3],
        "n_layers_per_attn_res_block": 1,
        "loop_config": {
            "core_start": 1,
            "core_end": 3,
            "num_passes": 3,
            "gate_rank": 16,
            "is_varloop": True,
        },
    },
    "mtp_config": {
        **TINY_CONFIG["mtp_config"],
        "hidden_size": 2304,
        "target_layer_ids": [3, 8],
    },
}


def write_config(path, config: dict[str, Any]) -> str:
    """Write `config` as the `config.json` of model directory `path`."""
    (path / "config.json").write_text(json.dumps(config))
    return str(path)
