# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MOE align block size function.

Run `pytest tests/kernels/moe/test_moe_align_block_size.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    batched_moe_align_block_size,
    moe_align_block_size,
)
from vllm.platforms import current_platform
from vllm.utils.kernel_probe import set_kernel_usable
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import set_random_seed

NUM_TOKENS = [1, 3, 256, 2256, 4096]
NUM_EXPERTS = [32, 160, 256, 257]
TOP_KS = [1, 2, 16, 32]
BLOCK_SIZES = [32, 128]
set_random_seed(0)


@pytest.mark.parametrize("num_experts,num_tokens", [(32, 8), (256, 8), (256, 1024)])
@pytest.mark.parametrize("use_expert_map", [False, True])
def test_moe_align_scatter_masks_invalid_routes(
    num_experts, num_tokens, use_expert_map
):
    topk_ids = torch.arange(num_tokens * 4, device="cuda").reshape(-1, 4)
    topk_ids %= num_experts
    topk_ids[0] = torch.tensor([-1, num_experts, 0, 1], device="cuda")
    expert_map = None
    valid = (topk_ids >= 0) & (topk_ids < num_experts)
    if use_expert_map:
        expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
        expert_map[::2] = torch.arange(
            num_experts // 2, device="cuda", dtype=torch.int32
        )
        valid &= topk_ids % 2 == 0

    sorted_ids, expert_ids, num_padded, scatter_idx = moe_align_block_size(
        topk_ids,
        32,
        num_experts,
        expert_map,
        ignore_invalid_experts=True,
        return_scatter_idx=True,
    )
    expected = torch.arange(topk_ids.numel(), device="cuda", dtype=torch.int32)
    expected[~valid.flatten()] = -1
    torch.testing.assert_close(scatter_idx, expected[:, None])
    active_sorted = sorted_ids[: num_padded.item()]
    active_sorted = active_sorted[active_sorted < topk_ids.numel()]
    torch.testing.assert_close(active_sorted.sort().values, expected[expected >= 0])

    baseline = moe_align_block_size(
        topk_ids, 32, num_experts, expert_map, ignore_invalid_experts=True
    )
    assert len(baseline) == 3
    torch.testing.assert_close(num_padded, baseline[2])
    torch.testing.assert_close(expert_ids, baseline[1])


def _group_tokens_by_expert(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    valid_length: int,
    total_tokens: int,
) -> dict:
    num_blocks = valid_length // block_size
    expert_tokens: dict[int, list[int]] = {}

    for block_idx in range(num_blocks):
        expert_id = expert_ids[block_idx].item()
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, valid_length)

        block_tokens = sorted_ids[block_start:block_end]
        valid_tokens = block_tokens[block_tokens < total_tokens]

        if expert_id not in expert_tokens:
            expert_tokens[expert_id] = []
        expert_tokens[expert_id].extend(valid_tokens.tolist())
    return expert_tokens


def _verify_expert_level_sorting(
    actual_sorted_ids: torch.Tensor,
    golden_sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    valid_length: int,
    total_tokens: int,
):
    """Verify that actual_sorted_ids follows the correct expert-level sorting.
    The kerne limplementation may or may not preserve original token order
    in topk_ids in the final sorted_ids however this does not impact quality.
    """
    # Group tokens by expert from the golden implementation
    golden_expert_tokens = _group_tokens_by_expert(
        golden_sorted_ids, expert_ids, block_size, valid_length, total_tokens
    )

    actual_expert_tokens = _group_tokens_by_expert(
        actual_sorted_ids, expert_ids, block_size, valid_length, total_tokens
    )

    assert set(golden_expert_tokens.keys()) == set(actual_expert_tokens.keys()), (
        f"Expert IDs mismatch: golden={set(golden_expert_tokens.keys())}, "
        f"actual={set(actual_expert_tokens.keys())}"
    )

    for expert_id in golden_expert_tokens:
        golden_tokens = torch.tensor(
            golden_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        actual_tokens = torch.tensor(
            actual_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        assert torch.equal(
            torch.sort(golden_tokens)[0], torch.sort(actual_tokens)[0]
        ), (
            f"Expert {expert_id} token mismatch: "
            f"golden={golden_expert_tokens[expert_id]}, "
            f"actual={actual_expert_tokens[expert_id]}"
        )


def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Golden torch implementation of moe_align_block_size.

    This function aligns the token distribution across experts to be compatible
    with block size for matrix multiplication by sorting tokens by expert and
    padding to block boundaries.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = topk_ids.numel() * block_size

    flattened_token_indices = torch.arange(
        topk_ids.numel(), device=topk_ids.device, dtype=torch.int32
    )
    flattened_expert_ids = topk_ids.flatten()
    sorted_expert_ids, sort_indices = torch.sort(flattened_expert_ids, stable=True)
    sorted_token_indices = flattened_token_indices[sort_indices]

    expert_token_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        mask = sorted_expert_ids == expert_id
        expert_token_counts[expert_id] = mask.sum()

    expert_padded_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        original_count = expert_token_counts[expert_id]
        if expert_map is not None and expert_map[expert_id] == -1:
            continue
        if original_count > 0:
            expert_padded_counts[expert_id] = (
                (original_count + block_size - 1) // block_size
            ) * block_size

    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.full(
        (max_num_blocks,), -1, dtype=torch.int32, device=topk_ids.device
    )

    current_pos = 0
    current_block = 0
    for expert_id in range(num_experts):
        if expert_map is not None and expert_map[expert_id] == -1:
            continue

        expert_mask = sorted_expert_ids == expert_id
        expert_tokens = sorted_token_indices[expert_mask]
        num_expert_tokens = expert_tokens.shape[0]

        if num_expert_tokens > 0:
            sorted_token_ids[current_pos : current_pos + num_expert_tokens] = (
                expert_tokens
            )

            expert_blocks_needed = expert_padded_counts[expert_id] // block_size

            expert_id_new = expert_id
            if expert_map is not None:
                expert_id_new = expert_map[expert_id]
            expert_ids[current_block : current_block + expert_blocks_needed] = (
                expert_id_new
            )

            current_pos += expert_padded_counts[expert_id]
            current_block += expert_blocks_needed

    total_padded_tokens = expert_padded_counts.sum()
    num_tokens_post_pad = torch.tensor(
        [total_padded_tokens], dtype=torch.int32, device=topk_ids.device
    )

    return sorted_token_ids, expert_ids, num_tokens_post_pad


def test_moe_align_block_size_rejects_noncontiguous_topk_ids():
    wide_topk_ids = torch.arange(16, device="cuda", dtype=torch.int32).reshape(2, 8) % 4
    topk_ids = wide_topk_ids[:, :4]
    assert not topk_ids.is_contiguous()

    with pytest.raises(RuntimeError, match="topk_ids must be contiguous"):
        moe_align_block_size(topk_ids, block_size=4, num_experts=4)


@pytest.mark.parametrize("m", NUM_TOKENS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("pad_sorted_ids", [False, True])
def test_moe_align_block_size(
    m: int, topk: int, num_experts: int, block_size: int, pad_sorted_ids: bool
):
    """Test moe_align_block_size without expert mapping."""
    topk_ids = torch.zeros((m, topk), device="cuda", dtype=torch.int32)
    for i in range(m):
        experts = torch.randperm(num_experts, device="cuda")[:topk]
        topk_ids[i] = experts

    actual_sorted_ids, actual_expert_ids, actual_num_tokens = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        pad_sorted_ids=pad_sorted_ids,
    )
    golden_sorted_ids, golden_expert_ids, golden_num_tokens = (
        torch_moe_align_block_size(
            topk_ids=topk_ids,
            block_size=block_size,
            num_experts=num_experts,
            pad_sorted_ids=pad_sorted_ids,
        )
    )

    torch.testing.assert_close(actual_num_tokens, golden_num_tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual_expert_ids, golden_expert_ids, atol=0, rtol=0)

    # For sorted_token_ids, verify block-level correctness rather than exact
    # order Tokens within each expert's blocks can be in any order, but expert
    # regions must be correct
    _verify_expert_level_sorting(
        actual_sorted_ids,
        golden_sorted_ids,
        actual_expert_ids,
        block_size,
        actual_num_tokens.item(),
        m * topk,
    )

    total_tokens = m * topk
    assert actual_num_tokens.item() % block_size == 0, (
        "num_tokens_post_pad should be divisible by block_size"
    )
    assert actual_num_tokens.item() >= total_tokens, (
        "num_tokens_post_pad should be at least total_tokens"
    )
    valid_tokens = actual_sorted_ids[actual_sorted_ids < total_tokens]
    assert len(valid_tokens) == total_tokens, (
        f"Should have exactly {total_tokens} valid tokens, got {len(valid_tokens)}"
    )
    actual_num_blocks = cdiv(int(actual_num_tokens.item()), block_size)
    assert (actual_expert_ids[:actual_num_blocks] >= 0).all() and (
        actual_expert_ids[:actual_num_blocks] < num_experts
    ).all(), "expert_ids should contain valid expert indices"


@pytest.mark.parametrize("m", [16, 32, 2048])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("num_experts", [8, 64])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("mask_inactive_experts", [False, True])
def test_moe_align_block_size_with_expert_map(
    m: int,
    topk: int,
    num_experts: int,
    block_size: int,
    mask_inactive_experts: bool,
):
    """Test moe_align_block_size with expert mapping (EP scenario)."""
    expert_map = torch.full((num_experts,), -1, device="cuda", dtype=torch.int32)
    local_experts = list(range(0, num_experts, 2))
    for i, expert_id in enumerate(local_experts):
        expert_map[expert_id] = i

    topk_ids = torch.empty((m, topk), device="cuda", dtype=torch.int32)
    for i in range(m):
        experts = torch.randperm(num_experts, device="cuda")[:topk]
        for k in range(topk):
            topk_ids[i, k] = (
                experts[k]
                if (experts[k] in local_experts) or not mask_inactive_experts
                else -1
            )
    topk_ids[0, 0] = -1

    actual_sorted_ids, actual_expert_ids, actual_num_tokens = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        expert_map=expert_map,
        ignore_invalid_experts=True,
    )
    golden_sorted_ids, golden_expert_ids, golden_num_tokens = (
        torch_moe_align_block_size(
            topk_ids=topk_ids,
            block_size=block_size,
            num_experts=num_experts,
            expert_map=expert_map,
        )
    )

    torch.testing.assert_close(actual_num_tokens, golden_num_tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual_expert_ids, golden_expert_ids, atol=0, rtol=0)
    _verify_expert_level_sorting(
        actual_sorted_ids,
        golden_sorted_ids,
        actual_expert_ids,
        block_size,
        actual_num_tokens.item(),
        m * topk,
    )


def test_moe_align_block_size_deterministic():
    m, topk, num_experts, block_size = 128, 2, 32, 64

    torch.manual_seed(42)
    topk_ids = torch.randint(
        0, num_experts, (m, topk), device="cuda", dtype=torch.int32
    )

    # expect the results to be reproducible
    results = []
    for _ in range(5):
        sorted_ids, expert_ids, num_tokens = moe_align_block_size(
            topk_ids=topk_ids, block_size=block_size, num_experts=num_experts
        )
        results.append((sorted_ids.clone(), expert_ids.clone(), num_tokens.clone()))

    for i in range(1, len(results)):
        assert torch.equal(results[0][0], results[i][0]), (
            "sorted_ids should be deterministic"
        )
        assert torch.equal(results[0][1], results[i][1]), (
            "expert_ids should be deterministic"
        )
        assert torch.equal(results[0][2], results[i][2]), (
            "num_tokens should be deterministic"
        )


@pytest.mark.parametrize("max_tokens_per_batch", [13, 16, 512])
@pytest.mark.parametrize("num_experts", [8, 16, 32, 64])
@pytest.mark.parametrize("block_size", [8, 16, 32, 48, 64])
@pytest.mark.parametrize(
    "token_distribution", ["random", "alternating_empty", "skewed"]
)
def test_batched_moe_align_block_size(
    max_tokens_per_batch: int,
    num_experts: int,
    block_size: int,
    token_distribution: str,
):
    def ref_outputs(
        expert_num_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E = expert_num_tokens.size(0)

        # Round up so each batch can be split to blocks evenly.
        Msum = round_up(max_tokens_per_batch, block_size) * E
        ref_sorted_ids = torch.empty((Msum,), dtype=torch.int32)
        ref_expert_ids = torch.empty((Msum // block_size,), dtype=torch.int32)
        ref_num_tokens_post_pad = torch.empty((1,), dtype=torch.int32)

        # Initialize
        sentinel = E * max_tokens_per_batch
        ref_sorted_ids.fill_(sentinel)
        ref_expert_ids.fill_(-1)

        # Fill ref_sorted_ids
        i = 0
        for expert_id, expert_nt in enumerate(expert_num_tokens):
            token_offset = expert_id * max_tokens_per_batch
            for j in range(expert_nt):
                ref_sorted_ids[i] = token_offset + j
                i += 1
            # round up i to the next block_size
            i = round_up(i, block_size)

        ref_num_tokens_post_pad[0] = i

        # Fill expert_ids
        nt_ceil_sum = 0
        for expert_id, expert_nt in enumerate(expert_num_tokens):
            expert_ids_offset = nt_ceil_sum // block_size
            ceil_expert_nt = round_up(int(expert_nt.item()), block_size)
            num_blocks = ceil_expert_nt // block_size
            for x in range(num_blocks):
                ref_expert_ids[expert_ids_offset + x] = expert_id
            nt_ceil_sum += ceil_expert_nt

        return (
            ref_sorted_ids.to("cuda"),
            ref_expert_ids.to("cuda"),
            ref_num_tokens_post_pad.to("cuda"),
        )

    # Compute expert_num_tokens
    expert_num_tokens = torch.randint(
        low=0,
        high=max_tokens_per_batch + 1,
        size=(num_experts,),
        device="cpu",
        dtype=torch.int32,
    )
    if token_distribution == "alternating_empty":
        expert_num_tokens[::2] = 0
    elif token_distribution == "skewed":
        expert_num_tokens.zero_()
        expert_num_tokens[0] = max_tokens_per_batch
        expert_num_tokens[-1] = 1

    # ref outputs
    ref_sorted_ids, ref_expert_ids, ref_num_tokens_post_pad = ref_outputs(
        expert_num_tokens
    )

    # outputs
    sorted_ids, expert_ids, num_tokens_post_pad = batched_moe_align_block_size(
        max_tokens_per_batch, block_size, expert_num_tokens.to("cuda")
    )

    assert ref_sorted_ids.size() == sorted_ids.size(), (
        f"{ref_sorted_ids.size()} vs {sorted_ids.size()}"
    )
    assert ref_expert_ids.size() == expert_ids.size(), (
        f"{ref_expert_ids.size()} vs {expert_ids.size()}"
    )
    assert ref_num_tokens_post_pad.size() == num_tokens_post_pad.size(), (
        f"{ref_num_tokens_post_pad.size()} vs {num_tokens_post_pad.size()}"
    )
    torch.testing.assert_close(ref_sorted_ids, sorted_ids, atol=0, rtol=0)
    torch.testing.assert_close(ref_expert_ids, expert_ids, atol=0, rtol=0)
    torch.testing.assert_close(
        ref_num_tokens_post_pad, num_tokens_post_pad, atol=0, rtol=0
    )


def _align_buffers(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> list[torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    return [
        topk_ids.new_empty(max_num_tokens_padded, dtype=torch.int32),
        topk_ids.new_empty(cdiv(max_num_tokens_padded, block_size), dtype=torch.int32),
        topk_ids.new_empty(1, dtype=torch.int32),
    ]


def _random_topk_ids(
    m: int, topk: int, num_experts: int, device: str, with_expert_map: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:topk] for _ in range(m)]
    ).to(torch.int32)
    topk_ids[0, 0] = -1
    if not with_expert_map:
        return topk_ids, None
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    expert_map[::2] = torch.arange(
        cdiv(num_experts, 2), dtype=torch.int32, device=device
    )
    return topk_ids, expert_map


@pytest.mark.parametrize("m", [1, 3, 256, 999])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("num_experts", [32, 257])
@pytest.mark.parametrize("block_size", [32, 128])
@pytest.mark.parametrize("with_expert_map", [False, True])
def test_moe_align_block_size_torch_matches_golden(
    m: int, topk: int, num_experts: int, block_size: int, with_expert_map: bool
):
    """The torch implementation of the op against the golden reference."""
    from vllm._custom_ops import _moe_align_block_size_torch

    topk_ids, expert_map = _random_topk_ids(
        m, topk, num_experts, "cpu", with_expert_map
    )
    sorted_ids, expert_ids, num_tokens = _align_buffers(
        topk_ids, block_size, num_experts
    )
    _moe_align_block_size_torch(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens,
        expert_map,
    )
    golden = torch_moe_align_block_size(
        topk_ids, block_size, num_experts, expert_map=expert_map
    )
    # Both keep the routes of an expert in their original order.
    torch.testing.assert_close(num_tokens, golden[2], atol=0, rtol=0)
    torch.testing.assert_close(expert_ids, golden[1], atol=0, rtol=0)
    torch.testing.assert_close(sorted_ids, golden[0], atol=0, rtol=0)


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU-only fallback")
@pytest.mark.parametrize("m", [1, 33, 2048])
@pytest.mark.parametrize("topk", [2, 8])
@pytest.mark.parametrize("num_experts", [64, 256])
@pytest.mark.parametrize("with_expert_map", [False, True])
def test_moe_align_block_size_torch_matches_native_xpu(
    m: int, topk: int, num_experts: int, with_expert_map: bool
):
    from vllm import _custom_ops as ops

    if not ops._moe_align_block_size_usable():
        pytest.skip("_moe_C.moe_align_block_size is not usable on this device")
    block_size = 64
    topk_ids, expert_map = _random_topk_ids(
        m, topk, num_experts, "xpu", with_expert_map
    )
    native = _align_buffers(topk_ids, block_size, num_experts)
    portable = _align_buffers(topk_ids, block_size, num_experts)
    torch.ops._moe_C.moe_align_block_size(
        topk_ids, num_experts, block_size, *native, expert_map
    )
    ops._moe_align_block_size_torch(
        topk_ids,
        num_experts,
        block_size,
        portable[0],
        portable[1],
        portable[2],
        expert_map,
    )

    torch.testing.assert_close(portable[2], native[2], atol=0, rtol=0)
    # Blocks past num_tokens_post_pad hold -1 here (as in the CUDA op) but 0
    # in the XPU op; no consumer reads them, the fused MoE kernels return for
    # `pid_m * BLOCK_SIZE_M >= num_tokens_post_padded` before loading
    # expert_ids, so compare the used blocks only.
    num_blocks = native[2].item() // block_size
    torch.testing.assert_close(
        portable[1][:num_blocks], native[1][:num_blocks], atol=0, rtol=0
    )
    assert (portable[1][num_blocks:] == -1).all()
    # The native op does not fix the order of routes within an expert.
    _verify_expert_level_sorting(
        portable[0],
        native[0],
        native[1],
        block_size,
        native[2].item(),
        topk_ids.numel(),
    )
    num_padded = native[2].item()
    assert (portable[0][num_padded:] == topk_ids.numel()).all()
    assert (native[0][num_padded:] == topk_ids.numel()).all()


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU-only fallback")
@pytest.mark.parametrize("with_expert_map", [False, True])
def test_moe_align_block_size_forced_fallback_xpu(monkeypatch, with_expert_map: bool):
    from vllm import _custom_ops as ops

    calls = 0
    torch_impl = ops._moe_align_block_size_torch

    def counting_impl(*args, **kwargs):
        nonlocal calls
        calls += 1
        return torch_impl(*args, **kwargs)

    monkeypatch.setattr(ops, "_moe_align_block_size_torch", counting_impl)
    set_kernel_usable("_moe_C.moe_align_block_size", False)
    try:
        topk_ids, expert_map = _random_topk_ids(256, 8, 64, "xpu", with_expert_map)
        actual = moe_align_block_size(
            topk_ids, 64, 64, expert_map, ignore_invalid_experts=True
        )
    finally:
        set_kernel_usable("_moe_C.moe_align_block_size", None)

    assert calls == 1
    golden = torch_moe_align_block_size(topk_ids, 64, 64, expert_map=expert_map)
    for got, expected in zip(actual, golden):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
