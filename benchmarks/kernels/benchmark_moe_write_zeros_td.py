# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: TD-scatter vs pointer-store for write_zeros_to_output
(the fused-MoE zero-fill store for EP-pruned expert blocks).

Per AGENTS.md, kernel perf work belongs in benchmarks/kernels/, not tests/.
This script also makes the write_zeros_to_output TD-vs-pointer numbers
independently reproducible: an earlier review round on this PR flagged that
the original "TD path is slower at 3 of 4 sizes" claim came from a
task-local script not checked into the PR, so it couldn't be re-run by a
reviewer. This is the checked-in replacement, used to justify
resolve_moe_write_zeros_use_td()'s default-off gating.

Isolates write_zeros_to_output via a thin @triton.jit launcher (no GEMM, no
fused_moe() overhead) with a synthetic sorted_token_ids array padded the
same way moe_align_block_size() pads it: padding sentinel = num_valid_tokens,
dropped by token_mask (pointer path) / the descriptor row bound (TD path).

Run inside the vLLM container:
    python3 benchmarks/kernels/benchmark_moe_write_zeros_td.py [--repeats 3]
"""

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_default_config,
    write_zeros_to_output,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type  # "cuda" (also reported by ROCm) or "xpu"

# (m, n, k): first row matches test_fused_moe_all_experts_pruned exactly. The
# rest are real model (m, n) pairs -- m = small decode-like vs large EP-batch
# token count, n = B.size(1) in fused_moe_kernel (either the gate_up
# projection width 2 * intermediate_size, or the down projection width
# hidden_size), per benchmark_moe.py's get_model_params() convention:
#   - Mixtral-8x7B: hidden=4096, intermediate=14336
#   - DeepSeek-V3:  hidden=7168, moe_intermediate=2048
# k has no effect on this kernel's cost (no arithmetic, pure zero store) --
# kept only as some value for label/config-lookup parity with the unit
# test, using n's model's hidden_size. E/TOPK below are likewise fixed
# (Mixtral-style) across all rows -- only get_default_config's tuning
# heuristics consume them, not the benchmarked store itself.
PROBLEM_SIZES = [
    (83, 512, 256),
    (32, 4096, 4096),  # Mixtral-8x7B down-proj, small decode-like batch
    (2048, 28672, 4096),  # Mixtral-8x7B gate_up-proj, EP-realistic batch
    (8192, 2048, 7168),  # DeepSeek-V3 gate_up-proj, large EP batch
]
E, TOPK = 8, 2  # only feeds get_default_config's tuning heuristics


@triton.jit
def _write_zeros_launcher(
    c_ptr,
    stride_cm,
    stride_cn,
    N,
    sorted_token_ids_ptr,
    num_valid_tokens,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    USE_TD: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    write_zeros_to_output(
        c_ptr,
        stride_cm,
        stride_cn,
        pid_n,
        N,
        offs_token,
        token_mask,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        tl.bfloat16,
        num_valid_tokens,
        USE_TD=USE_TD,
    )


def _make_inputs(m, n, block_m, device):
    em = triton.cdiv(m, block_m) * block_m  # padded, like moe_align_block_size
    sorted_token_ids = torch.arange(em, device=device, dtype=torch.int64)
    sorted_token_ids = torch.where(sorted_token_ids < m, sorted_token_ids, m)
    c = torch.full((m + 1, n), 7.0, device=device, dtype=torch.bfloat16)
    return c, sorted_token_ids, em


def bench_one(m, n, k, use_td):
    set_random_seed(42)
    config = get_default_config(m, E, n, k, TOPK, dtype=None)
    block_m, block_n = config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"]
    c, sorted_token_ids, em = _make_inputs(m, n, block_m, DEVICE)
    grid = (triton.cdiv(em, block_m), triton.cdiv(n, block_n))

    def run():
        _write_zeros_launcher[grid](
            c,
            c.stride(0),
            c.stride(1),
            n,
            sorted_token_ids,
            m,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            USE_TD=use_td,
        )

    ms, _, _ = triton.testing.do_bench(run, quantiles=[0.5, 0.2, 0.8])
    return ms * 1000  # ms -> us


def main():
    parser = FlexibleArgumentParser(
        description="Benchmark write_zeros_to_output: TD scatter vs pointer store."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="process-level repeats; report median-of-medians",
    )
    args = parser.parse_args()

    has_td = hasattr(tl, "make_tensor_descriptor")
    if not has_td:
        print(
            "Triton lacks tl.make_tensor_descriptor -- TD path unavailable, "
            "reporting pointer-path only."
        )

    print(f"device={DEVICE}")
    print(f"{'m':>6} {'n':>6} {'k':>6} {'pointer_us':>11} {'td_us':>11} {'delta%':>8}")
    for m, n, k in PROBLEM_SIZES:
        offs, ons = [], []
        for _ in range(args.repeats):
            offs.append(bench_one(m, n, k, use_td=False))
            if has_td:
                ons.append(bench_one(m, n, k, use_td=True))
        off_med = sorted(offs)[len(offs) // 2]
        if ons:
            on_med = sorted(ons)[len(ons) // 2]
            delta = (on_med - off_med) / off_med * 100
            print(
                f"{m:>6} {n:>6} {k:>6} {off_med:>11.2f} {on_med:>11.2f} {delta:>+7.2f}%"
            )
        else:
            print(f"{m:>6} {n:>6} {k:>6} {off_med:>11.2f} {'n/a':>11} {'n/a':>8}")


if __name__ == "__main__":
    main()
