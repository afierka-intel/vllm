# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm._custom_ops import (
    merge_attn_states as merge_attn_states_cuda,
)
from vllm._custom_ops import (
    scaled_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_merge_attn_states import (
    merge_attn_states as merge_attn_states_triton,
)


# Naive PyTorch Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
# can be used to combine partial attention results (in the split-KV case)
def merge_attn_states_torch(
    output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse: torch.Tensor,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse: torch.Tensor,  # [NUM_HEADS, NUM_TOKENS]
    output_lse: torch.Tensor | None = None,  # [NUM_HEADS, NUM_TOKENS]
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,  # scalar, per-tensor FP8 scale
):
    # Apply prefill_tokens_with_context mask if needed
    if prefill_tokens_with_context is None:
        prefill_tokens_with_context = output.shape[0]
    p_lse = prefix_lse
    s_lse = suffix_lse
    # inf -> -inf
    p_lse[p_lse == torch.inf] = -torch.inf
    s_lse[s_lse == torch.inf] = -torch.inf
    # max_lse [NUM_HEADS, NUM_TOKENS]
    max_lse = torch.maximum(p_lse, s_lse)

    mask = torch.ones((prefix_lse.shape[1], 1, 1), device=p_lse.device)
    mask[prefill_tokens_with_context:].fill_(0)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_lse_exp = torch.exp(p_lse)
    s_lse_exp = torch.exp(s_lse)
    out_se = p_lse_exp + s_lse_exp
    if output_lse is not None:
        output_lse = torch.log(out_se) + max_lse
        output_lse[prefill_tokens_with_context:] = suffix_lse[
            prefill_tokens_with_context:
        ]
    p_scale = p_lse_exp / out_se  # [NUM_HEADS, NUM_TOKENS]
    s_scale = s_lse_exp / out_se  # [NUM_HEADS, NUM_TOKENS]
    p_scale = torch.transpose(p_scale, 0, 1).unsqueeze(2)  # [NUM_TOKENS, NUM_HEADS, 1]
    s_scale = torch.transpose(s_scale, 0, 1).unsqueeze(2)  # [NUM_TOKENS, NUM_HEADS, 1]
    output = prefix_output * p_scale * mask + suffix_output * (
        s_scale * mask + (1 - mask)
    )
    if output_scale is not None:
        shape = output.shape
        output, _ = scaled_fp8_quant(output.float().view(-1, shape[-1]), output_scale)
        output = output.view(shape)
    return output, output_lse


NUM_BATCH_TOKENS = [256, 512, 613, 1024, 1536, 4096]
NUM_QUERY_HEADS = [4, 8, 16, 32, 48, 64]
HEAD_SIZES = [32, 48, 64, 96, 128, 256]
DTYPES = [torch.float32, torch.half, torch.bfloat16]

all_case_info: list[tuple] = []


DEVICE = current_platform.device_type


@pytest.mark.parametrize("merge_fn", [merge_attn_states_cuda, merge_attn_states_triton])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.half, torch.bfloat16])
def test_merge_attn_states_both_empty(merge_fn, output_dtype) -> None:
    """When a token is empty on both sides (both LSE -inf), the 0/0 softmax
    scales must not surface as NaN in the merged output."""
    if merge_fn is merge_attn_states_cuda and not current_platform.is_cuda():
        pytest.skip("CUDA merge_attn_states custom op requires CUDA")
    num_tokens, num_heads, head_size = 6, 8, 128
    prefix_output = torch.zeros(
        num_tokens, num_heads, head_size, device=DEVICE, dtype=output_dtype
    )
    prefix_lse = torch.randn(num_heads, num_tokens, device=DEVICE)
    suffix_output = torch.zeros(
        num_tokens, num_heads, head_size, device=DEVICE, dtype=output_dtype
    )
    suffix_lse = torch.randn(num_heads, num_tokens, device=DEVICE)

    # Tokens 2 and 3 are empty on both sides.
    empty = slice(2, 4)
    prefix_lse[:, empty] = float("-inf")
    suffix_lse[:, empty] = float("-inf")

    output = torch.empty_like(prefix_output)
    merge_fn(output, prefix_output, prefix_lse, suffix_output, suffix_lse)

    assert not output.isnan().any()


class _LaunchSpy:
    """Records the USE_TD value of every merge_attn_states_kernel launch."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.use_td: list[bool] = []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            self.use_td.append(kwargs["USE_TD"])
            return launch(*args, **kwargs)

        return run


def _spy_kernel(monkeypatch) -> _LaunchSpy:
    import vllm.v1.attention.ops.triton_merge_attn_states as mod

    spy = _LaunchSpy(mod.merge_attn_states_kernel)
    monkeypatch.setattr(mod, "merge_attn_states_kernel", spy)
    return spy


def _merge_ref(prefix_output, prefix_lse, suffix_output, suffix_lse, n_ctx, scale):
    """fp32 reference with the kernel's inf / both-empty handling."""
    p_lse = torch.where(prefix_lse == torch.inf, -torch.inf, prefix_lse)
    s_lse = torch.where(suffix_lse == torch.inf, -torch.inf, suffix_lse)
    max_lse = torch.maximum(p_lse, s_lse)
    p_se = torch.exp(p_lse - max_lse)
    s_se = torch.exp(s_lse - max_lse)
    out_se = p_se + s_se
    empty = max_lse == -torch.inf
    out_lse = torch.where(empty, -torch.inf, torch.log(out_se) + max_lse)
    p_scale = (p_se / out_se).T.unsqueeze(-1)
    s_scale = (s_se / out_se).T.unsqueeze(-1)
    out = prefix_output.float() * p_scale + suffix_output.float() * s_scale
    out = torch.where(empty.T.unsqueeze(-1), 0.0, out)
    out[n_ctx:] = suffix_output[n_ctx:].float()
    out_lse[:, n_ctx:] = suffix_lse[:, n_ctx:]
    if scale is not None:
        fp8 = torch.finfo(current_platform.fp8_dtype())
        out = (out * (1.0 / scale)).clamp(fp8.min, fp8.max)
        out = out.to(current_platform.fp8_dtype()).float() * scale
    return out, out_lse


@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("prefill_tokens_with_context", [None, 37])
@pytest.mark.parametrize("out_pad", [0, 32])
@pytest.mark.parametrize("in_pad", [0, 64])
@pytest.mark.parametrize("head_size", [64, 96, 128])
@pytest.mark.parametrize("input_dtype", DTYPES)
@torch.inference_mode()
def test_merge_attn_states_triton_td(
    monkeypatch,
    input_dtype: torch.dtype,
    head_size: int,
    in_pad: int,
    out_pad: int,
    prefill_tokens_with_context: int | None,
    use_fp8: bool,
):
    """The Triton tensor-descriptor path must match the pointer path exactly,
    including head_stride > head_size (padded inputs/output), FP8 output, the
    no-context copy path and inf/-inf LSEs, and must not write the padding
    columns. Non-power-of-2 head sizes must fall back to the pointer path."""
    spy = _spy_kernel(monkeypatch)
    torch.manual_seed(0)
    num_tokens, num_heads = 67, 8
    n_ctx = prefill_tokens_with_context or num_tokens
    out_dtype = current_platform.fp8_dtype() if use_fp8 else input_dtype
    scale = 0.05
    output_scale = (
        torch.tensor([scale], dtype=torch.float32, device=DEVICE) if use_fp8 else None
    )

    def padded_randn(pad: int) -> torch.Tensor:
        shape = (num_tokens, num_heads, head_size + pad)
        return torch.randn(shape, device=DEVICE).to(input_dtype)[..., :head_size]

    prefix_output = padded_randn(in_pad)
    suffix_output = padded_randn(in_pad)
    prefix_lse = torch.randn(num_heads, num_tokens, device=DEVICE)
    suffix_lse = torch.randn(num_heads, num_tokens, device=DEVICE)
    prefix_lse[:, 1] = float("inf")  # FA2-style empty prefix
    suffix_lse[:, 2] = float("inf")  # FA2-style empty suffix
    prefix_lse[:, 3] = suffix_lse[:, 3] = float("-inf")  # both empty
    prefix_lse[:, 4] = float("-inf")

    results = []
    for use_td in (False, True):
        out_buf = torch.full(
            (num_tokens, num_heads, head_size + out_pad), 7.0, device=DEVICE
        ).to(out_dtype)
        output_lse = torch.empty(num_heads, num_tokens, device=DEVICE)
        merge_attn_states_triton(
            out_buf[..., :head_size],
            prefix_output,
            prefix_lse.clone(),
            suffix_output,
            suffix_lse.clone(),
            output_lse,
            prefill_tokens_with_context,
            output_scale,
            use_td=use_td,
        )
        assert (out_buf[..., head_size:].float() == 7.0).all(), "padding written"
        results.append((out_buf[..., :head_size].float(), output_lse))

    assert spy.use_td == [False, head_size in (64, 128)]
    (ptr_out, ptr_lse), (td_out, td_lse) = results
    torch.testing.assert_close(td_out, ptr_out, atol=0, rtol=0)
    torch.testing.assert_close(td_lse, ptr_lse, atol=0, rtol=0)

    ref_out, ref_lse = _merge_ref(
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        n_ctx,
        scale if use_fp8 else None,
    )
    if use_fp8:
        atol, rtol, td_out = 1.5e-1, 1.5e-1, td_out * scale
    elif input_dtype == torch.bfloat16:
        atol, rtol = 1e-3, 1e-2
    else:
        atol, rtol = 1e-3, 1e-3
    torch.testing.assert_close(td_out, ref_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(td_lse, ref_lse, atol=1e-3, rtol=1e-3)


def test_merge_attn_states_triton_td_is_opt_in(monkeypatch) -> None:
    """The TD path is taken only when requested: an unset VLLM_TRITON_USE_TD
    keeps the pointer path on every platform, including XPU."""
    spy = _spy_kernel(monkeypatch)
    shape = (16, 8, 128)
    prefix_output = torch.randn(shape, device=DEVICE, dtype=torch.half)
    suffix_output = torch.randn(shape, device=DEVICE, dtype=torch.half)
    prefix_lse = torch.randn(8, 16, device=DEVICE)
    suffix_lse = torch.randn(8, 16, device=DEVICE)
    for env in (None, "0", "1"):
        if env is None:
            monkeypatch.delenv("VLLM_TRITON_USE_TD", raising=False)
        else:
            monkeypatch.setenv("VLLM_TRITON_USE_TD", env)
        output = torch.empty_like(prefix_output)
        merge_attn_states_triton(
            output, prefix_output, prefix_lse, suffix_output, suffix_lse
        )
    assert spy.use_td == [False, False, True]


def generate_markdown_table():
    global all_case_info
    table_header = (
        "| tokens | heads | headsize | dtype "
        "| device | torch | triton | cuda | speedup |"
    )
    table_separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"

    def shortly_dtype(dtype: torch.dtype) -> str:
        return str(dtype).removeprefix("torch.")

    def shortly_device(device: str) -> str:
        return device.removeprefix("NVIDIA").strip()

    print(table_header)
    print(table_separator)
    for info in all_case_info:
        (
            num_tokens,
            num_heads,
            head_size,
            dtype,
            device,
            avg_time_torch_kernel,
            avg_time_triton_kernel,
            avg_time_cuda_kernel,
            performance_improved,
        ) = info
        dtype = shortly_dtype(dtype)
        device = shortly_device(device)
        print(
            f"| {num_tokens} | {num_heads} | {head_size} "
            f"| {dtype} | {device} | {avg_time_torch_kernel:.5f}ms "
            f"| {avg_time_triton_kernel:.5f}ms "
            f"| {avg_time_cuda_kernel:.5f}ms "
            f"| {performance_improved:.4f}x |"
        )


@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("prefill_tokens_with_context", [None, 128])
@pytest.mark.parametrize("num_tokens", NUM_BATCH_TOKENS)
@pytest.mark.parametrize("num_query_heads", NUM_QUERY_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("input_dtype", DTYPES)
@torch.inference_mode()
def test_merge_attn_states(
    prefill_tokens_with_context: int | None,
    num_tokens: int,
    num_query_heads: int,
    head_size: int,
    input_dtype: torch.dtype,
    use_fp8: bool,
):
    if not current_platform.is_cuda():
        pytest.skip(
            "Currently only support compare triton merge_attn_states "
            "with custom cuda merge_attn_states kernel"
        )

    NUM_TOKENS = num_tokens
    NUM_HEADS = num_query_heads
    HEAD_SIZE = head_size

    # When use_fp8 is set, inputs stay as input_dtype (bf16/fp16/fp32)
    # and output becomes FP8.
    output_dtype = input_dtype
    output_scale = None
    if use_fp8:
        output_dtype = current_platform.fp8_dtype()
        output_scale = torch.tensor([0.05], dtype=torch.float32, device="cuda")

    print(
        f"\nNUM_TOKENS:{NUM_TOKENS}, NUM_HEADS:{NUM_HEADS}, "
        f"HEAD_SIZE:{HEAD_SIZE}, input_dtype: {input_dtype}, "
        f"output_dtype: {output_dtype}, use_fp8: {use_fp8}, "
        f"prefill_tokens_with_context: {prefill_tokens_with_context}, "
        f"Device: {current_platform.get_device_name()}"
    )

    # prefix_lse and suffix_lse contain inf and normal values
    prefix_lse = torch.randn(NUM_HEADS, NUM_TOKENS, dtype=torch.float32, device="cuda")
    suffix_lse = torch.randn(NUM_HEADS, NUM_TOKENS, dtype=torch.float32, device="cuda")

    # Generate boolean masks
    mask_prefix = torch.rand(NUM_HEADS, NUM_TOKENS) < 0.1
    mask_suffix = torch.rand(NUM_HEADS, NUM_TOKENS) < 0.1
    # Ensure that the same position is not True at the same time
    combined_mask = torch.logical_and(mask_prefix, mask_suffix)
    mask_prefix = torch.logical_and(mask_prefix, ~combined_mask)
    mask_suffix = torch.logical_and(mask_suffix, ~combined_mask)

    prefix_lse[mask_prefix] = float("inf")
    suffix_lse[mask_suffix] = float("inf")

    # Other input tensors (need to be initialized but
    # no actual calculation needed)
    output = torch.zeros(
        (NUM_TOKENS, NUM_HEADS, HEAD_SIZE), dtype=output_dtype, device="cuda"
    )
    output_lse = torch.zeros(
        (NUM_HEADS, NUM_TOKENS), dtype=torch.float32, device="cuda"
    )
    prefix_output = torch.randn(
        (NUM_TOKENS, NUM_HEADS, HEAD_SIZE), dtype=input_dtype, device="cuda"
    )
    suffix_output = torch.randn(
        (NUM_TOKENS, NUM_HEADS, HEAD_SIZE), dtype=input_dtype, device="cuda"
    )

    warmup_times = 2
    repeat_times = 20

    output_torch = output.clone()
    output_lse_torch = output_lse.clone()
    total_time_torch_kernel = 0
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)

    # 0. Run the Torch kernel
    prefix_lse_torch = prefix_lse.clone()
    suffix_lse_torch = suffix_lse.clone()
    for _ in range(warmup_times):
        output_torch, output_lse_torch = merge_attn_states_torch(
            output_torch,
            prefix_output,
            prefix_lse_torch,
            suffix_output,
            suffix_lse_torch,
            output_lse_torch,
            prefill_tokens_with_context,
            output_scale,
        )
    torch.accelerator.synchronize()

    for _ in range(repeat_times):
        start.record()
        output_torch, output_lse_torch = merge_attn_states_torch(
            output_torch,
            prefix_output,
            prefix_lse_torch,
            suffix_output,
            suffix_lse_torch,
            output_lse_torch,
            prefill_tokens_with_context,
            output_scale,
        )
        end.record()
        torch.accelerator.synchronize()
        total_time_torch_kernel += start.elapsed_time(end)

    avg_time_torch_kernel = total_time_torch_kernel / repeat_times

    # 1. Run the Triton kernel
    output_ref_triton = output.clone()
    output_lse_ref_triton = output_lse.clone()

    total_time_triton_kernel = 0
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)

    for _ in range(warmup_times):
        merge_attn_states_triton(
            output_ref_triton,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse_ref_triton,
            prefill_tokens_with_context,
            output_scale,
        )
    torch.accelerator.synchronize()

    for _ in range(repeat_times):
        start.record()
        merge_attn_states_triton(
            output_ref_triton,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse_ref_triton,
            prefill_tokens_with_context,
            output_scale,
        )
        end.record()
        torch.accelerator.synchronize()
        total_time_triton_kernel += start.elapsed_time(end)

    avg_time_triton_kernel = total_time_triton_kernel / repeat_times

    # 2. Run the CUDA kernel
    total_time_cuda_kernel = 0
    output_cuda = output.clone()
    output_lse_cuda = output_lse.clone()

    for _ in range(warmup_times):
        merge_attn_states_cuda(
            output_cuda,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse_cuda,
            prefill_tokens_with_context,
            output_scale,
        )
    torch.accelerator.synchronize()

    for _ in range(repeat_times):
        start.record()
        merge_attn_states_cuda(
            output_cuda,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse_cuda,
            prefill_tokens_with_context,
            output_scale,
        )
        end.record()
        torch.accelerator.synchronize()
        total_time_cuda_kernel += start.elapsed_time(end)

    avg_time_cuda_kernel = total_time_cuda_kernel / repeat_times

    # 3. Performance compare
    performance_improved = avg_time_triton_kernel / avg_time_cuda_kernel
    print(f" Torch time: {avg_time_torch_kernel:.6f}ms")
    print(f"Triton time: {avg_time_triton_kernel:.6f}ms")
    print(
        f"  CUDA time: {avg_time_cuda_kernel:.6f}ms, "
        f"Performance: {performance_improved:.5f}x"
    )
    print("-" * 100)

    # 4. Correctness compare
    # Liger Kernel: Efficient Triton Kernels for LLM Training
    # https://arxiv.org/pdf/2410.10989, 3.3 Correctness
    # use rtol = 1e-2 for bfloat16.
    if use_fp8:
        # Compare in dequantized space (multiply back by scale) so that
        # absolute differences reflect real precision, not amplified FP8
        # quantization steps.
        atol, rtol = 1e-1, 1e-1
        assert output_scale is not None
        scale = output_scale.item()
    elif output_dtype == torch.bfloat16:
        atol, rtol = 1e-3, 1e-2
        scale = 1.0
    else:
        atol, rtol = 1e-3, 1e-3
        scale = 1.0

    def diff(a: torch.Tensor, b: torch.Tensor):
        max_diff = torch.max(torch.abs(a.float() - b.float()))
        return max_diff

    # Use Triton output as reference because we want to replace
    # the Triton kernel with custom CUDA kernel for merge attn
    # states operation.
    output_ref = output_ref_triton
    output_lse_ref = output_lse_ref_triton
    torch.testing.assert_close(
        output_cuda.float() * scale,
        output_ref.float() * scale,
        atol=atol,
        rtol=rtol,
    )
    print(
        "Output all match, max abs diff (dequantized):"
        if use_fp8
        else "Output all match, max abs diff:"
    )
    _diff = diff(output_ref.float() * scale, output_torch.float() * scale)
    print(f"(Triton vs Torch) : {_diff}")
    _diff = diff(output_torch.float() * scale, output_cuda.float() * scale)
    print(f"  (CUDA vs Torch) : {_diff}")
    _diff = diff(output_ref.float() * scale, output_cuda.float() * scale)
    print(f"  (CUDA vs Triton): {_diff}")
    print("-" * 100)

    torch.testing.assert_close(
        output_lse_cuda.float(), output_lse_ref.float(), atol=atol, rtol=rtol
    )
    print("Output LSE all match, max abs diff:")
    print(f"(Triton vs Torch) : {diff(output_lse_torch, output_lse_ref)}")
    print(f"  (CUDA vs Torch) : {diff(output_lse_torch, output_lse_cuda)}")
    print(f"  (CUDA vs Triton): {diff(output_lse_ref, output_lse_cuda)}")
    print("-" * 100)

    print(
        "All output values test passed! All inf values "
        "are correctly replaced with -inf."
    )
    print("-" * 100)

    device = current_platform.get_device_name()
    all_case_info.append(
        (
            NUM_TOKENS,
            NUM_HEADS,
            HEAD_SIZE,
            output_dtype,
            device,
            avg_time_torch_kernel,
            avg_time_triton_kernel,
            avg_time_cuda_kernel,
            performance_improved,
        )
    )
    if len(all_case_info) == (
        len(NUM_BATCH_TOKENS) * len(HEAD_SIZES) * len(NUM_QUERY_HEADS) * len(DTYPES)
    ):
        generate_markdown_table()
