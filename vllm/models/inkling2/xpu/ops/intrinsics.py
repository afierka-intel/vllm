# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Math helpers of the Inkling2 XPU Triton kernels.

`nvidia/ops/*` uses inline PTX for these (approximate `rsqrt`, `ex2`, `lg2`,
`rcp`, `tanh`, an opaque `mov.b32`, an L2 prefetch). Intel's Triton backend
has no PTX, so the XPU copies use portable Triton. The numerics differ from
CUDA's approximations by design:

- `rsqrt`: `div_rn(1, sqrt_rn(x))`, at most 1 ulp from the exact value
  (CUDA: `rsqrt.approx.f32`, up to 2 ulp).
- `ex2` / `lg2`: `tl.exp2` / `tl.log2` (CUDA: FTZ `ex2` / `lg2.approx`).
- `sigmoid`: `1 / (1 + exp2(-x * log2(e)))` (CUDA: `ex2` / `rcp.approx`).
- `tanh`: libdevice `tanh` (CUDA: `tanh.approx.f32`, about 2^-11 relative
  error).

CUDA keeps a K group's partial sum out of the next MMA's accumulator with an
opaque move and prefetches the next lines into L2; neither has a portable
counterpart, so the XPU kernels do without them.
"""

from vllm.triton_utils import tl, tldevice, triton

LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def rsqrt(x):
    return tl.div_rn(1.0, tl.sqrt_rn(x))


@triton.jit
def ex2(x):
    return tl.exp2(x)


@triton.jit
def lg2(x):
    return tl.log2(x)


@triton.jit
def sigmoid(x):
    return 1.0 / (1.0 + tl.exp2(-x * LOG2E))


@triton.jit
def tanh(x):
    return tldevice.tanh(x)
