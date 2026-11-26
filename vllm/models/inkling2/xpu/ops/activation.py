# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU copy of the SiTU helpers of `vllm/models/inkling2/nvidia/ops/activation.py`
(upstream `b7186d137a`).

Adapted: the inline-PTX sigmoid and `tanh.approx` are the portable
`intrinsics.sigmoid` / `intrinsics.tanh`; the NVFP4 / MXFP8 quantizing
epilogue (`swiglu_quantize`) is not copied (the XPU experts take bf16
activations).

Softcapped SwiGLU (SiTU) of the Inkling2 experts, in fp32 on bf16 inputs.
"""

from vllm.triton_utils import tl, triton

from .intrinsics import sigmoid, tanh

FP32_MAX = tl.constexpr(3.4028234663852886e38)


@triton.jit
def _div_rn(x, DIVISOR: tl.constexpr):
    """Bitwise `tl.div_rn(x, DIVISOR)` without the division routine: the
    product with the rounded reciprocal and one FMA correction step (whose
    residual is clamped for infinite `x`) are exact for these divisors on
    every float32 input."""
    tl.static_assert(DIVISOR == 4.0 or DIVISOR == 25.0)
    q = x * (1.0 / DIVISOR)
    if DIVISOR != 4.0:
        residual = tl.maximum(tl.fma(q, DIVISOR, -x), -FP32_MAX)
        q = tl.fma(-residual, 1.0 / DIVISOR, q)
    return q


@triton.jit
def _softcap(x, CAP: tl.constexpr):
    if CAP > 0:
        x = CAP * tanh(_div_rn(x, CAP))
    return x


@triton.jit
def _situ(X, row, start, gamma, N: tl.constexpr, BLOCK: tl.constexpr, GATE_CAP, UP_CAP):
    """fp32 values of the bf16 SiTU of columns `[start, start + BLOCK)`."""
    # Interleaved (gate, up) bf16 pairs as one word each, loaded so that each
    # thread holds whole 16-column blocks: column 16 * b + 4 * i + j at
    # [i, b, j].
    i = tl.arange(0, 4)[:, None, None]
    b = tl.arange(0, BLOCK // 16)[None, :, None]
    j = tl.arange(0, 4)[None, None, :]
    cols = start + 16 * b + 4 * i + j
    pairs = tl.load(X.to(tl.pointer_type(tl.uint32)) + row * N + cols)
    pairs = tl.reshape(tl.permute(pairs, (1, 0, 2)), (BLOCK,))
    gate = (pairs << 16).to(tl.float32, bitcast=True)
    up = (pairs & 0xFFFF0000).to(tl.float32, bitcast=True)
    gate_sigmoid = sigmoid(gate)
    gate = _softcap(gate, GATE_CAP)
    up = _softcap(up, UP_CAP)
    return (((gate * gate_sigmoid) * up) * gamma).to(tl.bfloat16).to(tl.float32)
