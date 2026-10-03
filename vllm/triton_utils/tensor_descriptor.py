# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def use_tensor_descriptor(override: bool | None = None) -> bool:
    """Tri-state VLLM_TRITON_USE_TD: unset=auto (on for XPU), 1/0=force on/off."""
    from vllm import envs
    from vllm.platforms import current_platform

    if override is None:
        override = envs.VLLM_TRITON_USE_TD
    if override is not None:
        return override
    return current_platform.is_xpu()


def tensor_descriptor_compatible(*tensors: "torch.Tensor | None") -> bool:
    """Whether every tensor can back an in-kernel tensor descriptor whose shape
    spans its full last dim (block rows run along it). ``None`` (an unused
    optional operand) is accepted.
    """
    for t in tensors:
        if t is None:
            continue
        elem = t.element_size()
        width = t.shape[-1] * elem
        # Intel 2D block IO: min surface width/pitch 64 B (16 B multiples,
        # pitch >= width); this also satisfies TMA's 16 B rule. Strides of
        # size-1 dims are checked too: kernels may still pass them as a pitch.
        min_pitch = max(64, width)
        if (
            t.stride(-1) != 1
            or t.data_ptr() % 16 != 0
            or width < 64
            or width % 16 != 0
            or any(s * elem < min_pitch or s * elem % 16 for s in t.stride()[:-1])
        ):
            return False
    return True
