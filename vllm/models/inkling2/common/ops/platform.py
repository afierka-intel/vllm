# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side platform predicates of the Inkling2 kernels.

Each names one capability, so that the shared code takes the CUDA fast
paths of `nvidia/` exactly where they ran before and the portable Triton
paths everywhere else (e.g. XPU).
"""

import functools
from collections.abc import Mapping
from types import MappingProxyType


@functools.cache
def use_sm100_kernels() -> bool:
    """Whether to run the Blackwell kernels (tml_fa4, tcgen05 attention,
    TRT-LLM-gen expert GEMMs, e2m1 conversions). Hopper uses Humming experts;
    SM12x uses Triton experts and does not support tcgen05."""
    from vllm.platforms import current_platform

    return current_platform.is_device_capability_family(100)


@functools.cache
def use_pdl() -> bool:
    """Whether the Triton kernels use programmatic dependent launch
    (griddepcontrol) and TMA descriptors, both SM90+ (named and defined as in
    vllm-project/vllm#69165). ROCm reports its gfx capability (gfx942 is
    9.4), so it launches as before the portable paths. XPU reports no device
    capability: there the kernels run stream ordered."""
    from vllm.platforms import current_platform

    return current_platform.has_device_capability(90)


@functools.cache
def pdl_launch_kwargs() -> Mapping[str, bool]:
    """Launch options of the PDL kernels: `launch_pdl=use_pdl()` (CUDA and
    ROCm, as before the portable paths), none on XPU, whose Triton rejects
    the option even when it is False."""
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        return MappingProxyType({})
    return MappingProxyType({"launch_pdl": use_pdl()})


@functools.cache
def use_kernel_autotune() -> bool:
    """Whether the portable Triton kernels autotune their tiles: XPU, unless
    `VLLM_BATCH_INVARIANT` asks for run-to-run and batch-size invariant
    results, which a timing-based pick per token bucket cannot give. Then,
    as on CUDA (and as Hopper's Humming experts take their batch-invariant
    heuristic configs), the fixed launch configs run. CUDA keeps the fixed
    configs (tiles, warps, stages) chosen for Hopper and Blackwell, so its
    launches are unchanged."""
    from vllm import envs
    from vllm.platforms import current_platform

    return current_platform.is_xpu() and not envs.VLLM_BATCH_INVARIANT


@functools.cache
def use_tma_k_groups() -> bool:
    """Whether `k_group_partials` runs its host TensorDescriptor kernel for
    large batches: with `use_pdl` (SM90+, as in vllm-project/vllm#69165; ROCm
    as before the portable paths). XPU runs the K-group kernel, whose
    tensor-descriptor path follows `use_tensor_descriptor()`."""
    from vllm.platforms import current_platform

    return use_pdl() and not current_platform.is_xpu()


@functools.cache
def use_native_conv_mix() -> bool:
    """Whether `conv_mix` runs the `csrc/inkling2` conv/mix op: everywhere
    but SM12x and XPU, which run `short_conv` + `attn_res`. Off XPU this is
    the decision of the CUDA kernels before the portable paths."""
    from vllm.platforms import current_platform

    return not (
        current_platform.is_xpu() or current_platform.is_device_capability_family(120)
    )
