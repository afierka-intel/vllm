# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform scope of the Inkling2 dispatch predicates.

On CUDA (and ROCm) the predicates keep the decisions the CUDA kernels made
before the portable paths. The platform and its device capability are
emulated on the current device's platform object.
"""

import pytest

from vllm.models.inkling2.common.ops import platform as predicates
from vllm.platforms import current_platform

_PREDICATES = (
    predicates.use_sm100_kernels,
    predicates.use_pdl,
    predicates.pdl_launch_kwargs,
    predicates.use_native_conv_mix,
    predicates.use_native_e4m3,
)


@pytest.fixture
def emulate(monkeypatch):
    """`emulate(platform, capability)`: `current_platform` reports `platform`
    ("cuda", "rocm" or "xpu") and device capability `capability` (e.g. 90
    for SM90, 94 for gfx942; XPU reports none). The cached predicates are
    recomputed."""

    def clear():
        for predicate in _PREDICATES:
            predicate.cache_clear()

    def apply(platform: str, capability: int | None = None) -> None:
        assert platform != "xpu" or capability is None
        monkeypatch.setattr(current_platform, "is_cuda", lambda: platform == "cuda")
        monkeypatch.setattr(current_platform, "is_rocm", lambda: platform == "rocm")
        monkeypatch.setattr(current_platform, "is_xpu", lambda: platform == "xpu")
        monkeypatch.setattr(
            current_platform,
            "is_device_capability_family",
            lambda family, device_id=0: (
                capability is not None and capability // 10 == family // 10
            ),
        )
        monkeypatch.setattr(
            current_platform,
            "has_device_capability",
            lambda minimum, device_id=0: (
                capability is not None and capability >= minimum
            ),
        )
        clear()

    clear()
    yield apply
    monkeypatch.undo()
    clear()


@pytest.mark.parametrize(
    "platform,capability,kwargs",
    [
        # The pin launched every PDL kernel with launch_pdl=True.
        ("cuda", 90, {"launch_pdl": True}),
        ("cuda", 100, {"launch_pdl": True}),
        ("cuda", 120, {"launch_pdl": True}),
        ("rocm", 94, {"launch_pdl": True}),
        ("rocm", 95, {"launch_pdl": True}),
        # vllm-project/vllm#69165: launch_pdl=use_pdl() below SM90.
        ("cuda", 80, {"launch_pdl": False}),
        ("cuda", 89, {"launch_pdl": False}),
        # Intel's Triton rejects the option, even when it is False.
        ("xpu", None, {}),
    ],
)
def test_pdl_launch_kwargs_per_platform(emulate, platform, capability, kwargs):
    """`pdl_launch_kwargs()` is `launch_pdl=use_pdl()` everywhere but XPU,
    which gets no option at all; `use_pdl` (and so the griddepcontrol calls
    of `pdl.py`) follows SM90+ (#69165)."""
    emulate(platform, capability)
    assert dict(predicates.pdl_launch_kwargs()) == kwargs
    assert predicates.use_pdl() == kwargs.get("launch_pdl", False)


def test_xpu_predicates_take_portable_paths(emulate):
    """XPU: `short_conv` + `attn_res`, stream-ordered launches (no
    `launch_pdl` option at all), software e4m3, no SM10x kernels."""
    emulate("xpu")
    assert not predicates.use_native_conv_mix()
    assert not predicates.use_pdl()
    assert dict(predicates.pdl_launch_kwargs()) == {}
    assert not predicates.use_native_e4m3()
    assert not predicates.use_sm100_kernels()
