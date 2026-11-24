# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform scope of the Inkling2 dispatch predicates.

On CUDA the predicates keep the decisions the CUDA kernels made before the
portable paths. The platform and its device capability are
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
    "platform,capability,kwargs,pdl",
    [
        # The pin launched every PDL kernel with launch_pdl=True.
        ("cuda", 90, {"launch_pdl": True}, True),
        ("cuda", 100, {"launch_pdl": True}, True),
        ("cuda", 120, {"launch_pdl": True}, True),
        # vllm-project/vllm#69165: launch_pdl=use_pdl() below SM90.
        ("cuda", 80, {"launch_pdl": False}, False),
        ("cuda", 89, {"launch_pdl": False}, False),
        # The AMD (HIP) and Intel Triton backends have no `launch_pdl`
        # option and reject it, even when it is False.
        ("rocm", 94, {}, True),
        ("rocm", 95, {}, True),
        ("xpu", None, {}, False),
    ],
)
def test_pdl_launch_kwargs_per_platform(emulate, platform, capability, kwargs, pdl):
    """`pdl_launch_kwargs()` is `launch_pdl=use_pdl()` on CUDA and no option
    elsewhere; `use_pdl` (and so the griddepcontrol calls of `pdl.py`)
    follows SM90+ (#69165)."""
    emulate(platform, capability)
    assert dict(predicates.pdl_launch_kwargs()) == kwargs
    assert predicates.use_pdl() == pdl


def test_no_literal_launch_pdl_outside_pdl_launch_kwargs():
    """No launch site of the shared code (`vllm/models/inkling2/` outside the
    CUDA-only `nvidia/` package) passes a literal `launch_pdl=` keyword:
    every PDL launch unpacks `**pdl_launch_kwargs()`, the only place that
    sets the option, so the shared kernels launch on Triton backends
    without it (XPU, ROCm)."""
    import ast
    import pathlib

    import vllm.models.inkling2 as package

    root = pathlib.Path(package.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        if path.relative_to(root).parts[0] == "nvidia":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                found += [
                    f"{path.relative_to(root)}:{node.lineno}"
                    for keyword in node.keywords
                    if keyword.arg == "launch_pdl"
                ]
    assert not found, found


def test_xpu_predicates_take_portable_paths(emulate):
    """XPU: `short_conv` + `attn_res`, stream-ordered launches (no
    `launch_pdl` option at all), software e4m3, no SM10x kernels."""
    emulate("xpu")
    assert not predicates.use_native_conv_mix()
    assert not predicates.use_pdl()
    assert dict(predicates.pdl_launch_kwargs()) == {}
    assert not predicates.use_native_e4m3()
    assert not predicates.use_sm100_kernels()
