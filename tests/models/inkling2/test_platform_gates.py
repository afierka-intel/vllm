# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform scope of the Inkling2 dispatch predicates and fail-loud gates.

The XPU gates fire on XPU only; on CUDA (and ROCm) the predicates keep the
decisions the CUDA kernels made before the portable paths. The platform and
its device capability are emulated on the current device's platform object.
"""

import json
import sys

import pytest
import torch

from tests.models.inkling2.test_model import TINY_CONFIG
from tests.models.inkling2.utils import DEVICE
from vllm.models.inkling2.common.model import _check_platform_support
from vllm.models.inkling2.common.moe import flat_experts_class
from vllm.models.inkling2.common.ops import linear as linear_ops
from vllm.models.inkling2.common.ops import platform as predicates
from vllm.models.inkling2.nvidia.ops.moe_sm90 import HummingExperts
from vllm.models.inkling2.speculator_config import Inkling2DSparkSpeculatorConfig
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


CUDA_FAMILIES = [90, 100, 110, 120]


@pytest.mark.parametrize("family", CUDA_FAMILIES)
def test_cuda_predicates_keep_cuda_decisions(emulate, family):
    """CUDA: the native conv/mix op on Hopper and SM10x (as in
    vllm-project/vllm#69165), PDL launches, the SM10x kernels on SM10x only,
    and the Humming experts on the flat-row path."""
    emulate("cuda", family)
    assert predicates.use_native_conv_mix() == (family in (90, 100))
    assert predicates.use_pdl()
    assert dict(predicates.pdl_launch_kwargs()) == {"launch_pdl": True}
    assert predicates.use_sm100_kernels() == (family == 100)
    for quant_format in ("nvfp4", "mxfp8"):
        assert flat_experts_class(quant_format) is HummingExperts


@pytest.mark.parametrize("capability", [90, 94, 95])
def test_rocm_keeps_cuda_dispatch(emulate, capability):
    """ROCm takes none of the XPU decisions: the conv/mix op, the Humming
    experts and PDL launches, as before the portable paths."""
    emulate("rocm", capability)
    assert predicates.use_native_conv_mix()
    assert predicates.use_pdl()
    for quant_format in ("nvfp4", "mxfp8"):
        assert flat_experts_class(quant_format) is HummingExperts


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


@pytest.mark.parametrize("quant_format", ["nvfp4", "mxfp8"])
def test_xpu_routed_experts_fail_loud(emulate, quant_format):
    """XPU has no routed-expert GEMM yet: the layer fails at model init,
    before any weight loads, naming the format and the device."""
    emulate("xpu")
    with pytest.raises(NotImplementedError, match=quant_format.upper()):
        flat_experts_class(quant_format)


def _vllm_config(tmp_path, lora: bool):
    from vllm.config import LoRAConfig, ModelConfig, VllmConfig

    (tmp_path / "config.json").write_text(json.dumps(TINY_CONFIG))
    return VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="bfloat16",
            skip_tokenizer_init=True,
            tokenizer_mode="hf",
            max_model_len=2048,
        ),
        lora_config=LoRAConfig(max_lora_rank=8) if lora else None,
    )


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_model_init_gates_skip_cuda_and_rocm(tmp_path, emulate, platform):
    """Image inputs and LoRA pass the model-init gate off XPU."""
    config = _vllm_config(tmp_path, lora=True)
    emulate(platform, 90 if platform == "cuda" else 94)
    _check_platform_support(config, vision=True)


@pytest.mark.parametrize(
    "lora,vision,match",
    [
        (True, False, r"does not support LoRA$"),
        (False, True, r"does not support image inputs \(serve text only"),
        (True, True, r"image inputs .*; LoRA$"),
    ],
    ids=["lora", "image", "image-and-lora"],
)
def test_xpu_rejects_lora_and_image_inputs_at_init(
    tmp_path, emulate, lora, vision, match
):
    config = _vllm_config(tmp_path, lora=lora)
    emulate("xpu")
    with pytest.raises(NotImplementedError, match=match):
        _check_platform_support(config, vision=vision)


def test_xpu_text_model_passes_init_gate(tmp_path, emulate):
    config = _vllm_config(tmp_path, lora=False)
    emulate("xpu")
    _check_platform_support(config, vision=False)


def test_xpu_rejects_dspark_speculative_config(emulate):
    """The DSpark draft config is rejected on XPU before it is adapted."""
    emulate("xpu")
    with pytest.raises(NotImplementedError, match="DSpark .* not supported on XPU"):
        Inkling2DSparkSpeculatorConfig.adapt_draft_model_config(None)


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_linear_small_batches_without_flashinfer(monkeypatch, emulate, rows):
    """Off SM10x, `linear` of up to 32 rows never queries FlashInfer, which
    is a CUDA-only dependency (unimportable here), and is `F.linear`."""
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", None)
    emulate(*(("xpu", None) if current_platform.is_xpu() else ("cuda", 90)))
    torch.manual_seed(0)
    x = torch.randn(rows, 512, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(256, 512, device=DEVICE, dtype=torch.bfloat16)
    actual = linear_ops.linear(x, weight)
    torch.testing.assert_close(
        actual, torch.nn.functional.linear(x, weight), atol=0, rtol=0
    )


def test_fp8_kv_cache_rejected_at_model_init_off_sm12x(tmp_path, dist_init):
    """FP8 KV caches load PTX E4M3 conversions: off SM12x (e.g. on XPU)
    the attention layers reject them when the model is built."""
    from vllm.config import (
        CacheConfig,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.models.inkling2.common.model import Inkling2ForConditionalGeneration

    (tmp_path / "config.json").write_text(json.dumps(TINY_CONFIG))
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="bfloat16",
            skip_tokenizer_init=True,
            tokenizer_mode="hf",
            max_model_len=2048,
            limit_mm_per_prompt={"image": 0},
        ),
        cache_config=CacheConfig(cache_dtype="fp8_e4m3"),
    )
    with set_current_vllm_config(config), torch.device(DEVICE):
        if current_platform.is_device_capability_family(120):
            Inkling2ForConditionalGeneration(vllm_config=config)
        else:
            with pytest.raises(ValueError, match="KV cache dtype must be one of"):
                Inkling2ForConditionalGeneration(vllm_config=config)
