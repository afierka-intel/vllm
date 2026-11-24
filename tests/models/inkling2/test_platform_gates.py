# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform scope of the Inkling2 dispatch predicates and fail-loud gates.

The XPU gates fire on XPU only; on CUDA (and ROCm) the predicates keep the
decisions the CUDA kernels made before the portable paths. The platform and
its device capability are emulated on the current device's platform object.
"""

import functools
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
    predicates.use_kernel_autotune,
    predicates.use_tma_k_groups,
    predicates.use_native_conv_mix,
)


@pytest.fixture
def emulate(monkeypatch):
    """`emulate(platform, family)`: `current_platform` reports `platform`
    ("cuda", "rocm" or "xpu") and, on CUDA, compute capability family
    `family` (e.g. 90). The cached predicates are recomputed."""

    def clear():
        for predicate in _PREDICATES:
            predicate.cache_clear()

    def apply(platform: str, family: int | None = None) -> None:
        cuda = platform == "cuda"
        monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
        monkeypatch.setattr(current_platform, "is_rocm", lambda: platform == "rocm")
        monkeypatch.setattr(current_platform, "is_xpu", lambda: platform == "xpu")
        monkeypatch.setattr(
            current_platform,
            "is_device_capability_family",
            lambda capability, device_id=0: cuda and capability == family,
        )
        monkeypatch.setattr(
            current_platform,
            "has_device_capability",
            lambda capability, device_id=0: (
                cuda and family is not None and family >= capability
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
    """CUDA: the native conv/mix op on every family but SM12x (SM110
    included), PDL launches, the fixed launch configs, the SM10x kernels on
    SM10x only, and the Humming experts on the flat-row path."""
    emulate("cuda", family)
    assert predicates.use_native_conv_mix() == (family != 120)
    assert predicates.use_pdl()
    assert dict(predicates.pdl_launch_kwargs()) == {"launch_pdl": True}
    assert not predicates.use_kernel_autotune()
    assert predicates.use_sm100_kernels() == (family == 100)
    for quant_format in ("nvfp4", "mxfp8"):
        assert flat_experts_class(quant_format, 8, 4.0, 25.0) is HummingExperts


def test_rocm_keeps_cuda_dispatch(emulate):
    """ROCm takes none of the XPU decisions: the conv/mix op and the
    Humming experts, as before the portable paths."""
    emulate("rocm")
    assert predicates.use_native_conv_mix()
    assert not predicates.use_kernel_autotune()
    for quant_format in ("nvfp4", "mxfp8"):
        assert flat_experts_class(quant_format, 8, 4.0, 25.0) is HummingExperts


def test_xpu_predicates_take_portable_paths(emulate):
    """XPU: `short_conv` + `attn_res`, stream-ordered launches (no
    `launch_pdl` option at all), autotuned tiles, no SM10x kernels."""
    emulate("xpu")
    assert not predicates.use_native_conv_mix()
    assert not predicates.use_pdl()
    assert dict(predicates.pdl_launch_kwargs()) == {}
    assert predicates.use_kernel_autotune()
    assert not predicates.use_sm100_kernels()
    assert not predicates.use_tma_k_groups()


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
    emulate(platform, 90 if platform == "cuda" else None)
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
    emulate("xpu" if current_platform.is_xpu() else "cuda", 90)
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


@pytest.mark.skipif(
    not current_platform.is_xpu(), reason="vllm-xpu-kernels grouped GEMM routing"
)
def test_xpu_mxfp8_experts_only_on_xe2_grouped_gemm(monkeypatch):
    """MXFP8 runs only where `XpuFusedMoe` takes the xe_2 grouped GEMM (bf16
    activations, e4m3 weights); elsewhere it fails at init. NVFP4 does not
    depend on it."""
    from vllm.models.inkling2.xpu.experts import (
        XpuMxfp8Experts,
        XpuNvfp4Experts,
        mxfp8_experts_supported,
        xpu_flat_experts,
    )

    # The CI card (Arc Pro B70) is an Xe2 part.
    assert mxfp8_experts_supported()
    mxfp8 = xpu_flat_experts("mxfp8", 8, 4.0, 25.0)
    assert isinstance(mxfp8, functools.partial) and mxfp8.func is XpuMxfp8Experts

    monkeypatch.setenv("FORCE_XE3P_KERNEL", "1")
    assert not mxfp8_experts_supported()
    with pytest.raises(NotImplementedError, match="xe_3 GEMM"):
        xpu_flat_experts("mxfp8", 8, 4.0, 25.0)
    monkeypatch.delenv("FORCE_XE3P_KERNEL")

    xpu_ops = torch.ops._xpu_C
    monkeypatch.setattr(xpu_ops, "is_xe2_arch", lambda device_index=-1: False)
    monkeypatch.setattr(xpu_ops, "is_xe3_arch", lambda device_index=-1: False)
    assert not mxfp8_experts_supported()
    with pytest.raises(NotImplementedError, match="xe_2 grouped GEMM"):
        xpu_flat_experts("mxfp8", 8, 4.0, 25.0)
    nvfp4 = xpu_flat_experts("nvfp4", 8, 4.0, 25.0)
    assert isinstance(nvfp4, functools.partial) and nvfp4.func is XpuNvfp4Experts
