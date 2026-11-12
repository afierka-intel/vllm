# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4,
    quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import (
    quant_dequant_mxfp6,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    FP8_DTYPE,
    QuantKey,
    get_fp8_min_max,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp6E2M3Dynamic,
    kMxfp6E3M2Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_quark

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

logger = init_logger(__name__)

# MX activation formats; their quant-dequant needs amd-quark.
_ACTIVATION_QUANT_DEQUANT_FUNCS: dict[
    QuantKey, Callable[[torch.Tensor], torch.Tensor]
] = {
    kMxfp4Dynamic: quant_dequant_mxfp4,
    kMxfp6E3M2Dynamic: partial(quant_dequant_mxfp6, quant_dtype="fp6_e3m2"),
    kMxfp6E2M3Dynamic: partial(quant_dequant_mxfp6, quant_dtype="fp6_e2m3"),
}


def quant_dequant_fp8_dynamic(
    x: torch.Tensor, group_size: int, eps: float = 1e-10
) -> torch.Tensor:
    """Dynamic symmetric FP8 quant-dequant of x over groups of the last dim
    (group_size == -1: one scale per token).

    Same arithmetic as per_token_group_quant_fp8 with linear scales:
    scale = max(amax, eps) / fp8_max, q = clamp(x / scale) cast to FP8.
    """
    fp8_min, fp8_max = get_fp8_min_max()
    k = x.shape[-1]
    g = k if group_size == -1 else group_size
    assert k % g == 0, f"last dim {k} is not a multiple of group size {g}"
    xg = x.float().unflatten(-1, (k // g, g))
    scale = xg.abs().amax(dim=-1, keepdim=True).clamp(min=eps) * (1.0 / fp8_max)
    q = (xg / scale).clamp(fp8_min, fp8_max).to(FP8_DTYPE)
    return (q.float() * scale).flatten(-2).to(x.dtype)


def _fp8_dynamic_group_size(key: QuantKey | None) -> int | None:
    """Group size of a dynamic symmetric per-token / per-token-group FP8
    activation key (-1 = per token), or None for any other key."""
    if (
        key is None
        or key.dtype != FP8_DTYPE
        or key.scale.dtype != torch.float32  # excludes MXFP8 (E8M0 scales)
        or key.scale.static
        or not key.symmetric
        or key.scale2 is not None
    ):
        return None
    rows, cols = key.scale.group_shape
    if rows != 1 or cols == 0 or cols < -1:
        return None
    return cols


def _activation_quant_dequant_func(
    key: QuantKey | None,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    if key is None:
        return lambda x: x
    if key in _ACTIVATION_QUANT_DEQUANT_FUNCS:
        return _ACTIVATION_QUANT_DEQUANT_FUNCS[key]
    group_size = _fp8_dynamic_group_size(key)
    if group_size is not None:
        return partial(quant_dequant_fp8_dynamic, group_size=group_size)
    return None


class EmulationMxfp4LinearKernel(MxFp4LinearKernel):
    """Software emulation fallback for MXFP4 weights (dequant + F.linear) with
    unquantized, MXFP4/MXFP6 or dynamic per-token(-group) FP8 activations.

    Activations get the requested quant-dequant, so the result keeps the
    checkpoint's activation precision (e.g. W4A8 stays W4A8).
    """

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        super().__init__(config)
        qdq = _activation_quant_dequant_func(config.activation_quant_key)
        assert qdq is not None
        self.quant_dequant_func = qdq

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def _supports_quant_scheme(
        cls,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (
            weight_key == kMxfp4Static
            and _activation_quant_dequant_func(activation_key) is not None
        )

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if _activation_quant_dequant_func(config.activation_quant_key) is None:
            return False, (
                "only supports MXFP4, MXFP6, dynamic per-token(-group) FP8 or "
                "unquantized activations"
            )

        # Weight dequant has a PyTorch fallback; only the MX activation
        # quant-dequant needs amd-quark.
        if config.activation_quant_key in _ACTIVATION_QUANT_DEQUANT_FUNCS and (
            not has_quark()
        ):
            return False, "amd-quark package not available"

        if (
            current_platform.is_rocm()
            and current_platform.supports_mx()
            and config.activation_quant_key != kMxfp4Dynamic
        ):
            logger.warning_once(
                "The current platform supports native MXFP4/MXFP6 computation, "
                f"but kernels for activation_quant_key={config.activation_quant_key} "
                f"are not yet integrated in vLLM. Using EmulationMxfp4LinearKernel, "
                "with simulated weight dequantization and activation "
                "QDQ (quantize and dequantize), with the linear "
                "layers computed in high precision."
            )

        if not current_platform.supports_mx():
            logger.warning_once(
                "The current platform does not support native MXFP4 "
                "computation. Using EmulationMxfp4LinearKernel, with simulated weight "
                "dequantization and activation QDQ (quantize and dequantize), with "
                "the linear layers computed in high precision."
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Match the MXFP8 emulation path when explicitly enabled: dequantize
        # once at load time so inference uses a plain high-precision linear.
        # Otherwise retain packed weights and dequantize per invocation.
        if envs.VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD:
            weight = dequant_mxfp4(layer.weight, layer.weight_scale, torch.bfloat16)
            replace_parameter(layer, "weight", weight.contiguous())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qdq_x = self.quant_dequant_func(x)

        # Packed MXFP4 weights use one byte for two values. A weight with
        # elements wider than one byte was already dequantized at load time.
        if layer.weight.element_size() >= 2:
            return F.linear(qdq_x, layer.weight.to(x.dtype), bias)

        dq_w = dequant_mxfp4(layer.weight, layer.weight_scale, x.dtype)
        return F.linear(qdq_x, dq_w, bias)
