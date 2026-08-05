# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    _backend_incompatibility_reason,
    _convert_moe_wna16_humming_tensors,
    convert_to_wna16_moe_kernel_format,
    map_wna16_backend,
)
from vllm.model_executor.layers.quantization import moe_wna16
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    MoeWNA16Method,
)


def test_map_wna16_backend_supports_triton():
    assert map_wna16_backend("triton") == WNA16MoEBackend.TRITON


@pytest.mark.parametrize(
    ("backend", "quant_config", "may_have_zp", "may_have_bias", "expected"),
    [
        (
            WNA16MoEBackend.TRITON,
            AutoAWQConfig(4, 128, True, False),
            True,
            False,
            "AutoAWQ weight layout",
        ),
        (
            WNA16MoEBackend.TRITON,
            AutoGPTQConfig(4, 128, True, True, False, {}, {}),
            False,
            False,
            "activation ordering",
        ),
        (
            WNA16MoEBackend.TRITON,
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.GROUP,
            ),
            False,
            False,
            "activation ordering",
        ),
        (
            WNA16MoEBackend.TRITON,
            AutoGPTQConfig(4, 128, False, True, False, {}, {}),
            False,
            True,
            "bias",
        ),
        (
            WNA16MoEBackend.MARLIN,
            MoeWNA16Config(
                linear_quant_method="gptq",
                weight_bits=4,
                group_size=128,
                has_zp=False,
                lm_head_quantized=False,
                modules_to_not_convert=None,
                full_config={},
            ),
            False,
            False,
            "MoeWNA16 checkpoint layout",
        ),
        (
            # The XPU repack helper contracts on the AutoGPTQ int32 K-first
            # layout, so it must not be offered a MoeWNA16 uint8 N-first
            # checkpoint -- on XPU it is the only `auto` candidate, so a
            # missing rejection here silently shapes the weights wrong.
            WNA16MoEBackend.XPU,
            MoeWNA16Config(
                linear_quant_method="gptq",
                weight_bits=4,
                group_size=128,
                has_zp=False,
                lm_head_quantized=False,
                modules_to_not_convert=None,
                full_config={},
            ),
            False,
            False,
            "MoeWNA16 checkpoint layout",
        ),
    ],
)
def test_wna16_oracle_rejects_incompatible_quant_structures(
    backend, quant_config, may_have_zp, may_have_bias, expected
):
    from tests.kernels.moe.utils import make_dummy_moe_config

    moe_config = make_dummy_moe_config()

    reason = _backend_incompatibility_reason(
        backend=backend,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=may_have_zp,
        may_have_bias=may_have_bias,
        allow_tile_padding=True,
    )

    assert reason is not None
    assert expected in reason


def test_wna16_oracle_keeps_triton_available_for_moe_wna16():
    """Triton is the backend that reads the MoeWNA16 layout; keep it eligible.

    `--moe-backend triton` is the only way to run a MoeWNA16 checkpoint on XPU
    now that the XPU backend rejects that layout, so a broader rejection would
    leave the checkpoint with no backend at all.
    """
    from tests.kernels.moe.utils import make_dummy_moe_config

    reason = _backend_incompatibility_reason(
        backend=WNA16MoEBackend.TRITON,
        moe_config=make_dummy_moe_config(),
        quant_config=MoeWNA16Config(
            linear_quant_method="gptq",
            weight_bits=4,
            group_size=128,
            has_zp=False,
            lm_head_quantized=False,
            modules_to_not_convert=None,
            full_config={},
        ),
        may_have_zp=False,
        may_have_bias=False,
        allow_tile_padding=True,
    )

    assert reason is None


def test_auto_dispatch_falls_back_to_triton_when_xpu_rejects_moe_wna16(monkeypatch):
    """`--moe-backend auto` must not dead-end for a MoeWNA16 checkpoint on XPU.

    XPU's native kernel is the only `auto` candidate and it rejects MoeWNA16's
    layout outright (see the guard in `_backend_incompatibility_reason`), so
    without a fallback candidate this leaves the user with a bare
    `NotImplementedError` for a checkpoint that TRITON can serve correctly --
    confirmed by `test_wna16_oracle_keeps_triton_available_for_moe_wna16` above.
    There is no tradeoff in falling back: TRITON is the only backend this
    layout can ever run on, so `auto` choosing it is not a silent
    performance surprise, it is the only working choice.
    """
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.oracle import int_wna16
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        INT4_DTYPE,
        QuantKey,
        kInt4StaticGroupScale,
    )

    monkeypatch.setattr(int_wna16.current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(int_wna16.current_platform, "is_cpu", lambda: False)

    # A real weight_key, matching how MoeWNA16Method builds it for num_bits=4 --
    # the kernel-level is_supported_config() check (below the oracle-level
    # rejection this test targets) rejects a None weight_key outright, which
    # would make this test pass for the wrong reason if it were skipped instead.
    weight_key = QuantKey(INT4_DTYPE, kInt4StaticGroupScale)

    backend, _experts_cls = int_wna16.select_wna16_moe_backend(
        config=make_dummy_moe_config(
            num_experts=8, hidden_dim=4096, intermediate_size=1024
        ),
        weight_key=weight_key,
        quant_config=MoeWNA16Config(
            linear_quant_method="gptq",
            weight_bits=4,
            group_size=128,
            has_zp=False,
            lm_head_quantized=False,
            modules_to_not_convert=None,
            full_config={},
        ),
        may_have_zp=False,
        may_have_bias=False,
    )
    assert backend == int_wna16.WNA16MoEBackend.TRITON


def test_compressed_tensors_weights_are_transposed_for_triton():
    quant_config = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=32,
    )
    w13 = torch.arange(16, dtype=torch.int32).reshape(1, 2, 8)
    w2 = torch.arange(12, dtype=torch.int32).reshape(1, 2, 6)
    w13_scale = torch.arange(32, dtype=torch.float16).reshape(1, 4, 8)
    w2_scale = torch.arange(18, dtype=torch.float16).reshape(1, 3, 6)

    converted = convert_to_wna16_moe_kernel_format(
        backend=WNA16MoEBackend.TRITON,
        layer=torch.nn.Module(),
        quant_config=quant_config,
        input_dtype=None,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
    )

    assert converted is not None
    assert torch.equal(converted[0], w13.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(converted[1], w2.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(converted[2], w13_scale.transpose(1, 2).contiguous())
    assert torch.equal(converted[3], w2_scale.transpose(1, 2).contiguous())


def test_moe_wna16_setup_forwards_selected_backend(monkeypatch):
    method = object.__new__(MoeWNA16Method)
    method.experts_cls = object
    method.wna16_backend = WNA16MoEBackend.HUMMING
    method.moe = object()
    quant_config = object()
    method.get_fused_moe_quant_config = lambda layer: quant_config
    layer = SimpleNamespace(_expert_routing_tables=lambda: (None, None, None))
    captured = {}
    kernel = object()

    def fake_make_wna16_moe_kernel(**kwargs):
        captured.update(kwargs)
        return kernel

    monkeypatch.setattr(moe_wna16, "make_wna16_moe_kernel", fake_make_wna16_moe_kernel)

    method._setup_kernel(layer)

    assert method.moe_kernel is kernel
    assert captured["backend"] == WNA16MoEBackend.HUMMING


def test_moe_wna16_humming_adapter_repacks_uint8_tensors():
    qweight = torch.arange(32, dtype=torch.uint8).reshape(1, 4, 8)
    scales = torch.arange(16, dtype=torch.float16).reshape(1, 4, 4)
    qzeros = torch.arange(16, dtype=torch.uint8).reshape(1, 8, 2)

    converted = _convert_moe_wna16_humming_tensors(
        {"qweight": qweight, "scales": scales, "qzeros": qzeros},
        has_zero_point=True,
    )

    assert torch.equal(converted["weight"], qweight.view(torch.int32))
    assert converted["weight"].shape == (1, 4, 2)
    assert torch.equal(converted["weight_scale"], scales)
    expected_qzeros = (
        qzeros.transpose(-1, -2)
        .contiguous()
        .view(torch.int32)
        .transpose(-1, -2)
        .contiguous()
    )
    assert torch.equal(converted["zero_point"], expected_qzeros)
    assert converted["zero_point"].shape == (1, 2, 2)


def test_moe_wna16_uses_humming_quant_config(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import humming_utils

    method = object.__new__(MoeWNA16Method)
    method.wna16_backend = WNA16MoEBackend.HUMMING
    layer = object()
    quant_config = object()
    monkeypatch.setattr(
        humming_utils,
        "get_humming_moe_quant_config",
        lambda actual_layer, *args, **kwargs: (
            quant_config if actual_layer is layer else None
        ),
    )

    assert method.get_fused_moe_quant_config(layer) is quant_config


def test_moe_wna16_propagates_packed_modules_mapping_to_linear_delegate():
    """A fused linear layer must reach the delegate as quantized, not unquantized.

    `MoeWNA16Config.get_quant_method` rebuilds the linear delegate with
    `AutoGPTQConfig.from_config(self.full_config)`, which sees only the raw HF
    quantization dict. `packed_modules_mapping` is attached to the parent config
    later by the model loader, so unless it is forwarded the delegate cannot tell
    that `gate_up_proj` is a fusion of `gate_proj` and `up_proj`. It then matches
    nothing in `modules_in_block_to_quantize` and hands back
    `UnquantizedLinearMethod` -- and the checkpoint's `qweight` has nowhere to go.
    """
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod

    full_config = {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "quant_method": "gptq",
        # As emitted by AutoGPTQ: shard names, never the fused name.
        "modules_in_block_to_quantize": [["mlp.gate_proj", "mlp.up_proj"]],
    }
    config = MoeWNA16Config(
        linear_quant_method="gptq",
        weight_bits=4,
        group_size=128,
        has_zp=False,
        lm_head_quantized=False,
        modules_to_not_convert=None,
        full_config=full_config,
    )
    config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    assert "gate_up_proj" not in str(full_config["modules_in_block_to_quantize"]), (
        "premise: the checkpoint must list shard names, not the fused name"
    )

    layer = ColumnParallelLinear.__new__(ColumnParallelLinear)
    method = config.get_quant_method(layer, "model.layers.0.mlp.gate_up_proj")

    assert isinstance(method, AutoGPTQLinearMethod), (
        f"fused layer resolved to {type(method).__name__}; the delegate did not "
        "receive packed_modules_mapping, so its shards matched nothing"
    )


def test_xpu_platform_supports_moe_wna16():
    """Regression guard: XPU must keep `moe_wna16` in its quantization allowlist.

    Nothing else in this test file imports `vllm.platforms.xpu` (that module
    registers ops from the compiled `vllm_xpu_kernels` package, so it only
    imports cleanly on a real Intel XPU stack) -- skip everywhere else instead
    of failing collection on CUDA/CPU CI.
    """
    try:
        from vllm.platforms.xpu import XPUPlatform
    except ImportError:
        pytest.skip("vllm_xpu_kernels not importable outside an XPU stack")

    assert "moe_wna16" in XPUPlatform.supported_quantization
