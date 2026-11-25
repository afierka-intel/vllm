# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 on XPU at model level: the platform dispatch and its import
isolation, the features XPU rejects (image inputs, LoRA, DSpark, FP8 KV, BF16
experts), the loop controllers, and tiny random-weight engines of both
architectures (TP=1, BF16 KV)."""

import json
import subprocess
import sys

import pytest
import torch

from tests.models.inkling2.xpu.utils import (
    DEVICE,
    TINY_CONFIG,
    TINY_LOOP_CONFIG,
    requires_xpu,
    write_config,
)
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    LoRAConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.platforms import current_platform

pytestmark = requires_xpu

_ISOLATION_PROBE = """
import json, sys
import vllm.models.inkling2.xpu.loop
import vllm.models.inkling2.xpu.model
from vllm.models import inkling2

classes = {name: getattr(inkling2, name).__module__ for name in inkling2.__all__}
print("PROBE=" + json.dumps({"classes": classes, "modules": sorted(sys.modules)}))
"""
_FORBIDDEN_MODULES = (
    "vllm.models.inkling2.nvidia",
    "vllm.models.inkling2.amd",
    "vllm.models.inkling.nvidia",
    "flashinfer",
    "humming",
)


def test_xpu_dispatch_imports_no_cuda_package():
    """On XPU the package exports resolve to `xpu/`, and importing its entry
    modules (in a fresh interpreter) loads no CUDA / ROCm copy of Inkling or
    Inkling 2 and neither FlashInfer nor Humming."""
    result = subprocess.run(
        [sys.executable, "-c", _ISOLATION_PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    line = next(x for x in result.stdout.splitlines() if x.startswith("PROBE="))
    probe = json.loads(line.removeprefix("PROBE="))
    assert probe["classes"] == {
        "Inkling2DSparkModel": "vllm.models.inkling2.xpu.model",
        "Inkling2ForConditionalGeneration": "vllm.models.inkling2.xpu.model",
        "Inkling2LoopForConditionalGeneration": "vllm.models.inkling2.xpu.loop",
    }
    loaded = [
        name
        for name in probe["modules"]
        if any(name == m or name.startswith(m + ".") for m in _FORBIDDEN_MODULES)
    ]
    assert not loaded, loaded


def _vllm_config(tmp_path, config=TINY_CONFIG, cache_dtype="auto", **model_kwargs):
    model_kwargs.setdefault("limit_mm_per_prompt", {"image": 0})
    return VllmConfig(
        model_config=ModelConfig(
            model=write_config(tmp_path, config),
            dtype="bfloat16",
            skip_tokenizer_init=True,
            tokenizer_mode="hf",
            max_model_len=2048,
            **model_kwargs,
        ),
        cache_config=CacheConfig(cache_dtype=cache_dtype),
    )


@pytest.mark.parametrize(
    "model_kwargs,rejected",
    [
        ({"limit_mm_per_prompt": {"image": 0}}, False),
        ({"limit_mm_per_prompt": {}, "language_model_only": True}, False),
        ({"limit_mm_per_prompt": {}}, True),
    ],
    ids=["no-images", "language-model-only", "default-image-limit"],
)
def test_image_inputs_are_rejected_at_model_init(tmp_path, model_kwargs, rejected):
    """Text-only serving (an image limit of 0, or `--language-model-only`)
    passes the model-init gate; the default image limit fails it, naming
    both ways to serve text only."""
    from vllm.models.inkling2.xpu.model import (
        _check_platform_support,
        xpu_unsupported_features,
    )

    config = _vllm_config(tmp_path, **model_kwargs)
    features = xpu_unsupported_features(config)
    if not rejected:
        assert features == []
        _check_platform_support(config)
        return
    assert [f.split(" (")[0] for f in features] == ["image inputs"]
    with pytest.raises(NotImplementedError, match="--language-model-only") as error:
        _check_platform_support(config)
    assert "--limit-mm-per-prompt" in str(error.value)


def test_lora_is_rejected_at_config(tmp_path):
    """The XPU classes do not implement `SupportsLoRA`, so a LoRA config
    fails when the engine config is built."""
    model_config = _vllm_config(tmp_path).model_config
    with pytest.raises(ValueError, match="LoRA is not supported for Inkling2"):
        VllmConfig(model_config=model_config, lora_config=LoRAConfig(max_lora_rank=8))


def test_dspark_draft_fails_to_build(tmp_path):
    """The DSpark draft architecture is a stub on XPU that fails at init."""
    from vllm.models.inkling2 import Inkling2DSparkModel

    with pytest.raises(NotImplementedError, match="does not support DSpark"):
        Inkling2DSparkModel(vllm_config=_vllm_config(tmp_path))


def _build_model(config):
    from vllm.models.inkling2.xpu.loop import Inkling2LoopForConditionalGeneration
    from vllm.models.inkling2.xpu.model import Inkling2ForConditionalGeneration

    looped = config.model_config.hf_config.text_config.loop_config is not None
    cls = (
        Inkling2LoopForConditionalGeneration
        if looped
        else Inkling2ForConditionalGeneration
    )
    with set_current_vllm_config(config), torch.device(DEVICE):
        return cls(vllm_config=config)


def test_fp8_kv_cache_is_rejected_at_model_init(tmp_path, dist_init):
    """An FP8 KV cache fails when the attention layers are built: the XPU
    kernels store and read BF16 K/V only."""
    config = _vllm_config(tmp_path, cache_dtype="fp8_e4m3")
    with pytest.raises(ValueError, match="kv-cache-dtype auto / bfloat16 only"):
        _build_model(config)


def test_bf16_expert_weights_are_rejected_at_load(tmp_path, dist_init):
    """A checkpoint with BF16 routed experts (instead of NVFP4 / MXFP8)
    fails at load, naming the tensor."""
    model = _build_model(_vllm_config(tmp_path))
    name = "model.llm.layers.0.mlp.experts.w13_weight"
    shape = model.get_parameter("model.layers.0.mlp.experts.w13_weight").shape
    weight = torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(ValueError, match="BF16 expert weights are not supported"):
        model.load_weights([(name, weight)])


@pytest.mark.parametrize("tokens", [1, 300])
def test_loop_blend_matches_per_stream_controllers(tmp_path, dist_init, tokens):
    """A loop core layer's stream-batched controllers (shared `dest` and
    norm weight, as a local layer's q/k/v) equal each stream's reference
    controller followed by its RMSNorm."""
    from vllm.models.inkling2.xpu.loop import Inkling2LoopDecoderLayer
    from vllm.models.inkling2.xpu.ops.norm import rms_norm

    torch.manual_seed(0)
    model = _build_model(_vllm_config(tmp_path, TINY_LOOP_CONFIG))
    layer = next(m for m in model.modules() if isinstance(m, Inkling2LoopDecoderLayer))
    layer.controller_down.normal_(std=0.02)
    layer.controller_gain_forget.normal_(std=0.1)
    dim, streams = 2304, 3
    dest = torch.randn(1, tokens, dim, device=DEVICE).expand(streams, -1, -1)
    message = torch.randn(streams, tokens, dim, device=DEVICE)
    norm_weight = torch.randn(1, dim, device=DEVICE).bfloat16().expand(streams, -1)
    out = layer.blend(slice(0, streams), dest, message, norm_weight, 1e-6)
    assert out.dtype == torch.bfloat16 and out.shape == (streams, tokens, dim)
    for s, controller in enumerate(list(layer.controllers.values())[:streams]):
        ref = rms_norm(controller(dest[s], message[s]), norm_weight[s], 1e-6)
        torch.testing.assert_close(out[s].float(), ref.float(), atol=2e-2, rtol=2e-2)


def _memory_fraction(fraction: float, min_gib: float = 4.0) -> float:
    """`fraction` of the device memory, at least `min_gib` GiB."""
    total = current_platform.get_device_total_memory() / 2**30
    return max(fraction, min_gib / total)


def _engine_kwargs(looped: bool, **kwargs):
    return {
        "load_format": "dummy",
        "skip_tokenizer_init": True,
        "tokenizer_mode": "hf",
        "gpu_memory_utilization": _memory_fraction(0.10 if looped else 0.035),
        "max_model_len": 2048,
        "max_num_seqs": 4,
        # Bounds warmup memory and chunks the long prompts.
        "max_num_batched_tokens": 256,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "limit_mm_per_prompt": {"image": 0},
        **kwargs,
    }


@pytest.mark.parametrize(
    "config", [TINY_CONFIG, TINY_LOOP_CONFIG], ids=["text", "loop"]
)
def test_tiny_model_generates(vllm_runner, tmp_path, config):
    """One local and one full layer (the loop: prelude, three passes of a
    local and a full core layer, coda) load with dummy weights and decode
    through the paged KV and conv-state caches and XPU graphs, for a
    chunked 298-token prompt next to a short one."""
    looped = config is TINY_LOOP_CONFIG
    prompts = [list(range(2, 300)), [5, 5, 5]]
    with vllm_runner(write_config(tmp_path, config), **_engine_kwargs(looped)) as vllm:
        outputs = vllm.llm.generate(
            [{"prompt_token_ids": p} for p in prompts],
            SamplingParams(
                max_tokens=24, temperature=0, ignore_eos=True, detokenize=False
            ),
        )
    for output in outputs:
        token_ids = output.outputs[0].token_ids
        assert len(token_ids) == 24
        assert all(0 <= t < 1000 for t in token_ids)


@pytest.mark.parametrize(
    "config,text_overrides",
    [
        (TINY_CONFIG, {}),
        (TINY_LOOP_CONFIG, {}),
        (TINY_LOOP_CONFIG, {"loop_num_passes": 1}),
    ],
    ids=["text", "loop", "loop-one-pass"],
)
def test_tiny_model_decode_matches_prefill(
    vllm_runner, tmp_path, config, text_overrides
):
    """Greedy decode steps (reading the KV and conv-state caches of earlier
    tokens; each loop pass its own) give the tokens and logprobs of a single
    prefill over the same sequence, also at an overridden pass count
    (variable-loop checkpoints)."""
    looped = config is TINY_LOOP_CONFIG
    prompt = list(range(2, 202))
    kwargs = _engine_kwargs(looped, hf_overrides={"text_config": text_overrides})
    with vllm_runner(write_config(tmp_path, config), **kwargs) as vllm:
        decoded = vllm.llm.generate(
            [{"prompt_token_ids": prompt}],
            SamplingParams(
                max_tokens=16,
                temperature=0,
                logprobs=0,
                ignore_eos=True,
                detokenize=False,
            ),
        )[0].outputs[0]
        full = vllm.llm.generate(
            [{"prompt_token_ids": prompt + list(decoded.token_ids)}],
            SamplingParams(
                max_tokens=1, prompt_logprobs=0, ignore_eos=True, detokenize=False
            ),
        )[0]
    assert full.prompt_logprobs is not None and decoded.logprobs is not None
    for step, token in enumerate(decoded.token_ids):
        step_logprob = decoded.logprobs[step][token].logprob
        prefill = full.prompt_logprobs[len(prompt) + step]
        assert prefill is not None and token in prefill
        assert abs(prefill[token].logprob - step_logprob) < 2e-2
        assert prefill[token].rank == 1


def _chunked_mixed_prompt_logprobs(vllm_runner, model, looped, enforce_eager):
    kwargs = _engine_kwargs(
        looped,
        max_num_seqs=16,
        enforce_eager=enforce_eager,
        # The 256-token piecewise graph is captured on 16 verify-sized requests.
        compilation_config={"cudagraph_capture_sizes": [8, 16, 256]},
    )
    # The short prompts decode while the long one prefills in 256-token chunks.
    prompts = [[7, 8, 9], [5, 5, 5], list(range(2, 702))]
    with vllm_runner(model, **kwargs) as vllm:
        outputs = vllm.llm.generate(
            [{"prompt_token_ids": p} for p in prompts],
            SamplingParams(
                max_tokens=8,
                temperature=0,
                ignore_eos=True,
                detokenize=False,
                prompt_logprobs=0,
            ),
        )
    logprobs = []
    for o in outputs:
        assert o.prompt_logprobs is not None
        logprobs.append(
            [next(iter(d.values())).logprob for d in o.prompt_logprobs[1:] if d]
        )
    return logprobs


@pytest.mark.parametrize(
    "config", [TINY_CONFIG, TINY_LOOP_CONFIG], ids=["text", "loop"]
)
def test_graphs_match_eager(vllm_runner, tmp_path, config):
    """Piecewise XPU graphs are captured on a dummy batch and replayed for any
    batch of the same token count, so attention plans from the live batch: a
    chunked prompt prefilling alongside decodes matches eager."""
    looped = config is TINY_LOOP_CONFIG
    model = write_config(tmp_path, config)
    graph = _chunked_mixed_prompt_logprobs(vllm_runner, model, looped, False)
    eager = _chunked_mixed_prompt_logprobs(vllm_runner, model, looped, True)
    for g, e in zip(graph, eager):
        torch.testing.assert_close(torch.tensor(g), torch.tensor(e), rtol=0, atol=1e-2)


def test_dspark_is_rejected_at_engine_init(vllm_runner, tmp_path, capfd):
    """A DSpark speculative config fails in the engine core when the target
    model is built, with the gate's error (logged by the engine core)."""
    kwargs = _engine_kwargs(
        False, speculative_config={"method": "dspark", "num_speculative_tokens": 7}
    )
    with pytest.raises(RuntimeError, match="Engine core initialization failed"):
        vllm_runner(write_config(tmp_path, TINY_CONFIG), **kwargs)
    err = capfd.readouterr().err
    assert "Inkling2 on XPU does not support DSpark speculative decoding" in err
