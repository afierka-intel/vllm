# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU mHC ops: the fused RMSNorm (``norm_weight``) must be applied.

The SYCL ``mhc_pre`` / ``mhc_fused_post_pre`` ops have no norm argument, so
``forward_xpu`` applies the layer RMSNorm on the returned layer input, like the
CUDA/ROCm fallbacks. Models such as GLM-5.3-Flash rely on it: their input and
post-attention RMSNorms run only through these ops.
"""

import pytest
import torch

from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch
from vllm.model_executor.layers.mhc import MHCFusedPostPreOp, MHCPreOp
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="XPU mHC ops required"
)

DEVICE = "xpu"
HC_MULT = 4
MIX_HC = (2 + HC_MULT) * HC_MULT


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * weight.float()).to(x.dtype)


def _mhc_params(hidden_size: int):
    fn = torch.randn(MIX_HC, HC_MULT * hidden_size, device=DEVICE) * 0.02
    scale = torch.tensor([0.5, 0.25, 1.0], device=DEVICE)
    base = torch.randn(MIX_HC, device=DEVICE)
    weight = torch.empty(hidden_size, dtype=torch.bfloat16, device=DEVICE)
    weight.uniform_(0.5, 1.5)
    return fn, scale, base, weight


# rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
_HC_ARGS = (1e-6, 1e-6, 1e-6, 2.0, 20)
_NORM_EPS = 1e-5


@pytest.mark.parametrize("num_tokens", [1, 7, 128])
@pytest.mark.parametrize("hidden_size", [4096])
def test_mhc_pre_xpu_applies_norm(
    num_tokens: int, hidden_size: int, default_vllm_config
):
    set_random_seed(0)
    residual = torch.randn(
        num_tokens, HC_MULT, hidden_size, dtype=torch.bfloat16, device=DEVICE
    )
    fn, scale, base, weight = _mhc_params(hidden_size)
    args = (residual, fn, scale, base, *_HC_ARGS)
    op = MHCPreOp()

    post_raw, comb_raw, x_raw = op.forward_xpu(*args)
    post, comb, x = op.forward_xpu(*args, norm_weight=weight, norm_eps=_NORM_EPS)

    # Mixes are untouched by the norm; the layer input is exactly normalized.
    torch.testing.assert_close(post, post_raw, atol=0, rtol=0)
    torch.testing.assert_close(comb, comb_raw, atol=0, rtol=0)
    torch.testing.assert_close(
        x, _rms_norm_ref(x_raw, weight, _NORM_EPS), atol=2e-2, rtol=2e-2
    )

    # And the whole op matches the torch reference + RMSNorm.
    ref_post, ref_comb, ref_x = mhc_pre_torch(*args)
    torch.testing.assert_close(post, ref_post, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(comb, ref_comb, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        x, _rms_norm_ref(ref_x, weight, _NORM_EPS), atol=5e-2, rtol=5e-2
    )


@pytest.mark.parametrize("num_tokens", [1, 7, 128])
@pytest.mark.parametrize("hidden_size", [4096])
def test_mhc_fused_post_pre_xpu_applies_norm(
    num_tokens: int, hidden_size: int, default_vllm_config
):
    set_random_seed(0)
    residual = torch.randn(
        num_tokens, HC_MULT, hidden_size, dtype=torch.bfloat16, device=DEVICE
    )
    x_in = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=DEVICE)
    post_mix = torch.rand(num_tokens, HC_MULT, 1, device=DEVICE)
    comb_mix = torch.rand(num_tokens, HC_MULT, HC_MULT, device=DEVICE)
    fn, scale, base, weight = _mhc_params(hidden_size)
    args = (x_in, residual, post_mix, comb_mix, fn, scale, base, *_HC_ARGS)
    op = MHCFusedPostPreOp()

    res_raw, post_raw, comb_raw, x_raw = op.forward_xpu(*args)
    res, post, comb, x = op.forward_xpu(*args, norm_weight=weight, norm_eps=_NORM_EPS)

    torch.testing.assert_close(res, res_raw, atol=0, rtol=0)
    torch.testing.assert_close(post, post_raw, atol=0, rtol=0)
    torch.testing.assert_close(comb, comb_raw, atol=0, rtol=0)
    torch.testing.assert_close(
        x, _rms_norm_ref(x_raw, weight, _NORM_EPS), atol=2e-2, rtol=2e-2
    )

    ref_res = mhc_post_torch(x_in, residual, post_mix, comb_mix)
    ref_post, ref_comb, ref_x = mhc_pre_torch(ref_res, fn, scale, base, *_HC_ARGS)
    torch.testing.assert_close(res, ref_res, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(post, ref_post, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(comb, ref_comb, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        x, _rms_norm_ref(ref_x, weight, _NORM_EPS), atol=5e-2, rtol=5e-2
    )
