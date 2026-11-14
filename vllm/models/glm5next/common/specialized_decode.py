# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-neutral entry to the GLM-5.3-Flash specialized decode settings.

The specialized decode (`additional_config={"decode_backend":
"latency_specialized"}`) lives in nvidia/; on XPU it is unavailable, so this
module resolves the setting without importing nvidia/ there.
"""

from vllm.platforms import current_platform


def specialized_decode_config(vllm_config):
    """The deployment's specialized decode settings, None for the stock
    decode. Raises when the specialized decode is requested where it is not
    supported."""
    if not current_platform.is_xpu():
        from ..nvidia.specialized_decode.runtime import (
            specialized_decode_config as nvidia_specialized_decode_config,
        )

        return nvidia_specialized_decode_config(vllm_config)
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend != "auto":
        raise ValueError(
            f"GLM-5.3-Flash decode_backend={backend!r} is not supported on XPU; "
            "use 'auto'."
        )
    return None
