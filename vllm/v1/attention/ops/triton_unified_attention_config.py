# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load device-specific offline attention configs without runtime benchmarking."""

import hashlib
import json
import logging
import os
from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, cast

from vllm import envs

logger = logging.getLogger(__name__)
is_batch_invariant = envs.VLLM_BATCH_INVARIANT
_BUNDLED_DIR = Path(__file__).parent / "configs" / "unified_attention"
_UNSET = object()
_override = _UNSET


@dataclass(frozen=True)
class AttentionConfig:
    block_m: int
    tile_size: int
    num_warps: int | None = None
    num_stages: int | None = None
    grf_mode: str | None = None
    heads_per_program: int | str = "MAX"

    def get_heads_per_program(self, num_queries_per_kv: int) -> int:
        return (
            num_queries_per_kv
            if self.heads_per_program == "MAX"
            else int(self.heads_per_program)
        )

    def launch_kwargs(self, backend: str) -> dict:
        options: dict[str, Any] = {
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }
        if backend == "xpu":
            options["grf_mode"] = self.grf_mode
        return {name: value for name, value in options.items() if value is not None}


@dataclass(frozen=True)
class AttentionStaticKey:
    block_size: int
    head_size: int
    num_queries_per_kv: int
    use_td: bool
    kv_quant_mode: int
    use_softcap: bool
    sliding_window: int
    q_dtype: str
    k_dtype: str
    v_dtype: str
    out_dtype: str


@dataclass(frozen=True)
class AttentionDynamicKey:
    is_3d: bool
    num_segments: int
    q_len_bucket: int
    kv_len_bucket: int
    num_seqs: int
    query_work: int

    def regime(self) -> tuple:
        return (
            self.is_3d,
            self.num_segments,
            self.q_len_bucket,
            self.kv_len_bucket,
        )


@dataclass(frozen=True)
class AttentionKey(AttentionStaticKey, AttentionDynamicKey):
    def static_key(self) -> AttentionStaticKey:
        return AttentionStaticKey(
            **{
                field.name: getattr(self, field.name)
                for field in fields(AttentionStaticKey)
            }
        )

    def dynamic_key(self) -> AttentionDynamicKey:
        return AttentionDynamicKey(
            **{
                field.name: getattr(self, field.name)
                for field in fields(AttentionDynamicKey)
            }
        )


def make_key(
    *,
    block_size,
    head_size,
    is_3d,
    num_queries_per_kv,
    use_td,
    max_seqlen_q,
    max_seqlen_k,
    kv_quant_mode,
    use_softcap,
    sliding_window,
    q_dtype,
    k_dtype,
    v_dtype,
    out_dtype,
    num_segments,
    num_seqs,
    query_work,
) -> AttentionKey:
    return AttentionKey(
        block_size=block_size,
        head_size=head_size,
        is_3d=is_3d,
        num_queries_per_kv=num_queries_per_kv,
        use_td=use_td,
        q_len_bucket=sum(max_seqlen_q > bound for bound in (1, 16, 512)),
        kv_len_bucket=sum(max_seqlen_k > bound for bound in (256, 1024, 4096, 8192)),
        kv_quant_mode=int(kv_quant_mode),
        use_softcap=use_softcap,
        sliding_window=max(0, sliding_window),
        **dict(
            zip(
                ("q_dtype", "k_dtype", "v_dtype", "out_dtype"),
                (
                    str(dtype).removeprefix("torch.")
                    for dtype in (q_dtype, k_dtype, v_dtype, out_dtype)
                ),
            )
        ),
        num_segments=num_segments,
        num_seqs=num_seqs,
        query_work=query_work,
    )


def use_3d_attention(
    *,
    num_seqs,
    max_seqlen_q,
    block_q,
    seq_threshold_3D,
    num_par_softmax_segments,
    softmax_segm_output,
    softmax_segm_max,
    softmax_segm_expsum,
    is_xpu,
    sliding_window,
    use_causal=True,
    use_per_seq_causal=False,
    use_mm_prefix=False,
    tuned_large_head=False,
) -> bool:
    """Choose the execution path independently of offline configs."""
    from vllm.v1.attention.ops.triton_unified_attention import MAX_3D_QUERY_LEN

    return not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or (
            max_seqlen_q > 1
            and (
                max_seqlen_q > MAX_3D_QUERY_LEN
                or not use_causal
                or use_per_seq_causal
                or use_mm_prefix
                or tuned_large_head
            )
        )
        or num_seqs * -(-max_seqlen_q // block_q) > seq_threshold_3D
        or is_batch_invariant
        # Keep short-window attention on the optimized 2D path.
        or (is_xpu and 0 < sliding_window <= 1024)
    )


def supports_profile_layout(q, k, v) -> bool:
    num_query_heads, head_size = q.shape[1:]
    num_kv_heads = k.shape[2]
    qkv_stride = ((num_query_heads + 2 * num_kv_heads) * head_size, head_size, 1)
    interleaved_stride = (
        v.shape[1] * num_kv_heads * 2 * head_size,
        num_kv_heads * 2 * head_size,
        2 * head_size,
        1,
    )
    return (q.is_contiguous() or q.stride() == qkv_stride) and (
        (k.is_contiguous() and v.is_contiguous())
        or k.stride() == v.stride() == interleaved_stride
    )


def make_attention_key(
    *,
    q,
    k,
    v,
    out,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    window_size,
    softcap,
    use_td=False,
    kv_quant_mode=0,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    is_3d=None,
    **_unused,
) -> AttentionKey:
    """Derive the selection key from attention inputs without running a kernel."""
    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads, head_size = q.shape[1:]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    is_xpu = q.device.type == "xpu"
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    if is_3d is None:
        # Default (pre-config) BLOCK_Q of the runtime wrapper.
        block_m = max(32 if is_xpu else 16, 1 << (num_queries_per_kv - 1).bit_length())
        is_3d = use_3d_attention(
            num_seqs=num_seqs,
            max_seqlen_q=max_seqlen_q,
            block_q=block_m // num_queries_per_kv,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            is_xpu=is_xpu,
            sliding_window=sliding_window_val,
        )

    return make_key(
        block_size=block_size,
        head_size=head_size,
        is_3d=is_3d,
        num_queries_per_kv=num_queries_per_kv,
        use_td=use_td,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        kv_quant_mode=kv_quant_mode,
        use_softcap=softcap > 0,
        sliding_window=sliding_window_val,
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        v_dtype=v.dtype,
        out_dtype=out.dtype,
        num_segments=num_par_softmax_segments if is_3d else 1,
        num_seqs=num_seqs,
        query_work=q.shape[0] * q.shape[1],
    )


def _power_of_two(value) -> bool:
    return type(value) is int and value > 0 and value & (value - 1) == 0


def validate_config(config: AttentionConfig, key: AttentionKey, backend: str) -> bool:
    """Check launch structure before benchmarking or applying a profile."""
    if key.num_queries_per_kv <= 0:
        return False
    if type(config.block_m) is not int or config.block_m not in (16, 32, 64, 128):
        return False
    if type(config.tile_size) is not int or config.tile_size not in (
        16,
        32,
        64,
        128,
        256,
    ):
        return False
    if config.heads_per_program != "MAX" and (
        type(config.heads_per_program) is not int or config.heads_per_program <= 0
    ):
        return False
    heads = config.get_heads_per_program(key.num_queries_per_kv)
    if key.num_queries_per_kv % heads:
        return False
    # Smaller groups currently support TD prefill only.
    if heads != key.num_queries_per_kv and (
        backend != "xpu"
        or not key.use_td
        or key.is_3d
        or key.q_len_bucket == 0
        or not _power_of_two(key.head_size)
    ):
        return False
    if config.block_m % heads:
        return False
    if config.num_warps is not None and (
        type(config.num_warps) is not int
        or config.num_warps not in (1, 2, 4, 8, 16, 32)
    ):
        return False
    if config.num_stages is not None and (
        type(config.num_stages) is not int or config.num_stages not in (1, 2, 3, 4)
    ):
        return False
    if config.grf_mode is not None and type(config.grf_mode) is not str:
        return False
    if backend != "xpu" and config.grf_mode not in (None, "default"):
        return False
    if key.block_size < config.tile_size or key.block_size % config.tile_size:
        return False
    if key.q_dtype.startswith("float8_") and config.tile_size < 32:
        return False
    if key.use_td and _power_of_two(key.head_size) and _power_of_two(heads):
        return _power_of_two(config.block_m // heads)
    return True


@contextmanager
def override_config(config: AttentionConfig | None):
    """Force a config during offline tuning; None selects the heuristic."""
    global _override
    previous = _override
    _override = config
    try:
        yield
    finally:
        _override = previous


@cache
def _device_identity(index: int) -> dict:
    import torch
    import triton

    return {
        "device_name": torch.xpu.get_device_name(index),
        "triton_version": triton.__version__,
    }


def get_profile_identity(device) -> dict:
    import torch

    device = torch.device(device)
    if device.type != "xpu":
        raise ValueError("Offline attention profiles currently target XPU")
    index = device.index if device.index is not None else torch.xpu.current_device()
    return dict(_device_identity(index))


def profile_filename(identity: dict, static_key: AttentionStaticKey) -> str:
    device = identity["device_name"].replace(" ", "_")
    dtypes = (
        static_key.q_dtype,
        static_key.k_dtype,
        static_key.v_dtype,
        static_key.out_dtype,
    )
    dtype = hashlib.sha256(
        json.dumps(dtypes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:8]
    return (
        f"H={static_key.head_size},G={static_key.num_queries_per_kv}"
        f",B={static_key.block_size},dev={device},dt={dtype}"
        f",td={int(static_key.use_td)},kq={static_key.kv_quant_mode}"
        f",sc={int(static_key.use_softcap)},w={static_key.sliding_window}.json"
    )


def _parse_profile(
    data: dict, device_name: str, static_key: AttentionStaticKey
) -> dict:
    if data["device_name"] != device_name or data["static_key"] != asdict(static_key):
        raise ValueError("profile device or static key mismatch")
    tables: dict[tuple, Any] = {}
    for entry in data["entries"]:
        dynamic = AttentionDynamicKey(**entry["key"])
        if (
            dynamic.query_work <= 0
            or dynamic.num_seqs <= 0
            or not _power_of_two(dynamic.num_segments)
            or (not dynamic.is_3d and dynamic.num_segments != 1)
            or dynamic.q_len_bucket not in range(4)
            or dynamic.kv_len_bucket not in range(5)
        ):
            raise ValueError("invalid dynamic key")
        config = (
            AttentionConfig(**entry["config"]) if entry["config"] is not None else None
        )
        key = AttentionKey(**asdict(static_key), **asdict(dynamic))
        if config is not None and not validate_config(config, key, "xpu"):
            raise ValueError("invalid attention config")
        anchors = tables.setdefault(dynamic.regime(), {}).setdefault(
            dynamic.num_seqs, {}
        )
        if dynamic.query_work in anchors:
            raise ValueError("duplicate query-work point")
        anchors[dynamic.query_work] = config
    return {
        regime: {
            num_seqs: (
                tuple(sorted(anchors)),
                tuple(anchors[work] for work in sorted(anchors)),
            )
            for num_seqs, anchors in sequence_groups.items()
        }
        for regime, sequence_groups in tables.items()
    }


@cache
def _load_profile(path: str, device_name: str, static_key: AttentionStaticKey) -> dict:
    try:
        with open(path) as source:
            return _parse_profile(json.load(source), device_name, static_key)
    except FileNotFoundError:
        return {}
    except (OSError, KeyError, TypeError, ValueError) as error:
        logger.warning("Ignoring attention profile %s: %s", path, error)
        return {}


@cache
def _profile_sources(index: int, static_key: AttentionStaticKey, folder: str | None):
    identity = _device_identity(index)
    filename = profile_filename(identity, static_key)
    paths = [Path(folder) / filename] if folder else []
    paths.append(_BUNDLED_DIR / filename)
    return tuple(
        _load_profile(str(path), identity["device_name"], static_key) for path in paths
    )


@lru_cache(maxsize=4096)
def _resolve_profile(key: AttentionKey, index: int, folder: str | None):
    for tables in _profile_sources(index, key.static_key(), folder):
        table = tables.get(key.regime())
        if table is None:
            continue
        # TODO: Evaluate joint seq-count/query-work selection.
        num_seqs = min(table, key=lambda n: (abs(n - key.num_seqs), n))
        anchors, configs = table[num_seqs]
        pos = bisect_left(anchors, key.query_work)
        candidates = (max(0, pos - 1), min(pos, len(anchors) - 1))
        pos = min(
            candidates, key=lambda i: (abs(anchors[i] - key.query_work), anchors[i])
        )
        return configs[pos]
    return None


@cache
def _has_device_profiles(index: int, folder: str | None) -> bool:
    device_name = _device_identity(index)["device_name"].replace(" ", "_")
    marker = f",dev={device_name},dt="
    directories = [Path(folder), _BUNDLED_DIR] if folder else [_BUNDLED_DIR]
    for directory in directories:
        try:
            if any(
                marker in path.name and path.suffix == ".json" and path.is_file()
                for path in directory.iterdir()
            ):
                return True
        except FileNotFoundError:
            continue
        except OSError as error:
            logger.warning(
                "Cannot inspect attention configs in %s: %s", directory, error
            )
    return False


def needs_config_key(device) -> bool:
    """Skip profile work on devices without configs; always honor tuning overrides."""
    if _override is not _UNSET:
        return True
    if device.type != "xpu":
        return False
    index = device.index
    if index is None:
        import torch

        index = torch.xpu.current_device()
    return _has_device_profiles(index, os.environ.get("VLLM_TUNED_CONFIG_FOLDER"))


def clear_profile_cache():
    """Reload files and decisions after replacing offline profiles."""
    _resolve_profile.cache_clear()
    _profile_sources.cache_clear()
    _load_profile.cache_clear()
    _device_identity.cache_clear()
    _has_device_profiles.cache_clear()


def resolve_config(key: AttentionKey, device, *, eligible: bool = True):
    if _override is not _UNSET:
        override = cast(AttentionConfig | None, _override)
        if override is not None and (
            not eligible or not validate_config(override, key, device.type)
        ):
            raise ValueError("Forced attention config is invalid for this workload")
        return override
    if not eligible or device.type != "xpu":
        return None
    index = device.index
    if index is None:
        import torch

        index = torch.xpu.current_device()
    return _resolve_profile(key, index, os.environ.get("VLLM_TUNED_CONFIG_FOLDER"))


def write_profile(path, identity: dict, entries, *, static_key: AttentionStaticKey):
    data = dict(identity, static_key=asdict(static_key))
    data["entries"] = [
        {
            "key": asdict(dynamic),
            "config": asdict(config) if config is not None else None,
        }
        for dynamic, config in entries
    ]
    _parse_profile(data, identity["device_name"], static_key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)
