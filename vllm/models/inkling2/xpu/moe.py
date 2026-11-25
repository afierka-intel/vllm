# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU copy of `vllm/models/inkling2/nvidia/moe.py` (upstream `b7186d137a`).
Adapted: Hopper's flat-row flow only (`select_experts`, a `FlatExperts`
backend of `experts.py`, `combine`); no SM10x / SM12x routing, Humming,
sequence-parallel push or expert LoRA.

Inkling2 mixture-of-experts layer (routed experts only, NVFP4 or MXFP8).

Experts are sharded contiguously over the TP group (EP size = TP size); each
rank returns the full all-reduced bf16 output. Checkpoint names under
`model.llm.layers.N.mlp` map 1:1 to parameters.
"""

from collections.abc import Callable, Sequence
from typing import Literal, Protocol

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs

from .experts import xpu_flat_experts
from .ops.moe import combine, select_experts

QuantFormat = Literal["nvfp4", "mxfp8"]


class FlatExperts(Protocol):
    """Expert GEMMs over the flat (token * topk + slot) rows of
    `select_experts`: built after loading as `cls(weights, shapes, nvfp4)`
    from the layer's checkpoint-layout weights, block scales and NVFP4
    global scales of `w13` (interleaved gate / up rows) and `w2`. A call
    returns the bf16 `[T * topk, H]` down-projection rows of the local slots
    of `ids` (-1: not local), with the softcapped SiTU (`gate_cap`,
    `up_cap`) of the `w13` output scaled by the routing `weights`; the other
    rows are left unset.

    A backend whose `combines` is True (the SYCL fused MoE) returns the
    combined bf16 `[T, H]` output instead: the routing weight scales each
    down-projection row in fp32 and the sum is rounded once to bf16, as
    `combine` does. Without the attribute it is False.

    The backends are built from the layer's Parameters and keep them, in the
    checkpoint layout or rewritten in place into their kernel layout, so the
    weights stay module state."""

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
    ) -> torch.Tensor: ...


class Inkling2Gate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.use_gate_bias
        num_experts, hidden = config.n_routed_experts, config.hidden_size
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        for param in (self.weight, self.bias):
            set_weight_attrs(param, {"weight_loader": default_weight_loader})


class Inkling2Experts(nn.Module):
    def __init__(self, config, quant_format: QuantFormat) -> None:
        super().__init__()
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        assert config.n_routed_experts % self.ep_size == 0
        self.num_experts = config.n_routed_experts // self.ep_size
        self.topk = config.num_experts_per_tok
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.nvfp4 = quant_format == "nvfp4"
        hidden, inter = config.hidden_size, config.intermediate_size
        self.hidden_size, self.intermediate_size = hidden, inter

        pack, group = (2, 16) if self.nvfp4 else (1, 32)
        weight_dtype = torch.uint8 if self.nvfp4 else torch.float8_e4m3fn
        scale_dtype = torch.float8_e4m3fn if self.nvfp4 else torch.uint8
        self.shapes = ((2 * inter, hidden), (hidden, inter))
        for name, (n, k) in zip(("w13", "w2"), self.shapes):
            self._register(f"{name}_weight", (n, k // pack), weight_dtype)
            self._register(f"{name}_weight_scale", (n, k // group), scale_dtype)
            if self.nvfp4:
                self._register(f"{name}_weight_amax", (), torch.float32)
                self._register(
                    f"{name}_weight_outer_scale_denominator",
                    (1,),
                    torch.float32,
                    per_expert=False,
                )

        start = self.ep_rank * self.num_experts
        self.register_buffer(
            "expert_bounds",
            torch.tensor([start, start + self.num_experts], dtype=torch.int32),
            persistent=False,
        )
        # The expert GEMM backend, built after loading; unsupported formats
        # fail here.
        self.flat_experts: FlatExperts | None = None
        self.flat_experts_cls = xpu_flat_experts(
            quant_format, self.topk, self.gate_cap, self.up_cap
        )

    def _register(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        per_expert: bool = True,
    ) -> None:
        if per_expert:
            shape = (self.num_experts, *shape)
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        loader = self._load_expert_shard if per_expert else self._load_full
        set_weight_attrs(param, {"weight_loader": loader})
        self.register_parameter(name, param)

    def _refuse_reload(self, param: nn.Parameter) -> None:
        """Refuse a weight load into a built layer, before anything is
        written: the backend computes on the Parameters in its own layout
        (and NVFP4 global scales were taken at the build), so a reload,
        direct or layerwise (whose copy-back writes the loaded layout into
        them), would corrupt the experts."""
        if self.flat_experts is not None:
            raise RuntimeError(
                f"Inkling2Experts: {type(self.flat_experts).__name__} computes on "
                "the routed experts' Parameters in its own layout; reloading "
                "their weights into a built layer is not supported "
                f"(parameter of shape {tuple(param.shape)})."
            )

    def _load_expert_shard(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        start = self.ep_rank * self.num_experts
        self._load_full(param, loaded[start : start + self.num_experts])

    def _load_full(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._refuse_reload(param)
        assert loaded.shape == param.shape and loaded.dtype == param.dtype, (
            f"{loaded.shape} {loaded.dtype} != {param.shape} {param.dtype}"
        )
        param.data.copy_(loaded)

    def process_weights_after_loading(self) -> None:
        if self.flat_experts is not None:
            raise RuntimeError(
                "Inkling2Experts.process_weights_after_loading was already "
                "called: the expert kernels were built from the loaded "
                "weights, which now hold the kernel layout. Reloading the "
                "routed experts' weights into a built layer is not supported."
            )
        weights = {}
        for name in ("w13", "w2"):
            amax, denominator = self.scales(name)
            global_scale = None
            if amax is not None and denominator is not None:
                global_scale = amax.data / denominator.data
            weights[name] = (
                getattr(self, f"{name}_weight"),
                getattr(self, f"{name}_weight_scale"),
                global_scale,
            )
        self.flat_experts = self.flat_experts_cls(weights, self.shapes, self.nvfp4)

    def scales(self, name: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Outer scale amax and denominator of a projection's NVFP4 weights."""
        if not self.nvfp4:
            return None, None
        return (
            getattr(self, f"{name}_weight_amax"),
            getattr(self, f"{name}_weight_outer_scale_denominator"),
        )

    def forward(self, x: torch.Tensor, routing) -> torch.Tensor:
        """bf16 `[T, H]` -> bf16 `[T, H]` reduced over the expert-parallel
        group, for the `routing` of `select_experts`. The routing weight is
        applied inside the SwiGLU, before the down projection, except by a
        `combines` backend, which applies it to the down-projection rows in
        fp32 while summing them."""
        assert self.flat_experts is not None, "process_weights_after_loading"
        ids, gammas, src2dst = routing
        down = self.flat_experts(
            x,
            ids,
            gammas,
            src2dst,
            self.num_experts * self.ep_size,
            self.gate_cap,
            self.up_cap,
        )
        if getattr(self.flat_experts, "combines", False):
            # The fp32 combine was rounded once to bf16 by the experts.
            out = down
        else:
            # Each rank's fp32 combine is rounded once to bf16 before the
            # all-reduce.
            out = combine(down, src2dst, self.topk, x.dtype)
        return out if self.ep_size == 1 else tensor_model_parallel_all_reduce(out)


class Inkling2RoutingCapture(nn.Module):
    """A routed-experts capture row (`RoutedExpertsCaptureSource`): with
    `enable_return_routed_experts`, vLLM binds `capture_fn` to record each
    token's expert ids at row `layer_id` of the capture buffer."""

    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.capture_fn: Callable[[torch.Tensor], None] | None = None


class Inkling2MoE(nn.Module):
    def __init__(
        self, config, quant_format: QuantFormat, visit_ids: Sequence[int] = (0,)
    ) -> None:
        """`visit_ids`: the routed-experts capture row of each pass of the
        layer (`Inkling2ModelConfig.layer_visit_ids`)."""
        super().__init__()
        assert config.n_shared_experts == 0
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2Experts(config, quant_format)
        self.routing_captures = nn.ModuleList(
            Inkling2RoutingCapture(visit) for visit in visit_ids
        )
        self.register_buffer(
            "all_experts",
            torch.tensor([0, config.n_routed_experts], dtype=torch.int32),
            persistent=False,
        )

    def routed_expert_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Each token's top-k expert ids `[T, topk]` over all experts: the
        selection of `select_experts`."""
        gate = self.gate
        ids, _, _ = select_experts(
            x, gate.weight, gate.bias, gate.topk, gate.route_scale, self.all_experts
        )
        return ids

    def forward(self, x: torch.Tensor, pass_idx: int = 0) -> torch.Tensor:
        """bf16 [T, H] -> bf16 [T, H], reduced over the expert-parallel group.
        Padding tokens (`is_padding` of the forward context) are routed to
        no expert and come out as 0. With routed-experts capture, the expert
        ids are recorded at the row of loop pass `pass_idx`."""
        x = x.contiguous()
        capture_fn = self.routing_captures[pass_idx].capture_fn
        if capture_fn is not None:
            capture_fn(self.routed_expert_ids(x))
        gate, experts = self.gate, self.experts
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
        selected = select_experts(
            x,
            gate.weight,
            gate.bias,
            gate.topk,
            gate.route_scale,
            experts.expert_bounds,
            is_padding,
        )
        return experts(x, selected)
