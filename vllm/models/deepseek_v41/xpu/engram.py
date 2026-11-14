# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU Engram host tables stored as pinned chunks.

On XPU the device reads an offloaded Engram table through a UVA view, which
needs pinned host memory. The caching host allocator rounds every pinned
request up to a power of two, and a single host allocation larger than the
device memory fails (B60: 22.7 GiB, B70: 31.9 GiB), so a TP<=4 rank slice of a
DeepSeek-V4.1-Flash table (25.4 GB -> 32 GiB) cannot be pinned in one piece.

`XPUChunkedEngramEmbedding` keeps such a slice as several pinned chunks of at
most `CHUNK_BYTES`, each laid out as `rows x row_bytes` values followed by
`rows x scale_dim` scales, and hands their device views to the lookup kernel's
existing `PEER_ROWS` path (a per-chunk base-pointer table), so the kernel is
unchanged. Slices that fit in one chunk use the parent class as before.
"""

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from ..common.engram import ParallelEngramEmbedding

logger = init_logger(__name__)

# Largest pinned allocation that survives power-of-two rounding on every BMG card.
CHUNK_BYTES = 16 * 1024**3
# Dummy-weight convention of the parent class: FP8 value 1.0 and ue8m0 scale 1.0.
_FP8_ONE = 0x38
_UE8M0_ONE = 127


class XPUChunkedEngramEmbedding(ParallelEngramEmbedding):
    _chunks: list[torch.Tensor] | None = None

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        scale_dim = self.dim // self.block_size
        bytes_per_row = self.row_bytes + scale_dim
        rows = self.part_num_embeddings
        if (
            torch.get_default_device().type == "meta"
            or not self.cpu_offload
            or self.peer_rows
            or rows * bytes_per_row <= CHUNK_BYTES
        ):
            return super()._allocate_weights()

        chunk_rows = CHUNK_BYTES // bytes_per_row
        num_chunks = -(-rows // chunk_rows)
        self._chunks = []
        for _ in range(num_chunks):
            chunk = torch.empty(
                chunk_rows * bytes_per_row,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            chunk[: chunk_rows * self.row_bytes].fill_(_FP8_ONE)
            chunk[chunk_rows * self.row_bytes :].fill_(_UE8M0_ONE)
            self._chunks.append(chunk)
        self._chunk_views = [
            get_accelerator_view_from_cpu_tensor(c) for c in self._chunks
        ]
        device = self._chunk_views[0].device
        self.register_buffer(
            "peer_ptrs",
            torch.tensor(
                [v.data_ptr() for v in self._chunk_views],
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        # The lookup kernel's PEER_ROWS path: owner = local row // PEER_ROWS.
        self.peer_rows = chunk_rows
        self._weight_loader = self._load_into_chunks
        logger.info(
            "Engram table slice of %d rows (%.2f GiB) stored as %d pinned chunks "
            "of %d rows on XPU.",
            rows,
            rows * bytes_per_row / 1024**3,
            num_chunks,
            chunk_rows,
        )
        weight_dtype = torch.uint8 if self.packed_fp4 else torch.float8_e4m3fn
        # Placeholders: the rows live in the chunks; loading writes there.
        return (
            torch.empty(0, self.row_bytes, dtype=weight_dtype, device="cpu"),
            torch.empty(0, scale_dim, dtype=torch.uint8, device="cpu"),
        )

    def _load_into_chunks(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        assert self._chunks is not None
        if loaded_weight.dtype == torch.float8_e8m0fnu:
            loaded_weight = loaded_weight.view(torch.uint8)
        is_scale = param is self.weight_scale_inv
        width = self.dim // self.block_size if is_scale else self.row_bytes
        rows = self.part_num_embeddings
        shard = loaded_weight.narrow(0, param.engram_vocab_start, rows).view(
            torch.uint8
        )
        assert shard.shape == (rows, width), (shard.shape, rows, width)
        chunk_rows = self.peer_rows
        base = chunk_rows * self.row_bytes if is_scale else 0
        for k, chunk in enumerate(self._chunks):
            start = k * chunk_rows
            n = min(chunk_rows, rows - start)
            if n <= 0:
                break
            dst = chunk[base : base + chunk_rows * width].view(chunk_rows, width)
            dst[:n].copy_(shard[start : start + n])

    def process_weights_after_loading(self) -> None:
        # The parent synchronises a peer group; chunks are rank-local.
        if self._chunks is None:
            super().process_weights_after_loading()
