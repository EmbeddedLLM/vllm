# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered vLLM allocations mapped to rank-local, logical-order KV objects.

Objects concatenate layers in name order, with each page in logical H/N/C
order. Allocation capacity, physical packing and kernel block splitting do
not change the stored bytes. This format requires matching parallel topology;
it does not yet reshard objects between different producer/consumer TP sizes.
"""

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from fractions import Fraction
from itertools import product
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.store import (
    BufferSlice,
    MemoryRegion,
    TransferObject,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheLayout,
    UniformTypeKVCacheSpecs,
    create_kv_cache_views,
)


@dataclass(frozen=True)
class CacheTopology:
    """Parallel axes that determine the meaning of each rank's stored KV."""

    tp_size: int = 1
    pp_size: int = 1
    pcp_size: int = 1
    dcp_size: int = 1
    cp_interleave: int = 1

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("Cache topology dimensions must be positive integers")
        if self.tp_size % self.dcp_size:
            raise ValueError("DCP size must divide TP size")


def _identity(value: Any) -> Any:
    """Serialize semantic spec fields without addresses or unstable reprs."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                f.name: _identity(getattr(value, f.name)) for f in fields(value)
            },
        }
    if isinstance(value, Enum):
        return _identity(value.value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Fraction):
        return {"fraction": [value.numerator, value.denominator]}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Cache specification field names must be strings")
        return {key: _identity(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_identity(item) for item in value]
    if value is None or type(value) in (str, int, bool, float):
        return value
    raise ValueError(f"Unsupported cache specification field type: {type(value)}")


def _logical_runs(
    shape: tuple[int, ...], strides: tuple[int, ...], element_size: int
) -> tuple[tuple[int, int], ...]:
    """Contiguous byte runs in logical order, coalescing dense suffixes."""
    size = element_size
    split = len(shape)
    for index in reversed(range(len(shape))):
        if shape[index] != 1 and strides[index] != size:
            break
        size *= shape[index]
        split = index
    runs: list[tuple[int, int]] = []
    for indices in product(*(range(dim) for dim in shape[:split])):
        offset = sum(i * stride for i, stride in zip(indices, strides))
        if runs and runs[-1][0] + runs[-1][1] == offset:
            previous, length = runs[-1]
            runs[-1] = (previous, length + size)
        else:
            runs.append((offset, size))
    return tuple(runs)


@dataclass(frozen=True)
class _LayerPages:
    region: MemoryRegion
    offset: int
    block_stride: int
    num_blocks: int
    runs: tuple[tuple[int, int], ...]


class UMBPLayout:
    """Derive transfer slices from the current vLLM allocator contract.

    No native registration or I/O is performed here. The worker registers
    ``regions`` once and retains this layout until all transfers have drained.
    Cache-manager block ownership remains a scheduler responsibility.
    """

    def __init__(
        self,
        config: KVCacheConfig,
        caches: dict[str, torch.Tensor],
        topology: CacheTopology,
    ) -> None:
        if config.kv_cache_layout is None:
            raise ValueError("UMBP requires the resolved KV cache layout")
        physical_layout = KVCacheLayout[config.kv_cache_layout]
        self._groups: dict[int, tuple[_LayerPages, ...]] = {}
        regions: dict[int, MemoryRegion] = {}
        placements = {}
        for placement in config.kv_cache_tensors:
            if (
                placement.size <= 0
                or placement.layer_stride <= 0
                or placement.block_stride <= 0
                or placement.offset < 0
            ):
                raise ValueError("KV placement extents and strides must be positive")
            for index, name in enumerate(placement.layers):
                if name in placements:
                    raise ValueError(f"Duplicate KV cache placement for {name}")
                placements[name] = (placement, index)
        identities = []
        for group_id in config.transfer_group_ids:
            group = config.kv_cache_groups[group_id]
            pages = []
            layers = []
            for name in sorted(group.layer_names):
                if name not in placements or name not in caches:
                    raise ValueError(f"Missing KV cache allocation for {name}")
                placement, index = placements[name]
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                cache = caches[name]
                if cache.device.type not in ("cpu", "cuda"):
                    raise ValueError("UMBP supports CPU and CUDA/ROCm allocations")
                storage = cache.untyped_storage()
                ptr = storage.data_ptr()
                if storage.nbytes() < placement.size:
                    raise ValueError("KV allocation is smaller than its placement")
                region = regions.get(ptr)
                if region is None:
                    region = MemoryRegion(
                        ptr, storage.nbytes(), storage, cache.device.index
                    )
                    regions[ptr] = region
                elif (
                    region.size != storage.nbytes()
                    or region.device != cache.device.index
                ):
                    raise ValueError("Aliased KV allocations disagree on size/device")
                num_blocks = config.num_blocks_of(placement)
                if num_blocks <= 0:
                    raise ValueError("KV allocation must contain at least one block")
                offset = placement.offset + index * placement.layer_stride
                runs: tuple[tuple[int, int], ...]
                if not spec.has_layer_views:
                    if (
                        cache.dtype != torch.int8
                        or cache.ndim != 1
                        or cache.storage_offset() != 0
                        or not cache.is_contiguous()
                    ):
                        raise ValueError("Raw KV state must expose its backing buffer")
                    runs = ((0, spec.page_size_bytes),)
                else:
                    if cache.ndim != 4 or cache.shape[0] % num_blocks:
                        raise ValueError("KV view must contain whole manager blocks")
                    ratio = cache.shape[0] // num_blocks
                    if ratio <= 0 or spec.block_size % ratio:
                        raise ValueError("Invalid kernel/manager block ratio")
                    raw = torch.empty(0, dtype=torch.int8, device=cache.device).set_(
                        storage, 0, (storage.nbytes(),), (1,)
                    )
                    expected = create_kv_cache_views(
                        raw,
                        spec,
                        num_blocks,
                        physical_layout,
                        placement,
                        spec.block_size // ratio,
                    )[index]
                    if (
                        cache.shape != expected.shape
                        or cache.stride() != expected.stride()
                        or cache.storage_offset() != expected.storage_offset()
                        or cache.dtype != expected.dtype
                    ):
                        raise ValueError(
                            f"KV view disagrees with allocation for {name}"
                        )
                    elem = cache.element_size()
                    # H, kernel-block, N, C folds split kernel blocks into the
                    # same logical H/N/C bytes as an unsplit manager page.
                    runs = _logical_runs(
                        (cache.shape[1], ratio, cache.shape[2], cache.shape[3]),
                        tuple(cache.stride(dim) * elem for dim in (1, 0, 2, 3)),
                        elem,
                    )
                page = _LayerPages(
                    region, offset, placement.block_stride, num_blocks, runs
                )
                for start, size in runs:
                    BufferSlice(
                        region,
                        offset + (num_blocks - 1) * page.block_stride + start,
                        size,
                        0,
                    )
                pages.append(page)
                layers.append((name, _identity(spec)))
            if not pages:
                raise ValueError("Transferable KV groups must have layers")
            self._groups[group_id] = tuple(pages)
            identities.append(
                (group_id, group.role.value, group.is_eagle_group, layers)
            )
        if not self._groups:
            raise ValueError("No transferable KV cache groups")
        self.regions = tuple(regions.values())
        self.max_object_bytes = max(
            sum(size for page in pages for _, size in page.runs)
            for pages in self._groups.values()
        )
        self.num_shards = topology.tp_size * topology.pp_size * topology.pcp_size
        self.prefix_cacheable_group_ids = frozenset(config.prefix_cacheable_group_ids)
        self.identity = json.dumps(
            ("umbp-logical-rank-v1", asdict(topology), identities),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def block_object(self, key: str, *, group_id: int, block_id: int) -> TransferObject:
        """Describe one complete group block without copying any KV bytes."""
        if type(group_id) is not int or group_id not in self._groups:
            raise ValueError("Unknown or non-transferable KV cache group")
        parts = []
        object_offset = 0
        for page in self._groups[group_id]:
            if type(block_id) is not int or not 0 <= block_id < page.num_blocks:
                raise ValueError("KV block ID is outside the group's allocation")
            for offset, size in page.runs:
                parts.append(
                    BufferSlice(
                        page.region,
                        page.offset + block_id * page.block_stride + offset,
                        size,
                        object_offset,
                    )
                )
                object_offset += size
        return TransferObject(key, object_offset, tuple(parts))
