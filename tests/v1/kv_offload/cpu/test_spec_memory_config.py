# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU offload backing-memory configuration in offloading specs."""

import mmap
from typing import Any
from unittest.mock import MagicMock

import vllm.v1.kv_offload.cpu.spec as cpu_spec_module
import vllm.v1.kv_offload.tiering.spec as tiering_spec_module
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.memory import (
    HUGEPAGE_2MB,
    CPUOffloadMemoryBackend,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

PAGE_SIZE = mmap.PAGESIZE
BLOCK_SIZE = 16


def _make_offloading_config(
    extra_config: dict[str, Any],
    *,
    world_size: int = 1,
    worker_kv_bytes_per_block: int = PAGE_SIZE,
) -> OffloadingConfig:
    return OffloadingConfig(
        groups=(
            OffloadingGroupConfig(
                tokens_per_block=BLOCK_SIZE,
                layer_names=("layer",),
            ),
        ),
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=False,
        extra_config=extra_config,
        engine_id="spec-memory-config-test",
        model=OffloadingModelConfig(name="test-model", dtype="float32"),
        cache=OffloadingCacheConfig(
            tokens_per_hash=BLOCK_SIZE,
            blocks_per_chunk=1,
        ),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=world_size,
            tp_size=world_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
    )


def test_cpu_offloading_spec_default_keeps_shared_shm_backend() -> None:
    spec = CPUOffloadingSpec(
        _make_offloading_config({"cpu_bytes_to_use": 4 * PAGE_SIZE})
    )

    assert spec.cpu_memory_config.backend == CPUOffloadMemoryBackend.DEFAULT
    assert spec.cpu_memory_config.effective_backend == CPUOffloadMemoryBackend.SHM
    assert spec.num_blocks == 4


def test_cpu_spec_threads_memory_config_to_shared_worker(monkeypatch, tmp_path) -> None:
    region_calls: list[dict[str, Any]] = []

    def fake_region(**kwargs: Any) -> MagicMock:
        region_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(cpu_spec_module, "SharedOffloadRegion", fake_region)
    monkeypatch.setattr(cpu_spec_module, "CPUOffloadingWorker", MagicMock())
    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(
        cpu_spec_module.torch.accelerator, "current_device_index", lambda: 0
    )
    spec = CPUOffloadingSpec(
        _make_offloading_config(
            {
                "cpu_bytes_to_use": 4 * PAGE_SIZE,
                "cpu_memory_backend": "hugetlbfs",
                "cpu_memory_path": str(tmp_path),
                "cpu_numa_node": 1,
            }
        )
    )

    spec.create_worker(MagicMock())

    assert region_calls[0]["memory_config"] is spec.cpu_memory_config
    assert spec.cpu_memory_config.numa_node == 1


def test_tiering_spec_threads_config_to_scheduler_and_worker(
    monkeypatch, tmp_path
) -> None:
    created_regions: list[MagicMock] = []

    def fake_region(**kwargs: Any) -> MagicMock:
        region = MagicMock()
        region.rank = kwargs["rank"]
        region.memory_config = kwargs["memory_config"]
        region.create_kv_memoryview.return_value = memoryview(bytearray(PAGE_SIZE))
        created_regions.append(region)
        return region

    monkeypatch.setattr(tiering_spec_module, "SharedOffloadRegion", fake_region)
    monkeypatch.setattr(tiering_spec_module, "CPUOffloadingWorker", MagicMock())
    monkeypatch.setattr(
        tiering_spec_module.torch.accelerator, "current_device_index", lambda: 1
    )
    spec = TieringOffloadingSpec(
        _make_offloading_config(
            {
                "cpu_bytes_to_use": 8 * PAGE_SIZE,
                "spec_name": "TieringOffloadingSpec",
                "cpu_memory_backend": "hugetlbfs",
                "cpu_memory_path": str(tmp_path),
                "cpu_numa_node": 1,
                "secondary_tiers": [],
            },
            world_size=2,
        )
    )

    spec.get_manager()
    spec.create_worker(MagicMock())

    assert created_regions[0].rank is None
    assert created_regions[1].rank == 1
    assert all(
        region.memory_config is spec.cpu_memory_config for region in created_regions
    )


def test_tiering_num_blocks_use_logical_bytes_not_hugepage_padding(tmp_path) -> None:
    spec = TieringOffloadingSpec(
        _make_offloading_config(
            {
                "cpu_bytes_to_use": PAGE_SIZE + 1,
                "spec_name": "TieringOffloadingSpec",
                "cpu_memory_backend": "hugetlbfs",
                "cpu_memory_path": str(tmp_path),
            }
        )
    )

    logical_size = spec.num_blocks * spec.kv_bytes_per_chunk
    assert spec.num_blocks == 1
    assert logical_size == PAGE_SIZE
    assert spec.cpu_memory_config.mapped_size(logical_size) == HUGEPAGE_2MB
