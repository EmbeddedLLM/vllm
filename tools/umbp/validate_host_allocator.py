# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and benchmark the shared CPU offload allocator.

Run this inside the same container and with the same device/mount permissions
as vLLM. The allocator always cleans up its backing file before exit.
"""

import argparse
import json
import os
import time
import uuid
from typing import Any

import regex as re
import torch

from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.cpu.memory import CPUOffloadMemoryConfig
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("shm", "hugetlbfs"), default="shm")
    parser.add_argument(
        "--path",
        help="Backing directory; required for hugetlbfs (for example /dev/hugepages)",
    )
    parser.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--hugepage-size", choices=("2MB", "1GB"), default="2MB")
    parser.add_argument("--numa-node", type=int, default=-1)
    parser.add_argument("--no-prefault", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()
    if args.backend == "hugetlbfs" and not args.path:
        parser.error("--path is required for --backend hugetlbfs")
    if args.bytes <= 0:
        parser.error("--bytes must be positive")
    if args.iterations <= 0 or args.warmup_iterations < 0:
        parser.error("iteration counts must be non-negative and --iterations > 0")
    return args


def _numa_mapping(path: str) -> tuple[str, dict[int, int], int | None]:
    with open("/proc/self/numa_maps", encoding="utf-8") as numa_maps:
        line = next((line for line in numa_maps if path in line), "")
    if not line:
        raise RuntimeError(f"mapping {path!r} is absent from /proc/self/numa_maps")
    pages_by_node = {
        int(node): int(pages) for node, pages in re.findall(r"\bN(\d+)=(\d+)\b", line)
    }
    page_size_match = re.search(r"\bkernelpagesize_kB=(\d+)", line)
    page_size_kib = int(page_size_match.group(1)) if page_size_match else None
    return line.rstrip(), pages_by_node, page_size_kib


def _result_code(result: Any) -> int:
    return int(getattr(result, "value", result))


def _measure_copy_bandwidth(
    cpu_tensor: torch.Tensor,
    device: int,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, float | bool | int]:
    torch.accelerator.set_device_index(device)
    gpu_tensor = torch.empty_like(cpu_tensor, device=f"cuda:{device}")
    cudart = torch.cuda.cudart()
    register_result = cudart.cudaHostRegister(
        cpu_tensor.data_ptr(), cpu_tensor.numel(), 0
    )
    if _result_code(register_result) != 0:
        raise RuntimeError(f"cudaHostRegister failed: {register_result}")

    try:
        cpu_tensor.fill_(37)
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        torch.accelerator.synchronize()
        cpu_tensor.zero_()
        cpu_tensor.copy_(gpu_tensor, non_blocking=True)
        torch.accelerator.synchronize()
        round_trip_correct = bool(torch.all(cpu_tensor == 37).item())
        if not round_trip_correct:
            raise RuntimeError("GPU round-trip byte comparison failed")

        for _ in range(warmup_iterations):
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)
            cpu_tensor.copy_(gpu_tensor, non_blocking=True)
        torch.accelerator.synchronize()

        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        end.record()
        end.synchronize()
        h2d_seconds = start.elapsed_time(end) / 1000

        start.record()
        for _ in range(iterations):
            cpu_tensor.copy_(gpu_tensor, non_blocking=True)
        end.record()
        end.synchronize()
        d2h_seconds = start.elapsed_time(end) / 1000
    finally:
        unregister_result = cudart.cudaHostUnregister(cpu_tensor.data_ptr())
        if _result_code(unregister_result) != 0:
            raise RuntimeError(f"cudaHostUnregister failed: {unregister_result}")

    transferred_bytes = cpu_tensor.numel() * iterations
    return {
        "gpu_device": device,
        "gpu_round_trip_correct": round_trip_correct,
        "host_registered": True,
        "h2d_gbps": transferred_bytes / h2d_seconds / 1e9,
        "d2h_gbps": transferred_bytes / d2h_seconds / 1e9,
    }


def main() -> None:
    args = _parse_args()
    memory_config = CPUOffloadMemoryConfig.from_extra_config(
        {
            "cpu_memory_backend": args.backend,
            "cpu_memory_path": args.path,
            "cpu_hugepage_block_size": args.hugepage_size,
            "cpu_numa_node": args.numa_node,
            "cpu_prefault": not args.no_prefault,
        }
    )
    logical_size = round_up(args.bytes, SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT)
    engine_id = f"host-allocator-smoke-{os.getpid()}-{uuid.uuid4().hex}"
    started = time.perf_counter()
    region = SharedOffloadRegion(
        engine_id=engine_id,
        num_blocks=1,
        rank=0,
        kv_bytes_per_block=logical_size,
        cpu_page_size=logical_size,
        memory_config=memory_config,
    )
    allocation_seconds = time.perf_counter() - started
    result: dict[str, Any]
    mmap_path = region.mmap_path
    try:
        assert region._base is not None
        numa_line, pages_by_node, kernel_page_size_kib = _numa_mapping(region.mmap_path)
        if memory_config.numa_node is not None and memory_config.prefault:
            resident_bytes = sum(pages_by_node.values()) * (
                (kernel_page_size_kib or 0) * 1024
            )
            if set(pages_by_node) != {memory_config.numa_node} or (
                resident_bytes != region.mapped_size_bytes
            ):
                raise RuntimeError(
                    "NUMA placement mismatch: "
                    f"requested node {memory_config.numa_node}, "
                    f"observed {pages_by_node} ({resident_bytes} resident bytes)"
                )
        if args.backend == "hugetlbfs" and kernel_page_size_kib != (
            memory_config.hugepage_block_size // 1024
        ):
            raise RuntimeError(
                "hugetlbfs page-size mismatch: "
                f"expected {memory_config.hugepage_block_size // 1024} KiB, "
                f"observed {kernel_page_size_kib} KiB"
            )

        result = {
            "backend": args.backend,
            "path": args.path,
            "prefault": memory_config.prefault,
            "requested_numa_node": memory_config.numa_node,
            "logical_size_bytes": region.total_size_bytes,
            "mapped_size_bytes": region.mapped_size_bytes,
            "allocation_seconds": allocation_seconds,
            "kernel_page_size_kib": kernel_page_size_kib,
            "pages_by_numa_node": pages_by_node,
            "numa_maps_line": numa_line,
            "backing_file": region.mmap_path,
        }
        if args.skip_gpu:
            result.update(
                {
                    "host_registered": False,
                    "gpu_round_trip_correct": None,
                    "h2d_gbps": None,
                    "d2h_gbps": None,
                }
            )
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch does not expose a CUDA/ROCm GPU")
            result.update(
                _measure_copy_bandwidth(
                    region._base,
                    args.device,
                    args.iterations,
                    args.warmup_iterations,
                )
            )
    finally:
        region.cleanup()
        if os.path.exists(mmap_path):
            raise RuntimeError(f"allocator cleanup left backing file {mmap_path!r}")
    result["backing_file_removed"] = True
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
