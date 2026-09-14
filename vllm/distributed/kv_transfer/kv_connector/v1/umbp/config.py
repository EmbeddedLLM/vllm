# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit per-client MoRI 1.2.3.post1 policy and storage ownership.

No mounts, model caches, master processes or shared cache directories are
created here. SSD roots must already exist. Only fresh private subdirectories
are used, and the store removes them after native teardown, never before it.
"""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.store import UMBPStore
from vllm.logger import init_logger

logger = init_logger(__name__)
_MAX_BYTES = (1 << 63) - 1


def _integer(
    value: int, name: str, minimum: int = 0, maximum: int = _MAX_BYTES
) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _text(value: str, name: str, *, empty: bool = False) -> None:
    if (
        not isinstance(value, str)
        or (not value and not empty)
        or any(char.isspace() or ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{name} must be a nonempty string without whitespace/control")


def _ssd_root(root: str) -> None:
    if (
        not isinstance(root, str)
        or not root
        or root != root.strip()
        or any(ord(char) < 32 or char == "," for char in root)
        or not Path(root).is_absolute()
        or ".." in Path(root).parts
    ):
        raise ValueError("SSD roots must be absolute paths without '..' or ','")


@dataclass(frozen=True)
class UMBPNodeConfig:
    """Runtime worker identity; never part of the shared cache key namespace.

    For a cluster, each client needs a unique node ID and peer-service port,
    plus advertised and RDMA hosts reachable by the other workers.
    """

    node_id: str
    node_address: str = "127.0.0.1"
    io_engine_host: str = ""
    peer_service_port: int = 0
    io_engine_port: int = 0

    def __post_init__(self) -> None:
        _text(self.node_id, "node_id")
        _text(self.node_address, "node_address")
        _text(self.io_engine_host, "io_engine_host", empty=True)
        for name in ("peer_service_port", "io_engine_port"):
            _integer(getattr(self, name), name, maximum=65535)


class _OwnedPaths:
    """Retain exact created paths, including a failed cleanup for later retry."""

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def directory(self, parent: Path | None = None) -> Path:
        path = Path(tempfile.mkdtemp(prefix="vllm-umbp-", dir=parent))
        self.paths.append(path)
        return path

    def close(self) -> None:
        while self.paths:
            path = self.paths[-1]
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                if path.exists():
                    raise
            self.paths.pop()


@dataclass(frozen=True)
class UMBPStoreConfig:
    """Per-worker storage budgets, not per-host budgets.

    At least one medium needs positive capacity. SSD capacity is the total
    across all supplied roots, which may grow from SSD0 to SSD0..SSD7 without
    changing existing mount names. Every worker creates its own subdirectories.
    page_size_bytes must agree across peers and fit their largest KV object.
    ranged_scratch_bytes sizes EACH of two independent remote I/O arenas.
    ssd_staging_slots also bounds native data batches, including DRAM-only
    clients of remote SSD peers. Use a common limit no larger than the smallest
    peer arena. Every KV object fits one page by the open_store geometry check.
    """

    page_size_bytes: int
    dram_capacity_bytes: int = 0
    ssd_capacity_bytes: int = 0
    ssd_roots: tuple[str, ...] = ()
    ranged_scratch_bytes: int = 64 << 20
    ssd_staging_slots: int = 16
    master_address: str = ""
    workers: int = 2
    max_pending: int = 8

    def __post_init__(self) -> None:
        _integer(self.page_size_bytes, "page_size_bytes", 4096, _MAX_BYTES // 8)
        if self.page_size_bytes % 4096:
            raise ValueError("page_size_bytes must be a multiple of 4096")
        for name in ("dram_capacity_bytes", "ssd_capacity_bytes"):
            value = getattr(self, name)
            _integer(value, name)
            if value and value < self.page_size_bytes:
                raise ValueError(f"{name} must hold at least one page")
        if not (self.dram_capacity_bytes or self.ssd_capacity_bytes):
            raise ValueError("At least one storage medium needs positive capacity")
        if not isinstance(self.ssd_roots, tuple):
            raise ValueError("ssd_roots must be an immutable tuple")
        for root in self.ssd_roots:
            _ssd_root(root)
        if bool(self.ssd_roots) != bool(self.ssd_capacity_bytes):
            raise ValueError("SSD capacity and roots must be configured together")
        if self.ssd_roots and (
            self.ssd_capacity_bytes // len(self.ssd_roots) < self.page_size_bytes
        ):
            raise ValueError("Each SSD root must have capacity for at least one page")
        _integer(self.ranged_scratch_bytes, "ranged_scratch_bytes", 1, _MAX_BYTES // 2)
        for name in ("ssd_staging_slots", "workers", "max_pending"):
            _integer(getattr(self, name), name, 1, (1 << 31) - 1)
        if self.ssd_staging_slots * self.page_size_bytes > _MAX_BYTES:
            raise ValueError("SSD staging budget overflows the native byte range")
        _text(self.master_address, "master_address", empty=True)
        if self.master_address:
            address = urlsplit("//" + self.master_address)
            if (
                not address.hostname
                or not address.port
                or address.username is not None
                or address.path
                or address.query
                or address.fragment
            ):
                raise ValueError("master_address must be host:port without credentials")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "UMBPStoreConfig":
        """Parse the connector's storage config without silently ignored keys."""
        if not isinstance(values, dict) or any(
            not isinstance(key, str) for key in values
        ):
            raise ValueError("UMBP storage config must be an object")
        unknown = values.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown UMBP storage options: {sorted(unknown)}")
        parsed = dict(values)
        if isinstance(parsed.get("ssd_roots"), list):
            parsed["ssd_roots"] = tuple(parsed["ssd_roots"])
        return cls(**parsed)

    def _policy(self, ssd_paths: tuple[Path, ...]) -> dict[str, Any]:
        backends: dict[str, Any] = {}
        tiers: list[dict[str, Any]] = []
        if self.dram_capacity_bytes:
            backends["dram"] = {
                "type": "dram",
                "capacity": f"{self.dram_capacity_bytes}B",
            }
            tiers.append({"name": "host", "backends": {"dram": 1}})
        if ssd_paths:
            backends["ssd"] = {
                "type": "ssd",
                "capacity": f"{self.ssd_capacity_bytes}B",
                "path": ",".join(str(path) for path in ssd_paths),
                "staging_slots": self.ssd_staging_slots,
            }
            if tiers:
                tiers[0].update(offload_to=["disk"], offload_trigger="on_evict")
            tiers.append({"name": "disk", "backends": {"ssd": 1}})
            if self.dram_capacity_bytes:
                tiers[-1].update(promote_trigger="on_read", promote_mode="copy")
        return {
            "schema_version": 1,
            "entry_tier": tiers[0]["name"],
            "backends": backends,
            "tiers": tiers,
        }

    def open_store(self, node: UMBPNodeConfig, *, max_object_bytes: int) -> UMBPStore:
        """Validate geometry, then initialize native storage with owned paths.

        max_object_bytes is derived from the registered worker layout, not an
        unverified user estimate. Callers must close the returned store.
        """
        _integer(max_object_bytes, "max_object_bytes", 1)
        if max_object_bytes > self.page_size_bytes:
            raise ValueError("UMBP page must fit the largest KV object")
        if max_object_bytes > self.ranged_scratch_bytes:
            raise ValueError(
                "Each UMBP ranged scratch arena must fit the largest object"
            )
        if self.master_address:
            if not node.io_engine_host or not node.peer_service_port:
                raise ValueError("A shared master requires RDMA host and peer port")
        elif node.io_engine_host or node.peer_service_port or node.io_engine_port:
            raise ValueError("Network endpoints require a shared master")
        if os.getenv("UMBP_WORKLOAD_TRACE_PATH"):
            raise ValueError(
                "Unset UMBP_WORKLOAD_TRACE_PATH; use captured service logs"
            )
        roots = tuple(Path(root).resolve(strict=True) for root in self.ssd_roots)
        for root in roots:
            _ssd_root(str(root))
        if len(set(roots)) != len(roots) or any(not root.is_dir() for root in roots):
            raise ValueError("SSD roots must be distinct existing directories")

        from mori.cpp import UMBPConfig, UMBPDistributedConfig

        owned = _OwnedPaths()
        try:
            ssd_paths = tuple(owned.directory(root) for root in roots)
            policy_path = owned.directory() / "policy.json"
            policy_path.write_text(
                json.dumps(self._policy(ssd_paths)), encoding="utf-8"
            )
            native = UMBPConfig()
            # All allocation comes from the explicit policy. This otherwise
            # unused value prevents WithEmbeddedDefaults shrinking the page.
            native.dram.capacity_bytes = max(
                self.dram_capacity_bytes, self.page_size_bytes * 8
            )
            native.ssd.enabled = False
            distributed = UMBPDistributedConfig()
            distributed.master_config.master_address = self.master_address
            distributed.master_config.node_id = node.node_id
            distributed.master_config.node_address = node.node_address
            distributed.master_config.auto_heartbeat = bool(self.master_address)
            distributed.io_engine.host = node.io_engine_host
            distributed.io_engine.port = node.io_engine_port
            distributed.peer_service_port = node.peer_service_port
            distributed.dram_page_size = self.page_size_bytes
            distributed.ranged_scratch_size = self.ranged_scratch_bytes
            # Re-cache from a GPU destination is not a valid CPU memcpy source.
            # Also avoid hidden duplicate network traffic from locality prefetch.
            distributed.cache_remote_fetches = False
            distributed.ranged_locality_prefetch = False
            distributed.local_first = True
            distributed.backend_policy_path = str(policy_path)
            native.distributed = distributed
            logger.info(
                "UMBP node %s owns storage paths: %s", node.node_id, owned.paths
            )
            store = UMBPStore.from_native_config(
                native,
                workers=self.workers,
                max_pending=self.max_pending,
                max_batch_objects=self.ssd_staging_slots,
                cleanup=owned.close,
            )
        except BaseException:
            owned.close()
            raise
        return store
