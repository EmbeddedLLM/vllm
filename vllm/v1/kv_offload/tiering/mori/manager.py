# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRI UMBP secondary tier for native KV cache offloading."""

import ctypes
import hashlib
import json
import threading
import time
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, cast

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
    TransferJob,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class MoriAsyncLookupManager(AsyncLookupManager):
    def __init__(self, tier: "MoriSecondaryTierManager") -> None:
        super().__init__(tier_type=tier.tier_type)
        self._tier = tier

    @override
    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        return self._tier._client.batch_exists(self._tier._encode_keys(keys))


class MoriSecondaryTierManager(SecondaryTierManager):
    """Distributed DRAM/SSD tier backed by MoRI's UMBP data plane."""

    medium: ClassVar[Medium] = Medium.STORAGE

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        dram_capacity_bytes: int,
        dram_high_watermark: float | None = None,
        dram_low_watermark: float | None = None,
        dram_use_hugepages: bool | None = None,
        dram_hugepage_size: int | None = None,
        dram_numa_node: int | None = None,
        dram_prefault: bool | None = None,
        ssd_enabled: bool = False,
        ssd_storage_dir: str = "/tmp/vllm_umbp",
        ssd_capacity_bytes: int = 0,
        ssd_high_watermark: float | None = None,
        ssd_low_watermark: float | None = None,
        ssd_backend: str | None = None,
        ssd_segment_size_bytes: int | None = None,
        copy_pipeline_worker_threads: int | None = None,
        copy_pipeline_queue_depth: int | None = None,
        copy_pipeline_batch_max_ops: int | None = None,
        eviction_policy: str = "lru",
        auto_promote_on_read: bool = True,
        master_address: str | None = None,
        node_id: str = "",
        node_address: str = "127.0.0.1",
        io_engine_port: int = 0,
        peer_service_port: int = 0,
        staging_buffer_size: int = 256 * 1024 * 1024,
        ssd_staging_buffer_size: int = 256 * 1024 * 1024,
        ssd_staging_buffer_slots: int = 16,
        cache_remote_fetches: bool | None = None,
        cache_remote_admission: bool | None = None,
        dram_page_size: int | None = None,
        io_threads: int = 4,
        key_prefix: str | None = None,
        enable_kv_events: bool = False,
        locality: str | None = None,
    ) -> None:
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        if dram_capacity_bytes <= 0:
            raise ValueError("dram_capacity_bytes must be greater than zero")
        if ssd_enabled and ssd_capacity_bytes <= 0:
            raise ValueError(
                "ssd_capacity_bytes must be greater than zero when SSD is enabled"
            )
        if io_threads <= 0:
            raise ValueError("io_threads must be greater than zero")

        try:
            from mori.umbp import UMBPClient, UMBPConfig, UMBPDistributedConfig
        except ImportError as exc:
            raise ImportError(
                "The MoRI UMBP tier requires amd-mori built with BUILD_UMBP=ON"
            ) from exc

        config = UMBPConfig.from_environment()
        config.dram.capacity_bytes = dram_capacity_bytes
        self._set_if_not_none(config.dram, "high_watermark", dram_high_watermark)
        self._set_if_not_none(config.dram, "low_watermark", dram_low_watermark)
        self._set_if_not_none(config.dram, "use_hugepages", dram_use_hugepages)
        self._set_if_not_none(config.dram, "hugepage_size", dram_hugepage_size)
        self._set_if_not_none(config.dram, "numa_node", dram_numa_node)
        self._set_if_not_none(config.dram, "prefault", dram_prefault)
        config.ssd.enabled = ssd_enabled
        config.ssd.storage_dir = ssd_storage_dir
        config.ssd.capacity_bytes = ssd_capacity_bytes
        self._set_if_not_none(config.ssd, "high_watermark", ssd_high_watermark)
        self._set_if_not_none(config.ssd, "low_watermark", ssd_low_watermark)
        self._set_if_not_none(config.ssd, "ssd_backend", ssd_backend)
        self._set_if_not_none(config.ssd, "segment_size_bytes", ssd_segment_size_bytes)
        self._set_if_not_none(
            config.copy_pipeline, "worker_threads", copy_pipeline_worker_threads
        )
        self._set_if_not_none(
            config.copy_pipeline, "queue_depth", copy_pipeline_queue_depth
        )
        self._set_if_not_none(
            config.copy_pipeline, "batch_max_ops", copy_pipeline_batch_max_ops
        )
        config.eviction.policy = eviction_policy
        config.eviction.auto_promote_on_read = auto_promote_on_read
        if master_address:
            distributed = UMBPDistributedConfig()
            distributed.master_config.master_address = master_address
            distributed.master_config.node_id = (
                node_id or offloading_spec.config.engine_id
            )
            distributed.master_config.node_address = node_address
            distributed.io_engine.host = node_address
            distributed.io_engine.port = io_engine_port
            distributed.peer_service_port = peer_service_port
            distributed.staging_buffer_size = staging_buffer_size
            distributed.ssd_staging_buffer_size = ssd_staging_buffer_size
            distributed.ssd_staging_buffer_slots = ssd_staging_buffer_slots
            self._set_if_not_none(
                distributed, "cache_remote_fetches", cache_remote_fetches
            )
            self._set_if_not_none(
                distributed, "cache_remote_admission", cache_remote_admission
            )
            self._set_if_not_none(distributed, "dram_page_size", dram_page_size)
            config.distributed = distributed

        self.locality = Locality(locality) if locality is not None else None
        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        self._client = UMBPClient(config)
        self._executor = ThreadPoolExecutor(
            max_workers=io_threads, thread_name_prefix="vllm_kv_mori"
        )
        self._futures: dict[
            Future[list[bool]], tuple[int, bool, float, tuple[OffloadKey, ...]]
        ] = {}
        self._pending_results: list[JobResult] = []
        self._lock = threading.Lock()
        self._lookup_manager = MoriAsyncLookupManager(self)

        assert primary_kv_view.strides is not None
        self._block_size = primary_kv_view.strides[0]
        self._base_addr = ctypes.addressof(ctypes.c_char.from_buffer(primary_kv_view))
        self._registered = self._client.register_memory(
            self._base_addr, primary_kv_view.nbytes
        )
        if not self._registered:
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError("MoRI failed to register the primary KV buffer")
        self._key_prefix = key_prefix or self._default_key_prefix(offloading_spec)

    @staticmethod
    def _set_if_not_none(target: Any, name: str, value: Any) -> None:
        if value is not None:
            setattr(target, name, value)

    @staticmethod
    def _default_key_prefix(offloading_spec: "OffloadingSpec") -> str:
        config = offloading_spec.config
        identity = {
            "model": config.model.name,
            "dtype": config.model.dtype,
            "groups": [
                (group.tokens_per_block, group.layer_names) for group in config.groups
            ],
            "chunk": config.cache.blocks_per_chunk,
            "bytes": cast(Any, offloading_spec).kv_bytes_per_chunk,
            "portable": config.parallel.is_parallelism_agnostic,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"vllm:{digest}:"

    def _encode_keys(self, keys: Iterable[OffloadKey]) -> list[str]:
        return [f"{self._key_prefix}{bytes(key).hex()}" for key in keys]

    def _pointers(self, block_ids: Iterable[int]) -> list[int]:
        return [
            self._base_addr + int(block_id) * self._block_size for block_id in block_ids
        ]

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    def _submit(self, job: TransferJob, is_load: bool) -> None:
        offload_keys = tuple(job.keys)
        keys = self._encode_keys(offload_keys)
        pointers = self._pointers(job.block_ids)
        sizes = [self._block_size] * len(keys)
        operation = (
            self._client.batch_get_into_ptr
            if is_load
            else self._client.batch_put_from_ptr
        )
        started = time.monotonic()
        future = self._executor.submit(operation, keys, pointers, sizes)
        with self._lock:
            self._futures[future] = (
                job.job_id,
                is_load,
                started,
                offload_keys,
            )

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        self._submit(job_metadata, is_load=False)

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        self._submit(job_metadata, is_load=True)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        with self._lock:
            completed = [future for future in self._futures if future.done()]
            metadata = [(future, self._futures.pop(future)) for future in completed]
            results, self._pending_results = self._pending_results, []
        for future, (job_id, is_load, started, keys) in metadata:
            successful_keys = None
            try:
                statuses = future.result()
                success = len(statuses) == len(keys) and all(statuses)
                completed_keys = tuple(
                    key for key, status in zip(keys, statuses) if status
                )
                if is_load and not success:
                    successful_keys = completed_keys
                    failed_keys = [
                        key for key, status in zip(keys, statuses) if not status
                    ]
                    if len(statuses) < len(keys):
                        failed_keys.extend(keys[len(statuses) :])
                    self._lookup_manager.mark_miss(failed_keys)
                    logger.warning("MoRI UMBP load job %d had missing blocks", job_id)
                elif not is_load and self.events is not None:
                    # Prefix reuse stops at the first unavailable block. Do not
                    # advertise successful suffix blocks after a failed store:
                    # consumers cannot resolve their parent chain or use them
                    # for prefix-cache routing.
                    prefix_length = 0
                    for status in statuses[: len(keys)]:
                        if not status:
                            break
                        prefix_length += 1
                    if prefix_length:
                        # Make the completed physical placement visible before
                        # the KV event bridge's bounded lookup window expires.
                        # DistributedClient.flush() wakes MoRI's heartbeat
                        # shipper; standalone clients treat it as a storage
                        # flush.
                        if not self._client.flush():
                            logger.warning(
                                "MoRI UMBP placement flush failed for job %d", job_id
                            )
                        self.events.append(
                            OffloadingEvent(
                                keys=list(keys[:prefix_length]),
                                medium=self.medium,
                                removed=False,
                                locality=self.locality,
                            )
                        )
                if len(statuses) != len(keys):
                    logger.warning(
                        "MoRI UMBP transfer job %d returned %d statuses for %d blocks",
                        job_id,
                        len(statuses),
                        len(keys),
                    )
            except Exception:
                logger.exception("MoRI UMBP transfer job %d failed", job_id)
                success = False
                if is_load:
                    self._lookup_manager.mark_miss(keys)
            results.append(
                JobResult(
                    job_id=job_id,
                    success=success,
                    successful_keys=successful_keys or None,
                    transfer_time=time.monotonic() - started,
                )
            )
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def has_pending_work(self) -> bool:
        with self._lock:
            return bool(self._futures)

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def drain_jobs(self) -> None:
        while self.has_pending_work():
            results = list(self.get_finished_jobs())
            if results:
                with self._lock:
                    self._pending_results.extend(results)

    @override
    def shutdown(self) -> None:
        self._lookup_manager.shutdown()
        self.drain_jobs()
        self._executor.shutdown(wait=True)
        if self._registered:
            self._client.deregister_memory(self._base_addr)
            self._registered = False
