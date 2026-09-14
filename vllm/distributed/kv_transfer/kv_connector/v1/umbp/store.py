# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned, bounded asynchronous access to MoRI 1.2.3.post1 ranged storage.

The connector must fence GPU computation before submitting a store and must
not publish a loaded block before its successful result. A failed read may
have modified its destination. Cancelling a running future cannot cancel DMA.
"""

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from math import isfinite
from threading import Event, Lock
from typing import Any, Literal

from vllm.logger import init_logger

logger = init_logger(__name__)
_MAX_ADDRESS = (1 << 64) - 1


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or not 0 < value <= _MAX_ADDRESS:
        raise ValueError(f"{name} must be a positive 64-bit integer")


def validate_store_retry_timeout(value: float) -> None:
    if type(value) not in (int, float) or not 0 <= value <= 60 or not isfinite(value):
        raise ValueError("store_retry_timeout_s must be finite and in [0, 60]")


@dataclass(frozen=True)
class MemoryRegion:
    """Allocation retained through registration, queued I/O and deregistration.

    Args:
        ptr: Base address of an allocation owned by ``owner``.
        size: Allocation size in bytes.
        owner: Strong reference to the actual allocation, not just its address.
        device: GPU ordinal, or None for CPU memory.
    """

    ptr: int
    size: int
    owner: Any = field(compare=False, repr=False)
    device: int | None = None

    def __post_init__(self) -> None:
        _positive_int(self.ptr, "ptr")
        _positive_int(self.size, "size")
        if self.ptr + self.size > _MAX_ADDRESS or self.owner is None:
            raise ValueError(
                "Memory region must have an owner and a valid address range"
            )
        if self.device is not None and (
            type(self.device) is not int or self.device < 0
        ):
            raise ValueError("device must be a nonnegative GPU ordinal or None")


@dataclass(frozen=True)
class BufferSlice:
    """A slice of registered memory mapped to a byte offset in a stored object."""

    region: MemoryRegion
    offset: int
    size: int
    object_offset: int

    def __post_init__(self) -> None:
        _positive_int(self.size, "size")
        for value in (self.offset, self.object_offset):
            if type(value) is not int or value < 0:
                raise ValueError("Offsets must be nonnegative integers")
        if self.offset + self.size > self.region.size:
            raise ValueError("Buffer slice extends outside its memory region")


@dataclass(frozen=True)
class TransferObject:
    """One independently committed key; all of its ranges share one result."""

    key: str
    size: int
    slices: tuple[BufferSlice, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key or "\0" in self.key:
            raise ValueError("An object needs a nonempty key without NUL")
        _positive_int(self.size, "object size")
        if not isinstance(self.slices, tuple) or not self.slices:
            raise ValueError("Object slices must be a nonempty immutable tuple")
        if any(part.object_offset + part.size > self.size for part in self.slices):
            raise ValueError("A slice extends outside its stored object")


class StoreBusyError(RuntimeError):
    """Admission rejected without starting or queueing any native I/O."""


class UMBPStore:
    """Own the native client, registrations and bounded in-flight operations.

    ``native_client`` is exclusively owned by this store. Worker shutdown must
    call close(), which drains native calls before deregistering allocations.
    No cache clear is issued: the store can contain another engine's KV.
    Data batches are split at object boundaries; the future remains pending
    through every chunk. The limit must fit all participating SSD arenas.
    Failed immutable PUTs may be retried within one operation-wide budget;
    successful objects are never resubmitted. This does not bound a native
    call's duration or turn an unsuccessful write into a published cache entry.
    """

    def __init__(
        self,
        native_client: Any,
        memory_location_type: Any,
        *,
        workers: int = 2,
        max_pending: int = 8,
        max_batch_objects: int = 16,
        store_retry_timeout_s: float = 0,
        cleanup: Callable[[], None] | None = None,
    ) -> None:
        try:
            _positive_int(workers, "workers")
            _positive_int(max_pending, "max_pending")
            _positive_int(max_batch_objects, "max_batch_objects")
            validate_store_retry_timeout(store_retry_timeout_s)
            required = (
                "batch_exists",
                "batch_get_ranges_into_ptr",
                "batch_put_ranges_from_ptr",
                "register_memory",
                "deregister_memory",
                "supports_ranged_io",
            )
            if any(
                not callable(getattr(native_client, name, None)) for name in required
            ):
                raise RuntimeError("UMBP requires the MoRI 1.2.3.post1 ranged I/O API")
            if not native_client.supports_ranged_io():
                raise RuntimeError("This UMBP deployment does not support ranged I/O")
            self._executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="umbp-kv"
            )
        except BaseException:
            # Do not retain the native client in this frame's traceback while
            # a failed factory call removes its private storage directories.
            native_client = None
            raise
        self._client = native_client
        self._cleanup = cleanup
        self._locations = memory_location_type
        self._regions: dict[int, MemoryRegion] = {}
        self._lock = Lock()
        self._close_lock = Lock()
        self._closing = False
        self._pending = 0
        self._max_pending = max_pending
        self._max_batch_objects = max_batch_objects
        self._store_retry_timeout_s = store_retry_timeout_s
        self._stop_retries = Event()

    @classmethod
    def from_native_config(
        cls,
        config: Any,
        *,
        workers: int = 2,
        max_pending: int = 8,
        max_batch_objects: int = 16,
        store_retry_timeout_s: float = 0,
        cleanup: Callable[[], None] | None = None,
    ) -> "UMBPStore":
        _positive_int(workers, "workers")
        _positive_int(max_pending, "max_pending")
        _positive_int(max_batch_objects, "max_batch_objects")
        validate_store_retry_timeout(store_retry_timeout_s)
        from mori.cpp import MemoryLocationType, UMBPClient

        return cls(
            UMBPClient(config),
            MemoryLocationType,
            workers=workers,
            max_pending=max_pending,
            max_batch_objects=max_batch_objects,
            store_retry_timeout_s=store_retry_timeout_s,
            cleanup=cleanup,
        )

    def _check_open(self) -> None:
        if self._closing:
            raise RuntimeError("UMBP store is closing or closed")

    def register_region(self, region: MemoryRegion) -> None:
        with self._lock:
            self._check_open()
            existing = self._regions.get(region.ptr)
            if existing is region:
                return
            if any(
                region.ptr < other.ptr + other.size
                and other.ptr < region.ptr + region.size
                for other in self._regions.values()
            ):
                raise ValueError("Registered memory regions must not overlap")
            location = (
                self._locations.CPU if region.device is None else self._locations.GPU
            )
            device = -1 if region.device is None else region.device
            if (
                self._client.register_memory(region.ptr, region.size, location, device)
                is not True
            ):
                raise RuntimeError("UMBP memory registration failed")
            self._regions[region.ptr] = region

    def lookup(self, keys: tuple[str, ...]) -> Future[tuple[bool, ...]]:
        if not isinstance(keys, tuple) or any(
            not isinstance(key, str) or not key or "\0" in key for key in keys
        ):
            raise ValueError("Lookup keys must be an immutable tuple of valid keys")
        return self._submit("lookup", keys)

    def load(self, objects: tuple[TransferObject, ...]) -> Future[tuple[bool, ...]]:
        return self._submit("load", objects)

    def store(self, objects: tuple[TransferObject, ...]) -> Future[tuple[bool, ...]]:
        return self._submit("store", objects)

    def _validate_objects(
        self, operation: str, objects: tuple[TransferObject, ...]
    ) -> None:
        if not isinstance(objects, tuple):
            raise ValueError("Transfer objects must be an immutable tuple")
        if operation == "store" and len({obj.key for obj in objects}) != len(objects):
            raise ValueError("A store batch must not write the same key twice")
        destinations = []
        for obj in objects:
            parts = sorted(obj.slices, key=lambda part: part.object_offset)
            boundary = 0
            for part in parts:
                if self._regions.get(part.region.ptr) is not part.region:
                    raise ValueError("Transfer uses an unregistered memory region")
                if operation == "store":
                    if part.object_offset != boundary:
                        raise ValueError("Store slices must exactly tile the object")
                    boundary += part.size
                else:
                    ptr = part.region.ptr + part.offset
                    destinations.append((ptr, ptr + part.size))
            if operation == "store" and boundary != obj.size:
                raise ValueError("Store slices must exactly tile the object")
        destinations.sort()
        if any(
            end > start for (_, end), (start, _) in zip(destinations, destinations[1:])
        ):
            raise ValueError("Load destinations must not overlap")

    def _submit(
        self,
        operation: Literal["lookup", "load", "store"],
        items: tuple[Any, ...],
    ) -> Future[tuple[bool, ...]]:
        with self._lock:
            self._check_open()
            if operation != "lookup":
                self._validate_objects(operation, items)
            if self._pending >= self._max_pending:
                raise StoreBusyError("UMBP pending-operation limit reached")
            self._pending += 1
            try:
                future = self._executor.submit(self._execute, operation, items)
            except BaseException:
                self._pending -= 1
                raise
        future.add_done_callback(self._release_slot)
        return future

    def _release_slot(self, future: Future) -> None:
        with self._lock:
            self._pending -= 1

    def _execute(self, operation: str, items: tuple[Any, ...]) -> tuple[bool, ...]:
        if not items:
            return ()
        if operation == "lookup":
            return self._execute_batch(operation, items)
        results: list[bool] = []
        deadline = time.monotonic() + self._store_retry_timeout_s
        for start in range(0, len(items), self._max_batch_objects):
            chunk = items[start : start + self._max_batch_objects]
            try:
                outcome = self._execute_batch(operation, chunk)
                if operation == "store" and not all(outcome):
                    outcome = self._retry_store(chunk, outcome, deadline, start)
            except Exception as error:
                logger.warning(
                    "UMBP %s raised %s in chunk starting at %d (%d objects)",
                    operation,
                    type(error).__name__,
                    start,
                    len(chunk),
                )
                raise
            failed = [start + i for i, success in enumerate(outcome) if not success]
            if failed:
                logger.warning(
                    "UMBP %s failed for %d/%d objects in chunk starting at %d; "
                    "failed indices (first 16): %s",
                    operation,
                    len(failed),
                    len(chunk),
                    start,
                    failed[:16],
                )
            results.extend(outcome)
        return tuple(results)

    def _retry_store(
        self,
        items: tuple[TransferObject, ...],
        outcome: tuple[bool, ...],
        deadline: float,
        start: int,
    ) -> tuple[bool, ...]:
        # SSD read leases can consume the arena even after GET completion.
        # The pinned API reports only booleans, not a retryable error class.
        results = list(outcome)
        attempts = 0
        delay = 0.01
        while not all(results):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self._stop_retries.wait(min(delay, remaining)):
                break
            if time.monotonic() >= deadline:
                break
            failed = [i for i, success in enumerate(results) if not success]
            retried = self._execute_batch("store", tuple(items[i] for i in failed))
            attempts += 1
            for i, success in zip(failed, retried, strict=True):
                results[i] = success
            delay = min(delay * 2, 0.1)
        if attempts:
            logger.info(
                "UMBP store retry: chunk_start=%d, attempts=%d, "
                "initial_failed=%d, remaining_failed=%d",
                start,
                attempts,
                outcome.count(False),
                results.count(False),
            )
        return tuple(results)

    def _execute_batch(
        self, operation: str, items: tuple[Any, ...]
    ) -> tuple[bool, ...]:
        if operation == "lookup":
            result = self._client.batch_exists(list(items))
        else:
            keys = [obj.key for obj in items]
            ptrs = [
                [part.region.ptr + part.offset for part in obj.slices] for obj in items
            ]
            sizes = [[part.size for part in obj.slices] for obj in items]
            offsets = [[part.object_offset for part in obj.slices] for obj in items]
            if operation == "store":
                result = self._client.batch_put_ranges_from_ptr(
                    keys, [obj.size for obj in items], ptrs, sizes, offsets
                )
            else:
                result = self._client.batch_get_ranges_into_ptr(
                    keys, ptrs, sizes, offsets
                )
        if (
            not isinstance(result, list | tuple)
            or len(result) != len(items)
            or any(type(success) is not bool for success in result)
        ):
            raise RuntimeError("UMBP returned malformed per-object results")
        return tuple(result)

    def close(self) -> None:
        with self._close_lock:
            with self._lock:
                self._closing = True
                self._stop_retries.set()
            self._executor.shutdown(wait=True, cancel_futures=False)
            # On deregistration failure retain the native client and allocation
            # owners. A later close can retry; unsafe release is never a fallback.
            for ptr in list(self._regions):
                self._client.deregister_memory(ptr)
                del self._regions[ptr]
            # v1.2.3.post1 has a C++ Close/destructor but no Python close binding.
            # Dropping our exclusive client reference invokes native teardown.
            self._client = None
            if self._cleanup is not None:
                self._cleanup()
                self._cleanup = None
