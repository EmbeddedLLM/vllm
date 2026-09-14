# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in UMBP KVConnector wiring, pending native serving validation.

Use the external module-path mechanism during development. Dense full-attention
and MLA use the offload planner and optional pool-mediated P/D protocol. Aligned
Mamba checkpoints support ordinary offload; hybrid P/D remains a separate gate.
"""

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, replace
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.config import (
    UMBPNodeConfig,
    UMBPStoreConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.key import UMBPKeySpace
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.layout import (
    CacheTopology,
    UMBPLayout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.lifecycle import (
    UMBPWorkerLifecycle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.metadata import (
    UMBPConnectorMetadata,
    UMBPWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.pd import UMBPHandoffPlanner
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.planner import UMBPRequestPlanner
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPTransferScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.store import UMBPStore
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import UMBPTransferWorker
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import ForwardContext
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVQuantMode,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request


@dataclass(frozen=True)
class UMBPHandshake(KVConnectorHandshakeMetadata):
    rank: int
    generation: str
    namespace: str
    max_pending: int


def _generation_epoch(generations: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(generations, separators=(",", ":")).encode()
    ).hexdigest()


class UMBPConnector(KVConnectorBase_V1, SupportsHMA):
    """One scheduler planner and one independently owned store per worker.

    SupportsHMA supplies the all-group finish hook; configuration validation still
    rejects groups whose request-boundary semantics are not implemented here.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config
        allowed = {
            "deployment",
            "model",
            "revision",
            "storage",
            "nodes",
            "lookup_timeout",
            "enable_pd",
            "handoff_timeout",
        }
        if extra.keys() - allowed:
            raise ValueError(f"Unknown UMBP options: {sorted(extra.keys() - allowed)}")
        for key in ("deployment", "model", "revision"):
            if not isinstance(extra.get(key), str) or not extra[key].strip():
                raise ValueError(f"UMBP requires explicit {key} identity")
        # Include the resolved model configuration (RoPE overrides, quantization,
        # dtype, etc.), not just an operator's model label and weight revision.
        self._identity: tuple[str, str, str] = (
            extra["deployment"],
            json.dumps((extra["model"], vllm_config.model_config.compute_hash())),
            extra["revision"],
        )
        self._storage_config = UMBPStoreConfig.from_dict(extra.get("storage", {}))
        timeout = extra.get("lookup_timeout", 5.0)
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("lookup_timeout must be positive and finite")
        self._lookup_timeout = float(timeout)
        self._enable_pd = extra.get(
            "enable_pd", self._kv_transfer_config.kv_role != "kv_both"
        )
        if type(self._enable_pd) is not bool:
            raise ValueError("enable_pd must be a boolean")
        if not self._enable_pd and self._kv_transfer_config.kv_role != "kv_both":
            raise ValueError("P/D roles require enable_pd")
        timeout = extra.get("handoff_timeout", 30.0)
        if type(timeout) not in (int, float) or not 0 < timeout <= 3600:
            raise ValueError("handoff_timeout must be in (0, 3600] seconds")
        self._handoff_timeout = float(timeout)
        if self._enable_pd and not self._storage_config.master_address:
            raise ValueError("P/D handoff requires a shared UMBP master")
        parallel = vllm_config.parallel_config
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError("PP/CP request planning is not implemented for UMBP yet")
        if vllm_config.speculative_config is not None:
            raise ValueError("UMBP speculative boundary save planning is pending")
        if not vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("UMBP offload requires vLLM prefix-cache records")
        if vllm_config.cache_config.cache_dtype not in ("auto", "float16", "bfloat16"):
            raise ValueError("Quantized KV needs a validated runtime-scale identity")
        if not kv_cache_config.kv_cache_groups:
            raise ValueError("UMBP requires transferable KV cache groups")
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            specs = (
                tuple(spec.kv_cache_specs.values())
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else (spec,)
            )
            if (
                not group.enable_kv_transfer
                or group.host_resident
                or group.is_eagle_group
                or any(
                    type(item) not in (FullAttentionSpec, MLAAttentionSpec, MambaSpec)
                    or not item.prefix_cacheable
                    for item in specs
                )
            ):
                raise ValueError("UMBP hybrid/non-prefix boundary planning is pending")
            for item in specs:
                if isinstance(item, MambaSpec):
                    if self._enable_pd:
                        raise ValueError("UMBP hybrid P/D export planning is pending")
                    if (
                        not isinstance(spec, MambaSpec)
                        or item.mamba_cache_mode != "align"
                        or item.num_speculative_blocks
                        or vllm_config.cache_config.mamba_cache_mode != "align"
                    ):
                        raise ValueError("UMBP recurrent offload requires Mamba align")
                    if any(
                        dtype not in (torch.float16, torch.bfloat16, torch.float32)
                        for dtype in item.dtypes
                    ):
                        raise ValueError("UMBP recurrent state requires floating dtype")
                    continue
                assert isinstance(item, FullAttentionSpec)
                if item.non_causal:
                    raise ValueError("Non-causal KV cannot use dense prefix reuse")
                if item.kv_quant_mode != KVQuantMode.NONE or item.dtype not in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                ):
                    raise ValueError(
                        "Quantized KV needs a validated runtime-scale identity"
                    )
        if any(item.host_resident for item in kv_cache_config.kv_cache_tensors):
            raise ValueError("UMBP host KV requires pool-qualified block IDs")
        self._topology = CacheTopology(tp_size=parallel.tensor_parallel_size)
        self._nodes = extra.get("nodes")
        if self._storage_config.master_address and self._nodes is None:
            raise ValueError("A shared UMBP pool requires explicit per-rank nodes")
        if self._nodes is not None:
            if (
                not isinstance(self._nodes, list)
                or len(self._nodes) != self._topology.tp_size
            ):
                raise ValueError("nodes must contain one configuration per TP worker")
            self._nodes = tuple(UMBPNodeConfig(**node) for node in self._nodes)
            if len({node.node_id for node in self._nodes}) != len(self._nodes):
                raise ValueError("Worker node IDs must be unique")
            endpoints = [
                (node.node_address, node.peer_service_port)
                for node in self._nodes
                if node.peer_service_port
            ]
            if len(set(endpoints)) != len(endpoints):
                raise ValueError("Worker peer endpoints must be unique")
        self._handshake: UMBPHandshake | None = None
        self._worker_generations: tuple[str, ...] = ()
        self._planner: UMBPRequestPlanner | None = None
        self._transfers: UMBPTransferScheduler | None = None
        self._store: UMBPStore | None = None
        self._layout: UMBPLayout | None = None
        self._keyspace: UMBPKeySpace | None = None
        self._worker: UMBPWorkerLifecycle | None = None
        self._handoff: UMBPHandoffPlanner | None = None
        self._closed = False

    @property
    def requires_kv_delivery(self) -> bool:
        return self._enable_pd and self._kv_transfer_config.is_kv_producer

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        if self.role != KVConnectorRole.SCHEDULER or self._kv_cache_manager is None:
            raise RuntimeError("Bind the scheduler cache manager before its handshake")
        if self._planner is not None:
            raise RuntimeError(
                "An active UMBP engine cannot replace its worker generation"
            )
        ranks = frozenset(range(self._topology.tp_size))
        if metadata.keys() != ranks or any(
            not isinstance(item, UMBPHandshake)
            or item.rank != rank
            or item.max_pending != self._storage_config.max_pending
            for rank, item in metadata.items()
        ):
            raise ValueError("Incomplete or inconsistent UMBP worker handshake")
        values = []
        for rank in sorted(ranks):
            item = metadata[rank]
            assert isinstance(item, UMBPHandshake)
            values.append(item)
        namespaces = {item.namespace for item in values}
        if len(namespaces) != 1:
            raise ValueError("UMBP worker cache identities disagree")
        if len({item.generation for item in values}) != len(values):
            raise ValueError("UMBP worker generations must be independently generated")
        self._worker_generations = tuple(item.generation for item in values)
        epoch = _generation_epoch(self._worker_generations)
        self._transfers = UMBPTransferScheduler(
            self._kv_cache_manager,
            epoch=epoch,
            ranks=ranks,
            max_pending=self._storage_config.max_pending,
        )
        if self._enable_pd:
            self._handoff = UMBPHandoffPlanner(
                self._kv_cache_manager,
                self._transfers,
                namespace=values[0].namespace,
                producer=self._kv_transfer_config.is_kv_producer,
                consumer=self._kv_transfer_config.is_kv_consumer,
                timeout=self._handoff_timeout,
            )
        self._planner = UMBPRequestPlanner(
            self._kv_cache_manager,
            self._transfers,
            lookup_timeout=self._lookup_timeout,
            handoff=self._handoff,
        )

    def on_new_request(self, request: Request) -> None:
        assert self._planner is not None, "UMBP worker handshake is required"
        self._planner.on_new_request(request)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self._planner is not None
        return self._planner.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        assert self._planner is not None
        self._planner.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> UMBPConnectorMetadata:
        assert self._planner is not None
        return replace(
            self._planner.build_connector_meta(scheduler_output),
            worker_generations=self._worker_generations,
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        assert self._planner is not None
        self._planner.update_connector_output(connector_output)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._planner is not None
        return self._planner.request_finished(request)

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        assert self._planner is not None
        self._planner.register_finished_partial_tail(request, partial_tail_offloads)
        # Admitted stores hold independent block references, not the request.
        return False

    def has_pending_push_work(self) -> bool:
        return (
            self._transfers is not None and self._transfers.has_pending_push_work()
        ) or (self._handoff is not None and self._handoff.has_pending())

    def has_pending_block_frees(self) -> bool:
        return (
            self._transfers is not None and self._transfers.has_pending_block_frees()
        ) or (self._handoff is not None and self._handoff.has_pending())

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if (
            self.role != KVConnectorRole.WORKER
            or self._closed
            or self._store is not None
        ):
            raise RuntimeError("KV cache registration requires a fresh UMBP worker")
        layout = UMBPLayout(self._kv_cache_config, kv_caches, self._topology)
        rank = get_tp_group().rank_in_group
        generation = uuid.uuid4().hex
        node = (
            self._nodes[rank]
            if self._nodes is not None
            else UMBPNodeConfig(node_id=f"vllm-{generation}-{rank}")
        )
        keyspace = UMBPKeySpace(
            *self._identity,
            layout.identity,
            self._vllm_config.cache_config.prefix_caching_hash_algo,
        )
        store = self._storage_config.open_store(
            node, max_object_bytes=layout.max_object_bytes
        )
        try:
            for region in layout.regions:
                store.register_region(region)
        except BaseException:
            store.close()
            raise
        self._store, self._layout, self._keyspace = store, layout, keyspace
        self._handshake = UMBPHandshake(
            rank, generation, keyspace.prefix, self._storage_config.max_pending
        )

    def get_handshake_metadata(self) -> UMBPHandshake | None:
        return self._handshake

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UMBPConnectorMetadata) or self._closed:
            raise ValueError("UMBP requires its own metadata on an open worker")
        assert self._store is not None and self._layout is not None
        assert self._keyspace is not None and self._handshake is not None
        if (
            len(metadata.worker_generations) != self._topology.tp_size
            or metadata.worker_generations[self._handshake.rank]
            != self._handshake.generation
            or _generation_epoch(metadata.worker_generations) != metadata.epoch
        ):
            raise ValueError("Metadata does not include this worker's fresh generation")
        if self._worker is None:
            # The scheduler derives one epoch from every worker's fresh nonce.
            # Registration is already complete before startup handshake.
            self._worker = UMBPWorkerLifecycle(
                UMBPTransferWorker(
                    self._store,
                    self._layout,
                    self._keyspace,
                    epoch=metadata.epoch,
                    rank=self._handshake.rank,
                    max_pending=self._storage_config.max_pending,
                ),
                max_pending=self._storage_config.max_pending,
            )
        event = torch.Event(device=current_platform.device_type)
        event.record()
        self._worker.start_step(metadata, event)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass  # No current-forward or layerwise loads.

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any
    ) -> None:
        pass  # Whole-group jobs start after forward, outside graph capture.

    def wait_for_save(self) -> None:
        pass  # Cache-manager pins protect asynchronous jobs across steps.

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        if self._worker is None:
            return KVConnectorTransferResults()
        return self._worker.get_transfer_results(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self._worker.get_block_ids_with_load_errors() if self._worker else set()

    def build_connector_worker_meta(self) -> UMBPWorkerMetadata | None:
        return self._worker.build_connector_worker_meta() if self._worker else None

    def shutdown(self) -> None:
        self._closed = True
        if self._worker is not None:
            self._worker.close()
        elif self._store is not None:
            self._store.close()
        self._worker = None
        self._store = None
        self._layout = None
        self._keyspace = None
