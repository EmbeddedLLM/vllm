# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm DeepSeek-V4 model state for ATOM-style cache integration.

This module deliberately lives under the ROCm DeepSeek-V4 model package rather
than the common GPU worker.  Model runner v2 lets a model provide its own
``ModelState`` implementation, which is the right place for DSV4 request-lived
state such as SWA rings and compressor state rings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.amd.v4_kernels.compress_plan import (
    CompressPlan,
    make_compress_plans,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata, has_deep_gemm
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class _CpuGpuInt32Buffer:
    """Fixed host/GPU int32 buffer for CUDAGraph-stable ATOM plans."""

    def __init__(
        self,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        self.cpu = torch.empty(
            shape,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.np = self.cpu.numpy()
        self.gpu = torch.empty(shape, dtype=torch.int32, device=device)

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            view_cpu = self.cpu
            view_gpu = self.gpu
        else:
            view_cpu = self.cpu[:n]
            view_gpu = self.gpu[:n]
        if view_gpu.numel() > 0:
            view_gpu.copy_(view_cpu, non_blocking=True)
        return view_gpu


@dataclass(frozen=True)
class DeepseekV4RocmAtomStateBuffers:
    """Persistent ATOM-style request-state buffers.

    The scheduler-owned KV allocation contains the SWA and compressed token
    caches. These buffers only hold compressor state that is not block-managed.
    """

    csa_main_kv_state: torch.Tensor
    csa_main_score_state: torch.Tensor
    csa_idx_kv_state: torch.Tensor
    csa_idx_score_state: torch.Tensor
    hca_main_kv_state: torch.Tensor
    hca_main_score_state: torch.Tensor
    active_layer_ids: tuple[int, ...]
    csa_layer_ids: tuple[int, ...]
    hca_layer_ids: tuple[int, ...]


@dataclass(frozen=True)
class DeepseekV4RocmAtomUnifiedKVBuffers:
    """ATOM-style per-layer unified KV pools.

    Each layer gets one contiguous pool.  The prefix is the persistent SWA ring
    addressed by request state slot, and the tail is the compressed CSA/HCA
    block cache addressed by vLLM physical block IDs.
    """

    unified_kv: tuple[torch.Tensor, ...]
    unified_kv_by_layer: dict[int, torch.Tensor]
    compressed_kv_cache: dict[int, torch.Tensor]
    compressed_kv_scales: dict[int, torch.Tensor | None]
    compressed_kv_layout: dict[int, str]
    active_layer_ids: tuple[int, ...]
    num_blocks: int
    swa_pages: int
    k1_csa: int
    k2_hca: int


@dataclass(frozen=True)
class DeepseekV4RocmAtomDecodeBuffers:
    """Persistent decode index buffers for ATOM unified-KV attention."""

    swa_indptr_cpu_tensor: torch.Tensor
    csa_indptr_cpu_tensor: torch.Tensor
    hca_indptr_cpu_tensor: torch.Tensor
    safe_batch_cpu_tensor: torch.Tensor
    pos_plus_one_cpu_tensor: torch.Tensor
    tmp_lens_cpu_tensor: torch.Tensor
    swa_lens_cpu_tensor: torch.Tensor
    csa_lens_cpu_tensor: torch.Tensor
    hca_lens_cpu_tensor: torch.Tensor
    swa_indptr_cpu: np.ndarray
    csa_indptr_cpu: np.ndarray
    hca_indptr_cpu: np.ndarray
    safe_batch_cpu: np.ndarray
    valid_cpu: np.ndarray
    pos_plus_one_cpu: np.ndarray
    tmp_lens_cpu: np.ndarray
    swa_lens_cpu: np.ndarray
    csa_lens_cpu: np.ndarray
    hca_lens_cpu: np.ndarray
    swa_indptr: torch.Tensor
    csa_indptr: torch.Tensor
    hca_indptr: torch.Tensor
    swa_indices: torch.Tensor
    csa_indices: torch.Tensor
    hca_indices: torch.Tensor
    max_swa_indices: int
    max_csa_indices: int
    max_hca_indices: int


@dataclass(frozen=True)
class DeepseekV4RocmAtomPrefillBuffers:
    """Persistent prefill index buffers for ATOM unified-KV attention."""

    extend_indptr_cpu_tensor: torch.Tensor
    prefix_swa_indptr_cpu_tensor: torch.Tensor
    prefix_csa_indptr_cpu_tensor: torch.Tensor
    prefix_hca_indptr_cpu_tensor: torch.Tensor
    skip_prefix_len_csa_cpu_tensor: torch.Tensor
    cu_q_per_seq_cpu_tensor: torch.Tensor
    extend_indptr_cpu: np.ndarray
    prefix_swa_indptr_cpu: np.ndarray
    prefix_csa_indptr_cpu: np.ndarray
    prefix_hca_indptr_cpu: np.ndarray
    skip_prefix_len_csa_cpu: np.ndarray
    cu_q_per_seq_cpu: np.ndarray
    extend_indptr: torch.Tensor
    prefix_swa_indptr: torch.Tensor
    prefix_csa_indptr: torch.Tensor
    prefix_hca_indptr: torch.Tensor
    skip_prefix_len_csa: torch.Tensor
    cu_q_per_seq: torch.Tensor
    extend_indices: torch.Tensor
    prefix_swa_indices: torch.Tensor
    prefix_csa_indices: torch.Tensor
    prefix_hca_indices: torch.Tensor
    max_extend_indices: int
    max_prefix_swa_indices: int
    max_prefix_csa_indices: int
    max_prefix_hca_indices: int


class DeepseekV4RocmAtomPrefillCache:
    """Mutable per-forward cache for ATOM paged-prefill metadata.

    The metadata object is recreated for each scheduler step, while all layer
    modules see the same instance during that forward.  This lets ROCm DSV4
    attention reuse common prefill indptr/index work across layers without
    introducing model-runner or worker state.
    """

    def __init__(self, index_topk: int) -> None:
        self.index_topk = int(index_topk)
        self.indptr_key: tuple[int, int, bool] | None = None
        self.totals: tuple[int, int, int, int] = (0, 0, 0, 0)
        self.common_indices_key: tuple[int, int, int, int, int, int] | None = None
        self.hca_indices_key: (
            tuple[
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                tuple[int, ...],
                tuple[int, int, int, int, int, int],
            ]
            | None
        ) = None
        self.csa_translate_key: tuple[object, ...] | None = None
        self.indptr_hits = 0
        self.indptr_writes = 0
        self.common_indices_hits = 0
        self.common_indices_writes = 0
        self.hca_indices_hits = 0
        self.hca_indices_writes = 0
        self.csa_translate_hits = 0
        self.csa_translate_writes = 0


class DeepseekV4RocmAtomDecodeCache:
    """Mutable per-forward cache for ATOM paged-decode index buffers."""

    def __init__(self) -> None:
        self.common_indices_key: tuple[int, int, int, int] | None = None
        self.hca_indices_key: tuple[Any, ...] | None = None
        self.csa_translate_key: tuple[Any, ...] | None = None
        self.common_indices_hits = 0
        self.common_indices_writes = 0
        self.hca_indices_hits = 0
        self.hca_indices_writes = 0
        self.csa_translate_hits = 0
        self.csa_translate_writes = 0


@dataclass(frozen=True)
class DeepseekV4RocmAtomSWADecodeMetadata:
    """Minimal SWA metadata for ROCm DSV4 pure ATOM decode."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    query_start_loc_cpu: np.ndarray | None = None
    is_valid_token: None = None
    token_to_req_indices: None = None
    decode_swa_indices: None = None
    decode_swa_lens: None = None
    decode_swa_ragged_indices: None = None
    decode_swa_ragged_indptr: None = None
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0
    max_query_len: int = 1
    prefill_seq_lens: torch.Tensor | None = None
    prefill_gather_lens: torch.Tensor | None = None
    tile_sched_swaonly: None = None
    tile_sched_c4a: None = None
    tile_sched_c128a: None = None


@dataclass(frozen=True)
class DeepseekV4RocmAtomMLADecodeMetadata:
    """Minimal compressed-MLA metadata for ROCm DSV4 pure ATOM decode."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    req_id_per_token: torch.Tensor
    topk_tokens: int
    c128a_global_decode_topk_indices: None = None
    c128a_decode_topk_lens: None = None
    c128a_prefill_topk_indices: None = None
    c128a_decode_topk_ragged_indices: None = None
    c128a_decode_topk_ragged_indptr: None = None
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_prefill_tokens: int = 0


@dataclass(frozen=True)
class DeepseekV4RocmAtomCompressorDecodeMetadata:
    """Minimal metadata carrier for ATOM main compressor pure decode."""


@dataclass(frozen=True)
class DeepseekV4RocmAtomStateMetadata:
    """Per-forward ATOM-style request-state metadata.

    ``state_slot_mapping`` is the important bridge: vLLM model runner v2 keeps a
    stable request-state slot for each live request, and ATOM kernels use that
    same concept to address SWA and compressor rings.
    """

    state_slot_mapping: torch.Tensor
    state_slot_mapping_cpu: np.ndarray
    num_actual_reqs: int
    num_reqs: int
    num_actual_tokens: int
    num_tokens: int
    win_with_spec: int
    swa_pages: int
    chunk_start_per_seq: torch.Tensor
    chunk_start_per_seq_cpu: np.ndarray
    positions: torch.Tensor
    positions_cpu: np.ndarray
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    batch_id_per_token: torch.Tensor
    batch_id_per_token_cpu: np.ndarray
    n_committed_csa_per_seq: torch.Tensor
    n_committed_csa_per_seq_cpu: np.ndarray
    n_committed_hca_per_seq: torch.Tensor
    n_committed_hca_per_seq_cpu: np.ndarray
    decode_swa_total: int = 0
    decode_csa_total: int = 0
    decode_hca_total: int = 0
    decode_max_hca_len: int = 0
    indexer_decode_block_table: torch.Tensor | None = None
    indexer_decode_schedule_metadata: torch.Tensor | None = None
    indexer_decode_requires_padding: bool = False
    indexer_decode_num_tokens: int = 0
    buffers: DeepseekV4RocmAtomStateBuffers | None = None
    unified_kv_buffers: DeepseekV4RocmAtomUnifiedKVBuffers | None = None
    decode_buffers: DeepseekV4RocmAtomDecodeBuffers | None = None
    decode_cache: DeepseekV4RocmAtomDecodeCache | None = None
    prefill_buffers: DeepseekV4RocmAtomPrefillBuffers | None = None
    prefill_cache: DeepseekV4RocmAtomPrefillCache | None = None
    compress_plans: dict[int, CompressPlan] | None = None


def get_deepseek_v4_rocm_atom_state(
    metadata: Any,
) -> DeepseekV4RocmAtomStateMetadata | None:
    return getattr(metadata, "deepseek_v4_rocm_atom_state", None)


class DeepseekV4RocmAtomModelState(DefaultModelState):
    """ModelState carrying ATOM request-slot metadata for DSV4 on ROCm.

    This first integration slice does not replace vLLM's physical KV cache
    layout.  It establishes the request-lifetime state contract needed by the
    ATOM SWA, compressor, and unified-KV kernels without changing GPU workers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        hf_config = vllm_config.model_config.hf_config
        self.window_size = int(getattr(hf_config, "sliding_window", 0) or 0)
        self.num_spec_tokens = int(vllm_config.num_speculative_tokens or 0)
        self.win_with_spec = self.window_size + self.num_spec_tokens
        self.swa_pages = self.max_num_reqs * self.win_with_spec

        self.head_dim = int(getattr(hf_config, "head_dim", 0) or 0)
        self.index_head_dim = int(getattr(hf_config, "index_head_dim", 0) or 0)
        self.compress_ratios = tuple(int(r) for r in hf_config.compress_ratios)
        self.index_topk = int(getattr(hf_config, "index_topk", 0) or 0)
        self.max_model_len = int(vllm_config.model_config.max_model_len)
        atom_block_size = int(vllm_config.cache_config.block_size)
        if atom_block_size % 128 != 0:
            raise ValueError(
                "ROCm DeepSeek-V4 ATOM model state requires a KV block size "
                f"that is a multiple of 128, got {atom_block_size}."
            )
        self.k1_csa = atom_block_size // 4
        self.k2_hca = atom_block_size // 128
        self._additional_atom_models: list[nn.Module] = []
        self._atom_state_buffers: DeepseekV4RocmAtomStateBuffers | None = None
        self._atom_unified_kv_buffers: DeepseekV4RocmAtomUnifiedKVBuffers | None = None
        self._atom_swa_only_kv_by_layer: dict[int, torch.Tensor] = {}
        self._compress_plan_buffers = self._allocate_compress_plan_buffers()
        self._req_id_to_atom_slot: dict[str, int] = {}

        self._state_slot_mapping = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self._state_slot_mapping_cpu_tensor = self._new_pinned_int32(
            (self.max_num_reqs,),
            fill=0,
        )
        self._state_slot_mapping_cpu = self._state_slot_mapping_cpu_tensor.numpy()
        self._batch_id_per_token = torch.full(
            (self.max_num_tokens,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._batch_id_per_token_cpu_tensor = self._new_pinned_int32(
            (self.max_num_tokens,),
            fill=-1,
        )
        self._batch_id_per_token_cpu = self._batch_id_per_token_cpu_tensor.numpy()
        self._chunk_start_per_seq = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self._chunk_start_per_seq_cpu_tensor = self._new_pinned_int32(
            (self.max_num_reqs,),
            fill=0,
        )
        self._chunk_start_per_seq_cpu = self._chunk_start_per_seq_cpu_tensor.numpy()
        self._n_committed_csa_per_seq = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self._n_committed_hca_per_seq = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self._n_committed_csa_per_seq_cpu_tensor = self._new_pinned_int32(
            (self.max_num_reqs,),
            fill=0,
        )
        self._n_committed_csa_per_seq_cpu = (
            self._n_committed_csa_per_seq_cpu_tensor.numpy()
        )
        self._n_committed_hca_per_seq_cpu_tensor = self._new_pinned_int32(
            (self.max_num_reqs,),
            fill=0,
        )
        self._n_committed_hca_per_seq_cpu = (
            self._n_committed_hca_per_seq_cpu_tensor.numpy()
        )
        self._positions_cpu_tensor = self._new_pinned_int32(
            (self.max_num_tokens,),
            fill=0,
        )
        self._positions_cpu = self._positions_cpu_tensor.numpy()
        self._draft_positions_cpu_tensor = self._new_pinned_int32(
            (self.max_num_reqs,),
            fill=0,
        )
        self._draft_positions_cpu = self._draft_positions_cpu_tensor.numpy()
        self._scheduled_tokens_cpu = np.empty(self.max_num_reqs, dtype=np.int32)
        self._computed_tokens_cpu = np.empty(self.max_num_reqs, dtype=np.int32)
        self._context_lens_cpu = np.empty(self.max_num_reqs, dtype=np.int32)
        self._req_arange_cpu = np.arange(self.max_num_reqs, dtype=np.int32)
        self._atom_decode_buffers = self._allocate_atom_decode_buffers()
        self._atom_prefill_buffers = self._allocate_atom_prefill_buffers()
        self._atom_unified_kv_from_vllm_bound = False
        self._indexer_kv_cache_group_idx: int | None = None
        self._indexer_storage_block_size = self.k1_csa
        self._indexer_decode_schedule_metadata = torch.zeros(
            (num_compute_units(self.device.index or 0) + 1, 2),
            dtype=torch.int32,
            device=self.device,
        )
        self._indexer_compressed_slot_mapping = torch.full(
            (self.max_num_tokens,),
            -1,
            dtype=torch.int64,
            device=self.device,
        )

        self._atom_state_buffers = self._allocate_atom_state_buffers()
        self._bind_atom_state_buffers(self._atom_state_buffers)

    @staticmethod
    def _new_pinned_int32(
        shape: tuple[int, ...],
        *,
        fill: int | None = None,
    ) -> torch.Tensor:
        tensor = torch.empty(
            shape,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        if fill is not None:
            tensor.fill_(fill)
        return tensor

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self._req_id_to_atom_slot[new_req_data.req_id] = req_index
        self._reset_atom_request_slot(req_index)

    def remove_request(self, req_id: str) -> None:
        self._req_id_to_atom_slot.pop(req_id, None)
        super().remove_request(req_id)

    def _reset_atom_request_slot(self, req_index: int) -> None:
        if req_index < 0 or req_index >= self.max_num_reqs:
            return

        buffers = self._atom_state_buffers
        if buffers is not None:
            for kv_state in (
                buffers.csa_main_kv_state,
                buffers.csa_idx_kv_state,
                buffers.hca_main_kv_state,
            ):
                if kv_state.numel():
                    kv_state[:, req_index].zero_()
            for score_state in (
                buffers.csa_main_score_state,
                buffers.csa_idx_score_state,
                buffers.hca_main_score_state,
            ):
                if score_state.numel():
                    score_state[:, req_index].fill_(-float("inf"))

        unified = self._atom_unified_kv_buffers
        zero_split_swa = self._atom_unified_kv_from_vllm_bound
        if unified is not None:
            start = req_index * self.win_with_spec
            end = start + self.win_with_spec
            for layer_kv in unified.unified_kv:
                layer_kv[start:end].zero_()
            zero_split_swa = zero_split_swa or not unified.unified_kv
        if zero_split_swa:
            for _, attn in self._iter_active_attn_modules():
                atom_swa_kv = getattr(attn, "atom_swa_kv", None)
                if atom_swa_kv is not None and atom_swa_kv.numel():
                    atom_swa_kv[req_index].zero_()

    def _allocate_compress_plan_buffers(
        self,
    ) -> dict[int, dict[str, _CpuGpuInt32Buffer]]:
        capacities: dict[int, dict[str, _CpuGpuInt32Buffer]] = {}
        for ratio in sorted({4, 128}.intersection(self.compress_ratios)):
            capacities[ratio] = {
                "compress": _CpuGpuInt32Buffer(
                    (max(1, self.max_num_tokens), 4),
                    self.device,
                ),
                "write": _CpuGpuInt32Buffer(
                    (max(1, self.max_num_tokens), 4),
                    self.device,
                ),
            }
        return capacities

    def _allocate_atom_decode_buffers(self) -> DeepseekV4RocmAtomDecodeBuffers:
        max_tokens = max(1, self.max_num_tokens)
        max_hca_blocks = (self.max_model_len + 127) // 128
        max_swa_indices = max_tokens * max(1, self.window_size)
        max_csa_indices = max_tokens * max(1, self.window_size + self.index_topk)
        max_hca_indices = max_tokens * max(1, self.window_size + max_hca_blocks)
        swa_indptr_cpu_tensor = torch.empty(
            max_tokens + 1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        csa_indptr_cpu_tensor = torch.empty(
            max_tokens + 1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        hca_indptr_cpu_tensor = torch.empty(
            max_tokens + 1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        safe_batch_cpu_tensor = torch.empty(
            max_tokens,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        pos_plus_one_cpu_tensor = torch.empty(
            max_tokens,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        tmp_lens_cpu_tensor = torch.empty(
            max_tokens,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        swa_lens_cpu_tensor = torch.empty(
            max_tokens,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        csa_lens_cpu_tensor = torch.empty(
            max_tokens,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        hca_lens_cpu_tensor = torch.empty(
            max_tokens,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        return DeepseekV4RocmAtomDecodeBuffers(
            swa_indptr_cpu_tensor=swa_indptr_cpu_tensor,
            csa_indptr_cpu_tensor=csa_indptr_cpu_tensor,
            hca_indptr_cpu_tensor=hca_indptr_cpu_tensor,
            safe_batch_cpu_tensor=safe_batch_cpu_tensor,
            pos_plus_one_cpu_tensor=pos_plus_one_cpu_tensor,
            tmp_lens_cpu_tensor=tmp_lens_cpu_tensor,
            swa_lens_cpu_tensor=swa_lens_cpu_tensor,
            csa_lens_cpu_tensor=csa_lens_cpu_tensor,
            hca_lens_cpu_tensor=hca_lens_cpu_tensor,
            swa_indptr_cpu=swa_indptr_cpu_tensor.numpy(),
            csa_indptr_cpu=csa_indptr_cpu_tensor.numpy(),
            hca_indptr_cpu=hca_indptr_cpu_tensor.numpy(),
            safe_batch_cpu=safe_batch_cpu_tensor.numpy(),
            valid_cpu=np.empty(max_tokens, dtype=np.bool_),
            pos_plus_one_cpu=pos_plus_one_cpu_tensor.numpy(),
            tmp_lens_cpu=tmp_lens_cpu_tensor.numpy(),
            swa_lens_cpu=swa_lens_cpu_tensor.numpy(),
            csa_lens_cpu=csa_lens_cpu_tensor.numpy(),
            hca_lens_cpu=hca_lens_cpu_tensor.numpy(),
            swa_indptr=torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            ),
            csa_indptr=torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            ),
            hca_indptr=torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            ),
            swa_indices=torch.empty(
                max_swa_indices,
                dtype=torch.int32,
                device=self.device,
            ),
            csa_indices=torch.empty(
                max_csa_indices,
                dtype=torch.int32,
                device=self.device,
            ),
            hca_indices=torch.empty(
                max_hca_indices,
                dtype=torch.int32,
                device=self.device,
            ),
            max_swa_indices=max_swa_indices,
            max_csa_indices=max_csa_indices,
            max_hca_indices=max_hca_indices,
        )

    def _allocate_atom_prefill_buffers(self) -> DeepseekV4RocmAtomPrefillBuffers:
        max_tokens = max(1, self.max_num_tokens)
        max_hca_blocks = (self.max_model_len + 127) // 128
        max_extend_indices = max_tokens * max(1, self.window_size)
        max_prefix_swa_indices = max_tokens * max(1, self.window_size)
        max_prefix_csa_indices = max_tokens * max(1, self.window_size + self.index_topk)
        max_prefix_hca_indices = max_tokens * max(1, self.window_size + max_hca_blocks)

        def pinned(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.empty(
                shape,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )

        extend_indptr_cpu_tensor = pinned((max_tokens + 1,))
        prefix_swa_indptr_cpu_tensor = pinned((max_tokens + 1,))
        prefix_csa_indptr_cpu_tensor = pinned((max_tokens + 1,))
        prefix_hca_indptr_cpu_tensor = pinned((max_tokens + 1,))
        skip_prefix_len_csa_cpu_tensor = pinned((max_tokens,))
        cu_q_per_seq_cpu_tensor = pinned((self.max_num_reqs,))

        return DeepseekV4RocmAtomPrefillBuffers(
            extend_indptr_cpu_tensor=extend_indptr_cpu_tensor,
            prefix_swa_indptr_cpu_tensor=prefix_swa_indptr_cpu_tensor,
            prefix_csa_indptr_cpu_tensor=prefix_csa_indptr_cpu_tensor,
            prefix_hca_indptr_cpu_tensor=prefix_hca_indptr_cpu_tensor,
            skip_prefix_len_csa_cpu_tensor=skip_prefix_len_csa_cpu_tensor,
            cu_q_per_seq_cpu_tensor=cu_q_per_seq_cpu_tensor,
            extend_indptr_cpu=extend_indptr_cpu_tensor.numpy(),
            prefix_swa_indptr_cpu=prefix_swa_indptr_cpu_tensor.numpy(),
            prefix_csa_indptr_cpu=prefix_csa_indptr_cpu_tensor.numpy(),
            prefix_hca_indptr_cpu=prefix_hca_indptr_cpu_tensor.numpy(),
            skip_prefix_len_csa_cpu=skip_prefix_len_csa_cpu_tensor.numpy(),
            cu_q_per_seq_cpu=cu_q_per_seq_cpu_tensor.numpy(),
            extend_indptr=torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            ),
            prefix_swa_indptr=torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            ),
            prefix_csa_indptr=torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            ),
            prefix_hca_indptr=torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            ),
            skip_prefix_len_csa=torch.empty(
                max_tokens,
                dtype=torch.int32,
                device=self.device,
            ),
            cu_q_per_seq=torch.empty(
                self.max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            ),
            extend_indices=torch.empty(
                max_extend_indices,
                dtype=torch.int32,
                device=self.device,
            ),
            prefix_swa_indices=torch.empty(
                max_prefix_swa_indices,
                dtype=torch.int32,
                device=self.device,
            ),
            prefix_csa_indices=torch.empty(
                max_prefix_csa_indices,
                dtype=torch.int32,
                device=self.device,
            ),
            prefix_hca_indices=torch.empty(
                max_prefix_hca_indices,
                dtype=torch.int32,
                device=self.device,
            ),
            max_extend_indices=max_extend_indices,
            max_prefix_swa_indices=max_prefix_swa_indices,
            max_prefix_csa_indices=max_prefix_csa_indices,
            max_prefix_hca_indices=max_prefix_hca_indices,
        )

    def _iter_active_attn_modules(self) -> list[tuple[int, nn.Module]]:
        modules: list[tuple[int, nn.Module]] = []
        for model in (self.model, *self._additional_atom_models):
            for module in model.modules():
                if not (
                    hasattr(module, "swa_cache_layer")
                    and hasattr(module, "compress_ratio")
                    and hasattr(module, "head_dim")
                    and hasattr(module, "prefix")
                ):
                    continue
                try:
                    layer_id = extract_layer_index(module.prefix)
                except Exception:
                    continue
                modules.append((layer_id, module))
        return sorted(modules, key=lambda item: item[0])

    def register_atom_model(self, model: nn.Module) -> None:
        """Include an auxiliary model (for example MTP) in ATOM state.

        Registration happens after target and draft weights are loaded but
        before KV-cache profiling/initialization, so all draft allocations are
        visible to the normal vLLM memory profiler and are bound from the same
        KVCacheConfig as target layers.
        """
        if model is self.model or any(
            model is item for item in self._additional_atom_models
        ):
            return
        if self._atom_unified_kv_buffers is not None:
            raise RuntimeError(
                "ATOM auxiliary models must be registered before KV binding."
            )
        self._additional_atom_models.append(model)
        self._atom_state_buffers = self._allocate_atom_state_buffers()
        self._bind_atom_state_buffers(self._atom_state_buffers)
        for layer_id, attn in self._iter_active_attn_modules():
            if int(getattr(attn, "compress_ratio", 0)) > 1:
                continue
            unified = self._atom_swa_only_kv_by_layer.get(layer_id)
            if unified is None:
                unified = torch.zeros(
                    (self.swa_pages, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                self._atom_swa_only_kv_by_layer[layer_id] = unified
            attn.atom_unified_kv = unified
            attn.atom_swa_kv = unified.view(
                self.max_num_reqs, self.win_with_spec, self.head_dim
            )
            attn.atom_swa_pages = self.swa_pages

    def _allocate_atom_state_buffers(self) -> DeepseekV4RocmAtomStateBuffers:
        active_attn = self._iter_active_attn_modules()
        active_layer_ids = tuple(layer_id for layer_id, _ in active_attn)
        csa_layer_ids = tuple(
            layer_id
            for layer_id, module in active_attn
            if int(module.compress_ratio) == 4
        )
        hca_layer_ids = tuple(
            layer_id
            for layer_id, module in active_attn
            if int(module.compress_ratio) == 128
        )

        state_dtype = torch.float32
        ring_extra = self.num_spec_tokens
        csa_state_len = 2 * 4 + ring_extra
        hca_state_len = 128 + ring_extra

        csa_main_shape = (
            len(csa_layer_ids),
            self.max_num_reqs,
            csa_state_len,
            2 * self.head_dim,
        )
        csa_idx_shape = (
            len(csa_layer_ids),
            self.max_num_reqs,
            csa_state_len,
            2 * self.index_head_dim,
        )
        hca_shape = (
            len(hca_layer_ids),
            self.max_num_reqs,
            hca_state_len,
            self.head_dim,
        )

        def zeros(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.zeros(shape, dtype=state_dtype, device=self.device)

        def neg_inf(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.full(
                shape,
                -float("inf"),
                dtype=state_dtype,
                device=self.device,
            )

        buffers = DeepseekV4RocmAtomStateBuffers(
            csa_main_kv_state=zeros(csa_main_shape),
            csa_main_score_state=neg_inf(csa_main_shape),
            csa_idx_kv_state=zeros(csa_idx_shape),
            csa_idx_score_state=neg_inf(csa_idx_shape),
            hca_main_kv_state=zeros(hca_shape),
            hca_main_score_state=neg_inf(hca_shape),
            active_layer_ids=active_layer_ids,
            csa_layer_ids=csa_layer_ids,
            hca_layer_ids=hca_layer_ids,
        )
        logger.info(
            "Allocated ROCm DSV4 ATOM state buffers: active_layers=%d, "
            "csa_layers=%d, hca_layers=%d, win_with_spec=%d",
            len(active_layer_ids),
            len(csa_layer_ids),
            len(hca_layer_ids),
            self.win_with_spec,
        )
        return buffers

    def _bind_atom_state_buffers(self, buffers: DeepseekV4RocmAtomStateBuffers) -> None:
        csa_pos = {layer_id: pos for pos, layer_id in enumerate(buffers.csa_layer_ids)}
        hca_pos = {layer_id: pos for pos, layer_id in enumerate(buffers.hca_layer_ids)}

        for layer_id, attn in self._iter_active_attn_modules():
            attn.atom_win_with_spec = self.win_with_spec
            attn.atom_swa_pages = self.swa_pages

            compressor = getattr(attn, "compressor", None)
            if compressor is not None:
                ratio = int(getattr(compressor, "compress_ratio", 0))
                if ratio == 4:
                    pos = csa_pos[layer_id]
                    compressor.atom_kv_state = buffers.csa_main_kv_state[pos]
                    compressor.atom_score_state = buffers.csa_main_score_state[pos]
                elif ratio == 128:
                    pos = hca_pos[layer_id]
                    compressor.atom_kv_state = buffers.hca_main_kv_state[pos]
                    compressor.atom_score_state = buffers.hca_main_score_state[pos]

            indexer = getattr(attn, "indexer", None)
            if indexer is not None:
                pos = csa_pos[layer_id]
                inner = getattr(indexer, "compressor", None)
                if inner is not None:
                    inner.atom_kv_state = buffers.csa_idx_kv_state[pos]
                    inner.atom_score_state = buffers.csa_idx_score_state[pos]

    def _make_storage_view(
        self,
        source: torch.Tensor,
        *,
        dtype: torch.dtype,
        offset_bytes: int,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        dtype_size = get_dtype_size(dtype)
        if offset_bytes % dtype_size != 0:
            raise ValueError(
                f"Cannot view storage at byte offset {offset_bytes} as {dtype}."
            )
        stride = torch.empty(shape, dtype=dtype, device=source.device).stride()
        view = torch.empty((), dtype=dtype, device=source.device)
        view.set_(
            source.untyped_storage(),
            offset_bytes // dtype_size,
            shape,
            stride,
        )
        return view

    def _try_bind_atom_unified_kv_from_vllm(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> bool:
        if self._atom_unified_kv_from_vllm_bound:
            return True

        num_blocks = int(getattr(kv_cache_config, "num_blocks", 0) or 0)
        if num_blocks <= 0:
            return False

        active_attn = self._iter_active_attn_modules()
        atom_attn: list[tuple[int, nn.Module]] = []
        for layer_id, attn in active_attn:
            if int(getattr(attn, "compress_ratio", 0)) <= 1:
                continue
            if not hasattr(attn, "atom_vllm_unified_kv_prefix_bytes"):
                # Ratio/layer-scoped ATOM attention keeps the native vLLM KV
                # layout for fallback sparse-MLA layers. Only layers that
                # emitted the ATOM spec participate in this binding path.
                continue
            atom_attn.append((layer_id, attn))
            kv_cache = getattr(attn, "kv_cache", None)
            if kv_cache is None or not isinstance(kv_cache, torch.Tensor):
                return False
            if kv_cache.numel() == 0:
                return False
            compressed_layout = getattr(attn, "atom_vllm_compressed_layout", "dense")
            if compressed_layout == "fp8_ds_mla":
                if (
                    kv_cache.dtype != torch.uint8
                    or kv_cache.dim() != 3
                    or int(kv_cache.shape[-1]) != 584
                ):
                    raise RuntimeError(
                        "ROCm DSV4 ATOM packed fp8_ds_mla KV was requested, "
                        "but vLLM allocated an incompatible compressed tail "
                        f"for layer {getattr(attn, 'prefix', '<unknown>')}: "
                        f"dtype={kv_cache.dtype}, shape={tuple(kv_cache.shape)}."
                    )
                continue
            if kv_cache.dtype != self.dtype and not (
                hasattr(attn, "atom_split_kv_swa")
                and hasattr(attn, "atom_split_kv_compressed")
                and hasattr(attn, "atom_split_kv_scales")
            ):
                logger.warning_once(
                    "ROCm DSV4 ATOM vLLM-owned mixed KV cannot bind yet: "
                    "layer %s compressed tail dtype is %s and split KV "
                    "views are not available. Falling back to the "
                    "model-state side allocation.",
                    getattr(attn, "prefix", "<unknown>"),
                    kv_cache.dtype,
                )
                return False

        if not atom_attn:
            logger.warning_once(
                "ROCm DSV4 ATOM vLLM-owned unified KV was requested, but no "
                "compressed attention layer emitted the ATOM KV cache spec."
            )
            return False

        unified_kv: list[torch.Tensor] = []
        unified_kv_by_layer: dict[int, torch.Tensor] = {}
        compressed_kv_cache: dict[int, torch.Tensor] = {}
        compressed_kv_scales: dict[int, torch.Tensor | None] = {}
        compressed_kv_layout: dict[int, str] = {}
        unified_layer_ids: list[int] = []

        for layer_id, unified in self._atom_swa_only_kv_by_layer.items():
            unified_kv.append(unified)
            unified_kv_by_layer[layer_id] = unified
            unified_layer_ids.append(layer_id)

        for layer_id, attn in atom_attn:
            ratio = int(getattr(attn, "compress_ratio", 0))

            prefix_bytes = int(getattr(attn, "atom_vllm_unified_kv_prefix_bytes", 0))
            swa_pages = int(getattr(attn, "atom_vllm_unified_kv_swa_pages", 0))
            if prefix_bytes <= 0 or swa_pages <= 0:
                return False
            k_per_block = self.k1_csa if ratio == 4 else self.k2_hca
            kv_cache = attn.kv_cache
            compressed_layout = getattr(attn, "atom_vllm_compressed_layout", "dense")
            if compressed_layout == "fp8_ds_mla":
                if (
                    kv_cache.dtype != torch.uint8
                    or kv_cache.dim() != 3
                    or int(kv_cache.shape[0]) != num_blocks
                    or int(kv_cache.shape[1]) != k_per_block
                    or int(kv_cache.shape[2]) != 584
                ):
                    raise RuntimeError(
                        "ROCm DSV4 ATOM packed fp8_ds_mla KV shape mismatch "
                        f"for layer {getattr(attn, 'prefix', '<unknown>')}: "
                        f"expected=({num_blocks}, {k_per_block}, 584), "
                        f"got dtype={kv_cache.dtype}, shape={tuple(kv_cache.shape)}."
                    )
                storage_bytes = kv_cache.untyped_storage().nbytes()
                expected_bytes = prefix_bytes + num_blocks * k_per_block * 584
                if storage_bytes < expected_bytes:
                    raise RuntimeError(
                        "ROCm DSV4 ATOM packed fp8_ds_mla KV storage is too "
                        f"small for layer {getattr(attn, 'prefix', '<unknown>')}: "
                        f"storage_bytes={storage_bytes}, "
                        f"expected_bytes={expected_bytes}, "
                        f"swa_pages={swa_pages}, num_blocks={num_blocks}, "
                        f"k_per_block={k_per_block}."
                    )
                atom_swa_dtype = getattr(
                    attn, "atom_vllm_unified_kv_swa_dtype", self.dtype
                )
                swa = self._make_storage_view(
                    kv_cache,
                    dtype=atom_swa_dtype,
                    offset_bytes=0,
                    shape=(swa_pages, self.head_dim),
                )
                compressed = kv_cache
                unified_layer_ids.append(layer_id)
                compressed_kv_cache[layer_id] = compressed
                compressed_kv_scales[layer_id] = None
                compressed_kv_layout[layer_id] = "fp8_ds_mla"

                attn.atom_swa_kv = swa.view(
                    self.max_num_reqs, self.win_with_spec, self.head_dim
                )
                attn.atom_direct_swa_bound = True
                attn.atom_swa_pages = swa_pages
                attn.atom_compressed_kv_cache = compressed
                attn.atom_split_kv_swa = attn.atom_swa_kv
                attn.atom_split_kv_compressed = compressed
                attn.atom_split_kv_scales = None
                attn.atom_split_kv_layout = "fp8_ds_mla"
                if hasattr(attn, "atom_unified_kv"):
                    delattr(attn, "atom_unified_kv")
                compressor = getattr(attn, "compressor", None)
                if compressor is not None:
                    compressor.atom_kv_cache = compressed
                    compressor.atom_kv_scales = None
                    compressor.atom_kv_layout = "fp8_ds_mla"
                continue

            if kv_cache.dtype != self.dtype:
                compressed = getattr(attn, "atom_split_kv_compressed", None)
                if compressed is None:
                    return False
                compressed_kv_cache[layer_id] = compressed
                compressed_kv_scales[layer_id] = getattr(
                    attn, "atom_split_kv_scales", None
                )
                compressed_kv_layout[layer_id] = getattr(
                    attn, "atom_split_kv_layout", "dense"
                )
                unified_layer_ids.append(layer_id)
                attn.atom_swa_pages = swa_pages
                attn.atom_compressed_kv_cache = compressed
                compressor = getattr(attn, "compressor", None)
                if compressor is not None:
                    compressor.atom_kv_cache = compressed
                    compressor.atom_kv_scales = getattr(
                        attn, "atom_split_kv_scales", None
                    )
                    compressor.atom_kv_layout = getattr(
                        attn, "atom_split_kv_layout", "dense"
                    )
                continue

            total_pages = swa_pages + num_blocks * k_per_block
            expected_bytes = total_pages * self.head_dim * get_dtype_size(self.dtype)
            storage_bytes = kv_cache.untyped_storage().nbytes()
            if storage_bytes < expected_bytes:
                raise RuntimeError(
                    "ROCm DSV4 ATOM vLLM-owned unified KV storage is too "
                    f"small for layer {getattr(attn, 'prefix', '<unknown>')}: "
                    f"storage_bytes={storage_bytes}, expected_bytes={expected_bytes}, "
                    f"swa_pages={swa_pages}, num_blocks={num_blocks}, "
                    f"k_per_block={k_per_block}, head_dim={self.head_dim}, "
                    f"dtype={self.dtype}."
                )
            unified = self._make_storage_view(
                kv_cache,
                dtype=self.dtype,
                offset_bytes=0,
                shape=(total_pages, self.head_dim),
            )
            compressed = unified[swa_pages:].view(
                num_blocks,
                k_per_block,
                self.head_dim,
            )
            unified_kv.append(unified)
            unified_kv_by_layer[layer_id] = unified
            unified_layer_ids.append(layer_id)
            compressed_kv_cache[layer_id] = compressed
            compressed_kv_scales[layer_id] = None
            compressed_kv_layout[layer_id] = "dense"

            attn.atom_unified_kv = unified
            attn.atom_swa_kv = unified[:swa_pages].view(
                self.max_num_reqs, self.win_with_spec, self.head_dim
            )
            attn.atom_direct_swa_bound = True
            attn.atom_swa_pages = swa_pages
            attn.atom_compressed_kv_cache = compressed
            attn.atom_split_kv_swa = attn.atom_swa_kv
            attn.atom_split_kv_compressed = compressed
            attn.atom_split_kv_scales = None
            attn.atom_split_kv_layout = "dense"
            compressor = getattr(attn, "compressor", None)
            if compressor is not None:
                compressor.atom_kv_cache = compressed
                compressor.atom_kv_scales = getattr(attn, "atom_split_kv_scales", None)
                compressor.atom_kv_layout = "dense"

        self._atom_unified_kv_buffers = DeepseekV4RocmAtomUnifiedKVBuffers(
            unified_kv=tuple(unified_kv),
            unified_kv_by_layer=unified_kv_by_layer,
            compressed_kv_cache=compressed_kv_cache,
            compressed_kv_scales=compressed_kv_scales,
            compressed_kv_layout=compressed_kv_layout,
            active_layer_ids=tuple(unified_layer_ids),
            num_blocks=num_blocks,
            swa_pages=self.swa_pages,
            k1_csa=self.k1_csa,
            k2_hca=self.k2_hca,
        )
        self._atom_unified_kv_from_vllm_bound = True
        ratio_counts: dict[int, int] = {}
        layout_counts: dict[str, int] = {}
        for _, attn in atom_attn:
            ratio = int(getattr(attn, "compress_ratio", 0))
            if ratio > 1:
                ratio_counts[ratio] = ratio_counts.get(ratio, 0) + 1
                layout = getattr(attn, "atom_split_kv_layout", "dense")
                layout_counts[layout] = layout_counts.get(layout, 0) + 1
        logger.info(
            "Bound ROCm DSV4 ATOM unified KV views from vLLM-owned KV storage: "
            "active_layers=%d, ratio_counts=%s, num_blocks=%d, swa_pages=%d, "
            "win_with_spec=%d, head_dim=%d, dtype=%s, layout_counts=%s",
            len(unified_layer_ids),
            ratio_counts,
            num_blocks,
            self.swa_pages,
            self.win_with_spec,
            self.head_dim,
            self.dtype,
            layout_counts,
        )
        return True

    def _maybe_allocate_atom_unified_kv(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        if self._try_bind_atom_unified_kv_from_vllm(kv_cache_config):
            return

        num_blocks = int(getattr(kv_cache_config, "num_blocks", 0) or 0)
        if num_blocks <= 0:
            logger.warning_once(
                "Skipping ROCm DSV4 ATOM unified KV allocation because "
                "kv_cache_config.num_blocks is unavailable."
            )
            return
        raise RuntimeError(
            "ROCm DeepSeek-V4 requires ATOM KV views backed by vLLM-owned "
            "storage, but ModelState could not bind the allocated KV cache."
        )

    def _build_atom_state_metadata(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        compress_plans: dict[int, CompressPlan] | None,
        *,
        request_length_views: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> DeepseekV4RocmAtomStateMetadata:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        num_actual_reqs = input_batch.num_reqs
        pure_decode_one_token = False
        if num_actual_reqs:
            if request_length_views is None:
                scheduled, computed, context_lens = (
                    self._copy_actual_request_length_views(input_batch)
                )
            else:
                scheduled, computed, context_lens = request_length_views

            self._state_slot_mapping_cpu[:num_actual_reqs] = input_batch.idx_mapping_np[
                :num_actual_reqs
            ]

            actual_tokens = int(input_batch.num_tokens)
            pure_decode_one_token = (
                actual_tokens == num_actual_reqs
                and scheduled.size == num_actual_reqs
                and bool(np.all(scheduled == 1))
            )
            self._state_slot_mapping[:num_actual_reqs].copy_(
                input_batch.idx_mapping[:num_actual_reqs],
                non_blocking=True,
            )
            self._chunk_start_per_seq_cpu[:num_actual_reqs] = computed
            self._chunk_start_per_seq[:num_actual_reqs].copy_(
                self._chunk_start_per_seq_cpu_tensor[:num_actual_reqs],
                non_blocking=True,
            )
            if pure_decode_one_token:
                batch_id_per_token = self._req_arange_cpu[:num_actual_reqs]
            else:
                batch_id_per_token = np.repeat(
                    self._req_arange_cpu[:num_actual_reqs],
                    scheduled,
                )
            self._batch_id_per_token_cpu[:actual_tokens] = batch_id_per_token
            if num_tokens > actual_tokens:
                self._batch_id_per_token_cpu[actual_tokens:num_tokens] = -1
            self._batch_id_per_token[:num_tokens].copy_(
                self._batch_id_per_token_cpu_tensor[:num_tokens],
                non_blocking=True,
            )
        if num_actual_reqs:
            if pure_decode_one_token:
                self._positions_cpu[:num_actual_reqs] = computed
            else:
                token_cursor = 0
                for req_idx, num_scheduled in enumerate(scheduled):
                    num_scheduled_int = int(num_scheduled)
                    if num_scheduled_int <= 0:
                        continue
                    start_pos = int(input_batch.num_computed_tokens_np[req_idx])
                    token_end = token_cursor + num_scheduled_int
                    self._positions_cpu[token_cursor:token_end] = np.arange(
                        start_pos,
                        start_pos + num_scheduled_int,
                        dtype=np.int32,
                    )
                    token_cursor = token_end
            if num_tokens > actual_tokens:
                self._positions_cpu[actual_tokens:num_tokens] = 0

            self._n_committed_csa_per_seq_cpu[:num_actual_reqs] = context_lens // 4
            self._n_committed_hca_per_seq_cpu[:num_actual_reqs] = context_lens // 128
            self._n_committed_csa_per_seq[:num_actual_reqs].copy_(
                self._n_committed_csa_per_seq_cpu_tensor[:num_actual_reqs],
                non_blocking=True,
            )
            self._n_committed_hca_per_seq[:num_actual_reqs].copy_(
                self._n_committed_hca_per_seq_cpu_tensor[:num_actual_reqs],
                non_blocking=True,
            )
        if num_reqs > num_actual_reqs:
            self._state_slot_mapping[num_actual_reqs:num_reqs].zero_()
            self._state_slot_mapping_cpu[num_actual_reqs:num_reqs] = 0
            self._chunk_start_per_seq[num_actual_reqs:num_reqs].zero_()
            self._chunk_start_per_seq_cpu[num_actual_reqs:num_reqs] = 0
            self._n_committed_csa_per_seq[num_actual_reqs:num_reqs].zero_()
            self._n_committed_hca_per_seq[num_actual_reqs:num_reqs].zero_()
            self._n_committed_csa_per_seq_cpu[num_actual_reqs:num_reqs] = 0
            self._n_committed_hca_per_seq_cpu[num_actual_reqs:num_reqs] = 0
        elif num_actual_reqs == 0:
            self._batch_id_per_token_cpu[:num_tokens] = -1
            self._batch_id_per_token[:num_tokens].fill_(-1)
            self._positions_cpu[:num_tokens] = 0
            self._chunk_start_per_seq_cpu[:num_reqs] = 0
            self._chunk_start_per_seq[:num_reqs].zero_()
        # Pure decode consumes CPU arrays only while building graph-stable
        # indptrs below; the model forward reads the GPU tensors. Avoid per-step
        # numpy allocations for snapshots that would otherwise die unused.
        copy_cpu_snapshots = not pure_decode_one_token

        def _cpu_metadata_view(array: np.ndarray, size: int) -> np.ndarray:
            view = array[:size]
            return view.copy() if copy_cpu_snapshots else view

        metadata = DeepseekV4RocmAtomStateMetadata(
            state_slot_mapping=self._state_slot_mapping[:num_reqs],
            state_slot_mapping_cpu=_cpu_metadata_view(
                self._state_slot_mapping_cpu,
                num_reqs,
            ),
            num_actual_reqs=num_actual_reqs,
            num_reqs=num_reqs,
            num_actual_tokens=int(input_batch.num_tokens),
            num_tokens=num_tokens,
            win_with_spec=self.win_with_spec,
            swa_pages=self.swa_pages,
            chunk_start_per_seq=self._chunk_start_per_seq[:num_reqs],
            chunk_start_per_seq_cpu=_cpu_metadata_view(
                self._chunk_start_per_seq_cpu,
                num_reqs,
            ),
            positions=input_batch.positions[:num_tokens],
            positions_cpu=_cpu_metadata_view(self._positions_cpu, num_tokens),
            query_start_loc=input_batch.query_start_loc[: num_reqs + 1],
            seq_lens=input_batch.seq_lens[:num_reqs],
            batch_id_per_token=self._batch_id_per_token[:num_tokens],
            batch_id_per_token_cpu=_cpu_metadata_view(
                self._batch_id_per_token_cpu,
                num_tokens,
            ),
            n_committed_csa_per_seq=self._n_committed_csa_per_seq[:num_reqs],
            n_committed_csa_per_seq_cpu=_cpu_metadata_view(
                self._n_committed_csa_per_seq_cpu,
                num_reqs,
            ),
            n_committed_hca_per_seq=self._n_committed_hca_per_seq[:num_reqs],
            n_committed_hca_per_seq_cpu=_cpu_metadata_view(
                self._n_committed_hca_per_seq_cpu,
                num_reqs,
            ),
            buffers=self._atom_state_buffers,
            unified_kv_buffers=self._atom_unified_kv_buffers,
            decode_buffers=self._atom_decode_buffers,
            decode_cache=DeepseekV4RocmAtomDecodeCache(),
            prefill_buffers=self._atom_prefill_buffers,
            prefill_cache=DeepseekV4RocmAtomPrefillCache(self.index_topk),
            compress_plans=compress_plans,
        )
        self._prepare_atom_decode_metadata(metadata)
        return metadata

    def _prepare_atom_decode_metadata(
        self,
        metadata: DeepseekV4RocmAtomStateMetadata,
    ) -> None:
        """Build graph-stable ATOM decode indptrs outside model forward."""

        buffers = metadata.decode_buffers
        if buffers is None:
            return

        T = int(metadata.num_tokens)
        if T <= 0:
            buffers.swa_indptr_cpu[:1] = 0
            buffers.csa_indptr_cpu[:1] = 0
            buffers.hca_indptr_cpu[:1] = 0
            buffers.swa_indptr[:1].copy_(
                buffers.swa_indptr_cpu_tensor[:1], non_blocking=True
            )
            buffers.csa_indptr[:1].copy_(
                buffers.csa_indptr_cpu_tensor[:1], non_blocking=True
            )
            buffers.hca_indptr[:1].copy_(
                buffers.hca_indptr_cpu_tensor[:1], non_blocking=True
            )
            return

        positions_cpu = metadata.positions_cpu[:T]
        batch_cpu = metadata.batch_id_per_token_cpu[:T]
        valid = buffers.valid_cpu[:T]
        safe_batch = buffers.safe_batch_cpu[:T]
        pos_plus_one = buffers.pos_plus_one_cpu[:T]
        tmp_lens = buffers.tmp_lens_cpu[:T]
        swa_lens = buffers.swa_lens_cpu[:T]
        csa_lens = buffers.csa_lens_cpu[:T]
        hca_lens = buffers.hca_lens_cpu[:T]

        np.greater_equal(batch_cpu, 0, out=valid)
        np.maximum(batch_cpu, 0, out=safe_batch)
        np.add(positions_cpu, 1, out=pos_plus_one)

        np.minimum(pos_plus_one, self.window_size, out=swa_lens)
        swa_lens *= valid

        np.floor_divide(pos_plus_one, 4, out=tmp_lens)
        np.take(
            metadata.n_committed_csa_per_seq_cpu,
            safe_batch,
            out=csa_lens,
        )
        np.minimum(tmp_lens, csa_lens, out=csa_lens)
        np.minimum(csa_lens, self.index_topk, out=csa_lens)
        csa_lens *= valid

        np.take(
            metadata.n_committed_hca_per_seq_cpu,
            safe_batch,
            out=hca_lens,
        )
        hca_lens *= valid
        hca_max = int(hca_lens.max()) if T else 0

        buffers.swa_indptr_cpu[:1] = 0
        buffers.csa_indptr_cpu[:1] = 0
        buffers.hca_indptr_cpu[:1] = 0
        np.cumsum(swa_lens, out=buffers.swa_indptr_cpu[1 : T + 1])
        np.add(swa_lens, csa_lens, out=csa_lens)
        np.cumsum(csa_lens, out=buffers.csa_indptr_cpu[1 : T + 1])
        np.add(swa_lens, hca_lens, out=hca_lens)
        np.cumsum(hca_lens, out=buffers.hca_indptr_cpu[1 : T + 1])

        object.__setattr__(metadata, "decode_swa_total", int(buffers.swa_indptr_cpu[T]))
        object.__setattr__(metadata, "decode_csa_total", int(buffers.csa_indptr_cpu[T]))
        object.__setattr__(metadata, "decode_hca_total", int(buffers.hca_indptr_cpu[T]))
        object.__setattr__(
            metadata,
            "decode_max_hca_len",
            hca_max,
        )

        if metadata.decode_swa_total > buffers.max_swa_indices:
            raise RuntimeError(
                "ATOM SWA decode index buffer too small: "
                f"{metadata.decode_swa_total} > {buffers.max_swa_indices}."
            )
        if metadata.decode_csa_total > buffers.max_csa_indices:
            raise RuntimeError(
                "ATOM CSA decode index buffer too small: "
                f"{metadata.decode_csa_total} > {buffers.max_csa_indices}."
            )
        if metadata.decode_hca_total > buffers.max_hca_indices:
            raise RuntimeError(
                "ATOM HCA decode index buffer too small: "
                f"{metadata.decode_hca_total} > {buffers.max_hca_indices}."
            )

        buffers.swa_indptr[: T + 1].copy_(
            buffers.swa_indptr_cpu_tensor[: T + 1], non_blocking=True
        )
        buffers.csa_indptr[: T + 1].copy_(
            buffers.csa_indptr_cpu_tensor[: T + 1], non_blocking=True
        )
        buffers.hca_indptr[: T + 1].copy_(
            buffers.hca_indptr_cpu_tensor[: T + 1], non_blocking=True
        )

    def _attach_indexer_decode_metadata(
        self,
        metadata: DeepseekV4RocmAtomStateMetadata,
        attn_metadata: dict[str, Any],
    ) -> None:
        """Hoist the generic indexer decode schedule into ATOM ModelState.

        The current ROCm DSV4 ATOM indexer fastpath still needs the vLLM-built
        block table and DeepGEMM schedule metadata. Keep that dependency in
        ModelState preparation instead of reaching back into per-layer generic
        indexer metadata from model forward.
        """

        try:
            from vllm.v1.attention.backends.mla.indexer import (
                DeepseekV32IndexerMetadata,
            )
        except ImportError:
            return

        for layer_metadata in attn_metadata.values():
            if not isinstance(layer_metadata, DeepseekV32IndexerMetadata):
                continue
            if layer_metadata.num_prefills > 0 or layer_metadata.num_decodes <= 0:
                continue
            decode_metadata = layer_metadata.decode
            if decode_metadata is None:
                continue
            object.__setattr__(
                metadata,
                "indexer_decode_requires_padding",
                bool(decode_metadata.requires_padding),
            )
            object.__setattr__(
                metadata,
                "indexer_decode_num_tokens",
                int(layer_metadata.num_decode_tokens),
            )
            if decode_metadata.requires_padding:
                return
            object.__setattr__(
                metadata,
                "indexer_decode_block_table",
                decode_metadata.block_table,
            )
            object.__setattr__(
                metadata,
                "indexer_decode_schedule_metadata",
                decode_metadata.schedule_metadata,
            )
            return

    def _find_indexer_kv_cache_group(
        self,
        attn_groups: list[list[AttentionGroup]],
    ) -> tuple[int, int] | None:
        cached_idx = self._indexer_kv_cache_group_idx
        if cached_idx is not None and cached_idx < len(attn_groups):
            return cached_idx, self._indexer_storage_block_size

        for group_idx, groups in enumerate(attn_groups):
            for group in groups:
                backend_name = group.backend.get_name()
                if backend_name != "DEEPSEEK_V4_INDEXER":
                    continue
                storage_block_size = int(
                    getattr(group.kv_cache_spec, "storage_block_size", 0) or 0
                )
                if storage_block_size <= 0:
                    storage_block_size = self.k1_csa
                self._indexer_kv_cache_group_idx = group_idx
                self._indexer_storage_block_size = storage_block_size
                return group_idx, storage_block_size
        return None

    def _is_pure_decode_one_token_batch(
        self,
        input_batch: InputBatch,
        scheduled_tokens: np.ndarray | None = None,
    ) -> bool:
        num_reqs = int(input_batch.num_reqs)
        if num_reqs <= 0 or int(input_batch.num_tokens) != num_reqs:
            return False
        scheduled = (
            input_batch.num_scheduled_tokens[:num_reqs]
            if scheduled_tokens is None
            else scheduled_tokens
        )
        return scheduled.size == num_reqs and bool(np.all(scheduled == 1))

    @staticmethod
    def _without_attn_backends(
        attn_groups: list[list[AttentionGroup]],
        backend_names: frozenset[str],
    ) -> list[list[AttentionGroup]]:
        return [
            [group for group in groups if group.backend.get_name() not in backend_names]
            for groups in attn_groups
        ]

    @staticmethod
    def _is_atom_main_compressor_group(group: AttentionGroup) -> bool:
        if group.backend.get_name() != "CompressorBackend":
            return False
        # Main DSV4 compressor state-cache specs are based on head_dim=512:
        # ratio 4 -> 2 * 2 * 512 = 2048, ratio 128 -> 2 * 1 * 512 = 1024.
        # The indexer-inner compressor is head_dim=128 and has head_size=512;
        # it still needs generic CompressorMetadata for the native cache write.
        return int(getattr(group.kv_cache_spec, "head_size", 0) or 0) > 512

    def _without_atom_main_compressor_groups(
        self,
        attn_groups: list[list[AttentionGroup]],
    ) -> list[list[AttentionGroup]]:
        return [
            [
                group
                for group in groups
                if not self._is_atom_main_compressor_group(group)
            ]
            for groups in attn_groups
        ]

    def _can_skip_generic_atom_decode_metadata(
        self,
        input_batch: InputBatch,
        pure_decode_one_token: bool | None = None,
    ) -> bool:
        if not current_platform.is_rocm():
            return False
        if pure_decode_one_token is None:
            pure_decode_one_token = self._is_pure_decode_one_token_batch(input_batch)
        # Mixed decode+prefill needs native sparse metadata to build its
        # prefill rows; pure one-token decode uses ModelState metadata directly.
        return pure_decode_one_token

    def _attn_groups_use_only_bound_atom_layers(
        self,
        attn_groups: list[list[AttentionGroup]],
    ) -> bool:
        buffers = self._atom_state_buffers
        if buffers is None:
            return False
        bound_layer_ids = frozenset(buffers.active_layer_ids)
        saw_sparse_group = False
        sparse_backends = frozenset(
            (
                "DEEPSEEK_SPARSE_SWA",
                "FLASHMLA_SPARSE_DSV4",
                "ROCM_FLASHMLA_SPARSE_DSV4",
            )
        )
        for groups in attn_groups:
            for group in groups:
                if group.backend.get_name() not in sparse_backends:
                    continue
                saw_sparse_group = True
                for layer_name in group.layer_names:
                    try:
                        layer_id = extract_layer_index(layer_name)
                    except Exception:
                        return False
                    if layer_id not in bound_layer_ids:
                        return False
        return saw_sparse_group

    def _can_skip_generic_atom_compressor_metadata(
        self,
        input_batch: InputBatch,
        pure_decode_one_token: bool | None = None,
    ) -> bool:
        if not current_platform.is_rocm():
            return False
        if pure_decode_one_token is None:
            pure_decode_one_token = self._is_pure_decode_one_token_batch(input_batch)
        return pure_decode_one_token

    def _attach_minimal_atom_compressor_decode_metadata(
        self,
        attn_metadata: dict[str, Any],
        attn_groups: list[list[AttentionGroup]],
    ) -> bool:
        attached = False
        compressor_metadata = DeepseekV4RocmAtomCompressorDecodeMetadata()
        for groups in attn_groups:
            for group in groups:
                if not self._is_atom_main_compressor_group(group):
                    continue
                for layer_name in group.layer_names:
                    attn_metadata[layer_name] = compressor_metadata
                    attached = True
        return attached

    def _attach_minimal_atom_decode_metadata(
        self,
        metadata: DeepseekV4RocmAtomStateMetadata,
        attn_metadata: dict[str, Any],
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        input_batch: InputBatch | None = None,
    ) -> bool:
        """Attach lightweight metadata for ATOM decode/prefill.

        The direct ATOM decode kernels build/read their own SWA/CSA/HCA indices
        from ModelState and vLLM block tables. They do not need the dense/ragged
        decode tables created by the generic sparse MLA/SWA metadata builders.
        Mixed batches are supported only when vLLM has ordered all decode
        tokens before prefill tokens, matching the ROCm attention forward split.
        """

        attached = False
        (
            num_decodes,
            num_prefills,
            num_decode_tokens,
            num_prefill_tokens,
            max_query_len,
            query_start_loc_cpu,
        ) = self._atom_minimal_decode_prefill_counts(metadata, input_batch)
        prefill_seq_lens = (
            metadata.seq_lens[num_decodes : num_decodes + num_prefills]
            if num_prefills > 0
            else None
        )
        prefill_gather_lens = (
            torch.minimum(
                prefill_seq_lens,
                torch.full_like(prefill_seq_lens, self.window_size),
            )
            if prefill_seq_lens is not None
            else None
        )
        try:
            from vllm.v1.attention.backends.mla.indexer import (
                DeepseekV32IndexerMetadata,
            )
        except ImportError:
            DeepseekV32IndexerMetadata = None
        for group_idx, groups in enumerate(attn_groups):
            if group_idx >= len(block_tables) or group_idx >= len(slot_mappings):
                continue
            block_table = block_tables[group_idx]
            slot_mapping = slot_mappings[group_idx]
            for group in groups:
                backend_name = group.backend.get_name()
                if backend_name == "DEEPSEEK_SPARSE_SWA":
                    block_size = int(getattr(group.kv_cache_spec, "block_size", 0) or 0)
                    if block_size <= 0:
                        continue
                    swa_metadata = DeepseekV4RocmAtomSWADecodeMetadata(
                        block_table=block_table,
                        slot_mapping=slot_mapping,
                        block_size=block_size,
                        seq_lens=metadata.seq_lens,
                        query_start_loc=metadata.query_start_loc,
                        query_start_loc_cpu=query_start_loc_cpu,
                        num_decodes=num_decodes,
                        num_decode_tokens=num_decode_tokens,
                        num_prefills=num_prefills,
                        num_prefill_tokens=num_prefill_tokens,
                        max_query_len=max_query_len,
                        prefill_seq_lens=prefill_seq_lens,
                        prefill_gather_lens=prefill_gather_lens,
                    )
                    for layer_name in group.layer_names:
                        attn_metadata[layer_name] = swa_metadata
                        attached = True
                elif backend_name in (
                    "FLASHMLA_SPARSE_DSV4",
                    "ROCM_FLASHMLA_SPARSE_DSV4",
                ):
                    block_size = int(getattr(group.kv_cache_spec, "block_size", 0) or 0)
                    if block_size <= 0:
                        continue
                    mla_metadata = DeepseekV4RocmAtomMLADecodeMetadata(
                        block_table=block_table,
                        slot_mapping=slot_mapping,
                        block_size=block_size,
                        num_reqs=int(metadata.num_reqs),
                        max_query_len=1,
                        max_seq_len=int(self.max_model_len),
                        num_actual_tokens=int(metadata.num_actual_tokens),
                        query_start_loc=metadata.query_start_loc,
                        req_id_per_token=metadata.batch_id_per_token[
                            : metadata.num_tokens
                        ],
                        topk_tokens=int(self.index_topk),
                        num_decodes=num_decodes,
                        num_decode_tokens=num_decode_tokens,
                        num_prefills=num_prefills,
                        num_prefill_tokens=num_prefill_tokens,
                    )
                    for layer_name in group.layer_names:
                        existing_metadata = attn_metadata.get(layer_name)
                        if DeepseekV32IndexerMetadata is not None and isinstance(
                            existing_metadata, DeepseekV32IndexerMetadata
                        ):
                            attn_metadata[
                                layer_name + ".__rocm_atom_indexer_metadata"
                            ] = existing_metadata
                        attn_metadata[layer_name] = mla_metadata
                        attached = True
        return attached

    def _atom_minimal_decode_prefill_counts(
        self,
        metadata: DeepseekV4RocmAtomStateMetadata,
        input_batch: InputBatch | None,
    ) -> tuple[int, int, int, int, int, np.ndarray | None]:
        if input_batch is None:
            return (
                int(metadata.num_reqs),
                0,
                int(metadata.num_tokens),
                0,
                1,
                None,
            )
        num_reqs = int(input_batch.num_reqs)
        scheduled = input_batch.num_scheduled_tokens[:num_reqs]
        if scheduled.size <= 0:
            return (
                int(metadata.num_reqs),
                0,
                int(metadata.num_tokens),
                0,
                1,
                getattr(input_batch, "query_start_loc_np", None),
            )
        decode_mask = scheduled <= 1
        num_decodes = int(np.count_nonzero(decode_mask))
        if not (
            num_decodes == num_reqs
            or (
                num_decodes > 0
                and np.all(decode_mask[:num_decodes])
                and not np.any(decode_mask[num_decodes:])
            )
        ):
            # The minimal metadata assumes decode tokens precede prefill tokens.
            # If the scheduler ever changes that ordering, keep the pure-decode
            # shape so downstream ATOM paths fail closed instead of reading
            # mismatched prefill slices.
            return (
                int(metadata.num_reqs),
                0,
                int(metadata.num_tokens),
                0,
                1,
                getattr(input_batch, "query_start_loc_np", None),
            )
        num_prefills = num_reqs - num_decodes
        num_decode_tokens = int(scheduled[:num_decodes].sum())
        num_prefill_tokens = int(scheduled[num_decodes:].sum())
        query_start_loc_cpu = getattr(input_batch, "query_start_loc_np", None)
        if query_start_loc_cpu is not None:
            query_start_loc_cpu = query_start_loc_cpu[: num_reqs + 1]
        max_query_len = int(scheduled.max()) if scheduled.size else 1
        return (
            num_decodes,
            num_prefills,
            num_decode_tokens,
            num_prefill_tokens,
            max_query_len,
            query_start_loc_cpu,
        )

    def _attach_direct_indexer_decode_metadata(
        self,
        metadata: DeepseekV4RocmAtomStateMetadata,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        attn_groups: list[list[AttentionGroup]],
        pure_decode_one_token: bool | None = None,
    ) -> bool:
        """Attach indexer decode metadata directly from ModelState inputs.

        This covers the deployment hot path: pure decode, one token per live
        request, no MTP flattening or padding expansion. Mixed/prefill/spec
        batches intentionally fall back to the generic indexer metadata builder.
        """

        num_reqs = int(metadata.num_actual_reqs)
        num_tokens = int(metadata.num_actual_tokens)
        if num_reqs <= 0 or num_tokens != num_reqs:
            return False
        if pure_decode_one_token is None:
            pure_decode_one_token = self._is_pure_decode_one_token_batch(input_batch)
        if not pure_decode_one_token:
            return False

        group_info = self._find_indexer_kv_cache_group(attn_groups)
        if group_info is None:
            return False
        group_idx, storage_block_size = group_info
        if group_idx >= len(block_tables):
            return False

        block_table = block_tables[group_idx]
        if block_table.shape[0] < num_reqs:
            return False

        if current_platform.is_cuda() and has_deep_gemm():
            seq_lens = metadata.n_committed_csa_per_seq[:num_reqs].unsqueeze(-1)
            self._indexer_decode_schedule_metadata[:] = get_paged_mqa_logits_metadata(
                seq_lens,
                storage_block_size,
                num_compute_units(self.device.index or 0),
            )

        object.__setattr__(metadata, "indexer_decode_requires_padding", False)
        object.__setattr__(metadata, "indexer_decode_num_tokens", num_tokens)
        object.__setattr__(metadata, "indexer_decode_block_table", block_table)
        object.__setattr__(
            metadata,
            "indexer_decode_schedule_metadata",
            self._indexer_decode_schedule_metadata,
        )
        return True

    def _copy_actual_request_length_views(
        self,
        input_batch: InputBatch,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_reqs = int(input_batch.num_reqs)
        scheduled = self._scheduled_tokens_cpu[:num_reqs]
        computed = self._computed_tokens_cpu[:num_reqs]
        context_lens = self._context_lens_cpu[:num_reqs]
        if num_reqs > 0:
            np.copyto(
                scheduled,
                input_batch.num_scheduled_tokens[:num_reqs],
                casting="unsafe",
            )
            np.copyto(
                computed,
                input_batch.num_computed_tokens_np[:num_reqs],
                casting="unsafe",
            )
            np.add(computed, scheduled, out=context_lens)
        return scheduled, computed, context_lens

    def _build_compress_plans(
        self,
        input_batch: InputBatch,
        request_length_views: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> dict[int, CompressPlan] | None:
        num_reqs = input_batch.num_reqs
        if num_reqs <= 0 or not self._compress_plan_buffers:
            return None

        if request_length_views is None:
            extend_lens_cpu, _, context_lens_cpu = (
                self._copy_actual_request_length_views(input_batch)
            )
        else:
            extend_lens_cpu, _, context_lens_cpu = request_length_views
        return self._build_compress_plans_from_arrays(
            extend_lens_cpu,
            context_lens_cpu,
        )

    def _build_compress_plans_from_arrays(
        self,
        extend_lens_cpu: np.ndarray,
        context_lens_cpu: np.ndarray,
    ) -> dict[int, CompressPlan] | None:
        if extend_lens_cpu.size <= 0 or not self._compress_plan_buffers:
            return None
        ratios_overlap = [
            (ratio, ratio == 4) for ratio in sorted(self._compress_plan_buffers)
        ]
        is_decode_like = bool(extend_lens_cpu.size) and int(extend_lens_cpu.max()) <= 1
        # vLLM captures compressor kernels in CUDA/HIP graphs. The graph stores
        # the launch grid from capture time, so replay must keep the plan tensor
        # shape fixed and sentinel-fill inactive rows. Otherwise stale rows from
        # a prior larger batch are replayed by the captured kernel.
        fixed_capacity_per_ratio: dict[int, int] = {}
        fixed_write_capacity_per_ratio: dict[int, int] | None = None
        if is_decode_like:
            fixed_write_capacity_per_ratio = {}
            for ratio, buffers in self._compress_plan_buffers.items():
                per_seq_max = max(1, (self.num_spec_tokens + ratio) // ratio)
                decode_cap = min(
                    buffers["compress"].np.shape[0],
                    self.max_num_reqs * per_seq_max,
                )
                fixed_capacity_per_ratio[ratio] = decode_cap
                fixed_write_capacity_per_ratio[ratio] = min(
                    buffers["write"].np.shape[0],
                    self.max_num_reqs * per_seq_max,
                )
        else:
            fixed_capacity_per_ratio = {
                ratio: buffers["compress"].np.shape[0]
                for ratio, buffers in self._compress_plan_buffers.items()
            }
        plans = make_compress_plans(
            extend_lens_cpu,
            context_lens_cpu,
            ratios_overlap,
            plan_buffers=self._compress_plan_buffers,
            decode_capacity_per_ratio=fixed_capacity_per_ratio,
            write_capacity_per_ratio=fixed_write_capacity_per_ratio,
        )
        # ATOM's graph path uses decode-tight write-plan slices. Launching
        # update_compressor_states over the full prefill-capacity write buffer
        # leaves thousands of sentinel rows for every HCA layer and has proven
        # unsafe on the vLLM captured decode path. Keep the base allocation and
        # data pointer stable, but narrow the tensor view for decode-like fwds.
        if is_decode_like:
            for ratio, plan in plans.items():
                per_seq_max = max(1, (self.num_spec_tokens + ratio) // ratio)
                write_cap = min(
                    plan.write_plan_gpu.shape[0],
                    self.max_num_reqs * per_seq_max,
                )
                if plan.num_write > write_cap:
                    raise RuntimeError(
                        "ATOM decode write-plan capacity is too small for "
                        f"ratio={ratio}: num_write={plan.num_write}, "
                        f"capacity={write_cap}."
                    )
                plan.write_plan_gpu = plan.write_plan_gpu[:write_cap]
        return plans

    def build_legacy_runner_metadata(
        self,
        *,
        num_actual_reqs: int,
        num_reqs: int,
        num_actual_tokens: int,
        num_tokens: int,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        req_indices_cpu: np.ndarray,
        num_scheduled_tokens_cpu: np.ndarray,
        num_computed_tokens_cpu: np.ndarray,
        kv_cache_config: KVCacheConfig,
    ) -> DeepseekV4RocmAtomStateMetadata:
        """Build ATOM metadata for the legacy GPUModelRunner path."""

        self._maybe_allocate_atom_unified_kv(kv_cache_config)

        if num_actual_reqs:
            self._state_slot_mapping_cpu[:num_actual_reqs] = np.arange(
                num_actual_reqs,
                dtype=np.int32,
            )
            self._state_slot_mapping[:num_actual_reqs].copy_(
                self._state_slot_mapping_cpu_tensor[:num_actual_reqs],
                non_blocking=True,
            )

            req_indices = np.ascontiguousarray(
                req_indices_cpu[:num_actual_tokens],
                dtype=np.int32,
            )
            self._batch_id_per_token_cpu[:num_actual_tokens] = req_indices
            if num_tokens > num_actual_tokens:
                self._batch_id_per_token_cpu[num_actual_tokens:num_tokens] = -1
            self._batch_id_per_token[:num_tokens].copy_(
                self._batch_id_per_token_cpu_tensor[:num_tokens],
                non_blocking=True,
            )

            computed = np.ascontiguousarray(
                num_computed_tokens_cpu[:num_actual_reqs],
                dtype=np.int32,
            )
            self._chunk_start_per_seq_cpu[:num_actual_reqs] = computed
            self._chunk_start_per_seq[:num_actual_reqs].copy_(
                self._chunk_start_per_seq_cpu_tensor[:num_actual_reqs],
                non_blocking=True,
            )
            scheduled = np.ascontiguousarray(
                num_scheduled_tokens_cpu[:num_actual_reqs],
                dtype=np.int32,
            )
            counters = np.zeros(num_actual_reqs, dtype=np.int32)
            for token_idx, req_idx in enumerate(req_indices):
                if req_idx < 0 or req_idx >= num_actual_reqs:
                    self._positions_cpu[token_idx] = 0
                    continue
                self._positions_cpu[token_idx] = computed[req_idx] + counters[req_idx]
                counters[req_idx] += 1
            if num_tokens > num_actual_tokens:
                self._positions_cpu[num_actual_tokens:num_tokens] = 0

            context_lens = np.ascontiguousarray(computed + scheduled, dtype=np.int32)
            self._n_committed_csa_per_seq_cpu[:num_actual_reqs] = context_lens // 4
            self._n_committed_hca_per_seq_cpu[:num_actual_reqs] = context_lens // 128
            self._n_committed_csa_per_seq[:num_actual_reqs].copy_(
                self._n_committed_csa_per_seq_cpu_tensor[:num_actual_reqs],
                non_blocking=True,
            )
            self._n_committed_hca_per_seq[:num_actual_reqs].copy_(
                self._n_committed_hca_per_seq_cpu_tensor[:num_actual_reqs],
                non_blocking=True,
            )
        else:
            self._batch_id_per_token_cpu[:num_tokens] = -1
            self._batch_id_per_token[:num_tokens].fill_(-1)
            self._positions_cpu[:num_tokens] = 0

        if num_reqs > num_actual_reqs:
            self._state_slot_mapping_cpu[num_actual_reqs:num_reqs] = 0
            self._state_slot_mapping[num_actual_reqs:num_reqs].zero_()
            self._chunk_start_per_seq_cpu[num_actual_reqs:num_reqs] = 0
            self._chunk_start_per_seq[num_actual_reqs:num_reqs].zero_()
            self._n_committed_csa_per_seq_cpu[num_actual_reqs:num_reqs] = 0
            self._n_committed_hca_per_seq_cpu[num_actual_reqs:num_reqs] = 0
            self._n_committed_csa_per_seq[num_actual_reqs:num_reqs].zero_()
            self._n_committed_hca_per_seq[num_actual_reqs:num_reqs].zero_()

        compress_plans = self._build_compress_plans_from_arrays(
            np.ascontiguousarray(
                num_scheduled_tokens_cpu[:num_actual_reqs],
                dtype=np.int32,
            ),
            np.ascontiguousarray(
                num_computed_tokens_cpu[:num_actual_reqs]
                + num_scheduled_tokens_cpu[:num_actual_reqs],
                dtype=np.int32,
            ),
        )

        metadata = DeepseekV4RocmAtomStateMetadata(
            state_slot_mapping=self._state_slot_mapping[:num_reqs],
            state_slot_mapping_cpu=self._state_slot_mapping_cpu[:num_reqs].copy(),
            num_actual_reqs=num_actual_reqs,
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            num_tokens=num_tokens,
            win_with_spec=self.win_with_spec,
            swa_pages=self.swa_pages,
            chunk_start_per_seq=self._chunk_start_per_seq[:num_reqs],
            chunk_start_per_seq_cpu=self._chunk_start_per_seq_cpu[:num_reqs].copy(),
            positions=positions[:num_tokens],
            positions_cpu=self._positions_cpu[:num_tokens].copy(),
            query_start_loc=query_start_loc[: num_reqs + 1],
            seq_lens=seq_lens[:num_reqs],
            batch_id_per_token=self._batch_id_per_token[:num_tokens],
            batch_id_per_token_cpu=self._batch_id_per_token_cpu[:num_tokens].copy(),
            n_committed_csa_per_seq=self._n_committed_csa_per_seq[:num_reqs],
            n_committed_csa_per_seq_cpu=(
                self._n_committed_csa_per_seq_cpu[:num_reqs].copy()
            ),
            n_committed_hca_per_seq=self._n_committed_hca_per_seq[:num_reqs],
            n_committed_hca_per_seq_cpu=(
                self._n_committed_hca_per_seq_cpu[:num_reqs].copy()
            ),
            buffers=self._atom_state_buffers,
            unified_kv_buffers=self._atom_unified_kv_buffers,
            decode_buffers=self._atom_decode_buffers,
            prefill_buffers=self._atom_prefill_buffers,
            compress_plans=compress_plans,
        )
        self._prepare_atom_decode_metadata(metadata)
        return metadata

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        request_length_views = self._copy_actual_request_length_views(input_batch)
        pure_decode_one_token = self._is_pure_decode_one_token_batch(
            input_batch,
            request_length_views[0],
        )
        skip_generic_atom_decode_metadata = self._can_skip_generic_atom_decode_metadata(
            input_batch,
            pure_decode_one_token,
        ) and self._attn_groups_use_only_bound_atom_layers(attn_groups)
        skip_generic_atom_compressor_metadata = (
            self._can_skip_generic_atom_compressor_metadata(
                input_batch,
                pure_decode_one_token,
            )
            and any(
                self._is_atom_main_compressor_group(group)
                for groups in attn_groups
                for group in groups
            )
        )
        skip_backend_names: set[str] = set()
        if skip_generic_atom_decode_metadata:
            skip_backend_names.update(
                (
                    "DEEPSEEK_SPARSE_SWA",
                    "FLASHMLA_SPARSE_DSV4",
                    "ROCM_FLASHMLA_SPARSE_DSV4",
                )
            )
        metadata_attn_groups = (
            self._without_attn_backends(attn_groups, frozenset(skip_backend_names))
            if skip_backend_names
            else attn_groups
        )
        if skip_generic_atom_compressor_metadata:
            metadata_attn_groups = self._without_atom_main_compressor_groups(
                metadata_attn_groups
            )
        attn_metadata = super().prepare_attn(
            input_batch=input_batch,
            cudagraph_mode=cudagraph_mode,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            attn_groups=metadata_attn_groups,
            kv_cache_config=kv_cache_config,
            for_capture=for_capture,
        )

        self._maybe_allocate_atom_unified_kv(kv_cache_config)
        compress_plans = self._build_compress_plans(
            input_batch,
            request_length_views,
        )
        atom_state = self._build_atom_state_metadata(
            input_batch,
            cudagraph_mode,
            compress_plans,
            request_length_views=request_length_views,
        )
        if not self._attach_direct_indexer_decode_metadata(
            atom_state,
            input_batch,
            block_tables,
            attn_groups,
            pure_decode_one_token,
        ):
            self._attach_indexer_decode_metadata(atom_state, attn_metadata)
        if skip_generic_atom_decode_metadata and not (
            self._attach_minimal_atom_decode_metadata(
                atom_state,
                attn_metadata,
                block_tables,
                slot_mappings,
                attn_groups,
                input_batch=input_batch,
            )
        ):
            raise RuntimeError(
                "Could not attach minimal ROCm DeepSeek-V4 decode metadata."
            )
        if skip_generic_atom_compressor_metadata and not (
            self._attach_minimal_atom_compressor_decode_metadata(
                attn_metadata,
                attn_groups,
            )
        ):
            raise RuntimeError(
                "Could not attach minimal ROCm DeepSeek-V4 compressor metadata."
            )
        seen: set[int] = set()
        for metadata in attn_metadata.values():
            metadata_id = id(metadata)
            if metadata_id in seen:
                continue
            seen.add(metadata_id)
            object.__setattr__(metadata, "deepseek_v4_rocm_atom_state", atom_state)
        return attn_metadata

    def augment_draft_decode_attn_metadata(
        self,
        attn_metadata: dict[str, Any],
        *,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        input_buffers: InputBuffers,
        idx_mapping: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, Any]:
        """Attach ATOM request state to later autoregressive MTP forwards."""
        del attn_groups, block_tables, slot_mappings
        self._maybe_allocate_atom_unified_kv(kv_cache_config)

        if num_reqs <= 0:
            return attn_metadata
        if num_tokens_padded < num_reqs or num_reqs_padded < num_reqs:
            raise ValueError("Invalid padded shape for MTP draft decode.")

        self._state_slot_mapping[:num_reqs].copy_(idx_mapping[:num_reqs])
        if num_reqs_padded > num_reqs:
            self._state_slot_mapping[num_reqs:num_reqs_padded].zero_()

        self._batch_id_per_token_cpu[:num_reqs] = self._req_arange_cpu[:num_reqs]
        if num_tokens_padded > num_reqs:
            self._batch_id_per_token_cpu[num_reqs:num_tokens_padded] = -1
        self._batch_id_per_token[:num_tokens_padded].copy_(
            self._batch_id_per_token_cpu_tensor[:num_tokens_padded],
            non_blocking=True,
        )

        self._draft_positions_cpu_tensor[:num_reqs].copy_(
            input_buffers.positions[:num_reqs],
            non_blocking=False,
        )
        self._positions_cpu[:num_reqs] = self._draft_positions_cpu[:num_reqs]
        if num_tokens_padded > num_reqs:
            self._positions_cpu[num_reqs:num_tokens_padded] = 0

        self._chunk_start_per_seq_cpu[:num_reqs] = self._positions_cpu[:num_reqs]
        self._chunk_start_per_seq[:num_reqs].copy_(
            self._chunk_start_per_seq_cpu_tensor[:num_reqs],
            non_blocking=True,
        )
        np.add(
            self._positions_cpu[:num_reqs],
            1,
            out=self._context_lens_cpu[:num_reqs],
        )
        np.floor_divide(
            self._context_lens_cpu[:num_reqs],
            4,
            out=self._n_committed_csa_per_seq_cpu[:num_reqs],
        )
        np.floor_divide(
            self._context_lens_cpu[:num_reqs],
            128,
            out=self._n_committed_hca_per_seq_cpu[:num_reqs],
        )
        self._n_committed_csa_per_seq[:num_reqs].copy_(
            self._n_committed_csa_per_seq_cpu_tensor[:num_reqs],
            non_blocking=True,
        )
        self._n_committed_hca_per_seq[:num_reqs].copy_(
            self._n_committed_hca_per_seq_cpu_tensor[:num_reqs],
            non_blocking=True,
        )
        if num_reqs_padded > num_reqs:
            self._chunk_start_per_seq[num_reqs:num_reqs_padded].zero_()
            self._n_committed_csa_per_seq[num_reqs:num_reqs_padded].zero_()
            self._n_committed_hca_per_seq[num_reqs:num_reqs_padded].zero_()

        atom_state = DeepseekV4RocmAtomStateMetadata(
            state_slot_mapping=self._state_slot_mapping[:num_reqs_padded],
            state_slot_mapping_cpu=self._state_slot_mapping_cpu[:num_reqs_padded],
            num_actual_reqs=num_reqs,
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_reqs,
            num_tokens=num_tokens_padded,
            win_with_spec=self.win_with_spec,
            swa_pages=self.swa_pages,
            chunk_start_per_seq=self._chunk_start_per_seq[:num_reqs_padded],
            chunk_start_per_seq_cpu=self._chunk_start_per_seq_cpu[:num_reqs_padded],
            positions=input_buffers.positions[:num_tokens_padded],
            positions_cpu=self._positions_cpu[:num_tokens_padded],
            query_start_loc=input_buffers.query_start_loc[: num_reqs_padded + 1],
            seq_lens=input_buffers.seq_lens[:num_reqs_padded],
            batch_id_per_token=self._batch_id_per_token[:num_tokens_padded],
            batch_id_per_token_cpu=self._batch_id_per_token_cpu[:num_tokens_padded],
            n_committed_csa_per_seq=self._n_committed_csa_per_seq[:num_reqs_padded],
            n_committed_csa_per_seq_cpu=(
                self._n_committed_csa_per_seq_cpu[:num_reqs_padded]
            ),
            n_committed_hca_per_seq=self._n_committed_hca_per_seq[:num_reqs_padded],
            n_committed_hca_per_seq_cpu=(
                self._n_committed_hca_per_seq_cpu[:num_reqs_padded]
            ),
            buffers=self._atom_state_buffers,
            unified_kv_buffers=self._atom_unified_kv_buffers,
            decode_buffers=self._atom_decode_buffers,
            decode_cache=DeepseekV4RocmAtomDecodeCache(),
            prefill_buffers=self._atom_prefill_buffers,
            prefill_cache=DeepseekV4RocmAtomPrefillCache(self.index_topk),
            compress_plans=None,
        )
        self._prepare_atom_decode_metadata(atom_state)
        for metadata in {id(value): value for value in attn_metadata.values()}.values():
            object.__setattr__(metadata, "deepseek_v4_rocm_atom_state", atom_state)
        return attn_metadata
