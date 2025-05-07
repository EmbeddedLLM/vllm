# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionType)
from vllm.attention.ops.rocm_aiter_paged_attn import AITERPagedAttention
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class ROCMAiterBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_V1"

    @staticmethod
    def get_impl_cls() -> type[ROCMAITERImpl]:
        return ROCMAITERImpl

    @staticmethod
    def get_metadata_cls() -> type[ROCMAiterMetadata]:
        return ROCMAiterMetadata

    @staticmethod
    def get_builder_cls() -> type[ROCMAiterMetadataBuilder]:
        return ROCMAiterMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)


@dataclass
class ROCMAiterMetadata:

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int

    # The number of key/value heads
    num_kv_heads: int
    # The dimension of the attention heads
    head_dim: int

    # The data type of the query
    slot_mapping: torch.Tensor

    qo_indptr: torch.Tensor

    @property
    def query_start_loc(self):
        # The GPUModelRunner expects to be able to access this property.
        return self.qo_indptr

    def __post_init__(self):
        supported_head_sizes = ROCMAiterBackend.get_supported_head_sizes()
        if self.head_dim is not None and self.head_dim \
                not in supported_head_sizes:
            raise ValueError(
                f"Only {supported_head_sizes} are supported for head_dim,",
                f" received {self.head_dim}.")


class ROCMAiterMetadataBuilder:

    def __init__(self, runner: GPUModelRunner):
        self.runner = runner

    def reorder_batch(self, input_batch: InputBatch,
                      scheduler_output: SchedulerOutput) -> bool:
        return False

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int):

        max_seq_len = self.runner.seq_lens_np[:num_reqs].max()

        seq_lens = self.runner.seq_lens_cpu[:num_reqs].to(self.runner.device,
                                                          non_blocking=True)
        block_table = (
            self.runner.input_batch.block_table.get_device_tensor()[:num_reqs])

        slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True).long()

        qo_indptr = self.runner.query_start_loc_cpu[:num_reqs + 1].to(
            self.runner.device, non_blocking=True)

        attn_metadata = ROCMAiterMetadata(
            num_actual_tokens=num_actual_tokens,
            num_kv_heads=self.runner.num_kv_heads,
            head_dim=self.runner.head_size,
            slot_mapping=slot_mapping,
            block_table=block_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            max_query_len=max_query_len,
            qo_indptr=qo_indptr)

        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class ROCMAITERImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> None:
        self.aiter_kv_scales_initialized = False

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_cache_dtype = kv_cache_dtype
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "ROCMAiterBackend")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ROCMAiterMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashInfer.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run.
            return output

        if (kv_cache.dtype.itemsize == 1
                and not self.aiter_kv_scales_initialized
                and kv_cache.shape != torch.Size([0])):
            num_blocks = kv_cache.shape[1]
            block_size = kv_cache.shape[2] // (self.num_kv_heads *
                                               self.head_size)
            k_scale = torch.empty((self.num_kv_heads, num_blocks * block_size),
                                  dtype=torch.float32,
                                  device=kv_cache.device)
            v_scale = torch.empty((self.num_kv_heads, num_blocks * block_size),
                                  dtype=torch.float32,
                                  device=kv_cache.device)
            self.aiter_kv_scales_initialized = True
            k_scale.fill_(layer._k_scale.item())
            v_scale.fill_(layer._v_scale.item())
            layer._k_scale = k_scale
            layer._v_scale = v_scale

        num_actual_tokens = attn_metadata.num_actual_tokens

        key_cache, value_cache = AITERPagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size)

        AITERPagedAttention.write_to_paged_cache(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        if attn_metadata.max_query_len > 1:
            print("flash_attn_varlen_func")
            from aiter import flash_attn_varlen_func
            output[:num_actual_tokens] = flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
                alibi_slopes=self.alibi_slopes,
            )
        AITERPagedAttention.forward_decode(
            query=query[:num_actual_tokens],
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            max_seq_len=attn_metadata.max_seq_len,
            kv_cache_dtype=self.kv_cache_dtype,
            num_kv_heads=attn_metadata.num_kv_heads,
            scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
            output=output[:num_actual_tokens],
        )

        return output
