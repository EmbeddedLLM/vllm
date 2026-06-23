# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.library import Library
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.models.deepseek_v4.common.ops import (
    fused_indexer_q_rope_quant,
    fused_q_kv_rmsnorm,
    scale_indexer_weights,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
)
from vllm.utils.torch_utils import direct_register_custom_op, get_dtype_size
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import (
    DeepseekV4AtomMLAAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
)

logger = init_logger(__name__)
_ATOM_ATTENTION_ENABLED = (
    current_platform.is_rocm()
    and os.environ.get("VLLM_ROCM_DSV4_ATOM_ATTENTION", "0") == "1"
)
_ATOM_ATTENTION_RATIOS = frozenset(
    part.strip()
    for part in os.environ.get("VLLM_ROCM_DSV4_ATOM_ATTENTION_RATIOS", "").split(",")
    if part.strip()
)
_ATOM_ATTENTION_LAYERS = frozenset(
    part.strip()
    for part in os.environ.get("VLLM_ROCM_DSV4_ATOM_ATTENTION_LAYERS", "").split(",")
    if part.strip()
)
_ATOM_ROCM_DSV4_ENABLED = current_platform.is_rocm() and (
    _ATOM_ATTENTION_ENABLED
    or os.environ.get("VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR", "0") == "1"
    or os.environ.get("VLLM_ROCM_DSV4_ATOM_UNIFIED_KV", "0") == "1"
)
_ATOM_UNIFIED_KV_FROM_VLLM_ENABLED = (
    os.environ.get("VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM", "0") == "1"
)
_ATOM_MIXED_KV_ENABLED = (
    os.environ.get("VLLM_ROCM_DSV4_ATOM_MIXED_KV", "0") == "1"
)
_ATOM_INDEXER_FASTPATH_ENABLED = (
    os.environ.get("VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH", "0") == "1"
)
_ATOM_INDEXER_DISPATCH_ENABLED = (
    current_platform.is_rocm()
    and os.environ.get("VLLM_ROCM_DSV4_ATOM_INDEXER_DISPATCH", "0") == "1"
)
_ATOM_INDEXER_SEQUENCE_ENABLED = (
    current_platform.is_rocm()
    and os.environ.get("VLLM_ROCM_DSV4_ATOM_INDEXER_SEQUENCE", "0") == "1"
)
_ATOM_INDEXER_FASTPATH_NEEDS_SENTINEL_FILL = not (
    _ATOM_ATTENTION_ENABLED
    and _ATOM_UNIFIED_KV_FROM_VLLM_ENABLED
    and not _ATOM_ATTENTION_RATIOS
    and not _ATOM_ATTENTION_LAYERS
)
_ATOM_USE_INDEX_CACHE_OVERRIDE = os.environ.get(
    "VLLM_ROCM_DSV4_ATOM_USE_INDEX_CACHE"
)
_ATOM_INDEX_TOPK_FREQ_OVERRIDE = os.environ.get(
    "VLLM_ROCM_DSV4_ATOM_INDEX_TOPK_FREQ"
)
_ATOM_INDEX_TOPK_PATTERN_OVERRIDE = os.environ.get(
    "VLLM_ROCM_DSV4_ATOM_INDEX_TOPK_PATTERN"
)
_ATOM_INDEXER_DISPATCH_REGISTRY: dict[str, Any] = {}
_ATOM_AITER_FALLBACK_LIB = Library("aiter", "FRAGMENT")


def _atom_attention_enabled_for_ratio(ratio: int) -> bool:
    if not _ATOM_ATTENTION_ENABLED:
        return False
    if not _ATOM_ATTENTION_RATIOS:
        return True
    return str(max(1, int(ratio))) in _ATOM_ATTENTION_RATIOS


def _atom_attention_enabled_for_layer_id(layer_id: int | None) -> bool:
    if not _ATOM_ATTENTION_LAYERS:
        return True
    if layer_id is None:
        return False
    return str(int(layer_id)) in _ATOM_ATTENTION_LAYERS


def _atom_torch_op_exists(namespace: str, op_name: str) -> bool:
    try:
        namespace_obj = getattr(torch.ops, namespace)
        getattr(namespace_obj, op_name)
    except AttributeError:
        return False
    return True


def _atom_indexer_score_topk(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    layer_name: str,
    topk: int,
) -> torch.Tensor:
    indexer = _ATOM_INDEXER_DISPATCH_REGISTRY[layer_name]
    return indexer.indexer_score_topk(q_fp8, weights, topk)


def _atom_indexer_score_topk_fake(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    layer_name: str,
    topk: int,
) -> torch.Tensor:
    return torch.empty(
        (q_fp8.shape[0], topk),
        dtype=torch.int32,
        device=q_fp8.device,
    )


if not _atom_torch_op_exists("aiter", "indexer_score_topk"):
    direct_register_custom_op(
        op_name="indexer_score_topk",
        op_func=_atom_indexer_score_topk,
        mutates_args=[],
        fake_impl=_atom_indexer_score_topk_fake,
        target_lib=_ATOM_AITER_FALLBACK_LIB,
        tags=(torch.Tag.needs_fixed_stride_order,),
    )


def _atom_parse_index_topk_pattern() -> str | list[str] | None:
    pattern = _ATOM_INDEX_TOPK_PATTERN_OVERRIDE
    if pattern is None:
        return None
    pattern = pattern.strip()
    if not pattern:
        return None
    if "," in pattern:
        return [part.strip() for part in pattern.split(",")]
    return pattern


def _atom_use_index_cache(config: Any) -> bool:
    if not (_ATOM_ROCM_DSV4_ENABLED and current_platform.is_rocm()):
        return False
    if _ATOM_USE_INDEX_CACHE_OVERRIDE is not None:
        return _ATOM_USE_INDEX_CACHE_OVERRIDE == "1"
    return bool(getattr(config, "use_index_cache", False))


def _atom_index_topk_freq(config: Any) -> int:
    if _ATOM_INDEX_TOPK_FREQ_OVERRIDE is not None:
        try:
            return int(_ATOM_INDEX_TOPK_FREQ_OVERRIDE)
        except ValueError:
            return 1
    return int(getattr(config, "index_topk_freq", 1))


def _atom_index_topk_pattern(config: Any) -> Any | None:
    override = _atom_parse_index_topk_pattern()
    if override is not None:
        return override
    return getattr(config, "index_topk_pattern", None)


def _atom_v4_index_topk_refreshes(config: Any, layer_id: int) -> bool:
    pattern = _atom_index_topk_pattern(config)
    if pattern is not None:
        return not (
            0 <= layer_id < len(pattern) and str(pattern[layer_id]).upper() == "S"
        )

    index_topk_freq = _atom_index_topk_freq(config)
    if index_topk_freq <= 0:
        raise ValueError("index_topk_freq must be a positive integer")

    compress_ratios = getattr(config, "compress_ratios", ())
    csa_ordinal = (
        sum(1 for ratio in compress_ratios[: layer_id + 1] if int(ratio) == 4) - 1
    )
    if csa_ordinal < 0:
        return False
    return csa_ordinal % index_topk_freq == 0


def _atom_should_skip_v4_index_topk(config: Any, layer_id: int) -> bool:
    if not _atom_use_index_cache(config):
        return False
    compress_ratios = getattr(config, "compress_ratios", ())
    if not (0 <= layer_id < len(compress_ratios)):
        return False
    if int(compress_ratios[layer_id]) != 4:
        return False
    if _atom_v4_index_topk_refreshes(config, layer_id):
        return False

    return any(
        int(compress_ratios[prev_layer]) == 4
        and _atom_v4_index_topk_refreshes(config, prev_layer)
        for prev_layer in range(layer_id - 1, -1, -1)
    )


def _resolve_dsv4_kv_cache_dtype(
    use_flashmla_fp8_layout: bool,
    kv_cache_dtype: str,
    cache_config: CacheConfig | None,
) -> tuple[str, torch.dtype]:
    """Map ``(layout, --kv-cache-dtype)`` to ``(cache_dtype_str, torch_dtype)``.

    Both layouts are paged; they differ in the per-token block format. The
    FlashMLA fp8 layout (FlashMLA / ROCm Aiter) is the ``fp8_ds_mla`` format:
    UE8M0 block-scaled fp8 packed as ``uint8`` (the canonical ``fp8_ds_mla``
    string is written back onto ``cache_config`` so the page-size specs pick
    the 576B per-token slot). Otherwise (FlashInfer) each token's KV row is
    stored in its plain element dtype — bf16 or per-tensor FP8 E4M3.
    """
    if use_flashmla_fp8_layout:
        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.
        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 FlashMLA fp8 layout only supports fp8 kv-cache, "
            f"got {kv_cache_dtype}"
        )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8

    # Plain bf16 / per-tensor fp8 KV row (FlashInfer).
    if kv_cache_dtype.startswith("fp8"):
        return kv_cache_dtype, torch.float8_e4m3fn
    # auto / bfloat16 -> plain bf16 KV row.
    return kv_cache_dtype, torch.bfloat16


class DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC):
    """DeepseekV4 MLA attention layer.

    The platform-specific sparse-MLA forward (``forward_mqa`` /
    ``get_padded_num_q_heads`` / ``_o_proj`` / ``backend_cls``) is provided by a
    subclass — ``DeepseekV4FlashMLAAttention`` / ``DeepseekV4FlashInferMLAAttention``
    (CUDA) or ``DeepseekV4ROCMAiterMLAAttention`` (ROCm) — selected by the
    platform-specific deepseek_v4 model module. The base is never instantiated
    directly.
    """

    # Provided by the platform subclass.
    backend_cls: ClassVar[type[AttentionBackend]]
    # KV-cache per-token block format (both layouts are paged). True (default)
    # = FlashMLA / ROCm fp8_ds_mla (UE8M0 block-scaled fp8 packed as uint8);
    # False = FlashInfer plain bf16 / per-tensor fp8 KV row.
    use_flashmla_fp8_layout: ClassVar[bool] = True
    # Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
    # workspace allocated in _forward_prefill and is also read by the dummy-run
    # path to pre-reserve that workspace.
    PREFILL_CHUNK_SIZE: ClassVar[int] = 4

    @classmethod
    @abstractmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        """Q head count the q/output buffers are allocated at.

        The layer allocates the q/output buffers at
        ``[N, get_padded_num_q_heads(n_local_heads), head_dim]``. Must satisfy
        ``result >= num_heads``. Backends with no padding constraint return
        ``num_heads``.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Platform-specific sparse MLA forward; writes attention into ``output``."""
        raise NotImplementedError

    @abstractmethod
    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Inverse-RoPE + wo_a + wo_b output projection (platform-specific)."""
        raise NotImplementedError

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)

        self.layer_id = layer_id
        self.prefix = prefix  # Alias for compatibility with compressor
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        # NOTE(zyongye) Compress ratio can't be 0
        # we do this for because MTP layer is not included
        # in the compress ratio list
        if layer_id < config.num_hidden_layers:
            self.compress_ratio = max(1, config.compress_ratios[layer_id])
        else:
            self.compress_ratio = 1
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        # Padded Q head count is dictated by the platform subclass.
        self.padded_heads = self.get_padded_num_q_heads(self.n_local_heads)
        # Sink padded to the same head count, initialized to -inf (no sink
        # effect). Weight loading fills the first n_local_heads slots.
        self.attn_sink = nn.Parameter(
            torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )

        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        # Initialize rotary embedding before the indexer/compressor consume it.
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.topk_indices_buffer = topk_indices_buffer

        self.indexer = None
        self.skip_topk = False
        if self.compress_ratio == 4:
            # Only C4A uses sparse attention and hence has indexer.
            # aux_stream_list[2] is free here (outer GEMMs joined) for the inner
            # overlap of wq_b+fused_indexer_q_rope_quant vs compressor. None on
            # ROCm, where aux_stream_list is None.
            self.skip_topk = _atom_should_skip_v4_index_topk(config, layer_id)
            indexer_aux_stream = (
                aux_stream_list[2] if aux_stream_list is not None else None
            )
            self.indexer = DeepseekV4Indexer(
                vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
                aux_stream=indexer_aux_stream,
            )

        # Will be None on ROCm for now.
        self.aux_stream_list = aux_stream_list
        # [0]: GEMM start / post-GEMM event0. [1..3]: GEMM done events;
        # [1] doubles as post-GEMM event1. Reuse is safe: GEMM fully joins
        # before post-GEMM starts.
        self.ln_events = [torch.cuda.Event() for _ in range(4)]

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        # ---- Attention / KV-cache setup ----
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        # Resolve the kv-cache dtype from this backend's block format (a
        # ClassVar set by the subclass): fp8_ds_mla (UE8M0 block-scaled fp8 as
        # uint8) for FlashMLA / ROCm, vs a plain bf16 / per-tensor fp8 row for
        # FlashInfer. The same resolution drives the SWA cache tensor dtype
        # below.
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            self.use_flashmla_fp8_layout, cache_config.cache_dtype, cache_config
        )

        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        # Register with compilation context for metadata lookup.
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])

        # Create the compressor for layers with compress_ratio > 1; after the
        # attention setup above so its KV-cache prefix (self.prefix) is set.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Metadata-independent input GEMMs + RMSNorm stay in the captured
        # graph; the metadata-dependent rest (q up-proj + kv-insert, indexer,
        # compressor, MLA attention) runs in the eager break.
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm_parallel_execute(hidden_states)
        )
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )

        # attention_impl is wrapped with @eager_break_during_capture: this is
        # where the breakable cudagraph capture breaks (the attention op runs
        # eagerly between captured graph segments).
        self.attention_impl(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
        )
        o = o_padded[:, : self.n_local_heads, :]

        # Inverse-RoPE + wo_a + wo_b output projection (platform-specific).
        return self._o_proj(o, positions)

    def attn_gemm_parallel_execute(self, hidden_states) -> tuple[Any, ...]:
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            assert len(aux_streams) >= 3
            aux_streams = aux_streams[:3]

        # fused_wqa_wkv (heaviest) on default; the three lighter input GEMMs
        # on aux streams 0..2 when their owning module exists. ln_events[0]
        # is the fan-out start event; ln_events[1..3] are per-aux done events.
        # On ROCm, aux_streams is None and execute_in_parallel runs serially.
        aux_fns: list[Callable[[], Any] | None] = [None, None, None]

        if self.compressor is not None:
            # Local ref so the closure keeps a non-None type for mypy.
            compressor = self.compressor

            def compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[0] = compressor_kv_score

        if self.indexer is not None:
            indexer = self.indexer

            def indexer_weights_proj() -> torch.Tensor:
                # ReplicatedLinear returns (output, bias); bias is None.
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            def indexer_compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    indexer.compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[1] = indexer_weights_proj
            aux_fns[2] = indexer_compressor_kv_score

        def fused_wqa_wkv() -> torch.Tensor:
            # MergedColumnParallelLinear returns (output, bias); bias is None.
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            return qr_kv

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            fused_wqa_wkv,
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=hidden_states.shape[0]
            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        )

        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    @eager_break_during_capture
    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        # wq_b + kv_insert (+ MLA compressor when an indexer is present) ride
        # on the default stream so q stays on its consumer stream (forward_mqa
        # downstream reads q on default). Indexer/compressor go on aux for
        # overlap with default's GEMM + cache write.
        if self.indexer is not None and not self.skip_topk:
            aux_streams = self.aux_stream_list
            indexer = self.indexer
            # Local ref so the closure keeps a non-None type for mypy.
            assert self.compressor is not None
            compressor = self.compressor

            def wq_b_kv_insert() -> torch.Tensor:
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                return q

            # 3-way overlap (matches TRT-LLM PR #14142 Level 1): default runs
            # wq_b+kv_insert; slot [0] runs the full indexer; slot [1] runs the
            # MLA compressor. Slot [2] is reserved for the indexer's inner
            # overlap. ROCm (aux_streams is None) falls back to sequential.
            q, _ = execute_in_parallel(
                wq_b_kv_insert,
                [
                    lambda: indexer(
                        hidden_states,
                        qr,
                        indexer_kv_score,
                        indexer_weights,
                        positions,
                        self.indexer_rotary_emb,
                    ),
                    lambda: compressor(kv_score, positions, self.rotary_emb),
                ],
                self.ln_events[0],
                [self.ln_events[1], self.ln_events[2]],
                [aux_streams[0], aux_streams[1]] if aux_streams is not None else None,
                enable=aux_streams is not None,
            )
        elif self.compressor is not None:
            # wq_b + kv_insert on default, compressor on aux.
            aux_stream = (
                self.aux_stream_list[0] if self.aux_stream_list is not None else None
            )
            compressor = self.compressor

            def wq_b_kv_insert() -> torch.Tensor:
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                return q

            q, _ = maybe_execute_in_parallel(
                wq_b_kv_insert,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        else:
            # SWA-only layer: no compressor, no overlap.
            q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
            q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.forward_mqa(q, kv, positions, out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        if not isinstance(attn_metadata, dict):
            # Profile run: kernel doesn't fire; produce a padded tensor so
            # downstream FlashMLA gets the right shape.
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q,
                    (0, 0, 0, self.padded_heads - self.n_local_heads),
                    value=0.0,
                )
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        # The fused insert ops require int64 position_ids; the runner's positions
        # buffer is already int64, so no cast is needed.
        assert positions.dtype == torch.int64
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        cache_dtype = swa_kv_cache.dtype

        # kv is unchanged; attention reads kv solely via swa_kv_cache.
        if cache_dtype == torch.uint8:
            # Legacy FlashMLA UE8M0 paged path. Horizontally fused:
            #   Q side:  per-head RMSNorm (no weight) + GPT-J RoPE, zero-filling
            #            the padding head slots; the kernel allocates and returns
            #            the padded q tensor.
            #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert.
            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
            return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q,
                kv,
                swa_kv_cache_2d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.padded_heads,
                self.eps,
                swa_metadata.block_size,
            )

        # FlashInfer full-cache path: the [num_blocks, block_size, 512] cache
        # stores the KV row in its plain dtype (no Q padding). bf16 rewrites q
        # in place; per-tensor fp8 writes a separately-allocated fp8 q and
        # quantizes the KV row.
        block_size = swa_metadata.block_size
        swa_kv_cache_3d = swa_kv_cache.view(-1, block_size, self.head_dim)
        if cache_dtype == torch.bfloat16:
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
                q,
                kv,
                swa_kv_cache_3d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.eps,
                block_size,
            )
            return q

        # per-tensor fp8 (torch.float8_e4m3fn)
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
            q,
            kv,
            q_fp8,
            swa_kv_cache_3d,
            swa_metadata.slot_mapping,
            positions,
            cos_sin_cache,
            self._flashinfer_fp8_kv_scale,
            self._flashinfer_fp8_q_scale_inv,
            self.eps,
            block_size,
        )
        return q_fp8

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        # FlashMLA uses the fp8_ds_mla block format (UE8M0 block-scaled fp8 as
        # uint8, 576B aligned); FlashInfer stores a plain bf16 / per-tensor fp8
        # row with no extra alignment.
        is_flashmla = self.kv_cache_dtype == "fp8_ds_mla"
        spec_cls = MLAAttentionSpec
        extra_kwargs: dict[str, Any] = {}
        spec_dtype = torch.uint8 if is_flashmla else self.kv_cache_torch_dtype
        spec_cache_dtype = self.kv_cache_dtype
        spec_alignment = 576 if is_flashmla else None
        atom_vllm_owned_kv = (
            current_platform.is_rocm()
            and _ATOM_UNIFIED_KV_FROM_VLLM_ENABLED
            and _atom_attention_enabled_for_ratio(self.compress_ratio)
            and _atom_attention_enabled_for_layer_id(getattr(self, "layer_id", None))
        )
        atom_block_size = int(vllm_config.cache_config.block_size)
        if (
            (_ATOM_ROCM_DSV4_ENABLED or atom_vllm_owned_kv)
            and atom_block_size % 128 != 0
        ):
            raise ValueError(
                "ROCm DeepSeek-V4 ATOM kernels require --block-size to be a "
                "multiple of lcm(4,128)=128 so CSA/HCA compressed entries fit "
                f"inside each KV block; got {atom_block_size}."
            )
        if atom_vllm_owned_kv:
            atom_swa_dtype = vllm_config.model_config.dtype
            win_with_spec = self.window_size + int(
                vllm_config.num_speculative_tokens or 0
            )
            atom_swa_pages = vllm_config.scheduler_config.max_num_seqs * win_with_spec
            atom_swa_prefix_bytes = (
                atom_swa_pages * self.head_dim * get_dtype_size(atom_swa_dtype)
            )
            self.atom_vllm_unified_kv_prefix_bytes = atom_swa_prefix_bytes
            self.atom_vllm_unified_kv_swa_pages = atom_swa_pages
            self.atom_vllm_unified_kv_swa_dtype = atom_swa_dtype
            self.atom_vllm_compressed_scale_bytes_per_page = 0
            self.atom_vllm_compressed_layout = "dense"
            spec_cls = DeepseekV4AtomMLAAttentionSpec
            if _ATOM_MIXED_KV_ENABLED:
                if self.head_dim != 512:
                    raise ValueError(
                        "ROCm DeepSeek-V4 ATOM packed FP8 KV requires "
                        f"head_dim=512, got {self.head_dim}."
                    )
                # ATOM's documented DSV4 FP8 KV slot is the packed
                # fp8_ds_mla format: 448 FP8 NoPE bytes + 64 BF16 RoPE
                # values (128 bytes) + 8 UE8M0 scale bytes = 584 bytes per
                # compressed token. Do not expose the earlier all-FP8+fp32
                # sidecar experiment as the default mixed layout.
                spec_dtype = torch.uint8
                spec_cache_dtype = "fp8_ds_mla"
                atom_scale_bytes_per_page = 0
                self.atom_vllm_compressed_layout = "fp8_ds_mla"
            else:
                # Existing ATOM ROCm kernels take one homogeneous unified KV
                # tensor. Use model dtype for the compressed tail in this
                # vLLM-owned-storage slice by default.
                spec_dtype = atom_swa_dtype
                spec_cache_dtype = "bf16"
                atom_scale_bytes_per_page = 0
            spec_alignment = None
            extra_kwargs = {
                "atom_swa_prefix_bytes": atom_swa_prefix_bytes,
                "atom_swa_pages": atom_swa_pages,
                "atom_swa_dtype": atom_swa_dtype,
                "atom_compressed_kv_dtype": spec_dtype,
                "atom_compressed_layout": self.atom_vllm_compressed_layout,
                "atom_compressed_scale_dtype": torch.float32
                if atom_scale_bytes_per_page
                else None,
                "atom_compressed_scale_bytes_per_page": atom_scale_bytes_per_page,
            }
            self.atom_vllm_compressed_scale_bytes_per_page = (
                atom_scale_bytes_per_page
            )
        return spec_cls(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=spec_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=spec_cache_dtype,
            alignment=spec_alignment,
            model_version="deepseek_v4",
            **extra_kwargs,
        )

    def post_bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind ROCm ATOM unified views as soon as vLLM KV storage exists.

        ModelState also validates these views when it prepares attention
        metadata, but CUDA/HIP graph capture can happen before that first real
        metadata pass.  Creating the views at KV-cache bind time ensures graph
        capture sees the same tensors replay will use.
        """
        if (
            not current_platform.is_rocm()
            or not _ATOM_UNIFIED_KV_FROM_VLLM_ENABLED
            or self.compress_ratio <= 1
            or not hasattr(self, "atom_vllm_unified_kv_prefix_bytes")
        ):
            return
        if kv_cache.numel() == 0:
            return

        atom_swa_dtype = getattr(
            self, "atom_vllm_unified_kv_swa_dtype", self.kv_cache_torch_dtype
        )
        swa_pages = int(getattr(self, "atom_vllm_unified_kv_swa_pages", 0) or 0)
        prefix_bytes = int(
            getattr(self, "atom_vllm_unified_kv_prefix_bytes", 0) or 0
        )
        if swa_pages <= 0 or prefix_bytes <= 0:
            return
        compressed_layout = getattr(self, "atom_vllm_compressed_layout", "dense")
        if compressed_layout not in ("dense", "fp8_ds_mla"):
            raise RuntimeError(
                "ROCm DSV4 ATOM vLLM-owned KV bind got unsupported compressed "
                f"layout {compressed_layout!r} for {self.prefix}."
            )
        expected_tail_width = 584 if compressed_layout == "fp8_ds_mla" else self.head_dim
        if kv_cache.dim() != 3 or kv_cache.shape[-1] != expected_tail_width:
            raise RuntimeError(
                "ROCm DSV4 ATOM vLLM-owned KV bind expected compressed tail "
                f"[num_blocks, k_per_block, {expected_tail_width}], got "
                f"{tuple(kv_cache.shape)} for {self.prefix}."
            )

        num_blocks = int(kv_cache.shape[0])
        k_per_block = int(kv_cache.shape[1])
        tail_pages = num_blocks * k_per_block
        scale_bytes_per_page = int(
            getattr(self, "atom_vllm_compressed_scale_bytes_per_page", 0) or 0
        )
        if compressed_layout == "fp8_ds_mla":
            if kv_cache.dtype != torch.uint8:
                raise RuntimeError(
                    "ROCm DSV4 ATOM packed FP8 KV expects compressed tail dtype "
                    f"torch.uint8, got {kv_cache.dtype}."
                )
            tail_page_bytes = 584
        else:
            tail_page_bytes = self.head_dim * get_dtype_size(kv_cache.dtype)
        expected_bytes = prefix_bytes + tail_pages * (
            tail_page_bytes + scale_bytes_per_page
        )
        storage_bytes = kv_cache.untyped_storage().nbytes()
        if storage_bytes < expected_bytes:
            raise RuntimeError(
                "ROCm DSV4 ATOM vLLM-owned KV storage is too small for "
                f"{self.prefix}: storage_bytes={storage_bytes}, "
                f"expected_bytes={expected_bytes}, swa_pages={swa_pages}, "
                f"num_blocks={num_blocks}, k_per_block={k_per_block}, "
                f"head_dim={self.head_dim}, tail_dtype={kv_cache.dtype}, "
                f"scale_bytes_per_page={scale_bytes_per_page}."
            )

        swa_stride = torch.empty(
            (swa_pages, self.head_dim),
            dtype=atom_swa_dtype,
            device=kv_cache.device,
        ).stride()
        swa_flat = torch.empty((), dtype=atom_swa_dtype, device=kv_cache.device)
        swa_flat.set_(
            kv_cache.untyped_storage(),
            0,
            (swa_pages, self.head_dim),
            swa_stride,
        )
        win_with_spec = swa_pages // self.max_num_reqs
        self.atom_swa_kv = swa_flat.view(
            self.max_num_reqs,
            win_with_spec,
            self.head_dim,
        )
        self.atom_win_with_spec = win_with_spec
        self.atom_swa_pages = swa_pages
        self.atom_compressed_kv_cache = kv_cache
        self.atom_split_kv_swa = self.atom_swa_kv
        self.atom_split_kv_compressed = kv_cache
        self.atom_split_kv_scales = None
        self.atom_split_kv_layout = compressed_layout
        if (
            compressed_layout == "dense"
            and kv_cache.dtype == atom_swa_dtype
            and scale_bytes_per_page == 0
        ):
            total_pages = swa_pages + tail_pages
            unified_stride = torch.empty(
                (total_pages, self.head_dim),
                dtype=atom_swa_dtype,
                device=kv_cache.device,
            ).stride()
            unified = torch.empty((), dtype=atom_swa_dtype, device=kv_cache.device)
            unified.set_(
                kv_cache.untyped_storage(),
                0,
                (total_pages, self.head_dim),
                unified_stride,
            )
            self.atom_unified_kv = unified
        elif compressed_layout == "fp8_ds_mla":
            # Packed tail embeds its UE8M0 scales in the 584-byte slot.
            pass
        else:
            if scale_bytes_per_page <= 0:
                raise RuntimeError(
                    "ROCm DSV4 ATOM mixed KV requires per-page scale bytes "
                    f"for {self.prefix}."
                )
            if kv_cache.dtype != torch.float8_e4m3fnuz:
                raise RuntimeError(
                    "ROCm DSV4 ATOM mixed KV expects compressed tail dtype "
                    f"torch.float8_e4m3fnuz, got {kv_cache.dtype}."
                )
            scale_groups = self.head_dim // 64
            expected_scale_bytes = scale_groups * get_dtype_size(torch.float32)
            if scale_bytes_per_page != expected_scale_bytes:
                raise RuntimeError(
                    "ROCm DSV4 ATOM mixed KV scale bytes mismatch: "
                    f"got {scale_bytes_per_page}, expected {expected_scale_bytes}."
                )
            if (prefix_bytes + tail_page_bytes) % get_dtype_size(torch.float32) != 0:
                raise RuntimeError(
                    "ROCm DSV4 ATOM mixed KV scale sidecar is not fp32 aligned."
                )
            page_stride = (tail_page_bytes + scale_bytes_per_page) // get_dtype_size(
                torch.float32
            )
            scale_storage_offset = (
                prefix_bytes + tail_page_bytes
            ) // get_dtype_size(torch.float32)
            scale_base = torch.empty((), dtype=torch.float32, device=kv_cache.device)
            scale_base.set_(
                kv_cache.untyped_storage(),
                scale_storage_offset,
                (tail_pages, scale_groups),
                (page_stride, 1),
            )
            self.atom_split_kv_scales = scale_base
        if self.compressor is not None:
            self.compressor.atom_kv_cache = kv_cache
            self.compressor.atom_kv_scales = self.atom_split_kv_scales
            self.compressor.atom_kv_layout = compressed_layout


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # DeepseekV4 aligns indexer pages to FlashMLA's 576B so they can pack with
            # the indexer's compressor state cache. V3.2 keeps the legacy layout.
            alignment=576,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache
        logger.info_once(
            "Using %s indexer cache for Lightning Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # NOTE(yifan): FP8 indxer cache use the same layout as V3.2:
        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
        # For FP4 indexer cache, we still allocate the same amount of memory as FP8,
        # but only use the first half of the memory.
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = DeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )

        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
        )

        # None on ROCm — maybe_execute_in_parallel falls back to sequential.
        self.aux_stream = aux_stream
        self.ln_events: list[torch.cuda.Event] = [
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]
        _ATOM_INDEXER_DISPATCH_REGISTRY[self.prefix] = self

    def _maybe_atom_decode_indexer_fastpath(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor | None:
        """ATOM-style CSA indexer decode path over ModelState metadata.

        This intentionally starts as a narrow opt-in path: pure decode,
        one token per sequence, no padding. It bypasses the generic
        SparseAttnIndexer custom-op wrapper while reusing the same aiter
        paged-logits/top-k kernels.
        """
        if (
            not _ATOM_INDEXER_FASTPATH_ENABLED
            or not current_platform.is_rocm()
            or self.use_fp4_kv
            or self.topk_indices_buffer is None
        ):
            return None

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None

        from vllm.models.deepseek_v4.amd.model_state import (
            get_deepseek_v4_rocm_atom_state,
        )
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            _top_k_per_row_decode,
            rocm_fp8_paged_mqa_logits,
        )

        parent_prefix = self.prefix.rsplit(".indexer", 1)[0]
        swa_metadata = attn_metadata.get(f"{parent_prefix}.swa_cache")
        atom_state = get_deepseek_v4_rocm_atom_state(swa_metadata)
        if atom_state is None:
            return None

        block_table = atom_state.indexer_decode_block_table
        schedule_metadata = atom_state.indexer_decode_schedule_metadata
        if (
            atom_state.indexer_decode_requires_padding
            or block_table is None
            or schedule_metadata is None
        ):
            return None

        num_decode_tokens = int(atom_state.indexer_decode_num_tokens)
        num_reqs = int(atom_state.num_actual_reqs)
        if (
            num_reqs <= 0
            or num_decode_tokens <= 0
            or num_decode_tokens != num_reqs
            or int(atom_state.num_actual_tokens) != num_decode_tokens
            or hidden_states.shape[0] < num_decode_tokens
        ):
            return None

        q_decode = q_fp8[:num_decode_tokens].reshape(
            num_reqs,
            1,
            self.n_head,
            self.head_dim,
        )
        kv_cache = self.k_cache.kv_cache.unsqueeze(-2)
        n_committed = atom_state.n_committed_csa_per_seq[:num_reqs]

        if _ATOM_INDEXER_FASTPATH_NEEDS_SENTINEL_FILL:
            self.topk_indices_buffer[: hidden_states.shape[0]] = -1
        logits = rocm_fp8_paged_mqa_logits(
            q_decode,
            kv_cache,
            weights[:num_decode_tokens],
            n_committed,
            block_table[:num_reqs],
            schedule_metadata,
            max_model_len=self.max_model_len,
        )
        topk_indices = self.topk_indices_buffer[:num_decode_tokens, : self.topk_tokens]
        _top_k_per_row_decode(
            logits,
            1,
            n_committed,
            topk_indices,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            self.topk_tokens,
        )
        return self.topk_indices_buffer

    def _maybe_atom_indexer_sequence(
        self,
        q: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """ATOM-style explicit indexer Q quant + weight scale sequence.

        The default vLLM path fuses RoPE, FP8 quantization, and weight scaling
        into ``fused_indexer_q_rope_quant``. ATOM keeps the weight scaling as
        an explicit ``scale_indexer_weights`` op before indexer scoring. Keep
        this as an opt-in preview path so the exact op boundary can be
        benchmarked without changing the validated default.
        """
        if not _ATOM_INDEXER_SEQUENCE_ENABLED or self.use_fp4_kv:
            return None

        ops.rotary_embedding(
            positions,
            q,
            None,
            self.head_dim,
            rotary_emb.cos_sin_cache,
            False,
            self.head_dim - self.rope_dim,
            False,
        )
        q_fp8, q_scale = per_token_group_quant_fp8(
            q.view(-1, self.head_dim).contiguous(),
            self.head_dim,
            use_ue8m0=True,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1).contiguous()
        weights = scale_indexer_weights(
            indexer_weights.contiguous(),
            q_scale,
            self.softmax_scale * self.n_head**-0.5,
        )
        return q_fp8, weights

    def indexer_score_topk(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """ATOM-compatible indexer dispatch target.

        ATOM calls ``torch.ops.aiter.indexer_score_topk(q_fp8, weights,
        prefix, topk)`` after the compressor has already written indexer K to
        the cache. vLLM keeps that cache and metadata in this module, so the
        local fallback op dispatches back here.
        """
        if topk != self.topk_tokens:
            raise ValueError(
                f"Unexpected indexer topk {topk}; expected {self.topk_tokens}."
            )
        if self.use_fp4_kv:
            raise RuntimeError("ATOM indexer dispatch is only enabled for FP8 cache.")

        atom_topk = self._maybe_atom_decode_indexer_fastpath(
            q_fp8,
            q_fp8,
            weights,
        )
        if atom_topk is not None:
            return atom_topk
        return self.indexer_op(q_fp8, q_fp8, None, weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        compressor = self.compressor

        def wq_b_and_q_quant():
            # ReplicatedLinear returns (output, bias); bias is None.
            q, _ = self.wq_b(qr)
            q = q.view(-1, self.n_head, self.head_dim)
            atom_sequence = self._maybe_atom_indexer_sequence(
                q,
                indexer_weights,
                positions,
                rotary_emb,
            )
            if atom_sequence is not None:
                return atom_sequence
            return fused_indexer_q_rope_quant(
                positions,
                q,
                rotary_emb.cos_sin_cache,
                indexer_weights,
                self.softmax_scale,
                self.n_head**-0.5,
                use_fp4=self.use_fp4_kv,
            )

        # compressor returns None and writes K to the indexer KV cache; the
        # join orders that write before indexer_op (skip_k_cache_insert=True).
        (q_quant, weights), k = maybe_execute_in_parallel(
            wq_b_and_q_quant,
            lambda: compressor(compressed_kv_score, positions, rotary_emb),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        if _ATOM_INDEXER_DISPATCH_ENABLED and not self.use_fp4_kv:
            return torch.ops.aiter.indexer_score_topk(
                q_quant,
                weights,
                self.prefix,
                self.topk_tokens,
            )
        atom_topk = self._maybe_atom_decode_indexer_fastpath(
            hidden_states,
            q_quant,
            weights,
        )
        if atom_topk is not None:
            return atom_topk
        return self.indexer_op(hidden_states, q_quant, k, weights)
