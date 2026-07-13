# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    save_partial_states,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


@functools.cache
def _get_atom_runtime_helpers() -> tuple[Any, Any, Any]:
    from vllm.models.deepseek_v4.amd.model_state import (
        get_deepseek_v4_rocm_atom_state,
    )
    from vllm.models.deepseek_v4.amd.v4_kernels import (
        fused_compress_attn,
        update_compressor_states,
    )

    return (
        get_deepseek_v4_rocm_atom_state,
        fused_compress_attn,
        update_compressor_states,
    )


def _validate_atom_packed_fp8_kv_cache(
    kv_cache: torch.Tensor | None,
    atom_kv_scales: torch.Tensor | None,
) -> None:
    if kv_cache is None:
        raise RuntimeError(
            "ATOM packed FP8 compressor requested but atom_kv_cache is not bound."
        )
    if atom_kv_scales is not None:
        raise RuntimeError("ATOM packed fp8_ds_mla tail has embedded UE8M0 scales.")
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.dim() != 3
        or kv_cache.shape[-1] != 584
    ):
        raise RuntimeError(
            "ATOM packed FP8 compressor expects uint8 "
            f"[num_blocks, k_per_block, 584], got dtype={kv_cache.dtype}, "
            f"shape={tuple(kv_cache.shape)}."
        )


def _atom_attention_enabled_for_ratio(ratio: int) -> bool:
    return current_platform.is_rocm() and ratio > 1


def _atom_attention_enabled_for_layer(prefix: str) -> bool:
    return True


class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]
    query_start_loc: torch.Tensor | None = None  # [num_reqs + 1]


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size

        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
            query_start_loc=common_attn_metadata.query_start_loc,
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # fp8_ds_mla is the UE8M0 paged layout and needs 576B alignment. Plain
        # full-cache rows share state pages with contiguous KV pages, so padding
        # would break page matching.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    """DeepSeek V4 KV/score compressor.

    Owns the linear / norm / state-cache / ape state and the shared forward
    prologue (kv/score split, save_partial_states launch). The
    compress → norm → RoPE → store step is dispatched to a triton kernel
    (``compress_norm_rope_store_triton``) by default, except for the NVIDIA
    head_dim=128 indexer path which uses the cutedsl kernel
    (``compress_norm_rope_store_cutedsl``) for better performance.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache
        try:
            self._atom_layer_id = extract_layer_index(prefix)
        except ValueError:
            self._atom_layer_id = None

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )
        self._atom_hca_flat_block_table: torch.Tensor | None = None

        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

    def _atom_rotary_cos_sin(
        self,
        rotary_emb,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        half_rope = self.rope_head_dim // 2
        cos_sin_cache = rotary_emb.cos_sin_cache
        cache_key = (
            cos_sin_cache.data_ptr(),
            cos_sin_cache.device,
            cos_sin_cache.dtype,
            tuple(cos_sin_cache.shape),
            half_rope,
            dtype,
        )
        cached = getattr(self, "_atom_split_cos_sin_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1], cached[2]

        if cos_sin_cache.dtype == dtype:
            cos_cache = cos_sin_cache[..., :half_rope]
            sin_cache = cos_sin_cache[..., half_rope : 2 * half_rope]
        else:
            cos_cache = cos_sin_cache[..., :half_rope].to(dtype=dtype).contiguous()
            sin_cache = (
                cos_sin_cache[..., half_rope : 2 * half_rope]
                .to(dtype=dtype)
                .contiguous()
            )
        self._atom_split_cos_sin_cache = (cache_key, cos_cache, sin_cache)
        return cos_cache, sin_cache

    def _maybe_atom_main_compressor_forward(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,
        attn_metadata: dict[str, Any],
    ) -> bool:
        if not _atom_attention_enabled_for_ratio(
            self.compress_ratio
        ) or not _atom_attention_enabled_for_layer(self.prefix):
            # The ATOM compressor writes the ROCm unified KV cache. vLLM's
            # native sparse attention reads the original vLLM KV cache, so this
            # path is only valid when the paired ATOM attention reader is on.
            return False
        if not current_platform.is_rocm() or self.head_dim not in (128, 512):
            return False
        if self.head_dim == 128:
            # The ATOM fused-compress path added here matches the FP8 indexer
            # cache contract. Keep it separately gated until the large-batch
            # lmeval OOM/JIT pressure is resolved; MXFP4 keeps using vLLM's
            # native writer.
            return False

        (
            get_deepseek_v4_rocm_atom_state,
            fused_compress_attn,
            update_compressor_states,
        ) = _get_atom_runtime_helpers()

        state_metadata = attn_metadata.get(self.state_cache.prefix)
        atom_state = get_deepseek_v4_rocm_atom_state(state_metadata)
        if atom_state is None or atom_state.compress_plans is None:
            raise RuntimeError(
                "ROCm DeepSeek-V4 requires ModelState compressor plans for "
                "the ATOM main compressor."
            )
        plan = atom_state.compress_plans.get(self.compress_ratio)
        if plan is None:
            raise RuntimeError(
                "ATOM main compressor requested but no CompressPlan exists "
                f"for compress_ratio={self.compress_ratio}."
            )

        kv_state = getattr(self, "atom_kv_state", None)
        score_state = getattr(self, "atom_score_state", None)
        if kv_state is None or score_state is None:
            raise RuntimeError(
                "ATOM compressor requested but atom_kv_state or "
                "atom_score_state is not bound."
            )

        kv_cache = getattr(self, "atom_kv_cache", None)

        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        block_tables = None
        kv_slot_mapping = None
        cache_scale = None
        atom_kv_scales = getattr(self, "atom_kv_scales", None)
        atom_kv_layout = getattr(self, "atom_kv_layout", "dense")
        packed_fp8_ds_mla = atom_kv_layout == "fp8_ds_mla"
        mixed_tail_quant = False
        quant = self.head_dim == 128 or mixed_tail_quant
        quant_group_size = None
        preshuffle = True
        use_ue8m0 = True
        fp8_max = None
        if packed_fp8_ds_mla:
            _validate_atom_packed_fp8_kv_cache(kv_cache, atom_kv_scales)
            block_tables = k_cache_metadata.block_table
            k_per_block = k_cache_metadata.block_size // self.compress_ratio
            fp8_max = 448.0
        elif self.head_dim == 128:
            k_cache_layer = self._static_forward_context[self.k_cache_prefix]
            raw_kv_cache = k_cache_layer.kv_cache
            if raw_kv_cache.dtype != torch.uint8:
                return False
            minimal_block_table = getattr(k_cache_metadata, "block_table", None)
            minimal_block_size = int(getattr(k_cache_metadata, "block_size", 0) or 0)
            raw_k_per_block = int(raw_kv_cache.shape[1])
            if (
                minimal_block_table is not None
                and minimal_block_size == raw_k_per_block
                and minimal_block_table.is_contiguous()
            ):
                # Pure-decode ROCm ModelState metadata can provide the
                # compressed indexer block table directly. Prefer it over
                # direct slot mapping so fused_compress_attn may dispatch to
                # the aiter/flydsl fused compressor, whose public wrapper only
                # supports block-table scatter.
                block_tables = minimal_block_table
                k_per_block = minimal_block_size
            else:
                kv_slot_mapping = k_cache_metadata.slot_mapping
                k_per_block = raw_k_per_block
            flat_cache = raw_kv_cache.view(raw_kv_cache.shape[0], -1)
            value_elems = k_per_block * self.head_dim
            fp8_dtype = current_platform.fp8_dtype()
            kv_cache = (
                flat_cache[:, :value_elems]
                .view(fp8_dtype)
                .view(
                    raw_kv_cache.shape[0],
                    k_per_block,
                    self.head_dim,
                )
            )
            cache_scale = (
                flat_cache[:, value_elems:]
                .view(torch.float32)
                .view(
                    raw_kv_cache.shape[0],
                    k_per_block,
                )
            )
            fp8_fnuz = getattr(torch, "float8_e4m3fnuz", None)
            fp8_max = 224.0 if fp8_dtype == fp8_fnuz else 448.0
        elif mixed_tail_quant:
            block_tables = k_cache_metadata.block_table
            k_per_block = k_cache_metadata.block_size // self.compress_ratio
            cache_scale = atom_kv_scales
            fp8_max = 224.0
            quant_group_size = 64
            preshuffle = False
            # The sidecar is fp32, so keep raw amax/fp8 scales for the first
            # correctness pass instead of UE8M0 rounding used by indexer FP8.
            use_ue8m0 = False
        else:
            if kv_cache is None:
                raise RuntimeError(
                    "ATOM main compressor requested but atom_kv_cache is not bound."
                )
            if kv_cache.dtype == torch.float8_e4m3fnuz:
                raise RuntimeError(
                    "ATOM mixed main compressor found an FP8 compressed tail "
                    "but no bound fp32 scale sidecar. Check "
                    "DeepseekV4Attention.post_bind_kv_cache() and "
                    "ModelState split-KV binding."
                )
            block_tables = k_cache_metadata.block_table
            k_per_block = k_cache_metadata.block_size // self.compress_ratio
        kv_atom = kv if kv.dtype == torch.bfloat16 else kv.to(torch.bfloat16)
        score_atom = (
            score if score.dtype == torch.bfloat16 else score.to(torch.bfloat16)
        )
        cos_cache, sin_cache = self._atom_rotary_cos_sin(rotary_emb, kv_atom.dtype)
        fused_compress_attn(
            kv_in=kv_atom,
            score_in=score_atom,
            kv_state=kv_state,
            score_state=score_state,
            plan=plan,
            state_slot_mapping=atom_state.state_slot_mapping,
            ape=self.ape,
            rms_weight=self.norm.weight,
            rms_eps=self.rms_norm_eps,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            kv_cache=kv_cache,
            block_tables=block_tables,
            k_per_block=k_per_block,
            overlap=self.overlap,
            ratio=self.compress_ratio,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            quant=quant,
            cache_scale=cache_scale,
            use_ue8m0=use_ue8m0,
            quant_group_size=quant_group_size,
            preshuffle=preshuffle,
            fp8_max=fp8_max,
            kv_slot_mapping=kv_slot_mapping,
            packed_fp8_ds_mla=packed_fp8_ds_mla,
        )
        update_compressor_states(
            kv_atom,
            score_atom,
            self.ape,
            kv_state,
            score_state,
            write_plan=plan.write_plan_gpu,
            num_write=plan.num_write,
            state_slot_mapping=atom_state.state_slot_mapping,
            ratio=self.compress_ratio,
            overlap=self.overlap,
        )
        swa_metadata = attn_metadata.get(f"{self.k_cache_prefix}.swa_cache")
        num_prefills = int(getattr(swa_metadata, "num_prefills", 0) or 0)
        if num_prefills > 0:
            return True
        return True

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        if self._maybe_atom_main_compressor_forward(
            kv,
            score,
            positions,
            rotary_emb,
            attn_metadata,
        ):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        pdl_kwargs = (
            {}
            if current_platform.is_rocm() or current_platform.is_xpu()
            else {"launch_pdl": False}
        )

        # Store the KV and score (with fused APE addition) in the state.
        # NOTE: PDL is disabled — both this kernel and the compress kernels
        # below depend on preceding kernel outputs (kv/score from the cublas
        # GEMM; state_cache from this kernel) but neither emits/waits on PDL
        # grid dependency primitives, so launch_pdl=True caused a
        # read-after-write race and non-deterministic output.
        save_partial_states(
            kv=kv,
            score=score,
            ape=self.ape,
            positions=positions,
            state_cache=state_cache,
            slot_mapping=slot_mapping,
            block_size=block_size,
            state_width=state_width,
            compress_ratio=self.compress_ratio,
            pdl_kwargs=pdl_kwargs,
        )

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        # Plain-row V4 reads a contiguous bf16 / per-tensor fp8 cache row; the
        # fp8_ds_mla path uses the UE8M0 paged uint8 layout.
        store_full_kv = self.head_dim == 512 and kv_cache.dtype != torch.uint8
        store_full_fp8 = kv_cache.dtype == torch.float8_e4m3fn
        fp8_scale = (
            getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
            if store_full_fp8
            else None
        )

        # cutedsl (head=512) accepts the full-cache flags; triton (indexer/AMD)
        # does not, so the two callables have different signatures.
        compress_norm_rope_store_fn: Any
        if current_platform.is_cuda() and self.head_dim == 512:
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
                compress_norm_rope_store_cutedsl,
            )

            # head=512 on CUDA always uses cutedsl, for both the fp8_ds_mla
            # layout and the plain full-cache layout. The full-cache flags
            # are consumed only here.
            compress_norm_rope_store_fn = compress_norm_rope_store_cutedsl
            extra_kwargs: dict[str, Any] = dict(
                store_full_kv=store_full_kv,
                store_full_fp8=store_full_fp8,
                fp8_scale=fp8_scale,
            )
        else:
            # Indexer path (head_dim == 128) or non-CUDA GPUs (AMD, XPU, etc.).
            compress_norm_rope_store_fn = compress_norm_rope_store_triton
            extra_kwargs = {}

        compress_norm_rope_store_fn(
            state_cache=state_cache,
            num_actual=num_actual,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            pdl_kwargs=pdl_kwargs,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            use_fp4_cache=self.use_fp4_cache,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
            token_stride=self._token_stride,
            scale_dim=self._scale_dim,
            **extra_kwargs,
        )
