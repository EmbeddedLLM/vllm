# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


_PACKED_LAYOUT_TERMS = (
    "fp8_ds_mla",
    "packed_fp8_ds_mla",
)
_PACKED_SLOT_TERMS = (
    "584",
    "embedded",
    "compressed_kv_layout",
    "swa_pages",
    "split_kv",
)

_COMPRESSOR_MODULES = (
    "aiter.ops.flydsl.kernels.fused_compress_attn",
    "aiter.ops.flydsl.kernels.fused_compress_attn_hca",
)
_ATTENTION_MODULES = (
    "aiter.ops.pa_sparse_prefill_opus",
    "aiter.ops.triton.attention.pa_decode_sparse",
    "aiter.ops.triton.attention.unified_attention_sparse_mla",
    "aiter.ops.triton.attention.pa_mqa_logits",
)


@dataclass(frozen=True)
class AtomNativeAbiStatus:
    """Detected native ATOM ABI support in the installed aiter package."""

    aiter_available: bool
    packed_fp8_ds_mla_compressor: bool
    packed_fp8_ds_mla_attention: bool
    mhc_fused_post_pre: bool
    maybe_dual_stream_forward: bool
    checked_modules: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def has_full_native_packed_main_path(self) -> bool:
        return self.packed_fp8_ds_mla_compressor and self.packed_fp8_ds_mla_attention

    def missing_summary(self) -> str:
        return ", ".join(self.missing) if self.missing else "<unknown>"

    def checked_modules_summary(self) -> str:
        return ", ".join(self.checked_modules) if self.checked_modules else "<none>"


def _safe_import(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _public_text(module: ModuleType) -> str:
    """Return lowercase public signatures plus module source when available."""
    pieces: list[str] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        try:
            pieces.append(f"{name}{inspect.signature(obj)}")
        except Exception:
            pieces.append(name)
    file_name = getattr(module, "__file__", None)
    if file_name:
        path = Path(file_name)
        if path.suffix == ".py" and path.exists():
            try:
                pieces.append(path.read_text())
            except OSError:
                pass
    return "\n".join(pieces).lower()


def _module_mentions_packed_dsv4_abi(module: ModuleType) -> bool:
    text = _public_text(module)
    return any(term in text for term in _PACKED_LAYOUT_TERMS) or (
        "584" in text
        and any(term in text for term in _PACKED_SLOT_TERMS)
    )


def _any_module_mentions_packed_dsv4_abi(
    module_names: tuple[str, ...],
) -> tuple[bool, tuple[str, ...]]:
    checked: list[str] = []
    for module_name in module_names:
        module = _safe_import(module_name)
        if module is None:
            continue
        checked.append(module_name)
        if _module_mentions_packed_dsv4_abi(module):
            return True, tuple(checked)
    return False, tuple(checked)


def probe_atom_native_abi() -> AtomNativeAbiStatus:
    """Inspect installed aiter for native packed DSV4 ATOM ABI support.

    The current deployed vLLM packed layout is a BF16 SWA prefix plus a
    ``uint8[..., 584]`` compressed ``fp8_ds_mla`` tail. Generic MLA, OPUS
    homogeneous sparse prefill, and indexer logits kernels are useful but do
    not satisfy the native main attention/compressor ABI unless their public
    surface explicitly exposes that packed contract.
    """
    aiter = _safe_import("aiter")
    if aiter is None:
        return AtomNativeAbiStatus(
            aiter_available=False,
            packed_fp8_ds_mla_compressor=False,
            packed_fp8_ds_mla_attention=False,
            mhc_fused_post_pre=False,
            maybe_dual_stream_forward=False,
            checked_modules=(),
            missing=("aiter",),
        )

    has_compressor, compressor_checked = _any_module_mentions_packed_dsv4_abi(
        _COMPRESSOR_MODULES
    )
    has_attention, attention_checked = _any_module_mentions_packed_dsv4_abi(
        _ATTENTION_MODULES
    )
    mhc_fused_post_pre = hasattr(aiter, "mhc_fused_post_pre")
    torch = _safe_import("torch")
    torch_aiter_ops = (
        getattr(getattr(torch, "ops", object()), "aiter", object())
        if torch is not None
        else object()
    )
    maybe_dual_stream_forward = hasattr(
        aiter, "maybe_dual_stream_forward"
    ) or hasattr(torch_aiter_ops, "maybe_dual_stream_forward")

    missing: list[str] = []
    if not has_compressor:
        missing.append("packed_fp8_ds_mla_compressor")
    if not has_attention:
        missing.append("packed_fp8_ds_mla_attention")
    if not mhc_fused_post_pre:
        missing.append("mhc_fused_post_pre")
    if not maybe_dual_stream_forward:
        missing.append("maybe_dual_stream_forward")

    return AtomNativeAbiStatus(
        aiter_available=True,
        packed_fp8_ds_mla_compressor=has_compressor,
        packed_fp8_ds_mla_attention=has_attention,
        mhc_fused_post_pre=mhc_fused_post_pre,
        maybe_dual_stream_forward=maybe_dual_stream_forward,
        checked_modules=compressor_checked + attention_checked,
        missing=tuple(missing),
    )


def require_atom_native_abi() -> AtomNativeAbiStatus:
    """Return native ABI status or raise with an actionable error.

    Use this only on setup paths. The probe imports and inspects aiter modules
    and should not run in per-token hot paths.
    """
    status = probe_atom_native_abi()
    if status.has_full_native_packed_main_path:
        return status
    raise RuntimeError(
        "VLLM_ROCM_DSV4_REQUIRE_NATIVE_ATOM_ABI=1 was set, but the installed "
        "aiter package does not expose the native packed DeepSeek-V4 ATOM "
        "attention/compressor ABI required for fp8_ds_mla. Missing: "
        f"{status.missing_summary()}. Checked modules: "
        f"{status.checked_modules_summary()}."
    )


__all__ = (
    "AtomNativeAbiStatus",
    "probe_atom_native_abi",
    "require_atom_native_abi",
)
