# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias

import torch
from torch import nn
from torch.nn import Parameter
from typing_extensions import Self, dataclass_transform

from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter

WeightParameterType: TypeAlias = ModelWeightParameter | PackedvLLMParameter
WeightScaleParameterType: TypeAlias = (
    ChannelQuantScaleParameter
    | PerTensorScaleParameter
    | BlockQuantScaleParameter
    | GroupQuantScaleParameter
)
MaybeWeightScaleParameterType: TypeAlias = WeightScaleParameterType | None
MaybeInputScaleParameterType: TypeAlias = PerTensorScaleParameter | None
MaybeZeroPointParameterType: TypeAlias = PackedvLLMParameter | None


@dataclass
class LinearParamsMetadata:
    """Non-tensor metadata for linear params.

    Holds layer-level information kernels need at process- or apply-time
    (partition sizes, logical widths, dtype, block-quant shape, etc.) that is
    not an ``nn.Parameter``. Subclasses extend with scheme-specific fields.

    Sourced from layer attributes set in ``create_weights``; built and stashed
    on the layer by ``LinearParamsBase.build_metadata``.
    """

    input_size_per_partition: int
    output_size_per_partition: int
    logical_widths: list[int]
    orig_dtype: torch.dtype


def field_validator(field_name: str):
    """Mark a method as the validator for ``field_name``. Called with
    ``(value,)`` from ``__post_init__``; ``LinearParamsBase`` re-raises any
    ``AssertionError`` as ``ClassName.field: <reason>``.
    """

    def decorate(fn):
        fn.__linear_params_validator_for__ = field_name
        # Static — validators are stateless `(value) -> None` checks.
        return staticmethod(fn)

    return decorate


def model_validator(fn):
    """Mark a method as a cross-field validator. Called with ``(self,)``
    from ``__post_init__`` after every field validator has run.
    """
    fn.__linear_params_validator_for__ = None
    return fn


@dataclass_transform(field_specifiers=(field,))
class _LinearParamsMeta(type):
    """Metaclass for ``LinearParamsBase``. At class-definition time, applies
    ``@dataclass`` to ``cls``, materializes ``cls.__field_names__`` from
    ``dataclasses.fields(cls)`` (skipping ``ClassVar``s, kept as a plain
    tuple for torch.compile-friendly hot paths), and runs the
    ``@field_validator`` typo guard. Validator *collection* happens in
    ``LinearParamsBase.__init_subclass__`` (which runs first).
    """

    def __init__(cls, name, bases, ns, **kwargs):
        super().__init__(name, bases, ns, **kwargs)
        dataclass(cls)  # type: ignore[arg-type]
        cls.__field_names__ = tuple(f.name for f in dataclasses.fields(cls))  # type: ignore[arg-type]
        validators = getattr(cls, "__validators__", [])
        declared = set(cls.__field_names__)
        unknown = {t for t, _ in validators if t is not None} - declared
        assert not unknown, (
            f"{cls.__name__} declares @field_validator for unknown "
            f"field(s): {unknown}; declared fields are {declared}"
        )


class LinearParamsBase(metaclass=_LinearParamsMeta):
    """Pydantic-style base for canonical linear-param dataclasses.

    Subclasses declare typed fields and attach a ``@field_validator(name)``
    per field; cross-field invariants go on a method decorated with
    ``@model_validator``. ``__post_init__`` runs every collected validator
    exactly once, at construction. In-place mutation does not re-trigger
    validation — matches the kernel post-processing flow where weights are
    mutated to a kernel-specific format after canonicalization.

    Subclasses do **not** need ``@dataclass`` — ``_LinearParamsMeta`` applies
    it and populates ``__field_names__`` automatically.

    The ``metadata`` field is special: not an ``nn.Parameter`` and not
    validated. Subclasses point ``__metadata_cls__`` at a quant-specific
    metadata dataclass; ``register_params_in_layer`` builds an instance
    from layer attributes and stashes it as ``layer.metadata``.
    """

    __field_names__: ClassVar[tuple[str, ...]] = ()
    __validators__: ClassVar[list[tuple[str | None, Callable]]] = []

    # Override in subclasses to use a quant-specific metadata dataclass.
    __metadata_cls__: ClassVar[type[LinearParamsMetadata]] = LinearParamsMetadata
    metadata: LinearParamsMetadata = field(kw_only=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        validators: list[tuple[str | None, Callable]] = []
        for klass in reversed(cls.__mro__):
            for attr in vars(klass).values():
                raw = attr.__func__ if isinstance(attr, staticmethod) else attr
                target = getattr(
                    raw, "__linear_params_validator_for__", dataclasses.MISSING
                )
                if target is not dataclasses.MISSING:
                    validators.append((target, raw))
        cls.__validators__ = validators

    def __post_init__(self) -> None:
        cls = type(self)
        for target, v in cls.__validators__:
            # target=None -> model validator, called with self.
            # target=field -> field validator, skipped when field is None.
            if target is None:
                v(self)
                continue
            value = getattr(self, target)
            if value is None:
                continue
            try:
                v(value)
            except AssertionError as e:
                raise AssertionError(f"{cls.__name__}.{target}: {e}") from e

    @classmethod
    def register_params_in_layer(
        cls, layer: nn.Module, **parameters: nn.Parameter | None
    ) -> None:
        expected = set(cls.__field_names__) - {"metadata"}
        unknown = set(parameters) - expected
        assert not unknown, (
            f"Unknown parameter(s) {unknown}; expected subset of {expected}"
        )
        for name, param in parameters.items():
            if param is not None:
                layer.register_parameter(name, param)
        cls.build_metadata(layer)

    @classmethod
    def build_metadata(cls, layer: nn.Module) -> None:
        """Build ``__metadata_cls__`` from ``layer.*`` attributes and stash
        as ``layer.metadata``. Missing fields fall back to dataclass defaults.
        """
        md_cls = cls.__metadata_cls__
        kwargs = {
            f.name: getattr(layer, f.name)
            for f in dataclasses.fields(md_cls)
            if hasattr(layer, f.name)
        }
        layer.metadata = md_cls(**kwargs)

    @classmethod
    def read_params_from_layer(cls, layer: nn.Module) -> Self:
        # Typed view over `layer.<field>`; skips __init__/validators.
        # Iterates the precomputed __field_names__ tuple so the loop is
        # straight-line under torch.compile (no dataclasses.fields call
        # on the apply_weights hot path). Construct cls(**kwargs) directly
        # when validation is needed.
        obj = object.__new__(cls)
        for name in cls.__field_names__:
            setattr(obj, name, getattr(layer, name, None))
        return obj

    def evolve_and_verify(self, **changes) -> Self:
        """Return a validated copy with ``changes`` applied to fields."""
        return dataclasses.replace(self, **changes)

    def update_params_in_layer(self, layer: nn.Module) -> None:
        # `metadata` is not an nn.Parameter and is owned by create_weights /
        # build_metadata, not by the canonicalize -> kernel post-load flow.
        # Skip it so replace_parameter is never handed a non-tensor value.
        for name in type(self).__field_names__:
            if name == "metadata":
                continue
            replace_parameter(layer, name, getattr(self, name))


@dataclass
class Fp8LinearParamsMetadata(LinearParamsMetadata):
    """FP8-specific non-tensor metadata."""

    weight_block_size: list[int] | None = None
    is_bmm: bool = False
    bmm_batch_size: int = 0


class Fp8LinearParams(LinearParamsBase):
    """Canonical FP8 linear weight representation.

    Exactly one of ``weight_scale`` (per-tensor / per-channel / per-token)
    or ``weight_scale_inv`` (block-scaled) is populated, never both.
    ``convert_to_canonical`` in ``Fp8LinearMethod`` transposes the loaded
    weight to ``(K, N)`` for the per-tensor/-channel/-token paths; the
    block-scaled path keeps the loaded ``(N, K)``.

    Following vLLM's ``create_weights`` convention:
    ``N = metadata.output_size_per_partition``,
    ``K = metadata.input_size_per_partition``.

    Canonical shapes per path:

    * per-tensor / per-channel / per-token:
        - weight: ``(K, N)``
        - weight_scale:
            * per-tensor: ``(1,)`` or ``(N,)`` scalar-like
            * per-channel / per-token: ``(N,)`` or ``(1, N)``
        - weight_scale_inv: ``None``

      "per-token" applies on the activation; the weight stays per-channel.

    * block-scaled:
        - weight: ``(N, K)``
        - weight_scale: ``None``
        - weight_scale_inv: ``(ceil(N/bn), ceil(K/bk))``
    """

    __metadata_cls__: ClassVar[type[LinearParamsMetadata]] = Fp8LinearParamsMetadata
    metadata: Fp8LinearParamsMetadata = field(kw_only=True)

    weight: WeightParameterType
    weight_scale: MaybeWeightScaleParameterType = None
    weight_scale_inv: MaybeWeightScaleParameterType = None
    input_scale: MaybeInputScaleParameterType = None
    input_scale_ub: Parameter | None = None
    bias: Parameter | None = None

    # Required by Marlin.
    workspace: Parameter | None = None

    @field_validator("weight")
    def _check_weight(t: WeightParameterType) -> None:
        assert t.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz), (
            f"dtype must be float8_e4m3fn or float8_e4m3fnuz, got {t.dtype}"
        )
        assert t.data.ndim == 2, f"must be 2D, got ndim={t.data.ndim}"

    @field_validator("weight_scale")
    def _check_weight_scale(t: WeightScaleParameterType) -> None:
        # Per-tensor / per-channel / per-token path.

        fp32_min = torch.finfo(torch.float32).min
        assert t.dtype == torch.float32, f"dtype must be float32, got {t.dtype}"
        assert t.data.numel() > 0, "must be non-empty"
        assert not torch.any(t.data == fp32_min), (
            "still contains the fp32-min sentinel; weight loading or "
            "requantize_with_max_scale never ran"
        )
        assert torch.all(t.data > 0), (
            "must be strictly positive (after requantize_with_max_scale)"
        )

    @field_validator("weight_scale_inv")
    def _check_weight_scale_inv(t: WeightScaleParameterType) -> None:
        # Block-scaled path. dtype is float32 or e8m0 (when is_scale_e8m0).
        # Shape is 2D: [ceil(N/block_n), ceil(K/block_k)].
        assert t.dtype in (torch.float32, torch.float8_e8m0fnu), (
            "dtype must be float32 or float8_e8m0fnu (block-scaled path), "
            f"got {t.dtype}"
        )
        assert t.data.ndim == 2, (
            f"must be 2D [ceil(N/block_n), ceil(K/block_k)], got ndim={t.data.ndim}"
        )
        if t.dtype == torch.float32:
            fp32_min = torch.finfo(torch.float32).min
            assert not torch.any(t.data == fp32_min), (
                "still contains the fp32-min sentinel; weight loading never ran"
            )
            assert torch.all(t.data > 0), "must be strictly positive"

    @field_validator("input_scale")
    def _check_input_scale(t: PerTensorScaleParameter) -> None:
        # Only present for static activation quant. Per-shard scales are
        # collapsed via .max() into a scalar in process_weights_after_loading.
        fp32_min = torch.finfo(torch.float32).min
        assert t.dtype == torch.float32, f"dtype must be float32, got {t.dtype}"
        assert t.data.numel() == 1, (
            "must be a scalar (per-shard scales collapsed via .max() in "
            f"process_weights_after_loading), got numel={t.data.numel()}"
        )
        assert not torch.any(t.data == fp32_min), (
            "still contains the fp32-min sentinel; static input scale was never loaded"
        )
        assert torch.all(t.data > 0), "must be strictly positive"

    @field_validator("input_scale_ub")
    def _check_input_scale_ub(t: Parameter) -> None:
        # Upper bound on the dynamic input scale (FBGEMM-FP8). Sourced
        # from the quant config, not the checkpoint, so no fp32-min
        # sentinel is possible. Scalar float32, strictly positive.
        assert t.dtype == torch.float32, f"dtype must be float32, got {t.dtype}"
        assert t.data.numel() == 1, (
            f"must be a scalar upper bound, got numel={t.data.numel()}"
        )
        assert torch.all(t.data > 0), "must be strictly positive"

    @model_validator
    def _exactly_one_scale_path(self) -> None:
        ws_set = self.weight_scale is not None
        wsi_set = self.weight_scale_inv is not None
        assert ws_set != wsi_set, (
            f"{type(self).__name__} must have exactly one of "
            "`weight_scale` (per-tensor / per-channel / per-token path) or "
            "`weight_scale_inv` (block-scaled path) set; got "
            f"weight_scale={ws_set}, weight_scale_inv={wsi_set}"
        )

    @model_validator
    def _weight_shape_matches_partition(self) -> None:
        n = self.metadata.output_size_per_partition
        k = self.metadata.input_size_per_partition
        is_block = self.metadata.weight_block_size is not None
        expected = (n, k) if is_block else (k, n)
        assert tuple(self.weight.shape) == expected, (
            f"weight shape {tuple(self.weight.shape)} does not match "
            f"canonical {expected} for "
            f"{'block-scaled' if is_block else 'per-tensor/per-channel/per-token'}"
            f" path (N={n}, K={k})"
        )
