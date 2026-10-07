# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import collections
import glob
import os
import time
from collections.abc import Generator
from copy import copy
from typing import Any

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    runai_safetensors_weights_iterator,
)
from vllm.transformers_utils.s3_utils import glob as s3_glob
from vllm.transformers_utils.utils import is_s3

logger = init_logger(__name__)


class ShardedStateLoader(BaseModelLoader):
    """Model loader that directly loads each worker's model state dict, which
    enables a fast load path for large tensor-parallel models where each worker
    only needs to read its own shard rather than the entire checkpoint. See
    `examples/features/sharded_state/save_sharded_state_offline.py` for creating
    a sharded checkpoint.
    """

    DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else copy(load_config.model_loader_extra_config)
        )
        self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{load_config.model_loader_extra_config.keys()}"
            )

    @staticmethod
    def _filter_subtensors(
        tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        same_storage_groups: dict[Any, list[tuple[str, torch.Tensor]]] = (
            collections.defaultdict(list)
        )
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            # Stride-aware: bit-identical to the previous
            # `tensor.view(-1)[-1].data_ptr() + tensor.element_size()` for
            # contiguous tensors, but does not raise on non-contiguous ones.
            span = sum((s - 1) * st for s, st in zip(tensor.shape, tensor.stride()))
            return tensor.data_ptr() + (span + 1) * tensor.element_size()

        result: dict[str, torch.Tensor] = {}
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 < k:
                        # Same tensors, keep the one with the smaller key.
                        break
                else:
                    result[k] = t
        return result

    def _prepare_weights(self, model_name_or_path: str, revision: str | None):
        if is_s3(model_name_or_path) or os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            allow_patterns = ["*.safetensors"]
            return download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model, model_config.revision)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank

        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        local_model_path = model_weights

        rank = get_tensor_model_parallel_rank()
        pattern = os.path.join(
            local_model_path,
            self.pattern.format(rank=rank, part="*"),
        )

        filepaths = []
        if is_s3(local_model_path):
            file_pattern = f"*{self.pattern.format(rank=rank, part='*')}"
            filepaths = s3_glob(path=local_model_path, allow_pattern=[file_pattern])
        else:
            filepaths = glob.glob(pattern)
        if not filepaths:
            # TODO: support un-sharded checkpoints too
            raise ValueError(
                f"Could not find checkpoint files '{pattern}', only "
                f"pre-sharded checkpoints are currently supported!"
            )
        # Look up against the RAW model state dict — no _filter_subtensors on
        # the load path. Filtering demanded exact key-set equality between the
        # save-time model and this fresh init, which breaks on GLM5Next
        # `_active_layers` alias tensors; copy_ through an alias writes the
        # same storage as its canonical tensor, so raw lookup is safe.
        from safetensors.torch import safe_open

        raw_state_dict = model.state_dict()
        checkpoint_keys: set[str] = set()
        for path in filepaths:
            with safe_open(path, framework="pt") as f:
                checkpoint_keys.update(f.keys())
        missing_from_raw = sorted(checkpoint_keys - raw_state_dict.keys())
        if missing_from_raw:
            raise ValueError(
                f"Checkpoint keys not present in model.state_dict(): "
                f"{len(missing_from_raw)} keys "
                f"(checkpoint_n={len(checkpoint_keys)}, "
                f"raw_n={len(raw_state_dict)}): {missing_from_raw}"
            )
        logger.info(
            "Sharded state load: checkpoint_n=%d raw_n=%d",
            len(checkpoint_keys),
            len(raw_state_dict),
        )
        counter_before_loading_weights = time.perf_counter()
        loaded_keys: set[str] = set()
        for key, tensor in self.iterate_over_files(filepaths):
            # If loading with LoRA enabled, additional padding may
            # be added to certain parameters. We only load into a
            # narrowed view of the parameter data.
            param = raw_state_dict[key]
            param_data = param.data
            param_shape = param.shape
            for dim, size in enumerate(tensor.shape):
                if size < param_shape[dim]:
                    param_data = param_data.narrow(dim, 0, size)
            if tensor.shape != param_shape:
                logger.warning(
                    "loading tensor of shape %s into parameter '%s' of shape %s",
                    tensor.shape,
                    key,
                    param_shape,
                )
            param_data.copy_(tensor)
            loaded_keys.add(key)
        counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "Loading weights took %.2f seconds",
            counter_after_loading_weights - counter_before_loading_weights,
        )

        def byte_range(tensor: torch.Tensor) -> tuple[int, int]:
            # Same stride-aware span formula as `_filter_subtensors`.
            span = sum((s - 1) * st for s, st in zip(tensor.shape, tensor.stride()))
            start = tensor.data_ptr()
            return start, start + (span + 1) * tensor.element_size()

        # Coverage gate: every raw key not loaded from the checkpoint must be
        # byte-covered by a loaded tensor in the same storage group (its
        # canonical alias was loaded); anything else is missing weight data.
        loaded_ranges: dict[Any, list[tuple[int, int]]] = collections.defaultdict(list)
        for key in loaded_keys:
            tensor = raw_state_dict[key]
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                loaded_ranges[tensor.device, ptr].append(byte_range(tensor))

        leftover_covered = 0
        uncovered = []
        for key, tensor in raw_state_dict.items():
            if key in loaded_keys:
                continue
            if not tensor.numel():
                leftover_covered += 1  # zero-byte; no data to lose
                continue
            start, end = byte_range(tensor)
            group = loaded_ranges[(tensor.device, tensor.untyped_storage().data_ptr())]
            if any(k_start <= start and end <= k_end for k_start, k_end in group):
                leftover_covered += 1
            else:
                uncovered.append(key)
        if uncovered:
            raise ValueError(
                f"Sharded state load: {len(uncovered)} model keys not covered "
                f"by loaded checkpoint tensors: {uncovered}"
            )
        logger.info(
            "COVERAGE_OK loaded=%d leftover_covered=%d/total_raw=%d",
            len(loaded_keys),
            leftover_covered,
            len(raw_state_dict),
        )

    def iterate_over_files(
        self, paths
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        if self.load_config.load_format == "runai_streamer_sharded":
            yield from runai_safetensors_weights_iterator(paths, True)
        else:
            from safetensors.torch import safe_open

            for path in paths:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        yield key, tensor

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from safetensors.torch import save_file

        from vllm.distributed import get_tensor_model_parallel_rank

        if pattern is None:
            pattern = ShardedStateLoader.DEFAULT_PATTERN
        rank = get_tensor_model_parallel_rank()
        part_idx = 0
        total_size = 0
        full_state_dict = model.state_dict()
        noncontiguous_keys = [
            key for key, tensor in full_state_dict.items() if not tensor.is_contiguous()
        ]
        logger.info(
            "Sharded state save (rank %d): %d/%d non-contiguous tensors in "
            "model.state_dict(): %s",
            rank,
            len(noncontiguous_keys),
            len(full_state_dict),
            noncontiguous_keys,
        )
        state_dict = ShardedStateLoader._filter_subtensors(full_state_dict)
        dropped_keys = [key for key in full_state_dict if key not in state_dict]
        logger.info(
            "Sharded state save (rank %d): %d keys dropped by "
            "_filter_subtensors (normally storage-covered views): %s",
            rank,
            len(dropped_keys),
            dropped_keys,
        )
        ShardedStateLoader._verify_byte_coverage(full_state_dict, state_dict, rank)
        state_dict_part: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            param_size = tensor.nelement() * tensor.element_size()
            if max_size is not None and total_size + param_size > max_size:
                filename = pattern.format(rank=rank, part=part_idx)
                save_file(
                    state_dict_part,
                    os.path.join(path, filename),
                )
                part_idx += 1
                total_size = 0
                state_dict_part = {}
            # safetensors rejects non-contiguous tensors; byte accounting is
            # unchanged (same numel * element_size).
            state_dict_part[key] = (
                tensor if tensor.is_contiguous() else tensor.contiguous()
            )
            total_size += param_size
        if len(state_dict_part) > 0:
            filename = pattern.format(rank=rank, part=part_idx)
            save_file(
                state_dict_part,
                os.path.join(path, filename),
            )

    @staticmethod
    def _verify_byte_coverage(
        state_dict: dict[str, torch.Tensor],
        kept_state_dict: dict[str, torch.Tensor],
        rank: int | None = None,
    ) -> None:
        """Hard gate: every original tensor's byte range must point into a
        kept tensor in the same storage group, otherwise filtering would
        silently drop data that the saved checkpoint never contains.
        """

        def byte_range(tensor: torch.Tensor) -> tuple[int, int]:
            # Same stride-aware span formula as `_filter_subtensors`.
            span = sum((s - 1) * st for s, st in zip(tensor.shape, tensor.stride()))
            start = tensor.data_ptr()
            return start, start + (span + 1) * tensor.element_size()

        kept_ranges: dict[Any, list[tuple[int, int]]] = collections.defaultdict(list)
        for tensor in kept_state_dict.values():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                kept_ranges[tensor.device, ptr].append(byte_range(tensor))

        uncovered = []
        for key, tensor in state_dict.items():
            if not tensor.numel():
                continue  # zero-byte tensor; no data to lose when skipped
            start, end = byte_range(tensor)
            group = kept_ranges[(tensor.device, tensor.untyped_storage().data_ptr())]
            if not any(k_start <= start and end <= k_end for k_start, k_end in group):
                uncovered.append(key)
        if uncovered:
            message = (
                f"Sharded state save (rank {rank}): COVERAGE_FAIL "
                f"uncovered={len(uncovered)} "
                f"keys: {uncovered}"
            )
            logger.error("%s", message)
            raise RuntimeError(message)
        logger.info(
            "Sharded state save (rank %d): COVERAGE_OK kept=%d/orig=%d",
            rank,
            len(kept_state_dict),
            len(state_dict),
        )
