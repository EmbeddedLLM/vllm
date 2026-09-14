# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned storage identities shared by P/D and ordinary offload lookups."""

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class UMBPKeySpace:
    """Compatibility identity, independent of request/engine ID and P/D role.

    ``layout`` must describe the canonical bytes, including cache dtype,
    block geometry, layer order and shard mapping. It is not a backend name.
    Block hashes come from vLLM, including its salt/adapter/multimodal identity;
    callers must not substitute a router's independently computed token hash.
    """

    deployment: str
    model: str
    revision: str
    layout: str
    hash_algorithm: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.deployment,
                self.model,
                self.revision,
                self.layout,
                self.hash_algorithm,
            )
        ):
            raise ValueError("UMBP cache identity fields must be nonempty strings")

    @cached_property
    def prefix(self) -> str:
        fields = (
            self.deployment,
            self.model,
            self.revision,
            self.layout,
            self.hash_algorithm,
        )
        digest = hashlib.sha256(
            json.dumps(fields, ensure_ascii=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"vllm-umbp:v1:{digest}"

    def block_key(self, block_hash: bytes, *, group: int, shard: int) -> str:
        if not isinstance(block_hash, bytes) or not block_hash:
            raise ValueError("A nonempty vLLM block hash is required")
        if any(type(index) is not int or index < 0 for index in (group, shard)):
            raise ValueError("Group and canonical shard indices must be nonnegative")
        return f"{self.prefix}:g{group}:s{shard}:{block_hash.hex()}"
