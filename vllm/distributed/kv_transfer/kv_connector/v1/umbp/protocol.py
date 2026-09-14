# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned P/D handle; a handle is not a claim that its data is ready."""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HandoffHandle:
    namespace: str
    producer_epoch: str
    nonce: str
    token_boundary: int
    boundary_hash: str
    expires_at_ms: int
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported UMBP handoff version")
        if (
            not isinstance(self.namespace, str)
            or not self.namespace.startswith("vllm-umbp:v1:")
            or len(self.namespace) != 77
        ):
            raise ValueError("Invalid UMBP handoff namespace")
        for value, length in (
            (self.namespace[13:], 64),
            (self.producer_epoch, 64),
            (self.nonce, 32),
        ):
            if (
                not isinstance(value, str)
                or len(value) != length
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError("Invalid handoff identity digest")
        if (
            not isinstance(self.boundary_hash, str)
            or not 2 <= len(self.boundary_hash) <= 256
            or len(self.boundary_hash) % 2
            or any(char not in "0123456789abcdef" for char in self.boundary_hash)
        ):
            raise ValueError("Invalid handoff boundary hash")
        if any(
            type(value) is not int or not 0 < value < (1 << 63)
            for value in (self.token_boundary, self.expires_at_ms)
        ):
            raise ValueError("Handoff boundary and expiry must be positive integers")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> "HandoffHandle":
        if not isinstance(value, dict):
            raise ValueError("UMBP handoff must be an object")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError("Invalid UMBP handoff fields") from error

    def ready_key(self, rank: int) -> str:
        if type(rank) is not int or rank < 0:
            raise ValueError("Readiness requires a nonnegative shard index")
        identity = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(identity.encode()).hexdigest()
        return f"{self.namespace}:pd-ready:{digest}:s{rank}"
