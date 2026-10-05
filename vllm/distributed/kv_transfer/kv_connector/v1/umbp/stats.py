# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared UMBP connector transfer statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)


@dataclass
class UMBPStoreConnectorStats(KVConnectorStats):
    """Serializable interval counters shared by every UMBP runtime."""

    data: dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        operation: str,
        *,
        submitted: int = 0,
        completed: int = 0,
        failed: int = 0,
        num_bytes: int = 0,
        duration_seconds: float | None = None,
    ) -> None:
        entry = self.data.setdefault(
            operation,
            {
                "submitted": 0,
                "completed": 0,
                "failed": 0,
                "num_bytes": 0,
                "duration_seconds": [],
            },
        )
        entry["submitted"] += submitted
        entry["completed"] += completed
        entry["failed"] += failed
        entry["num_bytes"] += num_bytes
        if duration_seconds is not None:
            entry.setdefault("duration_seconds", []).append(duration_seconds)

    def reset(self) -> None:
        self.data.clear()

    def aggregate(self, other: KVConnectorStats) -> UMBPStoreConnectorStats:
        if not isinstance(other, UMBPStoreConnectorStats):
            raise TypeError("cannot aggregate incompatible UMBP stats")
        result = UMBPStoreConnectorStats()
        for operation, values in [*self.data.items(), *other.data.items()]:
            result.record(
                operation,
                submitted=values.get("submitted", 0),
                completed=values.get("completed", 0),
                failed=values.get("failed", 0),
                num_bytes=values.get("num_bytes", 0),
            )
            result.data[operation]["duration_seconds"].extend(
                values.get("duration_seconds", ())
            )
        return result

    def reduce(self) -> dict[str, int | float]:
        result: dict[str, int | float] = {}
        for operation, values in self.data.items():
            for key, value in values.items():
                if key == "duration_seconds":
                    if value:
                        result[f"{operation}_duration_avg_ms"] = round(
                            fmean(value) * 1e3, 3
                        )
                    continue
                result[f"{operation}_{key}"] = value
        return result

    def is_empty(self) -> bool:
        return not self.data


class UMBPStorePromMetrics(KVConnectorPromMetrics):
    """Prometheus counters for shared UMBP transfer outcomes."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )
        labels = labelnames + ["operation"]
        self._submitted = self._counter_cls(
            name="vllm:umbp_transfer_submitted_total",
            documentation="Number of UMBP objects submitted.",
            labelnames=labels,
        )
        self._completed = self._counter_cls(
            name="vllm:umbp_transfer_completed_total",
            documentation="Number of UMBP objects completed.",
            labelnames=labels,
        )
        self._failed = self._counter_cls(
            name="vllm:umbp_transfer_failed_total",
            documentation="Number of UMBP objects failed.",
            labelnames=labels,
        )
        self._bytes = self._counter_cls(
            name="vllm:umbp_transfer_bytes_total",
            documentation="Bytes in successfully completed UMBP transfer ranges.",
            labelnames=labels,
        )
        self._duration = self._histogram_cls(
            name="vllm:umbp_transfer_duration_seconds",
            documentation=(
                "End-to-end duration of a UMBP transfer job, including queueing."
            ),
            buckets=[
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
                30.0,
                60.0,
            ],
            labelnames=labels + ["tier"],
        )

    def observe(
        self,
        transfer_stats_data: dict[str, Any] | None,
        engine_idx: int = 0,
    ) -> None:
        if not transfer_stats_data:
            return
        for operation, values in transfer_stats_data.items():
            labels = self.per_engine_labelvalues[engine_idx] + [operation]
            self._submitted.labels(*labels).inc(values.get("submitted", 0))
            self._completed.labels(*labels).inc(values.get("completed", 0))
            self._failed.labels(*labels).inc(values.get("failed", 0))
            self._bytes.labels(*labels).inc(values.get("num_bytes", 0))
            duration_labels = labels + ["unknown"]
            duration_metric = self._duration.labels(*duration_labels)
            for duration in values.get("duration_seconds", ()):
                duration_metric.observe(duration)
