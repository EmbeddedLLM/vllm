# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

from benchmarks.kv_offload import benchmark_umbp_vs_cpu as benchmark


def test_mori_metrics_extracts_successful_ssd_reads(monkeypatch):
    response = Mock()
    response.text = """
mori_umbp_ssd_read_total{node="a",status="ok"} 2
mori_umbp_ssd_read_total{node="a",status="error"} 7
mori_umbp_ssd_read_total{node="b",status="ok"} 3
mori_umbp_ssd_read_bytes_total{node="a"} 4096
mori_umbp_ssd_read_bytes_total{node="b"} 8192
"""
    monkeypatch.setattr(benchmark.requests, "get", Mock(return_value=response))

    metrics = benchmark._mori_metrics("http://master:9091/metrics")

    assert metrics == {"ssd_reads": 5, "ssd_read_bytes": 12288}
    response.raise_for_status.assert_called_once_with()


def test_wait_for_mori_ssd_read_rejects_unproven_path(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_mori_metrics",
        Mock(return_value={"ssd_reads": 1, "ssd_read_bytes": 4096}),
    )
    times = iter((0.0, 1.0))
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(times))

    try:
        benchmark._wait_for_mori_ssd_read(
            "http://master:9091/metrics",
            {"ssd_reads": 1, "ssd_read_bytes": 4096},
            timeout=0.5,
        )
    except RuntimeError as exc:
        assert "did not prove" in str(exc)
    else:
        raise AssertionError("missing SSD evidence must fail")
