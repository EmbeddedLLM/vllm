# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm integration coverage for the MoRI UMBP KV-cache tier."""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.utils import RemoteOpenAIServer
from tests.v1.utils import get_prometheus_metrics
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_ports_list

MODEL = "Qwen/Qwen3-0.6B"
RUN_INTEGRATION = os.getenv("RUN_UMBP_INTEGRATION_TEST") == "1"
RUN_TP2_INTEGRATION = os.getenv("RUN_UMBP_TP2_INTEGRATION_TEST") == "1"

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only test"),
    pytest.mark.timeout(600),
]


class MoriRemoteOpenAIServer(RemoteOpenAIServer):
    """Launch vLLM with the pytest interpreter that imports MoRI."""

    def _start_server(
        self,
        model: str,
        vllm_serve_args: list[str],
        env_dict: dict[str, str] | None,
    ) -> None:
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if env_dict is not None:
            env.update(env_dict)
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            model,
            *vllm_serve_args,
        ]
        self.proc = subprocess.Popen(
            command,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True,
        )


def _umbp_master_executable() -> str:
    executable = shutil.which(os.getenv("UMBP_MASTER_BIN", "umbp_master"))
    if executable is not None:
        return executable
    try:
        import mori
        import mori.umbp  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"amd-mori with BUILD_UMBP=ON is required: {exc}")
    candidate = Path(mori.__file__).parent / "umbp_master"
    if not candidate.is_file():
        pytest.fail(f"umbp_master was not found at {candidate}")
    return str(candidate)


def _wait_for_port(port: int, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"umbp_master exited with status {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    pytest.fail("umbp_master did not become ready")


def _metric_total(server: RemoteOpenAIServer, name: str) -> float:
    return sum(get_prometheus_metrics(server).get(name, {}).values())


def _prompt(index: int) -> str:
    body = " ".join(f"word{token}" for token in range(36))
    return f"UMBP GPU integration fixture {index}: {body}"


def _complete(server: RemoteOpenAIServer, prompt: str) -> str:
    response = server.get_client().completions.create(
        model=MODEL,
        prompt=prompt,
        max_tokens=4,
        temperature=0,
    )
    return response.choices[0].text


def _run_mori_restore_test(tensor_parallel_size: int) -> None:
    master_port, metrics_port, io_port, peer_port = get_open_ports_list(4)
    master = subprocess.Popen(
        [
            _umbp_master_executable(),
            f"127.0.0.1:{master_port}",
            str(metrics_port),
        ]
    )
    try:
        _wait_for_port(master_port, master)
        tier_config = {
            "type": "mori",
            "dram_capacity_bytes": 1 << 30,
            "master_address": f"127.0.0.1:{master_port}",
            "node_id": f"vllm-umbp-gpu-integration-tp{tensor_parallel_size}",
            "node_address": "127.0.0.1",
            "io_engine_port": io_port,
            "peer_service_port": peer_port,
            "io_threads": 2,
            "key_prefix": "vllm:gpu-integration:",
        }
        transfer_config = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": 16 << 20,
                "offload_prompt_only": False,
                "secondary_tiers": [tier_config],
            },
        }
        server_args = [
            "--enforce-eager",
            "--enable-prefix-caching",
            "--max-model-len",
            "224",
            "--num-gpu-blocks-override",
            "16",
            "--gpu-memory-utilization",
            "0.2",
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--kv-transfer-config",
            json.dumps(transfer_config),
        ]
        with MoriRemoteOpenAIServer(MODEL, server_args) as server:
            first_output = _complete(server, _prompt(0))
            for index in range(1, 7):
                _complete(server, _prompt(index))

            written = 0.0
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                written = _metric_total(
                    server, "vllm:kv_offload_tiering_write_bytes_total"
                )
                if written > 0:
                    break
                time.sleep(0.5)
            assert written > 0, "no KV blocks were cascaded into UMBP"

            time.sleep(6)
            read_before = _metric_total(
                server, "vllm:kv_offload_tiering_read_bytes_total"
            )
            restored_output = _complete(server, _prompt(0))

            read_after = read_before
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                read_after = _metric_total(
                    server, "vllm:kv_offload_tiering_read_bytes_total"
                )
                if read_after > read_before:
                    break
                time.sleep(0.5)

            assert restored_output == first_output
            assert read_after > read_before, "evicted KV was not restored from UMBP"
            assert _metric_total(server, "vllm:kv_offload_total_bytes_total") > 0
    finally:
        master.terminate()
        try:
            master.wait(timeout=15)
        except subprocess.TimeoutExpired:
            master.kill()
            master.wait(timeout=5)


@pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="set RUN_UMBP_INTEGRATION_TEST=1 to run the real MoRI test",
)
def test_mori_restores_evicted_gpu_kv_from_umbp() -> None:
    """Restore an evicted prefix through UMBP on one GPU."""
    _run_mori_restore_test(tensor_parallel_size=1)


@pytest.mark.skipif(
    not RUN_TP2_INTEGRATION,
    reason="set RUN_UMBP_TP2_INTEGRATION_TEST=1 to run the TP=2 MoRI test",
)
def test_mori_restores_evicted_tp2_kv_from_umbp() -> None:
    """Restore an evicted tensor-parallel prefix through UMBP on two GPUs."""
    _run_mori_restore_test(tensor_parallel_size=2)
