# Reproducing the Kimi-K3 UMBP SSD validation

This document reproduces the passing Kimi-K3 validation completed on
2026-09-01. It covers the exact source and model revisions, runtime build,
container launch, correctness and effectiveness checks, evidence capture, and
zero-residue cleanup.

The validated topology is one vLLM server using all eight MI355X GPUs on
`crs-m2m-ext-vllm-003.us-east2-a.compute.internal`. vLLM uses its
`OffloadingConnector`, a 2 GiB shared CPU primary tier, a 2 GiB MoRI DRAM tier,
and a 64 GiB file-backed MoRI SSD tier on the local ext4 NVMe filesystem. This
test does not run llm-d or llm-d-router; those repositories are recorded below
for provenance but are outside the request and KV data path.

## Immutable provenance

Use credential-free repository URLs. Never put a GitHub token in a remote URL,
command transcript, or validation artifact.

| Component | Sanitized repository or artifact | Branch | Revision | Used by this validation |
| --- | --- | --- | --- | --- |
| vLLM UMBP validation base | `https://github.com/EmbeddedLLM/vllm.git` | `umbp` | `7aaa9957356b6b87b257e49dca36b5ad35994f64` | Yes |
| vLLM nightly source and image | `vllm/vllm-openai-rocm` | nightly tag | `1dc464d42681d22f38caf1fdc1eb632dc4421c45` | Yes |
| Nightly plus UMBP merge tree | synthetic Git tree | n/a | `d9faeffe5744f7ec540414612667fc6984d5add1` | Yes |
| Direct-GPU prefetch fix | `https://github.com/EmbeddedLLM/vllm.git` | `umbp` | `1cf7681d1e099af8ce383e7b270f51424669d7dd` | Yes |
| MoRI | `https://github.com/ROCm/mori.git` | `umbp-physical-placement` | `6bacfcc952a490f8782998a17b2ddf6b4878641d` | Yes |
| MoRI `msgpack-c` submodule | `https://github.com/msgpack/msgpack-c.git` | pinned | `9b801f087ab7434f2ab1ab3c0f48a966c19d3b70` | Yes |
| MoRI `spdlog` submodule | `https://github.com/gabime/spdlog.git` | pinned | `4a9ccf7e38e257feecce0c579a782741254eaeef` | Yes |
| MoRI `spdk` submodule pointer | `https://github.com/spdk/spdk.git` | pinned, not initialized | `2ef883ef96e79c3cc16da02f667a7a58c2453f2f` | No; `USE_SPDK=OFF` |
| llm-d-router | `https://github.com/EmbeddedLLM/llm-d-router.git` | `umbp-prefetch-checkpoint` | `b0201b2c20c45ef57d915f2aea0930984b3c55ae` | No |
| llm-d | `https://github.com/EmbeddedLLM/llm-d.git` | `main` | `d557f83e1e5f1e5a6ed54ef154c554cf6c775e33` | No |
| GEAK skill source | `https://github.com/tanpinsiang/GEAK.git` | `private/codex-runtime-main` | `9afa7a8ca7236a0da7b6cb76aed8971c3cd3b321` | Orchestration and evidence policy |
| Kimi-K3 model | `moonshotai/Kimi-K3` | immutable Hub revision | `a590ce090cb049c93a33dfe8c208ec652aa20503` | Yes |

The exact image ID observed for the nightly tag was
`sha256:40e19c756e3dc9ffc9117770904d40376c7d3bf529cc76ddc379cde7ac4dae2d`.
Fail preflight if the tag resolves to another image ID.

The exact deployed runtime source is not represented by one Git commit. It is
the synthetic merge tree `d9faeffe...` plus the archived direct-GPU prefetch
patch with SHA-256
`f462f0d7e7bb011c703a708745f1a4294238cd53de7e15e65ac0d637ac7a73bd`.
That byte-identical change was subsequently committed to the vLLM `umbp`
branch as `1cf7681d1e099af8ce383e7b270f51424669d7dd`. It adds a non-executed
lookahead token only to explicit GPU/HBM-prefetch requests and judges
completion against requested full blocks. Normal serving and CPU-tier
prefetch do not use this path. Without the change, hybrid lookup reserves the
final token for logit recomputation, aligns 767 tokens to zero, and falsely
reports a miss at Kimi-K3's 768-token cache boundary.

The MoRI, llm-d-router, and llm-d checkouts are clean. The requested local Git
identity used for the vLLM commits is:

```bash
git -C /home/ubuntu/vllmumbp/repos/vllm config --local \
  user.email "tunjian.tan@embeddedllm.com"
git -C /home/ubuntu/vllmumbp/repos/vllm config --local \
  user.name "tjtanaa"
```

## Preserved reproduction inputs

The canonical controller bundle is:

```text
/home/ubuntu/vllmumbp/local-logs/geak-runs/
  geak-e2e-vllm-20260901T022450Z-b12b5aaa
```

The preserved source archives and scripts are the easiest way to reproduce the
exact run. Verify them before staging:

| File relative to the bundle | SHA-256 |
| --- | --- |
| `artifacts/vllm-merged.tar.gz` | `3d5200f30ed3e61cefc4ab776a31da739ca1d728978a0085d90fd14fedd3fd7d` |
| `artifacts/mori.tar.gz` | `bcfa9f3c8523b84d061492ed844ca38a8f855a754653c0bcbca634bb51a67bb2` |
| `artifacts/mori-submodules.tar.gz` | `ac7987d8abf6bf3b031795eac4e345b00bfe05ae5c98aa5a85a131a7b0369b1a` |
| `evidence/hbm-prefetch-lookahead.patch` | `f462f0d7e7bb011c703a708745f1a4294238cd53de7e15e65ac0d637ac7a73bd` |
| `config/prepare-runtime.sh` | `60d01869103ba45cc687df263b843833a20c775e3fe64e1ebdc37dfd8ea56300` |
| `config/launch-kimi-k3.sh` | `e7588ec9364311b45bd61af3b3c39122a9ab9f850adabe83b812dd71031ae31e` |
| `config/validate_hbm_prefetch.py` | `0bb3ca1e9b0c7564208880bfeb886f3f1a891d1d8fe0ee547508685bc2289df4` |
| `config/validate_kimi_k3_ssd_readback.py` | `768defc2de04469a6494ba3e987d7f517a3340951e0e53bea8ce25fd7c5ac99f` |
| `config/inspect_mori_ssd_segments.py` | `a611b75a49842045843666048a375ce51093b715e4fe2cd1f51ec05335035600` |
| `config/chat-smoke.json` | `40940b8f73c2171eff0ed23a553616fa2e88779aae7f969ce887c484dc0a3659` |
| `config/tool-smoke.json` | `5ee47f07887707fa0184773e1cc4d185fe0cbdd04eb5030701a6f15f623a96f4` |

The merge tree can also be verified from the two vLLM commits without using the
archive:

```bash
KIMI_NIGHTLY_COMMIT=1dc464d42681d22f38caf1fdc1eb632dc4421c45
KIMI_UMBP_COMMIT=7aaa9957356b6b87b257e49dca36b5ad35994f64
KIMI_EXPECTED_TREE=d9faeffe5744f7ec540414612667fc6984d5add1
KIMI_ACTUAL_TREE="$(git -C /home/ubuntu/vllmumbp/repos/vllm \
  merge-tree --write-tree "$KIMI_NIGHTLY_COMMIT" "$KIMI_UMBP_COMMIT")"
test "$KIMI_ACTUAL_TREE" = "$KIMI_EXPECTED_TREE"
```

## 1. Define a fresh run

Never reuse the original run ID. Keep every task-created remote file below one
new child of `/mnt/umbp-ssd0`, and keep all command transcripts on the
controller VPS.

```bash
KIMI_CONTROLLER_ROOT=/home/ubuntu/vllmumbp
KIMI_INPUT_BUNDLE="$KIMI_CONTROLLER_ROOT/local-logs/geak-runs/geak-e2e-vllm-20260901T022450Z-b12b5aaa"
KIMI_RUNNER=crs-m2m-ext-vllm-003.us-east2-a.compute.internal
KIMI_RUN_ID="kimi-k3-umbp-repro-$(date -u +%Y%m%dT%H%M%SZ)"
KIMI_RUN_ROOT="/mnt/umbp-ssd0/$KIMI_RUN_ID"
KIMI_CONTROLLER_LOG="$KIMI_CONTROLLER_ROOT/local-logs/kimi-k3-umbp-reproductions/$KIMI_RUN_ID"
KIMI_IMAGE=vllm/vllm-openai-rocm:nightly-1dc464d42681d22f38caf1fdc1eb632dc4421c45
KIMI_IMAGE_ID=sha256:40e19c756e3dc9ffc9117770904d40376c7d3bf529cc76ddc379cde7ac4dae2d
KIMI_MODEL_REVISION=a590ce090cb049c93a33dfe8c208ec652aa20503
KIMI_MODEL_ROOT=/shared_vllm/huggingfacehub/models--moonshotai--Kimi-K3
KIMI_MODEL_SNAPSHOT="$KIMI_MODEL_ROOT/snapshots/$KIMI_MODEL_REVISION"
KIMI_SERVER_CONTAINER="$KIMI_RUN_ID-server"
install -d -m 0750 "$KIMI_CONTROLLER_LOG"
```

Use `ssh -G "$KIMI_RUNNER"` to review the resolved destination before the
first connection.

## 2. Preflight the runner, image, SSD, GPUs, and model

This validation requires eight idle MI355X GPUs, Docker, the exact image, and
the ext4 filesystem mounted from `/dev/nvme0n1` at `/mnt/umbp-ssd0`.

```bash
set -o pipefail
ssh -o BatchMode=yes "$KIMI_RUNNER" '
  set -euo pipefail
  findmnt -T /mnt/umbp-ssd0 -o SOURCE,FSTYPE,TARGET,OPTIONS
  test "$(findmnt -T /mnt/umbp-ssd0 -no SOURCE)" = /dev/nvme0n1
  test "$(findmnt -T /mnt/umbp-ssd0 -no FSTYPE)" = ext4
  docker version
  rocm-smi --showproductname --showmemuse --showuse
  docker ps --format "{{.ID}} {{.Names}} {{.Status}}"
' 2>&1 | tee "$KIMI_CONTROLLER_LOG/preflight.log"

KIMI_OBSERVED_IMAGE_ID="$(ssh -o BatchMode=yes "$KIMI_RUNNER" \
  "docker image inspect '$KIMI_IMAGE' --format '{{.Id}}'")"
test "$KIMI_OBSERVED_IMAGE_ID" = "$KIMI_IMAGE_ID"
```

Check both GPU utilization and processes or containers before claiming the
GPUs are free. Do not stop unrelated workloads.

The model was pre-existing on shared NFS and must remain read-only. Do not
download anything if these checks fail:

```bash
ssh -o BatchMode=yes "$KIMI_RUNNER" "
  set -euo pipefail
  test -f '$KIMI_MODEL_SNAPSHOT/config.json'
  test \"\$(find '$KIMI_MODEL_SNAPSHOT' -name '*.safetensors' | wc -l)\" -eq 96
  test -z \"\$(find '$KIMI_MODEL_ROOT' -name '*.incomplete' -print -quit)\"
  findmnt -T /shared_vllm -o SOURCE,FSTYPE,TARGET,OPTIONS
  du -sb '$KIMI_MODEL_ROOT'
" 2>&1 | tee "$KIMI_CONTROLLER_LOG/model-preflight.log"
```

The observed model tree size was 1,560,936,091,448 bytes. A different byte
count may reflect harmless Hub metadata changes, but the pinned snapshot,
96 shards, required configuration files, and empty incomplete-file inventory
are mandatory.

## 3. Stage the immutable source and scripts on NVMe

Create and mark the exact remote run root:

```bash
ssh -o BatchMode=yes "$KIMI_RUNNER" "
  set -euo pipefail
  test ! -e '$KIMI_RUN_ROOT'
  test -z \"\$(docker ps -aq --filter 'label=geak.codex.run_id=$KIMI_RUN_ID')\"
  install -d -m 0750 \
    '$KIMI_RUN_ROOT' \
    '$KIMI_RUN_ROOT/config' \
    '$KIMI_RUN_ROOT/src' \
    '$KIMI_RUN_ROOT/runtime' \
    '$KIMI_RUN_ROOT/cache' \
    '$KIMI_RUN_ROOT/home' \
    '$KIMI_RUN_ROOT/results' \
    '$KIMI_RUN_ROOT/tmp' \
    '$KIMI_RUN_ROOT/umbp-ssd'
  chmod 1777 '$KIMI_RUN_ROOT/tmp'
  printf '%s\\n' '$KIMI_RUN_ID' >'$KIMI_RUN_ROOT/.geak-run-id'
"
```

Copy only the recorded inputs:

```bash
scp \
  "$KIMI_INPUT_BUNDLE/artifacts/vllm-merged.tar.gz" \
  "$KIMI_INPUT_BUNDLE/artifacts/mori.tar.gz" \
  "$KIMI_INPUT_BUNDLE/artifacts/mori-submodules.tar.gz" \
  "$KIMI_RUNNER:$KIMI_RUN_ROOT/"

scp \
  "$KIMI_INPUT_BUNDLE/config/prepare-runtime.sh" \
  "$KIMI_INPUT_BUNDLE/config/launch-kimi-k3.sh" \
  "$KIMI_INPUT_BUNDLE/config/validate_hbm_prefetch.py" \
  "$KIMI_INPUT_BUNDLE/config/validate_kimi_k3_ssd_readback.py" \
  "$KIMI_INPUT_BUNDLE/config/inspect_mori_ssd_segments.py" \
  "$KIMI_INPUT_BUNDLE/config/chat-smoke.json" \
  "$KIMI_INPUT_BUNDLE/config/tool-smoke.json" \
  "$KIMI_INPUT_BUNDLE/evidence/hbm-prefetch-lookahead.patch" \
  "$KIMI_RUNNER:$KIMI_RUN_ROOT/config/"
```

Verify the hashes from the table above, then extract and apply the final patch:

```bash
ssh -o BatchMode=yes "$KIMI_RUNNER" "
  set -euo pipefail
  test \"\$(cat '$KIMI_RUN_ROOT/.geak-run-id')\" = '$KIMI_RUN_ID'
  cd '$KIMI_RUN_ROOT'
  echo '3d5200f30ed3e61cefc4ab776a31da739ca1d728978a0085d90fd14fedd3fd7d  vllm-merged.tar.gz' | sha256sum -c -
  echo 'bcfa9f3c8523b84d061492ed844ca38a8f855a754653c0bcbca634bb51a67bb2  mori.tar.gz' | sha256sum -c -
  echo 'ac7987d8abf6bf3b031795eac4e345b00bfe05ae5c98aa5a85a131a7b0369b1a  mori-submodules.tar.gz' | sha256sum -c -
  echo 'f462f0d7e7bb011c703a708745f1a4294238cd53de7e15e65ac0d637ac7a73bd  config/hbm-prefetch-lookahead.patch' | sha256sum -c -
  tar -xzf vllm-merged.tar.gz -C src
  tar -xzf mori.tar.gz -C src
  tar -xzf mori-submodules.tar.gz -C src/mori
  patch --dry-run -d src/vllm -p1 <config/hbm-prefetch-lookahead.patch
  patch -d src/vllm -p1 <config/hbm-prefetch-lookahead.patch
  chmod 0755 config/prepare-runtime.sh config/launch-kimi-k3.sh
"
```

## 4. Build MoRI and prepare the exact Python runtime

The preparation script copies ABI-matched vLLM extensions from the exact
nightly image, creates a Python 3.12 system-site-packages virtual environment,
pins `pybind11==3.0.4`, builds MoRI for `gfx950` with UMBP enabled and SPDK
disabled, and runs 96 focused tests.

```bash
ssh -o BatchMode=yes "$KIMI_RUNNER" "
  docker run --rm \
    --name '$KIMI_RUN_ID-prepare' \
    --label 'geak.codex.run_id=$KIMI_RUN_ID' \
    --network host \
    --shm-size 6g \
    -v '$KIMI_RUN_ROOT:/workspace/task:rw' \
    --entrypoint /bin/bash \
    '$KIMI_IMAGE' \
    /workspace/task/config/prepare-runtime.sh
" 2>&1 | tee "$KIMI_CONTROLLER_LOG/prepare-runtime.log"
```

Expected result: 96 tests pass and the import probe prints the Mori tier class,
physical-placement API, and Kimi-K3 combined parser.

## 5. Launch Kimi-K3 with the exact validated configuration

The launch script contains the full vLLM command. In addition to the arguments
requested for Kimi-K3, it pins the model revision and configures UMBP, prefix
caching, Kimi's 768-token aligned Mamba boundary, prompt-cache metrics, and a
bounded smoke cache.

The smoke-only settings are:

- `--max-model-len 98304`
- `--num-gpu-blocks-override 136`
- `--enable-prefix-caching`
- `--mamba-cache-mode align`
- `--prefix-cache-retention-interval 768`
- `--enable-prompt-tokens-details`

They make eviction deterministic; they are not proposed production capacity
settings.

```bash
ssh -o BatchMode=yes "$KIMI_RUNNER" "
  docker run -d \
    --name '$KIMI_SERVER_CONTAINER' \
    --label 'geak.codex.run_id=$KIMI_RUN_ID' \
    --network host \
    --shm-size 6g \
    --cap-add IPC_LOCK \
    --cap-add SYS_NICE \
    --cap-add SYS_PTRACE \
    --ulimit memlock=-1:-1 \
    --ulimit nofile=65536:65536 \
    --device /dev/kfd \
    --device /dev/dri \
    --device /dev/infiniband \
    -v /shared_vllm/huggingfacehub:/app/model/hub:ro \
    -v /sys:/sys:ro \
    -v /mnt/umbp-ssd0:/app/umbp-ssd0:rw \
    -v '$KIMI_RUN_ROOT:/workspace/task:rw' \
    -v '$KIMI_RUN_ROOT/tmp:/tmp:rw' \
    -e HF_HOME=/app/model \
    -e HF_HUB_CACHE=/app/model/hub \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_ROCM_USE_AITER=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    -e VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1 \
    -e AITER_BF16_FP8_MOE_BOUND=0 \
    -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
    -e VLLM_UMBP_DEBUG_STORE=1 \
    --entrypoint /bin/bash \
    '$KIMI_IMAGE' \
    /workspace/task/config/launch-kimi-k3.sh
" 2>&1 | tee "$KIMI_CONTROLLER_LOG/container-id.log"
```

In a separate controller terminal, stream the only complete server log to the
VPS:

```bash
set -o pipefail
ssh -o BatchMode=yes "$KIMI_RUNNER" \
  "docker logs -f '$KIMI_SERVER_CONTAINER'" \
  2>&1 | tee "$KIMI_CONTROLLER_LOG/server.log"
```

Wait for health without assuming that low instantaneous GPU utilization means
startup has stopped:

```bash
ssh -o BatchMode=yes "$KIMI_RUNNER" '
  set -euo pipefail
  for attempt in $(seq 1 120); do
    if curl -fsS http://127.0.0.1:18003/health >/dev/null; then
      curl -fsS http://127.0.0.1:18003/v1/models
      exit 0
    fi
    sleep 5
  done
  exit 124
' 2>&1 | tee "$KIMI_CONTROLLER_LOG/health.log"
```

The launch log must show the pinned NFS snapshot, 96 shards, TP=8, AITER
MXFP4/SITUV2 experts, full-decode-only graph capture, four hybrid KV groups,
and the MoRI SSD directory under `/workspace/task/umbp-ssd`.

## 6. Run focused patch tests

```bash
ssh -o BatchMode=yes "$KIMI_RUNNER" "
  docker exec \
    -e PYTHONPATH=/workspace/task/src/mori/python:/workspace/task/src/vllm \
    '$KIMI_SERVER_CONTAINER' \
    bash -lc '
      cd /workspace/task/src/vllm
      /workspace/task/runtime/venv/bin/python -m pytest -q \
        tests/v1/engine/test_kv_prefetch.py \
        tests/v1/core/test_scheduler.py::test_hbm_prefetch_completion_caches_then_releases_without_output \
        tests/v1/core/test_scheduler.py::test_hbm_prefetch_completion_ignores_synthetic_lookahead_token \
        tests/v1/core/test_scheduler.py::test_hbm_prefetch_admission_caps_pending_blocks_and_requests \
        tests/v1/core/test_scheduler.py::test_hbm_prefetch_admission_caps_gpu_bytes \
        tests/v1/core/test_scheduler.py::test_hbm_prefetch_expiry_cancels_pending_and_discards_terminal
    '
" 2>&1 | tee "$KIMI_CONTROLLER_LOG/hbm-prefetch-tests.log"
```

Expected result: 7 tests pass.

## 7. Force SSD spill and validate restoration

The validator creates 37 distinct 769-token prompts. Four hybrid groups per
prompt produce 148 retained objects, exceeding the 136-block HBM cap. The
six-second pacing allows MoRI's asynchronous DRAM-to-SSD cascade to complete
before the next prefix.

```bash
ssh -o BatchMode=yes "$KIMI_RUNNER" "
  docker exec \
    -e PYTHONPATH=/workspace/task/config:/workspace/task/src/mori/python:/workspace/task/src/vllm \
    '$KIMI_SERVER_CONTAINER' \
    /workspace/task/runtime/venv/bin/python \
    /workspace/task/config/validate_kimi_k3_ssd_readback.py \
    --output /workspace/task/results/kimi-k3-umbp-ssd-readback.json
" 2>&1 | tee "$KIMI_CONTROLLER_LOG/ssd-readback.log"
```

Require the validator result and full generated-token equality:

```bash
ssh -o BatchMode=yes "$KIMI_RUNNER" "
  jq -e '
    .passed == true and
    .cpu_prefetch.status == \"ready\" and
    .cpu_prefetch.ready_blocks == 4 and
    .gpu_prefetch.status == \"ready\" and
    .restored.cached_tokens == 768 and
    .ssd_copy_count >= 4 and
    .ssd_copy_bytes > 0 and
    .tier_read_bytes > 0 and
    .cpu_to_gpu_bytes > 0 and
    .reference.token_ids == .restored.token_ids and
    .first_token_logprob_abs_diff <= 0.25
  ' '$KIMI_RUN_ROOT/results/kimi-k3-umbp-ssd-readback.json'
"
```

The recorded passing run produced:

| Measurement | Value |
| --- | ---: |
| MoRI SSD copies | 148 |
| MoRI SSD-copy bytes | 25,140,658,176 |
| Mori-tier read bytes | 679,477,248 |
| CPU-to-GPU bytes | 619,241,472 |
| Restored cached tokens | 768 |
| Reference and restored token IDs | `[220, 220, 15]` |
| First-token logprob absolute difference | 0.0 |
| Reference latency | 1,869.49 ms, cold/JIT-affected |
| Restored latency | 170.38 ms |

Latency is diagnostic, not an acceptance threshold. Startup/JIT state and
Linux page cache make a single pair unsuitable as a production benchmark.

On a fresh run, segments 0 through 3 belong to the target prefix. Prove that
the master reports all four physical placements as SSD:

```bash
ssh -o BatchMode=yes "$KIMI_RUNNER" "
  docker exec \
    -e PYTHONPATH=/workspace/task/src/mori/python \
    '$KIMI_SERVER_CONTAINER' \
    /workspace/task/runtime/venv/bin/python \
    /workspace/task/config/inspect_mori_ssd_segments.py \
    --ssd-dir /workspace/task/umbp-ssd 0 1 2 3
" 2>&1 | tee "$KIMI_CONTROLLER_LOG/mori-placement.json"
```

Each object should be 169,869,312 bytes with tier `UMBPTierType.SSD`, node
`kimi-k3-umbp-003`, and peer `192.168.0.69:17003`.

The kernel NVMe sector counter may remain unchanged because these newly written
files can still be served from Linux page cache. The acceptance proof is the
combination of authoritative MoRI SSD placement, positive Mori-tier read bytes,
positive CPU-to-GPU bytes, cached-token reuse, and exact output equivalence.

## 8. Validate reasoning and tool parsing

```bash
ssh -o BatchMode=yes "$KIMI_RUNNER" "
  set -euo pipefail
  KIMI_CHAT=\$(curl -fsS -H 'Content-Type: application/json' \
    --data-binary '@$KIMI_RUN_ROOT/config/chat-smoke.json' \
    http://127.0.0.1:18003/v1/chat/completions)
  KIMI_TOOL=\$(curl -fsS -H 'Content-Type: application/json' \
    --data-binary '@$KIMI_RUN_ROOT/config/tool-smoke.json' \
    http://127.0.0.1:18003/v1/chat/completions)
  printf '%s\\n' \"\$KIMI_CHAT\" | jq -e '.choices[0].message.content | test(\"42\")'
  printf '%s\\n' \"\$KIMI_TOOL\" | jq -e '
    .choices[0].finish_reason == \"tool_calls\" and
    .choices[0].message.tool_calls[0].function.name == \"get_weather\" and
    ((.choices[0].message.tool_calls[0].function.arguments | fromjson).city | ascii_downcase) == \"boston\"
  '
" 2>&1 | tee "$KIMI_CONTROLLER_LOG/chat-tool-smoke.log"
```

The passing response returned content `42` with reasoning separated, and a
valid `get_weather({"city":"Boston"})` call with
`finish_reason="tool_calls"`.

## 9. Recover evidence before teardown

```bash
scp \
  "$KIMI_RUNNER:$KIMI_RUN_ROOT/results/kimi-k3-umbp-ssd-readback.json" \
  "$KIMI_CONTROLLER_LOG/"
sha256sum "$KIMI_CONTROLLER_LOG/kimi-k3-umbp-ssd-readback.json"

ssh -o BatchMode=yes "$KIMI_RUNNER" \
  'curl -fsS http://127.0.0.1:18003/metrics' \
  2>&1 | tee "$KIMI_CONTROLLER_LOG/vllm-metrics.txt" >/dev/null
ssh -o BatchMode=yes "$KIMI_RUNNER" \
  'curl -fsS http://127.0.0.1:19091/metrics' \
  2>&1 | tee "$KIMI_CONTROLLER_LOG/mori-metrics.txt" >/dev/null
ssh -o BatchMode=yes "$KIMI_RUNNER" \
  "docker inspect '$KIMI_SERVER_CONTAINER'" \
  2>&1 | tee "$KIMI_CONTROLLER_LOG/server-inspect.json" >/dev/null
```

Verify the copied JSON with the `jq` predicate from step 7 before removing its
remote source.

## 10. Remove only this run and audit cleanup

Stop the controller-side `docker logs -f` command first. Then remove only
objects carrying the new run label and the exact marker-validated root. Use
host sudo for the root because compiler/JIT files are created by container
root.

```bash
ssh -o BatchMode=yes "$KIMI_RUNNER" "
  set -euo pipefail
  KIMI_CLEAN_RUN_ID='$KIMI_RUN_ID'
  KIMI_CLEAN_RUN_ROOT='$KIMI_RUN_ROOT'
  test \"\$KIMI_CLEAN_RUN_ROOT\" = '/mnt/umbp-ssd0/$KIMI_RUN_ID'
  test -f \"\$KIMI_CLEAN_RUN_ROOT/.geak-run-id\"
  test \"\$(cat \"\$KIMI_CLEAN_RUN_ROOT/.geak-run-id\")\" = \"\$KIMI_CLEAN_RUN_ID\"
  while IFS= read -r KIMI_CONTAINER_ID; do
    test -n \"\$KIMI_CONTAINER_ID\" && docker rm -f \"\$KIMI_CONTAINER_ID\"
  done < <(docker ps -aq --filter \"label=geak.codex.run_id=\$KIMI_CLEAN_RUN_ID\")
  case \"\$KIMI_CLEAN_RUN_ROOT\" in
    /mnt/umbp-ssd0/kimi-k3-umbp-repro-*) ;;
    *) exit 1 ;;
  esac
  sudo rm -rf --one-file-system -- \"\$KIMI_CLEAN_RUN_ROOT\"
  test ! -e \"\$KIMI_CLEAN_RUN_ROOT\"
  test -z \"\$(docker ps -aq --filter \"label=geak.codex.run_id=\$KIMI_CLEAN_RUN_ID\")\"
"
```

Run the independent auditor from the controller:

```bash
python3 /home/ubuntu/.codex/skills/geak/scripts/audit_cleanup.py \
  --runner "$KIMI_RUNNER" \
  --run-id "$KIMI_RUN_ID" \
  --run-root "$KIMI_RUN_ROOT" \
  --weight-root /shared_vllm/huggingfacehub \
  --model moonshotai/Kimi-K3 \
  --revision "$KIMI_MODEL_REVISION" \
  --weight-origin preexisting \
  --port 18003 \
  --port 19091 \
  --port 15558 \
  --port 16003 \
  --port 17003 \
  2>&1 | tee "$KIMI_CONTROLLER_LOG/cleanup-audit.json"
jq -e '.cleanup_complete == true' "$KIMI_CONTROLLER_LOG/cleanup-audit.json"
```

Preserve the pre-existing nightly image and the read-only Kimi-K3 model cache.
The cleanup must report no task containers, images, networks, volumes,
processes, listeners, or run root, and no new incomplete model files.

## Known environment-specific points

- The launch script records runner 003's data-plane address
  `192.168.0.69`. Revalidate and change `node_address` only when intentionally
  reproducing on another authorized host.
- Runner 004 was not used in this validation because `/mnt/umbp-ssd0` was not
  mounted during the 2026-09-01 preflight.
- `AITER_JIT_DIR` is placed under the NVMe-backed run root because Docker's
  root filesystem did not have enough space for Kimi-K3 JIT artifacts.
- `pybind11==3.0.4` must match the exact nightly's AITER ABI.
- The 768-token retention interval is required for Kimi-K3's aligned hybrid
  boundary. A 769-token request leaves one token for recomputation while
  retaining a reusable 768-token boundary.
- MoRI is the UMBP storage implementation in this run. There is no separate
  in-engine MoRI scheduler and no llm-d router in the single-server test.
