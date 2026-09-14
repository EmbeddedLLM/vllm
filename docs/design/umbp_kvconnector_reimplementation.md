# UMBP reimplementation through KVConnectors

Status: **in progress; not yet a usable or validated connector**.

## Target and preserved baseline

Reimplement the UMBP integration through vLLM's KVConnector scheduler/worker
contract, supporting **both P/D disaggregation and KV offloading**. The old
tiering adapter is a reference for requirements, not the implementation being
promoted. Passing storage unit tests alone does not complete this objective.

| Component | Pinned starting commit |
| --- | --- |
| EmbeddedLLM/vllm `umbpkvconnector` | `1678b396270406c27fcab8f5b86b21fd305ac605` |
| ROCm/mori `v1.2.3.post1` | `67632e80e2e492184b589904b63225f82d45537c` |
| Old vLLM `umbp`, preserved separately | `aea18ed6021e03600fe16988238f366b683f7591` |
| Old MoRI implementation, preserved separately | `3f20d9a6e7c079d8ef1e682adfed07afc883a030` |
| Existing llm-d-router reference | `7541552c71642f2756517a7aeb3c0c35720c06d0` |
| Existing llm-d reference | `d557f83e1e5f1e5a6ed54ef154c554cf6c775e33` |

The MoRI annotated tag object is
`b7e7ed3e7314fade2099693bb6e700f300ac40c1`. The release requirement supersedes
the old `Requirements.md` reference to MoRI 1.2.2. No prior native results
establish correctness of these new commits. Old source, checkpoint tags and
uncommitted architecture drafts remain untouched in their original worktrees.

## Release contract changes

The pinned [MoRI client interface](https://github.com/ROCm/mori/blob/67632e80e2e492184b589904b63225f82d45537c/src/umbp/include/umbp/umbp_client.h)
uses one distributed-client implementation for embedded masterless and
master-led deployments. The old separate local backend is gone.

The [Python bindings](https://github.com/ROCm/mori/blob/67632e80e2e492184b589904b63225f82d45537c/src/pybind/pybind_umbp.cpp)
expose ranged multi-buffer reads/writes, per-key boolean results, and explicit
CPU/GPU allocation registration. Each stored object's ranges must cover all
its bytes before publication. Reads may request a subset of object bytes.
GPU registration must include the GPU ordinal; a pointer alone is insufficient.

There is no Python `UMBPClient.close()` binding in this release. The initial
wrapper therefore owns the native handle exclusively, drains all calls,
deregisters allocations, and drops the handle to invoke native destruction.
It never calls `clear()` during shutdown. Native lifecycle verification is
still required, including failures inside native teardown.

External HBM/DRAM placement reports are advisory, not proof that a corresponding
UMBP object can be read. The new implementation must not equate a routing hint
with completed, servable KV.

## Intended architecture and ownership

The planned `UMBPConnector` has producer, consumer and combined-cache roles.
P/D handoff uses the shared hash-addressed UMBP pool, allowing the decoder to
combine its local prefix with stored missing blocks. This is actual P/D only
when the producer's KV is delivered and used by a separate decode engine;
ordinary prefix reuse between full-serving replicas is not a substitute.

```text
llm-d routing / prefetch control
          |
          +--> Prefill engine: UMBPConnector scheduler <-> worker
          |                         |                       |
          |                  block/job ownership       registered KV
          |                         |                       |
          |                  handoff completion       UMBP ranged I/O
          |                         |                       |
          +--> Decode engine: UMBPConnector <---- shared MoRI pool
                                   |              DRAM / file-backed SSD
                              missing KV only
                                   |
                              decode compute

Non-disaggregated engine: the same connector load/store path for offloading.
```

This diagram is the intended end state, not a claim of implemented wiring.
The existing MoRIIO direct-P2P connector is a reference for handshake and rank
completion behavior; it is not being relabeled as UMBP offloading.

### Mandatory invariants

- vLLM's cache manager owns allocation and prefix publication. Connector jobs
  retain the exact GPU blocks until every required worker acknowledges them.
- Store submission must follow model-compute completion and relevant earlier
  transfers. Native completion, not queue submission, releases ownership.
- A P/D handoff cannot advertise complete KV after only one rank or cache
  group finishes. Late completions are scoped to a job generation, not just
  a reusable request ID. Timeouts/partial failure cannot become cache hits.
- Decoder lookup starts from its local computed prefix and loads only the
  missing required groups/shards. Incomplete or incompatible KV must lead to
  safe recomputation, never model execution on unwritten cache blocks.
- Compatibility includes deployment namespace, model revision, canonical
  cache layout/dtype/block geometry/shard mapping and hash algorithm.
  vLLM block hashes retain salt/adapter/multimodal identity. P/D role and
  engine ID are deliberately not part of compatible shared-cache keys.
- Ranged objects may contain multiple layer buffers. Write ranges must tile
  the object exactly; registered extents and read destinations are validated.
- Queue capacity is bounded. Rejected admission starts no native operation.
  Queued cancellation is distinct from cancellation of already-running DMA.
- Failed reads may have changed destination bytes. Report load errors through
  KVConnector APIs and do not publish those blocks. Offload misses and failed
  best-effort stores cannot hang ordinary requests indefinitely.
- No dependency on the old tier-manager registration or synthetic core-only
  offload implementation. Any required engine hook must be a generic connector
  lifecycle/control contract, not a separate UMBP execution path.

## Feature migration and acceptance matrix

| Requirement | Destination / evidence needed | Current state |
| --- | --- | --- |
| Native API and owned ranged storage | `umbp/store.py`; CPU contracts, then real native pointer tests | CPU contracts implemented; native pending |
| Compatible P/D/offload keys | `umbp/key.py`; canonical descriptor integration and identity isolation | Key namespace implemented; descriptor wiring pending |
| Scheduler lookup and allocation | KVConnector lookup, metadata, block ownership, asynchronous feedback | Pending |
| Worker load/store and errors | Registered cache views, compute fences, completion snapshots and error block IDs | Pending |
| P/D handoff and incremental transfer | Producer/consumer protocol, all-rank commit, local-prefix reuse, two-engine transfer accounting | Pending |
| Single-node DRAM and ext4 SSD offload | MoRI embedded deployment, forced eviction, byte/source reconciliation | Pending |
| Multi-node DRAM/SSD restore | Master-led deployment, peer failures, safe recompute and recovery | Pending |
| TP and hybrid cache geometry | Exact layouts, complete group/shard restoration, cancellation and preemption | Pending |
| Placement and MoRI scheduling | Current authoritative placement API, lifecycle events, tier/locality/cost routing | Pending |
| CPU/HBM prefetch and admission | Token-identity control, connector-owned load, TTL/cancel/drain/no-model invariants | Pending |
| llm-d integration and fault recovery | Routing, prefetch, replay/gaps, staleness, reconnect/fail-open tests | Pending |
| Qwen correctness and accuracy | Matched no-connector/offload/P-D outputs and GSM8K evidence | Pending |
| Kimi K3 acceptance | Requested TP8/AITER settings, hybrid layouts, long-context restore and semantic tests | Pending |
| Transfer effectiveness | Source counters, physical transfer proof, matched warmup/latency/throughput protocol | Pending |
| Reproduction and zero remote residue | Exact source/runtime/model manifests, VPS logs, audited cleanup | No remote work yet |

The existing eight-GPU host descriptions are hints until revalidated. Use only
authorized hosts, pre-existing ext4 SSD0 mounts, and disposable per-run/rank
subdirectories. Inspect established model caches before downloading anything.
Retain existing weights and recover all evidence to the VPS before removing
task-created remote resources. SPDK, filesystem creation and kernel changes
are not part of this filesystem-backed implementation.

## Implementation sequence

1. Establish native ranged-storage and compatible-key contracts. Keep native
   dependencies lazy and test pointer shape, failure and lifetime behavior.
2. Implement connector configuration and scheduler/worker metadata together
   with real cache-layout mapping. Register the connector only when its
   scheduler/worker hooks are implemented, not as a placeholder.
3. Compose local offload and P/D producer/consumer handoff in tests using the
   actual vLLM scheduler/block manager. Prove all-rank/group completion,
   prefix reuse, pressure, preemption, same-ID reuse and failure fallback.
4. Validate native MoRI GPU/CPU/file-SSD transfers and two-node serving.
   Pin exact release artifacts and prove the chosen transport at runtime.
5. Port placement-aware routing and prefetch through the connector contract;
   execute the required distributed correctness, accuracy and performance
   matrix, including Qwen and Kimi. Record remaining gaps explicitly.

## First implementation checkpoint: CPU contracts

Test design:

- Module purpose: safely translate connector cache objects into MoRI native
  ranged operations without losing ownership or reporting false success.
- I/O contract: immutable object/slice descriptors, bounded asynchronous
  submission, and one boolean result for each submitted key.
- Guarded failures: pointer overruns, incompatible cache identities, partial
  object publication, malformed results, queue overflow, and early release
  during cancellation or shutdown.
- Cheapest useful level: unit tests with actual CPU buffers and a native-API
  fake, deterministic thread barriers and weak-reference lifetime checks.
  These do not prove HIP, RDMA, SSD behavior or scheduler correctness.

The initial combined run passed **47 CPU tests**. Tests import the new
worktree's code with the existing VPS test venv (Torch `2.13.0+cpu`). No native
MoRI package or GPU is used in this test. The unbuilt checkout warns that
`vllm._version` is absent; no compiled-artifact or native-runtime claim is made.

From the new vLLM worktree, use a venv containing the CPU test dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 "$TEST_VENV/bin/python" -m pytest \
  --confcutdir=tests/v1/kv_connector/unit \
  tests/v1/kv_connector/unit/test_umbp_store.py \
  tests/v1/kv_connector/unit/test_umbp_key.py -q
```

`--confcutdir` excludes unrelated model-serving root fixtures, not code under
test. Existing connector regression suites and full plugin import/configuration
checks must run during scheduler/worker integration. No native validation or
end-to-end serving result has been inherited from the previous UMBP work.
