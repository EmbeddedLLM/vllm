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

Two additional release details affect the configuration layer:

- `distributed.ranged_scratch_size` defaults to zero, disabling the remote
  ranged-I/O arenas. Configure it explicitly to fit at least the largest
  object, budgeting separate GET and PUT arenas on every worker.
- The legacy distributed `medium` selects **one** local medium. Merely enabling
  DRAM and SSD is not a tiering policy. Simultaneous peer-local DRAM/SSD tiering
  requires an explicit `backend_policy_path`; without that policy, different
  peers may serve different media, but this is not local promotion/demotion.

These are source observations from the pinned
[configuration contract](https://github.com/ROCm/mori/blob/67632e80e2e492184b589904b63225f82d45537c/src/umbp/include/umbp/common/config.h),
not native behavior established by the CPU tests below.

### Configuration test design

The configuration adapter translates explicit per-worker capacity, SSD roots,
network endpoints and actual layout object sizes into the pinned native API.
Its contract is to reject invalid input before starting a client, generate an
explicit DRAM/SSD backend policy, and own only the directories it creates.
Unit tests will exercise unknown fields, insufficient page/scratch sizing,
partial network configuration, native startup failure and cleanup ordering.
The existing store tests provide real CPU buffers and deterministic I/O
barriers; native-config fakes test translation, not SSD or RDMA operation.

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
| Explicit storage configuration | `umbp/config.py`; page/scratch checks, DRAM/SSD policy and private-path lifetime | CPU translation/lifetime tests pass; native policy lowering pending |
| Compatible P/D/offload keys | `umbp/key.py`, `umbp/layout.py`; descriptor integration and identity isolation | Logical rank-local descriptors implemented; scheduler handshake pending |
| Scheduler lookup and allocation | KVConnector lookup, metadata, block ownership, asynchronous feedback | Job lifecycle uses real cache-manager pins; request planner/startup wiring pending |
| Worker load/store and errors | Registered cache views, compute fences, completion snapshots and error block IDs | All-rank finalization/error snapshots CPU-tested through both runner collectors; public connector wiring pending |
| P/D handoff and incremental transfer | Producer/consumer protocol, all-rank commit, local-prefix reuse, two-engine transfer accounting | Pending |
| Single-node DRAM and ext4 SSD offload | MoRI embedded deployment, forced eviction, byte/source reconciliation | Pending |
| Multi-node DRAM/SSD restore | Master-led deployment, peer failures, safe recompute and recovery | Pending |
| TP and hybrid cache geometry | Exact layouts, complete group/shard restoration, cancellation and preemption | Layout and rank-barrier CPU tests pass; real scheduler/hybrid semantics pending |
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

## Second implementation checkpoint: layout and transfer lifecycle

Implemented components, still without a registered `UMBPConnector`:

- `umbp/layout.py` validates views against vLLM's allocator and deduplicates
  registration of shared backing allocations. Objects concatenate sorted layers
  in logical H/N/C order. Physical LBHNC/LBNHC/LHBNC/BLHNC/BLNHC/BHLNC packing,
  cache capacity and kernel-block splitting do not alter the stored bytes.
  Padded bytes are omitted from ordinary layer views. Group-specific specs,
  layer identity, dtype/encoding, topology and CP interleave enter the identity.
- `umbp/worker.py` owns registered allocations, bounds admitted jobs, defers I/O
  until the supplied compute fence passes, and rejects conflicting transfers by
  physical byte ranges, including aliases across cache groups. A failed compute
  fence is fatal; it cannot be interpreted as completed work or a cache miss.
  Native queue pressure is retried within the bounded admitted set. Native
  exceptions produce failed per-object receipts, including destinations that
  may already have changed. Cancellation never claims to stop running DMA.
- `umbp/metadata.py` uses an engine generation plus monotonically increasing
  job sequence, independent of reusable request IDs. Rank receipts aggregate
  idempotently. The completion barrier waits for every required rank even after
  a failure, and requires every object to succeed before publication.

Test design: the mapper's contract is logical group-block bytes in/out, with
no overwrite outside the selected block; the worker's contract is an immutable
job plus compute fence in and final per-rank outcomes out. Guarded failures
include wrong strides, packed-layer overwrite, kernel-block ordering, early
release, alias conflicts, partial native writes, duplicate acknowledgements and
stale generations. CPU buffers plus deterministic thread barriers are the
cheapest useful tests. The existing `KVOutputAggregator` is also exercised with
the new worker metadata; this does not exercise scheduler ownership.

The combined suite passed **114 CPU tests** (112 UMBP component tests plus two
existing output-aggregator regressions). It includes all 36 source/destination
physical-layout pairs with packed layers and unequal cache capacities, split
kernel blocks, padded hybrid/Mamba geometry, per-layer uniform-wrapper specs,
native read cancellation, queued cancellation, shutdown/fence draining, native
failure after a partial write, and multi-rank/multi-group completion barriers.

```bash
PYTHONDONTWRITEBYTECODE=1 "$TEST_VENV/bin/python" -m pytest \
  --confcutdir=tests/v1/kv_connector/unit \
  tests/v1/kv_connector/unit/test_umbp_store.py \
  tests/v1/kv_connector/unit/test_umbp_key.py \
  tests/v1/kv_connector/unit/test_umbp_layout.py \
  tests/v1/kv_connector/unit/test_umbp_worker.py \
  tests/v1/kv_connector/unit/test_output_aggregator.py -q
```

Limitations and next integration work remain mandatory:

1. Integrate the configuration/policy adapter with actual connector hooks.
   The scheduler must pin blocks, handle rejected admission with explicit failed
   rank receipts, and deliver invalid load block IDs and request completions.
   Worker job admission must follow increasing sequence order; a retired job
   cannot be replayed to restart I/O. Constructor and shutdown ownership need
   to be checked again in the real model-runner lifecycle.
2. Exchange worker-derived identities before scheduler lookup. Scheduler configs
   may flatten a `UniformTypeKVCacheSpecs` wrapper, so reconstructing the worker
   identity from a representative scheduler layer is not sound. Quantization
   scales/calibration not represented by a cache spec need an explicit compatible
   identity or rejection; the current tests do not establish their portability.
3. The current format deliberately requires matching TP/PP/CP topology. This is
   not heterogeneous-TP resharding. Mamba byte-layout tests do not prove recurrent
   checkpoint validity. Sliding windows, partial tails, preemption and boundary
   state must be composed with the actual cache coordinator before serving.
4. Non-prefix-cacheable state cannot use a prefix hash. The worker rejects such
   jobs until request-scoped handoff keys and boundary-state handling exist.
   Do not silently exclude required groups and then advertise a full P/D hit.
5. A two-worker CPU buffer round trip is not two-engine P/D. Still required:
   producer completion/decoder handoff, local-prefix-aware lookup, all-rank
   publication, model output/accuracy tests, real MoRI GPU/RDMA/SSD evidence,
   routing/prefetch migration, transfer-effectiveness measurements and cleanup.

All applicable pre-commit checks must pass for each local checkpoint. The
unrelated Actionlint hook is skipped for these Python/Markdown-only changes:
its Go toolchain installer fails with a download/version-resolution error,
and no `.github/workflows/` files are changed. Repository lint policy is
unchanged. Full transcripts and failed intermediate runs remain on the VPS.

## Third implementation checkpoint: explicit storage policy

`UMBPStoreConfig` now builds the pinned native configuration with an explicit
schema-v1 policy for DRAM-only, SSD-only or DRAM→SSD storage. With both media,
the policy spills on eviction and requests copy-promotion on read. These are
configured native policies, not promotion/demotion measurements. No public
`--kv-transfer-config` recipe is ready until connector registration and hooks
are implemented.

The adapter requires an explicit pool page size, positive selected capacities,
and pre-existing absolute SSD roots. It rejects unknown options, missing network
endpoints for a shared master, and page/scratch sizes smaller than the worker
layout's `max_object_bytes`. Every peer must agree on page size; heterogeneous
worker sizes need a deployment-wide maximum, not independent auto-sizing.
The otherwise unused top-level DRAM capacity prevents the release's embedded
factory from silently shrinking that page size; real allocation is specified
only by the generated backend policy.

Budgets are **per worker**, not per eight-GPU host. Account for the selected
DRAM capacity, **two** ranged scratch arenas, SSD staging slots times page size,
and native transport/metadata overhead. SSD capacity is total across the
listed roots and is split by the native sharded tier. For this deployment,
use the existing `/mnt/umbp-ssd0` bind mount after a fresh runner preflight;
adding `/mnt/umbp-ssd1` through `/mnt/umbp-ssd7` later does not rename SSD0.
Directory existence alone does not establish that it is mounted, local NVMe,
ext4, large enough or free of other users' workloads. Those remain preflight
checks, not facts inferred by this adapter.

Each client creates fresh `vllm-umbp-*` subdirectories beneath those roots,
plus a private temporary policy directory. The paths are logged before native
startup. They survive running I/O and failed deregistration. Shutdown drains
the store, deregisters allocations and releases the native client before
deleting only these owned directories; cleanup failures can be retried.
Existing root contents, mounts, model caches and shared pool data are never
cleared. Abrupt process termination can still leave directories behind: retain
the startup manifest on the VPS and audit cleanup after recovering evidence.
Native destructor failure/peer-read draining still needs native validation.
The adapter rejects an ambient `UMBP_WORKLOAD_TRACE_PATH` to prevent unowned
trace files; collect service stdout/stderr to the VPS instead.

The pinned policy parser selects file-backed SSDs, with native direct-I/O and
CRC defaults. **This does not prove physical SSD I/O or checksum verification
on ranged reads.** The pinned
[SSD range interface](https://github.com/ROCm/mori/blob/67632e80e2e492184b589904b63225f82d45537c/src/umbp/include/umbp/local/tiers/ssd_tier.h)
explicitly omits whole-record CRC verification on ranged reads. Corruption
injection and an integrity strategy must be resolved before claiming that
all corrupted cache objects reliably fall back to recomputation. Likewise,
native O_DIRECT may fall back on unsupported filesystems; measure actual device
I/O in the SSD acceptance run. GPU-destination re-cache and implicit locality
prefetch are disabled in this adapter to avoid invalid host-copy assumptions
and hidden duplicate traffic.

The combined suite now passes **148 CPU tests** (146 UMBP tests plus the two
existing output-aggregator regressions). New coverage includes schema/sizing,
DRAM-only and SSD-only policies, tiering edges, two-client directory isolation,
symlink/comma-path ambiguity, startup failure, cleanup after native draining,
deregistration/cleanup retries and ambient trace rejection. Native config and
client construction are fakes; no HIP, RDMA, actual SSD or model result follows.

Reproduction on the VPS, from this worktree:

```bash
set -o pipefail
TEST_VENV=/home/ubuntu/vllmumbp/repos/vllm/.venv
PYTHONDONTWRITEBYTECODE=1 "$TEST_VENV/bin/python" -m pytest \
  --confcutdir=tests/v1/kv_connector/unit \
  tests/v1/kv_connector/unit/test_umbp_config.py \
  tests/v1/kv_connector/unit/test_umbp_store.py \
  tests/v1/kv_connector/unit/test_umbp_key.py \
  tests/v1/kv_connector/unit/test_umbp_layout.py \
  tests/v1/kv_connector/unit/test_umbp_worker.py \
  tests/v1/kv_connector/unit/test_output_aggregator.py -q
```

Evidence: `local-logs/umbp-kvconnector-reimplementation-20260914/` on the VPS.
The earlier `cpu-tests-r7.log` contains 144 passes; `cpu-tests-r8.log` contains
148 passes after additional cleanup and path tests; the final formatted-source
run is `cpu-tests-r9.log` (148 passed). Retain intermediate lint
failures alongside `config-pre-commit-r4.log`, the passing final lint run.
Scheduler ownership, worker-layout
exchange, connector hooks, P/D, native integrity, model validation, routing and
prefetch remain required work, not waived by these component results.

### Scheduler/worker lifecycle test design

The next component owns scheduler block references and carries bounded lookup,
transfer and finalization jobs through existing connector metadata. Inputs are
vLLM cache records/allocated destinations and generation-scoped worker receipts;
outputs are immutable job batches, all-rank outcomes and receive/error snapshots.
Tests must catch premature reuse after request free/preemption, shared-prefix
overwrite, wrong cache-record identities, rank failure/replay and publication
before all native accesses end. Use the real KVCacheManager/BlockPool and existing
CPU worker/native fixtures first, then the actual model-runner output collector.
These composed tests are still not model serving or a complete P/D protocol.

## Fourth implementation checkpoint: scheduler ownership and runner snapshots

`umbp/scheduler.py` now owns bounded lookup/transfer jobs against a real
`KVCacheManager`. Store sources must match actual group/hash cache records;
load destinations must belong to the request, be exclusively owned and not
already published to prefix caching. Every job pins its exact physical GPU
blocks. Request free/cancellation does not release native ownership, and a
same-string request ID cannot cancel a different Request incarnation. Only one
receive job may be outstanding per request because the runner completion API
identifies requests rather than individual jobs.

`umbp/lifecycle.py` translates ordered job metadata into the existing transfer
worker, bounded native lookups, per-rank receipts and runner completion/error
snapshots. Each rank looks up its own compatible keys, so masterless private
worker pools are not incorrectly treated as a rank-0 shared store. All-rank
per-object conjunction determines lookup results. Queue pressure retries within
the admitted set; cancellation drains running lookups and never invents hits.

The receive lifecycle deliberately retains ownership through error delivery:

```text
scheduler pins exact blocks -> worker compute fence -> native operation
            |                                             |
            |                      every rank reports terminal object results
            |                                             |
            +-> scheduler sends all-rank finalization <----+
                                  |
                workers emit finished_recving + invalid block IDs
                                  |
                retirement acknowledgements in the same output snapshots
                                  |
                scheduler consumes snapshots, then releases load pins
```

Stores can release their pins after all native completions, but receives keep
them until every finalization snapshot has been consumed. This closes the gap
where an old failed-load block ID could otherwise be reused before the scheduler
handles its error. Completed jobs continue counting against admission until
retirement; empty engine steps remain necessary while any phase is pending.
Duplicate rank feedback is idempotent, and a malformed feedback batch cannot
partially release other jobs' references. Rejected worker transfers still wait
for their compute fence before issuing failure receipts.

Validation: **164 CPU tests passed**, 15 expected warnings, 16.12 seconds,
in `cpu-tests-r11.log` on the VPS. This includes 161 UMBP component/composed
tests, two existing output-aggregator regressions, and the unchanged connector
metadata-cleanup regression. The final run used `HF_HUB_OFFLINE=1`: no model
or configuration was downloaded. Both v1's `KVConnectorModelRunnerMixin` and
v2's `ActiveKVConnector` collect real lifecycle completion/error/retirement
snapshots through a thin mock connector wrapper. GPU execution and native MoRI
remain fakes; these are not full Scheduler request-state or model-serving tests.

```bash
set -o pipefail
TEST_VENV=/home/ubuntu/vllmumbp/repos/vllm/.venv
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 "$TEST_VENV/bin/python" -m pytest \
  --confcutdir=tests/v1/kv_connector/unit \
  tests/v1/kv_connector/unit/test_umbp_config.py \
  tests/v1/kv_connector/unit/test_umbp_store.py \
  tests/v1/kv_connector/unit/test_umbp_key.py \
  tests/v1/kv_connector/unit/test_umbp_layout.py \
  tests/v1/kv_connector/unit/test_umbp_worker.py \
  tests/v1/kv_connector/unit/test_umbp_lifecycle.py \
  tests/v1/kv_connector/unit/test_output_aggregator.py \
  tests/v1/kv_connector/unit/test_kv_connector_lifecycle.py -q
```

There is still **no registered or usable UMBPConnector**. The next work must
connect request planning, worker initialization/layout handshake and actual GPU
event creation to these components, then implement the P/D handle/readiness
protocol and missing-prefix-only loading. A transfer finalization is not a
published P/D readiness record. Hybrid boundary selection, request-scoped
non-prefix state, host-pool-qualified IDs, native integrity and model/accuracy/
performance evidence remain required. Do not treat unsupported configurations
as waived requirements or offer a serving command that does not yet work.
