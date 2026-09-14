# UMBP reimplementation through KVConnectors

Status: **in progress; limited native Qwen offload and same-/two-host P/D
smokes passed; broader multi-node/TP, pressure, accuracy and performance
acceptance pending**. Current Python-source checkpoint:
`408c3b752eaf18dd9c055d9ab5bac3208c5071df`. The serving tests reused older nightly
vLLM native binaries. The latest two-host DRAM/SSD cold/partial P/D results,
source verification and cleanup evidence are recorded in the final section.

## Target and preserved baseline

Reimplement the UMBP integration through vLLM's KVConnector scheduler/worker
contract, supporting **both P/D disaggregation and KV offloading**. The old
tiering adapter is a reference for requirements, not the implementation being
promoted. Passing storage unit tests alone does not complete this objective.

The target also includes a direct MoRI-IO RDMA P/D fast path for fresh KV still
resident in prefill HBM, alongside UMBP reuse/offload. The current prototype
implements pool-mediated P/D only. Direct delivery and matched MoRIIO performance
validation are separate, unimplemented milestones; pool smoke passes do not
establish either of them.

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
Its current P/D path uses the shared hash-addressed UMBP pool, allowing the
decoder to combine its local prefix with stored missing blocks. The target adds
a direct HBM-to-HBM path under the same connector-owned lifecycle. This is actual
P/D only when the producer's KV is delivered and used by a separate decode
engine; ordinary prefix reuse between full-serving replicas is not a substitute.

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

This diagram shows the pool-mediated branch, not the complete dual-path target
or a claim of completed llm-d integration. The existing MoRIIO direct-P2P
connector is a reference for transport, handshake and rank completion behavior;
direct P/D is not being relabeled as UMBP offloading.

### Direct RDMA P/D milestone: planned, not implemented

RDMA transport and a direct GPU-to-GPU route are different properties. The
pinned MoRI release's
[MoriIoEngine](https://github.com/ROCm/mori/blob/67632e80e2e492184b589904b63225f82d45537c/src/umbp/distributed/transfer/mori_io_engine.cpp)
creates an RDMA backend and issues MoRI-IO batch reads/writes. However, the
current connector uses pool PUT/GET: remote ranged reads land in native pool
slots or scratch space before copying into the decoder's KV destinations.
Registering GPU pointers does not remove those intermediate buffers.

```text
Fresh KV still resident in producer HBM (planned direct path):
  Prefill HBM ---------------- MoRI-IO RDMA ----------------> Decode HBM

Current pool path, example with producer-local DRAM placement:
  Prefill HBM -> Prefill DRAM -- MoRI-IO RDMA --> Decode DRAM/scratch
                                                       |
                                                       +--> Decode HBM
```

Pool placement changes the number and location of copies/network transfers.
SSD is optional and adds storage I/O when selected. Direct SSD-to-HBM GDS is a
separate storage-read optimization, not the direct P/D path above. Fresh
handoffs are expected to pay extra pool admission, copies and readiness cost;
the size of that cost is unmeasured against a matched MoRIIO baseline. Pool
reuse can still avoid prefill recomputation and retain KV beyond HBM capacity.

Implement this milestone before claiming direct-P/D performance parity:

1. **Reuse transport and negotiate capability.** Reuse or factor existing
   MoRIIO transport/handshake code rather than duplicate the RDMA stack. Decide
   with maintainers whether this is a shared adapter inside `UMBPConnector` or
   coordinated connector composition. Negotiate a versioned transfer mode,
   engine generation, compatible layout/shards and worker-registered ranges.
   Router hints are not RDMA descriptors or authority to access memory. No
   current configuration switch enables this milestone.
2. **Select a path for missing KV.** Preserve the decoder's valid local prefix.
   Plan direct delivery for eligible fresh producer-resident ranges and pool
   GET for compatible stored ranges, using validated availability/cost policy.
   First validate forced direct and forced pool paths; add automatic selection
   only after each works independently. Partition mixed requests into disjoint
   destinations with one completion owner per range.
3. **Preserve direct-transfer overlap.** Allocate/register decoder destinations,
   pin producer sources and destinations, then fence each transmitted layer or
   range. Evaluate MoRIIO's layer-wise WRITE overlap and READ mode separately.
   Direct delivery must not require a preceding pool PUT, SSD access or a pool
   ready marker. Publish direct receive success only after all required
   ranks/groups complete. Pool delivery keeps its existing store/readiness gates.
4. **Coordinate offload and fallback.** Optional background pool persistence
   must not gate direct decode admission. Bound it separately and retain shared
   sources until every transfer using them finishes. A failed optional store
   must not invalidate a successful direct receive. On direct failure, suppress
   publication and drain/fence native writers before pool fallback, recompute
   or destination reuse; a timeout does not cancel RDMA. Discard partial state
   and reject stale receipts so two paths cannot race writes or publish twice.
5. **Prove lifecycle correctness.** Extend existing connector unit/composed
   tests for direct-only, pool-only and mixed-prefix requests, unavailable or
   evicted sources, incompatible peers, delayed/failed ranks, partial writes,
   cancellation, same-ID reuse and concurrent offload. Poison native receive
   buffers and compare restored bytes plus model tokens. Transport counters
   must prove HBM endpoints and expose staging/fallback; a fastpath-required
   test fails if it silently uses the pool or host payload staging.
6. **Measure effectiveness.** Run the three mandatory cases below: direct-only,
   direct-plus-offload and offload-only. Keep unchanged `MoRIIOConnector` and
   no-connector/recompute as additional references, not replacements. Pin source,
   runtime/model, attention/layout, TP, GPU/NIC allocation, transport tuning and
   workload. Separate cold, warm, partial-prefix and pressure cases. Record
   repeated-trial handoff latency, TTFT, ITL, throughput, tail latency, actual
   NIC/storage bytes, host/GPU copies, queue/readiness time and compute/transfer
   overlap. Set a numerical non-regression budget against the matched direct
   baseline before judging results; do not infer parity from correctness smokes
   or fewer requested objects. Preserve failures and reproduction logs on VPS.

#### Three-case benchmark contract

| Case | Fresh KV handoff | UMBP persistence and reuse |
| --- | --- | --- |
| Direct-only | Prefill HBM to decode HBM over MoRI-IO RDMA | Disabled: no pool PUT or GET |
| Direct-plus-offload | Direct RDMA, without waiting for pool persistence | Background PUT of selected reusable KV; pool reuse enabled for subsequent requests |
| Offload-only | Pool-mediated P/D: producer PUT, readiness, decoder GET | Enabled; direct HBM-to-HBM P/D disabled |

Here, offload-only names a P/D delivery mode, not a single-engine offload test
and not a ban on RDMA: the pool's cross-host transport may still use MoRI-IO
RDMA. These are proposed benchmark modes, not existing configuration flags.
The combined case must actually persist the selected KV; a run with no
background writes does not measure direct-plus-offload.

- **Fresh-handoff comparison:** use the same cold prompts, local-prefix state,
  hardware, model, parallelism, layout, request concurrency and transport
  settings. Direct-plus-offload versus direct-only isolates persistence cost;
  offload-only versus direct-only measures the storage-mediated handoff cost.
  Record physical bytes and actual direct/pool paths, not only configured modes.
- **Reuse comparison:** replay identical warm and partial-prefix workloads,
  including controlled GPU eviction/pressure, under the same cache policy.
  Verify pool publication before crediting a reusable copy; distinguish pool
  hits from GPU-local hits and report recomputed tokens. Direct-only cannot
  silently use a pool. For fresh ranges in the combined case, pool fallback
  must be reported separately, not counted as a successful direct fast path.
- **Storage subcases:** run DRAM-only and SSD-only pool variants for the two
  pool-enabled cases, with matched placement, capacities, admission/dedup policy
  and selected KV bytes. Keep producer/decoder-side persistence placement fixed
  within each comparison; treat any alternative placement as a separate sweep.
- **Account for asynchronous work:** report time until KV is reusable, completed
  versus failed/deferred PUTs, outstanding queue bytes, GPU source-retention
  time and host/GPU memory use, alongside handoff/TTFT/ITL, throughput, tails,
  copies and NIC/storage bytes. Measure sustained load as well as isolated
  handoffs. Record the end-of-window backlog and its drain time so deferred
  persistence is not hidden outside the benchmark, without charging that drain
  as synchronous decode latency. Preserve output/byte correctness in all cases.

See the [RFC's dual-path protocol](umbp_kvconnector_rfc.md#direct-rdma-and-pool-mediated-pd)
for the proposed public contract. These are planned tests and behavior, not new
validation results or authorization to run remote experiments.

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
| Native API and owned ranged storage | `umbp/store.py`; CPU contracts, then real native pointer tests | CPU contracts and eight native CPU/GPU DRAM/SSD buffer cases passed; broader pressure/fault validation pending |
| Explicit storage configuration | `umbp/config.py`; page/scratch checks, DRAM/SSD policy and private-path lifetime | CPU tests and native DRAM-only/SSD-only startup/cleanup passed; pressure and tier migration pending |
| Compatible P/D/offload keys | `umbp/key.py`, `umbp/layout.py`; descriptor integration and identity isolation | Worker-derived namespace handshake and generation exchange implemented; same-/two-host matching-TP1 P/D smokes passed, broader native topology pending |
| Scheduler lookup and allocation | KVConnector lookup, metadata, block ownership, asynchronous feedback | Dense offload and P/D planners exercised through real Scheduler; hybrid boundaries pending |
| Worker load/store and errors | Registered cache views, compute fences, completion snapshots and error block IDs | Real connector hooks and post-forward event recording wired; CPU runner tests pass, native GPU ordering pending |
| Pool-mediated P/D handoff and incremental transfer | Producer/consumer protocol, all-rank commit, local-prefix reuse, two-engine transfer accounting | Same-host DRAM and two-host DRAM/SSD Qwen cold/partial P/D passed; native multi-rank, pressure/fault and hybrid handoff pending |
| Direct HBM-to-HBM RDMA P/D | Shared MoRIIO transport, capability negotiation, fenced direct delivery and no mandatory pool write | Planned, not implemented; existing pool P/D results do not establish this path |
| Coordinated direct/pool delivery and offload | Disjoint missing-range ownership, independent background persistence, drain-before-fallback and aggregate completion | Planned; not enabled by an existing connector option |
| Three-case P/D effectiveness and direct-path preservation | Direct-only versus direct-plus-offload versus offload-only, plus unchanged MoRIIO reference; endpoint, copy, physical-byte, backlog and overlap evidence | Pending; no direct-path parity or speedup claim |
| Single-node DRAM and ext4 SSD offload | MoRI embedded deployment, forced eviction, byte/source reconciliation | Qwen local-cache-reset smoke passed at `2dc83e970`; forced HBM overwrite/pressure and performance pending |
| Multi-node DRAM/SSD restore | Master-led deployment, peer failures, safe recompute and recovery | Forced cross-host native buffer reads and P/D smokes passed; ordinary multi-node offload, pressure and peer/master faults pending |
| TP and hybrid cache geometry | Exact layouts, complete group/shard restoration, cancellation and preemption | Layout/rank barriers and unequal dense-group scheduler tests pass; native TP and hybrid boundary semantics pending |
| Placement and MoRI scheduling | Current authoritative placement API, lifecycle events, tier/locality/cost routing | Pending |
| CPU/HBM prefetch and admission | Token-identity control, connector-owned load, TTL/cancel/drain/no-model invariants | Pending |
| llm-d integration and fault recovery | Routing, prefetch, replay/gaps, staleness, reconnect/fail-open tests | Pending |
| Qwen correctness and accuracy | Matched no-connector/offload/P-D outputs and GSM8K evidence | One-prompt offload and same-/two-host P/D token matches passed; matched GSM8K acceptance pending |
| Kimi K3 acceptance | Requested TP8/AITER settings, hybrid layouts, long-context restore and semantic tests | Pending |
| Transfer effectiveness | Source counters, physical transfer proof, matched warmup/latency/throughput protocol | Pending |
| Reproduction and zero remote residue | Exact source/runtime/model manifests, VPS logs, audited cleanup | Native/offload and P/D attempts recovered to VPS with per-attempt cleanup audits; repeat for every run |

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
5. Implement and independently validate the direct RDMA P/D milestone above:
   shared MoRIIO transport, forced direct/pool modes, safe mixed-path ownership,
   background offload and failure fallback. Compare against unchanged MoRIIO
   before enabling automatic selection or claiming performance preservation.
6. Port placement-aware routing and prefetch through the connector contract;
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

At that checkpoint there was **no registered or usable UMBPConnector**. The next work was to
connect request planning, worker initialization/layout handshake and actual GPU
event creation to these components, then implement the P/D handle/readiness
protocol and missing-prefix-only loading. A transfer finalization is not a
published P/D readiness record. Hybrid boundary selection, request-scoped
non-prefix state, host-pool-qualified IDs, native integrity and model/accuracy/
performance evidence remain required. Do not treat unsupported configurations
as waived requirements or offer a serving command that does not yet work.

### Request-path and startup test design

Wire the existing job owner into actual KVConnector hooks. Inputs are real
Request hashes, allocated group tables, worker-derived layout identities and
SchedulerOutput; outputs are asynchronous lookup/load/store metadata and the
existing completion/error APIs. The cheapest composed test runs the real
Scheduler and CPU allocation/native-store fixtures through a real connector,
checking cold store, forced local eviction, missing-prefix-only restore, failed
receive recomputation, final-logit recompute, admission pressure and abort.
Extend the lifecycle suite and reuse core scheduler/request fixtures. Startup
tests must reject incompatible shards/configurations before any storage writes.
This step establishes the ordinary offload request path; P/D readiness, hybrid
boundary state and native serving remain required subsequent integration work.

## Fifth implementation checkpoint: offload request path and startup hooks

`umbp/connector.py` now implements `KVConnectorBase_V1` and the all-group finish
hook. The existing factory loads it using
`kv_connector_module_path=vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector`
and `kv_connector=UMBPConnector`. It is not added to the built-in registry and
has not been validated with a native vLLM/MoRI serving process.

Worker registration resolves the actual allocation layout, opens the owned
store, and registers the allocations before the existing startup handshake.
Workers report the namespace derived from their layout, explicit deployment /
model / revision identity, the resolved model-configuration hash and cache hash
algorithm. The scheduler requires all TP worker identities and job limits to
agree. Fresh per-worker nonces derive a new shared transfer epoch; worker or
engine IDs do not enter storage keys. Immutable weight revision is an operator
assertion, not a checksum of downloaded weights; verify that assertion at native
preflight. Model-config hashing is conservative and can cause misses across
otherwise equivalent configurations/paths; it must not be relaxed without tests.
Every metadata batch carries the nonce vector. Before binding even its first
epoch, each worker verifies its own fresh nonce and the derived epoch, rejecting
old engine metadata before it can launch I/O.

The worker uses `torch.Event(device=current_platform.device_type)` recorded
when the existing runner calls asynchronous `start_load_kv` after its forward
launch. Layer hooks perform no I/O, and `wait_for_save` does not block on native
completion. The prior job owner/finalization protocol protects allocations until
safe release. Actual ROCm event/zeroing/graph ordering is still a native test gate.

`umbp/planner.py` now selects real Request hashes and full cache records:

- Asynchronous lookup starts beyond the current local prefix and combines
  every rank's per-object results. A hole limits the hit to the common contiguous
  prefix, aligned across group block sizes. Read opt-out skips external lookup.
- Allocation reserves a job slot before promising asynchronous tokens. The
  existing scheduler owns the WAITING_FOR_REMOTE_KVS transition, publication,
  invalid-block recovery and final-logit recompute.
- Newly completed full blocks are pinned before post-forward stores. Save
  cursors avoid scanning the whole saved prefix on each decode step. Successfully
  restored objects are not immediately stored again.
- Lookup deadlines trigger ordinary recompute without waiting for native I/O
  to finish. They do not release native ownership. Aborted receives remain
  owned through finalization/error-snapshot retirement.
- Engine `has_requests()` keeps empty steps running through native completion
  and retirement even when no user request remains.

### Configuration surface at this checkpoint

This is a development interface, not an approved MI355X serving recipe.
Use `kv_role=kv_both` for the implemented ordinary offload path. The extra config
accepts only the following fields:

| Field | Contract |
| --- | --- |
| `deployment`, `model`, `revision` | Required nonempty compatibility identities; revision must identify the actual immutable weights |
| `storage` | Required `UMBPStoreConfig` dictionary described above; capacities and scratch are per worker |
| `nodes` | For a shared master, one explicit `UMBPNodeConfig` dictionary per TP rank; unique node IDs and peer endpoints |
| `lookup_timeout` | Positive finite seconds, default 5; bounds lookup/admission waiting, not native I/O lifetime |

Startup currently rejects P/D roles, hybrid/non-prefix/host-resident groups,
PP/CP, speculative decoding, non-causal attention, disabled prefix caching,
and KV formats requiring an unvalidated quantization-scale identity. Dense
FullAttentionSpec and MLAAttentionSpec are wired. These exclusions are temporary
implementation gaps, **not reductions of the required final scope**. In
particular, Kimi K3 hybrid acceptance is not established by this checkpoint.

### Evidence and next integration gate

`cpu-tests-r14.log`: **181 passed**, 15 expected warnings, 17.00 seconds.
This includes 178 UMBP tests plus three existing connector/output regressions.
The full offline command in the fourth checkpoint is unchanged and now includes
the extended lifecycle suite. To run just the request/startup tests:

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/vllmumbp/repos/vllm/.venv/bin/python -m pytest \
  --confcutdir=tests/v1/kv_connector/unit \
  tests/v1/kv_connector/unit/test_umbp_lifecycle.py \
  -k 'full_scheduler or connector_ or pool_reuse or new_worker' -q
```

The composed tests instantiate the actual factory, connector, Scheduler,
KVCacheManager/BlockPool and v1 runner collector. They demonstrate cold save,
local-cache eviction, byte-checked missing-prefix restore, chunked prefill,
unequal group sizes, lookup gaps, timeout recompute, failed receive recovery,
abort/drain, read opt-out and model-identity isolation between separate engines.
They use one CPU worker, synthetic forward bytes/tokens, a fake compute event,
and an in-memory native-API fake. The earlier two-rank and v2 collector tests
remain component evidence; this is not a real TP/GPU/model-accuracy result.
Inter-engine pool reuse in these tests is ordinary offload, **not P/D**.

Next implement the actual P/D handle/readiness record and consumer wait/fallback
through these hooks, including missing-prefix-only delivery. Hybrid exact
boundary/checkpoint and request-specific non-prefix state must then compose with
that protocol; PP/CP/speculative support and integrity checks remain required.
Native MoRI lifecycle, SSD/RDMA/TP, Qwen/GSM8K/Kimi, placement/routing/prefetch and
llm-d gates are unchanged. No remote state or model weights were created by this
checkpoint, and all evidence remains on the VPS.

### P/D protocol test design

Add producer finish-time export and consumer readiness-gated loads through the
existing connector jobs. Inputs are a versioned namespace/boundary-hash handle,
real cache records and all-rank receipts; outputs are ready markers, incremental
loads and the existing finished-sending/receive-error snapshots. Tests must catch
early readiness, missing ranks, failed stores, stale/mismatched handles, deadline
fallback, cancellation and source reuse before native completion. Reuse the CPU
worker fixtures and real Scheduler harness, with distinct prefill/decode engines
and the producer's actual EngineCoreOutput transfer parameters. Those tests do
not establish native/model correctness. MoRI's pinned Python binding has no
per-key delete/TTL: markers are immutable pool objects with logical handle expiry,
reclaimed by ordinary eviction/client teardown, not by clearing shared state.

Extend the existing two-worker job fixture to verify that a delayed final rank
prevents readiness and that any rank's export failure selects release rather
than publication. Use the real single-worker Scheduler pair for stale identity,
expiry, eviction after readiness, configured receive-error policy and aborts
during a blocked existence probe. The observable contract is no early GET,
no compute on unreceived KV, and no leaked/reused allocation ownership. These
tests need CPU buffers and controlled native barriers, not a new model harness.

## Sixth implementation checkpoint: dense P/D handoff

`umbp/protocol.py` defines a strict versioned handle with a compatible namespace,
producer generation, unique nonce, chained prefix boundary hash and expiry.
`umbp/pd.py` plans producer export and decoder receive through the same bounded
job ledger as ordinary offload. The connector uses existing request-finish
holds, worker metadata and finished-sending/receive-error APIs; there are no new
UMBP-only engine-core RPCs or synthetic prefetch requests in this checkpoint.

### Handoff and ownership

1. The producer finishes prefill and returns its handle in the actual
   `EngineCoreOutput.kv_transfer_params`. Returning `True` from the finish hook
   retains the source allocation. The handle is not a readiness claim.
2. Workers export complete aligned prefix objects after their compute fences.
   Only the scheduler's successful all-rank/all-object outcome admits a ready
   marker publication job. Failed or expired exports instead admit release.
3. Each producer worker publishes a one-byte immutable marker in its shard key
   space. Control-job finalization emits finished-sending on every rank, after
   which the normal Scheduler frees the held producer allocation.
4. The decoder validates namespace, schema, boundary hash, alignment and expiry,
   then allocates only the missing prefix beyond its valid local cache. Workers
   wait for their rank's marker and visibility of every requested missing
   object before issuing native GETs. A local marker can precede heartbeat
   publication of peer-held KV routes. The Scheduler does not execute a
   request while its receive is pending.
5. Receives use the existing all-rank finalization and error-snapshot retirement
   barriers. Missing/evicted objects and allocated-receive timeouts honor vLLM's
   `kv_load_failure_policy` (`recompute` or `fail`). Invalid hints detected before
   allocation are advisory misses and use local compute, not receive failures.

Existence probes carry no KV pointers. A cancelled/expired receive can retire
before a blocked existence call returns; the store still owns/counts that call
and drains it at shutdown. A running GET is different: cancellation or timeout
must not release its destination while native code may still write it.

### Development configuration and request contract

Use the existing module-path connector configuration from checkpoint five.
The additional options in `kv_connector_extra_config` are:

| Option | Contract |
| --- | --- |
| `enable_pd` | Boolean; defaults to true for `kv_producer`/`kv_consumer`, false for `kv_both`. Explicitly disabling it with a P/D-only role is rejected. |
| `handoff_timeout` | Seconds in `(0, 3600]`, default 30. Bounds producer handle acceptance and pre-GET readiness waiting, not running native memory access. |
| `storage.master_address`, `nodes` | P/D requires a shared master and explicit unique peer identity/endpoints per TP rank. Embedded offload remains supported without P/D. |

For the tested producer path, set `max_tokens=1` and request
`kv_transfer_params={"do_remote_decode": true}`. Pass the returned transfer
parameters unchanged to the consumer, along with the prompt **including the
producer's first sampled token**. The returned parameters set
`do_remote_prefill=true` and contain `umbp_handoff`; do not reconstruct or invent
that handle in a router. The CPU tests exercise EngineCoreOutput, not an HTTP
proxy. Native HTTP/proxy and llm-d integration remain separate validation work.

`kv_both` with `enable_pd=true` accepts per-request P/D flags while ordinary
requests still use offload. Enabling P/D on a producer-capable engine advertises
mandatory KV delivery to the Scheduler; per-request export still requires the
explicit flag. `kv_both` with P/D disabled keeps the prior offload behavior.

Export boundaries are rounded down to the least common multiple of dense group
block sizes. A 16-token prompt with 4-token blocks transfers at most 16 prefix
tokens and the decoder computes the producer's sampled token. For a 13-token
prompt, it transfers at most 12 and computes two tokens (partial tail plus the
sampled token). A valid decoder-local prefix reduces actual GET object counts.
This is not zero-recompute handoff for arbitrary hybrid or partial boundary state.
Independent engines also need matching hash algorithms and hash seeds: a shared
CPU test process does not validate cross-process hash initialization.

### Evidence and remaining gates

`cpu-tests-r16.log`: **199 passed**, 15 expected warnings, 18.27 seconds, using
the eight-suite offline command from checkpoint four. This includes 16 P/D
cases and ordinary offload with P/D enabled/disabled. To isolate the P/D cases:

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/vllmumbp/repos/vllm/.venv/bin/python -m pytest \
  --confcutdir=tests/v1/kv_connector/unit \
  tests/v1/kv_connector/unit/test_umbp_lifecycle.py -k test_pd_ -q
```

The two-worker component test proves that a delayed rank prevents publication
and any export failure selects release. Separate single-worker Scheduler pairs
prove handle-before-readiness ordering, byte-correct missing-prefix loads,
partial tails, unequal dense groups, invalid handles, post-readiness eviction,
recompute/fail behavior and aborts during blocked readiness probes. The tests
use real CPU buffers/cache managers and fake native I/O, GPU events and model
bytes/tokens. They establish neither real TP/P-D transport nor model accuracy.
The failed r15 run was a test-fixture argument error, corrected in r16; it is
retained as failed evidence. The earlier mypy fixture annotation is corrected.

MoRI remains unchanged at the pinned release. Its Python client lacks per-key
delete/TTL, so logical expiry does not reclaim ready objects. Native retention
under pressure, bounded marker metadata, teardown and ranged-SSD CRC integrity
still require proof or implementation changes. Never clear a shared pool to
clean up one request. Clock skew may reduce handle availability; no clock or
deadline grants permission to reuse incompatible KV or free in-flight buffers.

Native GPU/DRAM/ext4 SSD and two-node serving, Qwen/GSM8K/Kimi, hybrid/non-prefix
boundary semantics, parallel/speculative modes, routing and CPU/HBM prefetch
remain required. No native MoRI, remote host or model workload was used for this
checkpoint. All logs remain on the VPS; no remote residue was created.

### Native ranged-store test design

Extend the existing store suite with opt-in native cases requiring an explicit
`UMBP_NATIVE_TEST_ROOT` on the validated ext4 task mount. Exercise real MoRI
configuration/registration and ranged PUT/GET across CPU/GPU source and target
combinations for DRAM-only and SSD-only policy. Verify reordered full-object
assembly, unaligned partial reads, untouched destination padding, misses and
private-path cleanup after close. This is native storage-adapter evidence, not
full model serving, TP/P-D, corruption detection or direct-SSD fastpath proof.

### Native runtime preparation (2026-09-14)

The first native build uses the unchanged MoRI release commit
`67632e80e2e492184b589904b63225f82d45537c` and vLLM code checkpoint
`5e4e04707979189a3258e5f24213580abd16c901`. The eight opt-in tests are a separate
source overlay, SHA256
`4c723b125f838cdd3f1ef2a9ec869783f0e0d69f7d3eaeab146cdf9d11f31645`.
The complete CPU suite remains **199 passed, 8 skipped**, 15 expected warnings,
17.66 seconds (`cpu-tests-r17.log`); native tests are not counted as CPU passes.

The authorized 003 runner's existing image is
`vllm/vllm-openai-rocm:nightly-1dc464d42681d22f38caf1fdc1eb632dc4421c45`, image ID
`sha256:40e19c756e3dc9ffc9117770904d40376c7d3bf529cc76ddc379cde7ac4dae2d`.
It contains Python 3.12.13, glibc 2.35, ROCm 7.2.3 and Torch
`2.12.0+git6bbd260`, but its installed MoRI distribution is only 1.0.0.
The release's Python 3.12 wheel targets manylinux 2.39, so it is not used.

Build the pinned source against the image instead. Freeze its spdlog submodule
at `4a9ccf7e38e257feecce0c579a782741254eaeef` and msgpack-c at
`9b801f087ab7434f2ab1ab3c0f48a966c19d3b70`. Resolve Ubuntu Jammy build packages
on the VPS against the image's exported package status, then extract their DEBs
into a task-local dependency prefix; do not install packages into either OS.
This also avoids 003's broken default Docker bridge without reconfiguring it.

CMake explicitly enables UMBP and gfx950, disables SPDK, examples, C++ tests,
collectives and EP AOT compilation, and builds `mori_pybinds`, `umbp_master`
and `umbp_standalone_server` with eight compile workers. This is a UMBP test
runtime, not a claim that every MoRI feature was built. All 134 build steps
succeeded. The initial source-import check then failed on Torch's default cache
username lookup for container UID 1000. Explicit task-local
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` redirect compiler caches, but a
second unconditional Torch username lookup still needs a user entry. The next
runner attempt mounts a task-authored, password-free `/etc/passwd` read-only
with UID 1000 named `ubuntu`; the base image and host remain unchanged.
CMake reports hipFile absent and
the GDS engine disabled, so this runtime cannot validate direct SSD-to-HBM I/O.

All prepared scripts, frozen inputs, package hashes, failure transcripts and
recovered runtime archives are on the VPS under
`local-logs/umbp-kvconnector-reimplementation-20260914/`. In particular:

- `native-r1-manifest.md`: resource, isolation, mount and cleanup contract.
- `native-r1-input/build.sh`: full CMake command and source-import checks.
- `native-r1-input/native.sh`: source/GPU provenance assertions and native pytest.
- `native-r1-controller.sh`: preflight, source staging, isolated execution,
  evidence recovery, hash comparison and exact-target cleanup.
- `native-r1-build-2.log`: compiler/dependency evidence and initial import error.
- `native-r1-build-2-recovery.tar`: compiled runtime, SHA256
  `7e69200124bea49f0bd1ca6d9bc68fdc867729e218e0558593f1b7a62a2ca9d0`.

The controller uses a read-only, capability-dropped, non-root container with
task sources read-only and only its marker-owned SSD0 runtime directory writable.
GPU tests additionally expose only GPU 0's render node and KFD, with an existing
SSD-directory `flock` and fresh occupancy checks; no lock file is left behind.
The injected setup-failure test recovered matching evidence and removed its
remote root. Native build artifacts are recovered to the VPS before deletion,
not retained on 003. No model weights are needed or downloaded for this phase.

### Native ranged-store results

Attempt `native-4` passed the no-GPU source-import probe, then exposed one
gfx950 GPU and imported the connector from `/src/vllm` and MoRI's newly compiled
`/work/package/mori/libmori_pybinds.so`. **8 native tests passed**, 31 unrelated
store cases deselected, 15 expected warnings, 7.21 seconds. The matrix covers
CPU-to-CPU, CPU-to-GPU, GPU-to-CPU and GPU-to-GPU buffers through both DRAM-only
and SSD-only policy. It checks byte-exact full assembly, unaligned partial
loads, untouched padding, lookup/read misses and private SSD-path removal after
normal `store.close()`. The final task SSD test directory was empty.

The full transcript is `native-r1-native-4.log`; pytest XML is recovered from
`runtime/evidence/native-ranged.xml` in the corresponding recovery archive.
Earlier failed attempts remain recorded as failures, not native-test passes.
The source snapshot lacks vLLM's compiled extensions and version file, and the
import warnings are retained. These buffer tests do not require those vLLM
extensions; installing/building and checking them is mandatory before model
serving. MoRI's image distribution metadata still describes its old package:
the tested native release is identified by source commit, import origins and
compiled-library hashes, not that metadata.

To replay from the retained VPS bundle, use a new attempt name (the controller
refuses to overwrite an existing recovery archive):

```bash
cd /home/ubuntu/vllmumbp
set -o pipefail
task_bundle=/home/ubuntu/vllmumbp/local-logs/umbp-kvconnector-reimplementation-20260914
bash "$task_bundle/native-r1-controller.sh" native replay-1 \
  2>&1 | tee "$task_bundle/native-r1-replay-1.log"
```

This command rechecks the SSD mount, free space, device mapping and current GPU
occupancy, verifies the recovered build's hash, runs both no-device probes and
native tests, then recovers evidence before exact-root cleanup. It requires the
same existing image and prepared VPS inputs; it is not a public clean-machine
installer. Review the manifest before changing the host, image, paths or devices.

Remaining native gates include cancellation/drain under in-flight native I/O,
storage pressure, tier promotion/demotion, corruption detection, master/peer
failures, multi-node transfers, and real scheduler/model execution. This result
does not establish RDMA, GDS, TP/P-D serving, model accuracy or speedup. The full
Qwen/GSM8K/Kimi, hybrid/non-prefix, parallel/speculative and routing/prefetch
requirements remain unchanged.

### Full-engine Qwen smoke and SSD batch admission (2026-09-14)

The first full-engine attempt at `ea55a38003e11c26d9d7b4e6c9b646135ba8f231`
used the current Python source with the pinned image's precompiled vLLM
extensions. Their source commit is `1dc464d42681d22f38caf1fdc1eb632dc4421c45`,
not the current vLLM commit. Native sources/dependency requirements differ, so
this is limited runtime-compatibility evidence, not an exact-source native
build validation. MoRI is the newly built, unchanged 1.2.3.post1 release.

Qwen/Qwen3-0.6B uses the pre-existing snapshot
`c1899de289a04d12100db370d81485cdf75e47ca`, mounted read-only from its Hub tree.
TP1, BF16, eager execution, block size 16, 160 GPU blocks, max model length
2048, max sequences 4, max batched tokens 512 and GPU utilization 0.1 are
development settings, not a substitute for the Kimi TP8 acceptance recipe.
One 584-token chat prompt produces 15 greedy tokens; clear only the local
prefix cache with `reset_prefix_cache(reset_connector=False)` and repeat.
Require matching output token IDs, zero baseline cached tokens, and positive
external-cache hits and restored tokens in both offload arms. The configured
receive-failure policy is `fail`, so a failed GET cannot pass by recomputing.

The first attempt (`serve-r1-smoke-1.log`) passed the baseline, then failed
before connector serving when AITER copied 4,591,816,539 bytes of installed
JIT modules into a 4 GiB tmpfs. Its `get_user_jit_dir()` supports
`AITER_JIT_DIR`; revision 2 points that at task-owned SSD storage, seeds links
to immutable image modules, and sets `TMPDIR` to task-owned SSD storage too.
No host/image package, HOME, kernel or filesystem configuration is changed.
Revision 2 explicitly selects `ROCM_AITER_UNIFIED_ATTN` and LBHNC in every arm.
The global `VLLM_ROCM_USE_AITER=0` remains unchanged; it does not disable this
explicit attention-backend selection. Source/native import paths and image
module hashes are captured before GPU execution.

At `ea55a3800`, revision 2 produced valid negative SSD evidence:

| Arm | Cold cached tokens | Restored tokens | Output matches baseline |
| --- | ---: | ---: | --- |
| No connector | 0 | 0 | Yes |
| DRAM only | 0 | 576 | Yes |
| SSD only, before batching fix | 0 | 0 (receive failed) | No output |

SSD lookup found 576 tokens, but the 36-object GET exceeded the 16-page
staging arena. Native `SsdBackend::BatchResolve` correctly rejects a working
set larger than the whole arena as non-retryable. Thus external lookup hits
alone are not successful-transfer evidence. The complete run failed and was
recovered in `serve-r2-smoke-1-recovery.tar`, SHA256
`6642b4ee0023ccb350ca23ee912a776881db77009d0f89052166446eec045dd1`, before
marker-owned cleanup. The earlier passing DRAM arm is not an SSD pass.

The correction splits native GET/PUT calls at whole-object boundaries using
`ssd_staging_slots` as the maximum objects per call. `open_store` already
requires each KV object to fit one page. Even DRAM-only clients that may read
remote SSDs must use a common bound no larger than the smallest peer arena;
automatic remote-capacity negotiation is not implemented. Lookup batching is
unchanged. One future/admission slot covers every chunk, keeps all registrations
and scheduler-owned blocks alive, preserves ordered per-object outcomes and
fails on a malformed chunk before submitting later chunks. There is no early
publication or increased SSD staging allocation.

Regression tests extend the existing store/config suites: multi-chunk byte
correctness with a middle missing object, later-chunk failure, cancellation/
shutdown while a chunk is running, configuration-to-native batch bounds and
invalid batch limits. Use the eight-suite CPU command above with `--confcutdir`;
the corrected invocation passed **204 tests, eight native cases skipped** in
18.27 seconds (`cpu-tests-r18.log`).
Omitting that option on this CPU-only VPS triggers the unrelated global GPU
cleanup fixture, as retained in the first local batching-test logs.

The pinned SSD backend retains read staging under a default 3-second lease
and retries transient arena pressure. Chunking can therefore wait between
reads. This fixes per-call geometry, not transfer QoS, fairness, lease-release
efficiency or latency; those need separate native pressure/performance work.
The matched Qwen revalidation below exercises the correction with native SSD
storage. GSM8K, eviction pressure, TP/P-D, hybrid models and two-node acceptance
remain outstanding.

The VPS evidence bundle contains `serve-r2-manifest.md`,
`serve-r2-controller.sh`, `serve-r2-input/`, source/module hashes, the injected
cleanup-failure receipt and full logs. The exact original comparison command is:

```bash
set -o pipefail
bash local-logs/umbp-kvconnector-reimplementation-20260914/serve-r2-controller.sh smoke smoke-1 \
  2>&1 | tee local-logs/umbp-kvconnector-reimplementation-20260914/serve-r2-smoke-1.log
```

That attempt is finished; do not overwrite it. The controller requires a fresh
attempt name and an absent, marker-owned root. No P/D, GDS or speedup claim is
made from this local-prefix-reset smoke.

#### Verified batching correction: Qwen revision 3

At exact Python-source commit `2dc83e970aa6c2ae0f7df3e3d2456694c7d0cb0f`,
all three arms passed with the same model, GPU 0 on 003, image, MoRI library,
attention backend/layout and resource budgets as revision 2. The source archive
SHA256 is `22d2bca1422789f3f57d5f8370a759cbc2836bbc707c5e6f6fbf87334f9b5c70`;
it excludes the uncommitted RFC. The older-image vLLM native-binary limitation
described above still applies.

| Arm | Cold cached tokens | Restored tokens | Output matches baseline | Repeat request wall time |
| --- | ---: | ---: | --- | ---: |
| No connector | 0 | 0 | Yes, 15 greedy tokens | 0.144324 s |
| DRAM only | 0 | 576 | Yes, 15 greedy tokens | 0.155199 s |
| SSD only, bounded batches | 0 | 576 | Yes, 15 greedy tokens | 6.463459 s |

The prompt contains 584 tokens. Local prefix-cache hits are zero throughout;
external hits are zero for cold requests and 576 for both offload repeats.
Every repeat returns the same token IDs as its cold request and the baseline.
This is a limited native full-engine offload correctness pass. Local cache reset
does not establish that old HBM bytes were overwritten; actual pressure/eviction
and separate-engine P/D remain required. No GSM8K or throughput result is implied.

SSD's 36-object GET becomes 16/16/4 calls at unchanged staging capacity. Its
6.46-second repeat is consistent with two turnovers of the 3-second native read
leases, but no profiler attribution was collected. Do not treat these single
request times as a performance benchmark or shorten leases without proving
transfer lifetime safety.

Final CPU evidence is `cpu-tests-r19.log`: **204 passed, eight native tests
skipped**, 15 warnings, 18.21 seconds. This supersedes r18 as the final local
rerun; the eight earlier passing native buffer tests remain separate evidence.

The exact completed comparison command, run from `/home/ubuntu/vllmumbp`, was:

```bash
set -o pipefail
bash local-logs/umbp-kvconnector-reimplementation-20260914/serve-r3-controller.sh smoke smoke-1 \
  2>&1 | tee local-logs/umbp-kvconnector-reimplementation-20260914/serve-r3-smoke-1.log
```

Do not rerun that evidence name. The controller refuses an existing recovery
archive. A future replay needs a fresh attempt name plus fresh GPU, filesystem,
model and cleanup checks. Frozen inputs and provenance are in
`serve-r3-input/`, `serve-r3-manifest.md` and `serve-r3-launch-hashes.txt` within
`local-logs/umbp-kvconnector-reimplementation-20260914/`.

The controller finished with exit 0, `qwen_smoke_private_paths_removed=true`
and `cleanup_complete=true`. The entire runtime was recovered as
`serve-r3-smoke-1-recovery.tar.gz` (769,471,616 bytes), SHA256
`1d957f08a351e61b27ff17037f493df2d405929e1ff1ffeb7bf988d0e6e164e8`, and
compared with the deterministic remote stream before marker-owned cleanup.
The archive omits non-file IPC sockets, as reported by tar. The injected
setup-failure gate had already proved recovery/cleanup with expected exit 42.

Independent verification, also run from the VPS workspace:

```bash
set -o pipefail
/home/ubuntu/vllmumbp/repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/verify-qwen-smoke.py \
  local-logs/umbp-kvconnector-reimplementation-20260914/serve-r3-smoke-1.log \
  local-logs/umbp-kvconnector-reimplementation-20260914/serve-r3-smoke-1-recovery.tar.gz \
  2>&1 | tee local-logs/umbp-kvconnector-reimplementation-20260914/verify-qwen-r3.log
```

The verifier passed, comparing all six log receipts with recovered JSON,
expected tokens/cache counters, actual attention/layout markers and cleanup.
Its negative control rejects the r2 lookup-hit-but-failed-GET case. See
`verify-qwen-r3.log` and `verify-qwen-negative-control.log`.

The independent 14:33 UTC audit (`serve-r3-final-audit.sh` and its `.log`)
confirmed the exact r3 root and all run-labelled Docker resources absent, no
KFD processes, all eight GPUs at baseline VRAM, and the directory lock available.
The original image is preserved. The pre-existing Qwen weight/tokenizer sizes
and config/tokenizer-config hashes match; no incomplete files exist. No weight
download, host configuration change or work on 004 occurred. All recovered
runtime, logs and scripts remain on the VPS, not on the GPU host.

## P/D native validation and visibility correction (2026-09-14)

The test uses a release-built MoRI master, separate native clients and separate
Qwen producer/consumer engines on GPU0/GPU1 of runner003. The exact model
revision, image/native-library pins, runtime toolchain and older-native-vLLM
binary caveat are the same as the offload smoke above. The new engine source
is `efac63da24ddba5374d4a55ad30923c400c239a4`; MoRI remains at
`67632e80e2e492184b589904b63225f82d45537c`.

### Failed attempts and transport-only diagnostic

All paths in this section are relative to the VPS bundle
`local-logs/umbp-kvconnector-reimplementation-20260914/`.

| Attempt | Actual result | Recovery SHA256 |
| --- | --- | --- |
| `pd-r1-smoke-1` | Master loader failed to locate `libmori_io.so`; no model or native-client test | `47139291acd4fa80951253d60680e5e6fce6a511a3bd5912574ed2f8a0e5426a` |
| `pd-r2-smoke-1` | Loader fixed; first native PUT failed with ionic_0 RTR transition errno19 under network-none; no model P/D | `97314eadca602a83df0fe483ceba5237d079cbc23c538da11ad63466a077f7e3` |
| `pd-r3-transport-1` | Host-network native cross-process PUT/consumer-local GET passed; all 4096 bytes verified; no model mounted | `c51dd2d4ca28c133895ec31166cd20a4234f65d24c83bd333195358cb0860bf3` |
| `pd-r4-smoke-1` | Native transport and baseline passed; producer emitted handle576/first token, consumer KV load failed, partial case not reached | `ca79b391d69002d876ac10b9739edf6f4840a917b39870b69eeaf0eaaac8db8a` |

Each attempt recovered its entire runtime as `<attempt>-recovery.tar.gz`,
hash-checked before exact marker-owned deletion. The independent
`pd-r1-final-audit.log` through `pd-r4-final-audit.log` passed. The r4 archive
is 769,479,024 bytes; its four receipts correctly fail the seven-receipt model
verifier (`verify-pd-r4.log`). Failed GETs are not accepted based on an external
hit counter. r4 used distinct producer/consumer GPU UUIDs and `failure_policy=fail`.

The standalone master needs `/work/package/mori` on `LD_LIBRARY_PATH` in
addition to the relocated protobuf/gRPC libraries. A no-GPU `ldd` check now
precedes execution. The successful transport shape uses host networking within
the user's existing recipe scope, not host interface/kernel/firewall changes.
Master port 48700 and native I/O ports 49201/49202 bind loopback; peer ports 49101/49102 bind
all interfaces. Metrics are disabled. This is not network isolation or a
single-variable proof that network namespace alone caused the r2 failure.

### Readiness visibility regression

In the pinned native release, a successful PUT commits bytes but can precede
heartbeat delivery of the object's route to the master. `BatchExists` is
local-first. A ready marker placed on the decoder can therefore be visible
before producer-held KV is routable from that decoder. The former marker-only
gate allowed an early GET; r4 failed before the next heartbeat.

Checkpoint `efac63da2` gates P/D GETs on the rank marker **and every missing
object in that receive job**. It uses the same bounded asynchronous lookup,
backoff and deadline; no fixed sleep, heartbeat tuning or GET retry. Objects
can still be evicted after the probe, so normal native GET outcomes and receive
failure policy remain authoritative. No ordinary offload path changes.

Test design: extend the existing composed Scheduler fixture to expose the
marker while withholding data visibility. Require no GET/computation until
the data becomes visible, then one successful receive and no leaked blocks.
Keep separate eviction-after-probe coverage for both fail and recompute.
`pd-visibility-negative.log` shows the new test failing on the old code because
GET ran early. `cpu-tests-r20.log`: **205 passed, eight native tests skipped**,
18.34 seconds, using the eight-suite offline command above. Applicable code
and commit hooks passed (`pd-visibility-hooks.log`, `pd-visibility-commit.log`);
unrelated actionlint was skipped. The local code commit contains only the
worker gate and its tests; the existing design drafts were preserved.

Native retry 5 is separately frozen in `pd-r5-manifest.md`,
`pd-r5-launch-hashes.txt`, `pd-r5-controller.sh` and `pd-r5-input/`.
Source archive SHA256:
`b14bf8f7c753c40f216bf3e689c1b191c576d5717d6a1588f6654b735b58c3be`.
The setup-failure gate passed with expected exit 42 and cleanup true, recovery
SHA256 `a8bd2af195055a77ebfd0fd82f36d851c815bdd593edbb46c56fe7fa269affa2`.
The native retry result was independently verified as described below.

### Verified separate-engine Qwen P/D: retry 5

`pd-r5-smoke-1.log` completed with exit 0, all case markers, private-path removal
and controller cleanup. `verify-pd-r5.log` independently matched all seven
receipts against the recovered JSON, checked three actual backend/layout
selections and distinct producer/consumer GPU UUIDs, and confirmed the exact
token/count checks. Both producer and decoder use `LLM` engine APIs, not an
HTTP server or llm-d router. Both are on host 003; this is not a multi-node pass.

Common model/engine settings: pre-existing Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, TP1/BF16/eager, block size 16,
160 GPU blocks, max length 2048, max sequences 4, max batch tokens 512, GPU utilization 0.1,
`ROCM_AITER_UNIFIED_ATTN` and LBHNC. Native storage is 512 MiB DRAM per client,
4 MiB pages, 64 MiB GET and PUT arenas, 16-object batches; receive policy is fail.
The task/runtime caches use the bound ext4 SSD0; this run does not test SSD KV.

| Case | Decoder local tokens | Decoder external tokens | Native decoder GET chunks | Total GET objects | Output |
| --- | ---: | ---: | --- | ---: | --- |
| Cold P/D | 0 | 576 | 16 / 16 / 4 | 36 | First producer token + 14 decoder tokens match baseline's 15 IDs |
| Partial-prefix P/D | 256 | 320 | 16 / 4 | 20 | Same 15 IDs; valid local prefix retained |

The original prompt has 584 tokens. Producer emits its actual boundary-576
handle and first token; decoder receives the unchanged handle plus 585 prompt
tokens and reports 576 cached tokens. Separate salts isolate the cases. For
partial P/D the decoder first warms 257 tokens, retaining 256 full local tokens;
the producer also reuses those 256 from the pool before exporting its complete
aligned prefix. This is not a second cold-prefill comparison.

The baseline request took 0.758 s; cold and partial decoder requests took 3.064 s
and 4.612 s respectively. These are un-warmed single-request wall times, not a
benchmark or speedup. The readiness delay aligns with the default 5-second
heartbeat publication cycle; no latency attribution/profile or heartbeat
tuning was performed. Native startup also reported repeated empty-heartbeat
sequence-gap/full-sync warnings, retained in the log for follow-up.

#### Native count evidence and failed byte-accounting gate

`pd-r5-get-evidence.log` reconciles the decoder's five native ranged GET calls
with the cold/partial request windows: 36 versus 20 objects. Each object is
1,835,008 bytes by the registered layout/PUT log, so expected logical payloads
are 66,060,288 and 36,700,160 bytes. These are derived payload sizes, **not measured
physical traffic or validated native byte-counter totals**. Cold reads used 18
local and 18 remote objects; partial reads used 20 remote objects. Different pool
placement means fewer total GET objects does not imply fewer network bytes.

The strict `verify-pd-ranged.py` byte gate **failed**, retained in
`verify-pd-r5-ranged.log`; it was not relaxed into a pass. Native debug totals
were 33,030,144 bytes for cold GETs and 0 for partial GETs. Source inspection of
the unchanged release's `ServeWholeObjectUnitsFromMedium` shows its
`remote_bytes` output parameter is never updated, unlike the arena path.
That path can land full remote objects in a local medium slot and copy to GPU
even with `cache_remote_fetches=false`; that flag is not a blanket prohibition
on this ranged-read placement. Native byte metrics/admission/copy accounting
need separate fixes or instrumentation before effectiveness claims.
`summarize-pd-get-evidence.py` verifies only object counts and explicitly emits
`strict_byte_gate_passed=false`. Token-level smoke acceptance is separate.

#### Exact commands and teardown

From `/home/ubuntu/vllmumbp` on the VPS, after the recorded failure gate:

```bash
set -o pipefail
bash local-logs/umbp-kvconnector-reimplementation-20260914/pd-r5-controller.sh smoke smoke-1 \
  2>&1 | tee local-logs/umbp-kvconnector-reimplementation-20260914/pd-r5-smoke-1.log
/home/ubuntu/vllmumbp/repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/verify-pd-smoke.py \
  local-logs/umbp-kvconnector-reimplementation-20260914/pd-r5-smoke-1.log \
  local-logs/umbp-kvconnector-reimplementation-20260914/pd-r5-smoke-1-recovery.tar.gz \
  2>&1 | tee local-logs/umbp-kvconnector-reimplementation-20260914/verify-pd-r5.log
bash local-logs/umbp-kvconnector-reimplementation-20260914/pd-final-audit.sh r5 \
  2>&1 | tee local-logs/umbp-kvconnector-reimplementation-20260914/pd-r5-final-audit.log
```

Do not rerun the completed evidence name: select a fresh attempt name and repeat
the preflight/frozen-manifest/failure-cleanup gates. The controller records
exact mounts, source paths and runtime commands; the manifest defines all
expected outcomes without a recompute fallback.

Recovery archive `pd-r5-smoke-1-recovery.tar.gz`: 769,513,919 bytes, SHA256
`858a2209f741341071aceb6b2dff4a919e3bc9eed461f818bf3db487f5fbfe83`.
The controller compared the local archive with a second deterministic remote
stream before deleting `/mnt/umbp-ssd0/umbp-kvc-pd-20260914-r5`. Dead IPC sockets
were omitted by tar with warnings; all file evidence remains on the VPS.
Independent audit at 15:28 UTC confirmed the root, labelled Docker resources,
task listeners and KFD processes absent, all GPU memory back to baseline,
directory lock available, original image preserved and pre-existing model
size/config/tokenizer checks unchanged. No weights were downloaded; the protected
`/shared_vllm/huggingfacehub/models--Qwen--Qwen3-0.6B` tree remains intact. Host 004
was not used. GEAK evidence/cleanup guidance informed the run; no GEAK agent,
authentication, host kernel or filesystem provisioning was involved.

Remaining gates include native multi-node/TP and SSD P/D, forced HBM/storage
pressure, lifetime/fault/corruption cases, hybrid/Kimi state, exact-current
vLLM native binaries, matched GSM8K, routing/prefetch/llm-d and performance.
Neither the same-host smoke nor 205 CPU tests complete the implementation goal.

## Two-host native validation (2026-09-14)

This experiment retains Python source `efac63da24ddba5374d4a55ad30923c400c239a4`
and unchanged MoRI release `67632e80e2e492184b589904b63225f82d45537c`.
It uses physical GPU 0 on each authorized MI355X host, one TP1 Qwen engine per
role, the same pre-existing read-only model revision, and separate DRAM-only
and ext4 SSD-only arms. Runtime/JIT/SSD scratch is under each host's SSD0 run
root; the shared NFS model cache is not used for KV scratch. There is no model
download, GDS/SPDK enablement, filesystem provisioning or kernel change.

### Test contract and environment preparation

The native gate completes a 4096-byte PUT on 003 **before** registering the
consumer on 004. The consumer then poisons its destination and verifies every
received byte; this tests a cross-host GET, not the earlier consumer-local GET.
For each medium, a fresh no-connector engine supplies the 584-token-prompt,
15-output-token baseline. Separate producer/consumer engines then exercise
cold P/D (576 external/0 local tokens) and a 256-token local-prefix case
(320 external/256 local). The actual producer handle and first token are
forwarded unchanged. Output IDs must match the complete baseline exactly,
with no failed-GET recomputation fallback.

The master is on 003; native peer/I/O services use each host's management IP
and `ionic_0`. Host networking is explicit: native listeners are reachable on
deployment interfaces, not isolated to the Docker network. Only the selected
GPU render node, KFD and one verbs device are exposed. The controller uses
SSH stdio for actor commands/results; no extra application command server or
runner credentials are installed in the containers. Read-only root/source/model
mounts, capability dropping, fixed UID/GID, bounded resources, immediate GPU
checks and existing-directory cooperative locks match the prior safety model.

Keep the same pinned image contents across hosts. The nightly tag
`vllm/vllm-openai-rocm:nightly-1dc464d42681d22f38caf1fdc1eb632dc4421c45`
exists on 003 but was absent on 004. A registry tag pull returned not-found,
while read-only manifest inspection proved that the exact immutable digest
remains available. The fourth frozen attempt pulls by digest and restores
the original tag on 004:

```bash
docker pull vllm/vllm-openai-rocm@sha256:40e19c756e3dc9ffc9117770904d40376c7d3bf529cc76ddc379cde7ac4dae2d
docker image tag \
  vllm/vllm-openai-rocm@sha256:40e19c756e3dc9ffc9117770904d40376c7d3bf529cc76ddc379cde7ac4dae2d \
  vllm/vllm-openai-rocm:nightly-1dc464d42681d22f38caf1fdc1eb632dc4421c45
```

These commands are inside the marker-owned lifecycle, not standalone cleanup
instructions. Both Docker daemons are 29.7.1 and expose the manifest digest as
the image ID. The controller verifies that identity before execution. No
pre-existing image needs deletion: 004's Docker filesystem had 167.8 GB free.
Its existing ext4 loop-backed Docker filesystem uses an NFS backing file;
the experiment neither moves it nor changes the daemon. The added tag/digest
on 004 must be removed after recovery; the original image on 003 is preserved.
Public images cannot be relabelled without changing identity, so the exact
baseline-absent tag/digest and ownership marker control this image cleanup.

Preparation failures remain evidence, not serving passes:

- Attempt 1: `setsid` forked and its parent returned before the remote script
  remained attached to SSH. Neither image installation nor model execution ran.
- Attempt 2: `setsid --wait` fixed supervision; live no-GPU launcher termination
  was added to the failure gate. 003 passed source/import probes, but 004's
  registry tag pull failed. No native or model actor ran.
- Attempt 3: a bounded image relay through the VPS was explicitly cancelled
  after the exact registry digest was found. The controller exited 130 and
  its transfer SSH processes terminated; no model actor ran.

All three attempts recovered their task runtime state to the VPS, compared
SHA256 against a second remote stream, and removed their exact run roots.
Independent audits passed against the original image/container inventories,
model snapshot checks, GPU memory/process baselines and task port/path checks.
The verifier rejects the first failed attempt rather than accepting partial
setup evidence.

### Fourth attempt: three passing cases, SSD partial receive failed

The two-host experiment at source `efac63da2` reached actual model execution
on 003 and 004 using the same pinned runtime. Its full acceptance gate failed;
do not label this a complete multi-node pass. The observed cases are:

| Storage | Decoder case | External/local tokens | Decoder wall time | Result |
| --- | --- | --- | --- | --- |
| DRAM | Cold | 576/0 | 2.864 s | Baseline token IDs matched |
| DRAM | Partial prefix | 320/256 | 3.437 s | Baseline token IDs matched |
| SSD | Cold | 576/0 | 7.519 s | Baseline token IDs matched |
| SSD | Partial prefix | 320/256 planned, not completed | 119.798 s | Error, zero output tokens |

Each successful P/D response concatenates the producer's first token and the
decoder's 14 tokens to match the 15-token no-connector baseline exactly.
No-connector request times were 0.769 s in the DRAM arm and 0.174 s in the SSD
arm. These are single smoke observations, with JIT/readiness/storage waiting
and different warmup histories, not matched performance measurements.

Before either model arm, a native producer PUT on 003 completed before the
consumer on 004 registered. A poisoned CPU destination on 004 then verified
all 4096 bytes. Native GET logs show `local=0 remote=1` for each storage mode;
this establishes a cross-host native data transfer, unlike the earlier
consumer-local transport probe. It does not repair the native byte-accounting
gap or establish network bandwidth. The SSD backend explicitly fell back from
unavailable `io_uring` to POSIX; no GDS/SPDK path was tested.

DRAM decoder native GET batches contain 16/16/4 objects for cold and 16/4 for
partial. SSD decoder GETs contain only the cold 16/16/4 batches: no partial
GET was issued before its readiness deadline. The SSD partial receipt reports
`finish_reason=error`, no output IDs and zero completed cached tokens, despite
the scheduler's counters advertising 320 external and 256 local tokens.
Admission counters are not evidence of completed transfers.

The SSD producer logged data PUTs but only one ready-marker PUT across the two
exports. The second export therefore lacks a logged ready publication. Its
per-object PUT return values and scheduler export outcome were not captured,
so the logs do **not** yet establish which object or native condition failed.
Staging-arena-busy and heartbeat-sequence warnings are preserved as diagnostic
evidence, not asserted as the root cause. Keep the failed case and the original
120-second deadline; add per-object outcome evidence before a targeted retry.
Never bypass readiness or silently recompute to turn this gate green.

The original `verify-multinode-r1.py` still requires all four passing cases.
The separate `summarize-multinode-r4-results.py` checks the three successful
cases **and requires the observed SSD failure**, including missing partial GETs
and the error receipt. A successful evidence report is not a successful model
acceptance gate.

Independent verification finished: the unchanged full-arm verifier exited 1
at token equality for SSD partial P/D. The partial-evidence reporter exited 0
with `partial_evidence_verified=true` and `multinode_pd_smoke_verified=false`.
It compared every actor RPC receipt with the recovered JSON, checked both
archive and native-library hashes, actual image/import/backend/layout markers,
distinct host GPU UUIDs, handoff/token chains, cache-counter deltas and native
GET/ready-PUT counts. This preserves the failed gate while independently
substantiating the three passing cases.

The controller terminated with exit 1 at the SSD partial token-equality
assertion. Both recovery archives were verified against a second remote
SHA256 stream before deleting the task roots:

| Host | Recovered bytes | Archive SHA256 |
| --- | --- | --- |
| 003 | 769,516,820 | `83c9491158e1061aa32f6a6cef8362e3207954500aeec69c35dfcf250fb1c813` |
| 004 | 769,396,766 | `f70b750c8fe6c7f96defc344fc123dda17ea94d4e25e4361734364d3873f7761` |

Independent post-clean audits passed on both hosts. They checked absent task
roots, process working directories, labelled Docker objects and task listeners;
no KFD processes; baseline VRAM on all eight GPUs; SSD0 mount identity; unchanged
pre-existing model snapshot checks; and exact pre-run Docker image/container
inventories. A separate read-only check found no surviving Docker image
save/load processes from image preparation. 004's task-created image/tag/digest
was removed; 003's pre-existing image was preserved. No weights were downloaded
or removed. GEAK's recovery-before-deletion contract governed teardown.

### Two-host commands and evidence

All paths below are on the controller VPS under
`/home/ubuntu/vllmumbp/local-logs/umbp-kvconnector-reimplementation-20260914`.
The frozen launch commands were:

```bash
cd /home/ubuntu/vllmumbp
sha256sum -c local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r4-launch-hashes.txt
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r4-controller.py failure gate1
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r4-controller.py smoke smoke1
```

The injected failure gate intentionally exits 42 after testing live no-GPU
launcher termination and verified cleanup. The smoke exits 1 for the model
failure described above. Do not overwrite the existing `gate1`/`smoke1`
directories: a new run requires fresh attempt names, runner preflight and
review of its immutable manifest. Do not run two attempts concurrently.
The controller owns remote setup, execution, recovery and exact cleanup;
do not launch its container fragments independently.

The read-only evidence/cleanup commands are:

```bash
cd /home/ubuntu/vllmumbp
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/verify-multinode-r1.py \
  local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r4-smoke1
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/summarize-multinode-r4-results.py \
  local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r4-smoke1
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r4-audit.py \
  local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r4-smoke1
```

Each invocation was logged through `tee` with `set -o pipefail`, preserving its
actual exit status. Full logs are `multinode-r4-failure-gate1.log`,
`multinode-r4-smoke1.log`, `verify-multinode-r4.log`,
`multinode-r4-case-report.log`, `multinode-r4-final-audit.log` and
`multinode-r4-image-transfer-process-audit.log`. Per-actor logs, JSON receipts,
SSH RPC envelopes and both recovery archives are in `multinode-r4-smoke1/`.

Frozen identities:

- Controller: `3895fe850fefde0ed5b117cd1b0e0cf694fa519477d364e004f9483a9c504ad8`.
- Host lifecycle: `964183bc372eb73b7fbc9b2f6b6b2585ca20124ca9f1763a713dd9ff71c1bc36`.
- Manifest: `46b45302257ac27dae084c614d3f17fea4a24e7d818ce4385d0c59d82446ae77`.
- Strict verifier: `4204d6dac4e0eacde64b8cb45a8a18679d91bc0a3ca47960bb705fef92fe0ceb`.
- Partial-evidence reporter: `52808f114426e63dc25eaa89469b5ac9bb3c2de5adb626a6bf7cf3da70e8adc8`.
- Cleanup auditor: `cf97af66e9c693657a57a2e5917d9ce3ac94b9ee3f2b09d7a1ed13a79a60d7c1`.

The complete per-input hashes are in `multinode-r4-launch-hashes.txt`. The source
archive is still `efac63da24ddba5374d4a55ad30923c400c239a4`, paired with MoRI
`67632e80e2e492184b589904b63225f82d45537c` and native library
`d2c9b02ed7da9a727e43f1f8155b97b7fb7fea71e6fd9ac31e09cc9a47702c60`.
No implementation source or model weight changed in this two-host attempt.
The earlier-image vLLM native-extension limitation remains: this is current
Python connector code with precompiled native extensions, not an exact-current
vLLM native build. llm-d and mori-sched do not participate in these offline
`LLM` tests; their previous repository pins above remain provenance only.

## Native SSD staging pressure and bounded PUT retries

A native-only diagnostic on 003 reproduced read-to-write staging pressure at
the unchanged `efac63da2` source and pinned MoRI binary. A fresh singleton peer
used the same 4 MiB pages and 16 staging slots as the model experiment. It
stored 16 distinct 4096-byte objects, passed a new-key write-only control, then
read all 16 into a poisoned buffer and verified all 65,536 bytes. A subsequent
new-key PUT failed 62 times before succeeding on attempt 63 at 3.112 seconds.
The DRAM control succeeded on its first attempt at 0.000175 seconds. Both arms
verified a final 4096-byte GET into poisoned bytes.

The native source retains SSD read staging leases for 3000 ms, sharing the arena
with writes. The diagnostic's factory log confirms 16 staging pages and POSIX
SSD I/O. This demonstrates a native transient failure mechanism; it does not
retroactively identify every failed object in the earlier model export, whose
native return values were not logged. Diagnostic retries were an intervention
in the harness, not a passing production connector result.

Independent `verify-pressure-r1.py` compared RPC envelopes with recovered JSON,
checked the archive/native-library hashes, source/import markers, native
GET/PUT counts and both control outcomes. `verify-pressure-r1.log` reports
`native_pressure_evidence_verified=true` and `model_fix_verified=false`.
No model was mounted, downloaded or executed. All user buffers were CPU
allocations; the established native runtime retained the selected GPU0/verbs
device exposure. The native binary and vLLM source archive were unchanged.

The pressure controller exited 0. Its recovery archive has 671,241,695 bytes and
SHA256 `053098e5dd644789136b1cffe4497a658db638a468771292e138cbe4c78afedb`.
The preceding live no-GPU failure gate exited 42 and recovered SHA256
`ff88e8ed76507c1446fbdf029545ca1fb1c1ad8a8b5a2c5131cb3d768a3745ab`.
003's task root/container were removed only after verification; independent
cleanup audit passed, preserving original Docker inventories and model checks.
004 was not used by this probe. The image was pre-existing and preserved.

The frozen manifest, launch hashes, controller and native probe are
`pressure-r1-manifest.md`, `pressure-r1-launch-hashes.txt`,
`pressure-r1-controller.py` and `pressure-r1-input/pressure.py` in the same VPS
bundle. Commands executed from `/home/ubuntu/vllmumbp`, with pipefail/tee logs:

```bash
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/pressure-r1-controller.py failure gate1
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/pressure-r1-controller.py smoke probe1
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/verify-pressure-r1.py \
  local-logs/umbp-kvconnector-reimplementation-20260914/pressure-r1-probe1
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/pressure-r1-audit.py \
  local-logs/umbp-kvconnector-reimplementation-20260914/pressure-r1-probe1
```

### Retry contract at `408c3b752`

Commit `408c3b752eaf18dd9c055d9ab5bac3208c5071df` adds
`storage.store_retry_timeout_s`, default 5 seconds, finite range 0–60; zero
disables retries. The low-level directly constructed `UMBPStore` defaults to
zero; the connector's configuration factory supplies its configured budget.
DRAM-only peers also receive the setting because they may write remote SSDs.

Only failed immutable PUT objects are retried. Successful objects are not
resubmitted. One monotonic budget is shared across all chunks of an operation,
with 10 ms initial backoff doubling to 100 ms. Native exceptions and malformed
results remain failed futures, not retry candidates; GETs are not retried.
Because the pinned API returns only booleans, it cannot distinguish transient
capacity failures from other false outcomes. Persistent false results consume
the bounded budget and remain failures; no success is inferred from retries.

The future retains source buffers and scheduler ownership through all retries.
Shutdown wakes retry waits, then drains native calls before deregistration.
This budget does not bound a running native call, extend a P/D handle's expiry,
or relax readiness. Exhausted exports still cannot publish ready records.
Diagnostics include bounded failed-object indices and export failure counts,
without cache keys, buffer addresses or native exception messages.

CPU regression evidence: `cpu-tests-r24.log`, 225 passed, 8 native skips,
21.05 seconds. Tests cover selective retry/order, operation-wide exhaustion,
malformed/exception outcomes during retry, unchanged GET failure behavior,
shutdown/ownership and scheduler readiness before/after retry success or
failure. `retry-pre-commit-r3.log` and `retry-code-commit.log` passed all
applicable hooks. Earlier formatting/type-check failures are retained.
Model-level verification of this new source is a separate gate below; the
native diagnostic and CPU tests do not themselves pass it.

## Two-host DRAM and SSD P/D retest at `408c3b752`

The `multinode-r5` controller and independent full-arm verifier both passed all
four cold/partial model cases. Recovered-source verification and the read-only
post-cleanup audit also passed. This completes these limited TP1 smoke gates,
not the broader multi-node feature acceptance matrix.
The test preserves the previous model/native cases, 120-second handoff expiry,
baseline token oracle, fail-on-receive-error policy and pinned MoRI binary.
Only the vLLM source checkpoint and its verified source-install identity changed.

| Medium | Decoder case | External tokens | Local tokens | Producer + decoder token IDs match baseline | Decoder wall time |
| --- | --- | ---: | ---: | --- | ---: |
| DRAM | Cold | 576 | 0 | Yes, 15 tokens | 3.222611 s |
| DRAM | Partial prefix | 320 | 256 | Yes, 15 tokens | 3.375032 s |
| ext4 SSD | Cold | 576 | 0 | Yes, 15 tokens | 5.122930 s |
| ext4 SSD | Partial prefix | 320 | 256 | Yes, 15 tokens | 2.791323 s |

The ordinary no-connector baselines took 0.765024 and 0.164925 seconds for the
584-token prompt and identical 15-token output. Cold P/D returned one token
from 003 and 14 from 004. The partial case first warmed 257 prompt tokens on
004, preserving 256 locally reusable tokens. Both media first passed the
forced cross-host 4096-byte poisoned-buffer transport check. These are smoke
wall times including readiness/JIT effects, not a repeated matched benchmark.

The SSD producer logged `chunk_start=16, attempts=2, initial_failed=7,
remaining_failed=0`. Thus this model run actually exercised selective retries;
the final export and receive completed under the unchanged deadlines. It does
not prove general staging fairness or replace the earlier native-only diagnostic.
The failed `multinode-r4` SSD partial case and physical-byte-accounting failure
remain preserved. No physical traffic savings, pressure robustness or GDS are
claimed. SSD used host-staged POSIX fallback on the existing local ext4 SSD0;
the read-only model weights remained on the separate shared NFS cache.

### Retest provenance and commands

- vLLM Python source: `408c3b752eaf18dd9c055d9ab5bac3208c5071df`.
- MoRI: unchanged `67632e80e2e492184b589904b63225f82d45537c`, tag `v1.2.3.post1`.
- MoRI native library SHA256: `d2c9b02ed7da9a727e43f1f8155b97b7fb7fea71e6fd9ac31e09cc9a47702c60`.
- Both runtimes: `vllm/vllm-openai-rocm:nightly-1dc464d42681d22f38caf1fdc1eb632dc4421c45`,
  exact digest `sha256:40e19c756e3dc9ffc9117770904d40376c7d3bf529cc76ddc379cde7ac4dae2d`.
  Native vLLM extensions are still the image's older binaries, not a current-source rebuild.
- Model: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`;
  pre-existing on both hosts, no downloads.
- llm-d-router/llm-d pins in the baseline table remain provenance only;
  neither llm-d nor mori-sched participates in this offline `LLM` test.

From `/home/ubuntu/vllmumbp`, commands executed with `set -o pipefail` and both
streams logged through `tee`:

```bash
sha256sum -c local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r5-launch-hashes.txt
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r5-controller.py failure gate1
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r5-controller.py smoke smoke1
```

The injected no-GPU failure gate exited 42 after verifying bounded live-launcher
termination, recovery and cleanup on both hosts. Never overwrite existing
attempt directories or launch overlapping attempts. A rerun requires fresh
attempt names, runner preflight and manifest review. The controller owns setup,
GPU execution, evidence recovery and exact cleanup; do not launch its fragments
individually.

Read-only verification commands after controller completion:

```bash
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/verify-multinode-r1.py \
  local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r5-smoke1
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/verify-multinode-r5-source.py \
  local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r5-smoke1
PYTHONDONTWRITEBYTECODE=1 repos/vllm/.venv/bin/python \
  local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r5-audit.py \
  local-logs/umbp-kvconnector-reimplementation-20260914/multinode-r5-smoke1
```

Frozen SHA256 identities:

- Controller: `da992ad97290368809e6af2cd5b50a3594e3dd01780047e7e7c15580f43d9d14`.
- Host lifecycle: `1c1f441222b820a149b595c3e07d3da16e3b11fe7018f93ac77ea447e3dafcc5`.
- Manifest: `15480d3e5cee65dab8ae2a24a1fa65011e90b116a1ae1641d82b98d176be7d05`.
- Source archive: `3a18076a55fc705631de97319fa04e3bb08ffd397529db7a02501a683546a895`.
- Installed-source hash list: `33805af9d1d8cdd6f63289ce8c89ed737892ef3d21b118e21c4db13edcf94157`.
- Unchanged strict model verifier: `4204d6dac4e0eacde64b8cb45a8a18679d91bc0a3ca47960bb705fef92fe0ceb`.
- Additional source verifier: `c1f4f715fa797148c316c87e72446656d01eb2fd046ae113358a46111ec05e44`.
- Cleanup auditor: `5cf1ee14c67346a728957676b8a493f460fbe986e096f5a551bf8e703f01a6ce`.

Full launch/per-input hashes and verifier hashes are in
`multinode-r5-launch-hashes.txt` and `multinode-r5-verifier-hashes.txt`.
The log is `multinode-r5-smoke1.log`; per-actor logs, RPC envelopes, result JSON
and recovery artifacts are in `multinode-r5-smoke1/` on the VPS.

The controller exited 0 after verifying both recovery archives against second
stable remote streams and removing each exact marked task root and container.
The image added on 004 was removed; 003's pre-existing image was retained.

| Host | Recovery bytes | SHA256 |
| --- | ---: | --- |
| 003 | 769,625,463 | `08805795fdd2b69e5b82b90e1790903fed8c9d11064301b4ed05dee3d4d2a3a6` |
| 004 | 769,401,320 | `d2909f4773d61458115da7af5f8b79ab70752ed76d22a735ee8439c309706a39` |

All three independent checks exited 0:

- `verify-multinode-r5.log`: the unchanged full-arm verifier compared every RPC
  receipt with the recovered JSON, verified archives/native binary, distinct
  host GPU UUIDs, actual attention/layout, handoff and baseline token equality,
  external/local cache counters and native GET batches of 16/16/4 then 16/4
  in each medium. It reports `multinode_pd_smoke_verified=true`,
  `is_benchmark=false` and `physical_byte_accounting_verified=false`.
- `verify-multinode-r5-source.log`: all 13 installed UMBP Python modules in
  each recovered runtime match the frozen `408c3b752` hash list; actual installed
  version/import markers match. It reports `multinode_source_identity_verified=true`.
- `multinode-r5-final-audit.log`: exact task roots, processes, listeners and
  labelled Docker resources absent; GPU memory returned to baseline on all
  eight GPUs per host, with no KFD processes. Original container/image
  inventories, SSD0 filesystem identity and pre-existing model size/config
  checks were preserved. It reports `multinode_post_cleanup_audit_pass=true`.

No task-created files or resources remain on either GPU host. The retained
VPS archives allow reproduction review; these checks do not verify model
accuracy, physical transfer bytes or sustained staging-pressure behavior.
