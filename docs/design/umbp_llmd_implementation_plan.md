# UMBP and llm-d implementation plan

Last updated: 2026-08-23

## Direct pre-admission HBM preload progress

- Added `target_tier: "gpu"` to the prefetch API and propagated it through the
  async engine/core-client boundary.
- Added an internal `hbm_prefetch_only` scheduler request. It uses normal
  connector lookup, allocation, and load metadata but never runs the model.
- On load completion, the scheduler publishes successfully loaded blocks with
  `KVCacheManager.cache_blocks()`, releases request ownership, suppresses
  frontend completion output, and reports `ready`, `partial`, or `miss`.
- Added idempotent start/poll behavior and cancellation for pending HBM loads.
- Added a scheduler unit test for cache publication, reference release, and the
  no-model/no-output invariant. API and existing CPU-prefetch tests pass.
- Current guardrails: one token-only prompt, complete cache blocks, and optional
  cache salt. LoRA and multimodal HBM preload are rejected until full admission
  metadata can be carried without risking a hash mismatch.
- Added `benchmarks/kv_offload/validate_hbm_prefetch.py` and ran it against a
  local Qwen/Qwen3-0.6B AMD server using `CPUOffloadingSpec`: preload `ready`,
  14,680,064 CPU-to-GPU bytes, 3.20 ms recorded transfer time (4.27 GiB/s),
  52.97 ms end-to-end preload polling time, 128/128 cached prompt tokens, and
  14.46 ms post-preload TTFT. This proves the DRAM-to-HBM path in this
  container; the secondary-tier/SSD path still requires a run with
  `--require-tier-read` and a tier configuration that forces spill.
- Extended the live validator with a correctness oracle: compare the seed token,
  a normal HBM cache-hit reference, and the restored token, then compare the
  reference/restored selected-token log probabilities. The AMD rerun passed
  with identical output, identical logprob (`abs_diff=0.0`), 128/128 cached
  prompt tokens, and 14,680,064 physical CPU-to-GPU bytes.

## Objective

Reproduce the useful parts of AMD's scheduler-aware UMBP deployment with vLLM
as the model server and llm-d as the cluster control plane. This document is
both the implementation roadmap and the execution log.

The target is not to make every UMBP hit faster than a local filesystem read.
It is to route, prefetch, and overlap transfers so that shared KV avoids more
expensive recomputation without adding storage stalls to the request critical
path.

## Ownership boundary

| Capability | Owner | Reason |
| --- | --- | --- |
| GPU/CPU KV allocation and block lifecycle | vLLM | Engine-local state |
| UMBP puts, gets, promotion, and registered buffers | vLLM + MoRI | Data plane |
| Tier-specific transfer metrics and readiness | vLLM + MoRI | Only the data plane observes completion |
| Asynchronous restore and engine admission | vLLM | Must coordinate GPU blocks and model execution |
| Fleet-wide KV index | llm-d Router/EPP | Cluster-scoped state |
| Prefix, tier, load, and predicted-cost scoring | llm-d Router/EPP | Cross-replica decision |
| Replica selection and flow control | llm-d Router/EPP | Request gateway responsibility |
| Kubernetes discovery, health, and rollout | llm-d | Deployment control plane |
| UMBP placement and remote-transfer execution | MoRI | Distributed storage data plane |

The standalone cache-aware proxy in this repository is a development tool. It
must not become a second production router alongside llm-d.

## Blog capability mapping

| Blog capability | Existing state | Required work | Status |
| --- | --- | --- | --- |
| Pointer-based batched L3 I/O | Registered CPU buffer and UMBP batch APIs | Benchmark chunk sizes | In progress |
| KV placement feed | Self-describing vLLM events plus UMBP bridge | Feed tier/locality into llm-d index directly | Partial |
| Precise prefix routing | llm-d precise scorer retains `DeviceTier` | Add measured restore cost, not another tier index | Partial |
| Load-aware routing | llm-d queue/utilization scorers | Compose with tier-cost scorer | Upstream llm-d |
| Restore-versus-recompute decision | llm-d measured-cost scorer | Calibrate from deployment measurements | Partial |
| Loadback prefetch | Async demand promotion plus scheduler coordinator | Add engine control endpoint and key resolver | Partial |
| Admission after prefetch | Coordinator reports ready/partial/miss | Integrate with engine control endpoint | Partial |
| Zero-CU SDMA restore | Generic CPU-to-GPU copy | Profile ROCm copy path, then add dedicated transfer stream/path | Planned |
| Remote DRAM reads | MoRI distributed client | Two-node correctness and performance validation | TODO |
| SSD spill | MoRI copy-on-commit | Expose tuning and per-medium proof; fix burst admission | In progress |
| Incremental P/D transfer | vLLM KV transfer framework | Reuse decode-side prefix and send only missing blocks | Separate project |
| Agent TTL/session hints | Not present | Extend request metadata, events, llm-d policy, and MoRI policy | Future |

## Phased execution

### Phase 0: measurement correctness

- [x] Add forced-eviction client benchmark.
- [x] Compare recompute, CPU DRAM, UMBP DRAM, filesystem, and UMBP SSD.
- [x] Reject conclusions when transfer counters do not prove restoration.
- [ ] Add MoRI per-medium DRAM/SSD read counters to the vLLM metrics surface.
- [x] Require one completed restoration per timed trial.
- [x] Record lookup, queue, tier read, CPU-to-GPU, and total critical-path time.

Acceptance: every timed request is classified as recompute, HBM, CPU, UMBP
DRAM, local SSD, remote DRAM, or remote SSD without inference from latency.

### Phase 1: deployment tuning surface

- [x] Basic DRAM, SSD, distributed staging, promotion, and executor settings.
- [x] Apply `UMBPConfig.from_environment()` before explicit vLLM overrides.
- [x] Expose DRAM watermarks, hugepages, NUMA binding, and prefaulting.
- [x] Expose SSD backend, segment size, watermarks, and copy-pipeline controls.
- [x] Expose remote-fetch caching and distributed DRAM page size.
- [x] Document burst-capacity sizing and staging-slot sizing.

Acceptance: production MoRI settings can be expressed without patching vLLM,
and explicit JSON values take precedence over environment values.

### Phase 2: llm-d placement-aware routing

- [x] vLLM can emit self-describing GPU and CPU KV events.
- [x] UMBP bridge can register HBM/DRAM placement with the master.
- [x] Define event representation for UMBP DRAM and SSD location, locality,
  source node, and estimated bandwidth.
- [x] Confirm llm-d's precise index and scorer retain per-replica `DeviceTier`.
- [x] Add a measured tier restore-versus-recompute scorer and compose it with
  llm-d's existing queue/utilization scorers in the scheduling profile.
- [x] Preserve speculative indexing and stale-event TTL behavior.

Acceptance: llm-d chooses among local recompute, local UMBP restore, remote
UMBP restore, and a replica-local HBM hit using observable cost inputs.

### Phase 3: asynchronous prefetch contract

- [x] Audit existing behavior: vLLM starts secondary-to-CPU promotion while a
  forwarded request is waiting and returns `HIT_PENDING` until it completes.
- [x] Distinguish this demand-triggered promotion from pre-forward prefetch.
- [x] Define the external request using model, cache salt, token IDs, target
  tier, and correlation ID. Router-local hashes are deliberately excluded:
  vLLM must derive keys with its own model, multimodal, and hash configuration.
- [x] Add vLLM control-plane endpoints for prefetch start, status, and cancel.
- [x] Add a scheduler-side lifecycle that reserves/promotes CPU destination
  blocks without admitting model execution.
- [x] Promote prefetched blocks through the normal offload lookup path.
- [x] Add an alpha llm-d PreRequest plugin that POSTs a bounded, idempotent
  prefetch to the selected endpoint before request forwarding.
- [x] Add bounded llm-d readiness waiting and cancel/fall back on deadline.
- [x] Bound concurrent prefetch state and promoted-byte admission in vLLM.

Acceptance: a prefetched request produces identical output, never consumes
unbounded cache capacity, and can overlap restore with queue/model work.

### Phase 4: transfer-path optimization

Phase 4 is not complete. Distributed correctness harnesses may be implemented
before it, but the concurrency matrix and performance claims require Phase 4
profiling and tuning first.

- [ ] Sweep `blocks_per_chunk` and UMBP batch sizes.
- [x] Verify pinned, hugepage-backed, NUMA-local CPU buffers.
- [ ] Profile ROCm SDMA use and synchronization granularity.
- [ ] Batch CPU-to-GPU copies on a dedicated stream.
- [ ] Tune remote QPs and NIC selection; start with four QPs from the blog.
- [ ] Add SPDK support on dedicated NVMe and compare with `io_uring`.

Acceptance: each stage is within an agreed percentage of its measured hardware
roofline, and prefetch removes most restore time from request TTFT.

### Phase 5: distributed validation

- [x] Optional AMD TP=2 local UMBP test.
- [x] Two-replica same-node remote-placement test.
- [ ] Two-node RDMA read and failure-fallback test.
- [ ] llm-d routing integration test with stale and conflicting placements.
- [ ] Concurrency 1/8/32/128 performance matrix.
- [ ] Long-context production-model evaluation and accuracy check.

Acceptance: report p50/p90/p99 TTFT, throughput, hit rate, recomputed tokens,
bytes per tier, admission failures, and correctness for every topology.

## llm-d integration contract

The proposed flow is:

1. vLLM publishes self-describing block events with medium and locality.
2. llm-d indexes all replicas of each consecutive block prefix.
3. The llm-d data producer tokenizes once; vLLM derives its compatible hashes.
4. The scorer estimates queue plus restore or recompute cost per endpoint.
5. The picker selects an endpoint and creates speculative index entries.
6. When restoration is needed, llm-d sends a bounded prefetch request.
7. vLLM reports ready, partial, miss, or failed before the deadline.
8. llm-d forwards normally when ready, or instructs/falls back to recompute.
9. Confirmed KV events replace speculative state.

This follows llm-d's existing EPP parser → data producer → scorer → picker flow
and keeps cluster policy out of the vLLM engine.

## Execution log

### 2026-08-23

- Audited the AMD/Moonshot UMBP blog and MoRI 1.2.2 configuration surface.
- Audited llm-d precise prefix routing, KV indexer, predicted-latency producer,
  flow control, and model-server boundary.
- Confirmed `EmbeddedLLM/llm-d` is a fork; upstream development is under
  `llm-d/llm-d` and related `llm-d/llm-d-router`/`llm-d-kv-cache` repositories.
- Assigned cross-replica indexing, cost scoring, and request routing to llm-d.
- Assigned transfer execution, prefetch readiness, and GPU admission to vLLM.
- Began Phase 1 because trustworthy capacity and transport tuning is a
  prerequisite for prefetch and cost-model work.
- Completed the Phase 1 configuration patch: MoRI environment overlays now
  load first, explicit tier JSON wins, and vLLM exposes DRAM topology,
  watermarks, SSD backend/segments, SSD copy pipeline, and remote-cache knobs.
- Added a unit test covering environment-versus-explicit precedence and every
  newly exposed configuration group; the combined offload suite now passes
  (`55 passed`, including four prefetch lifecycle tests).
- Traced the offloading scheduler loadback path. It already batches promotion
  per request, submits it asynchronously at schedule end, and lets other work
  run while the request reports `HIT_PENDING`. The missing blog capability is
  earlier initiation by llm-d before normal request forwarding, plus optional
  GPU preloading/overlap; a second storage implementation is not required.
- Added `KVOffloadPrefetchCoordinator` in vLLM with idempotent start, poll,
  cancel, ready, partial, and miss states. It reuses `OffloadingManager.lookup`
  and schedule-end batching, so prefetched blocks enter the existing CPU
  prefix-cache path. Its mock suite passed (`4 passed`).
- Added the llm-d Router alpha `umbp-prefetch` PreRequest plugin in the sibling
  `/app/umbp/llm-d-router` checkout. It sends model/cache identity and prompt
  token IDs to the selected endpoint and supports fail-open or fail-closed
  behavior. Mock HTTP tests cover payload selection and failures.
- Installed the repository-required Go 1.26.6 toolchain under
  `/app/umbp/.tools/go1.26.6`, formatted all three touched Go files, and passed
  `go test ./pkg/epp/framework/plugins/requestcontrol/prerequest/umbpprefetch`.
- Added the production vLLM `/v1/kv_cache/prefetch` start/status/cancel API and
  routed it through EngineCore's utility channel, keeping offload-manager
  mutations on the scheduler process. vLLM derives chained hashes using its
  configured hash function and converts them to every KV group's chunk keys.
- Added API contract tests and reran the offloading lifecycle suite (`54
  passed`); focused Ruff, formatting, mypy, SPDX, and repository checks passed.
- Current scope is CPU loadback. GPU preload and byte-based promoted-cache
  admission remain before production enablement.
- Extended the llm-d plugin with configurable readiness polling. It waits only
  within `waitTimeout`, cancels the vLLM prefetch on expiry, and follows its
  fail-open policy so the original request can recompute. Plugin and runner
  package tests pass after formatting with Go 1.26.6.
- Bounded vLLM prefetch control state with separate pending-request and block
  caps plus pending and terminal TTLs. Stale pending contexts are finalized
  before removal, terminal results remain briefly available for idempotent
  retries, and saturation is surfaced as HTTP 409 for router fallback.
- Extended cache identity across llm-d and vLLM for multimodal content hashes,
  placeholder ranges, cache salt, and explicitly configured LoRA names. vLLM
  applies the same LoRA → multimodal → salt extra-key ordering as normal
  request hashing and derives tower-LoRA multimodal identifiers locally.
- Audited HBM preload ownership. vLLM GPU blocks are allocated to admitted
  requests and tracked by their block tables, so a safe pre-admission preload
  needs a reservation token that can later be claimed or evicted; directly
  copying into unowned GPU blocks was rejected.
- Added the llm-d alpha `restore-cost-scorer`. It evaluates every prompt block
  using the cheapest configured available tier or recomputation, supports
  per-tier fixed setup costs, and composes with existing queue/utilization
  scorers. Added a deployable UMBP EPP example whose placeholder costs must be
  replaced by measurements from the target model and topology.

### 2026-08-24

- Completed bounded HBM-prefetch admission. GPU-targeted preloads now share the
  configured pending-request, pending-block, pending TTL, and terminal TTL
  limits, and add `prefetch_max_pending_gpu_bytes` using the actual KV-cache
  allocation's bytes per GPU block. The block cap is also clamped to physical
  GPU cache capacity.
- Pending TTL expiry cancels connector work and releases request-owned blocks;
  terminal records expire independently. Idempotent results retain their block
  and byte accounting without a separate unbounded EngineCore map.
- Added tests for request, block, byte, pending-expiry, terminal-expiry, cache
  publication, and no-model/no-output behavior. The full scheduler suite passed
  (`154 passed`).
- Tightened the forced-eviction benchmark to bracket metrics around every timed
  request. CPU/filesystem/UMBP modes now fail on the first trial without a
  physical CPU-to-GPU restore; filesystem and UMBP modes additionally require
  a secondary-tier read per trial. Results report the proven restoration count
  and minimum per-trial byte deltas.
- Added lookup sync/async, request queue, secondary-tier read, CPU-to-GPU,
  server end-to-end, and client TTFT/latency measurements to the benchmark.
  Fixed metric parsing to match exact Prometheus sample names so histogram
  buckets cannot inflate sums. Stage times are reported separately because
  asynchronous lookup and transfer stages can overlap.
- Confirmed the current MoRI Python boundary returns only batch success flags.
  Exact UMBP DRAM-versus-SSD attribution requires a MoRI result/metrics API
  extension; vLLM must not infer the medium from latency or SSD enablement.
- Extended KV store/remove events with append-only physical placement hints:
  `storage_tier`, `source_node`, and (for stores)
  `estimated_bandwidth_bps`. `medium` continues to identify the vLLM cache
  layer and `locality` remains relative to the publisher. The hints default to
  unknown and preserve decoding in both directions with older schemas. They
  must only be populated from an authoritative placement observation; MoRI's
  master remains authoritative when blocks can migrate between DRAM and SSD.
- Updated llm-d's vLLM adapter to decode those hints from map and positional
  encodings into generic KV events (checkpoint `6f6492d`). Updated its
  event pool to prefer an authoritative physical `storage_tier` when building
  dedup scopes and KV-index entries, making `dram` and `ssd` available to the
  existing precise-prefix and restore-cost pipeline (checkpoint
  `30d90a0`). Targeted `pkg/kvevents`, adapter, KV-index, and restore-cost tests
  pass with Go 1.26.6.
- Checkpoint
  `da4bdb0` retains locality, source node, and bandwidth in index entries with
  symmetric store/remove identity. Checkpoint `1bacaef` aggregates conservative
  placement metadata across precise-prefix hits and lets the restore-cost
  scorer derive per-block transfer time from `kvBlockBytes / bandwidth`, add a
  configured locality penalty, and fall back to measured static tier cost when
  bandwidth is unknown. The example EPP configuration documents the new knobs.
- llm-d checkpoint `303550c` adds an optional TTL specifically for authoritative
  physical placement hints. Refreshes replace generation-tagged timers;
  explicit removals, pod clears, and shutdown cancel them. Expiry evicts the
  exact enriched index identity while leaving ordinary GPU/CPU events and the
  existing speculative-entry TTL unchanged. `go test -race ./pkg/kvevents` and
  the full `go test ./pkg/...` suite pass, including a real-index
  store-to-expiry integration test.

## Open decisions

1. Whether the HTTP control endpoint should remain public or move behind a
   dedicated authenticated internal listener before production enablement.
2. Whether llm-d should consume authoritative MoRI placement by polling the
   master, subscribing to a future placement feed, or through a sidecar that
   emits the optional standard KV-event placement hints.
3. How llm-d communicates a recompute-versus-restore decision without adding
   UMBP-specific fields to the OpenAI request schema.

## Why authoritative placement requires a MoRI change

The router must distinguish real, currently servable UMBP placement from
advisory cache metadata. The existing MoRI APIs do not provide a safe way to do
that from a periodically polling control plane:

- `MatchExternalKv` queries the external KV index populated by vLLM cache
  events. MoRI documents these entries as advisory and not servable through
  `ResolveKey`; they do not prove that the UMBP data plane currently holds the
  block in DRAM or SSD.
- `BatchRouteGet` queries real servable placement, but it is a data-plane read
  operation: it records access, updates routing counters, and grants leases.
  Polling it from vLLM or llm-d would contaminate measurements, alter eviction
  behavior, and could keep cold blocks resident merely because the control
  plane inspected them.
- `BatchLookup` is read-only but returns only existence flags. It cannot supply
  the owning node, physical tier, size, or peer address required for locality-
  and bandwidth-aware routing.

An experimental cross-project implementation therefore added a read-only
`BatchInspect` RPC to MoRI. It returns the actual servable node, tier, size, and
peer address without recording access or granting a lease. The Python
`UMBPMasterClient` exposes the same operation for a bounded placement polling
adapter. This keeps the MoRI master authoritative while allowing llm-d's
placement TTL to fail closed if refresh stops.

The experiment was committed locally as `8521f3dc`, validated, and never
pushed. It was subsequently reverted by `a1bbcf20` so MoRI remains unchanged.
The rationale stays here because exact physical placement would still require
an equivalent upstream API. Validation completed before the revert:

- `cmake --build build -j2` passed, including protobuf regeneration, the master,
  client, and Python bindings.
- The full Python master-client suite passed (`45 passed`).
- The new integration test performs a real distributed UMBP put, flushes its
  heartbeat, observes its servable DRAM placement through `BatchInspect`,
  verifies a missing key remains absent, and verifies the advisory external KV
  index remains empty.

Without this MoRI API addition, exact DRAM/SSD-aware routing must remain
disabled. Falling back to `MatchExternalKv` would be inaccurate, while polling
`BatchRouteGet` would change the behavior being measured.

### Logical availability proxy

The production design keeps MoRI unchanged. The vLLM bridge retains the
original KV-event stream and labels successful storage events as logical
`UMBP` availability. It deliberately leaves physical tier, owner, locality,
and bandwidth empty. llm-d can route to an endpoint with UMBP availability
using calibrated restore cost, but cannot claim exact DRAM/SSD placement.

The proxy remains intentionally outside the vLLM engine. That keeps master
integration and llm-d-specific policy out of the inference data path while
preserving the standard KV-event boundary. The next distributed validation is
to run the correctness sequence on two physical nodes and use existing MoRI
transport and aggregate tier metrics to validate the path.

The same-host validator stores before the second UMBP client joins, restores
through that second client, verifies all payload bytes, and checks the logical
`UMBP` event. This completes the two-replica functional item only; it does not
close physical placement, two-node RDMA, or performance validation.

llm-d checkpoint `761016e` recognizes this distinction. Logical `UMBP`
lifecycle entries remain indexed until their remove event, while physical DRAM
and SSD hints retain fail-closed placement expiry. The sample restore-cost
profile includes a deployment-calibrated `umbp` tier entry. Full Go package
tests, `go vet`, and the KV-event race suite passed. Container-backed presubmit
could not run because Docker and Podman are unavailable; its earlier signature
gate also rejects six unsigned checkpoint commits already present on the
branch.

The forced-eviction benchmark can consume the unmodified MoRI master's
Prometheus endpoint. Strict SSD mode brackets every timed request and waits for
both `mori_umbp_ssd_read_total{status="ok"}` and
`mori_umbp_ssd_read_bytes_total` to advance. A missing delta fails the run.
Other UMBP results are labeled `logical-umbp-only`; released MoRI has no DRAM
read counter, so absence of an SSD delta is not used to infer DRAM placement.

The remote-read validator supports separate source and reader processes for
two physical hosts. The source publishes its deterministic block before the
reader joins and remains alive until the reader verifies every byte and stores
an acknowledgement. The local two-process-equivalent run passed with 8,192
bytes. Cross-node RDMA and failure fallback remain unchecked until the harness
is run on the target network.

Real AMD validation on 2026-08-26 passed the Qwen/Qwen3-0.6B UMBP restoration
test on TP=1 and TP=2. Both runs forced GPU eviction, required UMBP tier-read
and CPU-to-GPU counters, and checked generated-output equality. The TP=2 run
restored 11,010,048 bytes through both measured stages. These runs close local
GPU correctness only; they do not replace the Phase 4 sweep or profiling.

On 2026-08-31, the shared CPU-primary allocator passed 97 focused tests in the
exact ROCm nightly image and real 2 MiB hugetlbfs/NUMA placement checks on an
MI355X host. Separate 64 MiB GPU 0/node 0 and GPU 4/node 1 smokes verified
`cudaHostRegister`, byte-exact GPU round trips, and approximately 56.7 GB/s H2D
and 56.4 GB/s D2H. The retained `tools/umbp/validate_host_allocator.py` tool
checks `/proc/self/numa_maps`, physical page size, cleanup, correctness, and
bandwidth. These single-size smokes close the allocator verification item but
are not the required block/batch/concurrency performance sweep.

The reproducible node bootstrap is
`tools/umbp/bootstrap_distributed_node.sh`. It pins the three validated project
revisions, builds the released MoRI source without local patches, installs vLLM
and its test dependencies in a Python 3.12 uv environment, checks GPU and UMBP
imports, and reports visible RDMA addresses. It deliberately leaves privileged
host networking, hugepage, and NVMe configuration to the operator. The script
and accompanying usage documentation passed `bash -n` and the targeted
pre-commit suite, including ShellCheck.

1. Whether SDMA loadback belongs in the generic CPU offload worker so all
   secondary tiers benefit.
