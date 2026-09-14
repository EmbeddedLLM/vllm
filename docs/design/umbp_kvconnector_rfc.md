# [RFC]: UMBP KVConnector for P/D Disaggregation and Multi-Tier KV Offloading

Draft for the vLLM RFC issue form. Not submitted to GitHub.
Implementation status and evidence are dated 2026-09-14.

## Motivation

Long-context and agentic workloads repeatedly revisit prefixes after their KV
has left GPU memory. Disaggregated serving introduces a related problem: a
prefill engine produces KV that a separate decode engine needs, potentially
with part of that prefix already resident on the decoder. We want one
connector-owned data path that can handle both situations while preserving
vLLM's cache ownership and failure-recovery rules.

vLLM already has P/D and offloading connectors. This proposal adds a MoRI UMBP
backend and its associated control-plane integration; it does not propose
replacing the KVConnector framework or treating existing connectors as incapable
of these workloads. In particular, the existing
[MoRIIOConnector](https://github.com/vllm-project/vllm/blob/1678b396270406c27fcab8f5b86b21fd305ac605/docs/features/moriio_connector_usage.md)
provides a direct P/D transfer path. A reusable, tiered UMBP object pool is a
different storage and lifetime model.

The intended benefits are:

- Restore evicted KV from local or remote DRAM/SSD when cheaper than recompute.
- Deliver prefill KV through the same pool, loading only the decoder's missing
  compatible state rather than retransferring its valid local prefix.
- Expose reliable placement and transfer-cost signals to a router without
  making a router's cache prediction authoritative for engine correctness.
- Support proactive prefetch using connector-managed transfers and ordinary
  cache-manager ownership, without a separate UMBP-only engine execution path.

This is motivated in part by
[SGLang RFC #27898](https://github.com/sgl-project/sglang/issues/27898), which
discusses UMBP-backed storage with scheduler-visible placement and policy. This
RFC adapts those goals to vLLM's scheduler/worker KVConnector contract; SGLang
benchmark or implementation claims are not evidence for this implementation.

## Proposed Change

### 1. Add an optional UMBPConnector

Implement a `KVConnectorBase_V1` connector supporting both:

1. **KV offloading:** a serving engine stores reusable KV and restores it after
   local eviction, using embedded storage or a master-led distributed pool.
2. **P/D disaggregation:** a prefill engine hands off computed KV to a separate
   decode engine through that pool, with explicit readiness and failure handling.

The first functional milestone must demonstrate both paths. Passing storage
unit tests or showing prefix reuse between ordinary serving replicas is not
sufficient to claim P/D support.

MoRI is imported only when this connector is selected. Ordinary vLLM serving
must not need MoRI, a UMBP master, llm-d or mori-sched. The initial dependency
target is [MoRI v1.2.3.post1](https://github.com/ROCm/mori/releases/tag/v1.2.3.post1),
commit `67632e80e2e492184b589904b63225f82d45537c`.

The prototype may use `kv_connector_module_path` while interfaces are reviewed.
Register an in-tree connector only with implemented scheduler/worker hooks,
supported-configuration checks, and reproducible serving validation.

### 2. Keep responsibilities explicit

```text
                    Request routing / optional llm-d policy
                         |                    |
                     prefill               decode
                         v                    v
                 +----------------+   +----------------+
                 | vLLM scheduler |   | vLLM scheduler |
                 | + UMBPConnector|   | + UMBPConnector|
                 | block ownership|   | local-prefix  |
                 | handoff state  |   | + missing KV  |
                 +--------+-------+   +--------+-------+
                          | metadata / rank receipts |
                 +--------v-------+   +--------v-------+
                 | worker: fences |   | worker: fences |
                 | registered KV  |   | registered KV  |
                 +--------+-------+   +--------+-------+
                          | ranged PUT         | ranged GET
                          +----------+---------+
                                     v
                           MoRI UMBP object pool
                    local / remote DRAM and file-backed SSD
                                     |
                      placement / capacity / transfer signals
                                     v
                           optional routing policy

 Single-engine offloading uses the same scheduler/worker/pool interfaces.
 Engine HBM placement hints are separate from readable pool objects.
```

This diagram describes the proposal, not a completed deployment.

| Component | Owns |
| --- | --- |
| vLLM cache manager and scheduler | GPU block allocation, valid prefix/checkpoint state, model admission and recomputation |
| Connector scheduler | Lookup, pinned transfer jobs, per-request handoff, all-rank completion and publication decisions |
| Connector worker | Allocation registration, byte layout, compute/transfer fences, native I/O and final outcomes |
| MoRI UMBP | Object storage, pool placement, transport and configured storage/eviction policy |
| llm-d or another router | Replica selection and bounded prefetch coordination; not GPU cache ownership |

MoRI storage policy, vLLM token scheduling, and router request scheduling are
different responsibilities. Neither llm-d nor mori-sched is a prerequisite for
basic connector correctness. A MoRI-aware routing adapter is a separately
validated part of the full integration.

### 3. Define compatible, versioned storage identities

Use vLLM-owned block hashes rather than hashes reconstructed independently from
prompt text by a router. Namespacing must include deployment isolation, model
and immutable weight revision, cache format/encoding, layer/group identity,
parallel shard mapping and hash algorithm. Salt, adapter and multimodal identity
must retain the semantics of vLLM's cache keys.

The initial payload is one logical cache-group block per worker shard, assembled
from its layer buffers using MoRI's ranged API. Its canonical byte order is
sorted layer name followed by logical H/N/C order. Cache capacity, GPU addresses,
physical layer packing and kernel-block splitting are not semantic identity.
The initial format requires matching TP/PP/CP topology; heterogeneous-TP
resharding needs a separately versioned and tested mapping.

Exchange worker-derived layout identities before lookup. A representative
scheduler cache spec is insufficient when workers retain different per-layer
specs. Quantization scale/calibration compatibility must also be established;
otherwise reject reuse even if the apparent tensor shapes match.

For hybrid models, use validated group-specific prefix records and recurrent
checkpoint boundaries. Do not save a mutable recurrent state under a positional
prefix hash merely because it occupies that request's current block-table slot.
Non-prefix-cacheable state needs a request-scoped handoff identity, not a reusable
prefix key. Unsupported required groups must fail configuration or trigger an
explicit supported recomputation path, never disappear from the hit calculation.

### 4. Compose offload with the scheduler lifecycle

Use existing connector hooks for lookup, post-allocation state, scheduler
metadata, worker registration, load/store execution, worker feedback and request
completion. Shared framework changes, if required, should be connector-neutral
and reviewed independently.

The proposed flow is:

1. Look up only the state beyond the request's valid local prefix. Check every
   required cache group and shard, respecting sliding-window and recurrent-state
   semantics. Keep lookup asynchronous and bounded; storage I/O must not block
   the scheduler's execution loop.
2. Allocate destinations through the cache manager. Pin the exact source or
   destination blocks for each generation-scoped transfer job before submission.
3. Workers wait for prior compute/transfer use of those bytes, then issue ranged
   I/O. Metadata does not contain arbitrary caller-supplied memory pointers.
4. Collect final per-object receipts from every required worker. Duplicate
   feedback cannot count as another rank. A failure on one rank does not permit
   freeing blocks that another rank may still be accessing.
5. Publish successful loads through vLLM's cache manager. Report failed receive
   requests and affected block IDs through the connector failure APIs so the
   configured recompute/fail policy can run without consuming unwritten KV.

Best-effort offload failure is a future miss, not a mandatory-delivery failure.
P/D handoff has stronger completion requirements and must be represented as such.
Finite lookup/admission deadlines must not be confused with safe cancellation
of a running memory operation.

### 5. Make P/D readiness and incremental loading explicit

The proposed handoff carries a versioned handle identifying the model/cache
namespace, producer generation, required token boundary, cache groups and shards.
Sending that handle to the decoder is **not** a claim that the KV is ready.

```text
Prefill compute -> pin exportable state -> fenced stores on all ranks
       -> all required objects successful -> publish handoff-ready record

Decode local-prefix lookup -> validate handle and readiness
       -> allocate/load only missing required state
       -> all required receives successful -> admit decode computation
```

The ready record's storage, lifetime and cleanup are part of the protocol, not
implicit consequences of the last rank submitting a PUT. UMBP's per-object write
completion is not an atomic transaction across all ranks/groups. Reusable prefix
objects may be shared; request-specific boundary state and readiness records
remain generation scoped.

Readiness describes completed production, not a guarantee against later
eviction or peer loss. Reads still validate outcomes. On stale handles,
incompatibility, timeout or missing data, discard the unusable state and follow
the configured recompute/error policy. Exact retention/lease behavior is an open
design question; do not assume an unverified native lease API.

Validation must measure the decoder's actual missing-state transfers and
recomputed tokens, including the final-logit token, rather than equating a pool
lookup hit with saved prefill work.

### 6. Adapt configuration to the pinned MoRI API

Expose explicit operator settings for namespace/model identity, serving role,
offload/P-D policy, bounded lookup/transfer admission, per-rank storage budgets,
master/peer addresses, native policy, and scratch arena sizes. Exact option
names are subject to review; this RFC does not provide a runnable configuration
for the unfinished connector.

The pinned native
[ranged interface and bindings](https://github.com/ROCm/mori/blob/67632e80e2e492184b589904b63225f82d45537c/src/pybind/pybind_umbp.cpp)
provide per-key outcomes and CPU/GPU registration. Writes must cover a complete
object; reads may request ranges. Register the allocation once, including the
GPU ordinal. Shutdown drains operations and deregisters allocations, without
clearing another engine's shared objects. The Python binding lacks `close()`;
the wrapper must own the native handle and verify its destruction behavior.

In the pinned
[configuration contract](https://github.com/ROCm/mori/blob/67632e80e2e492184b589904b63225f82d45537c/src/umbp/include/umbp/common/config.h),
remote ranged scratch defaults to zero. Size both GET/PUT arenas explicitly.
The legacy distributed `medium` selects one medium; peer-local DRAM/SSD tiering
requires an explicit backend policy. Enabling both capacities is not proof of
promotion/demotion. Distinguish a heterogeneous pool from a tiered local peer.

Start native validation with DRAM and file-backed ext4 SSD. Keep each worker's
writable state isolated. SPDK, filesystem provisioning and kernel changes are
not required by this proposal. Transport details such as SDMA, RDMA, staging or
zero-copy must be observed, not inferred from the connector name.

### 7. Add placement-aware routing and bounded prefetch

The full integration also includes an optional placement/event adapter and
llm-d policy integration. Routing may combine validated placement, replica load,
capacity and measured transfer/recompute costs. External engine-HBM reports are
advisory; they do not make a pool object readable.

Define prefetch start/status/cancel semantics using engine-validated token/cache
identity, a bounded byte/job budget, expiry and idempotency. CPU promotion and
HBM preload are distinct operations. HBM preload must use ordinary connector
allocation/load/publication rules while executing no model forward or sampling.
Expiry suppresses publication but does not release memory while I/O is running.

Use or extend an agreed connector-neutral control/hint mechanism. Do not add a
parallel family of UMBP-only core RPCs. Stale events, replay gaps, restarts and
control-plane failure must fall back to ordinary request routing safely.
Integrate traffic priority/admission so background offload or speculative
prefetch cannot consume every resource needed for demand reads and P/D.

### 8. Correctness, effectiveness and rollout gates

| Gate | Required evidence |
| --- | --- |
| Optional dependency and opt-out | Import/serve without MoRI; unchanged no-connector behavior |
| Storage/lifetime contracts | Real buffers; invalid ranges, partial I/O, queue limits, fences, cancellation, drain and owner release |
| Composed scheduler lifecycle | Actual scheduler/block pool; pressure, preemption, same-ID reuse, delayed ranks and failed load recovery |
| Native single-node offload | Forced GPU eviction with verified DRAM and ext4 SSD restore sources |
| Native multi-node offload | Remote DRAM/SSD reads, peer/master faults and safe recovery |
| True P/D | Separate prefill/decode engines, all-rank readiness, nonzero local-prefix reuse and missing-state-only reads |
| TP/hybrid state | Full attention, MLA and recurrent groups; boundary validity, group failures and exact supported shard mappings |
| Model correctness | Qwen3-0.6B development tests, matched GSM8K baseline/offload/P-D; Kimi K3 TP8 hybrid and long-context acceptance |
| Routing/prefetch | llm-d adapter, placement staleness/replay recovery, cancellation/expiry, no-model HBM preload and routing opt-out |
| Performance | Matched repeated trials with TTFT, ITL, throughput, tail latency, recomputed tokens and actual transfer bytes |
| Reproduction | Repository/runtime/model pins, commands, workload, logs, failure evidence and cleanup audits |

Compare no connector/recompute, native CPU offload, UMBP tiers and direct P/D
where supported. Report both improvements and regressions across cold, warm,
partially reusable and pressure-heavy workloads. Separate lookup, queueing,
storage/transport and GPU materialization costs. Count actual I/O: a successful
deduplicated PUT is not evidence that its logical bytes crossed a NIC or SSD.
Do not claim a universal speedup or merge readiness from smoke tests.

Proposed implementation sequence:

1. Storage, identities, layout mapping and transfer-lifetime contracts.
2. Native configuration plus scheduler/worker hooks, followed by composed
   offload and P/D handoff tests. Both paths gate the functional milestone.
3. Native GPU/DRAM/SSD and two-node model correctness, including hybrid state.
4. Placement-aware routing, connector-controlled prefetch and fault recovery.
5. Matched accuracy/performance evaluation and publication of reproduction data.

### 9. Alternatives and trade-offs

- **MoRIIO P/D plus a separate offloader:** retains a direct transfer path and
  may win for cold handoffs, but requires coordination of two data paths and
  ownership models. Benchmark it; the shared pool is not assumed faster.
- **UMBP only as an offloading backend:** useful, but does not by itself define
  P/D readiness, incremental decoder loading or request-specific boundary state.
- **Reuse general offloading components:** preferred where contracts match;
  factor common ownership/layout logic instead of maintaining another generic
  cache manager. Whether the upstream form is a dedicated connector or a shared
  backend with a thin P/D adapter is a maintainer-facing design decision.
- **Pool-mediated P/D:** enables reuse and decouples engines, at the cost of
  storage admission, metadata, staging and potentially extra copies. Bounded
  memory and metadata pressure must be measured under real serving concurrency.

### 10. Questions for maintainers

1. Should this upstream as a dedicated UMBPConnector, or reuse the offloading
   connector with a common storage interface and a thin P/D handoff layer?
2. Which existing scheduler ownership/prefix-record abstractions should be used
   for all-group save sources and generation-scoped completion?
3. What is the preferred connector-neutral readiness/control/hint surface for
   router-driven prefetch and a P/D handle whose data is not ready yet?
4. Should the initial wire format require identical topology, or is portable
   heterogeneous-TP sharding required for the first upstream feature milestone?
5. What retention guarantee should pool-mediated P/D promise, and what should
   happen when a ready object's storage lease cannot be maintained?
6. What native CI coverage and ongoing ROCm ownership are required before in-tree
   registration? Which shared changes should be reviewed separately?

## Feedback Period

At least two weeks after posting, with particular attention to KVConnector,
cache-manager/hybrid-state, P/D, ROCm and routing maintainers.

## CC List

To be selected by the submitter. No maintainer handles are added automatically.

## Any Other Things

### Related work and overlap

Coordinate with existing work rather than creating a second general framework:

- [Multi-tier offloading RFC #38260](https://github.com/vllm-project/vllm/issues/38260)
  discusses reusable multi-tier offloading architecture.
- [Prefix-record save sources RFC #52837](https://github.com/vllm-project/vllm/issues/52837)
  addresses source identity/ownership for valid cached state.
- [Transfer QoS RFC #46016](https://github.com/vllm-project/vllm/issues/46016)
  discusses prioritizing critical P/D traffic alongside background transfers.
- [KV hint envelope RFC #53421](https://github.com/vllm-project/vllm/issues/53421)
  proposes a versioned orchestrator-neutral hint surface.
- [Generic control RPC discussion #51636](https://github.com/vllm-project/vllm/issues/51636)
  is marked closed as duplicate; it is related discussion, not an accepted API
  that this proposal assumes exists.

The distinct proposal here is a release-pinned UMBP data plane with a concrete
joint offload/P-D lifecycle, plus placement/prefetch adapters. The initial related
issue search is not an exhaustive duplicate-work determination; refresh issues,
PRs and their current implementations before submission or opening a PR.

### Current prototype and evidence limits

The local prototype starts from `EmbeddedLLM/vllm:umbpkvconnector` at
`1678b396270406c27fcab8f5b86b21fd305ac605`, with these unpublished checkpoints:

- `d9692bb0f9ec30a72a5307c338c2199f0c4b74cb`: ranged storage and key identities.
- `6b71857f514ac5b5b568c1bfeaca9fa7e7de6f35`: page mapping and fenced transfers.
- `2f891e734513fa26d3c469c62b7dba64b3c2f656`: explicit native storage policy,
  layout-derived sizing checks and private SSD-path lifetime.

MoRI remains unchanged at release commit
`67632e80e2e492184b589904b63225f82d45537c`. The preserved llm-d-router reference
is `7541552c71642f2756517a7aeb3c0c35720c06d0`; llm-d is
`d557f83e1e5f1e5a6ed54ef154c554cf6c775e33`. Neither has yet been ported or
validated against this new connector implementation.

Current evidence: **148 CPU tests passed** (146 UMBP component tests and two
existing output-aggregation regressions). These cover physical layouts,
generation/rank receipts, policy translation, private-path isolation and
lifetime/failure cases using CPU buffers and a native-API fake. They do not
establish HIP/RDMA/SSD correctness, actual vLLM
scheduler ownership, complete P/D serving, model accuracy or performance.
No UMBPConnector is registered yet. Earlier integration results from other
branches/releases are not carried over as validation of this prototype.

One native integrity gap needs explicit resolution: the pinned
[SSD ranged-read interface](https://github.com/ROCm/mori/blob/67632e80e2e492184b589904b63225f82d45537c/src/umbp/include/umbp/local/tiers/ssd_tier.h)
does not verify whole-record CRCs on ranged reads. Enabling the native CRC
option alone is therefore insufficient evidence of corruption detection on
this path. The implementation needs an integrity strategy and fault-injection
evidence before claiming that corrupted KV reliably triggers recomputation.

See the adjacent `umbp_kvconnector_reimplementation.md` for commands and detailed
acceptance tracking. Public reproduction artifacts must accompany a future PR;
the local commit IDs above are not currently public download links.

### Deployment and privacy

Follow vLLM's [inter-node security guidance](https://github.com/vllm-project/vllm/blob/1678b396270406c27fcab8f5b86b21fd305ac605/docs/usage/security.md).
Keep pool/control endpoints inside the deployment trust boundary. A cache
namespace or content hash is not authentication or encryption. Protect stored
KV as model-derived user data; bound management requests and avoid exposing raw
tokens, pointers, credentials or per-request identifiers in metric labels.
Neither normal shutdown nor a local reset may clear unrelated shared KV.

This RFC and prototype were prepared with AI assistance. Human review of the
proposal, every changed line and model evaluation evidence is required before
submitting an implementation PR.

### Before submitting

- [ ] Human author reviews and owns the proposal and recorded evidence.
- [ ] Refresh related issues/PRs and resolve duplicate or overlapping work.
- [ ] Complete the RFC issue form's documentation-chatbot prerequisite; it has
  not been performed by this draft.
- [ ] Select the CC list and confirm the feedback period.
- [ ] Publish any referenced prototype/evidence links intended for reviewers.
- [ ] Obtain explicit approval to post the issue; this file alone posts nothing.
