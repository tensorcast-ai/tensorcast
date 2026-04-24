---
slug: composite-materialization-and-vectored-direct-write
title: Composite Materialization and Vectored Direct-Write
status: implemented
areas: ["core", "daemon", "docs", "benchmarks", "integrations"]
created: 2026-04-19
last_updated: 2026-04-23
related_code:
  - docs/designs/0088-unified-artifact-profiles-with-shared-dataplane.md
  - docs/designs/0087-unified-artifact-runtime-and-routed-byte-artifact-architecture.md
  - docs/designs/0089-core-backed-body-handles-and-backing-policy.md
  - docs/designs/0108-tensor-aware-materialization-strategy-plane.md
  - core/store/materialization/contracts/loading_spec.h
  - core/store/materialization/dataplane/contracts/source.h
  - core/store/materialization/dataplane/contracts/sink.h
  - core/store/materialization/dataplane/runtime/pump.h
  - core/store/materialization/dataplane/runtime/pump.cc
  - core/store/materialization/dataplane/sources/byte_range_mapped_source.cc
  - core/store/materialization/dataplane/sources/remote_key_source.h
  - core/store/materialization/dataplane/sources/remote_key_source.cc
  - core/store/materialization/dataplane/sinks/target_layout_host_sink.cc
  - core/store/runtime/ingestion/materialization_facade.h
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/communicator/routing/types.h
  - core/communicator/routing/connection.h
  - core/communicator/routing/adapter.h
  - core/communicator/routing/routing_context.h
  - core/communicator/engine/engine.h
  - core/communicator/engine/protocol.h
  - core/communicator/transport/request.h
  - core/communicator/transport/partition_tensor.h
  - core/communicator/transport/rdma_transport.h
  - proto/tensorcast/daemon/v2/store_daemon.proto
links:
  plan: ../plans/0115-composite-materialization-and-vectored-direct-write.md
  dependencies:
    - ./0088-unified-artifact-profiles-with-shared-dataplane.md
    - ./0087-unified-artifact-runtime-and-routed-byte-artifact-architecture.md
    - ./0089-core-backed-body-handles-and-backing-policy.md
    - ./0108-tensor-aware-materialization-strategy-plane.md
---

# Summary

Add one generic shared capability below the existing strategy plane:

- first-class composite source -> composite target materialization in the shared
  dataplane,
- and one routed, protocol-neutral vectored pull fast path that RDMA can
  realize efficiently.

The intent is to stop treating multi-item RDMA direct-write as a
byte-artifact-specific optimization. Instead:

- `MaterializationFacade` and the shared dataplane own composite execution
  semantics,
- communicator owns efficient execution of a vectored pull plan,
- RDMA transport owns chained WR realization,
- and byte-artifact batch-get becomes only one consumer of that shared
  capability.

This design keeps the repository's current architectural boundaries:

- `0088` still owns the "one shared dataplane" rule,
- `0108` still owns semantic truth, source acquisition, and strategy placement
  in `MaterializationFacade`,
- `0087` and `0089` still own byte-artifact authority and `BodyHandle`
  lifetimes,
- and `0115` owns the new common execution contract below that seam.

One boundary matters for the next RDMA optimization step: `0115` owns the sink-
side composite execution contract and the routed vectored pull API, but it does
not by itself remove producer-side staged response windows for CPU sources.
That direct-readable source-side follow-on remains owned by `0087-01`.

# Implementation Status

As of `2026-04-20`, the generic capability owned by `0115` is implemented in
the repository.

Landed outcomes:

- `DirectWriteOp`, default `SeekableSource::readv_into_at(...)`,
  `supports_batched_direct_write_at()`, bounded `pump_ranges(...)` batching,
  and typed batch byte or op-count config are live in the shared dataplane.
- `ByteRangeMappedSource`, `RemoteKeySource`, and fallback-bearing mux paths
  now implement pre-issue batched lowering, explicit branch freeze semantics,
  and deterministic capability-miss-before-issue behavior for the vectored
  direct-write path.
- routed communicator `ReadPlan`, `PreparedReadPlan`, request-scoped
  `LocalRegion` registration, additive wire ops, and centralized validation of
  one-authority, one-route-context, non-overlap, and CPU-first-cut invariants
  are landed across routing, engine, and transport code.
- RDMA realizes `ReadPlan` with per-segment `local_lkey`, multi-local-region
  placement splitting, multi-QP posting, and request-local ACK and staged
  credit bookkeeping. MTCP intentionally returns a pre-issue capability miss
  in the first cut rather than hiding many-scalar emulation inside
  communicator.
- `MaterializationFacade` now owns a private
  `materialize_mapped_sources_into_target(...)` helper, internal
  single-source blockers required for composite execution are removed, and the
  public materialization API shape remains unchanged.
- routed byte-artifact batch-get already consumes this generic capability on
  the get path. Current RDMA share-remote reruns log
  `materialize_mode=single_source_composite batched_direct_write=true`,
  confirming that the consumer uses the shared `0115` seams rather than a
  byte-artifact-private communicator API.

Remaining follow-on work outside `0115`:

- byte-artifact source-side direct-readable RDMA servicing remains owned by
  `0087-01`: current producer CPU source paths may still copy retained backing
  bytes into pinned staged response windows before remote reads, and windowing
  is still governed by staged-flow rules rather than descriptor/control limits,
- future MTCP native vectored direct-write, if pursued, must implement the
  same shared contract rather than introducing a transport-private API.

```mermaid
flowchart LR
  A["Request<br>selection + target layout"] --> B["MaterializationFacade<br>strategy + source acquisition"]
  B --> C["Composite execution contract<br>mapped source ops + IntoTargetLayout"]
  C --> D["pump_ranges<br>bounded direct-write batches"]
  D --> E["SeekableSource::readv_into_at"]
  E --> F["RemoteKeySource<br>ReadPlan lowering"]
  F --> G["RoutingContext / Communicator<br>read_plan"]
  G --> H["RDMA transport<br>vectored READ WRs"]
  H --> I["Final CPU target buffers"]
```

# Goals / Non-Goals

## Goals

- Add one shared dataplane execution contract for multiple source spans into
  multiple target spans.
- Preserve `ByteRangeMap` and `ByteRangeMappedSource` as the exact fallback IR
  and explainability surface.
- Keep all public SDK materialization APIs unchanged.
- Keep the target-side authority abstraction as `IntoTargetLayout` and
  `DirectWriteGrant`, not transport-specific destination keys.
- Add one routed, protocol-neutral communicator API that can execute a vectored
  pull plan against one selected remote endpoint.
- Let RDMA implement the sink-side vectored pull API as one request that may
  post many WRs without forcing a sink full-pack mirror.
- Keep producer-side direct-readable source servicing as an additive consumer-
  side follow-on in `0087-01` rather than expanding `0115` into source export
  policy or staged-credit semantics.
- Preserve MTCP and staged fallback behavior as valid fallback realizations.

## Non-Goals

- Add a byte-artifact-only RDMA batch transport API.
- Introduce `local_memory_keys[]` as a new destination publication abstraction.
- Change `ArtifactSelection`, `MaterializeIntoTarget`, or
  `MaterializeIntoMappedTarget` public shapes.
- Teach communicator about `ByteRangeMap`, `IntoTargetLayout`, or
  byte-artifact-specific routing semantics.
- Solve multi-endpoint striping or multi-source load balancing inside
  communicator in this cut.
- Add remote direct writes into caller-owned CUDA regions.

# Prior Constraints Reviewed

## `0088`: one shared dataplane

Keep.

This design extends the shared dataplane rather than adding another copy
runtime. `byte_artifact` and future profiles must consume the same execution
contract.

## `0108`: strategy belongs in `MaterializationFacade`

Keep.

`0115` does not move strategy out of `MaterializationFacade`. It only defines
the execution contract that a chosen plan may use after source acquisition and
lane selection are complete.

## `0087` and `0089`: `BodyHandle` stays transport-neutral

Keep.

`BodyHandle` remains the source-side export seam for routed byte artifacts, but
the new generic capability is not defined in terms of `BodyHandle`. `BodyHandle`
is one producer of compatible sources, not the owner of transport behavior.

## Current `DirectWriteGrant` shape

Keep transport-neutral, but widen enough to name stable local backing.

`DirectWriteGrant` remains transport-neutral. It must not grow RDMA-specific
`lkey`, communicator session ids, or destination publication keys in the first
cut.

However, the target authority may attach additive stable local backing metadata
to a direct-write window when that window is carved from one long-lived local
backing object such as a daemon-managed `HOST_SHARED` slab. This metadata is
still local placement and lifetime information, not transport state.

# Architecture & Interfaces

## 1. Ownership Boundary

The new capability is split across four layers.

1. `MaterializationFacade` and shared dataplane own semantic lowering into a
   composite execution contract.
2. `SeekableSource` implementations own source-side execution of batched
   direct-write operations.
3. communicator routing and engine own a routed vectored pull API.
4. RDMA transport owns efficient realization of that API with many WRs and
   request-scoped destination registration.

Normative rules:

1. Strategy chooses whether a request uses this path. Communicator does not.
2. Communicator executes a plan. It does not derive semantic coverage.
3. One `ReadPlan` targets one resolved remote endpoint and one selected routed
   channel. Multi-hop or multi-endpoint striping remains higher-level planner
   work.
4. Destination memory remains initiator-local. The issued destination window is
   request-scoped even when it is carved from one longer-lived stable local
   backing. Destination memory is not published as a new global key family.
5. Generic execution semantics remain strict. Narrower profile-specific retry
   or overwrite policies may only layer above this seam; they do not relax the
   generic communicator or materialization contract.
6. Producer-side response-window shape, staged-credit bypass, and source-backing
   lifetime policy remain outside `0115`. Once a consumer exposes a compatible
   remote source as one logical `SeekableSource`, `0115` owns the sink-side
   batched direct-write execution against that source.

## 2. Shared Dataplane Contract

The shared dataplane needs a first-class batched direct-write operation shape.

Minimal C++ contract:

```cpp
struct DirectWriteOp {
  uint64_t src_offset = 0;
  uint64_t dest_va_offset = 0;
  uint64_t bytes = 0;
};

class SeekableSource : public Source {
 public:
  virtual bool supports_batched_direct_write_at() const;
  virtual absl::StatusOr<size_t> readv_into_at(
      absl::Span<const DirectWriteOp> ops,
      const DirectWriteGrant& grant);
};
```

Semantics:

1. `DirectWriteOp` is expressed in source byte-space and target VA byte-space.
2. The default `readv_into_at(...)` implementation loops through
   `read_into_at(...)` to preserve compatibility.
3. `supports_batched_direct_write_at()` is a stricter hint than
   `supports_direct_write_at()`. It means the source owns a real pre-issue
   batch prepare boundary, so `Unimplemented` or `FailedPrecondition` from
   `readv_into_at(...)` may be treated as a capability miss before any writes
   are assumed committed.
4. Sources that inherit the default `readv_into_at(...)` loop must keep
   `supports_batched_direct_write_at()` disabled because the loop may already
   have executed earlier ops when a later op fails.
5. `pump_ranges(...)` remains the shared entrypoint, but its direct-write path
   now groups bounded batches of `DirectWriteOp` and issues one source call per
   batch instead of one source call per window.
6. `TargetLayoutHostSink::plan_direct_write(...)` remains the target authority
   seam. The sink already supports multiple windows, so no target abstraction
   split is needed.
7. A target authority may attach transport-neutral stable local backing
   metadata to a direct-write window when it can prove that the window is a
   slice of one long-lived local backing object with quiesced teardown
   semantics.
8. Stable local backing metadata is local placement state only. It is not part
   of artifact identity, routed truth, remote source capability, or request
   semantics.
9. `supports_direct_write_at()` is only a coarse, context-free hint. It must
   not be treated as the final fast-path eligibility decision.
10. Final fast-path eligibility is decided during pre-issue prepare and
    lowering when route, protocol, source shape, target backing, and fallback
    topology are known.
11. Route-dependent capability must not be permanently cached during source
    construction. Construction-time helpers may cache only context-free facts.
12. A source that advertises `supports_batched_direct_write_at()` must reject
    deterministic batch validation errors before issuing any target writes.
    Partial target mutation is only permitted after the batch has crossed the
    issue boundary and a real execution failure occurs.

Bounded batching is mandatory:

- one batch must be capped by bytes and op count,
- grant lifetime must remain bounded,
- and future config must live under typed runtime config rather than ambient
  environment.

Recommended typed config names:

- `engine.materialization_strategy.direct_write_batch_bytes`
- `engine.materialization_strategy.direct_write_batch_ops`

## 3. Internal Materialization Entry

The shared runtime needs an internal multi-source helper rather than a
byte-artifact-specific execution path.

Minimal internal helper:

```cpp
absl::StatusOr<loading::MaterializeIntoTargetResult>
materialize_mapped_sources_into_target(
    const DeviceKey& target_device,
    const loading::IntoTargetLayout& target_layout,
    std::vector<std::shared_ptr<loader::SeekableSource>> sources,
    const loader::ByteRangeMap& mapping,
    const loading::MaterializeHints& hints,
    loading::MaterializationSource source_kind);
```

Rules:

1. `IArtifactLoader` may remain a single-source opening interface. This design
   does not require a public multi-source loader API.
2. Existing single-source helpers become wrappers around the internal
   multi-source helper.
3. `mapping.num_sources == 1` convenience restrictions must be retired where
   they block internal composite execution.
4. `ByteRangeMappedSource` remains the generic composite-source executor and
   fallback surface.

## 4. `ByteRangeMappedSource` and `RemoteKeySource`

`ByteRangeMappedSource` and `RemoteKeySource` must both become batch-aware.

### `ByteRangeMappedSource`

Responsibilities:

1. flatten composite `DirectWriteOp` batches into underlying source runs,
2. zero-fill PAD runs directly into the grant,
3. group contiguous runs by underlying source when possible,
4. delegate grouped batches to underlying `readv_into_at(...)`,
5. fall back to per-op `read_into_at(...)` only when the underlying source does
   not advertise `supports_batched_direct_write_at()`, or when a batched source
   returns a pre-issue capability miss,
6. never infer safe batched fallback from the default `readv_into_at(...)`
   compatibility loop.

### `RemoteKeySource`

Responsibilities:

1. accept a batch of `DirectWriteOp`,
2. intersect those ops with grant windows and the remote
   `memory_keys[]/buffer_sizes[]` layout,
3. lower the result into one routed communicator `ReadPlan`,
4. execute that plan through communicator when the fast path is available,
5. fall back to the existing single-read path only before the new plan has been
   issued.

### Capability Freeze and Fallback-bearing Sources

The shared contract needs one explicit pre-issue freeze point for sources whose
runtime topology includes fallback branches.

Normative rules:

1. A fallback-bearing source may enter vectored direct-write only after the
   current batch has been frozen to one concrete execution branch such as
   `primary_only` or `fallback_only`.
2. If that freeze cannot be completed before issue, the source must return a
   pre-issue capability miss and let the caller use the existing staged path.
3. This rule preserves the generic `no fallback after issue` contract while
   still allowing higher-level consumers such as byte-artifact batch-get to use
   vectored RDMA when the batch has been frozen to the primary branch.
4. The generic contract does not define in-place branch switching after issue.
   If a narrower profile later wants whole-request overwrite retry semantics,
   it must prove stronger invariants and own that behavior above `0115`.

This preserves an important layering rule:

- composite source semantics remain in the materialization layer,
- communicator sees only a vectored pull plan,
- and byte-artifact batch-get consumes the result without defining it.

## 5. Routed Communicator Read Plan

The communicator fast path must be protocol-neutral and routed, not an RDMA
helper hidden inside `RemoteKeySource`.

Minimal API shape:

```cpp
struct ReadRouteContext {
  std::string local_endpoint_id;
  std::string remote_endpoint_id;
  ConnectionProtocol protocol = ConnectionProtocol::kAuto;
  int16_t rail_id = -1;
};

enum class StableLocalBackingKind {
  kNone = 0,
  kHostSharedRegion = 1,
};

struct StableLocalBackingRef {
  StableLocalBackingKind kind = StableLocalBackingKind::kNone;
  std::string backing_id;
  uint64_t backing_base_addr = 0;
  uint64_t backing_bytes = 0;
  int dev_type = base::COMMUNICATE_ENGINE_DEV_CPU;
  int dev_id = 0;
};

struct LocalRegion {
  uint64_t addr = 0;
  uint64_t bytes = 0;
  int dev_type = base::COMMUNICATE_ENGINE_DEV_CPU;
  int dev_id = 0;
  std::optional<StableLocalBackingRef> stable_backing;
};

struct SourceSlice {
  std::string authority_id;
  ReadRouteContext route;
  std::string tensor_key;
  uint64_t remote_offset = 0;
  uint64_t bytes = 0;
};

struct ReadPlanSlice {
  uint32_t source_slice_index = 0;
  uint32_t local_region_index = 0;
  uint64_t source_slice_offset = 0;
  uint64_t local_region_offset = 0;
  uint64_t bytes = 0;
};

struct ReadPlan {
  std::vector<LocalRegion> local_regions;
  std::vector<SourceSlice> source_slices;
  std::vector<ReadPlanSlice> slices;
};

transport::future_read_result_t read_plan(const ReadPlan& plan);
```

First-cut note:

- `SourceSlice` carries explicit `authority_id` and repeated
  `ReadRouteContext`. This is intentionally redundant so one-authority and
  one-route-context invariants stay explicit and centrally validated before the
  request crosses the issue boundary. Later prepared-plan lowering may collapse
  the shared route context after validation.

Rules:

1. `ReadPlan` is request-scoped and local. It is not persisted or published.
2. `ReadPlan` must target one resolved remote endpoint and one selected routed
   channel.
3. All `SourceSlice` entries in one `ReadPlan` must belong to one remote export
   authority and execute on one selected route / protocol / rail context.
4. `ReadPlan` may describe many source slices and many local regions.
5. `local_regions` are the correct abstraction for destination memory.
   `local_memory_keys[]` is explicitly rejected because destination memory is
   not a globally published source capability.
6. `LocalRegion` may optionally carry one transport-neutral stable local
   backing reference. This names the longer-lived local placement object that
   owns the requested `[addr, bytes)` window.
7. Stable local backing references are daemon-local and additive. They do not
   change request semantics if absent, and they must not be serialized into a
   remote destination capability.
8. The first concrete `StableLocalBackingKind` is `kHostSharedRegion`,
   intended for long-lived daemon-managed `HOST_SHARED` slabs whose local
   mapping, bounds, and teardown are already owned by the daemon.
9. The `LocalRegion` API remains extensible enough for future CPU, GPU, or
   mixed-region implementations.
10. The first cut only accepts CPU `local_regions`, and all accepted regions
    must share one local registration / memory domain and one selected rail.
11. Routing owns `read_plan(...)` selection just as it owns `read_tensor(...)`
    selection today.
12. MTCP may initially return a pre-issue capability miss such as
   `Unimplemented` for `read_plan(...)`. Callers may then fall back to the
   existing staged path.
13. MTCP's first-cut capability miss is not a permanent semantic limit. A
    future MTCP implementation may realize native vectored direct-write under
    the same contract.
14. Unsupported transports must not hide a many-single-read emulation inside
    communicator `read_plan(...)`; unsupported fast paths must fail before
    issue.
15. `ReadPlan` construction and validation are still pre-issue. Mixed
    authority, mixed route context, overlapping destination spans, invalid
    `LocalRegion` coverage, and unsupported transport or region types must be
    rejected before the plan crosses the issue boundary.
16. The issue boundary is not `ReadPlan` creation. The issue boundary is when a
    validated logical plan has been prepared into a transport-ready,
    request-scoped prepared object and the routed request is handed to
    communicator / engine for execution.

### Request-scoped Prepared Object

`ReadPlan` is the logical execution intent. The runtime still needs a prepared,
request-scoped object that owns realized destination registration and
transport-ready segment lowering.

Minimal internal direction:

```cpp
struct PreparedSourcePlacement {
  uint32_t local_region_index = 0;
  uint64_t local_region_offset = 0;
  uint64_t source_slice_offset = 0;
  uint64_t bytes = 0;
};

struct PreparedLocalRegion {
  LocalRegion logical_region;
  int16_t rail_id = -1;
  std::string nic_name;
  tensor_t tensor;
};

struct PreparedReadPlan {
  ReadPlan logical_plan;
  std::string remote_endpoint_id;
  ConnectionProtocol protocol = ConnectionProtocol::kAuto;
  int16_t rail_id = -1;
  std::string local_nic;
  uint64_t total_bytes = 0;
  std::vector<PreparedLocalRegion> local_regions;
  std::vector<std::vector<PreparedSourcePlacement>> placements_by_source_slice;
};
```

Rules:

1. `PreparedReadPlan` is request-scoped and local. It is never published as a
   reusable destination capability.
2. `PreparedReadPlan` is the intended replacement seam for vectored requests
   that do not fit the single `(addr, bytes)` ownership model of
   `PartitionTensor`.
3. Engine owns preparation, local-region registration or stable-backing lookup,
   transport-specific lowering, and cleanup for this object.
4. `PartitionTensor` may remain the owner of the legacy single-region path, but
   it is not the long-term abstraction for vectored plan execution.
5. `PreparedReadPlan` is the first object that may cross from pre-issue setup
   into issue. Deterministic preparation failures here remain pre-issue and
   must not imply any target mutation.
6. `ReadPlanSlice` carries `source_slice_offset`, and prepared placement
   coverage for each `SourceSlice` must exactly cover `[0, source.bytes)`
   without gaps or overlap before the plan crosses the issue boundary.
7. The current first cut also requires all accepted local regions to resolve to
   one selected RDMA NIC / rail so the request owns one transport-local memory
   domain.
8. When `logical_region.stable_backing` is present and resolved, preparation
   may reuse one or more preregistered registration chunks that belong to that
   backing object rather than registering a request-scoped destination tensor
   for the narrower window.

### Stable Local Backing and Chunked MR Reuse

The request-local `LocalRegion` window and the longer-lived placement object
behind it are different semantic layers and should stay separate.

The accepted widening is:

- `TargetLayoutHostSink::plan_direct_write(...)` may report that a requested
  window belongs to one stable local backing object,
- `ReadPlan::LocalRegion` threads that transport-neutral backing reference to
  communicator,
- and communicator may then realize request-local writes with MR reuse over the
  selected registration chunks inside that backing object instead of
  request-scoped `reg_mr` on each window.

For long-lived KV `HOST_SHARED` slabs, stable local backing remains
slab-scoped semantically, but RDMA MR reuse must be registration-chunk-scoped
operationally. The first landed whole-slab eager-preregistration geometry is
now explicitly rejected for this workload because large backing-wide MRs
regressed small-window completion latency on RDMA batch-get.

Normative rules:

1. Stable local backing describes local placement only. It does not make the
   backing object a routed capability and it does not let remote daemons write
   directly into caller-visible memory.
2. The requested destination window remains `[addr, bytes)` on one issued
   request. Stable backing only widens the registration envelope available to
   the local engine.
3. First cut support is only for daemon-managed long-lived `HOST_SHARED`
   backing with one daemon-local stable identity and one daemon-local mapping.
4. For that first cut, the accepted optimization policy is
   slab-scoped stable backing plus a lazy rail-local registration-chunk MR
   cache. The runtime must not preregister the full slab as one MR for KV
   sink-side direct-write reuse.
5. Registration chunks are an execution detail, not a new semantic identity.
   Correctness remains owned by slab lifetime plus allocator slot and
   generation rules above communicator.
6. First-cut registration chunks must be fixed-width and slot-aligned inside
   one backing. The accepted daemon YAML surface is:

   ```yaml
   communicator:
     enable_rdma: true
     rdma:
       enable_stable_local_mr_reuse: true
       stable_local_mr_reuse_chunk_slots: 1  # default: 1
       stable_local_mr_reuse_prewarm_workers: 0  # default: disabled; >0 enables async prewarm
   ```

   `stable_local_mr_reuse_chunk_slots` is the number of allocator slots per
   registration chunk. `chunk_bytes = chunk_slots * slot_bytes`.
   `stable_local_mr_reuse_prewarm_workers` bounds communicator-local
   background preregistration concurrency and is also the canonical enable
   switch: unset or `0` disables async prewarm, while values `>0` enable one
   prewarm job per visible rail on a bounded worker pool.
   The deprecated legacy alias
   `stable_local_mr_reuse_eager_prereg_all_rails` is accepted only for
   backward compatibility and maps to `prewarm_workers=1` when true if the
   canonical field is unset.
7. Chunk sizing must be driven by slot/page working-set locality rather than by
   total slab size. The default is `1` slot per chunk; any larger value must
   remain slot-aligned and bounded.
8. Preparation on one selected rail may ensure and cache only the registration
   chunks overlapped by the accepted `LocalRegion`s. Reuse on one selected rail
   must not depend on every other candidate rail already being prepared.
9. Activation may publish backing eligibility before all chunk MRs exist.
   First-cut serveability is not gated on full preregistration completion.
   Background prewarm is best-effort latency optimization only; requests may
   still lazily register selected chunks on cache miss.
10. First-cut background prewarm is rail-local. Activation may enqueue at most
    one prewarm walk per visible rail, and each walk may scan chunk indices for
    that rail. There is no mandatory orchestrator or request-path readiness
    barrier for full prewarm completion.
11. The backing registry lock is a short publish/remove lock only. It must not
    cover preregistration loops or request-time chunk activation.
12. `reg_mr` must not execute while holding a backing-scoped mutex shared by
    unrelated rails or chunks. The maximum serialization scope is one
    `(stable_backing_id, rail_id, chunk_index)` entry.
13. Request and prewarm must share one entry-level registration state machine.
    At most one registrar may own one `(rail, chunk)` at a time; other
    contenders must reuse or wait for that entry's outcome rather than issuing
    duplicate `reg_mr`.
14. A request may wait only for an in-flight registrar of the same
    `(rail, chunk)` entry. Unrelated rails, chunks, and remote shards must
    remain independent.
15. If chunk registration or lookup fails on the selected rail, the
    `HOST_SHARED` slab remains a valid local placement object, but the request
    must fall back before issue to request-scoped destination registration.
    Incomplete background prewarm alone is not a failure.
16. Teardown must quiesce in-flight requests and background prewarm tasks that
    reference the backing before deregistering any live chunk MRs on rails
    where they were activated and before releasing the local mapping.
17. Request-scoped registration remains the required fallback for local
    regions that do not carry stable backing or whose chunk-cache path is
    unavailable.
18. Stable backing does not weaken allocator slot or generation semantics.
    `slot_index`, `slot_generation`, stale-completion filtering, and
    visibility rules remain owned above communicator.

### Accepted Async Prewarm Concurrency Model

The accepted production direction for sink-side stable local MR reuse is:

- keep stable backing slab-scoped semantically,
- keep MR reuse rail-local and chunk-scoped operationally,
- allow activation to enqueue best-effort background prewarm,
- and remove request/prewarm interference by narrowing registration
  synchronization to one `(backing, rail, chunk)` entry.

The intended state split is:

- `backing registry`
  - one short-lived lock guards publish/remove of
    `stable_backing_id -> StableLocalBackingState`.
  - this lock is not part of the hot preregistration path.
- `backing lifecycle`
  - one backing-scoped lifecycle state owns `retiring`, `inflight_uses`, and
    `activation_keepalive`.
  - both request-path users and prewarm workers acquire a use token so
    deactivate can drain them uniformly.
- `rail registration table`
  - one rail-scoped map resolves `rail_id -> RailState`.
  - each `RailState` owns its chunk-entry map.
- `chunk entry`
  - one entry is keyed by `(stable_backing_id, rail_id, chunk_index)`.
  - one entry owns the only synchronization that may guard one actual
    `reg_mr()` call.

The accepted execution rule is:

- request path and prewarm path call the same entry-level
  `get_or_register_chunk(...)` routine,
- that routine may decide who is the registrar under the entry lock,
- but the actual `reg_mr()` must happen after releasing every coarse
  backing-scoped lock,
- and the registrar later publishes `Ready` or `Failed` back into the same
  entry before waking any waiters.

This means:

- unrelated rails can preregister in parallel,
- unrelated chunks on the same rail can preregister in parallel,
- a request only waits when it collides with the same `(rail, chunk)` already
  being registered by another request or a prewarm worker,
- and background prewarm never needs to block request service for the whole
  backing.

The accepted first-cut activation/prefetch policy is:

- activation publishes the stable backing immediately once local geometry and
  keepalive are valid,
- communicator then enqueues one best-effort prewarm job per visible rail onto
  a bounded worker pool,
- worker concurrency is controlled by
  `communicator.rdma.stable_local_mr_reuse_prewarm_workers`,
- the canonical configuration is unset or `0` for disabled, and `>0` for
  enabled; the common first-cut value is `1`, which keeps behavior simple and
  avoids adding more thread contention on top of existing remote-shard fanout,
- and requests remain correct even if no prewarm job has yet reached a needed
  chunk because lazy chunk registration remains valid on miss.

### Chunk Entry State Machine

The accepted first-cut chunk entry lifecycle is:

```mermaid
stateDiagram-v2
    [*] --> Empty
    Empty --> Registering: request/prewarm claims registrar
    Failed --> Registering: retry claims registrar
    Registering --> Ready: reg_mr succeeds
    Registering --> Failed: reg_mr fails or backing retires
    Ready --> Ready: cache hit / reuse
    Ready --> [*]: backing retire then dereg_mr
    Failed --> [*]: backing retire
```

Operational notes:

- `Empty`
  - no MR exists yet for this `(rail, chunk)`.
- `Registering`
  - exactly one registrar owns the in-flight `reg_mr()`.
  - other contenders for the same entry may wait for completion, but unrelated
    entries do not participate.
- `Ready`
  - the entry has a reusable MR and may satisfy request-path preparation
    without request-scoped `reg_mr`.
- `Failed`
  - the last registration attempt failed or the backing retired before the
    result could be published.
  - request path may still choose deterministic fallback before issue.

### Activation, Request, and Retire Flow

```mermaid
sequenceDiagram
    participant Store as Store / SGLang
    participant Comm as Communicator
    participant Pool as Prewarm Worker Pool
    participant Req as Request Path
    participant Entry as Chunk Entry

    Store->>Comm: activate_stable_local_backing(backing)
    Comm->>Comm: publish backing state
    Comm->>Pool: enqueue one prewarm job per visible rail

    Req->>Comm: prepare_read_plan(...)
    Comm->>Entry: lookup (rail, chunk)
    alt entry Ready
        Entry-->>Comm: reusable MR
    else entry Registering
        Entry-->>Comm: wait same entry only
    else entry Empty / Failed
        Comm->>Entry: claim registrar
        Comm->>Comm: reg_mr() outside coarse backing lock
        Comm->>Entry: publish Ready / Failed
    end

    Comm->>Comm: deactivate_stable_local_backing(backing)
    Comm->>Comm: stop new users and drain request + prewarm uses
    Comm->>Entry: dereg_mr for ready entries
```

### Minimal wire schema

The wire shape should remain internal and additive. A minimal internal protocol
family is:

```cpp
enum {
  ENGINE_OP_READ_PLAN_REQUEST,
  ENGINE_OP_READ_PLAN_RESPONSE_EX,
  ENGINE_OP_READ_PLAN_FAILED,
  ENGINE_OP_RDMA_READ_PLAN_DONE_EX,
};

struct ProtoReadPlanRequestHeader {
  uint8_t transport_type = 0;
  int16_t rail_id = -1;
  uint8_t reserved = 0;
  uint64_t request_id = 0;
  uint32_t num_source_slices = 0;
};

struct ProtoReadPlanSourceSlice {
  char tensor_key[kMaxTensorNameLen];
  uint32_t source_slice_index = 0;
  uint64_t remote_offset = 0;
  uint64_t bytes = 0;
};

struct ProtoReadPlanResponseExHeader {
  uint8_t transport_type = 0;
  uint8_t staged = 0;
  char nic_name[kMaxDevName];
  uint64_t request_id = 0;
  uint32_t num_segments = 0;
  uint32_t window_seq = 0;
  uint32_t credit_granted = 0;
  uint8_t more_segments = 0;
  uint8_t zero_copy = 0;
  int16_t rail_id = -1;
};

struct ProtoReadPlanResponseExSeg {
  uint32_t source_slice_index = 0;
  uint64_t source_slice_offset = 0;
  uint64_t addr = 0;
  uint32_t bytes = 0;
  uint32_t rkey = 0;
};

struct ProtoReadPlanFailed {
  uint64_t request_id = 0;
  uint32_t reason = 0;
};

struct ProtoRdmaReadPlanDoneExHeader {
  uint64_t request_id = 0;
  uint32_t num_segments = 0;
  uint32_t window_seq = 0;
  uint8_t final_window = 0;
  uint8_t reserved[3] = {};
};
```

The source daemon does not need local addresses. It only needs the source slice
set and must return staged or direct segments keyed by request id, window, and
slice identity.

Normative rules:

1. `source_slice_index` and `source_slice_offset` are sink-side placement
   metadata. They do not replace request-local ACK or staged-lease identity.
2. Source-side staged credit release remains keyed by request-local
   `window_seq + segment_idx`.
3. `READ_PLAN_FAILED` is additive and keyed by `request_id`; it does not change
   legacy tensor-key / offset failure decoding.
4. `READ_PLAN_RESPONSE_EX` carries transport, NIC, rail, staged, and zero-copy
   metadata so the sink can route the response through the same handshake and
   profiling machinery as legacy RDMA reads.
5. `RDMA_READ_PLAN_DONE_EX` carries request-local staged credit release for
   plan windows and remains keyed by `window_seq + segment_idx` rather than
   semantic slice identity.

## 6. RDMA Transport Realization

RDMA can already post many WRs, but the current transport contract assumes one
request-scoped local MR. That is too narrow.

Required widening:

```cpp
struct RdmaReadSeg {
  uint64_t local_addr = 0;
  uint32_t local_lkey = 0;
  uint32_t length = 0;
  uint64_t remote_addr = 0;
  uint32_t rkey = 0;
  uint32_t window_seq = 0;
  uint32_t segment_idx = 0;
};
```

Rules:

1. engine owns preparation of `local_regions` on the selected rail, either by
   request-scoped registration or by stable-backing chunk reuse,
2. transport consumes resolved `local_addr + local_lkey` pairs,
3. one RDMA request may post many WRs whose SGEs use different `lkey` values,
4. request progress, completion, and profiling remain aggregated at the engine
   layer.
5. The additive `local_lkey` field has already landed on the existing
   `RdmaReadSeg` type instead of introducing a parallel `RdmaReadSegEx`
   transport shape.
6. RDMA lowering may split one response segment into multiple WRs when the
   response range overlaps multiple prepared placements. All resulting WRs keep
   the same logical `window_seq + segment_idx` metadata while carrying the
   resolved `local_addr + local_lkey` for each overlap chunk.
7. Partial `ibv_post_send` failure is a hard request failure: already-posted
   WRs remain inflight and may still complete locally, but the request result
   becomes terminally failed and unposted WRs are not counted.
8. When a prepared local region resolves through stable-backing MR reuse,
   `local_lkey` is derived from the selected registration chunk MR that covers
   the overlap while `local_addr` still points at the narrower request-local
   window inside that backing object.
9. The first-cut stable-backing reuse path is CPU-only and `HOST_SHARED`-only.
   GPU or mixed-region backing may reuse the same abstraction later, but is not
   implied by this design.

`PartitionTensor` is not the long-term owner of this path because it is a
single `(addr, bytes)` abstraction. The vectored path should use a separate
request-scoped prepared object and local region set.

## 7. Control and Data Flow

```mermaid
sequenceDiagram
  participant U as Caller
  participant MF as MaterializationFacade
  participant P as pump_ranges
  participant BMS as ByteRangeMappedSource
  participant RKS as RemoteKeySource
  participant RC as RoutingContext
  participant CE as Communicator
  participant SD as Source Daemon

  U->>MF: materialize into target
  MF->>P: execute shared dataplane plan
  P->>BMS: readv_into_at(batch, grant)
  BMS->>RKS: readv_into_at(flattened batch, grant)
  RKS->>RC: read_plan(plan)
  RC->>CE: read_plan(plan)
  CE->>SD: READ_PLAN request
  SD-->>CE: READ_PLAN_RESPONSE_EX or READ_PLAN_FAILED
  CE-->>CE: post vectored RDMA READ WRs
  CE->>SD: RDMA_READ_PLAN_DONE_EX for staged windows
  CE-->>RKS: future result
  RKS-->>BMS: bytes committed
  BMS-->>P: batch success
  P-->>MF: target bytes visible
```

## 8. Failure Model

The key rule is that fallback is only allowed before a vectored direct-write
batch has been issued.

```mermaid
stateDiagram-v2
  [*] --> Prepared
  Prepared --> FallbackReady: setup failure before issue
  Prepared --> Issued: read_plan issued
  FallbackReady --> StagedFallback
  Issued --> Completed: all segments complete
  Issued --> FailedAfterIssue: transport or completion failure
  FailedAfterIssue --> PoisonedTarget
  StagedFallback --> Completed
  PoisonedTarget --> [*]
  Completed --> [*]
```

Normative rules:

1. If batching, source-freeze, routing, registration, or transport selection
   fails before the plan is issued, callers may fall back to the existing
   staged path.
2. A batch is considered issued once the routed plan request has been handed
   off such that destination memory may subsequently be dirtied.
3. Once a vectored direct-write batch is issued, in-place staged fallback is
   forbidden.
4. Destination spans in one issued plan must be non-overlapping. Overlap must
   be rejected before issue.
5. Post-issue failure must surface as a hard request failure and follow the
   existing poisoned-target semantics of the caller.
6. The generic contract does not define hidden branch switching, hidden
   transport emulation, or hidden overwrite retry after issue.
7. A narrower profile may layer whole-request retry or overwrite-retry above
   this contract only if it proves stronger invariants such as immutable
   sources, equivalent fallback bytes, and private unpublished targets.
8. These rules avoid silent double-write, partial overwrite, and ambiguous
   commit semantics.

## 9. Configuration and Observability

Required direction:

1. batch sizing must be typed config, not environment variables,
2. communicator must expose read-plan counters and latency,
3. materialization must expose whether a request used:
   - `direct_write_mode=single_op`
   - `direct_write_mode=batched_ops`
   - `direct_write_mode=read_plan`
   - `direct_write_mode=staged_fallback`
4. fallback cause must be explicit:
   - unsupported_source
   - unsupported_transport
   - registration_failure
   - routing_failure
   - pre_issue_setup_failure
5. read-plan preparation must expose whether each accepted `LocalRegion` used:
   - `local_registration_mode=request_scoped`
   - `local_registration_mode=stable_backing_reuse`
6. stable-backing logs and metrics must identify the backing kind, selected
   rail, chunk geometry, whether the request hit the registration-chunk
   cache, had to wait on an in-flight registrar, or had to register chunks
   lazily.
7. the accepted first-cut daemon YAML surfaces are:
   - `communicator.rdma.stable_local_mr_reuse_chunk_slots`, default `1`
   - `communicator.rdma.stable_local_mr_reuse_prewarm_workers`, default disabled
8. stable-backing prewarm observability must expose whether a backing was
   activated lazily or with background prewarm, how many rail jobs were
   enqueued, and whether completion lagged request service.

Recommended metrics:

- `tc_tx_direct_batch_ops_total`
- `tc_tx_direct_batch_bytes_total`
- `tc_comm_read_plan_requests_total`
- `tc_comm_read_plan_segments_total`
- `tc_comm_read_plan_fallback_total`
- `tc_comm_read_plan_local_region_registrations_total{mode,backing_kind}`
- `tc_comm_read_plan_local_region_prereg_failures_total{backing_kind}`
- `tc_comm_stable_backing_chunk_cache_events_total{event,backing_kind}`
- `tc_comm_stable_backing_chunk_cache_bytes{backing_kind}`
- `tc_comm_stable_backing_chunk_wait_ms_total{reason,backing_kind}`
- `tc_comm_stable_backing_chunk_registrars_total{result,backing_kind}`
- `tc_comm_stable_backing_prewarm_jobs_total{state,backing_kind}`

## 10. Naming Compliance

The proposed interface family follows repository naming rules.

- `DirectWriteOp`, `StableLocalBackingKind`, `StableLocalBackingRef`,
  `LocalRegion`, `SourceSlice`, `ReadPlanSlice`, `ReadPlan`, and `RdmaReadSeg`
  are `PascalCase` types.
- `readv_into_at(...)`, `materialize_mapped_sources_into_target(...)`, and
  `read_plan(...)` are `snake_case` functions or methods.
- future config fields such as `direct_write_batch_bytes` and
  `direct_write_batch_ops` remain `snake_case`.
- existing constants such as `ENGINE_OP_*` remain `ALL_CAPS`.

# Trade-offs & Risks

- This design adds one more internal contract family and a new communicator
  wire path.
- Request-scoped destination registration may add overhead until stable-backing
  reuse or other coalescing rules are tuned.
- Stable-backing chunk reuse replaces per-request `reg_mr` with rail-local
  chunk activation and cache management. This adds one more cache policy
  surface and more MR objects per backing than the rejected whole-slab design.
- Chunk sizes that are too small raise MR cardinality and first-touch
  activation churn; chunk sizes that are too large recreate the address
  translation locality problems observed with whole-slab preregistration.
- Quiesced teardown is stricter when one stable backing may hold chunk MRs on
  multiple rails because all in-flight users of that backing must drain before
  deregistration.
- Strict "no in-place fallback after issue" semantics make failures louder, but
  that is preferable to silent target corruption.
- MTCP may lag behind RDMA in immediate benefit because the first optimized
  production implementation is RDMA.

# Compatibility & Acceptance Criteria

Compatibility rules:

1. Public SDK and daemon request APIs do not change.
2. `ByteRangeMap` remains the exact fallback IR.
3. `DirectWriteGrant` stays transport-neutral in the first cut.
4. MTCP and staged paths remain valid fallbacks.
5. `0087` byte-artifact batch-get may consume this capability later but may not
   redefine it.
6. Stronger profile-level retry policy, if ever added, must layer above this
   contract rather than weaken the generic failure model.
7. Stable local backing metadata remains local placement state and must not be
   promoted into routed identity or remote destination capability.

Acceptance criteria:

1. shared dataplane can execute bounded batched direct-write without going
   through byte-artifact-specific loops,
2. `RemoteKeySource` can lower a batch into one routed communicator plan after
   pre-issue source freeze,
3. communicator can execute one vectored pull plan against one remote
   authority, one endpoint, and one selected route context,
4. RDMA can post many WRs across a request-scoped prepared local region set,
5. long-lived daemon-managed `HOST_SHARED` target backing can be represented as
   additive stable local backing and may reuse one or more slot-aligned
   registration-chunk MRs on the selected rail,
6. the `LocalRegion` interface remains extensible while the first cut accepts
   CPU regions only,
7. pre-issue fallback works,
8. post-issue failure never silently falls back in place,
9. source-side staged ACK and lease release remain keyed by request-local
   `window_seq + segment_idx`,
10. MTCP behavior remains functionally unchanged when the fast path is
    unavailable.

# References

- [0088 Shared Dataplane](./0088-unified-artifact-profiles-with-shared-dataplane.md)
- [0087 Routed Byte Artifact Architecture](./0087-unified-artifact-runtime-and-routed-byte-artifact-architecture.md)
- [0089 Core-Backed Body Handles](./0089-core-backed-body-handles-and-backing-policy.md)
- [0108 Strategy Plane](./0108-tensor-aware-materialization-strategy-plane.md)
