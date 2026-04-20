---
slug: composite-materialization-and-vectored-direct-write
title: Composite Materialization and Vectored Direct-Write Plan
status: implemented
areas: ["core", "daemon", "docs", "benchmarks", "integrations", "tests"]
created: 2026-04-19
last_updated: 2026-04-20
related_code:
  - core/store/materialization/dataplane/contracts/source.h
  - core/store/materialization/dataplane/runtime/pump.h
  - core/store/materialization/dataplane/runtime/pump.cc
  - core/store/materialization/dataplane/runtime/tests/pump_direct_test.cc
  - core/store/materialization/dataplane/sources/byte_range_mapped_source.cc
  - core/store/materialization/dataplane/sources/tests/mux_seekable_source_test.cc
  - core/store/materialization/dataplane/sources/tests/remote_key_source_routing_fallback_test.cc
  - core/store/materialization/dataplane/sources/remote_key_source.h
  - core/store/materialization/dataplane/sources/remote_key_source.cc
  - core/store/runtime/ingestion/materialization_facade.h
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/runtime/ingestion/materialization_facade_test.cc
  - core/communicator/routing/types.h
  - core/communicator/routing/connection.h
  - core/communicator/routing/adapter.h
  - core/communicator/routing/routing_context.h
  - core/communicator/routing/routing_context_test.cc
  - core/communicator/engine/engine.h
  - core/communicator/engine/protocol.h
  - core/communicator/engine/rdma_engine_test.cc
  - core/communicator/engine/staging_flow_controller_test.cc
  - core/communicator/transport/request.h
  - core/communicator/transport/request_test.cc
  - core/communicator/transport/partition_tensor.h
  - core/communicator/transport/rdma_transport.h
  - docs/plans/0087-01-routed-byte-artifact-batch-get-rdma-zero-copy-realization.md
links:
  design: ../designs/0115-composite-materialization-and-vectored-direct-write.md
---

# Objective

Implement the shared execution capability defined by `0115`:

- first-class composite source -> composite target materialization in the
  shared dataplane,
- routed communicator `read_plan(...)`,
- and RDMA vectored direct-write realization over request-scoped destination
  regions.

This plan intentionally owns the generic capability, not the byte-artifact
consumer path. `0087-01` may consume the result later but must not redefine it.

# Starting State & Grounding

At plan start, before the implementation below landed:

- `SeekableSource` exposes `read_at(...)` and `read_into_at(...)`, but not a
  first-class batched direct-write contract.
- The default `SeekableSource::readv_into_at(...)` compatibility loop is not a
  safe signal for pre-issue vectored capability fallback because it may already
  have issued earlier scalar writes when a later op fails.
- Batched-capable sources must now converge on two-phase semantics: deterministic
  validation errors are pre-issue and must not mutate the target before the
  batch is issued.
- `pump_ranges(...)` already owns the shared direct-write fast path, but it
  still plans and executes one window at a time.
- `ByteRangeMappedSource` already represents composite sources, but current
  convenience helpers still reject `mapping.num_sources != 1` in important
  paths.
- `RemoteKeySource` already supports direct-write into CPU windows, but it
  still issues one logical read at a time.
- communicator routing and engine only expose `read_tensor(...)`, not a public
  vectored pull plan.
- `RdmaTransport::read_multi(...)` can post many WRs, but its current request
  model assumes one local MR for the whole request.
- `0087-01` currently depends on a generic capability that does not yet exist:
  sink-side no-mirror is landed, but source-side no-pack segmented export and
  true batch direct-write still need the common primitive owned here.
- The repo already has useful test anchors that should absorb most of this
  work rather than creating an entirely separate test stack:
  `pump_direct_test.cc`, `mux_seekable_source_test.cc`,
  `remote_key_source_routing_fallback_test.cc`, `routing_context_test.cc`,
  `rdma_engine_test.cc`, `staging_flow_controller_test.cc`,
  `request_test.cc`, and `materialization_facade_test.cc`.

# Phases & Milestones

- [x] Phase 1: Contract Freeze
  - [x] Milestone 1.1: Land `0115` design and this plan as the sole owner of
    the generic capability.
  - [x] Milestone 1.2: Freeze the `DirectWriteOp`, `readv_into_at(...)`, and
    `ReadPlan` contract shapes before implementation starts.
  - [x] Milestone 1.3: Freeze the failure rule that forbids in-place staged
    fallback after a vectored direct-write batch is issued.
  - [x] Milestone 1.4: Freeze the rule that `supports_direct_write_at()` is a
    coarse hint and that final fast-path eligibility is decided only during
    pre-issue prepare.
  - [x] Milestone 1.5: Freeze the rule that fallback-bearing sources must be
    pre-issue frozen to one execution branch before entering vectored
    direct-write.

- [x] Phase 2: Shared Dataplane Batched Direct-Write
  - [x] Milestone 2.1: Add `DirectWriteOp` and default
    `SeekableSource::readv_into_at(...)`.
    - [x] 2.1.a: Introduce `DirectWriteOp` in the shared source contract with
      stable byte-space semantics.
    - [x] 2.1.b: Add the default `readv_into_at(...)` loop implementation and
      thread it through wrapper or adapter sources that currently forward only
      `read_at(...)` and `read_into_at(...)`.
    - [x] 2.1.c: Preserve backward compatibility for sources that do not
      override batching.
    - [x] 2.1.test: Extend
      `core/store/materialization/dataplane/runtime/tests/pump_direct_test.cc`
      or an adjacent source-contract suite to cover zero-op no-op,
      ordered multi-op execution, exact byte accounting, and first-error
      propagation of the default loop.
  - [x] Milestone 2.2: Change `pump_ranges(...)` direct-write execution from
    one-window calls to bounded batched calls.
    - [x] 2.2.a: Add a bounded op-batch builder keyed by byte limit and op
      count limit.
    - [x] 2.2.b: Keep grant lifetime and target-window validity bounded to one
      issued batch.
    - [x] 2.2.c: Emit explicit direct-write mode and fallback cause observability.
    - [x] 2.2.test: Extend
      `core/store/materialization/dataplane/runtime/tests/pump_direct_test.cc`
      to cover single-batch multi-window success, splitting by op count,
      splitting by byte limit, and pre-issue fallback without partial target
      corruption.
  - [x] Milestone 2.3: Add typed config plumbing for batch byte and op limits.
    - [x] 2.3.a: Add typed config fields for batch byte and op limits.
    - [x] 2.3.b: Plumb defaults through the runtime path that owns
      `pump_ranges(...)`.
    - [x] 2.3.c: Make batch sizing observable in logs or metrics for debugging.
    - [x] 2.3.test: Add config-plumbing tests in
      `core/store/runtime/ingestion/materialization_pump_options_test.cc`
      plus direct batching regressions in
      `core/store/materialization/dataplane/runtime/tests/pump_direct_test.cc`
      to verify defaults, explicit overrides, and limit enforcement.

- [x] Phase 3: Source Implementations
  - [x] Milestone 3.1: Teach `RemoteKeySource` to lower batched direct-write
    work into one communicator `ReadPlan` after pre-issue source freeze.
    - [x] 3.1.a: Add explicit pre-issue branch freeze for fallback-bearing
      remote sources.
    - [x] 3.1.b: Lower one batch of `DirectWriteOp` into
      `SourceSlice` plus `ReadPlanSlice` over `memory_keys[]/buffer_sizes[]`.
      - `RemoteKeySource::readv_into_at(...)` now lowers routed batches into a
        logical communicator `ReadPlan` with explicit `authority_id` and
        `ReadRouteContext`.
      - If routed `read_plan(...)` returns an immediate
        `Unimplemented`/`FailedPrecondition` capability miss, the source falls
        back before issue to the pre-frozen routed per-op execution branch.
    - [x] 3.1.c: Return capability miss before issue when routing, protocol,
      or source-freeze preconditions are not met.
    - [x] 3.1.test: Extend
      `core/store/materialization/dataplane/sources/tests/remote_key_source_routing_fallback_test.cc`
      to cover routed success, routing miss before issue, MTCP
      `Unimplemented` before issue, and source-freeze gating.
  - [x] Milestone 3.2: Teach `ByteRangeMappedSource` to flatten, group, and
    batch delegate `DirectWriteOp` work.
    - [x] 3.2.a: Flatten batched direct-write work across contiguous source
      runs while preserving target offsets.
    - [x] 3.2.b: Zero-fill PAD runs directly into the destination grant.
    - [x] 3.2.c: Delegate grouped runs to underlying `readv_into_at(...)`
      where supported and fall back to per-op `read_into_at(...)` otherwise.
    - [x] 3.2.d: Pre-validate and lower the whole batch before issuing any
      target writes so deterministic validation errors remain pre-issue.
    - [x] 3.2.test: Add a dedicated `ByteRangeMappedSource` direct-write test
      suite if needed, or extend existing dataplane source tests, to cover PAD
      zero-fill, grouped contiguous delegation, mixed-source grouping, and
      per-op fallback.
  - [x] Milestone 3.3: Preserve exact staged and per-op fallback when the
    lower layer does not support batching.
    - [x] 3.3.a: Make pre-issue fallback causes explicit and deterministic.
    - [x] 3.3.b: Ensure batch lowering never silently degrades into hidden
      post-issue fallback.
    - [x] 3.3.test: Add regression tests that distinguish
      capability-miss-before-issue from hard-failure-after-issue.
  - [x] Milestone 3.4: Reject issue-time in-place fallback for mux or other
    fallback-bearing sources that could not be frozen before issue.
    - [x] 3.4.a: Make mux capability depend on pre-issue freeze rather than
      unconditional branch probing.
    - [x] 3.4.b: Preserve existing staged `read_at(...)` fallback behavior
      outside the vectored direct-write path.
    - [x] 3.4.test: Extend
      `core/store/materialization/dataplane/sources/tests/mux_seekable_source_test.cc`
      to cover pre-issue freeze success, capability miss when freeze is
      impossible, and explicit rejection of issue-time branch switching.

Phase 3 note:

- The shared dataplane now distinguishes coarse scalar direct-write capability
  from safe batched capability via `supports_batched_direct_write_at()`. Only
  sources that advertise the latter may return `Unimplemented` or
  `FailedPrecondition` from `readv_into_at(...)` as a pre-issue capability
  miss that callers may fall back from.
- Batched-capable sources now follow two-phase semantics:
  `pre-validate/lower -> issue`. Deterministic validation failures must happen
  before any target mutation; partial target mutation is reserved for
  post-issue execution failures only.

- [x] Phase 4: Routed Communicator Read Plan
  - [x] Milestone 4.1: Add routed communicator `ReadPlan` API types and
    `read_plan(...)`.
    - [x] 4.1.a: Add logical `ReadPlan` types in the routing seam.
      - Landed in `core/communicator/routing/types.h` as `ReadRouteContext`,
        `LocalRegion`, `SourceSlice`, `ReadPlanSlice`, and `ReadPlan`.
      - `SourceSlice` currently carries explicit `authority_id` and
        `ReadRouteContext` so one-authority / one-route-context validation is
        explicit before issue.
    - [x] 4.1.b: Add entrypoints in routing, route-channel, adapter, and
      engine layers without changing public SDK APIs.
      - `read_plan(...)` now threads through `RoutingContext::Communicator`,
        `RouteChannel`, `Connection`, `ConnectionAdapter`, and
        `engine::Communicator`.
      - Current engine / local-adapter realization is still an additive stub:
        unsupported paths return an immediate pre-issue capability miss rather
        than hiding many-single-read emulation inside communicator.
    - [x] 4.1.c: Keep `read_tensor(...)` intact as the legacy path.
      - Legacy routed `read_tensor(...)` remains unchanged and is still used by
        `RemoteKeySource` as the explicit pre-issue fallback branch.
    - [x] 4.1.test: Extend
      `core/communicator/routing/routing_context_test.cc` or add a dedicated
      `read_plan` validation suite for route construction, protocol selection,
      and request-shape validation.
      - Added dedicated `core/communicator/routing/read_plan_validation_test.cc`.
  - [x] Milestone 4.2: Add engine-side request-scoped local region management,
    `PreparedReadPlan`, progress, and profiling for plan execution.
    - [x] 4.2.a: Introduce `PreparedReadPlan` as the request-scoped prepared
      object replacing the single-region ownership model of `PartitionTensor`
      for this path.
      - Landed in `core/communicator/transport/request.h` as
        `PreparedReadPlan`, `PreparedLocalRegion`, and
        `PreparedSourcePlacement`, with additive `ReadRequest` construction for
        request-scoped routed plans.
    - [x] 4.2.b: Implement CPU-only local-region registration and cleanup for
      the first cut.
      - `engine::Communicator::read_plan(...)` now registers each accepted CPU
        `LocalRegion` as a request-scoped `PartitionTensor`, enforces one local
        NIC / rail for the first cut, and relies on prepared-plan / request
        teardown for cleanup.
      - Local-region preparation still enforces one selected local NIC / rail
        in the first cut, but source-slice placement is no longer restricted to
        one full-slice placement.
    - [x] 4.2.c: Propagate progress and profiling fields through the prepared
      request lifecycle.
      - `ReadRequest` now carries explicit `total_bytes`, plan-mode progress,
        segment-count ACK windows, and additive RDMA profiling counters for
        routed plan requests.
    - [x] 4.2.test: Extend `core/communicator/transport/request_test.cc` and
      `core/communicator/engine/cpu_ce_test.cc` or adjacent engine tests to
      cover request-scoped cleanup, progress aggregation, and CPU-only region
      validation.
      - Covered by `core/communicator/transport/request_test.cc`,
        including explicit request-lifetime ownership / cleanup coverage for
        prepared local regions,
        `core/communicator/routing/read_plan_validation_test.cc`, and
        adjacent RDMA engine tests rather than `cpu_ce_test.cc`.
  - [x] Milestone 4.3: Add additive wire ops for plan request, plan response,
    and plan ACK.
    - [x] 4.3.a: Add additive protocol structs and engine dispatch.
      - Added `ENGINE_OP_READ_PLAN_REQUEST`,
        `ENGINE_OP_READ_PLAN_RESPONSE_EX`, `ENGINE_OP_READ_PLAN_FAILED`, and
        `ENGINE_OP_RDMA_READ_PLAN_DONE_EX` plus additive `ProtoReadPlan*`
        payloads.
    - [x] 4.3.b: Keep the current read-response wire path intact for legacy
      callers.
      - Legacy `read_tensor(...)` request / response / ACK flow remains intact.
    - [x] 4.3.c: Ensure plan ACK paths can coexist with existing staged-window
      bookkeeping.
      - `ReadRequest` now supports both offset-based ACK windows and
        segment-count plan ACK windows inside the same request machinery.
    - [x] 4.3.test: Extend
      `core/communicator/engine/rdma_engine_test.cc` to cover plan request
      decode, plan response encode/decode, and additive ACK flow.
      - `rdma_engine_test.cc` now covers missing-plan failures,
        `READ_PLAN_FAILED` resolution by request id, and
        `READ_PLAN_RESPONSE_EX` decode / ACK queueing.
  - [x] Milestone 4.4: Keep one-plan-one-authority, one-route-context, and
    non-overlapping-destination invariants explicit and validated.
    - [x] 4.4.a: Reject mixed-authority plans before issue.
    - [x] 4.4.b: Reject mixed route-context or mixed rail-domain plans before
      issue.
    - [x] 4.4.c: Reject overlapping destination spans before issue.
    - [x] 4.4.test: Add negative validation tests for mixed authority, mixed
      route context, and overlapping spans.
      - Central pre-issue validation now lives in
        `core/communicator/routing/types.cc::validate_read_plan(...)`.
  - [x] Milestone 4.5: Keep the `LocalRegion` interface extensible while the
    first cut implements CPU-only regions.
    - [x] 4.5.a: Preserve `dev_type` and `dev_id` in logical plan types.
    - [x] 4.5.b: Implement explicit first-cut rejection of non-CPU regions.
    - [x] 4.5.test: Add validation tests that CPU regions pass while GPU or
      mixed-region plans fail with pre-issue capability or validation errors.

Execution closure:

- The generic capability owned by `0115` is now landed end-to-end in core,
  communicator, transport, and shared-runtime code.
- Follow-on byte-artifact source-side no-pack export remains owned by
  `0087-01`; it consumes this generic capability rather than extending it.

- [x] Phase 5: RDMA Realization
  - [x] Milestone 5.1: Generalize the RDMA segment shape to support per-segment
    local `lkey`.
    - [x] 5.1.a: Introduce the per-segment local registration shape used by the
      transport boundary.
      - Landed early as a Phase 4 prerequisite by extending
        `transport::RdmaTransport::RdmaReadSeg` with additive `local_lkey`.
    - [x] 5.1.b: Keep request-level completion and profiling aggregation stable
      while segment-local `lkey` varies.
      - Existing request-level completion, progress, and profiling remain
        aggregated in `ReadRequest`; transport now consumes per-segment lkeys
        without changing request ownership.
    - [x] 5.1.test: Extend `core/communicator/engine/rdma_engine_test.cc` to
      cover at least two local registrations within one logical request.
  - [x] Milestone 5.2: Post vectored WR chains for one request across a local
    region set.
    - [x] 5.2.a: Build WR chains over resolved local region entries rather than
      one request-global MR.
      - `ReadPlanSlice` now carries `source_slice_offset`, prepared placement
        coverage is validated exactly per source slice, and
        `BuildPreparedPlanRdmaSegments(...)` can split one response segment
        into multiple RDMA WRs across prepared local regions.
    - [x] 5.2.b: Preserve multi-QP posting and completion ordering assumptions.
      - `rdma_engine_test.cc` now exercises a two-QP request and verifies the
        request's WRs are distributed across per-QP inflight queues without
        regressing request-level completion accounting.
    - [x] 5.2.c: Define failure behavior for partial post before request
      completion.
      - `RdmaTransport::read_multi(...)` now carries an explicit hard-failure
        rule for partial post: already-posted WRs remain inflight locally, the
        request becomes terminally failed, and no unposted WRs are counted.
    - [x] 5.2.test: Add RDMA engine regressions for multi-WR posting across
      multiple local regions, including partial-post failure handling.
  - [x] Milestone 5.3: Keep ACK, flow-credit, and staged-window release rules
    correct under the new request shape.
    - [x] 5.3.a: Ensure staged windows remain releasable when a prepared plan
      produces multiple segments or windows.
      - Added controller and engine regressions for multi-window staged
        release; `RDMA_READ_PLAN_DONE_EX` now has explicit coverage showing one
        acknowledged window releases only its staged segments while other
        windows remain inflight.
    - [x] 5.3.b: Keep flow-credit ledger behavior invariant under the new
      request shape.
      - `staging_flow_controller_test.cc` and `rdma_engine_test.cc` now assert
        staged credit returns exactly per released lease or window, preserving
        request-shape invariants under multi-window prepared plans.
    - [x] 5.3.test: Extend
      `core/communicator/engine/staging_flow_controller_test.cc` and
      `core/communicator/engine/rdma_engine_test.cc` for multi-window staged
      release and credit restoration.
  - [x] Milestone 5.4: Preserve request-local ACK identity keyed by
    `window_seq + segment_idx` rather than semantic slice identity.
    - [x] 5.4.a: Keep slice metadata available for placement and debugging only.
      - Added regressions where different staged windows intentionally reuse
        the same semantic slice-offset metadata while ACK release still follows
        request-local window identity only.
    - [x] 5.4.b: Ensure staged lease registry and ACK release remain keyed by
      request-local window and segment identity.
      - `StageLeaseKey` remains `{request_key, window_seq, segment_idx}` and
        ACK handlers are now covered by tests that release one window while an
        identically shaped sibling window remains staged.
    - [x] 5.4.test: Add regressions that reuse the same source slice identity
      across different windows while verifying the correct lease is released by
      `window_seq + segment_idx`.

- [x] Phase 6: Shared Runtime Integration
  - [x] Milestone 6.1: Add an internal
    `materialize_mapped_sources_into_target(...)` helper in
    `MaterializationFacade`.
    - [x] 6.1.a: Add one internal multi-source helper while preserving existing
      public entrypoints as wrappers.
      - Landed as private `MaterializationFacade::materialize_mapped_sources_into_target(...)`,
        which owns the shared `ByteRangeMappedSource -> sink -> pump_ranges(...)`
        execution for CPU/GPU target layouts over `std::vector<SeekableSource>`.
    - [x] 6.1.b: Route existing single-source paths through the helper where it
      is safe to do so.
      - `materialize_mapped_loader_into_target(...)` now acts as a single-source
        wrapper that opens one source and delegates execution to the shared
        helper.
    - [x] 6.1.test: Extend
      `core/store/runtime/ingestion/materialization_facade_test.cc` to cover
      composite execution over multiple sources into one target.
      - Added a private-helper regression for composite CPU sources into one
        target plus a wrapper regression that verifies the mapped-loader path
        now resolves through the shared helper.
  - [x] Milestone 6.2: Remove internal single-source restrictions that block
    composite execution.
    - [x] 6.2.a: Audit `mapping.num_sources == 1` restrictions and classify
      them into removable, wrapper-only, and still-semantic.
      - Removed the global single-source finalization guard in
        `derive_effective_source_maps(...)` so lowering can carry composite
        `ByteRangeMap` state forward.
      - Removed the early `mapping.num_sources == 1` rejection in
        `materialize_mapped_into_target(...)` so failures surface at actual
        source binding time instead of at facade entry.
      - Kept `materialize_mapped_loader_into_target(...)` and
        `ingest_mapped_loader_into_replica(...)` as explicit single-source
        wrappers because `IArtifactLoader` still resolves one source today.
      - Kept the scalar typed-local path restrictions that remain semantic for
        non-facade work items in this phase.
    - [x] 6.2.b: Remove only the restrictions that block internal composite
      execution for this design.
      - `materialize_mapped_into_target(...)` now reports a precise
        `FailedPrecondition` when execution maps require more sources than the
        current source resolver wires, instead of reporting a legacy
        single-source contract violation.
    - [x] 6.2.test: Add regression tests for multi-source `ByteRangeMap`
      lowering where the legacy public API shape remains unchanged.
      - Added a mapped-target regression that proves composite lowering now
        reaches the source-binding stage and fails with a source-resolution gap
        when only one concrete source is available.
      - Preserved wrapper-level regression coverage so the public mapped-loader
        API still rejects `mapping.num_sources != 1`.
  - [x] Milestone 6.3: Keep public materialization APIs unchanged.
    - [x] 6.3.a: Confirm no public request or daemon API shape changes are
      needed.
      - `MaterializationFacade` public method signatures remain unchanged.
      - The new composite mapped-source helper stays private-only; no request
        schema, daemon RPC, or external facade contract now accepts a
        multi-source payload directly.
    - [x] 6.3.b: Keep compatibility wrappers explicit rather than relying on
      accidental call-site behavior.
      - `materialize_mapped_loader_into_target(...)` stays as the public
        single-loader wrapper that performs loader-specific validation before
        delegating into the internal composite helper.
      - `ingest_mapped_loader_into_replica(...)` keeps the public
        logical/physical artifact-id split plus the single-loader contract
        instead of collapsing into the internal helper shape.
    - [x] 6.3.test: Keep existing public-facing facade tests green and add a
      compatibility regression if wrapper behavior changes.
      - Added regressions for `ingest_mapped_loader_into_replica(...)` that
        verify logical-vs-physical artifact semantics remain intact on the
        public path and that `mapping.num_sources != 1` is still rejected.

- [x] Phase 7: Validation and Dependent Consumers
  - [x] Milestone 7.1: Add unit tests for batched direct-write behavior,
    fallback, and failure semantics.
    - [x] 7.1.a: Collect the minimum direct-write regression matrix spanning
      source contract, pump batching, mux freeze, and facade integration.
      - `pump_direct_test` now covers the shared `DirectWriteOp` contract,
        default `readv_into_at(...)` loop behavior, batch splitting, staged
        fallback on pre-issue plan failure, and post-issue hard-fail surfacing.
      - `target_layout_host_sink_test` covers multi-window host direct-write
        grant planning across split target storages.
      - `byte_range_mapped_source_direct_test` covers grouped direct write,
        PAD zero-fill, scalar-child fallback, pre-validation, pre-issue
        vectored fallback, and hard vectored failure propagation.
      - `remote_key_source_routing_fallback_test` covers routed lowering,
        capability miss without RDMA, route freeze failure before issue, and
        logical `ReadPlan` lowering that falls back before issue.
      - `mux_seekable_source_test` covers branch freeze on direct write,
        capability miss, and explicit rejection of issue-time branch switching.
      - `materialization_facade_test` now covers public mapped-loader facade
        integration for batched direct writes and pre-issue vectored fallback.
      - `materialization_pump_options_test` covers runtime option wiring for
        batch bytes / op-count limits.
    - [x] 7.1.b: Require each lower-layer milestone to land its blocking unit
      tests before the next phase depends on it.
      - The current blocking matrix is green under Bazel for:
        `pump_direct_test`,
        `byte_range_mapped_source_direct_test`,
        `remote_key_source_routing_fallback_test`,
        `mux_seekable_source_test`,
        `target_layout_host_sink_test`,
        `materialization_pump_options_test`,
        and `materialization_facade_test`.
  - [x] Milestone 7.2: Add communicator tests for routed `read_plan(...)`,
    MTCP fallback, and RDMA multi-lkey posting.
    - [x] 7.2.a: Keep communicator correctness gates green for both RDMA and
      first-cut MTCP fallback.
      - `read_plan_validation_test` covers one-authority / one-route-context
        invariants, local-region shape validation, CPU-only first-cut region
        support, and exact source-slice coverage rules.
      - `routing_context_test` covers communicator-side adapter selection and
        route mismatch rejection before issue.
      - `adapter_test` now pins both first-cut capability misses:
        `read_plan(...)` on a non-RDMA communicator and routed MTCP
        `read_plan(...)` on an RDMA-enabled communicator both fail before
        issue with explicit `Unimplemented`.
      - `request_test` continues to cover read-plan request identity,
        segment-count ACK bookkeeping, and prepared-region lifetime retention.
    - [x] 7.2.b: Cover both direct and staged RDMA release behavior.
      - `rdma_engine_test` already covers prepared-plan request-key routing,
        per-segment local lkey preservation across multiple local regions,
        prepared-placement splitting, multi-QP vectored posting, partial post
        failure bookkeeping, and staged `RDMA_READ_PLAN_DONE_EX` release keyed
        by request-local `window_seq/segment_idx`.
      - `rdma_engine_test` now also covers the direct path explicitly:
        zero-copy `READ_PLAN_RESPONSE_EX` windows do not enqueue staged ACK
        state on the sink, and `RDMA_READ_PLAN_DONE_EX` for a zero-copy window
        is a no-op with respect to staged lease release.
  - [x] Milestone 7.3: Demonstrate that `0087-01` can consume this capability
    without defining a byte-artifact-private RDMA batch API.
    - [x] 7.3.a: Add a consumer-side contract checklist showing which generic
      primitives byte-artifact is allowed to call.
      - The byte-artifact consumer is now constrained to the generic
        `0115` seams:
        `ByteRangeMappedSource`,
        `DirectWriteOp`,
        `SeekableSource::readv_into_at(...)`,
        `RemoteKeySource`,
        routed communicator `read_plan(...)`,
        and the shared `MaterializationFacade` execution path above them.
      - No byte-artifact-private communicator or RDMA batch API was added in
        this cut.
    - [x] 7.3.b: If a targeted consumer test is added later, make it assert
      generic contract use rather than benchmark parity.
      - The current consumer evidence is integration-level:
        routed share-remote RDMA reruns on `20260420-155026`,
        `20260420-161513`, and `20260420-161949` all show cached-token reuse
        together with daemon logs
        `materialize_mode=single_source_composite batched_direct_write=true`.
      - Any future targeted consumer test must assert those generic contract
        surfaces and mode signals rather than TTFT parity alone.

# Tasks

- [x] Define and land the shared dataplane contract.
  - Add `DirectWriteOp`.
  - Add default `readv_into_at(...)`.
  - Keep `supports_direct_write_at()` as a coarse hint only.
  - Update direct-write tests to cover multiple destination windows.

- [x] Rework `pump_ranges(...)`.
  - Build bounded op batches instead of single-window calls.
  - Keep staged fallback before issue.
  - Add explicit pre-issue source-freeze hooks for fallback-bearing sources.
  - Surface explicit mode and fallback metrics.

- [x] Implement source batching.
  - `RemoteKeySource` batches by remote key and grant window.
  - `RemoteKeySource` lowers only pre-issue-frozen branches into `ReadPlan`.
  - `ByteRangeMappedSource` batches by underlying source and PAD policy.
  - Fallback-bearing mux paths return capability miss before issue when freeze
    cannot be completed.

- [x] Implement communicator `ReadPlan`.
  - Add routing types and engine entrypoint.
  - Add request-scoped `PreparedReadPlan` and destination region registration.
  - Keep one-plan-one-endpoint invariant.
  - Keep one-plan-one-authority / one-route-context invariant.
  - Reject overlapping destination spans before issue.
  - Keep first cut CPU-only while preserving extensible `LocalRegion` API.

- [x] Implement RDMA transport realization.
  - Support per-WR `lkey`.
  - Preserve request-level completion aggregation.
  - Preserve request-local stage lease and ACK identity safety.

- [x] Integrate with shared runtime.
  - Add internal multi-source helper.
  - Remove internal `num_sources == 1` blockers where safe.
  - Keep public API unchanged.

- [x] Validate consumer fit.
  - Do not add byte-artifact-only communicator APIs.

# Suggested Test Placement

- [x] `core/store/materialization/dataplane/runtime/tests/pump_direct_test.cc`
  owns batching, split-by-limit, pre-issue fallback, and post-issue hard-fail
  regressions for the shared dataplane direct-write path.
- [x] `core/store/materialization/dataplane/sources/tests/remote_key_source_routing_fallback_test.cc`
  owns routed lowering success, routing miss, MTCP capability miss, and
  source-freeze gating for `RemoteKeySource`.
- [x] `core/store/materialization/dataplane/sources/tests/mux_seekable_source_test.cc`
  owns mux-specific pre-issue freeze and rejection of issue-time branch
  switching.
- [x] `core/store/materialization/dataplane/sources/tests/byte_range_mapped_source_direct_test.cc`
  owns PAD zero-fill, grouped delegation, pre-validation, pre-issue vectored
  fallback, and post-issue hard-failure regressions for composite mapped
  direct-write sources.
- [x] `core/store/runtime/ingestion/materialization_pump_options_test.cc`
  owns typed runtime option wiring for direct-write batch byte and op-count
  limits.
- [x] `core/communicator/routing/routing_context_test.cc` plus
  `core/communicator/routing/read_plan_validation_test.cc` own route-context,
  authority, exact source-slice coverage, and CPU-only local-region
  validation.
- [x] `core/communicator/transport/request_test.cc` owns request-scoped
  progress, completion, and prepared-request accounting regressions.
- [x] `core/communicator/engine/rdma_engine_test.cc` owns additive wire-path,
  multi-lkey RDMA posting, staged ACK release, and mixed multi-window RDMA
  regressions.
- [x] `core/communicator/engine/staging_flow_controller_test.cc` owns
  request-local lease identity and staged-credit restoration invariants.
- [x] `core/store/runtime/ingestion/materialization_facade_test.cc` owns
  internal multi-source helper coverage and wrapper compatibility regressions.

# Test / Rollout / Backout

## Acceptance checks

- [x] Shared dataplane can batch direct-write work without controller-local
  loops.
- [x] `RemoteKeySource` can issue one routed plan for multiple direct-write
  ops after pre-issue source freeze.
- [x] communicator can execute one `ReadPlan` across multiple source slices and
  local regions within one authority / route context.
- [x] RDMA can post many WRs with per-segment local registrations.
- [x] The first cut accepts CPU `local_regions` only, while the interface
  remains extensible to GPU or mixed regions later.
- [x] Pre-issue fallback works and post-issue in-place fallback never happens.
- [x] Request-local ACK and staged lease release remain keyed by
  `window_seq + segment_idx`.
- [x] MTCP keeps existing staged behavior when the plan fast path is
  unavailable by returning a pre-issue capability miss in the first cut.

## Test plan

- [x] Dataplane direct-write tests:
  - batched multi-window success
  - default `readv_into_at(...)` zero-op no-op
  - default `readv_into_at(...)` first-error propagation
  - split by op-count limit
  - split by byte-limit
  - pre-issue source freeze for fallback-bearing sources
  - pre-issue fallback
  - post-issue failure surfaces hard error
- [x] `RemoteKeySource` tests:
  - routed plan success
  - routing failure fallback before issue
  - fallback-bearing source capability miss before issue
  - MTCP `Unimplemented` fallback
  - multi-key lowering into one logical `ReadPlan`
- [x] communicator tests:
  - `read_plan(...)` routing and connection selection
  - one-authority / one-route-context validation
  - CPU-only `LocalRegion` validation
  - overlapping destination rejection
  - RDMA response segmentation and ACK release
  - per-segment `lkey` posting
  - partial WR-post failure handling
- [x] shared runtime tests:
  - internal multi-source helper
  - exact fallback coverage with `ByteRangeMap`
  - removal of internal `num_sources == 1` blockers where intended

## Per-milestone test gates

- [x] Phase 2 may not complete until the shared dataplane direct-write tests
  cover default batching semantics and split-by-limit behavior.
- [x] Phase 3 may not complete until `RemoteKeySource`, mux freeze, and
  `ByteRangeMappedSource` fallback tests all distinguish pre-issue capability
  miss from post-issue hard failure.
- [x] Phase 4 may not complete until `ReadPlan` validation rejects mixed
  authority, mixed route context, non-CPU first-cut regions, and overlapping
  destination spans.
- [x] Phase 5 may not complete until RDMA tests cover multi-lkey posting and
  staged ACK release keyed by `window_seq + segment_idx`.
- [x] Phase 6 may not complete until materialization facade tests cover
  multi-source internal execution without public API changes.

## Rollout

- Land the contract and fallback rules before consumer integration.
- Keep plan fast path behind typed config until correctness and metrics are in
  place.
- Enable RDMA first.
- Keep MTCP on the current staged path until explicit evidence justifies a
  native plan implementation under the same contract.

## Backout

- Disable `read_plan(...)` use and fall back to existing single-read or staged
  paths.
- Keep `readv_into_at(...)` defaulting to per-op loops so the contract can
  remain while the optimization is turned off.
- Do not back out by restoring byte-artifact-specific copy loops as the owner of
  the optimization.

# Risks & Tracking

- [ ] Request-scoped destination registration may become a measurable setup
  cost.
- [ ] RDMA wire and ACK changes may regress staged-credit correctness if not
  covered by dedicated tests.
- [ ] Capability-hint misuse could accidentally cache route-dependent fast-path
  eligibility too early; review pre-issue freeze semantics carefully.
- [ ] Multi-source helper changes may accidentally widen public API or collapse
  `ByteRangeMap` fallback semantics if ownership boundaries blur.
  reject those changes and route them back through `0115`.
