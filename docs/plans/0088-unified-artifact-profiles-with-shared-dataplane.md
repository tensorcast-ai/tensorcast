---
slug: unified-artifact-profiles-with-shared-dataplane
title: Plan - Unified Artifact Profiles with Shared Dataplane
status: in_progress
areas: ["core", "daemon", "sdk", "global_store", "proto", "docs"]
created: 2026-03-08
last_updated: 2026-03-08
related_code:
  - docs/designs/0088-unified-artifact-profiles-with-shared-dataplane.md
  - core/store/materialization/contracts/loading_spec.h
  - core/store/runtime/ingestion/materialization_facade.h
  - core/store/runtime/ingestion/materialization_facade.cc
  - daemon/service/controllers/byte_artifact_controller.cc
  - daemon/service/byte_artifact_body_store.cc
  - daemon/service/byte_artifact_route_resolver.cc
  - daemon/service/payload_transport_broker.cc
  - daemon/service/controllers/external_target_access_service.h
  - proto/tensorcast/daemon/v2/store_daemon.proto
links:
  design: ../designs/0088-unified-artifact-profiles-with-shared-dataplane.md
---

# Objective

Hard-cut TensorCast to a profile-aware artifact architecture in which:

- `byte_artifact` remains an artifact profile,
- authority and routing are profile-specific,
- dataplane execution is shared and remains core-owned,
- no byte-artifact-private copy substrate remains in the target state.

# Latest Status

Updated 2026-03-08:

- Completed in this change:
  - introduced `ArtifactLoweringPlan` / `ArtifactLoweringResult` as the formal internal profile->core lowering IR
  - added a core-owned `execute_artifact_lowering_plan(...)` entrypoint in
    `MaterializationFacade` / `IngestionRuntime` / `StoreEngine`
  - added a core-owned `materialize_mapped_loader_into_target(...)` execution path so resolved loaders can enter the
    existing shared sink/pump dataplane without going through disk/GS source resolution
  - implemented `StoreEngine::ingest_from_buffer_internal(...)` so in-memory buffers can enter the core runtime
  - added a core-owned mapped-loader -> replica execution path so non-disk/non-GS sources can lower into core-backed
    replica targets with shared ingestion events
  - added shared synthetic flat-payload helpers in `loading_spec.h` for canonical payload index bytes and identity
    byte-range maps
  - added payload-ref loader/source adapters in `PayloadTransportBroker` so capability-backed payloads can lower to
    standard seekable-loader contracts
  - promoted `payload_ref` issuance and chunk reads to operate on core-backed body handles instead of raw payload
    strings on the steady-state path
  - rewired `BatchGetIntoRegion` to materialize byte-artifact payloads through the shared core dataplane instead of
    daemon-owned `write_item` copy loops
  - rewired `BatchPutIfAbsentFromRegion` and `HomeBatchPutIfAbsent` to lower their sources into shared executor plans
    and commit core-backed body handles
  - removed direct `cudaMemcpyHostToDevice` / `cudaMemcpyDeviceToHost` execution from
    `byte_artifact_region_layout.cc` and the byte-artifact controller path
  - centralized front-door route acquisition / refresh / redirect cache mutation into `ByteArtifactRouteResolver`;
    controller code no longer mutates `shard_routes` or `owned_shard_leases` directly
  - promoted `ArtifactProfileRegistry` into a real runtime factory internally by introducing
    `ArtifactProfileRuntime` and a concrete byte-artifact runtime implementation
  - removed the compatibility-style static profile-behavior wrappers from `ArtifactProfileRegistry`; byte-artifact
    callers now use explicit profile runtime dispatch
  - replaced raw byte-payload storage in `ByteArtifactBodyStore` / `ByteArtifactRuntimeState` with explicit
    core-backed `BodyHandle` storage and validation
  - bound body-handle cleanup on expiry / conflict / join to core replica retire semantics instead of leaving staged
    replicas unmanaged
- Still in progress:
  - targeted Bazel / Python validation for the new lowering path has not been completed in this update
  - direct-write / pinned-staging / shared-memory optimization coverage for byte-artifact put flows still needs
    explicit verification
  - controller ownership can still be simplified further now that lowering and execution are separated

# Current State & Grounding

Current baseline:

- semantic unification is partially in place:
  - `ArtifactSelection` is the shared selector,
  - plan and node-agent use typed artifact actions,
  - `byte_artifact` is defined as a profile in `0087`.
- authority split is partially in place:
  - Global Store owns shard-home leases only,
  - daemon controllers own home-batch orchestration.
- dataplane unification is materially in place for byte-artifact get/put:
  - `MaterializationFacade::materialize_into_target(...)` already provides the shared source/map/sink execution path,
  - `execute_artifact_lowering_plan(...)` now provides the shared core-owned lowering execution entrypoint for both
    into-target and replica-target flows,
  - `InlineBufferLoader`, `CpuMemorySource`, `GpuMemorySource`, `TargetLayoutGpuSink`, and `pump_ranges(...)` already
    exist,
  - byte-artifact get/put paths now lower through shared loaders/maps/targets rather than controller-owned copy loops.

Grounding references:

- shared dataplane substrate:
  - `core/store/materialization/contracts/loading_spec.h`
  - `core/store/materialization/dataplane/contracts/inline_buffer_loader.h`
  - `core/store/materialization/dataplane/sources/memory_source.h`
  - `core/store/materialization/dataplane/sinks/target_layout_gpu_sink.h`
  - `core/store/materialization/dataplane/runtime/pump.h`
  - `core/store/runtime/ingestion/materialization_facade.cc`
- current byte-artifact divergence:
  - `daemon/service/controllers/byte_artifact_controller.cc`
  - `daemon/service/byte_artifact_body_store.cc`
  - `daemon/service/byte_artifact_route_resolver.cc`
  - `daemon/service/byte_artifact_region_layout.cc`
  - `daemon/service/payload_transport_broker.cc`

Hard-cut constraints:

- no compatibility layer,
- no dual path retained for old behavior,
- no “temporary forever” byte-artifact-private dataplane,
- no daemon/service-owned second execution runtime parallel to `core/store`.

Concrete current-state observations that drive this plan:

- `daemon/service/controllers/byte_artifact_controller.cc` now constructs lowering plans and dispatches them into the
  core executor for both byte-artifact get and put flows.
- `daemon/service/byte_artifact_region_layout.cc` now stops at placement/source adaptation; steady-state copies happen
  in core/store.
- `core/store/runtime/ingestion/materialization_facade.cc` already has the exact execution skeleton we want to reuse:
  source init, byte-range planning, target sink setup, pinned staging, `pump_ranges(...)`, and target close.
- `core/store/store_engine.cc` no longer blocks in-memory source ingestion; byte-artifact bodies can enter core-backed
  replica lifecycles directly.
- `daemon/service/byte_artifact_route_resolver.cc` owns route acquisition / refresh / redirect cache mutation, while
  controller code focuses on validation, lowering, and dispatch.

# Phases & Milestones

- [ ] Phase 1: Freeze the target layering and prohibit further dataplane drift
  - [ ] Milestone 1.1: declare `byte_artifact` authority/routing as the only allowed specialization boundary.
  - [ ] Milestone 1.2: forbid any new byte-artifact-specific copy loop or memory-movement primitive outside the shared
    dataplane.
  - [x] Milestone 1.3: define explicit internal lowering contracts from profile runtime to shared dataplane inputs.
  - [x] Milestone 1.4: declare the shared executor as core-owned (`MaterializationFacade` / `StoreEngine`) and forbid
    service-layer execution drift.
  - [ ] Milestone 1.5: document current code paths that are temporary exceptions and must be deleted by the end of this
    plan.

- [ ] Phase 2: Make profile runtime first-class
  - [x] Milestone 2.1: promote `ArtifactProfileRegistry` from validator helper to a profile-runtime factory/registry,
    not a central switch controller.
  - [x] Milestone 2.2: introduce explicit `ArtifactProfileRuntime` ownership interfaces for authority resolution and
    lowering.
  - [x] Milestone 2.3: centralize byte-artifact routing and redirect logic into a single routing service.
  - [x] Milestone 2.4: remove controller ownership of route cache mutation logic.
  - [x] Milestone 2.5: define `ArtifactLoweringPlan` as the canonical profile->core lowering IR.

- [ ] Phase 3: Complete shared in-memory source support in core runtime
  - [x] Milestone 3.1: complete the memory-backed ingest path in core runtime so in-memory artifact bodies can enter
    the shared materialization substrate directly.
  - [x] Milestone 3.2: define standard source adapters for inline, CPU memory, GPU memory, and capability-backed
    payload sources.
  - [ ] Milestone 3.3: verify CPU memfd, CUDA IPC, direct-write, and pinned-buffer execution remain shared code paths.
  - [ ] Milestone 3.4: add or finalize the standard source contract for payload-capability-backed reads.
  - [x] Milestone 3.5: add one shared builder for byte-artifact synthetic canonical payload metadata used by profile
    lowering and core execution.

- [ ] Phase 4: Replace byte-artifact body bytes with body handles
  - [x] Milestone 4.1: refactor byte-artifact metadata state to store invariant/TTL/epoch metadata plus body handles,
    not raw payloads as the primary contract.
  - [x] Milestone 4.2: make payload references resolve to standard source adapters instead of bespoke copy loops.
  - [ ] Milestone 4.3: establish lifecycle pruning and capacity controls on metadata/body handles without reintroducing
    a dedicated dataplane.
  - [x] Milestone 4.4: ensure body handles can point to shared-memory or replica-backed storage without introducing a
    profile-private export mechanism.
  - [x] Milestone 4.5: bind body handles to the existing core lease / publish / retire / reconcile lifecycle instead of
    inventing a byte-artifact-private lifecycle.

- [ ] Phase 5: Lower byte-artifact get/put into the shared executor
  - [x] Milestone 5.1: refactor `BatchGetIntoRegion` to lower into shared source/map/target execution.
  - [x] Milestone 5.2: refactor `BatchPutIfAbsentFromRegion` and `HomeBatchPutIfAbsent` to lower into shared source
    and target execution.
  - [x] Milestone 5.3: remove byte-artifact-private direct `cudaMemcpy` target write/read paths from steady-state
    execution.
  - [ ] Milestone 5.4: ensure the byte-artifact path can exploit the same direct-write, pinned staging, and shared
    memory optimizations as other artifact flows.

- [ ] Phase 6: Simplify daemon ownership and API surfaces
  - [ ] Milestone 6.1: reduce `ByteArtifactController` to request validation, batching, authority dispatch, and result
    aggregation.
  - [ ] Milestone 6.2: ensure worker directory caching and payload transport stay shared daemon infrastructure.
  - [ ] Milestone 6.3: align Python, node-agent, and engine-adapter surfaces with the finalized profile/runtime split.
  - [ ] Milestone 6.4: remove dead state holders and duplicate caches once ownership is centralized.
  - [ ] Milestone 6.5: route byte-artifact completion, publish-context, and observability through the same shared
    operation semantics used by the core ingestion/materialization runtime.

- [ ] Phase 7: Hard-cut cleanup
  - [ ] Milestone 7.1: delete obsolete byte-artifact-private copy helpers and duplicate route-policy logic.
  - [ ] Milestone 7.2: update design/docs so the new shared-dataplane architecture is the canonical target state.
  - [ ] Milestone 7.3: lock acceptance checks and prevent regressions through targeted tests and repository searches.
  - [ ] Milestone 7.4: prove that a future profile can be added through profile runtime + lowering only, without
    requiring a new steady-state copy engine.

# Tasks

- Core runtime
  - finish memory-backed ingestion support in `StoreEngine` / `MaterializationFacade`
  - expose standard source adapters for profile runtimes
  - define the canonical core-owned execution entrypoint that consumes `ArtifactLoweringPlan`
  - add one shared builder for byte-artifact synthetic canonical payload metadata
  - keep `IntoTargetLayout` and shared sink/pump path as the only execution substrate
  - target files:
    - `core/store/store_engine.h`
    - `core/store/store_engine.cc`
    - `core/store/runtime/ingestion/materialization_facade.h`
    - `core/store/runtime/ingestion/materialization_facade.cc`
    - `core/store/materialization/contracts/loading_spec.h`
    - `core/store/materialization/dataplane/contracts/inline_buffer_loader.h`
    - `core/store/materialization/dataplane/sources/memory_source.h`

- Daemon profile runtime
  - define explicit profile runtime interfaces
  - keep `ArtifactProfileRegistry` as lookup/factory only
  - move shard-home acquisition/refresh/redirect policy out of controllers
  - refactor byte-artifact metadata to body-handle-based storage
  - ensure profile runtime stops at `ArtifactLoweringPlan` and does not execute copy logic
  - target files:
    - `daemon/service/artifact_profile_registry.h`
    - `daemon/service/artifact_profile_registry.cc`
    - `daemon/service/byte_artifact_authority_service.h`
    - `daemon/service/byte_artifact_authority_service.cc`
    - `daemon/service/byte_artifact_route_resolver.h`
    - `daemon/service/byte_artifact_route_resolver.cc`
    - `daemon/service/byte_artifact_body_store.h`
    - `daemon/service/byte_artifact_body_store.cc`
    - `daemon/service/byte_artifact_runtime_state.h`

- Byte-artifact dataplane lowering
  - add lowering from sealed byte bodies and payload capabilities to shared source objects
  - lower byte-artifact operations into `ArtifactLoweringPlan` rather than direct executor argument assembly
  - add lowering from byte-artifact placement descriptors to `IntoTargetLayout`
  - route get/put through shared executor rather than bespoke host-buffer loops
  - route completion and publish semantics through shared ingestion/materialization observability
  - target files:
    - `daemon/service/controllers/byte_artifact_controller.cc`
    - `daemon/service/byte_artifact_region_layout.h`
    - `daemon/service/byte_artifact_region_layout.cc`
    - `daemon/service/payload_transport_broker.h`
    - `daemon/service/payload_transport_broker.cc`
    - `daemon/service/controllers/external_target_access_service.h`
    - `daemon/service/controllers/external_target_access_service.cc`

- SDK / NodeAgent / Engine Adapter
  - keep `Artifact`, `ArtifactSelection`, and typed artifact actions stable
  - move any profile-specific helper placement so artifact semantics stay canonical and engine adapter remains an
    integration boundary

- Docs
  - keep `0087` as prior consolidation/current-state context
  - make `0088` the target architecture for deeper shared-dataplane unification

# Acceptance Checks

- repository search finds no new byte-artifact-specific copy engine outside the shared dataplane modules
- shared execution remains owned by `core/store`; daemon/service introduces no second steady-state copy runtime
- byte-artifact dataplane execution uses shared source/map/sink/pump substrate rather than controller-owned
  `cudaMemcpy` loops
- `ArtifactProfileRegistry` acts as profile lookup/factory and profile runtimes own specialization; no giant central
  switch controller owns cross-profile execution
- route acquisition and redirect policy are centralized in one routing service
- byte-artifact lowering produces one canonical lowering shape and one shared synthetic payload-metadata builder
- payload capabilities lower to standard source abstractions
- CPU shared memory, GPU shared memory, CUDA IPC, direct-write, pinned staging, and P2P execution remain shared code
  paths for byte-artifact flows
- body handles bind to the existing core lease / publish / retire / reconcile lifecycle
- lowered byte-artifact execution emits shared operation / completion / publish-context semantics
- no compatibility shims, no dual stack, and no retained obsolete byte-artifact-private dataplane

Repository searches expected to converge to target shape:

- `rg -n "InlineBufferSource loading not yet implemented" core/store`
  - expected result: no hits
- `rg -n "cudaMemcpyHostToDevice|cudaMemcpyDeviceToHost" daemon/service/byte_artifact* daemon/service/controllers/byte_artifact_controller.cc`
  - expected result: no steady-state dataplane hits
- `rg -n "FetchPayloadRefChunk.*payload.append|payload.append\\(" daemon/service/controllers/byte_artifact_controller.cc`
  - expected result: no bespoke remote payload reassembly loop
- `rg -n "owned_shard_leases|shard_routes" daemon/service/controllers/byte_artifact_controller.cc`
  - expected result: controller no longer mutates route ownership state directly
- `rg -n "pump_ranges|TargetLayoutGpuSink|ByteRangeCompiler" daemon/service`
  - expected result: no steady-state execution logic outside bridging/validation helpers

# Test / Rollout / Backout

Tests:

- core and daemon unit tests around:
  - shared source adapters
  - target sink and pump execution
  - byte-artifact get/put lowering
  - shard-home redirect and fencing
  - CPU memfd / CUDA IPC / direct-write coverage
  - shared publish-context / completion event semantics for lowered byte-artifact flows
- Python tests around:
  - byte-artifact identity and selection helpers
  - plan/node-agent artifact actions
  - binding and publish flows that must continue to consume the unified artifact semantics
  - tensor-dict and byte-artifact flows sharing the same selection and lowering invariants

Required command set:

```bash
source .venv/bin/activate
pytest tests/python/test_byte_artifact_identity.py \
  tests/python/test_selection_identity_vectors.py \
  tests/python/test_kvcache_adapter.py \
  tests/python/node_agent/test_plan_execution.py \
  tests/python/api/test_plan_spec.py
```

```bash
bazel test //daemon:grpc_service_impl_batch_runtime_test \
  --test_env=TENSORCAST_CUDA_BACKEND=fake \
  --test_output=errors

bazel test //daemon:grpc_service_impl_batch_redirect_e2e_test \
  --test_env=TENSORCAST_CUDA_BACKEND=fake \
  --test_output=errors
```

```bash
bazel test //daemon:grpc_service_impl_cpu_memfd_e2e_test \
  --test_env=TENSORCAST_CUDA_BACKEND=fake \
  --test_output=errors
```

Rollout:

- hard cut only
- land the shared-dataplane lowering path and delete obsolete private paths in the same execution line
- keep the target architecture coherent at every phase; do not preserve old behavior with hidden fallbacks
- if a phase cannot remove the old dataplane path yet, the phase is not complete

Backout:

- backout means reverting the refactor branch, not preserving compatibility inside the target architecture
- do not add runtime flags, dual wiring, or permanent legacy adapters as a “safe” fallback

# Risks & Tracking

- [ ] Risk: authority/routing refactor lands before shared lowerings are ready and temporarily increases duplication.
- [ ] Risk: body-handle refactor introduces lifecycle leaks without explicit prune/capacity enforcement.
- [ ] Risk: partial refactors accidentally preserve controller-owned copy loops and create a hidden second dataplane.
- [ ] Risk: shared runtime work stalls on unfinished in-memory ingestion support and leaves byte-artifact flows stranded.
- [ ] Risk: `ArtifactProfileRegistry` becomes a giant switch-based controller instead of a runtime factory.
- [ ] Risk: synthetic canonical payload metadata drifts between daemon/profile code and core runtime.
