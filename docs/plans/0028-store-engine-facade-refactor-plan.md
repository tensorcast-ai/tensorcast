---
slug: 0028-store-engine-facade-refactor-plan
title: StoreEngine Facade Refactor Execution Plan
areas: ["core"]
related_code:
  - core/store/store_engine.h
  - core/store/store_engine.cc
  - core/store/components/**
  - core/store/materialization/**
  - tensorcast/global_store/**
links:
  design: ../designs/0028-store-engine-facade-refactor.md
---

# Objective

Translate design 0028 into a series of bounded execution plans. The first three plans focus on C++ restructuring (component catalog, ingestion pipeline, telemetry/registration), while the final plan covers Python and protobuf surfaces when StoreEngine APIs change.

# Current State & Grounding

- `core/store/store_engine.cc` centralizes lifecycle, ingestion, registration, and telemetry in a single 1.8K+ LOC class, making testing and ownership difficult.
- `core/store/components/registration/artifact_registration_manager.cc` and the associated glue in `store_engine.cc` tie worker identity and Global Store interactions directly to the engine.
- Disk vs. P2P ingestion logic diverges between `StoreEngine::ingest_from_disk_internal`, `StoreEngine::ingest_from_p2p_internal`, and helpers in `core/store/materialization/planning/chunk_aware_strategy.cc`.
- Documentation (`core/store/README.md`, `docs/architecture/p2p-transfer-strategies.md`, `docs/plans/0027-materialization-unification.md`) references legacy helpers and does not describe the future pipeline-based flow.

# Phases & Milestones

## Plan A – Component Catalog & Replica Service (C++)

- [x] Milestone A1: Add `core/store/components/runtime/component_catalog.{h,cc}` that initializes DeviceManager, ReplicaRegistry, MetricsCollector, CommunicationManager, and GlobalStoreClient with existing invariants.
- [x] Milestone A2: Introduce `core/store/components/runtime/replica_service.{h,cc}` encapsulating replica lookups, eviction retries, resident queries, and UMA exports; expose a new Bazel target.
- [x] Milestone A3: Update `StoreEngine` construction to rely on ComponentCatalog/ReplicaService, delete redundant members, and refresh `core/store/BUILD`. Ensure existing `//core/store:store_engine*_test` suites pass.

### Scope & Dependencies

- Extract all constructor/destructor wiring from `StoreEngine` into a reusable `ComponentCatalog`, including fake CUDA toggles and pinned buffer pool setup currently in `store_engine.cc`.
- Encapsulate replica lifecycle helpers and UMA export code into `ReplicaService`, ensuring it can be injected wherever `ReplicaRegistry` was previously accessed directly.
- Touch points: `core/store/store_engine.{h,cc}`, `core/store/components/{device_manager,replica_registry,metrics_collector,communication_manager,global_store_client}.cc`, `core/store/BUILD`.

**Status:** Catalog + ReplicaService are in-tree and `StoreEngine` now owns only those collaborators. Targeted service tests (`runtime_services_test`) cover catalog startup and inline replica flows; structured load/unload event hooks landed with the TelemetryService work in Plan C.

### Execution Steps

1. **Catalog skeleton:** Create `component_catalog.h` with typed getters (`ReplicaRegistry& replica_registry()`, etc.) and lifecycle hooks (`Start()`, `Shutdown()`). Port option validation (chunk alignment, pinned buffer multiples) from `StoreEngine::Init`.
2. **Pinned buffer ownership:** Move the pinned buffer pool creation/destruction into the catalog so services share a single pool instance; ensure CUDA vs. fake-CUDA flags propagate via catalog options.
3. **Replica service implementation:** Encapsulate helper methods (`get_or_create_replica`, `wait_replica_ready`, `get_resident_devices`, eviction retries) in `replica_service.cc`, emitting callbacks for load/unload events.
4. **StoreEngine constructor rewrite:** Replace direct member ownership with injected catalog/service pointers. Update shutdown ordering to delegate to the catalog.
5. **BUILD/targets:** Add `component_catalog_lib` and `replica_service_lib` targets, wire them into `store_engine_lib`, and expose friend targets required by tests.

All five steps landed in CL XXX; remaining TODOs for Plan A are (a) dedicated unit tests for the new services and (b) wiring structured load/unload events once TelemetryService is implemented.

### Testing & Acceptance

- Extend `core/store/components/component_catalog_test.cc` (or add a new suite) to cover initialization order, fake CUDA toggles, and pinned pool reuse.
- Update `//core/store:store_engine_test` to construct `StoreEngine` via the catalog, ensuring regression coverage for resident device queries and eviction retries.
- Acceptance: no remaining direct uses of `DeviceManager`/`ReplicaRegistry` from `StoreEngine` except through the new services; all catalog services start/stop cleanly on test shutdown.

### Rollback Notes

- Retain the previous `StoreEngine` constructor path in git history; if issues arise, revert the catalog/service commits together and re-run `bazel test //core/store:store_engine_test`.

## Plan B – Ingestion Pipeline & Materialization Coordinator (C++)

- [x] Milestone B1: Create `core/store/materialization/runtime/pipeline/*` with SourceAdapter, Metadata, Allocation, Verification, and Handle stages that subsume the internal disk/P2P helpers.
- [x] Milestone B2: Build `MaterializationCoordinator` that wraps `materialization::control::MaterializationService` using the pipeline/replica services, wiring it into `StoreEngine::materialize_replica`.
- [x] Milestone B3: Port ingestion/materialization tests to pipeline-level suites (`core/store/materialization/runtime/pipeline/tests/*`) and remove `core/store/store_engine_p2p_*` tests.

### Scope & Dependencies

- Replace `StoreEngine::ingest_from_disk_internal` and `..._p2p_internal` with a staged pipeline under `core/store/materialization/runtime/pipeline/`. **DONE** – SourceAdapter, Metadata, Allocation, Verification, and Handle stages now share a common `IngestionContext` and own all disk/P2P flows.
- Introduce `IngestionContext` objects passed between stages plus event hooks for telemetry and Global Store publishing. **Context done** – structured event hooks will be added alongside TelemetryService in Plan C.
- Refactor `materialization/control` to use a dedicated `MaterializationCoordinator` that prebuilds `MaterializationDeps`. **DONE** – the coordinator owns the pipeline and replica/global-store dependencies, and `StoreEngine::materialize_replica` delegates to it.
- Touch points: `core/store/store_engine.{h,cc}`, `core/store/materialization/planning/*.cc`, `core/store/materialization/dataplane/**/*`, `core/store/materialization/control/*`, Bazel targets/tests.

### Execution Steps

1. **Pipeline scaffolding:** Define `IngestionContext` with source info, planning output, replica handles, and telemetry timers. Implement SourceAdapter stage (disk + P2P) to normalize requests and fetch canonical indices.
2. **Metadata + allocation stages:** Move view planning (`chunk_aware_strategy`) and replica allocation/eviction loops into respective stage files. Ensure allocation stage records pinned buffer wait metrics via `ReplicaService`.
3. **Verification + handle stages:** Reuse `verification_utils` for hashing/safetensors updates and create a Handle stage that performs CUDA IPC export, constructs `ReplicaHandle`, and fires success/failure events.
4. **Materialization coordinator:** Introduce `MaterializationCoordinator` that owns the pipeline, replica service, and Global Store publisher dependencies. Update `StoreEngine::materialize_replica` to call into the coordinator.
5. **Delete legacy helpers/tests:** Remove the old internal ingestion methods and the P2P-specific tests once new pipeline tests cover equivalent scenarios.

**Outstanding for Plan B:** all telemetry/global-store hooks were delivered in Plan C; pipeline + coordinator milestones remain green under `//core/store:store_engine_test`.

### Testing & Acceptance

- Add suites under `core/store/materialization/runtime/pipeline/tests/` covering disk happy path, P2P retry with eviction, verification failure, and safetensors backfill. (P2P happy path + verification failure exist today; disk/safetensors coverage to be layered in once TelemetryService hooks land.)
- Update `materialization_service_test` to inject fake pipeline/replica services to validate AUTO mode.
- Acceptance: all ingestion entry points go through the pipeline; `materialize_replica` no longer rebuilds lambdas per call; deleted tests replaced by pipeline coverage.

### Rollback Notes

- Keep the legacy ingestion helpers behind a git tag until pipeline stabilization; if regressions happen, revert pipeline commits and re-enable the old helpers/tests in the same change.

## Plan C – Telemetry, Registration, and Documentation (C++)

- [x] Milestone C1: Extract `GlobalStorePublisher` and `RegistrationFacade` services, remove the loader registry adapter, and hook them to lifecycle events.
- [x] Milestone C2: Implement `TelemetryService` consuming ReplicaService/IngestionPipeline events and exposing snapshot queries (`get_resident_devices`, memory gauges) via the facade.
- [x] Milestone C3: Update `core/store/README.md`, `docs/architecture/p2p-transfer-strategies.md`, and `docs/plans/0027-materialization-unification.md` to describe the new services and event-driven ingestion flow.

### Scope & Dependencies

- Move Global Store interactions and registration wiring out of `StoreEngine` into dedicated runtime services, fed by ComponentCatalog identity data.
- Ensure telemetry queries consume consistent snapshots from DeviceManager/ReplicaService without direct StoreEngine coupling.
- Update documentation to describe the facade/services rather than legacy helpers.
- Touch points: `core/store/components/registration/*`, `core/store/components/global_store_client.*`, `core/store/components/metrics_collector.*`, `core/store/materialization/control/materialization_service.cc`, docs listed above.

### Execution Steps

1. **GlobalStorePublisher:** Introduce `core/store/components/runtime/global_store_publisher.{h,cc}` responsible for replica register/unregister, variant updates, and key mapping CRUD. Subscribe to ingestion/registration events.
2. **RegistrationFacade rename:** Move `artifact_registration_manager.{h,cc}` into `registration_facade.{h,cc}`, wiring worker identity and Global Store callbacks via ComponentCatalog.
3. **TelemetryService:** Create `telemetry_service.{h,cc}` to aggregate replica/device stats, memory usage, and chunk states. Provide read-only APIs consumed by `StoreEngine` facade methods.
4. **Remove loader registry adapter:** Delete `materialization/control/loader_registry_adapter.{h,cc}` and adjust orchestrator/materialization service to request loaders directly through pipeline/dataplane registries.
5. **Documentation updates:** Refresh module READMEs and architecture docs with diagrams describing the new runtime services, ensuring doc sync compliance.

### Testing & Acceptance

- Extend registration tests to cover the new facade (begin/commit/abort/keep_alive flows) via ComponentCatalog-provided worker identity.
- Add telemetry unit tests verifying snapshot consistency and event-driven metric updates.
- Acceptance: no mention of the loader registry adapter remains; `StoreEngine` telemetry/registration methods delegate entirely to the new services; doc diffs accompany the code change.
- Verification: `bazel test //core/store:store_engine_test --define use_fake_cuda=true --test_output=errors`.

### Rollback Notes

- If telemetry/registration regressions appear, revert the new services alongside doc updates, restoring the previous `artifact_registration_manager` path and loader registry adapter.

### Rollback Notes

- If Python/SDK regressions appear, revert the sync commits and re-run `bash tools/build_proto_python.sh` to regenerate stubs for the previous proto version before republishing wheels.

# Tasks

- **Component catalog bring-up (DONE):** Catalog + ReplicaService landed along with targeted service tests (`core/store/components/runtime/runtime_services_test.cc`) that exercise startup/shutdown ordering and inline-replica flows. Telemetry hooks will arrive with Plan C's service extraction.
- **Pipeline + coordinator integration (DONE):** Disk and P2P ingestion routes now flow through the staged runtime pipeline and `MaterializationCoordinator`, with regression coverage in `core/store/materialization/runtime/pipeline/tests/*` replacing the old StoreEngine P2P suites. Verified via `bazel build //core/store/materialization/control:all --define use_fake_cuda=true` and `bazel build //core/store:store_engine --define use_fake_cuda=true`.
- **Telemetry & registration services:** Rewire registration/Global Store calls to dedicated services, emit structured events for success/failure paths, and delete legacy adapters.
- **Python parity & proto updates:** After C++ facade changes land, sync generated protobufs, update Python wrappers, and ensure cross-language API compatibility.

# Test / Rollout / Backout

- Run targeted Bazel suites per plan: `bazel test //core/store:store_engine_test`, `//core/store/materialization/runtime/pipeline:all_tests`, and registration/materialization suites to cover new services.
- For Plan D, execute `uv run pytest tests/python/...` plus any downstream integration suites before release; regenerate protobuf bindings via `bash tools/build_proto_python.sh`.
- Rollout strategy: land Plan A→C sequentially with feature branches gated by CI; only begin Plan D once the C++ facade stabilizes.
- Backout: revert the latest plan-specific commits and re-enable the previous StoreEngine implementation (kept in version control) if regressions emerge; regenerate protobufs to match the reverted API before re-running tests.

# Risks & Tracking

- **Initialization order bugs:** Mitigate via integration tests that construct ComponentCatalog with missing deps and by enforcing strict sequencing in constructors.
- **Event propagation gaps:** Add unit tests for pipeline stages and telemetry subscribers to ensure failure events still trigger Global Store/metrics updates.
- **Doc drift:** Tie documentation updates to Milestone C3/D3 and block merges without README/doc diffs.
- **Python lag:** Explicitly gate Plan D on detection of API/proto deltas so SDK consumers do not see inconsistent behaviors between languages.
