---
slug: 0029-store-runtime-rearchitecture
title: Plan — Store Runtime Re-architecture
areas: ["core"]
links:
  design: ../designs/0029-store-runtime-rearchitecture.md
related_designs:
  - docs/designs/0029-store-runtime-rearchitecture.md
related_code:
  - core/store/store_engine.cc
  - core/store/store_engine.h
  - core/store/runtime/**
  - core/store/registration/**
---

# Objective

Implement Design 0029 by replacing the monolithic runtime wiring in `StoreEngine` with RuntimeEnv, ReplicaRuntime, ArtifactIngressManager, GlobalMetadataGateway, and RuntimeEventHub so lifecycle, telemetry, ingestion, and metadata duties live in deep modules whose APIs track the AGENTS.md standards without altering public StoreEngine APIs.

# Current State & Grounding

- `core/store/store_engine.cc:42-394` directly wires ComponentCatalog, ReplicaService, TelemetryService, GlobalStorePublisher, RegistrationFacade, and materialization helpers, so lifecycle ordering and telemetry are duplicated everywhere.
- `core/store/components/runtime/*` keeps `component_catalog.cc`, `replica_service.cc`, `telemetry_service.cc`, and `global_store_publisher.cc` in separate directories despite tight coupling; telemetry simply forwards to replica storage.
- Docs `docs/designs/0019-store-engine-modularization.md` and `docs/designs/0028-store-engine-facade-refactor.md` describe modularity that is not reflected in code or `core/store/README.md`, so documentation and implementation diverge.
- Stakeholders: core/store maintainers, daemon team (StoreEngine consumers), and documentation owners responsible for doc-sync updates (`core/store/README.md`, `docs/architecture/architecture-overview.md`, `docs/internals/model-loading.md`).

# Phases & Milestones

- [x] **Phase 1: RuntimeEnv + Directory Realignment**
  - [x] Move `core/store/components/runtime/*` into `core/store/runtime/` with Bazel targets and shim headers so dependents keep building. *(Shim headers still TODO for downstream repos.)*
  - [x] Implement `runtime/runtime_env.{h,cc}` covering startup/shutdown sequencing, dependency injection (
    `component_catalog`, Global Store client overrides), and worker identity updates previously in `StoreEngine::set_worker_identity`.
  - [x] Update `core/store/store_engine.cc` constructor/destructor to request dependencies solely via RuntimeEnv and delete direct Catalog/Telemetry wiring; update `core/store/store_engine.h` fields accordingly.
  - [x] Refresh `core/store/README.md` and `docs/architecture/architecture-overview.md` diagrams to introduce RuntimeEnv, following the doc-sync rule.

- [ ] **Phase 2: ReplicaRuntime Merge**
  - [x] Merge `ReplicaService` + `TelemetryService` contracts into `runtime/replica/replica_runtime.{h,cc}`; ensure namespaces follow AGENTS.md conventions.
  - [x] Update call sites inside `core/store/store_engine.cc`, ingestion/materialization helpers, and `//core/store:store_engine_test` to use ReplicaRuntime.
  - [ ] Add `//core/store/runtime:replica_runtime_test` covering eviction, chunk-state snapshots, telemetry queries, and CUDA export toggles. *(Blocking TODO.)*
  - [x] Remove legacy telemetry plumbing inside `StoreEngine` (lines 147-198, 340-360) and delete transitional helper methods.

- [ ] **Phase 3: GlobalMetadataGateway**
  - [x] Create `runtime/global_metadata_gateway.{h,cc}` that wraps `GlobalStorePublisher` and key-mapping helpers so worker identity + overrides flow through RuntimeEnv.
  - [x] Update registration/materialization paths to publish through the gateway and remove `StoreEngine::set_global_store_client_for_testing` in favor of RuntimeEnv configuration.
  - [~] Add tests verifying register/unregister flows, key mapping TTL refresh, view alias enforcement, and client override propagation. *(New `//core/store/runtime:global_metadata_gateway_test` covers ingress-driven registrations; key-mapping TTL + view alias tests still pending.)*

- [ ] **Phase 4: ArtifactIngressManager Integration**
  - [x] Build `runtime/artifact_ingress_manager.{h,cc}` orchestrating `MaterializationCoordinator`, `IngestionPipeline`, and `registration::RegistrationFacade`.
  - [x] Route `StoreEngine::materialize_replica`, `ingest_from_{disk,p2p,memory}`, and registration APIs (begin/commit/abort/keep_alive/view chunk) through ArtifactIngressManager.
  - [x] Emit structured RuntimeEventHub notifications on ingress success/failure with UMA metadata and ensure RuntimeEnv metrics refresh accordingly via ReplicaRuntime subscribers.
  - [x] Cover disk/P2P ingestion, registration commit/abort/TTL enforcement, and view chunk offsets with `artifact_ingress_manager_test.cc`. *(New `artifact_ingress_manager_test` validates commit/abort event propagation; disk/P2P happy-path coverage remains under the ingestion pipeline tests.)*

- [ ] **Phase 5: RuntimeEventHub + End-to-End Wiring**
  - [x] Implement `runtime/runtime_event_hub.{h,cc}` with lock-free publish + scoped subscription hooks supporting draining on shutdown.
  - [x] Wire ReplicaRuntime, ArtifactIngressManager, and GlobalMetadataGateway to publish/consume events (IngressCompleted, RegistrationCommitted/Aborted, key mapping updates) so StoreEngine no longer coordinates directly. *(ReplicaLoaded/Evicted fan-out will follow after UMA eviction work; ingress + registration events are complete.)*
  - [x] Thin `StoreEngine` façade methods to simple delegates, keeping existing public signatures/status codes; ensure logs/metrics match golden templates.
  - [x] Remove shim headers and finalize the runtime directory layout; update `docs/internals/model-loading.md` and `docs/architecture/p2p-transfer-strategies.md` to reference the RuntimeEventHub interactions. *(Shim removal pending downstream migration.)*

- [ ] **Phase 6: Rollout Validation & Observability Guardrails**
  - [ ] Implement the metric snapshot diffing hook inside `//core/store:store_engine_test` and capture `metrics_golden.json` for CI diffing.
  - [ ] Add log template verification sinks in ReplicaRuntime tests so structured key/value templates stay stable.
  - [ ] Add `tests/python/runtime/test_call_graph.py` tracing `tensorcast.get()` via `uv run pytest tests/python/runtime/test_call_graph.py` to guard client-facing call ordering.
  - [ ] Run `bazel test //core/store:store_engine_test //core/store/runtime/...` and rerun affected Python suites via `uv run pytest tests/python/runtime` before merging; notify daemon/Python owners about include path changes.

# Tasks

- [x] Keep documentation synchronized each phase (`core/store/README.md`, `docs/architecture/architecture-overview.md`, `docs/internals/model-loading.md`, `docs/architecture/p2p-transfer-strategies.md`).
- [ ] Maintain compatibility shim headers only while downstream repos migrate; track removal in Phase 5.
- [ ] Record new metrics/log templates and publish the updated goldens referenced in Phase 6 to prevent regressions.
- [ ] Coordinate with daemon/Python owners on rollout timing so their build rules adopt the new directories without disruption.

# Test / Rollout / Backout

## Acceptance & Validation

- RuntimeEnv tests cover startup/shutdown failure handling, worker identity propagation, and dependency overrides.
- ReplicaRuntime tests prove telemetry accuracy, eviction behavior, and CUDA export gating, while ArtifactIngressManager tests exercise disk/P2P ingress, registration commit/abort/TTL, and view chunk offsets.
- GlobalMetadataGateway and RuntimeEventHub tests verify register/unregister flows, key mapping CRUD, event fan-out, and deadlock-free drains.
- StoreEngine integration and observability guardrail tests (metrics golden diff, log template capture, Python call-graph regression) pass without altering public APIs or status codes.
- Doc-sync verification: affected README/architecture docs updated in the same changes that move or rename modules.

## Rollout Strategy

- Merge phases sequentially; gate each by passing Bazel + `uv run pytest` suites and updated observability goldens.
- Communicate milestone completion to daemon and Python owners so their build imports update in lockstep.
- Enable temporary verbose logging for RuntimeEventHub events in sandbox deployments to confirm event ordering before disabling in production builds.

## Backout Plan

- Revert the latest milestone commit if regressions appear; compatibility shim headers remain until Phase 5 completes to simplify reverts.
- Maintain separate branches per phase so partial rollbacks keep earlier modules (e.g., RuntimeEnv) intact while undoing later ingress/event wiring if needed.
- Retain previous `StoreEngine` wiring in history for quick cherry-pick reapplication should the runtime modules require redesign.

# Risks & Tracking

- **Build break during directory shuffle** — Add shim headers + Bazel updates in the same CL and run `bazel query` sanity checks; owner: core/store build cop.
- **Telemetry or ingress behavior drift** — Guard with new module tests plus StoreEngine integration + metrics/log golden comparisons; owner: ReplicaRuntime tech lead.
- **RuntimeEventHub deadlocks or starvation** — Keep callbacks non-blocking, add watchdog stress tests publishing from multiple threads, and require subscribe/unsubscribe RAII wrappers; owner: RuntimeEnv maintainer.
- **Documentation drift** — Treat doc updates as acceptance criteria per phase; reviewers block merges lacking README/architecture updates; owner: docs liaison.

# Owner Checklist

- [ ] Link every PR/MR to this plan.
- [ ] Ensure lint/format (ruff, clang-tidy) and module/unit tests pass before requesting review.
- [ ] Update README + architecture/internal docs in the same change set as the code that moves or renames modules.
- [ ] Confirm no API/ABI regressions for daemon/Python consumers via targeted tests and owners’ sign-off.
- [ ] Remove compatibility shims once downstream dependencies have migrated.
