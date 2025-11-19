---
slug: 0030-store-runtime-consolidation
title: Plan — Store Runtime Consolidation
areas: ["core"]
links:
  design: ../designs/0030-store-runtime-consolidation.md
related_designs:
  - docs/designs/0030-store-runtime-consolidation.md
related_code:
  - core/store/runtime/**
  - core/store/materialization/**
  - core/store/components/**
  - core/store/store_engine.{h,cc}
---

# Objective

Deliver Design 0030 by consolidating StoreEngine’s runtime into RuntimeContext, ReplicaRuntime, IngestionRuntime, and MetadataGateway. The plan deletes ArtifactIngressManager, MaterializationCoordinator, MaterializationService, RegistrationFacade, and GlobalMetadataGateway as public concepts while keeping StoreEngine’s facade API stable and maintaining doc-sync between `core/store/docs/architecture.md`, module READMEs, and the new directory layout.

# Current State & Grounding

- `core/store/docs/architecture.md:14-167` shows ArtifactIngressManager, MaterializationCoordinator, MaterializationService, RegistrationFacade, GlobalMetadataGateway, and RuntimeEventHub referencing each other, producing cycles (e.g., AIM ↔ GMG ↔ EventHub ↔ AIM).
- `core/store/runtime/` historically mixed subdirectories (`component_catalog`, `artifact_ingress_manager`, `global_metadata_gateway`, `runtime_event_hub`) rather than the target `context/replica/ingestion/metadata` hierarchy; this phase begins by moving ingestion code under `runtime/ingestion/`.
- Event dispatch is coupled to `RuntimeEventHub` (`core/store/runtime/runtime_event_hub.{h,cc}`) which owns knowledge of ReplicaRuntime, ArtifactIngressManager, and GMG, making testing and replacement difficult.
- The legacy ingestion stack spread across `core/store/runtime/artifact_ingress_manager.*`, `core/store/materialization/control/*`, and `core/store/materialization/runtime/pipeline/*` duplicated orchestration and required `StoreEngine` to coordinate success/failure metrics.
- `IngestionRuntime::Config` still exposes `test_disk_result_status` / `test_p2p_result_status`, so unit tests short-circuit ingestion instead of exercising the real pipeline and cannot observe emitted events without wiring a full `RuntimeContext`.
- Registration facade (`core/store/components/registration/registration_facade.*`) and `runtime/global_metadata_gateway.*` duplicate metadata ownership, forcing clients to juggle two services for any publish flow.
- Stakeholders: core/store runtime maintainers (primary owners), doc owners (architecture docs + READMEs), daemon & Python client owners (ensure facade unchanged). Tests exercised via Bazel (`//core/store/runtime/...`, `//core/store:store_engine_test`) and Python (`uv run pytest tests/python/runtime/...`).

# Phases & Milestones

- [x] **Phase 1 — RuntimeContext foundation**
  - [x] Move `component_catalog.{h,cc}` and `runtime_event_hub.{h,cc}` into `core/store/runtime/context/` with Bazel targets renamed to `runtime_context_*`.
  - [x] Embed the event dispatcher inside RuntimeContext: provide `EventPublisher` and `EventSubscription` handles injected into runtimes, wiring the Bazel dependency on `@folly//folly:mpmc_queue`. Delete direct references to ReplicaRuntime, ArtifactIngressManager, and GMG from the dispatcher.
  - [x] Update `StoreEngine` construction (`core/store/store_engine.cc:40-220`) to depend on RuntimeContext handles rather than separate catalog/event hub members.
  - [x] Build a concurrent publisher stress test (e.g., `runtime_context_concurrency_test.cc`) that exercises multiple producers sharing the Folly MPMC queue to validate ordering/backpressure guarantees referenced in the design.
  - [x] Add `//core/store/runtime/context:runtime_context_test` covering startup/shutdown order, worker identity propagation, and event publish/drain semantics.
  - [x] Update docs (`core/store/docs/architecture.md`, `core/store/README.md`) to explain RuntimeContext and retire RuntimeEventHub terminology.

- [x] **Phase 2 — ReplicaRuntime integration**
  - [x] Relocate `replica_runtime.*` into `core/store/runtime/replica/` and wire it to RuntimeContext’s event publishers/subscriptions.
  - [x] Emit `replica_loaded`, `replica_evicted`, and `remote_access_toggled` via the new event handles; remove any direct wiring to ArtifactIngressManager/GlobalMetadataGateway.
  - [x] Update UMA/telemetry documentation sections in `core/store/docs/architecture.md` and `core/store/store/README.md` to reflect ReplicaRuntime’s new event flows.
  - [x] Extend `//core/store/runtime/replica:replica_runtime_test` to cover event emission and subscription plumbing.

- [ ] **Phase 3 — IngestionRuntime consolidation**
  - [x] Create `core/store/runtime/ingestion/ingestion_runtime.{h,cc}` that exposes StoreEngine ingestion APIs and embeds existing pipeline/planner helpers.
  - [x] Migrate ArtifactIngressManager responsibilities: disk/P2P/inline ingestion, materialize orchestration, registration hooks. Delete `artifact_ingress_manager.*` after callers switch.
  - [x] Move MaterializationCoordinator + MaterializationService code into `ingestion_runtime/` (retaining planner helpers under `pipeline/`), reducing surface area to a single runtime module.
  - [x] Wire RuntimeContext event publishers into IngestionRuntime for `ingestion_started`, `ingestion_completed`, and `ingestion_failed`, ensuring `publish_context_id` is propagated for dedupe.
- [x] Introduce the `IngestionEventSink` interface plus the production `RuntimeContextEventSink` wrapper so event publication can be overridden without instantiating a full RuntimeContext in tests.
- [x] Define `IngestionTestHooks` (`before_pipeline_start`, `mutate_completion_event`, `override_result`) with logging/metric guards so tests can short-circuit requests while production builds leave the hooks null.
- [x] Expose `IngestionRuntimeDependencies` (pipeline/coordinator factories, event sink override, structured hooks) and the helper library under `core/store/runtime/ingestion/testing/` (`FakeIngestionPipeline`, `RecordingEventSink`, `ScopedIngestionRuntimeTestHarness`) behind `testonly = 1` Bazel targets so unit tests can install deterministic fakes without production-only flags.
- [x] Remove `test_disk_result_status` / `test_p2p_result_status` from `IngestionRuntime::Config` and update all call sites to rely on injected dependencies.
  - [x] Extend unit tests (`core/store/runtime/ingestion/ingestion_runtime_test.cc`) to cover disk & P2P flows plus event emission; rerun `bazel test //core/store/materialization/runtime/pipeline/...` to ensure dataplane coverage. *(Pipeline suite rerun is still pending; tracked in Phase 6 test plan.)*
  - [x] Update ingestion documentation (`core/store/docs/architecture.md`, `docs/internals/model-loading.md`, `docs/architecture/p2p-transfer-strategies.md`) to describe the consolidated runtime and new events.

- [ ] **Phase 4 — MetadataGateway merge**
  - [x] Introduce `core/store/runtime/metadata/metadata_gateway.{h,cc}` merging GlobalMetadataGateway and RegistrationFacade responsibilities (registration CRUD, TTL keep-alive, CUDA IPC export).
  - [x] Replace direct RegistrationFacade usage across StoreEngine and ingestion code with MetadataGateway calls; remove redundant helper headers and Bazel targets.
  - [x] Subscribe MetadataGateway to RuntimeContext `ingestion_completed` events for auto-publish flows; ensure handlers remain non-blocking and dedupe explicit vs. event-driven publishes via `publish_context_id`.
  - [x] Build `//core/store/runtime/metadata:metadata_gateway_test` validating register/unregister flows, TTL refresh, CUDA export metadata, ingestion-event-triggered publish, and dedupe semantics.
  - [x] Update documentation (`core/store/docs/architecture.md`, `docs/architecture/architecture-overview.md`, module READMEs) to capture the merged gateway and publish lifecycle.

- [ ] **Phase 5 — Directory + build graph cleanup**
  - [x] Restructure `core/store/runtime/` to the final layout (`context/`, `replica/`, `ingestion/`, `metadata/`); relocate headers/sources and update Bazel `sc_cc_library` targets plus deps.
  - [x] Delete superseded files: `runtime/global_metadata_gateway.*`, `components/registration/registration_facade.*`, `runtime/artifact_ingress_manager.*`, `materialization/control/materialization_coordinator.*`, `materialization/control/materialization_service.*` (leave pipeline/data plane under `materialization/dataplane/`).
  - [x] Remove all compatibility shim headers/aliases—downstreams must use the RuntimeContext/Replica/Ingestion/Metadata APIs directly because the project has not shipped yet.
  - [x] Drop `RuntimeEnv::register_shutdown_dependency` and rename the StoreEngine field to `ingestion_runtime_` so no code references the ArtifactIngressManager/ComponentCatalog era.
  - [x] Update documentation (`core/store/docs/architecture.md`, `docs/architecture/architecture-overview.md`, `docs/internals/model-loading.md`, `docs/architecture/p2p-transfer-strategies.md`) to reflect the new topology and directory map (doc-sync rule).

- [ ] **Phase 6 — Validation, rollout, and cleanup**
  - [ ] Run integration suites: `bazel test //core/store:store_engine_test //core/store/runtime/...`
  - [x] Remove temporary shims; ensure include paths and Bazel targets referenced by daemon/Python builds point to the new directories (complete: only final APIs remain).
  - [ ] Final documentation sweep verifying terminology, diagrams, and references to removed services are gone.

# Tasks

- [ ] Maintain doc-sync for every code change via `core/store/docs/architecture.md`, module READMEs, and related guides.
- [ ] Keep AGENTS.md alignment: ensure new APIs documented with naming compliance and error-handling rationale.
- [ ] Coordinate with daemon/Python owners to validate no StoreEngine API changes propagate outside runtime internals.
- [x] Track removal of compatibility shims in issue tracker; downstreams now rely exclusively on the consolidated runtime layout.

# Acceptance, Testing, and Rollout

## Acceptance Criteria

- StoreEngine facade signatures/status codes remain identical; daemon/Python integration tests pass unchanged.
- RuntimeContext event ordering guarantees documented in Design 0030 hold under stress tests.
- IngestionRuntime and MetadataGateway wholly replace ArtifactIngressManager, MaterializationCoordinator, MaterializationService, RegistrationFacade, and GlobalMetadataGateway—no residual references remain in `core/store/runtime/` or `core/store/store_engine.{h,cc}`.
- Documentation updated to describe the three-runtime architecture with RuntimeContext.
- `IngestionRuntime::Config` no longer exposes `test_disk_result_status` / `test_p2p_result_status`; overrides flow through `IngestionRuntimeDependencies`.
- Ingestion events default to the RuntimeContext publisher whenever no event-sink override is supplied, so production behavior matches the pre-injection path.
- `//core/store/runtime/ingestion:ingestion_runtime_test` covers disk and P2P success/failure flows using the `core/store/runtime/ingestion/testing/` helpers to capture events deterministically.

## Test Plan

- `bazel test //core/store/runtime:runtime_context_test //core/store/runtime/metadata:metadata_gateway_test //core/store/materialization/runtime/pipeline/...`
- `bazel test //core/store/runtime/ingestion:ingestion_runtime_test --define=use_fake_cuda=true` (covers injected pipelines, event sinks, and structured hooks for disk + P2P flows)
- `bazel test //core/store:store_engine_test //core/store:materialization_contracts_test`
- `uv run pytest tests/python/runtime/test_call_graph.py`
- Stress test RuntimeContext by running the concurrent publisher test under `--runs_per_test=50` to exercise the Folly MPMC queue under load.
- Optional stress/soak: enable verbose RuntimeContext logging in integration env to confirm event ordering.

## Rollout / Backout

- Roll out phase-by-phase; each phase must land with green Bazel + pytest runs and doc updates.
- Compatibility shim headers/Bazel aliases have been removed; downstreams must follow the consolidated runtime layout, and future work will not reintroduce adapters.
- Backout: revert the latest phase commit; because RuntimeContext and runtimes are modular, reverts can target affected directories without disturbing prior phases.

# Risks & Mitigations

- **Event queue starvation/deadlock** — Keep handlers lightweight, add stress tests with concurrent publishers, and document max handler latency; owner: RuntimeContext maintainer.
- **Ingestion regression** — Use existing pipeline tests plus new IngestionRuntime suites; add tracing around `ingestion_completed` to validate single-fire semantics; owner: IngestionRuntime lead.
- **Test hook misuse in production** — Keep `IngestionTestHooks` declarations in `core/store/runtime/ingestion/testing/` with `testonly = 1`, guard overrides with `ABSL_CHECK` + metrics/logging when `override_result` fires, and ensure production binaries never link the helper targets; owner: IngestionRuntime lead.
- **Metadata publish drift** — Ensure MetadataGateway continues to receive ingestion completions (direct pipeline callbacks) and run golden tests comparing Global Store RPC payloads; owner: MetadataGateway lead.
- **Documentation drift** — Enforce doc-sync checklist; reviewers block runtime changes lacking architecture README updates; owner: docs liaison.

# Owner Checklist

- [ ] Reference this plan in every CL/PR touching runtime consolidation.
- [ ] Run Bazel + pytest commands listed above before requesting review.
- [ ] Update documentation alongside code changes (doc-sync rule).
- [ ] Coordinate with daemon/Python owners for rollout timing and include path updates.
- [x] Remove compatibility shims once downstreams confirm migration (complete: final APIs only).
