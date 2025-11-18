---
slug: 0027-materialization-unification
title: Materialization Control/Data-plane Unification (Plan)
status: complete
areas: ["core"]
related_code:
  - core/store/materialization/**
  - core/store/docs/**
links:
  design: ../designs/0027-materialization-unification.md
---

# Objective

Deliver the layering described in design 0027 by splitting control logic, contracts, and dataplane runtimes into `core/store/materialization/`, adding loader registry/source router seams, and ensuring StoreEngine depends only on the control package. The end state provides narrow Bazel targets, stable APIs between orchestration and loaders, and updated documentation so future work extends the right layer.

# Current State & Grounding

- `StoreEngine::materialize_replica()` now delegates exclusively to `core/store/materialization/control/materialization_service.h` and depends on `MaterializationDeps` bundles assembled in `make_materialization_deps()` (`core/store/store_engine.cc:1150-1280`). The control layer contains no direct includes of disk or P2P loaders anymore.
- `MaterializationService`, `MaterializeOrchestrator`, and `ReplicaRegistrationHelper` reside in `core/store/materialization/control/` with unit tests under the same package, so orchestration logic is fully isolated from dataplane runtime code.
- Loader contracts, pumps, sources/sinks, metadata helpers, and verification utilities all live within `core/store/materialization/dataplane/**`, and StoreEngine-facing code imports them via the registry interfaces introduced in the contracts layer.
- The legacy `core/store/loading` and `core/store/loader` trees (forwarding headers plus Bazel aliases) have been removed. Every Bazel target now references the narrow `//core/store/materialization/...` packages directly, enforcing the layering described in this plan.

# Phases & Milestones

- [x] **Phase Bridge – Materialization Shim**
  - [x] Milestone B.1: Introduce `MaterializationRequest` plus validation tests so callers stop duplicating canonical-id/device checks (`core/store/loading/materialization/materialization_request.{h,cc}`, `//core/store/materialization/contracts:materialization_request_test`).
  - [x] Milestone B.2: Factor the reuse/COPY_ONLY/LOAD_ONLY/AUTO logic into `MaterializationService` with injectable deps (`core/store/loading/materialization/materialization_service.{h,cc}`, `_test.cc`).
  - [x] Milestone B.3: Wire `StoreEngine` → `MaterializationService` via `MaterializationDeps` and route AUTO through `MaterializeOrchestrator` using the new `MaterializationBackend` interface (`core/store/store_engine.cc:1150-1280`, `core/store/loading/materialization/materialize_orchestrator.cc:18-140`).

- [x] **Phase 0 – Contracts Scaffolding**
  - [x] Milestone 0.1: Create `core/store/materialization/{contracts,planning,control,dataplane}` directories with stub `BUILD` files plus README placeholders describing dependencies; add Bazel aliases exposing `//core/store/loading:*` to keep downstreams compiling.
  - [x] Milestone 0.2: Move `materialization_request.{h,cc}` and its tests into `materialization/contracts/`, leaving deprecated forwarding headers under `core/store/loading/` to ease incremental landings.
  - [x] Milestone 0.3: Split `loading_spec.h` into `contracts/view/{view_spec.h,view_plan.h,view_id.h}` and adjust consumers; introduce `IArtifactLoaderRegistry` and `ArtifactSourceRouter` interfaces with lightweight fakes for tests.
  - [x] Milestone 0.4: Update `//core/store/materialization/contracts:loading_spec` and `//core/store/materialization/control:materialization_service` BUILD files to depend on the contracts target and ensure IWYU confirms no dataplane headers leak in.

- [x] **Phase 1 – Control Layer Migration**
  - [x] Milestone 1.1: Relocate `materialization_service`, `materialize_orchestrator`, and `replica_registration_helper` into `materialization/control/` and update Bazel deps to point at contracts plus StoreEngine components.
  - [x] Milestone 1.2: Introduce a temporary adapter in `materialization/control/loader_registry_adapter.{h,cc}` that implements `IArtifactLoaderRegistry` via the existing disk/P2P loaders; update control unit tests to consume registry/router fakes.
  - [x] Milestone 1.3: Rewire `StoreEngine::materialize_replica` (`core/store/store_engine.cc:1238-1495`) to include the new control headers and depend on `//core/store/materialization/control:materialization_service_lib`; keep runtime wiring identical by passing the adapter implementation.
  - [x] Milestone 1.4: Refresh documentation references in `core/store/README.md` and `core/store/docs/architecture.md` to describe the contracts/control boundary established in this phase.

- [x] **Phase 2 – Dataplane Relocation**
  - [x] Milestone 2.1: Move loader contract headers (`core/store/loader/loader.h`, `source.h`, `sink.h`, `buffer_pool.h`, `inline_buffer_loader.h`) into `materialization/dataplane/contracts/` with namespace adjustments and forwarding headers left behind.
  - [x] Milestone 2.2: Shift runtime/buffering files (`pump*.{h,cc}`, `streaming_buffer_adapter.{h,cc}` and associated tests) under `materialization/dataplane/runtime/` with private Bazel visibility.
  - [x] Milestone 2.3: Relocate disk/P2P source implementations (`file_partition_source*.{h,cc}`, `multi_safetensors_source*.{h,cc}`, `chunk_source_adapter*.{h,cc}`) into `materialization/dataplane/sources/` along with tests.
  - [x] Milestone 2.4: Move metadata and verification helpers (`canonical_index.{h,cc}`, `index_reader.{h,cc}`, `safetensors_util.{h,cc}`, `verification_utils.{h,cc}` and unit tests) into `materialization/dataplane/{metadata,verification}/`, validating the move via `bazel test --define=use_fake_cuda=true //core/store/materialization/control:materialization_service_test`.
  - [x] Milestone 2.5: Update Bazel targets so that `//core/store/loader` simply re-exports the new dataplane packages for downstream consumers that have not yet migrated.

- [x] **Phase 3 – Compatibility Removal**
- [x] Milestone 3.1: Delete the deprecated `core/store/loading/**` headers and update all includes/Bazel deps to point directly at `core/store/materialization/{contracts,control,planning}`.
- [x] Milestone 3.2: Remove the `//core/store/loader` alias package, repoint every consumer (daemon, checkpoint tooling, replication tests, examples) to the narrow dataplane targets, and drop the shim BUILD file entirely.
- [x] Milestone 3.3: Sweep documentation (`AGENTS.md`, `core/store/materialization/README.md`, `docs/architecture/p2p-transfer-strategies.md`, plan/design 0027) so no guidance references the legacy directories.

> **Plan 0028 follow-up:** The temporary `loader_registry_adapter` noted in Phase 1 has now been deleted. Disk and P2P ingestion flow exclusively through `materialization::runtime::pipeline::IngestionPipeline`, which emits events consumed by TelemetryService and GlobalStorePublisher. Successful loads are registered with Global Store automatically, so StoreEngine/MaterializeOrchestrator never call the old helper methods directly.

- [x] **Phase 4 – Validation & Rollout**
  - [x] Milestone 4.1: Run `bazel test --define=use_fake_cuda=true //core/store/materialization/control:materialization_service_test` to ensure control-only deps rebuild cleanly after the directory deletions.
  - [x] Milestone 4.2: Capture follow-up verification guidance (`bazel test //core/store:store_engine_test`, `bazel test //daemon:grpc_service_impl_registration_test --define=use_fake_cuda=true`, `uv run pytest tests/python/test_store_session_api.py -k materialize`) for CI/regression gates.
  - [x] Milestone 4.3: Document the final module boundaries in release notes and project docs so new features target the right package without relying on the removed shims.

# Tasks

- Guard against regressions by running `rg 'core/store/(loading|loader)'` in presubmits so no one attempts to resurrect the removed include paths.
- Keep the `materialization/contracts/tests/` fakes current so control-layer unit tests stay hermetic without depending on dataplane runtime code.
- Adjust namespace declarations throughout moved files (`namespace materialization::dataplane`) and ensure canonical includes follow UMA V3 policies.
- Update module documentation (`core/store/README.md`, `core/store/docs/architecture.md`) and AGENT guidance whenever a phase changes responsibilities, satisfying the doc-sync rule.
- Add `BUILD` visibility and dependency comments clarifying which packages may depend on which (control ← contracts & registry only) to make future violations obvious.
- Prefer using shell-level move commands (e.g., `mv old_path new_path`) for relocations before resorting to scripted rewrites so history stays clear and reversible.

# Test / Rollout / Backout

- **Unit tests**: `bazel test //core/store/materialization/contracts:materialization_request_test`, `bazel test //core/store/materialization/control:materialization_service_test`, and the `//core/store/materialization/dataplane/...` suites (superseding the legacy `//core/store:materialization_*` and `//core/store:loader_*` targets).
- **Integration tests**: `bazel test //core/store:store_engine_test`, `bazel test //daemon:grpc_service_impl_registration_test --define=use_fake_cuda=true`.
- **Python smoke**: `uv run pytest tests/python/test_store_session_api.py -k materialize`, `uv run pytest tests/python/test_safetensors_loading.py`.
- **Rollout**: Merge phases sequentially, verify dashboards (materialize latency, chunk repair hits) after each landing, and broadcast the new directory structure to contributors.
- **Backout**: Because forwarding headers and the loader-registry adapter remain until Phase 4, revert individual phase commits to fall back to the previous layout without schema or API changes.

# Risks & Tracking

- **Large-scale file moves** can cause rebases and review fatigue. Mitigation: land per-table chunks (contracts, runtime, metadata, verification) and keep forwarding headers until all consumers move.
- **Hidden dependency leaks** may reappear if new code includes dataplane headers directly. Mitigation: enforce Bazel query guardrails and add presubmit checks referencing `tools/lint/check_uma_aliases.sh`.
- **Behavior drift** when wiring the registry/router may change fallback order or error surfaces. Mitigation: copy logic verbatim first, add focused unit tests covering COPY_ONLY/AUTO flows, and compare logs between old/new builds.
- **Documentation drift** if AGENT rules and READMEs lag behind code. Mitigation: bake doc updates into each phase’s definition of done and block landings that lack doc-sync review.
