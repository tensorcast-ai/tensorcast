---
slug: store-engine-materialization-service
title: Materialization Service Extraction (Plan)
status: draft
areas: ["core"]
related_code:
  - core/store/store_engine.cc
  - core/store/store_engine.h
  - core/store/materialization/**
links:
  design: ../designs/0025-store-engine-materialization-service.md
---

# Objective

Implement the design in 0025 by extracting the COPY / LOAD / AUTO logic out of `StoreEngine::materialize_replica` and into a dedicated `loading::MaterializationService`, backed by a `MaterializationRequest` helper. The plan delivers a smaller `StoreEngine`, testable service code, and updated documentation without changing external APIs.

# Current State & Grounding

- `StoreEngine::materialize_replica` spans `core/store/store_engine.cc:1238-1495`, intertwining hint validation, registry queries, COPY_ONLY GPU logic, disk ingestion, and orchestrator fallback.
- The header (`core/store/store_engine.h:89-238`) exposes only monolithic APIs; there is no separation for request/response modeling or service-like helpers.
- COPY_ONLY mode allocates replicas inline (`core/store/store_engine.cc:1363-1469`) and manually builds `ReplicaHandle`, which makes the path hard to test without a full `StoreEngine`.
- Disk ingestion helper `ingest_from_disk_internal` already exists, but the decision logic for when/how to call it lives inside `materialize_replica`.
- Documentation (core/store/README.md, docs/architecture/architecture-overview.md) still describes `StoreEngine` as the owner of all load orchestration paths.

# Phases & Milestones

- [ ] **Phase 0 – Request Context Scaffolding**
  - [ ] Milestone 0.1: Add `loading/materialization/materialization_request.{h,cc}` with `MaterializationRequest::Create` that encapsulates canonical ID derivation, view ID validation, and device ordinal checks (lifting logic from `store_engine.cc:1244-1302`).
  - [ ] Milestone 0.2: Cover the helper with focused unit tests (invalid ordinals, COPY_ONLY without artifact_id, disk path requirements).

- [ ] **Phase 1 – Materialization Service Implementation**
  - [ ] Milestone 1.1: Introduce `loading/materialization/materialization_service.{h,cc}` plus Bazel target; define `MaterializationDeps` and stub helpers (`TryReuseReplica`, `CopyFromPeer`, `LoadFromDisk`, `RunAuto`).
  - [ ] Milestone 1.2: Move COPY_ONLY logic, disk fallback, and AUTO orchestrator glue from `store_engine.cc` into the new helpers without behavior changes, reusing `ComputeViewDataHash`.
  - [ ] Milestone 1.3: Write unit tests for `MaterializationService` covering existing paths (GPU reuse, COPY_ONLY fallback failure, disk load success/failure, orchestrator fallback).

- [ ] **Phase 2 – StoreEngine Integration & Cleanup**
  - [ ] Milestone 2.1: Replace the body of `StoreEngine::materialize_replica` with request creation + service invocation; provide a private `make_materialization_deps()` utility.
  - [ ] Milestone 2.2: Remove now-redundant inline helpers from `store_engine.cc` and adjust includes to point at the new headers.
  - [ ] Milestone 2.3: Update `core/store/README.md` and `docs/architecture/architecture-overview.md` to describe the new layering (doc-sync rule).

- [ ] **Phase 3 – Validation & Rollout**
  - [ ] Milestone 3.1: Run Bazel + Python suites (`bazel test //core/store:store_engine_test`, `bazel test //daemon:grpc_service_impl_registration_test --define=use_fake_cuda=true`, `uv run pytest tests/python/test_store_session_api.py -k materialize`) and land any fixes.
  - [ ] Milestone 3.2: Perform targeted benchmarking (if possible) or at least stress-copy scenarios to confirm there is no regression in COPY_ONLY latency; document results in the PR description.
  - [ ] Milestone 3.3: Announce the refactor in release notes / internal changelog so future contributors know to extend the service instead of `StoreEngine`.

# Tasks

- Add Bazel BUILD entries for the new files (using `sc_cc_library`) and include them from `core/store/store_engine.cc`.
- Ensure `MaterializationService` only includes what it needs (forward declare heavy components) to keep compile times low.
- Provide dependency injection hooks for tests (mockable replica registry/device manager/memory pool). Consider using lightweight fakes similar to `core/store/materialization/dataplane/view/tests/view_planner_test.cc`.
- Update AGENTS/README guidance to call out the requirement to route new materialization flows through the service.
- Keep tracing/metrics semantics identical by moving the relevant `SC_TRACE_INIT_GUARD`, logging, and view hashing calls.

# Test / Rollout / Backout

- **Unit tests**: `bazel test //core/store/materialization/contracts:materialization_request_test`, `bazel test //core/store/materialization/control:materialization_service_test`.
- **Integration tests**: `bazel test //core/store:store_engine_test`, `bazel test //daemon:grpc_service_impl_registration_test --define=use_fake_cuda=true`.
- **Python smoke**: `uv run pytest tests/python/test_store_session_api.py -k materialize`.
- **Rollout**: merge design + implementation in a single PR, monitor daemon metrics (materialize latency, replica cache hits).
- **Backout**: revert to the commit prior to the extraction; no data migrations involved, so rollback is a standard git revert.

# Risks & Tracking

- **Behavior drift**: copying logic into helpers might subtly change error propagation. Mitigation: move code verbatim first, then refactor, and lock it down with new unit tests.
- **Dependency cycles**: adding new headers in `loader/` could pull in `store_engine.h`. Mitigation: keep headers minimal and rely on forward declarations.
- **Test coverage gaps**: new helpers introduce fresh code paths that lack tests today. Mitigation: prioritize unit tests in Phase 1 and reuse existing integration suites.
- **Future bypasses**: engineers might keep adding logic directly to `StoreEngine`. Mitigation: document the boundary and enforce it via code review + AGENTS guidance.
