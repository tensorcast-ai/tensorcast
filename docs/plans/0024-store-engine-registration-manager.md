---
slug: store-engine-registration-manager
title: Store Engine Registration Manager Extraction (Plan)
areas: ["core","daemon"]
links:
  design: ../designs/0024-store-engine-registration-manager.md
related_code:
  - core/store/store_engine.*
  - core/store/components/registration/**
  - daemon/**
---

# Objective

Extract the RFC-0006 registration lifecycle from `StoreEngine` into `components::ArtifactRegistrationManager` (ARM) without preserving legacy compatibility branches. The plan implements the design in 0024 by creating the new component, migrating begin/commit/TTL/view logic, delegating from `StoreEngine`, and updating documentation/tests to reflect the new ownership boundary.

# Current State

- `StoreEngine` owns registration state via `PendingRegistrationEntry`, `pending_regs_`, and related mutexes (`core/store/store_engine.h:395-454`).
- Begin/commit/view APIs are monolithic functions inside `core/store/store_engine.cc:1898-2710`, tightly coupled to replica creation, device management, and Global Store interactions. There is no released client depending on this internal structure, so we can delete the old code once the manager exists.
- Tests for registration ride on daemon/service suites (e.g., `daemon/grpc_service_impl_registration_test.cc`) because there is no isolated component to exercise.
- Documentation (core/store/README.md, architecture guides) still states that StoreEngine “directly manages” registration.

# Phases & Milestones

- [x] **Phase 0 – Scaffolding & Interfaces**
  - [x] Milestone 0.1: Create `core/store/components/registration/` with `registration_facade.{h,cc}` (formerly `artifact_registration_manager.{h,cc}`), define `RegistrationResources`, `WorkerIdentity`, `ReplicaFactory`, and stub methods (`Begin`, `Commit`, `Abort`, `KeepAlive`, `IngestViewChunk`, `GetViewIngestedBytes`).
  - [x] Milestone 0.2: Add BUILD targets (`sc_cc_library` + tests) and wire includes for loaders/hash utilities.
  - [ ] Milestone 0.3: Write initial unit tests skeleton (`registration_facade_test.cc`) asserting constructor wiring and resource validation.

- [x] **Phase 1 – Begin Flow Migration**
  - [x] Milestone 1.1: Move `StoreEngine::begin_register_artifact` logic into ARM’s `Begin`, reusing helper functions (`sum_view_write_bytes`, canonical range merging) lifted from `store_engine.cc`.
  - [x] Milestone 1.2: ARM stores pending contexts (IPC handle, replica ptr, TTL metadata) and exposes the same `RegistrationBeginResult`; `StoreEngine` delegates to ARM.
  - [ ] Milestone 1.3: Add unit tests for Begin path (view planning validation, eviction fallback, TTL initialization).

- [x] **Phase 2 – Commit, View Ingestion, TTL**
  - [x] Milestone 2.1: Port `commit_registered_artifact` into ARM, including zero-fill of partial views, hash computation, and Global Store updates; ensure `WorkerIdentity` data flows through.
  - [x] Milestone 2.2: Port `ingest_view_registration_chunk`, `get_view_registration_ingested_bytes`, `keep_alive_registered_artifact`, and `abort_registered_artifact` into ARM with equivalent error handling.
  - [ ] Milestone 2.3: Expand ARM unit tests to cover view ingestion (SERVER vs CLIENT), TTL expiration cleanup, and abort semantics.

- [x] **Phase 3 – StoreEngine Integration & Cleanup**
  - [x] Milestone 3.1: Add `std::unique_ptr<ArtifactRegistrationManager> registration_manager_;` to `StoreEngine`, instantiate it in the constructor, and expose a `set_worker_identity` forwarder.
  - [x] Milestone 3.2: Remove `PendingRegistrationEntry`, `pending_regs_`, and related helpers from `StoreEngine`; public APIs become thin wrappers that call the manager with no fallback to legacy paths.
  - [x] Milestone 3.3: Ensure metrics/tracing calls either move into ARM or remain covered via `RegistrationResources`.

- [x] **Phase 4 – Testing, Docs, and Rollout**
  - [ ] Milestone 4.1: Extend C++ coverage (`bazel test //core/store:store_engine_test`, new ARM test target) and daemon registration integration tests to ensure no regressions.
  - [x] Milestone 4.2: Update `core/store/README.md`, `docs/architecture/architecture-overview.md`, and `docs/internals/adding-metrics.md` (if necessary) to call out ARM ownership.
  - [x] Milestone 4.3: Add release notes / migration guidance (minimal, since APIs stay stable) and verify any internal tooling referencing `PendingRegistrationEntry` now imports ARM headers.

# Tasks

- Implement helper utilities in `core/store/components/registration/helpers.h` (optional) for view-plan math and TTL enforcement shared across methods.
- Introduce tracing/metrics plumbing inside ARM so Begin/Commit spans remain observable (reuse `SC_TRACE_INIT_GUARD`).
- Provide a `ReplicaFactory` lambda inside `StoreEngine` that wraps `get_or_create_replica` to avoid circular dependencies.
- Ensure `global_store_client_` remains optional: ARM must tolerate a null shared_ptr and skip GS updates gracefully, mirroring current behavior.
- Delete or relocate now-unused static functions from `store_engine.cc` once their equivalents live inside ARM.

# Acceptance / Validation

- **Build smoke:** `bazel build //core/store:store_engine` (latest run from this change to ensure the new component integrates cleanly).
- Unit tests: `bazel test //core/store/components/registration:registration_facade_test`.
- Integration tests: `bazel test //daemon:grpc_service_impl_registration_test --define=use_fake_cuda=true` and `bazel test //core/store:store_engine_test`.
- Python/API regression (optional sanity): `uv run pytest tests/python/test_store_session_api.py -k register`.
- Manual verification: stress register/unregister flows on a dev node, confirm TTL expiry metrics and Global Store updates behave as before.

# Rollout & Backout

- Rollout: merge the ARM component and delegation changes together; no feature flag or dual-stack period is needed. Monitor daemon metrics (`tc_register_pending_gauge`, `tc_register_commit_seconds`) to ensure they remain stable.
- Backout: revert to the previous commit where `StoreEngine` owned registration (docs + code). Since RPC contracts are unchanged, rollback risk is limited to code structure.

# Risks & Mitigations

- **Behavioral drift during migration**: minimize edits while copying logic; rely on comprehensive unit/integration tests.
- **Thread-safety regressions**: enforce mutex discipline inside ARM, add tests for concurrent Begin/Commit operations.
- **Documentation debt**: update READMEs/architecture docs as part of Phase 4 to prevent future confusion.
- **Build regression**: ensure new directories leverage `sc_cc_library` and hook into existing Bazel deps to avoid IWYU issues.
