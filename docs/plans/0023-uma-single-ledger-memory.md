---
slug: uma-single-ledger-memory
title: UMA Single-Ledger Memory (Plan)
areas: ["core"]
links:
  design: ../designs/0023-uma-single-ledger-memory.md
related_code:
  - core/common/memory/**
  - core/store/replica/**
  - core/store/store_engine.cc
---

# Objective

Maintain the single-ledger UMA architecture where CPU virtual address reservation, chunk telemetry, and export bookkeeping are owned solely by `UnifiedMemoryAuthority`. StoreEngine/daemon callers consume UMA snapshots, auxiliary CPU-VA abstractions remain retired, and documentation reflects the UMA-only layering.

# Current State

- UMA’s embedded `CpuArena` owns CPU virtual address reservation, pin leases, and chunk telemetry; there is no parallel virtual-address subsystem.
- StoreEngine/Replica APIs consume UMA snapshots (`snapshot_cpu_chunks` / `get_chunk_states_uma`), and direct-write/export helpers expose UMA-owned keepalives.
- Bazel targets reference only UMA artifacts, and UMA-focused tests (`core/store/replica/unified_memory_authority_test.cc`, `core/store/replica/unified_memory_authority_plan_commit_test.cc`, daemon telemetry smoke tests) cover allocation, plan/commit, CPU preemptible marking, export flows, and the regression cases formerly covered by the virtual-address suite.
- Documentation (`core/store/README.md`, `docs/architecture/architecture-overview.md`, `docs/internals/preemptible-memory.md`, `AGENTS.md`) describes the UMA-only architecture and is in sync with the implementation.

# Phases & Milestones

- [x] **Phase 0: Ledger & Telemetry Grounding**
  - [x] Milestone 0.1: Introduce UMA snapshot helpers (e.g., `snapshot_cpu_chunks`) and switch StoreEngine/daemon telemetry to them (`core/store/store_engine.cc`, daemon observers).
  - [x] Milestone 0.2: Remove `core/store/replica/chunk_meta.h` and update tests to rely on UMA `ChunkRecord` data.

- [x] **Phase 1: Fold CpuArena into UMA**
  - [x] Milestone 1.1: Move mmap/mlock/pin logic into a private `CpuArena` struct inside UMA (`core/store/replica/unified_memory_authority.{h,cc}`); provide internal helpers (`allocate_cpu_region`, `advise_preemptible`, `pin_span`, `map_file_segments`, etc.).
  - [x] Milestone 1.2: Rewire UMA APIs (`mark_cpu_chunks_preemptible`, `post_gpu_load_policy`, `set_exported`, `grant_direct_write`, `plan_load`) to call the new helpers using chunk offsets from `ChunkRecord`.

- [x] **Phase 2: Remove Auxiliary VA Surface**
  - [x] Milestone 2.1: Drop redundant virtual-address files/BUILD deps and remove `va_space_` from StoreEngine (`core/store/store_engine.{h,cc}`) so UMA allocates CPU regions directly.
  - [x] Milestone 2.2: Replace exported pin keepalives with UMA-owned opaque handles (`std::shared_ptr<void>`), ensuring public APIs expose only UMA constructs.

- [x] **Phase 3: Documentation & Validation**
  - [x] Milestone 3.1: Update `docs/internals/preemptible-memory.md`, `docs/architecture/architecture-overview.md`, and `core/store/README.md` to describe the single-ledger UMA design.
  - [x] Milestone 3.2: Add UMA-focused regression coverage (allocation, write hooks, preemptible marking, export/direct-write flows) via `core/store/replica/unified_memory_authority_test.cc`, `core/store/replica/unified_memory_authority_plan_commit_test.cc`, and daemon-side telemetry tests.

# Tasks

- [x] Create UMA snapshot APIs returning immutable views of `ChunkRecord` (path: `core/store/replica/unified_memory_authority.h`) and use them in StoreEngine telemetry surfaces (`core/store/store_engine.cc:2683-2708`) and daemon metrics utilities.
- [x] Inline the CPU allocation flow into UMA by implementing the nested `CpuArena` helper responsible for mmap/mlock/pin operations while keeping per-artifact mutex semantics.
- [x] Rework export and direct-write keepalives to store UMA-owned structs containing the scoped pin handles; ensure `set_exported` and `grant_direct_write` continue to prevent premature `munlock`.
- [x] Remove redundant virtual-address references from Bazel targets (`core/common/memory/BUILD.bazel`, `core/store/BUILD.bazel`, tests`) and delete those files after UMA builds cleanly.
- [x] Add UMA-specific tests that cover allocation, write hooks, preemptible marking, pinning, export keepalive, and eviction paths (`core/store/replica/unified_memory_authority_test.cc`, `core/store/replica/unified_memory_authority_plan_commit_test.cc`, daemon telemetry smoke tests).
- [x] Refresh documentation: adjust diagrams and narrative sections in `docs/internals/preemptible-memory.md` and `docs/architecture/high-availability-design.md` (if applicable) to show UMA-only architecture; ensure AGENTS/README updates satisfy the doc-sync rule.

# Acceptance Checks

- UMA is the only CPU virtual-address authority (`core/store/replica/unified_memory_authority.h`), and StoreEngine builds without auxiliary VA dependencies.
- StoreEngine and daemon telemetry read chunk state exclusively via UMA snapshot helpers (`core/store/store_engine.cc`, daemon metrics utilities), eliminating duplicate metadata paths.
- UMA allocation/pinning/export flows are covered by Bazel tests such as `bazel test --define=use_fake_cuda=true //core/store/replica:unified_memory_authority_test //core/store/replica:unified_memory_authority_plan_commit_test` and daemon telemetry tests.
- Documentation describing memory authority (e.g., `core/store/README.md`, `docs/internals/preemptible-memory.md`, `docs/architecture/architecture-overview.md`) matches the UMA single-ledger model.
- Direct-write/export call paths continue to pass daemon/store integration tests (e.g., `bazel test //daemon:grpc_service_impl_registration_test`).

# Status Update (current branch)

- ✅ UMA constructs its own CPU arena (`CpuArena`) with mmap/mlock/pin logic in `unified_memory_authority.{h,cc}`, and all export/direct-write APIs use UMA-owned keepalives.
- ✅ `ChunkMeta` has been removed; StoreEngine/Replica wiring was switched to UMA snapshots and Bazel targets depend only on UMA artifacts.
- ✅ Documentation (README, architecture docs, preemptible-memory guide, AGENTS) reflects the UMA-only architecture, and smoke tests run with `bazel test --define=use_fake_cuda=true //core/store/replica:unified_memory_authority_plan_commit_test //core/store/replica:unified_memory_authority_test` plus daemon telemetry suites cover UMA allocation/plan/commit/export paths.

**Next focus:** continue monitoring UMA metrics/telemetry coverage and expand stress/regression suites as new features land.

# Test / Rollout / Backout

- Unit (Python): `uv run pytest tests/python/ -k load` to exercise loader flows that rely on UMA telemetry once refactored.
- Unit (C++): `bazel test //core/store:store_engine_test //core/store/replica:replica_tests --define=use_fake_cuda=true`.
- Integration: `bazel test //daemon:grpc_service_impl_registration_test` and relevant StoreEngine ingest tests to confirm export/pin flows.
- Backout plan: revert the UMA single-ledger change set if regression is detected; no schema or protocol changes are involved, so rollback is a standard git revert plus re-running the above test suites.

# Risks & Tracking

- **Large refactor scope**: Collapsing the layering touches many files; mitigate by landing phases sequentially and keeping UMA snapshot conversions separate from CpuArena embedding.
- **Lock ordering regressions**: Folding CpuArena inside UMA changes mutex interactions. Document and assert lock ordering in UMA tests; consider thread sanitizer runs on `bazel test //core/store/... --config=tsan`.
- **Telemetry gaps**: Switching to UMA snapshots could break dashboards if fields change. Coordinate with observability owners, update metric docs, and validate counters in `bazel test //daemon:metrics_test` (if present).
