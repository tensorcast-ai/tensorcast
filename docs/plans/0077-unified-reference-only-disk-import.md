---
slug: unified-reference-only-disk-import
title: Unified Reference-Only Disk Import Hard-Cut Plan
status: draft
areas: ["daemon", "core", "sdk"]
related_code:
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/service/controllers/materialization_disk_resolve_utils.{h,cc}
  - daemon/service/controllers/disk_artifact_service.{h,cc}
  - daemon/service/controllers/materialization_controller.{h,cc}
  - daemon/service/grpc_service_impl_rpc_delegates.cc
  - daemon/state/local_disk_import_catalog.{h,cc}
  - daemon/app/daemon_app.cc
  - daemon/state/daemon_options.h
  - daemon/state/daemon_kernel.{h,cc}
  - daemon/util/path_utils.{h,cc}
  - core/store/runtime/ingestion/materialization_service.cc
  - core/store/materialization/runtime/pipeline/verification_stage.cc
  - core/store/materialization/dataplane/verification/verification_utils.cc
  - core/store/materialization/dataplane/metadata/index_reader.cc
  - core/store/materialization/contracts/loading_spec.h
  - core/store/store_engine.{h,cc}
  - tensorcast/api/store/__init__.py
  - tensorcast/daemon_ctl.py
  - tensorcast/api/store/README.md
  - daemon/README.md
  - docs/architecture/architecture-overview.md
  - docs/architecture/p2p-transfer-strategies.md
  - docs/architecture/high-availability-design.md
  - docs/internals/model-loading.md
  - docs/architecture/api/materialization-flow.md
  - daemon/service/resolve_artifact_from_disk_test.cc
  - daemon/service/materialize_into_target_validation_test.cc
  - tests/python/test_safetensors_loading.py
links:
  design: ../designs/0077-unified-reference-only-disk-import.md
---

# Objective

Deliver a hard-cut migration to one import model only: reference-only registration with unified source authority, read-only source semantics, and zero compatibility retention.

# Current State and Grounding

Current code still has multiple legacy branches and interfaces:

- Resolve RPC + resolve stream APIs:
  - `proto/tensorcast/daemon/v2/store_daemon.proto`
  - `daemon/service/grpc_service_impl_rpc_delegates.cc`
- Sidecar/backfill/source-write branches:
  - `daemon/service/controllers/materialization_disk_resolve_utils.cc`
- Local import catalog still modeled as separate owner:
  - `daemon/service/controllers/disk_artifact_service.cc`
  - `daemon/state/local_disk_import_catalog.{h,cc}`
- Core verification/data-plane can still write descriptor/index/verification files:
  - `core/store/runtime/ingestion/materialization_service.cc`
- SDK/daemon client still consume resolve stream contracts:
  - `tensorcast/api/store/__init__.py`
  - `tensorcast/daemon_ctl.py`
- Docs across architecture/api/internals still reference resolve/disk-hint semantics:
  - `docs/architecture/architecture-overview.md`
  - `docs/architecture/p2p-transfer-strategies.md`
  - `docs/architecture/high-availability-design.md`
  - `docs/internals/model-loading.md`
  - `docs/architecture/api/materialization-flow.md`

This plan intentionally removes these branches and interfaces in one coordinated change set.

# Execution Principles (Extend + Refactor)

- No compatibility shims, no dual-write, no fallback mode toggles.
- One source authority owner only (`ArtifactSourceRegistry`).
- One canonicalization authority only (core).
- One import RPC family only.
- One progress/error model only.
- Any temporary branch created during refactor must be removed before merge.

# Phases and Milestones

- [ ] Phase 1: Source Authority Unification
  - [ ] Milestone 1.1: Refactor `LocalDiskImportCatalog` into daemon-wide `ArtifactSourceRegistry` shape (single owner for local import + managed disk bindings).
  - [ ] Milestone 1.2: Materialization source resolution reads only from typed source bindings, not path strings.
  - [ ] Milestone 1.3: Remove any parallel source-owner state stores.

- [ ] Phase 2: Core Authority + Read-Only Source Enforcement
  - [ ] Milestone 2.1: Reuse core canonical/index/hash builders exclusively; remove duplicate daemon-side builder logic.
  - [ ] Milestone 2.2: Add `source_mutation_policy=READ_ONLY` in import/materialization reference flows.
  - [ ] Milestone 2.3: Disable descriptor/index/verification source writes for reference-only imports.

- [ ] Phase 3: RPC/SDK Hard Cut (No Compatibility)
  - [ ] Milestone 3.1: Add `ImportArtifactFromPath` + `ImportArtifactFromPathStream` proto surfaces.
  - [ ] Milestone 3.2: Add machine-readable stream `error_code` and fixed phase set.
  - [ ] Milestone 3.3: Remove `ResolveArtifactFromDisk*` proto RPCs and all daemon delegate/controller usage.
  - [ ] Milestone 3.4: Remove SDK and daemon client resolve APIs; `from_disk` migrates to import APIs only.

- [ ] Phase 4: Runtime Root Consistency
  - [ ] Milestone 4.1: Place import metadata under daemon runtime root (`$TENSORCAST_HOME/hosts/<host_id>/runtime/daemons/<daemon_id>/import`).
  - [ ] Milestone 4.2: Startup probe and bootstrap for registry DB under unified root.
  - [ ] Milestone 4.3: Startup hard-fail with explicit `IMPORT_ROOT_UNAVAILABLE` if root cannot be initialized.

- [ ] Phase 5: Hard Deletion Cleanup
  - [ ] Milestone 5.1: Delete sidecar mirror/backfill helpers and sidecar-specific fields.
  - [ ] Milestone 5.2: Delete resolve-only progress enums/outcome metrics/legacy env knobs.
  - [ ] Milestone 5.3: Delete redundant response/interface fields that expose resolved disk paths.
  - [ ] Milestone 5.4: Delete tests that assert removed compatibility behavior; replace with new contract tests.

- [ ] Phase 6: Documentation Global Consistency
  - [ ] Milestone 6.1: Update daemon/sdk docs to reflect import-only contract and removed resolve API.
  - [ ] Milestone 6.2: Update architecture/api/internals docs to remove `hints.disk_path`/resolve semantics.
  - [ ] Milestone 6.3: Ensure no remaining docs describe sidecar/backfill as runtime behavior.

- [x] Phase 7: 0073 Retirement
  - [x] Milestone 7.1: Delete `docs/designs/0073-standalone-daemon-disk-hints.md`.
  - [x] Milestone 7.2: Delete `docs/plans/0073-standalone-daemon-disk-hints.md`.
  - [x] Milestone 7.3: Remove all references to 0073 and ensure design lineage converges on 0077 only.

# Task Checklist

- [ ] Source authority refactor
  - [ ] Define typed `ArtifactSourceRegistry` binding model (`managed`, `local_import`) with one primary key (`artifact_id`).
  - [ ] Refactor `disk_artifact_service` and materialization source resolution to use registry only.
  - [ ] Remove residual parallel source lookup helpers after cutover.

- [ ] Core authority and read-only policy
  - [ ] Route safetensors canonical/index/hash generation through existing core APIs (`index_reader`/`safetensors_util`) only.
  - [ ] Introduce read-only mutation policy in ingestion request/hints path.
  - [ ] Ensure read-only mode blocks:
    - [ ] `write_descriptor_if_absent`
    - [ ] index backfill writes
    - [ ] verification metadata writes for import reference paths

- [ ] Proto + RPC hard cut
  - [ ] Add import RPC/messages/events and machine-readable `error_code`.
  - [ ] Remove `ResolveArtifactFromDisk` and `ResolveArtifactFromDiskStream` from proto/service/delegates/controllers.
  - [ ] Regenerate protobuf outputs (`bash tools/build_proto_python.sh`).
  - [ ] Remove resolve-only daemon client methods in `tensorcast/daemon_ctl.py`.

- [ ] SDK cutover
  - [ ] Update `Store.from_disk` to use import RPC stream/unary only.
  - [ ] Remove resolve-specific stream parsing helpers and fallback assumptions.
  - [ ] Preserve progress bar support using new stream phase/event schema.

- [ ] Runtime root unification
  - [ ] Create import metadata root under daemon runtime root.
  - [ ] Add startup diagnostics for resolved root.
  - [ ] Add startup failure coverage when root probe/bootstrap fails.

- [ ] Hard deletion cleanup
  - [ ] Remove sidecar fields from catalog entries and spans.
  - [ ] Remove resolve-phase enum usage and resolve outcome counters that no longer apply.
  - [ ] Remove redundant response fields that leak resolved disk path semantics.
  - [ ] Remove compatibility env knobs used only by resolve legacy flow.

- [ ] Global documentation consistency
  - [ ] Update:
    - [ ] `daemon/README.md`
    - [ ] `tensorcast/api/store/README.md`
    - [ ] `docs/architecture/architecture-overview.md`
    - [ ] `docs/architecture/p2p-transfer-strategies.md`
    - [ ] `docs/architecture/high-availability-design.md`
    - [ ] `docs/internals/model-loading.md`
    - [ ] `docs/architecture/api/materialization-flow.md`
  - [ ] Verify no docs still mention resolve APIs or sidecar/backfill runtime behavior.

- [x] 0073 retirement
  - [x] Delete design and plan files for 0073.
  - [x] Remove references to 0073 in remaining docs.

# Test, Rollout, and Backout

## Test Plan

- C++ daemon:
  - `bazel test //daemon:resolve_artifact_from_disk_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- Core ingestion:
  - `bazel test //core/store/runtime/ingestion:materialization_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- Python:
  - `source .venv/bin/activate && pytest tests/python/test_safetensors_loading.py`
  - `source .venv/bin/activate && pytest tests/python/test_store_session_api.py`

Required additional tests:

- import succeeds for safetensors-only directories without source metadata files.
- import/materialization do not write descriptor/index/verification files to source.
- source mutation after import fails with `FAILED_PRECONDITION` and `SOURCE_MUTATED`.
- non-loopback import is denied.
- startup fails when import root bootstrap fails.
- stream events are monotonic and have exactly one terminal event.
- stream error terminal includes machine-readable `error_code`.
- no RPC/service/client symbol named `ResolveArtifactFromDisk*` remains.

## Rollout

- Land as one coordinated breaking change (proto + daemon + core + sdk + tests + docs).
- No compatibility shim phase.
- Merge is blocked until hard-deletion checklist is complete.

## Backout

- Full revert only.
- No partial backout because this is a semantic replacement and interface hard cut.

# Risks and Tracking

- [ ] Risk: Intermediate commits temporarily leave dual behavior.
  - Mitigation: keep hard deletion in same merge window; do not ship partial states.
- [ ] Risk: Read-only enforcement causes regressions in tests that relied on backfill.
  - Mitigation: rewrite tests to assert new contract, remove old behavior assertions.
- [ ] Risk: Large doc surface drifts from code.
  - Mitigation: treat doc consistency checklist as release gate, not post-task cleanup.

# Progress Log

| Date | Stage | Status | Notes |
| --- | --- | --- | --- |
| 2026-02-14 | Plan authored | Completed | Initial hard-cut plan created from design 0077 with explicit compatibility/redundancy cleanup and 0073 retirement scope. |
| 2026-02-14 | 0073 retirement | Completed | Deleted `docs/designs/0073-standalone-daemon-disk-hints.md` and `docs/plans/0073-standalone-daemon-disk-hints.md`; cleaned direct references. |
