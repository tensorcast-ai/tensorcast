---
slug: selection-first-artifact-retrieval
title: Selection-First Artifact Retrieval And Materialization Hard-Cut Plan
status: completed
areas: ["sdk", "daemon", "core", "proto"]
created: 2026-02-14
last_updated: 2026-02-15
related_code:
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/view_composer.py
  - tensorcast/api/store/materialization.py
  - tensorcast/api/_materialize.py
  - tensorcast/api/plan/plan.py
  - tensorcast/api/store/inplace_slot.py
  - tensorcast/api/store/binding.py
  - tensorcast/daemon_ctl.py
  - tensorcast/node_agent/executor.py
  - tensorcast/common/selection_identity.py
  - core/common/selection_identity.cc
  - core/common/selection_identity_test.cc
  - proto/tensorcast/common/v1/common.proto
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/service/controllers/replica_materialization_service.cc
  - daemon/service/controllers/materialization_payload_utils.cc
  - daemon/service/controllers/target_materialization_service.cc
  - daemon/service/controllers/materialization_target_plan_utils.cc
  - daemon/service/controllers/materialization_controller.h
  - daemon/service/controllers/materialization_controller.cc
  - daemon/service/controllers/target_publish_service.cc
  - daemon/service/grpc_service_impl.h
  - daemon/service/grpc_service_impl_rpc_delegates.cc
  - daemon/state/retention_registry.cc
  - docs/architecture/README.md
  - docs/architecture/architecture-overview.md
  - docs/architecture/api/materialization-flow.md
  - docs/architecture/p2p-transfer-strategies.md
  - tests/python/api/test_artifact_handle.py
  - tests/python/api/test_artifact_tensor_subset.py
  - tests/python/api/test_materialization_pipeline_v2.py
  - tests/python/api/test_prefetch_operation.py
  - tests/python/test_store_view_api.py
  - tests/python/test_dense_piece_assembly_sealing_acceptance.py
  - tests/python/test_binding.py
  - tests/python/cli/test_cli_surface.py
  - tests/python/daemon/test_cpp_daemon_lifecycle.py
  - tests/python/utils/daemon.py
  - daemon/service/materialize_into_target_validation_test.cc
  - daemon/service/grpc_service_impl_no_lease_materialize_test.cc
  - tests/python/test_selection_identity_vectors.py
links:
  design: ../designs/0078-selection-first-artifact-retrieval.md
---

# Objective

Deliver a hard-cut migration to a single system-wide selection contract:

- one selector and identity source: `ArtifactSelection`
- one request identity source: `selection.artifact_id`
- one selection execution path across replica, target, and mapped-target materialization
- key resolution remains control-path (`ResolveKeyMapping`) and is removed from data-path materialization RPCs
- responses carry resolved selection identity for cache hydration, observability, publish, and retention

No compatibility shim and no dual-line behavior are retained.

# Current State and Grounding

Current code still has split selector and identity channels:

- SDK retrieval still exposes duplicated call-site filtering (`names`) and mixed view/subset shape:
  - `tensorcast/api/store/artifact.py`
- materialization pipeline/client plumbing still carries parallel selector fields (`tensor_names`, `view_subset_hash`, `view_id/view_spec`):
  - `tensorcast/api/store/materialization.py`
  - `tensorcast/api/_materialize.py`
  - `tensorcast/daemon_ctl.py`
- materialization proto still has fragmented request-local selectors and by-key data-path RPC:
  - `proto/tensorcast/daemon/v2/store_daemon.proto`
- daemon payload shaping and target planning still rely on request-local selection fields:
  - `daemon/service/controllers/materialization_payload_utils.cc`
  - `daemon/service/controllers/replica_materialization_service.cc`
  - `daemon/service/controllers/materialization_target_plan_utils.cc`
  - `daemon/service/controllers/target_materialization_service.cc`
- plan/node-agent still reconstruct selection by re-entering artifact APIs (`artifact.view(names=...)` path):
  - `tensorcast/api/plan/plan.py`
  - `tensorcast/node_agent/executor.py`

Selection primitives already exist and should be reused directly:

- selector envelope: `tensorcast.common.v1.ArtifactSelection` in `proto/tensorcast/common/v1/common.proto`
- hash parity utilities:
  - `tensorcast/common/selection_identity.py`
  - `core/common/selection_identity.cc`
- selection usage in binding and retention paths:
  - `tensorcast/api/store/inplace_slot.py`
  - `daemon/service/controllers/target_publish_service.cc`
  - `daemon/state/retention_registry.cc`

Documentation also lags target architecture:

- architecture docs still recommend key-based materialization RPCs:
  - `docs/architecture/README.md`
  - `docs/architecture/architecture-overview.md`
  - `docs/architecture/api/materialization-flow.md`
  - `docs/architecture/p2p-transfer-strategies.md`

## Baseline Constraints

- pre-launch branch; compatibility retention is intentionally out of scope
- no feature flags or environment toggles for legacy behavior
- stale APIs and stale docs must be removed in the same merge window

# Execution Principles

- Hard cut only. Remove old signatures and RPCs instead of deprecating.
- Compile-time and signature-level failures are preferred for stale call sites.
- Selection hashing behavior must remain identical between Python and C++.
- Controllers must validate selection once, then propagate one resolved selection plan.
- Data path is artifact-id only; key indirection is control-path only.

# Phases and Milestones

- [x] Phase 1: Contract Freeze and Selector Scaffolding
  - [x] Milestone 1.1: Finalize canonical `ArtifactSelection` validation matrix (required fields and mismatch semantics).
  - [x] Milestone 1.2: Define daemon-internal resolved selector structure (for example `ResolvedSelectionPlan`) shared by replica/target/mapped-target paths.
  - [x] Milestone 1.3: Add cross-language golden vectors for subset hash and selection hash normalization.

- [x] Phase 2: SDK Public API Hard Cut
  - [x] Milestone 2.1: Remove `names` from `Artifact.view`.
  - [x] Milestone 2.2: Remove `names` from `Artifact.tensor_dict` and `Artifact.tensor_dict_async`.
  - [x] Milestone 2.3: Keep `Artifact.tensor(name, ...)` via single-name subset derivation only.
  - [x] Milestone 2.4: Update all SDK tests and call sites to `artifact.subset(...).tensor_dict(...)`.

- [x] Phase 3: SDK Selection-Native Pipeline and Key Control-Path Cut
  - [x] Milestone 3.1: Add one internal selection builder on `Artifact` and route retrieval/materialization through it.
  - [x] Milestone 3.2: Refactor `MaterializationPipeline` and `_materialize.py` to submit selection-only materialization requests.
  - [x] Milestone 3.3: Remove `materialize_by_key_v2` from `tensorcast/daemon_ctl.py`; resolve key before materialization.
  - [x] Milestone 3.4: Ensure plan and binding/inplace flows reuse the same selector builder contract.

- [x] Phase 4: Proto and RPC Hard Cut
  - [x] Milestone 4.1: `MaterializeReplicaRequest` uses `ArtifactSelection selection` and removes legacy selector/identity fields.
  - [x] Milestone 4.2: `MaterializeIntoTargetRequest` uses `ArtifactSelection selection` and removes `artifact_ref` and legacy selector fields.
  - [x] Milestone 4.3: `MaterializeIntoMappedTargetRequest` adopts the same selector contract.
  - [x] Milestone 4.4: Add `resolved_selection` to materialization responses.
  - [x] Milestone 4.5: Remove `MaterializeByKey` RPC/messages and regenerate outputs (`bash tools/build_proto_python.sh`).

- [x] Phase 5: Daemon Controller and Execution Convergence
  - [x] Milestone 5.1: Parse/validate selection once in controllers; reject invalid/mismatched selection with deterministic status codes.
  - [x] Milestone 5.2: Recompute and verify `view_subset_hash`, `selection_hash`, and `logical_layout_hash` in daemon.
  - [x] Milestone 5.3: Build payload descriptors from resolved selection only.
  - [x] Milestone 5.4: Unify replica/target/mapped-target selector handling through one resolved selection plan.
  - [x] Milestone 5.5: Remove by-key controller and gRPC delegate surfaces.

- [x] Phase 6: Plan and Node-Agent Selection-Native Convergence
  - [x] Milestone 6.1: Stop reconstructing selection through `Artifact.view(..., names=...)` in node-agent.
  - [x] Milestone 6.2: Execute prefetch/transform actions directly from incoming `ArtifactSelection`.
  - [x] Milestone 6.3: Keep idempotency fingerprint derivation anchored on `logical_layout_hash` plus `selection_hash`.

- [x] Phase 7: Publish and Retention Identity Convergence
  - [x] Milestone 7.1: Target publish scope uses resolved selection from materialization response path.
  - [x] Milestone 7.2: Retention acquisition consumes validated selection identity contract (`artifact_id`, `logical_layout_hash`, `selection_hash`) without legacy selector fallback.
  - [x] Milestone 7.3: Selection order/membership semantics remain consistent with target publish validation.

- [x] Phase 8: Tests, CI Gates, and Doc Sync
  - [x] Milestone 8.1: Add/refresh Python and C++ tests for selector validation and response identity.
  - [x] Milestone 8.2: Add hard grep gates for removed RPCs/signatures and removed request-local selectors.
  - [x] Milestone 8.3: Update architecture docs that still recommend `MaterializeByKey` data path.
  - [x] Milestone 8.4: Ensure examples and docs use target-state APIs only.

# Task Checklist

- [x] SDK API and selector builder
  - [x] `tensorcast/api/store/artifact.py`
  - [x] `tensorcast/api/store/view_composer.py`
  - [x] `tensorcast/api/store/materialization.py`
  - [x] `tensorcast/api/_materialize.py`
  - [x] `tensorcast/daemon_ctl.py`

- [x] Plan and node-agent
  - [x] `tensorcast/api/plan/plan.py`
  - [x] `tensorcast/node_agent/executor.py`

- [x] Proto and generated outputs
  - [x] `proto/tensorcast/daemon/v2/store_daemon.proto`
  - [x] remove `MaterializeByKey` request/response/service RPC
  - [x] `bash tools/build_proto_python.sh`

- [x] Daemon controller and services
  - [x] `daemon/service/controllers/replica_materialization_service.cc`
  - [x] `daemon/service/controllers/target_materialization_service.cc`
  - [x] `daemon/service/controllers/materialization_target_plan_utils.cc`
  - [x] `daemon/service/controllers/materialization_payload_utils.cc`
  - [x] `daemon/service/controllers/materialization_controller.h`
  - [x] `daemon/service/controllers/materialization_controller.cc`
  - [x] `daemon/service/grpc_service_impl.h`
  - [x] `daemon/service/grpc_service_impl_rpc_delegates.cc`

- [x] Publish and retention consistency
  - [x] `daemon/service/controllers/target_publish_service.cc`
  - [x] `daemon/state/retention_registry.cc`

- [x] Tests and contract gates
  - [x] Python tests under `tests/python/api/` and `tests/python/test_store_view_api.py`
  - [x] `tests/python/test_binding.py`
  - [x] daemon/core Bazel tests for target validation and no-lease materialization

- [x] Docs sync
  - [x] `docs/architecture/README.md`
  - [x] `docs/architecture/architecture-overview.md`
  - [x] `docs/architecture/api/materialization-flow.md`
  - [x] `docs/architecture/p2p-transfer-strategies.md`

# Test, Rollout, and Backout

## Test Plan

- C++ (Bazel):
  - `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - `bazel test //daemon:materialize_into_mapped_target_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - `bazel test //daemon:grpc_service_impl_no_lease_materialize_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - `bazel test //daemon:import_artifact_from_path_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - `bazel test //core/store/runtime/ingestion:materialization_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - `bazel test //core/store:store_engine_view_test --test_env=TENSORCAST_CUDA_BACKEND=fake`

- Python:
  - `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py`
  - `source .venv/bin/activate && pytest tests/python/api/test_artifact_tensor_subset.py`
  - `source .venv/bin/activate && pytest tests/python/api/test_materialization_pipeline_v2.py`
  - `source .venv/bin/activate && pytest tests/python/test_store_view_api.py`
  - `source .venv/bin/activate && pytest tests/python/test_store_session_api.py`
  - `source .venv/bin/activate && pytest tests/python/test_binding.py`

Required behavior checks:

- [x] `artifact.view(..., names=...)` is rejected by signature/API shape.
- [x] `artifact.subset(names).tensor_dict(...)` works and retrieval no longer accepts call-site `names`.
- [x] subset-only selection keeps `view_id == ""` and stable subset hash.
- [x] all key-based workflows resolve mapping before materialization.
- [x] materialization responses carry `resolved_selection` matching daemon-resolved selector.
- [x] target publish scope and retention handle flows consume consistent selection identity.
- [x] mapped-target and replica paths follow the same selector validation semantics.

## DoD Contract Gates

- [x] removed key materialization symbols in code paths:
  - `rg "MaterializeByKey" proto tensorcast daemon`
- [x] removed by-key SDK API surface:
  - `rg "materialize_by_key_v2|materialize_by_key\(" tensorcast/daemon_ctl.py`
- [x] removed retrieval `names` API parameters:
  - `rg "def (view|tensor_dict|tensor_dict_async)\([^\n]*names" tensorcast/api/store/artifact.py`
- [x] materialization request schema no longer carries duplicated selector fields:
  - `rg "artifact_ref|oneof view_identity" proto/tensorcast/daemon/v2/store_daemon.proto`
  - `rg "reserved .*tensor_names|reserved .*view_subset_hash" proto/tensorcast/daemon/v2/store_daemon.proto`
- [x] architecture docs no longer recommend key-based materialization data path:
  - `rg "MaterializeByKey" docs/architecture/README.md docs/architecture/architecture-overview.md docs/architecture/api/materialization-flow.md docs/architecture/p2p-transfer-strategies.md`

## Execution Status (2026-02-14)

- Completed hard-cut migration in SDK/proto/daemon/node-agent paths, including removal of `MaterializeByKey` and request-local selector fields from materialization RPCs.
- Added cross-language parity vectors for selection identity:
  - `tests/python/test_selection_identity_vectors.py`
  - `core/common/selection_identity_test.cc`
- Verified representative test and contract-gate commands:
  - `bash tools/build_proto_python.sh` (success)
  - `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors` (pass)
  - `bazel test //core/common:selection_identity_test --test_output=errors` (pass)

## Execution Status (2026-02-15)

- Post-cut stabilization completed from root-cause analysis (no compatibility rollback):
  - Fixed full-selection target publishability scope construction in `daemon/service/controllers/materialization_target_plan_utils.cc` so canonical full selection does not carry subset-only tensor-name markers.
  - Updated Python prefetch test fixture to match selection-first client contract (`get_artifact_index_by_id`) in `tests/python/api/test_prefetch_operation.py`.
  - Updated daemon test launcher sizing in `tests/python/utils/daemon.py` to derive engine pinned-pool capacity from streaming chunks and visible GPU count, matching current startup preflight requirements.
  - Migrated remaining legacy `MaterializeReplicaRequest(artifact_id, view_id, ...)` usage to `selection=ArtifactSelection(...)` in `tests/python/test_dense_piece_assembly_sealing_acceptance.py`.
  - Hardened CLI surface test isolation in `tests/python/cli/test_cli_surface.py` by mocking `runtime.status()` to avoid external local-session interference.
  - Hardened daemon lifecycle integration test in `tests/python/daemon/test_cpp_daemon_lifecycle.py` by using a free P2P port and explicit `127.0.0.1` endpoints.
- Full verification status:
  - `source .venv/bin/activate && pytest tests/python` -> `463 passed, 1 warning`.
  - `bazel build //daemon:tensorcast_daemon` -> success.
  - `bazel test //core/... --verbose_failures --test_tag_filters="-stress,-rdma,-multi_gpu" --test_output=errors --test_summary=detailed` -> all selected tests passed.
  - `source .venv/bin/activate && pytest tests/python/api/test_plan_spec.py tests/python/test_deferred_loader.py` (8 passed)
  - `source .venv/bin/activate && pytest tests/python/test_selection_identity_vectors.py` (3 passed)
  - `rg "MaterializeByKey" proto tensorcast daemon` (no matches)
  - `rg "materialize_by_key_v2|materialize_by_key\(" tensorcast/daemon_ctl.py` (no matches)
  - `rg "def (view|tensor_dict|tensor_dict_async)\([^\n]*names" tensorcast/api/store/artifact.py` (no matches)
  - `rg "artifact_ref|oneof view_identity" proto/tensorcast/daemon/v2/store_daemon.proto` (no matches)
  - `rg "MaterializeByKey" docs/architecture/README.md docs/architecture/architecture-overview.md docs/architecture/api/materialization-flow.md docs/architecture/p2p-transfer-strategies.md` (no matches)

## Rollout

- single coordinated merge for sdk + proto + daemon + core + tests + docs
- no staged dual-path rollout
- merge gate requires all DoD contract gates and required behavior checks

## Backout

- full revert only
- partial revert is disallowed because API and proto contracts are hard-cut

# Risks and Tracking

- [x] Risk: cross-layer hard cut causes broad compile/test failures.
  - Mitigation: phase-order execution (SDK first, proto next, daemon after), then full suite gate.

- [x] Risk: hidden stale call sites still use removed signatures or by-key RPC.
  - Mitigation: hard grep gates in CI and no-compat runtime behavior.

- [x] Risk: selection identity drift between Python and C++ during refactor.
  - Mitigation: shared golden vectors plus parity tests for subset and selection hashes.

- [x] Risk: mapped-target path diverges from replica/target selector semantics.
  - Mitigation: unified resolved selection plan and mirrored validation tests.

- [x] Risk: docs drift keeps old materialization guidance and confuses follow-on changes.
  - Mitigation: docs sync checklist is merge-blocking for this hard cut.
