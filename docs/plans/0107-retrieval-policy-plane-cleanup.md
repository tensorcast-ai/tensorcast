---
slug: retrieval-policy-plane-cleanup
title: Retrieval Policy Plane Cleanup Plan
status: draft
areas: ["sdk", "daemon", "core", "proto", "docs", "tests"]
created: 2026-03-18
last_updated: 2026-03-18
related_code:
  - tensorcast/api/store/types.py
  - tensorcast/api/_config.py
  - tensorcast/api/store/materialization.py
  - tensorcast/api/store/artifact.py
  - tensorcast/api/_materialize.py
  - tensorcast/daemon_ctl.py
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/service/controllers/materialization_policy_utils.h
  - daemon/service/controllers/materialization_policy_utils.cc
  - daemon/service/controllers/replica_materialization_service.cc
  - daemon/service/controllers/target_materialization_service.cc
  - daemon/service/controllers/owned_binding_service.cc
  - core/store/materialization/contracts/loading_spec.h
  - core/store/materialization/control/materialize_orchestrator.cc
  - core/store/runtime/ingestion/materialization_facade.cc
  - tests/python/api/test_fallback_options.py
  - tests/python/api/test_materialization_pipeline_v2.py
  - tests/python/api/test_artifact_handle.py
  - tests/python/test_binding.py
  - daemon/service/materialize_into_target_validation_test.cc
links:
  design: ../designs/0107-retrieval-policy-plane-cleanup.md
---

# Objective

Replace artifact-handle-owned fallback semantics with an execution-scoped
retrieval-policy model, while converging SDK, daemon, core, and owned-binding
flows on one structured transport contract.

# Current State & Grounding

Current implementation is split across the wrong boundaries:

- handle-owned fallback state lives in:
  - `tensorcast/api/store/types.py`
  - `tensorcast/api/store/artifact.py`
- SDK materialization lowers fallback into request `preference` plus
  `source_policy` in multiple call sites:
  - `tensorcast/api/store/materialization.py`
  - `tensorcast/api/store/artifact.py`
  - `tensorcast/api/_materialize.py`
  - `tensorcast/daemon_ctl.py`
- daemon normalizes the same decision again:
  - `daemon/service/controllers/materialization_policy_utils.cc`
  - `daemon/service/controllers/replica_materialization_service.cc`
  - `daemon/service/controllers/target_materialization_service.cc`
  - `daemon/service/controllers/owned_binding_service.cc`
- core execution already consumes structured hints:
  - `core/store/materialization/control/materialize_orchestrator.cc`
  - `core/store/runtime/ingestion/materialization_facade.cc`

Constraints:

- SDK must remain daemon-mediated.
- `ArtifactSelection` remains the only retrieval selection contract.
- `StorePolicy` remains the only durability/placement contract.
- rollout `source_mode` from `0104` must not be reused for retrieval naming.
- migration must include owned-binding flows, not only materialization RPCs.

# Phases & Milestones

- [ ] Phase 1: Freeze The Retrieval Policy Plane
  - [ ] Milestone 1.1: Finalize the plane boundary between `ArtifactSelection`,
    `StorePolicy`, retrieval policy, and rollout strategy.
  - [ ] Milestone 1.2: Finalize canonical structured retrieval policy
    (`preference`, `allow_p2p`, `allow_disk`).
  - [ ] Milestone 1.3: Finalize public preset sugar and canonical lowering
    table.

- [ ] Phase 2: Single Transport Channel
  - [ ] Milestone 2.1: Remove request-level `preference` from materialization
    and owned-binding RPCs.
  - [ ] Milestone 2.2: Make `source_policy` the only retrieval transport field.
  - [ ] Milestone 2.3: Keep proto `SourcePolicy` structured; do not introduce a
    mode-only wire enum.

- [ ] Phase 3: Daemon And Core Convergence
  - [ ] Milestone 3.1: Share one daemon normalization helper across replica,
    target, mapped-target, and owned-binding flows.
  - [ ] Milestone 3.2: Align orchestrator and ingestion facade behavior to the
    same structured policy contract.
  - [ ] Milestone 3.3: Keep wait-for-shared-disk retry behavior derived from the
    structured policy, not from a second public state machine.

- [ ] Phase 4: SDK Execution-Scope Migration
  - [ ] Milestone 4.1: Introduce execution-scoped retrieval policy on
    `GetArtifactOptions` or equivalent.
  - [ ] Milestone 4.2: Move `replica_uuid` and disk verification behavior out of
    `FallbackOptions`.
  - [ ] Milestone 4.3: Add one SDK compatibility adapter from legacy
    `FallbackOptions` and string shortcuts to canonical retrieval policy.

- [ ] Phase 5: Artifact Handle Cleanup
  - [ ] Milestone 5.1: Stop serializing retrieval policy in `Artifact`.
  - [ ] Milestone 5.2: Remove `Artifact.with_fallback(...)`.
  - [ ] Milestone 5.3: Remove handle-owned fallback references from public docs
    and tests.

- [ ] Phase 6: Hard Cut
  - [ ] Milestone 6.1: Remove `FallbackOptions`.
  - [ ] Milestone 6.2: Remove fallback string shortcuts.
  - [ ] Milestone 6.3: Verify no retrieval public surface uses `source_mode`.

# Task Breakdown

- [ ] Proto
  - [ ] Update `proto/tensorcast/daemon/v2/store_daemon.proto`.
  - [ ] Regenerate Python protobuf outputs via `bash tools/build_proto_python.sh`.

- [ ] Daemon
  - [ ] Update `daemon/service/controllers/materialization_policy_utils.h`.
  - [ ] Update `daemon/service/controllers/materialization_policy_utils.cc`.
  - [ ] Update `daemon/service/controllers/replica_materialization_service.cc`.
  - [ ] Update `daemon/service/controllers/target_materialization_service.cc`.
  - [ ] Update `daemon/service/controllers/owned_binding_service.cc`.

- [ ] Core
  - [ ] Update `core/store/materialization/contracts/loading_spec.h`.
  - [ ] Update `core/store/materialization/control/materialize_orchestrator.cc`.
  - [ ] Update `core/store/runtime/ingestion/materialization_facade.cc`.

- [ ] SDK
  - [ ] Update `tensorcast/api/_config.py`.
  - [ ] Update `tensorcast/api/store/types.py`.
  - [ ] Update `tensorcast/api/store/materialization.py`.
  - [ ] Update `tensorcast/api/store/artifact.py`.
  - [ ] Update `tensorcast/api/_materialize.py`.
  - [ ] Update `tensorcast/daemon_ctl.py`.
  - [ ] Add one canonical retrieval-policy normalization helper and route all
    retrieval call sites through it.

- [ ] Tests
  - [ ] Update `tests/python/api/test_fallback_options.py`.
  - [ ] Update `tests/python/api/test_materialization_pipeline_v2.py`.
  - [ ] Update `tests/python/api/test_artifact_handle.py`.
  - [ ] Update binding-related Python tests.
  - [ ] Update daemon validation tests.
  - [ ] Update core materialization tests.

- [ ] Docs
  - [ ] Update `docs/architecture/api/api-design.md`.
  - [ ] Update `docs/architecture/api/materialization-flow.md`.
  - [ ] Update `tensorcast/api/store/README.md`.

# Test Plan

Python:

- `source .venv/bin/activate && pytest tests/python/api/test_fallback_options.py`
- `source .venv/bin/activate && pytest tests/python/api/test_materialization_pipeline_v2.py`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pytest tests/python/test_binding.py`

C++:

- `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`
- `bazel test //daemon:materialize_into_mapped_target_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`
- `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`

Conformance checks:

- [ ] no materialization or owned-binding request uses top-level `preference`
- [ ] one shared daemon normalization path is used across all retrieval-related
  RPCs
- [ ] orchestrator and facade both consume the same structured policy contract
- [ ] artifact serialization no longer persists retrieval policy
- [ ] no retrieval public naming uses `source_mode`

# DoD Checks

- [ ] `rg "preference\\s*=|set_preference\\(" tensorcast/daemon_ctl.py daemon/service/controllers`
- [ ] `rg "with_fallback|ArtifactSerializedFallback|fallback" tensorcast/api/store/artifact.py`
- [ ] `rg "class FallbackOptions|FallbackPreference" tensorcast/api/store/types.py`
- [ ] `rg "source_mode" tensorcast/api/store tensorcast/api/_config.py docs/architecture/api tensorcast/api/store/README.md`
- [ ] `rg "SourcePolicy source_policy" proto/tensorcast/daemon/v2/store_daemon.proto`

# Rollout And Backout

Rollout:

- land proto, daemon, core, SDK, tests, and docs together
- keep one compatibility adapter for legacy fallback callers during migration
- remove the adapter only after all call sites and docs switch to the new
  execution-scoped retrieval contract

Backout:

- revert the change set as a whole if SDK and daemon contracts drift
- do not partially restore handle-owned fallback semantics

# Risks And Tracking

- [ ] Risk: transport and core semantics diverge during partial migration.
  - Mitigation: stage proto/daemon/core before hard-cut SDK cleanup.

- [ ] Risk: binding flows are missed and keep a second retrieval-policy model.
  - Mitigation: include `CreateOwnedBinding` and `RefillOwnedBinding` in the
    core checklist and tests.

- [ ] Risk: compatibility logic grows instead of shrinking.
  - Mitigation: keep one canonical helper and delete old adapters aggressively
    after migration.

# Owner Checklist

- [ ] SDK owner sign-off on execution-scoped retrieval policy.
- [ ] Daemon owner sign-off on single-channel transport policy.
- [ ] Core owner sign-off on structured canonical hints.
- [ ] Binding owner sign-off on owned-binding convergence.
- [ ] Docs owner sign-off on terminology separation from rollout `source_mode`.
