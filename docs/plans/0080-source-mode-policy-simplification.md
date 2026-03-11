---
slug: source-mode-policy-simplification
title: Unified Source-Mode Retrieval Policy Plan
status: draft
areas: ["sdk", "daemon", "core", "proto", "docs", "tests"]
created: 2026-02-15
last_updated: 2026-03-11
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
  - daemon/service/controllers/materialization_index_source_utils.cc
  - core/store/materialization/contracts/loading_spec.h
  - core/store/materialization/control/materialize_orchestrator.cc
  - core/store/runtime/ingestion/materialization_facade.cc
  - tests/python/test_store_session_api.py
  - tests/python/api/test_config_models.py
  - tests/python/api/test_fallback_options.py
  - core/store/runtime/ingestion/materialization_facade_test.cc
  - docs/architecture/api/api-design.md
  - docs/architecture/api/materialization-flow.md
  - tensorcast/api/store/README.md
links:
  design: ../designs/0080-source-mode-policy-simplification.md
---

# Objective

Execute a hard simplification from multi-field source policy (`prefer` and allow flags in multiple layers) to one mode-based policy model with one SDK entry point and one RPC transport contract.

# Current State and Grounding

Current implementation has policy split and drift:

- SDK duplicates source intent:
  - `tensorcast/api/store/types.py` (`FallbackOptions.prefer` and allow flags)
  - `tensorcast/api/_config.py` (`GetArtifactOptions` execution options)
- materialization pipeline primarily resolves policy from fallback:
  - `tensorcast/api/store/materialization.py`
- transport duplicates source policy:
  - `proto/tensorcast/daemon/v2/store_daemon.proto` (`preference` plus `source_policy`)
  - `tensorcast/daemon_ctl.py` (request merge behavior)
- daemon policy evaluation and ordering differ across flows:
  - `daemon/service/controllers/replica_materialization_service.cc`
  - `daemon/service/controllers/target_materialization_service.cc`
  - `daemon/service/controllers/materialization_index_source_utils.cc`
- core execution has mode hints but receives legacy semantics:
  - `core/store/materialization/control/materialize_orchestrator.cc`
  - `core/store/runtime/ingestion/materialization_facade.cc`

Constraints:

- SDK must remain daemon-mediated.
- disk path injection remains forbidden for retrieval.
- rollout must keep testability across Python and C++.
- legacy callers currently can express p2p-only through `allow_disk=false`, which is not a long-term public mode.

# Phases and Milestones

- [ ] Phase 1: Contract Freeze and Mode Definition
  - [ ] Milestone 1.1: Finalize `SourceMode` semantics table (`auto`, `local_only`, `disk_first`, `disk_only`).
  - [ ] Milestone 1.2: Lock common rule that local replica short-circuit is always first.
  - [ ] Milestone 1.3: Lock wait behavior contract (`wait_for_shared_disk_ms` invalid with `local_only`).
  - [ ] Milestone 1.4: Lock deterministic compat normalization table from legacy tuple (`prefer`, `allow_*`, `prefer_disk`) to mode/action.

- [ ] Phase 2: Proto and Daemon Policy Convergence
  - [ ] Milestone 2.1: Add `SourceMode` enum and make it the only policy expression in `SourcePolicy`.
  - [ ] Milestone 2.2: Remove legacy request `preference` path from materialization RPCs.
  - [ ] Milestone 2.3: Implement one shared daemon evaluator for replica/target/mapped-target flows.
  - [ ] Milestone 2.4: Converge canonical-index source-choice logic with source mode semantics.

- [ ] Phase 3: Core Execution Semantics Convergence
  - [ ] Milestone 3.1: Align orchestrator ordering with `SourceMode`.
  - [ ] Milestone 3.2: Align `materialization_facade` ordering and strictness with orchestrator.
  - [ ] Milestone 3.3: Ensure strict-mode behavior (`local_only`, `disk_only`) has deterministic source disallow handling.

- [ ] Phase 4: SDK Surface Simplification
  - [ ] Milestone 4.1: Introduce SDK `SourceMode` and `FallbackOptions.source_mode`.
  - [x] Milestone 4.2: Keep `GetArtifactOptions` execution-only and free of source-selection semantics.
  - [ ] Milestone 4.3: Remove legacy `FallbackOptions.prefer`, `allow_p2p`, `allow_disk`, `prefer_disk`.
  - [ ] Milestone 4.4: Ensure all retrieval callsites build one mode-only source policy.

- [ ] Phase 5: Compatibility Window and Removal Gates
  - [ ] Milestone 5.1: Stage a warning window for legacy fields.
  - [ ] Milestone 5.2: Add metrics and logs for legacy field usage.
  - [ ] Milestone 5.3: Keep temporary compat adapters for legacy p2p semantics (`p2p_first_compat`, `p2p_only_compat`, internal-only).
  - [ ] Milestone 5.4: Perform hard cut removal once legacy usage reaches gate threshold.

- [ ] Phase 6: Tests and Documentation Sync
  - [ ] Milestone 6.1: Add source-mode conformance matrix tests across replica/target/mapped-target.
  - [ ] Milestone 6.2: Update Python API and session tests to mode-only APIs.
  - [ ] Milestone 6.3: Update architecture and SDK docs to remove `prefer` and allow-flag mental model.

# Task Breakdown

- [ ] Proto and generated code
  - [ ] Update `proto/tensorcast/daemon/v2/store_daemon.proto` for `SourceMode` contract.
  - [ ] Regenerate protobuf outputs via `bash tools/build_proto_python.sh`.

- [ ] Daemon
  - [ ] Refactor `daemon/service/controllers/materialization_policy_utils.h`.
  - [ ] Refactor `daemon/service/controllers/materialization_policy_utils.cc`.
  - [ ] Update `daemon/service/controllers/replica_materialization_service.cc`.
  - [ ] Update `daemon/service/controllers/target_materialization_service.cc`.
  - [ ] Update `daemon/service/controllers/materialization_index_source_utils.cc`.

- [ ] Core
  - [ ] Update `core/store/materialization/contracts/loading_spec.h`.
  - [ ] Update `core/store/materialization/control/materialize_orchestrator.cc`.
  - [ ] Update `core/store/runtime/ingestion/materialization_facade.cc`.

- [ ] SDK and client
  - [ ] Update `tensorcast/api/store/types.py`.
  - [ ] Update `tensorcast/api/_config.py`.
  - [ ] Update `tensorcast/api/store/materialization.py`.
  - [ ] Update `tensorcast/api/_materialize.py`.
  - [ ] Update `tensorcast/daemon_ctl.py`.
  - [ ] Add one centralized legacy-normalization helper and route all SDK callsites through it.
  - [ ] Add legacy mapping adapter table in code comments and tests (`prefer` to `source_mode`), including temporary `p2p_first_compat` and `p2p_only_compat` behavior.

- [ ] Tests
  - [ ] Update/add Python tests:
    - [ ] `tests/python/api/test_config_models.py`
    - [ ] `tests/python/api/test_fallback_options.py`
    - [ ] `tests/python/test_store_session_api.py`
    - [ ] `tests/python/api/test_store_runtime_auto_fallback.py`
  - [ ] Update/add C++ tests:
    - [ ] `core/store/runtime/ingestion/materialization_facade_test.cc`
    - [ ] daemon materialization validation suites
    - [ ] `daemon/service/materialize_into_target_validation_test.cc`

- [ ] Docs
  - [ ] Update `docs/architecture/api/api-design.md`.
  - [ ] Update `docs/architecture/api/materialization-flow.md`.
  - [ ] Update `tensorcast/api/store/README.md`.

# Test, Rollout, and Backout

## Test Plan

C++:

- `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`
- `bazel test //daemon:materialize_into_mapped_target_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`
- `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`

Python:

- `source .venv/bin/activate && pytest tests/python/api/test_config_models.py`
- `source .venv/bin/activate && pytest tests/python/api/test_fallback_options.py`
- `source .venv/bin/activate && pytest tests/python/test_store_session_api.py`

Conformance checks:

- [ ] mode matrix behavior is identical for replica/target/mapped-target.
- [ ] `local_only` never reaches P2P or disk.
- [ ] `disk_only` never reaches P2P.
- [ ] `wait_for_shared_disk_ms` with `local_only` is rejected.
- [ ] no remaining source semantics in `GetArtifactOptions`.
- [ ] no remaining legacy transport policy fields.
- [ ] legacy p2p-compatible inputs emit deprecation warning during compat window and are rejected after hard cut.
- [ ] one shared normalization truth-table test covers SDK and daemon behavior.

## DoD Contract Gates

- [ ] `rg "prefer\\s*:" tensorcast/api/_config.py tensorcast/api/store/types.py`
- [ ] `rg "allow_p2p|allow_disk|prefer_disk" tensorcast/api/store/types.py`
- [ ] `rg "preference\\s*=|set_preference\\(" tensorcast/daemon_ctl.py daemon/service/controllers`
- [ ] `rg "SourcePreference" proto/tensorcast/daemon/v2/store_daemon.proto`
- [ ] `rg "prefer=|allow_p2p|allow_disk" docs/architecture/api/materialization-flow.md tensorcast/api/store/README.md`
- [ ] `rg "p2p_first_compat|p2p_only_compat" tensorcast daemon core` (expect no matches after hard cut)

## Rollout

- Land proto, daemon, core, SDK, tests, and docs in one coordinated change set.
- Enable temporary warning telemetry for legacy field usage before removal.
- Promote hard cut only after conformance matrix is green and metrics gate is met:
  - `tc.store.source_mode.legacy_input_total` seven-day moving average is zero.
  - `tc.store.source_mode.compat_mode_total{compat_mode=~"p2p_.*"}` is zero for 14 consecutive days.

## Backout

- Revert as a whole if conformance semantics regress.
- Avoid partial backout that leaves SDK and daemon contracts mismatched.

# Risks and Tracking

- [ ] Risk: mode semantics diverge between replica and target flows.
  - Mitigation: one shared evaluator and conformance tests.

- [ ] Risk: old client code breaks abruptly.
  - Mitigation: bounded warning window before hard removal.

- [ ] Risk: stale documentation perpetuates old mental model.
  - Mitigation: doc-sync gate and grep-based CI checks.

- [ ] Risk: hidden p2p-only dependents surface late.
  - Mitigation: explicit compat metrics and hard-cut gate on p2p compat usage.

# Owner Checklist

- [ ] SDK owner sign-off on one-field source policy API.
- [ ] Daemon owner sign-off on single-channel transport policy.
- [ ] Core owner sign-off on execution-order invariants.
- [ ] Proto owner sign-off on policy schema simplification.
- [ ] Test owner sign-off on conformance matrix.
- [ ] Docs owner sign-off on migration wording.
