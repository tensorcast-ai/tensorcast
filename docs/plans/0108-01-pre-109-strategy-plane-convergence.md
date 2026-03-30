---
slug: pre-109-strategy-plane-convergence
title: Pre-109 Strategy Plane Convergence Plan
status: completed
areas: ["core", "daemon", "proto", "docs", "tests", "benchmarks"]
created: 2026-03-28
last_updated: 2026-03-28
related_code:
  - docs/designs/0107-retrieval-policy-plane-cleanup.md
  - docs/designs/0108-tensor-aware-materialization-strategy-plane.md
  - docs/designs/0109-batched-owner-file-collective-executor.md
  - proto/tensorcast/config/v1/daemon_config.proto
  - core/store/store_engine_options.h
  - core/store/runtime/ingestion/materialization_strategy_types.h
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/runtime/ingestion/materialization_service.cc
  - core/store/replica/replica.cc
  - core/store/replica/replica_load_controller.cc
  - core/store/replica/collective_disk_loader.cc
  - daemon/service/controllers/materialization_policy_utils.cc
  - daemon/service/controllers/replica_materialization_service.cc
  - daemon/service/controllers/target_materialization_service.cc
  - docs/internals/disk-load-strategy.md
links:
  design: ../designs/0108-tensor-aware-materialization-strategy-plane.md
  related:
    - ../plans/0107-retrieval-policy-plane-cleanup.md
    - ../plans/0109-batched-owner-file-collective-executor.md
---

# Objective

Complete the architectural and code-level convergence required before `0109`
can be implemented cleanly and considered for default routing.

This plan is the prerequisite for `0109`. Its purpose is to remove the
remaining split ownership between:

- retrieval policy versus execution topology context,
- common-runtime strategy planning versus replica-layer branch order,
- long-term executor contracts versus the current collective/local-batched
  prototypes,
- boolean rollout flags versus typed cost-model budgets and thresholds.

This plan begins only after the runtime-critical request-normalization work from
`docs/plans/0107-retrieval-policy-plane-cleanup.md` Phases 1 through 3 lands.

# Sequencing Note

- This is the only active execution plan for remaining `0108` runtime
  convergence work.
- It depends on the internal request-normalization boundary from `0107`.
- It must complete before `docs/plans/0109-batched-owner-file-collective-executor.md`
  starts.
- It does not own the later SDK/public hard cut from `0107`.
- Because the project is pre-launch, the end state should remove temporary
  convergence scaffolding rather than preserve parallel strategy owners.

# Current State & Grounding

- Ordinary `materialize_replica` still reaches disk executor choice through the
  replica layer. `core/store/replica/replica.cc` builds
  `CollectiveDiskLoadInput` / `LocalBatchedDiskLoadInput`, and
  `core/store/replica/replica_load_controller.cc` still executes the final
  `collective -> local_batched -> generic` branch ordering.
- `core/store/runtime/ingestion/materialization_facade.cc` already has the
  mapped-target strategy seam, `ResolvedSourceBinding`, and
  `ExecutionCommitReport`, but the ordinary replica path is not yet converged on
  the same planning owner.
- `core/store/runtime/ingestion/materialization_strategy_types.h` has the first
  internal contracts (`ResolvedMaterializationPlan`, `ResolvedSourceBinding`,
  `ExecutionCommitReport`) but still lacks the shared execution-environment and
  strategy-plan contracts needed for cost-model-driven `AUTO`.
- `daemon/service/controllers/materialization_policy_utils.cc` still resolves
  retrieval policy by merging legacy request `preference` with `SourcePolicy`
  instead of consuming one normalized internal request context.
- `core/store/replica/collective_disk_loader.cc` currently mixes:
  - same-host group coordination,
  - eager owner-file preload through `owned_payload`,
  - optional root whole-source preload,
  - ordinary collective execution,
  - local-batched execution,
  - mapped collective execution.
  This file is still both prototype owner and executor runtime.
- Current owner assignment is based on file index modulo world size via
  `compute_file_segments(...)` instead of weighted ownership from selected bytes.
- `daemon/service/controllers/replica_materialization_service.cc` keeps
  collective context as an explicit request field today, but the design boundary
  between retrieval policy and execution topology was only clarified in the
  updated `0107` document.
- `daemon/service/controllers/target_materialization_service.cc` still carries
  target-specific transport-hint parsing, so topology facts are not yet lowered
  through one shared request boundary before reaching core strategy code.
- `proto/tensorcast/config/v1/daemon_config.proto` and
  `core/store/store_engine_options.h` already carry
  `engine.materialization_strategy`, but current fields are mostly enable flags
  and preference enums; they do not yet expose the budgets and thresholds needed
  by `0108` and `0109`.
- Test coverage is still thin at the prototype seams. The existing focused test
  in `core/store/replica/collective_disk_loader_test.cc` only checks one
  local-batched fallback case.

# Completion Summary

This plan is complete.

The repository now has the pre-`0109` convergence seams that this document was
created to require:

- ordinary `GPU <- DISK` startup reaches a common-runtime
  `ExecutionStrategyPlan` before executor choice,
- retrieval policy and execution-topology context lower separately through one
  normalized internal request context,
- typed `engine.materialization_strategy` config carries the budgets,
  thresholds, planner cache sizing, and group-assemble timeout required by the
  follow-on owner-file executor,
- `Replica` / `ReplicaLoadController` execute a selected plan rather than
  owning `AUTO`,
- eager `owned_payload` preload and root whole-source preload remain non-default
  prototype collective scaffolding and are explicitly marked for cleanup by
  `0109`.

# Phases & Milestones

- [x] Phase 1: Freeze The New Planning Inputs
  - [x] Milestone 1.1: `ExecutionEnvironmentFacts` and
    `ExecutionStrategyPlan` landed in
    `core/store/runtime/ingestion/materialization_strategy_types.h`.
  - [x] Milestone 1.2: Retrieval policy and execution topology context are
    normalized separately in daemon helpers and common runtime inputs.
  - [x] Milestone 1.3: Typed daemon strategy config now includes budgets,
    thresholds, planner cache sizing, and group-assemble timeout fields.

- [x] Phase 2: Converge Ordinary Replica Strategy Ownership
  - [x] Milestone 2.1: Ordinary `GPU <- DISK` executor candidacy and `AUTO`
    selection now live in `MaterializationFacade`.
  - [x] Milestone 2.2: Ordinary replica startup now consumes the same shared
    `RepresentationWorkPlan` and strategy contracts as mapped-target and
    `into_target` flows.
  - [x] Milestone 2.3: `Replica` / `ReplicaLoadController` are executor runners
    and commit plumbing, not the architecture owner of `AUTO`.

- [x] Phase 3: Extract Coordinator And Prototype Boundaries
  - [x] Milestone 3.1: Same-host group timeout policy is now typed config.
  - [x] Milestone 3.2: Local-batched and collective execution are selected by
    the common-runtime plan seam instead of replica-layer branch order.
  - [x] Milestone 3.3: Eager `owned_payload` preload and root whole-source
    preload are explicitly marked as prototype-only paths with a cleanup target.

- [x] Phase 4: Unify Reporting, Diagnostics, And Gating
  - [x] Milestone 4.1: `ExecutionCommitReport` semantics and executor
    diagnostics apply to ordinary replica startup as well as mapped-target
    flows.
  - [x] Milestone 4.2: Strategy input, chosen executor, rejected candidates,
    residual bytes, and commit coverage diagnostics are emitted from the common
    seam.
  - [x] Milestone 4.3: Rollout and fallback behavior is driven by typed config
    rather than hidden branch order or ambient state.

- [x] Phase 5: Verification Baseline For 0109
  - [x] Milestone 5.1: Ordinary correctness baselines are covered for generic,
    local-batched, and current collective paths through the stabilized strategy
    seam and targeted regression tests.
  - [x] Milestone 5.2: Planner and routing evidence now exists through ordinary
    strategy-plan logs and typed timeout/budget inputs.
  - [x] Milestone 5.3: `0109` can now land as a new executor on top of a stable
    strategy boundary rather than as another prototype branch.

- [x] Phase 6: Hard-Cut Cleanup Of Convergence Scaffolding
  - [x] Milestone 6.1: Redundant ordinary strategy ownership has been removed
    from replica-layer branch order.
  - [x] Milestone 6.2: Temporary coordinator/prototype shims are isolated to
    the non-default collective prototype path.
  - [x] Milestone 6.3: No rollout-only or compatibility-only path remains the
    architecture owner of the hot ordinary disk startup path.

# Tasks

- [x] Contracts and shared types
- [x] Daemon normalization and topology context
- [x] Common-runtime strategy convergence
- [x] Replica-layer cleanup
- [x] Typed config and defaults
- [x] Observability and diagnostics
- [x] Documentation sync

# Test / Rollout / Backout

## Acceptance checks

- [x] Ordinary replica `GPU <- DISK` startup no longer relies on replica-layer
  branch order as the architecture owner of `AUTO`.
- [x] Retrieval policy and execution topology context are normalized
  separately.
- [x] Strategy-plane diagnostics exist for ordinary replica startup and
  mapped-target execution.
- [x] One normalized request boundary from `0107` feeds the strategy plane
  instead of per-controller topology parsing.
- [x] Typed config carries the budgets and thresholds needed by the follow-on
  owner-file batched executor.
- [x] The eager owner-file and whole-source preload paths remain non-default and
  are explicitly marked as prototype scaffolding.
- [x] Temporary convergence scaffolding is removed or isolated out of the
  steady hot path.

## Test plan

- [x] `bash tools/build_proto_python.sh`
- [x] `bazel test //core/common:daemon_config_io_test --test_output=errors`
- [x] `bazel test //core/store/runtime/ingestion:materialization_service_test --test_output=errors`
- [x] `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_output=errors`
- [x] `bazel test //core/store/replica:collective_disk_loader_test --test_output=errors`
- [x] `bazel test //core/store:store_engine_test --test_output=errors`
- [x] `bazel test //daemon:materialize_into_mapped_target_test --test_output=errors`
- [x] `bazel test //daemon:materialize_into_target_validation_test --test_output=errors`
- [x] `source .venv/bin/activate && pytest tests/python/test_store_view_api.py`
- [x] Shared-FS and host-local benchmark recapture is now explicitly owned by
  `0109` rollout and default-routing validation, not by this strategy-seam
  closure.

## Rollout

- [x] The convergence work is landed behind conservative typed defaults.
- [x] Local-batched remains the host-local/default stopgap while the current
  collective prototype stays non-default.
- [x] No default collective promotion is claimed for `0109` until the follow-on
  validation matrix passes.
- [x] `0109` may now start because the `0107` prerequisite and this plan's
  acceptance checks are complete.
- [x] Temporary strategy-ownership scaffolding has been deleted or isolated
  rather than left as long-term architecture.

## Backout

- [x] Backout remains a unit revert of the convergence change set if ordinary
  replica startup or mapped-target execution regress.
- [x] Backout must not restore ambient env policy or make replica-layer branch
  order the permanent strategy owner again.

# Risks & Tracking

- [x] Ordinary replica convergence no longer leaves two strategy owners in the
  runtime.
- [x] Topology context no longer leaks into retrieval policy or into
  executor-private hints.
- [x] Typed config now includes the threshold and budget family required by the
  real follow-on cost model.
- [x] Prototype collective paths are isolated and non-default so `0109` can
  replace them cleanly.
