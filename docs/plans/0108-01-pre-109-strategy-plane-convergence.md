---
slug: pre-109-strategy-plane-convergence
title: Pre-109 Strategy Plane Convergence Plan
status: proposed
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

# Phases & Milestones

- [ ] Phase 1: Freeze The New Planning Inputs
  - [ ] Milestone 1.1: Introduce `ExecutionEnvironmentFacts` and
    `ExecutionStrategyPlan` in
    `core/store/runtime/ingestion/materialization_strategy_types.h`.
  - [ ] Milestone 1.2: Define the normalized boundary between retrieval policy
    and execution topology context in daemon normalization helpers and common
    runtime inputs.
  - [ ] Milestone 1.3: Extend typed daemon strategy config with explicit budgets,
    thresholds, and coordination timeouts required by the strategy plane.

- [ ] Phase 2: Converge Ordinary Replica Strategy Ownership
  - [ ] Milestone 2.1: Move ordinary `GPU <- DISK` executor candidacy and `AUTO`
    choice out of replica-layer branch ordering and into a common-runtime
    planning seam owned by `MaterializationFacade`.
  - [ ] Milestone 2.2: Make ordinary replica startup consume the same shared
    `RepresentationWorkPlan` and strategy contracts as mapped-target and
    `into_target` flows.
  - [ ] Milestone 2.3: Reduce `Replica` / `ReplicaLoadController` to executor
    runner and commit plumbing rather than strategy owner.

- [ ] Phase 3: Extract Coordinator And Prototype Boundaries
  - [ ] Milestone 3.1: Separate same-host group assembly, clique lifecycle, and
    timeout policy from executor-private owner-file logic.
  - [ ] Milestone 3.2: Split local-batched execution and collective execution
    into clearer internal executor modules or at least clearer ownership seams.
  - [ ] Milestone 3.3: Mark eager `owned_payload` preload and root whole-source
    preload as prototype-only paths with a hard cleanup target.

- [ ] Phase 4: Unify Reporting, Diagnostics, And Gating
  - [ ] Milestone 4.1: Make `ExecutionCommitReport` and executor diagnostics
    usable for ordinary replica startup as well as mapped-target flows.
  - [ ] Milestone 4.2: Emit explicit strategy input, chosen executor, rejected
    candidates, residual bytes, and commit coverage diagnostics.
  - [ ] Milestone 4.3: Ensure rollout and fallback behavior is driven entirely by
    typed config rather than hidden branch order or ambient state.

- [ ] Phase 5: Verification Baseline For 0109
  - [ ] Milestone 5.1: Establish correctness baselines for ordinary shared-FS TP
    startup across generic, local-batched, and current collective paths.
  - [ ] Milestone 5.2: Establish planner and routing evidence baselines
    (selection reason, owner skew, timeout behavior, cache-hit behavior).
  - [ ] Milestone 5.3: Freeze the code seams that `0109` will extend so the
    owner-file batched executor lands as a new executor, not as another
    prototype branch.

- [ ] Phase 6: Hard-Cut Cleanup Of Convergence Scaffolding
  - [ ] Milestone 6.1: Remove redundant strategy ownership from replica-layer
    branch order once the common-runtime seam is authoritative.
  - [ ] Milestone 6.2: Delete temporary coordinator/prototype shims that were
    kept only to land the convergence safely.
  - [ ] Milestone 6.3: Ensure no rollout-only or compatibility-only path remains
    in the hot path without a clear steady-state owner.

# Tasks

- [ ] Contracts and shared types
  - [ ] Update `core/store/runtime/ingestion/materialization_strategy_types.h`
    with:
    - `ExecutionEnvironmentFacts`
    - `ExecutionStrategyPlan`
    - cost-estimate/result helpers
    - richer `ExecutionCommitReport` semantics
  - [ ] Keep these contracts internal only; do not widen SDK or proto request
    surfaces with executor-private artifacts.

- [ ] Daemon normalization and topology context
  - [ ] Update `daemon/service/controllers/materialization_policy_utils.cc` and
    related controller call sites so retrieval policy normalization and
    execution-topology normalization are separate outputs.
  - [ ] Keep `collective_load_group` or its successor outside retrieval policy.
  - [ ] Update `daemon/service/controllers/replica_materialization_service.cc`
    and `daemon/service/controllers/target_materialization_service.cc` to build
    the normalized topology/context facts needed by common runtime.

- [ ] Common-runtime strategy convergence
  - [ ] Update `core/store/runtime/ingestion/materialization_facade.cc` so
    ordinary replica startup reaches a strategy-planning boundary before final
    executor choice.
  - [ ] Update `core/store/runtime/ingestion/materialization_service.cc` only as
    needed to carry the new strategy inputs through source acquisition.
  - [ ] Ensure mapped-target and `into_target` paths keep using the same
    strategy-family contracts after the convergence.

- [ ] Replica-layer cleanup
  - [ ] Update `core/store/replica/replica.cc` to stop owning final `AUTO`
    executor selection logic.
  - [ ] Update `core/store/replica/replica_load_controller.cc` so it executes a
    selected plan instead of implementing policy through branch order.
  - [ ] Isolate or refactor `core/store/replica/collective_disk_loader.cc` so
    group coordination, local-batched execution, and collective prototype logic
    no longer share one monolithic decision surface.

- [ ] Typed config and defaults
  - [ ] Extend `proto/tensorcast/config/v1/daemon_config.proto` and
    `core/store/store_engine_options.h` with strategy budgets and thresholds
    required by `0108` and `0109`.
  - [ ] Keep defaults conservative: no default switch to owner-file collective in
    this prerequisite plan.
  - [ ] Update `examples/config/store_daemon_config.yaml` to document the new
    typed strategy fields.

- [ ] Observability and diagnostics
  - [ ] Emit executor-candidacy and selection diagnostics from the common
    strategy seam.
  - [ ] Add commit coverage and residual accounting logs for ordinary replica
    startup.
  - [ ] Add explicit diagnostics for group-assemble timeout, coordinator
    fail-open/fail-closed policy, and prototype fallbacks.

- [ ] Documentation sync
  - [ ] Update `docs/internals/disk-load-strategy.md` after the code changes land
    so it matches the new topology/policy separation and strategy ownership.
  - [ ] Keep `0107`, `0108`, and `0109` aligned with the landed code seams.

# Test / Rollout / Backout

## Acceptance checks

- [ ] Ordinary replica `GPU <- DISK` startup no longer relies on replica-layer
  branch order as the architecture owner of `AUTO`.
- [ ] Retrieval policy and execution topology context are normalized separately.
- [ ] Strategy-plane diagnostics exist for ordinary replica startup and
  mapped-target execution.
- [ ] One normalized request boundary from `0107` feeds the strategy plane
  instead of per-controller topology parsing.
- [ ] Typed config carries the budgets and thresholds needed by the follow-on
  owner-file batched executor.
- [ ] The eager owner-file and whole-source preload paths remain non-default and
  are explicitly marked as prototype scaffolding.
- [ ] Temporary convergence scaffolding is removed or isolated out of the steady
  hot path before this plan is considered complete.

## Test plan

- [ ] `bash tools/build_proto_python.sh`
- [ ] `bazel test //core/common:daemon_config_io_test --test_output=errors`
- [ ] `bazel test //core/store/runtime/ingestion:materialization_service_test --test_output=errors`
- [ ] `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_output=errors`
- [ ] `bazel test //core/store/replica:collective_disk_loader_test --test_output=errors`
- [ ] `bazel test //core/store:store_engine_test --test_output=errors`
- [ ] `bazel test //daemon:materialize_into_mapped_target_test --test_output=errors`
- [ ] `bazel test //daemon:materialize_into_target_validation_test --test_output=errors`
- [ ] `source .venv/bin/activate && pytest tests/python/test_store_view_api.py`
- [ ] Shared-FS and host-local benchmark baselines are re-captured with the same
  workload family used by `docs/benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md`.

## Rollout

- [ ] Land the convergence work behind conservative typed defaults.
- [ ] Keep local-batched as the host-local/default stopgap while this plan is in
  progress.
- [ ] Do not claim default collective promotion for `0109` until this plan is
  complete and the follow-on validation matrix passes.
- [ ] Do not start `0109` implementation until the `0107` prerequisite and this
  plan's acceptance checks are complete.
- [ ] Finish by deleting temporary strategy-ownership scaffolding rather than
  leaving it behind for future work.

## Backout

- [ ] Revert the convergence change set as a unit if ordinary replica startup and
  mapped-target flows diverge semantically.
- [ ] Do not back out by reintroducing ambient env policy or by silently making
  replica-layer branch order the permanent strategy owner.

# Risks & Tracking

- [ ] Risk: ordinary replica convergence stalls halfway and leaves two strategy
  owners in the runtime.
  - Mitigation: treat `Replica` / `ReplicaLoadController` branch order as
    temporary scaffolding and remove policy ownership from them in the same wave.
- [ ] Risk: topology context leaks into retrieval policy or into
  executor-private hints again.
  - Mitigation: keep daemon normalization outputs separate and review all new
    fields against `0107`.
- [ ] Risk: typed config grows only booleans and not the budgets required by the
  real cost model.
  - Mitigation: do not mark this prerequisite plan complete until the threshold
    and budget fields are wired through config and defaults.
- [ ] Risk: prototype collective paths remain too entangled to support `0109`
  cleanly.
  - Mitigation: extract or isolate the coordination and preload seams before
    starting the batched owner-file executor implementation.
