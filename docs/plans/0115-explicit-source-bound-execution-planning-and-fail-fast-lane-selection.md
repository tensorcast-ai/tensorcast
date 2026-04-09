---
slug: explicit-source-bound-execution-planning-and-fail-fast-lane-selection
title: Explicit Source-Bound Execution Planning and Fail-Fast Lane Selection Plan
status: draft
areas: ["core", "daemon", "docs", "tests", "sdk"]
related_code:
  - docs/designs/0115-explicit-source-bound-execution-planning-and-fail-fast-lane-selection.md
  - docs/designs/0114-collective-first-binding-realization-for-tp-serving-startup.md
  - daemon/service/controllers/representation_transform_builder.h
  - daemon/service/controllers/representation_transform_builder.cc
  - daemon/service/controllers/materialization_target_plan_utils.h
  - daemon/service/controllers/materialization_target_plan_utils.cc
  - daemon/service/controllers/materialization_target_plan_utils_test.cc
  - daemon/service/controllers/owned_binding_service.cc
  - core/store/runtime/ingestion/materialization_strategy_types.h
  - core/store/runtime/ingestion/materialization_facade.h
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/runtime/ingestion/materialization_facade_test.cc
  - core/store/store_engine.h
  - core/store/store_engine.cc
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - tensorcast/types.py
  - tests/python/test_binding.py
links:
  design: ../designs/0115-explicit-source-bound-execution-planning-and-fail-fast-lane-selection.md
---

# Objective

Implement `0115` so source-bound mapped and binding execution becomes:

- explicitly planned before executor setup,
- fail-fast when no policy-compliant plan exists,
- performance-preserving through reuse of existing lowerings and fast paths,
- and free of runtime implicit fallback.

# Current State & Grounding

- Current semantic truth already exists in
  [materialization_strategy_types.h](/data/workspace/tensorcast-280/core/store/runtime/ingestion/materialization_strategy_types.h),
  `RepresentationTransformContract`, and `RepresentationWorkPlan`.
- Current controller lowering still leaks executor-private maps into
  `ResolvedMaterializationPlan` inside
  [materialization_target_plan_utils.cc](/data/workspace/tensorcast-280/daemon/service/controllers/materialization_target_plan_utils.cc).
- Current source-bound execution still makes runtime fallback decisions in
  [materialization_facade.cc](/data/workspace/tensorcast-280/core/store/runtime/ingestion/materialization_facade.cc).
- Current planner diagnostics are derived indirectly from
  `ResolvedMaterializationPlan` in
  [owned_binding_service.cc](/data/workspace/tensorcast-280/daemon/service/controllers/owned_binding_service.cc).
- Existing repository policy already requires explicit lane allocation before
  execution in
  [0108](/data/workspace/tensorcast-280/docs/designs/0108-tensor-aware-materialization-strategy-plane.md#L602)
  and
  [0114](/data/workspace/tensorcast-280/docs/designs/0114-collective-first-binding-realization-for-tp-serving-startup.md#L774).

# Phases & Milestones

- [ ] Phase 1: Separate semantic plan from source-bound lowering artifacts
  - [ ] Add core-owned `SourceBoundPolicy`, `SourceBoundLoweringStats`, and `SourceBoundLoweringArtifacts` types in [materialization_strategy_types.h](/data/workspace/tensorcast-280/core/store/runtime/ingestion/materialization_strategy_types.h)
  - [ ] Stop storing `collective_compatibility_map` and `executor_private_generic_fallback_map` on `ResolvedMaterializationPlan`
  - [ ] Update controller helpers so `build_resolved_mapped_materialization_plan(...)` returns semantic truth only, while `PreparedSourceBoundExecution` carries lowering artifacts separately

- [ ] Phase 2: Build an explicit source-bound strategy plan
  - [ ] Extend `ExecutionStrategyPlan` with `source_bound_lane_plan`
  - [ ] Implement `build_source_bound_execution_strategy_plan(...)` in core/runtime
  - [ ] Normalize daemon public `CollectivePolicy` into internal `SourceBoundPolicy`
  - [ ] Derive `SourceBoundPlanDiagnostics` from the explicit strategy plan instead of from optional executor-private maps

- [ ] Phase 3: Cut execution over to exact-plan semantics
  - [ ] Update `StoreEngine` and `MaterializationFacade` internal entrypoints to consume the explicit strategy plan
  - [ ] Remove source-bound runtime `value_or(...)` fallbacks and `handled=false` generic continuation
  - [ ] Preserve source-ordered fast path for explicit generic plans
  - [ ] Preserve collective lane source finalization without letting it change lane ownership

- [ ] Phase 4: Diagnostics, tests, and transitional cleanup
  - [ ] Add additive planner diagnostics fields if the implementation needs mode or local pad or fill split on the daemon wire
  - [ ] Update Python typed diagnostics in [tensorcast/types.py](/data/workspace/tensorcast-280/tensorcast/types.py) if proto fields change
  - [ ] Remove or demote transitional map semantics after tests prove no runtime implicit fallback remains
  - [ ] Update architectural docs if the final implementation meaning differs from the current `0114` wording

# Tasks

- [ ] Add focused unit tests for planner mode selection:
  - `pure_collective`
  - `collective_first_mixed`
  - `generic_only`
  - `local_typed_only`
  - explicit reject
- [ ] Add coverage-invariant tests proving complete coverage or explicit failure
- [ ] Add fail-fast tests proving `REQUIRE_COLLECTIVE` rejects before generic execution starts
- [ ] Add source-bound execution tests proving generic backend bytes are planned, not inferred
- [ ] Add Python diagnostics tests for the new planner facts if proto fields change

# Test / Rollout / Backout

## Test Plan

- [ ] `bazel test //daemon:materialization_target_plan_utils_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- [ ] `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- [ ] `bazel test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- [ ] `source .venv/bin/activate && pytest tests/python/test_binding.py`
- [ ] If daemon proto changes: `bash tools/build_proto_python.sh`

## Rollout

- [ ] Land planner types and controller/runtime separation first with tests
- [ ] Land executor cutover second
- [ ] Land additive diagnostics third
- [ ] Delete transitional fallback logic only after all source-bound tests pass

## Backout

- [ ] Back out the explicit strategy-plan consumption while keeping the new docs if coverage or performance regressions appear
- [ ] Do not back out semantic-plan cleanup partially; restore the prior full behavior only with a coherent revert

# Risks & Tracking

- [ ] Risk: coverage partition bugs could over-reject requests
  - Tracking: add coverage-invariant tests before executor cutover
- [ ] Risk: strategy planning introduces hot-path overhead
  - Tracking: reuse existing lowering maps and measure test-time executor-path stability
- [ ] Risk: diagnostics proto growth drifts away from the accepted architecture
  - Tracking: keep planner facts on `SourceBoundPlanDiagnostics` and actual facts on execution diagnostics
