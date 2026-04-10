---
slug: tensor-aware-materialization-strategy-plane
title: Tensor-Aware Materialization Strategy Plane Plan
status: in_progress
areas: ["core", "daemon", "sdk", "integrations", "proto", "docs", "tests", "benchmarks"]
created: 2026-03-23
last_updated: 2026-04-10
related_code:
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/runtime/ingestion/materialization_strategy_types.h
  - core/store/runtime/ingestion/source_bound_strategy_planner.cc
  - core/store/materialization/benchmarks/safetensors_load_strategy_benchmark_main.cc
  - core/store/replica/collective_disk_loader.cc
  - core/store/replica/replica.cc
  - core/store/replica/replica_load_controller.cc
  - daemon/service/controllers/materialization_target_plan_utils.cc
  - daemon/service/controllers/owned_binding_service.cc
  - proto/tensorcast/config/v1/daemon_config.proto
  - docs/internals/disk-load-strategy.md
links:
  design: ../designs/0108-tensor-aware-materialization-strategy-plane.md
---

# Objective

Track the remaining `0108`-owned strategy-plane work after the owner cleanup
that leaves:

- `0108` as the sole long-term strategy-plane owner,
- `0112` as the same-binding serving-path owner,
- `0109-01` as the owner-file collective rollout plan,
- and `0112-01` as the mounted rollout and delete-gate plan.

Active ownership on this plan is intentionally narrow:

- land the dedicated shared-runtime local tensor-aware executor for the ordinary
  host-local `GPU <- DISK` path,
- finish the prototype reabsorption that still bypasses the shared strategy seam
  on ordinary disk startup,
- and close the planner, correctness, safety, and parity gates that belong to
  the shared strategy plane itself.

Out of scope for this plan:

- owner-file collective executor rollout, mixed-residual policy, and shared-FS
  defaulting, which now live under `0109-01`,
- mounted same-binding serving rollout, operator-visible evidence hardening, and
  delete-gate cleanup, which now live under `0112-01`,
- and any downstream `internal-vllm` integration-local workflow that does not
  change TensorCast strategy-plane behavior.

# Current State & Grounding

- `0108` remains the sole long-term strategy-plane design owner:
  - `ResolvedMaterializationPlan` remains semantic-only,
  - `ExecutionStrategyPlan` remains the common runtime strategy carrier,
  - and source-bound explicit lane planning is now part of `0108` itself rather
    than a separate design.
- Ordinary disk startup already reaches the converged shared seam:
  - `MaterializationFacade` owns `AUTO`,
  - `Replica` / `ReplicaLoadController` consume a selected plan instead of
    owning executor branch order,
  - typed config lives under `engine.materialization_strategy`.
- Source-bound mapped and binding execution is no longer `0108`'s active rollout
  owner:
  - explicit source-bound planning is already landed in
    `core/store/runtime/ingestion/source_bound_strategy_planner.cc`,
  - audited Step3p5 same-binding mounted closure is now a `0112`-owned path-level
    result,
  - and `0108` treats that path as a no-regression consumer of the shared
    strategy seam.
- The main remaining `0108` gap is still the ordinary host-local path:
  - `executor_preference=TENSOR_AWARE_LOCAL` is not yet backed by a dedicated
    shared-runtime local executor for the target workloads that matter,
  - host-local full real-target-layout validation already passes correctness, but
    `tensorcast` is still slightly behind the `auto` baseline on the measured
    full-layout run,
  - replica-side local-batched hooks and prototype ownership are still present
    and should be retired only after the shared-runtime executor closes parity.
- `0108` must not reclaim work already delegated elsewhere:
  - `0109-01` owns owner-file collective executor evidence, policy, and cleanup,
  - `0112-01` owns mounted operator evidence, delete-gates, and deferred
    second-family validation.

# Phases & Milestones

- [x] Phase 1: Documentation Convergence
  - [x] Milestone 1.1: fold source-bound explicit planning rules into `0108`
    design ownership.
  - [x] Milestone 1.2: Move owner-file collective rollout tracking to `0109-01`.
  - [x] Milestone 1.3: Move same-binding mounted rollout and delete-gate work to
    `0112-01`.

- [ ] Phase 2: Host-Local Ordinary Local Executor
  - [ ] Milestone 2.1: Land a dedicated shared-runtime local tensor-aware
    executor for ordinary host-local disk startup.
  - [ ] Milestone 2.2: Cover contiguous, dim1-pack, and dedup-copy dominant
    workloads through that executor.
  - [ ] Milestone 2.3: Preserve exact residual fallback and planner-owned
    selection reasons when the local executor does not admit a request.

- [ ] Phase 3: Optional Source-Bound Local Convergence
  - [ ] Milestone 3.1: Decide, from evidence, whether source-bound mapped or
    binding requests still need a dedicated local executor beyond the current
    explicit lane-plan trunk.
  - [ ] Milestone 3.2: If such work is needed, extend it only through explicit
    lane-plan-backed execution rather than a second mapped fast path.
  - [ ] Milestone 3.3: If such work is not needed, document that outcome and
    keep `0112` source-bound execution on the current explicit strategy trunk.

- [ ] Phase 4: Prototype Reabsorption
  - [ ] Milestone 4.1: Retire replica-layer local-batched late hooks after the
    shared-runtime executor closes parity for the owned workload family.
  - [ ] Milestone 4.2: Remove ordinary-path prototype-only bypasses and stale
    executor ownership seams that no longer serve the hot path.
  - [ ] Milestone 4.3: Keep `ByteRangeMap` fallback exact and explainable after
    cleanup.

- [ ] Phase 5: Validation And Graduation
  - [ ] Milestone 5.1: Add deterministic planner and residual-coverage tests for
    the shared strategy seam.
  - [ ] Milestone 5.2: Prove host-local parity or better versus the current best
    `auto` / `fastsafetensors` baselines on the owned workloads.
  - [ ] Milestone 5.3: Revalidate the audited source-bound path when `0108`
    changes touch shared strategy code.

# Tasks

- [ ] Land the ordinary host-local local tensor-aware executor.
  - Reuse benchmark-proven planning ideas from
    `core/store/materialization/benchmarks/safetensors_load_strategy_benchmark_main.cc`.
  - Support direct contiguous reads to final layout.
  - Support staged row-block reads plus GPU 2D pack for dim1 patterns.
  - Support source-slice dedup and D2D copy reuse where it helps ordinary
    host-local startup.
  - Emit explicit planner reasons when requests fall back to generic execution.

- [ ] Close the owned correctness and safety gaps.
  - Add planner-level C++ tests for deterministic lowering.
  - Add disablement and partial-eligibility coverage so executor gating widens
    explicit fallback rather than suppressing bytes.
  - Add regression coverage for residual-accounting and commit-report
    consistency.
  - Keep external-target safety and publication correctness on the shared
    runtime paths that `0108` still touches.

- [ ] Retire ordinary-path prototype ownership once the shared executor is ready.
  - Remove replica-layer local-batched late hooks after parity is proven.
  - Remove obsolete ordinary-path prototype-only branches that no longer serve
    the hot path.
  - Keep source-bound explicit planning and mounted rollout out of this cleanup
    unless a touched shared seam requires no-regression validation.

- [ ] Maintain clean scope boundaries with companion plans.
  - Do not add owner-file collective executor policy work here; track it in
    `0109-01`.
  - Do not add mounted Step3p5 rollout or delete-gate tasks here; track them in
    `0112-01`.
  - If shared runtime changes affect those paths, record only the no-regression
    validation dependency here and keep the execution owner unchanged.

# Test / Rollout / Backout

## Acceptance checks

- [ ] Ordinary host-local startup has a dedicated shared-runtime local
  tensor-aware executor for the intended workload family.
- [ ] Planner disablement, partial eligibility, and fallback preserve exact byte
  coverage.
- [ ] Replica-layer late hooks no longer own the ordinary hot path once the new
  executor is authoritative.
- [ ] Host-local end-to-end TensorCast matches or beats the owned baseline on
  the intended workloads.
- [ ] Changes touching the shared strategy seam do not regress the audited
  source-bound same-binding path owned by `0112`.

## Test plan

- [ ] Ordinary host-local exact-workload checks:
  - exact `879` workload
  - exact `614` sliced workload
  - one planner-vs-generic benchmark on host-local SSD
- [ ] Real target-layout checks:
  - subset real-target-layout digest parity on host-local SSD
  - full real-target-layout digest parity on host-local SSD
- [ ] Planner and fallback checks:
  - executor disabled
  - partial eligibility
  - mixed execution plus residual fallback
  - explicit rejection when coverage is not proven
- [ ] Shared-seam no-regression checks when touched:
  - `bazel test //core/store/runtime/ingestion:materialization_facade_test`
  - `bazel test //core/store/runtime/ingestion:materialization_service_test`
  - `bazel test //daemon:materialize_into_mapped_target_test`

## Rollout

- keep new local execution behind typed `engine.materialization_strategy`
  controls until host-local parity is proven,
- validate the new local executor first under targeted preference before any
  default-policy change,
- keep owner-file collective rollout and same-binding mounted rollout delegated to
  their companion plans,
- and treat source-bound Step3p5 validation as a no-regression dependency rather
  than `0108`'s primary graduation gate.

## Backout

- back out by forcing explicit generic fallback through typed config,
- do not restore replica-layer branch-order policy as the architecture owner of
  `AUTO`,
- do not reintroduce executor-private request hints or runtime implicit fallback
  to recover from a failed rollout.

# Risks & Tracking

- [ ] Risk: the local tensor-aware executor lands too narrowly and fails to close
  the host-local parity gap.
  - Mitigation: measure exact-workload and real target-layout runs before
    retiring the existing prototype hooks.

- [ ] Risk: source-bound local optimization work reopens a second mapped fast
  path.
  - Mitigation: any future source-bound local executor work must extend the
    explicit lane-plan model already owned by `0108`.

- [ ] Risk: cleanup removes ordinary-path scaffolding before the new executor is
  fully authoritative.
  - Mitigation: gate deletion on parity evidence and deterministic fallback
    coverage tests.

- [ ] Risk: `0108` scope drifts back into owner-file collective rollout or
  mounted serving cleanup.
  - Mitigation: keep those TODOs only in `0109-01` and `0112-01`, and treat them
    here as external dependencies only.
