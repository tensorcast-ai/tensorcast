---
slug: batched-owner-file-collective-rollout-and-residual-policy
title: Batched Owner-File Collective Rollout and Residual Policy Plan
status: in_progress
areas: ["core", "daemon", "docs", "tests", "benchmarks", "serving"]
created: 2026-04-10
last_updated: 2026-04-13
related_code:
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/runtime/ingestion/materialization_strategy_types.h
  - core/store/replica/collective_disk_loader.cc
  - core/store/replica/replica_load_controller.cc
  - core/store/store_engine_options.h
  - docs/internals/disk-load-strategy.md
links:
  design: ../designs/0109-batched-owner-file-collective-executor.md
---

# Objective

Track the remaining executor-specific rollout work for `0109` after the
documentation reorganization that makes:

- `0108` the sole strategy-plane owner,
- `0112` the same-binding serving-path owner,
- and this companion plan the active checklist for owner-file collective
  rollout, mixed-residual policy, benchmark evidence, and prototype cleanup.

# Current State & Grounding

- The phase-1 owner-file batched executor is already landed:
  - ordinary disk startup requires explicit shared-source proof before
    owner-file collective becomes eligible under `AUTO`,
  - `collective_disk_load` already has bounded owner batches and no longer
    depends on eager whole-owner payload residency for the intended path,
  - and the selected owner-file collective route no longer falls back to root
    whole-source preload when bounded owner planning is chosen.
- The current shipped executor scope is intentionally narrow:
  - the collective runtime is still collective-lane-only,
  - true generic residual bytes still reject collective eligibility,
  - mixed execution remains strategy-owned above the executor.
- The current repo-local executor policy is now explicit rather than implicit:
  - source-bound owner-file collective remains zero-residual-only by default,
  - enabling mixed residual requires typed config
    `owner_file_collective_allow_mixed_residual=true`,
  - and operator-visible execution diagnostics now expose actual unique-source
    bytes, peer-transfer bytes, peak temporary bytes, batch count, and dedup
    savings.
- The active follow-up items are executor-specific:
  - mixed-residual policy,
  - shared-FS and JFS evidence recapture,
  - default-policy graduation,
  - and final deletion of legacy owner preload scaffolding.
- Current host-local evidence remains intentionally separate:
  - qwen2.5 TP-local host-local benchmarking currently favors the exact generic
    path over dim1-staged local execution,
  - so `0109-01` must continue to treat host-local no-regression as a hard gate
    and must not infer any broader `AUTO` preference change from shared-source
    wins alone.
- Current shared-source qwen2.5 TP4 serving evidence also needs refreshed
  capture after a repo-local executor-seam fix:
  - the first safe TP4 mounted packet reached readiness and showed
    `OwnerFileCollectiveExecutor`, but it also showed a planning/execution
    mismatch (`planned_collective_admitted_bytes ~= 1.3MiB` vs
    `actual_collective_committed_bytes ~= 16.38GiB`),
  - local debugging traced that mismatch to copy-plan lowering sending the
    entire mapped byte space into `compatibility_lowered_map`,
  - fixing that seam exposed a second repo-local bug where
    `executor_generic_data_map` was derived from the narrowed compatibility map
    rather than the full `generic_fallback_map`, so source-bound plans
    correctly failed closed with `generic_backend_coverage_unproven`,
  - the current repo-local fixes now:
    - keep compatibility maps scoped to compatibility-admitted bytes,
    - derive executor generic coverage from the full generic fallback map,
    - admit dim0 concat work to the collective lane where runtime support
      already exists,
    - and infer source-only dim0/dim1 shard copies as partitioned typed work,
  - `2026-04-13` rerun evidence on the updated daemon now shows planning and
    execution aligned again on qwen2.5 TP4:
    - `planned_collective_admitted_bytes=16382928896`,
    - `actual_collective_committed_bytes=16382928896`,
    - `actual_generic_backend_bytes=0`,
    - `dominant_executor=OwnerFileCollectiveExecutor`,
    - and mounted startup reaches `/health=200`, `/v1/models`, and
      `/v1/completions`,
  - but the executor is still not rollout-ready on this workload:
    - TensorCast startup remains about `84.6s` to `85.4s`,
    - while the same safe TP4 `safetensors` baseline remains about `6.36s` to
      `6.41s`,
    - and the dominant slow phases visible in the mounted logs are still
      `ResolvePublicDiskSource` (~`54s`) followed by `RefillOwnedBinding`
      (~`38s` to `47s`).
- This plan should not absorb strategy-plane redesign work from `0108` or
  mounted same-binding path ownership from `0112`.

# Phases & Milestones

- [ ] Phase 1: Evidence And Policy Baseline
  - [ ] Milestone 1.1: recapture exact-workload and serving evidence for the
    current owner-file batched executor on the intended shared-source workloads.
  - [x] Milestone 1.2: make mixed-residual policy decisions against measured
    evidence instead of leaving them implicit.
  - [x] Milestone 1.3: freeze the executor-side delete gates for legacy preload
    scaffolding.

- [ ] Phase 2: Rollout And Graduation
  - [ ] Milestone 2.1: define when shared-source workloads should prefer
    owner-file collective under `AUTO`.
  - [ ] Milestone 2.2: prove no regression on host-local SSD workloads that
    should remain on local execution.
  - [ ] Milestone 2.3: prove wins or justified use on the intended shared-source
    serving and benchmark matrix.

- [ ] Phase 3: Cleanup
  - [x] Milestone 3.1: delete eager owner preload and related prototype-only
    collective scaffolding once the bounded batched path is authoritative.
  - [ ] Milestone 3.2: retire stale docs and rollout notes that still describe a
    closure-only document as the execution owner.

# Tasks

- [x] Decide the mixed-residual policy explicitly.
  - Keep zero-residual-only if evidence still shows it is the correct executor
    boundary.
  - Or document and implement a bounded mixed-residual policy if evidence shows
    it is required.
  - In either case, keep local typed work and true generic residual accounting
    strategy-owned rather than executor-invented.

- [ ] Recapture executor evidence on the intended shared-source cases.
  - Exact-workload comparisons on shared storage.
  - Serving readiness and mounted startup evidence on the intended family set.
  - Explicit measurements for unique-source bytes, peer-transfer bytes, peak
    temporary bytes, batch counts, and dedup savings.

- [x] Freeze the defaulting and backout policy.
  - Document when `AUTO` may prefer owner-file collective.
  - Keep host-local or unproven-source workloads on local or generic execution.
  - Preserve typed backout through config instead of branch-order policy.

- [x] Delete legacy collective scaffolding after proof.
  - Remove eager owner payload preload from any still-supported steady path.
  - Remove root whole-source preload from the final default collective route.
  - Retire prototype-only executor branches that no longer serve the active
    rollout.

# Test / Rollout / Backout

## Acceptance checks

- [x] The owner-file batched executor has an explicit residual policy and delete
  gate.
- [ ] Shared-source evidence is sufficient to justify its intended rollout.
- [ ] Host-local workloads that should stay local do not regress.
- [ ] Legacy preload scaffolding is deleted only after the bounded batched path
  is proven.

## Test plan

- [ ] Exact-workload benchmark on shared source media.
- [ ] Serving startup validation on the intended shared-source family set.
- [ ] Host-local no-regression benchmark against the local executor path.
- [ ] Collective diagnostics validation:
  - collective unique-source bytes
  - peer-transfer bytes
  - peak temporary bytes
  - batch count
  - dedup savings

## Rollout

- keep owner-file collective behind typed `engine.materialization_strategy`
  policy until executor evidence is sufficient,
- prefer explicit shared-source proof and typed policy over filesystem folklore,
- and coordinate with `0108` and `0112` only through the shared strategy seam
  rather than reopening those designs.

## Backout

- back out by changing typed executor preference or collective eligibility
  policy,
- do not restore replica-layer branch order as the policy owner,
- do not reintroduce broad eager preload behavior as the steady-state path.

# Risks & Tracking

- [ ] Risk: mixed residual policy is left ambiguous and leaks back into runtime
  folklore.
  - Mitigation: document one explicit policy here and enforce it through tests
    and typed diagnostics.

- [ ] Risk: the executor wins on shared storage but regresses host-local startup.
  - Mitigation: require host-local no-regression evidence before any broader
    `AUTO` preference change.

- [ ] Risk: prototype cleanup happens before the bounded batched path has enough
  mounted evidence.
  - Mitigation: tie deletion to explicit evidence and keep the delete gates in
    this plan only.
