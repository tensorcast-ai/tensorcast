---
slug: batched-owner-file-collective-rollout-and-residual-policy
title: Batched Owner-File Collective Rollout and Residual Policy Plan
status: in_progress
areas: ["core", "daemon", "docs", "tests", "benchmarks", "serving"]
created: 2026-04-10
last_updated: 2026-04-10
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
  - root whole-source preload remains only as legacy fallback scaffolding.
- The current shipped executor scope is intentionally narrow:
  - the collective runtime is still collective-lane-only,
  - true generic residual bytes still reject collective eligibility,
  - mixed execution remains strategy-owned above the executor.
- The active follow-up items are executor-specific:
  - mixed-residual policy,
  - shared-FS and JFS evidence recapture,
  - default-policy graduation,
  - and final deletion of legacy owner preload scaffolding.
- This plan should not absorb strategy-plane redesign work from `0108` or
  mounted same-binding path ownership from `0112`.

# Phases & Milestones

- [ ] Phase 1: Evidence And Policy Baseline
  - [ ] Milestone 1.1: recapture exact-workload and serving evidence for the
    current owner-file batched executor on the intended shared-source workloads.
  - [ ] Milestone 1.2: make mixed-residual policy decisions against measured
    evidence instead of leaving them implicit.
  - [ ] Milestone 1.3: freeze the executor-side delete gates for legacy preload
    scaffolding.

- [ ] Phase 2: Rollout And Graduation
  - [ ] Milestone 2.1: define when shared-source workloads should prefer
    owner-file collective under `AUTO`.
  - [ ] Milestone 2.2: prove no regression on host-local SSD workloads that
    should remain on local execution.
  - [ ] Milestone 2.3: prove wins or justified use on the intended shared-source
    serving and benchmark matrix.

- [ ] Phase 3: Cleanup
  - [ ] Milestone 3.1: delete eager owner preload and related prototype-only
    collective scaffolding once the bounded batched path is authoritative.
  - [ ] Milestone 3.2: retire stale docs and rollout notes that still describe a
    closure-only document as the execution owner.

# Tasks

- [ ] Decide the mixed-residual policy explicitly.
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

- [ ] Freeze the defaulting and backout policy.
  - Document when `AUTO` may prefer owner-file collective.
  - Keep host-local or unproven-source workloads on local or generic execution.
  - Preserve typed backout through config instead of branch-order policy.

- [ ] Delete legacy collective scaffolding after proof.
  - Remove eager owner payload preload from any still-supported steady path.
  - Remove root whole-source preload from the final default collective route.
  - Retire prototype-only executor branches that no longer serve the active
    rollout.

# Test / Rollout / Backout

## Acceptance checks

- [ ] The owner-file batched executor has an explicit residual policy and delete
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
