---
slug: artifact-realization-and-cluster-rollout
title: Artifact Realization, Worker-Local Stable Readiness, and Cluster Rollout (Plan)
links:
  design: ../designs/0104-artifact-realization-and-cluster-rollout.md
---

# Objective

Add the missing layer between `0056` set orchestration, `0102` engine manifest projection, and `0103` volatile
publication:

- one explicit worker-local stable realization action,
- one cluster rollout contract that composes worker realization and instance realization honestly,
- and one source-retention model that allows transient source backing for training-to-inference rollout.

# Current State & Grounding

- `0056` already owns `ArtifactSetRef`, daemon ingress, and worker-set warmup, but it intentionally keeps
  `prefetch_set` at a worker-local readiness floor rather than silently claiming `local_stable_ready`.
  - [0056-programmable-framework-adv.md](/data/workspace/tensorcast-1/docs/designs/0056-programmable-framework-adv.md)
- `0102` already keeps the canonical instance vocabulary at:
  - `manifest`
  - `publish`
  - `hydrate`
  - `evict_local`
  and owns the bridge from integration-side carriers into `ArtifactSetRef`.
  - [0102-engine-artifact-integration-and-high-cardinality-manifest-orchestration.md](/data/workspace/tensorcast-1/docs/designs/0102-engine-artifact-integration-and-high-cardinality-manifest-orchestration.md)
- `0103` already narrows `publish` to volatile publication subjects and explicitly says multi-target replication intent
  belongs to orchestration above one publish subject.
  - [0103-volatile-publication-subjects-and-multi-replica-semantics.md](/data/workspace/tensorcast-1/docs/designs/0103-volatile-publication-subjects-and-multi-replica-semantics.md)
- Current worker-local stable admission already exists and is centered on:
  - [local_stable_tier_service.cc](/data/workspace/tensorcast-1/daemon/service/controllers/local_stable_tier_service.cc)
  - [registration-flow.md](/data/workspace/tensorcast-1/docs/architecture/api/registration-flow.md#local-stable-tier)
  - [policy-persistence.md](/data/workspace/tensorcast-1/docs/architecture/api/policy-persistence.md)
- Current remote stable persistence remains persistence-oriented and must not be treated as rollout-ready worker-local
  stable realization:
  - [persistence_manager.cc](/data/workspace/tensorcast-1/daemon/state/persistence_manager.cc)
  - [policy-persistence.md](/data/workspace/tensorcast-1/docs/architecture/api/policy-persistence.md)
- Current data plane already has the transport profiles rollout should reuse:
  - GPU export and direct RDMA capability flags in [memory_export_registry.cc](/data/workspace/tensorcast-1/core/store/replica/memory_export_registry.cc)
  - target-side direct write and remote reads in [remote_key_source.cc](/data/workspace/tensorcast-1/core/store/materialization/dataplane/sources/remote_key_source.cc)
  - orchestration in [materialize_orchestrator.cc](/data/workspace/tensorcast-1/core/store/materialization/control/materialize_orchestrator.cc)
- Current source ingress already has a strong local CPU path through memfd publish:
  - [0082-cpu-memfd-zero-copy-publish.md](/data/workspace/tensorcast-1/docs/designs/0082-cpu-memfd-zero-copy-publish.md)
  - [weight_publisher.py](/data/workspace/tensorcast-1/tensorcast/tools/weight_publisher.py)

# Phases & Milestones

- [ ] Phase 1: Close the semantic layer and plan surface
  - [ ] Milestone 1.1: Add `0104` design + plan and cross-link it from `0056`, `0102`, and rollout-facing docs.
  - [ ] Milestone 1.2: Decide the plan-surface shape:
    - new `RealizeSetAction`, or
    - equivalent explicit readiness selector on a worker-set action.
  - [ ] Milestone 1.3: Keep `prefetch_set` semantics unchanged and document why rollout needs a distinct contract.
  - [ ] Milestone 1.4: Freeze worker-local readiness vocabulary:
    - `local_replica_ready`
    - `local_stable_ready`
    - `instance_hydrated`
  - [ ] Milestone 1.5: Freeze the public issue-time SDK surface:
    - `RegisterArtifactOptions.realization_strategy`
    - `put/register(strategy=...)` as convenience syntax
    - `RegisteredArtifact.realization_operation()` as the follow-up observation hook

- [ ] Phase 2: Implement worker-local stable realization
  - [ ] Milestone 2.1: Add worker-layer action and result carrier for `realize_set`.
  - [ ] Milestone 2.2: Route `realize_set(..., readiness_kind="local_stable_ready")` through existing local stable
        admission machinery.
  - [ ] Milestone 2.3: Ensure the action fails or degrades honestly when only replica warmup succeeds.
  - [ ] Milestone 2.4: Emit path-profile diagnostics and readiness-kind diagnostics.

- [ ] Phase 3: Add rollout compilation and operation semantics
  - [ ] Milestone 3.1: Add internal rollout intent / compiler layer that lowers to canonical plan actions.
  - [ ] Milestone 3.2: Reuse `Operation[PlanResult]` / `0096` / `0100` continuation semantics rather than inventing a
        rollout-private continuation family.
  - [ ] Milestone 3.3: Support source-retention modes:
    - `drop_on_rollout_ready`
    - `keep_until_ttl`
    - `best_effort`
  - [ ] Milestone 3.4: Keep optional target-side child `publish` composition separate from worker realization.

- [ ] Phase 4: Path selection and diagnostics
  - [ ] Milestone 4.1: Implement source-ingress and worker-realization path selection as two separate stages.
  - [ ] Milestone 4.2: Prefer existing shared dataplane profiles:
    - `cpu_memfd_local_ingress`
    - `gpu_direct_rdma_pull_to_cpu_stable`
    - `gpu_staged_rdma_pull_to_cpu_stable`
    - `cpu_direct_write_pull_to_cpu_stable`
    - staged or disk fallback
  - [ ] Milestone 4.3: Keep default user surface parameter-light and expose selected-path diagnostics instead of many
        knobs.

- [ ] Phase 5: Integration and deployment closeout
  - [ ] Milestone 5.1: Rebase `WeightPublisher` or successor workflow helper onto rollout intent and `realize_set`.
  - [ ] Milestone 5.2: Keep `0102` manifest bridge the only owner of engine carrier -> `ArtifactSetRef`.
  - [ ] Milestone 5.3: Validate training-source transient closeout after target rollout barrier.
  - [ ] Milestone 5.4: Update rollout-facing operator docs once code paths are real.

# Tasks

- Add or amend proto and SDK worker action surfaces for explicit realization semantics.
- Add `RegisterArtifactOptions.realization_strategy` and top-level `put/register(strategy=...)` sugar without
  conflating the new field with existing `plan` or `policy`.
- Add typed worker-layer realization result carriers with explicit readiness kind.
- Reuse `LocalStableTierService` instead of inventing a rollout-private stable contract.
- Add rollout compiler helpers that lower to canonical plan actions and `Operation[PlanResult]`.
- Extend `RegisteredArtifact` observation with realization follow-up access while keeping `RegisteredArtifact` as the
  single issue-result top-level type.
- Ensure worker-target and instance-target scope remain separate in rollout execution.
- Reuse current shared data plane for:
  - source selection
  - transport path selection
  - target worker materialization
  - stable admission
- Do not interpret current persistence remote stable success as rollout success.
- Add source-retention closeout logic for transient training issuers.
- Add documentation and telemetry for selected path profile and readiness barrier.

# Test / Rollout / Backout

## Test Plan

- Worker-local realization semantics
  - add targeted daemon and SDK tests proving:
    - `prefetch_set` remains `local_replica_ready`
    - `realize_set(..., local_stable_ready)` reaches true stable admission or fails honestly
    - `put/register(strategy=...)` lowers to the same worker-realization semantics as explicit rollout compilation
- Data-path selection
  - add tests covering:
    - CPU memfd local ingress
    - GPU direct or staged source capability selection
    - CPU or staged pull fallback
    - disk fallback when source capability is unavailable
- Rollout composition
  - add plan and integration tests covering:
    - source issue -> worker realize_set -> instance hydrate
    - optional child target publish without changing rollout semantics
    - `drop_on_rollout_ready` source closeout
- Weight rollout integration
  - add or adapt E2E coverage under:
    - [weight_publisher.py](/data/workspace/tensorcast-1/tensorcast/tools/weight_publisher.py)
    - [weight-publisher.md](/data/workspace/tensorcast-1/docs/deployment/weight-publisher.md)

## Rollout

1. Land docs and semantic naming first.
2. Land worker-local `realize_set` before any higher-level rollout helper.
3. Land rollout compiler and source-retention modes on top of the same canonical plan spine.
4. Migrate deployment helpers after worker realization semantics are stable.

## Backout

- Keep `prefetch_set` unchanged.
- Keep current `publish` subject semantics unchanged.
- If rollout helpers regress, fall back to explicit plan composition over existing `prefetch_set` and instance actions
  while preserving the new docs as the target semantic model.

# Risks & Tracking

- Risk: `realize_set` becomes a second warmup action with ambiguous overlap with `prefetch_set`.
  - Tracking: keep readiness-kind requirements explicit and document the semantic split everywhere.
- Risk: rollout helper code starts acting like a second execution substrate.
  - Tracking: require every helper path to compile into canonical plan actions and `Operation[PlanResult]`.
- Risk: the issue-time SDK surface (`strategy`) is mistaken for a replacement of `plan`, `policy`, or full rollout.
  - Tracking: document and test the exact layer split:
    - `plan` = local ingress and registration plan
    - `policy` = backing and retention
    - `strategy` = issue-time worker realization intent
- Risk: implementation quietly treats persistence remote stable success as rollout success.
  - Tracking: add explicit negative tests and docs callouts.
- Risk: source-transient rollout drops source backing before the worker-local barrier is truly met.
  - Tracking: gate source closeout strictly on rollout barrier and add failure-path coverage.
- Risk: path selection grows many hidden knobs or environment-variable branches.
  - Tracking: require diagnostics-first and unified-config-first path tuning.
