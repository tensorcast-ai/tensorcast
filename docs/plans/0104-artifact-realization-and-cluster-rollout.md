---
slug: artifact-realization-and-cluster-rollout
title: Artifact Realization, Worker-Local Stable Readiness, and Cluster Rollout (Plan)
links:
  design: ../designs/0104-artifact-realization-and-cluster-rollout.md
---

# Objective

Land the two-layer kernel from `0104` without forking the programmable
framework:

- Layer 1: worker-local stable realization over `ArtifactSetRef`
- Layer 2: rollout orchestration over canonical plan actions and `Operation[T]`

Implementation rule:

- every new surface must either be a canonical action,
- or compile into canonical `PlanSpec` fragments and generic operation
  observation,
- never a second execution substrate.

# Current State and Grounding

- `0055` already gives:
  - `Plan`
  - `Operation[T]`
  - stable target identity rules
  - the `PlanSpec -> daemon / NodeAgent / EngineAdapter` execution spine
- `0056` already gives:
  - `ArtifactSetRef`
  - `prefetch_set`
  - explicit `local_replica_ready` floor
  - reserved cluster transport slot
  - daemon-served signals direction, but not yet full directory closure
- shared directory closure and `NodeAgentDirectory` are now being factored into
  `0106`; `0104` consumes that prerequisite rather than redefining it
- `0102` already gives:
  - canonical instance verbs:
    - `manifest`
    - `publish`
    - `hydrate`
    - `evict_local`
  - manifest bridge into `ArtifactSetRef`
- `0103` already gives:
  - volatile target-publication semantics
  - child `publish` ownership
- current stable-admission code already exists, but is registration-bound:
  - `LocalStableTierService`
  - registration-time `local_stable_tier`
- current dataplane already exists and must be reused:
  - `MaterializeOrchestrator`
  - `RemoteKeySource`
  - `MemoryExportRegistry`
- current peer-daemon transport helpers are still data-plane oriented; there is
  no dependency-ready inter-daemon control-plane subplan dispatcher yet
- current helper stack is still persistence-driven:
  - `WeightPublisher`
  - `wait_persistence`
  - external reload

# Execution Strategy

The implementation program should be staged in dependency order.

Do not start from rollout helpers.
Start from target identity and worker-local stable realization.

# Phases and Milestones

- [ ] Phase 1: Freeze the semantic split and target model
  - [ ] Milestone 1.1: Rewrite `0104` around the two-layer kernel:
    - worker-local realization kernel
    - rollout orchestration kernel
  - [ ] Milestone 1.2: Freeze target identity rules:
    - worker targets are `daemon_id`
    - instance targets are `instance_id`
  - [ ] Milestone 1.3: Freeze the first typed rollout carriers:
    - `WorkerRealizationIntent`
    - `ArtifactRolloutIntent`
    - `RolloutBarrierRef`
  - [ ] Milestone 1.4: Keep `prefetch_set` semantics unchanged and document why
        `realize_set` must be a separate action.
  - [ ] Milestone 1.5: Freeze the issue-time strategy split:
    - `plan` = local ingress / registration plan
    - `policy` = backing / durability
    - `strategy` = post-issue worker realization intent

- [ ] Phase 2: Close daemon-served directory and target resolution prerequisites
  - [ ] Milestone 2.1: Add daemon-side directory surfaces:
    - `ListWorkers`
    - `ListInstances`
    - `GetWorkerCapacity`
  - [ ] Milestone 2.2: Add `InstanceDirectoryCache` beside
        `WorkerDirectoryCache`.
  - [ ] Milestone 2.3: Add `DirectoryController` and wire it through daemon RPC
        delegates.
  - [ ] Milestone 2.4: Extend SDK signals to use the daemon-served directory:
    - `runtime.signals().list_workers(...)`
    - `runtime.signals().list_instances(...)`
    - `runtime.signals().get_worker_capacity(...)`
  - [ ] Milestone 2.5: Ensure rollout code no longer depends on direct SDK calls
        to Global Store for worker / instance listing.

- [ ] Phase 3: Extract the worker-local stable realization kernel
  - [ ] Milestone 3.1: Add `StableRealizationService` as reusable worker-local
        stable realization logic.
  - [ ] Milestone 3.2: Refactor `LocalStableTierService` into a registration
        adapter over `StableRealizationService`.
  - [ ] Milestone 3.3: Keep `ReplicaMaterializationService` as the loading /
        allocation owner and reuse it from the new kernel.
  - [ ] Milestone 3.4: Project explicit realization outcomes:
    - actual readiness kind
    - degraded reason
    - selected path profile
  - [ ] Milestone 3.5: Add explicit negative coverage proving persistence remote
        stable success is not rollout success.

- [ ] Phase 4: Land `realize_set` on the canonical plan spine
  - [ ] Milestone 4.1: Extend `plan.proto` with:
    - `RealizeSetAction`
    - placement enum
    - readiness-kind enum
  - [ ] Milestone 4.2: Extend `node_agent.v1.ArtifactSetResult` and SDK result
        decoding with:
    - `observed_readiness_kind`
    - `path_profile`
    - `degraded_reason`
  - [ ] Milestone 4.3: Add `WorkerStepBuilder.realize_set(...)`.
  - [ ] Milestone 4.4: Implement `realize_set` in the local Python plan runner.
  - [ ] Milestone 4.5: Implement `realize_set` in daemon ingress
        `ExecutePlan(terminal_only)`.
  - [ ] Milestone 4.6: Verify `prefetch_set` and `realize_set` stay behaviorally
        distinct.

- [ ] Phase 5: Land the first rollout orchestration wave
  - [ ] Milestone 5.1: Add rollout compiler carriers:
    - `ArtifactRolloutIntent`
    - `CompiledRollout`
    - `RolloutBarrierRef`
    - `ArtifactRolloutResult`
  - [ ] Milestone 5.2: Add `RolloutController` as the front door that compiles
        rollout into canonical `PlanSpec` fragments.
  - [ ] Milestone 5.3: Add `StartRollout` RPC returning generic
        `operation.v1.OperationRef`.
  - [ ] Milestone 5.4: Add SDK `Runtime.rollout(...)` as the public front door
        over rollout compilation and `StartRollout`.
  - [ ] Milestone 5.5: Reuse the generic operation plane for rollout
        observation:
    - no rollout-private wait/status RPC family
    - no rollout-private continuation family
  - [ ] Milestone 5.6: Keep the first dependency-ready rollout slice to the
        `workers_ready` barrier over worker-only subplans.
  - [ ] Milestone 5.7: Keep `cluster_action` reserved but do not make it the v1
        owner of rollout truth.
  - [ ] Milestone 5.8: Do not claim multi-daemon rollout controller closure
        until a real inter-daemon control-plane dispatch substrate exists.

- [ ] Phase 6: Add issue-time worker realization strategy
  - [ ] Milestone 6.1: Add `RegisterArtifactOptions.realization_strategy`.
  - [ ] Milestone 6.2: Add `register(..., strategy=...)` as the first
        dependency-ready issue-time convenience surface.
  - [ ] Milestone 6.3: Add `RegisteredArtifact.realization_operation()` and
        additive result fields.
  - [ ] Milestone 6.4: Validate contradictory input rejection:
    - `mode="worker_realization"` with empty `worker_daemon_ids`
    - `source_closeout="keep_until_ttl"` without `source_closeout_ttl_ms`
    - `wait_for="workers_ready"` without worker realization
  - [ ] Milestone 6.5: Keep `put(..., strategy=...)` explicitly out of the first
        wave until source semantics are made honest for that path.

- [ ] Phase 7: Expand rollout child-stage composition
  - [ ] Milestone 7.1: Define typed child-stage composition for optional
        publication and hydration stages.
  - [ ] Milestone 7.2: Keep child publication composition separate from worker
        realization and use only canonical `0103` publish actions.
  - [ ] Milestone 7.3: Keep child hydration composition separate from worker
        realization and use only canonical `0102` hydrate actions.
  - [ ] Milestone 7.4: Gate dependency-ready rollout child stages on daemon
        ingress support for local instance-target execution behind the target
        daemon.
  - [ ] Milestone 7.5: Expose `RolloutBarrierRef` as the typed child contract
        needed by `0105`.

- [ ] Phase 8: Migrate helpers and operator workflows
  - [ ] Milestone 8.1: Rebase `WeightPublisher` onto explicit rollout or
        `register(..., strategy=...)`.
  - [ ] Milestone 8.2: Keep `WeightPublisher` thin:
    - issue
    - wait for worker barrier
    - trigger reload / activation
    - GC old versions
  - [ ] Milestone 8.3: Ensure `wait_persistence` remains durability-oriented and
        is not repurposed as rollout completion.
  - [ ] Milestone 8.4: Update rollout-facing docs once the code path is real.

# Concrete Work Items

- Add `StableRealizationService` and narrow `LocalStableTierService` into a
  registration adapter.
- Add `RealizeSetAction` and wire it through:
  - `plan.proto`
  - SDK plan builders
  - local Python plan execution
  - daemon ingress plan execution
- Extend set-result carriers rather than adding a second rollout-private result
  family.
- Add daemon-served directory and capacity RPCs plus SDK signal wrappers.
- Add `RolloutController` and `StartRollout`.
- Add SDK `Runtime.rollout(...)`.
- Reuse generic `Operation[T]` and `DaemonGlobalStoreOperation` for rollout
  observation.
- Add `RegisterArtifactOptions.realization_strategy`.
- Add `register(..., strategy=...)`.
- Add `RegisteredArtifact.realization_operation()`.
- Keep `put(..., strategy=...)` out of the first wave.
- Add typed rollout barrier outputs for `0105`.

# Test, Rollout, and Backout

## Test Plan

- Worker-local realization semantics
  - prove `prefetch_set` still means `local_replica_ready`
  - prove `realize_set(..., local_stable_ready)` reaches true stable admission or
    reports honest degradation / failure
  - prove local runner and daemon ingress agree on readiness semantics
- Directory and target resolution
  - prove daemon-served `list_workers` / `list_instances` / `get_worker_capacity`
    expose freshness metadata and fail closed on stale data
  - prove rollout compilation resolves `daemon_id -> address` only through daemon
    caches
- Path selection
  - cover:
    - CPU memfd local ingress
    - GPU direct or staged RDMA pull to stable DRAM
    - CPU direct write fallback
    - disk fallback when source capability is unavailable
- Rollout orchestration first wave
  - cover:
    - worker-only rollout to `workers_ready`
    - `drop_on_barrier` source closeout
    - rollout observation through generic `Operation[T]`
- Strategy integration
  - prove `register(..., strategy=...)` lowers to the same worker-realization
    semantics as explicit rollout compilation
  - prove invalid strategy combinations fail fast
- Helper migration
  - adapt `WeightPublisher` coverage so rollout barrier and persistence barrier
    remain distinct assertions

## Rollout

1. Land semantic and target-model docs first.
2. Land daemon-served directory and stable realization kernel before rollout
   helpers.
3. Land `realize_set` before any issue-time strategy sugar.
4. Land worker-only rollout barrier before optional child publication or hydrate
   stages.
5. Migrate deployment helpers only after the worker barrier is stable and
   observable through `Operation[T]`.

## Backout

- Keep `prefetch_set` unchanged.
- Keep `publish` semantics from `0103` unchanged.
- Keep current persistence semantics unchanged.
- If rollout controller work regresses, callers can still:
  - issue artifacts normally
  - use explicit worker `prefetch_set` / `realize_set`
  - trigger activation out of band
- If issue-time strategy work regresses, keep explicit rollout or explicit plan
  composition as the canonical path.

# Risks and Tracking

- Risk: `realize_set` collapses into a second warmup action.
  - Tracking: keep readiness kind explicit and keep `prefetch_set` semantics
    frozen in tests and docs.
- Risk: rollout introduces a second execution substrate.
  - Tracking: require rollout compilation to emit canonical `PlanSpec` fragments
    only.
- Risk: target resolution forks into a daemon path and an SDK-direct GS path.
  - Tracking: make daemon-served directory closure a prerequisite, not an
    optional refinement.
- Risk: persistence state is again mistaken for rollout readiness.
  - Tracking: add explicit negative tests and distinct result fields.
- Risk: `strategy=` is mistaken for a replacement of `plan` or `policy`.
  - Tracking: keep the layer split explicit in docs, validation, and tests.
- Risk: `put(..., strategy=...)` ships before source semantics are honest.
  - Tracking: keep it out of the first wave.
- Risk: rollout child publication / hydration gets shipped before daemon ingress
  can host local instance-target execution.
  - Tracking: keep those milestones explicitly follow-on and gate them on ingress
    readiness.
