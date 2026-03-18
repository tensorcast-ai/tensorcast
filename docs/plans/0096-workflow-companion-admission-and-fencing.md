---
slug: workflow-companion-admission-and-fencing
title: Plan - Workflow Semantic Plane Publish-First Closeout
status: in_progress
areas: ["daemon", "sdk", "docs", "tests"]
created: 2026-03-09
last_updated: 2026-03-17
related_code:
  - docs/designs/0096-workflow-companion-admission-and-fencing.md
  - docs/designs/0100-distributed-authority-handoff-security-and-public-surfaces.md
  - docs/designs/0093-backing-identity-and-retained-backing-ownership.md
  - docs/designs/0094-unified-lifecycle-kernel-and-capability-families.md
  - docs/designs/0055-programmable-framework.md
  - docs/designs/0060-tensor-work-queue.md
  - daemon/state/lifecycle_kernel.h
  - daemon/state/target_publication_registry.h
  - daemon/state/target_publication_registry.cc
  - daemon/service/controllers/target_publish_service.cc
  - proto/tensorcast/operation/v1/operation.proto
  - tensorcast/api/operation.py
  - tests/python/api/test_operation_semantics.py
links:
  design: ../designs/0096-workflow-companion-admission-and-fencing.md
---

# Objective

Finish the first workflow-owner closeout on top of `0100` by using `publish` as the single active implementation
target.

The target is:

- one real workflow replay or attach path,
- mapped through `0100`,
- into caller-visible `Operation[T]`,
- with owner-loss behavior that matches the declared recovery class.

# Current State & Grounding

- local workflow-semantic carriers, `WorkflowBindingProjection`, and explicit `WorkflowRecoveryClass` are already landed.
- `publish` remains the grounded local workflow owner and the first routed workflow sample path.
- `TargetPublicationRegistry` already keeps replay and currentness outside lifecycle state.
- the remaining gap is not generic workflow vocabulary.
  It is the first honest public projection for a real workflow owner.
- `0100` is now the direct dependency for the public continuation contract.

# Phases and Milestones

- [x] Phase 0: Scope freeze
  - [x] Milestone 0.1: freeze `publish` as the first workflow-owner closeout target.
  - [x] Milestone 0.2: keep `0055` operation semantics and `0060` queue workflows as calibration constraints only.
  - [x] Milestone 0.3: keep `wait_not_ready` declared but out of dependency-ready scope for this phase.

- [x] Phase 1: Publish semantic mapping freeze
  - [x] Milestone 1.1: lock the publish outcome set:
        `admit_new`, `replay_terminal`, `replay_attachable`, `stale_current`, `fenced`, `cancelled`, `owner_lost`,
        `wait_not_ready`.
  - [x] Milestone 1.2: map those outcomes to `0100` owner replies and public projection classes.
  - [x] Milestone 1.3: lock owner-loss mapping for `ephemeral_process_local` publish-owned state.

- [x] Phase 2: First end-to-end public continuation
  - [x] Milestone 2.1: land `publish` replay to `attach_existing -> Operation[T]`.
  - [x] Milestone 2.2: ensure no workflow-private continuation surface appears beside `Operation[T]`.
  - [x] Milestone 2.3: keep `WorkflowCompanionRef` as the only persisted lifecycle-facing workflow binding.

- [x] Phase 3: Hardening and tests
  - [x] Milestone 3.1: add tests for replay attach projection, terminal replay reuse, stale current, fenced, and
        cancelled outcomes.
  - [x] Milestone 3.2: add tests for owner-loss behavior under `ephemeral_process_local`.
  - [x] Milestone 3.3: verify workflow answers stay outside lifecycle state.

- [x] Phase 4: Unblock next workflow owners
  - [x] Milestone 4.1: mark `publish` workflow continuation dependency-ready after tests land.
  - [x] Milestone 4.2: keep queue and next workflow owners blocked until the first publish closeout is complete.

# Latest Status

Implemented on top of the `0100` continuation contract:

- `publish` now exposes a real caller-visible continuation path through `StartPublishTargetReplica -> Operation[T]`
  and SDK `publish_replica_operation()` on slot and binding surfaces.
- publish workflow gate now returns `attach_existing` for in-flight replay and terminal projection for already-terminal
  replay.
- daemon `GetOperation` / `WaitOperation` now fail closed for publish owner loss and stale attachment under
  `ephemeral_process_local`.
- `WorkflowCompanionRef` remains the only persisted lifecycle-facing workflow binding; replay/currentness stay in
  `TargetPublicationRegistry`.
- legacy synchronous `PublishTargetReplica` remains as a compatibility path, but the new non-terminal continuation
  contract is carried only through `Operation[T]`.

# Tasks

- Workflow owner:
  - keep currentness and replay in `TargetPublicationRegistry`.
  - keep lifecycle admission separate from workflow-owned currentness and replay.
  - make owner-loss behavior explicit and fail closed.

- Mapping to `0100`:
  - route `replay_attachable` through `attach_existing`.
  - route `replay_terminal`, `stale_current`, `fenced`, and `cancelled` through immediate terminal projection.
  - leave `wait_not_ready` out of dependency-ready scope in this phase.

- Public surface:
  - depend on `0100` for the public-safe continuation descriptor carried by `Operation[T]`.
  - do not add workflow-private attach or replay handles.

- Documentation:
  - keep `0096` semantic-only.
  - keep `publish` as the first and only closeout owner in this phase.

# Test / Rollout / Backout

Tests:

- Python:
  - `source .venv/bin/activate && pytest tests/python/api/test_operation_semantics.py`
- Additional focused tests:
  - publish replay attach projection
  - publish terminal replay reuse
  - publish stale-current and fenced rejection
  - owner-loss fail-closed behavior for `ephemeral_process_local`

Acceptance checks:

- `publish` replay attach reaches caller-visible `Operation[T]`,
- no workflow-private continuation model appears beside `Operation[T]`,
- `WorkflowCompanionRef` remains the only persisted lifecycle-facing binding,
- workflow state does not collapse into lifecycle tables,
- owner-loss behavior is explicit and matches `WorkflowRecoveryClass`.

Rollout:

- land `publish` closeout first.
- keep queue and other workflow owners out of dependency-ready use until publish is complete.

Backout:

- keep current publish semantic state local and explicit.
- disable the public attach projection rather than exposing partial workflow-private carriers.

# Risks and Tracking

- Risk: `publish` grows workflow-specific surface area that does not generalize.
  - tracking: keep all public projection through `Operation[T]` and all protocol ownership in `0100`.
- Risk: owner-loss behavior is softened into best-effort replay reuse.
  - tracking: require fail-closed tests for `ephemeral_process_local`.
- Risk: queue or other owners start building on the path before `publish` is truly closed.
  - tracking: keep them explicitly calibration-only in design and plan.
