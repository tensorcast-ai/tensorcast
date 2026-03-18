---
slug: workflow-companion-admission-and-fencing
title: Workflow Semantic Plane for Gate, Observation, Replay, and Fencing
status: implemented
areas: ["daemon", "sdk", "docs", "tests"]
created: 2026-03-09
last_updated: 2026-03-17
related_code:
  - docs/designs/0093-backing-identity-and-retained-backing-ownership.md
  - docs/designs/0100-distributed-authority-handoff-security-and-public-surfaces.md
  - docs/designs/0094-unified-lifecycle-kernel-and-capability-families.md
  - docs/designs/0055-programmable-framework.md
  - docs/designs/0060-tensor-work-queue.md
  - daemon/state/lifecycle_kernel.h
  - daemon/service/controllers/target_publish_service.cc
  - daemon/state/target_publication_registry.h
  - daemon/state/target_publication_registry.cc
  - proto/tensorcast/operation/v1/operation.proto
  - tensorcast/api/operation.py
links:
  plan: ../plans/0096-workflow-companion-admission-and-fencing.md
  dependencies:
    - ./0093-backing-identity-and-retained-backing-ownership.md
    - ./0100-distributed-authority-handoff-security-and-public-surfaces.md
    - ./0094-unified-lifecycle-kernel-and-capability-families.md
    - ./0055-programmable-framework.md
    - ./0060-tensor-work-queue.md
---

# Summary

`0100` owns the distributed-authority substrate:

- routed request and owner-reply shape,
- shared distributed trust and reply admissibility,
- canonical path-family ownership,
- and public continuation rules.

`0093` owns the authority-to-dataplane bridge:

- `ResolvedSourceCapability` is the shared result shape used when a validated path is ready to lower into shared
  execution.

`0094` owns the local lifecycle baseline:

- lifecycle state holds bounded runtime promises,
- `WorkflowCompanionRef` is the lifecycle-facing pointer to workflow state,
- and `LifecycleUseGuard` marks bounded active use.

With those boundaries established, `0096` remains intentionally semantic-only.
It owns:

- workflow gate,
- workflow observation,
- workflow completion,
- workflow recovery honesty,
- and the mapping from workflow-owned answers into the `0100` owner-reply algebra.

This rewrite narrows the active closeout goal:

- `publish` is the first workflow owner that must close end to end,
- `0055` operation semantics and `0060` queue semantics remain calibration constraints,
- and no additional workflow owner should be activated until `publish` is fully projected through `Operation[T]`.

# Implementation Status

Implemented publish-first closeout on top of `0100`.

Current code now provides:

- shared local workflow-semantic carrier structs live in `lifecycle_kernel.h`,
- `TargetPublicationRegistry` keeps replay and currentness outside lifecycle state,
- `TargetPublishService` binds lifecycle capabilities through `WorkflowBindingProjection -> WorkflowCompanionRef`,
- `publish` replay now closes through `attach_existing -> Operation[T]`,
- daemon operation observation fails closed on publish owner loss and stale attachment,
- and `publish` remains the first grounded workflow owner with a routed sample path through `0100`.

# Dependency Readiness

`0096` now distinguishes semantic readiness from dependency readiness.

| Workflow owner or family | Semantic shape | Dependency readiness |
| --- | --- | --- |
| `publish` gate and currentness semantics | defined | ready |
| `publish` replay to immediate terminal reuse | defined | ready |
| `publish` replay to `attach_existing -> Operation[T]` | defined | ready |
| `publish` observation to `retry_later` | declared | not dependency-ready |
| `0055` operation plane as calibration | defined | calibration only |
| `0060` queue workflows as calibration | defined | calibration only |

Repository rule:

- child designs may depend on `publish` semantic gate, terminal mapping, and the first `attach_existing -> Operation[T]`
  closeout today,
- but they must not treat declared `wait_not_ready` continuation as dependency-ready.

# Problem Statement

## 1. The repository still needs one workflow-semantic contract

The architecture already knows that replay, currentness, wait, join, cancel, and status are not lifecycle state.
What it still lacks is one dependency-ready contract for how a real workflow owner projects those answers.

## 2. `WorkflowCompanionRef` is persisted binding, not distributed attach identity

`WorkflowCompanionRef` is the accepted lifecycle-facing binding carrier.
It is enough to persist which workflow state a capability refers to.

It is not, by itself:

- a distributed route handle,
- a public attach contract,
- or a substitute for the `0100` continuation surface.

## 3. `publish` is grounded, but not yet closed end to end

Current code already proves:

- `publish` separates current-for-target from lifecycle admission,
- replay and currentness live in `TargetPublicationRegistry`,
- and the local workflow recovery class is `ephemeral_process_local`.

But the repository still lacks one full path that proves:

- a workflow replay hit can become `attach_existing`,
- ingress can project that to `Operation[T]`,
- and owner loss is surfaced honestly.

# Goals / Non-Goals

## Goals

- Define one shared workflow-semantic contract.
- Keep `0096` semantic-only and keep routing or public continuation rules in `0100`.
- Close `publish` as the first workflow owner end to end.
- Make workflow recovery class and owner-loss mapping explicit.
- Keep `WorkflowCompanionRef` as the only persisted lifecycle-facing workflow binding in this phase.
- Keep `0055` operation semantics and `0060` queue semantics representable as calibration constraints.

## Non-Goals

- Define workflow-owner routing, hop auth, or public continuation rules. `0100` owns those.
- Move workflow truth into lifecycle tables.
- Activate additional workflow owners before `publish` is closed.
- Turn `0055` operation semantics or `0060` queue semantics into the first closeout owner in this phase.
- Split operation-style replay, wait, cancel, or status into a second top-level authority plane.

# Architecture & Interfaces

## 1. Workflow boundary

Workflow semantic state answers questions such as:

- is this request current,
- is this a replay or join,
- is this caller fenced,
- should an existing outcome be reused,
- should a caller attach to an existing outcome,
- what status should be surfaced,
- what terminal semantic result should be recorded.

Lifecycle state answers a different question:

- does a bounded runtime promise currently exist and may it be activated or used.

Boundary rules:

1. workflow truth remains outside lifecycle ownership,
2. `WorkflowCompanionRef` remains the lifecycle-facing binding carrier, not the workflow truth owner,
3. `0096` defines workflow-owned answer space,
4. `0100` owns path families, distributed routing, and public continuation rules,
5. `0093` owns `ResolvedSourceCapability` as the bridge into shared lowering.

## 2. First-owner policy

`publish` is the first workflow owner that must close end to end.

Repository rule:

- no additional workflow owner should be activated on a new public continuation path until `publish` is closed and
  tested,
- `0055` operation semantics and `0060` queue semantics remain calibration constraints used to validate contract
  generality, not to expand implementation scope in this phase.

## 3. Publish semantic contract

### 3.1 Current grounded semantic state

Current grounded publish workflow state is:

| Semantic state | Current owner | Recovery class |
| --- | --- | --- |
| current-for-target | `TargetPublicationRegistry.latest_by_target_` | `ephemeral_process_local` |
| replay by `publication_id` | `TargetPublicationRegistry.records_` | `ephemeral_process_local` |

Critical rules:

1. current-for-target remains workflow truth, not lifecycle truth,
2. replay remains workflow truth, not lifecycle truth,
3. because recovery is `ephemeral_process_local`, owner loss must fail honestly rather than pretending attachability
   survives restart or failover.

### 3.2 Publish outcome classes

The first closeout phase standardizes these publish-owned semantic outcomes:

| Outcome | Meaning |
| --- | --- |
| `admit_new` | no reusable publish outcome exists and the request is allowed to proceed |
| `replay_terminal` | a reusable terminal outcome already exists |
| `replay_attachable` | an attachable existing outcome exists |
| `stale_current` | request is not current for the target |
| `fenced` | caller is semantically fenced |
| `cancelled` | workflow owner marks the operation cancelled |
| `owner_lost` | previously attachable publish-owned state was lost according to recovery class |
| `wait_not_ready` | workflow has a non-terminal observation but no attachable existing outcome yet |

## 4. Publish mapping to `0100`

### 4.1 Canonical mapping table

| Publish workflow outcome | `0100` owner reply | Public projection | Readiness |
| --- | --- | --- | --- |
| `admit_new` | `continue_with_authority` on `gate, continue, then adopt` | later immediate lowering result after issuer adoption | ready |
| `replay_terminal` | `terminal` | immediate reused result | ready |
| `replay_attachable` | `attach_existing` | `Operation[T]` | first closeout target |
| `stale_current` | `terminal` | immediate semantic reject | ready |
| `fenced` | `terminal` | immediate semantic reject | ready |
| `cancelled` | `terminal` | immediate semantic reject | ready |
| `wait_not_ready` | `retry_later` | `Operation[T]` or another explicit public family | declared but not dependency-ready |

Rules:

1. `publish` does not define repository-wide path-family order; it reuses `0100`,
2. `replay_attachable` is the first workflow-owned non-terminal public closeout target,
3. `wait_not_ready` remains out of dependency-ready scope until the first attach path is closed,
4. `publish` must not expose `AuthorityAttachmentRef` directly to callers.

### 4.2 Owner-loss mapping

Because the current publish workflow recovery class is `ephemeral_process_local`, owner loss is mapped as follows:

- if a caller is attached to an existing publish-owned outcome and the owner loses that state, later observation must
  fail closed with `UNAVAILABLE` or another explicit owner-loss error,
- the system must not treat owner loss as if replay or currentness were still provable,
- a fresh root request after owner loss may re-evaluate semantic state from scratch and may no longer observe a replay
  hit.

This is stricter than optimistic reattach.
That strictness is required for honest recovery semantics.

## 5. Calibration constraints

### 5.1 `0055` operation plane

Long-term interpretation:

- semantic replay, join, wait, cancel, and status remain operation-owned,
- attachable existing-operation outcomes must use the `0100` continuation surface rather than a workflow-private
  routing model,
- bare `operation_id` is still not sufficient for distributed attach without public-safe continuation metadata.

### 5.2 `0060` queue workflows

Long-term interpretation:

- queue fencing, visibility, and blocking remain workflow-owned,
- queue semantics validate the contract but are not the first closeout target in this phase.

# Proto and SDK Implications

This design does not create a new public continuation family.

It depends on `0100` to provide:

- a public-safe continuation descriptor carried by `OperationRef` or an equivalent public surface,
- and `Operation[T]` as the caller-visible continuation shape.

This design requires:

- workflow-owned outcomes to map into `0100` owner replies,
- and `publish` to use that public continuation contract without introducing a family-private attach surface.

No persistent schema changes are required by this design itself.

# Error Model

- replay hit with attachable existing outcome
  - workflow-owned `replay_attachable`
  - maps to `attach_existing`
- replay hit with terminal reusable outcome
  - workflow-owned `replay_terminal`
  - maps to `terminal`
- stale current
  - workflow-owned semantic reject
- fenced
  - workflow-owned semantic reject
- cancelled
  - workflow-owned semantic reject
- owner loss for `ephemeral_process_local` attached publish state
  - explicit owner-loss error such as `UNAVAILABLE`
- wait not ready
  - workflow observation result, not lifecycle expiry

# Observability

Minimum required dimensions:

- `family`
- `workflow_owner_kind`
- `workflow_recovery_class`
- `workflow_decision_class`
- `workflow_projection_kind`
- `workflow_attachment_kind`

# Naming Compliance

| Interface | Language | Rule | Result |
| --- | --- | --- | --- |
| `WorkflowRecoveryClass` | C++ enum class | `PascalCase` | pass |
| `WorkflowIssueContext` | C++ struct | `PascalCase` | pass |
| `WorkflowBindingProjection` | C++ struct | `PascalCase` | pass |
| `WorkflowRedemptionContext` | C++ struct | `PascalCase` | pass |
| `WorkflowGateDecision` | C++ struct | `PascalCase` | pass |
| `WorkflowObservationResult` | C++ struct | `PascalCase` | pass |
| `WorkflowOutcomeProjection` | C++ struct | `PascalCase` | pass |

# Trade-offs & Risks

- Narrowing active scope to `publish` makes `0096` less broad in the short term.
  That is intentional because the repository now needs one dependency-ready workflow owner more than it needs more
  partially specified owner families.
- Keeping `wait_not_ready` outside the first closeout slows breadth, but it prevents `attach_existing` and
  `retry_later` from drifting simultaneously.
- Honest owner-loss behavior for `ephemeral_process_local` state is stricter than optimistic attach reuse.
  That strictness is correct.

# Compatibility & Acceptance Criteria

Acceptance criteria:

- `0096` remains semantic-only and depends on `0100` for public continuation rules,
- `publish` is the first fully closed workflow owner,
- `publish` replay to `attach_existing -> Operation[T]` is landed end to end,
- `WorkflowCompanionRef` remains the only persisted lifecycle-facing workflow binding,
- workflow replay, wait, status, and currentness remain outside lifecycle state,
- owner-loss behavior is explicit and consistent with `WorkflowRecoveryClass`,
- `0055` operation semantics and `0060` queue semantics remain representable without becoming the first implementation
  target,
- no workflow-private public continuation model appears beside `Operation[T]`.

# References

- `0100` for distributed routing, path-family ownership, attach refs, owner-reply algebra, and public continuation.
- `0093` for `ResolvedSourceCapability`.
- `0094` for lifecycle-facing workflow binding and active-use boundaries.
- `0055` for `Operation[T]` as the public continuation surface.
- `0060` for queue fencing and blocking semantics.
