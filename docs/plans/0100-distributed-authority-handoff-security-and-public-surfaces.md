---
slug: distributed-authority-handoff-security-and-public-surfaces
title: Plan - Unified Distributed Authority First Public Continuation Closeout
status: in_progress
areas: ["daemon", "sdk", "proto", "docs", "tests"]
created: 2026-03-10
last_updated: 2026-03-17
related_code:
  - docs/designs/0100-distributed-authority-handoff-security-and-public-surfaces.md
  - docs/designs/0096-workflow-companion-admission-and-fencing.md
  - docs/designs/0093-backing-identity-and-retained-backing-ownership.md
  - docs/designs/0094-unified-lifecycle-kernel-and-capability-families.md
  - docs/designs/0055-programmable-framework.md
  - daemon/state/distributed_security_kernel.h
  - daemon/state/routed_authority_protocol.h
  - daemon/service/grpc_service_impl_rpc_delegates.cc
  - daemon/service/controllers/transport_controller.cc
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/operation/v1/operation.proto
  - tensorcast/api/operation.py
links:
  design: ../designs/0100-distributed-authority-handoff-security-and-public-surfaces.md
---

# Objective

Close the first dependency-ready public continuation path on top of the already landed `0100` lower layers.

The target is no longer "more routed protocol surface."
The target is:

- one real `attach_existing` projection,
- into `Operation[T]`,
- with public-safe continuation metadata,
- and no leakage of daemon-internal routed carriers.

# Current State & Grounding

- `FrontDoorCredentialContext`, `PortableParsedCredential`, and forwardable evidence are already the active routed
  front-door seam.
- canonical routed request and owner-reply carriers and additive `RouteAuthorityStage` RPCs are already landed.
- `DistributedSecurityKernel` already enforces transport-derived peer identity, authority binding, delegated
  disclosure, and fail-closed reply admission.
- the first routed issuer child path is already active on top of that substrate.
- immediate `ready_for_lowering` and `terminal` paths are already dependency-ready.
- the remaining gap is the first non-terminal public continuation closeout.

Current practical blocker:

- existing public `OperationRef` shape is too weak to serve as an honest distributed attach contract by `operation_id`
  alone.

# Phases and Milestones

- [x] Phase 0: Scope freeze and dependency boundary
  - [x] Milestone 0.1: freeze `attach_existing` as the first public continuation closeout target.
  - [x] Milestone 0.2: keep `retry_later` declared but explicitly non-ready for dependent designs.
  - [x] Milestone 0.3: keep immediate lowering and terminal paths dependency-ready.

- [x] Phase 1: Public-safe continuation descriptor
  - [x] Milestone 1.1: define additive authority-scope and recovery metadata for `OperationRef` or equivalent public
        continuation payload.
  - [x] Milestone 1.2: ensure the public descriptor is sufficient for fail-closed reattach without exposing
        `AuthorityAttachmentRef`.
  - [x] Milestone 1.3: document exact mapping from public descriptor to owner-consumed reentry behavior.

- [x] Phase 2: First end-to-end `attach_existing` path
  - [x] Milestone 2.1: wire the first `OwnerStageReply(attach_existing)` path through ingress projection to
        `Operation[T]`.
  - [x] Milestone 2.2: use the `0096` publish replay or join path as the first concrete consumer.
  - [x] Milestone 2.3: ensure no bare `OwnerStageReply` or `AuthorityAttachmentRef` crosses the SDK boundary.

- [x] Phase 3: Hardening and failure honesty
  - [x] Milestone 3.1: add tests for owner loss, stale attachment, authority mismatch, and wrong-path fail-closed
        behavior.
  - [x] Milestone 3.2: verify `status`, `wait`, and `cancel` use the public continuation descriptor rather than bare
        `operation_id`.
  - [x] Milestone 3.3: verify recovery-class-specific error mapping is honest.

- [x] Phase 4: Documentation and dependent-design unblocking
  - [x] Milestone 4.1: mark `attach_existing` dependency-ready in `0100` after tests land.
  - [x] Milestone 4.2: keep `retry_later` blocked until it gets its own end-to-end closeout.
  - [x] Milestone 4.3: update dependent designs and plans to use only ready continuation paths.

# Latest Status

Implemented closeout for the first dependency-ready public continuation path:

- `OperationRef` now carries additive authority-scope, attachment-kind, recovery-class, and fencing metadata.
- `OperationContinuationMetadata` is persisted through operation snapshots so later `GetOperation` responses can
  reconstruct the public-safe continuation descriptor without a schema split.
- daemon `GetOperation` / `WaitOperation` now enforce publish continuation fail-closed admission on top of the public
  descriptor, covering owner loss, stale attachment, authority mismatch, and wrong-path metadata.
- the first real `attach_existing` producer is the `0096` publish workflow gate, and the first real public consumer is
  `StartPublishTargetReplica -> Operation[T]` / SDK `publish_replica_operation()` on slot and binding surfaces.
- `cancel()` still has no backend cancel RPC for Global Store operations, but it now consults the same continuation
  descriptor before returning fail-closed.

# Tasks

- Protocol and daemon:
  - keep `RoutedAuthorityRequest` and `OwnerStageReply` internal-facing.
  - preserve strict request or reply identity matching and fail-closed admission.
  - wire ingress projection from `attach_existing` into the public continuation surface.

- Operation public surface:
  - extend `proto/tensorcast/operation/v1/operation.proto` or an equivalent public continuation descriptor with
    authority-scope and recovery metadata.
  - update `tensorcast/api/operation.py` so distributed `status`, `wait`, and `cancel` retain and consume that
    descriptor.

- Integration path:
  - coordinate with `0096` so the first closeout path is the publish workflow replay or join path.

- Documentation:
  - keep declared path families separate from dependency-ready path families.
  - keep `0100` as the only active owner of distributed public continuation rules.

# Test / Rollout / Backout

Tests:

- Proto:
  - `bash tools/build_proto_python.sh`
  - `bazel test //proto/... --test_output=errors`
- SDK and daemon:
  - add focused tests for `attach_existing -> Operation[T]` projection
  - add focused tests for owner-loss and stale-attach fail-closed behavior

Acceptance checks:

- one concrete `attach_existing` path reaches caller-visible `Operation[T]`,
- no raw daemon-internal routed carrier crosses the SDK boundary,
- bare `operation_id` is no longer the only distributed attach identifier,
- immediate lowering paths remain unchanged,
- dependent designs can point to one dependency-ready continuation path instead of prose intent.

Rollout:

- do not activate new dependent continuation paths until the first path is closed.
- keep `retry_later` out of dependency-ready status until its own closeout lands.

Backout:

- keep immediate lowering and terminal paths active.
- disable the first public continuation projection rather than exposing partial daemon-internal carriers.

# Risks and Tracking

- Risk: public continuation metadata becomes another ad hoc public protocol.
  - tracking: require it to remain an additive `Operation[T]` contract rather than a parallel surface.
- Risk: `attach_existing` lands with insufficient owner-loss semantics.
  - tracking: require recovery-class-specific tests before marking dependency-ready.
- Risk: implementers reuse the partially closed path to justify `retry_later` or child-local continuation work.
  - tracking: keep readiness explicitly split in the design and dependent plans.
