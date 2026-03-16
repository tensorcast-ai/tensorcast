---
slug: distributed-authority-handoff-security-and-public-surfaces
title: Plan - Unified Distributed Authority Remaining Work
status: in_progress
areas: ["daemon", "sdk", "proto", "docs", "tests"]
created: 2026-03-10
last_updated: 2026-03-16
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
  - tensorcast/api/operation.py
links:
  design: ../designs/0100-distributed-authority-handoff-security-and-public-surfaces.md
---

# Objective

Close the remaining distributed-authority work after folding the former `0095`, `0099`, and `0101` streams into
`0100`.

The only active gap should now be the public continuation closeout on top of the already landed front-door seam,
routed protocol, security kernel, and routed issuer path.

# Current State & Grounding

- `FrontDoorCredentialContext`, `PortableParsedCredential`, and forwardable evidence are already the active routed front-door seam.
- canonical routed request or reply types and additive `RouteAuthorityStage` RPCs are already landed.
- `DistributedSecurityKernel` already enforces transport-derived peer identity, authority binding, delegated disclosure,
  and fail-closed reply admission.
- the first routed issuer child path is already active on top of that substrate.
- the remaining repository gap is not another lower-layer split; it is proving the first honest
  `attach_existing` or `retry_later` projection from `OwnerStageReply` to `Operation[T]` and finishing doc ownership
  cleanup so `0100` is the only active distributed-authority design.

# Remaining TODOs

- [ ] Prove the first family-level projection from `OwnerStageReply(attach_existing|retry_later)` to `Operation[T]`
  end to end on the `0096` consumer path.
- [ ] Add the remaining tests that verify public projection into `Operation[T]` without exposing
  `AuthorityAttachmentRef` or `OwnerStageReply` at the SDK boundary.
- [ ] Finish doc ownership cleanup so foundational designs describe `0100` as the single active owner of
  distributed-authority ingress, issuer handoff, routed protocol, and distributed trust semantics.
- [ ] Keep any next routed child on the same `0100` path-family and reply algebra instead of introducing a child-local
  protocol.

# Acceptance Checks

- [ ] `0100` is the only active design for distributed-authority ingress, routed protocol, issuer handoff, and trust.
- [ ] public long-lived continuation still converges on `Operation[T]` or another explicit family surface.
- [ ] no raw daemon-internal routed carriers cross the SDK boundary.
- [ ] byte-moving distributed success still terminates in `ResolvedSourceCapability` plus ingress-owned lowering.
