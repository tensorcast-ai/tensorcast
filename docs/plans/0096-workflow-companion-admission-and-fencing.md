---
slug: workflow-companion-admission-and-fencing
title: Plan - Workflow Semantic Plane Remaining Work
status: in_progress
areas: ["daemon", "sdk", "docs", "tests"]
created: 2026-03-09
last_updated: 2026-03-16
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
  - tensorcast/api/operation.py
  - tensorcast/api/store/handles.py
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/registration.py
  - tests/python/api/test_operation_semantics.py
  - tests/python/api/test_persistence_wireup.py
  - tensorcast/tools/weight_publisher.py
  - tests/python/tools/test_weight_publisher_pretrim.py
links:
  design: ../designs/0096-workflow-companion-admission-and-fencing.md
---

# Objective

Finish the workflow-semantic closeout on top of the consolidated `0100` distributed-authority substrate.

The remaining work is to make workflow-owned attach, replay, wait, and status project cleanly to `Operation[T]`
without leaking daemon-internal routed carriers, then choose any next routed workflow owner only after that public
surface is honest.

# Current State & Grounding

- local workflow-semantic carriers, `WorkflowBindingProjection`, and explicit `WorkflowRecoveryClass` are already landed.
- `publish` remains the grounded local workflow owner and `target_publication` is already the first routed workflow
  sample path.
- queue-style not-ready, fenced, and stale-current outcomes already stay in workflow vocabulary rather than collapsing
  into lifecycle expiry or not-found.
- the remaining gap is the public continuation closeout: the first complete `0055` attach or replay or status path
  needs to land on `Operation[T]` end to end without exposing `AuthorityAttachmentRef`.

# Remaining TODOs

- [ ] Finish the first workflow-owned `attach_existing` or `retry_later` projection to `Operation[T]`.
- [ ] Add workflow tests that verify replay or attach or status projection through `Operation[T]` without exposing
  `AuthorityAttachmentRef` or other daemon-internal routed carriers.
- [ ] Keep queue, fencing, and non-admit workflow outcomes in workflow vocabulary while the public continuation path is
  finalized.
- [ ] Choose and land the next routed workflow owner only after the public `Operation[T]` projection and tests are
  closed.

# Acceptance Checks

- [ ] workflow replay, wait, status, and currentness remain outside lifecycle state.
- [ ] `WorkflowCompanionRef` remains the only persisted lifecycle-facing workflow binding.
- [ ] caller-visible continuation converges on `Operation[T]`.
- [ ] no workflow-private public continuation model appears beside `Operation[T]`.
