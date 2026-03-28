---
slug: source-to-serving-builder-and-representation-publication
title: Source-to-Serving Builder and Representation Publication Plan
status: in_progress
areas: ["core", "daemon", "sdk", "integrations", "docs", "tests"]
created: 2026-03-25
last_updated: 2026-03-27
related_code:
  - docs/designs/0111-source-to-serving-builder-and-representation-publication.md
  - docs/designs/0110-artifact-representation-contract-and-transform-unification.md
  - docs/designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md
  - docs/designs/0084-binding-unified-model-and-contract.md
  - docs/designs/0085-distributed-binding-assembly-and-coordinator.md
  - docs/internals/model-loading.md
  - tensorcast/types.py
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/README.md
  - tensorcast/api/store/serving_builder.py
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/publication/v1/publication.proto
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/service/controllers/assembly_coordination_utils.cc
  - daemon/service/controllers/serving_artifact_manifest_utils.{h,cc}
links:
  design: ../designs/0111-source-to-serving-builder-and-representation-publication.md
---

# Objective

Track only the remaining follow-up after the repo-owned `0111` bridge landed.
The TensorCast-owned contract / carrier / helper / daemon-validation work is
already implemented in this repository. This document now tracks only the
remaining follow-on.

# Current Status

- Repo-owned `0111` implementation is landed in `tensorcast-280`.
- The remaining open items are no longer "finish the base implementation in
  this repo"; they are either:
  - repo-local hardening follow-on,
  - external integration convergence,
  - or evidence / rollout gated work.

# Remaining Repo-Local Follow-On

- Enforce stronger runtime-only finalize invariants beyond the current
  support-level and typed-carrier gates.
- Add any remaining bootstrap acceptance / runtime preflight coverage that is
  still repo-owned and does not depend on family-specific external evidence.

# External / Integration Follow-On

- Continue `/data/workspace/internal-vllm` migration away from any remaining
  legacy helper shapes.
- Move framework-fact ownership, semantic probes, and family-specific readiness
  inputs toward the shared Torch integration layer rather than this repo.
- Continue integration rollout work for:
  - finalize classification
  - realization protocol reporting
  - fast-path validation reporting separate from support level
  - topology-sensitive admission facts
  - hash migration
  - admission-level rollout
  - source-bind bootstrap remaining only as an explicit migration/bootstrap
    mode

# Evidence / Rollout Gated Follow-On

- Admit `SAME_BINDING_FAST_PATH` only for families with family-specific
  correctness evidence.
- Keep semantic validation gates for `BINDING_FINALIZE` families evidence-driven
  rather than turning them into generic placeholder checks.
- Keep topology-sensitive family admission above semantic/build identity unless
  there is concrete evidence that a topology fact changes canonical serving
  bytes.

# Guardrails

- Do not start topology-scoped TP4<->TP8 executor work in this plan.
- Do not introduce broad durable artifact-catalog schema expansion in this
  plan.
- Do not move integration-owned family readiness inventory wholesale into
  daemon/runtime in this plan.
- Do not silently widen `serving_build_digest`; any future widening must come
  with explicit compatibility/versioning.
- Do not let same-binding fast-path language become a universal contract for
  every admitted family.
