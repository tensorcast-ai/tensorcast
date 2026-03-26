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
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/service/controllers/assembly_coordination_utils.cc
links:
  design: ../designs/0111-source-to-serving-builder-and-representation-publication.md
---

# Objective

Track only the remaining follow-up after the initial `0111` bridge landed.
Completed rollout work has been folded back into the design, SDK docs, and
tests. This plan now tracks the unfinished bootstrap-builder,
finalize-admission, and integration-convergence work.

# Current State & Grounding

- `0110` remains the owner of semantic transform truth.
- `0111` now owns the builder and publication bridge on top of that truth, but
  the remaining open work is still limited to:
  - broader repository-owned builder orchestration beyond the current exact-copy
    `PURE_TRANSFORM` paths
  - node-local bootstrap builder semantics
  - `BINDING_FINALIZE` admission and support-level rollout
  - integration-side builder/publication identity convergence in
    `/data/workspace/internal-vllm`
- This plan must not expand into topology-scoped execution or durable artifact
  catalog schema work.

# Phases & Milestones

- [ ] Phase 1: Finish The `PURE_TRANSFORM` Boundary Cleanup
  - [ ] Milestone 1.1: Make source-to-serving `PURE_TRANSFORM` builds consume
    only repo-owned `0110` and `0111` carriers end-to-end.
  - [ ] Milestone 1.2: Mark integration-private manifest and hash carriers as
    target-state removals in docs.
  - [ ] Milestone 1.3: Update `0102` and integration-facing docs so serving
    closeout ownership and TensorCast-owned manifest/hash carriers are explicit.
- [ ] Phase 2: Support Node-Local Bootstrap Builder Without Making It Runtime Truth
  - [ ] Milestone 2.1: Define the node-local bootstrap builder workflow as a
    temporary or bootstrap build mode.
  - [ ] Milestone 2.2: Require bootstrap builder completion to seal or register a
    serving artifact before steady-state runtime switches to it.
  - [ ] Milestone 2.3: Keep source-bind bootstrap documented as different from
    steady-state serving bind or swap.
- [ ] Phase 3: Add `BINDING_FINALIZE` With Explicit Admission Gates
  - [ ] Milestone 3.1: Define the minimal framework facts consumed from the
    shared Torch layer and `FinalizeClass`.
  - [ ] Milestone 3.2: Enforce runtime-only finalize invariants before any family
    is admitted to steady-state serving bind or swap.
  - [ ] Milestone 3.3: Add semantic validation gates for
    `BINDING_FINALIZE` families before publication is admitted.
  - [ ] Milestone 3.4: Introduce `ServingSupportLevel`-based admission and keep
    blocked or bootstrap-only families out of steady-state serving publish.
- [ ] Phase 4: Integration Convergence
  - [ ] Milestone 4.1: Move `internal-vllm` off private builder/publication
    identity computation to `representation_contract_hash` plus
    `serving_build_digest`.
  - [ ] Milestone 4.2: Keep source-bind bootstrap only as an explicit migration
    or bootstrap mode, not as a second steady-state contract.

# Tasks

## Documentation tasks

- [ ] Update `docs/designs/0102-engine-artifact-integration-and-high-cardinality-manifest-orchestration.md`
  - keep engine projection ownership separate from serving closeout ownership.
- [ ] Update integration-facing docs in `/data/workspace/internal-vllm`
  - replace private manifest or hash ownership language with TensorCast-owned
    carriers.

## Core code tasks

- [ ] Add builder-side orchestration helpers for:
  - node-local bootstrap build
  - `BINDING_FINALIZE`

## Daemon and closeout tasks

- [ ] Add finalize-class and support-level enforcement to the serving
  publication admission path.
- [ ] Keep version-key activation and runtime bind/swap admission gated on the
  final support-level policy.

## External integration coordination tasks

- [ ] Coordinate `/data/workspace/internal-vllm` adapter work for:
  - finalize classification
  - hash migration
  - admission-level rollout

## Deferred follow-on tasks

- [ ] Do not start topology-scoped TP4<->TP8 executor work in this plan.
- [ ] Do not introduce broad durable artifact-catalog schema expansion in this
  plan.

# Test / Rollout / Backout

## Test plan

- [ ] Add bootstrap-builder acceptance coverage for serving-artifact switch.
- [ ] Add `BINDING_FINALIZE` admission coverage for explicitly validated
  families only.
- [ ] Add any remaining support-level runtime preflight coverage required before
  admitting new families.

## Rollout

- [ ] Roll out in strict order:
  - bootstrap builder
  - `BINDING_FINALIZE`
  - admitted-family runtime bind or swap convergence
- [ ] Keep source-bind bootstrap available only as an explicit migration mode.
- [ ] Do not activate additional admitted families until support-level and
  finalize validation gates exist.
- [ ] Do not preserve a second private publication truth in integrations.

## Backout

- [ ] Backout is branch-level revert of incomplete bootstrap or finalize work.
- [ ] Do not add a second long-lived manifest or hash format as a compatibility
  shell.
- [ ] If `BINDING_FINALIZE` proves under-specified, stop at `PURE_TRANSFORM` and
  bootstrap-builder support rather than weakening admission rules.

# Risks & Tracking

- [ ] Risk: finalize classification is too optimistic.
  - Mitigation: require explicit `FinalizeClass` and semantic validation before
    publication or steady-state admission.
- [ ] Risk: integrations continue to own private builder/publication identity.
  - Mitigation: make TensorCast-owned manifest and hash semantics mandatory
    before admitted serving publication.
- [ ] Risk: bootstrap builder lingers as de facto steady-state runtime truth.
  - Mitigation: keep support-level rollout explicit and measure convergence to
    serving-artifact bind or swap.
