---
slug: distributed-binding-assembly-and-coordinator-plan
title: Distributed Binding Assembly on the Existing Assembly and Layout Trunk Plan
links:
  design: ../designs/0085-distributed-binding-assembly-and-coordinator.md
areas: ["sdk", "daemon", "core", "proto", "global_store"]
related_code:
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/registration.py
  - tensorcast/api/_register.py
  - tensorcast/types.py
  - proto/tensorcast/layout/v1/layout.proto
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/operation/v1/operation.proto
  - schema.sql
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/registration_controller.cc
  - daemon/state/session_lifecycle.cc
  - core/store/components/global_store_client.h
  - core/store/components/global_store_client.cc
  - core/store/runtime/metadata/registration_backend.cc
  - tensorcast/global_store/repositories/assembly_slot_occupancy_repository.py
  - tensorcast/global_store/repositories/operation_repository.py
  - tensorcast/global_store/rpc/operation_rpc_handler.py
  - tensorcast/global_store/services/view_state_service.py
  - tests/python/test_binding.py
  - tests/python/test_assembly_attempt.py
  - tests/python/test_dense_piece_assembly_sealing_acceptance.py
  - tests/python/global_store/test_schema_persistence.py
  - tests/python/global_store/test_assembly_slot_occupancy_rpc.py
  - daemon/service/owned_binding_service_test.cc
  - daemon/service/grpc_service_impl_start_seal_assembly_test.cc
last_updated: 2026-03-18
---

# Objective

Close `0085` as the parent one-trunk design by:

- landing the attempt-domain hard cut from `0105`,
- converging all frontends onto one structural commit and seal trunk,
- and proving published-lineage semantics and parity across the accepted
  contract families.

This plan is now the umbrella integration plan.
It does not re-specify the executable attempt carriers.
Those are owned by the `0105` plan.

# Current State & Grounding

## Structural trunk that already exists

- `LayoutSpec` already defines immutable structural layout truth.
- `registration_controller.cc` and `registration_backend.cc` already compute
  deterministic `view_id`, coverage, proof material, and durable structural
  state.
- `view_state_service.py` already owns overlap and proof semantics.
- `assembly_id` and `artifact_bindings` already define the structural assembly
  workspace and post-seal binding.
- `StartSealAssembly` already drives the structural seal through the daemon.

## Useful implementation already landed

- `Store.start_assembly_attempt(...)` exists.
- `SubmitBindingContribution` exists.
- `assembly_slot_occupancies` exists as the durable occupancy projection.
- binding mutation paths already consult live-contribution state.
- `wait_assembly_attempt(...)` already decodes a `SealAssemblyResult` into
  `PublishedModelVersion`.

## What `0085` no longer owns directly

The following are no longer duplicated in this plan:

- immutable attempt-carrier shape,
- operation-snapshot split,
- `slot_key` public carrier definition,
- closeout-policy snapshotted carrier,
- and public continuation metadata.

Those are execution-owned by:

- [0105 design](../designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md)
- [0105 plan](./0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md)

## Remaining parent-level gaps after the `0105` hard cut

- shared structural commit helper parity is still incomplete,
- same-slot replacement still needs to be proven against the readiness-cut
  model,
- planner-backed `PP` and `EP` mapping still needs one canonical phase-1 story,
- `canonical_full` still needs to be kept explicit as a legal contract family,
- `PublishedModelVersion` lineage still needs full source and optional serving
  closeout parity,
- acceptance tests still do not prove that binding-backed contributors obey the
  same trunk semantics as direct piece registration.

# Execution Split To Lock

This plan assumes the following split and should be updated only in ways that
preserve it.

- `0105` plan owns the attempt-domain hard cut:
  - immutable spec
  - runtime projection
  - readiness cut
  - slot-key public identity
  - closeout-policy freeze
  - continuation hard cut
- `0085` plan owns one-trunk convergence:
  - shared structural helper extraction
  - cross-frontend parity
  - planner mapping and contract-family closure
  - final published-lineage parity
  - parent-level acceptance matrix and documentation closure

# Plan Decisions To Lock Before Coding

- One assembly trunk remains in the system.
- One layout contract trunk remains in the system.
- Attempt entry remains contract-first even if a phase-1 layout-seeded shorthand
  survives for compatibility.
- `LayoutSpec.expected_view_ids` remains only the layout-scoped seed for the
  disjoint `piece_partial` family.
- `canonical_full` remains a legal contract family member and does not become a
  full-coverage piece shortcut.
- Mutable closeout-policy configuration resolves on pre-attempt scope and is
  snapshotted before contributors are accepted.
- Structural commit semantics stay shared across frontends.
- Seal correctness consumes a captured readiness cut rather than post-cut
  ambient state.
- Published success means real lineage completion, not source seal alone.
- No training-only side path may be introduced.

# Non-Goals For This Execution Plan

- Do not define a second assembly implementation.
- Do not redefine the attempt-carrier hard cut outside `0105`.
- Do not widen `LayoutSpec` with topology or owner metadata.
- Do not make the SDK talk to Global Store directly.
- Do not promise `TP > 1` assembly in this phase.

# Phases & Milestones

- [ ] Phase 1: Consume the `0105` hard cut and re-baseline the trunk
  - [ ] Milestone 1: Land the `0105` plan so `0085` no longer depends on weak
    attempt carriers.
  - [ ] Milestone 2: Re-baseline all `0085`-owned docs and tests on the new
    immutable-spec, readiness-cut, and continuation rules.
  - [ ] Milestone 3: Remove any remaining one-trunk prose or helper logic that
    still talks as if slot identity and structural identity were the same.

- [ ] Phase 2: Extract one shared structural commit helper
  - [ ] Milestone 1: Refactor registration code so structural commit logic for
    deterministic identity, coverage, proof, and durable state is centralized.
  - [x] Milestone 2: Make binding-backed contribution and direct piece
    registration both lower onto that one helper path.
  - [x] Milestone 3: Keep `canonical_full` on the canonical structural path
    without forking publish semantics.

- [ ] Phase 3: Close phase-1 contract-family mapping
  - [ ] Milestone 1: Write down one canonical phase-1 planner-to-slot mapping
    for `PP` and `EP`.
  - [ ] Milestone 2: Ensure required `piece_partial` slots are derived once from
    deterministic planner output and not rebuilt ad hoc at publish time.
  - [ ] Milestone 3: Keep attempt entry contract-first even when
    `layout_id`-seeded shorthand is still supported for compatibility.
  - [ ] Milestone 4: Keep single-rank publish legal through explicit
    `canonical_full`.
  - [ ] Milestone 5: Keep `TP` explicitly deferred in code and docs.

- [ ] Phase 4: Finish published-lineage closeout on the same trunk
  - [ ] Milestone 1: Extend `PublishedModelVersion` so source lineage, optional
    serving lineage, immutable keys, and manifest facts are populated
    coherently.
  - [ ] Milestone 2: Ensure serving-facing workflows do not report success
    before required closeout facts exist.
  - [ ] Milestone 3: Keep source seal and serving publication inside one
    lineage rather than two unrelated success paths.

- [ ] Phase 5: Prove one-trunk parity with an acceptance matrix
  - [ ] Milestone 1: Add parity tests for direct piece registration,
    binding-backed `piece_partial`, and `canonical_full`.
  - [x] Milestone 2: Add same-slot replacement and liveness-loss coverage under
    the readiness-cut model.
  - [ ] Milestone 3: Add planner-backed `PP`, `EP`, and single-rank acceptance
    coverage.
  - [ ] Milestone 4: Update operator-facing docs only after the parity matrix is
    green.

# Tasks

## Workstream A: Shared Structural Helper

- [ ] Refactor `core/store/runtime/metadata/registration_backend.cc` so the
  following steps share one implementation:
  - structural identity derivation
  - canonical coverage derivation
  - proof digest generation
  - durable `view_state_service` update
- [x] Keep binding-backed contribution as a thin frontend wrapper over that
  helper path.
- [ ] Keep `view_state_service.py` as the sole authority for overlap and proof
  semantics.

## Workstream B: Contract-Family Closure

- [ ] Write down the exact phase-1 mapping from planner output to structural
  `ViewSpec`, `view_id`, and `slot_key`.
- [ ] Ensure planner or frontend-owned contract input canonicalizes before
  attempt creation rather than being rebuilt at publish or seal time.
- [ ] Keep `PP` and `EP` as disjoint-slot families on the same trunk.
- [x] Keep single-rank publish legal through `canonical_full`, not through a
  full-coverage piece shortcut.

## Workstream C: Published Lineage

- [ ] Extend result decode and daemon result paths so `PublishedModelVersion`
  contains real source lineage and optional serving lineage.
- [ ] Ensure `representation_contract_hash` comes from a real representation
  contract or manifest lineage.
- [ ] Keep success semantics aligned with configured closeout stages.

## Workstream D: Tests And Documentation

- [ ] Add cross-frontend parity tests before claiming the one-trunk refactor is
  complete.
- [x] Add readiness-cut-aware replacement and liveness-loss tests.
- [ ] Update `tensorcast/api/store/README.md` only when the implemented public
  behavior changes.
- [ ] Update `docs/guides/steptron-vllm-binding-integration.md` only after
  serving-lineage success semantics are real.

# Test / Rollout / Backout

## Required Acceptance Matrix

Python acceptance and integration:

- [ ] `source .venv/bin/activate && pytest tests/python/test_dense_piece_assembly_sealing_acceptance.py`
  - blocker: the rebuilt branch still has unrelated direct-piece/materialization
    regressions in the full file; targeted binding-attempt coverage is green
- [ ] `source .venv/bin/activate && pytest tests/python/test_assembly_attempt.py`
- [ ] `source .venv/bin/activate && pytest tests/python/test_binding.py`
- [ ] `source .venv/bin/activate && pytest tests/python/global_store/test_schema_persistence.py`
- [ ] `source .venv/bin/activate && pytest tests/python/global_store/test_assembly_slot_occupancy_rpc.py`

Daemon and core tests:

- [ ] `bazel test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [ ] `bazel test //daemon:grpc_service_impl_start_seal_assembly_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [ ] `bazel test //core/store:store_engine_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`

New tests that must exist before closure:

- [ ] binding-backed `piece_partial` and direct piece registration share the
  same structural semantics
- [x] `canonical_full` publish succeeds on the canonical structural path
- [x] same-slot replacement is legal only while the attempt is open and before
  the readiness cut
- [x] contributor liveness loss before the readiness cut prevents successful
  seal
- [ ] `PP`, `EP`, and single-rank contract families all seal through the same
  trunk
- [ ] serving-facing publish does not report success before required closeout
  facts exist

## Rollout Order

1. Land the `0105` hard cut first.
2. Land shared structural-helper extraction next.
3. Land contract-family mapping and planner parity next.
4. Land published-lineage closeout parity next.
5. Land acceptance-matrix tests and doc closure last.

## Backout Plan

- If the one-helper refactor regresses the branch, preserve the structural trunk
  and roll back only the helper extraction.
- Do not back out by reintroducing a training-only side path.
- Do not back out by weakening the `0105` hard cut or by restoring
  `expected_view_ids`-based public reconstruction.

# Risks & Tracking

- Risk: the repository lands `0105`, but binding still keeps a subtly different
  structural commit path.
  Mitigation: make parity tests a closure blocker.
- Risk: `PP` and `EP` stay hand-wavy and become ad hoc planner conventions.
  Mitigation: document one deterministic phase-1 mapping and test it.
- Risk: `PublishedModelVersion` remains too weak for serving workflows.
  Mitigation: treat closeout-lineage completeness as a tracked milestone, not a
  follow-up footnote.

# Owner Checklist

- [ ] one assembly trunk remains in the system
- [ ] one layout contract trunk remains in the system
- [ ] `0105` is the sole executable carrier specification for assembly attempts
- [ ] structural commit semantics are shared across frontends
- [ ] `PP`, `EP`, and `canonical_full` all remain on the same trunk
- [ ] published success means real lineage completion
- [ ] parity is proven by tests rather than by prose
- [ ] SDK never talks to Global Store directly
