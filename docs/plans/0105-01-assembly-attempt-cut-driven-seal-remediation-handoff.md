---
slug: assembly-attempt-cut-driven-seal-remediation-handoff-plan
title: Assembly Attempt Cut-Driven Seal Remediation Handoff Plan
links:
  design: ../designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md
areas: ["sdk", "daemon", "core", "global_store", "docs"]
related_code:
  - tensorcast/types.py
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/binding.py
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - core/store/components/global_store_client.h
  - core/store/components/global_store_client.cc
  - core/store/runtime/ingestion/materialization_facade.cc
  - daemon/service/controllers/assembly_coordination_utils.cc
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/materialization_post_seal_utils.cc
  - docs/designs/0085-distributed-binding-assembly-and-coordinator.md
  - docs/designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md
  - docs/plans/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md
last_updated: 2026-03-19
---

# Objective

Provide one ordered handoff plan to finish the `0085`/`0105` hard cut from the
current partially-closed state to a fully cut-driven, daemon-owned attempt
domain.

This plan is intentionally execution-oriented.
The code path described here is now closed through the intended implementation
surface.
The remaining work is reduced to clean verification on a rebuilt daemon/core
binary once the external Bazel queue is clear.

It assumes the repository already has:

- daemon-owned binding contribution lowering in the authoritative path,
- explicit attempt requirements in the public SDK surface,
- durable attempt rows, readiness cuts, and slot occupancy rows,
- and lightweight Python validation already green.

The closure work in this handoff eliminates the last semantic mismatch:

- attempt semantics are cut-first,
- readiness cut now carries full structural evidence,
- attempt seal consumes typed cut input instead of `allowed_view_ids`,
- and daemon admission now enforces requirement family semantics.

# Current State & Grounding

The current repository state is best summarized as follows.

- binding-backed contribution no longer performs final structural registration
  lowering in SDK code before submission; the daemon owns the authoritative
  registration path:
  - `tensorcast/api/store/binding.py`
  - `daemon/service/controllers/owned_binding_service.cc`
- the public requirement kernel already distinguishes `pp`, `ep`, and
  `canonical_full` contract families:
  - `tensorcast/types.py`
- the attempt path already requires explicit `AssemblyRequirementSetRef`
  instead of daemon-side `expected_view_ids` reconstruction:
  - `tensorcast/api/store/__init__.py`
  - `daemon/service/controllers/assembly_operation_service.cc`
- readiness cut capture now uses full `ViewInfo` evidence from the Global Store
  `list_views(...)` path and persists that evidence into the cut:
  - `daemon/service/controllers/assembly_operation_service.cc`
  - `proto/tensorcast/daemon/v2/store_daemon.proto`
  - `daemon/service/controllers/materialization_post_seal_utils.cc`
- the attempt seal path now captures and persists a readiness cut and then
  calls a cut-driven seal entrypoint that consumes explicit structural evidence
  or the explicit `canonical_full` path:
  - `daemon/service/controllers/assembly_operation_service.cc`
  - `core/store/runtime/ingestion/materialization_facade.cc`
- the daemon now validates `slot_id`, target kind, structural view identity,
  and `coverage_contract` so `pp`, `ep`, and `canonical_full` stay distinct in
  the authoritative path:
  - `daemon/service/controllers/assembly_coordination_utils.cc`

Observed validation state:

- lightweight Python tests around the public attempt surface, binding lowering,
  and view registration are green,
- new cut-driven/family-mismatch regression tests are added,
- dense-piece acceptance that relies on the rebuilt daemon binary and daemon-side
  Bazel verification are still pending only because another Bazel compile is in
  flight and this handoff avoids interfering with it.

# Execution Order

The work should be executed in the exact order below.
Do not skip ahead.

- [x] Phase 1: Freeze the spec boundary and acceptance bar
- [x] Phase 2: Make readiness cut full-fidelity
- [x] Phase 3: Replace workspace-driven seal with cut-driven seal
- [x] Phase 4: Enforce attempt contract-family semantics in daemon admission
- [ ] Phase 5: Close validation, rollout notes, and document cleanup

# Phases & Milestones

- [x] Phase 1: Freeze the spec boundary and acceptance bar
  - [x] Milestone 1: `0085` and `0105` say the same thing about what is
        parent-level invariant vs executable attempt-domain behavior
  - [x] Milestone 2: the team has one explicit definition of attempt success,
        canonical-full semantics, and post-cut correctness

- [x] Phase 2: Make readiness cut full-fidelity
  - [x] Milestone 1: the system can capture a cut without reconstructing
        incomplete structural evidence
  - [x] Milestone 2: `meta_digest` is derived from the same complete evidence
        that later validation and reuse logic consider authoritative

- [x] Phase 3: Replace workspace-driven seal with cut-driven seal
  - [x] Milestone 1: attempt seal consumes a typed readiness-cut-backed input
  - [x] Milestone 2: `canonical_full` becomes a first-class seal path instead of
        a best-effort fallback from the view-based engine

- [x] Phase 4: Enforce attempt contract-family semantics in daemon admission
  - [x] Milestone 1: `pp` and `ep` contributions are no longer interchangeable
        in the authoritative daemon path
  - [x] Milestone 2: contribution admission and attempt digest semantics are
        aligned with the canonical family mapping

- [ ] Phase 5: Close validation, rollout notes, and document cleanup
  - [ ] Milestone 1: all targeted Python acceptance and daemon Bazel checks are
        green
  - [ ] Milestone 2: remaining temporary guardrails and stale compatibility
        wording are either removed or clearly documented as temporary

# Detailed Plan

## Phase 1: Freeze the spec boundary and acceptance bar

Purpose:
- eliminate doc/spec ambiguity before changing more code,
- avoid making `0085` and `0105` diverge as the implementation hardens.

Tasks:

- [ ] Decide one of two documentation directions and apply it consistently:
  - [ ] keep `0085` as parent-thesis-only and move executable phase-1 family
        specifics into `0105`
  - [ ] or explicitly mark the exact family mapping in `0085` as normative and
        require `0105` to remain textually aligned
- [ ] Update:
  - [ ] `docs/designs/0085-distributed-binding-assembly-and-coordinator.md`
  - [ ] `docs/designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md`
  - [ ] `docs/plans/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md`
- [ ] Freeze the repository-wide definitions for:
  - [ ] what counts as `pp`
  - [ ] what counts as `ep`
  - [ ] what counts as `canonical_full`
  - [ ] what “published success” means for the current `source_publish_only`
        wave
  - [ ] whether attempt seal correctness is permitted to consult live workspace
        state after the cut is captured

Required outcome:

- after this phase, the implementation can treat `0105` as the sole executable
  contract for the attempt domain,
- and `0085` only constrains invariants that child designs must not violate.

## Phase 2: Make readiness cut full-fidelity

Purpose:
- make the cut a trustworthy, durable semantic object,
- stop computing evidence digests from partial workspace metadata.

Tasks:

- [x] Expand the cut-carried structural evidence model in
      `proto/tensorcast/daemon/v2/store_daemon.proto`
- [x] Extend the view-info retrieval path so daemon code can fetch the same
      complete structural evidence that `compute_view_meta_digest(...)` expects:
  - [x] repository direction: reuse full `list_views(...)` `ViewInfo` instead
        of the incomplete `ViewMetadata` path
- [x] Change readiness-cut capture in
      `daemon/service/controllers/assembly_operation_service.cc` to:
  - [x] fetch full view evidence for each required structural target
  - [x] persist that evidence into the cut
  - [x] compute `meta_digest` from that full evidence only
- [x] Reuse or extend the existing common helpers rather than duplicating
      normalization logic:
  - [x] `core/store/materialization/common/piece_view_state_utils.cc`
  - [x] `core/store/materialization/common/view_registration_normalization_utils.cc`

Required outcome:

- `AssemblyReadinessCut` is no longer a thin digest wrapper,
- and later seal or validation code can consume the cut without rereading live
  workspace truth.

## Phase 3: Replace workspace-driven seal with cut-driven seal

Purpose:
- make the cut the authority root for the sealing phase,
- stop relying on current `list_views(workspace_assembly_id)` as the effective
  seal input.

Tasks:

- [x] Introduce an attempt-specific seal input path in core/daemon.
  Recommended direction:
  - [x] add a helper or API equivalent to
        `seal_assembly_from_cut(workspace_assembly_id, readiness_cut, publish_canonical=true)`
- [x] In `core/store/runtime/ingestion/materialization_facade.cc`, separate
      plan construction into:
  - [x] a structural-view path built from cut-carried structural evidence
  - [x] a `canonical_full` path built from explicit canonical source evidence
- [x] In `daemon/service/controllers/assembly_operation_service.cc`, replace
      the current:
  - [x] capture cut
  - [x] remove `allowed_view_ids` reconstruction from the attempt path
  - [x] stop calling legacy `seal_assembly(...)` from attempt seal
  with:
  - [x] capture cut
  - [x] persist cut
  - [x] call cut-driven seal
- [x] Keep low-level legacy `Store.seal_assembly(...)` behavior intact for
      callers intentionally operating below the attempt surface

Required outcome:

- after the cut is captured, attempt correctness no longer depends on live
  workspace reconstruction,
- and `canonical_full` is a first-class path, not an error-plus-fallback path.

## Phase 4: Enforce attempt contract-family semantics in daemon admission

Purpose:
- preserve the semantic-kernel split that `0085` requires,
- ensure `pp` and `ep` are different in the authoritative daemon path, not just
  in Python helpers and docs.

Tasks:

- [x] Update `daemon/service/controllers/assembly_coordination_utils.cc`
      contribution admission logic to validate:
  - [x] `slot_id`
  - [x] target kind
  - [x] structural view id where relevant
  - [x] `coverage_contract`
- [x] Canonicalize and reject unknown contract-family strings centrally
- [x] Decide whether to keep `coverage_contract` as a string or promote it to a
      typed enum in proto.
  Recommended direction:
  - [x] short-term: strict daemon-side validation of the string
  - [ ] follow-on hardening: convert to a typed enum once the attempt wave is
        stable
- [x] Add explicit negative tests proving:
  - [x] `pp` contribution cannot satisfy `ep` requirements
  - [x] `ep` contribution cannot satisfy `pp` requirements
  - [x] `canonical_full` cannot masquerade as piece coverage

Required outcome:

- the canonical requirement kernel remains semantically distinct in the
  authoritative path,
- and bridge-local lowering does not silently collapse family semantics.

## Phase 5: Close validation, rollout notes, and document cleanup

Purpose:
- prove the new end state with the smallest meaningful test matrix,
- and leave the repo with one coherent story for the next owner.

Tasks:

- [ ] Rebuild and re-run the local Python extension and core bridge:
  - [ ] `source .venv/bin/activate && BUILD_CORE=1 BUILD_EXTENSION=1 python -vvv setup.py build_ext`
- [ ] Re-run the current failing/high-risk dense-piece subset:
  - [ ] `source .venv/bin/activate && TENSORCAST_CUDA_BACKEND=fake pytest tests/python/test_dense_piece_assembly_sealing_acceptance.py -k "canonical_full_attempt_publishes_lineage or same_slot_replacement_rejected_after_readiness_cut or liveness_loss"`
- [ ] Re-run targeted daemon Bazel coverage:
  - [ ] `bazel test //daemon:grpc_service_impl_start_seal_assembly_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [ ] Add or update targeted tests in:
  - [ ] `tests/python/test_binding.py`
  - [ ] `tests/python/test_store_view_api.py`
  - [ ] `tests/python/test_dense_piece_assembly_sealing_acceptance.py`
  - [ ] daemon/controller C++ tests covering cut-driven seal and contract-family
        mismatches
- [ ] Remove or clearly demote temporary closure patches once cut-driven seal is
      proven:
  - [ ] bounded wait for view appearance during cut capture
  - [ ] canonical source retry fallback in seal path
- [ ] Update docs:
  - [ ] `docs/designs/0085-distributed-binding-assembly-and-coordinator.md`
  - [ ] `docs/designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md`
  - [ ] `docs/plans/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md`
  - [ ] `tensorcast/api/store/README.md`

Required outcome:

- one consistent cut-driven story across code, tests, and docs,
- and no outstanding ambiguity about whether the hard cut is actually complete.

# Test / Rollout / Backout

## Test Plan

- [ ] Lightweight Python API tests
  - [ ] explicit requirements
  - [ ] daemon-owned contribution lowering
  - [ ] PP/EP mismatch rejection
- [ ] Dense-piece acceptance
  - [ ] canonical-full single-rank attempt
  - [ ] readiness-cut replacement closure
  - [ ] contributor liveness loss before cut
  - [ ] post-cut stability
- [ ] Daemon Bazel tests
  - [ ] start attempt
  - [ ] explicit seal transition
  - [ ] cut capture fidelity
  - [ ] cut-driven seal

## Rollout

- [ ] Land Phase 2 and Phase 4 before flipping the attempt path to cut-driven
      seal
- [ ] Land Phase 3 behind the attempt surface only; do not rewrite the legacy
      low-level structural seal surface in the same wave
- [ ] Keep temporary readiness/view guardrails until the dense-piece acceptance
      subset is green

## Backout

- [ ] If Phase 3 destabilizes the attempt path, revert only the attempt-path
      call from cut-driven seal back to the legacy seal entrypoint
- [ ] Preserve explicit requirements, daemon-owned contribution lowering, and
      slot occupancy semantics during any backout
- [ ] Do not restore SDK-side structural pre-registration as a rollback path

# Risks & Tracking

- [ ] Risk: `0085` and `0105` continue to drift
  - mitigation: finish Phase 1 first and require a single executable contract
- [ ] Risk: cut fidelity work expands into ad-hoc metadata duplication
  - mitigation: add one full `ViewInfo` retrieval path and reuse the same
    normalization helpers already being added in core
- [ ] Risk: canonical-full remains a hidden exception path
  - mitigation: give it an explicit cut-driven seal branch instead of growing
    more retries around `list_views(...)`
- [ ] Risk: PP/EP semantics stay cosmetic
  - mitigation: enforce `coverage_contract` in daemon admission before claiming
    the family split is done
- [ ] Risk: build/test cost hides regressions
  - mitigation: keep the validation subset small and stable, and always rerun it
    after core/daemon changes in this area

# Owner Checklist

- [ ] `0085` is parent-invariant only, or its normative phase-1 scope is made
      explicit
- [ ] `0105` remains the sole executable attempt-domain spec
- [ ] readiness cut carries full structural evidence
- [ ] readiness cut digest is derived from that full evidence
- [ ] attempt seal consumes the cut rather than reconstructing from live
      workspace state
- [ ] `canonical_full` is first-class in the seal path
- [ ] daemon admission distinguishes `pp`, `ep`, and `canonical_full`
- [ ] published success implies the required lineage and closeout facts exist
- [ ] docs, tests, and code say the same thing
