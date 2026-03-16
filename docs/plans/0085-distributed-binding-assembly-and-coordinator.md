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
  - proto/tensorcast/global_store/v1/global_store.proto
  - tensorcast/schema.sql
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/registration_controller.cc
  - daemon/state/session_lifecycle.cc
  - core/store/components/global_store_client.h
  - core/store/components/global_store_client.cc
  - core/store/runtime/metadata/registration_backend.cc
  - tensorcast/global_store/repositories/assembly_contribution_repository.py
  - tensorcast/global_store/repositories/operation_repository.py
  - tensorcast/global_store/rpc/assembly_contribution_rpc_handler.py
  - tensorcast/global_store/rpc/operation_rpc_handler.py
  - tensorcast/global_store/services/view_state_service.py
  - tests/python/test_binding.py
  - tests/python/test_assembly_attempt.py
  - tests/python/test_dense_piece_assembly_sealing_acceptance.py
  - tests/python/global_store/test_assembly_contribution_repository.py
  - tests/python/global_store/test_assembly_contribution_rpc.py
  - daemon/service/owned_binding_service_test.cc
  - daemon/service/grpc_service_impl_start_seal_assembly_test.cc
last_updated: 2026-03-16
---

# Objective

Re-baseline `0085` against the current implementation and land the remaining
work needed for the design to be true in both semantics and tests.

The required end state is still the same:

- TensorCast keeps one assembly trunk.
- TensorCast keeps one layout contract trunk.
- binding-backed publish remains one frontend onto the existing
  view-registration and seal path.
- attempt completeness is rooted in `LayoutSpec.expected_view_ids`.
- contributor liveness and binding mutation fencing are real correctness
  properties, not advisory metadata.
- attempt success means a real `PublishedModelVersion` lineage, not only a
  sealed source artifact.

This revision resets the implementation plan around the actual gaps found in
code review instead of the more optimistic status captured in the prior draft.

# Current State & Grounding

## Existing Trunk Pieces We Must Reuse

- `proto/tensorcast/layout/v1/layout.proto`
  - `LayoutSpec` is already immutable, content-addressed, and already carries
    `expected_view_ids`.
- `daemon/service/controllers/registration_controller.cc`
  - piece registration already computes `view_id`, coverage, proof material, and
    view-state writes through the existing assembly path.
- `tensorcast/global_store/services/view_state_service.py`
  - overlap and proof semantics already live here and already use the active
    `assembly_layout_bindings` root.
- `daemon/service/controllers/assembly_operation_service.cc`
  - `StartSealAssembly` already uses the durable `operation.proto` workflow and
    already snapshots current assembly state before seal.
- `tensorcast/api/store/binding.py`
  - `SealedBindingValue.contribute_to_assembly(...)` already compiles binding
    bytes onto the existing SDK registration APIs instead of inventing a second
    publish path.
- `docs/architecture/view-replicas-and-assembly.md`
  - the architecture doc already establishes one assembly workspace and one
    sealing path as the canonical trunk.

## Existing Implementation That Is Useful But Incomplete

- `Store.start_assembly_attempt(layout_id=...)` exists and returns
  `AssemblyAttemptRef`.
- `assembly_contributions` exists in `tensorcast/schema.sql`.
- `SubmitBindingContribution` exists and persists contributor rows keyed by
  `(assembly_id, view_id)`.
- `Binding.begin_update(...)` already checks for accepted contributor rows and
  fences mutation.
- `wait_assembly_attempt(...)` already decodes a `SealAssemblyResult` into
  `PublishedModelVersion`.

These are useful foundations, but they do not yet prove the design goals.

## Landed Since Re-Baselining

- `wait_assembly_attempt(...)` now drives a still-open attempt onto the same
  deterministic seal operation before waiting.
  - the SDK no longer depends on an external manual `StartSealAssembly` step
    for the normal attempt workflow
- `StartSealAssembly(layout_id=...)` now behaves as a fail-closed attempt
  transition.
  - if an attempt snapshot is missing or malformed, sealing fails instead of
    reconstructing a weaker contract
- assembly-contribution mutation paths now use a durable live predicate based
  on:
  - row state
  - `lease_expires_at`
  - contributor worker or daemon identity still active in Global Store
- durable row updates now use atomic CAS-style SQL predicates for:
  - slot claim on `(assembly_id, view_id)`
  - lease or state-scoped row transitions
- seal and binding-mutation fencing now consume the same durable live
  contributor predicate instead of lease-expiry-only checks

## Remaining Gaps That Still Block Full Design Closure

- `PublishedModelVersion` is still only partially populated.
  - source artifact lineage exists
  - serving lineage, immutable source or serving version keys, and manifest
    completion semantics remain incomplete
- same-slot replacement semantics are still unresolved.
  - the current implementation keeps one-slot occupancy strict and does not yet
    implement the design’s long-term replacement handoff
- tests now prove the attempt wait path and durable live-contributor predicate,
  but they still do not prove:
  - liveness loss mid-attempt under real daemon crash or restart
  - same-slot replacement semantics
  - binding, `register_view`, and LIP piece parity
  - full published model-version lineage including serving publication

## Current Test Baseline

Targeted suites currently pass and are still useful as a baseline:

- `tests/python/test_binding.py`
- `tests/python/test_assembly_attempt.py`
- `tests/python/test_binding_state.py`
- `tests/python/global_store/test_assembly_contribution_repository.py`
- `tests/python/global_store/test_assembly_contribution_rpc.py`
- `daemon:owned_binding_service_test`
- `daemon:grpc_service_impl_start_seal_assembly_test`

These validate SDK shape and repository plumbing, but they do not prove the
core `0085` invariants. They should be treated as smoke tests, not acceptance
proof.

The historical dense-piece acceptance baseline is still the closest end-to-end
trunk proof:

- `tests/python/test_dense_piece_assembly_sealing_acceptance.py`

That suite proves the existing assembly trunk can seal disjoint pieces, but it
still does not prove that binding-backed contributors obey the same semantics.

# Plan Decisions To Lock Before Coding

This execution plan makes the following implementation decisions explicit.
They are intended to remove ambiguity from `0085`, not to change its direction.

- `contribution_contract_hash` must be computed from an explicit
  `ContributionContractSnapshot`.
  - Hashing only `layout_id` plus `expected_view_ids` is non-compliant.
- phase-1 contract snapshots must explicitly distinguish:
  - binding-backed attempts that require durable live contributor occupancy
  - direct piece-registration seal snapshots that only rely on the registered
    current view set
- `StartAssemblyAttempt` must materialize a real coordinator operation record,
  or an equivalent durable coordinator fence, at attempt creation time.
  - Returning only a deterministic future operation id is not sufficient for
    submit-time fencing.
- a contribution is considered live only when all of the following hold:
  - the durable row is in `accepted`
  - the row's liveness fence is still current
  - the contributing daemon identity is still current enough for phase-1 seal
    correctness
- seal completeness must be checked against the live required contributor set,
  not against accepted rows alone.
- `StartSealAssembly` remains the sealing primitive.
  - this plan does not create a second coordinator subsystem
  - it transitions the already-created attempt coordinator into the running seal
    phase
- `wait_assembly_attempt(...)` reports success only when the attempt has reached
  the design's full published-lineage success condition.
  - source seal alone is not sufficient for final success when serving-side
    publication is required
- no new training-only slot plane or layout plane may be introduced.
  - `view_id` stays the phase-1 contribution identity
  - `layout_id` stays the global layout identity

# Non-Goals For This Execution Plan

- Do not introduce a second assembly trunk for binding-backed training.
- Do not widen `LayoutSpec` with frontend or topology ownership metadata.
- Do not make the SDK talk to Global Store directly.
- Do not treat `accepted` contributor rows as lease authority.
- Do not add TP-aware publish as part of this phase-1 completion work.
- Do not introduce a new durable top-level contract table unless the operation
  snapshot form proves insufficient during implementation.

# Phases & Milestones

- [ ] Phase 1: Correct the attempt contract and coordinator fence
  - [ ] Milestone 1: Represent an explicit `ContributionContractSnapshot` in the
    attempt snapshot and compute `contribution_contract_hash` from that explicit
    snapshot content.
  - [ ] Milestone 2: Use one shared helper for contract-snapshot construction
    and hashing so daemon controllers cannot drift.
  - [ ] Milestone 3: Make `StartAssemblyAttempt` create the live coordinator
    fence that later `SubmitBindingContribution` validates.
  - [ ] Milestone 4: Make `SubmitBindingContribution` reject missing, stale, or
    generation-mismatched coordinator fences.
  - [ ] Milestone 5: Keep `StartSealAssembly` as the seal transition on that
    same attempt fence instead of inventing a parallel coordinator path.

- [ ] Phase 2: Make contributor liveness durable enough for seal correctness
  - [ ] Milestone 1: Define the phase-1 durable liveness projection for
    `assembly_contributions`.
  - [ ] Milestone 2: Ensure contributor loss through PID exit, daemon loss, or
    coordinator loss transitions rows to a visible non-live state.
  - [ ] Milestone 3: Make seal compute completeness from the required live
    contributor set, not accepted rows alone.
  - [ ] Milestone 4: Keep binding mutation fencing tied to the same liveness
    notion so `Binding.begin_update(...)` and overwrite paths fence exactly the
    values still occupying live required slots.

- [ ] Phase 3: Finish the one-trunk implementation refactor
  - [ ] Milestone 1: Extract shared view or piece commit helpers from
    `registration_backend.cc`.
  - [ ] Milestone 2: Route binding-backed contribution through the same helper
    path used by existing piece registration for coverage, proof, hashing, and
    view-state update.
  - [ ] Milestone 3: Prove daemon-owned, client-owned, LIP-backed, and future
    byte-in sources can all land on the same commit path without behavior drift.

- [ ] Phase 4: Codify phase-1 mapping and single-rank behavior
  - [ ] Milestone 1: Decide and document the concrete phase-1 planner-to-view
    mapping for `PP` and `EP`.
  - [ ] Milestone 2: Ensure required `expected_view_ids` are computed once from
    deterministic planner output and not re-derived ad hoc during publish.
  - [ ] Milestone 3: Keep the single-rank case legal through
    `canonical_full`, not through a full-coverage piece shortcut.
  - [ ] Milestone 4: Keep TP explicitly deferred in both code and docs so the
    phase-1 trunk does not silently over-promise.

- [ ] Phase 5: Complete published model-version semantics
  - [ ] Milestone 1: Extend the daemon result path so
    `PublishedModelVersion` contains real source lineage, optional serving
    lineage, immutable keys, and manifest metadata.
  - [ ] Milestone 2: Define when attempt success is allowed to report success
    for serving-facing workflows.
  - [ ] Milestone 3: Keep source seal and serving publication inside one
    lineage rather than two unrelated success paths.

- [ ] Phase 6: Prove parity with an acceptance matrix
  - [ ] Milestone 1: Add missing negative tests for contract mismatch,
    coordinator mismatch, liveness loss, and required-view incompleteness.
  - [ ] Milestone 2: Add parity tests showing binding-backed contribution,
    `register_view`, and LIP piece flows use the same trunk semantics.
  - [ ] Milestone 3: Add planner-backed `PP`, `EP`, and single-rank
    `canonical_full` acceptance coverage.
  - [ ] Milestone 4: Re-run the full targeted Python and daemon C++ matrix and
    update the docs only when the one-trunk behavior is proven.

# Detailed Tasks

## Workstream A: Attempt Contract And Coordinator Fence

- [ ] Add an explicit phase-1 contribution-contract payload to
  `proto/tensorcast/daemon/v2/store_daemon.proto`.
  - Preferred shape:
    - keep `AssemblyAttemptRef` lightweight
    - carry the full snapshot in the coordinator operation snapshot
    - add enough proto structure inside `SealAssemblySnapshot` or a sibling
      message to serialize per-view contract semantics explicitly
- [ ] Define the minimum phase-1 per-view contract contents.
  - required `view_id`
  - required contribution kind
  - canonical contribution mapping source
    - semantic `ViewSpec`
    - or canonical-full sentinel
  - planner or coverage digest
  - any additional phase-1 flags needed to validate resubmission
- [ ] Add one shared daemon helper for:
  - constructing `ContributionContractSnapshot`
  - computing `contribution_contract_hash`
  - validating a submit request against the snapshot
- [ ] Remove duplicated contract-hash construction logic from:
  - `daemon/service/controllers/assembly_operation_service.cc`
  - `daemon/service/controllers/owned_binding_service.cc`
- [ ] Change `StartAssemblyAttempt` to create the live coordinator operation
  record instead of returning only a future deterministic operation id.
- [ ] Decide and document the operation state model for an attempt before seal.
  - acceptable options:
    - pending attempt state inside the same operation
    - running attempt state before seal
  - non-acceptable option:
    - submit path trusting an operation id that does not yet exist
- [ ] Make `SubmitBindingContribution` validate:
  - operation exists
  - operation kind matches the attempt
  - lease or generation is current
  - operation has not already transitioned to terminal failure or success
- [ ] Make `StartSealAssembly` transition the same attempt operation to the seal
  phase and reject mismatched attempt generations.

## Workstream B: Durable Liveness Projection

- [ ] Define the phase-1 durable "live contribution" predicate.
  - minimum inputs:
    - `assembly_contributions.state`
    - durable liveness fence such as `lease_expires_at`
    - contributor daemon identity visibility
  - the predicate must still work after a daemon crash where local finalizers do
    not run
- [ ] Decide how `lease_expires_at` is maintained.
  - acceptable direction:
    - explicit keepalive or refresh path tied to contribution lifetime
    - or contributor-daemon liveness projection plus bounded-expiry rows
  - non-acceptable direction:
    - relying only on in-memory finalizers
- [ ] Extend `assembly_contributions` repository helpers to query:
  - required live contributors by assembly
  - live rows for one `(binding_id, binding_value_id)`
  - stale or expired rows that need transition
- [ ] Extend `tensorcast/global_store/rpc/assembly_contribution_rpc_handler.py`
  and `core/store/components/global_store_client.{h,cc}` as needed so the daemon
  can update and query the durable liveness projection cleanly.
- [ ] Integrate contributor daemon identity with existing GS liveness truth.
  - `contributor_daemon_id` is already stored
  - seal must not treat a row from a known-dead contributor as live
- [ ] Keep graceful finalizers as an accelerator, not the sole authority.
  - PID exit should still mark rows stale quickly
  - coordinator abort should still mark rows aborted quickly
  - success should still release rows quickly

## Workstream C: Seal Correctness

- [ ] Make seal evaluate completeness in this order:
  - snapshot contract present and valid
  - current registered view set matches the snapshot for required views
  - required live contributor set satisfies the same required view ids
  - only then attempt final source seal
- [ ] Make seal reject:
  - missing required current views
  - missing required live contributors
  - contract-hash mismatch between current attempt state and snapshot
  - coordinator-generation mismatch
- [ ] Keep current post-seal migration or reuse behavior aligned with the same
  required-view set so post-seal optimizations do not bypass contract rules.

## Workstream D: Shared Registration Trunk

- [ ] Refactor `core/store/runtime/metadata/registration_backend.cc` so the
  following steps have one shared implementation:
  - view plan derivation
  - canonical coverage derivation
  - view hashing
  - proof digest generation
  - durable view-state update
- [ ] Ensure binding-backed contribution remains a thin frontend wrapper over
  the shared registration helper path.
- [ ] Keep `tensorcast/global_store/services/view_state_service.py` as the only
  authority for overlap and proof semantics.
- [ ] Ensure same `(assembly_id, view_id)` replacement stays idempotent and
  explicit on that one path.
- [ ] Add parity tests that compare:
  - existing piece registration
  - binding-backed `piece_partial`
  - future byte-in or other frontends when they land

## Workstream E: Phase-1 Mapping And Single-Rank Publish

- [ ] Write down the exact phase-1 mapping contract for planner output.
  - how one local ownership plan maps to deterministic `ViewSpec`
  - how those `ViewSpec` map to deterministic `view_id`
  - where that mapping is stored and versioned
- [ ] Add a concrete plan object or helper only if needed.
  - avoid introducing a second identity plane
  - avoid turning per-update publish into an ad hoc parameter soup
- [ ] Add explicit `PP` acceptance coverage.
  - one model split into deterministic disjoint stage or chunk views
  - expected view ids fixed up front
  - seal succeeds only after all required views arrive
- [ ] Add explicit `EP` acceptance coverage.
  - one model split into deterministic expert-group views
  - same completeness and replacement semantics as `PP`
- [ ] Add explicit single-rank `canonical_full` acceptance coverage.
  - no required `expected_view_ids`
  - no fake full-coverage piece shortcut

## Workstream F: Published Model-Version Lineage

- [ ] Extend `proto/tensorcast/daemon/v2/store_daemon.proto` and the SDK decode
  path so `PublishedModelVersion` can carry:
  - source artifact descriptor
  - optional serving artifact descriptor
  - immutable source version key
  - immutable serving version key
  - representation contract hash
  - serving manifest reference
- [ ] Define the success boundary in code.
  - if only source seal is required, source seal may be final success
  - if serving publication is configured, operation success must wait for
    serving artifact and immutable serving publication
- [ ] Update `tensorcast/api/store/__init__.py` so `wait_assembly_attempt(...)`
  preserves the stronger success semantics and does not imply full publication
  before the daemon has actually completed it.

## Workstream G: Documentation And Operator Guidance

- [ ] Update `docs/designs/0085-distributed-binding-assembly-and-coordinator.md`
  together with implementation so the design remains authoritative.
- [ ] Update `tensorcast/api/store/README.md` when public behavior changes.
- [ ] Update `docs/guides/steptron-vllm-binding-integration.md` only after the
  serving-lineage success semantics are real.
- [ ] Keep all status bullets and checkboxes truthful to the actual repo state.

# Test / Rollout / Backout

## Required Acceptance Matrix

Python acceptance and integration:

- [ ] `source .venv/bin/activate && pytest tests/python/test_dense_piece_assembly_sealing_acceptance.py`
- [ ] `source .venv/bin/activate && pytest tests/python/test_dual_daemon_global_store_tp_view.py`
- [x] `source .venv/bin/activate && pytest tests/python/test_assembly_attempt.py`
- [x] `source .venv/bin/activate && pytest tests/python/test_binding.py`
- [x] `source .venv/bin/activate && pytest tests/python/global_store/test_assembly_contribution_repository.py`
- [x] `source .venv/bin/activate && pytest tests/python/global_store/test_assembly_contribution_rpc.py`
- [ ] `source .venv/bin/activate && pytest tests/python/global_store/test_operation_rpc.py`
- [ ] `source .venv/bin/activate && pytest tests/python/global_store/test_services.py`

New Python tests that must be added:

- [ ] contract hash changes when per-view semantics change but `expected_view_ids`
  do not
- [ ] submit rejects stale or missing coordinator operation or generation
- [ ] required `expected_view_ids` missing prevents final seal
- [ ] same `(assembly_id, view_id)` contribution can be replaced idempotently
  inside one attempt
- [ ] contributor liveness loss before seal aborts or stales the attempt
- [ ] `Binding.begin_update(...)` rejects or waits while its current value is
  still a live required contributor
- [ ] binding-backed `piece_partial` matches equivalent piece-registration slot
  identity and replacement semantics
- [ ] single-rank `canonical_full` publish succeeds through the legal
  contribution kind
- [ ] successful attempt returns full `PublishedModelVersion` lineage when
  serving publication is configured
- [ ] serving-facing publish does not report success before immutable serving
  publication completes

C++ daemon and core tests:

- [x] `bazel test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [x] `bazel test //daemon:grpc_service_impl_start_seal_assembly_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [ ] `bazel test //daemon:grpc_service_impl_lip_piece_assembly_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [ ] `bazel test //core/store:store_engine_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`

New C++ tests that must be added:

- [ ] coordinator completeness is checked by required live expected-view set
- [ ] submit rejects stale coordinator generation
- [ ] attempt contract hash uses explicit snapshot semantics rather than only
  expected-view ids
- [ ] contributor finalizers plus crash-recovery path both transition rows out of
  the live set
- [ ] binding-backed contribution and existing piece registration share the same
  commit helper path

## Rollout Order

1. Land the doc and proto clarification first.
2. Land coordinator and contract correctness next.
3. Land durable liveness projection next.
4. Land seal live-set correctness next.
5. Land shared trunk refactor next.
6. Land planner-backed `PP` or `EP` coverage and single-rank coverage next.
7. Land full published-lineage success semantics last.
8. Update guides only after the full acceptance matrix is green.

## Backout Plan

- If coordinator or liveness hardening regresses the branch, revert the new
  contributor acceptance rules first and leave `assembly_contributions` schema
  unused rather than deleting schema history.
- Do not back out by introducing a training-only side path.
- Preserve existing `register_view`, LIP piece registration, `LayoutSpec`,
  `assembly_layout_bindings`, `artifact_bindings`, and `StartSealAssembly`
  behavior as the stable fallback trunk.
- Do not leave behind a partial state where publish can report success without
  required-view completeness or without releasing contributor fences.

# Risks & Tracking

- Risk: we accidentally bless the current weak contract hash as "good enough".
  Mitigation: make explicit contract snapshotting a phase-1 blocker.

- Risk: coordinator fencing remains an audit field instead of a real gate.
  Mitigation: require attempt creation to materialize the live coordinator
  operation that submit must verify.

- Risk: contributor liveness still depends on graceful finalizers only.
  Mitigation: require a durable liveness projection that survives daemon crash.

- Risk: seal continues to use accepted rows instead of live contributors.
  Mitigation: define and test one authoritative live-set predicate.

- Risk: shared trunk refactor slips and binding keeps a subtly different commit
  path.
  Mitigation: add parity tests and one shared helper extraction as a tracked
  milestone.

- Risk: `PublishedModelVersion` success semantics stay too weak for serving.
  Mitigation: make operation success depend on full configured lineage
  completion.

# Owner Checklist

- [ ] one assembly trunk remains in the system
- [ ] one layout contract trunk remains in the system
- [ ] `LayoutSpec.expected_view_ids` remains the phase-1 completeness root
- [ ] `ContributionContractSnapshot` is explicit and authoritative
- [ ] `view_id` remains the phase-1 contribution identity
- [ ] submit validates a live coordinator generation
- [ ] seal uses the required live contributor set, not accepted rows alone
- [ ] binding mutation fences are coherent with the same live-set semantics
- [ ] phase-1 `PP` and `EP` compile to disjoint views on the same trunk
- [ ] single-rank publish uses `canonical_full`, not a full-coverage piece
- [ ] `PublishedModelVersion` success semantics match real source or serving
      publication completion
- [ ] SDK never talks to Global Store directly

# Re-Baselined Status (2026-03-16)

What is genuinely landed and can be relied on:

- assembly-attempt API surface exists
- binding-backed contribution entrypoints exist
- contributor rows exist in persistent schema
- `wait_assembly_attempt(...)` now drives a still-open attempt onto the same
  deterministic seal operation before waiting
- `wait_assembly_attempt(...)` now passes the snapped coordinator generation
  when it transitions a pending attempt into seal
- `StartSealAssembly(layout_id=...)` now fails closed for missing or malformed
  attempt snapshots instead of rebuilding a weaker contract
- `StartSealAssembly(layout_id=...)` now rejects stale coordinator generations
  before it reacquires the attempt fence
- durable contributor liveness now depends on:
  - accepted row state
  - unexpired `lease_expires_at`
  - contributor worker or daemon identity still active in Global Store
- binding overwrite paths are fenced by that same durable live-contributor
  predicate
- view-slot occupancy updates for `assembly_contributions` now use atomic
  claim/CAS-style SQL predicates
- same-slot replacement on `(assembly_id, view_id)` is now allowed when the
  replacement stays inside the same live coordinator fence
- seal live-set completeness now evaluates the snapped contribution contract,
  including `canonical_full`, instead of only the non-canonical
  `expected_view_ids` subset
- contribution-slot identity mapping now lives in the shared
  `assembly_coordination_utils` helper instead of drifting between controllers
- the seal result path now consumes `assembly_runtime_policies` to publish
  immutable source and serving version keys plus serving manifest metadata when
  policy is configured
- fake-CUDA binding contribution now routes through the stable host upload path
  instead of requiring driver-backed CUDA IPC for the contribution upload
- `wait_assembly_attempt(...)` can now hand the attempt snapshot back into
  `StartSealAssembly` when the coordinator record is missing or stale during the
  wait-to-seal transition
- canonical-full sealing now has a source-assembly fallback path that does not
  require view rows when a local canonical replica already exists
- subset-backed binding attempts now mint deterministic piece `view_id`
  identity and submit piece registrations with explicit `tensor_names`
  carried through the daemon or seal trunk
- assembly-scoped same-view replacement now rewrites view leaf state instead of
  failing on stale leaf-digest conflicts during a legal in-attempt replacement
- dense-piece assembly remains the current end-to-end trunk baseline

What must still be treated as open until this plan is executed:

- full cross-frontend contribution-contract snapshot correctness
- one-helper shared registration commit path
- planner-backed `PP` and `EP` binding parity at final seal time
- full source and serving published-lineage success semantics
- binding-backed piece acceptance now gets through bind, contribution submit,
  and durable slot replacement, but seal still fails to consume those piece
  replicas in the daemon acceptance path
  - direct GS transport requests against the contributed view replicas succeed
    in local repro
  - the remaining blocker is specifically the daemon seal path still reporting
    `no available piece replicas for assembly_id=...`
  - canonical-full binding publish now seals and publishes lineage, but the
    follow-on sealed-artifact materialization path is still unstable in the
    fake-CUDA acceptance environment
  - the binding-backed acceptance cases in
    `tests/python/test_dense_piece_assembly_sealing_acceptance.py` are still
    not final acceptance proof until that seal-time piece-replica gap is
    closed
