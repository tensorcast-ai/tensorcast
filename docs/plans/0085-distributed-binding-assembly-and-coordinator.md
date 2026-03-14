---
slug: distributed-binding-assembly-and-coordinator-plan
title: Distributed Binding Assembly on the Existing Assembly and Layout Trunk Plan
links:
  design: ../designs/0085-distributed-binding-assembly-and-coordinator.md
areas: ["sdk", "daemon", "core", "proto", "global_store"]
related_code:
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/registration.py
  - tensorcast/api/_register.py
  - proto/tensorcast/layout/v1/layout.proto
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - tensorcast/schema.sql
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/service/controllers/registration_controller.cc
  - core/store/runtime/metadata/registration_backend.cc
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/materialization/dataplane/view/view_identity.h
  - tensorcast/global_store/services/view_state_service.py
  - tensorcast/global_store/rpc/layout_binding_rpc_handler.py
  - tensorcast/global_store/repositories/assembly_layout_binding_repository.py
  - tensorcast/global_store/repositories/artifact_binding_repository.py
  - tests/python/test_dense_piece_assembly_sealing_acceptance.py
  - tests/python/test_dual_daemon_global_store_tp_view.py
  - tests/python/global_store/test_operation_rpc.py
  - tests/python/global_store/test_services.py
---

# Objective

Implement distributed publish for binding-backed training by routing it onto the
repository’s **existing** assembly and layout trunk instead of creating a second
training-only assembly path.

Target end state:

- `LayoutSpec` remains the single global layout contract root
- `LayoutSpec.expected_view_ids` becomes the phase-1 expected contribution set
- each attempt snapshots an explicit per-view contribution contract above that
  expected-view root
- deterministic `view_id` remains the contribution identity
- `assembly_id` remains the concrete publish workspace
- binding-backed contribution compiles down to the same view/piece registration
  trunk already used by assembly today
- contributor liveness and mutation fencing become explicit without splitting the
  coordinator model
- attempt completion returns a published model-version lineage
  - sealed source artifact
  - optional serving artifact
  - immutable keys and manifest metadata

# Current State & Grounding

The repository already contains most of the trunk this feature needs. The plan
must deepen and connect those pieces rather than replace them.

- `proto/tensorcast/layout/v1/layout.proto`
  - `LayoutSpec` is already immutable and content-addressed.
  - It already carries `expected_view_ids`, overlap policy, and
    `proof_schema_version`.
  - This is the strongest signal that the current trunk already has a reusable
    contract root.

- `core/store/materialization/dataplane/view/view_identity.h`
  - deterministic `view_id` is already computed from semantic `ViewSpec` plus
    canonical index bytes
  - this is the existing contribution identity plane; we should not invent a
    second slot-id plane in phase 1

- `core/store/runtime/metadata/registration_backend.cc`
  - piece registration already computes:
    - `ViewSpec` plan
    - canonical coverage ranges
    - `view_data_hash`
    - proof digests
  - it already distinguishes `CANONICAL` vs `PIECE`
  - piece registration already requires `client_artifact_id (cgid)` for the
    target assembly workspace

- `tensorcast/global_store/services/view_state_service.py`
  - already upserts per-view metadata
  - already uses `replace_ranges(...)` for the same `(artifact_id, view_id)`
  - already loads the active `LayoutSpec` through `assembly_layout_bindings`
  - already enforces overlap/proof semantics against the current layout

- `tensorcast/global_store/rpc/layout_binding_rpc_handler.py`
  - `assembly_layout_bindings` already provide the versioned CAS root for
    `assembly_id -> layout_id`
  - handler already validates `layout.index_multihash` against the assembly’s
    canonical index

- `daemon/service/controllers/assembly_operation_service.cc`
  - `StartSealAssembly` already runs as a durable `operation.proto` workflow
  - it already snapshots `layout_id` and `assembly_layout_binding_version`
  - today it snapshots current views and seals them, but it does not yet treat
    `expected_view_ids` as the required completeness root

- `daemon/service/grpc_service_impl_lip_piece_assembly_test.cc`
  - existing tests already prove LIP-backed piece registration can participate in
    assembly
  - this is exactly the kind of trunk reuse we want for binding contributors

- `tests/python/test_dense_piece_assembly_sealing_acceptance.py`
  - current acceptance coverage already proves disjoint dense pieces can fill one
    assembly workspace and seal successfully
  - this is the closest existing end-to-end reality for phase-1 `PP` and `EP`

- `tests/python/test_dual_daemon_global_store_tp_view.py`
  - current TP-oriented coverage exists on the retrieval side through view
    materialization
  - it does not yet prove TP assembly semantics, so TP should stay explicitly
    deferred in the publish plan

- `core/store/README.md`
  - documents view registration and assembly sealing as existing StoreEngine
    responsibilities

- `0084` local binding contract
  - local binding now separates local sealed value from artifact promotion
  - this makes `SealedBindingValue` the right frontend surface for compiling
    local values onto the assembly trunk

Current gaps that this plan must close:

- no explicit daemon API to start a fresh assembly workspace from `layout_id`
- no explicit per-attempt contribution-contract snapshot above
  `LayoutSpec.expected_view_ids`
- no binding-contributor path that compiles to the existing piece registration
  helpers
- no persisted contributor identity/liveness keyed by `(assembly_id, view_id)`
- no integration with the daemon lease/guard/finalizer runtime from `0011`
- no mutation fence from open assembly contribution back into
  `Binding.begin_update(...)`
- no seal completeness rule that compares the current live view set against
  `LayoutSpec.expected_view_ids`
- no authoritative publish result that links source assembly to serving-artifact
  publication for downstream consumers such as vLLM

Baseline constraints this plan must respect:

- SDK must not talk to Global Store directly.
- `register_view`, LIP piece registration, and final `seal_assembly(...)` remain
  the one true assembly trunk.
- `view_id` remains the deterministic contribution identity plane.
- Every published version must use a fresh `assembly_id`.
- `LayoutSpec` must not absorb frontend topology ownership metadata.
- serving-facing workflows ultimately publish serving artifacts, not only source
  assembly artifacts.
- Phase-1 `PP` and `EP` must compile to disjoint partial views on the current
  trunk.
- `TP` is not the primary publish target in phase 1.

# Phases & Milestones

- [ ] Phase 1: Lock the one-trunk contract in code and docs
  - [ ] Milestone 1: Explicitly define phase-1 completeness as
    `LayoutSpec.expected_view_ids` satisfaction, not contributor count.
  - [ ] Milestone 2: Explicitly define phase-1 slot identity as existing
    deterministic `view_id`.
  - [ ] Milestone 3: Define an explicit per-attempt contribution contract
    snapshot above `expected_view_ids`.
  - [ ] Milestone 4: Define binding contribution as one frontend onto the
    existing piece/view registration trunk.
  - [ ] Milestone 5: Define fresh `assembly_id` per publish attempt as the only
    supported workspace lineage rule.
  - [ ] Milestone 6: Define published result as a model-version lineage rather
    than a bare artifact id.
  - [ ] Milestone 7: Define phase-1 distributed scope explicitly as:
    - `PP` and `EP` on disjoint partial views
    - `TP` deferred from publish

- [ ] Phase 2: Add assembly-attempt bootstrap on top of existing layout binding
  - [ ] Milestone 1: Add daemon RPCs to create a fresh `assembly_id` from
    `layout_id` and return an attempt/coordinator reference.
  - [ ] Milestone 2: Reuse `assembly_layout_bindings` CAS to bind the attempt to
    the active `layout_id`.
  - [ ] Milestone 3: Extend coordinator snapshot/state so `expected_view_ids`
    from the active layout are part of the attempt’s explicit seal preconditions.
  - [ ] Milestone 4: Snapshot the explicit contribution contract and expose a
    stable `contribution_contract_hash`.

- [ ] Phase 3: Codify phase-1 projection mapping for `PP` and `EP`
  - [ ] Milestone 1: Decide exactly how local `BindingContributionPlan` compiles
    to existing `ViewSpec` and deterministic `view_id` on the current trunk.
  - [ ] Milestone 2: Keep phase-1 output restricted to disjoint required views
    that match current overlap rules.
  - [ ] Milestone 3: Compute and persist required `expected_view_ids` once during
    layout or contract generation; do not derive them ad hoc on every publish.
  - [ ] Milestone 4: Avoid introducing a second projection or slot language in
    phase 1 if current `ViewSpec` plus `view_id` is sufficient.

- [ ] Phase 4: Add persisted contributor identity and liveness on the same view
  identity plane
  - [ ] Milestone 1: Add `assembly_contributions` keyed by `(assembly_id, view_id)`.
  - [ ] Milestone 2: Persist current occupant metadata:
    - `binding_id`
    - `binding_value_id`
    - `coverage_plan_hash`
    - contributor daemon identity
    - coordinator generation
    - lease identity and lease generation
    - lease expiry or equivalent liveness fence
  - [ ] Milestone 3: Project runtime `ContributionLease` state into
    `assembly_contributions` instead of treating the row as the lease authority.
  - [ ] Milestone 4: Add cleanup rules so success, abort, and contributor-loss
    all produce a visible terminal state.

- [ ] Phase 5: Compile binding contributors onto the existing registration trunk
  - [ ] Milestone 1: Factor shared view/piece commit helpers out of
    `registration_backend.cc` so binding-backed contribution does not duplicate
    coverage/proof/hash logic.
  - [ ] Milestone 2: Add daemon controllers that translate `SealedBindingValue`
    plus local contribution plan into the same logical registration flow used by
    existing piece registration for disjoint phase-1 views.
  - [ ] Milestone 3: Ensure daemon-owned, client-owned, LIP-backed, and future
    byte-in sources can all hit the same commit path.

- [ ] Phase 6: Add contributor liveness and binding mutation fencing
  - [ ] Milestone 1: Implement `ContributionLease` on top of the `0011`
    lease/guard/finalizer runtime.
  - [ ] Milestone 2: Make `Binding.begin_update(...)` fail or wait while a live
    `(binding_id, binding_value_id)` contribution row exists.
  - [ ] Milestone 3: Reuse existing lifecycle concepts so contributor loss marks
    the contribution stale or aborts the attempt.
  - [ ] Milestone 4: Make coordinator seal reject stale or expired required view
    contributors.

- [ ] Phase 7: Source seal, serving publication, and frontend parity tests
  - [ ] Milestone 1: Make final seal compare the current live required view set
    against `LayoutSpec.expected_view_ids`.
  - [ ] Milestone 2: Return a `PublishedModelVersion` result that records source
    lineage, optional serving lineage, and immutable keys.
  - [ ] Milestone 3: Publish final source or serving keys only after successful
    seal and required post-assembly stages.
  - [ ] Milestone 4: Add parity coverage showing that binding, `register_view`,
    and LIP piece flows all use the same trunk semantics.

# Tasks

- [ ] Extend `proto/tensorcast/daemon/v2/store_daemon.proto`
  - add daemon RPCs for:
    - starting an assembly attempt from `layout_id`
    - submitting binding-backed contributions
    - waiting on attempt result
  - return attempt contract digest and published model-version lineage in result
    protos
  - keep `StartSealAssembly` as the seal primitive, not a second coordinator
    system

- [ ] Extend `tensorcast/schema.sql`
  - add `assembly_contributions`
  - key it by `(assembly_id, view_id)`
  - include contributor identity, coordinator generation, lease identity, and
    liveness fields
  - do not add a new phase-1 contract table if current `LayoutSpec` and
    attempt snapshots are sufficient

- [ ] Add Global Store persistence and lookup for `assembly_contributions`
  - repository for current occupant rows
  - helper methods for completeness queries and cleanup
  - no parallel slot-id plane

- [ ] Extend `daemon/service/controllers/assembly_operation_service.cc`
  - create fresh assembly attempts
  - snapshot `layout_id`, `assembly_layout_binding_version`, and
    `expected_view_ids`
  - snapshot the explicit contribution contract and compute
    `contribution_contract_hash`
  - compare current live view set against required expected view set before seal
  - return a published model-version lineage, not only a sealed source artifact

- [ ] Extend `daemon/service/controllers/registration_controller.cc`
  - add a binding-backed frontend that routes onto the same logical piece/view
    registration helper path
  - do not duplicate coverage/proof/hash logic outside the existing registration
    backend

- [ ] Refactor `core/store/runtime/metadata/registration_backend.cc`
  - extract reusable helpers for:
    - view plan computation
    - coverage derivation
    - view hashing
    - proof digest generation
    - Global Store view-state update
  - keep one commit path for `register_view`-style and binding-backed
    contributions
  - ensure phase-1 disjoint partial-view contributions do not bypass current
    overlap validation

- [ ] Extend `tensorcast/global_store/services/view_state_service.py`
  - keep current overlap/proof enforcement as the one authority
  - ensure same `(assembly_id, view_id)` replacement remains idempotent and
    explicit
  - do not add binding-specific assembly semantics here beyond contributor
    bookkeeping hooks

- [ ] Add binding mutation fence wiring
  - integrate `ContributionLease` and `assembly_contributions` lookup with
    binding state
  - make `Binding.begin_update(...)` observe live contributor occupancy
  - release fences only on success, failure, or abort

- [ ] Decide and codify phase-1 `PP` / `EP` projection mapping
  - determine how one local binding’s tensor ownership maps onto required
    disjoint `view_id`
  - compute those required `view_id` once and store them in
    `LayoutSpec.expected_view_ids`
  - keep the mapping deterministic and planner-derived
  - avoid encoding this per-update as ad-hoc runtime parameters
  - define the degenerate single-rank case with a legal contribution kind such
    as `canonical_full`, not a full-coverage piece shortcut

- [ ] Integrate contributor liveness with `0011`
  - add `ContributionLease` subject and finalizers to the daemon lifecycle
    runtime
  - persist lease projection into `assembly_contributions`
  - keep row state and runtime lease state coherent under abort, PID exit, and
    coordinator loss

- [ ] Define published model-version output and source/serving lineage
  - seal one source artifact per attempt
  - optionally run source -> serving builder or publisher as part of the same
    lineage
  - return immutable keys and manifest metadata required by serving consumers

- [ ] Keep SDK control-path routing local-daemon only
  - attempt creation
  - contribution submission
  - wait/result fetch
  - no direct SDK Global Store contract logic

# Test / Rollout / Backout

Acceptance checks against existing suites:

- `source .venv/bin/activate && pytest tests/python/test_dense_piece_assembly_sealing_acceptance.py`
- `source .venv/bin/activate && pytest tests/python/test_dual_daemon_global_store_tp_view.py`
- `source .venv/bin/activate && pytest tests/python/global_store/test_operation_rpc.py`
- `source .venv/bin/activate && pytest tests/python/global_store/test_services.py`
- `bazel test //daemon:grpc_service_impl_publish_target_replica_test`
- `bazel test //daemon:grpc_service_impl_retire_published_replica_test`
- `bazel test //daemon:grpc_service_impl_lip_piece_assembly_test`
- `bazel test //core/store:store_engine_test`

New tests to add:

- [ ] Python: fresh assembly attempt binds `layout_id` and exposes expected view set
- [ ] Python: attempt creation snapshots the explicit contribution contract and
  exposes `contribution_contract_hash`
- [ ] Python: binding-backed contribution produces the same `view_id` identity as
  the equivalent `register_view` path
- [ ] Python: `PP`-style disjoint local chunks compile into distinct required
  `view_id` and seal successfully
- [ ] Python: `EP`-style disjoint expert subsets compile into distinct required
  `view_id` and seal successfully
- [ ] Python: same `(assembly_id, view_id)` contribution can be replaced
  idempotently inside one attempt
- [ ] Python: required `expected_view_ids` missing prevents final seal
- [ ] Python: contributor liveness loss before seal aborts or stales the attempt
- [ ] Python: `Binding.begin_update(...)` is blocked or rejected while its
  current value occupies a live required `view_id`
- [ ] Python: single-rank publish succeeds through a legal contribution kind
  such as `canonical_full`
- [ ] Python: successful attempt returns `PublishedModelVersion` with source
  lineage and, when configured, serving lineage
- [ ] Python: serving-facing publish does not report success before immutable
  serving key or manifest publication
- [ ] C++: coordinator completeness is checked by required expected view set
- [ ] C++: `ContributionLease` finalizers update durable contributor state and
  release mutation fences
- [ ] C++: binding-backed contribution and existing piece registration share the
  same view-state commit helper path

Rollout order:

1. Lock the trunk semantics first in docs and proto comments.
2. Codify the phase-1 `PP` / `EP` disjoint-view mapping second.
3. Land assembly-attempt bootstrap and `assembly_contributions` persistence
   third.
4. Refactor shared registration helpers fourth so new frontends reuse them.
5. Add binding-backed contribution fifth.
6. Turn on mutation fencing and completeness enforcement sixth.
7. Update the integration guide only after the one-trunk behavior is proven by
   tests.

Backout plan:

- revert binding-backed contribution entry points first
- leave `assembly_contributions` unused rather than trying to erase schema
  history
- preserve existing `register_view`, LIP piece registration, `seal_assembly(...)`,
  `LayoutSpec`, and `StartSealAssembly` behavior intact
- do not leave a partial path that publishes without expected-view completeness or
  without releasing contributor fences

# Risks & Tracking

- Risk: the implementation accidentally creates a second assembly trunk just for
  binding.
  Mitigation: require binding-backed contribution to reuse extracted
  registration/view-state commit helpers and existing `view_id` identity.

- Risk: `expected_view_ids` remains advisory and never becomes a real
  completeness gate.
  Mitigation: make coordinator seal explicitly compare current live required view
  set against `LayoutSpec.expected_view_ids`.

- Risk: contributor identity drifts away from the current view identity plane.
  Mitigation: key persisted contributor rows by `(assembly_id, view_id)`.

- Risk: contributor liveness is still too local compared with a durable
  coordinator.
  Mitigation: persist liveness fence data in `assembly_contributions` and make
  seal reject stale contributors.

- Risk: future `register`, `put`, and byte-in frontends grow separate partial
  assembly implementations.
  Mitigation: treat this plan as the trunk contract for all partial contribution
  frontends, not just training bindings.

- Risk: phase-1 projection mapping for `PP` and `EP` drifts into one-off
  frontend logic.
  Mitigation: make the mapping planner-derived and compile it to existing
  `ViewSpec` and `view_id`, with contract tests.

Owner checklist:

- [ ] one assembly trunk remains in the system
- [ ] one layout contract trunk remains in the system
- [ ] `LayoutSpec.expected_view_ids` is the phase-1 completeness root
- [ ] `view_id` is the phase-1 contribution identity
- [ ] phase-1 `PP` and `EP` use disjoint partial views on the same trunk
- [ ] binding-backed contribution reuses the existing registration/seal path
- [ ] every publish uses a fresh `assembly_id`
- [ ] SDK never talks to Global Store directly
