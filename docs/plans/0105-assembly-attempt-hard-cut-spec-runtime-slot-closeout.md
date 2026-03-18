---
slug: assembly-attempt-hard-cut-spec-runtime-slot-closeout-plan
title: Assembly Attempt Strong-Consistency Hard Cut Implementation Plan
links:
  design: ../designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md
areas: ["sdk", "daemon", "proto", "global_store", "core"]
related_code:
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/binding.py
  - tensorcast/types.py
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/operation/v1/operation.proto
  - proto/tensorcast/global_store/v1/global_store.proto
  - schema.sql
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/service/controllers/assembly_coordination_utils.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/registration_controller.cc
  - core/store/runtime/metadata/registration_backend.cc
  - core/store/components/global_store_client.cc
  - tensorcast/global_store/repositories/assembly_attempt_repository.py
  - tensorcast/global_store/repositories/assembly_readiness_cut_repository.py
  - tensorcast/global_store/repositories/assembly_slot_occupancy_repository.py
  - tensorcast/global_store/rpc/operation_rpc_handler.py
  - tests/python/test_assembly_attempt.py
  - tests/python/test_binding.py
  - tests/python/global_store/test_schema_persistence.py
  - tests/python/global_store/test_assembly_slot_occupancy_rpc.py
  - daemon/service/grpc_service_impl_start_seal_assembly_test.cc
last_updated: 2026-03-18
---

# Objective

Implement the `0105` hard cut as a first-class strong-consistency domain across
proto, schema, SDK, daemon, and Global Store.

The target end state is:

- one canonical requirement kernel,
- one separate readiness-policy kernel,
- one separate closeout-contract kernel,
- one immutable attempt record keyed by `attempt_id`,
- one structural workspace keyed by `workspace_assembly_id`,
- one workflow projection over durable truth,
- one explicit transition API for sealing,
- one durable readiness cut,
- one public continuation contract aligned with `0096` and `0100`,
- and zero compatibility aliases left in the authoritative path.

# Latest Status

Last synced: 2026-03-18

Completed in this execution wave:

- [x] Rebased the design set first:
  - [x] `0105` rewritten around durable attempt truth, explicit transition, and
        frontend-agnostic requirement kernel
  - [x] `0085` rewritten as the parent one-trunk thesis only
  - [x] `0096` updated so follow-on owners must keep `wait/status` observation
        side-effect-free
  - [x] `0100` updated so follow-on child owners must bind continuation scope to
        durable owner truth
- [x] Added first-wave schema for:
  - [x] `assembly_attempts`
  - [x] `assembly_readiness_cuts`
  - [x] `assembly_slot_occupancies`
- [x] Added Global Store Python repositories and RPC handlers for:
  - [x] durable attempt rows
  - [x] durable readiness cuts
  - [x] durable slot occupancy rows
- [x] Extended Global Store C++ client interfaces and request routing for the
      new attempt/readiness/slot-occupancy APIs
- [x] Extended daemon/store proto with:
  - [x] canonical attempt-domain messages
  - [x] explicit `SealAssemblyAttempt` RPC
  - [x] new public `AssemblyAttemptRef` fields
- [x] Regenerated Python proto code via `bash tools/build_proto_python.sh`
- [x] Updated Python public types and SDK surface:
  - [x] new assembly requirement/readiness/closeout models
  - [x] `AssemblyAttemptRef` now centers `attempt_id` and
        `workspace_assembly_id`
  - [x] explicit `Store.seal_assembly_attempt(...)`
  - [x] `Store.wait_assembly_attempt(...)` no longer triggers seal implicitly
- [x] Updated binding contribution SDK path to submit against durable
      `attempt_id` / `workspace_assembly_id`
- [x] Enforced the dependency-ready closeout cut in daemon admission:
  - [x] only `source_publish_only` is accepted in the current execution wave
  - [x] serving-facing closeout fields are rejected at attempt creation time
- [x] Updated lightweight Python tests to the new API shape

Validated so far:

- [x] `bash tools/build_proto_python.sh`
- [x] `source .venv/bin/activate && BUILD_CORE=1 BUILD_EXTENSION=1 python -vvv setup.py build_ext`
- [x] `source .venv/bin/activate && pytest tests/python/test_assembly_attempt.py tests/python/test_binding.py tests/python/global_store/test_schema_persistence.py tests/python/global_store/test_assembly_slot_occupancy_rpc.py`
  - result: `10 passed, 17 skipped`
- [x] `source .venv/bin/activate && TENSORCAST_CUDA_BACKEND=fake pytest tests/python/test_dense_piece_assembly_sealing_acceptance.py -k "canonical_full_attempt_publishes_lineage or same_slot_replacement_rejected_after_readiness_cut or liveness_loss"`
  - result: `3 passed`
- [x] Root-cause hardening for legacy direct-piece seal:
  - [x] reproduced `test_piece_bootstrap_and_seal` failure down to the low-level
        seal source selection path
  - [x] identified the authority mismatch: `seal_assembly()` always routed piece
        reads through transport keys, even when the piece replica was already
        durably present on the local daemon
  - [x] aligned `seal_assembly()` with the other assembly materialization paths
        so piece source selection is `local durable replica first`, and only
        falls back to remote transport keys when the local durable source is
        absent
  - [x] refactored the shared helper in
        `core/store/runtime/ingestion/materialization_facade.cc` so all three
        assembly-piece paths follow the same authority rule
- [x] `source .venv/bin/activate && TENSORCAST_CUDA_BACKEND=fake pytest tests/python/test_dense_piece_assembly_sealing_acceptance.py -k test_piece_bootstrap_and_seal`
  - result: `1 passed`
- [x] `source .venv/bin/activate && TENSORCAST_CUDA_BACKEND=fake pytest tests/python/test_dense_piece_assembly_sealing_acceptance.py`
  - result: `9 passed`
- [x] `source .venv/bin/activate && ruff check ...`

Still pending after this execution wave:

- [ ] daemon-side Bazel test verification after the current workspace Bazel queue
      is clear
- [ ] root-cause bridge fix: move binding contribution lowering from SDK-side
      workspace registration into daemon-owned attempt lowering
- [ ] fresh `BUILD_CORE=1 BUILD_EXTENSION=1 python -vvv setup.py build_ext`
      rerun for the latest local-piece-source refactor is still in flight while
      this status was synced

# Decision Summary

This plan assumes the repository is treated as one strong-consistency system.

Therefore the implementation must follow these rules:

1. durable semantic truth must live in first-class durable rows, not only in
   `operations.snapshot_proto`,
2. `wait_assembly_attempt(...)` must become pure observation,
3. transition to sealing must become explicit,
4. attempt identity must separate from structural workspace identity,
5. binding-local bridge enums must not remain canonical attempt truth,
6. mutable post-start policy rows must not remain authoritative,
7. compatibility-only storage aliases may be removed rather than preserved,
8. binding-backed attempt contribution lowering must be daemon-owned,
9. attempt success must imply usable result-artifact readability, not only a
   published descriptor.

# Recommended Scope Cut

Recommended first execution wave:

- land the strong-consistency attempt domain,
- make `source_publish_only` the only dependency-ready closeout contract kind,
- defer `0102` and `0104` child-contract wiring until typed child refs exist.

Reason:

- this keeps the hard cut honest,
- avoids hiding representation or rollout semantics in JSON,
- and limits the first code wave to one coherent semantic boundary.

Optional extension wave:

- if the team wants attempt success to cover representation publish or rollout
  gating immediately, extend `0102` and `0104` in the same program with typed
  child refs and explicit fail-closed resolution.

# Current-State Gaps To Remove

The hard-cut items below are now closed in code:

- `ContributionContractSnapshot` no longer remains in the authoritative path.
- `attempt_spec_hash` no longer remains in the authoritative path.
- `assembly_contributions.view_id` no longer remains slot identity.
- `assembly_runtime_policies` no longer remain authoritative attempt input.
- SDK `wait_assembly_attempt(...)` no longer triggers sealing implicitly.

Remaining gaps are concentrated in verification and rollout closure:

- proving daemon-side tests after the workspace Bazel queue clears,
- finishing the daemon-owned binding-lowering hard cut,
- tightening docs and operator-facing descriptions to the final target state,
- and keeping broader acceptance coverage around owner-loss and readiness-cut
  behavior green as follow-on refactors land.

# Target Durable Model

## New or replaced durable rows

- `assembly_attempts`
  - primary key: `attempt_id`
  - unique `workspace_assembly_id`
  - durable immutable attempt record or typed fields
- `assembly_readiness_cuts`
  - primary key: `attempt_id`
  - durable readiness-cut payload
- `assembly_slot_occupancies`
  - primary key: `(attempt_id, slot_id)`
  - keeps `structural_view_id` separate from `slot_id`
- optional `assembly_closeout_profiles`
  - keyed by `layout_id` or another explicit pre-attempt profile id
  - used only for pre-attempt closeout-contract resolution

## Existing rows that change role

- `operations`
  - remain workflow runtime and continuation projection only
  - stop serving as the only durable store for immutable attempt truth
- `artifact_bindings`
  - remain post-seal authority
- structural rows
  - remain unchanged as the structural substrate

## Rows to remove from the authoritative path

- removed `assembly_runtime_policies`
- removed `assembly_contributions.view_id` as slot identity
- removed compatibility-only embedded attempt-spec fallback bytes in public SDK refs
- `expected_view_ids` remains a layout-scoped bridge seed only and is no longer
  canonical attempt truth

# Execution Principles

- No API named `wait_*` may perform transition.
- No semantic digest may include `attempt_id` or `workspace_assembly_id`.
- No bridge-local enum may remain the repository-wide requirement kernel.
- No public continuation scope may be keyed only by workspace identity.
- No closeout semantics may be carried only as JSON in the authoritative path.

# Phases

- [x] Phase 1: Schema hard cut and durable repositories
- [x] Phase 2: Proto and public type rewrite
- [x] Phase 3: SDK surface and behavior rewrite
- [x] Phase 4: Daemon attempt creation and explicit transition rewrite
- [x] Phase 5: Binding frontend bridge rewrite
- [ ] Phase 6: Workflow continuation and recovery tightening
- [ ] Phase 7: Tests, docs, and rollout closure
- [ ] Optional Phase 8: `0102` / `0104` typed closeout child contracts

# Phase Details

## Phase 1: Schema Hard Cut And Durable Repositories

- [x] Add `assembly_attempts` to `schema.sql`.
- [x] Add `assembly_readiness_cuts` to `schema.sql`.
- [x] Add `assembly_slot_occupancies` to `schema.sql`.
- [x] Replace or retire `assembly_runtime_policies`.
- [ ] Update Global Store repositories:
  - [x] add `AssemblyAttemptRepository`
  - [x] add `AssemblyReadinessCutRepository`
  - [x] replace `AssemblyContributionRepository` with slot-occupancy semantics
  - [ ] narrow `OperationRepository` usage so it no longer acts as the durable
        attempt-spec store
- [x] Delete `assembly_contributions` from the authoritative path rather than
      retaining it as a migration helper.

Required rule:

- no later phase may keep immutable attempt truth only inside operation
  snapshots.

## Phase 2: Proto And Public Type Rewrite

- [x] Replace binding-shaped attempt carriers in
  `proto/tensorcast/daemon/v2/store_daemon.proto` with:
  - [x] `AssemblyTargetRef`
  - [x] `AssemblyRequirement`
  - [x] `AssemblyRequirementSetRef`
  - [x] `AssemblyReadinessPolicy`
  - [x] `AssemblyCloseoutContract`
  - [x] `AssemblyAttemptIntent`
  - [x] `AssemblyAttemptRecord`
  - [x] `AssemblyReadinessCut`
- [x] Add explicit attempt transition RPC:
  - [x] `SealAssemblyAttempt` or equivalent attempt-scoped transition RPC
- [ ] Keep low-level structural `StartSealAssembly` or equivalent workspace-only
  primitive below the attempt surface.
- [x] Update `OperationRef` usage rules in code so:
  - [x] `authority_scope_id = attempt_id`
  - [x] `target_artifact_id = workspace_assembly_id`
  - [x] `fencing_digest = attempt_intent_digest`
- [x] Update `tensorcast/types.py`:
  - [x] remove `expected_view_ids`
  - [x] remove embedded fallback attempt-spec bytes from public recovery path
  - [x] add strong-typed attempt ref fields

Required rule:

- public attempt refs must be durable-scope projections, not mini truth blobs.

## Phase 3: SDK Surface And Behavior Rewrite

- [x] Update `Store.start_assembly_attempt(...)` to accept canonical
  requirement-set, readiness-policy, and closeout-contract inputs.
- [x] Add `Store.seal_assembly_attempt(...)`.
- [x] Rewrite `Store.wait_assembly_attempt(...)` as pure observation.
- [x] Remove implicit seal trigger from SDK wait path.
- [x] Keep bare string operation waiting only on low-level `wait_operation(...)`.
- [x] Update `SealedBindingValue.contribute_to_assembly(...)` so it lowers to the
  canonical attempt bridge rather than to a binding-shaped attempt contract.

Required rule:

- SDK observation and SDK transition must be separate methods.

## Phase 4: Daemon Attempt Creation And Explicit Transition Rewrite

- [x] Rewrite `start_assembly_attempt(...)` in
  `assembly_operation_service.cc` so it:
  - [x] canonicalizes a frontend bridge input into `AssemblyRequirementSetRef`
  - [x] resolves readiness policy
  - [x] resolves or validates closeout contract
  - [x] computes `attempt_intent_digest`
  - [x] creates durable `assembly_attempts` row
  - [x] creates open workflow runtime
- [x] Rewrite attempt transition so explicit `seal_assembly_attempt(...)`:
  - [x] validates open phase
  - [x] moves runtime to sealing
  - [x] captures durable readiness cut
  - [x] starts structural seal and closeout
- [x] Remove fallback logic that reconstructs attempt truth from layout hints or
      embedded public bytes.

Required rule:

- daemon workflow code may project durable truth, but must not recreate it from
  weaker carriers.

## Phase 5: Binding Frontend Bridge Rewrite

- [x] Rewrite `assembly_coordination_utils.cc` around the new canonical kernel.
- [x] Treat binding-local enums and coverage-plan hashes as bridge-local only.
- [x] Lower binding contribution into:
  - [x] canonical `slot_id`
  - [x] `AssemblyTargetRef`
  - [x] occupancy claim against `(attempt_id, slot_id)`
- [ ] Move final contribution lowering authority from SDK/frontend code to
      daemon-owned attempt handlers.
- [ ] Keep structural registration path shared with direct `register_view`.
- [ ] Ensure daemon-side lowering applies active layout overlap policy,
      including replicated-tensor participation, before workspace registration
      commits.
- [x] Update `owned_binding_service.cc` so slot occupancy uses the new durable
  slot model rather than `(assembly_id, view_id)` aliasing.

Required rule:

- binding bridge code must not remain the owner of attempt truth or of the
  final structural lowering payload.

## Phase 6: Workflow Continuation And Recovery Tightening

- [x] Make attempt workflow continuation durable-scope and fail-closed.
- [x] Ensure owner-loss and malformed reentry are reported through the attempt
      owner contract, not through layout reconstruction.
- [x] Ensure `wait/status/cancel` operate on existing workflow state only.
- [ ] Ensure `retry_later` is not introduced for attempt workflow in this wave.
- [ ] Keep recovery-class reporting honest:
  - [ ] if durable rows are complete, report `cluster_durable`
  - [ ] otherwise do not claim that class

Required rule:

- no attempt continuation may depend on a weaker contract than `publish`
  already uses under `0096` and `0100`.

## Phase 6A: Result Availability Tightening

- [ ] Ensure `seal_assembly_attempt(...)` success means the returned
      `PublishedModelVersion.source_artifact_id` is readable through ordinary
      daemon artifact reads.
- [ ] If local registration, attachment, or post-seal replication is required
      for that invariant, complete it before reporting success.

Required rule:

- result projection must not claim a usable source artifact unless the daemon can
  satisfy ordinary post-success reads for it.

## Phase 7: Tests, Docs, And Rollout Closure

- [ ] Rewrite or replace tests for:
  - [ ] attempt-intent digest separation
  - [x] explicit transition versus observation
  - [x] durable slot occupancy keyed by `slot_id`
  - [ ] readiness-cut capture and replacement closure
  - [ ] closeout-contract freeze
  - [ ] owner-loss fail-closed behavior
- [ ] Update docs and operators only after code matches the new domain model.
- [ ] Delete stale compatibility docs that still describe:
  - [ ] `expected_view_ids`
  - [ ] `assembly_id` as attempt identity
  - [x] `wait` as implicit seal
  - [ ] mutable assembly-scoped runtime policy authority

## Optional Phase 8: `0102` / `0104` Typed Closeout Child Contracts

This phase is optional.
It should be taken only if the team wants attempt success to include
representation publish or rollout gating in the same implementation program.

### Option A: Defer `0102` / `0104` wiring

Recommended.

- [ ] keep `AssemblyCloseoutContract.kind = "source_publish_only"` as the only
  dependency-ready contract kind
- [ ] reserve extension points in proto and code, but do not wire them
- [ ] land the strong-consistency attempt domain first

Benefits:

- smaller code wave
- cleaner semantic closure
- no fake typed child refs

Cost:

- final attempt success initially stops at source published lineage

### Option B: Include `0102` / `0104` in the same execution program

Only choose this if the team explicitly wants broader attempt closeout now.

- [ ] modify `0102` to expose typed representation or manifest closeout refs
- [ ] modify `0104` to expose typed rollout barrier refs
- [ ] extend `AssemblyCloseoutContract`
- [ ] extend daemon closeout logic to resolve and validate those refs
- [ ] add cross-design tests for representation and rollout child closeout

Benefits:

- broader final-success scope in one wave

Costs:

- larger blast radius across engine integration and rollout semantics
- more cross-design code changes
- higher risk of mixing not-yet-ready child contracts into the first hard cut

Recommendation:

- do not include Option B in the first implementation wave unless there is a
  strong product reason to make attempt success broader than source publish
  immediately.

# Workstreams By Area

## Proto

- [ ] redefine attempt-domain messages
- [ ] add explicit attempt transition RPC
- [ ] remove deprecated attempt fallback fields from the authoritative path

## Schema / Global Store

- [ ] add durable attempt tables
- [ ] add repositories and RPC support
- [ ] stop using operation snapshots as immutable truth storage

## SDK

- [ ] new public attempt types
- [ ] explicit transition API
- [ ] wait/status rewrite
- [ ] binding frontend bridge update

## Daemon

- [ ] create durable attempt rows
- [ ] create durable readiness cut
- [ ] explicit transition handling
- [ ] occupancy rewrite keyed by `slot_id`

## Tests

- [ ] Python SDK tests
- [ ] Global Store repository tests
- [ ] daemon grpc and controller tests
- [ ] end-to-end attempt flow tests

# Validation Matrix

The implementation should not be considered complete until it proves:

- [ ] two attempts with the same semantic intent and different `attempt_id`
      produce the same `attempt_intent_digest`
- [ ] two attempts with different readiness policy do not share the same intent
      digest
- [x] `wait_assembly_attempt(...)` never triggers sealing
- [ ] explicit `seal_assembly_attempt(...)` is required before sealing begins
- [ ] slot occupancy is keyed by `(attempt_id, slot_id)` rather than by
      `(assembly_id, view_id)`
- [ ] contributor loss before the cut invalidates the attempt when readiness
      policy requires live occupancy
- [ ] contributor loss after the cut does not mutate the already-captured cut
- [ ] closeout-contract edits after attempt creation do not affect in-flight
      attempt semantics
- [ ] owner loss or malformed continuation fails closed
- [ ] `OperationRef.authority_scope_id` names durable attempt scope

# Suggested Test Commands For The Later Execution Wave

- [ ] `source .venv/bin/activate && pytest tests/python/test_assembly_attempt.py`
- [ ] `source .venv/bin/activate && pytest tests/python/test_binding.py`
- [x] `source .venv/bin/activate && pytest tests/python/global_store/test_schema_persistence.py tests/python/global_store/test_assembly_slot_occupancy_rpc.py`
- [ ] `bazel test //daemon:grpc_service_impl_start_seal_assembly_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`

# Risks

- Risk: code lands the new names but keeps old authority paths behind them.
  Mitigation: require repository and SDK tests to prove observation is
  side-effect-free and immutable truth is not operation-snapshot-only.
- Risk: binding bridge logic remains the de facto attempt owner.
  Mitigation: centralize canonical requirement and slot-occupancy helpers and
  test non-binding frontends against the same carriers.
- Risk: `0102` or `0104` pressure expands closeout scope too early.
  Mitigation: keep `source_publish_only` as the only dependency-ready closeout
  kind in the first execution wave.

# Owner Checklist

- [ ] canonical requirement kernel is frontend-agnostic
- [ ] readiness policy is separate from requirement identity
- [ ] closeout contract is typed and separate from workflow runtime
- [ ] attempt identity is separate from workspace identity
- [ ] immutable attempt truth is stored durably outside workflow snapshots
- [ ] explicit transition API exists
- [ ] `wait_assembly_attempt(...)` is observation only
- [ ] slot occupancy uses first-class `slot_id`
- [ ] continuation aligns with `0096` and `0100`
- [ ] first-wave closeout scope is explicit and honest
