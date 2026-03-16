---
slug: binding-unified-model-and-contract-plan
title: Binding Unified Model and Contract Plan
links:
  design: ../designs/0084-binding-unified-model-and-contract.md
areas: ["sdk", "daemon", "core", "proto"]
related_code:
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/inplace_slot.py
  - tensorcast/api/store/owned_binding_slot.py
  - tensorcast/api/store/owned_binding_layout.py
  - tensorcast/api/store/materialization.py
  - tensorcast/daemon_ctl.py
  - proto/tensorcast/common/v1/common.proto
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/target_materialization_service.cc
  - daemon/service/controllers/target_publish_service.cc
  - daemon/state/binding_registry.h
  - daemon/state/session_lifecycle.cc
  - tests/python/test_binding.py
  - tests/python/test_inplace_slot.py
  - tests/python/api/test_mapped_binding.py
  - tests/python/daemon/test_inplace_slot_swap_publish_e2e.py
---

# Objective

Implement the `0084` local binding model so TensorCast has one stable local
binding abstraction with an explicit mutable-to-sealed lifecycle and a clean
boundary to the existing artifact and assembly trunks.

Target end state:

- `Binding` is the stable local location
- `SealedBindingValue` is the authoritative current local immutable value
- `binding_layout_id` replaces local misuse of `layout_id`
- artifact-backed inference flows still use `swap(...)`
- training flows use `begin_update(...)` and `seal_current(...)`
- local sealing does **not** silently mint a second artifact identity plane
- promotion of local-only sealed values is deferred to the assembly trunk in
  `0085`

# Current State & Grounding

Current SDK and daemon behavior already provide most of the mechanics needed for
the final model, but they are still artifact-first and identity-overloaded.

- `tensorcast/api/store/binding.py`
  - `Binding` is a thin wrapper over slot implementations.
  - Public lifecycle is limited to `swap(...)`, `publish_replica()`, and
    `close()`.
  - `artifact_id` and `selection` are assumed to exist unconditionally.

- `tensorcast/api/store/inplace_slot.py`
  - client-owned binding already preserves pointer stability across refill
  - stores selection identity fields and uses daemon-minted `target_publication_token`
  - has no explicit `Mutable` window or local-only `seal_current(...)` path

- `tensorcast/api/store/owned_binding_slot.py`
  - daemon-owned binding already has stable refill and publish/retire support
  - still assumes create-time source artifact exists

- `tensorcast/api/store/owned_binding_layout.py`
  - already computes a deterministic local layout hash for owned bindings
  - still uses naming that is easy to confuse with global `layout_id`
  - already captures the ordered tensor metadata that `0085` will later need in
    order to derive deterministic disjoint contribution views

- `proto/tensorcast/daemon/v2/store_daemon.proto`
  - `CreateOwnedBinding` and `RefillOwnedBinding` are artifact-seeded
  - there is no layout-seeded create path with no initial source
  - there is no explicit binding update/seal lifecycle RPC

- `daemon/state/binding_registry.h`
  - already distinguishes `source_selection` from `current_selection`
  - does not yet track binding state, local value identity, or update epochs

- `daemon/service/controllers/target_publish_service.cc`
  - publish authority already assumes artifact-backed `ArtifactSelection`
  - this confirms the design constraint that local sealing must not auto-create a
    second artifact plane

- `docs/designs/0078-selection-first-artifact-retrieval.md`
  - `ArtifactSelection` is already the canonical artifact-plane selection
    contract
  - we must not synthesize parallel artifact semantics in the binding layer

- `docs/designs/0011-unified-session-lifecycle-leases.md`
  - daemon lifecycle already has one lease/guard/finalizer model
  - binding lifecycle must integrate with that model rather than introducing
    independent cleanup behavior

Baseline constraints this plan must respect:

- SDK code must not talk to Global Store directly.
- Existing `Artifact.bind(...)`, `Artifact.bind_into(...)`, and `swap(...)`
  behavior must remain available during migration.
- `layout_id` remains reserved for the existing Global Store `LayoutSpec`
  identity.
- Local-only `seal_current(...)` must not create a hidden artifact/key-routing
  contract.

# Phases & Milestones

- [x] Phase 1: Lock identity and terminology in code and types
  - [x] Milestone 1: Introduce `binding_layout_id` and stop using `layout_id`
    for local binding identity in Python-visible surfaces.
  - [x] Milestone 2: Introduce `SealedBindingValue` and `Binding.current_value`
    as the authoritative local model in `tensorcast/api/store/binding.py`.
  - [x] Milestone 3: Make `Binding.artifact_id` and `Binding.selection` explicit
    optional mirrors over artifact-backed current values only.
  - [x] Milestone 4: Update docstrings and type hints so `Mutable` and `Dirty`
    clearly mean “no current sealed value exists”.

- [x] Phase 2: Refactor slot/backend state around local value identity
  - [x] Milestone 1: Define a shared binding backend protocol for `InplaceSlot`
    and `OwnedBindingSlot` that exposes:
    - stable layout metadata
    - optional artifact-backed current metadata
    - local sealed value metadata
    - explicit state transitions
  - [x] Milestone 2: Add local value identity and generation tracking:
    - `binding_value_id`
    - `seal_generation`
    - `BindingUpdateEpoch`
  - [x] Milestone 3: Keep `Artifact.bind(...)` and `Artifact.bind_into(...)` as
    shortcuts layered over the new backend contract.

- [x] Phase 3: Add daemon-mediated layout-seeded create and local seal control
  paths
  - [x] Milestone 1: Extend daemon proto/controllers so layout-seeded binding
    creation does not require an initial source artifact.
  - [x] Milestone 2: Add explicit daemon-mediated local update/seal RPCs for both
    daemon-owned and client-owned bindings.
  - [x] Milestone 3: Regenerate Python protobufs with
    `bash tools/build_proto_python.sh` after proto changes.

- [x] Phase 4: Enforce artifact-boundary correctness
  - [x] Milestone 1: Wire `begin_update(...)` through existing retire/drain
    semantics so source visibility is gone before mutation begins.
  - [x] Milestone 2: Make `seal_current(...)` produce a local-only sealed value
    by default.
  - [x] Milestone 3: Restrict `publish_replica()` and `activate_key(...)` to
    artifact-backed current values only.
  - [x] Milestone 4: Reserve non-blocking update begin for a future
    `Operation[BindingUpdateEpoch]` rather than shipping an unsafe bare-epoch
    API.

- [x] Phase 4.5: Harden contract authority and failure semantics
  - [x] Milestone 1: Make daemon responses authoritative for current sealed-value
    identity on successful create/refill/commit/seal paths; remove SDK fallback
    synthesis of `binding_value_id` / `seal_generation`.
  - [x] Milestone 2: Make `BindingUpdateEpoch` binding-scoped and reject
    wrong-binding token reuse in both SDK and daemon paths.
  - [x] Milestone 3: Enforce the strict dirty rule so failed overwrite/seal
    paths clear `current_value` and artifact-backed mirrors.
  - [x] Milestone 4: Honor `wait_events` as a real barrier or fail fast on
    unsupported barrier inputs.
  - [x] Milestone 5: Make layout-seeded `Store.create_binding(..., mapping=...)`
    either fully wired or explicitly rejected; no silent ignore.

- [ ] Phase 5: Integrate with the existing assembly trunk
  - [x] Milestone 1: Expose enough local metadata on `SealedBindingValue` for the
    `0085` contributor path:
    - `binding_id`
    - `binding_layout_id`
    - `binding_value_id`
    - `seal_generation`
  - [x] Milestone 2: Ensure the local planner and layout objects preserve stable
    canonical tensor naming, ordering, and offsets so `0085` can derive disjoint
    contribution views for `PP` and `EP` without reparsing live framework
    objects.
  - [ ] Milestone 3: Make overwrite entry points observe contribution fences
    introduced by `0085`, including `Binding.begin_update(...)` and any
    byte-overwriting `swap(...)` path.
  - [ ] Milestone 4: Update
    `docs/guides/steptron-vllm-binding-integration.md` after the code shape is
    stable so guide text no longer implies local seal == artifact publish.

# Tasks

- [x] Update `tensorcast/api/store/binding.py`
  - add `SealedBindingValue`
  - add `current_value`
  - rename local layout identity to `binding_layout_id`
  - make `artifact_id` and `selection` optional mirrors
  - keep `swap(...)` as the artifact-backed inference convenience API

- [x] Refactor `tensorcast/api/store/inplace_slot.py`
  - carry explicit local binding state and current local value metadata
  - add `begin_update(...)` and `seal_current(...)`
  - clear artifact-backed source metadata when entering `Mutable`
  - keep current publish/retire plumbing only for artifact-backed values

- [x] Refactor `tensorcast/api/store/owned_binding_slot.py`
  - mirror the same state machine as `InplaceSlot`
  - keep daemon-owned current-value metadata in sync with SDK-visible state

- [x] Update `tensorcast/api/store/owned_binding_layout.py`
  - make local naming clearly separate from global `layout_id`
  - expose a stable `binding_layout_id`
  - preserve deterministic tensor metadata needed for later disjoint assembly
    view derivation

- [x] Extend `daemon/state/binding_registry.h`
  - add binding state
  - add local value identity and generation
  - add update epoch bookkeeping
  - keep `source_selection` vs `current_selection` distinction explicit

- [x] Extend `proto/tensorcast/daemon/v2/store_daemon.proto`
  - add layout-seeded binding create/adopt RPC shape
  - add explicit binding update/seal RPC shape
  - keep artifact routing anchored on existing `ArtifactSelection`

- [x] Update daemon controllers
  - `daemon/service/controllers/owned_binding_service.cc`
  - reused existing publish/retire authority in `target_materialization_service.cc`
    and `target_publish_service.cc` without adding a second publish plane
  - reuse existing publish and retire flows instead of inventing a second
    publish authority

- [x] Keep SDK control-path routing local-daemon only
  - any local seal bookkeeping, artifact-backed publish token minting, or
    mutation fencing goes through Store Daemon APIs
  - do not add SDK direct Global Store access

- [x] Harden SDK/daemon authority boundaries
  - remove silent SDK synthesis of current-value identity on successful daemon
    responses
  - make missing daemon-authored `current_value` on success a hard contract
    failure
  - keep `binding_value_id` trustworthy for `0085` contributor occupancy and
    mutation fencing

- [ ] Normalize overwrite control semantics
  - [x] make update tokens binding-scoped
  - [x] keep dirty transitions destructive to `current_value`
  - avoid partial state machines split across proto state and local SDK booleans

- [x] Complete layout-seeded mapped binding support
  - thread `mapping` through `Store.create_binding(...)`
  - persist or mirror mapped overwrite metadata for later `swap(...)`
  - reject unsupported mapped create forms explicitly

# Test / Rollout / Backout

Acceptance checks against existing suites:

- `source .venv/bin/activate && pytest tests/python/test_binding.py`
- `source .venv/bin/activate && pytest tests/python/test_inplace_slot.py`
- `source .venv/bin/activate && pytest tests/python/api/test_mapped_binding.py`
- `source .venv/bin/activate && pytest tests/python/daemon/test_inplace_slot_swap_publish_e2e.py`
- `bazel test //daemon:grpc_service_impl_publish_target_replica_test`
- `bazel test //daemon:grpc_service_impl_retire_published_replica_test`
- `bazel test //daemon:session_lifecycle_test`
- `bazel test //core/common:selection_identity_test`

New tests to add:

- [x] Python: layout-seeded `Store.create_binding(...)` with no initial artifact
- [x] Python: `Binding.begin_update(...)` clears `current_value` before mutation
- [x] Python: `Binding.seal_current(...)` returns a local-only current value with
  no artifact-backed selection
- [x] Python: `Binding.publish_replica()` fails with `FAILED_PRECONDITION` while
  the current value is local-only
- [x] Python: successful `swap(...)` and `seal_current(...)` both preserve tensor
  pointer stability
- [x] Python: wrong-binding `BindingUpdateEpoch` is rejected cleanly
- [x] Python: failed refill/overwrite clears `current_value` and enters `Dirty`
- [x] Python: successful control-path response missing `current_value` fails fast
  instead of minting local replacement identity
- [x] Python: `wait_events` are synchronized before transition RPCs
- [x] Python: layout-seeded `Store.create_binding(..., mapping=...)` forwards
  mapped overwrite metadata
- [ ] C++: binding registry state and generation transitions
- [ ] C++: daemon create/adopt/commit/seal RPC validation and retire-before-mutate
- [ ] C++: failed daemon-owned refill transitions the binding to `Dirty`

Rollout order:

1. Land type-model and naming changes first.
2. Refactor slot state and SDK surface second while preserving current inference
   behavior.
3. Add daemon proto/controller changes third.
4. Harden authority, dirty semantics, and update-token correctness before
   expanding more binding frontends.
5. Turn on layout-first create and local mutation flows after safety tests pass.
6. Integrate contribution fences from `0085` after the local model is stable.

Backout plan:

- revert new daemon binding RPCs together with the new SDK state machine
- preserve existing artifact-seeded `bind(...)`, `bind_into(...)`, and `swap(...)`
  behavior if the local mutation lifecycle proves unstable
- do not leave a partial `begin_update(...)` surface without matching
  `seal_current(...)` and drain guarantees

# Risks & Tracking

## Status Update (2026-03-14)

Implemented in this change set:

- Added `BindingLayout.binding_layout_id`, `Binding.current_value`,
  `SealedBindingValue`, and `BindingUpdateEpoch`.
- Made `Binding.artifact_id` / `Binding.selection` optional mirrors over the
  current artifact-backed sealed value only.
- Added `Binding.begin_update(...)`, `Binding.seal_current(...)`,
  `Binding.retire(...)`, and layout-seeded `Store.create_binding(...)`.
- Refactored `InplaceSlot` / `OwnedBindingSlot` around shared local-value
  metadata and explicit mutable/sealed transitions.
- Added daemon RPCs for `CreateBinding`, `CommitBindingArtifact`,
  `BeginBindingUpdate`, and `SealBinding`, and extended owned-binding responses
  with local value metadata.
- Hardened successful control paths so artifact-backed and local sealed-value
  identity stays daemon-authored; SDK now fails fast on missing authoritative
  `current_value` instead of silently minting replacement ids.
- Made `BindingUpdateEpoch` binding-scoped in both daemon token generation and
  SDK validation.
- Made failed overwrite/seal paths clear `current_value` and artifact-backed
  mirrors before surfacing `Dirty`.
- Honored `wait_events` as a synchronous transition barrier.
- Wired layout-seeded `Store.create_binding(..., mapping=...)` through the
  mapped overwrite contract.
- Regenerated Python protobufs with `bash tools/build_proto_python.sh`.

Verification status:

- Passed:
  - `source .venv/bin/activate && pytest tests/python/test_binding.py`
  - `source .venv/bin/activate && pytest tests/python/test_inplace_slot.py`
  - `source .venv/bin/activate && pytest tests/python/api/test_mapped_binding.py`
  - `source .venv/bin/activate && pytest tests/python/test_assembly_attempt.py`
  - `source .venv/bin/activate && pytest tests/python/daemon/test_inplace_slot_swap_publish_e2e.py`
  - `source .venv/bin/activate && ruff check tensorcast/api/store/artifact.py tensorcast/api/store/binding.py tensorcast/api/store/inplace_slot.py tensorcast/api/store/owned_binding_slot.py tensorcast/api/store/__init__.py tests/python/test_binding.py tests/python/test_inplace_slot.py tests/python/api/test_mapped_binding.py`
  - `bazel test //daemon:grpc_service_impl_publish_target_replica_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  - `bazel test //daemon:grpc_service_impl_retire_published_replica_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  - `bazel test //daemon:session_lifecycle_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  - `bazel test //core/common:selection_identity_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`

- Risk: existing callers assume `Binding.artifact_id` is always non-null.
  Mitigation: land `current_value` and optional typing before changing call
  sites; keep artifact-backed inference compatibility helpers.

- Risk: local binding code keeps leaking global `layout_id` terminology.
  Mitigation: rename aggressively at the boundary; reserve `layout_id` for the
  Global Store layout contract only.

- Risk: local seal accidentally becomes a hidden artifact plane again.
  Mitigation: keep publish/key activation artifact-backed only and route
  promotion through `0085`.

- Risk: daemon-owned and client-owned bindings fork into different state
  machines.
  Mitigation: one shared backend protocol, one dirty rule, and binding-scoped
  overwrite tokens; further state-dimension normalization remains follow-up work.

- Risk: binding lifecycle cleanup diverges from the daemon lease model.
  Mitigation: integrate with existing lifecycle and contribution fences instead
  of adding independent cleanup logic.

Owner checklist:

- [x] `Binding` stays the single stable local location abstraction
- [x] `SealedBindingValue` is the authoritative local immutable value handle
- [x] `binding_layout_id` replaces local misuse of `layout_id`
- [x] local sealing does not silently create artifact identity
- [x] publish and retire remain daemon-mediated
- [x] existing inference `swap(...)` flows stay intact
- [x] SDK does not talk to Global Store directly
- [x] successful control paths do not rely on SDK-synthesized current-value
  identity
- [x] wrong-binding update-token reuse is rejected
- [x] `Dirty` clears `current_value` and artifact-backed mirrors
- [x] layout-seeded mapped create is not silently ignored
