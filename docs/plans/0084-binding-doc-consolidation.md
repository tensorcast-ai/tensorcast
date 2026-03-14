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

- [ ] Phase 1: Lock identity and terminology in code and types
  - [ ] Milestone 1: Introduce `binding_layout_id` and stop using `layout_id`
    for local binding identity in Python-visible surfaces.
  - [ ] Milestone 2: Introduce `SealedBindingValue` and `Binding.current_value`
    as the authoritative local model in `tensorcast/api/store/binding.py`.
  - [ ] Milestone 3: Make `Binding.artifact_id` and `Binding.selection` explicit
    optional mirrors over artifact-backed current values only.
  - [ ] Milestone 4: Update docstrings and type hints so `Mutable` and `Dirty`
    clearly mean “no current sealed value exists”.

- [ ] Phase 2: Refactor slot/backend state around local value identity
  - [ ] Milestone 1: Define a shared binding backend protocol for `InplaceSlot`
    and `OwnedBindingSlot` that exposes:
    - stable layout metadata
    - optional artifact-backed current metadata
    - local sealed value metadata
    - explicit state transitions
  - [ ] Milestone 2: Add local value identity and generation tracking:
    - `binding_value_id`
    - `seal_generation`
    - `BindingUpdateEpoch`
  - [ ] Milestone 3: Keep `Artifact.bind(...)` and `Artifact.bind_into(...)` as
    shortcuts layered over the new backend contract.

- [ ] Phase 3: Add daemon-mediated layout-seeded create and local seal control
  paths
  - [ ] Milestone 1: Extend daemon proto/controllers so layout-seeded binding
    creation does not require an initial source artifact.
  - [ ] Milestone 2: Add explicit daemon-mediated local update/seal RPCs for both
    daemon-owned and client-owned bindings.
  - [ ] Milestone 3: Regenerate Python protobufs with
    `bash tools/build_proto_python.sh` after proto changes.

- [ ] Phase 4: Enforce artifact-boundary correctness
  - [ ] Milestone 1: Wire `begin_update(...)` through existing retire/drain
    semantics so source visibility is gone before mutation begins.
  - [ ] Milestone 2: Make `seal_current(...)` produce a local-only sealed value
    by default.
  - [ ] Milestone 3: Restrict `publish_replica()` and `activate_key(...)` to
    artifact-backed current values only.
  - [ ] Milestone 4: Reserve non-blocking update begin for a future
    `Operation[BindingUpdateEpoch]` rather than shipping an unsafe bare-epoch
    API.

- [ ] Phase 5: Integrate with the existing assembly trunk
  - [ ] Milestone 1: Expose enough local metadata on `SealedBindingValue` for the
    `0085` contributor path:
    - `binding_id`
    - `binding_layout_id`
    - `binding_value_id`
    - `seal_generation`
  - [ ] Milestone 2: Ensure the local planner and layout objects preserve stable
    canonical tensor naming, ordering, and offsets so `0085` can derive disjoint
    contribution views for `PP` and `EP` without reparsing live framework
    objects.
  - [ ] Milestone 3: Make `Binding.begin_update(...)` observe contribution fences
    introduced by `0085`.
  - [ ] Milestone 4: Update
    `docs/guides/steptron-vllm-binding-integration.md` after the code shape is
    stable so guide text no longer implies local seal == artifact publish.

# Tasks

- [ ] Update `tensorcast/api/store/binding.py`
  - add `SealedBindingValue`
  - add `current_value`
  - rename local layout identity to `binding_layout_id`
  - make `artifact_id` and `selection` optional mirrors
  - keep `swap(...)` as the artifact-backed inference convenience API

- [ ] Refactor `tensorcast/api/store/inplace_slot.py`
  - carry explicit local binding state and current local value metadata
  - add `begin_update(...)` and `seal_current(...)`
  - clear artifact-backed source metadata when entering `Mutable`
  - keep current publish/retire plumbing only for artifact-backed values

- [ ] Refactor `tensorcast/api/store/owned_binding_slot.py`
  - mirror the same state machine as `InplaceSlot`
  - keep daemon-owned current-value metadata in sync with SDK-visible state

- [ ] Update `tensorcast/api/store/owned_binding_layout.py`
  - make local naming clearly separate from global `layout_id`
  - expose a stable `binding_layout_id`
  - preserve deterministic tensor metadata needed for later disjoint assembly
    view derivation

- [ ] Extend `daemon/state/binding_registry.h`
  - add binding state
  - add local value identity and generation
  - add update epoch bookkeeping
  - keep `source_selection` vs `current_selection` distinction explicit

- [ ] Extend `proto/tensorcast/daemon/v2/store_daemon.proto`
  - add layout-seeded binding create/adopt RPC shape
  - add explicit binding update/seal RPC shape
  - keep artifact routing anchored on existing `ArtifactSelection`

- [ ] Update daemon controllers
  - `daemon/service/controllers/owned_binding_service.cc`
  - `daemon/service/controllers/target_materialization_service.cc`
  - `daemon/service/controllers/target_publish_service.cc`
  - reuse existing publish and retire flows instead of inventing a second
    publish authority

- [ ] Keep SDK control-path routing local-daemon only
  - any local seal bookkeeping, artifact-backed publish token minting, or
    mutation fencing goes through Store Daemon APIs
  - do not add SDK direct Global Store access

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

- [ ] Python: layout-seeded `Store.create_binding(...)` with no initial artifact
- [ ] Python: `Binding.begin_update(...)` clears `current_value` before mutation
- [ ] Python: `Binding.seal_current(...)` returns a local-only current value with
  no artifact-backed selection
- [ ] Python: `Binding.publish_replica()` fails with `FAILED_PRECONDITION` while
  the current value is local-only
- [ ] Python: successful `swap(...)` and `seal_current(...)` both preserve tensor
  pointer stability
- [ ] C++: binding registry state and generation transitions
- [ ] C++: daemon create/adopt/seal RPC validation and retire-before-mutate

Rollout order:

1. Land type-model and naming changes first.
2. Refactor slot state and SDK surface second while preserving current inference
   behavior.
3. Add daemon proto/controller changes third.
4. Turn on layout-first create and local mutation flows after safety tests pass.
5. Integrate contribution fences from `0085` after the local model is stable.

Backout plan:

- revert new daemon binding RPCs together with the new SDK state machine
- preserve existing artifact-seeded `bind(...)`, `bind_into(...)`, and `swap(...)`
  behavior if the local mutation lifecycle proves unstable
- do not leave a partial `begin_update(...)` surface without matching
  `seal_current(...)` and drain guarantees

# Risks & Tracking

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
  Mitigation: one shared backend protocol, one state model, one dirty rule.

- Risk: binding lifecycle cleanup diverges from the daemon lease model.
  Mitigation: integrate with existing lifecycle and contribution fences instead
  of adding independent cleanup logic.

Owner checklist:

- [ ] `Binding` stays the single stable local location abstraction
- [ ] `SealedBindingValue` is the authoritative local immutable value handle
- [ ] `binding_layout_id` replaces local misuse of `layout_id`
- [ ] local sealing does not silently create artifact identity
- [ ] publish and retire remain daemon-mediated
- [ ] existing inference `swap(...)` flows stay intact
- [ ] SDK does not talk to Global Store directly
