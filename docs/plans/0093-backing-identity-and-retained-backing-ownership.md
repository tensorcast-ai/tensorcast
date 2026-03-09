---
slug: backing-identity-and-retained-backing-ownership
title: Plan - Backing Identity, Retained Backing Ownership, and Shared Lifecycle Capability
status: completed
areas: ["core", "daemon", "docs", "tests"]
created: 2026-03-09
last_updated: 2026-03-09
related_code:
  - docs/designs/0093-backing-identity-and-retained-backing-ownership.md
  - core/store/runtime/ingestion/artifact_truth.h
  - core/store/runtime/ingestion/materialization_facade.h
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/store_engine.h
  - core/store/store_engine.cc
  - daemon/service/body_backing_types.h
  - daemon/service/body_backing_manager.h
  - daemon/service/body_backing_manager.cc
  - daemon/service/byte_artifact_body_store.h
  - daemon/service/byte_artifact_body_store.cc
  - daemon/service/byte_artifact_runtime_state.h
  - daemon/service/byte_artifact_authority_service.h
  - daemon/service/byte_artifact_authority_service.cc
  - daemon/service/payload_transport_broker.h
  - daemon/service/payload_transport_broker.cc
  - daemon/service/controllers/byte_artifact_controller.cc
  - daemon/state/retention_registry.h
  - daemon/state/retention_registry.cc
  - daemon/service/grpc_service_impl_batch_runtime_test.cc
  - daemon/service/grpc_service_impl_batch_redirect_e2e_test.cc
links:
  design: ../designs/0093-backing-identity-and-retained-backing-ownership.md
---

# Objective

Land the Phase 2 contract from `0093` so that:

- `BackingIdentity` is the exact shared projection of the live retained backing subject,
- repeated bindings of the same retained backing subject are distinguished by an explicit record generation rather than
  by pretending `BackingIdentity` alone is per-instance unique,
- `AuthorityRecord` explicitly points to retained backing truth,
- `BodyStore` becomes an authority index over claim state plus a backing-truth table,
- `ServingCapability` becomes lifecycle-kernel-owned rather than body-store-owned,
- the lifecycle kernel lands as a shared adapter over existing lease and token owners before any final physical
  convergence,
- high-cardinality routed artifacts derive backing intent from `StorePolicy`,
- policy-backed rematerialization can restore visibility only under explicit, actionable proof.

# Current State & Grounding

Current baseline observed in code:

- `BackingIdentity` currently carries only `physical_artifact_id` and `device`:
  - `core/store/runtime/ingestion/artifact_truth.h`
- shared lowering already emits `VerifiedContentDescriptor`, `VerificationRecord`, and `BackingIdentity`:
  - `core/store/runtime/ingestion/materialization_facade.cc`
- `BodyBackingManager` already stages and reuses core-backed bodies, but intent derivation is still mostly
  `BodyAccessClass` plus partial stable-policy lowering:
  - `daemon/service/body_backing_manager.cc`
- `ByteArtifactRuntimeState::AuthorityEntry` still stores the visible `BodyHandle` directly:
  - `daemon/service/byte_artifact_runtime_state.h`
- `ByteArtifactBodyStore` still acts as both authority index and backing holder:
  - `daemon/service/byte_artifact_body_store.cc`
- `BodyStore::get(...)` currently returns `ServingCapability` directly even though it does not own the lifecycle kernel:
  - `daemon/service/byte_artifact_body_store.cc`
- `PayloadTransportBroker` currently synthesizes its own capability resolution result:
  - `daemon/service/payload_transport_broker.cc`
- lifecycle ownership is already split across multiple existing systems:
  - `daemon/state/session_lifecycle.h`
  - `daemon/state/handle_lease_registry.h`
  - `daemon/state/retention_registry.h`
  - `daemon/state/placement_lease_tokens.h`
- existing tests already exercise:
  - claim truth surviving backing loss,
  - bounded `payload_ref` expiry,
  - current policy-driven stable admission helpers,
  - `daemon/service/grpc_service_impl_batch_runtime_test.cc`

Hard constraints carried from earlier design phases:

- `0090` remains authoritative on routed claim truth and recovery limits,
- `0091` remains authoritative on shared content truth,
- `0092` remains authoritative on the split between content, backing, lifecycle, and authority-local state,
- `StorePolicy` remains the only declarative policy surface,
- `BodyBackingIntent` must remain an internal lowering hint.

# Latest Status

Implemented in the current cut:

- `BackingIdentity` now carries the full `ReplicaKey` and the invariant
  `physical_artifact_id == replica_key.artifact_id` is enforced in staging and
  lifecycle minting.
- `BodyStore` is split into:
  - an authority index keyed by `artifact_id`,
  - a backing table keyed by `BackingIdentity`.
- authority state now references `retained_backing_identity` instead of owning a
  visible `BodyHandle`,
- backing state now carries `instance_generation`, `BackingLifecycleState`,
  backing-local `BodyHandle`, and replica visibility indexing,
- same-identity rebind now restores visibility by bumping
  `instance_generation`,
- `ServingCapability` is minted through shared lifecycle-kernel entrypoints used
  by both `HomeBatchGet` authority reads and `PayloadTransportBroker`
  resolution,
- body intent derivation now normalizes through `BodyPlacementContext`
  (`StorePolicy + route_role + locality + access_pattern`),
- runtime/core tests were updated and pass for:
  - `//daemon:grpc_service_impl_batch_runtime_test`,
  - `//daemon:grpc_service_impl_batch_redirect_e2e_test`,
  - `//daemon:persistence_manager_test`,
  - `//daemon:grpc_service_impl_wait_for_shared_disk_test`,
  - `//core/store/runtime/ingestion:materialization_facade_test`.

First-cut scope note:

- `policy_backed_path` is now implemented for managed shared-disk persistence,
  which is the recoverable path family already backed by explicit control-plane
  truth in this repository.
- remote-stable policy-backed families remain a later extension; they are not
  required for the completed `0093` cut because the design explicitly allowed
  the first executable family set to be narrower.

# Phases & Milestones

- [x] Phase 1: Upgrade shared backing truth
  - [x] Milestone 1.1: expand `BackingIdentity` to carry the exact full `ReplicaKey` and enforce
    `physical_artifact_id == replica_key.artifact_id`.
  - [x] Milestone 1.2: add `BackingLifecycleState`, `BackingRecord`, and `instance_generation` as the shared local
    backing-truth carrier.
  - [x] Milestone 1.3: keep `BodyDescriptor` as a projection from shared content truth plus `BackingIdentity`, not as a
    second truth family.
  - [x] Milestone 1.4: define same-identity rebind semantics so repeated rematerialization bumps generation instead of
    pretending `BackingIdentity` alone is per-instance unique.

- [x] Phase 2: Split `BodyStore` into authority index and backing truth
  - [x] Milestone 2.1: replace `AuthorityEntry.visible_body_handle` as the authoritative backing pointer with explicit
    `retained_backing_identity`.
  - [x] Milestone 2.2: add a backing table keyed by `BackingIdentity`.
  - [x] Milestone 2.3: move replica visibility indexing and core reconciliation to the backing layer instead of the
    authority layer.
  - [x] Milestone 2.4: make authority reads flow through `artifact_id -> AuthorityRecord -> BackingRecord`.
  - [x] Milestone 2.5: keep any `BodyHandle` stored in `BackingRecord` explicitly scoped as a backing-local core
    reference rather than lifecycle truth.

- [x] Phase 3: Introduce the shared lifecycle capability adapter
  - [x] Milestone 3.1: stop persisting or synthesizing `ServingCapability` inside `BodyStore`.
  - [x] Milestone 3.2: introduce lifecycle-kernel mint or release entrypoints that adapt existing lifecycle owners
    rather than inventing a brand-new registry first.
  - [x] Milestone 3.3: route `HomeBatchGet` through the lifecycle kernel before emitting inline payload or `payload_ref`.
  - [x] Milestone 3.4: route `PayloadTransportBroker` capability resolution through the same kernel instead of a second
    local contract.
  - [x] Milestone 3.5: thread `lifecycle_owner_ref` plus backing generation through capability mint results.

- [x] Phase 4: Make routed byte artifacts real `StorePolicy` clients
  - [x] Milestone 4.1: introduce normalized derivation input such as `BodyPlacementContext`.
  - [x] Milestone 4.2: replace `BodyAccessClass`-only intent derivation with `StorePolicy + route_role + locality +
    access_pattern`.
  - [x] Milestone 4.3: define and store `policy_visibility_ref` separately from live backing identity.
  - [x] Milestone 4.4: make `claimed_invisible -> claimed_visible` depend on actionable policy-backed proof, not merely
    on configured policy presence.
  - [x] Milestone 4.5: require `policy_visibility_ref` to carry the exact `VerifiedContentDescriptor` and forbid pure
    local-stable retention from counting as a policy-backed rematerialization path.

- [x] Phase 5: State-machine hardening and cleanup
  - [x] Milestone 5.1: implement explicit transitions for `active`, `invalidated`, `superseded`, `draining`, and
    `retired` backing states.
  - [x] Milestone 5.2: separate backing invalidation from claim loss in code and tests.
  - [x] Milestone 5.3: ensure replacement never mutates content truth and only rebinds claim ownership when the new
    backing is join-equivalent.
  - [x] Milestone 5.4: ensure claim deletion prevents future resurrection by policy-backed rematerialization.

- [x] Phase 6: Verification, tests, and observability
  - [x] Milestone 6.1: add tests for backing replacement, invalidation, draining, and retirement.
  - [x] Milestone 6.2: add tests for capability survival across backing loss and route handoff within expiry bounds.
  - [x] Milestone 6.3: add tests for `claimed_invisible -> claimed_visible` only when policy-backed proof is live and
    actionable.
  - [x] Milestone 6.4: add tests for same-identity rematerialization rebinding through generation bump.
  - [x] Milestone 6.5: expose metrics or structured logs for:
    - backing lifecycle transitions,
    - capability mint and release,
    - policy-backed visibility installation or loss.
  - [x] Milestone 6.6: document the recoverability boundary for authority truth, backing truth, policy visibility, and
    serving capability.

# Tasks

- Core shared truth
  - update `core/store/runtime/ingestion/artifact_truth.h` to harden `BackingIdentity`,
  - keep shared lowering results carrying the upgraded backing truth without reintroducing body-private truth types.

- Daemon authority and backing layering
  - refactor `daemon/service/byte_artifact_runtime_state.h` so authority state holds backing references instead of
    `BodyHandle`,
  - refactor `daemon/service/byte_artifact_body_store.*` around authority-table and backing-table responsibilities,
  - make `daemon/service/byte_artifact_authority_service.*` consume the split snapshots rather than body-store-owned
    capabilities.

- Lifecycle capability
  - introduce lifecycle-kernel mint or release helpers and thread them through:
    - `daemon/service/controllers/byte_artifact_controller.cc`,
    - `daemon/service/payload_transport_broker.*`,
    - `daemon/state/session_lifecycle.*`,
    - `daemon/state/handle_lease_registry.*`,
    - any future handle-lease front doors touched by the cut.

- Policy client convergence
  - refactor `daemon/service/body_backing_manager.*` so intent derivation consumes normalized placement context plus
    resolved `StorePolicy`,
  - plumb whatever policy-backed visibility reference is required from retention or persistence truth into routed
    authority decisions,
  - limit first-cut `policy_backed_path` support to path kinds that already have explicit recoverable control-plane
    truth.

- Tests
  - extend daemon runtime tests around:
    - backing loss preserving claim truth,
    - bounded capability expiry,
    - policy-backed restoration,
    - superseded backing drain behavior.

# Test / Rollout / Backout

Test plan:

- `bazel test //daemon:grpc_service_impl_batch_runtime_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- `bazel test //daemon:grpc_service_impl_batch_redirect_e2e_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- add or extend focused unit tests for:
  - `ByteArtifactBodyStore`,
  - lifecycle capability mint or release helpers,
  - `BodyBackingManager` policy derivation.

Rollout strategy:

- land shared backing truth changes first,
- then land authority/backing split,
- then cut capability minting over to the lifecycle kernel adapter over existing owners,
- finally enable policy-backed visibility restoration once the actionable-proof path exists.

Backout strategy:

- if lifecycle-kernel convergence causes regressions, keep `policy_backed_path` disabled and preserve `0090`'s strict
  `ready_backing`-only visibility rule until the kernel cut is corrected,
- do not back out to the ambiguous `{physical_artifact_id, device}` backing shape once callers depend on full
  `ReplicaKey` backing truth.

# Risks & Tracking

- The authority/backing split touches hot-path routed flows and can easily regress visibility semantics if partially
  landed.
- Capability convergence spans multiple front doors and multiple existing registries; partial rollout risks leaving two
  lifecycle contracts alive at once.
- Policy-backed visibility is correctness-sensitive. A stale or merely configured policy path must never be treated as
  actionable visibility proof.
- Replacement and draining semantics are easy to get wrong if content truth and backing truth are not kept separate in
  tests.
