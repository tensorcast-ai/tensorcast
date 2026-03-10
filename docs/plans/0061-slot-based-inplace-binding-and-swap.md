---
slug: 0061-slot-based-inplace-binding-and-swap
title: Plan - Slot-Based Inplace Binding and Safe Swap (Client-Owned VRAM)
links:
  design: ../designs/0061-slot-based-inplace-binding-and-swap.md
areas: ["sdk", "daemon", "core", "global_store", "proto"]
related_code:
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/deferred_loader.py
  - tensorcast/api/store/materialization.py
  - tensorcast/api/store/README.md
  - tests/python/test_deferred_loader.py
  - daemon/service/grpc_service_impl.cc
  - daemon/state/lip_manager.{h,cc}
  - daemon/state/transport_lock_manager.h
  - daemon/state/ipc_region_registry.{h,cc}
  - daemon/service/controllers/materialization_controller.cc
  - daemon/service/controllers/registration_controller.cc
  - proto/tensorcast/common/v1/capability_token.proto
  - proto/tensorcast/common/v1/common.proto
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/global_store/v1/global_store.proto
---

# Objective

Implement the slot-based inplace binding + safe swap flow defined in `docs/designs/0061-slot-based-inplace-binding-and-swap.md`:

- Introduce an SDK-owned `InplaceSlot` abstraction for stable client-owned VRAM layouts (pointer-stable tensor storages).
- Provide safe swap ordering: `preflight -> retire old -> overwrite -> (optional) publish new` (ByteSpace-aware).
- Adopt the repo’s explicit identity model:
  - `ByteSpaceRef` for routing (`CANONICAL` or `VIEW(view_id)`).
  - `ArtifactSelection` (`logical_layout_hash` + `selection_hash`) as the operation/layout identity bound into tokens and retries.
- Standardize the cross-layer **selection encoding contract** so “full coverage” vs “subset-packed” is decidable and
  `selection_hash` is stable across Python/C++ (proto3 empty-bytes normalization).
- Make swap safety match the system’s source of truth by using **Global Store transport drain** (`is_available` +
  `current_requests`) as the Phase-1 correctness barrier for overwrite; daemon-local locks remain best-effort hardening.
- Thread an explicit `operation_id` (SDK-generated UUID) through retire/materialize/publish and GS drain calls for
  correlation, debugging, and safe retries (without changing routing identity).
- Enable publishing a filled target as a routable memory replica **without re-hashing GPU memory** by using a daemon-minted
  write capability (`target_write_token`) and a new `PublishTargetReplica` RPC.
- Enforce publishability constraints: Phase-1 `publish_replica()` only supports full-coverage, routable ByteSpaces
  (canonical or view); packed subset layouts remain local-only until selection-aware routing exists.
- (Optional) Improve artifact handle ergonomics with a non-ambiguous `tc.artifact(ref)` shorthand.

# Status (2026-02-01)

- Implementation complete across SDK/daemon/global store; phases 0-4 are done.
- Slot swap/publish integration and ByteSpace-aware retire are wired end-to-end with tests.

# Current State & Grounding

- SDK deferred loader is implemented:
  - `tensorcast/api/store/deferred_loader.py` allocates a client-owned CUDA arena, registers it as a VRAM region, and returns placeholder tensors via `Tensor.set_(untyped_storage, offset, shape, stride)`.
  - `DeferredLoader.commit()` performs a single `materialize_into_target_v2(...)` call and returns `DeferredCommitResult`.
  - `publish=True` on `commit()` calls `Store.register(... plan=VRAM_LEASED, lease_in_place=True)`, which is "register new artifact id" (hash/index + optional GS publish), not "publish this target as a replica of an existing artifact".
  - Docs/tests currently encode this contract: `tensorcast/api/store/README.md` and `tests/python/test_deferred_loader.py`.
- Daemon `MaterializeIntoTarget` is local-only and has no "publish capability":
  - Handler: `daemon/service/controllers/materialization_controller.cc:materialize_into_target(...)`.
  - Proto: `proto/tensorcast/daemon/v2/store_daemon.proto:MaterializeIntoTargetResponse` has no `target_write_token`.
- Daemon retire surface is canonical-only and does not target ByteSpace explicitly:
  - RPC: `proto/tensorcast/daemon/v2/store_daemon.proto:DeregisterArtifact`.
  - Handler: `daemon/service/grpc_service_impl.cc:DeregisterArtifact` calls `LipManager::quiesce_artifact(...)`, optionally waits for staged exports to drain, revokes the commit lease, and then calls `StoreEngine::unregister_replica_from_global_store(artifact_id, device_id)`.
  - It does not accept `tensorcast.common.v1.ByteSpaceRef` and does not use `LipManager`’s view-scoped/routable lease tracking (`LipManager::attach_replica_id(...)`).
- Transport locks exist but are not integrated into retire-for-overwrite:
  - `LockTransportChunks`/`UnlockTransportChunks` RPCs exist; daemon-local tracking is in `daemon/state/transport_lock_manager.h`.
  - `DeregisterArtifact` does not gate on transport locks, so routable source reads could overlap with overwrite.
- Global Store transport sessions are the actual in-flight source of truth for P2P:
  - P2P source selection and in-flight counting use `RequestReplicaTransport`/`CompleteReplicaTransport` and `current_requests` in Global Store (`tensorcast/global_store/services/transport_service.py`).
  - The existing `LockTransportChunks` path is only meaningful for the legacy staged LIP export path; for UMA V3 it is explicitly “bookkeeping only” (`daemon/service/controllers/transport_controller.cc`).
  - GS has a stale-transport cleanup safety net (`TransportService.cleanup_expired_transports(...)`); drain RPCs should expose good diagnostics and remain bounded by deadlines.
- Global Store currently lacks a synchronous API for swap retire:
  - GS persists `is_available` and `current_requests` but does not expose an RPC to mark a specific `replica_id` unavailable or to wait for drain by `replica_id` (needed for swap correctness).
- There is an existing path to register a routable view lease + publish it to Global Store:
  - `daemon/service/controllers/registration_controller.cc` uses `LipManager::commit_routable_view_lease_in_place(...)` and `IGlobalStoreClient::register_memory_replica(...)`.
  - This is tied to the register/commit flow (LIP), not to `MaterializeIntoTarget` outputs.
- ByteSpaceRef is already a first-class type in common/global store protos:
  - `proto/tensorcast/common/v1/common.proto:ByteSpaceRef`
  - `proto/tensorcast/global_store/v1/global_store.proto:RegisterReplicaRequest.mem_info.byte_space`
- Selection identity utilities already exist and must be reused (no new ad-hoc identities):
  - Proto: `proto/tensorcast/common/v1/common.proto:ArtifactSelection`
  - Python: `tensorcast/common/selection_identity.py`
  - C++: `core/common/selection_identity.{h,cc}`
- Selection encoding currently diverges from the desired “full coverage” contract:
  - SDK region-backed layout building computes `view_subset_hash` for any selection (even full coverage) and may use `INDEX_KIND_VIEW` when an explicit `selection_order` is provided (`tensorcast/api/store/materialization.py`).
  - `DeferredLoader.plan(...)` currently sorts the planned names, which conflicts with “caller-controlled order” semantics for packed layouts (`tensorcast/api/store/deferred_loader.py`).
  - Daemon `MaterializeIntoTarget` currently sorts names and feeds the sorted list into view planning for subset layouts, which loses the caller’s packed-stream order (`daemon/service/controllers/materialization_controller.cc`).
  - Daemon rejects `index_kind=VIEW` for “packed reorder full coverage” (full canonical set + no view transform), which is required for local packed reorder layouts in the design (`daemon/service/controllers/materialization_controller.cc`).
  - `compute_view_subset_hash` empty-set semantics diverge across languages today (Python hashes `[]`; C++ returns empty bytes), which will make “full coverage” encoding and hash stability fragile unless normalized explicitly (`tensorcast/common/selection_identity.py`, `core/common/selection_identity.cc`).
- Capability tokens already exist and should be reused for `target_write_token` rather than inventing a parallel token system:
  - Proto: `proto/tensorcast/common/v1/capability_token.proto`
  - C++: `core/common/capability_token.{h,cc}`

# Phases & Milestones

- [x] Phase 0: Contract + proto scaffolding (additive-only)
  - [x] Milestone 0.1: Selection encoding contract alignment
    - [x] Make “full coverage” vs “subset-packed” encoding decidable across SDK/daemon (proto3 empty-bytes normalization for `view_subset_hash`).
    - [x] Ensure `selection_hash` is computed with identical semantics in Python and C++ for empty `view_subset_hash`.
    - [x] Normalize `view_subset_hash=b""` as “unset” before hashing:
      - [x] Python: `tensorcast/common/selection_identity.py`
      - [x] C++: `core/common/selection_identity.{h,cc}`
    - [x] Define and enforce “ordered tensor_names vs membership view_subset_hash” semantics:
      - [x] `tensor_names` order is the packed byte-stream order and MUST NOT be sorted away by the daemon.
      - [x] `view_subset_hash` is an order-independent membership hash over **sorted unique** names; the empty set MUST be encoded as `b""` (not a digest of `[]`).
      - [x] Allow `index_kind=VIEW` when `tensor_names` is non-empty even if the selection covers the full canonical set (packed reorder local-only).
  - [x] Milestone 0.2: Global Store retire/drain RPCs (required for swap safety)
    - [x] Add RPC(s) to mark a single replica unavailable (by `replica_id`) and to wait for drain (`current_requests==0`).
    - [x] Implement the server-side handlers (bounded waits + drain diagnostics).
    - [x] Add/extend `tests/python/global_store/test_grpc_service.py` coverage for:
      - [x] idempotent “mark unavailable”
      - [x] drain timeout returns actionable snapshots (`current_requests`, optional oldest transport age)
      - [x] `WaitReplicaDrain` is observation-only (must not mutate transports / force-decrement counters).
    - [x] Run codegen: `bash tools/build_proto_python.sh`.
  - [x] Milestone 0.3: Daemon proto scaffolding for slot retire/publish
    - [x] Extend `DeregisterArtifactRequest` with `tensorcast.common.v1.ByteSpaceRef byte_space` (compat wrapper; default canonical when absent).
    - [x] Extend `MaterializeIntoTargetResponse` with `bytes target_write_token` (optional).
    - [x] Add `operation_id` (optional) to swap-related RPC requests (materialize/publish/retire and GS drain RPCs).
    - [x] Add RPC `PublishTargetReplica(...)` and `RetirePublishedReplica(...)` (stub-first; return `UNIMPLEMENTED` until implemented).
    - [x] Extend `proto/tensorcast/common/v1/capability_token.proto` with a target-write audience + scope message.

- [x] Phase 1: SDK local slot object (no publish required)
  - [x] Milestone 1.1: `InplaceSlot` as the commit result (pointer stability baseline)
    - [x] `DeferredLoader.commit()` returns `InplaceSlot` and exposes `InplaceSlot.commit_result: DeferredCommitResult`.
    - [x] Add `packing="byte_space"` with canonical/view full-coverage encodings that are publishable.
    - [x] Fix `packing="plan"` to preserve caller-provided order (no implicit sorting).
  - [x] Milestone 1.2: Local-only swap surface (publish gated)
    - [x] Implement `InplaceSlot.swap(..., publish=False)` via `MaterializeIntoTarget` without changing tensor pointers.
  - [x] Milestone 1.3 (optional): `tc.artifact(ref)` shorthand (non-ambiguous)

- [x] Phase 2: Daemon retire-before-overwrite (GS drain as correctness barrier)
  - [x] Milestone 2.1: ByteSpace-aware targeting and cleanup by replica_id
    - [x] Retire targets `(artifact_id, view_id, device_id)` without canonical/view aliasing.
    - [x] Cleanup uses per-lease `replica_id` instead of worker-wide unregister-by-(artifact,device).
    - [x] `DeregisterArtifact` honors `byte_space` and routes into the same ByteSpace-aware cleanup path (default canonical).
  - [x] Milestone 2.2: `RetirePublishedReplica` blocks overwrite until safe
    - [x] Implement retire ordering: mark GS `is_available=false` → wait GS drain → unexport communicator keys → unregister GS replica.
    - [x] Optional best-effort daemon-local gates: drain staged LIP exports; ensure ref/use counts are clear.
  - [x] Milestone 2.3: SDK integration point
    - [x] `InplaceSlot.swap(publish=True)` (Phase 4) calls `RetirePublishedReplica(wait=True)` before overwrite when currently published.

- [x] Phase 3: Daemon publish-target replica (no GPU re-hash)
  - [x] Milestone 3.1: `target_write_token` mint + validation
  - [x] Milestone 3.2: Publish registers communicator keys + GS memory replica
  - [x] Milestone 3.3: Keepalive refreshes region TTLs

- [x] Phase 4: SDK publish/swap integration + validation
  - [x] Milestone 4.1: Slot state machine matches the design contract
  - [x] Milestone 4.2: Tests + docs updates

# Implementation Tasks

## Phase 0: Contract + proto scaffolding

- Selection encoding contract (cross-layer):
  - Align proto3 empty-bytes semantics for `view_subset_hash` with `selection_hash` computation (treat `b""` as unset).
  - Add/update docs as needed to make “full coverage vs subset-packed” encoding unambiguous for publishability.
  - Fix ordering semantics end-to-end:
    - `tensor_names` order MUST be preserved by the daemon and used for view planning / view index generation (subset-packed and packed reorder).
    - `view_subset_hash` MUST be computed over the set (sorted unique names) and MUST be empty for full coverage and packed reorder.
- Update Global Store proto (`proto/tensorcast/global_store/v1/global_store.proto`):
  - Add `MarkReplicaUnavailable(...)` to set `is_available=false` for a specific `replica_id` (idempotent).
  - Add `WaitReplicaDrain(...)` to wait for `current_requests==0` for a `replica_id` (bounded wait; returns snapshot).
    - Include enough drain diagnostics to debug timeouts (at least `current_requests`, ideally oldest in-progress transport age).
  - Both RPCs SHOULD accept an optional `operation_id` for correlation.
  - No schema changes expected: use existing `artifact_replicas.is_available` and `replica_counters.current_requests`.
- Update daemon proto (`proto/tensorcast/daemon/v2/store_daemon.proto`):
  - Extend `DeregisterArtifactRequest` with `tensorcast.common.v1.ByteSpaceRef byte_space` (compat wrapper; default canonical when omitted).
  - Extend `MaterializeIntoTargetResponse` with `bytes target_write_token` (optional).
  - Add optional `operation_id` to swap-related RPC requests (`MaterializeIntoTarget`, `PublishTargetReplica`, `RetirePublishedReplica`, and retire compat wrappers).
  - Add RPC `PublishTargetReplica(PublishTargetReplicaRequest) -> PublishTargetReplicaResponse`.
  - Add RPC `RetirePublishedReplica(RetirePublishedReplicaRequest) -> RetirePublishedReplicaResponse`.
- Update capability token proto (`proto/tensorcast/common/v1/capability_token.proto`):
  - Add a new `CapabilityAudience` for target write tokens and a scope message binding the written `ArtifactSelection`.
- Run proto generation: `bash tools/build_proto_python.sh`.
- Add minimal SDK plumbing for new RPCs (stubs only) so Python can compile/typecheck before full semantics land.

## Phase 1: SDK local slot object (no publish required)

- Introduce `InplaceSlot` (new module, e.g. `tensorcast/api/store/inplace_slot.py`):
  - Owns pointer-stable CUDA tensor views + the immutable template metadata needed for subsequent fills.
  - Carries the authoritative `ArtifactSelection` + derived `ByteSpaceRef`, using the selection encoding contract
    (including proto3 empty-bytes normalization for `view_subset_hash` when computing `selection_hash`).
  - (Optional) Records `layout_spec_id` when the template is derived from Global Store layout v2, and uses it in
    preflight validation for better debuggability.
  - Exposes `publish_replica()` (UNIMPLEMENTED until Phase 3 lands), `swap(...)`, `retire(...)`, `close()`.
- Update `DeferredLoader`:
  - `commit()` returns `InplaceSlot` instead of `DeferredCommitResult`.
  - Preserve observability/back-compat via `InplaceSlot.commit_result: DeferredCommitResult`.
  - Implement `packing="byte_space"`:
    - Place tensors at logical offsets of the selected index stream (canonical or view), without treating call order as a
      packed reorder.
    - Ensure canonical/view full-coverage encodings produce publishable selections (empty `tensor_names`, empty
      `view_subset_hash`).
    - Ensure the SDK does not accidentally force `index_kind=VIEW` for full-coverage byte-space layouts (avoid passing a
      `selection_order` / `tensor_names` list when full coverage is intended).
  - Keep `packing="append"` / `packing="plan"` as packed local layouts (not publishable in Phase 1).
    - Fix `plan(tensor_names=[...])` to preserve caller-provided order (do not sort).
  - Keep `publish=True` on `commit()` as “register new artifact id”; reserve `InplaceSlot.publish_replica()` for “publish
    as replica”.
- Update tests + docs:
  - Update `tests/python/test_deferred_loader.py` for the new return type and ordering semantics.
  - Update `tensorcast/api/store/README.md` to document slot vs register vs publish.
- (Optional) Artifact ref shorthand:
  - Add positional `ref` to `tensorcast.api.store:artifact` / `artifact_async` with reserved prefixes (`mi2:`/`cgid:`/`disk:`)
    and explicit ambiguity rejection.

## Phase 2: Daemon retire-before-overwrite (GS drain)

- Make retire targeting ByteSpace-aware:
  - Map `ByteSpaceRef(kind=CANONICAL)` → `view_id=""`, `ByteSpaceRef(kind=VIEW, id=view_id)` → `view_id=view_id`.
  - Reject `BYTE_SPACE_KIND_UNSPECIFIED` and `VIEW` with empty id (`INVALID_ARGUMENT`).
- Update `DeregisterArtifact` to accept and enforce `byte_space`:
  - Default missing `byte_space` to canonical for backward compatibility.
  - Ensure any quiesce/drain/cleanup is keyed by `(artifact_id, view_id, device_id)` so canonical and view do not alias.
  - Update daemon-local quiesce/drain bookkeeping to be ByteSpace-aware (avoid artifact-wide quiesce for view leases):
    - Replace/extend `LipManager::quiesce_artifact(...)` + `wait_exports_drained(...)` to key by `(artifact_id, view_id, device_id)`.
- Implement `RetirePublishedReplica` with GS drain as the correctness barrier:
  - Target by `lease_id` (preferred) or by `(artifact_id, byte_space, device_id, owner_pid)` (fallback).
  - Required semantics:
    1) Quiesce: stop new exports (ByteSpace-aware) and remove the replica from the daemon’s publishable inventory immediately.
    2) Stop new transports: call Global Store `MarkReplicaUnavailable(replica_id)` for the specific published replica.
    3) Drain: call Global Store `WaitReplicaDrain(replica_id, deadline)` until `current_requests==0`.
    4) Best-effort local drains: ensure any staged LIP exports are drained for the same key.
    5) Unexport: unregister communicator tensor keys / remote keys and release any held region refs.
    6) Unregister: remove the Global Store replica (by `replica_id`), using per-lease tracking (`LipManager::attach_replica_id(...)`).
  - On drain timeout: return `DEADLINE_EXCEEDED` and do not proceed to overwrite.

## Phase 3: Daemon publish-target replica (no GPU re-hash)

- Mint `target_write_token`:
  - After successful `MaterializeIntoTarget`, mint a capability token and return it in `MaterializeIntoTargetResponse`.
  - Ensure the token binds to the actual `ArtifactSelection` realized by the daemon (`artifact_id`, `view_id`, `view_subset_hash`, `logical_layout_hash`, `selection_hash`) and the resulting `ByteSpaceRef`.
- Implement `PublishTargetReplica`:
  - Validate `target_write_token` ownership (pid/device), expiry, and identity fields.
  - Enforce Phase-1 publishability: only canonical/view full-coverage ByteSpaces (reject subset-packed / packed reorder with `FAILED_PRECONDITION`).
  - Token retry semantics:
    - `PublishTargetReplica` SHOULD be idempotent for the same `(target_write_token, operation_id)` and MAY allow retrying publish within token TTL (to tolerate GS/transient failures).
    - Reject stale tokens after a newer materialization for the same target regions (ABA protection).
  - Register communicator remote keys for the target layout (client-owned VRAM regions), producing `remote_memory_keys` + `buffer_sizes`.
  - Publish to Global Store via existing client methods (memory replica registration) without re-hashing GPU bytes.
  - Return `lease_id` (daemon keepalive/revoke handle) and `replica_id` (GS identifier).
  - Failure modes:
    - GS unavailable → return a clear `FAILED_PRECONDITION` (or accept "local-only published" as a follow-on; do not invent ad-hoc env flags).
    - Comm engine disabled → `FAILED_PRECONDITION`.
- Keepalive integration:
  - Ensure the published lease keepalive path refreshes:
    - lease TTL
    - VRAM region TTLs referenced by the lease (so regions do not expire while replica is routable)

## Phase 4: SDK swap/publish integration + validation

- SDK `InplaceSlot.swap(...)`:
  - Preflight: resolve selection + validate daemon capability and layout compatibility before retiring any published replica.
  - If currently `Published`, call `RetirePublishedReplica(... wait=True)` (or `DeregisterArtifact` with ByteSpaceRef if the surface is unified).
  - Call `MaterializeIntoTarget` into the existing arena layout.
  - If `publish=True`, call `PublishTargetReplica(target_write_token, byte_space)`.
  - Update local state (`artifact_id`, `byte_space`, `published_lease_id`).
  - On failures after overwrite begins:
    - materialization failure → slot enters `Dirty` (bytes undefined)
    - publish failure → slot remains `FilledLocal` (bytes defined) and can retry `publish_replica()` later
- Python tests:
  - Update `tests/python/test_deferred_loader.py` to assert `commit()` returns `InplaceSlot` and `commit_result` preserves prior fields.
  - Add `tests/python/test_inplace_slot.py`:
    - pointer stability: tensor `data_ptr()` remains unchanged across swap (mock daemon client responses)
    - publish failure semantics: publish error leaves slot `FilledLocal` (bytes defined) and allows retry via `publish_replica()`
    - ref parsing: `tc.artifact("llama")` behavior + rejection of ambiguous forms
- C++ tests (Bazel):
  - Add daemon tests validating retire safety:
    - retire marks the GS replica unavailable (`MarkReplicaUnavailable`) before attempting drain
    - retire waits for GS drain (`WaitReplicaDrain`) and returns `DEADLINE_EXCEEDED` if it cannot reach `current_requests==0` in time
    - retire unregisters by `replica_id` (does not nuke unrelated replicas on the same worker/device)
    - (optional hardening) retire waits for any staged LIP exports to drain (legacy `LockTransportChunks` path), but does not treat daemon-local locks as the authoritative P2P signal
  - Add daemon tests validating publish token semantics:
    - token required and owner-checked
    - token pins ByteSpace/layout identity
    - (recommended) stale/replayed token rejected (single-use or newer-materialization invalidates old token)
  - Run with fake backend where applicable: `bazel test //daemon:... --test_env=TENSORCAST_CUDA_BACKEND=fake`.
- Docs:
  - Update `tensorcast/api/store/README.md` and (if needed) `docs/architecture/api/region-backed.md` / `docs/internals/tensor_dict_into_dataflow.md` to describe slot fill vs register vs publish.

# Test / Rollout / Backout

- Tests (Python):
  - `TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python/test_deferred_loader.py`
  - `TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python/test_inplace_slot.py`
  - `uv run pytest tests/python/global_store/test_grpc_service.py`
- Tests (C++):
  - `bazel test //daemon:... --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - `bazel test //proto/... --test_output=streamed`
- Lint/type checks (Python):
  - `uv run ruff check .`
  - `uv run ruff format .`
  - `uv run mypy ./tensorcast`
- Rollout:
  - Land Phase 1 SDK changes first with `publish_replica()` returning `UNIMPLEMENTED` until Phase 3 ships.
  - Keep legacy `DeferredCommitResult` available via `InplaceSlot.commit_result` to reduce downstream churn.
- Backout:
  - SDK-only rollback: keep `DeferredLoader.commit()` returning `DeferredCommitResult` and expose `commit_slot()` behind a minor version bump (if API break is unacceptable).
  - Daemon rollback: leave new RPCs additive; callers can gate by feature detection (UNIMPLEMENTED) and fall back to local-only slots.

# Acceptance Criteria (plan-level)

- A vLLM-style "meta-init + per-parameter binding + one commit" workflow works via `InplaceSlot`.
- `swap()` keeps tensor `data_ptr()` stable and enforces `preflight -> retire -> overwrite -> (optional) publish`.
- When swapping a published slot, overwrite does not begin until GS reports the old replica is unavailable and drained
  (`is_available=false` + `current_requests==0`), or `swap()` fails with `DEADLINE_EXCEEDED`.
- `swap(publish=True)` publishes the new bytes as a routable memory replica (no GPU re-hash), and the new replica becomes eligible for P2P routing.
- `publish_replica()` fails fast (`FAILED_PRECONDITION`) for non-publishable packed selections (subset-packed or reordered).
- Published client-owned leases do not expire due to region TTLs (keepalive refreshes region TTLs referenced by the lease).
- `tc.artifact("some-key")` behaves as `tc.artifact(key="some-key")`, and ambiguous `ref` forms are rejected.
- Packed-layout ordering is preserved end-to-end: `tensor_names` order used by SDK is the same order used by daemon planning and recorded in `ArtifactSelection`.

# Risks & Tracking

- Non-atomic swap: once overwrite begins, bytes cannot be rolled back; callers must serialize compute vs swap.
- Drain correctness: swap safety depends on GS `is_available/current_requests` being authoritative; leaked or stalled transports can block drain until cleanup. Mitigate with `MarkReplicaUnavailable` + `WaitReplicaDrain`, bounded deadlines, and actionable drain diagnostics (plus best-effort local staged-export drains).
- Publishability correctness: publishing a packed subset layout under a ByteSpaceRef would corrupt remote readers; mitigate by enforcing full-coverage + routable ByteSpace eligibility checks in both daemon and SDK before publish.
- ByteSpace ambiguity: mixing view/canonical without explicit selectors can retire the wrong replica; mitigate by requiring `ByteSpaceRef` on retire/publish and rejecting unspecified kinds.
- Region TTL expiry: published client-owned replicas must refresh region TTLs; missing refresh causes hard-to-debug P2P failures.
- API compatibility: `DeferredLoader.commit()` return-type change can break callers; mitigate via `InplaceSlot.commit_result` and clear docs.
