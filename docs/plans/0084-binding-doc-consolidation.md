---
slug: binding-owner-path-and-mapped-bind
title: Owner-Path Bind and Mapping Unification
links:
  design: ../designs/0084-binding-unified-model-and-contract.md
areas: ["sdk", "daemon", "core", "proto"]
related_code:
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/inplace_slot.py
  - tensorcast/api/store/mapped_binding.py
  - tensorcast/api/_materialize.py
  - tensorcast/daemon_ctl.py
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/service/controllers/target_materialization_service.cc
  - daemon/service/controllers/materialization_target_storage_utils.cc
  - daemon/service/controllers/materialization_replica_handle_utils.cc
  - daemon/state/handle_lease_registry.h
  - daemon/state/local_handle_server.cc
  - core/store/store_engine.h
---

# Objective

Make `Artifact.bind()` a real owner path instead of a thin wrapper over
`bind_into()`, while adding first-class `mapping` support and preserving one
public `Binding` concept.

The intended end state is:

- `bind()` allocates and owns the target location through the daemon
- `bind_into()` adopts caller-provided target tensors
- `bind(mapping=...)` keeps `bind()` ownership semantics and does not silently
  downgrade to `bind_into(...)`
- mapped and unmapped binding share the same public lifecycle, but may use
  different internal daemon pipelines
- `swap()` never asks the caller to restate `mapping`; it reuses the plan
  captured at bind time
- `swap(full_artifact_id)` replays the stored subset/view/mapping contract to
  find the exact rank-local partial bytes inside the full artifact

# Current State & Grounding

- `Artifact.bind()` currently allocates local `torch.empty_strided(...)`
  tensors in the SDK and then delegates directly to `bind_into(...)`.
  Current code: `tensorcast/api/store/artifact.py`.
- `Artifact.bind_into()` and `_bind_into_mapped()` only support
  caller-provided, user-owned CUDA memory. They register VRAM regions and write
  through `MaterializeIntoTarget` or `MaterializeIntoMappedTarget`.
  Current code: `tensorcast/api/store/artifact.py`.
- The daemon target-write RPCs reject any storage source other than
  `vram_region_id`, and `TargetStorageLease` only knows how to acquire region
  registrations. Current code:
  `daemon/service/controllers/target_materialization_service.cc`,
  `daemon/service/controllers/materialization_target_storage_utils.cc`.
- Reusing daemon-owned allocations as fake region-backed targets is not a sound
  shortcut. The daemon target-write path re-opens CUDA IPC handles in the daemon
  process, while real CUDA rejects same-process open of a handle exported by that
  same process. Current code:
  `core/cuda/cuda_backend_real.cc`,
  `daemon/service/controllers/materialization_target_storage_utils.cc`.
- The retrieval path already proves that TensorCast can allocate daemon-owned GPU
  memory, export a CUDA IPC handle, and reconstruct Python tensors on the SDK
  side. Current code:
  `proto/tensorcast/daemon/v2/store_daemon.proto`,
  `tensorcast/api/_materialize.py`.
- `Binding` and `InplaceSlot` still encode client-owned assumptions in both
  wording and implementation. Current code:
  `tensorcast/api/store/binding.py`,
  `tensorcast/api/store/inplace_slot.py`.
- The local handle plane and GPU handle lease machinery already solve the
  lifetime problem for daemon-exported memory. That machinery should be reused or
  generalized instead of inventing ad-hoc tensor lifetime tracking. Current code:
  `daemon/service/controllers/materialization_replica_handle_utils.cc`,
  `daemon/state/handle_lease_registry.h`,
  `daemon/state/local_handle_server.cc`.

Conclusion from current state:

- `bind()` cannot become a true owner path by layering more logic onto
  `bind_into()`.
- A daemon-owned binding registry and a dedicated owner-path RPC flow are
  required.
- The core copy engine can still be reused once the daemon has an in-process
  pointer to the owned allocation.

# Chosen Implementation Shape

## 1. Owner Path Uses Daemon-Owned Coalesced GPU Storage

The MVP owner-path representation for `bind()` is one daemon-owned coalesced GPU
allocation per binding.

Reasons:

- the current SDK already knows how to rebuild multiple tensor views from one
  CUDA IPC handle plus tensor descriptors
- one-handle export avoids designing a repeated-handle response surface in the
  first iteration
- arbitrary tensor shapes, strides, and logical offsets can still be represented
  with descriptors over one backing allocation
- the daemon can build a normal `IntoTargetLayout` over one in-process storage
  and reuse existing `StoreEngine` target-write routines

This deliberately does not promise that `bind()` preserves today's incidental
"one tensor, one CUDA allocation" implementation detail.

## 2. Add A Daemon `BindingRegistry`

Introduce a daemon-owned registry for bind-created locations.

Each `BindingRecord` stores at least:

- `binding_id`
- `owner_pid`
- `device_id` / `device_uuid`
- the coalesced GPU allocation and exported CUDA IPC handle
- tensor descriptors used to reconstruct SDK tensors
- persisted target layout metadata and `logical_layout_hash`
- captured `ArtifactSelection` identity
- captured selection replay envelope:
  subset membership/order, resolved `view_id` or `view_spec`, and any selection
  fields needed to deterministically rebuild the partial source selection on
  refill
- optional persisted `CopyPlan`
- latest `target_publication_token`
- lifecycle state such as `closed`, published routing state, and cleanup guards

The registry is daemon-local only. The SDK never talks to Global Store directly.

## 3. Add Dedicated Owner-Path Binding RPCs

Add a daemon RPC family for owner-path binding. The exact message names can be
finalized during implementation, but the protocol needs three operations:

- create owner binding:
  allocate daemon-owned storage, materialize initial bytes, export tensors, and
  return `binding_id`
- refill owner binding:
  swap a new artifact into an existing `binding_id` while preserving storage
  pointers seen by the SDK
- close owner binding:
  release the swap/publish control object and schedule final allocation cleanup

These RPCs must remain daemon-only control-plane APIs. No SDK direct-to-GS path
is allowed.

## 4. Reuse The Existing Core Target-Write Engine

The owner path should not introduce a second copy engine in `core/`.

Instead:

- the daemon owner-path controller constructs an in-process
  `loading::IntoTargetLayout` from the stored allocation base pointer
- unmapped bind uses `StoreEngine::materialize_into_target(...)`
- mapped bind uses `StoreEngine::materialize_mapped_into_target(...)`

This preserves one copy-plan model and one target-write dataplane in core, while
still allowing separate daemon control paths for owner vs adopt semantics.

## 5. Split SDK Binding Backends Under One Public `Binding`

Keep `Binding` public and stable, but stop assuming one backend.

Refactor toward:

- `InplaceSlot` (or renamed equivalent) for `bind_into()` adopt-path bindings
- `OwnedBindingSlot` for daemon-owned `bind()` bindings
- `Binding` as a thin wrapper over a slot/backend protocol with common methods:
  `tensors`, `artifact_id`, `selection`, `swap`, `publish_replica`, `close`

`Artifact.bind()` must call the owner-path backend directly. It must no longer
delegate to `bind_into()`.

## 6. Lifecycle Model For Owner-Path Bindings

The owner path must manage two related but different lifetimes:

- control lifetime:
  whether the binding may still `swap`, `publish_replica`, or `close`
- export lifetime:
  whether the daemon-owned GPU allocation must stay alive because the SDK still
  holds mapped tensors or the target is still published

The key rule is:

- `Binding.close()` ends control lifetime first
- allocation reclamation happens only after export lifetime also reaches zero

Each owner-path `BindingRecord` therefore needs at least three live references:

- control ref:
  owned by the active `binding_id` control object
- export refs:
  owned by exported tensor-handle leases held by SDK tensors reconstructed from
  the daemon CUDA IPC export
- publish ref:
  owned by any active published replica lease that still routes traffic to this
  target

Suggested owner-path states:

- `Allocating`
  daemon has accepted create, allocated storage, and is filling initial bytes
- `ReadyLocal`
  tensors are exported to the SDK, `swap()` is allowed, target is local-only
- `Published`
  routable publish lease exists; refill must retire and drain before overwrite
- `Dirty`
  overwrite started and failed; storage pointers remain stable but bytes are not
  publishable until a later successful refill
- `Closing`
  control object is closed or PID-exit cleanup has started; no new user actions
  are accepted, but the allocation may still be pinned by export refs
- `ClosedControl`
  control lifetime is gone, publish lease is retired, waiting only for export
  refs to reach zero
- `Reclaimed`
  daemon allocation and registry record are fully removed

State transitions:

- create RPC:
  `Allocating -> ReadyLocal` on success, otherwise immediate cleanup
- publish:
  `ReadyLocal -> Published`
- refill/swap:
  `Published -> ReadyLocal` via retire/drain, then overwrite; successful publish
  may return to `Published`
- overwrite failure:
  `ReadyLocal|Published -> Dirty`
- later successful refill:
  `Dirty -> ReadyLocal` or `Dirty -> Published`
- close:
  `ReadyLocal|Published|Dirty -> Closing -> ClosedControl`
- final reclaim:
  `ClosedControl -> Reclaimed` when export refs and publish refs are both zero

Close and reclaim semantics:

- `Binding.close()` must send an owner-path close RPC
- the daemon marks the record closed and rejects future swap/publish calls for
  that `binding_id`
- if published, the daemon retires the published replica as part of close
- the daemon drops the control ref immediately
- the daemon must not free the GPU allocation yet if exported tensors are still
  alive in the SDK

Export ref handling:

- owner-path create returns the exported CUDA IPC handle plus a GPU handle lease
  token
- SDK tensor reconstruction binds that lease token to tensor lifetime, reusing
  the same local-handle release mechanism used by existing materialization
  exports
- when the last exported tensor is dropped, the lease token release removes one
  export ref from the `BindingRecord`

PID-exit handling:

- if the client process dies without calling `close()`, daemon session lifecycle
  cleanup drops the control ref and any outstanding handle leases
- published replicas are retired best-effort during PID-exit cleanup
- once publish refs and export refs both reach zero, the allocation is reclaimed

This split is mandatory. A single "close means free" model is incorrect for
daemon-owned bindings because Python may still hold live tensor views after the
binding control object is gone.

## 7. `bind(mapping=...)` Performs SDK Inference, Server Revalidation

For mapped `bind()`:

- the SDK infers the destination layout from canonical index metadata, source
  view metadata, and `CopyPlan`
- the daemon revalidates the declared target layout and copy plan before
  allocating or refilling

This keeps user-facing validation errors deterministic and early, while still
protecting daemon correctness.

## 8. No Implicit Fallback

The implementation must fail hard rather than silently changing modes:

- `bind(mapping=...)` must not downgrade to unmapped `bind()`
- `bind(...)` must not silently call `bind_into(...)`
- owner-path RPC absence must be a hard `FAILED_PRECONDITION`, not a hidden
  compatibility fallback
- inference ambiguity must be a hard error
- owner-path execution must not widen an explicit source policy; it may inherit
  the normal `FallbackOptions` contract, but it must not silently become more
  permissive than the request asked for

## 9. Tensor-Parallel Replay Contract

The plan must support the intended TP serving workflow:

1. each rank binds once with its rank-local subset/view/mapping contract
2. TensorCast owns the rank-local VRAM for `bind(...)`
3. the model process maps tensor views from that VRAM
4. later updates call `binding.swap(full_artifact_id)` only

To make that work, the implementation must ensure:

- the create RPC persists the full replay contract in `BindingRecord`
- the refill RPC takes `binding_id` plus a new artifact reference, not a new
  copy plan
- refill resolves the new full artifact and reconstructs the exact rank-local
  partial selection from the stored replay contract
- refill rejects the new artifact if the reconstructed selection no longer
  matches the bound target layout contract

This is a hard requirement, not an optimization.

# Rejected Shortcuts

- Reusing `bind_into(...)` as the implementation of `bind(...)`:
  rejected because it forces caller-owned target memory semantics into the owner
  constructor.
- Allocating daemon-owned memory and then pretending it is a client region:
  rejected because the current target-write path only accepts `vram_region_id`,
  and region acquisition re-opens exported CUDA IPC handles in the daemon.
- Using `MaterializeReplica` as the whole bind implementation:
  rejected because it allocates artifact/view byte-space replicas, not arbitrary
  target layouts derived from `bind(mapping=...)`.

# Phases & Milestones

- [ ] Phase 1: Define owner-path protocol and state model
  - [ ] Milestone 1: Add a daemon `BindingRegistry` and `BindingRecord`
    lifecycle model for create, refill, close, and PID-exit cleanup.
  - [ ] Milestone 2: Add proto messages and gRPC methods for owner-path binding
    create, refill, and close.
  - [ ] Milestone 3: Decide how GPU handle lease tokens attach to binding-owned
    exports, and document the final cleanup contract.
  - [ ] Milestone 4: Make the refill request shape artifact-only
    (`binding_id` + full artifact reference + options), with selection replay and
    mapping replay sourced entirely from the stored binding state.

- [ ] Phase 2: Implement daemon owner-path allocation and refill
  - [ ] Milestone 1: Extract or introduce a daemon helper for GPU allocation +
    CUDA IPC export, reusing registration/materialization internals where
    sensible instead of duplicating CUDA export code.
  - [ ] Milestone 2: Build the owner-path controller that materializes into the
    binding-owned allocation via `StoreEngine::materialize_into_target(...)`.
  - [ ] Milestone 3: Add the mapped variant that materializes via
    `StoreEngine::materialize_mapped_into_target(...)` using the persisted copy
    plan.
  - [ ] Milestone 4: Persist `target_publication_token`, selection identity, and
    target layout metadata after create/refill.
  - [ ] Milestone 5: Persist and replay the full rank-local selection envelope
    so `swap(full_artifact_id)` can resolve the correct partial source bytes.

- [ ] Phase 3: Refactor SDK binding backends
  - [ ] Milestone 1: Introduce a backend protocol so `Binding` can wrap both
    adopt-path and owner-path implementations.
  - [ ] Milestone 2: Keep `bind_into()` on the existing explicit-target path.
  - [ ] Milestone 3: Move `bind()` to the new owner-path daemon RPCs for both
    mapped and unmapped cases.
  - [ ] Milestone 4: Rebuild owner-path tensors in the SDK from the daemon
    export response using the existing tensor restoration helpers.

- [ ] Phase 4: Add mapped bind inference and hard-failure semantics
  - [ ] Milestone 1: Add a layout inference helper in
    `tensorcast/api/store/mapped_binding.py` that derives owner-path tensor
    specs from `CopyPlan`.
  - [ ] Milestone 2: Keep the accepted subset strict: contiguous, unique dtype,
    unique rank/shape profile, full destination coverage, one slice dimension.
  - [ ] Milestone 3: Update `capacity_bytes` checks to use inferred target bytes
    for `bind(mapping=...)`.
  - [ ] Milestone 4: Add explicit errors and tests for non-inferable mappings
    and forbidden constructor fallback.
  - [ ] Milestone 5: Prove that mapped owner-path swap reuses the captured
    `CopyPlan` and does not accept a new one at swap time.

- [ ] Phase 5: Publish, cleanup, and end-to-end verification
  - [ ] Milestone 1: Reuse existing publish/retire behavior with
    `target_publication_token` for owner-path bindings.
  - [ ] Milestone 2: Make `Binding.close()` safe for daemon-owned exports by
    separating control-record close from final memory release.
  - [ ] Milestone 3: Add PID-exit cleanup for leaked owner-path bindings.
  - [ ] Milestone 4: Update SDK docs and examples after code and tests land.

# Detailed Tasks

- Add a new daemon state component, likely `daemon/state/binding_registry.{h,cc}`,
  rather than overloading `IpcRegionRegistry`.
- Add a controller/service pair for owner-path binding RPCs instead of extending
  `MaterializeIntoTarget` to fake owner semantics.
- Store enough replay metadata on create that refill can rebuild
  `ArtifactSelection` from the new full artifact id without any caller-provided
  subset/view/mapping hints.
- Reuse `TargetLayout` + tensor descriptor concepts where possible, but do not
  require owner-path requests to provide `vram_region_id`.
- Standardize the owner-path storage shape on one coalesced allocation in the
  first iteration. If multi-storage owner bindings are needed later, treat that
  as a follow-on extension rather than blocking the owner-path MVP.
- Factor a shared SDK helper for "response with CUDA IPC handle + tensor
  descriptors -> `dict[str, torch.Tensor]`" so owner-path bind does not duplicate
  retrieval reconstruction logic.
- Introduce an owner-path slot/backend, likely in a new file such as
  `tensorcast/api/store/owned_binding_slot.py`, instead of overloading
  `InplaceSlot` with conditionals for mutually exclusive ownership models.
- Keep `bind_into()` validation unchanged for user-owned memory. The owner path
  should not weaken or repurpose that contract.
- Make source policy explicit on new owner-path RPC calls; the path may inherit
  existing `FallbackOptions`, but it must never switch constructors or silently
  retry via another mode.
- Update `tensorcast/api/store/README.md` only after implementation and tests
  settle on the final owner-path lifecycle wording.

# Test / Rollout / Backout

Acceptance checks after implementation:

- `source .venv/bin/activate && pytest tests/python/test_binding.py`
- `source .venv/bin/activate && pytest tests/python/test_inplace_slot.py`
- `source .venv/bin/activate && pytest tests/python/api/test_mapped_binding.py`
- `source .venv/bin/activate && pytest tests/python/daemon/test_inplace_slot_swap_publish_e2e.py`
- `source .venv/bin/activate && pytest tests/python/tools/test_weight_publisher_e2e_tp_bind_retry.py`
- `source .venv/bin/activate && ruff check tensorcast/api/store tests/python`
- `bazel test //daemon:materialize_into_target_validation_test`
- `bazel test //daemon:materialize_into_mapped_target_test`
- `bazel test //daemon:grpc_service_impl_publish_target_replica_test`
- `bazel test //daemon:local_handle_lease_ttl_expiry_test`

New tests to add:

- Python unit tests for `bind(mapping=...)` success on inferable layouts
- Python unit tests that `bind(mapping=...)` hard-fails on ambiguous layouts
- Python unit tests that `bind()` no longer delegates to `bind_into()`
- Python unit tests that `swap()` on a mapped binding does not accept or require
  a copy plan argument
- Python and daemon tests that `swap(full_artifact_id)` replays stored
  subset/view/mapping and lands on the exact same rank-local partial layout
- Python or daemon integration tests for owner-path `Binding.close()` while
  tensors are still alive
- Bazel daemon tests for owner-path create/refill/close RPC behavior and PID-exit
  cleanup

Rollout:

- Land proto + daemon owner-path state model first.
- Add SDK backend split second, keeping `bind_into()` unchanged.
- Flip `Artifact.bind()` over to the owner path only after owner-path tests pass.
- Keep publish/retire behavior unchanged at the public API level during rollout.
- Do not ship an intermediate version where `bind(mapping=...)` silently falls
  back to another constructor path.

Backout:

- Revert the SDK call-site change so `Artifact.bind()` returns to the current
  local-allocation behavior.
- Leave `bind_into(..., mapping=...)` intact as the only mapped binding entry
  point if the owner path proves unstable.
- Back out daemon owner-path RPCs and registry together; do not keep a partial
  dead path that the SDK no longer calls.

# Risks & Tracking

- Risk: owner-path bind leaks daemon GPU memory when `Binding.close()` races
  with outstanding local tensor references.
  Mitigation: reuse or generalize handle lease tokens and local-handle release;
  do not free exported memory solely on `Binding.close()`.

- Risk: coalesced owner-path storage changes observable tensor-storage behavior
  relative to today's incidental per-tensor allocations.
  Mitigation: add compatibility tests for tensor values, shape/stride,
  `data_ptr()` stability across swap, and any assumptions the SDK currently
  exposes.

- Risk: mapped layout inference accepts an under-specified plan.
  Mitigation: keep the accepted subset strict and add explicit rejection tests.

- Risk: daemon and SDK validation drift for mapped owner-path bindings.
  Mitigation: share normalized layout and copy-plan invariants where possible and
  add parity tests that compare explicit-target vs owner-path outcomes for the
  same inferable plan.

- Risk: refill accidentally depends on caller-supplied partial selectors instead
  of stored binding state.
  Mitigation: make the refill RPC artifact-only and add tests that only pass a
  full artifact id.

- Risk: a tempting shortcut reintroduces constructor fallback.
  Mitigation: add explicit tests that assert `bind()` never routes through
  `bind_into()` and that owner-path RPC absence is terminal.

Owner checklist:

- [ ] `bind()` is implemented as an independent owner path.
- [ ] `bind_into()` remains the explicit-target primitive.
- [ ] `bind(mapping=...)` preserves owner semantics.
- [ ] `swap()` reuses the captured `CopyPlan` and never asks the caller for one.
- [ ] `swap(full_artifact_id)` is sufficient to recover the bound partial source
  contract.
- [ ] No implicit constructor fallback exists.
- [ ] No SDK direct-to-GS path is introduced.
- [ ] Publish and retire semantics remain daemon-mediated.
