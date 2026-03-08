---
slug: 0063-binding-first-inplace-updates
title: Binding-First Inplace Updates and Swap (Hide Slots)
areas: ["sdk", "daemon", "global_store", "proto"]
status: implemented
created: 2026-02-03
last_updated: 2026-02-03
related_code:
  - docs/designs/0039-artifact-first-sdk.md
  - docs/designs/0061-slot-based-inplace-binding-and-swap.md
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/deferred_loader.py
  - tensorcast/api/store/inplace_slot.py
  - tensorcast/api/store/materialization.py
  - tensorcast/api/store/runtime.py
  - tests/python/test_binding.py
  - tensorcast/global_store/grpc_service.py
  - proto/tensorcast/global_store/v1/global_store.proto
  - daemon/service/grpc_service_impl.cc
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - schema.sql
  - tensorcast/global_store/migrations/0018_key_mapping_generation.py
links:
  predecessors:
    - ./0039-artifact-first-sdk.md
    - ./0061-slot-based-inplace-binding-and-swap.md
  plan: ../plans/0063-binding-first-inplace-updates.md
---

# Summary

The current SDK supports vLLM-style inplace binding and safe swap via `DeferredLoader.commit() -> InplaceSlot`
(`docs/designs/0061-slot-based-inplace-binding-and-swap.md`). In practice, this exposes **too many concepts** (deferred
loader, commit, slot) and makes it hard to adopt swap in “I already have tensors” or “I just want a swap-able
TensorDict-like object” scenarios.

This design introduces a single user-facing abstraction: **`Binding`**.

- `Artifact` remains the immutable “value” handle (artifact-first, per `docs/designs/0039-artifact-first-sdk.md`).
- `Binding` is the mutable “variable / location” that owns (or adopts) a **stable target layout** and can be
  **refilled in place** while preserving tensor storage pointers.
- The existing `InplaceSlot` state machine remains, but becomes an **internal implementation detail** behind `Binding`
  (the concept does not need to be taught to end users).
- `Binding.swap(...)` optionally supports a **cluster activation** step (update a global key/alias to the new
  artifact), aligning swap with the intuitive lifecycle “old becomes inactive, new becomes active”.

This keeps the project consistent with existing invariants:

- Pointer stability and overwrite safety remain anchored in the existing slot/retire/publish flow (GS drain as the
  correctness barrier before overwrite).
- `artifact_id` remains an immutable content identity; “swap” is a mutation of a **binding**, not a mutation of an
  artifact.
- Views/slices remain purely local composition until used; a `Binding` created from `artifact.view(...)` implicitly
  reuses that selection on future swaps.

Long-term intent (全局长期视角):
- `Binding` is not only an API wrapper; it becomes the **lifecycle owner** for external-target materialization:
  region registration, retry/poison handling, and managing the published lease lifecycle (keepalive optional).
- All external-target writes (deferred commit, bind_into, region-backed into, binding swap) should converge on one
  shared internal implementation so correctness policies do not fork.

# Problem Statement (问题陈述)

We need a long-lived, pointer-stable “variable” abstraction that is easy to adopt in real model codebases:

- Today’s user-facing flow (`DeferredLoader` + `commit()` + `InplaceSlot`) is correct but **too ceremony-heavy** and
  forces users to learn internal orchestration concepts.
- We also need an ergonomic path for the “already have tensors” case (load into pre-allocated parameter buffers),
  while staying honest about the constraint that **only user-owned allocations** can be exported/registered.
- Long-running production usage surfaced two systemic gaps that must be addressed in the design (not only in code):
  - historically, published client-owned replicas were TTL-bound by default (e.g., 10 minutes) and could silently
    expire unless explicitly kept alive; this design makes `ttl_ms=0` the default for publish leases and relies on
    explicit retire + PID-exit cleanup as the primary lifecycle mechanism.
  - VRAM region IDs are TTL/poison bound capabilities; region-backed writes must self-heal instead of assuming
    registrations are permanent.

# Goals / Non-Goals

## Goals

- **Simplify the public surface**: most users only need `Binding` + `Artifact`.
- Enable **swap without deferred-load ceremony**: one call that returns a swap-able, mapping-like object.
- Support both:
  - “allocate + bind + fill” (vLLM/meta-init style) and
  - “adopt existing tensors + fill” (load into pre-allocated parameter buffers).
- Preserve the **safe swap contract** from 0061: `preflight -> retire old -> overwrite -> optional publish`.
- Provide a clear and consistent **lifecycle model**:
  - binding-local: old published replica becomes unavailable, then overwritten; new becomes current.
  - global: an optional “activation key” can be updated to point to the new artifact (with CAS support).
- Make published bindings **operationally safe by default**:
  - if `publish=True`, the binding publishes with `ttl_ms=0` by default (disable time-based expiry) and relies on
    explicit retire + PID-exit cleanup; if a non-zero TTL is used, keepalive is best-effort.
- Make bindings **robust to region TTL / region poisoning**:
  - swap/fill paths should self-heal by re-registering missing/expired regions and retrying once when safe.

## Non-Goals

- Make packed/subset layouts routable cluster-wide (still local-only until selection-aware routing exists).
- Provide thread-safe concurrent readers during overwrite (users must gate model execution around swap).
- Re-define artifact immutability: old `artifact_id`s are not “destroyed”; they can remain valid historical values.
- Require new ad-hoc environment variables or bypass the unified configuration system.

# Architecture & Interfaces

## (a) API Surface

### User-facing types

#### `Binding` (new)

`Binding` is a mapping-like container of tensors with an inplace-update primitive.

```python
binding = tc.artifact("llama-7b-v1").bind(device="cuda:0")
model = build_model_from_binding(binding)  # meta-init, or assign param.data
binding.swap("llama-7b-v2")
```

Proposed minimal surface:

- `Binding.tensors: Mapping[str, torch.Tensor]` (read-only view)
- `Binding.artifact_id: str` (identity of the currently loaded bytes; immutable “value id”, changes after swap)
- `Binding.selection: tensorcast.proto.common.v1.common_pb2.ArtifactSelection` (opaque identity; used for correctness/debugging)
- `Binding.swap(artifact_or_ref, *, publish=False, activate_key=None, expected_active_artifact_id=None, expected_active_generation=None, wait=True, drain_timeout_s=None, ctx=None) -> None`
- `Binding.close() -> None` (idempotent; best-effort retire/unregister)

Notes:
- `Binding` intentionally does **not** expose `InplaceSlot` in the public API or docs.
- We keep `publish=False` and `activate_key=None` as the default to preserve local-only ergonomics.
- We intentionally do not expose region IDs or lease IDs in the public surface; the binding manages them internally.
  For debugging, callers can use `binding.selection` and logs/metrics keyed by `selection_hash`.

### Construction APIs (artifact-first)

We add two methods to `Artifact` (available on both canonical and view artifacts):

- `Artifact.bind(device, *, packing="byte_space", capacity_bytes=None, publish=False, ctx=None) -> Binding`
  - Allocates a client-owned CUDA arena and returns placeholder tensors **already filled** from this artifact.
  - Equivalent to `artifact.deferred_loader(...).tensor(... for all).commit()` but hides `DeferredLoader/commit/slot`.

- `Artifact.bind_into(target_tensors, *, packing="byte_space", publish=False, ctx=None) -> Binding`
  - Adopts *user-owned* CUDA tensors (already allocated in the current process) as the target layout.
  - Performs a single `materialize_into_target_v2(...)` fill from this artifact into `target_tensors`.
  - Returns a `Binding` that can later `swap(...)` in place.

Ergonomics for views (the user’s requested shape):

```python
artifact_tp0 = tc.artifact("llama-7b-v1").view(slices=slices_tp0)
binding_tp0 = artifact_tp0.bind(device="cuda:0")     # captures slices_tp0 once
binding_tp0.swap("llama-7b-v2")                      # no need to restate slices
```

We explicitly do **not** add `Artifact.swap(...)` because swap needs a stable target location; `Artifact` is a value
handle, while `Binding` is the location handle.

### Packing modes (user-level explanation)

Packing remains the same underlying concept as 0061, but becomes an “advanced knob” on `bind/bind_into`:

| `packing` | What it means | Primary use | Publishable |
| --- | --- | --- | --- |
| `"byte_space"` | Place tensors at logical offsets of the selected ByteSpace stream (canonical or view). Call order does not define layout. | Weight swap where publish/routing may matter; best default | Yes (only when coverage is full) |
| `"append"` | Densely pack tensors in the order the user first requests them. | Quick meta-init without pre-planning | No (local-only) |
| `"plan"` | Densely pack tensors in a user-provided order (`plan(tensor_names=[...])`). | Stable packed reorder (local-only) | No (local-only) |

Default: `packing="byte_space"` for `Artifact.bind/bind_into`, because it yields the most predictable identity and the
only path to `publish=True`.

## (b) Core Idea (最新的核心思路)

### 1) Value vs variable: making lifecycle intuitive

We align the API with the lifecycle intuition “destruct old, construct new” while preserving content-addressed
invariants:

- **`Artifact`** is an immutable value: it names a byte stream (canonical or view) and can be materialized.
- **`Binding`** is a mutable variable/location: it owns (or adopts) a stable layout and can be updated in place.
- **`swap`** mutates the *binding’s contents* (bytes + associated selection metadata), not the artifact.

This distinction matters for correctness and for teaching:
users only need to learn “Artifact is data; Binding is what my model points at”.

### 2) Safe overwrite is a lifecycle operation, not a copy

Swap is not “load new weights” — it is a lifecycle transition:

1) If a published replica exists, retire it (mark unavailable and drain) so remote reads cannot overlap overwrite.
2) Overwrite bytes in-place to match the new artifact selection.
3) Optionally publish the new contents and (optionally) activate a global key/alias.

We preserve the 0061 failure semantics:

- Materialization failure after overwrite begins makes the binding **Dirty** (bytes undefined).
- Publish failure after overwrite leaves bytes correct but the binding remains local-only.

### 3) Views are first-class: selection is captured once

Creating a `Binding` from `artifact.view(...)` captures:

- the view’s ByteSpace identity (`CANONICAL` vs `VIEW(view_id)`)
- the selection fingerprint (`ArtifactSelection`, including `logical_layout_hash` and `selection_hash`)

Future swaps reuse the same selection automatically, enabling TP/PP sharding without repeating slice specs.

## (b2) System-level invariants

These constraints make `Binding` safe and maintainable at cluster scale.

### External-target writes must be unified (one policy surface)

All SDK flows that write into caller-owned CUDA memory must share one internal implementation and error policy:

- `DeferredLoader.commit()` (internal impl)
- `Artifact.bind_into(...)`
- `Binding.swap(...)`
- `MaterializationPipeline.get_into(...)` when region-backed is selected

Rationale:
- Today, region-backed `get_into` has dedicated region-cache invalidation on `DATA_LOSS/FAILED_PRECONDITION`, while
  `InplaceSlot.swap` does not; `Binding` must remove this split by construction.

### Region IDs are ephemeral; pointers are stable

- Tensor pointers/storages are the stable contract.
- VRAM region IDs are a *renewable capability* with TTL and poison semantics; the binding must treat them as ephemeral:
  - On region-related failures, the binding may re-register regions and retry once (when safe).
  - The binding must never assume a region registration lasts beyond TTL unless TTL is explicitly disabled
    (`ttl_ms=0`). When TTL is enabled, renew/keepalive is required to avoid spurious expiration.

### Publish is a “while-alive” capability (not durability)

- Publishing a client-owned replica is a capability bound to a live process (PID).
- Default intent for long-lived serving processes: disable time-based expiry (`ttl_ms=0`) and manage lifecycle explicitly:
  - Explicit retire is the primary mechanism for safe overwrite (drain before swap).
  - PID-exit cleanup is the crash-safety mechanism.
- When TTL is enabled (`ttl_ms>0`), keepalive is required to prevent silent expiry; TTL expiry is the safety net.
- Implementation note: today, `PublishTargetReplica` returns `lease_id == registration_id` (the write id), so keepalive
  can reuse `KeepAliveRegisterArtifact(registration_id=lease_id, ...)` without adding new RPCs.

### TTL semantics

This design touches multiple TTL-governed resources. The guiding rule is:

- **Short-lived capabilities (tokens, caches)** keep TTLs short and always enforced.
- **Long-lived resources (published leases, VRAM regions)** default to `ttl_ms=0` (disable time expiry) and rely on
  explicit lifecycle + PID-exit cleanup.

#### Summary table

| Surface | Where | Default | `ttl==0` meaning | After expiry |
| --- | --- | --- | --- | --- |
| `target_publication_token` | `MaterializeIntoTarget` → `target_publication_token` | 5 minutes | N/A | `PublishTargetReplica` fails (token no longer valid); caller must re-materialize to get a new token |
| VRAM region registration | `RegisterVramRegionRequest.ttl_ms` | `0` for Binding/Slot-managed regions | Disable time expiry (`expires_at` unset); cleanup via explicit unregister + PID-exit | Region is swept from daemon registry; later region-backed writes fail until re-registered (binding self-heal can retry once) |
| Published replica lease (LIP) | `PublishTargetReplicaRequest.ttl_ms` (optional) | unset → `0` | Disable time expiry (lease stays active while PID lives) | Daemon treats lease inactive; best-effort unregisters replica from Global Store and drops export mappings |
| Key mapping cache TTL | `ResolveKeyMappingResponse.cache_ttl_seconds` | server hint (e.g., 30s for IMMUTABLE; 0 for ALIAS) | Disable client caching for that key | SDK must re-resolve on each use (prevents stale activation for aliases) |

#### 1) Target publication token TTL (short-lived capability)

- **Default TTL**: 5 minutes (daemon policy).
- **Behavior**:
  - The token is a capability and is intentionally short-lived.
  - After expiry, `PublishTargetReplica` must fail because the daemon can no longer trust the write capability.
  - The caller should re-run `MaterializeIntoTarget` to obtain a fresh token.

#### 2) VRAM region TTL (`RegisterVramRegion`)

- **Default TTL in this design**: `ttl_ms=0` for regions managed by `Binding`/`InplaceSlot` (no time-based expiry).
- **`ttl_ms>0` behavior**:
  - Daemon returns `expires_at` and can sweep the region after it elapses.
  - If a region is still actively referenced at sweep time, the daemon may extend the effective expiry.
  - Callers that choose a non-zero TTL must keep the region alive (or be prepared to self-heal by re-registering).
- **`ttl_ms==0` behavior**:
  - Daemon does not time-sweep the region (the response omits `expires_at`).
  - Lifecycle is controlled by:
    - explicit `UnregisterVramRegion` (normal path), and
    - PID-exit cleanup (crash-safety path).
- **Guardrail**: `DaemonOptions.max_region_ttl` clamps `ttl_ms>0` (default 10 minutes) and does not apply when `ttl_ms==0`.

#### 3) Published replica TTL (`PublishTargetReplica` / LIP lease)

- **Default TTL in this design**: when `ttl_ms` is unset, it behaves as `ttl_ms=0` (disable time-based expiry).
- **`ttl_ms>0` behavior**:
  - The binding must keep the lease alive to prevent silent expiration.
  - Today this reuses `KeepAliveRegisterArtifact(registration_id=lease_id, ttl_ms=..., epoch=...)`.
  - If keepalive stops and TTL expires, the daemon will:
    - treat the lease as inactive for routing, and
    - best-effort unregister the replica from the Global Store and release export resources during sweep.
- **`ttl_ms==0` behavior**:
  - No time-based expiry; lifecycle is explicit `RetirePublishedReplica` (overwrite safety) + PID-exit cleanup (crash safety).

#### 4) Key mapping cache TTL (server-driven hint)

- `ResolveKeyMapping` returns `cache_ttl_seconds` as a **server hint** for SDK caching.
- The SDK should treat `cache_ttl_seconds==0` as **no caching** for that key (important for alias/activation keys).
- Rationale: keys that represent “active / latest” should be resolved fresh (or via a very short TTL) to avoid stale reads.

### Observability and idempotency (operation_id)

`Binding` operations are multi-RPC flows (retire → materialize → publish → optional activate). For debuggability and safe
retries, every logical operation MUST have a single `operation_id` and propagate it to all involved RPCs, consistent
with the `OperationId` guidance in 0061.

## (c) Implementation Approach (接口的具体实现方式)

This design is intentionally an integration layer over the already-implemented 0061 machinery.

### Binding internal shape

- New module: `tensorcast/api/store/binding.py`
- `Binding` wraps an internal `InplaceSlot` (or a thin adapter around it).
- `InplaceSlot` remains the “hard state machine” (dirty/published/retire/publish tokens).

`Binding` owns:
- `tensors` mapping (pointer-stable)
- current selection metadata (for debugging and publishability checks)
- region registrations and their lifetime (unregister on `close()`)
- optional published lease/replica identity (if publish is used)
- the published lease lifecycle (explicit retire on swap/close; optional keepalive only when TTL is enabled)
- the “region self-heal” logic for swap/fill paths (re-register + one retry when safe)

### Region registration strategy (长期: 支持子分配指针)

To make `bind_into` practical for real models, region registration must support pointers that are *not* allocation bases
(sub-allocations from caching allocators).

Requirement:
- Region registration should export CUDA IPC handles using “base pointer + base_offset” (via `cudaMemGetAddressRange`
  or equivalent). The daemon already supports `mapping_base_offset`; the SDK should leverage that instead of requiring
  `base_ptr` to be an allocation base.

This aligns `bind_into` with existing registration flows that already export handle-with-offset for storages.

### `Artifact.bind(...)` (allocate + fill)

Implementation sketch:

1) Create a `DeferredLoader` internally with `packing` (default `"byte_space"`).
2) Request tensors for the full selection automatically:
   - canonical: all `artifact.tensor_names`
   - view: all names in the view index (derived via view metadata or computed view index)
3) Call `.commit()` to perform a single `materialize_into_target_v2(...)`.
4) Wrap the resulting internal `InplaceSlot` as a `Binding`.
5) If `publish=True`, call the slot publish path (only allowed for publishable selections).

This provides the “I don’t want deferred load” user experience while still using client-owned VRAM for swap.

### `Artifact.bind_into(target_tensors, ...)` (adopt + fill)

Key constraints (important for correctness and honest ergonomics):

- **Only user-owned CUDA memory can be registered as a VRAM region**. CUDA IPC import pointers (typical for
  `artifact.tensor_dict(...)` results) generally cannot be re-exported via `cudaIpcGetMemHandle`, so they cannot be used
  as slot targets.
- Therefore `bind_into` supports tensors that were allocated in the current process (e.g., model parameter buffers or a
  `torch.empty` arena), not daemon-owned imports.

Implementation sketch:

1) Validate `target_tensors` are CUDA, contiguous, dtype/shape/stride match the selected index entries.
2) Register VRAM regions using allocation-base export (see “Region registration strategy” above), grouping tensors by
   allocation base so we do not create one region per tensor.
3) Build a region-backed `TargetLayout` via the existing `_build_region_backed_layout(...)` machinery.
4) Call `materialize_into_target_v2(...)` to fill bytes once.
5) Construct an internal slot state (same as `DeferredLoader.commit()` does) and wrap as `Binding`.

Failure/robustness requirements:
- If the fill fails with region poisoning or region missing/expired, invalidate the region cache entry and allow one
  re-register + retry when the binding still owns the underlying tensors (safe because pointers remain stable).

### Global activation (`activate_key=...`)

We need to represent “new artifact becomes effective globally” without breaking artifact immutability.
The correct abstraction is a **mutable key/alias** that points to a current artifact id.

Current state (important grounding):
- Global Store `UpsertKeyMapping` currently behaves as “insert-only” (conflict if the key already maps to a different
  `artifact_id`), even though the RPC name is *upsert*.
- The SDK caches key resolutions for 30s by default (`StoreRuntimeContext._configure_key_cache_ttl()`), which is unsafe
  for fast-moving “active” keys.

Proposed change:

1) Introduce a true **swap/update** operation for key mappings with compare-and-set (CAS).
2) Version key mappings (a monotonic `generation`) so clients can reason about staleness.
3) Update SDK caching to be **server-driven** for key mappings, so “activation keys” are safe by default.
   - Activation keys should resolve with `cache_ttl_seconds = 0` (do not cache).
   - Immutable version keys may still be cached for performance.

Concrete proto/API proposal (Global Store):

- Add:
  - `SwapKeyMapping(SwapKeyMappingRequest) returns (SwapKeyMappingResponse)`
    - fields: `key`, `new_artifact_id`, optional `expected_artifact_id`, optional `expected_generation`
    - response: `status`, `artifact_id`, `generation`
- Extend `ResolveKeyMappingResponse` with:
  - `generation` (monotonic version)
  - `cache_ttl_seconds` (server-controlled hint; activation keys must set this to `0`)

CAS semantics (normative):
- No-op update: if `new_artifact_id == current_artifact_id`, return success and do not bump `generation`.
- Conflict: if `expected_*` does not match, return a conflict status and include the current `(artifact_id, generation)`
  in the response for diagnostics and safe retries.
- If both `expected_artifact_id` and `expected_generation` are provided, both must match.

Concrete storage proposal (`schema.sql`):

- Add `generation BIGINT NOT NULL DEFAULT 0` to `key_mappings`.
- (Recommended) Add a key mutability marker so the server can safely drive caching policy:
  - `kind TEXT NOT NULL DEFAULT 'IMMUTABLE'` (values: `IMMUTABLE`, `ALIAS`)
  - `SwapKeyMapping` may require `kind='ALIAS'` (or implicitly create an `ALIAS` key when missing).
- On successful swap/update, increment `generation` and update `updated_at`.

SDK behavior:

- `Binding.swap(..., activate_key=...)` performs activation *after* a successful local overwrite, and only if `publish`
  succeeded when publish is requested.
- On activation success, the SDK invalidates the local runtime key cache entry for `activate_key`.
- For correctness across all readers (not just the writer), the SDK must respect `cache_ttl_seconds` returned by resolve
  RPCs (via daemon), overriding any default TTL for that key.

Daemon wrapper:

- Expose a daemon RPC (e.g., `SwapKeyMapping`) so SDK callers remain decoupled from the Global Store endpoint, matching
  the existing `PublishReplicaKey`/`ResolveKeyMapping` pattern.

# Scenarios & Tests (场景与测试)

## (a) Typical scenarios (典型的场景信息)

### Scenario 1: Meta-init model weights and later update in place

- Create a `Binding` from an artifact (alloc + fill).
- Build the model by assigning parameter storage to `binding.tensors[name]`.
- Swap to a new artifact id without reallocating model parameters.

### Scenario 2: Load into an already-allocated model (no deferred loader concepts)

- Allocate model parameters normally (or via another framework).
- Use `artifact.bind_into({name: param.data})` to fill into those buffers.
- Later call `binding.swap(...)` to update in place.

### Scenario 3: Tensor-parallel slices (TP0/TP1) with one-time slice spec

- Each rank creates `artifact_tp = artifact.view(slices=rank_slices)`.
- `binding_tp = artifact_tp.bind(device=local_gpu)`.
- On update, each rank calls `binding_tp.swap(new_artifact)`.
- A coordinator performs a single `activate_key="model:latest"` update once the cluster is ready.

### Scenario 4: Publish as a routable memory replica (full coverage only)

- `binding = tc.artifact("v2").bind(device="cuda:0", publish=True)`
- The binding becomes a routable memory replica without re-hashing GPU memory.

## (b) Sample tests (具体的样例测试)

PyTest-style examples (illustrative; exact fixtures depend on existing test harness):

1) Pointer stability
- Allocate a binding from `v1`, capture `data_ptr()` per tensor, swap to `v2`, assert pointers unchanged.

2) View reuse
- `binding = tc.artifact("v1").view(slices=...).bind(...)`
- `binding.swap("v2")` (no slices passed) and verify the selection identity matches the view’s selection.

3) Publishability gating
- Create a subset selection binding and assert `publish=True` fails fast with `FAILED_PRECONDITION`.
- Create a full-coverage binding and assert `publish=True` succeeds.

4) Activation CAS correctness
- Set key `model:latest` to `v1`.
- Swap to `v2` with `expected_active_artifact_id="v1"` succeeds.
- Swap to `v3` with `expected_active_artifact_id="v1"` fails with conflict.

5) Dirty-state semantics
- Induce a materialization failure during swap and assert the binding transitions to Dirty and blocks publish.

## (c) Operational workflow (我们的操作流程)

Recommended operator/user flow for online weight updates:

1) Choose naming:
   - Immutable versions: `model:v1`, `model:v2`, ...
   - One mutable active key: `model:latest`
2) Rollout:
   - Distribute/ingest `model:v2` (out of scope here; via `register/put`).
   - Each worker swaps its local bindings to `model:v2` (local safety: gate inference).
   - Publish replicas if routing should prefer local memory replicas.
   - Update `model:latest` to point to `model:v2` (CAS, one coordinator).
3) Backout:
   - Swap bindings back to `model:v1`.
   - Update `model:latest` back to `model:v1` (CAS).

# Refactor Plan (方案重构)

## (a) Refactor / migration plan (对现有方案的重构计划)

1) Add `Binding` as the preferred user-facing API:
   - Documented in SDK docs and examples.
   - `Artifact.bind` and `Artifact.bind_into` become the “happy path”.
2) Keep `DeferredLoader` available for advanced placeholder workflows, but:
   - Stop teaching `InplaceSlot` directly in user docs.
   - Consider returning `Binding` from `DeferredLoader.commit()` in a later breaking change, or add
     `DeferredLoader.bind()` as a non-breaking alias.
3) Update existing designs:
   - Add an “API update” note to `docs/designs/0039-artifact-first-sdk.md` and
     `docs/designs/0061-slot-based-inplace-binding-and-swap.md` pointing to this design as the new user-facing surface.

## (b) Implementation details (具体的实现细节)

### SDK changes

- Add `tensorcast/api/store/binding.py` (`Binding`).
- Add `Artifact.bind` and `Artifact.bind_into` in `tensorcast/api/store/artifact.py`.
- Ensure `tensorcast/api/store/__init__.py` re-exports `Binding` (but does not need to re-export `InplaceSlot`).
- Add tests under `tests/python/`:
  - binding pointer stability
  - view reuse
  - publishability gating
  - activation CAS

### Daemon / Global Store changes (for `activate_key`)

- Global Store:
  - Add `SwapKeyMapping` RPC + `generation` to key mapping resolution.
  - Update `tensorcast/global_store/grpc_service.py` to implement swap semantics.
  - Update `schema.sql` to store `generation`.
- Daemon:
  - Add wrapper RPC(s) in `proto/tensorcast/daemon/v2/store_daemon.proto` and implementation in
    `daemon/service/grpc_service_impl.cc`.
  - Preserve the existing `ResolveKeyMapping` surface for callers not aware of activation.

# Schema Changes

If we implement `activate_key`, we must version key mappings. Proposed patch to `schema.sql`:

```sql
ALTER TABLE key_mappings ADD COLUMN generation BIGINT NOT NULL DEFAULT 0;
ALTER TABLE key_mappings ADD COLUMN kind TEXT NOT NULL DEFAULT 'IMMUTABLE';
```

And update the swap/upsert path to increment `generation` on every successful update.

Migration note (重要):
- Global Store initializes DB schema from repo-root `schema.sql` only for new databases.
  Existing deployments require an explicit migration step to add these columns.

# Trade-offs & Risks

- **Cannot “upgrade” daemon-owned materializations into inplace bindings**: tensors returned by `artifact.tensor_dict`
  are typically CUDA IPC imports and are not valid targets for client-owned region registration. Mitigation: provide
  `Artifact.bind(...)` as the one-line, non-deferred user experience for swap, and `bind_into(...)` for truly
  user-owned tensors.
- **Published lease lifecycle**: published client-owned replicas are pid-scoped capabilities; if time-based TTL is used,
  keepalive is required and can be a source of operational flakiness. Mitigation: default `ttl_ms=0` (disable TTL) for
  long-lived published bindings; rely on explicit retire for overwrite and PID-exit cleanup for crash safety.
- **Region TTL / suballocation pointers**: naive region registration may fail in real allocator environments and may
  break swaps after TTL expiry. Mitigation: support `ttl_ms=0` (disable region TTL) for long-lived bindings; use
  allocation-base export + self-heal retry on swap/fill for robustness.
- **Key mapping mutability vs caching**: fast activation conflicts with client-side key caching. Mitigation: versioned
  mappings + server-driven cache TTL hints, plus clear operator guidance.
- **API churn**: introducing `Binding` while keeping `InplaceSlot` risks duplication. Mitigation: `Binding` becomes the
  documented “happy path”, while `InplaceSlot` remains internal/advanced.

# Alternatives & Rationale

- Keep `InplaceSlot` as the primary user-facing surface: correct but leaks internal state-machine concepts and forces
  users to learn slot/commit ergonomics even for the “just swap my weights” use case.
- Add `Artifact.swap(...)`: violates artifact immutability and conflates “value” with “location”; swap needs a stable
  target layout, which is by definition a property of the binding/location.
- Keep key mapping insert-only + client TTL caching: unsafe for fast-moving “active” keys; it creates unavoidable
  cluster-wide stale reads under normal caching and makes “activation” non-deterministic.

# Compatibility & Acceptance Criteria

Compatibility:
- This is an additive API at first (`Binding`, `Artifact.bind`, `Artifact.bind_into`).
- Existing `DeferredLoader` / `InplaceSlot` flows continue to work unchanged.

Acceptance:
- Users can perform swap with only `Artifact` + `Binding` concepts.
- `artifact.view(...).bind(...).swap(...)` works without restating slice specs.
- Pointer stability is preserved across swaps.
- Publishability checks match 0061 (full coverage only).
- When `publish=True`, the published lease remains valid over long runtimes without keepalive (default `ttl_ms=0`).
- Optional activation:
  - supports CAS (`expected_active_artifact_id` or `expected_generation`)
  - is safe with SDK caching (server-driven `cache_ttl_seconds`, activation keys return TTL=0)

# Naming Compliance (Python)

New public interfaces introduced by this design:

- Class (PascalCase): `Binding`
- Methods (snake_case): `Artifact.bind`, `Artifact.bind_into`, `Binding.swap`, `Binding.close`
- Properties (snake_case): `Binding.tensors`, `Binding.artifact_id`, `Binding.selection`

# Naming Compliance (Proto / C++)

If `activate_key` is implemented, new public RPCs/fields must follow repo conventions:

- Proto RPCs (PascalCase): `SwapKeyMapping`
- Proto fields (lower_snake_case): `expected_generation`, `cache_ttl_seconds`, `new_artifact_id`
- C++ methods (snake_case): `swap_key_mapping` (daemon/core wrappers), `resolve_key_mapping` (existing)

# References

- `docs/designs/0039-artifact-first-sdk.md`
- `docs/designs/0061-slot-based-inplace-binding-and-swap.md`
- `docs/designs/0011-unified-session-lifecycle-leases.md`
- `docs/designs/0048-ha-replica-visibility-and-retire.md`
- `docs/architecture/api/region-backed.md`
- `tensorcast/api/store/artifact.py`, `tensorcast/api/store/deferred_loader.py`, `tensorcast/api/store/inplace_slot.py`
- `proto/tensorcast/global_store/v1/global_store.proto`, `schema.sql`
