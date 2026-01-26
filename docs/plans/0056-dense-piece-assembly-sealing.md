---
title: Dense Piece Artifacts and Unsealed-to-Sealed Assembly (Plan)
areas: ["core","daemon","global_store","sdk","proto"]
status: in_progress
created: 2026-01-24
last_updated: 2026-01-26
links:
  design: ../designs/0056-dense-piece-assembly-sealing.md
---

# Summary

Implement dense piece artifacts (view replicas) for partial coverage, make routing/transport ByteSpaceRef-aware end-to-end, and add a sealing path from `assembly_id` (typically `cgid:`) to `mi2:` when canonical coverage is complete.

Current scope (normative; see design for full details):

- All routing/locks/exports/HA checksums are scoped by `ByteSpaceRef { kind, id }` (canonical vs view); no silent cross-ByteSpace fallback.
- Pieces are selection-only (`narrow` only; no `transpose`).
- Assembly synthesis is selection-only (target view must not require compute transforms).
- Pieces must be partial (identity/full-coverage views are rejected for piece registration).
- No LIP pieces (piece registration is implemented only for non-LIP plans).
- No legacy partial semantics: `allow_partial` / “canonical with holes/zero-fill” is removed; partial coverage is represented only via `registration_kind=PIECE`.

This plan now includes a step-by-step implementation order with ownership, dependencies, and concrete deliverables. It is intended to be executable without additional scaffolding.

# Status (2026-01-26)

- Proto/schema: ByteSpaceRef, view-aware replicas, variant coverage ranges, artifact bindings, requested_byte_space, and ListVariants are implemented; codegen regenerated.
- Global Store: view-aware replica routing, coverage persistence + overlap checks, bindings, and view-aware HA checksums are implemented.
- Daemon/Core: registration_kind enforced; view_id computed/validated; dense piece registration with padding zeroing + required view_data_hash; view-aware transport locks and communicator keys; assembly materialization and seal_assembly + binding resolution are implemented.
- SDK: added `register_piece` + `seal_assembly`, with `canonical_index_bytes` bootstrapping; `allow_partial` is deprecated in favor of `registration_kind="piece"`.
- Tests: updated daemon registration expectations for CGID index binding + view_id auto-compute; ran `bazel test //daemon:grpc_service_impl_registration_test` and `bazel test //core/store:registration_memory_replica_test` with `TENSORCAST_CUDA_BACKEND=fake`.
- Remaining: add/expand unit + integration tests (acceptance suite) and run the validation matrix.

# Implementation Order & Ownership

The order below minimizes cross-component breakage while keeping each step shippable. Each step lists primary owners and expected deliverables.

1) **Spec alignment (Owners: Docs)**
   - Deliverables: updated design/plan, explicit breaking-change semantics, ByteSpaceRef naming consistency.
2) **Proto + schema foundation (Owners: Proto, GS)**
   - Deliverables: proto changes, schema + migrations, regenerated stubs.
3) **GS model/repo/services (Owners: GS)**
   - Deliverables: view-aware replicas, variant range persistence, view-aware routing and list variants.
4) **Daemon RPC plumbing (Owners: Daemon)**
   - Deliverables: registration_kind handling, ByteSpaceRef in locks/routing, binding resolution hooks.
5) **Core registration split (Owners: Core)**
   - Deliverables: canonical-vs-piece registration paths, dense view storage, view hash.
6) **P2P key and export safety (Owners: Core)**
   - Deliverables: view-aware communicator keys and lock scoping.
7) **Assembly planning and execution (Owners: Core + Daemon)**
   - Deliverables: assembly plan, piece discovery, partial coverage errors.
8) **Sealing and binding resolution (Owners: Daemon + GS + Core)**
   - Deliverables: seal RPC, bindings table usage, post-seal read semantics.
9) **SDK/CLI updates (Owners: SDK)**
   - Deliverables: register_piece API, bootstrap path, sealing entrypoint.
10) **HA and recovery updates (Owners: Daemon + GS)**
   - Deliverables: checksums include ByteSpace identity.
11) **Tests (Owners: Core/Daemon/GS/SDK)**
   - Deliverables: unit + integration suites for v1 acceptance criteria.

# Current State (Grounding)

- Proto + schema now include ByteSpaceRef, view-aware replicas (`artifact_replicas.view_id`), variant coverage ranges, and artifact bindings.
- Global Store routes by ByteSpaceRef, persists variant coverage/overlap metadata, and exposes ListVariants for assembly planning.
- Core + daemon support dense piece registration, view-aware transport locks/keys, assembly materialization, and sealing with binding resolution.
- Tests/acceptance suite remain pending (see “Testing & Validation”).

# Phases & Milestones

## Phase 0: Spec alignment (close correctness gaps before coding)

Owners: Docs

- [x] (DONE) Update design/plan to reflect the current scope (selection-only pieces/assembly, no overlap across different view_ids, post-seal resolution semantics).
- [x] (DONE) Align with current implementation realities: view registration today enforces canonical-size buffers and uses `allow_partial` to zero-fill; piece registration must be a new code path and “canonical with holes/zero-fill” is removed.
- [x] (DONE) Confirm proto/API shapes use ByteSpaceRef consistently across routing/locks/exports/HA.

## Phase 1: Make replicas view-aware (control plane)

Owners: Proto, Global Store

Dependencies: Phase 0

- [x] (DONE) Extend `schema.sql` so replicas can be keyed by `(artifact_id, view_id)` (view_id = ByteSpaceRef `kind=VIEW` id; canonical uses NULL).
- [x] (DONE) Persist variant canonical coverage in SQL (`variants.canonical_size_bytes`, `variants.canonical_bytes_covered`) and add canonical coverage ranges (`variant_coverage_ranges` or equivalent manifest storage).
- [x] (DONE) Add `artifact_bindings` table for `assembly_id -> mi2_id` sealing outcomes.
- [x] (DONE) Add `tensorcast.common.v1.ByteSpaceRef` and plumb it through `tensorcast.common.v1.MemoryInfo` (`byte_space`).
- [x] (DONE) Extend `RequestReplicaTransportRequest` to include `requested_byte_space: ByteSpaceRef` and enforce ByteSpaceRef-aware routing.
- [x] (DONE) Extend daemon registration API to include `ViewRegistrationKind` (`CANONICAL` vs `PIECE`) and require it (no legacy defaulting).
- [x] (DONE) Regenerate proto bindings after changes (`bash tools/build_proto_python.sh`).

## Phase 2: Make transport view-safe (data plane)

Owners: Core, Daemon, Global Store

Dependencies: Phase 1 (proto + schema)

- [x] (DONE) Add ByteSpaceRef to chunk lock requests so P2P locking is unambiguous.
- [x] (DONE) Fix communicator tensor key generation to incorporate ByteSpace identity and prevent collisions.
- [x] (DONE) Audit any lock/lease/eviction/inventory selectors that currently key only by `artifact_id` and make them ByteSpace-aware (or explicitly canonical-only).
- [x] (DONE) Update HA checksum inputs (daemon + GS recovery) to include ByteSpace identity.

## Phase 3: Dense piece registration semantics

Owners: Core, Daemon, SDK, Global Store

Dependencies: Phase 1 and Phase 2

- [x] (DONE) Update SDK + daemon registration so "piece registration" stores dense view bytes (no canonical-size allocation, no zero-fill of canonical space).
- [x] (DONE) Enforce v1 constraints: reject `transpose` and reject identity/full-coverage views for piece registration.
- [x] (DONE) Enforce canonical index binding for `assembly_id` across all piece registrations.
- [x] (DONE) Introduce an explicit wire discriminator for canonical-vs-piece view registration (e.g., `ViewRegistrationKind`).
- [x] (DONE) Add a bootstrap path so the first piece registration can supply `canonical_index_bytes` without requiring pre-existing GS state.
- [x] (DONE) Add piece immutability rule for `(assembly_id, view_id)` (idempotent retry vs conflict).
- [x] (DONE) Zero padding bytes in the stored view buffer before hashing/publishing; compute `view_data_hash` over the full view ByteSpace.
- [x] (DONE) For CGID assemblies, persist `index_multihash` via `ArtifactDescriptor` on `RegisterReplica` and stop clearing it in LIP commit flows.

## Phase 4: Assembly materialization + sealing

Owners: Core, Daemon, Global Store

Dependencies: Phase 1 through Phase 3

- [x] (DONE) Add Global Store variant enumeration RPC (paginated list) so daemons can discover candidate piece views for assembly, including canonical coverage ranges.
- [x] (DONE) Add daemon/core assembly materialization path (serve a requested selection-only view by assembling from available pieces when the exact view replica is missing).
  - Include canonical requests for unsealed assemblies (ByteSpaceRef `kind=CANONICAL`): `artifact(assembly_id).tensor_dict()` assembles canonical from pieces when no canonical replica exists.
- [x] (DONE) Add `seal_assembly` flow to compute MI2 from assembled canonical bytes and persist `assembly_id -> mi2_id` bindings.
- [x] (DONE) Add binding resolution for reads (default redirect `assembly_id` -> `mi2_id` once sealed).
- [x] (DONE) Decide how view-cached replicas survive sealing: pieces remain under the assembly_id (no alias/migration); post-seal reads resolve to the sealed mi2_id by default.
- [x] (DONE) Decide and document canonical replica publish policy at seal time: `seal_assembly(..., publish_canonical=True)` materializes and publishes a canonical replica when requested (default in SDK).

# Component Ownership and File Map

This section assigns concrete file ownership for implementation. It is not exhaustive but captures the critical edit points.

## Proto / Schema (Owners: Proto, Global Store)

- `proto/tensorcast/daemon/v2/store_daemon.proto`
  - Add `ViewRegistrationKind` to `ViewRegistrationOptions`.
  - Add `ByteSpaceRef byte_space` to `LockTransportChunksRequest`.
- `proto/tensorcast/common/v1/common.proto`
  - Add `ByteSpaceRef` and `MemoryInfo.byte_space`.
- `proto/tensorcast/global_store/v1/global_store.proto`
  - Add `requested_byte_space: ByteSpaceRef` to `RequestReplicaTransportRequest`.
  - Add `ListVariants` (or extend `GetArtifactInfoById`).
  - Add `canonical_ranges` to `VariantUpsert` (or new RPC).
- `schema.sql` + GS migrations
  - `artifact_replicas.view_id`
  - `variants.canonical_size_bytes`, `variants.canonical_bytes_covered`
  - `variant_coverage_ranges`
  - `artifact_bindings`

## Global Store (Owner: Global Store)

- `tensorcast/global_store/grpc_service.py`
  - RegisterReplica honors `descriptor` and `mem_info.byte_space`.
  - RequestReplicaTransport enforces `requested_byte_space` when present.
  - ListVariants RPC returns coverage ranges.
- `tensorcast/global_store/models/replica.py`
  - Add ByteSpace identity fields (kind + id / view_id).
- `tensorcast/global_store/repositories/replica_repository.py`
  - Filter and lookup by `(artifact_id, view_id)` where applicable.
- `tensorcast/global_store/services/transport_service.py`
  - View-aware selection.
- `tensorcast/global_store/services/view_state_service.py`
  - Persist canonical range data (or manifest).
- `tensorcast/global_store/services/recovery_service.py`
  - Include ByteSpace identity in checksum.

## Daemon (Owner: Daemon)

- `daemon/service/controllers/registration_controller.cc`
  - Parse and require `ViewRegistrationKind` (no legacy defaulting).
- `daemon/service/controllers/transport_controller.cc`
  - Accept ByteSpaceRef in lock requests; skip LIP for view replicas.
- `daemon/state/lip_manager.cc`
  - Do not clear `index_multihash` for assembly CGID; enforce no LIP pieces.
- `daemon/ha/worker_lifecycle_manager.cc`
  - Include ByteSpace identity in state checksum.
- `daemon/service/controllers/materialization_controller.cc`
  - Resolve `assembly_id -> mi2_id` binding and pass ByteSpaceRef to routing.

## Core (Owner: Core)

- `core/store/runtime/metadata/registration_backend.cc`
  - Split canonical vs piece registration paths.
  - Allow `canonical_size_bytes != total_size_bytes` for pieces.
  - Zero padding bytes for pieces; compute view hash from view bytes.
  - Publish view metadata + canonical coverage ranges.
- `core/store/replica/memory_export_registry.cc`
  - Include ByteSpace identity in communicator tensor keys.
- `core/store/materialization/control/materialize_orchestrator.cc`
  - Use ByteSpaceRef-aware transport; avoid cross-ByteSpace fallback.
- `core/store/materialization/control` (new)
  - Assembly plan builder + execution (selection-only).

## SDK (Owner: SDK)

- `tensorcast/api/store/registration.py`
  - Add `register_piece` (or `register_view` with `registration_kind="piece"`).
  - Ensure bootstrap path accepts canonical index bytes.
- `tensorcast/api/_register.py`
  - Allow view_size_bytes total size for piece registration.
  - Reject non-selection-only pieces and identity views.

# Rollout

- Rollout sequence: proto+schema+GS view-aware routing first, then daemon/core piece path, then SDK exposure.
- This work is a breaking change for partial registration semantics; update SDK/daemon/GS together (no legacy fallback paths).

# Testing & Validation (Guidance)

- [ ] (TODO) Unit tests for view-aware key generation, view-aware routing selection, and piece-vs-canonical byte space separation.
- [ ] (TODO) Unit tests for canonical coverage range persistence + overlap detection (including "missing coverage metadata" errors).
- [ ] (TODO) Unit tests for padding zeroing + `view_data_hash` determinism.
- [ ] (TODO) Integration test for "put shard(0) pieces, get shard(1) view" across multiple daemons (TorchStore-like scenario).
- [ ] (TODO) Integration test for the concrete TP scenario:
  - register TP8 pieces (8 writers)
  - `tensor_dict()` assembles canonical correctly from pieces (unsealed)
  - TP4 consumers can materialize TP4 views from TP8 pieces (reshard TP8 -> TP4)
- [ ] (TODO) Regression tests ensuring MI2 canonical flows are unchanged.
- [ ] (TODO) Concurrency tests: idempotent piece retries; conflicting piece writes rejected; sealing race idempotence (`assembly_id` bound once).

# Completion Acceptance (Black-box + White-box)

This section is the v1 **release gate** for this design/plan. The work is considered "done" only when **all** black-box and white-box checks pass, and existing MI2 flows remain unchanged.

## Black-box acceptance (user-visible / E2E)

Black-box acceptance validates externally observable behavior via public SDK/CLI + gRPC behavior. It should not depend on reading internal code or querying DuckDB tables directly (Global Store RPCs are allowed for observation).

### Test environment matrix

- **CI / no GPU**: run under a recognized test environment with fake CUDA backend:
  - `TENSORCAST_CUDA_BACKEND=fake uv run pytest ...`
  - `bazel test ... --test_env=TENSORCAST_CUDA_BACKEND=fake`
- **Real GPU (recommended)**: re-run the same E2E suite without `TENSORCAST_CUDA_BACKEND` set.

### Scenario A: Bootstrap piece registration + dense semantics + index binding

Setup: start 1 Global Store and 1 daemon connected to that Global Store.

Steps (API-level):

1) Register the **first piece** for a new `assembly_id="cgid:..."` using the new piece pathway:
   - `Store.register_piece(...)`, or
   - `Store.register_view(..., registration_kind="piece")`
2) The registration MUST include `canonical_index_bytes` inline (bootstrapping requirement).
3) Use a **selection-only** view (v1: `narrow`/slices only; no `transpose`).
4) Materialize the same piece view and verify the bytes match what was registered.

Pass criteria:

- Commit returns a non-empty `view_id` (the piece is routable as a view replica).
- Commit returns `view_data_hash` for the piece (required for immutability).
- Global Store can resolve the assembly's canonical index:
  - `GetArtifactIndexById(assembly_id)` returns **exactly** the same bytes as provided at bootstrap.
- Global Store can resolve view metadata for the piece:
  - `GetArtifactInfoById(assembly_id, view_id=..., include_view_meta=true)` returns `{view_spec_json, view_size, view_data_hash}` consistent with the commit response.
- ByteSpaceRef + dense sizing is observable:
  - The registered replica is reported under the view ByteSpace (`MemoryInfo.byte_space.kind=VIEW`, `MemoryInfo.byte_space.id == view_id`).
  - The reported replica size equals `view_size_bytes` (e.g., `MemoryInfo.memory_size == view_size_bytes`).

### Scenario B: Piece immutability (idempotent retry vs conflict)

Steps:

1) Re-register the same `(assembly_id, view_id)` with identical bytes.
2) Re-register the same `(assembly_id, view_id)` with different bytes.

Pass criteria:

- (1) succeeds and returns the same `view_data_hash` (idempotent retry).
- (2) fails with `FAILED_PRECONDITION` and a clear "piece content conflict" signal.

### Scenario C: Overlap rejection across different view_ids (v1 policy)

Steps:

1) Register piece A and piece B under the same `assembly_id` with **different** `view_id`s.
2) Ensure their canonical coverage ranges overlap (e.g., two writers cover the same canonical interval).

Pass criteria:

- The second registration fails with `FAILED_PRECONDITION` and identifies overlap as the reason (ranges surfaced when available).

### Scenario D: v1 constraints are enforced (selection-only + no LIP pieces)

Steps / pass criteria:

- Omitting `registration_kind` MUST fail with `INVALID_ARGUMENT`.
- Setting `allow_partial=true` MUST fail with `FAILED_PRECONDITION` (legacy sparse semantics are removed).
- Piece registration with `transpose` MUST fail with `FAILED_PRECONDITION` (v1 forbids compute transforms for pieces).
- Assembly materialization targeting a view that requires compute (e.g., `transpose`) MUST fail with `FAILED_PRECONDITION` with actionable guidance.
- Piece registration attempted on LIP / `VRAM_LEASED` MUST fail (v1 forbids LIP pieces).
- Piece registration that normalizes to identity / full canonical coverage MUST fail (pieces must be partial in v1).

### Scenario E: Unsealed assembly reads (canonical + reshard)

Setup: register a set of pieces whose union fully covers the canonical byte space (e.g., TP8: 8 disjoint shards).

Steps:

1) Canonical read from an unsealed assembly:
   - `artifact(assembly_id).tensor_dict()` (no explicit view) when no canonical replica exists.
2) Reshard read:
   - request a different selection-only target view (e.g., TP4) and materialize it from TP8 pieces.
3) Verify view disambiguation:
   - materialize at least two different TP8 piece views and assert each returned payload matches the corresponding registered shard bytes.
4) Negative case:
   - attempt reads when coverage is incomplete.

Pass criteria:

- (1) returns correct canonical tensors (byte-for-byte) assembled from pieces.
- (2) returns correct target view tensors assembled from pieces, without requiring sparse canonical staging during registration.
- (3) returns the correct bytes for each requested piece view (guards against ByteSpace collisions / wrong-replica routing).
- (4) fails with `UNAVAILABLE` and includes machine-actionable missing canonical ranges (recommended: reuse `PartialCoverageDetail` semantics).
- Source attribution is correct (e.g., `MaterializeReplicaResponse.source` reports P2P or local replica, not accidental disk fallback).

### Scenario F: Sealing + redirect semantics

Setup: full canonical coverage exists.

Steps:

1) Call `seal_assembly(assembly_id)` and record the returned `mi2_id`.
2) Repeat sealing (idempotence) and run two sealers concurrently (race).
3) After sealing, read by `assembly_id` and by `mi2_id`.
4) Attempt to register additional pieces after sealing.

Pass criteria:

- Sealing returns a stable `mi2_id` and is idempotent under retries/races (single binding).
- Global Store records and serves `assembly_id -> mi2_id` binding (via the chosen binding RPC).
- Default read behavior redirects `assembly_id` to `mi2_id` after sealing.
- Post-seal piece writes are rejected with `FAILED_PRECONDITION`.
- At least one canonical replica is published for `mi2_id` (recommended default for durability/tooling).

### Suggested automated black-box suite (pytest)

Maintain a single focused E2E suite (recommended file name):

- `tests/python/test_dense_piece_assembly_sealing_acceptance.py`

Run (fake backend for CI):

```bash
TENSORCAST_CUDA_BACKEND=fake uv run pytest -q tests/python/test_dense_piece_assembly_sealing_acceptance.py
```

## White-box acceptance (invariants / regression coverage)

White-box acceptance proves the design's named ambiguity/collision points are actually fixed in this repo, and that the implementation is guarded by unit/integration tests.

### A) Proto + codegen

- After any `.proto` change: `bash tools/build_proto_python.sh`
- Proto CI gate: `bazel test //proto/... --test_output=errors`

### B) Schema + migrations (Global Store)

The DB schema (and any migrations) MUST include and exercise:

- `artifact_replicas.view_id` (nullable for canonical)
- `variants.canonical_size_bytes`, `variants.canonical_bytes_covered`
- `variant_coverage_ranges` (or equivalent manifest table) for overlap/completeness checks
- `artifact_bindings` (assembly -> mi2 sealing outcome)

Acceptance method:

- Add/extend Global Store schema persistence tests to assert the columns/tables exist and round-trip (DuckDB) correctly.
  - Suggested command: `uv run pytest -q tests/python/global_store/test_schema_persistence.py`

### C) Global Store behavior correctness

Acceptance method (python tests; extend existing suites as needed):

- View-aware replicas and routing:
  - `RequestReplicaTransport` MUST honor `requested_byte_space` (select only matching ByteSpace-scoped replicas).
  - `RequestReplicaTransport` MUST NOT silently fall back across ByteSpaces:
    - requested `VIEW` ByteSpaceRef MUST NOT return canonical replicas
    - requested `CANONICAL` ByteSpaceRef MUST NOT return view replicas
  - `GetArtifactInfoById(include_replicas=true)` MUST return view-scoped replicas without conflating them with canonical.
- Assembly CGID index binding:
  - `RegisterReplicaRequest.descriptor` MUST be honored so `cgid:` assemblies persist `index_multihash`.
  - `GetArtifactIndexById(assembly_id)` MUST work for assemblies (not only MI2).
- Checksum correctness:
  - `tensorcast/global_store/services/recovery_service.py::_compute_state_checksum` MUST incorporate ByteSpace identity to avoid inventory collisions.

Suggested commands (extend tests to cover the new ByteSpaceRef / binding behavior):

```bash
uv run pytest -q tests/python/global_store/test_grpc_service.py
uv run pytest -q tests/python/global_store/test_recovery_service_checksum.py
```

### D) Daemon: locks, LIP, HA

Acceptance method (bazel tests; extend/add where needed):

- Transport locks:
  - `LockTransportChunksRequest` MUST carry and plumb `byte_space: ByteSpaceRef` to scope locks to the correct replica identity.
  - Any LIP staged export fast path MUST remain canonical-only in v1 (view replicas bypass it).
- HA checksums:
  - `daemon/ha/worker_lifecycle_manager.cc::compute_state_checksum` MUST incorporate ByteSpace identity and match GS formatting.

Suggested commands (examples; add new targets as assembly/sealing lands):

```bash
bazel test //daemon:grpc_service_impl_registration_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors
bazel test //daemon:transport_lock_manager_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors
bazel test //daemon:worker_lifecycle_manager_sync_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors
```

### E) Core: registration split, export keys, assembly, sealing

Acceptance method (bazel tests; extend/add where needed):

- Registration split:
  - `core/store/runtime/metadata/registration_backend.cc` MUST implement distinct canonical-vs-piece registration semantics.
  - Canonical-kind view registration MUST reject partial coverage (no more canonical zero-fill).
  - Piece-kind registration MUST allow `canonical_size_bytes != total_size_bytes`, compute deterministic `view_data_hash` (after zeroing padding), and publish canonical coverage ranges.
- Export key safety:
  - `core/store/replica/memory_export_registry.cc` MUST include ByteSpace identity in communicator tensor key generation to avoid collisions.
- Assembly planning:
  - Deterministic `AssemblyPlan` generation for a fixed candidate set.
  - Overlap across different `view_id` rejected in v1; missing coverage returns `UNAVAILABLE` with missing ranges detail.
- Sealing:
  - Sealing computes `data_multihash` by streaming assembled canonical bytes (no required full canonical staging buffer for hashing) and records the binding.

Suggested commands (examples; add new targets for assembly/sealing planners):

```bash
bazel test //core/store/runtime/metadata:metadata_gateway_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors
bazel test //core/store/materialization/dataplane:view_planner_test --test_output=errors
bazel test //core/store/replica:replica_p2p_registration_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors
```

### F) Static "no known ambiguity left" grep gates (cheap, but effective)

These are intentionally narrow "tripwires" for regressions on the exact pitfalls called out in the design's "Correctness gaps".

- The placeholder fallback must be gone:
  - `rg -n \"request_view_transport fell back to canonical routing\" core/store/components/global_store_client.cc` returns no matches.
- Inventory/checksum must not be artifact-id-only anymore:
  - `rg -n \"artifact_id:node_id\" daemon/ha/worker_lifecycle_manager.cc` should show the ByteSpace-aware format (with view ByteSpace identity in the stable string).
