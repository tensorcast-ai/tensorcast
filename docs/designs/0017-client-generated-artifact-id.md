---
slug: client-generated-artifact-id
title: Client‑Generated Artifact ID (CGID)
areas: ["core","daemon","global_store","sdk","proto"]
links:
  prior_design: ./0007-content-addressed-artifact-id.md
  session_api: ./0014-store-session-api-modernization.md
  views: ./0016-artifact-view-v1.md
related_code:
  - core/store/**
  - daemon/**
  - tensorcast/global_store/**
  - tensorcast/api/**
---

# Summary

Introduce an alternative artifact identity kind, CGID (Client‑Generated ID), for high‑churn, runtime artifacts such as KVCache blocks used by PagedAttention. CGID complements, not replaces, the existing content‑addressed identity (mi2) defined in 0007. It removes server‑side hashing from hot registration paths while preserving simple routing, zero‑copy/P2P transport, and explicit lifecycle (register → serve → deregister).

Key outcomes
- Dual identity kinds: `artifact_id` can be either `mi2:...` (content‑addressed) or `cgid:...` (client‑generated).
- Low‑overhead registration: CGID path skips `index_multihash`/`data_multihash` computation; daemon trusts a client‑supplied identifier.
- Same verbs and flows: Keep `register/put/get/get_into` from 0014; a single optional `artifact_id` parameter (when set to a `cgid:`) selects the CGID path.
- Short‑lived/ephemeral by default: CGID artifacts are intended for runtime residency with TTL/explicit deregistration; no tree‑hash leaves or de‑dup semantics.

# Goals / Non‑Goals

Goals
- Eliminate commit‑time hashing for KV‑style high‑frequency registrations without changing the public API shape.
- Preserve transport correctness and isolation: existing transport lock + staged export guarantees continue to apply.
- Make identity space explicit and validated across SDK/daemon/Global Store (GS) so both `mi2:` and `cgid:` are first‑class.

Non‑Goals
- Do not remove or weaken mi2. Persistent, verifiable artifacts continue to use `mi2:`; CGID is additive.
- Do not change P2P/IPC mechanics or memory plans (LIP/coalesced) beyond identity handling.
- Do not introduce view planning, partial registration, or variant ByteSpaces here (see 0016). This document only establishes identity semantics.

# Architecture & Interfaces

## 1. Identity kinds and grammar

- Identity kinds: `IDKind = { MI2, CGID }`.
- `artifact_id` grammar (closed set of prefixes):
  - `mi2:` — unchanged (see 0007)
  - `cgid:` — client‑generated, opaque id. Validation:
    - ASCII, length 8..200
    - Allowed chars: RFC 3986 unreserved `[-._~A-Za-z0-9]` (covers hex and base64url without `=`)
    - Recommended profile for KVCache: `cgid:<block_hash_hex64>` where `<block_hash_hex64>` is caller‑computed SHA‑256 hex of the logical block (or other collision‑resistant hash used by the engine). The daemon does not recompute or verify content.

Semantics
- `cgid:` denotes an ephemeral identity: no de‑dup, no tree‑hash, no GS leaves. Routing and transport are keyed by the id string and device residency.
- `mi2:` denotes a persistent, verifiable identity with structural and data digests.

Shared helpers
- Python: `tensorcast.common.identity` defines `ArtifactIdKind`, `infer_artifact_id_kind()`, and `validate_client_generated_id()` so SDK and Global Store share grammar.
- C++: `core/common/artifact_identity.h/.cc` expose the same helpers for the daemon, store engine, and tests, eliminating ad-hoc validation.

## 2. SDK changes (0014 Store API compatible)

Public facade remains unchanged; registration consolidates to a single optional identity parameter:

```python
# Optional artifact_id selects CGID when provided as a "cgid:" string
tc.register(tensors, *,
            artifact_id: str | None = None,   # when set and startswith("cgid:"), use CGID; otherwise compute mi2
            key: str | None = None,
            ttl_ms: int | None = None) -> RegisteredArtifact

# Convenience wrappers (optional):
# tc.register_cgid(tensors, cgid: str, *, key=None, ttl_ms=None)
# tc.register_kv_block({"k": k, "v": v}, block_hash, *, ttl_ms=None)  # internally uses artifact_id=f"cgid:{block_hash}"
```

Rules
- If `artifact_id` is provided and starts with `cgid:`, SDK uses the CGID path (no hashing) and forwards the id to the daemon; LIP/coalesced plan selection and storage graph collection are unchanged.
- If `artifact_id` is not provided, SDK takes the mi2 path (content‑addressed) as today.
- Supplying an `artifact_id` that does not start with `cgid:` is invalid (SDK or daemon rejects to avoid forged mi2 ids).
- `ttl_ms` continues to control lease keepalive for LIP replicas; CGID favors short to medium TTL.
- `get/get_into` accept either `artifact_id` (which can be `mi2:` or `cgid:`) or `key`, unchanged from 0014.

## 3. Daemon changes (registration/commit identity)

Extend the registration RPCs to accept an optional client‑provided identity and derive behavior from it (no explicit hashing mode parameter):

Proto changes (now merged):
```proto
message BeginRegisterArtifactRequest {
  // existing fields ...
  string client_artifact_id = 1002;  // optional; when present and starts with "cgid:", server uses CGID path
}

message CommitRegisteredArtifactResponse {
  tensorcast.common.v1.ArtifactDescriptor artifact_descriptor = 2;
}
```
`tensorcast.common.v1.ArtifactDescriptor` includes `ArtifactIdKind id_kind = 7;`, and generated code in both languages exposes the strongly typed enum.

Server behavior
- `RegistrationController::begin()` validates `client_artifact_id` using the shared helper and records the desired identity kind in `RegistrationManager::RegMeta`.
- `LipManager::commit_lease_in_place()` and `StoreEngine::commit_registered_artifact()` branch on the recorded kind:
  - `MI2`: compute index/data multihashes, construct a `mi2:` descriptor, and populate digest fields.
  - `CGID`: skip hashing entirely, mirror the client-supplied identifier, and leave digest fields empty.
- `CommitLeaseResult` and `RegistrationCommitResult` propagate `id_kind` so downstream controllers, lifecycle leases, and responses remain consistent.

Transport and locking
- Unchanged. Transport locks and staged exports continue to require a stable `artifact_id` regardless of identity kind.

## 4. Global Store changes (identity & routing)

Identity recognition
- gRPC handlers use `infer_artifact_id_kind()` to classify ids and reject invalid prefixes. Repository/service layers persist the explicit kind so policies and queries can pivot on `id_kind`.

Persistence model
- `schema.sql` and `tensorcast/global_store/init.sql` add `artifacts.id_kind TEXT NOT NULL DEFAULT 'MI2'`, relax the digest columns to accept `NULL`, and extend `artifact_replicas` with an optional `expires_at` timestamp.
- Migration `tensorcast/global_store/migrations/0017_cgid.py` applies the same DDL for existing deployments; it is reversible.
- `ArtifactRepository` upserts descriptors with explicit `id_kind`. `mi2:` records retain digests; `cgid:` records store `NULL` digests.
- `ReplicaRepository` persists `expires_at` when supplied (favoring short-lived CGID replicas) while keeping existing write paths for MI2 replicas untouched.
- Key mappings continue to target either identity kind; consumers must be prepared for empty digest fields when resolving CGID entries.

Minimal schema evolution (illustrative; exact patch in the plan that changes `schema.sql`):
```sql
-- artifacts: add identity kind and relax digest nullability for CGID
ALTER TABLE artifacts
  ADD COLUMN id_kind TEXT NOT NULL DEFAULT 'MI2',
  ALTER COLUMN index_multihash DROP NOT NULL,
  ALTER COLUMN data_multihash DROP NOT NULL;

-- replicas: ensure TTL/ephemeral residency can be tracked uniformly
ALTER TABLE replicas
  ADD COLUMN expires_at TIMESTAMPTZ NULL;
```

Routing behavior
- `GetArtifactInfoById` must accept both id kinds. When the id has prefix `cgid:`, fields specific to mi2 (leaves, digests) are empty/omitted; callers should not request `include_leaves`.

## 5. Error model & invariants

Invariants
- `artifact_id` prefix determines validation and downstream capabilities.
- `cgid:` artifacts are not verifiable by tree hash within TensorCast; integrity is the caller’s responsibility. Transport correctness is still enforced (locks, PAD zero‑fill, range checks).
- Canonical index bytes, when present, must remain self‑consistent with the provided storage/alias metadata; the daemon continues to enforce index layout rules in registration.
- CGID descriptors intentionally propagate empty digest fields; consumers should treat `index_multihash` / `data_multihash` as unset when `id_kind` is `CGID`.

Errors
- `INVALID_ARGUMENT`: client supplied an `artifact_id`/`client_artifact_id` that does not start with `cgid:` or violates grammar.
- `FAILED_PRECONDITION`: caller requests mi2‑only features (e.g., `include_leaves`) for a `cgid:` id.
- `UNAVAILABLE`: no resident replica for a transient `cgid:` id and disk fallback is disabled/not applicable.

## 6. Observability & policy

- Tag metrics and logs with identity kind (derived from `artifact_id` prefix) to distinguish CGID vs MI2 paths (register latency, bytes, retries).
- Admission control: deployments may configure policies to allow/deny CGID usage, enforce namespaces (e.g., `cgid:kv:<hash>`), or limit per‑tenant cardinality/TTL.

# Compatibility & Acceptance Criteria

Compatibility
- SDK verbs (`register/put/get/get_into`) are unchanged; `artifact_id` parameters accept either `mi2:` or `cgid:`. Keys can continue to map to either kind.
- Existing mi2 behavior and data remain intact.
- LIP and coalesced memory plans, CUDA IPC export, and P2P transport are unaffected.

Acceptance criteria
- Registering with `artifact_id="cgid:..."` returns the same `artifact_id` and does not compute/return mi2 digests.
- `get/get_into` successfully materialize replicas by CGID across nodes, using existing transport locks.
- GS can list/resolve replicas for `cgid:` ids and omit mi2‑specific fields without breaking consumers.
- Mixed deployments (some clients using mi2, some using CGID) operate without behavioral regressions.

# Trade‑offs & Risks

Trade‑offs
- CGID trades away server‑side integrity verification and de‑duplication for speed and simplicity.
- Duplicate CGIDs (caller error) would alias independent content; deployments should namespace or sign CGIDs where necessary.

Risks & mitigations
- Collision/abuse: recommend namespacing (`cgid:<ns>:<id>`), tenancy scoping, and rate/cardinality limits.
- Misuse of mi2‑only APIs: explicit `id_kind` in responses and clear error codes reduce surprises; SDK wrappers can gate usage at call sites.

# References

- Identity (mi2): [0007 Content‑Addressed Artifact ID](./0007-content-addressed-artifact-id.md)
- Session API: [0014 Store‑Centric Artifact Session API](./0014-store-session-api-modernization.md)
- Views: [0016 Variant‑Aware Artifact Views](./0016-artifact-view-v1.md)

- Key code locations for the implementation:
  - `tensorcast/common/identity.py` and `core/common/artifact_identity.cc` — shared validation helpers and identity kind enum.
  - `tensorcast/api/_register.py` & `tensorcast/api/store.py` — SDK path that accepts CGID, skips hashing, and surfaces `ArtifactIdKind`.
  - `daemon/service/controllers/registration_controller.cc` & `daemon/lip_manager.cc` — daemon validation and CGID-aware commit path.
  - `core/store/store_engine.cc` — Store engine commit logic propagating identity kind to Global Store.
  - `tensorcast/global_store/repositories/*.py` & `tensorcast/global_store/grpc_service.py` — Global Store persistence and routing with `id_kind` and `expires_at` support.
