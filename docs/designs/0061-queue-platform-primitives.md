---
slug: queue-platform-primitives
title: Queue Platform Primitives (Retention Handles + Capability Tokens + Capability Directory)
description: Platform-level primitives that make descriptor-only queues and other control-plane workflows consistent: daemon-issued retention handles, a unified capability-token envelope, and Global Store capability discovery.
areas: ["daemon", "core", "global_store", "sdk", "proto"]
status: draft
created: 2026-01-31
last_updated: 2026-01-31
related_code:
  - docs/designs/0004-unified-runtime-config.md
  - docs/designs/0011-unified-session-lifecycle-leases.md
  - docs/designs/0034-stable-memory-tiers.md
  - docs/designs/0043-unified-pinned-memory-authority.md
  - docs/designs/0055-programmable-framework.md
  - docs/designs/0060-tensor-work-queue.md
  - core/store/memory_tier_budget.h
  - core/store/components/stable_dram_cache_manager.h
  - core/store/components/stable_dram_cache_policy.h
  - daemon/state/session_lifecycle.h
  - daemon/service/controllers/registration_controller.cc
  - proto/tensorcast/common/v1/common.proto
  - proto/tensorcast/plan/v1/plan.proto
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/global_store/v1/global_store.proto
  - tensorcast/global_store/README.md
links:
  plan: ../plans/0061-queue-platform-primitives.md
  config: ./0004-unified-runtime-config.md
  leases: ./0011-unified-session-lifecycle-leases.md
  stable_tiers: ./0034-stable-memory-tiers.md
  pinned_memory: ./0043-unified-pinned-memory-authority.md
  programmable: ./0055-programmable-framework.md
  queue: ./0060-tensor-work-queue.md
  schema: ../../schema.sql
---

# Summary

TensorCast already has strong primitives for:

- deterministic identities and retry semantics (`CallContext`, selection fingerprints),
- daemon-side lifecycle control (Lease/Guard/Finalizer),
- stable/preemptible budgeting (UMA + `MemoryTierBudget`),
- daemon-issued capability tokens (placement pins, target capabilities).

What is missing is a **platform-owned retention + token + discovery substrate** that allows *control-plane components*
(like a queue broker) to safely manage “keep this payload rematerializable for a bounded time” without relying on
PID-bound registration keepalives.

This design introduces three platform primitives:

1. **Retention Handles (daemon-scoped, capability-tokenized)**
   - A `RetentionHandle` is an explicit, renewable right that keeps a daemon-owned payload rematerializable (v1: local
     stable DRAM; v2+: optional durable upgrade).
   - Handles support **renew** and **release**, and release can **downgrade** retention (unlike today’s “upgrade-only”
     stable retention policy).
   - Handles are implemented using the daemon’s Lease/Guard/Finalizer discipline so cleanup is idempotent and bounded.

2. **Unified Capability Token Envelope**
   - Standardize a single versioned token envelope used by placement pins, retention handles, and future broker-issued
     tokens (e.g., queue WorkLease tokens).
   - Tokens are issuer-scoped (by `daemon_id`), time-bounded, and unforgeable.

3. **Capability Directory (Global Store, low-frequency)**
   - Extend worker/instance discovery to include a compact, bounded set of **capability flags** (e.g.,
     `queue_broker_enabled`), so clients and controllers can discover control-plane endpoints without static broker lists.
   - This is *directory/discovery only*. Hot-path state remains daemon-local.

These primitives are a **prerequisite** for the queue design in `0060-tensor-work-queue`, but are intentionally useful
beyond queues (programmatic orchestration, control-plane agents, and future catalog services).

---

# Goals / Non-Goals

## Goals

1. **Cross-process retention without PID coupling**
   - Retention MUST be renewable by non-owner processes (broker/controller) via a capability token, not by PID-bound
     registration keepalive.

2. **Release semantics that actually free resources**
   - Retention MUST be releasable, and release MUST be able to downgrade effective stable retention when all handles are
     released (so “owned cleanup” is real).

3. **Consistent, reusable token validation**
   - All daemon-issued tokens SHOULD share a single envelope and validation library (issuer identity, expiry, scope,
     signature) to avoid drift across subsystems.

4. **Low-frequency, bounded discovery**
   - Capability discovery MUST be low-write-rate, bounded-cardinality, and align with existing Global Store HA and
     identity practices (stable `daemon_id` over endpoint addresses).

## Non-Goals

- Implementing the queue itself (see `0060-tensor-work-queue`).
- Implementing a durable, global GC model for content-addressed artifacts.
- Re-defining selection identity (assumed already standardized).
- Making Global Store the SoT for high-frequency retention state.

---

# Architecture & Interfaces

## Retention Handles

### Intuition

A retention handle is a daemon-owned lease-like record that answers one question:

> “For the next *T* ms, will there exist at least one rematerialization path for this payload, subject to policy?”

In v1 the primary rematerialization path is **local stable DRAM** managed by the StoreEngine’s stable retention
machinery (`StableDramCacheManager` + UMA stable leases). In v2+ a durable upgrade may create additional paths.

### V1 rematerialization definition (required)

In v1, a caller should treat “rematerialization path exists” as:

- **Issuer-local, policy-scoped**: the issuing daemon can currently provide at least one serving path *from its own
  StoreEngine* for the referenced `ArtifactSelection`, consistent with the provided `StorePolicy`.
- **Stable-DRAM anchored (v1)**: the intended path is a daemon-owned CPU replica held in the stable tier (UMA stable
  lease / stable cache admission). This is **not** a cluster-wide durability guarantee and does not survive loss of the
  issuer daemon’s local state unless the `StorePolicy` also provides a durable fallback tier.

Acquire semantics boundary (required):

- `AcquireRetentionHandle` / `RenewRetentionHandle` / `ReleaseRetentionHandle` are **control-plane only** and MUST NOT
  trigger payload materialization, transfer, or registration. They only adjust retention intent for an already-known
  selection in the issuer daemon’s StoreEngine. Any “make bytes exist” work is performed by existing data-plane flows
  (e.g., `put`, persistence tasks).

This definition is intentionally minimal so it composes:

- With `0034-stable-memory-tiers` (stable leases are explicit, budgeted, and released),
- With `0011-unified-session-lifecycle-leases` (expiry and cleanup are bounded and idempotent),
- With `0060-tensor-work-queue` (broker can dead-letter on a clear “path lost” signal from the issuer).

### API (conceptual)

The daemon exposes a small API surface (exact proto placement is flexible):

- `AcquireRetentionHandle(selection, *, policy, ttl_ms, ctx) -> RetentionHandle`
- `RenewRetentionHandle(handle_token, *, extend_ttl_ms, ctx) -> RetentionHandle` (or updated expiry)
- `ReleaseRetentionHandle(handle_token, ctx) -> bool`

`RetentionHandle` is represented in the SDK as:

- `handle_id` (debuggable id)
- `expires_at_ms`
- `capability_token` (required for renew/release)
- `charged_bytes` (authoritative bytes charged for retention/accounting)
- optional `diagnostics` (why admitted/skipped, tier status)

### Key properties (normative)

- **Issuer-scoped**: a retention handle is always bound to one issuer daemon (`daemon_id`).
- **Time-bounded**: handles MUST expire; all renewals MUST be bounded by caller deadlines.
- **TTL caps (required)**: the issuer MUST clamp requested TTLs to a configured maximum so handles cannot pin resources
  indefinitely due to buggy callers or lost control-plane state.
- **Idempotent release**: releasing the same token multiple times MUST be safe.
- **Downgrade on last release**: when the last handle for a given canonicalized selection identity
  (`ArtifactSelection.logical_layout_hash` + `selection_hash`, issuer-scoped) is released or expires, the daemon MUST
  reduce effective retention intent so stable resources can be reclaimed.

### Error model (normative)

To keep queue semantics consistent across implementations, the retention API MUST follow a stable error taxonomy:

- `AcquireRetentionHandle`
  - `RESOURCE_EXHAUSTED`: the issuer cannot admit the request under current tier budgets/policies.
  - `FAILED_PRECONDITION`: the selection/policy is invalid or cannot be made rematerializable (e.g., missing/unknown
    selection identity).
  - `UNAVAILABLE`: the issuer is not able to serve control-plane requests (starting, fenced, shutting down).
- `RenewRetentionHandle`
  - `FAILED_PRECONDITION`: the handle token is invalid/expired/unknown (non-retryable; caller must treat as “path
    lost” for v1 queue semantics).
  - `UNAVAILABLE` / `DEADLINE_EXCEEDED`: transient issuer unreachability (retryable within bounded deadlines).
- `ReleaseRetentionHandle`
  - MUST be idempotent; repeated release MUST return success.
  - SHOULD NOT expose “unknown handle” as an error (treat as already-released) so callers can safely retry cleanup.

### Implementation sketch (daemon)

We introduce a daemon-local `RetentionRegistry` that:

- tracks active handles (by `handle_id`) and aggregates them into an **effective retention intent** per payload
  (refcount + max policy rank + max expiry deadline),
- enforces admission by charging against stable tier budget and applying stable retention policy via
  `StoreEngine::admit_stable_cache_policy(...)`,
- performs cleanup via the unified Lease/Guard/Finalizer system:
  - `DeadlineGuard(expires_at)` retires handles,
  - finalizer decrements refs and triggers downgrade logic when refs reach zero.

Important: this design intentionally pushes “correct, bounded cleanup” into the lifecycle system rather than
background scanning.

### Relationship to StorePolicy and MemoryTierBudget

- StorePolicy remains the single namespace for tier choice, retention semantics, and overflow/spill behavior.
- Stable DRAM retention consumes UMA stable bytes and is admitted via `MemoryTierBudget`.
- Retention handles MUST NOT introduce a separate “pinned memory” concept; pinned pools remain data-plane resources.

## Unified Capability Token Envelope

The platform standardizes a token envelope used for *all daemon-issued capabilities* (including broker-issued
capabilities when the broker runtime is embedded in the daemon).

Verification model (normative, v1):

- **Issuer-only**: tokens MUST be validated by the issuing daemon. Callers present the token back to the issuer; no
  third-party token verification or key distribution is required for v1 correctness.
- **Issuer binding**: tokens MUST include `issuer_daemon_id`, and the issuer MUST reject tokens whose
  `issuer_daemon_id != self.daemon_id`.

Token requirements (normative):

- **Unforgeable**: signed/authenticated with a key loaded from unified runtime config.
- **Versioned**: tokens MUST carry a `token_version` to allow key rotation and format evolution.
- **Issuer identity**: includes `issuer_daemon_id` (not the advertised address).
- **Audience binding**: includes an `audience` field (service or capability kind) to prevent token type confusion (e.g.,
  a retention token must not be accepted as a work-lease token).
- **Scope binding**: binds tightly to a scope-specific payload (e.g., placement pin id, retention handle id).
  - `scope` MUST be the deterministic Protobuf serialization of an audience-specific scope message so the issuer can
    parse and validate it consistently across languages and versions.
- **Expiry**: includes `expires_at_ms` and MUST be checked by the issuer.
- **Optional fencing**: when a capability authorizes writes to volatile control-plane state, the token MUST embed a
  fencing field (e.g., `queue_epoch`) so stale tokens fail fast after leader changes. Fencing MUST include both a
  stable fencing principal (e.g., `queue_operation_id`) and an epoch/generation value to avoid cross-queue confusion.

This design does not mandate a specific crypto primitive, but it MUST support fast verification (hot path) and key
rotation with bounded overlap.

### Envelope schema (conceptual)

The envelope is intentionally “small and boring” so it can be reused across C++ and Python, and reasoned about in
failure-handling logic:

Authoritative proto placement (recommended): `proto/tensorcast/common/v1/capability_token.proto` (package
`tensorcast.common.v1`).

```proto
message CapabilityTokenEnvelope {
  uint32 token_version = 1;        // key id / format version
  string issuer_daemon_id = 2;     // stable identity (not address)
  CapabilityAudience audience = 3; // prevents token type confusion
  bytes scope = 4;                // audience-specific payload (e.g., handle_id, lease_id)
  uint64 expires_at_ms = 5;        // absolute expiry

  // Optional fencing for volatile control-plane writes.
  oneof fencing {
    QueueEpochFencing queue_epoch = 10;
  }

  // Opaque authentication tag (MAC/signature over fields above).
  bytes auth_tag = 100;
}

message QueueEpochFencing {
  string queue_operation_id = 1; // e.g., "queue:v1:<queue_name>"
  uint64 queue_epoch = 2;        // Global Store operation lease_generation
}
```

Notes:

- SDKs SHOULD treat tokens as opaque `bytes` and never parse them.
- The issuer MUST validate `audience` before reading `scope`.

### Key management and rotation (required)

- Token keys MUST be loaded from the unified runtime config (no ad-hoc env vars).
- Issuers MUST support bounded-overlap key rotation:
  - tokens embed `token_version`,
  - the issuer accepts the active key and a bounded set of previous keys for verification.

### Migration and compatibility (required)

- Issuers SHOULD support dual-format verification for a bounded transition window when migrating existing capabilities
  (e.g., placement pin “lease tokens”) to the unified envelope. The token format MUST be unambiguously detectable by the
  issuer (via `token_version` or a stable format tag) so clients can roll independently.

### HA note (non-goal for v1, required to call out)

Issuer-only validation is sufficient for v1 correctness, but it constrains future HA shapes:

- If a capability must remain valid across issuer failover (e.g., an HA queue broker validating tokens on a different
  node), then the HA design must introduce a shared verification key or a replication strategy for issuer identity/keys.
- This design does not choose that mechanism; it only standardizes the fields needed so a future HA scheme can be added
  without changing client-facing APIs.

## Capability Directory (Global Store)

We extend Global Store discovery surfaces to include a compact capability set.

### What is stored

Per daemon/worker/instance record, store:

- stable identity (`daemon_id`, `instance_id` when applicable),
- routable endpoint attributes (address/port),
- **capability flags** (bounded enum/bitset), scoped to the record type:
  - **Worker/daemon capabilities** (daemon endpoint): e.g. `queue_broker_enabled`, `retention_handles_enabled`,
    `capability_tokens_v2_enabled`.
  - **Instance capabilities** (engine/agent endpoint): e.g. `execution_signals_enabled`, `node_agent_enabled`.

### Consistency rules

- Updates are low-frequency (heartbeat cadence is acceptable).
- Capability updates SHOULD be “update-on-change” to avoid write amplification (only write when the bitset differs).
- Directory responses are advisory for routing, not a correctness mechanism.
- Clients MUST cache directory results and MUST NOT query Global Store on every control-plane action (e.g., per enqueue,
  claim, or lease renew). Refresh MUST be bounded by explicit staleness budgets and backoff.
- Correctness MUST be enforced by issuer-scoped capability tokens and fencing (e.g., queue epoch tokens).

---

# Schema Changes (if any)

V1 minimal changes (recommended):

- Extend worker and instance discovery protos in `proto/tensorcast/global_store/v1/global_store.proto` to carry a
  bounded capability bitset (in `WorkerInfo` and `InstanceInfo`) and persist it alongside worker/instance identity in
  `schema.sql` (Global Store).

No high-frequency per-handle state is stored in Global Store.

---

# Trade-offs & Risks

- **More platform surface area**: retention handles and token libraries are shared infrastructure.
  - Mitigation: keep the API small; reuse lifecycle primitives; keep discovery low-frequency.
- **Retention downgrade semantics are new**: today stable retention is effectively upgrade-only.
  - Mitigation: introduce an explicit `RetentionRegistry` layer that owns downgrade logic rather than overloading
    stable cache policy upgrades.
- **Operational debugging**: failures may manifest as “path lost” later.
  - Mitigation: include diagnostics in handles and export metrics for active handles, bytes charged, and expirations.

---

# Compatibility & Acceptance Criteria

## Compatibility

- Does not change data-plane transport protocols.
- Uses unified runtime config; no new ad-hoc env vars.
- Works with fake CUDA backend for tests (as it is control-plane + stable tier bookkeeping).

## Acceptance Criteria

1. **Cross-process retention**
   - A process that is not the original producer can renew and release a retention handle using only the capability
     token, without PID coupling.
   - The retention handle response includes an authoritative `charged_bytes` so brokers/controllers can enforce
     consistent byte accounting without trusting callers.

2. **Release frees resources**
   - When the last handle is released/expired, effective retention intent downgrades and stable DRAM resources become
     reclaimable under memory pressure.

3. **Token consistency**
   - Placement pin tokens and retention handle tokens share the same envelope and validation library.
   - Tokens include audience binding and issuer binding; v1 uses issuer-only validation.
   - During migration, issuers support dual-format verification for a bounded transition window and fail fast on
     audience/type confusion.

4. **Discovery**
   - Clients can discover broker-enabled daemons via Global Store capability flags (no static broker list required for
     correctness).
   - SDK caches directory results with bounded staleness budgets (does not query Global Store per request).

---

# References

- Leases/Guards/Finalizers: `docs/designs/0011-unified-session-lifecycle-leases.md`
- Stable tier budgeting: `docs/designs/0034-stable-memory-tiers.md`
- Pinned memory boundary: `docs/designs/0043-unified-pinned-memory-authority.md`
- Programmable framework (CallContext/Operation/capabilities): `docs/designs/0055-programmable-framework.md`
- Queue design (uses these primitives): `docs/designs/0060-tensor-work-queue.md`
