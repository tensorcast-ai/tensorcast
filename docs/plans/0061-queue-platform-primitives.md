---
slug: queue-platform-primitives
title: Plan - Queue Platform Primitives (Retention Handles + Capability Tokens + Capability Directory)
links:
  design: ../designs/0061-queue-platform-primitives.md
areas: ["daemon", "core", "global_store", "sdk", "proto"]
related_code:
  - docs/designs/0061-queue-platform-primitives.md
  - docs/designs/0011-unified-session-lifecycle-leases.md
  - docs/designs/0034-stable-memory-tiers.md
  - docs/designs/0055-programmable-framework.md
  - core/store/components/stable_dram_cache_manager.h
  - daemon/state/session_lifecycle.h
  - proto/tensorcast/global_store/v1/global_store.proto
  - schema.sql
  - tensorcast/global_store/README.md
---

# Objective

Deliver the platform primitives required by `0060-tensor-work-queue`:

- daemon-issued, capability-tokenized **RetentionHandles** with renew + release (downgrade on last release),
- a unified **Capability Token envelope** used across subsystems,
- Global Store **capability discovery** so clients can find broker-enabled daemons without static broker lists.

Note: selection identity unification is already complete and is not part of this plan.

# Current State & Grounding

- Stable DRAM retention exists but is effectively “upgrade-only” (`StableDramCacheManager`) and does not expose a
  releasable handle abstraction.
- Lease/Guard/Finalizer exists in the daemon (`SessionLifecycleManager`) and is the correct place to express expirations
  and idempotent cleanup.
- Capability tokens exist for some flows (e.g., placement pins), but token shape/validation is not yet a single shared
  platform library.
- Global Store already runs a worker/instance registry with batched heartbeats and is the correct low-frequency
  directory for capability flags (but capability writes must be update-on-change to avoid write amplification).

# Phases & Milestones

- [x] Phase 0: Spec closure + proto shapes
  - [x] Milestone 0.1: Define `RetentionHandle` proto (include `charged_bytes`) + normative status/error taxonomy (acquire/renew/release) + “control-plane only” semantics
  - [x] Milestone 0.2: Define versioned capability-token envelope (issuer/audience/scope=deterministic proto/expiry/fencing) + key rotation + migration contract (dual-format verification)
  - [x] Milestone 0.3: Extend Global Store discovery protos with bounded capability flags (worker vs instance) + update-on-change write rules

- [x] Phase 1: Daemon/core retention implementation
  - [x] Milestone 1.1: Implement daemon-local `RetentionRegistry` with refcount + expiry aggregation
  - [x] Milestone 1.2: Wire handle expiry and release via `SessionLifecycleManager` finalizers
  - [x] Milestone 1.3: Implement downgrade-on-last-release semantics for stable DRAM retention

- [x] Phase 2: Token library consolidation
  - [x] Milestone 2.1: Implement shared token encode/verify library (C++ + Python)
  - [x] Milestone 2.2: Migrate placement pin tokens to the shared envelope
  - [x] Milestone 2.3: Use the shared envelope for retention-handle tokens
  - [x] Milestone 2.4: Add key rotation support (active + bounded previous keys) and compatibility tests (including dual-format verification during migration)

- [x] Phase 3: Global Store capability directory
  - [x] Milestone 3.1: Persist capability flags in schema and plumb through worker/instance heartbeats
  - [x] Milestone 3.2: Expose capability-filtered listing for clients/controllers
  - [x] Milestone 3.3: Define staleness budgets and cache semantics for directory clients (SDK/controllers)
  - [x] Milestone 3.4: Ensure capability writes are update-on-change (no per-heartbeat rewrites when unchanged)

- [x] Phase 4: SDK surface + tests
  - [x] Milestone 4.1: Add Python SDK helpers for acquire/renew/release retention handles
  - [x] Milestone 4.2: Add C++ unit tests for handle expiry/release/downgrade behavior
  - [x] Milestone 4.3: Add integration tests for discovery + token validation (fake CUDA backend)
  - [x] Milestone 4.4: Add tests for retention error classification (retryable vs non-retryable) and token expiry

# Tasks

- [x] Protos
  - [x] Add `proto/tensorcast/retention/v1/retention.proto` (or daemon service extension)
  - [x] Add `proto/tensorcast/common/v1/capability_token.proto` (canonical `CapabilityTokenEnvelope`; shared across subsystems)
  - [x] Add unified-config fields for capability token keys/rotation (daemon config)
  - [x] Add unified-config fields for retention handle TTL caps (issuer-side clamp)
  - [x] Add capability flags to `proto/tensorcast/global_store/v1/global_store.proto` (worker + instance scoped)
  - [x] Persist capability flags in `schema.sql` (bounded bitset; update-on-change)
  - [x] Regenerate Python stubs via `bash tools/build_proto_python.sh`

- [x] Daemon/Core
  - [x] Add `RetentionRegistry` module (daemon-local)
  - [x] Integrate with `StoreEngine::admit_stable_cache_policy(...)` and stable tier budgeting
  - [x] Integrate handle lifecycle with `SessionLifecycleManager`

- [x] Global Store
  - [x] Persist capability flags and expose via listing RPCs
  - [x] Add metrics for capability adoption (bounded labels)

- [x] SDK
  - [x] Add `tensorcast/retention/` module with typed APIs and CallContext plumbing
  - [x] Add a capability-directory client helper with bounded staleness + backoff (no per-request Global Store reads)

# Test / Rollout / Backout

- Tests
  - Python: `TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python/...`
  - C++: `bazel test //daemon:... --test_env=TENSORCAST_CUDA_BACKEND=fake`

- Rollout
  - Enable capability directory and retention handles behind config flags; validate on a single node first.

- Backout
  - Disable retention-handle usage and fall back to best-effort TTL-only stable retention (no cross-process renew).

# Risks & Tracking

- Risk: downgrade-on-last-release could regress existing stable retention behavior
  - Mitigation: make downgrade logic strictly opt-in for handle-managed payloads at first; add unit tests.
- Risk: token envelope change breaks existing placement pin clients
  - Mitigation: support dual-format verification during a transition window (versioned tokens).
