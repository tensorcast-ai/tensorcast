---
slug: tensor-work-queue
title: Plan - Distributed Tensor Work Queue (Descriptor-Only)
links:
  design: ../designs/0060-tensor-work-queue.md
areas: ["daemon", "core", "global_store", "sdk", "proto"]
related_code:
  - docs/designs/0060-tensor-work-queue.md
  - docs/designs/0055-programmable-framework.md
  - docs/designs/0011-unified-session-lifecycle-leases.md
  - docs/designs/0016-artifact-view-v1.md
  - docs/designs/0034-stable-memory-tiers.md
  - proto/tensorcast/plan/v1/plan.proto
  - proto/tensorcast/node_agent/v1/node_agent.proto
  - core/store/memory_tier_budget.h
  - core/store/components/stable_dram_cache_manager.h
  - daemon/state/session_lifecycle.h
  - tensorcast/api/store/artifact.py
  - tensorcast/api/operation.py
  - schema.sql
---

# Objective

Implement the descriptor-only, at-least-once MPMC Tensor Work Queue defined in `docs/designs/0060-tensor-work-queue.md`,
including retained-bytes limits, consumer credit gating, and an MVP upstream offload path (stage to stable DRAM on a
producer-local daemon via CUDA IPC + retention handles), with explicit fencing and retry-safe idempotency aligned with
`0055-programmable-framework`.

V1 baseline (implementation contract):

- V1 broker is **L0 (ephemeral)** and **single-partition per queue**. After broker leader restart, queue state MAY be
  lost; at-least-once is scoped to leader liveness.
- Queue identity is `QueueId = (queue_namespace, queue_name)`. Leadership fencing uses a deterministic
  `queue_operation_id = "queue:v1:" + UUID5("tensorcast.queue.v1", HEX(DETERMINISTIC_PROTO(QueueRef{queue_namespace, queue_name})))`.
- Queue leadership lease principal mapping (Global Store `AcquireOperationLease`):
  - `operation_id = queue_operation_id`
  - `kind = "queue_leader_v1"`
  - `target_artifact_id = queue_operation_id`
- Enqueue idempotency: when `idempotency_key` is provided (SDK: `CallContext.idempotency_key`), derive deterministic
  `work_id = "work:v1:" + UUID5("tensorcast.work.v1", HEX(DETERMINISTIC_PROTO(WorkIdDerivationInput{...})))`; otherwise generate a random `work_id`
  (enqueue retries may create duplicates).
- Queue RPC messages do not embed `CallContext`; SDK maps `ctx.deadline_ms` → RPC deadline and `ctx.idempotency_key` →
  `EnqueueWorkRequest.idempotency_key`.
- ClaimPolicy is consumer-side (ClaimWork waiting behavior), not a producer-specified EnqueueWork attribute.
- V1 supports only:
  - `payload_origin=queue_staged + retention_mode=handle` (primary upstream-offload path; broker validates + renews RetentionHandle; uses `charged_bytes`)
    - v1 restriction: queues that accept `payload_origin=queue_staged` MUST configure exactly one consumer group.
  - `payload_origin=external + retention_mode=policy_only`
    - v1: broker does not infer "path lost"; terminal failure is attempts-exceeded (or consumer terminal failure in v2+).
- V1 does not ship `UpdateCredit` / external "credit grants" or any broker-exposed `RenewRetain` surface; credit is a
  broker-owned per-(queue,group) ledger.

# Current State & Grounding

Existing primitives we build on:

- **Leases and lifecycle**: `daemon/state/session_lifecycle.h` (`0011-unified-session-lifecycle-leases`)
- **Stable tier budgeting**: `core/store/memory_tier_budget.h` (`0034-stable-memory-tiers`)
- **Stable DRAM retention manager**: `core/store/components/stable_dram_cache_manager.h`
- **Materialization pipeline**: `core/store/runtime/ingestion/materialization_service.h`,
  `core/store/materialization/control/materialize_orchestrator.h`
- **Programmable control-plane primitives**: `tensorcast/api/operation.py` and `docs/designs/0055-programmable-framework.md`
- **Views for partial pull**: `docs/designs/0016-artifact-view-v1.md`
- **Node-local plan execution** (recommended for `stage_and_enqueue` when deployed): `proto/tensorcast/plan/v1/plan.proto`,
  `proto/tensorcast/node_agent/v1/node_agent.proto` (`0055-programmable-framework`)
- **Global Store operation leases**: `proto/tensorcast/operation/v1/operation.proto` (Acquire/Keepalive/GetOperation)
- **Capability directory and worker identity**: `proto/tensorcast/global_store/v1/global_store.proto` (worker registry + capability flags)
- **Capability token envelope**: `proto/tensorcast/common/v1/capability_token.proto` (audience + issuer + queue epoch fencing)
- **Retention handles**: `proto/tensorcast/daemon/v2/store_daemon.proto` and `docs/designs/0034-stable-memory-tiers.md`
  (`AcquireRetentionHandle` / `RenewRetentionHandle` / `ReleaseRetentionHandle`)

Constraints:

- Python entry points must use `uv run ...` (repo rule).
- C++ build/test via Bazel; CUDA backend selection via `TENSORCAST_CUDA_BACKEND`.
- Config changes must use unified Protobuf config (`docs/designs/0004-unified-runtime-config.md`); no env var knobs.
- Global Store (v1): no new durable per-item queue state; reuse existing GetOperation + worker registry + capability flags.

# Proposed Code Layout (v1)

This is a suggested layout to keep the implementation modular and testable (adjust as needed, but keep the boundaries):

- Protos:
  - `proto/tensorcast/queue/v1/work_queue.proto` (WorkQueueService + messages)
- Daemon (C++):
  - `daemon/queue/queue_broker_runtime.*` (in-memory state machine + scheduler + quotas)
  - `daemon/queue/work_queue_service_impl.*` (gRPC adapter; leader-only behavior and error mapping)
  - `daemon/app/daemon_app.cc` (register additional gRPC service when enabled)
- Config:
  - `proto/tensorcast/config/v1/daemon_config.proto` (`DaemonConfig.queue_broker`)
  - `proto/tensorcast/config/v1/client_config.proto` (`ClientConfig.queue`)
- SDK (Python):
  - `tensorcast/queue/__init__.py`
  - `tensorcast/queue/client.py` (RPC client wrapper + leader resolver + caching)
  - `tensorcast/queue/api.py` (WorkQueue/Producer/Consumer/WorkItem)
  - `tensorcast/queue/stage.py` (`stage_and_enqueue` helper)
- Tests:
  - Python: `tests/python/queue/test_*.py`
  - C++: `bazel test //daemon/queue:...`

# Phases & Milestones

- [ ] Phase 0: Interfaces + config + SDK skeleton (L0)
  - [ ] Milestone 0.1: Lock platform conventions (OperationLease field mapping, deterministic UUID5 ids, unix-epoch ms time semantics, SDK CallContext mapping, token field naming, PayloadOrigin semantics)
  - [ ] Milestone 0.2: Define `tensorcast.queue.v1` protos (QueueRef + PayloadOrigin + Enqueue/Claim/Ack/Nack/Extend/Stats) and wire codegen
  - [ ] Milestone 0.3: Add unified config schema for the queue broker and client queue section (QueueId, per-queue limits, lease tuning, v1 queue-staged single-group restriction)
  - [ ] Milestone 0.4: Leader fencing + self-fencing via Global Store `operation.proto` leases (low-frequency)
  - [ ] Milestone 0.5: Leader routing contract (GetOperation + worker registry) + SDK caching + optional redirect hints
  - [ ] Milestone 0.6: Broker discovery via capability directory with bounded staleness + backoff (no per-request Global Store reads)
  - [ ] Milestone 0.7: Python SDK stubs with SDK-local `CallContext` mapping and basic ergonomics (`work_queue`, producer/consumer)
  - [ ] Milestone 0.8: Spec hardening checklist complete (v1 failure matrix, ClaimPolicy scope, supported combinations, queue-staged single-group restriction, policy-only terminal semantics, authoritative size derivation + caching, credit ledger rules)

- [ ] Phase 1: L0 broker runtime core (in-memory)
  - [ ] Milestone 1.1: State machine per `(queue, group)` with `DEADLETTER` terminal state
  - [ ] Milestone 1.2: Visibility timeout + ack/nack/extend idempotency + stale token rejection (epoch fencing)
  - [ ] Milestone 1.3: Per-key gating (`max_inflight_per_key`) and bounded waits (no shared-executor blocking)
  - [ ] Milestone 1.4: Self-fenced mode behavior (fail-closed, no token mint/accept, no queue-owned side effects)
  - [ ] Milestone 1.5: L0 safety limits (max_pending_items / max_pending_descriptor_bytes) to bound broker memory

- [ ] Phase 2: Resource contracts + retry safety (retain + credit)
  - [ ] Milestone 2.1: Retained-bytes accounting (`retention_mode=handle`; v1 queue-scoped) and limits
  - [ ] Milestone 2.2: Per-group `credit_limit_bytes` caps + reservation/release rules on claim/ack/nack/timeout (group ledger)
  - [ ] Milestone 2.3: Authoritative `size_bytes` derivation so quotas cannot be bypassed (RetentionHandle charged_bytes and metadata-only planning + caching)
  - [ ] Milestone 2.4: Enforce v1 supported combinations (`queue_staged+handle`, `external+policy_only`) and reject others fast (`FAILED_PRECONDITION`)
  - [ ] Milestone 2.5: RetainLease renew/expiry loop with retryable vs non-retryable classification (RetentionHandle taxonomy)
  - [ ] Milestone 2.6: Progress under admission constraints (no head-of-line deadlock) + `tc_queue_unprocessable_total` metric

- [ ] Phase 3: Upstream offload + queue-staged cleanup (v1 primary UX)
  - [ ] Milestone 3.1: `stage_and_enqueue` (SDK composition, optionally via Node Agent Plan step) with retry safety (`idempotency_key`) and bounded orphans
  - [ ] Milestone 3.2: Queue-staged payload cleanup finalizer (release RetentionHandle + best-effort local unload; no global deletion)
  - [ ] Milestone 3.3: Optional v2+: durability upgrade/persist path (requires a durable reclamation model; not in v1)

- [ ] Phase 4: Observability + tests
  - [ ] Milestone 4.1: Metrics (depth/inflight/attempt/credit/retained bytes) + traces with stable low-cardinality attributes
  - [ ] Milestone 4.2: Python tests for queue semantics (fake CUDA backend)
  - [ ] Milestone 4.3: C++ unit tests for broker state machine, fencing, and lease expiry

- [ ] Phase 5: L1 recovery + directory hardening (optional)
  - [ ] Milestone 5.1: Local WAL/KV persistence for broker state and restart recovery tests
  - [ ] Milestone 5.2: Add integration tests for directory caching + leader transition retries (staleness budgets, backoff)
  - [ ] Milestone 5.3: Optional queue-specific Global Store directory table (v2+) if capability directory is insufficient

# Tasks

- [ ] Protos
  - [ ] Define `tensorcast.queue.v1` protos (service + messages) under `proto/tensorcast/queue/v1/`
  - [ ] Ensure proto shape matches design conventions (no embedded `CallContext`; `PayloadOrigin` enum; `EnqueueWorkRequest.idempotency_key`; `retention_handle_token`; unix-epoch `*_expires_at_ms`)
  - [ ] Ensure WorkLease tokens use the existing unified capability token envelope and audience
    `CAPABILITY_AUDIENCE_QUEUE_WORK_LEASE` (`proto/tensorcast/common/v1/capability_token.proto`)
  - [ ] Generate Python stubs via `bash tools/build_proto_python.sh`

- [ ] Daemon (C++)
  - [ ] Add broker runtime module (state machine, timers, quotas)
  - [ ] Expose `tensorcast.queue.v1.WorkQueueService` on broker-enabled daemons (leader-only; followers return `UNAVAILABLE`
    and may attach leader hints)
  - [ ] Ensure dedicated control-plane executor to avoid starvation
  - [ ] Implement QueueId (`queue_namespace`, `queue_name`) plumbing and deterministic `queue_operation_id` derivation
    (`"queue:v1:" + UUID5("tensorcast.queue.v1", HEX(DETERMINISTIC_PROTO(QueueRef)))`)
  - [ ] Implement queue epoch fencing via Global Store operation leases (Acquire/Keepalive/Release; use `kind="queue_leader_v1"` and `target_artifact_id=queue_operation_id`)
  - [ ] Mint and validate WorkLease capability tokens (daemon-scoped issuer, audience-bound, queue-epoch fenced)
  - [ ] Implement canonical work descriptor derivation (canonicalize selection + derive authoritative `size_bytes`)
    - [ ] `retention_mode=handle`: validate token with issuer (renew) and use `charged_bytes`
    - [ ] `retention_mode=policy_only`: metadata-only planning (index + view planner) + caching
  - [ ] Implement deterministic `work_id` derivation for idempotent enqueue
    (`"work:v1:" + UUID5("tensorcast.work.v1", HEX(DETERMINISTIC_PROTO(WorkIdDerivationInput{...})))` when `idempotency_key` is provided)
  - [ ] Implement enqueue idempotency conflict detection (persist canonical descriptor fingerprint per `work_id`)
  - [ ] Enforce v1 supported combinations (`queue_staged+handle`, `external+policy_only`) and reject others (`FAILED_PRECONDITION`);
    validate the v1 queue-staged single-group restriction
  - [ ] Implement self-fenced mode gating (no token mint/accept; no queue-owned side effects)
  - [ ] Implement RetainLease renew/expiry loop using the RetentionHandle error taxonomy
  - [ ] Implement issuer routing for RetentionHandles (parse capability token envelope header to route by `issuer_daemon_id`)
  - [ ] Enforce progress rules (bounded scan-ahead via `scan_limit_items`) and emit `unprocessable` metrics on enqueue rejection
  - [ ] Enforce L0 safety limits (`max_pending_items`, `max_pending_descriptor_bytes`)
  - [ ] Implement `GetQueueStats` and core metrics/tracing hooks (depth/inflight/credit/retained bytes, leader fenced/epoch)
  - [ ] Implement v1 dead-letter reason taxonomy and counters (`ATTEMPTS_EXCEEDED`, `PATH_LOST`, `UNPROCESSABLE`, `INTERNAL`)
  - [ ] Implement queue-staged payload cleanup finalizer (release RetentionHandle + best-effort local unload; no global deletion)

- [ ] SDK (Python)
  - [ ] Add `tensorcast/queue/` module with `WorkQueue`, `QueueProducer`, `QueueConsumer`, `WorkItem`, `ClaimPolicy`
  - [ ] Align Python API defaults with v1: `enqueue(artifact=...)` defaults to `external+policy_only`;
    `stage_and_enqueue(...)` produces `queue_staged+handle`
  - [ ] Keep `CallContext` SDK-local: map `ctx.deadline_ms` to gRPC deadline and `ctx.idempotency_key` to `EnqueueWorkRequest.idempotency_key` (do not add CallContext fields to protos)
  - [ ] Implement leader resolver (QueueId -> queue_operation_id -> GetOperation + worker registry) with bounded staleness + refresh triggers
  - [ ] Implement capability-directory caching + refresh triggers (`UNAVAILABLE`/`FAILED_PRECONDITION`) with backoff
  - [ ] Implement `stage_and_enqueue` as SDK composition (`put` + acquire retention handle + enqueue), with an optional
    Node Agent Plan execution path when deployed
  - [ ] Ensure client-side retry guidance matches v1 L0 semantics (leader loss may drop state; attempt resets are allowed)

- [ ] Config
  - [ ] Extend `proto/tensorcast/config/v1/{daemon_config,client_config}.proto` for queue sections
  - [ ] Add per-queue `QueueConfig` to queue broker config (`queue_namespace`, `queue_name`, consumer groups, and v1 limits)
  - [ ] Add v1 scheduler/backlog limits (`visibility_timeout`, `max_inflight_per_key`, `scan_limit_items`, `max_pending_items`, `max_pending_descriptor_bytes`)
  - [ ] Add leader lease tuning (`leader_lease_ttl`, `leader_keepalive_interval`) and client directory cache TTL
  - [ ] Add per-queue `retained_bytes_limit` and `deadletter_max_attempts`; apply `retained_bytes_limit` only to `retention_mode=handle`
  - [ ] Enforce v1 consumer-group constraints (append-only groups; queue-staged queues require exactly one group)
  - [ ] Add example config snippets under `examples/config/` (store daemon + global store + node agent + client)

# Acceptance Checks

- [ ] QueueId is end-to-end: `queue_namespace` + `queue_name` are present in protos, config, SDK, metrics labels, and id derivations.
- [ ] `queue_operation_id` derivation is deterministic and identical across languages (`"queue:v1:" + UUID5("tensorcast.queue.v1", HEX(DETERMINISTIC_PROTO(QueueRef)))`).
- [ ] Queue leadership uses Global Store operation leases with the standardized fields (`operation_id=queue_operation_id`, `kind="queue_leader_v1"`, `target_artifact_id=queue_operation_id`).
- [ ] `work_id` is treated as an opaque string; when `idempotency_key` is provided (SDK: `ctx.idempotency_key`) it is deterministic
  (`"work:v1:" + UUID5("tensorcast.work.v1", HEX(DETERMINISTIC_PROTO(WorkIdDerivationInput{...})))`).
- [ ] Enqueue is idempotent under `idempotency_key`; enqueue conflicts fail `FAILED_PRECONDITION` without mutating the existing item.
- [ ] Queue protos do not embed `CallContext`; SDK maps `CallContext` to RPC deadlines and the enqueue idempotency field.
- [ ] WorkLease tokens are audience-bound and epoch-fenced; stale tokens after leader change fail `FAILED_PRECONDITION`.
- [ ] ClaimPolicy is a ClaimWork parameter (waiting behavior) and is not persisted as an EnqueueWork attribute.
- [ ] Credit ledger is enforced per `(queue,group)` and never exceeds `credit_limit_bytes`; `defer` claims still reserve credit.
- [ ] Head-of-line non-admissible items do not deadlock progress (bounded scan-ahead via `scan_limit_items`).
- [ ] L0 broker state is bounded by `max_pending_items` / `max_pending_descriptor_bytes` and rejects overload with `RESOURCE_EXHAUSTED`.
- [ ] `queue_staged+handle`: broker validates RetentionHandle on enqueue and uses issuer-returned `charged_bytes` as authoritative size; v1 enforces the queue-staged single-group restriction.
- [ ] Retention renew is issuer-routed by `issuer_daemon_id`; expiry/non-retryable issuer responses trigger `PATH_LOST` dead-lettering.
- [ ] No bytes ever traverse broker RPCs (descriptor-only).

# Test / Rollout / Backout

- Tests
  - Python: `TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python/...`
    - [ ] Enqueue idempotency + conflict detection
    - [ ] Claim/ack/nack/extend + visibility timeout redelivery (`attempt` increments while leader alive)
    - [ ] Credit enforcement + `defer` still reserves credit
    - [ ] Progress under head-of-line non-admissible items (scan-ahead)
    - [ ] L0 backlog limits (`max_pending_items`, `max_pending_descriptor_bytes`)
    - [ ] `stage_and_enqueue` retry safety (bounded orphans via retention TTL)
  - C++: `bazel test //daemon:... --test_env=TENSORCAST_CUDA_BACKEND=fake`
    - [ ] Token validation (audience + epoch fencing)
    - [ ] Self-fencing behavior (`UNAVAILABLE` and no side effects while fenced)
    - [ ] Retention renew loop error taxonomy + PATH_LOST dead-letter
    - [ ] Scheduler scan-ahead and per-key gating invariants

- Rollout
  - Start with broker-enabled daemon subset and a single queue for one workload.
  - If using `stage_and_enqueue`, ensure the upload daemon is node-local to producers (CUDA IPC staging is node-local).
  - Set conservative retained/credit limits; observe metrics and tune.
  - Required v1 prerequisites (config):
    - `DaemonConfig.capability_tokens` configured (WorkLease tokens are daemon-issued)
    - `DaemonConfig.retention_handles.enabled=true` on producer-local daemons used for staging
    - `DaemonConfig.capability_directory.enabled=true` so clients can discover broker-enabled daemons by capability flags
    - For `queue_staged+handle` queues: exactly one configured consumer group (v1 restriction)
    - RetentionHandle TTL caps sized to tolerate transient Global Store outages (see design guidance)

- Backout
  - Disable `DaemonConfig.queue_broker.enabled` and fall back to direct artifact-based transfer patterns.
  - In v1 L0, there is no durable broker state to migrate; backout is a restart + config change.

# Risks & Tracking

- Risk: control-plane starvation under heavy ingestion
  - Mitigation: dedicated control-plane threads + bounded waits, alert on queue RPC latency.
- Risk: retain lease leaks pin stable memory
  - Mitigation: lease finalizers, leak counters, periodic audits, deadletter on non-renewable retention.
- Risk: split-brain queue leadership yields inconsistent acks
  - Mitigation: queue epoch fencing + self-fencing via Global Store operation leases; reject stale tokens.
- Risk: Global Store outage triggers fencing and stops retention renewals, causing PATH_LOST dead-lettering for short TTLs
  - Mitigation: conservative TTL sizing (handle TTL >> leader lease keepalive budget) and explicit operator guidance/metrics.

# Owner Checklist

- [ ] Protos regenerated (`bash tools/build_proto_python.sh`) and Bazel proto deps updated as needed.
- [ ] Example configs updated under `examples/config/` and validated against unified config loader strictness.
- [ ] Metrics/tracing reviewed for cardinality (labels limited to QueueId+group; no per-work-id labels).
- [ ] Tests added for both Python and C++ paths; fake CUDA backend coverage included.
