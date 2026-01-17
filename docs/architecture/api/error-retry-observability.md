---
title: Error, Retry, Observability
description: Error model, retry policy, and key metrics
---

# Error, Retry, Observability

This document consolidates error semantics, retry behavior, and observability
signals for the SDK and daemon.

Related docs:

- Public contracts and where errors surface: [API Design](./api-design.md#contracts-and-invariants)
- Source selection and disk verification knobs: [Materialization Flow](./materialization-flow.md#fallback-and-source-preference)
- Persistence degradation semantics: [Policy & Persistence](./policy-persistence.md#task-states-and-degradation)

## Error Model

The SDK raises `ArtifactError` with:

- `status_code` aligned to gRPC canonical codes
- `retryable` hint for callers

Why both fields exist:

- `status_code` is a stable, interoperable taxonomy (matches gRPC canonical codes).
- `retryable` captures SDK knowledge about which errors are expected to clear
  with time (e.g. transient daemon unavailability) without callers needing to
  re-encode complex rules.

Type definitions:

- `ArtifactError`: [tensorcast/api/store/types.py](../../../tensorcast/api/store/types.py)
- Error mapping: [tensorcast/api/store/retry.py](../../../tensorcast/api/store/retry.py)

Common mappings:

- `INVALID_ARGUMENT` for input validation
- `FAILED_PRECONDITION` for state mismatches
- `NOT_FOUND` for missing artifacts or disk paths
- `RESOURCE_EXHAUSTED` for capacity issues
- `UNAVAILABLE` and `DEADLINE_EXCEEDED` for transient failures
- `DATA_LOSS` for integrity mismatches

### Caller guidance (typical handling patterns)

- `INVALID_ARGUMENT`, `FAILED_PRECONDITION`: fix the caller inputs or state; do not retry blindly.
- `NOT_FOUND`: treat as a missing artifact/key/path; retry only if you expect eventual consistency.
- `RESOURCE_EXHAUSTED`: retry may help (evictions/capacity changes), but also consider adjusting policy.
- `UNAVAILABLE`, `DEADLINE_EXCEEDED`, `ABORTED`: safe to retry within a deadline.
- `DATA_LOSS`: indicates corruption or verification failure; retrying may not help without changing source.

## Retry Policy

The Store runtime applies bounded retries for transient errors. Defaults are
implemented in `tensorcast/api/store/retry.py` and can be overridden via
`StoreOptions.retry_overrides`.

Why retries are bounded:

- Unbounded retries hide outages and create thundering herds.
- Deadlines enforce an upper bound on caller-perceived latency.
- Backoff+jitter reduces correlated retries across workers.

Default policy summary:

- `register`: 30s deadline, 2 attempts
- `put`: 45s deadline, 2 attempts
- `get`: 40s deadline, 3 attempts
- `get_into`: 40s deadline, 3 attempts

The SDK retries only when **all** of the following are true:

- the error is an `ArtifactError` with `retryable=True`
- `status_code` is one of `UNAVAILABLE`, `DEADLINE_EXCEEDED`, `ABORTED`
- the attempt count and overall deadline budget allow another attempt

## Observability Signals

SDK metrics:

- `tc_store_artifact_cache_hits_total`
- `tc_store_artifact_cache_misses_total`
- `tc_store_artifact_cache_evictions_total`
- `tc_store_artifact_cache_invalidations_total`
- `tc_store_operation_latency_seconds`
- `tc_store_operation_errors_total`
- `tc_store_operation_retries_total`
- `tc_store_batch_hits_total`, `tc_store_batch_coalesced_total`, `tc_store_batch_window_seconds`
- `tc_store_prefetch_events_total`
- `tc_store_region_backed_fallback_total`, `tc_store_region_backed_verification_skipped_total`

Daemon metrics:

- `tc_local_stable_tier_total{op,status,requirement}`
- `tc_local_stable_tier_seconds{op,status}`
- `tc_persist_tasks_active{state}`
- `tc_persist_errors_total{stage,reason}`
- `tc_persist_progress_ratio`

Logs and traces:

- SDK fallback logs include `tc.store.mode`, `tc.store.artifact_id`, and
  `tc.store.key`.
- Persistence logs include `task_id`, `plan_id`, `artifact_id`, and degraded
  reasons.

### What to watch (practical signals)

- Rising `tc_store_operation_errors_total{status="UNAVAILABLE"}` usually indicates daemon connectivity / overload.
- Sustained `tc_persist_tasks_active{state="running"}` with flat `tc_persist_progress_ratio` suggests persistence stalls.
- Increased `tc_store_region_backed_fallback_total{reason="layout_mismatch"}` indicates callers are attempting region-backed `get_into` with non-canonical targets.
- `tc_local_stable_tier_total{status="DEGRADED"}` spikes indicate local stable capacity pressure; correlate with policy usage and `overflow_policy`.

## Code Map

- Error mapping: [tensorcast/api/store/retry.py](../../../tensorcast/api/store/retry.py)
- SDK metrics: [tensorcast/api/_metrics.py](../../../tensorcast/api/_metrics.py)
- Daemon local-stable metrics: [daemon/service/controllers/registration_controller.cc](../../../daemon/service/controllers/registration_controller.cc)
- Daemon materialization metrics: [daemon/service/controllers/materialization_controller.cc](../../../daemon/service/controllers/materialization_controller.cc)
- Persistence metrics: [daemon/state/persistence_manager.cc](../../../daemon/state/persistence_manager.cc)
