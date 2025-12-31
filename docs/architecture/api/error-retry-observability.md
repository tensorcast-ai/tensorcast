---
title: Error, Retry, Observability
description: Error model, retry policy, and key metrics
---

# Error, Retry, Observability

This document consolidates error semantics, retry behavior, and observability
signals for the SDK and daemon.

## Error Model

The SDK raises `ArtifactError` with:

- `status_code` aligned to gRPC canonical codes
- `retryable` hint for callers

Common mappings:

- `INVALID_ARGUMENT` for input validation
- `FAILED_PRECONDITION` for state mismatches
- `NOT_FOUND` for missing artifacts or disk paths
- `RESOURCE_EXHAUSTED` for capacity issues
- `UNAVAILABLE` and `DEADLINE_EXCEEDED` for transient failures
- `DATA_LOSS` for integrity mismatches

## Retry Policy

The Store runtime applies bounded retries for transient errors. Defaults are
implemented in `tensorcast/api/store/retry.py` and can be overridden via
`StoreOptions.retry_overrides`.

Default policy summary:

- `register`: 30s deadline, 2 attempts
- `put`: 45s deadline, 2 attempts
- `get`: 40s deadline, 3 attempts
- `get_into`: 40s deadline, 3 attempts

## Observability Signals

SDK metrics:

- `tc_store_artifact_cache_hits_total`
- `tc_store_artifact_cache_misses_total`
- `tc_store_artifact_cache_evictions_total`
- `tc_store_artifact_cache_invalidations_total`
- `tc_store_operation_latency` and retry counters

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

## Code Map

- Error mapping: `../../../tensorcast/api/store/retry.py`
- SDK metrics: `../../../tensorcast/api/_metrics.py`
- Daemon metrics: `../../../daemon/service/controllers/registration_controller.cc`
- Persistence metrics: `../../../daemon/persistence_manager.cc`
