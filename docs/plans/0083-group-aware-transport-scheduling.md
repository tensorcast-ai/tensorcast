---
slug: group-aware-transport-scheduling
title: Group-Aware Replica Transport Scheduling Plan
status: proposed
areas: ["global_store", "core", "daemon", "proto", "tests", "docs"]
created: 2026-02-25
last_updated: 2026-02-25
related_code:
  - docs/designs/0083-group-aware-transport-scheduling.md
  - proto/tensorcast/global_store/v1/global_store.proto
  - proto/tensorcast/config/v1/global_store_config.proto
  - tensorcast/global_store/rpc/transport_rpc_handler.py
  - tensorcast/global_store/services/transport_service.py
  - tensorcast/global_store/repositories/replica_repository.py
  - tensorcast/global_store/repositories/transport_repository.py
  - core/store/components/global_store_client.h
  - core/store/components/global_store_client.cc
  - core/store/materialization/contracts/loading_spec.h
  - core/store/materialization/control/materialize_orchestrator.cc
  - core/store/runtime/ingestion/materialization_facade.cc
  - docs/benchmarks/20260222-weight-publisher-multihost-p2p-report.md
  - schema.sql
links:
  design: ../designs/0083-group-aware-transport-scheduling.md
---

# Objective

Implement a two-stage scheduler evolution:

1. **Stage A**: source-side balancing and diffusion improvements within current request flow.
2. **Stage B**: global pending-request dispatch with group completion objective.

At the same time, fix transport semantics so group progress is based on explicit success outcomes rather than transport lease closure.

# Current State and Root Causes

Current transport path:

- request entry: `tensorcast/global_store/rpc/transport_rpc_handler.py`
- service polling and claim loop: `tensorcast/global_store/services/transport_service.py`
- source selection and claim: `tensorcast/global_store/repositories/replica_repository.py`
- transport persistence: `tensorcast/global_store/repositories/transport_repository.py`
- C++ call sites: `core/store/materialization/control/materialize_orchestrator.cc`, `core/store/runtime/ingestion/materialization_facade.cc`

Root-cause constraints:

- `complete_replica_transport` is called on both success and failure paths, and stale cleanup force-completes in-progress rows.
- request-local loop cannot enforce cluster-wide fairness/completion goals.
- claim and transport-create are not fully atomic in one transaction.
- transport repository still uses positional `SELECT *` mapping.
- benchmark TP workloads are currently rank-serial per receiver, and one version expects one resolved `artifact_id` across ranks.

# Baseline Before Changes

Record baseline first on 40GB TP4/TP8, then verify at 320GB:

- source top-1 share and source HHI
- per-version publish-to-apply latency p50/p95/p99
- queue wait vs transfer execution split
- group completion spread (max-min, p95)
- diffusion first-hit latency (time from new source exportable to first assignment)
- transport completion outcome distribution (`success/failed/expired/cancelled`)

Baseline method must align with:

- `docs/benchmarks/20260222-weight-publisher-multihost-p2p-report.md`

# Phases and Milestones

- [ ] Phase 0: Prerequisite Hardening (must finish before scheduler changes)
  - [ ] Milestone 0.1: Replace `SELECT *` in `TransportRepository` with explicit projection and name-based mapping.
  - [ ] Milestone 0.2: Make source claim + transport record create atomic in one transaction.
  - [ ] Milestone 0.3: Add request-level idempotency key plumbing (`request_id`) and dedupe behavior.

- [ ] Phase 1: Protocol and Schema Foundation
  - [ ] Milestone 1.1: Add `TransportSchedulingGroup`, `requester_worker_id`, and `request_id` to `RequestReplicaTransportRequest`.
  - [ ] Milestone 1.2: Add completion outcome fields to `CompleteReplicaTransportRequest`.
  - [ ] Milestone 1.3: Extend `artifact_transports` schema with requester/group/outcome columns and indexes.
  - [ ] Milestone 1.4: Regenerate proto code via `bash tools/build_proto_python.sh`.

- [ ] Phase 2: Stage A Source-Balance Scheduler
  - [ ] Milestone 2.1: Implement source score terms (`replica_load`, `worker_load`, `recent_assignment_penalty`, `diffusion_bonus`).
  - [ ] Milestone 2.2: Add deterministic tie-break and bounded query path.
  - [ ] Milestone 2.3: Add metrics for top-1 share, HHI, and diffusion hit latency.
  - [ ] Milestone 2.4: Gate by config mode `SOURCE_BALANCE` with safe fallback to `LEGACY`.

- [ ] Phase 3: Stage A Integration and Validation
  - [ ] Milestone 3.1: Propagate optional scheduling hints through core path (`MaterializeHints` to `GlobalStoreClient`).
  - [ ] Milestone 3.2: Ensure outcome is reported from success/failure/cancel paths in orchestrator/facade.
  - [ ] Milestone 3.3: Validate no correctness regression under Stage A mode.

- [ ] Phase 4: Stage B Dispatcher Foundation
  - [ ] Milestone 4.1: Add `pending_transport_requests` table and repository API.
  - [ ] Milestone 4.2: Change request path to enqueue + dispatch instead of request-local polling.
  - [ ] Milestone 4.3: Implement timeout/cancel/expiration handling on pending requests.
  - [ ] Milestone 4.4: Gate by config mode `GROUP_DISPATCH`.

- [ ] Phase 5: Stage B Group Objective
  - [ ] Milestone 5.1: Enforce v1 group contract (same artifact and byte space in one group epoch).
  - [ ] Milestone 5.2: Compute group progress from completion outcomes (`SUCCESS` only for completed parts).
  - [ ] Milestone 5.3: Implement fairness floor + completion bias + starvation aging policy.

- [ ] Phase 6: Config and Operability
  - [ ] Milestone 6.1: Extend `GlobalStoreConfig` scheduler policy under unified config.
  - [ ] Milestone 6.2: Add scheduler mode/weights/aging/scan limits and strict loader validation.
  - [ ] Milestone 6.3: Update `examples/config/global_store_config.yaml` and related docs.

- [ ] Phase 7: Correctness and Regression Tests
  - [ ] Milestone 7.1: Add repository tests for worker-balance, diffusion, and queue dispatch fairness.
  - [ ] Milestone 7.2: Add service/RPC tests for request idempotency and completion outcome accounting.
  - [ ] Milestone 7.3: Add C++ tests for hint propagation and completion outcome propagation.

- [ ] Phase 8: Benchmark Replay and Acceptance Gates
  - [ ] Milestone 8.1: Re-run 40GB TP4 and tune Stage A.
  - [ ] Milestone 8.2: Re-run 40GB TP8 and tune Stage B.
  - [ ] Milestone 8.3: Run 320GB replay after 40GB gates pass.
  - [ ] Milestone 8.4: Update benchmark report with before/after tables and root-cause narrative.

- [ ] Phase 9: Rollout and Closure
  - [ ] Milestone 9.1: Rollout mode progression: `LEGACY -> SOURCE_BALANCE -> GROUP_DISPATCH`.
  - [ ] Milestone 9.2: Validate mixed-version behavior and fallback.
  - [ ] Milestone 9.3: Mark design/plan status with owner signoff and lock acceptance evidence.

# Test and Validation Matrix

Python tests (Global Store):

- `source .venv/bin/activate && pytest tests/python/global_store/test_repositories.py`
- `source .venv/bin/activate && pytest tests/python/global_store/test_services.py`
- `source .venv/bin/activate && pytest tests/python/global_store/test_grpc_service.py`
- `source .venv/bin/activate && pytest tests/python/global_store/test_transport_scheduler.py`

C++ tests (materialization hint and completion propagation):

- `bazel test //core/store/materialization/control:materialize_orchestrator_reselection_test`
- `bazel test //core/store/runtime/ingestion:materialization_facade_test`
- `bazel test //core/store/runtime/metadata:metadata_gateway_test`

Proto and schema hygiene:

- `bash tools/build_proto_python.sh`
- `bazel test //proto/... --test_output=streamed`

Benchmark validation:

- 40GB TP4/TP8 first for rapid iteration
- 320GB final replay after 40GB stabilization
- always include queue-wait vs transfer split and source-distribution plots

# Quantitative Success Gates

All gates must pass for completion:

- source concentration: top-1 source share reduced by >=20% vs baseline.
- source diversity: HHI reduced by >=25% vs baseline.
- group completion tail: p95 reduced by >=30% in TP4 and TP8 40GB cases.
- diffusion: new exportable source gets first assignment within configured bounded window.
- semantic correctness: non-success completion outcomes never increment `group_completed_parts`.
- no functional regression: key visibility, hash verification, and final artifact correctness unchanged.

# Rollout, Backout, and Risks

Rollout:

1. Merge with scheduler mode defaulting to `LEGACY`.
2. Enable `SOURCE_BALANCE` in staging benchmark lanes.
3. Enable `GROUP_DISPATCH` after Stage A gates pass and outcome reporting is validated.
4. Promote to larger replay only after 40GB gates hold stable.

Backout:

- Set scheduler mode back to `LEGACY`.
- Keep additive proto/schema fields; old clients continue to work.
- Keep dispatcher code path disabled when mode is not `GROUP_DISPATCH`.

Primary risks and mitigations:

- fairness bias misconfiguration: mitigate with bounded aging and canary metrics.
- DB overhead from queue scans: mitigate with explicit indexes and scan limits.
- mixed-version partial semantics: mitigate with mode gating and compatibility counters.
- behavior drift between modes: mitigate with dual-run validation on same replay inputs.

# Owner Checklist

- [ ] `schema.sql` updated with synchronized migration and indexes.
- [ ] proto code regenerated and committed.
- [ ] unified config proto and loader updates landed (no env-only knobs).
- [ ] benchmark report updated with quantitative evidence and mode labels.
- [ ] docs links and examples updated in same change set.
