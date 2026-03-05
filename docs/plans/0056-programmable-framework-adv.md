---
slug: 0056-programmable-framework-adv
title: Plan - Programmable Framework Advanced Runtime and Control Plane
status: proposed
areas: ["sdk", "daemon", "global_store", "proto", "integrations", "docs"]
created: 2026-03-04
last_updated: 2026-03-04
related_code:
  - docs/designs/0056-programmable-framework-adv.md
  - docs/designs/0084-unified-artifact-binding-kv-runtime.md
  - docs/designs/0055-programmable-framework.md
  - tensorcast/api/plan/plan.py
  - tensorcast/api/runtime.py
  - tensorcast/api/signals.py
  - tensorcast/node_agent/executor.py
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/plan/v1/plan.proto
  - proto/tensorcast/node_agent/v1/node_agent.proto
  - proto/tensorcast/global_store/v1/global_store.proto
links:
  design: ../designs/0056-programmable-framework-adv.md
---

# Objective

Deliver the advanced programmable framework control plane on top of `0055`:

- daemon-run plan ingress without semantic drift from local runner,
- daemon-served signals and directory snapshots with explicit staleness bounds,
- lane/policy propagation across plan and non-plan RPCs,
- cache blob routing and shard-fencing control flow only (cache blob data model and IO primitives are implemented via `0084`).

# Current State and Grounding

- `0055` already defines plan/spec semantics and local PlanExecutor behavior.
- `0056` design includes runtime, ingress, signals, and KV sections but lacks a paired execution plan.
- Cache-blob data model rules are defined in `0084`; `0056` remains focused on control-plane orchestration and routing.

# Phases and Milestones

- [ ] Phase 0: Scope split and contract alignment
  - [ ] Milestone 0.1: update `0056` to reference `0084` for KV data-path semantics.
  - [ ] Milestone 0.2: preserve one execution semantics between local and daemon-run PlanExecutor.
  - [ ] Milestone 0.3: freeze ingress and signals API boundaries.

- [ ] Phase 1: Runtime and ingress
  - [ ] Milestone 1.1: land process runtime (`connect/runtime`) and single entry daemon wiring.
  - [ ] Milestone 1.2: add daemon ingress `ExecutePlan` with deterministic operation ids.
  - [ ] Milestone 1.3: keep transitional fallback path aligned with same retry/idempotency rules.

- [ ] Phase 2: Signals and directory cache
  - [ ] Milestone 2.1: daemon-served `GetSignals/ListWorkers/ListInstances` with staleness metadata.
  - [ ] Milestone 2.2: GS watch streams and daemon-side cache controllers.
  - [ ] Milestone 2.3: expose SDK `TensorCastSignals` and `ExecutionSignals`.

- [ ] Phase 3: Cache blob routing and lease fencing control path
  - [ ] Milestone 3.1: add shard lease watch/cache and fail-closed ownership transitions.
  - [ ] Milestone 3.2: add home-scoped `CacheBlobBatch*` routing with fenced redirect behavior.
  - [ ] Milestone 3.3: ensure plan actions route through instance-agent boundary with no direct SDK->GS calls.

- [ ] Phase 4: Plan IR and node-agent execution
  - [ ] Milestone 4.1: add required actions (`prefetch_many`, `materialize_into`, cache aliases).
  - [ ] Milestone 4.2: execute instance-scoped actions via node-agent with bounded concurrency.
  - [ ] Milestone 4.3: avoid re-entrant deadlocks between instance-step execution and local cache blob frontend calls.

- [ ] Phase 5: Validation and rollout
  - [ ] Milestone 5.1: control-plane chaos tests for cache staleness, redirect, and lease generation changes.
  - [ ] Milestone 5.2: verify mixed-version compatibility between local runner and ingress mode.
  - [ ] Milestone 5.3: update docs and lock acceptance criteria.

# Tasks

- SDK/runtime:
  - implement runtime and plan submission plumbing in `tensorcast/api/runtime.py` and `tensorcast/api/plan/plan.py`.
  - plumb lane/policy metadata across non-plan RPC paths.

- Daemon:
  - add ingress controller, signals controller, and directory cache controller.
  - add mesh client support for plan fragment dispatch.

- Global Store:
  - add watch streams and shard lease RPCs required by daemon caches.
  - keep low-cardinality control-plane responsibilities only.

- Node agent:
  - execute instance-scoped actions safely with engine adapter boundaries.

# Test / Rollout / Backout

Tests:

- Python:
  - `source .venv/bin/activate && pytest tests/python/test_plan.py`
  - `source .venv/bin/activate && pytest tests/python/test_runtime.py`
- C++:
  - `bazel test //daemon:session_lifecycle_test`
  - `bazel test //daemon:grpc_service_impl_registration_test`
- Proto:
  - `bash tools/build_proto_python.sh`
  - `bazel test //proto/... --test_output=errors`

Rollout:

- enable ingress by daemon capability flag.
- keep local runner fallback until ingress parity is proven in staged clusters.

Backout:

- disable ingress and route `Plan.run()` back to caller-local runner.
- keep watch APIs additive and leave legacy polling path active.

# Risks and Tracking

- Risk: ingress introduces divergent retry/idempotency semantics.
  - tracking: deterministic step fingerprint checks between local and daemon modes.
- Risk: directory cache staleness harms routing correctness.
  - tracking: `as_of_ms`/`staleness_ms` budgets with alerting.
- Risk: shard lease churn causes oscillation.
  - tracking: lease generation churn metrics and fail-closed event counters.
