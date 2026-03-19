---
slug: daemon-served-directory-and-target-resolution
title: Daemon-Served Directory and Target Resolution (Plan)
links:
  design: ../designs/0106-daemon-served-directory-and-target-resolution.md
---

# Objective

Land one dependency-ready routing-directory contract for programmable callers so
stable target identity, worker route mapping, instance execution routing, and
local-only versus cluster-backed behavior stop drifting across SDK-direct GS
reads, daemon caches, rollout helpers, queue helpers, and future instance-step
routing.

# Status Update

Implemented in this change set:

- explicit instance execution facts in the Global Store schema and RPC surface
  via `execution_endpoint` and `execution_host_kind`
- Node Agent registration publishing explicit execution routing facts
- Node Agent registration and heartbeats resolving `worker_id` from the local
  daemon front door instead of Global Store worker-directory reads
- daemon-served worker and instance directory RPCs with bounded-staleness
  freshness metadata and `authority_mode`
- shared daemon caches for worker routes and instance execution routes
- Python `Runtime.directory()` plus `signals().list_*` compatibility delegation
- Python plan `Instance` objects carrying `daemon_id` and `execution_endpoint`
- daemon ingress forwarding compatible instance-targeted plans through the same
  route-directory contract to resolved Node Agent endpoints

Still pending after this cut:

- downstream migrations for `0104`, `0060`, and broader SDK-direct GS cleanup

# Current State and Grounding

- SDK still has `CapabilityDirectoryClient` as a direct Global Store reader.
- daemon already has `WorkerDirectoryCache`, but not a full public directory
  surface.
- `GetWorkerStatus` is the first daemon-served signal with freshness metadata,
  but those fields are local snapshot metadata rather than full directory-cache
  provenance.
- Global Store instance rows expose `signals_endpoint`, not a canonical
  execution endpoint fact.
- Python plan or runtime instance objects still carry `worker_id` and
  `signals_endpoint`, and local `Plan.run()` still fails instance steps instead
  of owning instance-host routing.
- no dependency-ready instance-execution directory exists yet for
  `instance_id -> execution_endpoint`.
- `0060` needs leader routing but should not grow a second worker-route dialect.

# Phases and Milestones

- [x] Phase 1: Freeze the routing contract and ownership split
  - [x] Milestone 1.1: freeze stable target identity by `daemon_id` and
        `instance_id`
  - [x] Milestone 1.2: freeze `0106` scope as stable-principal-to-route read
        model only, not workflow truth
  - [x] Milestone 1.3: freeze authority modes:
        `GLOBAL_STORE_BACKED` and `LOCAL_ONLY`
  - [x] Milestone 1.4: freeze long-term SDK split:
        `runtime.directory()` versus `runtime.signals()`

- [x] Phase 2: Anchor explicit instance execution facts
  - [x] Milestone 2.1: add `execution_endpoint` to the instance registry and
        protobuf surfaces
  - [x] Milestone 2.2: optionally add `execution_host_kind`
  - [x] Milestone 2.3: keep `signals_endpoint` observability-only
  - [x] Milestone 2.4: land a short-lived label-based migration shim only if
        needed, with explicit removal plan

- [x] Phase 3: Close daemon caches and daemon-served RPCs
  - [x] Milestone 3.1: keep `WorkerDirectoryCache` but upgrade it to shared
        route-cache semantics
  - [x] Milestone 3.2: add `InstanceExecutionDirectoryCache`
  - [x] Milestone 3.3: add daemon-served directory RPCs for worker and instance
        route reads
  - [x] Milestone 3.4: expose freshness metadata and `authority_mode`
  - [x] Milestone 3.5: fail closed when cache freshness exceeds budget

- [x] Phase 4: Migrate SDK and object models
  - [x] Milestone 4.1: add `Runtime.directory()`
  - [x] Milestone 4.2: keep `TensorCastSignals.list_*` only as compatibility
        delegation during migration
  - [x] Milestone 4.3: add `daemon_id` to Python `Instance` route objects and
        stop treating `signals_endpoint` as routing state
  - [x] Milestone 4.4: make `Plan.run()` and daemon ingress consume the same
        route-directory contract

- [ ] Phase 5: Migrate dependent owners
  - [ ] Milestone 5.1: switch `0104` rollout helpers to daemon-served route
        discovery
  - [ ] Milestone 5.2: make `0060` reuse `WorkerDirectory` for
        `leader_daemon_id -> endpoint`
  - [ ] Milestone 5.3: stop new programmable features from depending on
        SDK-direct GS directory reads

# Test, Rollout, and Backout

Tests:

- worker directory freshness and fail-closed behavior
- instance execution directory freshness and fail-closed behavior
- explicit `execution_endpoint` publication and readback correctness
- local-only singleton worker and instance directory behavior
- queue leader endpoint mapping reusing worker directory
- SDK directory wrappers over daemon-served route reads
- compatibility delegation from `signals().list_*` while migration is active

Rollout:

1. land explicit execution-endpoint facts before migrating rollout or
   instance-step users
2. land daemon caches and RPCs before migrating `0104`, `0060`, or local
   `Plan.run()` routing
3. keep `CapabilityDirectoryClient` as compatibility fallback during migration
4. keep `signals().list_*` additive until `runtime.directory()` consumers are
   stable

Backout:

- keep daemon-served directory additive
- keep `signals_endpoint` readable while the explicit execution-endpoint rollout
  is incomplete
- keep `CapabilityDirectoryClient` and old plan object shapes readable while the
  new route objects are not yet fully consumed
- preserve `CapabilityDirectoryClient` while migration is incomplete
