---
slug: source-side-remote-view-transport
title: Source-Side Remote View Transport Plan
status: in_progress
areas: ["global_store", "core", "daemon", "proto", "tests", "docs"]
created: 2026-03-15
last_updated: 2026-03-18
related_code:
  - docs/designs/0086-source-side-remote-view-transport.md
  - docs/architecture/p2p-transfer-strategies.md
  - proto/tensorcast/global_store/v1/global_store.proto
  - tensorcast/global_store/rpc/transport_rpc_handler.py
  - tensorcast/global_store/services/transport_service.py
  - tensorcast/global_store/repositories/replica_repository.py
  - core/store/components/global_store_client.h
  - core/store/components/global_store_client.cc
  - core/store/materialization/control/materialize_orchestrator.cc
  - core/store/runtime/metadata/metadata_gateway.cc
  - daemon/service/controllers/replica_materialization_service.cc
  - daemon/state/lip_manager.cc
  - daemon/state/types.h
  - schema.sql
links:
  design: ../designs/0086-source-side-remote-view-transport.md
---

# Objective

Implement view-aware remote transport so `request_view_transport(view)` can route either:

- an already-resident dense view replica; or
- a canonical source daemon that can derive and export the requested dense view byte-space on demand.

The target outcome is to eliminate destination-side canonical reconstruction for remote TP-sliced loads while preserving TensorCast view semantics and mixed-version correctness.

# Current State and Root Causes

Current behavior has four blocking gaps:

- view residency publication is best-effort because `record_view_residency` is still effectively a no-op for Global Store routing;
- `request_view_transport` only routes already-routable view sources;
- lookup miss falls back to canonical transport, so the destination daemon reconstructs the view locally;
- receiver-side canonical fallback reintroduces read amplification and strided repack on remote TP loads.

# Phases and Milestones

- [x] Phase 0: Contract Grounding
  - [x] Milestone 0.1: Lock the route kinds and transport semantics: `resident_view`, `derived_view_from_canonical`, `canonical_fallback`.
  - [x] Milestone 0.2: Define the minimal view residency record needed for routing, including `artifact_id`, `view_id`, `view_size_bytes`, optional `view_data_hash`, source identity, and placement.
  - [x] Milestone 0.3: Define lifecycle rules for resident views versus ephemeral transport-scoped derived views.

- [x] Phase 1: First-Class View Residency in Global Store
  - [x] Milestone 1.1: Implement Global Store persistence and query for view residency so `record_view_residency` becomes routable state.
  - [x] Milestone 1.2: Plumb view residency publication from daemon completion and registration paths.
  - [x] Milestone 1.3: Update `request_view_transport(view)` to route already-resident view sources before canonical fallback.

- [x] Phase 2: Route Model and Compatibility
  - [x] Milestone 2.1: Extend the transport response model so route kind is explicit and observable.
  - [x] Milestone 2.2: Carry view-scoped metadata needed for integrity and debugging, including size and optional hash semantics.
  - [x] Milestone 2.3: Keep mixed-version compatibility by retaining canonical fallback when source, destination, or Global Store lacks view-aware capability.

- [x] Phase 3: Source-Side Derived View Export
  - [x] Milestone 3.1: Extend view transport lookup into lookup-or-derive over eligible canonical sources.
  - [x] Milestone 3.2: Implement source-daemon derive-on-demand export that reuses existing local view dataplane primitives.
  - [x] Milestone 3.3: Bind derived-view exports to transport lease or TTL cleanup without turning them into durable global replicas by default.

- [x] Phase 4: Destination Daemon Integration
  - [x] Milestone 4.1: Teach the destination daemon to ingest dense view transport directly without canonical reconstruction.
  - [x] Milestone 4.2: Preserve the existing local materialize, register, and requester-export behavior after ingest completes.
  - [x] Milestone 4.3: Keep canonical fallback functional and correct as the last compatibility path.

- [x] Phase 5: Verification and Correctness
  - [x] Milestone 5.1: Ensure view payloads use view-scoped verification semantics and never reuse canonical verification incorrectly.
  - [x] Milestone 5.2: Validate `need_view_data_hash=false` behavior so skipped view hashing does not weaken byte-space correctness.
  - [x] Milestone 5.3: Add end-to-end correctness tests for resident-view routing, derive-on-demand routing, and canonical fallback.

- [x] Phase 6: Benchmark and Acceptance
  - [x] Milestone 6.1: Re-run remote TP load benchmarks and verify receiver-side amplification and pack disappear on routed view transport.
  - [x] Milestone 6.2: Compare `resident_view`, `derived_view_from_canonical`, and `canonical_fallback` route behavior with route-kind observability enabled.
  - [x] Milestone 6.3: Update architecture and benchmark docs with before/after results and rollout guidance.

- [ ] Phase 7: Ephemeral Derived-View Lifecycle Hardening
  - Note: the first slow `tp=4` remote-update regression came from over-conservative prepare admission rather than true budget exhaustion. Phase 7 hardening must keep fallback tied to real admission failure, not to an arbitrary serialization gate.
  - [ ] Milestone 7.1: Introduce daemon-owned derived-view entry state keyed by `(artifact_id, view_id, device)` so source-side exports are tracked as reusable ephemeral residents rather than one-shot transport cleanup artifacts.
  - [x] Task 7.1a: Define an explicit derived-view entry record with lifecycle state, local export handle, routing identity, TTL deadline, and `active_fetches`.
  - [ ] Task 7.1b: Add a daemon-local index keyed by `(artifact_id, view_id, device)` and a secondary lookup by exported replica identity for cleanup and observability.
  - [x] Task 7.1c: Split entry states at least into `preparing`, `ready`, and `draining`, with invariants documented at the manager boundary.
  - [ ] Task 7.1d: Make concurrent requests for the same key coalesce onto one prepare path instead of exporting duplicate dense views.
  - [x] Milestone 7.2: Implement sliding TTL semantics for derived views, with refresh only on successful data-plane use and not on control-plane probes or route lookup.
  - [x] Task 7.2a: Add a configurable TTL for source-side derived views with a conservative default suitable for repeated model-load/update bursts.
  - [x] Task 7.2b: Refresh TTL on successful data-plane admission/use only, not on `RequestViewTransport` lookup hits or other control-plane reads.
  - [x] Task 7.2c: Define how `preparing` entries age, including whether they are protected from expiry until first successful publish/use.
  - [x] Milestone 7.3: Implement pressure-aware admission and eviction for derived views so expired idle entries are reclaimed first, then oldest idle non-expired entries, while `preparing` and `active_fetches>0` entries remain protected.
  - [x] Task 7.3a: Track derived-view residency against the source daemon memory budget separately from durable canonical residency.
  - [x] Task 7.3b: Add an admission check before prepare/publish that can reclaim eligible entries before creating a new dense view.
  - [x] Task 7.3c: Implement eviction ordering: expired idle first, then oldest idle non-expired, never `preparing`, never `active_fetches>0`.
  - [x] Task 7.3d: Define the fallback policy when admission still fails after eviction: return canonical fallback rather than overcommitting source-daemon memory.
  - [x] Milestone 7.4: Implement ordered retirement semantics on the source daemon: mark draining, withdraw Global Store route, wait for in-flight fetches to drain, unregister export state, then release local memory.
  - [x] Task 7.4a: Introduce an explicit drain transition that blocks new fetches before local export teardown starts.
  - [x] Task 7.4b: Make Global Store route withdrawal happen before local export release so stale-route fetches cannot observe a dead endpoint.
  - [x] Task 7.4c: Wait for `active_fetches==0` before unregistering export state and releasing local backing memory.
  - [x] Task 7.4d: Ensure exceptional cleanup and normal TTL/pressure retirement share the same ordered teardown path.
  - [x] Milestone 7.5: Ensure canonical fallback remains the compatibility and resource-exhaustion backstop when derived-view admission cannot safely proceed.
  - [x] Task 7.5a: Keep route-kind observability explicit so `derived_view_from_canonical` and `canonical_fallback` remain distinguishable in logs and metrics.
  - [x] Task 7.5b: Preserve current compatibility fallback when source, destination, or Global Store lacks the needed view-aware capability.
  - [x] Task 7.5c: Add explicit fallback reasons for admission failure, route invalidation, and lifecycle-race handling.
  - [x] Milestone 7.6: Add lifecycle metrics and logs for creation, reuse hit, TTL refresh, drain, eviction, and fallback reason so repeated remote update/load workloads are diagnosable.
  - [x] Task 7.6a: Add counters or structured logs for create/reuse/refresh/expire/evict/drain/fallback events.
  - [x] Task 7.6b: Make logs include `(artifact_id, view_id, device)` and route kind so cross-daemon correlation is possible during remote benchmarks.
  - [x] Task 7.6c: Add enough observability to explain repeated-trial behavior such as “trial 2/3 reused source-side dense view from trial 1”.
  - [ ] Milestone 7.7: Add focused tests:
    - [x] Test 7.7a: unit test for derived-view cache keying and reuse, proving repeated identical requests hit the same source-side ephemeral export within TTL.
    - [x] Test 7.7b: unit test for sliding TTL refresh semantics, proving data-plane use extends lifetime while pure control-plane observation does not.
    - [x] Test 7.7c: unit test for eviction ordering, proving expired idle entries evict before non-expired idle entries and active/preparing entries are never reclaimed.
    - [ ] Test 7.7d: integration test for ordered retirement, proving route withdrawal happens before local export teardown and prevents stale-route connection failures.
    - [x] Test 7.7e: integration test for repeated multi-version remote update/load workloads, proving source-side derived views do not accumulate unbounded DRAM and later trials continue to succeed.
    - [x] Test 7.7f: regression test for fallback behavior under forced admission pressure, proving the system degrades to canonical transport without semantic breakage.

- [x] Phase 8: Deadline and Wait-Budget Semantics for Source-Side Upgrade
  - Note: the current remote-update failures show that source-side derived-view upgrade still inherits an internal ~30s wait budget from `pinned_allocation_timeout_ms`, even when the caller allows a much larger end-to-end deadline. This phase makes caller deadline authoritative and stops using pinned-allocation timeout as a distributed upgrade deadline.
  - [x] Milestone 8.1: Re-establish authoritative request-budget semantics for remote materialization and source-side upgrade.
  - [x] Task 8.1a: Change daemon-side materialization request budget derivation so it uses the incoming gRPC deadline as the authoritative request budget, instead of deriving request budget from `pinned_allocation_timeout_ms`.
  - [x] Task 8.1b: Keep `pinned_allocation_timeout_ms` scoped to local pinned/staging allocation waits only; do not copy it into `request_budget` or `transport_wait_timeout`.
  - [x] Task 8.1c: Make source-side upgrade wait budget derive from remaining authoritative request budget, so daemon B may keep waiting for daemon A's derived-view export until the caller deadline is nearly exhausted.
  - [x] Milestone 8.2: Make timeout and fallback semantics principled.
  - [x] Task 8.2a: Preserve canonical fallback for compatibility, route-unavailable, lifecycle-race, and admission/resource failures.
  - [x] Task 8.2b: Stop doing late canonical fallback when source-side upgrade merely exhausts the remaining request deadline; propagate timeout instead.
  - [x] Task 8.2c: Ensure logs and status reasons distinguish `admission/resource fallback` from `deadline exhausted while waiting for source-side export readiness`.
  - [x] Milestone 8.3: Add observability for budget propagation and upgrade waiting.
  - [x] Task 8.3a: Log caller request budget, gRPC deadline remaining, pinned timeout, transport wait timeout, and source-side prepare wait budget at materialization entry / upgrade decision points.
  - [x] Task 8.3b: Add route-selection / fallback logs that make it obvious whether a request stayed on `resident_view`, downgraded to canonical transport, or terminated due to deadline exhaustion.
  - [x] Milestone 8.4: Validate the new semantics against the known regression cases.
  - [x] Task 8.4a: Re-run `update_weight_remote` for `qwen3-32b tp=2/4` and confirm source-side upgrade no longer falls back after an accidental internal ~30s wait-budget cap.
  - [x] Task 8.4b: Re-run `load_weight_remote` and confirm the new request-budget semantics do not regress the existing relay fast path.

# Suggested Implementation Order for Phase 7

- [x] Step 7.A: Add the daemon-local derived-view entry manager and state machine.
- [x] Step 7.B: Move current source-side export registration onto the new manager so repeated requests reuse a stable in-memory entry instead of one-shot cleanup state.
- [x] Step 7.C: Add `active_fetches` accounting and drain semantics first, before TTL or eviction, so teardown ordering is correct.
  - Scope note: relay hot-path completion requires transport-session-backed source-side fetch accounting via daemon-to-daemon `BeginReplicaFetch` / `EndReplicaFetch`; daemon lock/unlock alone is not sufficient.
- [x] Step 7.D: Add sliding TTL refresh on data-plane use.
- [x] Step 7.E: Add admission plus eviction logic and wire it into prepare-time decisions.
- [ ] Step 7.F: Replace ad hoc cleanup paths with one ordered retirement path shared by TTL expiry, pressure eviction, and exceptional cleanup.
- [x] Step 7.G: Add observability for reuse, drain, eviction, and fallback reason.
- [x] Step 7.H: Land unit tests for cache keying, TTL refresh, and eviction ordering.
- [ ] Step 7.I: Land integration tests for ordered retirement and repeated multi-version remote update/load stability.

# Validation Matrix

Global Store and daemon control plane:

- view residency publish/query behavior
- route-kind response correctness
- mixed-version canonical fallback

Data path correctness:

- resident dense view transport matches requested `view_id`
- derived dense view transport matches requested `view_id`
- canonical fallback remains functionally unchanged

Lifecycle correctness:

- derived-view TTL reuse works across repeated fetches of the same `(artifact_id, view_id, device)`
- derived-view entries are not retained indefinitely after TTL expiry or pressure eviction
- stale resident-view routes are withdrawn before local export release
- active fetches are never torn down underneath an in-flight transfer

Benchmark acceptance:

- remote TP relay no longer shows receiver-side canonical amplification for routed view transport
- destination-side repack time is eliminated or near-zero on the routed path
- transport time converges toward source-side derivation plus wire transfer, not destination-side reconstruction
- repeated remote update/load trials do not regress due to leaked source-side ephemeral views

# Primary Risks

- routing stale or inconsistent view residency: mitigate with explicit lifecycle rules and lease-scoped cleanup.
- semantic drift between resident view and derived view transport: mitigate with shared view metadata contract and end-to-end byte-space tests.
- mixed-version rollout ambiguity: mitigate with explicit route kind and capability-gated canonical fallback.
- over-promoting ephemeral derived views into durable state: mitigate by keeping derive-on-demand export daemon-owned, TTL-scoped, and pressure-evictable by default.
- stale source-side routes after local export teardown: mitigate with ordered drain and route-withdraw-first retirement.
- repeated multi-version workloads exhausting source-daemon DRAM: mitigate with bounded ephemeral residency plus eviction/admission control.

# Owner Checklist

- [x] Design doc and plan stay aligned during implementation.
- [x] Proto and schema changes remain additive and compatibility-safe.
- [x] Route kind is explicit in metrics and logs.
- [x] Benchmark evidence demonstrates the intended TP>1 remote load improvement.
- [x] Lifecycle evidence demonstrates repeated remote load/update trials reuse derived views when hot and reclaim them when cold.
