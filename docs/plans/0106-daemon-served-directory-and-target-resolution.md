---
slug: daemon-served-directory-and-target-resolution
title: Daemon-Served Directory and Target Resolution (Plan)
links:
  design: ../designs/0106-daemon-served-directory-and-target-resolution.md
---

# Objective

Land one dependency-ready directory contract for programmable callers so target
resolution stops drifting across SDK-direct GS reads, daemon caches, rollout
helpers, and future instance-step routing.

# Current State and Grounding

- SDK still has `CapabilityDirectoryClient` as a direct Global Store reader.
- daemon already has `WorkerDirectoryCache`, but not a full public directory
  surface.
- `GetWorkerStatus` is the first daemon-served signal with freshness metadata.
- no dependency-ready `NodeAgentDirectory` exists yet for
  `instance_id -> node_agent_endpoint`.

# Phases and Milestones

- [ ] Phase 1: Freeze the shared contract
  - [ ] Milestone 1.1: freeze stable target identity by `daemon_id` and
        `instance_id`
  - [ ] Milestone 1.2: freeze daemon-served directory RPC surface
  - [ ] Milestone 1.3: freeze freshness evidence and fail-closed behavior
  - [ ] Milestone 1.4: freeze `NodeAgentDirectory` minimal contract

- [ ] Phase 2: Close daemon caches and RPC wiring
  - [ ] Milestone 2.1: add `InstanceDirectoryCache`
  - [ ] Milestone 2.2: add daemon RPCs for `ListWorkers`, `ListInstances`, and
        `GetWorkerCapacity`
  - [ ] Milestone 2.3: expose freshness metadata on those reads
  - [ ] Milestone 2.4: fail closed when cache freshness exceeds budget

- [ ] Phase 3: Migrate programmable callers
  - [ ] Milestone 3.1: add SDK `TensorCastSignals.list_workers(...)`
  - [ ] Milestone 3.2: add SDK `TensorCastSignals.list_instances(...)`
  - [ ] Milestone 3.3: add SDK `TensorCastSignals.get_worker_capacity(...)`
  - [ ] Milestone 3.4: stop new programmable features from depending on
        SDK-direct GS directory reads

# Test, Rollout, and Backout

Tests:

- worker directory freshness and fail-closed behavior
- instance directory freshness and fail-closed behavior
- `NodeAgentDirectory` resolution correctness
- SDK signal wrappers over daemon-served directory reads

Rollout:

1. land daemon caches and RPCs before migrating rollout or instance-step users
2. keep `CapabilityDirectoryClient` as compatibility fallback during migration

Backout:

- keep daemon-served directory additive
- preserve `CapabilityDirectoryClient` while migration is incomplete
