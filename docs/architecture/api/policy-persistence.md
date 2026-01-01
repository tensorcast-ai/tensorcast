---
title: StorePolicy And Persistence
description: Policy model, placement planning, and persistence tasks
---

# StorePolicy And Persistence

This document defines the policy model used by `register` and `put` and
explains how persistence and placement are executed.

Related docs:

- Public surface and contracts: [API Design](./api-design.md#storepolicy-and-persistence-hooks)
- Registration lifecycle and where policy is resolved: [Registration Flow](./registration-flow.md#local-stable-tier)
- Persistence status and on-call signals: [Error, Retry, Observability](./error-retry-observability.md)

## Why a unified policy?

TensorCast uses a single `StorePolicy` object to describe *durability* and
*placement intent* so that:

- applications declare outcomes (“must be on shared disk”) instead of implementation details
- the daemon can choose the correct mechanism (sync local stable admission vs async persistence)
- the system can report **degraded vs failed** outcomes in a consistent way

`StorePolicy` is not “just configuration”: it is a contract that influences
correctness (data loss avoidance) and operational behavior (admission/eviction).

## StorePolicy Model

`StorePolicy` is the single durability and placement declaration. It supports:

- Profile presets: `cache`, `durable`, `ha`, `cold`, `warm`, `pinned`.
- Explicit tiers via `must`, `should`, `may` lists.
- `overflow_policy` for local stable DRAM admission.
- `layout` to control shard layout.

The SDK validates policy shape and forwards it to the daemon. The daemon is the
authoritative resolver.

### What “must/should/may” mean

- `must`: required for correctness from the caller’s point of view. Failure is a hard error.
- `should`: best-effort. Failure downgrades the result to *degraded* but does not fail the primary operation.
- `may`: opportunistic. Failure is silent and does not degrade.

Why this matters:

- It allows a “fast local success” while still requesting background durability.
- It allows operators to distinguish “everything is durable” from “we returned
  something usable but not fully placed”.

### TierSpec (policy tier parameters)

Policy tiers are expressed as `TierSpec`.

- Python model: [tensorcast/api/_config.py](../../../tensorcast/api/_config.py) (`StorePolicy`, `TierSpec`)
- Proto model: [proto/tensorcast/daemon/v1/store_daemon.proto](../../../proto/tensorcast/daemon/v1/store_daemon.proto) (`StorePolicy`, `TierSpec`)

Fields:

| Field | What it does | Examples |
|---|---|---|
| `tier` | Which tier: `stable_dram` or `shared_disk`. | `TierSpec(tier="stable_dram", scope="local")` |
| `scope` | Where the tier should exist: `local`, `remote`, `any`. | `stable_dram(scope="remote")` for remote stable replicas. |
| `min_replicas` | Minimum replicas for the tier. | Currently only `1` is supported (see validation). |
| `retention_policy` | Local stable retention: `best_effort`, `ttl`, `pinned`. | `pinned` for “must stay resident”. |
| `retention_ttl_ms` | TTL for `retention_policy=ttl`. | e.g. `60_000` for 60s local caching. |

### Profiles (what they expand to)

Profiles are convenience presets. Expansion happens in both the SDK and daemon:

- SDK: [tensorcast/api/_config.py](../../../tensorcast/api/_config.py) (`StorePolicy.from_profile`)
- Daemon: [daemon/store_policy_resolver.cc](../../../daemon/store_policy_resolver.cc) (`profile_defaults`)

At a high level:

- `cache`: local stable (may, best-effort), overflow `evict`
- `durable`: shared disk (must) + local stable (should, best-effort)
- `ha`: shared disk (must) + remote stable (should) + local stable (should)
- `cold`: shared disk (must) + local stable (should, ttl) with a default TTL
- `warm`: local stable (should, best-effort), overflow `reject`
- `pinned`: local stable (must, pinned), overflow `reject`

## Policy Examples (recommended starting points)

```python
import tensorcast

# Default cache semantics (fast, not durable).
cache = tensorcast.StorePolicy(profile="cache")

# Durable: must land on shared disk, but also try to keep a local stable copy.
durable = tensorcast.StorePolicy(profile="durable")

# HA: durable + try to create a remote stable replica for fast cross-node reads.
ha = tensorcast.StorePolicy(profile="ha")

# Pinned local: fail if we cannot keep this resident in local stable memory.
pinned = tensorcast.StorePolicy(profile="pinned")
```

## Policy Validation And Resolution

Validation rules are enforced in two places:

- SDK: [tensorcast/api/_config.py](../../../tensorcast/api/_config.py)
- Daemon: [daemon/store_policy_resolver.cc](../../../daemon/store_policy_resolver.cc)

Key constraints:

- `shared_disk` requires scope `any` and `min_replicas=1` and forbids retention
  fields.
- `stable_dram` supports only `min_replicas=1` and retention only for local
  scopes.
- `must` local stable requires `retention_policy=pinned`.
- `overflow_policy=spill` requires shared disk in `must` or `should`.

### overflow_policy (local stable admission behavior)

`overflow_policy` controls what happens when local stable DRAM is under pressure:

- `evict`: allow best-effort eviction of non-pinned entries to make space.
- `reject`: refuse admission when capacity is insufficient (caller sees failure/degraded depending on requirement).
- `spill`: allow eviction only when durability requirements are satisfied (gated by the durability index).

Why spill is special:

- It is intended to prevent “evict the only durable copy” scenarios.
- It couples local admission/eviction decisions to persistence completion; see
  [Spill Gating And Durability Index](#spill-gating-and-durability-index).

### layout (sharding intent)

`layout` declares how the artifact should be treated for persistence planning:

- `auto`: daemon chooses based on size and tier requirements.
- `unsharded`: prefer a single logical unit (fewer parts, simpler placement).
- `sharded`: prefer shard planning (better parallelism and partial retries).

Why this exists:

- Sharding can improve throughput and failure isolation for very large artifacts.
- Unsharded can reduce overhead for smaller artifacts and simplify operator reasoning.

## Local Stable Tier Versus Persistence

Local stable DRAM can be satisfied synchronously at commit time. Remote stable
and shared disk are satisfied asynchronously through persistence tasks.

- Local stable `must` failures fail commit.
- Local stable `should` failures return degraded status.

## StartPersistence And QueryPersistenceStatus

The SDK triggers `StartPersistence` after registration when the resolved policy
includes shared disk or remote stable tiers. The daemon runs a background task
and exposes status via `QueryPersistenceStatus`.

Task results are attached to the SDK surface as `persistence_task_id` and can be
queried by task id or artifact id.

### StartPersistenceRequest / QueryPersistenceStatusResponse

Proto: [proto/tensorcast/daemon/v1/store_daemon.proto](../../../proto/tensorcast/daemon/v1/store_daemon.proto)

`StartPersistence`:

- Inputs: `artifact_id`, `policy`
- Output: `task_id`, `plan_id`, `state`, `progress`, `degraded_reason`

`QueryPersistenceStatus`:

- Query by `task_id` or `artifact_id`
- Returns:
  - task-level: `state`, `progress`, `degraded_reason`, `last_error`
  - shard-level: `shards[]` with `state/progress`, plus `target_nodes` and `lease_ids`

How to interpret shard fields:

- `target_nodes[i]` is the planned target for shard `i` (index-aligned).
- `lease_ids[i]` is non-empty once the daemon has acquired/acknowledged the lease for that shard/target.
- A task can be `degraded` while still progressing if optional tiers are failing.

## Placement Planning And Shards

The daemon requests a placement plan from the Global Store:

- Placement policies: `local_only`, `replicated`, `sharded`.
- Shard planning uses UMA chunk layout with a 128MB sharding threshold and
  64MB to 256MB shard caps.
- When remote stable capacity is insufficient, placement degrades to local only
  and reports a degraded reason.

## Task States And Degradation

Persistence tasks report:

- `pending`, `running`, `degraded`, `success`, `failed`
- Degraded states when optional tiers fail or placement downgrades.
- Failed states when required tiers fail.

## Spill Gating And Durability Index

Stable DRAM admission with `overflow_policy=spill` uses a durability index
maintained by persistence. Spill eviction is allowed only when required
non-local tiers are satisfied.

## Code Map

- Policy model: [tensorcast/api/_config.py](../../../tensorcast/api/_config.py)
- Policy resolver: [daemon/store_policy_resolver.cc](../../../daemon/store_policy_resolver.cc)
- Persistence manager: [daemon/persistence_manager.cc](../../../daemon/persistence_manager.cc)
- Global Store placement service: [tensorcast/global_store/services/placement_service.py](../../../tensorcast/global_store/services/placement_service.py)
- Daemon RPCs: [proto/tensorcast/daemon/v1/store_daemon.proto](../../../proto/tensorcast/daemon/v1/store_daemon.proto)
