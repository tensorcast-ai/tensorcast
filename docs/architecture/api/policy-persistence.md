---
title: StorePolicy And Persistence
description: Policy model, placement planning, and persistence tasks
---

# StorePolicy And Persistence

This document defines the policy model used by `register` and `put` and
explains how persistence and placement are executed.

## StorePolicy Model

`StorePolicy` is the single durability and placement declaration. It supports:

- Profile presets: `cache`, `durable`, `ha`, `cold`, `warm`, `pinned`.
- Explicit tiers via `must`, `should`, `may` lists.
- `overflow_policy` for local stable DRAM admission.
- `layout` to control shard layout.

The SDK validates policy shape and forwards it to the daemon. The daemon is the
authoritative resolver.

## Policy Validation And Resolution

Validation rules are enforced in two places:

- SDK: `tensorcast/api/_config.py`
- Daemon: `daemon/store_policy_resolver.cc`

Key constraints:

- `shared_disk` requires scope `any` and `min_replicas=1` and forbids retention
  fields.
- `stable_dram` supports only `min_replicas=1` and retention only for local
  scopes.
- `must` local stable requires `retention_policy=pinned`.
- `overflow_policy=spill` requires shared disk in `must` or `should`.

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

- Policy model: `../../../tensorcast/api/_config.py`
- Policy resolver: `../../../daemon/store_policy_resolver.cc`
- Persistence manager: `../../../daemon/persistence_manager.cc`
- Placement service: `../../../tensorcast/global_store/services/placement_service.py`
- Daemon RPCs: `../../../proto/tensorcast/daemon/v1/store_daemon.proto`
