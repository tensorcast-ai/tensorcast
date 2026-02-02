---
title: API Architecture
description: Public API design and internal API flows for TensorCast
---

# API Architecture

This section documents the public Python SDK surface and the internal API flows
(SDK ↔ daemon ↔ StoreEngine ↔ Global Store) that implement registration,
materialization, policy/persistence, and region-backed operations.

These docs are intended to answer, in order of importance:

- **What** the API/flow is (contract, semantics, invariants)
- **Why** it exists (tradeoffs, safety, performance, operational needs)
- **How** it works (RPC sequence, key data structures, and code map)

## Reading Start Points

- Application user: [API Design](./api-design.md)
- SDK maintainer: [API Design](./api-design.md) → [Registration Flow](./registration-flow.md) → [Materialization Flow](./materialization-flow.md)
- Daemon engineer: [Registration Flow](./registration-flow.md), [Policy & Persistence](./policy-persistence.md), [Region-Backed](./region-backed.md)
- Global Store engineer: [Policy & Persistence](./policy-persistence.md)
- Core Store engineer: [Registration Flow](./registration-flow.md) and [Policy & Persistence](./policy-persistence.md)
- Observability / on-call: [Error, Retry, Observability](./error-retry-observability.md)

## Document Map

- [API Design](./api-design.md): Public Python API surface, core concepts, and user-facing contracts.
- [Registration Flow](./registration-flow.md): `register`/`put`/`register_view` internals and the `Begin`/`Feed`/`Commit` lifecycle.
- [Materialization Flow](./materialization-flow.md): `Artifact` retrieval, source selection, selective materialization, and `get_into`/region-backed paths.
- [Policy & Persistence](./policy-persistence.md): `StorePolicy` model, placement planning, and persistence task lifecycle.
- [Region-Backed](./region-backed.md): VRAM region registration, region-referenced LIP, and quiesced teardown (`DeregisterArtifact`).
- [Error, Retry, Observability](./error-retry-observability.md): Error taxonomy, retry model, and key metrics/logging signals.
- [Artifact Views and Retrieval](../artifact-views-and-retrieval.md): Canonical view semantics and retrieval architecture.

## Concept Index

- Policy model and tier semantics: [policy-persistence.md#storepolicy-model](./policy-persistence.md#storepolicy-model)
- Durability and background tasks: [policy-persistence.md#startpersistence-and-querypersistencestatus](./policy-persistence.md#startpersistence-and-querypersistencestatus)
- Registration RPC lifecycle: [registration-flow.md#begin-feed-commit](./registration-flow.md#begin-feed-commit)
- Region-backed LIP reuse: [registration-flow.md#region-referenced-lip-storage](./registration-flow.md#region-referenced-lip-storage)
- Key mapping and `key` materialization: [materialization-flow.md#materialize-by-key-and-by-replica](./materialization-flow.md#materialize-by-key-and-by-replica)
- Materialization source selection: [materialization-flow.md#fallback-and-source-preference](./materialization-flow.md#fallback-and-source-preference)
- Region-backed `get_into`: [materialization-flow.md#region-backed-get_into-materializeintotarget-v2](./materialization-flow.md#region-backed-get_into-materializeintotarget-v2)
- Status codes and retries: [error-retry-observability.md#error-model](./error-retry-observability.md#error-model)
- View semantics and identity: [artifact-views-and-retrieval.md](../artifact-views-and-retrieval.md)

## Related Module Docs

- [tensorcast/api/README.md](../../../tensorcast/api/README.md)
- [tensorcast/api/store/README.md](../../../tensorcast/api/store/README.md)
- [daemon/README.md](../../../daemon/README.md)
- [tensorcast/global_store/README.md](../../../tensorcast/global_store/README.md)
- [core/store/README.md](../../../core/store/README.md)
