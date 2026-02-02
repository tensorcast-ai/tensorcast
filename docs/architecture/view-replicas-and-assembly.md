---
title: View Replicas and Assembly
description: Dense piece replicas, assembly lifecycle, sealing, and overlap semantics
---

# Summary

This document describes view replicas (pieces), their assembly lifecycle, and sealing into canonical MI2 identity.
It consolidates the architecture-level semantics for dense piece registration, assembly, and sealing, and it
distinguishes current behavior from planned extensions.

# Scope

In scope:
- Piece (dense view replica) semantics and invariants.
- Assembly lifecycle: unsealed assemblies, sealing, and post-seal behavior.
- Overlap and transform rules (current vs planned).

Out of scope (linked instead):
- Programmable control-plane operations: `docs/designs/0055-programmable-framework.md`.
- Byte-range execution engine details: `docs/internals/byte-range-mapping-and-execution.md`.
- SDK product positioning: `docs/designs/0039-artifact-first-sdk.md`.

# Core Concepts

## Assembly and piece identities

- **Assembly**: an unsealed artifact identity, typically `assembly_id = cgid:...`, bound to a canonical index.
- **Piece**: a dense view replica under `(assembly_id, view_id)`, stored in a view ByteSpace.
- **Sealed artifact**: a canonical MI2 identity (`mi2:index_multihash:data_multihash`) derived when full coverage
  exists and sealing completes.

Pieces are always dense in their own view ByteSpace; partial coverage is tracked via canonical coverage metadata
rather than by storing sparse canonical buffers.

## ByteSpace separation

- Canonical ByteSpace is anchored by `index_multihash`.
- View ByteSpace is anchored by `view_id`.
- Routing, transport locks, and replica state must disambiguate ByteSpaces to avoid collisions.

# Piece Registration (Current Behavior)

Current implemented constraints (v1):
- **Selection-only views**: pieces may use `narrow` but must not use `transpose`.
- **No overlaps**: overlapping canonical coverage across different `view_id` is rejected.
- **No LIP pieces**: lease-in-place (VRAM_LEASED) is not supported for piece registration.
- **Partial coverage required**: identity/full-coverage views are rejected for pieces; use canonical registration
  or sealing instead.
- **Dense bytes**: piece payloads are dense, and padding bytes introduced by alignment are zeroed.

These rules keep piece replicas deterministic and prevent sparse canonical allocations.

# Assembly Lifecycle

## Unsealed assembly

An assembly binds a canonical index to `assembly_id` without a canonical data hash. Pieces register against the
assembly and publish coverage ranges plus view metadata. The assembly can serve view materialization requests by
selecting and copying from available pieces.

## Sealing

Sealing transitions an assembly to a sealed MI2 identity:
- Canonical coverage is validated.
- `data_multihash` is computed.
- `assembly_id -> mi2` binding is persisted.
- Optional policies may materialize and publish a canonical replica.

Post-seal, canonical MI2 becomes authoritative; piece registrations for the sealed assembly are rejected, and
policies may migrate or retire cached views.

# Overlaps and Transforms (Planned Extensions)

Planned extensions (v2) introduce:
- **Transform-aware assembly**: allow transpose-based assembly using a structured execution graph.
- **Overlap tolerance**: support `REPLICATE_EQUAL` overlaps with proof commitments.
- **LIP pieces**: allow piece registration on VRAM_LEASED plans when ByteSpace-aware LIP exports exist.
- **LayoutSpec bindings**: content-addressed layout declarations for long-lived assemblies.

Until these are delivered, the v1 constraints above remain in force.

# Related Docs and Code Map

Related docs:
- `docs/architecture/artifact-views-and-retrieval.md`
- `docs/internals/byte-range-mapping-and-execution.md`
- `docs/architecture/api/materialization-flow.md`
- `docs/architecture/api/registration-flow.md`
- `docs/designs/0055-programmable-framework.md`

Code map (entry points):
- Piece registration and sealing: `core/store/runtime/metadata/registration_backend.{h,cc}`
- Assembly materialization: `core/store/runtime/ingestion/materialization_facade.{h,cc}`
- Daemon sealing RPC: `daemon/service/controllers/registration_controller.cc`
- Global Store view state: `tensorcast/global_store/services/view_state_service.py`
