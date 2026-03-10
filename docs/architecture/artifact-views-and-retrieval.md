---
title: Artifact Views and Retrieval
description: Canonical view semantics, planning/execution, and retrieval/materialization pipeline
---

# Summary

This document consolidates the canonical semantics for artifact views (ViewSpec, view identities, ByteSpaces) and
how those views flow through retrieval and materialization. It is the stable, human-readable reference for
variant-aware access, and it points to the detailed API and internal execution docs where appropriate.

If you need the step-by-step RPC and pipeline mechanics, start from
`docs/architecture/api/materialization-flow.md`. If you need the byte-range execution engine details, see
`docs/internals/byte-range-mapping-and-execution.md`.

# Scope

In scope:
- ViewSpec semantics (supported ops, normalization, identity handling).
- View identity, hash, and ByteSpace semantics.
- Planning and execution flow at the architecture level (ViewPlanner, ViewPlanSource, ViewIngestExecutor).
- Retrieval/materialization pipeline interactions, including into-target and deferred slice.

Out of scope (linked instead):
- Programmable framework and Operation semantics: `docs/designs/0055-programmable-framework.md`.
- Artifact-first SDK product narrative: `docs/designs/0039-artifact-first-sdk.md`.
- CPU shared-memory materialization: `docs/designs/0049-cpu-shared-memory-materialization.md`.

# Core Concepts

## Canonical index and ByteSpaces

- The canonical index (index v3) defines the canonical ByteSpace for an artifact.
- Canonical ByteSpace is anchored by `index_multihash` and covers `[0, total_size)`.
- A view defines a variant ByteSpace anchored by `view_id` and covers `[0, view_size)`.

When discussing coverage or verification, always specify which ByteSpace is referenced. Missing coverage is never
filled implicitly; missing byte ranges are surfaced as `UNAVAILABLE` with `PartialCoverageDetail`.

## ViewSpec

A ViewSpec describes per-tensor operations for a view. v1 supports:
- `narrow(dim, start, length)` (single dim, step = 1)
- `transpose(dim0, dim1)`

Rules:
- A tensor may use either `narrow` or `transpose`, not both.
- `narrow` is limited to one dimension per tensor.
- Omitted tensors are identity passes.

## View identity and hashes

- `view_id` is a deterministic identity for the variant ByteSpace. It is derived from a normalized ViewSpec plus the
  canonical index identity. Identity views are collapsed to the canonical path and omit `view_id`.
- `view_data_hash` is a TreeHash over the realized view ByteSpace (post-transform), and is distinct from `view_id`.
- `view_subset_hash` (aka `ViewSubset.subset_hash`) is an opaque digest for subset selection (for example, sorted
  `tensor_names`). It is not a view identity and must be treated as raw bytes.

# View Planning and Execution (Architecture)

View planning and execution are centralized in the C++ core so that retrieval and registration share the same math.

Key components:
- `ViewPlanner` builds a `ViewPlan` from canonical index JSON plus ViewSpec.
  - Emits a `SelectionPlan` (byte ranges in canonical space) and a `TransformPlan` (for transpose).
  - Emits `ViewWritePlan` for ingestion (inverse mapping).
- `ViewPlanSource` executes the selection plan and streams bytes from any `SeekableSource`.
  - Uses the unified byte-range execution engine (ByteRangeMap + ByteRangeProgram).
  - For narrow(axis=1) plans, strided coalescing packs data to avoid IOPS-bound reads.
- `ViewIngestExecutor` writes view bytes back into canonical storage using the inverse plan.

For execution details, see `docs/internals/byte-range-mapping-and-execution.md`.

# Retrieval and Materialization Pipeline

Retrieval always routes through the daemon and StoreEngine, regardless of source preference. At a high level:

1) Resolve the canonical index (disk descriptor or Global Store).
2) Normalize view identity (compute or validate `view_id` when the view is non-identity).
3) Build a view plan (SelectionPlan + TransformPlan + ViewWritePlan).
4) Select the source (existing replica, P2P, or disk) based on `SourcePolicy`.
5) Stream bytes through the data plane (ViewPlanSource + pump) into a replica or into a target layout.
6) Return view metadata (`view_index_json`, `view_data_hash`) alongside handles.

See `docs/architecture/api/materialization-flow.md` for the full control-flow and RPC sequence.

# Into-Target and Deferred Slice

`MaterializeIntoTarget` streams bytes into client-provided CUDA regions with a coalesced `TargetLayout`:
- The layout can be canonical or view-indexed.
- Packed subsets are supported and use `tensor_names` ordering to define view offsets.

Deferred slice loading is built on the same primitive:
- The SDK allocates a client-owned CUDA region, registers it, and defers I/O.
- `commit()` issues a single `MaterializeIntoTarget` with the subset order required by the view plan.

See `docs/architecture/api/region-backed.md` and `docs/internals/tensor_dict_into_dataflow.md` for details.

# SDK Surface (Where This Shows Up)

The public API surface is documented in `docs/architecture/api/api-design.md`. View semantics influence:
- `get_view` / `register_view` (view-aware retrieval and registration)
- `get_into` / `MaterializeIntoTarget` (region-backed, view-indexed materialization)
- `Artifact` handle methods that accept view specs or subset selection

# Error Model (View-Specific)

Common failure modes:
- `INVALID_ARGUMENT`: unknown tensor name, invalid dimensions, unsupported op mix, invalid ranges.
- `FAILED_PRECONDITION`: unsupported placement or transform (for example, view transforms not allowed for a given path).
- `UNAVAILABLE` + `PartialCoverageDetail`: missing byte coverage for the requested ByteSpace.

See `docs/architecture/api/error-retry-observability.md` for details.

# Current vs Planned Behavior

Current behavior (v1):
- Supported per-tensor ops: single-dimension `narrow` and `transpose`.
- Identity views fold to canonical retrieval/materialization.
- View-aware routing is best-effort; canonical fallbacks remain valid.

Planned extensions:
- Broader view op support (beyond `narrow`/`transpose`) once execution and validation are expanded.
- Transform-aware assembly and overlap semantics (see `docs/architecture/view-replicas-and-assembly.md`).

# Related Docs and Code Map

Related docs:
- `docs/architecture/api/api-design.md`
- `docs/architecture/api/materialization-flow.md`
- `docs/architecture/api/region-backed.md`
- `docs/internals/byte-range-mapping-and-execution.md`
- `docs/internals/canonical-index.md`
- `docs/internals/tensor-first-artifact-architecture.md`
- `docs/architecture/view-replicas-and-assembly.md`

Code map (entry points):
- View planning: `core/store/materialization/dataplane/view/view_planner.{h,cc}`
- View execution: `core/store/materialization/dataplane/view/view_plan_source.{h,cc}`
- View ingest: `core/store/materialization/dataplane/view/view_ingest_executor.{h,cc}`
- Byte-range execution: `core/store/materialization/dataplane/sources/byte_range_mapped_source.{h,cc}`
- Materialization control: `core/store/runtime/ingestion/materialization_service.{h,cc}`
