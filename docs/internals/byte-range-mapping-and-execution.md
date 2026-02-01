---
title: Byte-Range Mapping and Execution
description: ByteRangeMap/Program semantics, compilation, and execution pipeline
---

# Summary

This document describes the unified byte-range mapping and execution engine used across view selection, assembly,
hashing, and materialize-into-target flows. It focuses on execution semantics and invariants, and defers canonical
index format details to `docs/internals/canonical-index.md`.

# Scope

In scope:
- `ByteRangeMap` semantics (explicit PAD, no implicit coverage).
- Normalization, compilation, and execution (`ByteRangeProgram`).
- Strided coalescing and direct-write gating.
- Integration points across the data plane.

Out of scope:
- Canonical index schema details (see `docs/internals/canonical-index.md`).
- Control-plane orchestration plans.

# Core Concepts

## ByteRangeMap

`ByteRangeMap` is the canonical linear IR for mapping destination byte ranges to source ranges. Key rules:
- PAD bytes are explicit segments; they are never inferred at execution time.
- Gaps in destination coverage are invalid; normalization rejects them.
- Missing source coverage is reported upstream as `UNAVAILABLE` with `PartialCoverageDetail` and no map is produced.

## ByteRangeProgram

`ByteRangeProgram` is a compiled execution artifact derived from a normalized map. It organizes runs into
execution-friendly structures:
- `PadRun`: zero-fill ranges.
- `ContiguousRun`: single-source contiguous reads.
- `StridedRun`: single-source strided reads with packing.

The compiler never changes byte semantics; it only chooses execution strategy.

## Sized SeekableSource and short-read contract

All `SeekableSource` implementations are sized and must enforce a strong short-read contract for both `read_at`
and `read_into_at`. Execution assumes total byte length is stable and known at construction time.

# Execution Pipeline

1) **Build**: planners and builders construct `ByteRangeMap` with explicit PAD segments.
2) **Normalize**: sort/merge/validate; reject destination gaps and overlaps.
3) **Compile**: generate `ByteRangeProgram` with run-level strategies.
4) **Execute**: `ByteRangeMappedSource` runs the program against sized sources and writes into sinks or hashing.

Missing coverage is detected before compilation. The executor never invents bytes to hide missing coverage.

# Strided Coalescing

Strided execution coalesces repeated patterns (for example, narrow(axis=1) views) into fewer large reads and packs
results into the destination buffer. This shifts execution from IOPS-bound to bandwidth-bound without changing
logical byte ordering. Strided coalescing is per-source and never merges across sources.

# Direct-Write Gating

Mapped direct-write is enabled only when safe:
- Direct-write uses `read_into_at(src_offset, dest_va_offset, ...)` for explicit mapping.
- Programs containing any `StridedRun` disable direct-write (A1 gating).
- Current direct-write support targets CPU VA windows; GPU direct-write is deferred.

# Integration Points

This engine is used by:
- View selection execution (`ViewPlanSource`).
- Piece assembly and sealing (range-copy execution).
- Hashing of canonical and view ByteSpaces (`ByteRangeMappedSource`).
- `MaterializeIntoTarget` (range mapping into external targets).

Key code:
- Map builder: `core/store/materialization/dataplane/sources/byte_range_map_builder.{h,cc}`
- Mapper: `core/store/materialization/dataplane/sources/byte_range_mapped_source.{h,cc}`
- View source: `core/store/materialization/dataplane/view/view_plan_source.{h,cc}`
- Pump: `core/store/materialization/dataplane/runtime/pump.{h,cc}`

# Related Docs

- `docs/architecture/artifact-views-and-retrieval.md`
- `docs/architecture/view-replicas-and-assembly.md`
- `docs/architecture/api/materialization-flow.md`
- `docs/internals/canonical-index.md`
- `docs/internals/model-loading.md`
