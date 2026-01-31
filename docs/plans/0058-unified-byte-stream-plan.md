---
slug: unified-byte-stream-plan
title: Unified Byte Range Mapping (Implementation Plan)
areas: ["core", "daemon"]
status: implemented
created: 2026-01-27
last_updated: 2026-01-29
related_code:
  - core/store/materialization/contracts/view/view_plan.h
  - core/store/materialization/dataplane/view/view_plan_source.cc
  - core/store/materialization/dataplane/sources/byte_range_map_builder.cc
  - core/store/materialization/dataplane/sources/byte_range_mapped_source.cc
  - core/store/materialization/dataplane/sources/byte_range_program.cc
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/runtime/metadata/registration_backend.cc
links:
  design: ../designs/0058-unified-byte-stream-plan.md
---

# Objective

Implement the unified `ByteRangeMap` IR and `ByteRangeProgram` execution path with hard cutover, ensuring no performance regression
relative to existing view, assembly, and hashing paths.

# Current State & Grounding (pre-implementation)

- Linear mapping was duplicated across `SegmentPiece`, `AssemblySegment`, and `SelectionPlan::Range`.
- Execution logic was duplicated in `LinearizedGpuPlanSource`, `AssemblySource`, `PlanBackedSeekableSource`, and
  `ViewPlanSource`.
- Strided execution and block caching lived in
  `core/store/materialization/dataplane/view/view_plan_source.cc`.
- Direct-write assumed identity mapping (`src_offset == dest_va_offset`), blocking direct-write for view/assembly.
- Key call sites:
  - View execution: `core/store/materialization/dataplane/view/view_plan_source.cc`
  - Canonical linearization + hashing: `core/store/materialization/dataplane/sources/byte_range_map_builder.cc`,
    `core/store/runtime/metadata/registration_backend.cc`
  - Assembly and materialize-into-target: `core/store/runtime/ingestion/materialization_facade.cc`

# Phases & Milestones

- [x] Phase 0: Capability surfaces and config (pre-work)
  - [x] Milestone: Lock in naming (`ByteRangeMap` / `ByteRangeProgram`) and add contracts skeleton + BUILD wiring
  - [x] Milestone: Make `SeekableSource` a sized interface (`total_bytes`) and enforce the global short-read contract
    (short read = DataLoss except EOF clamping) for both `read_at` and `read_into_at`
  - [x] Milestone: Replace legacy identity direct-write with mapped direct-write
    (`supports_direct_write_at` / `read_into_at(src_offset, dest_va_offset, ...)`)
  - [x] Milestone: Update `pump_ranges` to use `supports_direct_write_at/read_into_at` only (no identity fallback)
  - [x] Milestone: Add `engine.byte_mapping` config under unified runtime config (`docs/designs/0004-unified-runtime-config.md`)
  - [x] Milestone: Define builder/normalization contract: builders emit explicit `kPad` segments and full coverage
    `ByteRangeMap`s; normalization rejects gaps as `INVALID_ARGUMENT`. Surface missing coverage as `UNAVAILABLE` with RPC
    detail `PartialCoverageDetail{HashSpaceRef, missing_ranges}` (units: bytes) and do not emit a map on coverage failure.
    For assembly, include canonical missing ranges (required) and mapped view/output missing ranges when assembling a view
    via selection-only mapping; coalesce and bound the range list to avoid oversized status details.
  - [x] Milestone: Define unified `tc_byte_range_*` metrics and remove view-specific naming
- [x] Phase 1: IR normalization and builders
  - [x] Milestone: Implement normalization + validation utilities (sort, merge, PAD fill, overlap/gap rejection for raw)
  - [x] Milestone: Add canonical index -> `ByteRangeMap` builder (replacement for segment plan)
  - [x] Milestone: Add multi-source assembly -> `ByteRangeMap` builder (replacement for `AssemblySegment`)
- [x] Phase 2: Compiler and execution primitives
  - [x] Milestone: Implement `ByteRangeCompiler` and `ByteRangeProgram`
  - [x] Milestone: Port strided heuristics + block cache behavior from `ViewPlanSource`
  - [x] Milestone: Implement `ByteRangeMappedSource` (PAD + contiguous + strided runs)
  - [x] Milestone: Implement A1 mapped direct-write eligibility: enable only when program contains no `StridedRun`
  - [x] Milestone: Add `GpuMemorySource` and `CpuMemorySource`
  - [x] Milestone: Implement mapped direct-write in `ByteRangeMappedSource` for PAD + contiguous runs (CPU VA grant only)
- [x] Phase 3: View path migration
  - [x] Milestone: `ViewPlanner` emits `ByteRangeMap` and selection metadata only
  - [x] Milestone: `ViewPlanSource` becomes a thin wrapper over `ByteRangeMappedSource` (no run-building)
  - [x] Milestone: View hashing paths use the unified mapper + program
- [x] Phase 4: Assembly + ingest path migration
  - [x] Milestone: Replace `AssemblySource` with `ByteRangeMappedSource`
  - [x] Milestone: Replace `PlanBackedSeekableSource` in materialize-into-target
  - [x] Milestone: Enable assembly direct-write into UMA CPU VA windows when available
- [x] Phase 5: Hashing path migration
  - [x] Milestone: Canonical hash uses `GpuMemorySource` + `ByteRangeMap`
  - [x] Milestone: View hash uses `ByteRangeMap` + compiled program
- [x] Phase 6: Cleanup and docs
  - [x] Milestone: Remove `SegmentPiece`, `AssemblySegment`, `SelectionPlan::Range`, and legacy adapters
  - [x] Milestone: Update module READMEs and dataplane docs (contracts + dataplane)

# Tasks

- Add contracts in `core/store/materialization/contracts/byte_range/` and update BUILD deps.
- Make `loader::SeekableSource` sized (`total_bytes`) and enforce the global short-read contract in all implementations.
- Replace legacy identity direct-write with `supports_direct_write_at` / `read_into_at`; update `pump_ranges`.
- Implement A1 direct-write eligibility in `ByteRangeMappedSource`: disable mapped direct-write when program contains any `StridedRun`.
- Add `engine.byte_mapping` config via unified runtime config; wire compiler thresholds from config (no env vars).
- Implement normalization + fingerprinting for `ByteRangeMap` with the “no implicit PAD” rule (builders emit `kPad`;
  normalization rejects gaps).
- Implement compiler + program in `core/store/materialization/dataplane/sources/`.
- Add a bounded compiled-program cache keyed by `(map_fingerprint, config_fingerprint)` (e.g., fixed-size LRU).
- Implement `ByteRangeMappedSource`, `GpuMemorySource`, `CpuMemorySource`.
- Update `ViewPlanner` (`core/store/materialization/dataplane/view/view_planner.cc`) to emit `ByteRangeMap`.
- Rewrite `ViewPlanSource` to execute compiled program; remove local run derivation.
- Replace assembly plan and mapper in `core/store/runtime/ingestion/materialization_facade.cc`.
- Replace canonical hashing and view hashing usage in `core/store/runtime/metadata/registration_backend.cc`.
- Update `RemoteKeySource` direct-write path to support mapped direct-write (`read_into_at(src_offset, dest_va_offset, ...)`) for non-identity mappings.
- Replace view-specific metrics with unified `tc_byte_range_*` metrics (no legacy compatibility required).
- Remove old IR structs and adapters, update README docs.

# Status (2026-01-29)

- Implementation complete; legacy adapters removed and ByteRangeMap/Program paths are now the single execution pipeline.
- Existing unit tests were updated for the new ByteRangeMap + sized `SeekableSource` API; new targeted tests listed below remain follow-ups.

# Test / Rollout / Backout

## Test Matrix (per path)

### View Path
- Unit tests:
  - Add `core/store/materialization/dataplane/sources/tests/byte_range_compiler_strided_test.cc` to validate
    strided run detection, block sizing, and cache behavior.
  - Update `core/store/materialization/dataplane/view/tests/view_plan_source_test.cc` to assert equivalence with
    compiled execution plan outputs (range coverage + PAD behavior).
  - Add targeted tests for short-read enforcement (short read = DataLoss except EOF clamping).
- Existing regression anchors:
  - `core/store/materialization/dataplane/view/tests/view_planner_test.cc`

### Assembly Path
- Unit tests:
  - Add `core/store/runtime/ingestion/tests/byte_range_assembly_map_test.cc` to validate multi-source
    `ByteRangeMap` assembly mapping and gap detection.
- Integration coverage:
  - Exercise `MaterializationFacade::assemble_from_pieces` on a synthetic plan using `ByteRangeMappedSource`.
  - Add a direct-write integration test covering mapped direct-write into a `DirectWriteCapable` sink (PAD + contiguous).

### Hash Path
- Unit tests:
  - Add `core/store/materialization/dataplane/sources/tests/byte_range_hash_source_test.cc` to validate canonical
    and view hashing streams are identical to prior results for PAD/data patterns.
- Integration coverage:
  - Validate `registration_backend` hashing paths for canonical + view (CPU and GPU) use the unified mapper.

- Tests: add unit coverage for normalization, compiler run detection, multi-source assembly mapping, and strided runs.
- Rollout: hard cutover in a single change set; no compatibility phase.
- Backout: not supported by design.

# Risks & Tracking

- Risk: strided heuristics not preserved. Mitigation: port and reuse current thresholds and caching logic.
- Risk: extra `read_at` calls in contiguous paths. Mitigation: aggressive run merging in compiler.
- Risk: mapped direct-write breaks identity-only assumptions. Mitigation: keep identity API fallback and gate mapped
  direct-write on capability + config kill-switch.
