---
slug: 0062-safetensors-canonical-bytespace
title: Safetensors Canonical ByteSpace Unification (Plan)
links:
  design: ../designs/0062-safetensors-canonical-bytespace.md
areas: ["core", "daemon", "sdk"]
related_code:
  - core/store/materialization/dataplane/metadata/index_reader.*
  - core/store/materialization/dataplane/metadata/safetensors_util.*
  - core/store/materialization/dataplane/view/view_planner.*
  - core/store/materialization/dataplane/view/view_plan_source.*
  - core/store/materialization/dataplane/sources/byte_range_map_builder.*
  - core/store/materialization/dataplane/sources/byte_range_mapped_source.*
  - core/store/materialization/dataplane/sources/byte_range_program.*
  - core/store/materialization/runtime/pipeline/metadata_stage.*
  - core/store/materialization/runtime/pipeline/source_adapter.*
  - core/store/runtime/ingestion/materialization_facade.*
  - core/store/store_engine_options.*
  - proto/tensorcast/config/v1/daemon_config.proto
created: 2026-02-02
last_updated: 2026-02-03
---

# Summary

Implement safetensors ByteSpace unification by splitting **canonical identity/layout** from **disk source layout**, then executing disk loads with a source-ordered window schedule so materialization fills the coalesced canonical ByteSpace without data conversion.

# Baseline (Pre-0062)

Grounding in current code paths:
- Safetensors index building uses payload offsets as canonical (`core/store/materialization/dataplane/metadata/index_reader.cc`, `BuildCanonicalIndexFromSafetensors`).
- Disk sources hide safetensors headers and expose payload-only Source ByteSpace (`SafetensorsSource`, `MultiSafetensorsSource` from design-0009).
- `MaterializeIntoTarget` builds a ByteRangeMap from canonical index only (`build_byte_range_map_from_canonical_index_json`), so it assumes `src_offset == dst_offset` and reads in canonical order (`core/store/runtime/ingestion/materialization_facade.cc`).
- Region-backed / `DeferredLoader(packing="byte_space")` require coalesced canonical layout; safetensors violates this and triggers `INVALID_ARGUMENT` (see SDK validation in `tensorcast/api/store/materialization.py`).
- View planning is defined over canonical index and should not depend on disk layout (`core/store/materialization/dataplane/view/view_planner.cc`, `ViewPlanSource`).

# Phases & Milestones

- [x] Phase 1: Split safetensors identity vs source layout
  - [x] Extend `loader::IndexInfo` with `source_index_json` (source layout) and `source_total_size_bytes` in `core/store/materialization/dataplane/metadata/index_reader.h`.
  - [x] Rename/wrap the current safetensors index builder as `BuildSourceIndexFromSafetensors` in `core/store/materialization/dataplane/metadata/safetensors_util.{h,cc}`.
  - [x] Add a canonical-from-source helper (coalesced layout) in a new module (preferred) or under `canonical_index.{h,cc}`:
    - Coalesce by sorted tensor names + 8-byte alignment (canonical identity must not depend on safetensors payload offsets).
    - Safetensors: `storage_offset` remains zero for all tensors.
  - [x] Align SDK "coalesced canonical" validation with the intended invariant:
    - Update `tensorcast/api/store/materialization.py` to validate element-size alignment of `segment_offset`
      (required for `DeferredLoader(packing="byte_space")`), rather than requiring
      `segment_offset == storage_offset * elem_bytes`.
    - Update existing unit tests that construct fake canonical index bytes (e.g., `tests/python/test_deferred_loader.py`)
      so they no longer encode the old, overly strict constraint.
  - [x] Update `index_reader.cc` to return:
    - `canonical_index_json` = coalesced canonical layout (identity)
    - `source_index_json` = source layout (disk planning only)
    - `index_multihash` computed from the coalesced canonical index bytes.
  - [x] Plumb source layout into disk-only hints:
    - [x] Add `source_index_json` to `loading::DiskMetadata` in `core/store/materialization/contracts/loading_spec.h`.
    - [x] Populate `hints.disk_metadata->source_index_json` in `core/store/materialization/runtime/pipeline/source_adapter.cc` / `metadata_stage.cc` for disk safetensors.

- [x] Phase 2: Canonicalize disk sources (canonical→source mapping)
  - [x] Add `build_byte_range_map_from_canonical_and_source_index_json(...)` in `core/store/materialization/dataplane/sources/byte_range_map_builder.{h,cc}`.
    - Strict validation: name set match, per-tensor size match, bounds checks; return `INVALID_ARGUMENT` on mismatch.
  - [x] Update map caching in `core/store/runtime/ingestion/materialization_facade.cc`:
    - Key includes canonical index hash + source layout hash to avoid collisions.
  - [x] Introduce a canonicalizing wrapper for disk sources when `disk_metadata.source_index_json` is present:
    - Wrap the base disk `SeekableSource` with the canonical→source map so the wrapper presents Canonical ByteSpace bytes.
  - [x] Keep view planning unchanged:
    - View specs continue to plan over canonical index bytes.
    - View execution uses `ViewPlanSource` on top of the canonicalized disk source (no `ViewPlanner` signature change).

- [x] Phase 3: Source-ordered disk window scheduling (byte-range engine)
  - [x] Implement source-ordered window execution as a byte-range strategy (not safetensors-specific):
    - Build per-source windows from map segments sorted by `src_offset`.
    - Merge gaps with bounded amplification and apply a staging cap; scatter into canonical destinations.
    - Ensure byte semantics are unchanged vs. destination-ordered execution (only scheduling differs).
  - [x] Add config fields to `StoreEngineOptions::ByteMappingConfig` in `core/store/store_engine_options.h`:
    - `disk_source_ordered_read`, `disk_source_merge_max_gap_bytes`, `disk_source_merge_max_amplification`, `disk_source_prefetch_depth`.
  - [x] Extend daemon config proto `proto/tensorcast/config/v1/daemon_config.proto` (unified runtime config) with the new fields and regenerate protobufs (`bash tools/build_proto_python.sh`).
  - [x] Wire config parsing in `daemon/app/server_main.cc`, `core/common/config/daemon_config_io.cc`,
    `tensorcast/daemon_runtime_config.py`, and `tensorcast/cli_utils/config.py`.
  - [x] Gate source-ordered scheduling by config + presence of `disk_metadata.source_index_json`.

- [x] Phase 4: Map composition for views (performance refinement)
  - [x] Add a `compose_byte_range_maps` helper so `view→canonical` can be composed with `canonical→source` into `view→source`.
  - [x] Use composition when a view is requested over a disk source with `source_index_json`, avoiding nested mapping sources and improving window fusion.

- [x] Phase 5: Tests and verification
  - [x] Add index conversion tests in `core/store/materialization/dataplane/metadata/tests/` (single + multi-file safetensors).
  - [x] Add mapping validation tests in `core/store/materialization/dataplane/sources/tests/` for canonical+source maps and composition.
  - [x] Add execution tests covering source-ordered windows (merge policy, amplification bounds, staging cap).
  - [x] Add Python test coverage for `DeferredLoader(packing=\"byte_space\")` on safetensors (implemented via safetensors index bytes in `tests/python/test_deferred_loader.py`).
  - [ ] Validate metrics (`tc_byte_range_base_read_calls_total`, window amplification ratio) and ensure view performance does not regress for `narrow`.

# Acceptance Checks

- Safetensors disk loading produces coalesced canonical index bytes (identity/layout) and a separate source layout payload.
- Region-backed `MaterializeIntoTarget` succeeds for safetensors with `packing="byte_space"`.
- Views remain canonical: `view_id` depends only on canonical index bytes + ViewSpec; view planning signature remains unchanged.
- View materialization matches outputs for `narrow` / `transpose` views.
- Source-ordered window scheduling reduces disk read calls and improves effective sequential read size on safetensors.

# Test Plan

- C++ unit tests in `core/store/materialization/dataplane/metadata/tests/` for index conversion.
- C++ unit tests for canonical→source mapping, composition, and source-ordered execution (`core/store/materialization/dataplane/sources/tests/`).
- Python tests for `DeferredLoader` and safetensors loading (`tests/python/test_safetensors_loading.py`).

# Rollout / Backout

- Rollout guarded by `StoreEngineOptions::ByteMappingConfig.disk_source_ordered_read`.
- If issues occur, disable source-ordered scheduling and fall back to destination-ordered execution (semantics unchanged).

# Risks and Mitigations

- Incorrect mapping causing silent corruption: enforce strict name/size checks and add regression tests.
- Nested mapping + view performance regression: implement map composition and add targeted view perf tests.
- Source-window scheduler complexity: cap amplification and add unit tests for merge logic.

# Owner Checklist

- [x] Update related docs if external behavior changes (region-backed safetensors).
- [x] Ensure unified runtime config is updated per `docs/designs/0004-unified-runtime-config.md`.
- [x] Validate naming compliance for new C++ APIs.

# Status Updates

- 2026-02-03: Added storage_root absolute-path validation tests and clarified daemon storage_path semantics in docs.
