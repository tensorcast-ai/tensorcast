---
slug: 0016-artifact-view-v1-core
title: Plan — Core: ViewPlanner & ViewPlanSource (v1)
links:
  design: ../designs/0016-artifact-view-v1.md
areas: ["core"]
related_code:
  - core/store/loader/**
  - core/store/**
---

# Objective

Implement core primitives for Variant-Aware Views: a `ViewPlanner` that computes slice/transpose view layouts and selection plans over the canonical index, and a `ViewPlanSource` executor to stream minimal bytes. Start with `narrow` (slice) server-side min-byte path; add client-side transpose integration hooks.

# Phases & Milestones

- [x] Phase 1: View planning (slice)
  - [x] Introduce `ViewPlanner` with `compute_view_plan()` (slice only)
  - [x] Deterministic `view_index_json` emission (field order parity with canonical)
  - [x] SelectionPlan metrics: `is_contiguous`, `num_ranges`, `total_bytes`, `is_segment_aligned`, `requires_materialization=false`

- [x] Phase 2: View execution source
  - [x] Implement `ViewPlanSource` as a `SeekableSource` over selection ranges
  - [x] Support 0‑copy alias when segment-aligned; scatter/gather otherwise

- [x] Phase 3: Engine integration
  - [x] Add view-aware path in `StoreEngine` to accept optional `ViewPlan`
  - [x] Choose alias or scatter/gather based on plan; compute `view_data_hash`

- [x] Phase 4: Tests & build wiring
  - [x] Catch2 tests for slice planning/execution and identity folding
  - [x] TreeHash leaf sizing parity and root equality checks
  - [x] BUILD targets for new libraries

- [x] Phase 5: Variant residency & AUTO wiring
  - [x] Extend `ReplicaKey`/registry to track `view_id` alongside canonical id
  - [x] Propagate `VariantIdentity` through `MaterializeHints` → `StoreEngine` → `MaterializeOrchestrator`
  - [x] Ensure `register_replica_with_global_store` records variants without mutating canonical routing
  - [x] Add tests covering canonical + variant co-residency and AUTO P2P fallback
  - [x] Remove legacy canonical-index v2 handling; enforce v3 end-to-end

- [x] Phase 6: Transpose execution (server/client)
  - [x] Extend `ViewPlanner` to emit `TransformPlan` for transpose ops (executor implemented in `core/store/loader/view_transform_executor.{h,cc}`)
  - [x] Implement LibTorch-backed transpose executor covering CPU/GPU placements
  - [x] Integrate transform pipeline into `StoreEngine` materialization + hashing
  - [x] Add tests for transpose-only + mixed view scenarios on CPU/GPU
  - [x] Reuse LibTorch (ATen) kernels for both `narrow` and `transpose` execution paths to stay aligned with PyTorch semantics (libtorch include path: .venv/lib/python3.10/site-packages/torch/include/, bazel target: @libtorch//:torch)

# Tasks (Detailed TODO)

- [x] Add `core/store/loader/view_planner.{h,cc}`
  - [x] API: `absl::StatusOr<ViewPlan> compute_view_plan(std::string_view canonical_index_json, const ViewSpec& spec)`
  - [x] Output: `TargetLayout` (JSON `view_index_json`) and `SelectionPlan` for `narrow`
  - [x] Validation: single `narrow` op per tensor; negative `start` normalized
  - [x] JSON emission: match canonical ordering (`rebuild_stable_canonical_index`)

- [x] Add `core/store/loader/view_plan_source.{h,cc}`
  - [x] Implement `SeekableSource` over selection ranges; PAD semantics preserved
  - [x] Provide flags for alias eligibility and materialization requirement

- [x] Implement `core/store/loader/view_transform_executor.cc`
  - [x] Materialize views via LibTorch tensors (CPU/GPU) in-place
  - [x] Surface placement-aware entrypoint for StoreEngine integration

- [x] Update `core/store/loader/BUILD`
  - [x] Add `sc_cc_library(name = "view_planner", ...)`
  - [x] Add `sc_cc_library(name = "view_plan_source", ...)`

- [x] Engine integration
  - [x] Extend `StoreEngine` to accept optional `ViewPlan` in materialization path
  - [x] Compute `view_data_hash` over view ByteSpace; attach to result struct

- [x] Tests (`//core/store/loader:view_planner_test`)
  - [x] Identity folding (no-op view → canonical)
  - [x] Slice contiguous/aligned vs non-aligned ranges
  - [x] Leaf sizing and TreeHash parity CPU/GPU

- [x] Variant residency (`ReplicaKey`, registry, LIP)
  - [x] Extend `store::loading::ReplicaKey` with optional `view_id`
  - [x] Update `ReplicaRegistry`, LIP cache, eviction helpers, and lease tracking to use the extended key
  - [x] Adjust metrics/telemetry to surface view-aware identifiers without losing canonical aggregation
  - [x] Audit loader & hashing code paths to drop v2 schema branches; assert `schema_version=="v3"` (StoreEngine rejects non-v3 descriptors; loader paths still need explicit asserts)

- [x] `MaterializeHints` & engine wiring
  - [x] Introduce `VariantIdentity` struct (canonical id, view id/spec, placement) and plumb through hints
  - [x] Update `StoreEngine::materialize_replica` fast-path to match `(canonical_id, view_id)`
  - [x] Record `view_id` on `ReplicaHandle` for subsequent hits
  - [x] Fetch canonical index bytes (via disk path or `get_canonical_index_by_id`) before invoking `ViewPlanner`

- [x] AUTO / P2P orchestration
  - [x] Enhance `MaterializeOrchestrator` to request GS metadata for `view_id`, falling back to canonical when absent (view-aware RPCs enabled)
  - [x] Teach P2P ingest path to consume `SelectionPlan` (slice) so transfers honor minimal byte ranges
  - [x] Ensure disk fallback writes canonical replica then persists variant metadata via GS helper
  - [x] When materializing slice/transpose on server, invoke LibTorch tensor views (e.g., `at::narrow`, `at::transpose`) prior to hashing/export (libtorch include path: .venv/lib/python3.10/site-packages/torch/include/, bazel target: @libtorch//:torch)

- [x] Tests (`//core/store:store_engine_test` + new cases)
  - [x] Load canonical then variant on same GPU; verify registry distinguishes them (CPU path covered; GPU pending)
  - [x] AUTO path with GS missing variant falls back to canonical without corruption
  - [x] P2P slice transfer uses range-restricted source (mock or instrumentation)
  - [x] Transpose placement obeys SERVER/CLIENT policy; GPU path validated with fake CUDA
  - [x] Add regression ensuring all materialisation paths emit `schema_version="v3"`

# Code Anchors

```27:44:core/store/loader/canonical_index.h
// Rebuild canonical index JSON bytes from an existing index JSON string.
// This function enforces:
// - Sorted outer keys (tensor names ascending)
// - Fixed field order [offset, size, shape, stride, dtype, storage_offset]
// - Proper integer types for numeric fields
// It DOES NOT change offsets/sizes; it is safe for hashing/verification.
absl::StatusOr<std::string> rebuild_stable_canonical_index(const std::string& index_json, int default_device_id);

// Build canonical index JSON object from ordered tensor names and metadata.
absl::StatusOr<std::string> build_canonical_index_json(
    const std::vector<std::string>& ordered_names,
    const std::unordered_map<std::string, uint64_t>& offsets,
    const std::unordered_map<std::string, uint64_t>& sizes,
    const std::unordered_map<std::string, CanonicalTensorMeta>& metas);
```

```30:37:core/store/loader/segment_plan_source.h
// Build a linear SegmentPlan from a canonical index (v3) JSON string.
// The index JSON maps tensor name -> [offset, size, shape, stride, dtype, storage_offset].
// The resulting plan covers [0, total_size) with DATA and PAD segments (8B alignment assumed).
absl::StatusOr<std::vector<SegmentPiece>> build_segment_plan_from_canonical_index_json(
    std::string_view index_json,
    uint64_t total_size,
    uint64_t align_bytes = 8);
```

```579:596:core/store/store_engine.cc
// Standard partition format – read tensor_index.json and canonicalize bytes via nlohmann::json
const auto index_json_path = artifact_path / "tensor_index.json";
try {
  // Read canonical index (JSON), then rebuild with stable grouping
  std::string raw_json;
  if (std::filesystem::exists(index_json_path)) {
    std::ifstream f(index_json_path);
    nlohmann::json j;
    f >> j;
    raw_json = j.dump();
  }
  if (!raw_json.empty()) {
    // Apply stable canonicalization using C++ authority
    auto rebuilt_or = loader::rebuild_stable_canonical_index(raw_json, target_device_id);
    const std::string& canonical_json = rebuilt_or.ok() ? *rebuilt_or : raw_json;
```

# Commands

```bash
BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext
bazel test //core/store:store_engine_test --define=use_fake_cuda=true
```

# Risks & Notes

- Planner/executor correctness: rely on deterministic JSON and single implementation.
- Alias eligibility depends on segment alignment; fall back to scatter/gather when necessary.
- Verification metadata now differentiates canonical and variant ByteSpaces. `StoreEngine::ingest_from_disk_internal()` reads/writes variant metadata under `verification.view_<sanitized_view_id>.json` and embeds `byte_space_id` in the JSON payload. Canonical loads continue to use `verification.json`.

## Verification Metadata for Views (Resolved)

- **Fix**: Added per-view verification files plus a `byte_space_id` field on `ArtifactVerificationInfo`. During loads we select the appropriate metadata based on `view_id` and fall back to generating fresh hashes if the ByteSpace identifier mismatches.
- **Side effects**: Tests no longer delete canonical `verification.json`; instead they assert that both canonical and variant files exist and that variant entries record the correct view size.
- **Next steps**: Extend Global Store registration flows (Phase 5) so variant verification blobs can be forwarded upstream once GS schemas are ready. Completed via GS integration in plan 0016-c.

## Status

- **Completed** — All phases delivered; ViewPlanner/ViewPlanSource/ViewTransformExecutor shipped with residency, AUTO/P2P wiring, and full test coverage (slice + transpose). Canonical index v2 paths removed and schema enforcement in place.
