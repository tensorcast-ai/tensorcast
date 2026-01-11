---
slug: 0019-store-engine-modularization
title: StoreEngine Modularization — Index/Verification/Eviction Services
areas: ["core"]
related_code:
  - core/store/store_engine.cc
  - core/store/store_engine.h
  - core/store/materialization/dataplane/**
  - core/store/components/**
links:
---

# Summary

`StoreEngine` has grown into a large monolith mixing orchestration with index canonicalization, verification, descriptor management, eviction, and device‑specific flows. This design modularizes the engine into focused services while preserving public behavior and APIs. The primary goal is to cut `core/store/store_engine.cc` significantly, remove duplicated logic, and create stable interfaces that reuse existing `loader/*` and `components/*` capabilities.

# Goals / Non‑Goals

Goals
- Reduce `store_engine.cc` code size and duplication with minimal behavior change.
- Separate concerns: index handling, verification/descriptor, eviction policy, and copy/load orchestration.
- Reuse `loader/*` primitives (canonical index, seekable sources, hashing) and keep them dependency‑light.
- Keep `components/*` focused on devices/registry/metrics/communication; no new cycles.
- Maintain API compatibility for `StoreEngine` public methods and status/logging semantics.

Non‑Goals
- No redesign of wire formats, protobufs, or Global Store APIs.
- No change to UMA/VS internals or `Replica` lifecycle semantics.
- No new external CLI/SDK surface; Python/daemon remain unchanged.

# Architecture & Interfaces

High‑level layering

```mermaid
flowchart LR
  SE[StoreEngine
  Orchestration] --> IDX[IndexService
  (loader/)]
  SE --> VER[VerificationService
  (loader/)]
  SE --> EVI[EvictionService
  (components/)]
  SE --> REG[ReplicaRegistry]
  SE --> DEV[DeviceManager]
  SE --> MET[MetricsCollector]
  SE --> COMM[CommunicationManager]
  SE --> GS[GlobalStoreClient]
```

Placement rationale
- IndexService and VerificationService live under `core/store/materialization/dataplane/` to reuse `canonical_index`, `source_hash`, `segment_plan_source`, and `view_plan_source` without depending on `components/*`.
- EvictionService lives under `core/store/components/` because it depends on `ReplicaRegistry`, `DeviceManager`, and `MetricsCollector`.
- `StoreEngine` becomes a thin coordinator that wires these services with the registry and device/comm/global store clients.

## IndexService (core/store/materialization/dataplane/metadata/index_reader.{h,cc})

Purpose: single entry to obtain canonical index bytes, index multihash, and total size across formats.

Proposed types

```cpp
struct IndexInfo {
  std::string canonical_index_json;
  std::string index_multihash;
  uint64_t total_size_bytes{0};
  bool is_safetensors{false};
};
```

Proposed functions

```cpp
absl::StatusOr<IndexInfo> read_from_artifact_dir(
    const std::filesystem::path& artifact_path,
    int target_device_id);

absl::StatusOr<IndexInfo> canonicalize_from_raw_json(
    std::string raw_json,
    int target_device_id);

absl::StatusOr<IndexInfo> build_from_safetensors(
    const std::vector<std::filesystem::path>& st_files);
```

Notes
- Internally reuse: `loader/canonical_index.*`, `loader/safetensors_util.*`.
- Consolidates currently duplicated logic in `ingest_from_disk_internal` and P2P paths.

## VerificationService (core/store/materialization/dataplane/verification/verification_utils.{h,cc})

Purpose: compute data/view hashes, reuse/generate verification metadata, and write descriptors when needed.

Proposed types

```cpp
struct MemoryView {
  common::memory::MemoryLocation location; // CPU or GPU
  void* base_ptr{nullptr};
  uint64_t size_bytes{0};
  std::optional<int> gpu_device_id; // when GPU
};

struct ViewHashResult {
  std::string multihash;
  std::vector<std::vector<uint8_t>> leaf_digests; // tree leaves
};
```

Proposed functions

```cpp
absl::StatusOr<std::string> compute_data_multihash(const MemoryView& mem);

absl::StatusOr<ViewHashResult> compute_view_tree_hash_and_leaves(
    loader::SeekableSource& base_source,
    uint64_t total_size,
    size_t leaf_chunk_bytes);

absl::Status reuse_or_generate_verification_json(
    const std::filesystem::path& artifact_dir,
    std::string expected_byte_space_id,
    const MemoryView& mem);

absl::Status write_descriptor_if_absent(
    const std::filesystem::path& artifact_dir,
    std::string_view index_multihash,
    std::string_view data_multihash,
    uint64_t total_size_bytes,
    std::string_view encoding);

absl::StatusOr<std::vector<std::vector<uint8_t>>> compute_canonical_leaf_digests(
    const MemoryView& mem,
    absl::Span<const uint64_t> leaf_indices,
    size_t chunk_bytes);
```

Notes
- Internally reuse: `loader/source_hash.*`, `segment_plan_source.*`, `view_plan_source.*`, and `core/common/artifact_verification.*`.
- Keeps descriptor writeback and verification.json reuse in one place; `StoreEngine` simply calls it.

## EvictionService (core/store/components/eviction_service.{h,cc})

Purpose: centralize eviction policies and device‑aware freeing logic.

Proposed functions

```cpp
absl::Status evict_for_gpu(
    components::ReplicaRegistry& registry,
    components::DeviceManager& device_manager,
    components::MetricsCollector& metrics,
    int device_id,
    size_t required_bytes);

absl::Status evict_for_cpu(
    components::ReplicaRegistry& registry,
    components::MetricsCollector& metrics,
    size_t required_pinned_pool_bytes);
```

Notes
- Merges the TU‑local `try_evict_gpu_memory_impl` with `StoreEngine::try_evict_memory_for_replica` semantics.
- `StoreEngine` invokes this once in load/retry paths, removing duplicate inline eviction code.

## StoreEngine changes (thin orchestration)

Key callsites become:
- Disk load: `IndexService` → optional `ViewPlanner` → `Replica::ensure_loaded_async` → `EvictionService` (if needed) → `VerificationService` (digest/verification/descriptor).
- P2P load: Obtain canonical index bytes (from Global Store) → `IndexService::canonicalize_from_raw_json` → same flow.
- Variant registration commit: `VerificationService::compute_view_tree_hash_and_leaves` and `compute_canonical_leaf_digests`; `StoreEngine` maps results into `global_store::v1::LeafWrite` before calling `GlobalStoreClient`.
- COPY_ONLY: optional `CopyService` (follow‑up) to encapsulate GPU→GPU instance copy using `Replica::copy_from`.

Small utilities to move out of `store_engine.cc`
- Alignment helpers, `sanitize_view_id_for_filename`, view spec JSON construction.

# Invariants & Error Model

Compatibility invariants
- Public `StoreEngine` methods and enums are unchanged.
- Status codes and messages remain consistent in normal and error paths (InvalidArgument, FailedPrecondition, NotFound, ResourceExhausted, DeadlineExceeded, DataLoss, etc.).
- Timeout behavior: 0 means “wait indefinitely” (retained via a shared helper used by both disk and P2P paths).
- Descriptor writeback for safetensors (when absent) follows current rules and encoding remains `json`.
- Verification metadata reuse respects `byte_space_id` matching; mismatch regenerates metadata (same as today).
- COPY_ONLY requires a GPU target and a GPU source on the same host; view hash presence/omission unchanged.

Observability
- Preserve existing log levels and messages where user‑visible; internal debug logs may be reduced but semantics unchanged.
- Retain OpenTelemetry spans/attributes in P2P path.

# Trade‑offs & Risks

Pros
- Large reduction in `store_engine.cc` code size; duplicated logic gathered into testable units.
- Clear layering with no new cycles: `loader/*` remains dependency‑light; `components/*` focuses on runtime/devices.
- Easier to evolve hashing/verification or eviction independently.

Risks
- Moving logic can change subtle file‑system edge cases; mitigate with targeted tests on malformed/missing descriptor/index.
- Include paths and BUILD files need careful updates to avoid accidental transitive dependencies.
- COPY_ONLY path depends on `InlineBufferSource` semantics; ensure behavior remains unchanged until loader support is expanded.

Mitigations
- Phase the refactor: first pure extractions (mechanical), then dedup, finally small orchestration cleanups.
- Add Catch2 unit tests for the new services reusing existing test data and helpers.

# Compatibility & Acceptance Criteria

Must‑pass checks
- Bazel and `uv run setup.py build_ext` builds succeed.
- All existing C++ tests in `core/store/materialization/dataplane/` and `components/` pass in both fake and real CUDA modes.
- Python tests that rely on `StoreEngine` behavior pass without changes.

Targeted behavioral tests
- IndexService: safetensors and partition index canonicalization and total size derivation.
- VerificationService: FULL_DIGEST CPU/GPU, verification.json reuse/mismatch regeneration, descriptor writeback.
- EvictionService: device‑scoped GPU eviction with required bytes accounting; CPU pinned pool eviction preserves metrics.
- Variant registration: canonical and variant leaf digests stable for the same inputs.

# References

- `core/store/store_engine.cc`, `core/store/store_engine.h` (current implementation)
- `core/store/materialization/dataplane/metadata/canonical_index.*`, `core/store/materialization/dataplane/metadata/safetensors_util.*`, `core/store/materialization/dataplane/metadata/source_hash.*`, `core/store/materialization/dataplane/sources/segment_plan_source.*`, `core/store/materialization/dataplane/view/view_plan_source.*`
- `core/store/components/{replica_registry,device_manager,metrics_collector}.h`
- 0007‑content‑addressed‑artifact‑id, [api-design](../architecture/api/api-design.md), 0016‑artifact‑view‑v1, 0018‑artifact‑view‑registration
