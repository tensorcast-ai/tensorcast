## RFC-0007 Content-Addressed Model ID (Concise)

### 1. Background and Goals

- Problem: The current `model_id` depends on external naming (path/source). Identical content across disk, memory, and P2P is hard to aggregate and address consistently.
- Goal: Adopt content addressing. Combine a structural fingerprint with a data fingerprint to produce a stable `model_id` so that disk, memory, and P2P replicas converge under the same ID and can be routed uniformly.
- Success criteria:
  - Identical content yields the same `model_id` regardless of origin.
  - Keep millisecond-level verification for load and P2P fast paths (KEY_POINTS/SEGMENT).
  - Controlled cost during compute/register phases (GPU-first, CPU fallback).

### 2. Design Overview

- ID spec (mi2): `model_id = "mi2:" + index_multihash + ":" + data_multihash` (Multihash + Multibase32; self-describing algorithm; default `sha2-256`).
- Structural fingerprint: Multihash over the Canonical Index bytes to produce `index_multihash`.
- Data fingerprint: Tree hash over the normalized linear data stream; Multihash applied to the Merkle root to produce `data_multihash`.
- Unified routing: Global Store uses `model_id` as the primary key to aggregate replicas; Local Store selects the optimal source by `model_id` (memory > local disk > remote).

### 3. Specification (Normative)

#### 3.1 Model ID Format

- `model_id = "mi2:" + index_multihash + ":" + data_multihash`
- Multihash: default `sha2-256`; supports future algorithm evolution via self-description.
- Multibase: default base32.

#### 3.2 Canonical Index (Input to Structural Fingerprint)

- Encoding: Prefer Canonical CBOR; allow Strict Canonical JSON during transition (ordered keys, fixed fields).
- Top-level keys: strictly ascending `tensor_name`.
- Record field order (fixed): `offset, size, shape, stride, dtype, storage_offset`.
- 8-byte alignment: consistent with RFC-0006.
- Authority: Canonical Index is generated in the C++ core (avoid cross-language divergence).

Stable grouping and layout (replacing unstable pointer-based grouping):

- Grouping key: `(dtype_code, device_id, group_key)` where `group_key = H(sorted(tensor_names_in_group))` and `H` matches the algorithm used for `index_multihash`.
- Ordering: groups ascending; within a group, `tensor_name` ascending; maintain 8B alignment.

#### 3.3 Normalized Linear Data Stream (Input to Data Fingerprint)

- Definition: The byte sequence read sequentially across the logical range `[0, total_size)` according to the Canonical Index’s coalesced layout.
- Sources (uniformly abstracted as `SeekableSource`):
  - Disk: read partitions in file-name order (`FilePartitionSource`).
  - P2P: stream via `RemoteKeySource`.
  - Memory: read directly from contiguous memory allocated during Begin (coalesced), GPU-preferred.

#### 3.4 Fingerprint Computation

- `index_multihash = MULTIHASH(canonical_index_bytes)`.
- `data_multihash = MULTIHASH(TREE_HASH(stream_bytes(0..total_size)))`.
- Tree hash: 1–16 MiB chunking, `sha2-256` leaves/root; prefer GPU parallelization, with CPU/PCIe fallback.
- Runtime verification: load/P2P use KEY_POINTS/SEGMENT; FULL verification is optional.

### 4. System Integration

#### 4.1 Global Store (Keyed by `model_id`)

- Tables:
  - `models(model_id PK, index_multihash, data_multihash, schema_version, encoding, hash_params_json, created_at, ...)`
  - `model_indices(index_multihash PK, schema_version, encoding, size_bytes, index_data BLOB, created_at, ...)`
  - `model_replicas(id PK, model_id FK, source_type ENUM('DISK','MEMORY','P2P'), location|disk_path, device_id, created_at, ...)`
- RPCs: `RegisterModelReplica(model_id, ...)`, `GetModelInfoById(model_id)`, `GetModelIndex(index_key)`.

#### 4.2 Local Store / StoreDaemon

- Begin: create in-memory-only model, allocate device memory, return `registration_id + cuda_ipc_handle`.
- Commit: compute `index_multihash` and `data_multihash` inside daemon/core, build `mi2`, return `ModelDescriptor`; register replica in GS at the same time.
- Prepare (single entry point): `prepare(DeviceKey, mode, hints)` supports `hints.model_id` (content addressing) and `hints.disk_path` (explicit disk). Never interpret `mi2:` as a filesystem path.

#### 4.3 Python API

- `register_tensor_dict(state_dict, ...) -> (state_dict, commit_info)`; `commit_info.model_id` starts with `mi2:`.
- Helpers: `generate_model_id_from_state_dict(...)`, `generate_model_id_from_path(...)` (for audit and migration).

### 5. ModelDescriptor (Returned by Commit)

- Fields (minimal set):
  - `model_id` (`mi2:...`), `index_multihash`, `data_multihash`
  - `schema_version`, `encoding` (recommend `cbor`)
  - `total_size`, `hash_params` (e.g., `chunk_size`, `fanout`)

- On-disk file: `model_descriptor.json`
  - Save side (`save_dict`/`save_tensors`) must write `model_descriptor.json` to the target directory alongside `tensor_index.(cbor|json)` (CBOR recommended).
  - Applies to standard data format directories (see “Data File Format Specification”) and `safetensors` directory structures.

### 6. Rollout Plan (Phases)

- A: Canonical Index (CBOR) with stable grouping/layout; define the linear space.
- B: Tree hash and Multihash wrapper; GPU-first, CPU fallback.
- C: Commit returns `ModelDescriptor`; align Python/Proto.
- D: GS/DB: key by `model_id`; `model_replicas.model_id` as FK; switch RPCs.
- E: CheckpointStore: `prepare(..., model_id=...)` routes via GS; light runtime verification; migration tool backfills IDs.

### 7. Current Status (Code Synced 2025-08-20)

- Completed (matches current repo implementation):
  - C++ (hashing and pipeline):
    - Added `core/store/loader/source_hash.{h,cc}`: unified tree hashing over `SeekableSource` (4 MiB leaves → Merkle root → multibase32 multihash).
    - Added `core/store/loader/disk_dir_hash.{h,cc}`: data multihash for standard partitioned directories (`tensor.data*`) via `FilePartitionSource` and the unified pipeline.
    - `core/common/model_hash.{h,cc}` now retains only GPU buffer data hashing (`compute_data_multihash_from_gpu(...)`) and Index multihash (`compute_index_multihash(...)`); directory hashing moved to the loader layer.
    - Unified P2P and memory (CPU/GPU) sources:
      - P2P uses `RemoteKeySource`; memory uses a minimal local source wrapper inside `source_hash.cc` (no new public classes/targets).
      - Convenience functions: `loader::compute_data_multihash_from_cpu_memory(...)`, `loader::compute_data_multihash_from_gpu_memory(...)` run through the same streaming pipeline as disk/P2P.
  - PyBind: new unified interfaces in `scstore/_C`:
    - `save_model_to_disk(tensor_names, tensor_data, meta_state_dict, path, config) -> descriptor`
    - `inspect_or_generate_descriptor(path) -> descriptor`
  - Python:
    - `save_dict(...)` now calls `save_model_to_disk(...)` to emit `tensor_index.(cbor|json)` + `model_descriptor.json` in one shot and returns the `descriptor`; removed redundant Python-side multihash/tree-hash/base32 logic.
    - `load_dict(...)` calls `inspect_or_generate_descriptor(...)` after loading; if missing, computes and writes `model_descriptor.json`; retains background KEY_POINTS/SEGMENT verification.
    - Removed `load_dict_pure_local(...)`; tests call `load_dict(...)` directly.
  - Tools and tests:
    - `scstore/tools/backfill_descriptor.py` delegates to `_C.inspect_or_generate_descriptor`; removed the redundant `--write-index` placeholder.
    - `core/testing/common.cc` and C++ tests call the unified loader pipeline (`loader::compute_data_multihash_from_disk_dir`).
    - Examples and tests (`examples/*`, `tests/python/*`) updated to the new interfaces.

- Next (TODO):
  - Canonical Index (C++ authority):
    - Generate Canonical Index in C++ (prefer CBOR; allow Strict Canonical JSON short-term) with stable grouping + 8B alignment; remove Python-side assembly and sorting entirely.
  - Performance and memory optimization:
    - Replace single-stream CPU reduction with multi-stream/multi-core GPU tree hashing; keep CPU/PCIe fallback.
    - Change leaf-digest "collect-all → reduce" into streaming reduction to reduce peak memory.
  - Standard-partition gate and strong verification (DiskLoader):
    - Gate is implemented: standard directories must contain `model_descriptor.json` and `tensor_index.(cbor|json)`; otherwise return `FailedPrecondition(MODEL_DESCRIPTOR_REQUIRED)` (`DiskLoader::initialize`).
  - Commit/Prepare path hardening:
    - Unify Canonical Index and `mi2` generation fully inside Commit (partially landed); refine AUTO routing and GS-side dedupe/aggregation; add strict verification switch (FULL/Merkle proof).
  - `safetensors` Canonical Index (C++):
    - Finish `BuildCanonicalIndexFromSafetensors(...)` (sorted outer keys, fixed field order, `storage_offset=0`, row-major stride) for backfilling directories missing descriptor/index.
  - Docs/Examples/CLI:
    - Align all to `save_model_to_disk` / `inspect_or_generate_descriptor` and `prepare(..., model_id=..., disk_path=...)`; add backfill guidance and error code table.

### 8. Key Trade-offs

- `tensor_index_key` alone cannot distinguish "same layout, different data", so it is not a suitable unique content identifier.
- Runtime continues to use KEY_POINTS/SEGMENT; `data_multihash` is computed only at save/register/Commit.
- Stable grouping must be environment-independent; otherwise cross-source consistency breaks.

### 9. Risks and Mitigations

- Historical models with unstable ordering: treat as different `model_id`s. Provide batch normalization and backfill tools.
- Commit overhead: support GPU-first/CPU fallback and async backfill; block only when strong consistency is required.
- Negative stride / mixed dtype / multi-device: covered by v2 semantics; stable grouping and 8B alignment are hard constraints.

### 10. Examples

```python
from scstore.torch_util import register_tensor_dict

sd, info = register_tensor_dict(model.state_dict(), "name", device_id=0)
assert info["model_id"].startswith("mi2:")
```

```python
# Loading (prefer content addressing)
store.prepare(device_key, mode="AUTO", model_id=info["model_id"])  # GS routing
```

### 11. Consistency with Verification System

- `data_multihash` is part of the unique identity and is computed only during save/register/Commit.
- Runtime uses KEY_POINTS/SEGMENT by default; FULL verification or Merkle partial proof can be enabled when necessary.

### 12. Compatibility and Migration

- During coexistence: `disk_path` can explicitly trigger disk loading; `mi2:` is only a content-addressed ID, never a path.
- Standard data format directories: loader requires `model_descriptor.json` and `tensor_index.(cbor|json)`; legacy directories must be processed by the migration tool first.
- `safetensors` directories: if `model_descriptor.json` exists, treat it as authoritative and verify consistency; if missing, allow computing and writing `model_descriptor.json` and Canonical Index after loading (when write permissions are available).


### 13. Disk Save and Load (disk_loader) Specification

#### 13.1 Save Side (`save_dict` / `save_tensors`)

- Must output:
  - `tensor_index.cbor` (or Strict Canonical JSON: `tensor_index.json` during transition)
  - `model_descriptor.json` (includes `model_id`, `index_multihash`, `data_multihash`, `schema_version`, `encoding`, `total_size`, `hash_params`)
- Standard data format (see `web-docs/.../data-format.md`): write partitions and index per spec; both files must be present in the directory.
- `safetensors`: when saving into a `safetensors` directory, also write `model_descriptor.json` at the directory root; for multi-file payloads, index/descriptor remain at the root.

#### 13.2 Load Side (DiskLoader)

- Standard partitions (see “Data File Format Specification”):
  - Directory requirements (target behavior): `model_descriptor.json` and `tensor_index.(cbor|json)` must exist; missing files return `FailedPrecondition(MODEL_DESCRIPTOR_REQUIRED)`.
  - Current implementation (transition period): if `model_descriptor.json` is missing, allow computing and writing it "after load completes"; will be tightened to hard gate.
  - Loading flow: read partitions by file-name order, build linear stream, and complete memory mapping.
  - Post-load verification (ensure ID matches data):
    - Immediately run lightweight verification (KEY_POINTS/SEGMENT) consistent with descriptor info.
    - When strict verification is enabled (current: `LoadingHints::verify == FULL_DIGEST`), compute or Merkle-verify the tree-hash root and ensure it matches `data_multihash`; otherwise return `DataCorruption(MODEL_ID_MISMATCH)`.
  - On success: if GS is available, register/update local replica by `model_id`.

- `safetensors` directories:
  - If `model_descriptor.json` exists: treat it as the authoritative `model_id`; after loading, verify per above; mismatch returns `DataCorruption(MODEL_ID_MISMATCH)`.
  - If `model_descriptor.json` does not exist:
    - After loading, compute `index_multihash`/`data_multihash` from the Canonical Index and linear data stream, and build the `model_id`.
    - Write `model_descriptor.json` to the directory (on failure return `PermissionDenied(DESCRIPTOR_NOT_WRITABLE)`); optionally persist the Canonical Index.
    - If GS is available, register the replica with the new `model_id`.

#### 13.3 Common Constraints and Errors

- Never interpret `mi2:` as a filesystem path.
- Suggested error codes:
  - `FailedPrecondition(MODEL_DESCRIPTOR_REQUIRED)`: standard data directory is missing descriptor
  - `DataCorruption(MODEL_ID_MISMATCH)`: post-load verification mismatches `model_id`
  - `PermissionDenied(DESCRIPTOR_NOT_WRITABLE)`: cannot generate/persist descriptor


### 14. Code-level Implementation Plan (Aligned with Current Repo)

- Save side (Python via unified C++ pipeline) — completed
  - File: `scstore/torch_util.py`
    - `save_dict(...)` prepares inputs and calls `_C.save_model_to_disk(...)` to emit `tensor_index.(cbor|json)` and `model_descriptor.json`, returning the `descriptor`.
    - Python no longer implements multihash/tree-hash/base32.
  - Files: `scstore/_C.pyi` / `scstore/csrc/checkpoint_py.cc`
    - Expose and implement `save_model_to_disk(...)` and `inspect_or_generate_descriptor(...)`, both backed by the C++ `SeekableSource` hashing + Canonical Index pipeline.

- Disk loading (C++, unified verification and backfill) — completed
  - Files: `core/store/loader/disk_loader.cc` / `disk_loader.h`
    - `DiskLoader::initialize()`:
      - Standard partitions: require `model_descriptor.json` + `tensor_index.(json|cbor)`; missing files return `FailedPrecondition(MODEL_DESCRIPTOR_REQUIRED)`; otherwise read and cache descriptor.
      - `safetensors`: allow missing `model_descriptor.json` (backfilled later).
  - File: `core/store/checkpoint_store.cc`
    - `CheckpointStore::load_from_disk_internal(...)` (constructs `ModelHandle` after `LOADED`):
      - Obtain GPU base and size; call `model_hash::compute_data_multihash_from_gpu(...)` to get `data_mh`.
      - Standard partitions: read `tensor_index.(json|cbor)`, build canonical bytes, compute `index_mh` via `compute_index_multihash(index_bytes, "")`.
      - `safetensors`:
        - If `model_descriptor.json` exists: use its `index_multihash`.
        - If missing: build v2 Canonical Index bytes using `BuildCanonicalIndexFromSafetensors(...)`, compute `index_mh`, and write `model_descriptor.json` (report `PermissionDenied` if unwritable).
      - If descriptor exists: strictly verify `descriptor.data_multihash == data_mh`; mismatch raises `DataCorruption(MODEL_ID_MISMATCH)`; optionally add KEY_POINTS/SEGMENT fast checks.
      - If GS available: register/update local replica by `model_id`.

- `safetensors` Canonical Index (C++) — completed
  - Files: `core/store/loader/safetensors_util.{h,cc}`
    - `absl::StatusOr<std::string> BuildCanonicalIndexFromSafetensors(const std::vector<std::filesystem::path>& files);`
      - Parse header JSON, construct v2 canonical mapping (sorted outer keys, fixed field order, `storage_offset=0`, row-major stride), serialize to Strict Canonical JSON bytes.

— Redundancy cleanup (see §17)
  - Python’s former multihash/tree-hash/base32 and partition collection functions removed; CLI/tools now use `_C` (done).
  - Legacy C++ directory-hash entry points migrated to `core/store/loader/*`; deprecated symbols deleted (done).

### 15. Test Plan

- C++:
  - Standard partition directory missing `model_descriptor.json` → `DiskLoader::initialize()` returns `FailedPrecondition`.
  - Standard partition directory with matching descriptor → `load_from_disk_internal()` succeeds; tampered data or descriptor → `MODEL_ID_MISMATCH`.
  - `safetensors`: both with/without descriptor (generation/verification/errors).
- Python:
  - `save_dict` via `_C.save_model_to_disk` produces canonical `tensor_index.json` + `model_descriptor.json`; assert `mi2:` prefix and that `total_size` is the logical size.

### 16. Migration and Delivery

- Migration tool: batch-generate `model_descriptor.json` for existing directories (standard partitions and `safetensors`).
- Delivery order:
  - Batch 1: save-side descriptor emission + standard partition loader gate.
  - Batch 2: post-load ID verification + `safetensors` backfill.
  - Batch 3: migration tool + GS registration hardening and end-to-end regression.


### 17. De-dup Roadmap (Redundant Implementation Cleanup)

- Objective: remove cross-language duplication so all fingerprint computation, encoding, and descriptor persistence flow through the unified C++ pipeline, reducing complexity and inconsistency risk.

- Python layer (effective immediately)
  - Delete files/functions (no compatibility shims):
    - In `scstore/content_addressing.py` remove:
      - `compute_index_multihash_from_index_bytes`
      - `compute_data_multihash_from_segments`
      - `collect_partition_segments_for_stream`
      - `collect_safetensors_segments_for_stream`
      - and private implementations for `_to_multibase_multihash_sha256`, tree hashing, and Base32 encoding
  - Retain but refactor:
    - `generate_model_id_from_path(...)`: call `_C.inspect_or_generate_descriptor(path)` and return the resulting `descriptor`; do not compute hashes in Python.
  - Tools and examples:
    - `scstore/tools/backfill_descriptor.py` now uses `_C.inspect_or_generate_descriptor` and drops the `--write-index` placeholder (done); ensure other CLIs/examples no longer import compute functions from `scstore.content_addressing` (self-audited).

- C++ layer
  - Removed entry points:
    - `compute_data_multihash_from_disk_dir` in `core/common/model_hash.h` removed.
    - Call sites migrated to `core/store/loader/disk_dir_hash.{h,cc}` or `source_hash.{h,cc}` (done).
  - Retained implementations:
    - `core/common/model_hash.{h,cc}` as the single place for multibase/multihash/base32 and GPU-buffer tree hashing.
    - Directories/partitions/remote sources go through `SeekableSource` + `source_hash` pipeline.

- Docs and runtime gates
  - Docs: clearly state "Python must not compute `index_multihash`/`data_multihash` or write `model_descriptor.json`".
  - Runtime: `DiskLoader` enforces descriptor presence for standard partitions; missing files return `FailedPrecondition(MODEL_DESCRIPTOR_REQUIRED)`; `safetensors` directories may be backfilled (with write permission).
  - Web docs: remove the outdated `load_dict_pure_local` developer guide from the sidebar to avoid confusion (done).

- Timeline
  - vNext (current branch): remove Python duplication and deprecated C++ symbols.
  - vNext+1: continue consolidating remaining entry points and docs.
  - vNext+2: eliminate all transitional text and comments.
