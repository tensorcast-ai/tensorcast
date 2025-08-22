# 0007-Safetensors Loader Integration

## Overview

### Problem statement
- Current loader stack supports only the partitioned binary checkpoint format (`tensor.data_*` + `tensor_index.json`). Many ecosystems (HF, PyTorch community) produce Safetensors. We need first-class, high-performance loading from Safetensors without forking the data path.

### Goals and success criteria
- Support loading directly from `.safetensors` files with minimal changes to `@loader/` pipeline.
- Keep a single unified loading pipeline: `SeekableSource` → `pump_ranges` → `PositionedSink`.
- Preserve performance: comparable throughput to partitioned files; no unnecessary copies or reformatting.
- Keep index handling simple: reuse existing “normalized v2” semantics where possible.
- Maintain verification support; produce/consume sizes that represent the actual tensor payload, not headers.

### High-level solution summary
- Add a new `SeekableSource` implementation, `SafetensorsSource`, which exposes a single Safetensors file's data buffer as a contiguous byte stream (offset 0 = data buffer start), hiding the header entirely from the pipeline.
- Add `MultiSafetensorsSource` that concatenates the data buffers of multiple `.safetensors` files into one logical byte space: `total_size = Σ data_size_i`. `read_at` transparently spans file boundaries.
- Extend `DiskLoader` to detect one or more `.safetensors` files when no `tensor.data*` partitions are present. If 1 file → `SafetensorsSource`; if N>1 → `MultiSafetensorsSource`.
- In Python, add header parsers to merge metadata from all `.safetensors` files into a unified v2-like in-memory index; assert no duplicate tensor keys across files; identity device offsets (no realignment).
- Update verification path to handle Safetensors by hashing only the data buffer(s) in file-name order.

## Current Architecture Analysis

### Key components
- `SeekableSource` interface

```17:34:core/store/loader/source.h
  virtual absl::StatusOr<size_t> read(void* dst, size_t max_bytes) = 0;
};

class SeekableSource : public Source {
 public:
  virtual absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) = 0;

  // Optional zero-copy capability: direct write into destination address space.
  // Default implementations disable the feature.
  [[nodiscard]] virtual bool supports_direct_write() const {
    return false;
  }
  virtual absl::StatusOr<size_t> read_into(uint64_t dest_va_offset, size_t bytes, const DirectWriteToken& /*token*/) {
    (void)dest_va_offset;
    (void)bytes;
    return absl::UnimplementedError("direct write not supported");
  }
};
```

- Pumping API

```199:206:core/store/loader/pump.cc
absl::Status pump_ranges(
    SeekableSource& src,
    PositionedSink& dst,
    BufferPool& pool,
    absl::Span<const Range> ranges,
    int concurrency) {
```

- Disk loader discovers partitions and returns a `FilePartitionSource`

```100:117:core/store/loader/disk_loader.cc
  // Find all partition files
  partition_paths_.clear();
  partition_sizes_.clear();
  artifact_size_ = 0;

  for (const auto& entry : std::filesystem::directory_iterator(artifact_dir)) {
    if (entry.is_regular_file()) {
      const std::string filename = entry.path().filename().string();
      // Support only the partition naming convention defined in
      // docs/developer-guides/core/checkpoint/data-format.md:
      //   tensor.data        (single-file artifact)
      //   tensor.data_<n>    (multi-partition artifact, 0-based index)
      if (filename.starts_with("tensor.data")) {
        partition_paths_.push_back(entry.path());
        size_t file_size = std::filesystem::file_size(entry.path());
        partition_sizes_.push_back(file_size);
        artifact_size_ += file_size;
      }
    }
  }
```

```121:125:core/store/loader/disk_loader.cc
  // If no partitions were detected with the supported patterns report an error.
  if (partition_paths_.empty()) {
    return absl::NotFoundError(absl::StrFormat("No artifact partition files found in: %s", artifact_dir.string()));
  }
```

```165:175:core/store/loader/disk_loader.cc
  loader::FilePartitionSource::Options source_opts;
  {
    absl::MutexLock lock(&mutex_);
    source_opts.partition_paths = partition_paths_;
    source_opts.partition_sizes = partition_sizes_;
    source_opts.total_size = artifact_size_;
    // chunk size is determined by MemoryManager's pinned pool; a default here is fine
    source_opts.chunk_size = 128 * 1024 * 1024;
    source_opts.use_direct_io = (artifact_size_ > 5ULL * 1024 * 1024 * 1024);
  }
  return std::make_unique<loader::FilePartitionSource>(std::move(source_opts));
```

- Transfer service fallback to source-specific size

```219:223:core/store/replica/transfer_service.cc
  // Fallback: use source total size when UMA doesn't know
  if (total_size == 0) {
    if (auto* fps = dynamic_cast<loader::FilePartitionSource*>(source.get())) {
      total_size = fps->total_size();
    }
  }
```

- `FilePartitionSource` exposes `total_size()` (non-virtual)

```38:44:core/store/loader/file_partition_source.h
  uint64_t total_size() const {
    return options_.total_size;
  }

  bool is_using_direct_io() const {
    return using_direct_io_;
  }
```

### Pain points / constraints
- Disk loader assumes only `tensor.data*` files exist; it errors if not found (no graceful probing for other formats).
- O_DIRECT and alignment assumptions are tailored to partitioned files. Safetensors has a small header; header offset is rarely 512-byte aligned, so O_DIRECT needs careful handling.
- `SeekableSource` lacks a standard size query; the transfer path does a concrete-type fallback to `FilePartitionSource`.

## Proposed Solution

### Core architectural changes
1) Introduce `SafetensorsSource : SeekableSource`
- Purpose: present a single Safetensors file's data buffer as a contiguous byte space starting at offset 0.
- Implementation:
  - On first use (or constructor), open file, read 8-byte little-endian `N`, read `N` bytes of UTF-8 header JSON, compute `data_start = 8 + N`, compute `data_size = file_size - data_start`.
  - `read_at(off, ...)` → POSIX `pread(fd, ..., data_start + off)`; return 0 on EOF.
  - `read(...)` keeps an internal offset with a small mutex (same pattern as `FilePartitionSource`).
  - O_DIRECT is disabled for this source (unaligned header offset makes it unreliable); rely on kernel page cache.
  - Provide a concrete `total_size()` method (non-virtual) analogous to `FilePartitionSource`.
  - Thread-safety: stateless reads are thread-safe; sequential `read()` guarded by a mutex (same as partitions).

2) Introduce `MultiSafetensorsSource : SeekableSource`
- Purpose: present multiple Safetensors files as a single concatenated byte space of their data buffers (headers hidden).
- Implementation:
  - Pre-parse all selected `.safetensors` files once to compute pairs `(fd_i, data_start_i, data_size_i)` and build prefix sums `base_offset_i = Σ_{k<i} data_size_k`.
  - `total_size() = Σ data_size_i`.
  - `read_at(off, ...)` locates file segment `i` where `base_offset_i ≤ off < base_offset_{i+1}`, then issues `pread(fd_i, ..., data_start_i + (off - base_offset_i))`, and spans files as needed.
  - O_DIRECT disabled; rely on page cache.
  - Thread-safety: stateless multi-file reads; sequential `read()` guarded by a small mutex.

3) Extend `DiskLoader` to detect one or more `.safetensors`
- `initialize()`:
  - If any `tensor.data*` exists → keep current behavior.
  - Else, probe for all `*.safetensors` files in the directory (N ≥ 1). Sort by filename for deterministic order.
    - For each file, pre-parse header to compute `data_start_i`, `data_size_i`; set `artifact_size_ = Σ data_size_i`.
    - Memoize file paths and their `(data_start_i, data_size_i)` for `open_source()`.
  - If neither partitions nor `.safetensors` exist → keep current error.
- `open_source()`:
  - If in safetensors mode with N==1, return `std::make_unique<SafetensorsSource>(path)`.
  - If in safetensors mode with N>1, return `std::make_unique<MultiSafetensorsSource>(paths)`.
  - Otherwise return `FilePartitionSource` as-is.

4) Keep TransferService unchanged, with a minor improvement
- UMA continues to be the source of truth for artifact size.
- Fallback path adds `dynamic_cast` to `SafetensorsSource` and `MultiSafetensorsSource` to read `total_size()` when UMA is missing (optional, low risk).

5) Python: build normalized meta from Safetensors header(s)
- Parse all `.safetensors` headers in the directory; ignore `__metadata__` values for the tensor index. Support merging `__metadata__` into an optional dictionary (keys must be unique, otherwise raise an error or ignore conflicts).
- For each file i: in its header, `data_offsets=[BEGIN, END]` are relative to the start of that file's data buffer; compute the global offset `GLOBAL_BEGIN = base_offset_i + BEGIN`, and the length `LEN = END - BEGIN`.
- Build a unified index: `tensor_meta_index[name] = (shape, row_major_stride(shape), torch_dtype, storage_offset=0)` and `tensor_data_index[name] = (GLOBAL_BEGIN, LEN)`.
- Add assertions: all tensor keys across `.safetensors` files must be pairwise unique; if a duplicate is found, immediately raise an error (configurable strict mode). `__metadata__` may likewise be merged uniquely or conflicting entries ignored.
- Device offsets: identity mapping on the local path; do not apply the 8-byte realignment heuristic from the partitioned format.
- Dtype mapping:
  - Support `F16/F32/F64/BF16`, `I8/I16/I32/I64`, `U8`, `BOOL`;
  - Types smaller than 1 byte (e.g., 4-bit) are not supported; raise a clear error.

6) Verification support
- For all `.safetensors` files, hash only each data buffer (excluding the header) in filename order; process files in filename order; record the aggregate `artifact_size = Σ data_size_i` and `partition_count = N` (optional).

### Why this design
- Hides all format differences at the `Source` layer. The loader pipeline sees a uniform, contiguous byte space for both formats.
- Preserves existing `pump_ranges` and sinks; minimizes change amplification.
- Keeps index/metadata responsibilities in Python (unchanged mental artifact for callers). No need to invent a new index schema.

### Alternatives considered
- Introduce a new "index adapter" layer that rewrites Safetensors header(s) into a single `tensor_index.json` on disk: rejected to avoid additional I/O and complexity.
- Extend `SeekableSource` with a `logical_size()` pure virtual: deferred. Would require changing all Sources; low benefit. This proposal uses a targeted `dynamic_cast` fallback.

## Implementation Plan

### Phases and tasks
1) Core C++ source
- Add `core/store/loader/safetensors_source.h/.cc`
- Implement header parsing, `read`/`read_at`, and `total_size()`
- Unit tests: header parsing, range correctness, EOF behavior

2) DiskLoader detection
- Modify `core/store/loader/disk_loader.cc` to probe one or more `.safetensors` when partitions are absent
- Pre-parse headers of all `.safetensors` to obtain each `data_size_i` and compute `artifact_size_ = Σ data_size_i`; memoize ordered paths
- `open_source()` returns `SafetensorsSource` (N==1) or `MultiSafetensorsSource` (N>1)

3) TransferService minor fallback
- Optional: extend `load_from_source` fallback to read size from `SafetensorsSource` when UMA does not provide it

4) Python integration
- Add header parser and dtype mapping in `scstore/torch_util.py`
- `load_dict` and `load_dict_pure_local` add a Safetensors path that constructs meta/data indices from the header and uses identity offsets
- Tests: CPU & GPU restoration from `.safetensors`

5) Verification
- Extend verification to handle `.safetensors` by hashing data buffer section only

6) Docs & examples
- Update developer docs to include Safetensors format alignment with our pipeline

### File modification table

| File | Change | Notes |
|----|----|----|
| `core/store/loader/safetensors_source.h/.cc` | Add | New `SeekableSource` for Safetensors |
| `core/store/loader/disk_loader.cc` | Update | Detect `.safetensors`; compute data_size; return `SafetensorsSource` |
| `core/store/replica/transfer_service.cc` | Update (optional) | Fallback: size from `SafetensorsSource`/`MultiSafetensorsSource` when UMA is empty |
| `core/store/loader/multi_safetensors_source.h/.cc` | Add | Concatenate multiple Safetensors data buffers into one logical source |
| `scstore/torch_util.py` | Update | Parse header; build meta/data indices; identity offsets |
| `core/checkpoint/*` | Update | Verification: handle `.safetensors` data buffer only |
| `core/store/loader/BUILD` | Update | Add new source, tests |
| `web-docs/docs/developer-guides/...` | Update | Add Safetensors loading doc |

### API changes
- None to public APIs. Internal addition of `SafetensorsSource`.

## Code References

- `SeekableSource` interface (read, read_at, direct-write capabilities):

```17:34:core/store/loader/source.h
  virtual absl::StatusOr<size_t> read(void* dst, size_t max_bytes) = 0;
};

class SeekableSource : public Source {
 public:
  virtual absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) = 0;
  [[nodiscard]] virtual bool supports_direct_write() const {
    return false;
  }
  virtual absl::StatusOr<size_t> read_into(uint64_t dest_va_offset, size_t bytes, const DirectWriteToken& /*token*/) {
    (void)dest_va_offset;
    (void)bytes;
    return absl::UnimplementedError("direct write not supported");
  }
};
```

- `pump_ranges` driving producer/consumer threads and positioned writes:

```199:206:core/store/loader/pump.cc
absl::Status pump_ranges(
    SeekableSource& src,
    PositionedSink& dst,
    BufferPool& pool,
    absl::Span<const Range> ranges,
    int concurrency) {
```

- Disk partition discovery and error path:

```100:117:core/store/loader/disk_loader.cc
  for (const auto& entry : std::filesystem::directory_iterator(artifact_dir)) {
    if (entry.is_regular_file()) {
      const std::string filename = entry.path().filename().string();
      if (filename.starts_with("tensor.data")) {
        partition_paths_.push_back(entry.path());
        size_t file_size = std::filesystem::file_size(entry.path());
        partition_sizes_.push_back(file_size);
        artifact_size_ += file_size;
      }
    }
  }
```

```121:125:core/store/loader/disk_loader.cc
  if (partition_paths_.empty()) {
    return absl::NotFoundError(absl::StrFormat("No artifact partition files found in: %s", artifact_dir.string()));
  }
```

- `open_source()` building a `FilePartitionSource` (to be complemented with Safetensors path):

```165:175:core/store/loader/disk_loader.cc
  source_opts.partition_paths = partition_paths_;
  source_opts.partition_sizes = partition_sizes_;
  source_opts.total_size = artifact_size_;
  source_opts.chunk_size = 128 * 1024 * 1024;
  source_opts.use_direct_io = (artifact_size_ > 5ULL * 1024 * 1024 * 1024);
  return std::make_unique<loader::FilePartitionSource>(std::move(source_opts));
```

- TransferService UMA size fallback (we will optionally extend with SafetensorsSource):

```219:223:core/store/replica/transfer_service.cc
  if (total_size == 0) {
    if (auto* fps = dynamic_cast<loader::FilePartitionSource*>(source.get())) {
      total_size = fps->total_size();
    }
  }
```

## Visual

```mermaid
flowchart LR
  subgraph Disk (Current)
    D1[DiskLoader] -->|open_source| FP[FilePartitionSource]
  end

  subgraph Safetensors (New)
    S1[DiskLoader] -->|open_source| SS1[SafetensorsSource]
    S1 -->|open_source (multi)| SSN[MultiSafetensorsSource]
  end

  FP --> P[pump_ranges]
  SS --> P
  P --> SNK[PositionedSink (GPU/CPU/DVMP)]
```

## Detailed Design Notes

### SafetensorsSource
- Constructor accepts `file_path`. On first read:
  - `read(fd, 8)` → `N : uint64_le`.
  - `read(fd, N)` → JSON header; verify it starts with `{` and consists of valid UTF-8 (ignore trailing spaces 0x20).
  - `data_start = 8 + N`, `data_size = file_size - data_start`.
  - Store values; subsequent `read_at(off, ...)` translates `off` relative to `data_start`.
- Error handling: return `InvalidArgument` for malformed header; `OutOfRange` for overrun; propagate `errno` for I/O.
- Concurrency: same patterns as `FilePartitionSource`.
- Direct I/O: disabled by design; kernel page cache suffices and simplifies alignment.

### DiskLoader changes
- On `initialize()`:
  - Attempt partition discovery first (preserves current behavior and doc contract).
  - If none found, scan for all `*.safetensors` files (N ≥ 1). For each, pre-parse the header to compute `data_start_i` and `data_size_i`; sort by filename; accumulate `artifact_size_`.
- On `open_source()`:
  - If safetensors paths are set and N==1, return `SafetensorsSource(path)`; if N>1, return `MultiSafetensorsSource(paths)`; otherwise build `FilePartitionSource`.

### MultiSafetensorsSource
- Mapping from the file set to a unified byte space:
  - `base_offset_i = Σ_{k<i} data_size_k`, global range `[base_offset_i, base_offset_i + data_size_i)`.
  - `read_at(off, ...)` supports cross-file reads: read remaining capacity in the current segment, then advance to the next segment until the request is satisfied or EOF.
- Thread-safety: same as `FilePartitionSource::read_at`; `read()` protected by a small mutex.
- Direct write: disabled.

### Python header parsing and metadata
- Parse all `.safetensors` in the directory and build a unified index:
  - `tensor_meta_index[name] = (shape, row_major_stride(shape), torch_dtype, storage_offset=0)`
  - `tensor_data_index[name] = (GLOBAL_BEGIN, LEN)`, where `GLOBAL_BEGIN = base_offset_i + BEGIN` and `LEN = END - BEGIN`
- Assertions (required): all tensor keys must be unique across files; if duplicated, immediately raise an error (configurable strict mode). `__metadata__` may likewise be merged uniquely or conflicting entries ignored.
- Device offsets: identity mapping on the local path; do not apply the 8-byte realignment heuristic from the partitioned format.
- Dtype mapping:
  - Support `F16/F32/F64/BF16`, `I8/I16/I32/I64`, `U8`, `BOOL`;
  - Types smaller than 1 byte (e.g., 4-bit) are not supported; raise a clear error.

### Verification integration
- Update verification logic: for each `.safetensors` file, hash only the data buffer (excluding the header) in filename order; record the aggregate `artifact_size = Σ data_size_i`.

## Trade-offs and risks
- O_DIRECT: disable for Safetensors (single/multi-file) to avoid alignment complexity; rely on page cache performance.
- Size visibility: still rely on UMA or concrete-type fallbacks; to avoid broad changes, do not expand the interface for now.
- Multi-file ordering: use a deterministic order by filename; custom ordering can be added later if needed.
- Duplicate tensor keys: this design strictly disallows them; if a permissive "last-write wins" strategy is desired, make it configurable; default is strict.
- Sharded Safetensors (sharding spec) are covered by the multi-file mode; for future cross-directory/remote indexing, introduce a more general virtual source.

## Test Plan
- Unit tests (C++):
  - Header parsing (single file): valid cases, abnormal `N`, first byte not `{`, trailing spaces accepted.
  - `read_at` (single file): boundaries and exact EOF.
  - `read_at` (multi-file): cross-file boundary segmented reads, short-read and EOF behavior; concurrency correctness.
- Integration tests:
  - CPU path: for single/multi `.safetensors` directories, pump to DVMP and restore tensors via the merged index and validate.
  - GPU path: pump to `GPUMemorySink`, restore via the merged index and validate values.
  - Verification: baseline samples match for single/multi-folder cases.
  - Key conflict: construct two files containing the same tensor name, verify that the assertion triggers.

## Success Metrics
- Functional correctness: byte-for-byte equality vs PyTorch-loaded tensors for a reference fixture.
- Performance: within ±5% throughput vs partitioned format on the same hardware for 4–20 GB payloads.
- Resource usage: no material increase in peak RSS; no OOMs on typical models.

## Progress Tracking

| Phase | Task | Status | Notes |
|----|---|-----|----|
| 1 | Add `SafetensorsSource` + unit tests | ✅ Implemented (tests TBD) | Added `core/store/loader/safetensors_source.{h,cc}` |
| 2 | Add `MultiSafetensorsSource` + unit tests | ✅ Implemented (tests TBD) | Added `core/store/loader/multi_safetensors_source.{h,cc}` |
| 3 | `DiskLoader` multi-file detection + open source | ✅ Implemented | Detects `*.safetensors`, computes data sizes, returns appropriate source |
| 4 | TransferService fallback (optional) | ✅ Implemented | Fallback reads size from `SafetensorsSource`/`MultiSafetensorsSource` |
| 5 | Python header parse & merge + load paths | ✅ Implemented | `scstore/torch_util.py` builds indices from `.safetensors` when no `tensor_index.json` |
| 6 | Verification updates (single & multi) | ⏳ Pending | |
| 7 | Docs & examples | ⏳ Pending | |

## Execution Status

- Implemented C++ sources `SafetensorsSource` and `MultiSafetensorsSource` that expose only the data buffers (header hidden) and provide `total_size()`; direct I/O disabled.
- Extended `DiskLoader::initialize()` to discover `.safetensors` when no `tensor.data*` is present, pre-parse headers to compute `data_size_i`, and aggregate `artifact_size_`. `open_source()` returns the safetensors sources accordingly.
- Updated `TransferService` to fallback to safetensors `total_size()` when UMA is missing.
- Updated Bazel targets in `core/store/loader/BUILD` and `core/store/replica/BUILD` to include the new sources.
- Python path: `scstore/torch_util.py` now parses safetensors headers to produce unified meta/data indices when `tensor_index.json` is absent. Dtype mapping covers F16/BF16/F32/F64 and integer/byte/bool types.

Open tasks:
- Unit tests for new sources and integration tests for single/multi-file safetensors.
- Verification path: update hashing to process only safetensors data buffers in filename order.
- Developer docs update and examples.

## Appendix: Format references
- Safetensors layout (little-endian):
  - 8 bytes: `N` (u64) header size
  - `N` bytes: UTF-8 JSON dict `{name: {dtype, shape, data_offsets:[BEGIN, END]}, ...}` (may end with spaces)
  - Rest: data buffer
- Offsets in header are relative to data buffer start. We reflect that by translating source offsets by `data_start`.


