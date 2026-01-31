// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/metadata/source_hash.h"

#include <algorithm>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/dataplane/sources/memory_source.h"

namespace tensorcast::store::loader {

using tensorcast::common::compute_tree_hash_root_sha256;
using tensorcast::common::multibase_multihash_sha256;
using tensorcast::common::sha256_digest_bytes;

absl::StatusOr<std::string> compute_data_multihash_from_seekable_source(
    SeekableSource& source,
    uint64_t total_size,
    size_t leaf_chunk_bytes,
    std::function<void(uint64_t hashed_leaf_count, uint64_t total_hash_leaves)> progress_cb) {
  if (total_size == 0) {
    return absl::InvalidArgumentError("total_size must be > 0");
  }
  if (leaf_chunk_bytes == 0) {
    return absl::InvalidArgumentError("leaf_chunk_bytes must be > 0");
  }

  std::vector<std::vector<uint8_t>> leaves;
  const uint64_t total_leaves = (total_size + leaf_chunk_bytes - 1) / leaf_chunk_bytes;
  leaves.reserve(static_cast<size_t>(total_leaves));

  std::vector<uint8_t> buffer(leaf_chunk_bytes);
  uint64_t processed = 0;
  uint64_t hashed = 0;
  while (processed < total_size) {
    const size_t to_read = static_cast<size_t>(std::min<uint64_t>(leaf_chunk_bytes, total_size - processed));
    auto n_or = source.read_at(processed, buffer.data(), to_read);
    if (!n_or.ok()) {
      return n_or.status();
    }
    const size_t got = n_or.value();
    if (got == 0) {
      // Unexpected short read
      return absl::DataLossError("short read while hashing source");
    }
    if (got != to_read) {
      return absl::DataLossError("short read while hashing source");
    }
    leaves.push_back(sha256_digest_bytes(absl::Span<const uint8_t>(buffer.data(), got)));
    processed += got;
    hashed += 1;
    if (progress_cb) {
      progress_cb(hashed, total_leaves);
    }
  }
  std::vector<uint8_t> root = compute_tree_hash_root_sha256(leaves);
  return multibase_multihash_sha256(root);
}

absl::StatusOr<std::string> compute_data_multihash_from_cpu_memory(
    gsl::not_null<const void*> base_ptr,
    uint64_t total_size,
    size_t leaf_chunk_bytes,
    std::function<void(uint64_t hashed_leaf_count, uint64_t total_hash_leaves)> progress_cb) {
  CpuMemorySource src(base_ptr, total_size);
  return compute_data_multihash_from_seekable_source(src, total_size, leaf_chunk_bytes, std::move(progress_cb));
}

absl::StatusOr<std::string> compute_data_multihash_from_gpu_memory(
    gsl::not_null<void*> device_ptr,
    uint64_t total_size,
    int device_id,
    size_t leaf_chunk_bytes) {
  return common::compute_data_multihash_from_gpu(device_ptr.get(), total_size, device_id, leaf_chunk_bytes);
}

} // namespace tensorcast::store::loader
