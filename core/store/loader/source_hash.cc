// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/source_hash.h"

#include <algorithm>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "core/common/model_hash.h"

namespace stepcast::store::loader {

namespace {

using stepcast::store::model_hash::compute_tree_hash_root_sha256;
using stepcast::store::model_hash::multibase_multihash_sha256;
using stepcast::store::model_hash::sha256_digest_bytes;

} // namespace

absl::StatusOr<std::string> compute_data_multihash_from_seekable_source(
    SeekableSource& source,
    uint64_t total_size,
    size_t leaf_chunk_bytes) {
  if (total_size == 0) {
    return absl::InvalidArgumentError("total_size must be > 0");
  }
  if (leaf_chunk_bytes == 0) {
    return absl::InvalidArgumentError("leaf_chunk_bytes must be > 0");
  }

  std::vector<std::vector<uint8_t>> leaves;
  leaves.reserve(static_cast<size_t>((total_size + leaf_chunk_bytes - 1) / leaf_chunk_bytes));

  std::vector<uint8_t> buffer(leaf_chunk_bytes);
  uint64_t processed = 0;
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
    leaves.push_back(sha256_digest_bytes(absl::Span<const uint8_t>(buffer.data(), got)));
    processed += got;
  }
  std::vector<uint8_t> root = compute_tree_hash_root_sha256(leaves);
  return multibase_multihash_sha256(root);
}

} // namespace stepcast::store::loader
