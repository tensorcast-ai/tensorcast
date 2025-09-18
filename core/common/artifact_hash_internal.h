// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace tensorcast::common::internal {

inline constexpr size_t kMaxChunkSize = 64ULL * 1024 * 1024;
inline constexpr size_t kMinLeafChunkBytes = 512ULL * 1024;
inline constexpr size_t kTargetLeafCount = 4096ULL;

absl::StatusOr<std::vector<uint8_t>> sha256_bytes(absl::Span<const uint8_t> bytes);
absl::StatusOr<std::vector<uint8_t>> compute_tree_hash_sha256(const std::vector<std::vector<uint8_t>>& leaf_digests);
std::string to_multibase_multihash_sha256(const std::vector<uint8_t>& digest);
std::vector<uint64_t> compute_chunk_lengths(uint64_t total_size, size_t chunk_size_bytes);
size_t determine_leaf_chunk_size(uint64_t total_size, size_t requested_chunk_bytes);

} // namespace tensorcast::common::internal
