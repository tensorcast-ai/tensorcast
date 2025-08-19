// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace stepcast::store::model_hash {

// Compute index multihash (multibase base32 over multihash sha2-256) from either
// canonical index bytes (preferred) or a precomputed sha256 hex key.
absl::StatusOr<std::string> compute_index_multihash(
    const std::optional<std::string>& index_data,
    std::string_view index_key_hex);

// Compute data multihash (multibase base32 over tree-hash sha2-256 root) for a
// contiguous GPU buffer [gpu_ptr, size). The implementation streams device→host
// in 4MiB leaves and reduces them as a Merkle tree.
absl::StatusOr<std::string> compute_data_multihash_from_gpu(void* gpu_ptr, uint64_t total_size, int device_id);

// DEPRECATED: Use loader::compute_data_multihash_from_disk_dir in
// core/store/loader when hashing a disk directory via SeekableSource.
absl::StatusOr<std::string> compute_data_multihash_from_disk_dir(const std::string& model_dir);

// Lightweight public wrappers to reuse hashing primitives across modules
// without duplicating implementations. These mirror the internal helpers
// used by the functions above.
std::vector<uint8_t> sha256_digest_bytes(absl::Span<const uint8_t> bytes);
std::vector<uint8_t> compute_tree_hash_root_sha256(const std::vector<std::vector<uint8_t>>& leaf_digests);
std::string multibase_multihash_sha256(const std::vector<uint8_t>& digest);

} // namespace stepcast::store::model_hash
