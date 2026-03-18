// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace tensorcast::common {

inline constexpr size_t kGpuHashDefaultLeafChunkBytes = 4ULL * 1024 * 1024;

// Compute index multihash (multibase base32 over multihash sha2-256) from either
// canonical index bytes (preferred) or a precomputed sha256 hex key.
absl::StatusOr<std::string> compute_index_multihash(
    const std::optional<std::string>& index_data,
    std::string_view index_key_hex);

// Compute data multihash (multibase base32 over tree-hash sha2-256 root) for a
// contiguous GPU buffer [gpu_ptr, size). Under the real CUDA backend this
// requires the NVRTC-based device hashing path; FakeCuda falls back to
// streaming device→host in 4MiB leaves.
absl::StatusOr<std::string> compute_data_multihash_from_gpu(
    void* gpu_ptr,
    uint64_t total_size,
    int device_id,
    size_t leaf_chunk_bytes = kGpuHashDefaultLeafChunkBytes);

// Pre-compile/load the NVRTC GPU hashing kernel for a specific device. This is
// a no-op under the FakeCuda backend.
absl::Status prewarm_gpu_hash_nvrtc_for_device(int device_id);

// Pre-compile/load the NVRTC GPU hashing kernel for every visible device. This
// is intended for daemon startup fail-fast checks and is a no-op under the
// FakeCuda backend or when no GPUs are visible.
absl::Status prewarm_gpu_hash_nvrtc_for_visible_devices();

// Lightweight public wrappers to reuse hashing primitives across modules
// without duplicating implementations. These mirror the internal helpers
// used by the functions above.
std::vector<uint8_t> sha256_digest_bytes(absl::Span<const uint8_t> bytes);
std::vector<uint8_t> compute_tree_hash_root_sha256(const std::vector<std::vector<uint8_t>>& leaf_digests);
std::string multibase_multihash_sha256(const std::vector<uint8_t>& digest);

} // namespace tensorcast::common
