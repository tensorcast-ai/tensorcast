// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "absl/status/statusor.h"

namespace tensorcast::store::loader {

// Compute data multihash for a standard partitioned disk directory by
// constructing a FilePartitionSource and streaming via the unified hashing
// pipeline. Returns multibase(base32)-encoded multihash of the Merkle root.
absl::StatusOr<std::string> compute_data_multihash_from_disk_dir(
    const std::string& artifact_dir,
    std::function<void(uint64_t processed_bytes, uint64_t total_bytes)> progress_cb = {});

} // namespace tensorcast::store::loader
