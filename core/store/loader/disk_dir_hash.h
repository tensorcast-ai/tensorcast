// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <string>

#include "absl/status/statusor.h"

namespace stepcast::store::loader {

// Compute data multihash for a standard partitioned disk directory by
// constructing a FilePartitionSource and streaming via the unified hashing
// pipeline. Returns multibase(base32)-encoded multihash of the Merkle root.
absl::StatusOr<std::string> compute_data_multihash_from_disk_dir(const std::string& artifact_dir);

} // namespace stepcast::store::loader
