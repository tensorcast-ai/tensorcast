// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "absl/status/statusor.h"

namespace stepcast::store::loader {

// Canonical tensor metadata used to construct the RFC-0007 index
struct CanonicalTensorMeta {
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::string dtype; // torch dtype string, e.g. "torch.float16"
  uint64_t storage_offset{0};
};

// Map torch dtype string to a stable integer code for ordering/grouping.
// The mapping is deterministic and only affects ordering, not semantics.
int torch_dtype_code(std::string_view dtype);

// Rebuild canonical index JSON bytes from an existing index JSON string.
// This function enforces:
// - Sorted outer keys (tensor names ascending)
// - Fixed field order [offset, size, shape, stride, dtype, storage_offset]
// - Proper integer types for numeric fields
// It DOES NOT change offsets/sizes; it is safe for hashing/verification.
absl::StatusOr<std::string> rebuild_stable_canonical_index(const std::string& index_json, int default_device_id);

// Build canonical index JSON object from ordered tensor names and metadata.
// - ordered_names: final write order
// - offsets: tensor name -> starting byte offset (8B aligned by writer)
// - sizes: tensor name -> logical size in bytes (max across aliases)
// - metas: tensor name -> CanonicalTensorMeta
absl::StatusOr<std::string> build_canonical_index_json(
    const std::vector<std::string>& ordered_names,
    const std::unordered_map<std::string, uint64_t>& offsets,
    const std::unordered_map<std::string, uint64_t>& sizes,
    const std::unordered_map<std::string, CanonicalTensorMeta>& metas);

} // namespace stepcast::store::loader
