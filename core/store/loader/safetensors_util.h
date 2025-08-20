// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace stepcast::store::loader {

// Information extracted from a safetensors file header
struct SafetensorsHeaderInfo {
  uint64_t header_length; // Length of the JSON header
  uint64_t data_start; // Offset where tensor data starts (8 + header_length)
  uint64_t data_size; // Size of the tensor data payload
};

// Parse the safetensors header from a file descriptor
// The fd should be positioned at the beginning of the file
// This function reads the 8-byte header length and validates it
absl::StatusOr<SafetensorsHeaderInfo> ParseSafetensorsHeader(int fd);

// Build RFC-0007 v2 canonical index JSON bytes from a set of .safetensors files.
// The resulting JSON has sorted outer keys and fixed inner field order:
// [offset, size, shape, stride, dtype, storage_offset]
// - offset: base_offset + data_offsets[0], where base_offset is accumulated payload size across files
// - size: data_offsets[1] - data_offsets[0]
// - shape: array from header
// - stride: row-major stride derived from shape
// - dtype: mapped torch dtype string (e.g., "torch.float16")
// - storage_offset: always 0 for safetensors headers
absl::StatusOr<std::string> BuildCanonicalIndexFromSafetensors(const std::vector<std::filesystem::path>& files);

} // namespace stepcast::store::loader