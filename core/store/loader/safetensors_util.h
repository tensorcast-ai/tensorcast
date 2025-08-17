// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstdint>

#include "absl/status/statusor.h"

namespace stepcast::store::loader {

// Information extracted from a safetensors file header
struct SafetensorsHeaderInfo {
  uint64_t header_length;   // Length of the JSON header
  uint64_t data_start;      // Offset where tensor data starts (8 + header_length)
  uint64_t data_size;       // Size of the tensor data payload
};

// Parse the safetensors header from a file descriptor
// The fd should be positioned at the beginning of the file
// This function reads the 8-byte header length and validates it
absl::StatusOr<SafetensorsHeaderInfo> ParseSafetensorsHeader(int fd);

}  // namespace stepcast::store::loader