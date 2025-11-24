// Copyright (c) 2025, TensorCast Team.
//
// Helpers for detecting host memory capacity (cgroup-aware).

#pragma once

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"

namespace tensorcast::common::memory {

// Detect total host memory capacity in bytes, preferring cgroup limits when
// present (memory.max) and falling back to /proc/meminfo MemTotal.
absl::StatusOr<uint64_t> detect_host_memory_capacity_bytes();

// Test-only hook to override host memory detection. Pass std::nullopt to clear.
void set_host_memory_capacity_override_for_testing(std::optional<uint64_t> bytes);

} // namespace tensorcast::common::memory
