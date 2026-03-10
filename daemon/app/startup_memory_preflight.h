// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tensorcast::daemon {

struct StartupMemoryAvailability {
  uint64_t available_bytes = 0;
  std::string source;
  std::optional<uint64_t> cgroup_memory_max_bytes;
  std::optional<uint64_t> cgroup_memory_current_bytes;
  std::optional<uint64_t> cgroup_inactive_file_bytes;
};

struct EnginePinnedConcurrencySizing {
  uint64_t required_slices = 0;
  int effective_gpu_count = 0;
};

// Reserve headroom=min(10% of required, 10 GiB).
uint64_t compute_startup_memory_headroom_bytes(uint64_t required_bytes);

// Computes startup sizing requirements for pinned_memory.classes[name=engine].
// In fake-CUDA mode we intentionally size to one logical GPU so local smoke
// and unit-test workflows do not require production-scale pinned pools.
EnginePinnedConcurrencySizing compute_engine_pinned_concurrency_sizing(
    uint32_t streaming_buffer_chunks,
    int detected_gpu_count,
    bool fake_cuda_backend);

// Returns an effective "available bytes" view for startup admission checks.
// Prefers cgroup v2 headroom when memory.max is set; otherwise falls back to
// /proc/meminfo MemAvailable.
absl::StatusOr<StartupMemoryAvailability> detect_startup_memory_available_bytes();

// Fail-fast admission check for daemon startup fixed allocations.
// `pinned_bytes`: sum of pinned pool bytes (pinned_memory.classes[].pool_bytes).
// `stable_bytes`: engine.memory_tiers.stable_bytes (0 when unset).
absl::Status preflight_startup_memory(uint64_t pinned_bytes, uint64_t stable_bytes);

// Test-only hook to override detected available bytes. Pass std::nullopt to clear.
void set_startup_memory_available_override_for_testing(std::optional<uint64_t> bytes);

} // namespace tensorcast::daemon
