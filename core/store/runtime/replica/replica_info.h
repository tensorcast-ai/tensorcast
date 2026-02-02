// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "core/common/memory/memory_location.h"
#include "core/store/materialization/contracts/loading_spec.h"

namespace tensorcast::store::runtime {

enum class ReplicaPublishState : std::uint8_t {
  kLocalOnly = 0,
  kPublishPending = 1,
  kPublished = 2,
  kRetiring = 3,
};

struct ReplicaInfo {
  loading::ReplicaKey key;
  std::string artifact_id;
  uint64_t size_bytes;
  common::memory::MemoryLocation cpu_state;
  common::memory::MemoryLocation gpu_state;
  int gpu_device_id;
  std::string gpu_device_uuid;
  bool is_registered_for_comm;
  std::chrono::time_point<std::chrono::system_clock> last_access_time;
  std::chrono::time_point<std::chrono::system_clock> load_time;
};

struct ReplicaInventoryEntry {
  loading::ReplicaKey key;
  uint64_t size_bytes{0};
  common::memory::MemoryLocation memory_location{common::memory::MemoryLocation::NONE};
  bool is_available{false};
  ReplicaPublishState publish_state{ReplicaPublishState::kLocalOnly};
  std::vector<std::string> remote_memory_keys;
  std::vector<uint64_t> buffer_sizes;
};

} // namespace tensorcast::store::runtime
