// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <string>

#include "core/common/memory/memory_location.h"

namespace tensorcast::store::components::runtime {

struct ReplicaInfo {
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

} // namespace tensorcast::store::components::runtime
