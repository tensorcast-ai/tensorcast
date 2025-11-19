// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "core/common/memory/memory_location.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"

namespace tensorcast::store::runtime {

enum class IngestionSource {
  kDisk = 0,
  kP2P = 1,
};

struct IngestionResultEvent {
  IngestionSource source{IngestionSource::kDisk};
  std::string artifact_id;
  DeviceKey target_device;
  common::memory::MemoryLocation target_location{common::memory::MemoryLocation::CPU};
  uint64_t bytes_transferred{0};
  double duration_seconds{0.0};
  absl::Status status;
  std::optional<loading::ReplicaKey> replica_key;
  std::optional<std::string> view_id;
  bool publish_to_global_store{true};
};

} // namespace tensorcast::store::runtime
