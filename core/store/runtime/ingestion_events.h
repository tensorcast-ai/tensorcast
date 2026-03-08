// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "core/common/memory/memory_location.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"

namespace tensorcast::store::runtime {

enum class IngestionSource : std::uint8_t {
  kDisk = 0,
  kP2P = 1,
  kMemory = 2,
};

struct IngestionStartedEvent {
  std::string request_id;
  std::string artifact_id;
  IngestionSource source{IngestionSource::kDisk};
  loading::ReplicaTarget target;
  loading::MaterializeMode materialize_mode{loading::MaterializeMode::AUTO};
  loading::ExportPolicy export_policy{loading::ExportPolicy::kNever};
  bool publish_to_global_store{true};
  std::string publish_context_id;
  std::optional<std::string> view_id;
};

struct IngestionResultEvent {
  std::string request_id;
  IngestionSource source{IngestionSource::kDisk};
  loading::MaterializeMode materialize_mode{loading::MaterializeMode::AUTO};
  loading::ExportPolicy export_policy{loading::ExportPolicy::kNever};
  std::string artifact_id;
  DeviceKey target_device;
  common::memory::MemoryLocation target_location{common::memory::MemoryLocation::CPU};
  uint64_t bytes_transferred{0};
  double duration_seconds{0.0};
  absl::Status status;
  std::optional<loading::ReplicaKey> replica_key;
  std::optional<std::string> view_id;
  bool publish_to_global_store{true};
  std::string publish_context_id;
};

using IngestionCompletedEvent = IngestionResultEvent;

} // namespace tensorcast::store::runtime
