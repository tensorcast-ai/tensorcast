// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/replica/replica_placement.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/store/device_registry.h"

namespace tensorcast::store::replica {

absl::StatusOr<DeviceKey> resolve_replica_config_device_key(const ReplicaConfig& config) {
  if (config.device_type == DeviceType::GPU) {
    if (config.local_device_id < 0) {
      return absl::InvalidArgumentError(
          "ReplicaConfig placement is inconsistent: device_type=GPU requires local_device_id >= 0");
    }
    return DeviceRegistry::instance().gpu_key(config.local_device_id);
  }

  if (config.device_type == DeviceType::CPU) {
    if (config.local_device_id >= 0) {
      return absl::InvalidArgumentError(
          "ReplicaConfig placement is inconsistent: device_type=CPU requires local_device_id=-1; use "
          "device_type=GPU for GPU replicas");
    }
    return DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  }

  return absl::InvalidArgumentError(
      absl::StrCat(
          "ReplicaConfig device_type=", to_string(config.device_type), " is not supported for local Replica creation"));
}

DeviceKey normalize_replica_device_key(DeviceKey key) {
  if (key.type == DeviceType::GPU) {
    return DeviceRegistry::instance().normalize(key);
  }
  if (key.type == DeviceType::CPU) {
    return DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  }
  return key;
}

} // namespace tensorcast::store::replica
