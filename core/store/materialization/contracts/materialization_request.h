// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "core/common/memory/memory_location.h"
#include "core/store/components/device_manager.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"

namespace tensorcast::store::loading {

class MaterializationRequest {
 public:
  static absl::StatusOr<MaterializationRequest> Create(
      DeviceKey target_device,
      MaterializeMode mode,
      const MaterializeHints& hints,
      const components::DeviceManager& device_manager,
      std::optional<DiskSource> disk_source = std::nullopt);

  [[nodiscard]] const ReplicaKey& replica_key() const {
    return replica_key_;
  }

  [[nodiscard]] const MaterializeHints& hints() const {
    return hints_;
  }

  [[nodiscard]] MaterializeMode mode() const {
    return mode_;
  }

  [[nodiscard]] const DeviceKey& target_device() const {
    return target_device_;
  }

  [[nodiscard]] common::memory::MemoryLocation target_location() const {
    return target_location_;
  }

  [[nodiscard]] const std::optional<std::string>& requested_view_id() const {
    return requested_view_id_;
  }

  [[nodiscard]] const std::string& canonical_artifact_id() const {
    return canonical_artifact_id_;
  }

  [[nodiscard]] bool target_is_gpu() const {
    return target_device_.type == DeviceType::GPU;
  }

  [[nodiscard]] bool target_is_cpu() const {
    return target_device_.type == DeviceType::CPU;
  }

  [[nodiscard]] bool has_disk_source() const {
    return disk_source_.has_value();
  }

  [[nodiscard]] const std::optional<DiskSource>& disk_source() const {
    return disk_source_;
  }

 private:
  MaterializeMode mode_ = MaterializeMode::AUTO;
  DeviceKey target_device_{};
  ReplicaKey replica_key_{};
  MaterializeHints hints_{};
  std::optional<std::string> requested_view_id_;
  std::string canonical_artifact_id_;
  std::optional<DiskSource> disk_source_;
  common::memory::MemoryLocation target_location_{common::memory::MemoryLocation::CPU};
};

} // namespace tensorcast::store::loading
