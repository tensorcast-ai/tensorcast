// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "core/store/device_types.h"
#include "daemon/service/byte_artifact_region_layout.h"
#include "daemon/service/controllers/materialization_target_storage_utils.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/ipc_region_registry.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class ExternalTargetAccessService {
 public:
  struct Dep {
    DeviceResolver& devices;
    IpcRegionRegistry& regions;
  };

  struct ValidatedTargetAccess {
    store::DeviceKey device;
    materialization_target_storage::TargetStorageLease storage_lease;
  };

  struct ValidatedSourceAccess {
    ByteArtifactRegionLayout layout;
  };

  explicit ExternalTargetAccessService(Dep d);

  [[nodiscard]] absl::Status ensure_local_region_peer(std::string_view peer, std::string_view rpc_name) const;

  [[nodiscard]] absl::StatusOr<ValidatedTargetAccess> validate_local_target_layout(
      std::string_view peer,
      std::string_view rpc_name,
      const v2::TargetLayout& layout,
      int owner_pid,
      std::string_view device_uuid) const;

  [[nodiscard]] absl::StatusOr<ValidatedSourceAccess> validate_local_source_layout(
      std::string_view peer,
      std::string_view rpc_name,
      const v2::TargetLayout& layout,
      int owner_pid,
      std::string_view device_uuid,
      const absl::flat_hash_map<std::string, std::uint64_t>& expected_lengths) const;

 private:
  [[nodiscard]] absl::Status validate_target_storage_device(
      const v2::TargetLayout& layout,
      const store::DeviceKey& device,
      std::string_view rpc_name) const;

  Dep d_;
};

} // namespace tensorcast::daemon
