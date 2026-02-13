// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <string_view>
#include "absl/status/statusor.h"
#include "core/common/memory/memory_location.h"
#include "core/store/components/global_store_client.h"
#include "core/store/device_types.h"
#include "gsl/pointers"

namespace tensorcast::store::materialization::control {

class ReplicaRegistrationHelper {
 public:
  // Registers the current process' replica of the given artifact with Global Store.
  // This is a thin wrapper around GlobalStoreClient::register_replica().
  static absl::StatusOr<std::string> register_local_replica(
      gsl::not_null<components::IGlobalStoreClient*> gs_client,
      std::string_view worker_id,
      std::string_view artifact_id,
      const DeviceKey& device,
      common::memory::MemoryLocation location,
      uint64_t size_bytes,
      std::optional<std::string_view> view_id = std::nullopt,
      std::optional<std::string_view> client_request_id = std::nullopt);
};

} // namespace tensorcast::store::materialization::control
