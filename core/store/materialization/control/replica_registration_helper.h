// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <string_view>
#include "absl/status/status.h"
#include "core/common/memory/memory_location.h"
#include "core/store/components/global_store_client.h"
#include "core/store/device_types.h"
#include "gsl/pointers"

namespace tensorcast::store::materialization::control {

class ReplicaRegistrationHelper {
 public:
  // Registers the current process' replica of the given artifact with Global Store.
  // This is a thin wrapper around GlobalStoreClient::register_replica().
  static absl::Status register_local_replica(
      gsl::not_null<components::IGlobalStoreClient*> gs_client,
      std::string_view worker_id,
      std::string_view artifact_id,
      const DeviceKey& device,
      common::memory::MemoryLocation location,
      uint64_t size_bytes);
};

} // namespace tensorcast::store::materialization::control
