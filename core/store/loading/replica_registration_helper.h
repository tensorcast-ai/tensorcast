// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <string_view>
#include "absl/status/status.h"
#include "core/store/components/global_store_client.h"
#include "core/store/device_types.h"
#include "core/store/model/model_location.h"

namespace stepcast::store {

class ReplicaRegistrationHelper {
 public:
  // Registers the current process' replica of the given model with Global Store.
  // This is a thin wrapper around GlobalStoreClient::register_model_replica().
  static absl::Status register_local_replica(
      GlobalStoreClient* gs_client,
      std::string_view model_id,
      const DeviceKey& device,
      ModelLocation location,
      uint64_t size_bytes);
};

} // namespace stepcast::store