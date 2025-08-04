// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <string_view>
#include "absl/status/statusor.h"
#include "core/store/device_types.h"
#include "core/store/loading/loading_spec.h"

namespace stepcast::store {

class GlobalStoreClient;
class ModelRegistry;
class DeviceManager;
class CheckpointStore;

// PrepareOrchestrator encapsulates the decision tree for model preparation
// (remote replica selection, P2P transport setup, disk fallback, and
// replica registration with Global Store).  It is intended to be used from
// CheckpointStore::prepare() when mode == AUTO.
class PrepareOrchestrator {
 public:
  PrepareOrchestrator(
      CheckpointStore* store,
      GlobalStoreClient* gs_client,
      ModelRegistry* registry,
      DeviceManager* device_manager);

  // Execute the preparation logic.
  absl::StatusOr<ModelHandle> run(std::string_view model_id, const DeviceKey& target_device, const LoadingHints& hints);

 private:
  CheckpointStore* store_;
  GlobalStoreClient* gs_client_;
  ModelRegistry* registry_;
  DeviceManager* device_manager_;
};

} // namespace stepcast::store