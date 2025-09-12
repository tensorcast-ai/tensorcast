// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <string_view>
#include "absl/status/statusor.h"
#include "core/store/components/global_store_client.h"
#include "core/store/device_types.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/store_engine.h"
#include "gsl/pointers"

namespace tensorcast::store::loading {

// MaterializeOrchestrator encapsulates the decision tree for replica materialization
// (remote replica selection, P2P transport setup, disk fallback, and
// replica registration with Global Store).  It is intended to be used from
// StoreEngine::materialize_replica() when mode == AUTO.
class MaterializeOrchestrator {
 public:
  MaterializeOrchestrator(gsl::not_null<StoreEngine*> store, gsl::not_null<components::GlobalStoreClient*> gs_client);

  // Execute the preparation logic.
  absl::StatusOr<ReplicaHandle> run(
      std::string_view artifact_id,
      const DeviceKey& target_device,
      const MaterializeHints& hints);

 private:
  gsl::not_null<StoreEngine*> store_;
  gsl::not_null<components::GlobalStoreClient*> gs_client_;
};

} // namespace tensorcast::store::loading
