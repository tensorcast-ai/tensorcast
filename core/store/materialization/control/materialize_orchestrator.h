// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/worker_identity.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/control/materialization_backend.h"
#include "gsl/pointers"

namespace tensorcast::store::materialization::control {

using tensorcast::store::loading::MaterializeHints;
using tensorcast::store::loading::ReplicaHandle;

// MaterializeOrchestrator encapsulates the decision tree for replica materialization
// (remote replica selection, P2P transport setup, disk fallback, and
// replica registration with Global Store).  It is intended to be used from
// StoreEngine::materialize_replica() when mode == AUTO.
class MaterializeOrchestrator {
 public:
  MaterializeOrchestrator(
      gsl::not_null<MaterializationBackend*> backend,
      gsl::not_null<components::IGlobalStoreClient*> gs_client,
      components::WorkerIdentity local_identity);

  // Execute the preparation logic.
  absl::StatusOr<ReplicaHandle> run(
      std::string_view artifact_id,
      const DeviceKey& target_device,
      const MaterializeHints& hints,
      const std::optional<loading::DiskSource>& disk_source);

 private:
  gsl::not_null<MaterializationBackend*> backend_;
  gsl::not_null<components::IGlobalStoreClient*> gs_client_;
  components::WorkerIdentity local_identity_;
};

} // namespace tensorcast::store::materialization::control
