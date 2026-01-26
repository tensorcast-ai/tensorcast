// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/control/replica_registration_helper.h"

#include "absl/log/log.h"

namespace tensorcast::store::materialization::control {

absl::Status ReplicaRegistrationHelper::register_local_replica(
    gsl::not_null<components::IGlobalStoreClient*> gs_client,
    std::string_view worker_id,
    std::string_view artifact_id,
    const DeviceKey& device,
    common::memory::MemoryLocation location,
    uint64_t size_bytes,
    std::optional<std::string_view> view_id) {
  auto reg_or = gs_client->register_replica(
      artifact_id,
      worker_id,
      device,
      location,
      size_bytes,
      /*max_concurrency=*/1,
      view_id);

  if (!reg_or.ok()) {
    LOG(WARNING) << "Failed to register replica replica with Global Store: " << reg_or.status();
    return reg_or.status();
  }
  VLOG(1) << "Registered local replica replica for " << artifact_id << " with replica_id " << *reg_or;
  return absl::OkStatus();
}

} // namespace tensorcast::store::materialization::control
