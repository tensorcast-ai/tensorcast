// Copyright (c) 2025, TensorCast Team.

#include "core/store/loading/replica_registration_helper.h"

#include "absl/log/log.h"

namespace tensorcast::store::loading {

absl::Status ReplicaRegistrationHelper::register_local_replica(
    gsl::not_null<components::GlobalStoreClient*> gs_client,
    std::string_view worker_id,
    std::string_view artifact_id,
    const DeviceKey& device,
    common::memory::MemoryLocation location,
    uint64_t size_bytes) {
  auto reg_or = gs_client->register_replica(
      artifact_id,
      worker_id,
      device,
      location,
      size_bytes,
      /*max_concurrency=*/1);

  if (!reg_or.ok()) {
    LOG(WARNING) << "Failed to register replica replica with Global Store: " << reg_or.status();
    return reg_or.status();
  }
  VLOG(1) << "Registered local replica replica for " << artifact_id << " with replica_id " << *reg_or;
  return absl::OkStatus();
}

} // namespace tensorcast::store::loading
