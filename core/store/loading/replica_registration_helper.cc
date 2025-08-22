// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loading/replica_registration_helper.h"

#include "absl/log/log.h"

namespace stepcast::store {

absl::Status ReplicaRegistrationHelper::register_local_replica(
    GlobalStoreClient* gs_client,
    std::string_view artifact_id,
    const DeviceKey& device,
    MemoryLocation location,
    uint64_t size_bytes) {
  if (gs_client == nullptr) {
    return absl::FailedPreconditionError("GlobalStoreClient is null");
  }

  // NOTE: For now we use the placeholder worker_id "local". In production this should
  // be replaced by the actual worker id obtained during worker registration.
  constexpr std::string_view kLocalWorkerId = "local";

  auto reg_or = gs_client->register_replica(
      artifact_id,
      kLocalWorkerId,
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

} // namespace stepcast::store