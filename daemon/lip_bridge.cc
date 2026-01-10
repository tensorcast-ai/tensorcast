// Copyright (c) 2025-2026, TensorCast Team.

// Implementation of LipBridge

#include "daemon/lip_bridge.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "core/store/device_registry.h"

namespace tensorcast::daemon {

absl::StatusOr<bool> LipBridge::try_satisfy_from_lip(
    absl::string_view artifact_id,
    int target_device_id,
    const std::function<void(const store::loading::ReplicaKey&)>& on_ready,
    v2::MemCopyHandle* out_handle) {
  if (!out_handle) {
    return absl::InvalidArgumentError("out_handle is null");
  }
  std::optional<LipLeaseEntry> lip_src = lip_.find_active_by_artifact_id(std::string(artifact_id));
  if (!lip_src.has_value())
    return false;
  if (lip_src->device_id == target_device_id) {
    return absl::FailedPreconditionError("lease_in_place not supported for same device_id consumers");
  }
  auto hbytes_or = lip_.copy_to_new_coalesced(
      target_device_id,
      lip_src->index_data,
      lip_src->total_size,
      absl::MakeSpan(lip_src->segments),
      absl::MakeSpan(lip_src->storages),
      lip_src->owner_pid);
  if (!hbytes_or.ok())
    return hbytes_or.status();

  // Prepare a ReplicaKey for callbacks
  store::loading::ReplicaKey rkey;
  rkey.artifact_id = std::string(artifact_id);
  rkey.device = store::DeviceRegistry::instance().gpu_key(target_device_id);
  rkey.replica = 0;
  on_ready(rkey);

  // Fill IPC handle bytes
  const auto& bytes = *hbytes_or;
  out_handle->set_cuda_ipc_handle(std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
  return true;
}

} // namespace tensorcast::daemon
