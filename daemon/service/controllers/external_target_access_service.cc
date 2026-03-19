// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/external_target_access_service.h"

#include <format>

#include "daemon/service/controllers/materialization_target_storage_utils.h"
#include "daemon/util/grpc_peer_utils.h"

namespace tensorcast::daemon {

using materialization_target_storage::AcquireTargetStoragesError;
using materialization_target_storage::TargetStorageLease;

ExternalTargetAccessService::ExternalTargetAccessService(Dep d) : d_(d) {}

absl::Status ExternalTargetAccessService::ensure_local_region_peer(std::string_view peer, std::string_view rpc_name)
    const {
  if (!is_loopback_grpc_peer(peer)) {
    return absl::PermissionDeniedError(std::format("{} is local-only (loopback/UDS)", rpc_name));
  }
  return absl::OkStatus();
}

absl::StatusOr<ExternalTargetAccessService::ValidatedTargetAccess> ExternalTargetAccessService::
    validate_local_target_layout(
        std::string_view peer,
        std::string_view rpc_name,
        const v2::TargetLayout& layout,
        int owner_pid,
        std::string_view device_uuid) const {
  auto peer_status = ensure_local_region_peer(peer, rpc_name);
  if (!peer_status.ok()) {
    return peer_status;
  }
  if (owner_pid <= 0) {
    return absl::InvalidArgumentError("pid is required");
  }
  if (layout.storages_size() == 0) {
    return absl::InvalidArgumentError("target_layout must include at least one storage entry");
  }
  if (device_uuid.empty()) {
    return absl::InvalidArgumentError("device_uuid is required");
  }

  const auto device = d_.devices.From(v2::DeviceType::DEVICE_TYPE_GPU, device_uuid, std::nullopt);
  auto device_status = validate_target_storage_device(layout, device, rpc_name);
  if (!device_status.ok()) {
    return device_status;
  }

  AcquireTargetStoragesError acquire_error = AcquireTargetStoragesError::kUnknown;
  auto storage_lease_or = TargetStorageLease::acquire(d_.regions, layout.storages(), owner_pid, &acquire_error);
  if (!storage_lease_or.ok()) {
    return storage_lease_or.status();
  }

  return ValidatedTargetAccess{
      .device = device,
      .storage_lease = std::move(*storage_lease_or),
  };
}

absl::StatusOr<ExternalTargetAccessService::ValidatedSourceAccess> ExternalTargetAccessService::
    validate_local_source_layout(
        std::string_view peer,
        std::string_view rpc_name,
        const v2::TargetLayout& layout,
        int owner_pid,
        std::string_view device_uuid,
        const absl::flat_hash_map<std::string, std::uint64_t>& expected_lengths) const {
  auto peer_status = ensure_local_region_peer(peer, rpc_name);
  if (!peer_status.ok()) {
    return peer_status;
  }
  if (owner_pid <= 0) {
    return absl::InvalidArgumentError("pid is required");
  }
  auto layout_or = ByteArtifactRegionLayout::acquire(d_.regions, layout, owner_pid, device_uuid, expected_lengths);
  if (!layout_or.ok()) {
    return layout_or.status();
  }
  return ValidatedSourceAccess{.layout = std::move(*layout_or)};
}

absl::Status ExternalTargetAccessService::validate_target_storage_device(
    const v2::TargetLayout& layout,
    const store::DeviceKey& device,
    std::string_view /*rpc_name*/) const {
  if (device.type != DeviceType::GPU || device.ordinal < 0) {
    return absl::InvalidArgumentError("device_uuid must resolve to a GPU device");
  }
  for (const auto& storage : layout.storages()) {
    if (storage.storage_source_case() != v2::StorageEntry::kVramRegionId) {
      return absl::InvalidArgumentError("Target storage must reference a vram_region_id");
    }
    if (storage.device_id() != device.ordinal) {
      return absl::InvalidArgumentError("storage.device_id does not match device_uuid");
    }
  }
  return absl::OkStatus();
}

} // namespace tensorcast::daemon
