// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/external_target_access_service.h"

#include <format>
#include <optional>

#include "daemon/service/controllers/materialization_target_storage_utils.h"
#include "daemon/util/grpc_peer_utils.h"

namespace tensorcast::daemon {

using materialization_target_storage::AcquireTargetStoragesError;
using materialization_target_storage::TargetStorageLease;

absl::Status validate_host_shared_target_region_classes(
    IpcRegionRegistry& regions,
    const v2::TargetLayout& layout,
    std::string_view rpc_name) {
  for (const auto& storage : layout.storages()) {
    if (storage.storage_source_case() != v2::StorageEntry::kRegionRef ||
        storage.region_ref().memory_kind() != v2::REGION_MEMORY_KIND_HOST_SHARED) {
      continue;
    }
    auto desc_or = regions.describe(storage.region_ref().region_id());
    if (!desc_or.ok()) {
      return desc_or.status();
    }
    if (desc_or->host_region_class == IpcRegionRegistry::HostRegionClass::kAllocator) {
      return absl::InvalidArgumentError(
          std::format(
              "{} does not support generic target validation for allocator-backed HOST_SHARED regions", rpc_name));
    }
  }
  return absl::OkStatus();
}

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
  auto device_or = resolve_target_storage_device(layout, device_uuid, rpc_name);
  if (!device_or.ok()) {
    return device_or.status();
  }

  AcquireTargetStoragesError acquire_error = AcquireTargetStoragesError::kUnknown;
  auto storage_lease_or = TargetStorageLease::acquire(d_.regions, layout.storages(), owner_pid, &acquire_error);
  if (!storage_lease_or.ok()) {
    return storage_lease_or.status();
  }
  if (d_.comm_manager != nullptr) {
    auto activate_status =
        materialization_target_storage::activate_stable_local_backings(*d_.comm_manager, storage_lease_or->storages());
    if (!activate_status.ok()) {
      return activate_status;
    }
  }
  auto host_region_class_status = validate_host_shared_target_region_classes(d_.regions, layout, rpc_name);
  if (!host_region_class_status.ok()) {
    return host_region_class_status;
  }

  return ValidatedTargetAccess{
      .device = *device_or,
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
  if (d_.comm_manager != nullptr) {
    auto activate_status = layout_or->activate_stable_local_backings(*d_.comm_manager);
    if (!activate_status.ok()) {
      return activate_status;
    }
  }
  return ValidatedSourceAccess{.layout = std::move(*layout_or)};
}

absl::StatusOr<store::DeviceKey> ExternalTargetAccessService::resolve_target_storage_device(
    const v2::TargetLayout& layout,
    std::string_view device_uuid,
    std::string_view /*rpc_name*/) const {
  std::optional<bool> host_shared_layout;
  std::optional<int32_t> vram_device_id;
  for (const auto& storage : layout.storages()) {
    bool storage_is_host_shared = false;
    switch (storage.storage_source_case()) {
      case v2::StorageEntry::kVramRegionId:
        if (storage.vram_region_id().empty()) {
          return absl::InvalidArgumentError("vram_region_id must not be empty");
        }
        storage_is_host_shared = false;
        break;
      case v2::StorageEntry::kRegionRef:
        if (storage.region_ref().region_id().empty()) {
          return absl::InvalidArgumentError("region_ref.region_id must not be empty");
        }
        switch (storage.region_ref().memory_kind()) {
          case v2::REGION_MEMORY_KIND_VRAM:
            storage_is_host_shared = false;
            break;
          case v2::REGION_MEMORY_KIND_HOST_SHARED:
            storage_is_host_shared = true;
            break;
          case v2::REGION_MEMORY_KIND_UNSPECIFIED:
          default:
            return absl::InvalidArgumentError("region_ref.memory_kind must be specified");
        }
        break;
      case v2::StorageEntry::STORAGE_SOURCE_NOT_SET:
      default:
        return absl::InvalidArgumentError("target storage must reference a region");
    }
    if (!host_shared_layout.has_value()) {
      host_shared_layout = storage_is_host_shared;
    } else if (*host_shared_layout != storage_is_host_shared) {
      return absl::InvalidArgumentError("mixed VRAM and HOST_SHARED target layouts are not supported");
    }
    if (storage_is_host_shared) {
      if (storage.device_id() != -1) {
        return absl::InvalidArgumentError("HOST_SHARED storage.device_id must be -1");
      }
      if (!device_uuid.empty()) {
        return absl::InvalidArgumentError("device_uuid must be empty for pure HOST_SHARED target layouts");
      }
      continue;
    }
    if (storage.device_id() < 0) {
      return absl::InvalidArgumentError("storage.device_id must be >= 0");
    }
    if (device_uuid.empty()) {
      return absl::InvalidArgumentError("device_uuid is required for VRAM target layouts");
    }
    if (!vram_device_id.has_value()) {
      vram_device_id = storage.device_id();
    } else if (*vram_device_id != storage.device_id()) {
      return absl::InvalidArgumentError("target_layout must use one VRAM device_id");
    }
  }
  if (!host_shared_layout.has_value()) {
    return absl::InvalidArgumentError("target_layout must include at least one storage entry");
  }
  if (*host_shared_layout) {
    return store::DeviceKey{
        .type = DeviceType::CPU,
        .ordinal = -1,
        .uuid = "",
    };
  }
  const auto device = d_.devices.From(v2::DeviceType::DEVICE_TYPE_GPU, device_uuid, std::nullopt);
  if (device.type != DeviceType::GPU || device.ordinal < 0) {
    return absl::InvalidArgumentError("device_uuid must resolve to a GPU device");
  }
  if (vram_device_id.has_value() && *vram_device_id != device.ordinal) {
    return absl::InvalidArgumentError("storage.device_id does not match device_uuid");
  }
  return device;
}

} // namespace tensorcast::daemon
