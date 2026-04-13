// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_target_storage_utils.h"

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "core/cuda/cuda_api.h"
#include "gsl/pointers"

namespace tensorcast::daemon::materialization_target_storage {

namespace {

struct ResolvedStorageSource {
  std::string region_id;
  tensorcast::daemon::IpcRegionRegistry::MemoryKind memory_kind{
      tensorcast::daemon::IpcRegionRegistry::MemoryKind::kVram};
};

absl::StatusOr<ResolvedStorageSource> resolve_storage_source(const v2::StorageEntry& storage) {
  switch (storage.storage_source_case()) {
    case v2::StorageEntry::kVramRegionId:
      if (storage.vram_region_id().empty()) {
        return absl::InvalidArgumentError("vram_region_id must not be empty");
      }
      return ResolvedStorageSource{
          .region_id = storage.vram_region_id(),
          .memory_kind = tensorcast::daemon::IpcRegionRegistry::MemoryKind::kVram,
      };
    case v2::StorageEntry::kCudaIpcHandle:
      return absl::InvalidArgumentError("storage entry must reference a managed vram region");
    case v2::StorageEntry::STORAGE_SOURCE_NOT_SET:
    default:
      return absl::InvalidArgumentError("storage entry must reference a region");
  }
}

} // namespace

std::string_view acquire_error_reason(AcquireTargetStoragesError error) {
  switch (error) {
    case AcquireTargetStoragesError::kRegionPoisoned:
      return "region_poisoned";
    case AcquireTargetStoragesError::kRegionMissing:
      return "region_missing";
    case AcquireTargetStoragesError::kMapFailed:
      return "map_failed";
    case AcquireTargetStoragesError::kDeviceMismatch:
      return "device_uuid_mismatch";
    case AcquireTargetStoragesError::kBounds:
      return "bounds";
    case AcquireTargetStoragesError::kUnknown:
    default:
      return "transfer_error";
  }
}

TargetStorageLease::TargetStorageLease(TargetStorageLease&& other) noexcept
    : registry_(other.registry_),
      acquired_region_ids_(std::move(other.acquired_region_ids_)),
      region_map_(std::move(other.region_map_)),
      storages_(std::move(other.storages_)) {
  other.registry_ = nullptr;
  other.acquired_region_ids_.clear();
}

TargetStorageLease& TargetStorageLease::operator=(TargetStorageLease&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release_now();
  registry_ = other.registry_;
  acquired_region_ids_ = std::move(other.acquired_region_ids_);
  region_map_ = std::move(other.region_map_);
  storages_ = std::move(other.storages_);
  other.registry_ = nullptr;
  other.acquired_region_ids_.clear();
  return *this;
}

TargetStorageLease::~TargetStorageLease() {
  release_now();
}

const std::vector<store::loading::IntoTargetStorage>& TargetStorageLease::storages() const {
  return storages_;
}

const std::vector<std::string>& TargetStorageLease::acquired_region_ids() const {
  return acquired_region_ids_;
}

absl::StatusOr<TargetStorageLease> TargetStorageLease::acquire(
    IpcRegionRegistry& registry,
    const google::protobuf::RepeatedPtrField<v2::StorageEntry>& storages,
    int owner_pid,
    AcquireTargetStoragesError* error) {
  if (error != nullptr) {
    *error = AcquireTargetStoragesError::kUnknown;
  }

  TargetStorageLease lease;
  lease.registry_ = &registry;
  lease.region_map_.reserve(storages.size());
  lease.acquired_region_ids_.reserve(storages.size());
  lease.storages_.reserve(storages.size());

  for (const auto& storage : storages) {
    auto storage_source_or = resolve_storage_source(storage);
    if (!storage_source_or.ok()) {
      return storage_source_or.status();
    }
    const auto& storage_source = *storage_source_or;
    auto it = lease.region_map_.find(storage_source.region_id);
    if (it == lease.region_map_.end()) {
      RegionMapping mapping;
      if (storage_source.memory_kind == IpcRegionRegistry::MemoryKind::kVram) {
        auto region_desc_or = registry.acquire(storage_source.region_id, owner_pid);
        if (!region_desc_or.ok()) {
          if (error != nullptr) {
            const bool poisoned = absl::IsFailedPrecondition(region_desc_or.status()) &&
                region_desc_or.status().message() == "region is poisoned";
            *error =
                poisoned ? AcquireTargetStoragesError::kRegionPoisoned : AcquireTargetStoragesError::kRegionMissing;
          }
          return region_desc_or.status();
        }
        auto handle_or = registry.get_handle_bytes(storage_source.region_id);
        if (!handle_or.ok()) {
          if (error != nullptr) {
            *error = AcquireTargetStoragesError::kRegionMissing;
          }
          return handle_or.status();
        }
        auto set_device_status = cuda::set_device(storage.device_id());
        if (!set_device_status.ok()) {
          if (error != nullptr) {
            *error = AcquireTargetStoragesError::kMapFailed;
          }
          return set_device_status;
        }
        auto map_or = cuda::IpcMapping::open(*handle_or, cuda::OpenOptions{.flags = cudaIpcMemLazyEnablePeerAccess});
        if (!map_or.ok()) {
          if (error != nullptr) {
            *error = AcquireTargetStoragesError::kMapFailed;
          }
          return map_or.status();
        }
        mapping.desc = *region_desc_or;
        mapping.mapping = std::make_unique<cuda::IpcMapping>(std::move(*map_or));
        mapping.base_ptr = mapping.mapping->get();
      } else {
        auto local_mapping_or = registry.acquire_host_shared_local_mapping(storage_source.region_id, owner_pid);
        if (!local_mapping_or.ok()) {
          if (error != nullptr) {
            const bool poisoned = absl::IsFailedPrecondition(local_mapping_or.status()) &&
                local_mapping_or.status().message() == "region is poisoned";
            *error =
                poisoned ? AcquireTargetStoragesError::kRegionPoisoned : AcquireTargetStoragesError::kRegionMissing;
          }
          return local_mapping_or.status();
        }
        mapping.desc = local_mapping_or->region;
        mapping.base_ptr = local_mapping_or->base_ptr;
      }

      auto [inserted_it, _] = lease.region_map_.emplace(storage_source.region_id, std::move(mapping));
      it = inserted_it;
      lease.acquired_region_ids_.push_back(storage_source.region_id);
    }

    const auto& region_desc = it->second.desc;
    if (region_desc.memory_kind == IpcRegionRegistry::MemoryKind::kVram) {
      if (region_desc.device_id != storage.device_id()) {
        if (error != nullptr) {
          *error = AcquireTargetStoragesError::kDeviceMismatch;
        }
        return absl::FailedPreconditionError("region device does not match storage device");
      }
    } else if (storage.device_id() != -1) {
      if (error != nullptr) {
        *error = AcquireTargetStoragesError::kDeviceMismatch;
      }
      return absl::FailedPreconditionError("HOST_SHARED storage.device_id must be -1");
    }

    const uint64_t region_end = storage.mapping_base_offset() + storage.storage_length();
    if (region_end > region_desc.size_bytes) {
      if (error != nullptr) {
        *error = AcquireTargetStoragesError::kBounds;
      }
      return absl::FailedPreconditionError("region-backed storage exceeds region bounds");
    }

    if (it->second.base_ptr == nullptr) {
      if (error != nullptr) {
        *error = AcquireTargetStoragesError::kMapFailed;
      }
      return absl::FailedPreconditionError("region base pointer is unavailable");
    }

    void* region_base_ptr =
        static_cast<uint8_t*>(it->second.base_ptr) + static_cast<uint64_t>(storage.mapping_base_offset());
    lease.storages_.push_back(
        store::loading::IntoTargetStorage{
            .base_ptr = gsl::not_null<void*>{region_base_ptr},
            .length = storage.storage_length(),
        });
  }

  return lease;
}

void TargetStorageLease::release_now() {
  if (registry_ == nullptr) {
    return;
  }
  for (const auto& region_id : acquired_region_ids_) {
    registry_->release(region_id).IgnoreError();
  }
  acquired_region_ids_.clear();
}

} // namespace tensorcast::daemon::materialization_target_storage
