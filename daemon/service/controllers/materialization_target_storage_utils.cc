// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_target_storage_utils.h"

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "core/cuda/cuda_api.h"
#include "gsl/pointers"

namespace tensorcast::daemon::materialization_target_storage {

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
    auto it = lease.region_map_.find(storage.vram_region_id());
    if (it == lease.region_map_.end()) {
      auto region_desc_or = registry.acquire(storage.vram_region_id(), owner_pid);
      if (!region_desc_or.ok()) {
        if (error != nullptr) {
          const bool poisoned = absl::IsFailedPrecondition(region_desc_or.status()) &&
              region_desc_or.status().message() == "region is poisoned";
          *error = poisoned ? AcquireTargetStoragesError::kRegionPoisoned : AcquireTargetStoragesError::kRegionMissing;
        }
        return region_desc_or.status();
      }
      auto handle_or = registry.get_handle_bytes(storage.vram_region_id());
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
      RegionMapping mapping{.desc = *region_desc_or, .mapping = std::make_unique<cuda::IpcMapping>(std::move(*map_or))};
      auto [inserted_it, _] = lease.region_map_.emplace(storage.vram_region_id(), std::move(mapping));
      it = inserted_it;
      lease.acquired_region_ids_.push_back(storage.vram_region_id());
    }

    const auto& region_desc = it->second.desc;
    if (region_desc.device_id != storage.device_id()) {
      if (error != nullptr) {
        *error = AcquireTargetStoragesError::kDeviceMismatch;
      }
      return absl::FailedPreconditionError("region device does not match storage device");
    }

    const uint64_t region_end = storage.mapping_base_offset() + storage.storage_length();
    if (region_end > region_desc.size_bytes) {
      if (error != nullptr) {
        *error = AcquireTargetStoragesError::kBounds;
      }
      return absl::FailedPreconditionError("region-backed storage exceeds region bounds");
    }

    void* region_base_ptr =
        static_cast<uint8_t*>(it->second.mapping->get()) + static_cast<uint64_t>(storage.mapping_base_offset());
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
