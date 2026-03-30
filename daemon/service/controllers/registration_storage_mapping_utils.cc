// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/registration_storage_mapping_utils.h"

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::daemon {

void release_region_refs(IpcRegionRegistry& registry, const absl::flat_hash_map<std::string, uint32_t>& refs) {
  for (const auto& [region_id, count] : refs) {
    for (uint32_t i = 0; i < count; ++i) {
      absl::Status st = registry.release(region_id);
      if (!st.ok()) {
        LOG(WARNING) << "release_region_refs: release failed for region=" << region_id << ": " << st;
      }
    }
  }
}

RegionPinGuard::RegionPinGuard(IpcRegionRegistry& registry) : registry_(registry) {}

RegionPinGuard::~RegionPinGuard() {
  if (!active_) {
    return;
  }
  release_region_refs(registry_, refs_);
}

void RegionPinGuard::add(const std::string& region_id) {
  ++refs_[region_id];
}

void RegionPinGuard::release() {
  active_ = false;
  refs_.clear();
}

const absl::flat_hash_map<std::string, uint32_t>& RegionPinGuard::refs() const {
  return refs_;
}

absl::StatusOr<cuda::IpcMapping*> get_or_open_mapping_for_storage(
    const RegisterStorageMeta& storage,
    absl::flat_hash_map<std::string, std::unique_ptr<cuda::IpcMapping>>& cache,
    RegionPinGuard& region_pin,
    IpcRegionRegistry& regions,
    int owner_pid) {
  const std::string cache_key =
      storage.has_handle() ? absl::StrCat("h:", storage.handle_bytes) : absl::StrCat("r:", storage.region_id);
  auto it = cache.find(cache_key);
  if (it != cache.end()) {
    return it->second.get();
  }

  std::unique_ptr<cuda::IpcMapping> mapping;
  if (storage.has_handle()) {
    auto set_device_status = cuda::set_device(storage.device_id);
    if (!set_device_status.ok()) {
      return set_device_status;
    }
    auto map_or =
        cuda::IpcMapping::open(storage.handle_bytes, cuda::OpenOptions{.flags = cudaIpcMemLazyEnablePeerAccess});
    if (!map_or.ok()) {
      return map_or.status();
    }
    mapping = std::make_unique<cuda::IpcMapping>(std::move(*map_or));
  } else if (storage.has_region()) {
    auto acq_or = regions.acquire(storage.region_id, owner_pid);
    if (!acq_or.ok()) {
      return acq_or.status();
    }
    region_pin.add(storage.region_id);
    auto handle_or = regions.get_handle_bytes(storage.region_id);
    if (!handle_or.ok()) {
      return handle_or.status();
    }
    auto set_device_status = cuda::set_device(storage.device_id);
    if (!set_device_status.ok()) {
      return set_device_status;
    }
    auto map_or = cuda::IpcMapping::open(*handle_or, cuda::OpenOptions{.flags = cudaIpcMemLazyEnablePeerAccess});
    if (!map_or.ok()) {
      return map_or.status();
    }
    mapping = std::make_unique<cuda::IpcMapping>(std::move(*map_or));
  } else {
    return absl::InvalidArgumentError("storage entry missing source handle or region");
  }

  auto [insert_it, _] = cache.emplace(cache_key, std::move(mapping));
  return insert_it->second.get();
}

} // namespace tensorcast::daemon
