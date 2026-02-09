// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "core/cuda/cuda_ipc.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/types.h"

namespace tensorcast::daemon {

void release_region_refs(IpcRegionRegistry& registry, const absl::flat_hash_map<std::string, uint32_t>& refs);

class RegionPinGuard {
 public:
  explicit RegionPinGuard(IpcRegionRegistry& registry);
  ~RegionPinGuard();

  void add(const std::string& region_id);

  void release();

  const absl::flat_hash_map<std::string, uint32_t>& refs() const;

 private:
  IpcRegionRegistry& registry_;
  absl::flat_hash_map<std::string, uint32_t> refs_;
  bool active_{true};
};

absl::StatusOr<cuda::IpcMapping*> get_or_open_mapping_for_storage(
    const RegisterStorageMeta& storage,
    absl::flat_hash_map<std::string, std::unique_ptr<cuda::IpcMapping>>& cache,
    RegionPinGuard& region_pin,
    IpcRegionRegistry& regions,
    int owner_pid);

} // namespace tensorcast::daemon
