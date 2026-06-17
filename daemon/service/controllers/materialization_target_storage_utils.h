// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/cuda/cuda_ipc.h"
#include "core/store/components/communication_manager.h"
#include "core/store/store_engine.h"
#include "daemon/state/ipc_region_registry.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::materialization_target_storage {

enum class AcquireTargetStoragesError {
  kUnknown,
  kRegionPoisoned,
  kRegionMissing,
  kMapFailed,
  kDeviceMismatch,
  kBounds,
};

std::string_view acquire_error_reason(AcquireTargetStoragesError error);

absl::Status activate_stable_local_backings(
    store::components::CommunicationManager& comm_manager,
    absl::Span<const store::loading::IntoTargetStorage> storages);

class TargetStorageLease {
 public:
  TargetStorageLease() = default;
  TargetStorageLease(const TargetStorageLease&) = delete;
  TargetStorageLease& operator=(const TargetStorageLease&) = delete;
  TargetStorageLease(TargetStorageLease&& other) noexcept;
  TargetStorageLease& operator=(TargetStorageLease&& other) noexcept;
  ~TargetStorageLease();

  const std::vector<store::loading::IntoTargetStorage>& storages() const;

  const std::vector<std::string>& acquired_region_ids() const;

  static absl::StatusOr<TargetStorageLease> acquire(
      IpcRegionRegistry& registry,
      const google::protobuf::RepeatedPtrField<v2::StorageEntry>& storages,
      int owner_pid,
      AcquireTargetStoragesError* error);

 private:
  struct SharedState {
    IpcRegionRegistry* registry{nullptr};
    std::vector<std::string> acquired_region_ids;

    ~SharedState();
  };

  struct RegionMapping {
    IpcRegionRegistry::RegionDescriptor desc;
    void* base_ptr{nullptr};
    std::unique_ptr<cuda::IpcMapping> mapping;
  };

  void release_now();

  std::shared_ptr<SharedState> shared_state_;
  absl::flat_hash_map<std::string, RegionMapping> region_map_;
  std::vector<store::loading::IntoTargetStorage> storages_;
};

} // namespace tensorcast::daemon::materialization_target_storage
