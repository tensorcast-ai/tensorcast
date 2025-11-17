// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/replica_registry.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/contracts/materialization_request.h"
#include "core/store/replica/replica.h"
#include "gsl/pointers"

namespace tensorcast::store::materialization::control {

using tensorcast::store::loading::DiskSource;
using tensorcast::store::loading::MaterializationRequest;
using tensorcast::store::loading::MaterializeHints;
using tensorcast::store::loading::MaterializeMode;
using tensorcast::store::loading::ReplicaHandle;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::loading::ReplicaTarget;

struct MaterializationDeps {
  MaterializationDeps(
      gsl::not_null<components::ReplicaRegistry*> registry,
      gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> pool)
      : replica_registry(registry), memory_pool(pool) {}

  gsl::not_null<components::ReplicaRegistry*> replica_registry;
  gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> memory_pool;
  size_t artifact_chunk_bytes = 0;
  std::chrono::milliseconds pinned_memory_timeout{0};
  int num_threads = 0;
  std::function<absl::StatusOr<ReplicaHandle>(const MaterializationRequest& request)> run_auto;
  std::function<absl::StatusOr<ReplicaHandle>(
      const std::string& artifact_identifier,
      const DiskSource& source,
      const ReplicaTarget& target,
      const MaterializeHints& hints)>
      ingest_from_disk;
  std::function<std::optional<std::string>(
      replica::Replica& replica,
      common::memory::MemoryLocation location,
      uint64_t view_size_bytes,
      std::optional<int> gpu_device_id)>
      compute_view_hash;
};

class MaterializationService {
 public:
  explicit MaterializationService(MaterializationDeps deps);

  absl::StatusOr<ReplicaHandle> Execute(const MaterializationRequest& request);

 private:
  MaterializationDeps deps_;

  absl::StatusOr<ReplicaHandle> TryReuseReplica(const MaterializationRequest& request) const;
  absl::StatusOr<ReplicaHandle> CopyFromPeer(const MaterializationRequest& request) const;
  absl::StatusOr<ReplicaHandle> LoadFromDisk(const MaterializationRequest& request) const;
  absl::StatusOr<ReplicaHandle> RunAuto(const MaterializationRequest& request) const;
  ReplicaHandle BuildHandle(
      const MaterializationRequest& request,
      const std::shared_ptr<replica::Replica>& replica,
      std::shared_future<absl::Status> ready_future) const;
};

} // namespace tensorcast::store::materialization::control
