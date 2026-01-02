// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "core/common/async_runtime.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/replica_registry.h"
#include "core/store/materialization/common/view_hash_utils.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/contracts/materialization_request.h"
#include "core/store/replica/replica.h"
#include "gsl/pointers"

namespace tensorcast::store::runtime::ingestion {

using loading::DiskSource;
using loading::MaterializationRequest;
using loading::MaterializeHints;
using loading::MaterializeMode;
using loading::ReplicaHandle;
using loading::ReplicaKey;
using loading::ReplicaTarget;

struct MaterializationDeps {
  MaterializationDeps(
      gsl::not_null<components::ReplicaRegistry*> registry,
      const gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>& pool)
      : replica_registry(registry), memory_pool(pool) {}

  gsl::not_null<components::ReplicaRegistry*> replica_registry;
  gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> memory_pool;
  std::shared_ptr<common::AsyncRuntime> async_runtime{nullptr};
  size_t artifact_chunk_bytes = 0;
  std::chrono::milliseconds pinned_memory_timeout{0};
  size_t streaming_buffer_chunks{16};
  int num_threads = 0;
  std::function<absl::StatusOr<ReplicaHandle>(const MaterializationRequest& request)> run_auto;
  std::function<absl::StatusOr<ReplicaHandle>(
      const std::string& artifact_identifier,
      const DiskSource& source,
      const ReplicaTarget& target,
      const MaterializeHints& hints)>
      ingest_from_disk;
  std::shared_ptr<ViewHashComputer> view_hash_computer;
};

class MaterializationService {
 public:
  explicit MaterializationService(MaterializationDeps deps);

  [[nodiscard]] absl::StatusOr<ReplicaHandle> execute(const MaterializationRequest& request);

 private:
  MaterializationDeps deps_;

  [[nodiscard]] absl::StatusOr<ReplicaHandle> try_reuse_replica(const MaterializationRequest& request) const;
  [[nodiscard]] absl::StatusOr<ReplicaHandle> copy_from_local_cpu(const MaterializationRequest& request) const;
  [[nodiscard]] absl::StatusOr<ReplicaHandle> copy_from_peer(const MaterializationRequest& request) const;
  [[nodiscard]] absl::StatusOr<ReplicaHandle> load_from_disk(const MaterializationRequest& request) const;
  [[nodiscard]] absl::StatusOr<ReplicaHandle> run_auto(const MaterializationRequest& request) const;
  [[nodiscard]] ReplicaHandle build_handle(
      const MaterializationRequest& request,
      const std::shared_ptr<replica::Replica>& replica,
      std::shared_ptr<common::ReadySignal<absl::Status>> ready_signal,
      loading::MaterializationSource source) const;
};

} // namespace tensorcast::store::runtime::ingestion
