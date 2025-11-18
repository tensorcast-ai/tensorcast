// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/components/worker_identity.h"
#include "core/store/materialization/common/view_hash_utils.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::components::runtime {

class ComponentCatalog {
 public:
  explicit ComponentCatalog(const StoreEngineOptions& options);
  ~ComponentCatalog();

  ComponentCatalog(const ComponentCatalog&) = delete;
  ComponentCatalog& operator=(const ComponentCatalog&) = delete;

  absl::Status start();
  void shutdown();

  DeviceManager& device_manager();
  ReplicaRegistry& replica_registry();
  MetricsCollector& metrics_collector();

  std::shared_ptr<CommunicationManager> communication_manager() const {
    return comm_manager_;
  }

  std::shared_ptr<common::memory::PinnedBufferPool> pinned_buffer_pool() const {
    return memory_pool_;
  }

  std::shared_ptr<IGlobalStoreClient> global_store_client() const {
    return global_store_client_;
  }

  std::shared_ptr<ViewHashComputer> view_hash_computer() const {
    return view_hash_computer_;
  }

  void set_global_store_client_for_testing(std::shared_ptr<IGlobalStoreClient> client);

  size_t artifact_chunk_bytes() const {
    return artifact_chunk_bytes_;
  }

  size_t memory_pool_size() const {
    return options_.memory_pool_size;
  }

  size_t tx_slice_bytes() const {
    return options_.tx_slice_bytes;
  }

  const StoreEngineOptions& options() const {
    return options_;
  }

  void set_worker_identity(WorkerIdentity identity);

  const WorkerIdentity& worker_identity() const {
    return worker_identity_;
  }

 private:
  void validate_options() const;
  absl::Status initialize_device_manager();
  void initialize_communication_manager();
  absl::Status initialize_global_store_client();

  StoreEngineOptions options_;
  size_t artifact_chunk_bytes_{0};
  std::shared_ptr<common::memory::PinnedBufferPool> memory_pool_;
  std::unique_ptr<DeviceManager> device_manager_;
  std::unique_ptr<ReplicaRegistry> replica_registry_;
  std::unique_ptr<MetricsCollector> metrics_collector_;
  std::shared_ptr<CommunicationManager> comm_manager_;
  std::shared_ptr<IGlobalStoreClient> global_store_client_;
  std::shared_ptr<ViewHashComputer> view_hash_computer_;
  WorkerIdentity worker_identity_;
  bool started_{false};
};

} // namespace tensorcast::store::components::runtime
