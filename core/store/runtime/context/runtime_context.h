// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "core/common/async_runtime.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/components/worker_identity.h"
#include "core/store/materialization/common/view_hash_utils.h"
#include "core/store/memory_tier_budget.h"
#include "core/store/runtime/context/runtime_context_events.h"
#include "core/store/runtime/ingestion/ingestion_event_hub.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::runtime {

class RuntimeContext {
 public:
  explicit RuntimeContext(const StoreEngineOptions& options);
  ~RuntimeContext();

  RuntimeContext(const RuntimeContext&) = delete;
  RuntimeContext& operator=(const RuntimeContext&) = delete;

  absl::Status start();
  void shutdown();

  components::DeviceManager& device_manager();
  components::ReplicaRegistry& replica_registry();
  components::MetricsCollector& metrics_collector();

  [[nodiscard]] std::shared_ptr<components::CommunicationManager> communication_manager() const {
    return comm_manager_;
  }

  [[nodiscard]] std::shared_ptr<MemoryTierBudget> memory_tier_budget() const {
    return memory_tier_budget_;
  }

  [[nodiscard]] std::shared_ptr<common::memory::PinnedBufferPool> pinned_buffer_pool() const {
    return memory_pool_;
  }

  [[nodiscard]] std::shared_ptr<components::IGlobalStoreClient> global_store_client() const {
    return global_store_client_;
  }

  [[nodiscard]] std::shared_ptr<ViewHashComputer> view_hash_computer() const {
    return view_hash_computer_;
  }

  [[nodiscard]] std::shared_ptr<common::AsyncRuntime> async_runtime() const {
    return async_runtime_;
  }

  [[nodiscard]] bool owns_async_runtime() const {
    return owns_async_runtime_;
  }

  void set_global_store_client_for_testing(std::shared_ptr<components::IGlobalStoreClient> client);

  [[nodiscard]] size_t artifact_chunk_bytes() const {
    return artifact_chunk_bytes_;
  }

  [[nodiscard]] size_t memory_pool_size() const {
    return options_.memory_pool_size;
  }

  [[nodiscard]] size_t tx_slice_bytes() const {
    return options_.tx_slice_bytes;
  }

  [[nodiscard]] const StoreEngineOptions& options() const {
    return options_;
  }

  void set_worker_identity(components::WorkerIdentity identity);

  [[nodiscard]] const components::WorkerIdentity& worker_identity() const {
    return worker_identity_;
  }

  RuntimeContextEvents::Publisher event_publisher();
  std::unique_ptr<RuntimeContextEvents::Subscription> subscribe_to_events(RuntimeContextEvents::Callback callback);
  void drain_events();
  [[nodiscard]] std::string mint_publish_context_id();
  ingestion::IngestionEventHub* ingestion_event_hub();
  const ingestion::IngestionEventHub* ingestion_event_hub() const;

 private:
  absl::Status validate_options() const;
  absl::Status initialize_device_manager();
  absl::Status initialize_communication_manager();
  absl::Status initialize_global_store_client();

  StoreEngineOptions options_;
  size_t artifact_chunk_bytes_{0};
  std::shared_ptr<common::memory::PinnedBufferPool> memory_pool_;
  std::unique_ptr<components::DeviceManager> device_manager_;
  std::unique_ptr<components::ReplicaRegistry> replica_registry_;
  std::unique_ptr<components::MetricsCollector> metrics_collector_;
  std::shared_ptr<components::CommunicationManager> comm_manager_;
  std::shared_ptr<components::IGlobalStoreClient> global_store_client_;
  std::shared_ptr<ViewHashComputer> view_hash_computer_;
  std::shared_ptr<MemoryTierBudget> memory_tier_budget_;
  std::shared_ptr<common::AsyncRuntime> async_runtime_;
  bool owns_async_runtime_{false};
  std::unique_ptr<RuntimeContextEvents> events_;
  std::unique_ptr<ingestion::IngestionEventHub> ingestion_event_hub_;
  components::WorkerIdentity worker_identity_;
  std::atomic<uint64_t> publish_context_counter_{1};
  bool started_{false};
};

} // namespace tensorcast::store::runtime
