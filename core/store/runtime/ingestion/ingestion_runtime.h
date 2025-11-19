// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "core/store/materialization/runtime/pipeline/ingestion_pipeline.h"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/runtime/ingestion/materialization_coordinator.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/runtime/metadata/metadata_gateway.h"
#include "core/store/runtime/replica/replica_runtime.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::runtime {

struct IngestionRequestMetadata {
  std::string request_id;
  std::string artifact_identifier;
  IngestionSource source;
  loading::ReplicaTarget target;
  std::string publish_context_id;
  bool publish_to_global_store;
  loading::MaterializeMode materialize_mode;
  loading::MaterializeHints hints;
};

class IngestionEventSink {
 public:
  virtual ~IngestionEventSink() = default;
  virtual void publish(RuntimeEventType type, IngestionResultEvent event) = 0;
};

class RuntimeContextEventSink final : public IngestionEventSink {
 public:
  explicit RuntimeContextEventSink(RuntimeContextEvents::Publisher publisher);
  void publish(RuntimeEventType type, IngestionResultEvent event) override;

 private:
  RuntimeContextEvents::Publisher publisher_;
};

struct IngestionTestHooks {
  std::function<void(const IngestionRequestMetadata&)> before_pipeline_start;
  std::function<void(IngestionResultEvent&)> mutate_completion_event;
  std::function<std::optional<absl::StatusOr<loading::ReplicaHandle>>()> override_result;
};

struct IngestionRuntimeDependencies final {
  using PipelineFactory = std::function<std::unique_ptr<materialization::runtime::pipeline::IngestionPipeline>(
      const materialization::runtime::pipeline::IngestionPipeline::Config&)>;
  using CoordinatorFactory = std::function<std::unique_ptr<ingestion::MaterializationCoordinator>(
      const ingestion::MaterializationCoordinator::Config&)>;

  PipelineFactory pipeline_factory;
  CoordinatorFactory coordinator_factory;
  std::shared_ptr<IngestionEventSink> event_sink_override;
  std::shared_ptr<IngestionTestHooks> test_hooks;
};

class IngestionRuntime {
 public:
  struct Config {
    RuntimeContext* runtime_context;
    ReplicaRuntime* replica_runtime;
    metadata::MetadataGateway* metadata_gateway;
    RuntimeContextEvents::Publisher event_publisher;
    std::filesystem::path storage_path;
    size_t artifact_chunk_bytes;
    std::chrono::milliseconds pinned_memory_timeout;
    int num_threads;
    const StoreEngineOptions* options;
    std::shared_ptr<const IngestionRuntimeDependencies> dependencies;
  };

  explicit IngestionRuntime(Config config);
  ~IngestionRuntime() = default;

  IngestionRuntime(const IngestionRuntime&) = delete;
  IngestionRuntime& operator=(const IngestionRuntime&) = delete;

  absl::StatusOr<loading::ReplicaHandle> materialize_replica(
      const DeviceKey& target_device,
      loading::MaterializeMode mode,
      const loading::MaterializeHints& hints);

  absl::StatusOr<loading::ReplicaHandle> ingest_from_disk(
      const std::string& artifact_identifier,
      const loading::DiskSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints);

  absl::StatusOr<loading::ReplicaHandle> ingest_from_p2p(
      const std::string& artifact_identifier,
      const P2PSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints);

  absl::Status register_replica_with_global_store(
      const loading::ReplicaKey& key,
      std::string_view artifact_id_override);

 private:
  std::string make_request_id(std::string_view prefix);
  [[nodiscard]] IngestionResultEvent make_ingestion_event_seed(
      const std::string& request_id,
      std::string_view artifact_identifier,
      IngestionSource source,
      const loading::ReplicaTarget& target,
      bool publish_to_global_store,
      const std::string& publish_context_id,
      loading::MaterializeMode mode,
      const loading::MaterializeHints& hints) const;
  void apply_event_defaults(IngestionResultEvent& event, const IngestionResultEvent& defaults) const;
  void publish_ingestion_event(RuntimeEventType type, IngestionResultEvent event) const;
  [[nodiscard]] std::optional<absl::StatusOr<loading::ReplicaHandle>> maybe_override_result() const;
  void maybe_invoke_before_pipeline_start(const IngestionRequestMetadata& metadata) const;
  void maybe_mutate_completion_event(IngestionResultEvent& event) const;
  void record_publish_context_for_replica(const loading::ReplicaKey& key, std::string_view publish_context_id);
  [[nodiscard]] std::optional<std::string> lookup_publish_context_for_replica(const loading::ReplicaKey& key) const;

  Config config_;
  RuntimeContextEvents::Publisher event_publisher_;
  std::shared_ptr<IngestionEventSink> ingestion_event_sink_;
  std::shared_ptr<IngestionTestHooks> test_hooks_;
  std::unique_ptr<materialization::runtime::pipeline::IngestionPipeline> pipeline_;
  std::unique_ptr<ingestion::MaterializationCoordinator> coordinator_;
  std::atomic<uint64_t> request_counter_{1};
  mutable absl::Mutex publish_context_mu_;
  absl::flat_hash_map<loading::ReplicaKey, std::string, loading::ReplicaKeyHash> publish_context_by_replica_
      ABSL_GUARDED_BY(publish_context_mu_);
};

} // namespace tensorcast::store::runtime
