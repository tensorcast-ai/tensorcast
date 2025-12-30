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
#include "core/store/device_types.h"
#include "core/store/materialization/control/materialization_backend.h"
#include "core/store/materialization/dataplane/sources/segment_plan_source.h"
#include "core/store/materialization/runtime/pipeline/ingestion_pipeline.h"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/runtime/ingestion/ingestion_event_hub.h"
#include "core/store/runtime/ingestion/materialization_service.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/runtime/metadata/metadata_gateway.h"
#include "core/store/runtime/replica/replica_runtime.h"
#include "core/store/store_engine_options.h"
#include "gsl/pointers"

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

struct MaterializationHooks {
  using PipelineFactory = std::function<std::unique_ptr<materialization::runtime::pipeline::IngestionPipeline>(
      const materialization::runtime::pipeline::IngestionPipeline::Config&)>;
  using ServiceFactory =
      std::function<std::unique_ptr<ingestion::MaterializationService>(ingestion::MaterializationDeps)>;
  using RegisterReplicaOverride = std::function<absl::Status(
      const loading::ReplicaKey& key,
      std::string_view artifact_override,
      std::string_view publish_context_id)>;

  PipelineFactory pipeline_factory;
  ServiceFactory materialization_service_factory;
  RegisterReplicaOverride register_replica_override;
  std::function<void(const IngestionRequestMetadata&)> before_pipeline_start;
  std::function<void(IngestionResultEvent&)> mutate_completion_event;
  std::function<std::optional<absl::StatusOr<loading::ReplicaHandle>>()> override_result;
};

} // namespace tensorcast::store::runtime

namespace tensorcast::store::runtime::ingestion {

namespace metadata = tensorcast::store::runtime::metadata;

class MaterializationFacade : public materialization::control::MaterializationBackend {
 public:
  struct Config {
    gsl::not_null<RuntimeContext*> runtime_context;
    gsl::not_null<ReplicaRuntime*> replica_runtime;
    gsl::not_null<metadata::MetadataGateway*> metadata_gateway;
    std::filesystem::path storage_path;
    size_t artifact_chunk_bytes;
    std::chrono::milliseconds pinned_memory_timeout;
    int num_threads;
    const StoreEngineOptions* options;
    std::shared_ptr<const MaterializationHooks> hooks;
  };

  explicit MaterializationFacade(Config config);
  ~MaterializationFacade();

  MaterializationFacade(const MaterializationFacade&) = delete;
  MaterializationFacade& operator=(const MaterializationFacade&) = delete;

  absl::StatusOr<loading::ReplicaHandle> materialize_replica(
      const DeviceKey& target_device,
      loading::MaterializeMode mode,
      const loading::MaterializeHints& hints);

  absl::StatusOr<loading::MaterializeIntoTargetResult> materialize_into_target(
      const DeviceKey& target_device,
      gsl::not_null<void*> target_ptr,
      uint64_t total_size,
      std::string_view canonical_index_json,
      uint64_t generation,
      const loading::MaterializeHints& hints);

  absl::StatusOr<loading::ReplicaHandle> ingest_from_disk(
      const std::string& artifact_identifier,
      const loading::DiskSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints) override;

  absl::StatusOr<loading::ReplicaHandle> ingest_from_p2p(
      const std::string& artifact_identifier,
      const P2PSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints) override;

  absl::Status register_replica_with_global_store(
      const loading::ReplicaKey& key,
      std::string_view artifact_id_override,
      std::string_view publish_context_id = {}) override;

  absl::StatusOr<loading::ReplicaHandle> ingest_from_disk(
      const std::string& artifact_identifier,
      const loading::DiskSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints,
      bool publish_to_global_store);

  absl::StatusOr<loading::ReplicaHandle> ingest_from_p2p(
      const std::string& artifact_identifier,
      const P2PSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints,
      bool publish_to_global_store);

 private:
  template <typename SourceT, typename RunnerFn>
  absl::StatusOr<loading::ReplicaHandle> run_pipeline_ingestion(
      IngestionSource source_type,
      const std::string& artifact_identifier,
      const SourceT& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints,
      bool publish_to_global_store,
      RunnerFn&& runner);

  absl::StatusOr<loading::ReplicaHandle> run_disk_ingestion_internal(
      const std::string& artifact_identifier,
      const loading::DiskSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints,
      bool publish_to_global_store);

  absl::StatusOr<loading::ReplicaHandle> run_p2p_ingestion_internal(
      const std::string& artifact_identifier,
      const P2PSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints,
      bool publish_to_global_store);

  std::string make_request_id(std::string_view prefix);
  [[nodiscard]] IngestionStartedEvent make_started_event(
      const std::string& request_id,
      std::string_view artifact_identifier,
      IngestionSource source,
      const loading::ReplicaTarget& target,
      const std::string& publish_context_id,
      bool publish_to_global_store,
      loading::MaterializeMode mode,
      const loading::MaterializeHints& hints) const;
  [[nodiscard]] IngestionResultEvent make_ingestion_event_seed(
      const std::string& request_id,
      std::string_view artifact_identifier,
      IngestionSource source,
      const loading::ReplicaTarget& target,
      bool publish_to_global_store,
      const std::string& publish_context_id,
      loading::MaterializeMode mode,
      const loading::MaterializeHints& hints) const;
  void publish_started_event(const IngestionStartedEvent& event) const;
  void publish_completed_event(IngestionCompletedEvent event) const;
  void apply_event_defaults(IngestionResultEvent& event, const IngestionResultEvent& defaults) const;
  [[nodiscard]] std::optional<absl::StatusOr<loading::ReplicaHandle>> maybe_override_result() const;
  void maybe_invoke_before_pipeline_start(const IngestionRequestMetadata& metadata) const;
  void maybe_mutate_completion_event(IngestionResultEvent& event) const;
  void record_publish_context_for_replica(const loading::ReplicaKey& key, std::string_view publish_context_id);
  [[nodiscard]] std::optional<std::string> lookup_publish_context_for_replica(const loading::ReplicaKey& key) const;

  Config config_;
  std::shared_ptr<const MaterializationHooks> hooks_;
  std::unique_ptr<materialization::runtime::pipeline::IngestionPipeline> pipeline_;
  std::unique_ptr<MaterializationService> materialization_service_;
  ingestion::IngestionEventHub* ingestion_event_hub_;
  std::atomic<uint64_t> request_counter_{1};
  mutable absl::Mutex segment_plan_mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<std::vector<loader::SegmentPiece>>> segment_plan_cache_
      ABSL_GUARDED_BY(segment_plan_mu_);
  mutable absl::Mutex publish_context_mu_;
  absl::flat_hash_map<loading::ReplicaKey, std::string, loading::ReplicaKeyHash> publish_context_by_replica_
      ABSL_GUARDED_BY(publish_context_mu_);
};

} // namespace tensorcast::store::runtime::ingestion
