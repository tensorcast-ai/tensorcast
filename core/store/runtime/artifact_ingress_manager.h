// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/store/components/registration/registration_facade.h"
#include "core/store/materialization/control/materialization_coordinator.h"
#include "core/store/materialization/runtime/pipeline/ingestion_pipeline.h"
#include "core/store/runtime/runtime_env.h"

namespace tensorcast::store::runtime {

class ArtifactIngressManager {
 public:
  struct Config {
    RuntimeEnv* env;
    ReplicaRuntime* replica_runtime;
    GlobalMetadataGateway* metadata_gateway;
    std::filesystem::path storage_path;
    size_t artifact_chunk_bytes;
    std::chrono::milliseconds pinned_memory_timeout;
    int num_threads;
    const StoreEngineOptions* options;
  };

  explicit ArtifactIngressManager(Config config);
  ~ArtifactIngressManager() = default;

  ArtifactIngressManager(const ArtifactIngressManager&) = delete;
  ArtifactIngressManager& operator=(const ArtifactIngressManager&) = delete;

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

  absl::StatusOr<components::RegistrationBeginResult> begin_registration(const components::ArtifactRegistration& reg);
  absl::StatusOr<components::RegistrationCommitResult> commit_registration(std::string_view registration_id);
  absl::Status abort_registration(std::string_view registration_id);
  absl::Status keep_alive_registration(std::string_view registration_id, uint32_t ttl_ms);
  absl::Status ingest_view_chunk(
      std::string_view registration_id,
      uint64_t view_offset,
      absl::Span<const std::byte> data);
  [[nodiscard]] absl::StatusOr<uint64_t> get_view_ingested_bytes(std::string_view registration_id) const;

 private:
  void publish_registration_event(
      RuntimeEventType type,
      std::string_view registration_id,
      const components::RegistrationCommitResult* result,
      const absl::Status& status) const;

  Config config_;
  RuntimeEventHub* event_hub_;
  std::unique_ptr<materialization::runtime::pipeline::IngestionPipeline> pipeline_;
  std::unique_ptr<materialization::control::MaterializationCoordinator> coordinator_;
  std::unique_ptr<components::RegistrationFacade> registration_facade_;
};

} // namespace tensorcast::store::runtime
