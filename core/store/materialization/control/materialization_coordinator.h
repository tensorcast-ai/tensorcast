// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <memory>

#include "absl/status/statusor.h"
#include "core/store/components/runtime/component_catalog.h"
#include "core/store/components/runtime/global_store_publisher.h"
#include "core/store/components/runtime/replica_service.h"
#include "core/store/materialization/control/materialization_backend.h"
#include "core/store/materialization/control/materialization_service.h"
#include "core/store/materialization/runtime/pipeline/ingestion_pipeline.h"

namespace tensorcast::store::materialization::control {

class MaterializationCoordinator : public MaterializationBackend {
 public:
  struct Config {
    components::runtime::ReplicaService* replica_service;
    materialization::runtime::pipeline::IngestionPipeline* pipeline;
    components::runtime::ComponentCatalog* component_catalog;
    components::runtime::GlobalStorePublisher* global_store_publisher;
    size_t artifact_chunk_bytes;
    std::chrono::milliseconds pinned_memory_timeout;
    int num_threads;
  };

  explicit MaterializationCoordinator(Config config);

  absl::StatusOr<loading::ReplicaHandle> Materialize(
      const DeviceKey& target_device,
      loading::MaterializeMode mode,
      const loading::MaterializeHints& hints);

  absl::StatusOr<loading::ReplicaHandle> ingest_from_p2p(
      const std::string& artifact_identifier,
      const P2PSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints) override;

  absl::StatusOr<loading::ReplicaHandle> ingest_from_disk(
      const std::string& artifact_identifier,
      const loading::DiskSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints) override;

  absl::Status register_replica_with_global_store(const loading::ReplicaKey& key, std::string_view artifact_id_override)
      override;

 private:
  MaterializationDeps MakeDeps() const;

  Config config_;
  MaterializationService materialization_service_;
};

} // namespace tensorcast::store::materialization::control
