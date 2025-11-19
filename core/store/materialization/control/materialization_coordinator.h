// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>

#include "absl/status/statusor.h"
#include "core/store/materialization/control/materialization_backend.h"
#include "core/store/materialization/control/materialization_service.h"
#include "core/store/materialization/runtime/pipeline/ingestion_pipeline.h"
#include "core/store/runtime/component_catalog.h"
#include "core/store/runtime/global_metadata_gateway.h"
#include "core/store/runtime/replica_runtime.h"

namespace tensorcast::store::materialization::control {

namespace store_runtime = tensorcast::store::runtime;

class MaterializationCoordinator : public MaterializationBackend {
 public:
  struct Config {
    store_runtime::ReplicaRuntime* replica_runtime;
    materialization::runtime::pipeline::IngestionPipeline* pipeline;
    store_runtime::ComponentCatalog* component_catalog;
    store_runtime::GlobalMetadataGateway* metadata_gateway;
    size_t artifact_chunk_bytes;
    std::chrono::milliseconds pinned_memory_timeout;
    int num_threads;
  };

  explicit MaterializationCoordinator(Config config);

  absl::StatusOr<loading::ReplicaHandle> materialize(
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
  [[nodiscard]] MaterializationDeps make_deps() const;

  Config config_;
  MaterializationService materialization_service_;
};

} // namespace tensorcast::store::materialization::control
