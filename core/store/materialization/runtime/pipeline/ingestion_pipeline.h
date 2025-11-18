// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <filesystem>
#include <string>

#include "absl/status/statusor.h"
#include "core/store/components/runtime/component_catalog.h"
#include "core/store/components/runtime/global_store_publisher.h"
#include "core/store/components/runtime/ingestion_events.h"
#include "core/store/components/runtime/replica_service.h"
#include "core/store/components/runtime/telemetry_service.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/runtime/pipeline/allocation_stage.h"
#include "core/store/materialization/runtime/pipeline/handle_stage.h"
#include "core/store/materialization/runtime/pipeline/ingestion_context.h"
#include "core/store/materialization/runtime/pipeline/metadata_stage.h"
#include "core/store/materialization/runtime/pipeline/source_adapter.h"
#include "core/store/materialization/runtime/pipeline/verification_stage.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::materialization::runtime::pipeline {

class IngestionPipeline {
 public:
  struct Config {
    std::filesystem::path storage_path;
    int num_threads;
    size_t artifact_chunk_bytes;
    std::chrono::milliseconds pinned_memory_timeout;
    const StoreEngineOptions* engine_options;
    components::runtime::ReplicaService* replica_service;
    components::runtime::ComponentCatalog* component_catalog;
    components::runtime::TelemetryService* telemetry_service;
    components::runtime::GlobalStorePublisher* global_store_publisher;
  };

  explicit IngestionPipeline(Config config);

  absl::StatusOr<loading::ReplicaHandle> ingest_from_disk(
      const std::string& artifact_identifier,
      const loading::DiskSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints,
      bool publish_to_global_store = true);

  absl::StatusOr<loading::ReplicaHandle> ingest_from_p2p(
      const std::string& artifact_identifier,
      const P2PSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints,
      bool publish_to_global_store = true);

 private:
  Config config_;
};

} // namespace tensorcast::store::materialization::runtime::pipeline
