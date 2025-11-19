// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <filesystem>
#include <string>

#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/runtime/pipeline/allocation_stage.h"
#include "core/store/materialization/runtime/pipeline/handle_stage.h"
#include "core/store/materialization/runtime/pipeline/ingestion_context.h"
#include "core/store/materialization/runtime/pipeline/metadata_stage.h"
#include "core/store/materialization/runtime/pipeline/source_adapter.h"
#include "core/store/materialization/runtime/pipeline/verification_stage.h"
#include "core/store/runtime/component_catalog.h"
#include "core/store/runtime/global_metadata_gateway.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/runtime/replica_runtime.h"
#include "core/store/runtime/runtime_event_hub.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::materialization::runtime::pipeline {

namespace store_runtime = tensorcast::store::runtime;

class IngestionPipeline {
 public:
  struct Config {
    std::filesystem::path storage_path;
    int num_threads;
    size_t artifact_chunk_bytes;
    std::chrono::milliseconds pinned_memory_timeout;
    const StoreEngineOptions* engine_options;
    store_runtime::ReplicaRuntime* replica_runtime;
    store_runtime::ComponentCatalog* component_catalog;
    store_runtime::GlobalMetadataGateway* metadata_gateway;
    store_runtime::RuntimeEventHub* event_hub;
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
