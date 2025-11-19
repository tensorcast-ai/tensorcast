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
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/runtime/metadata/metadata_gateway.h"
#include "core/store/runtime/replica/replica_runtime.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::materialization::runtime::pipeline {

namespace store_runtime = tensorcast::store::runtime;
namespace metadata = tensorcast::store::runtime::metadata;

class IngestionPipeline {
 public:
  struct Config {
    std::filesystem::path storage_path;
    int num_threads;
    size_t artifact_chunk_bytes;
    std::chrono::milliseconds pinned_memory_timeout;
    const StoreEngineOptions* engine_options;
    store_runtime::ReplicaRuntime* replica_runtime;
    store_runtime::RuntimeContext* runtime_context;
    metadata::MetadataGateway* metadata_gateway;
    store_runtime::RuntimeContextEvents::Publisher event_publisher;
  };

  explicit IngestionPipeline(Config config);
  virtual ~IngestionPipeline() = default;

  virtual absl::StatusOr<loading::ReplicaHandle> ingest_from_disk(
      const std::string& artifact_identifier,
      const loading::DiskSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints,
      bool publish_to_global_store = true,
      store_runtime::IngestionResultEvent* event_out = nullptr,
      std::string request_id = {},
      std::string publish_context_id = {});

  virtual absl::StatusOr<loading::ReplicaHandle> ingest_from_p2p(
      const std::string& artifact_identifier,
      const P2PSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints,
      bool publish_to_global_store = true,
      store_runtime::IngestionResultEvent* event_out = nullptr,
      std::string request_id = {},
      std::string publish_context_id = {});

 private:
  Config config_;
};

} // namespace tensorcast::store::materialization::runtime::pipeline
