// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/runtime/ingestion/materialization_facade.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/runtime/replica/replica_runtime.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::runtime {

struct IngestionRuntimeDependencies final {
  std::shared_ptr<MaterializationHooks> hooks;
};

class IngestionRuntime {
 public:
  struct Config {
    RuntimeContext* runtime_context;
    ReplicaRuntime* replica_runtime;
    metadata::MetadataGateway* metadata_gateway;
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
  Config config_;
  std::unique_ptr<ingestion::MaterializationFacade> materialization_facade_;
};

} // namespace tensorcast::store::runtime
