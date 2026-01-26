// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/ingestion/ingestion_runtime.h"

#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "gsl/pointers"

namespace tensorcast::store::runtime {

IngestionRuntime::IngestionRuntime(Config config) : config_(std::move(config)) {
  ABSL_CHECK(config_.runtime_context != nullptr) << "RuntimeContext is required";
  ABSL_CHECK(config_.replica_runtime != nullptr) << "ReplicaRuntime is required";
  ABSL_CHECK(config_.metadata_gateway != nullptr) << "MetadataGateway is required";
  ABSL_CHECK(config_.options != nullptr) << "StoreEngineOptions must not be null";

  ingestion::MaterializationFacade::Config facade_config{
      .runtime_context = gsl::not_null<RuntimeContext*>{config_.runtime_context},
      .replica_runtime = gsl::not_null<ReplicaRuntime*>{config_.replica_runtime},
      .metadata_gateway = gsl::not_null<metadata::MetadataGateway*>{config_.metadata_gateway},
      .storage_path = config_.storage_path,
      .artifact_chunk_bytes = config_.artifact_chunk_bytes,
      .pinned_memory_timeout = config_.pinned_memory_timeout,
      .num_threads = config_.num_threads,
      .options = config_.options,
      .hooks = config_.dependencies ? config_.dependencies->hooks : nullptr,
  };
  materialization_facade_ = std::make_unique<ingestion::MaterializationFacade>(std::move(facade_config));
}

absl::StatusOr<loading::ReplicaHandle> IngestionRuntime::materialize_replica(
    const DeviceKey& target_device,
    loading::MaterializeMode mode,
    const loading::MaterializeHints& hints) {
  return materialization_facade_->materialize_replica(target_device, mode, hints);
}

absl::StatusOr<loading::MaterializeIntoTargetResult> IngestionRuntime::materialize_into_target(
    const DeviceKey& target_device,
    const loading::IntoTargetLayout& target_layout,
    std::string_view canonical_index_json,
    uint64_t generation,
    const loading::MaterializeHints& hints) {
  return materialization_facade_->materialize_into_target(
      target_device, target_layout, canonical_index_json, generation, hints);
}

absl::StatusOr<loading::ReplicaHandle> IngestionRuntime::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return materialization_facade_->ingest_from_disk(
      artifact_identifier, source, target, hints, /*publish_to_global_store=*/true);
}

absl::StatusOr<loading::ReplicaHandle> IngestionRuntime::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return materialization_facade_->ingest_from_p2p(
      artifact_identifier, source, target, hints, /*publish_to_global_store=*/true);
}

absl::Status IngestionRuntime::register_replica_with_global_store(
    const loading::ReplicaKey& key,
    std::string_view artifact_id_override) {
  return materialization_facade_->register_replica_with_global_store(key, artifact_id_override);
}

absl::StatusOr<SealAssemblyResult> IngestionRuntime::seal_assembly(
    std::string_view assembly_id,
    bool publish_canonical) {
  return materialization_facade_->seal_assembly(assembly_id, publish_canonical);
}

} // namespace tensorcast::store::runtime
