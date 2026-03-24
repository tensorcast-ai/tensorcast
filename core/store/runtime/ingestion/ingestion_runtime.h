// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "core/store/materialization/dataplane/contracts/loader.h"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/runtime/ingestion/artifact_lowering_plan.h"
#include "core/store/runtime/ingestion/materialization_facade.h"
#include "core/store/runtime/ingestion/materialization_strategy_types.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/runtime/replica/replica_runtime.h"
#include "core/store/seal_assembly_result.h"
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
      const loading::MaterializeHints& hints,
      std::optional<loading::DiskSource> disk_source = std::nullopt);

  absl::StatusOr<loading::MaterializeIntoTargetResult> materialize_into_target(
      const DeviceKey& target_device,
      const loading::IntoTargetLayout& target_layout,
      std::string_view canonical_index_json,
      uint64_t generation,
      const loading::MaterializeHints& hints,
      std::optional<loading::DiskSource> disk_source = std::nullopt);

  absl::StatusOr<loading::MaterializeIntoTargetResult> materialize_mapped_into_target(
      const DeviceKey& target_device,
      const ingestion::strategy::ResolvedMaterializationPlan& resolved_plan,
      const loading::MaterializeHints& hints,
      std::optional<loading::DiskSource> disk_source);

  absl::StatusOr<loading::MaterializeIntoTargetResult> materialize_mapped_into_target(
      const DeviceKey& target_device,
      const ingestion::strategy::ResolvedMaterializationPlan& resolved_plan,
      const loading::MaterializeHints& hints);

  absl::StatusOr<loading::MaterializeIntoTargetResult> materialize_mapped_loader_into_target(
      const DeviceKey& target_device,
      const loading::IntoTargetLayout& target_layout,
      std::unique_ptr<IArtifactLoader> loader,
      const loader::ByteRangeMap& mapping,
      const loading::MaterializeHints& hints,
      loading::MaterializationSource source_kind);

  absl::StatusOr<ingestion::ArtifactLoweringResult> execute_artifact_lowering_plan(
      ingestion::ArtifactLoweringPlan plan);

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

  absl::StatusOr<loading::ReplicaHandle> materialize_view_from_assembly(
      std::string_view assembly_id,
      std::string_view target_artifact_id,
      std::string_view view_id,
      std::string_view view_spec_json,
      const DeviceKey& target_device,
      loading::TransformPlacement placement,
      const std::vector<std::string>* allowed_view_ids = nullptr);

  absl::Status register_replica_with_global_store(
      const loading::ReplicaKey& key,
      std::string_view artifact_id_override);

  absl::StatusOr<SealAssemblyResult> seal_assembly(
      std::string_view assembly_id,
      bool publish_canonical,
      ingestion::MaterializationFacade::SealProgressCallback progress_cb = {},
      const std::vector<std::string>* allowed_view_ids = nullptr);

  absl::StatusOr<SealAssemblyResult> seal_assembly_from_cut(
      std::string_view assembly_id,
      const ingestion::MaterializationFacade::SealAssemblyCutInput& cut_input,
      bool publish_canonical,
      ingestion::MaterializationFacade::SealProgressCallback progress_cb = {});

 private:
  Config config_;
  std::unique_ptr<ingestion::MaterializationFacade> materialization_facade_;
};

} // namespace tensorcast::store::runtime
