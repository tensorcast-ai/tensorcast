// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/control/materialization_coordinator.h"

#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "core/store/materialization/control/materialize_orchestrator.h"
#include "gsl/pointers"

namespace tensorcast::store::materialization::control {

MaterializationCoordinator::MaterializationCoordinator(Config config)
    : config_(std::move(config)), materialization_service_(make_deps()) {}

MaterializationDeps MaterializationCoordinator::make_deps() const {
  auto& registry = config_.replica_runtime->registry();
  auto pool = config_.component_catalog->pinned_buffer_pool();
  auto* backend = const_cast<MaterializationCoordinator*>(this);
  MaterializationDeps deps(
      gsl::not_null<components::ReplicaRegistry*>{&registry},
      gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{pool});
  deps.artifact_chunk_bytes = config_.artifact_chunk_bytes;
  deps.pinned_memory_timeout = config_.pinned_memory_timeout;
  deps.num_threads = config_.num_threads;
  deps.ingest_from_disk = [this](
                              const std::string& artifact_identifier,
                              const loading::DiskSource& source,
                              const loading::ReplicaTarget& target,
                              const loading::MaterializeHints& hints) {
    return config_.pipeline->ingest_from_disk(artifact_identifier, source, target, hints);
  };
  deps.view_hash_computer = config_.component_catalog->view_hash_computer();
  deps.run_auto = [this,
                   backend](const loading::MaterializationRequest& request) -> absl::StatusOr<loading::ReplicaHandle> {
    auto client = config_.component_catalog->global_store_client();
    if (!client || !client->is_connected()) {
      return absl::FailedPreconditionError("GlobalStoreClient not connected");
    }
    MaterializeOrchestrator orchestrator(
        gsl::not_null<MaterializationBackend*>{backend}, gsl::not_null<components::IGlobalStoreClient*>{client.get()});
    return orchestrator.run(request.canonical_artifact_id(), request.target_device(), request.hints());
  };
  return deps;
}

absl::StatusOr<loading::ReplicaHandle> MaterializationCoordinator::materialize(
    const DeviceKey& target_device,
    loading::MaterializeMode mode,
    const loading::MaterializeHints& hints) {
  auto request_or =
      loading::MaterializationRequest::Create(target_device, mode, hints, config_.replica_runtime->device_manager());
  if (!request_or.ok()) {
    return request_or.status();
  }
  return materialization_service_.Execute(request_or.value());
}

absl::StatusOr<loading::ReplicaHandle> MaterializationCoordinator::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return config_.pipeline->ingest_from_p2p(
      artifact_identifier, source, target, hints, /*publish_to_global_store=*/false);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationCoordinator::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return config_.pipeline->ingest_from_disk(
      artifact_identifier, source, target, hints, /*publish_to_global_store=*/false);
}

absl::Status MaterializationCoordinator::register_replica_with_global_store(
    const loading::ReplicaKey& key,
    std::string_view artifact_id_override) {
  if (config_.metadata_gateway == nullptr) {
    return absl::FailedPreconditionError("GlobalMetadataGateway not initialized");
  }
  return config_.metadata_gateway->register_replica(key, artifact_id_override);
}

} // namespace tensorcast::store::materialization::control
