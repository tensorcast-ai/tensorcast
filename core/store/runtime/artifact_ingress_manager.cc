// Copyright (c) 2025, TensorCast Team.

#include "core/store/runtime/artifact_ingress_manager.h"

#include <utility>

#include "absl/log/check.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/materialization/control/materialization_coordinator.h"
#include "core/store/replica/replica.h"
#include "gsl/pointers"

namespace tensorcast::store::runtime {

ArtifactIngressManager::ArtifactIngressManager(Config config)
    : config_(std::move(config)), event_hub_(config_.env != nullptr ? &config_.env->event_hub() : nullptr) {
  ABSL_CHECK(config_.env != nullptr) << "RuntimeEnv is required";
  ABSL_CHECK(config_.replica_runtime != nullptr) << "ReplicaRuntime is required";
  ABSL_CHECK(config_.metadata_gateway != nullptr) << "GlobalMetadataGateway is required";
  ABSL_CHECK(config_.options != nullptr) << "StoreEngineOptions must not be null";

  auto& catalog = config_.env->component_catalog();
  materialization::runtime::pipeline::IngestionPipeline::Config pipeline_config{
      .storage_path = config_.storage_path,
      .num_threads = config_.num_threads,
      .artifact_chunk_bytes = config_.artifact_chunk_bytes,
      .pinned_memory_timeout = config_.pinned_memory_timeout,
      .engine_options = config_.options,
      .replica_runtime = config_.replica_runtime,
      .component_catalog = &catalog,
      .metadata_gateway = config_.metadata_gateway,
      .event_hub = event_hub_,
  };
  pipeline_ = std::make_unique<materialization::runtime::pipeline::IngestionPipeline>(std::move(pipeline_config));

  materialization::control::MaterializationCoordinator::Config coordinator_config{
      .replica_runtime = config_.replica_runtime,
      .pipeline = pipeline_.get(),
      .component_catalog = &catalog,
      .metadata_gateway = config_.metadata_gateway,
      .artifact_chunk_bytes = config_.artifact_chunk_bytes,
      .pinned_memory_timeout = config_.pinned_memory_timeout,
      .num_threads = config_.num_threads,
  };
  coordinator_ = std::make_unique<materialization::control::MaterializationCoordinator>(std::move(coordinator_config));

  auto* device_manager = &catalog.device_manager();
  auto* replica_registry = &config_.replica_runtime->registry();
  auto* metrics_collector = &catalog.metrics_collector();

  components::RegistrationResources registration_resources{
      .device_manager = gsl::not_null<components::DeviceManager*>{device_manager},
      .replica_registry = gsl::not_null<components::ReplicaRegistry*>{replica_registry},
      .metrics_collector = gsl::not_null<components::MetricsCollector*>{metrics_collector},
      .memory_pool = gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{catalog.pinned_buffer_pool()},
      .communication_manager = catalog.communication_manager(),
  };

  components::ReplicaFactory replica_factory =
      [](const replica::ReplicaConfig& config) -> absl::StatusOr<std::shared_ptr<replica::Replica>> {
    auto create_or = replica::Replica::create(config);
    if (!create_or.ok()) {
      return create_or.status();
    }
    return std::shared_ptr<replica::Replica>(std::move(create_or.value()));
  };

  registration_facade_ = std::make_unique<components::RegistrationFacade>(
      std::move(registration_resources),
      std::move(replica_factory),
      config_.artifact_chunk_bytes,
      config_.pinned_memory_timeout,
      config_.metadata_gateway);
}

absl::StatusOr<loading::ReplicaHandle> ArtifactIngressManager::materialize_replica(
    const DeviceKey& target_device,
    loading::MaterializeMode mode,
    const loading::MaterializeHints& hints) {
  return coordinator_->materialize(target_device, mode, hints);
}

absl::StatusOr<loading::ReplicaHandle> ArtifactIngressManager::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return pipeline_->ingest_from_disk(artifact_identifier, source, target, hints, /*publish_to_global_store=*/true);
}

absl::StatusOr<loading::ReplicaHandle> ArtifactIngressManager::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return pipeline_->ingest_from_p2p(artifact_identifier, source, target, hints, /*publish_to_global_store=*/true);
}

absl::Status ArtifactIngressManager::register_replica_with_global_store(
    const loading::ReplicaKey& key,
    std::string_view artifact_id_override) {
  return coordinator_->register_replica_with_global_store(key, artifact_id_override);
}

absl::StatusOr<components::RegistrationBeginResult> ArtifactIngressManager::begin_registration(
    const components::ArtifactRegistration& reg) {
  if (!registration_facade_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_facade_->begin(reg);
}

absl::StatusOr<components::RegistrationCommitResult> ArtifactIngressManager::commit_registration(
    std::string_view registration_id) {
  if (!registration_facade_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  auto result_or = registration_facade_->commit(registration_id);
  if (!result_or.ok()) {
    publish_registration_event(RuntimeEventType::kRegistrationAborted, registration_id, nullptr, result_or.status());
    return result_or.status();
  }
  publish_registration_event(
      RuntimeEventType::kRegistrationCommitted, result_or->registration_id, &result_or.value(), absl::OkStatus());
  return result_or;
}

absl::Status ArtifactIngressManager::abort_registration(std::string_view registration_id) {
  if (!registration_facade_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  auto status = registration_facade_->abort(registration_id);
  publish_registration_event(RuntimeEventType::kRegistrationAborted, registration_id, nullptr, status);
  return status;
}

absl::Status ArtifactIngressManager::keep_alive_registration(std::string_view registration_id, uint32_t ttl_ms) {
  if (!registration_facade_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_facade_->keep_alive(registration_id, ttl_ms);
}

absl::Status ArtifactIngressManager::ingest_view_chunk(
    std::string_view registration_id,
    uint64_t view_offset,
    absl::Span<const std::byte> data) {
  if (!registration_facade_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_facade_->ingest_view_chunk(registration_id, view_offset, data);
}

absl::StatusOr<uint64_t> ArtifactIngressManager::get_view_ingested_bytes(std::string_view registration_id) const {
  if (!registration_facade_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_facade_->get_view_ingested_bytes(registration_id);
}

void ArtifactIngressManager::publish_registration_event(
    RuntimeEventType type,
    std::string_view registration_id,
    const components::RegistrationCommitResult* result,
    const absl::Status& status) const {
  if (event_hub_ == nullptr) {
    return;
  }
  RuntimeEvent event;
  event.type = type;
  RegistrationEvent payload;
  payload.registration_id = std::string(registration_id);
  payload.status = status;
  payload.committed = (type == RuntimeEventType::kRegistrationCommitted) && status.ok();
  if (result != nullptr) {
    payload.artifact_id = result->artifact_id;
    payload.device = result->device;
    payload.size_bytes = result->size_bytes;
    payload.view_id = result->view_id;
    payload.existed = result->existed;
  }
  event.payload = std::move(payload);
  event_hub_->publish(event);
}

} // namespace tensorcast::store::runtime
