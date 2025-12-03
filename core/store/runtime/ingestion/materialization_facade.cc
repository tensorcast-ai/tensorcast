// Copyright (c) 2025, TensorCast Team.

#include "core/store/runtime/ingestion/materialization_facade.h"

#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/store/materialization/control/materialize_orchestrator.h"

namespace tensorcast::store::runtime::ingestion {

namespace pipeline = tensorcast::store::materialization::runtime::pipeline;
using materialization::control::MaterializeOrchestrator;

MaterializationFacade::MaterializationFacade(Config config)
    : config_(std::move(config)),
      hooks_(config_.hooks),
      ingestion_event_hub_(config_.runtime_context->ingestion_event_hub()) {
  ABSL_CHECK(config_.runtime_context != nullptr) << "RuntimeContext is required";
  ABSL_CHECK(config_.replica_runtime != nullptr) << "ReplicaRuntime is required";
  ABSL_CHECK(config_.metadata_gateway != nullptr) << "MetadataGateway is required";
  ABSL_CHECK(config_.options != nullptr) << "StoreEngineOptions must not be null";
  ABSL_CHECK(ingestion_event_hub_ != nullptr) << "RuntimeContext missing ingestion event hub";

  pipeline::IngestionPipeline::Config pipeline_config{
      .storage_path = config_.storage_path,
      .num_threads = config_.num_threads,
      .artifact_chunk_bytes = config_.artifact_chunk_bytes,
      .pinned_memory_timeout = config_.pinned_memory_timeout,
      .engine_options = config_.options,
      .replica_runtime = config_.replica_runtime,
      .runtime_context = config_.runtime_context.get(),
  };
  if (hooks_ && hooks_->pipeline_factory) {
    pipeline_ = hooks_->pipeline_factory(pipeline_config);
  } else {
    pipeline_ = std::make_unique<pipeline::IngestionPipeline>(pipeline_config);
  }
  ABSL_CHECK(pipeline_ != nullptr) << "Ingestion pipeline factory returned null";

  auto& registry = config_.replica_runtime->registry();
  auto pinned_pool = config_.runtime_context->pinned_buffer_pool();
  MaterializationDeps deps(
      gsl::not_null<components::ReplicaRegistry*>{&registry},
      gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{pinned_pool});
  deps.artifact_chunk_bytes = config_.artifact_chunk_bytes;
  deps.pinned_memory_timeout = config_.pinned_memory_timeout;
  deps.num_threads = config_.num_threads;
  deps.view_hash_computer = config_.runtime_context->view_hash_computer();
  deps.ingest_from_disk = [this](
                              const std::string& artifact_identifier,
                              const loading::DiskSource& source,
                              const loading::ReplicaTarget& target,
                              const loading::MaterializeHints& hints) {
    return run_disk_ingestion_internal(artifact_identifier, source, target, hints, /*publish_to_global_store=*/false);
  };
  deps.run_auto = [this](const loading::MaterializationRequest& request) -> absl::StatusOr<loading::ReplicaHandle> {
    auto client = config_.runtime_context->global_store_client();
    if (!client || !client->is_connected()) {
      return absl::FailedPreconditionError("GlobalStoreClient not connected");
    }
    MaterializeOrchestrator orchestrator(
        gsl::not_null<materialization::control::MaterializationBackend*>{this},
        gsl::not_null<components::IGlobalStoreClient*>{client.get()});
    return orchestrator.run(request.canonical_artifact_id(), request.target_device(), request.hints());
  };

  if (hooks_ && hooks_->materialization_service_factory) {
    materialization_service_ = hooks_->materialization_service_factory(std::move(deps));
  } else {
    materialization_service_ = std::make_unique<MaterializationService>(std::move(deps));
  }
  ABSL_CHECK(materialization_service_ != nullptr) << "Materialization service factory returned null";
}

MaterializationFacade::~MaterializationFacade() = default;

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::materialize_replica(
    const DeviceKey& target_device,
    loading::MaterializeMode mode,
    const loading::MaterializeHints& hints) {
  auto request_or =
      loading::MaterializationRequest::Create(target_device, mode, hints, config_.replica_runtime->device_manager());
  if (!request_or.ok()) {
    return request_or.status();
  }
  return materialization_service_->execute(request_or.value());
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return run_disk_ingestion_internal(artifact_identifier, source, target, hints, /*publish_to_global_store=*/false);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  return run_disk_ingestion_internal(artifact_identifier, source, target, hints, publish_to_global_store);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return run_p2p_ingestion_internal(artifact_identifier, source, target, hints, /*publish_to_global_store=*/false);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  return run_p2p_ingestion_internal(artifact_identifier, source, target, hints, publish_to_global_store);
}

absl::Status MaterializationFacade::register_replica_with_global_store(
    const loading::ReplicaKey& key,
    std::string_view artifact_id_override,
    std::string_view publish_context_id) {
  std::string context = publish_context_id.empty() ? "" : std::string(publish_context_id);
  if (context.empty()) {
    auto stored_context = lookup_publish_context_for_replica(key);
    if (stored_context.has_value()) {
      context = *stored_context;
    }
  } else {
    record_publish_context_for_replica(key, context);
  }

  if (hooks_ && hooks_->register_replica_override) {
    return hooks_->register_replica_override(key, artifact_id_override, context);
  }
  return config_.metadata_gateway->register_replica(key, artifact_id_override, context);
}

template <typename SourceT, typename RunnerFn>
absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::run_pipeline_ingestion(
    IngestionSource source_type,
    const std::string& artifact_identifier,
    const SourceT& /*source*/,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store,
    RunnerFn&& runner) {
  const std::string request_id = make_request_id(source_type == IngestionSource::kDisk ? "disk" : "p2p");
  const std::string publish_context_id =
      publish_to_global_store ? config_.runtime_context->mint_publish_context_id() : std::string();
  const loading::MaterializeMode mode =
      source_type == IngestionSource::kP2P ? loading::MaterializeMode::COPY_ONLY : loading::MaterializeMode::LOAD_ONLY;
  IngestionRequestMetadata metadata{
      .request_id = request_id,
      .artifact_identifier = artifact_identifier,
      .source = source_type,
      .target = target,
      .publish_context_id = publish_context_id,
      .publish_to_global_store = publish_to_global_store,
      .materialize_mode = mode,
      .hints = hints,
  };
  maybe_invoke_before_pipeline_start(metadata);

  const IngestionStartedEvent started_event = make_started_event(
      request_id, artifact_identifier, source_type, target, publish_context_id, publish_to_global_store, mode, hints);
  publish_started_event(started_event);

  IngestionResultEvent defaults = make_ingestion_event_seed(
      request_id, artifact_identifier, source_type, target, publish_to_global_store, publish_context_id, mode, hints);

  if (auto override_result = maybe_override_result(); override_result.has_value()) {
    IngestionResultEvent event = defaults;
    if (!override_result->ok()) {
      event.status = override_result->status();
      maybe_mutate_completion_event(event);
      publish_completed_event(std::move(event));
      return override_result->status();
    }
    auto handle = std::move(override_result->value());
    event.replica_key = handle.key();
    maybe_mutate_completion_event(event);
    if (publish_to_global_store && !publish_context_id.empty()) {
      record_publish_context_for_replica(handle.key(), publish_context_id);
    }
    publish_completed_event(event);
    return handle;
  }

  IngestionResultEvent pipeline_event;
  auto handle_or = runner(request_id, publish_context_id, &pipeline_event);
  if (!handle_or.ok()) {
    IngestionResultEvent failure_event = pipeline_event.request_id.empty() ? defaults : pipeline_event;
    apply_event_defaults(failure_event, defaults);
    failure_event.status = handle_or.status();
    maybe_mutate_completion_event(failure_event);
    publish_completed_event(std::move(failure_event));
    return handle_or.status();
  }

  auto handle = std::move(handle_or.value());
  apply_event_defaults(pipeline_event, defaults);
  if (!pipeline_event.replica_key.has_value()) {
    pipeline_event.replica_key = handle.key();
  }
  maybe_mutate_completion_event(pipeline_event);
  if (publish_to_global_store && !pipeline_event.publish_context_id.empty()) {
    record_publish_context_for_replica(handle.key(), pipeline_event.publish_context_id);
  }
  publish_completed_event(pipeline_event);
  return handle;
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::run_disk_ingestion_internal(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  auto runner = [&](const std::string& request_id,
                    const std::string& publish_context_id,
                    IngestionResultEvent* event_out) {
    return pipeline_->ingest_from_disk(
        artifact_identifier, source, target, hints, publish_to_global_store, event_out, request_id, publish_context_id);
  };

  return run_pipeline_ingestion(
      IngestionSource::kDisk, artifact_identifier, source, target, hints, publish_to_global_store, runner);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::run_p2p_ingestion_internal(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  auto runner = [&](const std::string& request_id,
                    const std::string& publish_context_id,
                    IngestionResultEvent* event_out) {
    return pipeline_->ingest_from_p2p(
        artifact_identifier, source, target, hints, publish_to_global_store, event_out, request_id, publish_context_id);
  };

  return run_pipeline_ingestion(
      IngestionSource::kP2P, artifact_identifier, source, target, hints, publish_to_global_store, runner);
}

std::string MaterializationFacade::make_request_id(std::string_view prefix) {
  const uint64_t sequence = request_counter_.fetch_add(1, std::memory_order_relaxed);
  const int64_t timestamp = absl::ToUnixNanos(absl::Now());
  return absl::StrCat(prefix, "_", timestamp, "_", sequence);
}

IngestionResultEvent MaterializationFacade::make_ingestion_event_seed(
    const std::string& request_id,
    std::string_view artifact_identifier,
    IngestionSource source,
    const loading::ReplicaTarget& target,
    bool publish_to_global_store,
    const std::string& publish_context_id,
    loading::MaterializeMode mode,
    const loading::MaterializeHints& hints) const {
  IngestionResultEvent event;
  event.request_id = request_id;
  event.source = source;
  event.materialize_mode = mode;
  event.artifact_id = std::string(artifact_identifier);
  event.target_device = target.location.to_device_key();
  event.target_location = target.location.type;
  event.publish_to_global_store = publish_to_global_store;
  event.publish_context_id = publish_context_id;
  event.status = absl::OkStatus();
  if (hints.variant && hints.variant->view_id.has_value()) {
    event.view_id = hints.variant->view_id;
  }
  return event;
}

IngestionStartedEvent MaterializationFacade::make_started_event(
    const std::string& request_id,
    std::string_view artifact_identifier,
    IngestionSource source,
    const loading::ReplicaTarget& target,
    const std::string& publish_context_id,
    bool publish_to_global_store,
    loading::MaterializeMode mode,
    const loading::MaterializeHints& hints) const {
  IngestionStartedEvent started;
  started.request_id = request_id;
  started.artifact_id = std::string(artifact_identifier);
  started.source = source;
  started.target = target;
  started.publish_context_id = publish_context_id;
  started.publish_to_global_store = publish_to_global_store;
  started.materialize_mode = mode;
  if (hints.variant && hints.variant->view_id.has_value()) {
    started.view_id = hints.variant->view_id;
  }
  return started;
}

void MaterializationFacade::publish_started_event(const IngestionStartedEvent& event) const {
  if (ingestion_event_hub_ != nullptr) {
    ingestion_event_hub_->publish_started(event);
  }
}

void MaterializationFacade::publish_completed_event(IngestionCompletedEvent event) const {
  if (ingestion_event_hub_ != nullptr) {
    ingestion_event_hub_->publish_completed(event);
  }
}

void MaterializationFacade::apply_event_defaults(IngestionResultEvent& event, const IngestionResultEvent& defaults)
    const {
  if (event.request_id.empty()) {
    event.request_id = defaults.request_id;
  }
  if (event.artifact_id.empty()) {
    event.artifact_id = defaults.artifact_id;
  }
  event.source = defaults.source;
  event.materialize_mode = defaults.materialize_mode;
  event.target_device = defaults.target_device;
  event.target_location = defaults.target_location;
  if (!event.view_id.has_value() && defaults.view_id.has_value()) {
    event.view_id = defaults.view_id;
  }
  event.publish_to_global_store = defaults.publish_to_global_store;
  if (event.publish_context_id.empty()) {
    event.publish_context_id = defaults.publish_context_id;
  }
}

std::optional<absl::StatusOr<loading::ReplicaHandle>> MaterializationFacade::maybe_override_result() const {
  if (!hooks_ || !hooks_->override_result) {
    return std::nullopt;
  }
  return hooks_->override_result();
}

void MaterializationFacade::maybe_invoke_before_pipeline_start(const IngestionRequestMetadata& metadata) const {
  if (!hooks_ || !hooks_->before_pipeline_start) {
    return;
  }
  hooks_->before_pipeline_start(metadata);
}

void MaterializationFacade::maybe_mutate_completion_event(IngestionResultEvent& event) const {
  if (!hooks_ || !hooks_->mutate_completion_event) {
    return;
  }
  hooks_->mutate_completion_event(event);
}

void MaterializationFacade::record_publish_context_for_replica(
    const loading::ReplicaKey& key,
    std::string_view publish_context_id) {
  if (publish_context_id.empty()) {
    return;
  }
  absl::MutexLock lock(&publish_context_mu_);
  publish_context_by_replica_[key] = std::string(publish_context_id);
}

std::optional<std::string> MaterializationFacade::lookup_publish_context_for_replica(
    const loading::ReplicaKey& key) const {
  absl::MutexLock lock(&publish_context_mu_);
  auto it = publish_context_by_replica_.find(key);
  if (it == publish_context_by_replica_.end()) {
    return std::nullopt;
  }
  return it->second;
}

} // namespace tensorcast::store::runtime::ingestion
