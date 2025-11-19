// Copyright (c) 2025, TensorCast Team.

#include "core/store/runtime/ingestion/ingestion_runtime.h"

#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/replica/replica.h"
#include "core/store/runtime/ingestion/materialization_coordinator.h"
#include "gsl/pointers"

namespace tensorcast::store::runtime {

RuntimeContextEventSink::RuntimeContextEventSink(RuntimeContextEvents::Publisher publisher)
    : publisher_(std::move(publisher)) {}

void RuntimeContextEventSink::publish(RuntimeEventType type, IngestionResultEvent event) {
  if (!publisher_) {
    return;
  }
  RuntimeEvent runtime_event;
  runtime_event.type = type;
  runtime_event.payload = std::move(event);
  publisher_.publish(std::move(runtime_event));
}

IngestionRuntime::IngestionRuntime(Config config)
    : config_(std::move(config)), event_publisher_(config_.event_publisher) {
  ABSL_CHECK(config_.runtime_context != nullptr) << "RuntimeContext is required";
  ABSL_CHECK(config_.replica_runtime != nullptr) << "ReplicaRuntime is required";
  ABSL_CHECK(config_.metadata_gateway != nullptr) << "MetadataGateway is required";
  ABSL_CHECK(config_.options != nullptr) << "StoreEngineOptions must not be null";

  auto& context = *config_.runtime_context;
  if (config_.dependencies && config_.dependencies->event_sink_override) {
    ingestion_event_sink_ = config_.dependencies->event_sink_override;
  }
  if (!ingestion_event_sink_) {
    ingestion_event_sink_ = std::make_shared<RuntimeContextEventSink>(event_publisher_);
  }
  test_hooks_ = config_.dependencies ? config_.dependencies->test_hooks : nullptr;
  ABSL_CHECK(ingestion_event_sink_ != nullptr) << "Ingestion event sink must not be null";

  materialization::runtime::pipeline::IngestionPipeline::Config pipeline_config{
      .storage_path = config_.storage_path,
      .num_threads = config_.num_threads,
      .artifact_chunk_bytes = config_.artifact_chunk_bytes,
      .pinned_memory_timeout = config_.pinned_memory_timeout,
      .engine_options = config_.options,
      .replica_runtime = config_.replica_runtime,
      .runtime_context = &context,
      .metadata_gateway = config_.metadata_gateway,
      .event_publisher = event_publisher_,
  };
  if (config_.dependencies && config_.dependencies->pipeline_factory) {
    pipeline_ = config_.dependencies->pipeline_factory(pipeline_config);
  } else {
    pipeline_ = std::make_unique<materialization::runtime::pipeline::IngestionPipeline>(std::move(pipeline_config));
  }
  ABSL_CHECK(pipeline_ != nullptr) << "Ingestion pipeline factory returned null";

  ingestion::MaterializationCoordinator::Config coordinator_config{
      .replica_runtime = config_.replica_runtime,
      .pipeline = pipeline_.get(),
      .runtime_context = &context,
      .metadata_gateway = config_.metadata_gateway,
      .artifact_chunk_bytes = config_.artifact_chunk_bytes,
      .pinned_memory_timeout = config_.pinned_memory_timeout,
      .num_threads = config_.num_threads,
  };
  if (config_.dependencies && config_.dependencies->coordinator_factory) {
    coordinator_ = config_.dependencies->coordinator_factory(coordinator_config);
  } else {
    coordinator_ = std::make_unique<ingestion::MaterializationCoordinator>(std::move(coordinator_config));
  }
  ABSL_CHECK(coordinator_ != nullptr) << "Materialization coordinator factory returned null";
}

absl::StatusOr<loading::ReplicaHandle> IngestionRuntime::materialize_replica(
    const DeviceKey& target_device,
    loading::MaterializeMode mode,
    const loading::MaterializeHints& hints) {
  return coordinator_->materialize(target_device, mode, hints);
}

absl::StatusOr<loading::ReplicaHandle> IngestionRuntime::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  const std::string request_id = make_request_id("disk");
  const std::string publish_context_id = config_.runtime_context->mint_publish_context_id();
  const IngestionResultEvent defaults = make_ingestion_event_seed(
      request_id,
      artifact_identifier,
      IngestionSource::kDisk,
      target,
      /*publish_to_global_store=*/true,
      publish_context_id,
      loading::MaterializeMode::LOAD_ONLY,
      hints);
  publish_ingestion_event(RuntimeEventType::kIngestionStarted, defaults);

  IngestionRequestMetadata metadata{
      .request_id = request_id,
      .artifact_identifier = artifact_identifier,
      .source = IngestionSource::kDisk,
      .target = target,
      .publish_context_id = publish_context_id,
      .publish_to_global_store = true,
      .materialize_mode = loading::MaterializeMode::LOAD_ONLY,
      .hints = hints,
  };
  maybe_invoke_before_pipeline_start(metadata);

  if (auto override_result = maybe_override_result(); override_result.has_value()) {
    IngestionResultEvent stub_event = defaults;
    if (!override_result->ok()) {
      stub_event.status = override_result->status();
      maybe_mutate_completion_event(stub_event);
      publish_ingestion_event(RuntimeEventType::kIngestionFailed, std::move(stub_event));
      return override_result->status();
    }
    auto handle = std::move(override_result->value());
    maybe_mutate_completion_event(stub_event);
    record_publish_context_for_replica(handle.key(), stub_event.publish_context_id);
    publish_ingestion_event(RuntimeEventType::kIngestionCompleted, stub_event);
    return handle;
  }

  IngestionResultEvent pipeline_event;
  auto handle_or = pipeline_->ingest_from_disk(
      artifact_identifier,
      source,
      target,
      hints,
      /*publish_to_global_store=*/true,
      &pipeline_event,
      request_id,
      publish_context_id);
  if (!handle_or.ok()) {
    IngestionResultEvent failure_event = pipeline_event.request_id.empty() ? defaults : pipeline_event;
    apply_event_defaults(failure_event, defaults);
    failure_event.status = handle_or.status();
    maybe_mutate_completion_event(failure_event);
    publish_ingestion_event(RuntimeEventType::kIngestionFailed, std::move(failure_event));
    return handle_or.status();
  }

  auto handle = std::move(handle_or.value());
  apply_event_defaults(pipeline_event, defaults);
  maybe_mutate_completion_event(pipeline_event);
  record_publish_context_for_replica(handle.key(), pipeline_event.publish_context_id);
  publish_ingestion_event(RuntimeEventType::kIngestionCompleted, std::move(pipeline_event));
  return handle;
}

absl::StatusOr<loading::ReplicaHandle> IngestionRuntime::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  const std::string request_id = make_request_id("p2p");
  const std::string publish_context_id = config_.runtime_context->mint_publish_context_id();
  const IngestionResultEvent defaults = make_ingestion_event_seed(
      request_id,
      artifact_identifier,
      IngestionSource::kP2P,
      target,
      /*publish_to_global_store=*/true,
      publish_context_id,
      loading::MaterializeMode::COPY_ONLY,
      hints);
  publish_ingestion_event(RuntimeEventType::kIngestionStarted, defaults);

  IngestionRequestMetadata metadata{
      .request_id = request_id,
      .artifact_identifier = artifact_identifier,
      .source = IngestionSource::kP2P,
      .target = target,
      .publish_context_id = publish_context_id,
      .publish_to_global_store = true,
      .materialize_mode = loading::MaterializeMode::COPY_ONLY,
      .hints = hints,
  };
  maybe_invoke_before_pipeline_start(metadata);

  if (auto override_result = maybe_override_result(); override_result.has_value()) {
    IngestionResultEvent stub_event = defaults;
    if (!override_result->ok()) {
      stub_event.status = override_result->status();
      maybe_mutate_completion_event(stub_event);
      publish_ingestion_event(RuntimeEventType::kIngestionFailed, std::move(stub_event));
      return override_result->status();
    }
    auto handle = std::move(override_result->value());
    maybe_mutate_completion_event(stub_event);
    record_publish_context_for_replica(handle.key(), stub_event.publish_context_id);
    publish_ingestion_event(RuntimeEventType::kIngestionCompleted, stub_event);
    return handle;
  }

  IngestionResultEvent pipeline_event;
  auto handle_or = pipeline_->ingest_from_p2p(
      artifact_identifier,
      source,
      target,
      hints,
      /*publish_to_global_store=*/true,
      &pipeline_event,
      request_id,
      publish_context_id);
  if (!handle_or.ok()) {
    IngestionResultEvent failure_event = pipeline_event.request_id.empty() ? defaults : pipeline_event;
    apply_event_defaults(failure_event, defaults);
    failure_event.status = handle_or.status();
    maybe_mutate_completion_event(failure_event);
    publish_ingestion_event(RuntimeEventType::kIngestionFailed, std::move(failure_event));
    return handle_or.status();
  }

  auto handle = std::move(handle_or.value());
  apply_event_defaults(pipeline_event, defaults);
  maybe_mutate_completion_event(pipeline_event);
  record_publish_context_for_replica(handle.key(), pipeline_event.publish_context_id);
  publish_ingestion_event(RuntimeEventType::kIngestionCompleted, std::move(pipeline_event));
  return handle;
}

absl::Status IngestionRuntime::register_replica_with_global_store(
    const loading::ReplicaKey& key,
    std::string_view artifact_id_override) {
  const auto publish_context_id = lookup_publish_context_for_replica(key);
  if (publish_context_id.has_value()) {
    return coordinator_->register_replica_with_global_store(key, artifact_id_override, *publish_context_id);
  }
  return coordinator_->register_replica_with_global_store(key, artifact_id_override, {});
}

std::string IngestionRuntime::make_request_id(std::string_view prefix) {
  const uint64_t sequence = request_counter_.fetch_add(1, std::memory_order_relaxed);
  const int64_t timestamp = absl::ToUnixNanos(absl::Now());
  return absl::StrCat(prefix, "_", timestamp, "_", sequence);
}

IngestionResultEvent IngestionRuntime::make_ingestion_event_seed(
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

void IngestionRuntime::apply_event_defaults(IngestionResultEvent& event, const IngestionResultEvent& defaults) const {
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

void IngestionRuntime::publish_ingestion_event(RuntimeEventType type, IngestionResultEvent event) const {
  if (!ingestion_event_sink_) {
    return;
  }
  ingestion_event_sink_->publish(type, std::move(event));
}

std::optional<absl::StatusOr<loading::ReplicaHandle>> IngestionRuntime::maybe_override_result() const {
  if (!test_hooks_ || !test_hooks_->override_result) {
    return std::nullopt;
  }
  return test_hooks_->override_result();
}

void IngestionRuntime::maybe_invoke_before_pipeline_start(const IngestionRequestMetadata& metadata) const {
  if (!test_hooks_ || !test_hooks_->before_pipeline_start) {
    return;
  }
  test_hooks_->before_pipeline_start(metadata);
}

void IngestionRuntime::maybe_mutate_completion_event(IngestionResultEvent& event) const {
  if (!test_hooks_ || !test_hooks_->mutate_completion_event) {
    return;
  }
  test_hooks_->mutate_completion_event(event);
}

void IngestionRuntime::record_publish_context_for_replica(
    const loading::ReplicaKey& key,
    std::string_view publish_context_id) {
  if (publish_context_id.empty()) {
    return;
  }
  absl::MutexLock lock(&publish_context_mu_);
  publish_context_by_replica_[key] = std::string(publish_context_id);
}

std::optional<std::string> IngestionRuntime::lookup_publish_context_for_replica(const loading::ReplicaKey& key) const {
  absl::MutexLock lock(&publish_context_mu_);
  auto it = publish_context_by_replica_.find(key);
  if (it == publish_context_by_replica_.end()) {
    return std::nullopt;
  }
  return it->second;
}

} // namespace tensorcast::store::runtime
