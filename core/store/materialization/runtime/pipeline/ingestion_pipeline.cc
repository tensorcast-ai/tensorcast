// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/runtime/pipeline/ingestion_pipeline.h"

#include <chrono>
#include <optional>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "core/common/trace/trace_macros.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"

namespace tensorcast::store::materialization::runtime::pipeline {

namespace store_runtime = tensorcast::store::runtime;

namespace otel = opentelemetry;

namespace {

double compute_duration_seconds(const IngestionContext& ctx) {
  if (ctx.start_time.time_since_epoch().count() == 0) {
    return 0.0;
  }
  const auto duration = std::chrono::steady_clock::now() - ctx.start_time;
  return std::chrono::duration<double>(duration).count();
}

uint64_t determine_logical_size(const IngestionContext& ctx) {
  if (ctx.replica) {
    auto size_or = ctx.replica->get_artifact_size();
    if (size_or.ok()) {
      return *size_or;
    }
  }
  if (ctx.verification.logical_total_size > 0) {
    return ctx.verification.logical_total_size;
  }
  if (ctx.source_type == SourceType::kP2P) {
    return ctx.p2p.source.size_bytes;
  }
  if (ctx.disk.source.expected_size.has_value()) {
    return *ctx.disk.source.expected_size;
  }
  return 0;
}

void emit_ingestion_event(
    const IngestionPipeline::Config& config,
    const IngestionContext& ctx,
    const absl::Status& status,
    const loading::ReplicaHandle* handle) {
  const bool has_event_hub = config.event_hub != nullptr;
  if (!has_event_hub && !config.replica_runtime && !config.metadata_gateway) {
    return;
  }
  store_runtime::IngestionResultEvent event;
  event.source = ctx.source_type == SourceType::kP2P ? store_runtime::IngestionSource::kP2P
                                                     : store_runtime::IngestionSource::kDisk;
  event.artifact_id = ctx.artifact_identifier;
  event.target_device = ctx.target_device;
  event.target_location = ctx.target_location;
  event.bytes_transferred = determine_logical_size(ctx);
  event.duration_seconds = compute_duration_seconds(ctx);
  event.status = status;
  if (handle != nullptr) {
    event.replica_key = handle->replica_key;
    if (handle->replica_key.view_id.has_value()) {
      event.view_id = handle->replica_key.view_id;
    }
  } else if (ctx.hints.variant && ctx.hints.variant->view_id) {
    event.view_id = ctx.hints.variant->view_id;
  }
  event.publish_to_global_store = ctx.publish_to_global_store;

  if (has_event_hub) {
    store_runtime::RuntimeEvent runtime_event;
    runtime_event.type = store_runtime::RuntimeEventType::kIngressCompleted;
    runtime_event.payload = event;
    config.event_hub->publish(runtime_event);
    return;
  }
  if (config.replica_runtime) {
    config.replica_runtime->record_ingestion_result(event);
  }
  if (status.ok() && handle != nullptr && ctx.publish_to_global_store && config.metadata_gateway) {
    config.metadata_gateway->handle_ingestion_result(event);
  }
}

absl::Status initialize_context(
    const std::string& artifact_identifier,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    SourceType source_type,
    const IngestionPipeline::Config& config,
    bool publish_to_global_store,
    IngestionContext& ctx) {
  ctx.source_type = source_type;
  ctx.artifact_identifier = artifact_identifier;
  ctx.target = target;
  ctx.hints = hints;
  ctx.storage_path = config.storage_path;
  ctx.artifact_chunk_bytes = config.artifact_chunk_bytes;
  ctx.num_threads = config.num_threads;
  ctx.pinned_memory_timeout = config.pinned_memory_timeout;
  ctx.options = config.engine_options;
  ctx.replica_runtime = config.replica_runtime;
  ctx.component_catalog = config.component_catalog;
  ctx.target_device = target.location.to_device_key();
  ctx.target_is_gpu = ctx.target_device.type == DeviceType::GPU;
  ctx.target_location = ctx.target_is_gpu ? common::memory::MemoryLocation::GPU : common::memory::MemoryLocation::CPU;
  ctx.target_device_id = ctx.target_device.ordinal;
  ctx.publish_to_global_store = publish_to_global_store;
  if (ctx.target_is_gpu && !ctx.target_device.uuid.empty()) {
    auto device_result = ctx.replica_runtime->device_manager().find_device_by_uuid(ctx.target_device.uuid);
    if (!device_result.ok()) {
      return device_result.status();
    }
    ctx.target_device_id = device_result.value();
  }
  if (ctx.target_is_gpu) {
    const int num_gpus = ctx.replica_runtime->device_manager().get_num_gpus();
    if (ctx.target_device_id < 0 || ctx.target_device_id >= num_gpus) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid GPU device ordinal: ", ctx.target_device_id, " (", num_gpus, " devices available)"));
    }
  }
  return absl::OkStatus();
}

} // namespace

IngestionPipeline::IngestionPipeline(Config config) : config_(std::move(config)) {
  ABSL_DCHECK(config_.engine_options != nullptr);
  ABSL_DCHECK(config_.replica_runtime != nullptr);
  ABSL_DCHECK(config_.component_catalog != nullptr);
}

absl::StatusOr<loading::ReplicaHandle> IngestionPipeline::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  const std::string request_id = absl::StrCat("disk_", absl::ToUnixNanos(absl::Now()));
  SC_TRACE_INIT_GUARD(request_id, artifact_identifier, "ingest_from_disk_internal");

  IngestionContext ctx;
  absl::Status init_status =
      initialize_context(artifact_identifier, target, hints, SourceType::kDisk, config_, publish_to_global_store, ctx);
  if (!init_status.ok()) {
    return init_status;
  }
  ctx.start_time = std::chrono::steady_clock::now();
  ctx.publish_to_global_store = publish_to_global_store;

  auto fail = [&](const absl::Status& status) -> absl::StatusOr<loading::ReplicaHandle> {
    emit_ingestion_event(config_, ctx, status, nullptr);
    return status;
  };

  auto status = DiskSourceAdapter::prepare(source, ctx);
  if (!status.ok()) {
    return fail(status);
  }

  status = MetadataStage::process(ctx);
  if (!status.ok()) {
    return fail(status);
  }

  status = AllocationStage::allocate(ctx);
  if (!status.ok()) {
    return fail(status);
  }

  status = VerificationStage::verify(ctx);
  if (!status.ok()) {
    return fail(status);
  }

  auto handle_or = HandleStage::build(ctx);
  if (!handle_or.ok()) {
    return fail(handle_or.status());
  }
  auto handle = std::move(handle_or.value());
  emit_ingestion_event(config_, ctx, absl::OkStatus(), &handle);
  return handle;
}

absl::StatusOr<loading::ReplicaHandle> IngestionPipeline::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  const std::string request_id = absl::StrCat("p2p_", absl::ToUnixNanos(absl::Now()));
  SC_TRACE_INIT_GUARD(request_id, artifact_identifier, "ingest_from_p2p_internal");

  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.store");
  otel::trace::StartSpanOptions span_opts;
  span_opts.kind = otel::trace::SpanKind::kInternal;
  auto p2p_span = tracer->StartSpan("StoreEngine/P2PIngest", span_opts);
  otel::trace::Scope span_scope(p2p_span);
  p2p_span->SetAttribute("component", "StoreEngine");

  IngestionContext ctx;
  absl::Status init_status =
      initialize_context(artifact_identifier, target, hints, SourceType::kP2P, config_, publish_to_global_store, ctx);
  if (!init_status.ok()) {
    p2p_span->SetAttribute("error", true);
    p2p_span->AddEvent("p2p_ingest_error", {{"message", std::string(init_status.message())}});
    p2p_span->End();
    return init_status;
  }
  ctx.start_time = std::chrono::steady_clock::now();

  auto fail_with_status = [&](const absl::Status& status) -> absl::Status {
    p2p_span->SetAttribute("error", true);
    p2p_span->AddEvent("p2p_ingest_error", {{"message", std::string(status.message())}});
    p2p_span->End();
    emit_ingestion_event(config_, ctx, status, nullptr);
    return status;
  };

  auto prepare_status = P2PSourceAdapter::prepare(source, ctx);
  if (!prepare_status.ok()) {
    return fail_with_status(prepare_status);
  }

  if (hints.variant && hints.variant->view_id.has_value()) {
    p2p_span->SetAttribute("tc.view.id", *hints.variant->view_id);
  }
  p2p_span->SetAttribute("tc.source.address", ctx.p2p.source.ip);
  p2p_span->SetAttribute("tc.p2p.port", static_cast<int64_t>(ctx.p2p.source.port));
  p2p_span->SetAttribute("tc.size.bytes", static_cast<int64_t>(ctx.p2p.source.size_bytes));
  p2p_span->SetAttribute("tc.location", target.location.type == common::memory::MemoryLocation::GPU ? "gpu" : "cpu");

  auto metadata_status = MetadataStage::process(ctx);
  if (!metadata_status.ok()) {
    return fail_with_status(metadata_status);
  }

  auto alloc_status = AllocationStage::allocate(ctx);
  if (!alloc_status.ok()) {
    return fail_with_status(alloc_status);
  }

  auto verify_status = VerificationStage::verify(ctx);
  if (!verify_status.ok()) {
    return fail_with_status(verify_status);
  }

  auto handle_or = HandleStage::build(ctx);
  if (!handle_or.ok()) {
    return fail_with_status(handle_or.status());
  }

  auto handle = std::move(handle_or.value());
  emit_ingestion_event(config_, ctx, absl::OkStatus(), &handle);
  p2p_span->AddEvent("p2p_ingest_complete", {{"bytes", static_cast<int64_t>(ctx.p2p.source.size_bytes)}});
  p2p_span->End();
  return handle;
}

} // namespace tensorcast::store::materialization::runtime::pipeline
