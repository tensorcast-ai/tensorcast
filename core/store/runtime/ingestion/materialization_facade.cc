// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/ingestion/materialization_facade.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <functional>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/worker_identity.h"
#include "core/store/materialization/control/materialize_orchestrator.h"
#include "core/store/materialization/dataplane/loaders/disk_loader.h"
#include "core/store/materialization/dataplane/loaders/p2p_loader.h"
#include "core/store/materialization/dataplane/runtime/pump.h"
#include "core/store/materialization/dataplane/runtime/streaming_buffer_adapter.h"
#include "core/store/materialization/dataplane/sinks/gpu_memory_sink.h"

namespace tensorcast::store::runtime::ingestion {

namespace pipeline = tensorcast::store::materialization::runtime::pipeline;
using materialization::control::MaterializeOrchestrator;

namespace {

bool is_local_identity(const components::WorkerIdentity& local) {
  return !local.node_id.empty() || !local.node_address.empty();
}

bool is_local_replica(const components::RemoteReplicaInfo& remote, const components::WorkerIdentity& local) {
  if (!is_local_identity(local)) {
    return false;
  }
  if (!local.node_id.empty() && !remote.node_id.empty()) {
    return local.node_id == remote.node_id;
  }
  if (!local.node_address.empty() && !remote.node_address.empty() && local.node_address == remote.node_address) {
    if (local.p2p_port == 0 || remote.node_port == 0) {
      return true;
    }
    return local.p2p_port == remote.node_port;
  }
  return false;
}

absl::Status stale_local_route_status(std::string_view artifact_id) {
  return absl::UnavailableError(
      absl::StrCat("Global Store route stale for artifact_id=", artifact_id, "; retry or provide disk_path"));
}

} // namespace

class PlanBackedSeekableSource final : public loader::SeekableSource {
 public:
  PlanBackedSeekableSource(
      std::unique_ptr<loader::SeekableSource> source,
      std::shared_ptr<const std::vector<loader::SegmentPiece>> plan,
      uint64_t total_size)
      : source_(std::move(source)), plan_(std::move(plan)), total_size_(total_size) {}

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto st = read_at(current_offset_, dst, max_bytes);
    if (!st.ok()) {
      return st;
    }
    current_offset_ += *st;
    return st;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= total_size_) {
      return static_cast<size_t>(0);
    }
    size_t remaining = static_cast<size_t>(std::min<uint64_t>(bytes, total_size_ - offset));
    auto* out = static_cast<uint8_t*>(dst);

    size_t idx = 0;
    for (; idx < plan_->size(); ++idx) {
      const auto& p = (*plan_)[idx];
      if (offset < p.dst_offset + p.length) {
        break;
      }
    }
    if (idx == plan_->size()) {
      return static_cast<size_t>(0);
    }

    while (remaining > 0 && idx < plan_->size()) {
      const auto& p = (*plan_)[idx];
      const uint64_t local = offset - p.dst_offset;
      const size_t avail = static_cast<size_t>(p.length - local);
      const size_t take = std::min(remaining, avail);
      if (p.kind == loader::SegmentPiece::PAD) {
        std::memset(out, 0, take);
      } else {
        auto read_or = source_->read_at(p.src_offset + local, out, take);
        if (!read_or.ok()) {
          return read_or.status();
        }
        if (*read_or != take) {
          return absl::DataLossError("Short read while materializing into target");
        }
      }
      out += take;
      offset += take;
      remaining -= take;
      if (take == avail) {
        ++idx;
      }
    }
    return static_cast<size_t>(out - static_cast<uint8_t*>(dst));
  }

 private:
  std::unique_ptr<loader::SeekableSource> source_;
  std::shared_ptr<const std::vector<loader::SegmentPiece>> plan_;
  uint64_t total_size_{0};
  uint64_t current_offset_{0};
};

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
  deps.async_runtime = config_.runtime_context->async_runtime();
  deps.artifact_chunk_bytes = config_.artifact_chunk_bytes;
  deps.pinned_memory_timeout = config_.pinned_memory_timeout;
  deps.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);
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
        gsl::not_null<components::IGlobalStoreClient*>{client.get()},
        config_.runtime_context->worker_identity());
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

absl::StatusOr<loading::MaterializeIntoTargetResult> MaterializationFacade::materialize_into_target(
    const DeviceKey& target_device,
    gsl::not_null<void*> target_ptr,
    uint64_t total_size,
    std::string_view canonical_index_json,
    uint64_t generation,
    const loading::MaterializeHints& hints) {
  if (target_device.type != DeviceType::GPU) {
    return absl::InvalidArgumentError("materialize_into_target requires GPU target device");
  }
  if (total_size == 0) {
    return absl::InvalidArgumentError("materialize_into_target requires total_size > 0");
  }
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("materialize_into_target requires canonical index bytes");
  }
  if (hints.artifact_id.empty()) {
    return absl::InvalidArgumentError("materialize_into_target requires hints.artifact_id");
  }

  auto plan_key = [&]() -> std::string {
    auto mh_or = common::compute_index_multihash(std::optional<std::string>(canonical_index_json), "");
    if (mh_or.ok()) {
      return absl::StrCat(generation, ":", *mh_or);
    }
    const size_t fallback_hash = std::hash<std::string_view>{}(canonical_index_json);
    return absl::StrCat(generation, ":raw:", fallback_hash);
  }();

  std::shared_ptr<std::vector<loader::SegmentPiece>> plan_ptr;
  {
    absl::MutexLock lock(&segment_plan_mu_);
    auto it = segment_plan_cache_.find(plan_key);
    if (it != segment_plan_cache_.end()) {
      plan_ptr = it->second;
    }
  }
  if (!plan_ptr) {
    auto plan_or = loader::build_segment_plan_from_canonical_index_json(canonical_index_json, total_size);
    if (!plan_or.ok()) {
      return plan_or.status();
    }
    plan_ptr = std::make_shared<std::vector<loader::SegmentPiece>>(std::move(*plan_or));
    absl::MutexLock lock(&segment_plan_mu_);
    segment_plan_cache_.emplace(plan_key, plan_ptr);
  }

  auto run_source =
      [&](std::unique_ptr<IArtifactLoader> loader,
          loading::MaterializationSource source_kind) -> absl::StatusOr<loading::MaterializeIntoTargetResult> {
    auto init_status = loader->initialize();
    if (!init_status.ok()) {
      return init_status;
    }
    auto source_or = loader->open_source();
    if (!source_or.ok()) {
      return source_or.status();
    }

    auto plan_source = std::make_unique<PlanBackedSeekableSource>(std::move(*source_or), plan_ptr, total_size);

    const size_t slice_bytes = config_.runtime_context->tx_slice_bytes();
    if (slice_bytes == 0 || config_.artifact_chunk_bytes == 0) {
      return absl::FailedPreconditionError("tx_slice_bytes or artifact_chunk_bytes is zero");
    }
    const size_t num_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);
    auto session_spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
        /*num_chunks=*/num_chunks, slice_bytes, config_.runtime_context->pinned_buffer_pool());
    const std::chrono::milliseconds timeout =
        hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : config_.pinned_memory_timeout;
    auto init_spb_status = session_spb->initialize(timeout);
    if (!init_spb_status.ok()) {
      return init_spb_status;
    }
    loader::StreamingBufferAdapter adapter(session_spb);

    loader::GpuMemorySink::Options sink_opts{
        .gpu_base_ptr = target_ptr,
        .total_size = total_size,
        .chunk_size = config_.artifact_chunk_bytes,
        .device_id = target_device.ordinal,
    };
    loader::GpuMemorySink sink(std::move(sink_opts));

    const int concurrency = hints.pipeline_concurrency > 0 ? static_cast<int>(hints.pipeline_concurrency)
                                                           : std::max(1, config_.num_threads);
    std::array<loader::Range, 1> ranges{loader::Range{0, static_cast<size_t>(total_size)}};
    auto pump_status = loader::pump_ranges(
        *plan_source,
        sink,
        adapter,
        absl::MakeSpan(ranges),
        concurrency,
        config_.runtime_context->async_runtime()->blocking_executor());
    if (!pump_status.ok()) {
      return absl::DataLossError(absl::StrCat("materialize_into_target pump failed: ", pump_status.message()));
    }
    auto close_status = sink.close();
    if (!close_status.ok()) {
      return absl::DataLossError(absl::StrCat("materialize_into_target sink close failed: ", close_status.message()));
    }
    return loading::MaterializeIntoTargetResult{.source = source_kind};
  };

  auto gs_client = config_.runtime_context->global_store_client();
  const bool gs_connected = gs_client && gs_client->is_connected();
  const bool prefer_disk = hints.source_preference == loading::SourcePreference::kPreferDisk;
  const bool prefer_p2p = hints.source_preference == loading::SourcePreference::kPreferP2P;
  const bool allow_p2p = hints.allow_p2p;
  const bool allow_disk = hints.allow_disk;
  const bool has_disk_path = !hints.disk_path.empty();
  const auto& local_identity = config_.runtime_context->worker_identity();

  if (prefer_disk && !allow_disk) {
    return absl::InvalidArgumentError("source_policy disallows disk but preference=PREFER_DISK was requested");
  }
  if (prefer_p2p && !allow_p2p) {
    return absl::InvalidArgumentError("source_policy disallows P2P but preference=PREFER_P2P was requested");
  }

  if (prefer_disk && has_disk_path && allow_disk) {
    loading::DiskSource disk_src;
    disk_src.path = std::filesystem::path(hints.disk_path);
    disk_src.require_descriptor = tensorcast::common::is_mi2_artifact_id(hints.artifact_id);
    auto disk_or = run_source(std::make_unique<DiskLoader>(disk_src), loading::MaterializationSource::kDisk);
    if (disk_or.ok()) {
      return disk_or;
    }
    if (!gs_connected || !allow_p2p) {
      return disk_or.status();
    }
  }

  if (!gs_connected && (!has_disk_path || !allow_disk)) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  if (allow_p2p && gs_connected && !hints.artifact_id.empty()) {
    auto transport_or = gs_client->request_replica_transport(
        hints.artifact_id,
        local_identity.node_id,
        local_identity.node_address,
        local_identity.p2p_port,
        target_device,
        /*wait_timeout_ms=*/30000);
    if (transport_or.ok()) {
      const auto& session = *transport_or;
      const auto& remote = session.remote_replica;
      if (is_local_replica(remote, local_identity)) {
        LOG(WARNING) << "Global Store returned local replica for artifact_id=" << hints.artifact_id
                     << "; treating route as stale";
        auto complete_status = gs_client->complete_replica_transport(session.transport_id);
        if (!complete_status.ok()) {
          LOG(WARNING) << "complete_replica_transport after stale-local route returned error: " << complete_status;
        }
        if (!has_disk_path) {
          return stale_local_route_status(hints.artifact_id);
        }
      } else {
        P2PSource p2p_src;
        p2p_src.size_bytes = remote.memory_size;
        p2p_src.ip = remote.node_address;
        p2p_src.port = static_cast<uint16_t>(remote.node_port);
        p2p_src.memory_keys = remote.remote_memory_keys;
        p2p_src.buf_sizes = remote.buffer_sizes;
        p2p_src.verification_json = remote.verification_json;
        p2p_src.enable_checksum = false;
        p2p_src.location.type = remote.memory_type;
        p2p_src.location.device_id = remote.device_id;
        auto p2p_or = run_source(std::make_unique<P2PLoader>(p2p_src), loading::MaterializationSource::kP2P);
        auto complete_status = gs_client->complete_replica_transport(session.transport_id);
        if (!complete_status.ok()) {
          LOG(WARNING) << "complete_replica_transport returned error: " << complete_status;
        }
        if (p2p_or.ok()) {
          return p2p_or;
        }
        if (!allow_disk || !has_disk_path || prefer_p2p) {
          return p2p_or.status();
        }
      }
    } else if (!allow_disk || !has_disk_path) {
      return transport_or.status();
    }
  }

  if (allow_disk && has_disk_path) {
    loading::DiskSource disk_src;
    disk_src.path = std::filesystem::path(hints.disk_path);
    disk_src.require_descriptor = tensorcast::common::is_mi2_artifact_id(hints.artifact_id);
    return run_source(std::make_unique<DiskLoader>(disk_src), loading::MaterializationSource::kDisk);
  }

  if (!allow_p2p && !allow_disk) {
    return absl::FailedPreconditionError("source_policy disallows P2P and disk for materialize_into_target");
  }

  return absl::FailedPreconditionError("materialize_into_target requires disk_path or Global Store connectivity");
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
