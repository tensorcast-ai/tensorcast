// Copyright (c) 2025, TensorCast Team.

#include "core/store/runtime/metadata/metadata_gateway.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "core/store/materialization/control/replica_registration_helper.h"

namespace tensorcast::store::runtime::metadata {

namespace {

constexpr int kMaxPublishContextRecords = 1024;
constexpr absl::Duration kPublishContextTtl = absl::Minutes(10);

absl::StatusOr<std::string> canonical_artifact_id(
    const loading::ReplicaKey& key,
    std::string_view artifact_id_override) {
  if (!artifact_id_override.empty()) {
    if (!absl::StartsWith(artifact_id_override, "mi2:")) {
      return absl::InvalidArgumentError("artifact_id_override must be a canonical mi2: identifier");
    }
    return std::string(artifact_id_override);
  }
  return key.artifact_id;
}

class GlobalStoreRegistrationPublisher final : public RegistrationPublisher {
 public:
  GlobalStoreRegistrationPublisher(
      gsl::not_null<RuntimeContext*> runtime_context,
      std::shared_ptr<components::IGlobalStoreClient>* override_slot)
      : runtime_context_(runtime_context), override_slot_(override_slot) {}

  absl::Status publish_registration(const RegistrationPublication& publication) override {
    auto client_or = get_connected_client();
    if (!client_or.ok()) {
      return client_or.status();
    }
    auto client = std::move(*client_or);
    auto reg_or = client->register_memory_replica(
        publication.artifact_id,
        worker_id(),
        publication.device,
        publication.size_bytes,
        publication.tensor_index_key,
        publication.remote_memory_keys,
        publication.buffer_sizes,
        publication.tensor_index_data,
        publication.encoding,
        publication.schema_version,
        /*max_concurrency=*/1,
        publication.verification_json);
    if (!reg_or.ok()) {
      return reg_or.status();
    }
    return absl::OkStatus();
  }

  absl::Status update_variant_view(const components::VariantViewUpdate& update) override {
    auto client_or = get_connected_client();
    if (!client_or.ok()) {
      return client_or.status();
    }
    return (*client_or)->update_artifact_view_state(update);
  }

 private:
  std::string worker_id() const {
    const auto& identity = runtime_context_->worker_identity();
    return identity.worker_id.empty() ? std::string("local") : identity.worker_id;
  }

  absl::StatusOr<std::shared_ptr<components::IGlobalStoreClient>> get_connected_client() const {
    std::shared_ptr<components::IGlobalStoreClient> client = override_slot_ ? *override_slot_ : nullptr;
    if (!client) {
      client = runtime_context_->global_store_client();
    }
    if (!client || !client->is_connected()) {
      return absl::FailedPreconditionError("GlobalStoreClient not connected");
    }
    return client;
  }

  gsl::not_null<RuntimeContext*> runtime_context_;
  std::shared_ptr<components::IGlobalStoreClient>* override_slot_;
};

} // namespace

MetadataGateway::MetadataGateway(Config config)
    : runtime_context_(gsl::not_null<RuntimeContext*>{config.runtime_context}),
      replica_runtime_(gsl::not_null<ReplicaRuntime*>{config.replica_runtime}),
      event_publisher_(runtime_context_->event_publisher()),
      artifact_chunk_bytes_(config.artifact_chunk_bytes),
      pinned_memory_timeout_(config.pinned_memory_timeout),
      replica_factory_(config.replica_factory ? config.replica_factory : make_default_replica_factory()) {
  registration_publisher_ = std::make_unique<GlobalStoreRegistrationPublisher>(runtime_context_, &override_client_);
  registration_backend_ = std::make_unique<RegistrationBackend>(
      make_registration_resources(),
      replica_factory_,
      artifact_chunk_bytes_,
      pinned_memory_timeout_,
      registration_publisher_.get());
  if (runtime_context_->ingestion_event_hub() != nullptr) {
    ingestion_event_subscription_ = runtime_context_->ingestion_event_hub()->subscribe_completed(
        [this](const IngestionCompletedEvent& event) { handle_ingestion_result(event); });
  }
}

MetadataGateway::~MetadataGateway() = default;

bool MetadataGateway::is_connected() const {
  auto client = override_client_ ? override_client_ : runtime_context_->global_store_client();
  return client != nullptr && client->is_connected();
}

void MetadataGateway::set_client_override(std::shared_ptr<components::IGlobalStoreClient> client) {
  override_client_ = std::move(client);
  refresh_override_endpoint();
}

void MetadataGateway::refresh_override_endpoint() {
  if (!override_client_) {
    return;
  }
  const auto& identity = runtime_context_->worker_identity();
  override_client_->update_local_endpoint(
      identity.node_id, identity.node_address, identity.grpc_port, identity.p2p_port);
}

void MetadataGateway::handle_ingestion_result(const IngestionResultEvent& event) {
  if (!is_connected() || !event.publish_to_global_store) {
    return;
  }
  if (!event.status.ok()) {
    return;
  }
  if (!event.replica_key.has_value()) {
    return;
  }
  absl::Status st = register_replica(*event.replica_key, {}, event.publish_context_id);
  if (!st.ok()) {
    LOG(WARNING) << "HandleIngestionResult: register_replica failed for artifact=" << event.artifact_id << ": " << st;
  }
}

absl::Status MetadataGateway::register_replica(
    const loading::ReplicaKey& key,
    std::string_view artifact_id_override,
    std::string_view publish_context_id) {
  if (should_skip_publish_for_context(publish_context_id, key)) {
    VLOG(1) << "Publish context " << publish_context_id << " already registered for artifact=" << key.artifact_id
            << " device=" << key.device.to_string();
    return absl::OkStatus();
  }

  auto client_or = get_connected_client();
  if (!client_or.ok()) {
    return client_or.status();
  }
  auto client = std::move(*client_or);

  if (key.view_id.has_value()) {
    VLOG(1) << "MetadataGateway registering variant view_id=" << *key.view_id
            << " (canonical_artifact_id=" << key.artifact_id << ")";
  }

  auto size_or = replica_runtime_->get_replica_size(key);
  if (!size_or.ok()) {
    return size_or.status();
  }

  auto canonical_id_or = canonical_artifact_id(key, artifact_id_override);
  if (!canonical_id_or.ok()) {
    return canonical_id_or.status();
  }
  const std::string& artifact_id = *canonical_id_or;

  const common::memory::MemoryLocation loc =
      (key.device.type == DeviceType::GPU) ? common::memory::MemoryLocation::GPU : common::memory::MemoryLocation::CPU;

  absl::Status register_status = materialization::control::ReplicaRegistrationHelper::register_local_replica(
      gsl::not_null<components::IGlobalStoreClient*>{client.get()},
      worker_id(),
      artifact_id,
      key.device,
      loc,
      *size_or);
  if (!register_status.ok()) {
    return register_status;
  }

  if (key.view_id.has_value()) {
    auto variant_status = client->record_variant_residency(key.artifact_id, *key.view_id, *size_or);
    if (!variant_status.ok()) {
      if (absl::IsUnimplemented(variant_status)) {
        VLOG(1) << "Global Store does not yet accept variant residency updates: " << variant_status.message();
      } else {
        LOG(WARNING) << "record_variant_residency failed for view_id=" << *key.view_id << ": " << variant_status;
      }
    }
  }
  record_publish_context_result(publish_context_id, key, register_status);
  return register_status;
}

absl::Status MetadataGateway::unregister_replica(std::string_view artifact_id, int device_id) {
  auto client_or = get_connected_client();
  if (!client_or.ok()) {
    return client_or.status();
  }
  auto client = std::move(*client_or);
  return client->unregister_replica_by_worker(
      artifact_id, worker_id(), common::memory::MemoryLocation::GPU, static_cast<uint32_t>(device_id));
}

absl::StatusOr<components::KeyMapping> MetadataGateway::resolve_key_mapping(std::string_view key) const {
  auto client_or = get_connected_client();
  if (!client_or.ok()) {
    return client_or.status();
  }
  return (*client_or)->resolve_key_mapping(key);
}

absl::StatusOr<std::string> MetadataGateway::get_canonical_index(std::string_view artifact_id) const {
  auto client_or = get_connected_client();
  if (!client_or.ok()) {
    return client_or.status();
  }
  return (*client_or)->get_artifact_index_by_id(artifact_id);
}

absl::Status MetadataGateway::upsert_key_mapping(
    std::string_view key,
    std::string_view artifact_id,
    std::string_view disk_path,
    absl::Duration ttl) {
  auto client_or = get_connected_client();
  if (!client_or.ok()) {
    return client_or.status();
  }
  return (*client_or)->upsert_key_mapping(key, artifact_id, disk_path, ttl);
}

absl::Status MetadataGateway::revoke_key_mapping(std::string_view key) {
  auto client_or = get_connected_client();
  if (!client_or.ok()) {
    return client_or.status();
  }
  return (*client_or)->revoke_key_mapping(key);
}

absl::StatusOr<RegistrationBeginResult> MetadataGateway::begin_registration(const ArtifactRegistration& reg) {
  if (!registration_backend_) {
    return absl::FailedPreconditionError("registration backend is not initialized");
  }
  return registration_backend_->begin(reg);
}

absl::StatusOr<RegistrationCommitResult> MetadataGateway::commit_registration(std::string_view registration_id) {
  if (!registration_backend_) {
    return absl::FailedPreconditionError("registration backend is not initialized");
  }
  auto result_or = registration_backend_->commit(registration_id);
  if (!result_or.ok()) {
    publish_registration_event(RuntimeEventType::kRegistrationAborted, registration_id, nullptr, result_or.status());
    replica_runtime_->update_memory_pool_metrics();
    return result_or.status();
  }
  publish_registration_event(
      RuntimeEventType::kRegistrationCommitted, registration_id, &result_or.value(), absl::OkStatus());
  replica_runtime_->update_memory_pool_metrics();
  return result_or;
}

absl::Status MetadataGateway::abort_registration(std::string_view registration_id) {
  if (!registration_backend_) {
    return absl::FailedPreconditionError("registration backend is not initialized");
  }
  auto status = registration_backend_->abort(registration_id);
  publish_registration_event(RuntimeEventType::kRegistrationAborted, registration_id, nullptr, status);
  replica_runtime_->update_memory_pool_metrics();
  return status;
}

absl::Status MetadataGateway::keep_alive_registration(std::string_view registration_id, uint32_t ttl_ms) {
  if (!registration_backend_) {
    return absl::FailedPreconditionError("registration backend is not initialized");
  }
  return registration_backend_->keep_alive(registration_id, ttl_ms);
}

absl::Status MetadataGateway::ingest_view_chunk(
    std::string_view registration_id,
    uint64_t view_offset,
    absl::Span<const std::byte> data) {
  if (!registration_backend_) {
    return absl::FailedPreconditionError("registration backend is not initialized");
  }
  return registration_backend_->ingest_view_chunk(registration_id, view_offset, data);
}

absl::StatusOr<uint64_t> MetadataGateway::get_view_ingested_bytes(std::string_view registration_id) const {
  if (!registration_backend_) {
    return absl::FailedPreconditionError("registration backend is not initialized");
  }
  return registration_backend_->get_view_ingested_bytes(registration_id);
}

void MetadataGateway::publish_registration_event(
    RuntimeEventType type,
    std::string_view registration_id,
    const RegistrationCommitResult* result,
    const absl::Status& status) const {
  if (!event_publisher_) {
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
  event_publisher_.publish(std::move(event));
}

std::string MetadataGateway::worker_id() const {
  const auto& identity = runtime_context_->worker_identity();
  return identity.worker_id.empty() ? std::string("local") : identity.worker_id;
}

absl::StatusOr<std::shared_ptr<components::IGlobalStoreClient>> MetadataGateway::get_connected_client() const {
  auto client = override_client_ ? override_client_ : runtime_context_->global_store_client();
  if (!client || !client->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  return client;
}

RegistrationResources MetadataGateway::make_registration_resources() const {
  RegistrationResources resources{
      .device_manager = gsl::not_null<components::DeviceManager*>{&runtime_context_->device_manager()},
      .replica_registry = gsl::not_null<components::ReplicaRegistry*>{&replica_runtime_->registry()},
      .metrics_collector = gsl::not_null<components::MetricsCollector*>{&runtime_context_->metrics_collector()},
      .memory_pool =
          gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{runtime_context_->pinned_buffer_pool()},
      .communication_manager = runtime_context_->communication_manager(),
      .async_runtime = runtime_context_->async_runtime(),
      .memory_tier_budget = runtime_context_->memory_tier_budget(),
      .memory_tier_config = runtime_context_->options().memory_tier_config,
  };
  return resources;
}

ReplicaFactory MetadataGateway::make_default_replica_factory() const {
  return [](const replica::ReplicaConfig& config) -> absl::StatusOr<std::shared_ptr<replica::Replica>> {
    auto create_or = replica::Replica::create(config);
    if (!create_or.ok()) {
      return create_or.status();
    }
    return std::shared_ptr<replica::Replica>(std::move(create_or.value()));
  };
}

bool MetadataGateway::should_skip_publish_for_context(
    std::string_view publish_context_id,
    const loading::ReplicaKey& key) const {
  if (publish_context_id.empty()) {
    return false;
  }
  absl::MutexLock lock(&publish_context_mu_);
  auto it = publish_contexts_.find(publish_context_id);
  if (it == publish_contexts_.end()) {
    return false;
  }
  if (!it->second.status.ok()) {
    return false;
  }
  return it->second.key == key;
}

void MetadataGateway::record_publish_context_result(
    std::string_view publish_context_id,
    const loading::ReplicaKey& key,
    const absl::Status& status) {
  if (publish_context_id.empty()) {
    return;
  }
  PublishContextRecord record{
      .key = key,
      .status = status,
      .updated_at = absl::Now(),
  };
  absl::MutexLock lock(&publish_context_mu_);
  publish_contexts_[std::string(publish_context_id)] = std::move(record);
  if (publish_contexts_.size() > kMaxPublishContextRecords) {
    cleanup_publish_contexts_locked(absl::Now());
  }
}

void MetadataGateway::cleanup_publish_contexts_locked(absl::Time now) const {
  const absl::Time expiry = now - kPublishContextTtl;
  for (auto it = publish_contexts_.begin(); it != publish_contexts_.end();) {
    if (it->second.updated_at < expiry) {
      publish_contexts_.erase(it++);
      continue;
    }
    ++it;
  }
  if (publish_contexts_.size() <= kMaxPublishContextRecords) {
    return;
  }
  std::vector<std::pair<absl::Time, std::string>> ordered;
  ordered.reserve(publish_contexts_.size());
  for (const auto& entry : publish_contexts_) {
    ordered.emplace_back(entry.second.updated_at, entry.first);
  }
  std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& candidate : ordered) {
    if (publish_contexts_.size() <= kMaxPublishContextRecords) {
      break;
    }
    publish_contexts_.erase(candidate.second);
  }
}

} // namespace tensorcast::store::runtime::metadata
