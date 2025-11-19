// Copyright (c) 2025, TensorCast Team.

#include "core/store/runtime/global_metadata_gateway.h"

#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "core/store/materialization/control/replica_registration_helper.h"

namespace tensorcast::store::runtime {

namespace {

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

} // namespace

GlobalMetadataGateway::GlobalMetadataGateway(Config config)
    : component_catalog_(gsl::not_null<ComponentCatalog*>{config.component_catalog}),
      replica_runtime_(gsl::not_null<ReplicaRuntime*>{config.replica_runtime}),
      event_hub_(config.event_hub) {
  if (event_hub_ != nullptr) {
    ingress_subscription_ = event_hub_->subscribe([this](const RuntimeEvent& event) {
      if (event.type != RuntimeEventType::kIngressCompleted) {
        return;
      }
      const auto* ingress = std::get_if<IngestionResultEvent>(&event.payload);
      if (ingress == nullptr) {
        return;
      }
      handle_ingestion_result(*ingress);
    });
  }
}

bool GlobalMetadataGateway::is_connected() const {
  auto client = override_client_ ? override_client_ : component_catalog_->global_store_client();
  return client != nullptr && client->is_connected();
}

void GlobalMetadataGateway::set_client_override(std::shared_ptr<components::IGlobalStoreClient> client) {
  override_client_ = std::move(client);
  refresh_override_endpoint();
}

void GlobalMetadataGateway::refresh_override_endpoint() {
  if (!override_client_) {
    return;
  }
  const auto& identity = component_catalog_->worker_identity();
  override_client_->update_local_endpoint(
      identity.node_id, identity.node_address, identity.grpc_port, identity.p2p_port);
}

void GlobalMetadataGateway::handle_ingestion_result(const IngestionResultEvent& event) {
  if (!is_connected() || !event.publish_to_global_store) {
    return;
  }
  if (!event.status.ok()) {
    return;
  }
  if (!event.replica_key.has_value()) {
    return;
  }
  absl::Status st = register_replica(*event.replica_key, {});
  if (!st.ok()) {
    LOG(WARNING) << "HandleIngestionResult: register_replica failed for artifact=" << event.artifact_id << ": " << st;
  }
}

absl::Status GlobalMetadataGateway::register_replica(
    const loading::ReplicaKey& key,
    std::string_view artifact_id_override) {
  auto client_or = get_connected_client();
  if (!client_or.ok()) {
    return client_or.status();
  }
  auto client = std::move(*client_or);

  if (key.view_id.has_value()) {
    VLOG(1) << "GlobalMetadataGateway registering variant view_id=" << *key.view_id
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
  return register_status;
}

absl::Status GlobalMetadataGateway::unregister_replica(std::string_view artifact_id, int device_id) {
  auto client_or = get_connected_client();
  if (!client_or.ok()) {
    return client_or.status();
  }
  auto client = std::move(*client_or);
  return client->unregister_replica_by_worker(
      artifact_id, worker_id(), common::memory::MemoryLocation::GPU, static_cast<uint32_t>(device_id));
}

absl::StatusOr<components::KeyMapping> GlobalMetadataGateway::resolve_key_mapping(std::string_view key) const {
  auto client_or = get_connected_client();
  if (!client_or.ok()) {
    return client_or.status();
  }
  return (*client_or)->resolve_key_mapping(key);
}

absl::StatusOr<std::string> GlobalMetadataGateway::get_canonical_index(std::string_view artifact_id) const {
  auto client_or = get_connected_client();
  if (!client_or.ok()) {
    return client_or.status();
  }
  return (*client_or)->get_artifact_index_by_id(artifact_id);
}

absl::Status GlobalMetadataGateway::upsert_key_mapping(
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

absl::Status GlobalMetadataGateway::revoke_key_mapping(std::string_view key) {
  auto client_or = get_connected_client();
  if (!client_or.ok()) {
    return client_or.status();
  }
  return (*client_or)->revoke_key_mapping(key);
}

absl::Status GlobalMetadataGateway::publish_registration(const RegistrationPublication& publication) {
  if (!is_connected()) {
    return absl::OkStatus();
  }
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

absl::Status GlobalMetadataGateway::update_variant_view(const components::VariantViewUpdate& update) {
  if (!is_connected()) {
    return absl::OkStatus();
  }
  auto client_or = get_connected_client();
  if (!client_or.ok()) {
    return client_or.status();
  }
  return (*client_or)->update_artifact_view_state(update);
}

std::string GlobalMetadataGateway::worker_id() const {
  const auto& identity = component_catalog_->worker_identity();
  return identity.worker_id.empty() ? std::string("local") : identity.worker_id;
}

absl::StatusOr<std::shared_ptr<components::IGlobalStoreClient>> GlobalMetadataGateway::get_connected_client() const {
  auto client = override_client_ ? override_client_ : component_catalog_->global_store_client();
  if (!client || !client->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  return client;
}

} // namespace tensorcast::store::runtime
