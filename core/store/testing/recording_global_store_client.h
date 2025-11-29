// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "core/store/components/global_store_client.h"
#include "tensorcast/memory_tier/v1/memory_tier.pb.h"

namespace tensorcast::store::testing {

class RecordingGlobalStoreClient final : public components::IGlobalStoreClient {
 public:
  bool connected{true};
  bool allow_replica_transport{false};
  bool allow_view_transport{false};
  std::vector<std::string> view_requests;
  std::vector<std::string> replica_requests;
  std::vector<std::string> registered_replicas;
  std::vector<std::tuple<std::string, std::string, uint64_t>> recorded_variants;
  std::vector<components::VariantViewUpdate> view_updates;
  std::vector<components::MemoryTierStatusPayload> memory_tier_statuses;
  std::vector<components::MemoryTierLeaseDescriptor> memory_tier_leases;

  absl::Status initialize() override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> register_worker(
      std::string_view,
      std::string_view,
      uint32_t,
      uint32_t,
      uint64_t,
      uint64_t,
      bool,
      std::string_view) override {
    return absl::UnimplementedError("register_worker not supported in test stub");
  }

  absl::Status send_heartbeat(std::string_view, uint64_t, bool) override {
    return absl::UnimplementedError("send_heartbeat not supported in test stub");
  }

  absl::StatusOr<tensorcast::global_store::v1::WorkerHeartbeatResponse> send_heartbeat_enhanced(
      std::string_view,
      uint64_t,
      bool,
      uint64_t,
      std::string_view,
      const std::vector<std::string>&,
      int64_t,
      tensorcast::global_store::v1::ConnectionStatus) override {
    return absl::UnimplementedError("send_heartbeat_enhanced not supported in test stub");
  }

  absl::Status unregister_worker(std::string_view, bool) override {
    return absl::UnimplementedError("unregister_worker not supported in test stub");
  }

  absl::StatusOr<std::string> register_replica(
      std::string_view artifact_id,
      std::string_view,
      const tensorcast::store::DeviceKey&,
      common::memory::MemoryLocation,
      uint64_t,
      uint32_t) override {
    registered_replicas.emplace_back(artifact_id);
    return std::string("replica-0");
  }

  absl::Status record_variant_residency(
      std::string_view canonical_artifact_id,
      std::string_view view_id,
      uint64_t view_size_bytes,
      std::optional<std::string_view>) override {
    recorded_variants.emplace_back(std::string(canonical_artifact_id), std::string(view_id), view_size_bytes);
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> register_memory_replica(
      std::string_view artifact_id,
      std::string_view,
      const tensorcast::store::DeviceKey&,
      uint64_t,
      std::string_view,
      const std::vector<std::string>&,
      const std::vector<uint64_t>&,
      const std::optional<std::string>&,
      std::string_view,
      std::string_view,
      uint32_t,
      const std::optional<std::string>&) override {
    registered_replicas.emplace_back(std::string(artifact_id));
    return std::string("memory_replica");
  }

  absl::Status unregister_replica(std::string_view, std::string_view) override {
    return absl::UnimplementedError("unregister_replica not supported in test stub");
  }

  absl::Status unregister_replica_by_worker(
      std::string_view,
      std::string_view,
      std::optional<common::memory::MemoryLocation>,
      std::optional<uint32_t>) override {
    return absl::UnimplementedError("unregister_replica_by_worker not supported in test stub");
  }

  absl::Status update_artifact_view_state(const components::VariantViewUpdate& update) override {
    view_requests.emplace_back(update.view_id);
    view_updates.push_back(update);
    return absl::OkStatus();
  }

  absl::StatusOr<components::TransportSession> request_replica_transport(
      std::string_view artifact_id,
      std::string_view,
      std::string_view,
      uint32_t,
      const tensorcast::store::DeviceKey& target_device,
      uint32_t) override {
    replica_requests.emplace_back(std::string(artifact_id));
    if (!allow_replica_transport) {
      return absl::UnavailableError("replica transport disabled in RecordingGlobalStoreClient");
    }
    return make_transport_session(artifact_id, target_device);
  }

  absl::StatusOr<components::TransportSession> request_view_transport(
      std::string_view artifact_id,
      std::string_view view_id,
      std::string_view,
      std::string_view,
      uint32_t,
      const tensorcast::store::DeviceKey& target_device,
      uint32_t) override {
    view_requests.emplace_back(std::string(view_id));
    if (!allow_view_transport) {
      return absl::NotFoundError("view transport disabled in RecordingGlobalStoreClient");
    }
    return make_transport_session(artifact_id, target_device);
  }

  absl::Status complete_replica_transport(std::string_view) override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<components::RemoteReplicaInfo>> get_artifact_replicas(std::string_view) override {
    return absl::UnimplementedError("get_artifact_replicas not supported in test stub");
  }

  absl::StatusOr<std::vector<components::ChunkLocationInfo>> query_chunk_locations(
      std::string_view,
      const std::vector<uint32_t>&) override {
    return absl::UnimplementedError("query_chunk_locations not supported in test stub");
  }

  absl::StatusOr<std::pair<uint64_t, std::string>> synchronize_worker_state(
      const tensorcast::global_store::v1::WorkerLocalState&,
      bool,
      std::vector<tensorcast::global_store::v1::StateChange>*) override {
    return absl::UnimplementedError("synchronize_worker_state not supported in test stub");
  }

  absl::StatusOr<std::pair<uint64_t, std::string>> request_full_state_sync(
      std::string_view,
      uint64_t,
      std::vector<tensorcast::common::v1::ReplicaInfo>*) override {
    return absl::UnimplementedError("request_full_state_sync not supported in test stub");
  }

  bool is_connected() const override {
    return connected;
  }

  absl::Status batch_update_chunk_states(
      std::string_view,
      std::string_view,
      const std::vector<components::ChunkStateUpdate>&) override {
    return absl::UnimplementedError("batch_update_chunk_states not supported in test stub");
  }

  absl::StatusOr<components::KeyMapping> resolve_key_mapping(std::string_view) override {
    return absl::UnimplementedError("resolve_key_mapping not supported in test stub");
  }

  absl::StatusOr<std::string> get_artifact_index_by_id(std::string_view) override {
    return absl::UnimplementedError("get_artifact_index_by_id not supported in test stub");
  }

  absl::Status upsert_key_mapping(std::string_view, std::string_view, std::string_view, absl::Duration) override {
    return absl::UnimplementedError("upsert_key_mapping not supported in test stub");
  }

  absl::Status revoke_key_mapping(std::string_view) override {
    return absl::UnimplementedError("revoke_key_mapping not supported in test stub");
  }

  absl::Status publish_memory_tier_status(const components::MemoryTierStatusPayload& status) override {
    memory_tier_statuses.push_back(status);
    return absl::OkStatus();
  }

  absl::StatusOr<components::MemoryTierLeaseDescriptor> request_memory_tier_lease(
      const components::MemoryTierLeaseDescriptor& request) override {
    memory_tier_leases.push_back(request);
    return request;
  }

  absl::StatusOr<components::MemoryTierLeaseDescriptor> acknowledge_memory_tier_lease(
      const components::MemoryTierLeaseAckPayload& ack) override {
    components::MemoryTierLeaseDescriptor lease;
    lease.lease_id = ack.lease_id;
    lease.node_id = ack.node_id;
    lease.artifact_id = ack.artifact_id;
    lease.chunk_ids = ack.chunk_ids;
    lease.chunk_start = ack.chunk_start;
    lease.chunk_count = ack.chunk_count;
    lease.ledger_version = ack.ledger_version;
    lease.bytes = ack.bytes;
    lease.request_id = ack.request_id;
    lease.state = components::MemoryTierLeaseState::kActive;
    memory_tier_leases.push_back(lease);
    return lease;
  }

  absl::StatusOr<std::vector<components::MemoryTierLeaseDescriptor>> list_memory_tier_leases(
      std::string_view) override {
    return memory_tier_leases;
  }

  absl::StatusOr<components::MemoryTierLeaseDescriptor> revoke_memory_tier_lease(std::string_view lease_id) override {
    for (auto& l : memory_tier_leases) {
      if (l.lease_id == lease_id) {
        l.state = components::MemoryTierLeaseState::kRevoking;
        return l;
      }
    }
    return absl::NotFoundError("lease not found");
  }

  void update_local_endpoint(std::string, std::string, uint32_t, uint32_t) override {}

 private:
  static components::TransportSession make_transport_session(
      std::string_view artifact_id,
      const tensorcast::store::DeviceKey& target_device) {
    components::TransportSession session;
    (void)artifact_id;
    session.transport_id = "test-transport";
    session.remote_replica.node_id = "stub-node";
    session.remote_replica.node_address = "127.0.0.1";
    session.remote_replica.node_port = 12345;
    session.remote_replica.memory_size = 16;
    session.remote_replica.memory_type = common::memory::MemoryLocation::CPU;
    session.remote_replica.device_id = target_device.ordinal;
    session.remote_replica.remote_memory_keys = {"tensor.data_0"};
    session.remote_replica.buffer_sizes = {16};
    session.remote_replica.verification_json = "{}";
    return session;
  }
};

inline std::shared_ptr<components::IGlobalStoreClient> MakeRecordingGlobalStoreClient() {
  return std::make_shared<RecordingGlobalStoreClient>();
}

} // namespace tensorcast::store::testing
