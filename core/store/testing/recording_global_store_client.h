// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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
  std::vector<components::VariantInfo> variant_infos;
  std::optional<components::ArtifactBinding> artifact_binding;
  std::vector<components::MemoryTierStatusPayload> memory_tier_statuses;
  std::vector<components::MemoryTierLeaseDescriptor> memory_tier_leases;
  std::vector<components::PlacementPlanResult> placement_plans;
  std::vector<components::PersistenceReport> persistence_reports;
  bool allow_plan_placement{true};
  bool plan_degraded{false};
  bool deny_leases{false};
  bool fail_register_replica{false};
  bool fail_acknowledge_lease{false};
  std::string remote_node_id{"stub-remote"};
  std::string plan_degraded_reason{"insufficient_remote_capacity"};
  std::optional<std::string> canonical_index_json;

  absl::Status initialize() override {
    return absl::OkStatus();
  }

  absl::StatusOr<components::WorkerRegistrationInfo> register_worker(
      std::string_view,
      std::string_view,
      uint32_t,
      uint32_t,
      uint64_t,
      uint64_t,
      bool,
      std::string_view,
      std::string_view) override {
    return absl::UnimplementedError("register_worker not supported in test stub");
  }

  absl::StatusOr<tensorcast::global_store::v1::WorkerHeartbeatResponse> send_heartbeat_enhanced(
      std::string_view,
      uint64_t,
      bool,
      uint64_t,
      std::string_view,
      const std::vector<std::string>&,
      int64_t,
      tensorcast::global_store::v1::ConnectionStatus,
      const components::RpcOptions&,
      std::string_view) override {
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
      uint32_t,
      std::optional<std::string_view>) override {
    if (fail_register_replica) {
      return absl::UnavailableError("register_replica disabled in RecordingGlobalStoreClient");
    }
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
      const std::optional<std::string>&,
      std::optional<std::string_view>,
      const std::optional<common::v1::ArtifactDescriptor>&) override {
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

  absl::StatusOr<std::vector<components::VariantInfo>> list_variants(std::string_view) override {
    return variant_infos;
  }

  absl::StatusOr<components::ArtifactBinding> get_artifact_binding(std::string_view) override {
    if (!artifact_binding.has_value()) {
      return absl::NotFoundError("artifact binding not found");
    }
    return *artifact_binding;
  }

  absl::StatusOr<components::ArtifactBindingResult> upsert_artifact_binding(
      const components::ArtifactBinding& binding) override {
    bool created = !artifact_binding.has_value();
    artifact_binding = binding;
    components::ArtifactBindingResult result;
    result.binding = binding;
    result.created = created;
    return result;
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

  absl::StatusOr<std::vector<components::RemoteReplicaInfo>> get_artifact_replicas(
      std::string_view,
      std::optional<std::string_view>) override {
    return absl::UnimplementedError("get_artifact_replicas not supported in test stub");
  }

  absl::StatusOr<std::vector<components::ChunkLocationInfo>> query_chunk_locations(
      std::string_view,
      const std::vector<uint32_t>&) override {
    return absl::UnimplementedError("query_chunk_locations not supported in test stub");
  }

  absl::StatusOr<components::StateSyncResult> synchronize_worker_state(
      const tensorcast::global_store::v1::WorkerLocalState&,
      bool,
      const components::StateSyncToken&,
      const components::RpcOptions&) override {
    return absl::UnimplementedError("synchronize_worker_state not supported in test stub");
  }

  absl::StatusOr<components::FullStateSyncResult> request_full_state_sync(
      std::string_view,
      uint64_t,
      const components::StateSyncToken&,
      const components::RpcOptions&) override {
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
    if (canonical_index_json.has_value()) {
      return *canonical_index_json;
    }
    return absl::UnimplementedError("get_artifact_index_by_id not supported in test stub");
  }

  absl::StatusOr<components::ViewMetadata> get_view_metadata(std::string_view, std::string_view) override {
    return absl::UnimplementedError("get_view_metadata not supported in test stub");
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
    if (deny_leases) {
      return absl::UnavailableError("lease denied by test stub");
    }
    memory_tier_leases.push_back(request);
    return request;
  }

  absl::StatusOr<components::MemoryTierLeaseDescriptor> acknowledge_memory_tier_lease(
      const components::MemoryTierLeaseAckPayload& ack) override {
    if (fail_acknowledge_lease) {
      return absl::UnavailableError("lease ack denied in RecordingGlobalStoreClient");
    }
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

  absl::StatusOr<components::PlacementPlanResult> plan_placement(
      std::string_view artifact_id,
      tensorcast::global_store::v1::PlacementPolicy policy,
      const std::vector<components::PlacementShardSpec>& shards,
      std::string_view source_node_id) override {
    if (!allow_plan_placement) {
      return absl::UnavailableError("plan placement disabled in RecordingGlobalStoreClient");
    }
    components::PlacementPlanResult plan;
    plan.plan_id = absl::StrCat("plan-", placement_plans.size());
    plan.effective_policy = policy;
    plan.degraded = plan_degraded && policy != tensorcast::global_store::v1::PLACEMENT_POLICY_LOCAL_ONLY;
    plan.degraded_reason = plan.degraded ? plan_degraded_reason : "";
    for (const auto& shard : shards) {
      components::ShardPlacement placement;
      placement.shard = shard;
      components::PlacementTargetStatus target;
      target.node_id = std::string(source_node_id);
      target.target_state = tensorcast::global_store::v1::PLACEMENT_TARGET_STATE_PENDING;
      placement.targets.push_back(std::move(target));
      if (plan.degraded) {
        placement.degraded_reason = plan.degraded_reason;
      } else if (policy != tensorcast::global_store::v1::PLACEMENT_POLICY_LOCAL_ONLY) {
        components::PlacementTargetStatus remote;
        remote.node_id = remote_node_id;
        remote.target_state = tensorcast::global_store::v1::PLACEMENT_TARGET_STATE_PENDING;
        placement.targets.push_back(std::move(remote));
      }
      plan.placements.push_back(std::move(placement));
    }
    placement_plans.push_back(plan);
    return plan;
  }

  absl::Status report_persistence_status(const components::PersistenceReport& report) override {
    persistence_reports.push_back(report);
    return absl::OkStatus();
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
