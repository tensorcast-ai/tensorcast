// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
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
  bool replica_transport_not_found{false};
  std::vector<std::string> view_requests;
  std::vector<std::string> replica_requests;
  std::vector<std::optional<components::TransportSchedulingGroupHint>> view_request_groups;
  std::vector<std::optional<components::TransportSchedulingGroupHint>> replica_request_groups;
  std::vector<std::string> view_request_request_ids;
  std::vector<std::string> replica_request_request_ids;
  std::vector<std::string> view_request_requester_worker_ids;
  std::vector<std::string> replica_request_requester_worker_ids;
  std::vector<uint32_t> view_request_wait_timeouts_ms;
  std::vector<uint32_t> replica_request_wait_timeouts_ms;
  std::vector<std::string> completed_transport_ids;
  std::vector<components::TransportCompletionOutcome> completed_transport_outcomes;
  std::vector<std::string> completed_transport_outcome_details;
  std::vector<std::string> registered_replicas;
  std::vector<std::pair<std::string, std::string>> unregistered_replicas;
  std::vector<std::string> marked_unavailable;
  std::vector<std::string> drained_replicas;
  std::vector<std::string> call_sequence;
  std::vector<std::tuple<std::string, std::string, uint64_t>> recorded_views;
  std::vector<components::ViewStateUpdate> view_updates;
  std::vector<components::ViewInfo> view_infos;
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
  bool fail_disk_location_upsert{false};
  absl::Status unregister_replica_status{absl::OkStatus()};
  absl::Status unregister_replica_by_worker_status{absl::OkStatus()};
  bool drain_success{true};
  uint32_t drain_current_requests{0};
  std::string remote_node_id{"stub-remote"};
  std::string plan_degraded_reason{"insufficient_remote_capacity"};
  std::string remote_node_address{"127.0.0.1"};
  uint32_t remote_node_port{12345};
  std::optional<std::string> canonical_index_json;
  std::string cluster_id{"cluster-test"};
  std::vector<components::ArtifactDiskLocation> disk_locations;

  struct TransportReplicaInfo {
    std::string artifact_id;
    std::optional<std::string> view_id;
    uint64_t memory_size{0};
    std::vector<std::string> remote_memory_keys;
    std::vector<uint64_t> buffer_sizes;
    common::memory::MemoryLocation memory_type{common::memory::MemoryLocation::CPU};
    int device_id{0};
  };

  struct UnregisterReplicaByWorkerCall {
    std::string artifact_id;
    std::string worker_id;
    std::optional<common::memory::MemoryLocation> memory_type;
    std::optional<uint32_t> device_id;
  };

  std::unordered_map<std::string, TransportReplicaInfo> transport_replicas;
  std::vector<UnregisterReplicaByWorkerCall> unregister_replica_by_worker_calls;
  std::vector<components::TransportSession> scripted_transport_sessions;
  size_t scripted_transport_next{0};

  void push_scripted_transport_session(components::TransportSession session) {
    scripted_transport_sessions.push_back(std::move(session));
  }

  void clear_scripted_transport_sessions() {
    scripted_transport_sessions.clear();
    scripted_transport_next = 0;
  }

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
      std::string_view,
      uint64_t) override {
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
      std::string_view,
      uint64_t) override {
    return absl::UnimplementedError("send_heartbeat_enhanced not supported in test stub");
  }

  absl::Status unregister_worker(std::string_view, bool) override {
    return absl::UnimplementedError("unregister_worker not supported in test stub");
  }

  absl::StatusOr<std::vector<components::ActiveWorkerInfo>> list_active_workers(
      bool,
      uint64_t,
      const components::RpcOptions&) override {
    return absl::UnimplementedError("list_active_workers not supported in test stub");
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

  absl::Status record_view_residency(
      std::string_view canonical_artifact_id,
      std::string_view view_id,
      uint64_t view_size_bytes,
      std::optional<std::string_view>) override {
    recorded_views.emplace_back(std::string(canonical_artifact_id), std::string(view_id), view_size_bytes);
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> register_memory_replica(
      std::string_view artifact_id,
      std::string_view,
      const tensorcast::store::DeviceKey& device,
      uint64_t size_bytes,
      std::string_view,
      const std::vector<std::string>& remote_memory_keys,
      const std::vector<uint64_t>& buffer_sizes,
      const std::optional<std::string>&,
      std::string_view,
      std::string_view,
      uint32_t,
      const std::optional<std::string>&,
      std::optional<std::string_view> view_id,
      const std::optional<common::v1::ArtifactDescriptor>&) override {
    registered_replicas.emplace_back(std::string(artifact_id));
    TransportReplicaInfo info;
    info.artifact_id = std::string(artifact_id);
    if (view_id.has_value()) {
      info.view_id = std::string(*view_id);
    }
    info.memory_size = size_bytes;
    info.remote_memory_keys = remote_memory_keys;
    info.buffer_sizes = buffer_sizes;
    info.memory_type =
        (device.type == DeviceType::GPU) ? common::memory::MemoryLocation::GPU : common::memory::MemoryLocation::CPU;
    info.device_id = device.ordinal;
    transport_replicas[transport_key(info.artifact_id, view_id)] = std::move(info);
    return std::string("memory_replica");
  }

  absl::Status unregister_replica(std::string_view artifact_id, std::string_view replica_id) override {
    unregistered_replicas.emplace_back(std::string(artifact_id), std::string(replica_id));
    return unregister_replica_status;
  }

  absl::Status unregister_replica_by_worker(
      std::string_view artifact_id,
      std::string_view worker_id,
      std::optional<common::memory::MemoryLocation> memory_type,
      std::optional<uint32_t> device_id) override {
    unregister_replica_by_worker_calls.push_back(
        UnregisterReplicaByWorkerCall{
            .artifact_id = std::string(artifact_id),
            .worker_id = std::string(worker_id),
            .memory_type = memory_type,
            .device_id = device_id});
    return unregister_replica_by_worker_status;
  }

  absl::StatusOr<bool> mark_replica_unavailable(
      std::string_view,
      std::string_view replica_id,
      std::optional<std::string_view>,
      std::optional<std::string_view>) override {
    if (replica_id.empty()) {
      return absl::InvalidArgumentError("replica_id required");
    }
    call_sequence.emplace_back(absl::StrCat("mark:", replica_id));
    marked_unavailable.emplace_back(std::string(replica_id));
    return true;
  }

  absl::StatusOr<components::ReplicaDrainStatus> wait_replica_drain(
      std::string_view replica_id,
      uint32_t,
      std::optional<std::string_view>) override {
    if (replica_id.empty()) {
      return absl::InvalidArgumentError("replica_id required");
    }
    call_sequence.emplace_back(absl::StrCat("drain:", replica_id));
    drained_replicas.emplace_back(std::string(replica_id));
    components::ReplicaDrainStatus out;
    out.drained = drain_success;
    if (drain_success) {
      out.current_requests = 0;
    } else {
      out.current_requests = drain_current_requests > 0 ? drain_current_requests : 1;
    }
    return out;
  }

  absl::Status update_artifact_view_state(const components::ViewStateUpdate& update) override {
    view_requests.emplace_back(update.view_id);
    view_updates.push_back(update);
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<components::ViewInfo>> list_views(std::string_view) override {
    return view_infos;
  }

  absl::StatusOr<tensorcast::global_store::v1::AssemblyLayoutBinding> get_assembly_layout_binding(
      std::string_view) override {
    return absl::UnimplementedError("get_assembly_layout_binding not supported in RecordingGlobalStoreClient");
  }

  absl::StatusOr<tensorcast::layout::v1::LayoutSpecRecord> get_layout_spec(std::string_view) override {
    return absl::UnimplementedError("get_layout_spec not supported in RecordingGlobalStoreClient");
  }

  absl::Status attach_layout_to_artifact(std::string_view, std::string_view) override {
    return absl::UnimplementedError("attach_layout_to_artifact not supported in RecordingGlobalStoreClient");
  }

  absl::StatusOr<std::vector<std::string>> list_artifact_layouts(std::string_view) override {
    return absl::UnimplementedError("list_artifact_layouts not supported in RecordingGlobalStoreClient");
  }

  absl::StatusOr<tensorcast::global_store::v1::WriteTensorProofCommitmentsResponse> write_tensor_proof_commitments(
      const tensorcast::global_store::v1::WriteTensorProofCommitmentsRequest&) override {
    return absl::UnimplementedError("write_tensor_proof_commitments not supported in RecordingGlobalStoreClient");
  }

  absl::StatusOr<tensorcast::global_store::v1::CheckProofCommitmentsMatchResponse> check_proof_commitments_match(
      const tensorcast::global_store::v1::CheckProofCommitmentsMatchRequest&) override {
    return absl::UnimplementedError("check_proof_commitments_match not supported in RecordingGlobalStoreClient");
  }

  absl::StatusOr<tensorcast::global_store::v1::AssemblyRuntimePolicy> get_assembly_runtime_policy(
      std::string_view) override {
    return absl::UnimplementedError("get_assembly_runtime_policy not supported in RecordingGlobalStoreClient");
  }

  absl::StatusOr<tensorcast::operation::v1::AcquireOperationLeaseResponse> acquire_operation_lease(
      const tensorcast::operation::v1::AcquireOperationLeaseRequest&) override {
    return absl::UnimplementedError("acquire_operation_lease not supported in RecordingGlobalStoreClient");
  }

  absl::StatusOr<tensorcast::operation::v1::KeepaliveOperationLeaseResponse> keepalive_operation_lease(
      const tensorcast::operation::v1::KeepaliveOperationLeaseRequest&) override {
    return absl::UnimplementedError("keepalive_operation_lease not supported in RecordingGlobalStoreClient");
  }

  absl::StatusOr<tensorcast::operation::v1::ReleaseOperationLeaseResponse> release_operation_lease(
      const tensorcast::operation::v1::ReleaseOperationLeaseRequest&) override {
    return absl::UnimplementedError("release_operation_lease not supported in RecordingGlobalStoreClient");
  }

  absl::StatusOr<tensorcast::operation::v1::GetOperationResponse> get_operation(
      const tensorcast::operation::v1::GetOperationRequest&) override {
    return absl::UnimplementedError("get_operation not supported in RecordingGlobalStoreClient");
  }

  absl::Status update_operation(const tensorcast::operation::v1::UpdateOperationRequest&) override {
    return absl::UnimplementedError("update_operation not supported in RecordingGlobalStoreClient");
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
      uint32_t wait_timeout_ms,
      const std::optional<components::TransportSchedulingGroupHint>& scheduling_group,
      std::string_view requester_worker_id,
      std::string_view request_id) override {
    replica_requests.emplace_back(std::string(artifact_id));
    replica_request_groups.push_back(scheduling_group);
    replica_request_request_ids.emplace_back(std::string(request_id));
    replica_request_requester_worker_ids.emplace_back(std::string(requester_worker_id));
    replica_request_wait_timeouts_ms.push_back(wait_timeout_ms);
    if (scripted_transport_next < scripted_transport_sessions.size()) {
      return scripted_transport_sessions[scripted_transport_next++];
    }
    if (replica_transport_not_found) {
      return absl::NotFoundError("replica transport not found in RecordingGlobalStoreClient");
    }
    if (!allow_replica_transport) {
      return absl::UnavailableError("replica transport disabled in RecordingGlobalStoreClient");
    }
    auto it = transport_replicas.find(transport_key(artifact_id, std::nullopt));
    if (it == transport_replicas.end()) {
      return make_transport_session(artifact_id, target_device);
    }
    return make_transport_session(it->second, target_device);
  }

  absl::StatusOr<components::TransportSession> request_view_transport(
      std::string_view artifact_id,
      std::string_view view_id,
      std::string_view,
      std::string_view,
      uint32_t,
      const tensorcast::store::DeviceKey& target_device,
      uint32_t wait_timeout_ms,
      const std::optional<components::TransportSchedulingGroupHint>& scheduling_group,
      std::string_view requester_worker_id,
      std::string_view request_id) override {
    view_requests.emplace_back(std::string(view_id));
    view_request_groups.push_back(scheduling_group);
    view_request_request_ids.emplace_back(std::string(request_id));
    view_request_requester_worker_ids.emplace_back(std::string(requester_worker_id));
    view_request_wait_timeouts_ms.push_back(wait_timeout_ms);
    if (!allow_view_transport) {
      return absl::NotFoundError("view transport disabled in RecordingGlobalStoreClient");
    }
    auto it = transport_replicas.find(transport_key(artifact_id, view_id));
    if (it == transport_replicas.end()) {
      return absl::NotFoundError("view transport not found in RecordingGlobalStoreClient");
    }
    return make_transport_session(it->second, target_device);
  }

  absl::Status complete_replica_transport(
      std::string_view transport_id,
      components::TransportCompletionOutcome outcome,
      std::string_view outcome_detail) override {
    completed_transport_ids.emplace_back(std::string(transport_id));
    completed_transport_outcomes.push_back(outcome);
    completed_transport_outcome_details.emplace_back(std::string(outcome_detail));
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

  absl::StatusOr<components::StateSyncResult> reconcile_worker_state(
      std::string_view,
      std::string_view,
      const std::vector<tensorcast::common::v1::ReplicaInfo>&,
      bool,
      const components::StateSyncToken&,
      const components::RpcOptions&) override {
    return absl::UnimplementedError("reconcile_worker_state not supported in test stub");
  }

  absl::StatusOr<components::AcquireShardHomeLeaseResult> acquire_shard_home_lease(
      uint64_t,
      std::string_view,
      uint64_t,
      const components::RpcOptions&) override {
    return absl::UnimplementedError("acquire_shard_home_lease not supported in test stub");
  }

  absl::StatusOr<components::ShardHomeLeaseDescriptor> keepalive_shard_home_lease(
      std::string_view,
      uint64_t,
      const components::RpcOptions&) override {
    return absl::UnimplementedError("keepalive_shard_home_lease not supported in test stub");
  }

  absl::StatusOr<std::vector<components::ShardHomeLeaseKeepaliveOutcome>> batch_keepalive_shard_home_leases(
      const std::vector<components::ShardHomeLeaseKeepaliveInput>&,
      uint64_t,
      const components::RpcOptions&) override {
    return absl::UnimplementedError("batch_keepalive_shard_home_leases not supported in test stub");
  }

  absl::StatusOr<bool> release_shard_home_lease(std::string_view, const components::RpcOptions&) override {
    return absl::UnimplementedError("release_shard_home_lease not supported in test stub");
  }

  absl::StatusOr<components::ShardHomeRouteInfo> get_shard_home_lease(uint64_t, const components::RpcOptions&)
      override {
    return absl::UnimplementedError("get_shard_home_lease not supported in test stub");
  }

  absl::StatusOr<std::vector<components::ShardHomeRouteInfo>> batch_get_shard_home_leases(
      const std::vector<uint64_t>&,
      const components::RpcOptions&) override {
    return absl::UnimplementedError("batch_get_shard_home_leases not supported in test stub");
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

  absl::Status upsert_key_mapping(std::string_view, std::string_view, absl::Duration) override {
    return absl::UnimplementedError("upsert_key_mapping not supported in test stub");
  }

  absl::StatusOr<std::string> get_cluster_id() override {
    if (cluster_id.empty()) {
      return absl::NotFoundError("cluster_id unavailable");
    }
    return cluster_id;
  }

  absl::Status upsert_artifact_disk_location(
      std::string_view artifact_id,
      std::string_view cluster_id_value,
      std::string_view relative_path,
      tensorcast::global_store::v1::DiskLocationKind kind,
      bool is_deleted = false) override {
    if (fail_disk_location_upsert) {
      return absl::UnavailableError("disk_location_upsert_failed");
    }
    const absl::Time now = absl::Now();
    for (auto& entry : disk_locations) {
      if (entry.artifact_id == artifact_id && entry.cluster_id == cluster_id_value &&
          entry.relative_path == relative_path) {
        entry.kind = kind;
        const bool prior_deleted = entry.is_deleted;
        entry.is_deleted = entry.is_deleted || is_deleted;
        if (!prior_deleted && entry.is_deleted) {
          entry.deleted_at = now;
        }
        entry.updated_at = now;
        return absl::OkStatus();
      }
    }
    components::ArtifactDiskLocation entry;
    entry.artifact_id = std::string(artifact_id);
    entry.cluster_id = std::string(cluster_id_value);
    entry.relative_path = std::string(relative_path);
    entry.kind = kind;
    entry.is_deleted = is_deleted;
    entry.created_at = now;
    entry.updated_at = now;
    if (is_deleted) {
      entry.deleted_at = now;
    }
    disk_locations.push_back(std::move(entry));
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<components::ArtifactDiskLocation>> list_artifact_disk_locations(
      std::string_view artifact_id,
      bool include_deleted = false) override {
    std::vector<components::ArtifactDiskLocation> out;
    for (const auto& entry : disk_locations) {
      if (entry.artifact_id == artifact_id) {
        if (!include_deleted && entry.is_deleted) {
          continue;
        }
        out.push_back(entry);
      }
    }
    if (out.empty()) {
      return absl::NotFoundError("disk_locations_not_found");
    }
    return out;
  }

  absl::StatusOr<components::KeyMappingSwapResult> swap_key_mapping(
      std::string_view,
      std::string_view,
      std::optional<std::string_view>,
      std::optional<uint64_t>) override {
    return absl::UnimplementedError("swap_key_mapping not supported in test stub");
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
  std::string transport_key(std::string_view artifact_id, std::optional<std::string_view> view_id) const {
    std::string key(artifact_id);
    key.push_back('|');
    if (view_id.has_value()) {
      key.append(view_id->data(), view_id->size());
    }
    return key;
  }

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

  components::TransportSession make_transport_session(
      const TransportReplicaInfo& info,
      const tensorcast::store::DeviceKey&) const {
    components::TransportSession session;
    session.transport_id = "test-transport";
    session.remote_replica.node_id = remote_node_id;
    session.remote_replica.node_address = remote_node_address;
    session.remote_replica.node_port = remote_node_port;
    session.remote_replica.memory_size = info.memory_size;
    session.remote_replica.memory_type = info.memory_type;
    session.remote_replica.device_id = info.device_id;
    session.remote_replica.remote_memory_keys = info.remote_memory_keys;
    session.remote_replica.buffer_sizes = info.buffer_sizes;
    session.remote_replica.verification_json = "{}";
    return session;
  }
};

inline std::shared_ptr<components::IGlobalStoreClient> MakeRecordingGlobalStoreClient() {
  return std::make_shared<RecordingGlobalStoreClient>();
}

} // namespace tensorcast::store::testing
