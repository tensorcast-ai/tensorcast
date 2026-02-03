// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "core/store/components/global_store_client.h"

namespace tensorcast::store::testing {

// GlobalStoreClientStub is a test-friendly base that keeps unit tests resilient to
// IGlobalStoreClient interface growth. Tests should derive from this class and
// override only the RPCs they need.
class GlobalStoreClientStub : public components::IGlobalStoreClient {
 public:
  bool connected{true};

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
    return absl::UnimplementedError("register_worker not supported in GlobalStoreClientStub");
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
    return absl::UnimplementedError("send_heartbeat_enhanced not supported in GlobalStoreClientStub");
  }

  absl::Status unregister_worker(std::string_view, bool) override {
    return absl::UnimplementedError("unregister_worker not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<std::string> register_replica(
      std::string_view,
      std::string_view,
      const DeviceKey&,
      common::memory::MemoryLocation,
      uint64_t,
      uint32_t,
      std::optional<std::string_view>) override {
    return absl::UnimplementedError("register_replica not supported in GlobalStoreClientStub");
  }

  absl::Status record_view_residency(std::string_view, std::string_view, uint64_t, std::optional<std::string_view>)
      override {
    return absl::UnimplementedError("record_view_residency not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<std::string> register_memory_replica(
      std::string_view,
      std::string_view,
      const DeviceKey&,
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
    return absl::UnimplementedError("register_memory_replica not supported in GlobalStoreClientStub");
  }

  absl::Status unregister_replica(std::string_view, std::string_view) override {
    return absl::UnimplementedError("unregister_replica not supported in GlobalStoreClientStub");
  }

  absl::Status unregister_replica_by_worker(
      std::string_view,
      std::string_view,
      std::optional<common::memory::MemoryLocation>,
      std::optional<uint32_t>) override {
    return absl::UnimplementedError("unregister_replica_by_worker not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<bool> mark_replica_unavailable(
      std::string_view,
      std::string_view,
      std::optional<std::string_view>,
      std::optional<std::string_view>) override {
    return absl::UnimplementedError("mark_replica_unavailable not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<components::ReplicaDrainStatus> wait_replica_drain(
      std::string_view,
      uint32_t,
      std::optional<std::string_view>) override {
    return absl::UnimplementedError("wait_replica_drain not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<components::TransportSession> request_replica_transport(
      std::string_view,
      std::string_view,
      std::string_view,
      uint32_t,
      const DeviceKey&,
      uint32_t) override {
    return absl::UnimplementedError("request_replica_transport not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<components::TransportSession> request_view_transport(
      std::string_view,
      std::string_view,
      std::string_view,
      std::string_view,
      uint32_t,
      const DeviceKey&,
      uint32_t) override {
    return absl::UnimplementedError("request_view_transport not supported in GlobalStoreClientStub");
  }

  absl::Status complete_replica_transport(std::string_view) override {
    return absl::UnimplementedError("complete_replica_transport not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<std::vector<components::RemoteReplicaInfo>> get_artifact_replicas(
      std::string_view,
      std::optional<std::string_view>) override {
    return absl::UnimplementedError("get_artifact_replicas not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<std::vector<components::ChunkLocationInfo>> query_chunk_locations(
      std::string_view,
      const std::vector<uint32_t>&) override {
    return absl::UnimplementedError("query_chunk_locations not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<components::StateSyncResult> synchronize_worker_state(
      const tensorcast::global_store::v1::WorkerLocalState&,
      bool,
      const components::StateSyncToken&,
      const components::RpcOptions&) override {
    return absl::UnimplementedError("synchronize_worker_state not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<components::FullStateSyncResult> request_full_state_sync(
      std::string_view,
      uint64_t,
      const components::StateSyncToken&,
      const components::RpcOptions&) override {
    return absl::UnimplementedError("request_full_state_sync not supported in GlobalStoreClientStub");
  }

  bool is_connected() const override {
    return connected;
  }

  absl::Status batch_update_chunk_states(
      std::string_view,
      std::string_view,
      const std::vector<components::ChunkStateUpdate>&) override {
    return absl::UnimplementedError("batch_update_chunk_states not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<components::KeyMapping> resolve_key_mapping(std::string_view) override {
    return absl::UnimplementedError("resolve_key_mapping not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<std::string> get_artifact_index_by_id(std::string_view) override {
    return absl::UnimplementedError("get_artifact_index_by_id not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<components::ViewMetadata> get_view_metadata(std::string_view, std::string_view) override {
    return absl::UnimplementedError("get_view_metadata not supported in GlobalStoreClientStub");
  }

  absl::Status upsert_key_mapping(std::string_view, std::string_view, std::string_view, absl::Duration) override {
    return absl::UnimplementedError("upsert_key_mapping not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<components::KeyMappingSwapResult> swap_key_mapping(
      std::string_view,
      std::string_view,
      std::optional<std::string_view>,
      std::optional<uint64_t>) override {
    return absl::UnimplementedError("swap_key_mapping not supported in GlobalStoreClientStub");
  }

  absl::Status revoke_key_mapping(std::string_view) override {
    return absl::UnimplementedError("revoke_key_mapping not supported in GlobalStoreClientStub");
  }

  void update_local_endpoint(std::string, std::string, uint32_t, uint32_t) override {}

  absl::Status update_artifact_view_state(const components::ViewStateUpdate&) override {
    return absl::UnimplementedError("update_artifact_view_state not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<std::vector<components::ViewInfo>> list_views(std::string_view) override {
    return absl::UnimplementedError("list_views not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<tensorcast::global_store::v1::AssemblyLayoutBinding> get_assembly_layout_binding(
      std::string_view) override {
    return absl::UnimplementedError("get_assembly_layout_binding not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<tensorcast::layout::v1::LayoutSpecRecord> get_layout_spec(std::string_view) override {
    return absl::UnimplementedError("get_layout_spec not supported in GlobalStoreClientStub");
  }

  absl::Status attach_layout_to_artifact(std::string_view, std::string_view) override {
    return absl::UnimplementedError("attach_layout_to_artifact not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<std::vector<std::string>> list_artifact_layouts(std::string_view) override {
    return absl::UnimplementedError("list_artifact_layouts not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<tensorcast::global_store::v1::WriteTensorProofCommitmentsResponse> write_tensor_proof_commitments(
      const tensorcast::global_store::v1::WriteTensorProofCommitmentsRequest&) override {
    return absl::UnimplementedError("write_tensor_proof_commitments not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<tensorcast::global_store::v1::CheckProofCommitmentsMatchResponse> check_proof_commitments_match(
      const tensorcast::global_store::v1::CheckProofCommitmentsMatchRequest&) override {
    return absl::UnimplementedError("check_proof_commitments_match not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<tensorcast::global_store::v1::AssemblyRuntimePolicy> get_assembly_runtime_policy(
      std::string_view) override {
    return absl::UnimplementedError("get_assembly_runtime_policy not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<tensorcast::operation::v1::AcquireOperationLeaseResponse> acquire_operation_lease(
      const tensorcast::operation::v1::AcquireOperationLeaseRequest&) override {
    return absl::UnimplementedError("acquire_operation_lease not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<tensorcast::operation::v1::KeepaliveOperationLeaseResponse> keepalive_operation_lease(
      const tensorcast::operation::v1::KeepaliveOperationLeaseRequest&) override {
    return absl::UnimplementedError("keepalive_operation_lease not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<tensorcast::operation::v1::ReleaseOperationLeaseResponse> release_operation_lease(
      const tensorcast::operation::v1::ReleaseOperationLeaseRequest&) override {
    return absl::UnimplementedError("release_operation_lease not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<tensorcast::operation::v1::GetOperationResponse> get_operation(
      const tensorcast::operation::v1::GetOperationRequest&) override {
    return absl::UnimplementedError("get_operation not supported in GlobalStoreClientStub");
  }

  absl::Status update_operation(const tensorcast::operation::v1::UpdateOperationRequest&) override {
    return absl::UnimplementedError("update_operation not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<components::ArtifactBinding> get_artifact_binding(std::string_view) override {
    return absl::UnimplementedError("get_artifact_binding not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<components::ArtifactBindingResult> upsert_artifact_binding(
      const components::ArtifactBinding&) override {
    return absl::UnimplementedError("upsert_artifact_binding not supported in GlobalStoreClientStub");
  }

  absl::StatusOr<components::PlacementPlanResult> plan_placement(
      std::string_view,
      tensorcast::global_store::v1::PlacementPolicy,
      const std::vector<components::PlacementShardSpec>&,
      std::string_view) override {
    return absl::UnimplementedError("plan_placement not supported in GlobalStoreClientStub");
  }

  absl::Status report_persistence_status(const components::PersistenceReport&) override {
    return absl::UnimplementedError("report_persistence_status not supported in GlobalStoreClientStub");
  }
};

} // namespace tensorcast::store::testing
