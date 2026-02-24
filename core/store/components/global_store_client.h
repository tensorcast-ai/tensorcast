// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "core/common/memory/memory_location.h"
#include "core/store/device_types.h"
#include "core/store/replica/chunk_state.h"
#include "grpcpp/grpcpp.h"
#include "gsl/pointers"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/global_store/v1/global_store.grpc.pb.h"
#include "tensorcast/global_store/v1/global_store.pb.h"
#include "tensorcast/layout/v1/layout.pb.h"
#include "tensorcast/memory_tier/v1/memory_tier.grpc.pb.h"
#include "tensorcast/operation/v1/operation.pb.h"

namespace tensorcast::store::components {

namespace global_store = tensorcast::global_store::v1;
namespace layout = tensorcast::layout::v1;
namespace operation = tensorcast::operation::v1;

// Configuration for Global Store Client
struct GlobalStoreClientConfig {
  std::string global_store_address = "localhost:50051";
  absl::Duration connection_timeout = absl::Seconds(30);
  absl::Duration rpc_timeout = absl::Seconds(60);
  uint32_t max_retries = 3;
  absl::Duration retry_backoff = absl::Milliseconds(100);
  std::string cluster_token;
};

struct RpcOptions {
  std::optional<absl::Duration> timeout;
  std::optional<uint32_t> max_retries;
  std::optional<absl::Duration> retry_backoff;
  // Optional cancellation predicate inherited from upstream callers.
  // When provided, RPC retries/backoff should stop as soon as it returns true.
  std::function<bool()> cancel_check;
};

struct StateSyncToken {
  uint64_t generation{0};
  uint64_t request_seq{0};
};

enum class ReconcileResultKind {
  kUnspecified = 0,
  kApplied,
  kNoop,
  kIgnoredStale,
  kRetryLater,
  kRebaseRequired,
  kFatal,
};

struct StateSyncResult {
  ReconcileResultKind result_kind{ReconcileResultKind::kUnspecified};
  uint32_t retry_after_ms{0};
  uint64_t new_state_version{0};
  std::string new_state_checksum;
  std::vector<global_store::StateChange> state_changes;
  std::vector<common::v1::ReplicaInfo> expected_replicas;
};

struct WorkerRegistrationInfo {
  std::string worker_id;
  uint64_t reconcile_generation{1};
  uint64_t expected_state_version{0};
  bool state_sync_required{false};
  uint32_t heartbeat_interval_ms{0};
};

// Information about a remote replica replica
struct RemoteReplicaInfo {
  std::string node_id;
  std::string node_address;
  uint32_t node_port;
  uint64_t memory_size;
  common::memory::MemoryLocation memory_type;
  uint32_t device_id;
  std::vector<std::string> remote_memory_keys;
  std::vector<uint64_t> buffer_sizes;
  std::string verification_json; // optional verification metadata (JSON)
  std::optional<std::string> view_id;
};

// Transport session for P2P transfers
struct TransportSession {
  std::string transport_id;
  RemoteReplicaInfo remote_replica;
  absl::Time start_time;
};

struct ReplicaDrainStatus {
  bool drained{false};
  uint32_t current_requests{0};
  std::optional<uint64_t> oldest_transport_age_ms;
};

struct ChunkLocationInfo {
  uint32_t chunk_idx;
  std::string node_id;
  std::string node_address;
  uint32_t p2p_port;
  replica::ChunkState state;
  float node_load_ratio;
  std::string device_uuid;
  uint32_t replica;
};

struct ChunkStateUpdate {
  std::string artifact_id;
  uint32_t chunk_idx{0};
  replica::ChunkState state{replica::ChunkState::COLD};
  std::string device_uuid;
  uint32_t replica{0};
};

struct KeyMapping {
  std::string artifact_id;
  std::string replica_uuid;
  std::string daemon_address;
  uint64_t generation{0};
  uint32_t cache_ttl_seconds{0};
};

struct ArtifactDiskLocation {
  std::string artifact_id;
  std::string cluster_id;
  std::string relative_path;
  global_store::DiskLocationKind kind{global_store::DISK_LOCATION_KIND_UNSPECIFIED};
  bool is_deleted{false};
  std::optional<absl::Time> created_at;
  std::optional<absl::Time> updated_at;
  std::optional<absl::Time> deleted_at;
};

struct KeyMappingSwapResult {
  bool ok{false};
  std::string artifact_id;
  uint64_t generation{0};
};

struct CanonicalRange {
  uint64_t offset{0};
  uint64_t length{0};
};

struct ViewStateUpdate {
  std::string artifact_id;
  std::string view_id;
  std::string view_spec_json;
  uint64_t view_size_bytes{0};
  std::optional<std::string> view_data_hash;
  bool mark_verified{false};
  uint64_t canonical_size_bytes{0};
  uint64_t canonical_bytes_covered{0};
  std::vector<CanonicalRange> canonical_ranges;
  std::vector<global_store::LeafWrite> leaf_writes;
  std::vector<global_store::PieceProofDigestWrite> proof_digests;
};

struct ViewInfo {
  std::string view_id;
  std::string view_spec_json;
  uint64_t view_size_bytes{0};
  std::optional<std::string> view_data_hash;
  std::optional<absl::Time> verified_at;
  uint64_t canonical_size_bytes{0};
  uint64_t canonical_bytes_covered{0};
  std::vector<CanonicalRange> canonical_ranges;
};

struct ArtifactBinding {
  std::string from_artifact_id;
  std::string to_artifact_id;
  global_store::ArtifactBindingKind kind{global_store::ARTIFACT_BINDING_KIND_UNSPECIFIED};
  std::optional<absl::Time> created_at;
};

struct ArtifactBindingResult {
  ArtifactBinding binding;
  bool created{false};
};

struct ViewMetadata {
  std::string view_spec_json;
  uint64_t view_size_bytes{0};
  std::optional<std::string> view_data_hash;
};

enum class MemoryTierLeaseKind { kStable, kPreemptible };
enum class MemoryTierLeaseState { kPending, kActive, kRevoking, kExpired };
enum class MemoryTierAckAction { kAcquired, kReleased };

struct MemoryTierStatusPayload {
  std::string node_id;
  std::string worker_id;
  uint64_t stable_total_bytes{0};
  uint64_t stable_used_bytes{0};
  uint64_t preemptible_total_bytes{0};
  uint64_t preemptible_marked_bytes{0};
  double faults_per_sec{0.0};
  uint64_t rehydrate_p99_ns{0};
  bool enable_preemptible{false};
  std::string memory_tier_config_json;
  uint64_t epoch_ns{0};
};

struct MemoryTierLeaseDescriptor {
  std::string lease_id;
  std::string node_id;
  MemoryTierLeaseKind kind{MemoryTierLeaseKind::kStable};
  std::string artifact_id;
  uint32_t chunk_start{0};
  uint32_t chunk_count{0};
  std::vector<uint32_t> chunk_ids;
  uint64_t ledger_version{0};
  uint64_t bytes{0};
  std::string workload_id;
  MemoryTierLeaseState state{MemoryTierLeaseState::kPending};
  std::string request_id;
  uint64_t ack_epoch_ns{0};
  uint64_t issued_at_ns{0};
  uint64_t expires_at_ns{0};
};

struct MemoryTierLeaseAckPayload {
  std::string lease_id;
  std::string node_id;
  MemoryTierAckAction action{MemoryTierAckAction::kAcquired};
  std::string artifact_id;
  std::vector<uint32_t> chunk_ids;
  uint32_t chunk_start{0};
  uint32_t chunk_count{0};
  uint64_t ledger_version{0};
  uint64_t bytes{0};
  std::string request_id;
  uint64_t ack_epoch_ns{0};
};

struct PlacementShardSpec {
  std::string shard_id;
  uint32_t shard_idx{0};
  uint64_t size_bytes{0};
  std::string content_digest;
  uint64_t byte_range_start{0};
  uint64_t byte_range_length{0};
  std::vector<uint64_t> chunk_ids;
};

struct PlacementTargetStatus {
  std::string node_id;
  std::string lease_id;
  global_store::PlacementTargetState target_state{global_store::PLACEMENT_TARGET_STATE_PENDING};
  std::string degraded_reason;
};

struct ShardPlacement {
  PlacementShardSpec shard;
  std::vector<PlacementTargetStatus> targets;
  std::string degraded_reason;
};

struct PlacementPlanResult {
  std::string plan_id;
  global_store::PlacementPolicy effective_policy{global_store::PLACEMENT_POLICY_UNSPECIFIED};
  bool degraded{false};
  std::string degraded_reason;
  std::vector<ShardPlacement> placements;
};

struct PersistenceShardReport {
  std::string shard_id;
  uint32_t shard_idx{0};
  global_store::PersistenceState state{global_store::PERSISTENCE_STATE_PENDING};
  double progress{0.0};
  std::string degraded_reason;
  std::string last_error;
  std::vector<PlacementTargetStatus> targets;
};

struct PersistenceReport {
  std::string task_id;
  std::string artifact_id;
  std::string plan_id;
  global_store::PersistenceState state{global_store::PERSISTENCE_STATE_PENDING};
  double progress{0.0};
  std::string last_error;
  std::string degraded_reason;
  std::vector<PersistenceShardReport> shards;
};

class IGlobalStoreClient {
 public:
  virtual ~IGlobalStoreClient() = default;

  virtual absl::Status initialize() = 0;

  virtual absl::StatusOr<WorkerRegistrationInfo> register_worker(
      std::string_view node_id,
      std::string_view node_address,
      uint32_t grpc_port,
      uint32_t p2p_port,
      uint64_t mem_pool_total_size,
      uint64_t mem_pool_available_size,
      bool is_recovery_registration = false,
      std::string_view previous_worker_id = {},
      std::string_view daemon_id = {},
      uint64_t capability_flags = 0) = 0;

  virtual absl::StatusOr<global_store::WorkerHeartbeatResponse> send_heartbeat_enhanced(
      std::string_view worker_id,
      uint64_t mem_pool_available_size,
      bool accepting_new_requests,
      uint64_t state_version,
      std::string_view state_checksum,
      const std::vector<std::string>& registered_artifact_ids,
      int64_t last_successful_sync,
      global_store::ConnectionStatus connection_status = global_store::CONNECTION_STATUS_CONNECTED,
      const RpcOptions& rpc_options = RpcOptions{},
      std::string_view daemon_id = {},
      uint64_t capability_flags = 0) = 0;

  virtual absl::Status unregister_worker(std::string_view worker_id, bool is_graceful_shutdown = true) = 0;

  virtual absl::Status unregister_worker_idempotent(
      std::string_view worker_id,
      bool is_graceful_shutdown = true,
      std::optional<std::string_view> client_request_id = std::nullopt) {
    (void)client_request_id;
    return unregister_worker(worker_id, is_graceful_shutdown);
  }

  virtual absl::StatusOr<std::string> register_replica(
      std::string_view artifact_id,
      std::string_view worker_id,
      const DeviceKey& device,
      common::memory::MemoryLocation location,
      uint64_t memory_size,
      uint32_t max_concurrency = 1,
      std::optional<std::string_view> view_id = std::nullopt) = 0;

  virtual absl::StatusOr<std::string> register_replica_idempotent(
      std::string_view artifact_id,
      std::string_view worker_id,
      const DeviceKey& device,
      common::memory::MemoryLocation location,
      uint64_t memory_size,
      uint32_t max_concurrency = 1,
      std::optional<std::string_view> view_id = std::nullopt,
      std::optional<std::string_view> client_request_id = std::nullopt) {
    (void)client_request_id;
    return register_replica(artifact_id, worker_id, device, location, memory_size, max_concurrency, view_id);
  }

  virtual absl::Status record_view_residency(
      std::string_view canonical_artifact_id,
      std::string_view view_id,
      uint64_t view_size_bytes,
      std::optional<std::string_view> view_data_hash = std::nullopt) = 0;

  virtual absl::StatusOr<std::string> register_memory_replica(
      std::string_view artifact_id,
      std::string_view worker_id,
      const DeviceKey& device,
      uint64_t memory_size,
      std::string_view tensor_index_key,
      const std::vector<std::string>& remote_memory_keys,
      const std::vector<uint64_t>& buffer_sizes,
      const std::optional<std::string>& tensor_index_data = std::nullopt,
      std::string_view encoding = "json",
      std::string_view schema_version = "v3",
      uint32_t max_concurrency = 1,
      const std::optional<std::string>& verification_json = std::nullopt,
      std::optional<std::string_view> view_id = std::nullopt,
      const std::optional<common::v1::ArtifactDescriptor>& descriptor = std::nullopt) = 0;

  virtual absl::StatusOr<std::string> register_memory_replica_idempotent(
      std::string_view artifact_id,
      std::string_view worker_id,
      const DeviceKey& device,
      uint64_t memory_size,
      std::string_view tensor_index_key,
      const std::vector<std::string>& remote_memory_keys,
      const std::vector<uint64_t>& buffer_sizes,
      const std::optional<std::string>& tensor_index_data = std::nullopt,
      std::string_view encoding = "json",
      std::string_view schema_version = "v3",
      uint32_t max_concurrency = 1,
      const std::optional<std::string>& verification_json = std::nullopt,
      std::optional<std::string_view> view_id = std::nullopt,
      const std::optional<common::v1::ArtifactDescriptor>& descriptor = std::nullopt,
      std::optional<std::string_view> client_request_id = std::nullopt) {
    (void)client_request_id;
    return register_memory_replica(
        artifact_id,
        worker_id,
        device,
        memory_size,
        tensor_index_key,
        remote_memory_keys,
        buffer_sizes,
        tensor_index_data,
        encoding,
        schema_version,
        max_concurrency,
        verification_json,
        view_id,
        descriptor);
  }

  virtual absl::Status unregister_replica(std::string_view artifact_id, std::string_view replica_id) = 0;

  // Deregister a replica using worker identity and optional selectors.
  virtual absl::Status unregister_replica_by_worker(
      std::string_view artifact_id,
      std::string_view worker_id,
      std::optional<common::memory::MemoryLocation> memory_type = std::nullopt,
      std::optional<uint32_t> device_id = std::nullopt) = 0;

  virtual absl::StatusOr<bool> mark_replica_unavailable(
      std::string_view artifact_id,
      std::string_view replica_id,
      std::optional<std::string_view> reason = std::nullopt,
      std::optional<std::string_view> operation_id = std::nullopt) = 0;

  virtual absl::StatusOr<ReplicaDrainStatus> wait_replica_drain(
      std::string_view replica_id,
      uint32_t timeout_ms,
      std::optional<std::string_view> operation_id = std::nullopt) = 0;

  virtual absl::StatusOr<TransportSession> request_replica_transport(
      std::string_view artifact_id,
      std::string_view source_node_id,
      std::string_view source_address,
      uint32_t source_port,
      const DeviceKey& target_device,
      uint32_t wait_timeout_ms = 30000) = 0;

  virtual absl::StatusOr<TransportSession> request_view_transport(
      std::string_view artifact_id,
      std::string_view view_id,
      std::string_view source_node_id,
      std::string_view source_address,
      uint32_t source_port,
      const DeviceKey& target_device,
      uint32_t wait_timeout_ms = 30000) = 0;

  virtual absl::Status complete_replica_transport(std::string_view transport_id) = 0;

  virtual absl::StatusOr<std::vector<RemoteReplicaInfo>> get_artifact_replicas(
      std::string_view artifact_id,
      std::optional<std::string_view> view_id = std::nullopt) = 0;

  virtual absl::StatusOr<std::vector<ChunkLocationInfo>> query_chunk_locations(
      std::string_view artifact_id,
      const std::vector<uint32_t>& chunk_indices) = 0;

  virtual absl::StatusOr<StateSyncResult> reconcile_worker_state(
      std::string_view worker_id,
      std::string_view daemon_id,
      const std::vector<common::v1::ReplicaInfo>& inventory,
      bool snapshot_request,
      const StateSyncToken& token,
      const RpcOptions& rpc_options = RpcOptions{}) = 0;

  virtual bool is_connected() const = 0;

  virtual absl::Status batch_update_chunk_states(
      std::string_view worker_id,
      std::string_view node_id,
      const std::vector<ChunkStateUpdate>& updates) = 0;

  virtual absl::StatusOr<KeyMapping> resolve_key_mapping(std::string_view key) = 0;

  virtual absl::StatusOr<KeyMapping> resolve_key_mapping_with_options(
      std::string_view key,
      const RpcOptions& rpc_options) {
    (void)rpc_options;
    return resolve_key_mapping(key);
  }

  virtual absl::StatusOr<std::string> get_artifact_index_by_id(std::string_view artifact_id) = 0;
  virtual absl::StatusOr<ViewMetadata> get_view_metadata(std::string_view artifact_id, std::string_view view_id) = 0;

  virtual absl::Status upsert_key_mapping(std::string_view key, std::string_view artifact_id, absl::Duration ttl) = 0;

  virtual absl::StatusOr<KeyMappingSwapResult> swap_key_mapping(
      std::string_view key,
      std::string_view new_artifact_id,
      std::optional<std::string_view> expected_artifact_id,
      std::optional<uint64_t> expected_generation) = 0;

  virtual absl::Status revoke_key_mapping(std::string_view key) = 0;

  virtual absl::StatusOr<std::string> get_cluster_id() = 0;

  virtual absl::Status upsert_artifact_disk_location(
      std::string_view artifact_id,
      std::string_view cluster_id,
      std::string_view relative_path,
      global_store::DiskLocationKind kind,
      bool is_deleted = false) = 0;

  virtual absl::StatusOr<std::vector<ArtifactDiskLocation>> list_artifact_disk_locations(
      std::string_view artifact_id,
      bool include_deleted = false) = 0;

  // Memory tier RPCs (optional; default implementations return Unimplemented)
  virtual absl::Status publish_memory_tier_status(const MemoryTierStatusPayload& status) {
    return absl::UnimplementedError("MemoryTierService not available");
  }

  virtual absl::StatusOr<MemoryTierLeaseDescriptor> request_memory_tier_lease(
      const MemoryTierLeaseDescriptor& request) {
    return absl::UnimplementedError("MemoryTierService not available");
  }

  virtual absl::StatusOr<MemoryTierLeaseDescriptor> acknowledge_memory_tier_lease(
      const MemoryTierLeaseAckPayload& ack) {
    return absl::UnimplementedError("MemoryTierService not available");
  }

  virtual absl::StatusOr<std::vector<MemoryTierLeaseDescriptor>> list_memory_tier_leases(std::string_view node_id) {
    return absl::UnimplementedError("MemoryTierService not available");
  }

  virtual absl::StatusOr<MemoryTierLeaseDescriptor> revoke_memory_tier_lease(std::string_view lease_id) {
    return absl::UnimplementedError("MemoryTierService not available");
  }

  virtual void update_local_endpoint(
      std::string node_id,
      std::string node_address,
      uint32_t grpc_port,
      uint32_t p2p_port) = 0;

  virtual absl::Status update_artifact_view_state(const ViewStateUpdate& update) = 0;

  virtual absl::StatusOr<std::vector<ViewInfo>> list_views(std::string_view artifact_id) = 0;

  // ========== Layout v2 ==========
  virtual absl::StatusOr<global_store::AssemblyLayoutBinding> get_assembly_layout_binding(
      std::string_view assembly_id) = 0;
  virtual absl::StatusOr<layout::LayoutSpecRecord> get_layout_spec(std::string_view layout_id) = 0;
  virtual absl::Status attach_layout_to_artifact(std::string_view mi2_id, std::string_view layout_id) = 0;
  virtual absl::StatusOr<std::vector<std::string>> list_artifact_layouts(std::string_view mi2_id) = 0;
  virtual absl::StatusOr<global_store::WriteTensorProofCommitmentsResponse> write_tensor_proof_commitments(
      const global_store::WriteTensorProofCommitmentsRequest& request) = 0;
  virtual absl::StatusOr<global_store::CheckProofCommitmentsMatchResponse> check_proof_commitments_match(
      const global_store::CheckProofCommitmentsMatchRequest& request) = 0;

  virtual absl::StatusOr<global_store::AssemblyRuntimePolicy> get_assembly_runtime_policy(
      std::string_view assembly_id) = 0;

  // ========== Unified Operations ==========
  virtual absl::StatusOr<operation::AcquireOperationLeaseResponse> acquire_operation_lease(
      const operation::AcquireOperationLeaseRequest& request) = 0;
  virtual absl::StatusOr<operation::KeepaliveOperationLeaseResponse> keepalive_operation_lease(
      const operation::KeepaliveOperationLeaseRequest& request) = 0;
  virtual absl::StatusOr<operation::ReleaseOperationLeaseResponse> release_operation_lease(
      const operation::ReleaseOperationLeaseRequest& request) = 0;
  virtual absl::StatusOr<operation::GetOperationResponse> get_operation(
      const operation::GetOperationRequest& request) = 0;
  virtual absl::Status update_operation(const operation::UpdateOperationRequest& request) = 0;

  virtual absl::StatusOr<ArtifactBinding> get_artifact_binding(std::string_view artifact_id) = 0;

  virtual absl::StatusOr<ArtifactBindingResult> upsert_artifact_binding(const ArtifactBinding& binding) = 0;

  virtual absl::StatusOr<PlacementPlanResult> plan_placement(
      std::string_view artifact_id,
      global_store::PlacementPolicy policy,
      const std::vector<PlacementShardSpec>& shards,
      std::string_view source_node_id) = 0;

  virtual absl::Status report_persistence_status(const PersistenceReport& report) = 0;
};

class TransportLease {
 public:
  TransportLease() = default;
  TransportLease(IGlobalStoreClient* client, std::string transport_id);

  TransportLease(const TransportLease&) = delete;
  TransportLease& operator=(const TransportLease&) = delete;

  TransportLease(TransportLease&& other) noexcept;
  TransportLease& operator=(TransportLease&& other) noexcept;

  ~TransportLease();

  const std::string& transport_id() const {
    return transport_id_;
  }

  bool is_active() const {
    return client_ != nullptr && !transport_id_.empty();
  }

  void release();

 private:
  void complete();

  IGlobalStoreClient* client_{nullptr};
  std::string transport_id_;
};

class GlobalStoreClient : public IGlobalStoreClient {
 public:
  explicit GlobalStoreClient(GlobalStoreClientConfig config);
  ~GlobalStoreClient() override;

  // Initialize connection to Global Store
  absl::Status initialize() override;

  // Worker registration and lifecycle
  absl::StatusOr<WorkerRegistrationInfo> register_worker(
      std::string_view node_id,
      std::string_view node_address,
      uint32_t grpc_port,
      uint32_t p2p_port,
      uint64_t mem_pool_total_size,
      uint64_t mem_pool_available_size,
      bool is_recovery_registration = false,
      std::string_view previous_worker_id = {},
      std::string_view daemon_id = {},
      uint64_t capability_flags = 0) override;

  // Enhanced heartbeat with HA state fields
  absl::StatusOr<global_store::WorkerHeartbeatResponse> send_heartbeat_enhanced(
      std::string_view worker_id,
      uint64_t mem_pool_available_size,
      bool accepting_new_requests,
      uint64_t state_version,
      std::string_view state_checksum,
      const std::vector<std::string>& registered_artifact_ids,
      int64_t last_successful_sync,
      global_store::ConnectionStatus connection_status = global_store::CONNECTION_STATUS_CONNECTED,
      const RpcOptions& rpc_options = RpcOptions{},
      std::string_view daemon_id = {},
      uint64_t capability_flags = 0) override;

  absl::Status unregister_worker(std::string_view worker_id, bool is_graceful_shutdown = true) override;
  absl::Status unregister_worker_idempotent(
      std::string_view worker_id,
      bool is_graceful_shutdown = true,
      std::optional<std::string_view> client_request_id = std::nullopt) override;

  // Replica management
  absl::StatusOr<std::string> register_replica(
      std::string_view artifact_id,
      std::string_view worker_id,
      const DeviceKey& device,
      common::memory::MemoryLocation location,
      uint64_t memory_size,
      uint32_t max_concurrency = 1,
      std::optional<std::string_view> view_id = std::nullopt) override;
  absl::StatusOr<std::string> register_replica_idempotent(
      std::string_view artifact_id,
      std::string_view worker_id,
      const DeviceKey& device,
      common::memory::MemoryLocation location,
      uint64_t memory_size,
      uint32_t max_concurrency = 1,
      std::optional<std::string_view> view_id = std::nullopt,
      std::optional<std::string_view> client_request_id = std::nullopt) override;

  // Record metadata for a view replica while keeping canonical routing unchanged.
  // This is a placeholder that will be backed by a dedicated Global Store RPC.
  absl::Status record_view_residency(
      std::string_view canonical_artifact_id,
      std::string_view view_id,
      uint64_t view_size_bytes,
      std::optional<std::string_view> view_data_hash = std::nullopt) override;

  // Register a GPU memory replica (in-memory tensor dict) with tensor index key.
  absl::StatusOr<std::string> register_memory_replica(
      std::string_view artifact_id,
      std::string_view worker_id,
      const DeviceKey& device,
      uint64_t memory_size,
      std::string_view tensor_index_key,
      const std::vector<std::string>& remote_memory_keys,
      const std::vector<uint64_t>& buffer_sizes,
      const std::optional<std::string>& tensor_index_data = std::nullopt,
      std::string_view encoding = "json",
      std::string_view schema_version = "v3",
      uint32_t max_concurrency = 1,
      const std::optional<std::string>& verification_json = std::nullopt,
      std::optional<std::string_view> view_id = std::nullopt,
      const std::optional<common::v1::ArtifactDescriptor>& descriptor = std::nullopt) override;
  absl::StatusOr<std::string> register_memory_replica_idempotent(
      std::string_view artifact_id,
      std::string_view worker_id,
      const DeviceKey& device,
      uint64_t memory_size,
      std::string_view tensor_index_key,
      const std::vector<std::string>& remote_memory_keys,
      const std::vector<uint64_t>& buffer_sizes,
      const std::optional<std::string>& tensor_index_data = std::nullopt,
      std::string_view encoding = "json",
      std::string_view schema_version = "v3",
      uint32_t max_concurrency = 1,
      const std::optional<std::string>& verification_json = std::nullopt,
      std::optional<std::string_view> view_id = std::nullopt,
      const std::optional<common::v1::ArtifactDescriptor>& descriptor = std::nullopt,
      std::optional<std::string_view> client_request_id = std::nullopt) override;

  absl::Status unregister_replica(std::string_view artifact_id, std::string_view replica_id) override;

  absl::Status unregister_replica_by_worker(
      std::string_view artifact_id,
      std::string_view worker_id,
      std::optional<common::memory::MemoryLocation> memory_type = std::nullopt,
      std::optional<uint32_t> device_id = std::nullopt) override;

  absl::StatusOr<bool> mark_replica_unavailable(
      std::string_view artifact_id,
      std::string_view replica_id,
      std::optional<std::string_view> reason = std::nullopt,
      std::optional<std::string_view> operation_id = std::nullopt) override;

  absl::StatusOr<ReplicaDrainStatus> wait_replica_drain(
      std::string_view replica_id,
      uint32_t timeout_ms,
      std::optional<std::string_view> operation_id = std::nullopt) override;

  // P2P transport coordination
  absl::StatusOr<TransportSession> request_replica_transport(
      std::string_view artifact_id,
      std::string_view source_node_id,
      std::string_view source_address,
      uint32_t source_port,
      const DeviceKey& target_device,
      uint32_t wait_timeout_ms = 30000) override;
  absl::StatusOr<TransportSession> request_view_transport(
      std::string_view artifact_id,
      std::string_view view_id,
      std::string_view source_node_id,
      std::string_view source_address,
      uint32_t source_port,
      const DeviceKey& target_device,
      uint32_t wait_timeout_ms = 30000) override;

  absl::Status complete_replica_transport(std::string_view transport_id) override;

  absl::StatusOr<std::vector<RemoteReplicaInfo>> get_artifact_replicas(
      std::string_view artifact_id,
      std::optional<std::string_view> view_id = std::nullopt) override;

  absl::StatusOr<std::vector<ChunkLocationInfo>> query_chunk_locations(
      std::string_view artifact_id,
      const std::vector<uint32_t>& chunk_indices) override;

  // HA State Synchronization
  absl::StatusOr<StateSyncResult> reconcile_worker_state(
      std::string_view worker_id,
      std::string_view daemon_id,
      const std::vector<common::v1::ReplicaInfo>& inventory,
      bool snapshot_request,
      const StateSyncToken& token,
      const RpcOptions& rpc_options = RpcOptions{}) override;

  bool is_connected() const override;

  absl::Status batch_update_chunk_states(
      std::string_view worker_id,
      std::string_view node_id,
      const std::vector<ChunkStateUpdate>& updates) override;

  absl::StatusOr<KeyMapping> resolve_key_mapping(std::string_view key) override;
  absl::StatusOr<KeyMapping> resolve_key_mapping_with_options(std::string_view key, const RpcOptions& rpc_options)
      override;

  absl::Status upsert_key_mapping(
      std::string_view key,
      std::string_view artifact_id,
      absl::Duration ttl = absl::ZeroDuration()) override;

  absl::StatusOr<KeyMappingSwapResult> swap_key_mapping(
      std::string_view key,
      std::string_view new_artifact_id,
      std::optional<std::string_view> expected_artifact_id,
      std::optional<uint64_t> expected_generation) override;

  absl::Status revoke_key_mapping(std::string_view key) override;

  absl::StatusOr<std::string> get_cluster_id() override;

  absl::Status upsert_artifact_disk_location(
      std::string_view artifact_id,
      std::string_view cluster_id,
      std::string_view relative_path,
      global_store::DiskLocationKind kind,
      bool is_deleted = false) override;

  absl::StatusOr<std::vector<ArtifactDiskLocation>> list_artifact_disk_locations(
      std::string_view artifact_id,
      bool include_deleted = false) override;

  absl::Status publish_memory_tier_status(const MemoryTierStatusPayload& status) override;
  absl::StatusOr<MemoryTierLeaseDescriptor> request_memory_tier_lease(
      const MemoryTierLeaseDescriptor& request) override;
  absl::StatusOr<MemoryTierLeaseDescriptor> acknowledge_memory_tier_lease(
      const MemoryTierLeaseAckPayload& ack) override;
  absl::StatusOr<std::vector<MemoryTierLeaseDescriptor>> list_memory_tier_leases(std::string_view node_id) override;
  absl::StatusOr<MemoryTierLeaseDescriptor> revoke_memory_tier_lease(std::string_view lease_id) override;

  absl::StatusOr<std::string> get_artifact_index_by_id(std::string_view artifact_id) override;
  absl::StatusOr<ViewMetadata> get_view_metadata(std::string_view artifact_id, std::string_view view_id) override;

  void update_local_endpoint(std::string node_id, std::string node_address, uint32_t grpc_port, uint32_t p2p_port)
      override;

  absl::Status update_artifact_view_state(const ViewStateUpdate& update) override;

  absl::StatusOr<std::vector<ViewInfo>> list_views(std::string_view artifact_id) override;

  absl::StatusOr<global_store::AssemblyLayoutBinding> get_assembly_layout_binding(
      std::string_view assembly_id) override;
  absl::StatusOr<layout::LayoutSpecRecord> get_layout_spec(std::string_view layout_id) override;
  absl::Status attach_layout_to_artifact(std::string_view mi2_id, std::string_view layout_id) override;
  absl::StatusOr<std::vector<std::string>> list_artifact_layouts(std::string_view mi2_id) override;
  absl::StatusOr<global_store::WriteTensorProofCommitmentsResponse> write_tensor_proof_commitments(
      const global_store::WriteTensorProofCommitmentsRequest& request) override;
  absl::StatusOr<global_store::CheckProofCommitmentsMatchResponse> check_proof_commitments_match(
      const global_store::CheckProofCommitmentsMatchRequest& request) override;
  absl::StatusOr<global_store::AssemblyRuntimePolicy> get_assembly_runtime_policy(
      std::string_view assembly_id) override;

  absl::StatusOr<operation::AcquireOperationLeaseResponse> acquire_operation_lease(
      const operation::AcquireOperationLeaseRequest& request) override;
  absl::StatusOr<operation::KeepaliveOperationLeaseResponse> keepalive_operation_lease(
      const operation::KeepaliveOperationLeaseRequest& request) override;
  absl::StatusOr<operation::ReleaseOperationLeaseResponse> release_operation_lease(
      const operation::ReleaseOperationLeaseRequest& request) override;
  absl::StatusOr<operation::GetOperationResponse> get_operation(const operation::GetOperationRequest& request) override;
  absl::Status update_operation(const operation::UpdateOperationRequest& request) override;

  absl::StatusOr<ArtifactBinding> get_artifact_binding(std::string_view artifact_id) override;

  absl::StatusOr<ArtifactBindingResult> upsert_artifact_binding(const ArtifactBinding& binding) override;

  absl::StatusOr<PlacementPlanResult> plan_placement(
      std::string_view artifact_id,
      global_store::PlacementPolicy policy,
      const std::vector<PlacementShardSpec>& shards,
      std::string_view source_node_id) override;

  absl::Status report_persistence_status(const PersistenceReport& report) override;

 private:
  // Helper for RPC retries
  template <typename Request, typename Response, typename RpcMethod>
  absl::Status execute_rpc_with_retry(
      const Request& request,
      Response* response,
      RpcMethod method,
      const std::string& method_name,
      const RpcOptions& rpc_options = RpcOptions{});

  // Convert between internal types and proto types
  static common::v1::MemoryType convert_to_proto_memory_type(common::memory::MemoryLocation location);
  static common::memory::MemoryLocation convert_from_proto_memory_type(common::v1::MemoryType type);
  absl::Status fill_memory_info(
      common::v1::MemoryInfo* info,
      const DeviceKey& device,
      common::memory::MemoryLocation location,
      uint64_t memory_size,
      std::optional<std::string_view> view_id = std::nullopt);
  static std::string build_client_request_id(std::string_view operation_kind, std::string_view canonical_payload);
  static RemoteReplicaInfo convert_from_proto_memory_info(const common::v1::MemoryInfo& info);

  const GlobalStoreClientConfig config_;
  const gsl::not_null<std::shared_ptr<grpc::Channel>> channel_;
  const gsl::not_null<std::unique_ptr<global_store::ClusterRuntimeService::Stub>> cluster_runtime_stub_;
  const gsl::not_null<std::unique_ptr<global_store::ArtifactCatalogService::Stub>> artifact_catalog_stub_;
  const gsl::not_null<std::unique_ptr<global_store::AssemblyViewService::Stub>> assembly_view_stub_;
  const gsl::not_null<std::unique_ptr<global_store::WorkflowOrchestrationService::Stub>> workflow_orchestration_stub_;
  const gsl::not_null<std::unique_ptr<global_store::ClusterAdminService::Stub>> cluster_admin_stub_;
  const gsl::not_null<std::unique_ptr<tensorcast::memory_tier::v1::MemoryTierService::Stub>> memory_tier_stub_;
  std::string worker_id_;
  std::string node_id_;
  std::string node_address_;
  uint32_t grpc_port_{0};
  uint32_t p2p_port_{0};
  mutable std::mutex mutex_;
};

} // namespace tensorcast::store::components
