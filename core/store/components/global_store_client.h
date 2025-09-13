// Copyright (c) 2025, TensorCast Team.

#pragma once

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
#include "core/store/replica/chunk_meta.h"
#include "grpcpp/grpcpp.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/global_store/v1/global_store.grpc.pb.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::store::components {

// Configuration for Global Store Client
struct GlobalStoreClientConfig {
  std::string global_store_address = "localhost:50051";
  absl::Duration connection_timeout = absl::Seconds(30);
  absl::Duration rpc_timeout = absl::Seconds(60);
  uint32_t max_retries = 3;
  absl::Duration retry_backoff = absl::Milliseconds(100);
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
};

// Transport session for P2P transfers
struct TransportSession {
  std::string transport_id;
  RemoteReplicaInfo remote_replica;
  absl::Time start_time;
};

class GlobalStoreClient {
 public:
  explicit GlobalStoreClient(const GlobalStoreClientConfig& config);
  ~GlobalStoreClient();

  // Initialize connection to Global Store
  absl::Status initialize();

  // Worker registration and lifecycle
  absl::StatusOr<std::string> register_worker(
      std::string_view node_id,
      std::string_view node_address,
      uint32_t grpc_port,
      uint32_t p2p_port,
      uint64_t mem_pool_total_size,
      uint64_t mem_pool_available_size,
      bool is_recovery_registration = false,
      std::string_view previous_worker_id = {});

  absl::Status send_heartbeat(
      std::string_view worker_id,
      uint64_t mem_pool_available_size,
      bool accepting_new_requests = true);

  // Enhanced heartbeat with HA state fields
  absl::StatusOr<global_store::v1::WorkerHeartbeatResponse> send_heartbeat_enhanced(
      std::string_view worker_id,
      uint64_t mem_pool_available_size,
      bool accepting_new_requests,
      uint64_t state_version,
      std::string_view state_checksum,
      const std::vector<std::string>& registered_artifact_ids,
      int64_t last_successful_sync,
      global_store::v1::ConnectionStatus connection_status = global_store::v1::CONNECTION_STATUS_CONNECTED);

  absl::Status unregister_worker(std::string_view worker_id, bool is_graceful_shutdown = true);

  // Replica management
  absl::StatusOr<std::string> register_replica(
      std::string_view artifact_id,
      std::string_view worker_id,
      const DeviceKey& device,
      common::memory::MemoryLocation location,
      uint64_t memory_size,
      uint32_t max_concurrency = 1);

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
      std::string_view schema_version = "v2",
      uint32_t max_concurrency = 1,
      const std::optional<std::string>& verification_json = std::nullopt);

  absl::Status unregister_replica(std::string_view artifact_id, std::string_view replica_id);

  // P2P transport coordination
  absl::StatusOr<TransportSession> request_replica_transport(
      std::string_view artifact_id,
      std::string_view source_node_id,
      std::string_view source_address,
      uint32_t source_port,
      const DeviceKey& target_device,
      uint32_t wait_timeout_ms = 30000);

  absl::Status complete_replica_transport(std::string_view transport_id);

  absl::StatusOr<std::vector<RemoteReplicaInfo>> get_artifact_replicas(std::string_view artifact_id);

  // Chunk-level queries for unified memory management
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

  absl::StatusOr<std::vector<ChunkLocationInfo>> query_chunk_locations(
      std::string_view artifact_id,
      const std::vector<uint32_t>& chunk_indices);

  // HA State Synchronization
  absl::StatusOr<std::pair<uint64_t, std::string>> synchronize_worker_state(
      const global_store::v1::WorkerLocalState& local_state,
      bool force_full_sync,
      std::vector<global_store::v1::StateChange>* out_changes);

  absl::StatusOr<std::pair<uint64_t, std::string>> request_full_state_sync(
      std::string_view worker_id,
      uint64_t current_state_version,
      std::vector<common::v1::ReplicaInfo>* out_expected_replicas);

  bool is_connected() const;

  // Batch chunk-state updates (VS telemetry -> Global Store)
  struct ChunkStateUpdate {
    std::string artifact_id;
    uint32_t chunk_idx{0};
    replica::ChunkState state{replica::ChunkState::COLD};
    std::string device_uuid;
    uint32_t replica{0};
  };

  absl::Status batch_update_chunk_states(
      std::string_view worker_id,
      std::string_view node_id,
      const std::vector<ChunkStateUpdate>& updates);

  // ========== RFC-0014: Key Mapping ==========
  struct KeyMapping {
    std::string artifact_id;
    std::string replica_uuid;
    std::string daemon_address;
    std::string disk_path;
  };

  // Resolve a human-friendly key to artifact identity and optional hints.
  absl::StatusOr<KeyMapping> resolve_key_mapping(std::string_view key);

  // Upsert a key mapping (conflict if mapping points to a different artifact).
  absl::Status upsert_key_mapping(
      std::string_view key,
      std::string_view artifact_id,
      std::string_view disk_path = {},
      absl::Duration ttl = absl::ZeroDuration());

  // Revoke an existing key mapping.
  absl::Status revoke_key_mapping(std::string_view key);

  // Fetch canonical tensor index bytes by artifact_id.
  absl::StatusOr<std::string> get_artifact_index_by_id(std::string_view artifact_id);

 private:
  // Helper for RPC retries
  template <typename Request, typename Response, typename RpcMethod>
  absl::Status execute_rpc_with_retry(
      const Request& request,
      Response* response,
      RpcMethod method,
      const std::string& method_name);

  // Convert between internal types and proto types
  static common::v1::MemoryType convert_to_proto_memory_type(common::memory::MemoryLocation location);
  static common::memory::MemoryLocation convert_from_proto_memory_type(common::v1::MemoryType type);
  static void fill_memory_info(
      common::v1::MemoryInfo* info,
      const DeviceKey& device,
      common::memory::MemoryLocation location,
      uint64_t memory_size);
  static RemoteReplicaInfo convert_from_proto_memory_info(const common::v1::MemoryInfo& info);

  GlobalStoreClientConfig config_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<global_store::v1::GlobalStoreService::Stub> stub_;
  std::string worker_id_;
  mutable std::mutex mutex_;
};

} // namespace tensorcast::store::components
