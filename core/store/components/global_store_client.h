// Copyright (c) 2025, StepCast Team. All rights reserved.

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
#include "core/store/device_types.h"
#include "core/store/model/chunk_meta.h"
#include "core/store/model/model_location.h"
#include "grpcpp/grpcpp.h"
#include "proto/global_store.grpc.pb.h"
#include "proto/global_store.pb.h"

namespace stepcast::store {

// Configuration for Global Store Client
struct GlobalStoreClientConfig {
  std::string global_store_address = "localhost:50051";
  absl::Duration connection_timeout = absl::Seconds(30);
  absl::Duration rpc_timeout = absl::Seconds(60);
  uint32_t max_retries = 3;
  absl::Duration retry_backoff = absl::Milliseconds(100);
};

// Information about a remote model replica
struct RemoteReplicaInfo {
  std::string node_id;
  std::string node_address;
  uint32_t node_port;
  uint64_t memory_size;
  ModelLocation memory_type;
  uint32_t device_id;
  std::vector<std::string> remote_memory_keys;
  std::vector<uint64_t> buffer_sizes;
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
      uint64_t mem_pool_available_size);

  absl::Status send_heartbeat(
      std::string_view worker_id,
      uint64_t mem_pool_available_size,
      bool accepting_new_requests = true);

  absl::Status unregister_worker(std::string_view worker_id, bool is_graceful_shutdown = true);

  // Model replica management
  absl::StatusOr<std::string> register_model_replica(
      std::string_view model_id,
      std::string_view worker_id,
      const DeviceKey& device,
      ModelLocation location,
      uint64_t memory_size,
      uint32_t max_concurrency = 1);

  // Register a GPU memory replica (in-memory tensor dict) with tensor index key.
  absl::StatusOr<std::string> register_memory_replica(
      std::string_view model_id,
      std::string_view worker_id,
      const DeviceKey& device,
      uint64_t memory_size,
      std::string_view tensor_index_key,
      const std::vector<std::string>& remote_memory_keys,
      const std::vector<uint64_t>& buffer_sizes,
      const std::optional<std::string>& tensor_index_data = std::nullopt,
      std::string_view encoding = "json",
      std::string_view schema_version = "v2",
      uint32_t max_concurrency = 1);

  absl::Status unregister_model_replica(std::string_view model_id, std::string_view replica_id);

  // P2P transport coordination
  absl::StatusOr<TransportSession> request_model_transport(
      std::string_view model_id,
      std::string_view source_node_id,
      std::string_view source_address,
      uint32_t source_port,
      const DeviceKey& target_device,
      uint32_t wait_timeout_ms = 30000);

  absl::Status complete_model_transport(std::string_view transport_id);

  // Model information queries
  absl::StatusOr<std::vector<RemoteReplicaInfo>> get_model_replicas(std::string_view model_id);

  // Chunk-level queries for unified memory management
  struct ChunkLocationInfo {
    uint32_t chunk_idx;
    std::string node_id;
    std::string node_address;
    uint32_t p2p_port;
    ChunkState state;
    float node_load_ratio;
    std::string device_uuid;
    uint32_t replica;
  };

  absl::StatusOr<std::vector<ChunkLocationInfo>> query_chunk_locations(
      std::string_view model_id,
      const std::vector<uint32_t>& chunk_indices);

  bool is_connected() const;

 private:
  // Helper for RPC retries
  template <typename Request, typename Response, typename RpcMethod>
  absl::Status execute_rpc_with_retry(
      const Request& request,
      Response* response,
      RpcMethod method,
      const std::string& method_name);

  // Convert between internal types and proto types
  static ::global_store::MemoryType convert_to_proto_memory_type(ModelLocation location);
  static ModelLocation convert_from_proto_memory_type(::global_store::MemoryType type);
  static void fill_memory_info(
      ::global_store::MemoryInfo* info,
      const DeviceKey& device,
      ModelLocation location,
      uint64_t memory_size);
  static RemoteReplicaInfo convert_from_proto_memory_info(const ::global_store::MemoryInfo& info);

  GlobalStoreClientConfig config_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<::global_store::GlobalModelStore::Stub> stub_;
  std::string worker_id_;
  mutable std::mutex mutex_;
};

} // namespace stepcast::store