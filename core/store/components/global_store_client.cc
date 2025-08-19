// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/components/global_store_client.h"

#include <unistd.h>
#include <chrono>
#include <thread>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "proto/global_store.grpc.pb.h"
#include "proto/global_store.pb.h"

namespace stepcast::store {

GlobalStoreClient::GlobalStoreClient(const GlobalStoreClientConfig& config) : config_(config) {}

GlobalStoreClient::~GlobalStoreClient() = default;

absl::Status GlobalStoreClient::initialize() {
  std::lock_guard<std::mutex> lock(mutex_);

  grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 30000);
  args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 20000);
  args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);

  channel_ = grpc::CreateCustomChannel(config_.global_store_address, grpc::InsecureChannelCredentials(), args);

  stub_ = ::global_store::GlobalModelStore::NewStub(channel_);

  // Test connection with a health check
  ::global_store::HealthCheckRequest req;
  ::global_store::HealthCheckResponse resp;

  grpc::ClientContext context;
  context.set_deadline(
      std::chrono::system_clock::now() + std::chrono::seconds(absl::ToInt64Seconds(config_.connection_timeout)));

  auto status = stub_->HealthCheck(&context, req, &resp);
  if (!status.ok()) {
    return absl::UnavailableError(
        absl::StrFormat(
            "Failed to connect to Global Store at %s: %s", config_.global_store_address, status.error_message()));
  }

  LOG(INFO) << "Successfully connected to Global Store at " << config_.global_store_address;
  return absl::OkStatus();
}

absl::StatusOr<std::string> GlobalStoreClient::register_worker(
    std::string_view node_id,
    std::string_view node_address,
    uint32_t grpc_port,
    uint32_t p2p_port,
    uint64_t mem_pool_total_size,
    uint64_t mem_pool_available_size) {
  ::global_store::RegisterWorkerRequest request;
  request.set_node_id(std::string(node_id));
  request.set_node_address(std::string(node_address));
  request.set_grpc_port(grpc_port);
  request.set_p2p_port(p2p_port);
  request.set_mem_pool_total_size(mem_pool_total_size);
  request.set_mem_pool_available_size(mem_pool_available_size);

  ::global_store::RegisterWorkerResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RegisterWorker(ctx, req, resp); },
      "RegisterWorker");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != ::global_store::OK) {
    return absl::InternalError(absl::StrFormat("RegisterWorker failed with status: %d", response.status()));
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    worker_id_ = response.worker_id();
  }

  LOG(INFO) << "Registered worker with ID: " << response.worker_id();
  return response.worker_id();
}

absl::Status GlobalStoreClient::send_heartbeat(
    std::string_view worker_id,
    uint64_t mem_pool_available_size,
    bool accepting_new_requests) {
  ::global_store::WorkerHeartbeatRequest request;
  request.set_worker_id(std::string(worker_id));
  request.set_mem_pool_available_size(mem_pool_available_size);
  request.set_accepting_new_requests(accepting_new_requests);

  ::global_store::WorkerHeartbeatResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->WorkerHeartbeat(ctx, req, resp); },
      "WorkerHeartbeat");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != ::global_store::OK) {
    return absl::InternalError(absl::StrFormat("WorkerHeartbeat failed with status: %d", response.status()));
  }

  return absl::OkStatus();
}

absl::Status GlobalStoreClient::unregister_worker(std::string_view worker_id, bool is_graceful_shutdown) {
  ::global_store::UnregisterWorkerRequest request;
  request.set_worker_id(std::string(worker_id));
  request.set_is_graceful_shutdown(is_graceful_shutdown);

  ::global_store::UnregisterWorkerResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->UnregisterWorker(ctx, req, resp); },
      "UnregisterWorker");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != ::global_store::OK) {
    return absl::InternalError(absl::StrFormat("UnregisterWorker failed with status: %d", response.status()));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::string> GlobalStoreClient::register_model_replica(
    std::string_view model_id,
    std::string_view worker_id,
    const DeviceKey& device,
    ModelLocation location,
    uint64_t memory_size,
    uint32_t max_concurrency) {
  ::global_store::RegisterModelReplicaRequest request;
  request.set_model_id(std::string(model_id));
  request.set_worker_id(std::string(worker_id));
  request.set_max_concurrency(max_concurrency);

  auto* mem_info = request.mutable_mem_info();
  fill_memory_info(mem_info, device, location, memory_size);

  ::global_store::RegisterModelReplicaResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RegisterModelReplica(ctx, req, resp); },
      "RegisterModelReplica");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != ::global_store::OK) {
    return absl::InternalError(absl::StrFormat("RegisterModelReplica failed with status: %d", response.status()));
  }

  LOG(INFO) << "Registered model replica: " << model_id << " with ID: " << response.replica_id();
  return response.replica_id();
}

absl::StatusOr<std::string> GlobalStoreClient::register_memory_replica(
    std::string_view model_id,
    std::string_view worker_id,
    const DeviceKey& device,
    uint64_t memory_size,
    std::string_view tensor_index_key,
    const std::vector<std::string>& remote_memory_keys,
    const std::vector<uint64_t>& buffer_sizes,
    const std::optional<std::string>& tensor_index_data,
    std::string_view encoding,
    std::string_view schema_version,
    uint32_t max_concurrency) {
  // NOTE: This implementation relies on proto/global_store.proto support for
  // memory replicas with tensor index key. If the server does not support the
  // new fields it will still accept the request but ignore extra data.

  ::global_store::RegisterModelReplicaRequest request;
  request.set_model_id(std::string(model_id));
  request.set_worker_id(std::string(worker_id));
  request.set_max_concurrency(max_concurrency);

  auto* mem_info = request.mutable_mem_info();
  fill_memory_info(mem_info, device, ModelLocation::GPU, memory_size);
  // If server supports memory replica fields, populate them via extension fields in MemoryInfo
  // For current proto, we include memory-replica metadata by overloading fields when available via
  // Global Store server. As a fallback, embed keys in the request's optional fields.

  // Mark as in-memory replica and attach tensor index key/metadata
  mem_info->set_is_memory_replica(true);
  if (!tensor_index_key.empty()) {
    mem_info->set_tensor_index_key(std::string(tensor_index_key));
  }
  // Best-effort creation metadata
  {
    const auto now = std::chrono::system_clock::now();
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    mem_info->set_creation_timestamp(static_cast<int64_t>(secs));
  }
  mem_info->set_source_process_id(std::to_string(getpid()));

  for (const auto& key : remote_memory_keys) {
    mem_info->add_remote_memory_keys(key);
  }
  for (const auto& sz : buffer_sizes) {
    mem_info->add_buffer_sizes(sz);
  }

  // Attach remote memory keys and buffer sizes if present
  // The header signature includes vectors; however, to maintain backward
  // compatibility with existing callers we do not require them.
  // Callers using enable_remote_instance_access should populate these via
  // the Model/CommunicationManager before calling this method.
  // NOTE: We cannot reference parameters by name due to preexisting signature –
  // adjust after refactor if needed.

  // Optionally include canonical index data for UPSERT on first write.
  if (tensor_index_data.has_value() && !tensor_index_data->empty()) {
    request.set_tensor_index_data(*tensor_index_data);
    if (!encoding.empty()) {
      request.set_encoding(std::string(encoding));
    }
    if (!schema_version.empty()) {
      request.set_schema_version(std::string(schema_version));
    }
  }
  // Note: descriptor attachment is handled at server-side using the parsed mi2 model_id
  // Optionally include descriptor when available to allow GS to upsert `models` directly.
  // Server will parse mi2 model_id and upsert models table as needed.

  ::global_store::RegisterModelReplicaResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RegisterModelReplica(ctx, req, resp); },
      "RegisterModelReplica(memory)");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != ::global_store::OK) {
    return absl::InternalError(absl::StrFormat("RegisterMemoryReplica failed with status: %d", response.status()));
  }

  LOG(INFO) << "Registered memory replica: " << model_id << " with ID: " << response.replica_id();
  return response.replica_id();
}

absl::Status GlobalStoreClient::unregister_model_replica(std::string_view model_id, std::string_view replica_id) {
  ::global_store::UnregisterModelReplicaRequest request;
  request.set_model_id(std::string(model_id));
  request.set_replica_id(std::string(replica_id));

  ::global_store::UnregisterModelReplicaResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->UnregisterModelReplica(ctx, req, resp); },
      "UnregisterModelReplica");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != ::global_store::OK) {
    return absl::InternalError(absl::StrFormat("UnregisterModelReplica failed with status: %d", response.status()));
  }

  return absl::OkStatus();
}

absl::StatusOr<TransportSession> GlobalStoreClient::request_model_transport(
    std::string_view model_id,
    std::string_view source_node_id,
    std::string_view source_address,
    uint32_t source_port,
    const DeviceKey& target_device,
    uint32_t wait_timeout_ms) {
  ::global_store::RequestModelReplicaTransportRequest request;
  request.set_model_id(std::string(model_id));
  request.set_source_node_id(std::string(source_node_id));
  request.set_source_address(std::string(source_address));
  request.set_source_port(source_port);
  request.set_wait_timeout_ms(wait_timeout_ms);

  auto* local_mem_info = request.mutable_local_memory_info();
  fill_memory_info(local_mem_info, target_device, ModelLocation::GPU, 0);

  ::global_store::RequestModelReplicaTransportResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RequestModelReplicaTransport(ctx, req, resp); },
      "RequestModelReplicaTransport");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != ::global_store::OK) {
    if (response.status() == ::global_store::NOT_FOUND) {
      return absl::NotFoundError(absl::StrFormat("No available replicas for model: %s", model_id));
    }
    return absl::InternalError(
        absl::StrFormat("RequestModelReplicaTransport failed with status: %d", response.status()));
  }

  TransportSession session;
  session.transport_id = response.transport_id();
  session.remote_replica = convert_from_proto_memory_info(response.remote_memory_info());
  session.start_time = absl::Now();

  LOG(INFO) << "Started P2P transport " << session.transport_id << " from " << session.remote_replica.node_id;

  return session;
}

absl::Status GlobalStoreClient::complete_model_transport(std::string_view transport_id) {
  ::global_store::CompleteModelReplicaTransportRequest request;
  request.set_transport_id(std::string(transport_id));

  ::global_store::CompleteModelReplicaTransportResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->CompleteModelReplicaTransport(ctx, req, resp); },
      "CompleteModelReplicaTransport");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != ::global_store::OK) {
    return absl::InternalError(
        absl::StrFormat("CompleteModelReplicaTransport failed with status: %d", response.status()));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<RemoteReplicaInfo>> GlobalStoreClient::get_model_replicas(std::string_view model_id) {
  ::global_store::GetModelInfoByIdRequest request;
  request.set_model_id(std::string(model_id));

  ::global_store::GetModelInfoByIdResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->GetModelInfoById(ctx, req, resp); },
      "GetModelInfoById");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != ::global_store::OK) {
    if (response.status() == ::global_store::NOT_FOUND) {
      return absl::NotFoundError(absl::StrFormat("Model not found: %s", model_id));
    }
    return absl::InternalError(absl::StrFormat("GetModelInfoById failed with status: %d", response.status()));
  }

  std::vector<RemoteReplicaInfo> replicas;
  for (const auto& mem_info : response.replicas()) {
    replicas.push_back(convert_from_proto_memory_info(mem_info));
  }

  return replicas;
}

bool GlobalStoreClient::is_connected() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return channel_ && channel_->GetState(false) == GRPC_CHANNEL_READY;
}

// Static conversion methods
::global_store::MemoryType GlobalStoreClient::convert_to_proto_memory_type(ModelLocation location) {
  switch (location) {
    case ModelLocation::GPU:
      return ::global_store::GPU;
    case ModelLocation::PAGEABLE_CPU:
      return ::global_store::RAM;
    case ModelLocation::DISK:
      return ::global_store::DISK;
    default:
      return ::global_store::DISK;
  }
}

ModelLocation GlobalStoreClient::convert_from_proto_memory_type(::global_store::MemoryType type) {
  switch (type) {
    case ::global_store::GPU:
      return ModelLocation::GPU;
    case ::global_store::RAM:
      return ModelLocation::PAGEABLE_CPU;
    case ::global_store::DISK:
      return ModelLocation::DISK;
    default:
      return ModelLocation::DISK;
  }
}

void GlobalStoreClient::fill_memory_info(
    ::global_store::MemoryInfo* info,
    const DeviceKey& device,
    ModelLocation location,
    uint64_t memory_size) {
  // Get hostname for node_id
  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) == 0) {
    info->set_node_id(hostname);
  }

  // TODO: Get actual IP address
  info->set_node_address("127.0.0.1");
  info->set_node_port(9090); // Default P2P port
  info->set_memory_size(memory_size);
  info->set_memory_type(convert_to_proto_memory_type(location));

  if (device.type == DeviceType::GPU) {
    info->set_device_id(device.ordinal);
  }
}

RemoteReplicaInfo GlobalStoreClient::convert_from_proto_memory_info(const ::global_store::MemoryInfo& info) {
  RemoteReplicaInfo replica;
  replica.node_id = info.node_id();
  replica.node_address = info.node_address();
  replica.node_port = info.node_port();
  replica.memory_size = info.memory_size();
  replica.memory_type = convert_from_proto_memory_type(info.memory_type());
  replica.device_id = info.device_id();

  for (const auto& key : info.remote_memory_keys()) {
    replica.remote_memory_keys.push_back(key);
  }

  for (const auto& size : info.buffer_sizes()) {
    replica.buffer_sizes.push_back(size);
  }

  return replica;
}

template <typename Request, typename Response, typename RpcMethod>
absl::Status GlobalStoreClient::execute_rpc_with_retry(
    const Request& request,
    Response* response,
    RpcMethod method,
    const std::string& method_name) {
  for (uint32_t attempt = 0; attempt <= config_.max_retries; ++attempt) {
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() + std::chrono::seconds(absl::ToInt64Seconds(config_.rpc_timeout)));

    auto status = method(&context, request, response);

    if (status.ok()) {
      return absl::OkStatus();
    }

    if (attempt < config_.max_retries) {
      auto backoff = config_.retry_backoff * (1 << attempt);
      LOG(WARNING) << "RPC " << method_name << " failed (attempt " << attempt + 1 << "/" << config_.max_retries + 1
                   << "): " << status.error_message() << ". Retrying in " << absl::ToInt64Milliseconds(backoff) << "ms";
      std::this_thread::sleep_for(std::chrono::milliseconds(absl::ToInt64Milliseconds(backoff)));
    }
  }

  return absl::UnavailableError(
      absl::StrFormat("RPC %s failed after %d retries", method_name, config_.max_retries + 1));
}

absl::StatusOr<std::vector<GlobalStoreClient::ChunkLocationInfo>> GlobalStoreClient::query_chunk_locations(
    std::string_view model_id,
    const std::vector<uint32_t>& chunk_indices) {
  ::global_store::QueryChunkLocationsRequest request;
  request.set_model_id(std::string(model_id));
  for (uint32_t idx : chunk_indices) {
    request.add_chunk_indices(idx);
  }

  ::global_store::QueryChunkLocationsResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->QueryChunkLocations(ctx, req, resp); },
      "QueryChunkLocations");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != ::global_store::OK) {
    if (response.status() == ::global_store::NOT_FOUND) {
      return absl::NotFoundError(absl::StrFormat("Model not found: %s", model_id));
    }
    return absl::InternalError(absl::StrFormat("QueryChunkLocations failed with status: %d", response.status()));
  }

  std::vector<ChunkLocationInfo> locations;
  locations.reserve(response.locations_size());

  for (const auto& loc : response.locations()) {
    ChunkLocationInfo info;
    info.chunk_idx = loc.chunk_idx();
    info.node_id = loc.node_id();
    info.node_address = loc.node_address();
    info.p2p_port = loc.p2p_port();

    // Convert proto ChunkState to our ChunkState
    switch (loc.state()) {
      case ::global_store::CHUNK_HOT:
        info.state = ChunkState::HOT;
        break;
      case ::global_store::CHUNK_LOCKED_TX:
        info.state = ChunkState::LOCKED_TX;
        break;
      case ::global_store::CHUNK_COPIED_GPU:
        info.state = ChunkState::COPIED_GPU;
        break;
      case ::global_store::CHUNK_COLD:
        info.state = ChunkState::COLD;
        break;
      case ::global_store::CHUNK_EVICTED:
        info.state = ChunkState::EVICTED;
        break;
      default:
        info.state = ChunkState::EVICTED; // Safe default
        break;
    }

    info.node_load_ratio = loc.node_load_ratio();
    info.device_uuid = loc.device_uuid();
    info.replica = loc.replica();

    locations.push_back(std::move(info));
  }

  return locations;
}

} // namespace stepcast::store