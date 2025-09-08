// Copyright (c) 2025, TensorCast Team.

#include "core/store/components/global_store_client.h"

#include <unistd.h>
#include <chrono>
#include <random>
#include <thread>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "core/common/otel/grpc_propagation.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/global_store/v1/global_store.grpc.pb.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::store::components {

// Backward-compatibility: map unversioned global_store symbols to v1
namespace global_store = tensorcast::global_store::v1;

using common::memory::MemoryLocation;
using replica::ChunkState;

GlobalStoreClient::GlobalStoreClient(const GlobalStoreClientConfig& config) : config_(config) {}

GlobalStoreClient::~GlobalStoreClient() = default;

absl::Status GlobalStoreClient::initialize() {
  std::lock_guard<std::mutex> lock(mutex_);

  grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 30000);
  args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 20000);
  args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);

  channel_ = grpc::CreateCustomChannel(config_.global_store_address, grpc::InsecureChannelCredentials(), args);

  stub_ = global_store::GlobalStoreService::NewStub(channel_);

  // Test connection with a health check
  global_store::HealthCheckRequest req;
  global_store::HealthCheckResponse resp;

  grpc::ClientContext context;
  context.set_deadline(
      std::chrono::system_clock::now() + std::chrono::seconds(absl::ToInt64Seconds(config_.connection_timeout)));

  // OpenTelemetry: create a client span for HealthCheck and inject context.
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kClient;
  auto span = tracer->StartSpan("GlobalStore/HealthCheck", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.global_store.GlobalStoreService");
  span->SetAttribute("rpc.method", "HealthCheck");
  tensorcast::common::otel::inject_into_client_metadata(context);

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
  global_store::RegisterWorkerRequest request;
  request.set_node_id(std::string(node_id));
  request.set_node_address(std::string(node_address));
  request.set_grpc_port(grpc_port);
  request.set_p2p_port(p2p_port);
  request.set_mem_pool_total_size(mem_pool_total_size);
  request.set_mem_pool_available_size(mem_pool_available_size);

  global_store::RegisterWorkerResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RegisterWorker(ctx, req, resp); },
      "RegisterWorker");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
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
  global_store::WorkerHeartbeatRequest request;
  request.set_worker_id(std::string(worker_id));
  request.set_mem_pool_available_size(mem_pool_available_size);
  request.set_accepting_new_requests(accepting_new_requests);

  global_store::WorkerHeartbeatResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->WorkerHeartbeat(ctx, req, resp); },
      "WorkerHeartbeat");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(absl::StrFormat("WorkerHeartbeat failed with status: %d", response.status()));
  }

  return absl::OkStatus();
}

absl::StatusOr<global_store::WorkerHeartbeatResponse> GlobalStoreClient::send_heartbeat_enhanced(
    std::string_view worker_id,
    uint64_t mem_pool_available_size,
    bool accepting_new_requests,
    uint64_t state_version,
    std::string_view state_checksum,
    const std::vector<std::string>& registered_artifact_ids,
    int64_t last_successful_sync,
    global_store::ConnectionStatus connection_status) {
  global_store::WorkerHeartbeatRequest request;
  request.set_worker_id(std::string(worker_id));
  request.set_mem_pool_available_size(mem_pool_available_size);
  request.set_accepting_new_requests(accepting_new_requests);
  request.set_state_version(state_version);
  request.set_state_checksum(std::string(state_checksum));
  for (const auto& id : registered_artifact_ids) {
    request.add_registered_artifact_ids(id);
  }
  request.set_last_successful_sync(last_successful_sync);
  request.set_global_store_status(connection_status);

  global_store::WorkerHeartbeatResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->WorkerHeartbeat(ctx, req, resp); },
      "WorkerHeartbeat(enhanced)");
  if (!status.ok())
    return status;
  if (response.status() != global_store::STATUS_OK && response.status() != global_store::STATUS_STATE_SYNC_REQUIRED) {
    return absl::InternalError(absl::StrFormat("WorkerHeartbeat(enhanced) failed with status: %d", response.status()));
  }
  return response;
}

absl::Status GlobalStoreClient::unregister_worker(std::string_view worker_id, bool is_graceful_shutdown) {
  global_store::UnregisterWorkerRequest request;
  request.set_worker_id(std::string(worker_id));
  request.set_is_graceful_shutdown(is_graceful_shutdown);

  global_store::UnregisterWorkerResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->UnregisterWorker(ctx, req, resp); },
      "UnregisterWorker");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(absl::StrFormat("UnregisterWorker failed with status: %d", response.status()));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::string> GlobalStoreClient::register_replica(
    std::string_view artifact_id,
    std::string_view worker_id,
    const DeviceKey& device,
    MemoryLocation location,
    uint64_t memory_size,
    uint32_t max_concurrency) {
  global_store::RegisterReplicaRequest request;
  request.set_artifact_id(std::string(artifact_id));
  request.set_worker_id(std::string(worker_id));
  request.set_max_concurrency(max_concurrency);

  auto* mem_info = request.mutable_mem_info();
  fill_memory_info(mem_info, device, location, memory_size);

  global_store::RegisterReplicaResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RegisterReplica(ctx, req, resp); },
      "RegisterReplica");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(absl::StrFormat("RegisterReplica failed with status: %d", response.status()));
  }

  LOG(INFO) << "Registered replica: " << artifact_id << " with ID: " << response.replica_id();
  return response.replica_id();
}

absl::StatusOr<std::string> GlobalStoreClient::register_memory_replica(
    std::string_view artifact_id,
    std::string_view worker_id,
    const DeviceKey& device,
    uint64_t memory_size,
    std::string_view tensor_index_key,
    const std::vector<std::string>& remote_memory_keys,
    const std::vector<uint64_t>& buffer_sizes,
    const std::optional<std::string>& tensor_index_data,
    std::string_view encoding,
    std::string_view schema_version,
    uint32_t max_concurrency,
    const std::optional<std::string>& verification_json) {
  // NOTE: This implementation relies on proto/global_store.proto support for
  // memory replicas with tensor index key. If the server does not support the
  // new fields it will still accept the request but ignore extra data.

  global_store::RegisterReplicaRequest request;
  request.set_artifact_id(std::string(artifact_id));
  request.set_worker_id(std::string(worker_id));
  request.set_max_concurrency(max_concurrency);

  auto* mem_info = request.mutable_mem_info();
  fill_memory_info(mem_info, device, MemoryLocation::GPU, memory_size);
  // If server supports memory replica fields, populate them via extension fields in MemoryInfo
  // For current proto, we include memory-replica metadata by overloading fields when available via
  // Global Store server. As a fallback, embed keys in the request's optional fields.

  // Mark as in-memory replica and attach tensor index key/metadata
  mem_info->set_is_memory_replica(true);
  if (!tensor_index_key.empty()) {
    mem_info->set_tensor_index_key(std::string(tensor_index_key));
  }
  if (verification_json.has_value() && !verification_json->empty()) {
    mem_info->set_verification_json(*verification_json);
  }
  // Best-effort creation metadata
  {
    const auto now = std::chrono::system_clock::now();
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    auto* ts = mem_info->mutable_creation_ts();
    ts->set_seconds(static_cast<int64_t>(secs));
    ts->set_nanos(0);
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
  // Callers using enable_remote_replica_access should populate these via
  // the Replica/CommunicationManager before calling this method.
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
  // Note: descriptor attachment is handled at server-side using the parsed mi2 artifact_id
  // Optionally include descriptor when available to allow GS to upsert the artifact registry directly.
  // Server will parse mi2 artifact_id and upsert the artifacts table as needed.

  global_store::RegisterReplicaResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RegisterReplica(ctx, req, resp); },
      "RegisterReplica(memory)");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(absl::StrFormat("RegisterMemoryReplica failed with status: %d", response.status()));
  }

  LOG(INFO) << "Registered memory replica: " << artifact_id << " with ID: " << response.replica_id();
  return response.replica_id();
}

absl::Status GlobalStoreClient::unregister_replica(std::string_view artifact_id, std::string_view replica_id) {
  global_store::UnregisterReplicaRequest request;
  request.set_artifact_id(std::string(artifact_id));
  request.set_replica_id(std::string(replica_id));

  global_store::UnregisterReplicaResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->UnregisterReplica(ctx, req, resp); },
      "UnregisterReplica");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(absl::StrFormat("UnregisterReplica failed with status: %d", response.status()));
  }

  return absl::OkStatus();
}

absl::StatusOr<TransportSession> GlobalStoreClient::request_replica_transport(
    std::string_view artifact_id,
    std::string_view source_node_id,
    std::string_view source_address,
    uint32_t source_port,
    const DeviceKey& target_device,
    uint32_t wait_timeout_ms) {
  global_store::RequestReplicaTransportRequest request;
  request.set_artifact_id(std::string(artifact_id));
  request.set_source_node_id(std::string(source_node_id));
  request.set_source_address(std::string(source_address));
  request.set_source_port(source_port);
  // Use standard duration
  auto* dur = request.mutable_wait_timeout_dur();
  dur->set_seconds(static_cast<int64_t>(wait_timeout_ms / 1000));
  dur->set_nanos(static_cast<int32_t>((wait_timeout_ms % 1000) * 1000000));

  auto* local_mem_info = request.mutable_local_memory_info();
  fill_memory_info(local_mem_info, target_device, MemoryLocation::GPU, 0);

  global_store::RequestReplicaTransportResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RequestReplicaTransport(ctx, req, resp); },
      "RequestReplicaTransport");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    if (response.status() == global_store::STATUS_NOT_FOUND) {
      return absl::NotFoundError(absl::StrFormat("No available replicas for replica: %s", artifact_id));
    }
    return absl::InternalError(absl::StrFormat("RequestReplicaTransport failed with status: %d", response.status()));
  }

  TransportSession session;
  session.transport_id = response.transport_id();
  session.remote_replica = convert_from_proto_memory_info(response.remote_memory_info());
  session.start_time = absl::Now();

  LOG(INFO) << "Started P2P transport " << session.transport_id << " from " << session.remote_replica.node_id;

  return session;
}

absl::Status GlobalStoreClient::complete_replica_transport(std::string_view transport_id) {
  global_store::CompleteReplicaTransportRequest request;
  request.set_transport_id(std::string(transport_id));

  global_store::CompleteReplicaTransportResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->CompleteReplicaTransport(ctx, req, resp); },
      "CompleteReplicaTransport");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(absl::StrFormat("CompleteReplicaTransport failed with status: %d", response.status()));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<RemoteReplicaInfo>> GlobalStoreClient::get_artifact_replicas(std::string_view artifact_id) {
  global_store::GetArtifactInfoByIdRequest request;
  request.set_artifact_id(std::string(artifact_id));

  global_store::GetArtifactInfoByIdResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->GetArtifactInfoById(ctx, req, resp); },
      "GetArtifactInfoById");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    if (response.status() == global_store::STATUS_NOT_FOUND) {
      return absl::NotFoundError(absl::StrFormat("Artifact not found: %s", artifact_id));
    }
    return absl::InternalError(absl::StrFormat("GetArtifactInfoById failed with status: %d", response.status()));
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

absl::Status GlobalStoreClient::batch_update_chunk_states(
    std::string_view worker_id,
    std::string_view node_id,
    const std::vector<ChunkStateUpdate>& updates) {
  if (updates.empty())
    return absl::OkStatus();
  global_store::BatchUpdateChunkStatesRequest req;
  req.set_worker_id(std::string(worker_id));
  req.set_node_id(std::string(node_id));
  for (const auto& u : updates) {
    auto* up = req.add_updates();
    up->set_artifact_id(u.artifact_id);
    up->set_chunk_idx(u.chunk_idx);
    switch (u.state) {
      case ChunkState::HOT:
        up->set_state(global_store::CHUNK_STATE_HOT);
        break;
      case ChunkState::LOCKED_TX:
        up->set_state(global_store::CHUNK_STATE_LOCKED_TX);
        break;
      case ChunkState::COPIED_GPU:
        up->set_state(global_store::CHUNK_STATE_COPIED_GPU);
        break;
      case ChunkState::EVICTED:
        up->set_state(global_store::CHUNK_STATE_EVICTED);
        break;
      case ChunkState::COLD:
      case ChunkState::PREEMPTIBLE:
      default:
        up->set_state(global_store::CHUNK_STATE_COLD);
        break;
    }
    up->set_device_uuid(u.device_uuid);
    up->set_replica(u.replica);
  }
  global_store::BatchUpdateChunkStatesResponse resp;
  auto st = execute_rpc_with_retry(
      req,
      &resp,
      [this](auto* ctx, const auto& r, auto* o) { return stub_->BatchUpdateChunkStates(ctx, r, o); },
      "BatchUpdateChunkStates");
  if (!st.ok())
    return st;
  if (resp.status() != global_store::STATUS_OK) {
    return absl::InternalError(absl::StrFormat("BatchUpdateChunkStates failed with status: %d", resp.status()));
  }
  return absl::OkStatus();
}

// Static conversion methods
tensorcast::common::v1::MemoryType GlobalStoreClient::convert_to_proto_memory_type(MemoryLocation location) {
  switch (location) {
    case MemoryLocation::GPU:
      return tensorcast::common::v1::MEMORY_TYPE_GPU;
    case MemoryLocation::PAGEABLE_CPU:
      return tensorcast::common::v1::MEMORY_TYPE_RAM;
    case MemoryLocation::DISK:
      return tensorcast::common::v1::MEMORY_TYPE_DISK;
    case MemoryLocation::REMOTE:
      return tensorcast::common::v1::MEMORY_TYPE_DISK; // Fallback mapping for REMOTE
    default:
      return tensorcast::common::v1::MEMORY_TYPE_DISK;
  }
}

MemoryLocation GlobalStoreClient::convert_from_proto_memory_type(tensorcast::common::v1::MemoryType type) {
  switch (type) {
    case tensorcast::common::v1::MEMORY_TYPE_GPU:
      return MemoryLocation::GPU;
    case tensorcast::common::v1::MEMORY_TYPE_RAM:
      return MemoryLocation::PAGEABLE_CPU;
    case tensorcast::common::v1::MEMORY_TYPE_DISK:
      return MemoryLocation::DISK;
    default:
      return MemoryLocation::DISK;
  }
}

void GlobalStoreClient::fill_memory_info(
    tensorcast::common::v1::MemoryInfo* info,
    const DeviceKey& device,
    MemoryLocation location,
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

RemoteReplicaInfo GlobalStoreClient::convert_from_proto_memory_info(const tensorcast::common::v1::MemoryInfo& info) {
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
  if (!info.verification_json().empty()) {
    replica.verification_json = info.verification_json();
  }

  return replica;
}

template <typename Request, typename Response, typename RpcMethod>
absl::Status GlobalStoreClient::execute_rpc_with_retry(
    const Request& request,
    Response* response,
    RpcMethod method,
    const std::string& method_name) {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  for (uint32_t attempt = 0; attempt <= config_.max_retries; ++attempt) {
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() + std::chrono::seconds(absl::ToInt64Seconds(config_.rpc_timeout)));

    // OpenTelemetry: create client span and inject W3C Trace Context.
    namespace otel = opentelemetry;
    auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
    otel::trace::StartSpanOptions opts;
    opts.kind = otel::trace::SpanKind::kClient;
    auto span = tracer->StartSpan(absl::StrFormat("GlobalStore/%s", method_name), opts);
    otel::trace::Scope scope(span);
    span->SetAttribute("rpc.system", "grpc");
    span->SetAttribute("rpc.service", "tensorcast.global_store.GlobalStoreService");
    span->SetAttribute("rpc.method", method_name);
    tensorcast::common::otel::inject_into_client_metadata(context);

    auto status = method(&context, request, response);
    if (!status.ok()) {
      span->SetAttribute("rpc.grpc.status_code", static_cast<int64_t>(status.error_code()));
      span->SetAttribute("rpc.grpc.message", status.error_message());
    }

    if (status.ok()) {
      return absl::OkStatus();
    }

    if (attempt < config_.max_retries) {
      auto base = config_.retry_backoff * (1 << attempt);
      // Jitter within +/- 50%
      double jitter = std::uniform_real_distribution<double>(0.5, 1.5)(rng);
      auto jittered = absl::Milliseconds(static_cast<int64_t>(absl::ToInt64Milliseconds(base) * jitter));
      LOG(WARNING) << "RPC " << method_name << " failed (attempt " << attempt + 1 << "/" << config_.max_retries + 1
                   << "): " << status.error_message() << ". Retrying in " << absl::ToInt64Milliseconds(jittered)
                   << "ms";
      std::this_thread::sleep_for(std::chrono::milliseconds(absl::ToInt64Milliseconds(jittered)));
    }
  }

  return absl::UnavailableError(
      absl::StrFormat("RPC %s failed after %d retries", method_name, config_.max_retries + 1));
}

absl::StatusOr<std::vector<GlobalStoreClient::ChunkLocationInfo>> GlobalStoreClient::query_chunk_locations(
    std::string_view artifact_id,
    const std::vector<uint32_t>& chunk_indices) {
  global_store::QueryChunkLocationsRequest request;
  request.set_artifact_id(std::string(artifact_id));
  for (uint32_t idx : chunk_indices) {
    request.add_chunk_indices(idx);
  }

  global_store::QueryChunkLocationsResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->QueryChunkLocations(ctx, req, resp); },
      "QueryChunkLocations");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    if (response.status() == global_store::STATUS_NOT_FOUND) {
      return absl::NotFoundError(absl::StrFormat("Artifact not found: %s", artifact_id));
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
      case global_store::CHUNK_STATE_HOT:
        info.state = ChunkState::HOT;
        break;
      case global_store::CHUNK_STATE_LOCKED_TX:
        info.state = ChunkState::LOCKED_TX;
        break;
      case global_store::CHUNK_STATE_COPIED_GPU:
        info.state = ChunkState::COPIED_GPU;
        break;
      case global_store::CHUNK_STATE_COLD:
        info.state = ChunkState::COLD;
        break;
      case global_store::CHUNK_STATE_EVICTED:
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

absl::StatusOr<std::pair<uint64_t, std::string>> GlobalStoreClient::synchronize_worker_state(
    const global_store::WorkerLocalState& local_state,
    bool force_full_sync,
    std::vector<global_store::StateChange>* out_changes) {
  if (!out_changes)
    return absl::InvalidArgumentError("out_changes must not be null");
  global_store::SynchronizeWorkerStateRequest request;
  request.set_worker_id(local_state.worker_id());
  *request.mutable_local_state() = local_state;
  request.set_force_full_sync(force_full_sync);

  global_store::SynchronizeWorkerStateResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->SynchronizeWorkerState(ctx, req, resp); },
      "SynchronizeWorkerState");
  if (!status.ok())
    return status;
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(absl::StrFormat("SynchronizeWorkerState failed with status: %d", response.status()));
  }
  out_changes->clear();
  out_changes->reserve(response.state_changes_size());
  for (const auto& ch : response.state_changes()) {
    out_changes->push_back(ch);
  }
  return std::make_pair(response.new_state_version(), response.new_state_checksum());
}

absl::StatusOr<std::pair<uint64_t, std::string>> GlobalStoreClient::request_full_state_sync(
    std::string_view worker_id,
    uint64_t current_state_version,
    std::vector<tensorcast::common::v1::ReplicaInfo>* out_expected_replicas) {
  if (!out_expected_replicas)
    return absl::InvalidArgumentError("out_expected_replicas must not be null");
  global_store::RequestFullStateSyncRequest request;
  request.set_worker_id(std::string(worker_id));
  request.set_current_state_version(current_state_version);

  global_store::RequestFullStateSyncResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RequestFullStateSync(ctx, req, resp); },
      "RequestFullStateSync");
  if (!status.ok())
    return status;
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(absl::StrFormat("RequestFullStateSync failed with status: %d", response.status()));
  }
  out_expected_replicas->clear();
  out_expected_replicas->reserve(response.expected_replicas_size());
  for (const auto& rep : response.expected_replicas()) {
    out_expected_replicas->push_back(rep);
  }
  return std::make_pair(response.new_state_version(), response.new_state_checksum());
}

// ========== RFC-0014: Key Mapping ==========

absl::StatusOr<GlobalStoreClient::KeyMapping> GlobalStoreClient::resolve_key_mapping(std::string_view key) {
  global_store::ResolveKeyMappingRequest request;
  request.set_key(std::string(key));

  global_store::ResolveKeyMappingResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->ResolveKeyMapping(ctx, req, resp); },
      "ResolveKeyMapping");

  if (!status.ok()) {
    return status;
  }
  if (response.status() != global_store::STATUS_OK) {
    return absl::NotFoundError("key not found");
  }

  KeyMapping out{
      .artifact_id = response.artifact_id(),
      .replica_uuid = response.replica_uuid(),
      .daemon_address = response.daemon_address(),
      .disk_path = response.disk_path(),
  };
  return out;
}

absl::Status GlobalStoreClient::upsert_key_mapping(
    std::string_view key,
    std::string_view artifact_id,
    std::string_view disk_path,
    absl::Duration ttl) {
  global_store::UpsertKeyMappingRequest request;
  request.set_key(std::string(key));
  request.set_artifact_id(std::string(artifact_id));
  request.set_disk_path(std::string(disk_path));
  if (ttl > absl::ZeroDuration()) {
    auto* d = request.mutable_ttl();
    d->set_seconds(absl::ToInt64Seconds(ttl));
    d->set_nanos(0);
  }

  global_store::UpsertKeyMappingResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->UpsertKeyMapping(ctx, req, resp); },
      "UpsertKeyMapping");
  if (!status.ok())
    return status;
  if (response.status() != global_store::STATUS_OK) {
    return absl::AlreadyExistsError("key mapping conflict or error");
  }
  return absl::OkStatus();
}

absl::Status GlobalStoreClient::revoke_key_mapping(std::string_view key) {
  global_store::RevokeKeyMappingRequest request;
  request.set_key(std::string(key));

  global_store::RevokeKeyMappingResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RevokeKeyMapping(ctx, req, resp); },
      "RevokeKeyMapping");
  if (!status.ok())
    return status;
  if (response.status() != global_store::STATUS_OK) {
    return absl::NotFoundError("key not found");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> GlobalStoreClient::get_artifact_index_by_id(std::string_view artifact_id) {
  global_store::GetArtifactIndexByIdRequest request;
  request.set_artifact_id(std::string(artifact_id));

  global_store::GetArtifactIndexByIdResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->GetArtifactIndexById(ctx, req, resp); },
      "GetArtifactIndexById");
  if (!status.ok())
    return status;
  // In proto3, singular bytes fields do not have presence; check emptiness instead.
  if (response.status() != global_store::STATUS_OK || response.tensor_index_data().empty()) {
    return absl::NotFoundError("artifact index not found");
  }
  return response.tensor_index_data();
}

} // namespace tensorcast::store::components
