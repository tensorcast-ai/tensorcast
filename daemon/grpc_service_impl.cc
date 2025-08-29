// Copyright (c) 2025, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include "absl/strings/str_format.h"
#include "core/store/device_registry.h"
#include "core/store/device_types.h"
#include "core/store/loading/loading_spec.h"
#include "daemon/status_utils.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

Status StoreDaemonServiceImpl::MaterializeReplica(
    grpc::ServerContext* ctx,
    const ::store_daemon::MaterializeReplicaRequest* req,
    ::store_daemon::MaterializeReplicaResponse* resp) {
  using ::store_daemon::MaterializeReplicaStatus;

  // Validate one-of inputs: exactly one of artifact_id or disk_path
  const bool has_artifact = req->has_artifact_id() && !req->artifact_id().empty();
  const bool has_disk = req->has_disk_path() && !req->disk_path().empty();
  if (has_artifact == has_disk) {
    return {StatusCode::INVALID_ARGUMENT, "Exactly one of artifact_id or disk_path must be provided"};
  }

  // Build DeviceKey
  const auto device = resolve_device(*req);

  // Build hints
  tensorcast::store::MaterializeHints hints;
  if (req->pinned_allocation_timeout_ms() > 0) {
    hints.pinned_timeout = std::chrono::milliseconds(req->pinned_allocation_timeout_ms());
  }
  if (has_artifact)
    hints.artifact_id = req->artifact_id();
  if (has_disk)
    hints.disk_path = req->disk_path();

  // Choose mode
  tensorcast::store::StoreEngine::MaterializeMode mode = tensorcast::store::StoreEngine::MaterializeMode::AUTO;
  if (has_disk) {
    mode = tensorcast::store::StoreEngine::MaterializeMode::LOAD_ONLY;
  }

  auto result = engine_->materialize_replica(device, mode, hints);
  if (!result.ok()) {
    // Allocation/init failed
    resp->set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(result.status());
  }

  const auto& handle = *result;
  // Store session mapping for Confirm/Unload
  if (!req->replica_uuid().empty()) {
    sessions_.put(req->replica_uuid(), handle.replica_key, handle.ready_future);
  }
  // Track initial PID reference if provided
  if (req->pid() > 0) {
    refs_.add_ref(handle.replica_key, req->pid());
  }

  // Populate response
  if (has_disk)
    resp->set_disk_path(req->disk_path());
  resp->set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  if (handle.cuda_ipc_handle.is_valid()) {
    auto* mem = resp->mutable_mem_handle();
    mem->set_cuda_ipc_handle(handle.cuda_ipc_handle.to_string());
  }
  return Status::OK;
}

Status StoreDaemonServiceImpl::ConfirmReplica(
    grpc::ServerContext* ctx,
    const ::store_daemon::ConfirmReplicaRequest* req,
    ::store_daemon::ConfirmReplicaResponse* resp) {
  resp->set_disk_path(req->disk_path());

  if (req->replica_uuid().empty()) {
    resp->set_code(0);
    return Status::OK;
  }

  auto entry = sessions_.get(req->replica_uuid());
  if (!entry.has_value()) {
    // Parity: unknown replica_uuid → code=0 OK
    resp->set_code(0);
    return Status::OK;
  }

  // Compute bounded wait timeout: min(30s, remaining gRPC deadline)
  std::chrono::milliseconds wait_ms(30000);
  const auto deadline = ctx->deadline();
  const auto now = std::chrono::system_clock::now();
  if (deadline != std::chrono::system_clock::time_point::max()) {
    if (deadline <= now) {
      wait_ms = std::chrono::milliseconds(0);
    } else {
      auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
      if (remaining < wait_ms)
        wait_ms = remaining;
    }
  }

  absl::Status st = entry->ready.wait_for(wait_ms) == std::future_status::ready
      ? entry->ready.get()
      : absl::DeadlineExceededError("confirm timeout");
  if (st.ok()) {
    resp->set_code(0);
    return Status::OK;
  }
  resp->set_code(1);
  return to_grpc_status(st);
}

Status StoreDaemonServiceImpl::UnloadReplica(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::UnloadReplicaRequest* req,
    ::store_daemon::UnloadReplicaResponse* resp) {
  resp->set_disk_path(req->disk_path());
  // Idempotent success for unknown entries
  if (req->replica_uuid().empty()) {
    resp->set_code(0);
    return Status::OK;
  }
  auto entry = sessions_.get(req->replica_uuid());
  if (!entry.has_value()) {
    resp->set_code(0);
    return Status::OK;
  }
  // Drop PID ref if provided; if refs remain, skip unload per parity
  if (req->has_pid()) {
    refs_.drop_ref(entry->key, req->pid());
    if (refs_.ref_count(entry->key) > 0) {
      resp->set_code(0);
      return Status::OK;
    }
  }
  const int rc = engine_->unload_replica(entry->key);
  if (rc == 0) {
    sessions_.erase(req->replica_uuid());
    resp->set_code(0);
    return Status::OK;
  }
  resp->set_code(1);
  return {StatusCode::INTERNAL, absl::StrFormat("unload_replica() returned %d", rc)};
}

Status StoreDaemonServiceImpl::ClearMem(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::ClearMemRequest* /*req*/,
    ::store_daemon::ClearMemResponse* /*resp*/) {
  const int rc = engine_->clear_mem();
  if (rc == 0)
    return Status::OK;
  return {StatusCode::INTERNAL, absl::StrFormat("clear_mem() returned %d", rc)};
}

Status StoreDaemonServiceImpl::GetServerConfig(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::GetServerConfigRequest* /*req*/,
    ::store_daemon::GetServerConfigResponse* resp) {
  resp->set_mem_pool_size(static_cast<int64_t>(engine_->get_mem_pool_size()));
  resp->set_chunk_size(static_cast<int64_t>(engine_->get_chunk_size()));
  return Status::OK;
}

// Destructor: stop sweepers
StoreDaemonServiceImpl::~StoreDaemonServiceImpl() {
  stop_sweepers();
}

// ──────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────

static tensorcast::store::DeviceKey default_gpu_key() {
  return tensorcast::store::DeviceRegistry::instance().gpu_key(0);
}

tensorcast::store::DeviceKey StoreDaemonServiceImpl::resolve_device(
    const ::store_daemon::MaterializeReplicaRequest& req) {
  using tensorcast::DeviceType;
  using tensorcast::store::DeviceKey;
  if (!req.device_uuid().empty()) {
    DeviceKey key{DeviceType::GPU, 0, req.device_uuid()};
    return tensorcast::store::DeviceRegistry::instance().normalize(key);
  }
  switch (req.target_device_type()) {
    case ::store_daemon::DeviceType::DEVICE_TYPE_CPU:
      return DeviceKey{DeviceType::CPU, -1, ""};
    case ::store_daemon::DeviceType::DEVICE_TYPE_DISK:
      // Treat as ingest-from-disk to default GPU for v1 parity
      return default_gpu_key();
    case ::store_daemon::DeviceType::DEVICE_TYPE_GPU:
    default:
      return default_gpu_key();
  }
}

tensorcast::store::DeviceKey StoreDaemonServiceImpl::resolve_device(const ::store_daemon::ConfirmReplicaRequest& req) {
  using tensorcast::DeviceType;
  using tensorcast::store::DeviceKey;
  switch (req.target_device_type()) {
    case ::store_daemon::DeviceType::DEVICE_TYPE_CPU:
      return DeviceKey{DeviceType::CPU, -1, ""};
    case ::store_daemon::DeviceType::DEVICE_TYPE_DISK:
      return default_gpu_key();
    case ::store_daemon::DeviceType::DEVICE_TYPE_GPU:
    default:
      return default_gpu_key();
  }
}

tensorcast::store::DeviceKey StoreDaemonServiceImpl::resolve_device(const ::store_daemon::UnloadReplicaRequest& req) {
  using tensorcast::DeviceType;
  using tensorcast::store::DeviceKey;
  switch (req.target_device_type()) {
    case ::store_daemon::DeviceType::DEVICE_TYPE_CPU:
      return DeviceKey{DeviceType::CPU, -1, ""};
    case ::store_daemon::DeviceType::DEVICE_TYPE_DISK:
      return default_gpu_key();
    case ::store_daemon::DeviceType::DEVICE_TYPE_GPU:
    default:
      return default_gpu_key();
  }
}

tensorcast::store::ReplicaKey StoreDaemonServiceImpl::make_replica_key(const std::string& artifact_id) {
  tensorcast::store::ReplicaKey key;
  key.artifact_id = artifact_id;
  key.device = default_gpu_key();
  key.replica = 0;
  return key;
}

Status StoreDaemonServiceImpl::WaitReplicaVerification(
    grpc::ServerContext* ctx,
    const ::store_daemon::ReplicaVerificationRequest* req,
    ::store_daemon::ReplicaVerificationResponse* resp) {
  // Find session by replica_uuid
  auto entry = sessions_.get(req->replica_uuid());
  if (!entry.has_value()) {
    // If unknown, treat as IN_PROGRESS (client may retry)
    resp->set_status(::store_daemon::VerificationStatus::VERIFICATION_STATUS_IN_PROGRESS);
    return Status::OK;
  }
  // Compute wait timeout from request or default 30s, bounded by gRPC deadline
  std::chrono::milliseconds wait_ms(30000);
  if (req->timeout_ms() > 0)
    wait_ms = std::chrono::milliseconds(req->timeout_ms());
  const auto deadline = ctx->deadline();
  const auto now = std::chrono::system_clock::now();
  if (deadline != std::chrono::system_clock::time_point::max()) {
    if (deadline <= now) {
      wait_ms = std::chrono::milliseconds(0);
    } else {
      auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
      if (remaining < wait_ms)
        wait_ms = remaining;
    }
  }
  // Check future state
  auto status = entry->ready.wait_for(wait_ms);
  if (status == std::future_status::timeout) {
    resp->set_status(::store_daemon::VerificationStatus::VERIFICATION_STATUS_IN_PROGRESS);
    return Status::OK;
  }
  absl::Status st = entry->ready.get();
  if (st.ok()) {
    resp->set_status(::store_daemon::VerificationStatus::VERIFICATION_STATUS_PASSED);
    return Status::OK;
  }
  resp->set_status(::store_daemon::VerificationStatus::VERIFICATION_STATUS_FAILED);
  resp->set_err_msg(std::string(st.message()));
  return to_grpc_status(st);
}

Status StoreDaemonServiceImpl::LockTransportChunks(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::LockChunksRequest* req,
    ::store_daemon::LockChunksResponse* resp) {
  // Resolve ReplicaKey using default v1 device policy (GPU:0)
  auto key = make_replica_key(req->artifact_id());
  std::vector<uint32_t> indices(req->chunk_indices().begin(), req->chunk_indices().end());
  auto st = engine_->lock_chunks(key, absl::MakeSpan(indices));
  if (!st.ok())
    return to_grpc_status(st);
  std::string token = locks_.mint_token();
  locks_.put(token, key, std::move(indices));
  resp->set_lock_token(token);
  return Status::OK;
}

Status StoreDaemonServiceImpl::UnlockTransportChunks(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::UnlockChunksRequest* req,
    ::store_daemon::UnlockChunksResponse* /*resp*/) {
  auto entry = locks_.get(req->lock_token());
  if (!entry.has_value()) {
    return {StatusCode::NOT_FOUND, "unknown lock token"};
  }
  auto st = engine_->unlock_chunks(entry->key, absl::MakeSpan(entry->chunk_indices), /*copied_gpu=*/false);
  if (!st.ok())
    return to_grpc_status(st);
  locks_.erase(req->lock_token());
  return Status::OK;
}

Status StoreDaemonServiceImpl::BeginRegisterArtifact(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::BeginRegisterArtifactRequest* req,
    ::store_daemon::BeginRegisterArtifactResponse* resp) {
  tensorcast::store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = req->artifact_id();
  reg.device_id = req->device_id();
  reg.total_size_bytes = req->total_size();
  reg.enable_p2p = req->enable_p2p();
  if (req->has_ttl_ms())
    reg.ttl_ms = req->ttl_ms();
  if (req->has_tensor_index_key()) {
    reg.tensor_index_key = req->tensor_index_key();
  } else if (req->has_tensor_index_data()) {
    reg.tensor_index_data = std::string(req->tensor_index_data().data().begin(), req->tensor_index_data().data().end());
    reg.schema_version = req->tensor_index_data().schema_version();
    reg.encoding = req->tensor_index_data().encoding();
  }
  auto begin_or = engine_->begin_register_artifact(reg);
  if (!begin_or.ok())
    return to_grpc_status(begin_or.status());
  const auto& out = begin_or.value();
  resp->set_registration_id(out.registration_id);
  // Copy IPC handle bytes
  resp->set_daemon_ipc_handle(
      reinterpret_cast<const char*>(out.cuda_ipc_handle_bytes.data()), out.cuda_ipc_handle_bytes.size());
  resp->set_device_id(out.device_id);
  resp->set_size(out.size_bytes);
  return Status::OK;
}

Status StoreDaemonServiceImpl::CommitRegisteredArtifact(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::CommitRegisteredArtifactRequest* req,
    ::store_daemon::CommitRegisteredArtifactResponse* resp) {
  auto commit_or = engine_->commit_registered_artifact(req->registration_id());
  if (!commit_or.ok())
    return to_grpc_status(commit_or.status());
  const auto& out = commit_or.value();
  resp->set_registration_id(out.registration_id);
  resp->set_artifact_id(out.artifact_id);
  resp->set_device_id(out.device_id);
  resp->set_size(out.size_bytes);
  // Note: some protobuf generators reserve accessor name 'descriptor' in C++.
  // For portability, we skip populating the nested descriptor here.
  return Status::OK;
}

Status StoreDaemonServiceImpl::AbortRegisteredArtifact(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::AbortRegisteredArtifactRequest* req,
    ::store_daemon::AbortRegisteredArtifactResponse* resp) {
  auto st = engine_->abort_registered_artifact(req->registration_id());
  if (!st.ok())
    return to_grpc_status(st);
  resp->set_ok(true);
  return Status::OK;
}

void StoreDaemonServiceImpl::start_sweepers() {
  stop_.store(false);
  sweep_sessions_th_ = std::thread([this]() {
    while (!stop_.load()) {
      for (const auto& k : sessions_.keys()) {
        sessions_.remove_if_expired(k);
      }
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  });
  sweep_locks_th_ = std::thread([this]() {
    while (!stop_.load()) {
      for (const auto& tok : locks_.tokens()) {
        auto expired = locks_.remove_if_expired(tok);
        if (expired.has_value()) {
          (void)engine_->unlock_chunks(expired->key, absl::MakeSpan(expired->chunk_indices), /*copied_gpu=*/false);
        }
      }
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  });
}

void StoreDaemonServiceImpl::stop_sweepers() {
  stop_.store(true);
  if (sweep_sessions_th_.joinable())
    sweep_sessions_th_.join();
  if (sweep_locks_th_.joinable())
    sweep_locks_th_.join();
}

Status StoreDaemonServiceImpl::GetWorkerStatus(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::GetWorkerStatusRequest* /*req*/,
    ::store_daemon::GetWorkerStatusResponse* resp) {
  resp->set_is_registered(false);
  resp->set_is_healthy(true);
  resp->set_is_shutting_down(false);
  resp->set_mem_pool_total_size(engine_->get_mem_pool_size());
  resp->set_mem_pool_available_size(engine_->get_available_memory());
  auto uptime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);
  resp->set_uptime_seconds(uptime.count());
  resp->set_worker_id("");
  return Status::OK;
}

Status StoreDaemonServiceImpl::GetDetailedStatus(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::GetDetailedStatusRequest* /*req*/,
    ::store_daemon::GetDetailedStatusResponse* resp) {
  resp->set_is_registered(false);
  resp->set_is_healthy(true);
  resp->set_is_shutting_down(false);
  auto uptime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);
  resp->set_uptime_seconds(uptime.count());
  resp->set_worker_id("");

  auto* mp = resp->mutable_memory_pool_info();
  mp->set_total_size_bytes(engine_->get_mem_pool_size());
  mp->set_available_bytes(engine_->get_available_memory());
  mp->set_allocated_bytes(engine_->get_mem_pool_size() - engine_->get_available_memory());
  mp->set_allocated_chunks_count(0);
  mp->set_chunk_size_bytes(engine_->get_chunk_size());

  for (const auto& info : engine_->get_all_replicas_info()) {
    if (info.gpu_state != tensorcast::store::MemoryLocation::NONE) {
      auto* gpu = resp->add_gpu_devices();
      gpu->set_device_id(info.gpu_device_id);
      gpu->set_device_uuid(info.gpu_device_uuid);
      gpu->set_total_memory_bytes(0);
      gpu->set_free_memory_bytes(0);
      gpu->set_used_memory_bytes(0);
      auto* r = gpu->add_loaded_replicas();
      r->set_artifact_id(info.artifact_id);
      r->set_artifact_size_bytes(info.size_bytes);
      r->set_location(::store_daemon::MemoryLocation::MEMORY_LOCATION_GPU);
      r->set_loaded_timestamp(
          std::chrono::duration_cast<std::chrono::seconds>(info.load_time.time_since_epoch()).count());
      r->set_last_access_timestamp(
          std::chrono::duration_cast<std::chrono::seconds>(info.last_access_time.time_since_epoch()).count());
      r->add_replica_uuids("");
      r->set_is_registered_for_comm(info.is_registered_for_comm);
    } else if (info.cpu_state != tensorcast::store::MemoryLocation::NONE) {
      auto* r = resp->add_cpu_replicas();
      r->set_artifact_id(info.artifact_id);
      r->set_artifact_size_bytes(info.size_bytes);
      r->set_location(::store_daemon::MemoryLocation::MEMORY_LOCATION_PAGEABLE_CPU);
      r->set_loaded_timestamp(
          std::chrono::duration_cast<std::chrono::seconds>(info.load_time.time_since_epoch()).count());
      r->set_last_access_timestamp(
          std::chrono::duration_cast<std::chrono::seconds>(info.last_access_time.time_since_epoch()).count());
      r->add_replica_uuids("");
      r->set_is_registered_for_comm(info.is_registered_for_comm);
    }
  }

  resp->mutable_communication_info()->set_enabled(true);
  resp->set_total_replicas_loaded(resp->gpu_devices_size());
  resp->set_total_artifact_size_bytes(0);
  resp->set_storage_path("");
  resp->set_num_worker_threads(0);
  return Status::OK;
}

Status StoreDaemonServiceImpl::GetLoadedReplicas(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::GetLoadedReplicasRequest* req,
    ::store_daemon::GetLoadedReplicasResponse* resp) {
  int32_t total = 0;
  uint64_t total_bytes = 0;
  for (const auto& info : engine_->get_all_replicas_info()) {
    if (req->has_artifact_id_filter() && info.artifact_id != req->artifact_id_filter())
      continue;
    if (req->has_device_id_filter() && info.gpu_device_id != req->device_id_filter())
      continue;
    auto* out = resp->add_replicas();
    out->set_artifact_id(info.artifact_id);
    out->set_device_id(info.gpu_device_id);
    out->set_ref_count(0);
    out->set_size_bytes(info.size_bytes);
    out->set_keep_for_global(false);
    out->set_last_access_timestamp(
        std::chrono::duration_cast<std::chrono::seconds>(info.last_access_time.time_since_epoch()).count());
    total++;
    total_bytes += info.size_bytes;
  }
  resp->set_total_replicas(total);
  resp->set_total_size_bytes(total_bytes);
  return Status::OK;
}

} // namespace tensorcast::daemon
