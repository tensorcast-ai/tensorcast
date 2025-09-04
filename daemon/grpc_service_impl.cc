// Copyright (c) 2025, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <nlohmann/json.hpp>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "core/common/otel/grpc_propagation.h"
#include "core/store/device_registry.h"
#include "core/store/device_types.h"
#include "core/store/loading/loading_spec.h"
#include "daemon/status_utils.h"
#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"

namespace tensorcast::daemon {

static inline bool tc_otel_truthy(const char* v) {
  if (!v)
    return false;
  std::string s(v);
  for (auto& c : s)
    c = static_cast<char>(::tolower(c));
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

Status StoreDaemonServiceImpl::MaterializeReplica(
    grpc::ServerContext* ctx,
    const v1::MaterializeReplicaRequest* req,
    v1::MaterializeReplicaResponse* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = tensorcast::common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/MaterializeReplica", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "MaterializeReplica");
  // Always attach artifact_id for correlation as requested
  if (req->has_artifact_id() && !req->artifact_id().empty()) {
    span->SetAttribute("tc.artifact.id", req->artifact_id());
  }
  // Avoid other high-cardinality attributes unless explicitly enabled
  const bool allow_hc = tc_otel_truthy(std::getenv("TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS"));
  if (allow_hc) {
    if (req->has_disk_path() && !req->disk_path().empty()) {
      span->SetAttribute("tc.disk.path", req->disk_path());
    }
    // device_uuid is high-cardinality; attach only when allowed
    span->SetAttribute("tc.device.uuid", req->device_uuid());
  }
  span->SetAttribute("tc.size.bytes", static_cast<int64_t>(req->size_bytes()));
  using v1::MaterializeReplicaStatus;

  // Reject new materialization while shutting down to align with Python daemon
  if (is_shutting_down_.load()) {
    resp->set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  // Validate one-of inputs: exactly one of artifact_id or disk_path
  const bool has_artifact = req->has_artifact_id() && !req->artifact_id().empty();
  const bool has_disk = req->has_disk_path() && !req->disk_path().empty();
  if (has_artifact == has_disk) {
    return {StatusCode::INVALID_ARGUMENT, "Exactly one of artifact_id or disk_path must be provided"};
  }

  // Build DeviceKey
  const auto device = resolve_device(*req);

  // Build hints
  store::loading::MaterializeHints hints;
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
    // Initialize verification registry entry and enqueue a background task to
    // update status to PASSED/FAILED after the ready_future resolves.
    set_verif_status(req->replica_uuid(), v1::VerificationStatus::VERIFICATION_STATUS_IN_PROGRESS);
    {
      absl::MutexLock l(&bg_tasks_mu_);
      verif_tasks_.push_back(VerifTask{req->replica_uuid(), handle.ready_future});
    }
  }
  // Track initial PID reference and keep_for_global if provided
  if (req->pid() > 0) {
    bool keep = req->keep_for_global();
    refs_.add_ref(handle.replica_key, req->pid(), keep);
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
    const v1::ConfirmReplicaRequest* req,
    v1::ConfirmReplicaResponse* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/ConfirmReplica", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "ConfirmReplica");
  if (tc_otel_truthy(std::getenv("TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS"))) {
    span->SetAttribute("tc.disk.path", req->disk_path());
  }
  span->SetAttribute("tc.device.id", static_cast<int64_t>(req->target_device_type()));
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
    grpc::ServerContext* ctx,
    const v1::UnloadReplicaRequest* req,
    v1::UnloadReplicaResponse* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/UnloadReplica", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "UnloadReplica");
  if (tc_otel_truthy(std::getenv("TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS"))) {
    if (!req->disk_path().empty())
      span->SetAttribute("tc.disk.path", req->disk_path());
    if (req->has_pid())
      span->SetAttribute("tc.pid", static_cast<int64_t>(req->pid()));
  }
  resp->set_disk_path(req->disk_path());
  // Python parity: DISK-target unload is a no-op idempotent success
  if (req->target_device_type() == v1::DeviceType::DEVICE_TYPE_DISK) {
    resp->set_code(0);
    return Status::OK;
  }
  // Two paths:
  // 1) If replica_uuid present and known -> use session key
  // 2) Else, fall back to disk_path + target device to form a ReplicaKey for idempotent unload
  store::loading::ReplicaKey key;
  bool have_key = false;
  if (!req->replica_uuid().empty()) {
    auto entry = sessions_.get(req->replica_uuid());
    if (entry.has_value()) {
      key = entry->key;
      have_key = true;
    }
  }
  if (!have_key) {
    if (!req->disk_path().empty()) {
      key.artifact_id = req->disk_path();
      key.device = resolve_device(*req);
      key.replica = 0;
      have_key = true;
    } else {
      // Idempotent success when no identification provided
      resp->set_code(0);
      return Status::OK;
    }
  }
  // Drop PID ref if provided; if refs remain, skip unload per parity
  if (req->has_pid()) {
    refs_.drop_ref(key, req->pid());
    if (refs_.ref_count(key) > 0) {
      resp->set_code(0);
      return Status::OK;
    }
  }
  const int rc = engine_->unload_replica(key);
  if (rc == 0) {
    if (!req->replica_uuid().empty()) {
      sessions_.erase(req->replica_uuid());
    }
    resp->set_code(0);
    return Status::OK;
  }
  resp->set_code(1);
  return {StatusCode::INTERNAL, absl::StrFormat("unload_replica() returned %d", rc)};
}

Status StoreDaemonServiceImpl::ClearMem(
    grpc::ServerContext* ctx,
    const v1::ClearMemRequest* /*req*/,
    v1::ClearMemResponse* /*resp*/) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/ClearMem", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "ClearMem");
  const int rc = engine_->clear_mem();
  if (rc == 0)
    return Status::OK;
  return {StatusCode::INTERNAL, absl::StrFormat("clear_mem() returned %d", rc)};
}

Status StoreDaemonServiceImpl::GetServerConfig(
    grpc::ServerContext* ctx,
    const v1::GetServerConfigRequest* /*req*/,
    v1::GetServerConfigResponse* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/GetServerConfig", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "GetServerConfig");
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

tensorcast::store::DeviceKey StoreDaemonServiceImpl::resolve_device(const v1::MaterializeReplicaRequest& req) {
  using tensorcast::DeviceType;
  using tensorcast::store::DeviceKey;
  if (!req.device_uuid().empty()) {
    DeviceKey key{DeviceType::GPU, 0, req.device_uuid()};
    return tensorcast::store::DeviceRegistry::instance().normalize(key);
  }
  switch (req.target_device_type()) {
    case v1::DeviceType::DEVICE_TYPE_CPU:
      return DeviceKey{DeviceType::CPU, -1, ""};
    case v1::DeviceType::DEVICE_TYPE_DISK:
      // Treat as ingest-from-disk to default GPU for v1 parity
      return default_gpu_key();
    case v1::DeviceType::DEVICE_TYPE_GPU:
    default:
      return default_gpu_key();
  }
}

tensorcast::store::DeviceKey StoreDaemonServiceImpl::resolve_device(const v1::ConfirmReplicaRequest& req) {
  using tensorcast::DeviceType;
  using tensorcast::store::DeviceKey;
  switch (req.target_device_type()) {
    case v1::DeviceType::DEVICE_TYPE_CPU:
      return DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
    case v1::DeviceType::DEVICE_TYPE_DISK:
      return default_gpu_key();
    case v1::DeviceType::DEVICE_TYPE_GPU:
    default:
      return default_gpu_key();
  }
}

tensorcast::store::DeviceKey StoreDaemonServiceImpl::resolve_device(const v1::UnloadReplicaRequest& req) {
  using tensorcast::DeviceType;
  using tensorcast::store::DeviceKey;
  switch (req.target_device_type()) {
    case v1::DeviceType::DEVICE_TYPE_CPU:
      return DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
    case v1::DeviceType::DEVICE_TYPE_DISK:
      return default_gpu_key();
    case v1::DeviceType::DEVICE_TYPE_GPU:
    default:
      return default_gpu_key();
  }
}

store::loading::ReplicaKey StoreDaemonServiceImpl::make_replica_key(const std::string& artifact_id) {
  store::loading::ReplicaKey key;
  key.artifact_id = artifact_id;
  key.device = default_gpu_key();
  key.replica = 0;
  return key;
}

Status StoreDaemonServiceImpl::WaitReplicaVerification(
    grpc::ServerContext* ctx,
    const v1::WaitReplicaVerificationRequest* req,
    v1::WaitReplicaVerificationResponse* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/WaitReplicaVerification", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "WaitReplicaVerification");
  if (tc_otel_truthy(std::getenv("TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS"))) {
    if (!req->replica_uuid().empty())
      span->SetAttribute("tc.replica.id", req->replica_uuid());
  }
  // Consult verification registry (populated by MaterializeReplica) to see
  // if we already have a terminal status. If so, return immediately; otherwise
  // proceed to wait on the readiness future with bounded timeout.
  std::optional<v1::VerificationStatus> known_status;
  std::string known_err;
  {
    absl::MutexLock l(&verif_mu_);
    auto it = verif_.find(req->replica_uuid());
    if (it != verif_.end()) {
      known_status = it->second.status;
      known_err = it->second.err;
      if (*known_status == v1::VerificationStatus::VERIFICATION_STATUS_PASSED ||
          *known_status == v1::VerificationStatus::VERIFICATION_STATUS_FAILED) {
        resp->set_status(*known_status);
        if (!known_err.empty())
          resp->set_err_msg(known_err);
        return Status::OK;
      }
    }
  }
  // If no session, treat as unknown UUID
  auto entry = sessions_.get(req->replica_uuid());
  if (!entry.has_value()) {
    resp->set_status(v1::VerificationStatus::VERIFICATION_STATUS_UNSPECIFIED);
    return Status::OK;
  }
  // Bounded wait
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
  auto st_wait = entry->ready.wait_for(wait_ms);
  if (st_wait == std::future_status::timeout) {
    return {StatusCode::DEADLINE_EXCEEDED, "verification wait timeout"};
  }
  absl::Status st = entry->ready.get();
  if (st.ok()) {
    resp->set_status(v1::VerificationStatus::VERIFICATION_STATUS_PASSED);
    return Status::OK;
  }
  resp->set_status(v1::VerificationStatus::VERIFICATION_STATUS_FAILED);
  resp->set_err_msg(std::string(st.message()));
  return to_grpc_status(st);
}

Status StoreDaemonServiceImpl::LockTransportChunks(
    grpc::ServerContext* ctx,
    const v1::LockTransportChunksRequest* req,
    v1::LockTransportChunksResponse* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/LockTransportChunks", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "LockTransportChunks");
  span->SetAttribute("tc.artifact.id", req->artifact_id());
  if (req->has_device_id())
    span->SetAttribute("tc.device.id", static_cast<int64_t>(req->device_id()));
  // Resolve ReplicaKey, prefer explicit device_id then infer residency
  auto key = make_replica_key(req->artifact_id());
  if (req->has_device_id()) {
    key.device = tensorcast::store::DeviceRegistry::instance().gpu_key(req->device_id());
  } else {
    int unique_gpu_device = -2; // -2=unknown, -1=none, >=0=single device
    for (const auto& info : engine_->get_all_replicas_info()) {
      if (info.artifact_id == req->artifact_id() && info.gpu_state == common::memory::MemoryLocation::GPU) {
        if (unique_gpu_device == -2) {
          unique_gpu_device = info.gpu_device_id;
        } else if (unique_gpu_device != info.gpu_device_id) {
          unique_gpu_device = -3; // ambiguous
          break;
        }
      }
    }
    if (unique_gpu_device >= 0) {
      key.device = tensorcast::store::DeviceRegistry::instance().gpu_key(unique_gpu_device);
    } else if (unique_gpu_device == -3) {
      return {StatusCode::INVALID_ARGUMENT, "ambiguous artifact residency across multiple GPUs; device_id required"};
    }
  }
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
    grpc::ServerContext* ctx,
    const v1::UnlockTransportChunksRequest* req,
    v1::UnlockTransportChunksResponse* /*resp*/) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/UnlockTransportChunks", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "UnlockTransportChunks");
  if (tc_otel_truthy(std::getenv("TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS"))) {
    span->SetAttribute("tc.lock.token", req->lock_token());
  }
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
    grpc::ServerContext* ctx,
    const v1::BeginRegisterArtifactRequest* req,
    v1::BeginRegisterArtifactResponse* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/BeginRegisterArtifact", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "BeginRegisterArtifact");
  span->SetAttribute("tc.artifact.id", req->artifact_id());
  span->SetAttribute("tc.device.id", static_cast<int64_t>(req->device_id()));
  span->SetAttribute("tc.size.bytes", static_cast<int64_t>(req->total_size()));
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
    grpc::ServerContext* ctx,
    const v1::CommitRegisteredArtifactRequest* req,
    v1::CommitRegisteredArtifactResponse* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/CommitRegisteredArtifact", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "CommitRegisteredArtifact");
  if (tc_otel_truthy(std::getenv("TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS"))) {
    span->SetAttribute("tc.registration.id", req->registration_id());
  }
  auto commit_or = engine_->commit_registered_artifact(req->registration_id());
  if (!commit_or.ok())
    return to_grpc_status(commit_or.status());
  const auto& out = commit_or.value();
  resp->set_registration_id(out.registration_id);
  resp->set_artifact_id(out.artifact_id);
  resp->set_device_id(out.device_id);
  resp->set_size(out.size_bytes);
  // Populate RFC-0007 content-addressed descriptor for parity with Python daemon
  // and client expectations.
  auto* desc = resp->mutable_descriptor_();
  desc->set_artifact_id(out.artifact_id);
  desc->set_index_multihash(out.index_multihash);
  desc->set_data_multihash(out.data_multihash);
  desc->set_schema_version(out.schema_version);
  desc->set_encoding(out.encoding);
  desc->set_total_size(out.size_bytes);
  return Status::OK;
}

Status StoreDaemonServiceImpl::AbortRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v1::AbortRegisteredArtifactRequest* req,
    v1::AbortRegisteredArtifactResponse* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/AbortRegisteredArtifact", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "AbortRegisteredArtifact");
  if (tc_otel_truthy(std::getenv("TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS"))) {
    span->SetAttribute("tc.registration.id", req->registration_id());
  }
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
  // Verification and auto-registration sweeper: updates verification status and
  // registers disk-ingested replicas with Global Store once ready.
  verif_sweeper_th_ = std::thread([this]() {
    using namespace std::chrono_literals;
    while (!stop_.load()) {
      // Collect completed verification tasks
      std::vector<std::pair<std::string, absl::Status>> verif_done;
      {
        absl::MutexLock l(&bg_tasks_mu_);
        for (auto it = verif_tasks_.begin(); it != verif_tasks_.end();) {
          if (it->ready.wait_for(0ms) == std::future_status::ready) {
            verif_done.emplace_back(it->uuid, it->ready.get());
            it = verif_tasks_.erase(it);
          } else {
            ++it;
          }
        }
      }
      for (auto& p : verif_done) {
        const std::string& uuid = p.first;
        const absl::Status& st = p.second;
        if (st.ok()) {
          set_verif_status(uuid, v1::VerificationStatus::VERIFICATION_STATUS_PASSED);
        } else {
          set_verif_status(uuid, v1::VerificationStatus::VERIFICATION_STATUS_FAILED, std::string(st.message()));
        }
      }

      // Collect completed auto-registration tasks
      std::vector<AutoRegTask> reg_ready;
      {
        absl::MutexLock l(&bg_tasks_mu_);
        for (auto it = auto_reg_tasks_.begin(); it != auto_reg_tasks_.end();) {
          if (it->ready.wait_for(0ms) == std::future_status::ready) {
            reg_ready.push_back(*it);
            it = auto_reg_tasks_.erase(it);
          } else {
            ++it;
          }
        }
      }
      for (auto& task : reg_ready) {
        // Wait for load completion result (ignore status; parity with previous behavior)
        (void)task.ready.get();
        // Try to read descriptor for mi2 ID
        std::string mi2_id;
        try {
          std::filesystem::path desc_path = std::filesystem::path(task.disk_path) / "artifact_descriptor.json";
          if (std::filesystem::exists(desc_path)) {
            std::ifstream f(desc_path);
            if (f.is_open()) {
              nlohmann::json j;
              f >> j;
              if (j.contains("artifact_id") && j["artifact_id"].is_string()) {
                mi2_id = j["artifact_id"].get<std::string>();
              } else if (
                  j.contains("index_multihash") && j.contains("data_multihash") && j["index_multihash"].is_string() &&
                  j["data_multihash"].is_string()) {
                mi2_id = absl::StrCat(
                    "mi2:", j["index_multihash"].get<std::string>(), ":", j["data_multihash"].get<std::string>());
              }
            }
          }
        } catch (...) {
          // Ignore descriptor parsing errors; fall back to key.artifact_id
        }
        auto st = engine_->register_replica_with_global_store(task.key, mi2_id);
        if (!st.ok()) {
          VLOG(1) << "Auto-register disk load failed: " << st;
        }
      }

      std::this_thread::sleep_for(500ms);
    }
  });
  // PID watcher: drop dead PID refs to avoid leaked references pinning memory
  pid_watcher_th_ = std::thread([this]() {
    while (!stop_.load()) {
      auto keys = refs_.keys();
      for (const auto& key : keys) {
        auto plist = refs_.pids(key);
        for (int32_t pid : plist) {
          // Check /proc/<pid>
          std::string proc_path = absl::StrCat("/proc/", pid);
          if (::access(proc_path.c_str(), F_OK) != 0) {
            refs_.drop_ref(key, pid);
          }
        }
      }
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  });
  // Optional periodic eviction policy: unload least-recently-used GPU replicas
  // on devices where used memory exceeds the configured fraction. Disabled by default.
  // Controlled via environment variables:
  //  - TC_DAEMON_ENABLE_PERIODIC_EVICTION: truthy to enable (default: false)
  //  - TC_DAEMON_GPU_MEMORY_LIMIT_FRACTION: threshold fraction (default: 0.90)
  //  - TC_DAEMON_EVICTION_CHECK_INTERVAL_MS: check interval in ms (default: 1000)
  const bool enable_periodic_eviction = tc_otel_truthy(std::getenv("TC_DAEMON_ENABLE_PERIODIC_EVICTION"));
  if (enable_periodic_eviction) {
    auto env_double = [](const char* name, double defval) -> double {
      if (const char* v = std::getenv(name)) {
        char* end = nullptr;
        double d = std::strtod(v, &end);
        if (end != v)
          return d;
      }
      return defval;
    };
    auto env_int = [](const char* name, int defval) -> int {
      if (const char* v = std::getenv(name)) {
        char* end = nullptr;
        int64_t x = std::strtol(v, &end, 10);
        if (end != v)
          return static_cast<int>(x);
      }
      return defval;
    };
    const double gpu_memory_limit_fraction = env_double("TC_DAEMON_GPU_MEMORY_LIMIT_FRACTION", 0.90);
    const int eviction_check_interval_ms = env_int("TC_DAEMON_EVICTION_CHECK_INTERVAL_MS", 1000);
    eviction_th_ = std::thread([this, gpu_memory_limit_fraction, eviction_check_interval_ms]() {
      using namespace std::chrono_literals;
      while (!stop_.load()) {
        // Iterate all GPU devices; if usage exceeds threshold, evict LRU replicas with no refs and not keep_for_global
        const int num_gpus = engine_->get_num_gpus();
        for (int dev = 0; dev < num_gpus; ++dev) {
          auto tot_or = engine_->get_device_total_memory(dev);
          auto free_or = engine_->get_device_free_memory(dev);
          if (!tot_or.ok() || !free_or.ok())
            continue;
          const auto total = static_cast<double>(*tot_or);
          const auto used = static_cast<double>(*tot_or - *free_or);
          if (total <= 0.0)
            continue;
          double ratio = used / total;
          if (ratio <= gpu_memory_limit_fraction)
            continue;

          // Build LRU list of candidates on this device
          struct Cand {
            store::loading::ReplicaKey key;
            std::chrono::time_point<std::chrono::system_clock> last_access;
            size_t size;
          };
          std::vector<Cand> cands;
          for (const auto& info : engine_->get_all_replicas_info()) {
            if (info.gpu_state == common::memory::MemoryLocation::NONE)
              continue;
            if (info.gpu_device_id != dev)
              continue;
            store::loading::ReplicaKey key{
                .artifact_id = info.artifact_id,
                .device = tensorcast::store::DeviceRegistry::instance().gpu_key(dev),
                .replica = 0};
            if (refs_.ref_count(key) > 0 || refs_.keep_for_global(key))
              continue;
            cands.push_back(
                Cand{.key = key, .last_access = info.last_access_time, .size = static_cast<size_t>(info.size_bytes)});
          }
          std::ranges::sort(cands, [](const Cand& a, const Cand& b) { return a.last_access < b.last_access; });

          // Evict until under threshold or no candidates
          for (const auto& c : cands) {
            if (ratio <= gpu_memory_limit_fraction)
              break;
            (void)engine_->unload_replica(c.key);
            // Recompute ratio after unload attempt
            auto f2 = engine_->get_device_free_memory(dev);
            if (!f2.ok())
              break;
            const double used2 = static_cast<double>(*tot_or - *f2);
            ratio = used2 / total;
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(eviction_check_interval_ms));
      }
    });
  }
}

void StoreDaemonServiceImpl::stop_sweepers() {
  stop_.store(true);
  if (sweep_sessions_th_.joinable())
    sweep_sessions_th_.join();
  if (sweep_locks_th_.joinable())
    sweep_locks_th_.join();
  if (pid_watcher_th_.joinable())
    pid_watcher_th_.join();
  if (verif_sweeper_th_.joinable())
    verif_sweeper_th_.join();
  if (eviction_th_.joinable())
    eviction_th_.join();
}

Status StoreDaemonServiceImpl::GetWorkerStatus(
    grpc::ServerContext* ctx,
    const v1::GetWorkerStatusRequest* /*req*/,
    v1::GetWorkerStatusResponse* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/GetWorkerStatus", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "GetWorkerStatus");
  resp->set_is_registered(is_registered());
  resp->set_is_healthy(true);
  resp->set_is_shutting_down(is_shutting_down_.load());
  resp->set_mem_pool_total_size(engine_->get_mem_pool_size());
  resp->set_mem_pool_available_size(engine_->get_available_memory());
  auto uptime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);
  resp->set_uptime_seconds(uptime.count());
  resp->set_worker_id(is_registered() ? worker_id() : "");
  return Status::OK;
}

Status StoreDaemonServiceImpl::GetDetailedStatus(
    grpc::ServerContext* ctx,
    const v1::GetDetailedStatusRequest* /*req*/,
    v1::GetDetailedStatusResponse* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/GetDetailedStatus", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "GetDetailedStatus");
  resp->set_is_registered(is_registered());
  resp->set_is_healthy(true);
  resp->set_is_shutting_down(is_shutting_down_.load());
  auto uptime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);
  resp->set_uptime_seconds(uptime.count());
  resp->set_worker_id(is_registered() ? worker_id() : "");

  auto* mp = resp->mutable_memory_pool_info();
  mp->set_total_size_bytes(engine_->get_mem_pool_size());
  mp->set_available_bytes(engine_->get_available_memory());
  mp->set_allocated_bytes(engine_->get_mem_pool_size() - engine_->get_available_memory());
  mp->set_allocated_chunks_count(0);
  mp->set_chunk_size_bytes(engine_->get_chunk_size());

  // Aggregate by GPU device to avoid duplicate device entries. Also compute totals.
  uint64_t total_bytes = 0;
  int32_t total_replicas = 0;
  struct GpuAgg {
    v1::GpuDeviceInfo* out;
    bool mem_filled{false};
  };
  absl::flat_hash_map<int, GpuAgg> gpu_map;

  for (const auto& info : engine_->get_all_replicas_info()) {
    if (info.gpu_state != common::memory::MemoryLocation::NONE) {
      auto it = gpu_map.find(info.gpu_device_id);
      if (it == gpu_map.end()) {
        auto* gpu = resp->add_gpu_devices();
        gpu->set_device_id(info.gpu_device_id);
        gpu->set_device_uuid(info.gpu_device_uuid);
        it = gpu_map.emplace(info.gpu_device_id, GpuAgg{gpu, false}).first;
      }
      // Populate GPU memory totals once per device
      if (!it->second.mem_filled) {
        size_t total_mem = 0;
        size_t free_mem = 0;
        if (auto t = engine_->get_device_total_memory(info.gpu_device_id); t.ok()) {
          total_mem = *t;
        }
        if (auto f = engine_->get_device_free_memory(info.gpu_device_id); f.ok()) {
          free_mem = *f;
        }
        it->second.out->set_total_memory_bytes(static_cast<uint64_t>(total_mem));
        it->second.out->set_free_memory_bytes(static_cast<uint64_t>(free_mem));
        uint64_t used = (total_mem > free_mem) ? static_cast<uint64_t>(total_mem - free_mem) : 0ULL;
        it->second.out->set_used_memory_bytes(used);
        it->second.mem_filled = true;
      }
      auto* r = it->second.out->add_loaded_replicas();
      r->set_artifact_id(info.artifact_id);
      r->set_artifact_size_bytes(info.size_bytes);
      r->set_location(v1::MemoryLocation::MEMORY_LOCATION_GPU);
      r->set_loaded_timestamp(
          std::chrono::duration_cast<std::chrono::seconds>(info.load_time.time_since_epoch()).count());
      r->set_last_access_timestamp(
          std::chrono::duration_cast<std::chrono::seconds>(info.last_access_time.time_since_epoch()).count());
      r->add_replica_uuids("");
      r->set_is_registered_for_comm(info.is_registered_for_comm);
      total_replicas += 1;
      total_bytes += info.size_bytes;
    }
    if (info.cpu_state != common::memory::MemoryLocation::NONE) {
      auto* r = resp->add_cpu_replicas();
      r->set_artifact_id(info.artifact_id);
      r->set_artifact_size_bytes(info.size_bytes);
      r->set_location(v1::MemoryLocation::MEMORY_LOCATION_PAGEABLE_CPU);
      r->set_loaded_timestamp(
          std::chrono::duration_cast<std::chrono::seconds>(info.load_time.time_since_epoch()).count());
      r->set_last_access_timestamp(
          std::chrono::duration_cast<std::chrono::seconds>(info.last_access_time.time_since_epoch()).count());
      r->add_replica_uuids("");
      r->set_is_registered_for_comm(info.is_registered_for_comm);
      total_replicas += 1;
      total_bytes += info.size_bytes;
    }
  }

  bool any_comm = false;
  for (const auto& info : engine_->get_all_replicas_info()) {
    any_comm = any_comm || info.is_registered_for_comm;
  }
  resp->mutable_communication_info()->set_enabled(any_comm);
  resp->set_total_replicas_loaded(total_replicas);
  resp->set_total_artifact_size_bytes(static_cast<int64_t>(total_bytes));
  resp->set_storage_path("");
  resp->set_num_worker_threads(0);
  return Status::OK;
}

void StoreDaemonServiceImpl::set_verif_status(const std::string& uuid, v1::VerificationStatus st, std::string err) {
  absl::MutexLock l(&verif_mu_);
  verif_[uuid] = VerifEntry{.status = st, .err = std::move(err)};
}

// Legacy GetLoadedReplicas removed; use V2

Status StoreDaemonServiceImpl::GetLoadedReplicasV2(
    grpc::ServerContext* ctx,
    const v1::GetLoadedReplicasV2Request* req,
    v1::GetLoadedReplicasV2Response* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/GetLoadedReplicasV2", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "GetLoadedReplicasV2");
  if (tc_otel_truthy(std::getenv("TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS"))) {
    if (req->has_artifact_id_filter())
      span->SetAttribute("tc.artifact.filter", req->artifact_id_filter());
  }

  // Collect all matching entries
  struct Entry {
    std::string artifact_id;
    int device_id;
    int32_t ref_count;
    std::vector<int32_t> pids;
    uint64_t size_bytes;
    bool keep_for_global;
    int64_t last_access_ts;
  };
  std::vector<Entry> entries;
  entries.reserve(64);
  for (const auto& info : engine_->get_all_replicas_info()) {
    int device_id = -1;
    if (info.gpu_state != common::memory::MemoryLocation::NONE) {
      device_id = info.gpu_device_id;
    }
    if (req->has_artifact_id_filter() && info.artifact_id.find(req->artifact_id_filter()) == std::string::npos)
      continue;
    if (req->has_device_id_filter() && device_id != req->device_id_filter())
      continue;

    store::loading::ReplicaKey key;
    key.artifact_id = info.artifact_id;
    key.device = (device_id >= 0) ? tensorcast::store::DeviceRegistry::instance().gpu_key(device_id)
                                  : tensorcast::store::DeviceKey{tensorcast::DeviceType::CPU, -1, ""};
    key.replica = 0;

    Entry e;
    e.artifact_id = info.artifact_id;
    e.device_id = device_id;
    e.ref_count = static_cast<int32_t>(refs_.ref_count(key));
    for (int32_t pid : refs_.pids(key))
      e.pids.push_back(pid);
    e.size_bytes = info.size_bytes;
    e.keep_for_global = refs_.keep_for_global(key);
    e.last_access_ts =
        std::chrono::duration_cast<std::chrono::seconds>(info.last_access_time.time_since_epoch()).count();
    entries.push_back(std::move(e));
  }

  const uint32_t page_size =
      req->has_pagination() && req->pagination().has_page_size() ? req->pagination().page_size() : 100;
  uint32_t start = 0;
  if (req->has_pagination() && req->pagination().has_page_token()) {
    try {
      start = static_cast<uint32_t>(std::stoul(req->pagination().page_token()));
    } catch (...) {
      start = 0;
    }
  }
  const uint32_t end = std::min<uint32_t>(start + page_size, static_cast<uint32_t>(entries.size()));
  for (uint32_t i = start; i < end; ++i) {
    const auto& e = entries[i];
    auto* out = resp->add_replicas();
    out->set_artifact_id(e.artifact_id);
    out->set_device_id(e.device_id);
    out->set_ref_count(e.ref_count);
    for (int32_t pid : e.pids)
      out->add_pids(pid);
    out->set_size_bytes(static_cast<int64_t>(e.size_bytes));
    out->set_keep_for_global(e.keep_for_global);
    // Populate standard timestamp
    auto* ts = out->mutable_last_access_ts();
    ts->set_seconds(e.last_access_ts);
    ts->set_nanos(0);
  }
  auto* pi = resp->mutable_page_info();
  if (end < entries.size()) {
    pi->set_next_page_token(std::to_string(end));
  } else {
    pi->set_next_page_token("");
  }
  pi->set_total_size(static_cast<uint32_t>(entries.size()));
  return Status::OK;
}

} // namespace tensorcast::daemon
