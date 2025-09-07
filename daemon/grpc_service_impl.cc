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
#include "core/common/artifact_hash.h"
#include "core/common/otel/grpc_propagation.h"
#include "core/store/components/global_store_client.h"
#include "core/store/device_registry.h"
#include "core/store/device_types.h"
#include "core/store/loader/segment_plan_source.h"
#include "core/store/loader/source_hash.h"
#include "core/store/loading/loading_spec.h"
#include "daemon/status_utils.h"
#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/metrics/provider.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"

namespace tensorcast::daemon {

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
  const bool allow_hc = opts_.allow_high_card_attrs;
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
      verif_tasks_.push_back(VerifTask{.uuid = req->replica_uuid(), .ready = handle.ready_future});
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
  if (opts_.allow_high_card_attrs) {
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

// RFC-0014: Materialize by key using Global Store mapping
Status StoreDaemonServiceImpl::MaterializeByKey(
    grpc::ServerContext* ctx,
    const v1::MaterializeByKeyRequest* req,
    v1::MaterializeByKeyResponse* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/MaterializeByKey", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "MaterializeByKey");
  span->SetAttribute("tc.key", req->key());

  using v1::MaterializeReplicaStatus;
  if (is_shutting_down_.load()) {
    resp->set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req->key().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "key is required"};
  }

  // Resolve key via a transient GlobalStoreClient (avoids exposing engine internals).
  store::components::GlobalStoreClientConfig cfg;
  auto gs = std::make_unique<store::components::GlobalStoreClient>(cfg);
  auto st = gs->initialize();
  if (!st.ok()) {
    resp->set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(st);
  }
  auto mapping_or = gs->resolve_key_mapping(req->key());
  if (!mapping_or.ok()) {
    resp->set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(mapping_or.status());
  }
  auto mapping = *mapping_or;
  span->SetAttribute("tc.artifact.id", mapping.artifact_id);

  const auto device = store::DeviceKey{.type = store::DeviceType::GPU, .ordinal = req->device_id(), .uuid = ""};
  store::loading::MaterializeHints hints;
  if (req->pinned_allocation_timeout_ms() > 0) {
    hints.pinned_timeout = std::chrono::milliseconds(req->pinned_allocation_timeout_ms());
  }
  hints.artifact_id = mapping.artifact_id;
  // RFC-0014: provide optional disk fallback when mapping carries disk_path
  if (!mapping.disk_path.empty()) {
    hints.disk_path = mapping.disk_path;
  }

  auto result = engine_->materialize_replica(device, store::StoreEngine::MaterializeMode::AUTO, hints);
  if (!result.ok()) {
    resp->set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(result.status());
  }
  const auto& handle = *result;
  // RFC-0014: if client supplied a replica_uuid, register a session for Confirm/Verification
  if (!req->replica_uuid().empty()) {
    sessions_.put(req->replica_uuid(), handle.replica_key, handle.ready_future);
    set_verif_status(req->replica_uuid(), v1::VerificationStatus::VERIFICATION_STATUS_IN_PROGRESS);
    {
      absl::MutexLock l(&bg_tasks_mu_);
      verif_tasks_.push_back(VerifTask{.uuid = req->replica_uuid(), .ready = handle.ready_future});
    }
  }
  if (handle.cuda_ipc_handle.is_valid()) {
    auto* mem = resp->mutable_mem_handle();
    mem->set_cuda_ipc_handle(handle.cuda_ipc_handle.to_string());
  }
  resp->set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  resp->set_artifact_id(mapping.artifact_id);
  resp->set_used_disk_path(mapping.disk_path);
  return grpc::Status::OK;
}

// RFC-0014: Publish key mapping – lightweight wrapper to Global Store
Status StoreDaemonServiceImpl::PublishReplicaKey(
    grpc::ServerContext* ctx,
    const v1::PublishReplicaKeyRequest* req,
    v1::PublishReplicaKeyResponse* resp) {
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  auto parent_ctx = common::otel::ExtractFromServerMetadata(*ctx);
  auto ctx_token = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kServer;
  auto span = tracer->StartSpan("StoreDaemon/PublishReplicaKey", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
  span->SetAttribute("rpc.method", "PublishReplicaKey");
  span->SetAttribute("tc.key", req->key());

  if (req->key().empty() || !req->has_descriptor() || req->descriptor().artifact_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "key and descriptor.artifact_id are required"};
  }
  if (is_shutting_down_.load()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  store::components::GlobalStoreClientConfig cfg;
  auto gs = std::make_unique<store::components::GlobalStoreClient>(cfg);
  auto st = gs->initialize();
  if (!st.ok()) {
    resp->set_ok(false);
    return to_grpc_status(st);
  }
  auto up = gs->upsert_key_mapping(req->key(), req->descriptor().artifact_id(), req->disk_path());
  if (!up.ok()) {
    resp->set_ok(false);
    resp->set_conflict_reason(std::string(up.message()));
    return to_grpc_status(up);
  }
  resp->set_ok(true);
  return grpc::Status::OK;
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
  if (opts_.allow_high_card_attrs) {
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
  if (!req->replica_uuid().empty()) {
    auto entry = sessions_.get(req->replica_uuid());
    if (entry.has_value()) {
      key = entry->key;
    } else {
      // Fall through to alternate identification methods
    }
  }
  if (key.artifact_id.empty()) {
    if (!req->disk_path().empty()) {
      key.artifact_id = req->disk_path();
      key.device = resolve_device(*req);
      key.replica = 0;
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
    DeviceKey key{.type = DeviceType::GPU, .ordinal = 0, .uuid = req.device_uuid()};
    return tensorcast::store::DeviceRegistry::instance().normalize(key);
  }
  switch (req.target_device_type()) {
    case v1::DeviceType::DEVICE_TYPE_CPU:
      return DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
    case v1::DeviceType::DEVICE_TYPE_DISK:
    case v1::DeviceType::DEVICE_TYPE_GPU:
    default:
      // Treat DISK as ingest-to-default GPU for v1 parity
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
  if (opts_.allow_high_card_attrs) {
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
  }
  {
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
  if (opts_.allow_high_card_attrs) {
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
  span->SetAttribute("tc.device.id", static_cast<int64_t>(req->device_id()));
  span->SetAttribute("tc.size.bytes", static_cast<int64_t>(req->total_size()));
  tensorcast::store::StoreEngine::ArtifactRegistration reg;
  // Use a temporary placeholder identifier for pre-commit residency; replaced at Commit with mi2:...
  reg.artifact_id = absl::StrCat("mem_reg:", absl::ToUnixNanos(absl::Now()), ":", getpid());
  reg.device_id = req->device_id();
  reg.total_size_bytes = req->total_size();
  // Plan handling
  reg.enable_p2p = true;
  if (req->has_ttl_ms())
    reg.ttl_ms = req->ttl_ms();
  // Build registration meta
  RegPlan plan = RegPlan::COALESCED;
  if (req->has_dvmp())
    plan = RegPlan::DVMP;
  else if (req->has_lease())
    plan = RegPlan::LEASE;
  RegMeta meta;
  meta.plan = plan;
  meta.total_size = req->total_size();
  meta.device_id = req->device_id();
  if (req->has_ttl_ms() && req->ttl_ms() > 0) {
    meta.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(req->ttl_ms());
    meta.ttl_ms = static_cast<uint32_t>(req->ttl_ms());
  }
  if (req->has_tensor_index_key()) {
    meta.index_key_hex = req->tensor_index_key();
  } else if (req->has_tensor_index_data()) {
    meta.index_data = std::string(req->tensor_index_data().data().begin(), req->tensor_index_data().data().end());
  }

  if (plan == RegPlan::COALESCED) {
    if (req->has_tensor_index_key())
      reg.tensor_index_key = req->tensor_index_key();
    if (req->has_tensor_index_data()) {
      reg.tensor_index_data = meta.index_data;
      reg.schema_version = req->tensor_index_data().schema_version();
      reg.encoding = req->tensor_index_data().encoding();
    }
    auto begin_or = engine_->begin_register_artifact(reg);
    if (!begin_or.ok()) {
      LOG(ERROR) << "BeginRegisterArtifact failed: " << begin_or.status();
      return to_grpc_status(begin_or.status());
    }
    const auto& out = begin_or.value();
    resp->set_registration_id(out.registration_id);
    auto* hs = resp->mutable_coalesced();
    hs->set_daemon_ipc_handle(
        reinterpret_cast<const char*>(out.cuda_ipc_handle_bytes.data()), out.cuda_ipc_handle_bytes.size());
    resp->set_device_id(out.device_id);
    resp->set_total_size(out.size_bytes);
    // Metrics: begin (coalesced)
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_begin_coalesced_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_begin_coalesced_total unavailable";
    }
    {
      absl::MutexLock l(&reg_mu_);
      reg_meta_[out.registration_id] = meta;
    }
  } else if (plan == RegPlan::DVMP) {
    // Engine-backed DVMP begin: allocate DVMP/UMA CPU region
    store::StoreEngine::ArtifactRegistration a;
    a.artifact_id = absl::StrCat("mem_reg:", absl::ToUnixNanos(absl::Now()), ":", getpid());
    if (req->has_tensor_index_key())
      a.tensor_index_key = req->tensor_index_key();
    if (req->has_tensor_index_data()) {
      a.tensor_index_data = std::string(req->tensor_index_data().data().begin(), req->tensor_index_data().data().end());
      a.schema_version = req->tensor_index_data().schema_version();
      a.encoding = req->tensor_index_data().encoding();
    }
    a.device_id = req->device_id();
    a.total_size_bytes = req->total_size();
    a.enable_p2p = true;
    if (req->has_ttl_ms())
      a.ttl_ms = req->ttl_ms();
    auto begin_or = engine_->begin_register_artifact_dvmp(a);
    if (!begin_or.ok())
      return to_grpc_status(begin_or.status());
    const auto& out = begin_or.value();
    resp->set_registration_id(out.registration_id);
    auto* hs = resp->mutable_dvmp()->mutable_stream();
    hs->set_token(out.registration_id);
    resp->set_device_id(out.device_id);
    resp->set_total_size(out.size_bytes);
    // Metrics: begin (dvmp)
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_begin_dvmp_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_begin_dvmp_total unavailable";
    }
    {
      absl::MutexLock l(&reg_mu_);
      reg_meta_[out.registration_id] = meta;
    }
  }
  {
    // Lease plan stub: return empty lease handshake
    std::string reg_id = absl::StrCat("reg_", absl::ToUnixNanos(absl::Now()), "_", getpid());
    resp->set_registration_id(reg_id);
    (void)resp->mutable_lease();
    resp->set_device_id(req->device_id());
    resp->set_total_size(req->total_size());
    // Metrics: begin (lease)
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_begin_lease_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_begin_lease_total unavailable";
    }
    absl::MutexLock l(&reg_mu_);
    reg_meta_[reg_id] = meta;
  }
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
  if (opts_.allow_high_card_attrs) {
    span->SetAttribute("tc.registration.id", req->registration_id());
  }
  // Decide path by plan
  RegMeta meta;
  {
    absl::MutexLock l(&reg_mu_);
    auto it = reg_meta_.find(req->registration_id());
    if (it != reg_meta_.end())
      meta = it->second;
  }
  if (meta.plan == RegPlan::DVMP) {
    // TTL enforcement: if expired, cleanup and fail fast
    if (meta.expiry.time_since_epoch().count() > 0 && std::chrono::steady_clock::now() > meta.expiry) {
      absl::MutexLock l(&reg_mu_);
      reg_meta_.erase(req->registration_id());
      try {
        static auto meter =
            opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
        static auto counter = meter->CreateDoubleCounter("tc_register_ttl_expired_commit_total");
        counter->Add(1.0);
      } catch (...) {
        VLOG(1) << "metrics counter tc_register_ttl_expired_commit_total unavailable";
      }
      return {StatusCode::DEADLINE_EXCEEDED, "registration expired (TTL)"};
    }
    // Delegate to engine to compute descriptor and map mi2 id
    auto commit_or = engine_->commit_registered_artifact(req->registration_id());
    if (!commit_or.ok())
      return to_grpc_status(commit_or.status());
    const auto& out = commit_or.value();
    auto* desc = resp->mutable_descriptor_();
    desc->set_artifact_id(out.artifact_id);
    desc->set_index_multihash(out.index_multihash);
    desc->set_data_multihash(out.data_multihash);
    desc->set_schema_version(out.schema_version);
    desc->set_encoding(out.encoding);
    desc->set_total_size(out.size_bytes);
    // Metrics: commit (dvmp)
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_commit_dvmp_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_commit_dvmp_total unavailable";
    }
    absl::MutexLock l(&reg_mu_);
    reg_meta_.erase(req->registration_id());
    return Status::OK;
  }
  if (meta.plan == RegPlan::LEASE) {
    if (meta.expiry.time_since_epoch().count() > 0 && std::chrono::steady_clock::now() > meta.expiry) {
      absl::MutexLock l(&reg_mu_);
      reg_meta_.erase(req->registration_id());
      reg_leases_.erase(req->registration_id());
      try {
        static auto meter =
            opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
        static auto counter = meter->CreateDoubleCounter("tc_register_ttl_expired_commit_total");
        counter->Add(1.0);
      } catch (...) {
        VLOG(1) << "metrics counter tc_register_ttl_expired_commit_total unavailable";
      }
      return {StatusCode::DEADLINE_EXCEEDED, "registration expired (TTL)"};
    }
    if (meta.index_data.empty() && meta.index_key_hex.empty()) {
      return {StatusCode::INVALID_ARGUMENT, "lease commit requires index (data or key)"};
    }
    if (meta.index_data.empty()) {
      return {StatusCode::INVALID_ARGUMENT, "lease commit requires canonical index bytes (v2 json)"};
    }
    // Build SegmentPlan
    auto plan_or =
        tensorcast::store::loader::build_segment_plan_from_canonical_index_json(meta.index_data, meta.total_size, 8);
    if (!plan_or.ok())
      return to_grpc_status(plan_or.status());
    auto plan = *plan_or;
    // Copy lease segments under lock
    std::vector<LeaseSegMeta> lease_vec;
    {
      absl::MutexLock l(&reg_mu_);
      auto itl = reg_leases_.find(req->registration_id());
      if (itl == reg_leases_.end())
        return {StatusCode::FAILED_PRECONDITION, "no lease segments fed"};
      lease_vec = itl->second;
    }
    // Map IPC handles for source segments
    struct Opened {
      int device_id;
      void* ptr;
      uint64_t base;
      uint64_t len;
    };
    std::vector<Opened> opened;
    opened.reserve(lease_vec.size());
    for (const auto& seg : lease_vec) {
      cudaIpcMemHandle_t h{};
      size_t n = std::min(sizeof(h), seg.handle_bytes.size());
      if (n > 0)
        std::memcpy(&h, seg.handle_bytes.data(), n);
      void* dev_ptr = nullptr;
      if (auto st = tensorcast::cuda::open_ipc_mem_handle(&dev_ptr, h, cudaIpcMemLazyEnablePeerAccess); !st.ok()) {
        for (const auto& o : opened)
          (void)tensorcast::cuda::close_ipc_mem_handle(o.ptr);
        return to_grpc_status(st);
      }
      opened.push_back(Opened{.device_id = seg.device_id, .ptr = dev_ptr, .base = seg.base_offset, .len = seg.length});
    }
    // Begin coalesced VRAM registration in engine (internal-only)
    store::StoreEngine::ArtifactRegistration areg;
    areg.artifact_id = absl::StrCat("mem_reg:", absl::ToUnixNanos(absl::Now()), ":", getpid());
    areg.tensor_index_key = meta.index_key_hex;
    areg.tensor_index_data = meta.index_data; // canonical JSON
    areg.schema_version = "v2";
    areg.encoding = "json";
    areg.device_id = meta.device_id;
    areg.total_size_bytes = meta.total_size;
    areg.enable_p2p = true;
    auto begin2_or = engine_->begin_register_artifact(areg);
    if (!begin2_or.ok()) {
      for (const auto& o : opened)
        (void)tensorcast::cuda::close_ipc_mem_handle(o.ptr);
      return to_grpc_status(begin2_or.status());
    }
    const auto& out2 = begin2_or.value();
    // Ensure pending registration is aborted on any error prior to successful commit
    struct RegAbortGuard {
      tensorcast::store::StoreEngine* engine;
      std::string id;
      bool active{true};
      ~RegAbortGuard() {
        if (active && engine) {
          (void)engine->abort_registered_artifact(id);
        }
      }
      void release() {
        active = false;
      }
    } abort_guard{.engine = engine_.get(), .id = out2.registration_id};
    // Open IPC handle for destination coalesced VRAM
    cudaIpcMemHandle_t hdst{};
    std::memcpy(&hdst, out2.cuda_ipc_handle_bytes.data(), sizeof(hdst));
    void* dst_dev = nullptr;
    if (auto st = tensorcast::cuda::open_ipc_mem_handle(&dst_dev, hdst, cudaIpcMemLazyEnablePeerAccess); !st.ok()) {
      for (const auto& o : opened)
        (void)tensorcast::cuda::close_ipc_mem_handle(o.ptr);
      return to_grpc_status(st);
    }
    // Zero PAD segments per plan to ensure deterministic padding
    (void)tensorcast::cuda::set_device(meta.device_id);
    for (const auto& p : plan) {
      if (p.kind != tensorcast::store::loader::SegmentPiece::PAD)
        continue;
      if (p.length == 0)
        continue;
      auto st =
          tensorcast::cuda::memset(static_cast<uint8_t*>(dst_dev) + p.dst_offset, 0, static_cast<size_t>(p.length));
      if (!st.ok()) {
        (void)tensorcast::cuda::close_ipc_mem_handle(dst_dev);
        for (const auto& o : opened)
          (void)tensorcast::cuda::close_ipc_mem_handle(o.ptr);
        return to_grpc_status(st);
      }
    }
    // Copy each leased segment to its explicit destination offset; order-free
    for (size_t j = 0; j < opened.size(); ++j) {
      const auto& o = opened[j];
      const uint64_t dst_off = lease_vec[j].dst_offset;
      // Validate destination range within [0, total_size)
      if (dst_off > meta.total_size || o.len > meta.total_size || dst_off + o.len > meta.total_size) {
        (void)tensorcast::cuda::close_ipc_mem_handle(dst_dev);
        for (const auto& k : opened)
          (void)tensorcast::cuda::close_ipc_mem_handle(k.ptr);
        return {StatusCode::OUT_OF_RANGE, "lease segment dst range out of bounds"};
      }
      auto st = tensorcast::cuda::memcpy(
          static_cast<uint8_t*>(dst_dev) + dst_off,
          static_cast<uint8_t*>(o.ptr) + o.base,
          static_cast<size_t>(o.len),
          cudaMemcpyDeviceToDevice);
      if (!st.ok()) {
        (void)tensorcast::cuda::close_ipc_mem_handle(dst_dev);
        for (const auto& k : opened)
          (void)tensorcast::cuda::close_ipc_mem_handle(k.ptr);
        return to_grpc_status(st);
      }
    }
    (void)tensorcast::cuda::device_synchronize();
    (void)tensorcast::cuda::close_ipc_mem_handle(dst_dev);
    for (const auto& o : opened)
      (void)tensorcast::cuda::close_ipc_mem_handle(o.ptr);
    // Commit the coalesced registration in engine to finalize and compute descriptor
    auto commit2_or = engine_->commit_registered_artifact(out2.registration_id);
    if (!commit2_or.ok())
      return to_grpc_status(commit2_or.status());
    // Successful commit; prevent abort on scope exit
    abort_guard.release();
    const auto& d = commit2_or.value();
    auto* desc = resp->mutable_descriptor_();
    desc->set_artifact_id(d.artifact_id);
    desc->set_index_multihash(d.index_multihash);
    desc->set_data_multihash(d.data_multihash);
    desc->set_schema_version(d.schema_version);
    desc->set_encoding(d.encoding);
    desc->set_total_size(d.size_bytes);
    // Metrics: commit (lease)
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_commit_lease_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_commit_lease_total unavailable";
    }
    // Cleanup lease meta
    absl::MutexLock l2(&reg_mu_);
    reg_meta_.erase(req->registration_id());
    reg_leases_.erase(req->registration_id());
    return Status::OK;
  }
  {
    auto commit_or = engine_->commit_registered_artifact(req->registration_id());
    if (!commit_or.ok())
      return to_grpc_status(commit_or.status());
    const auto& out = commit_or.value();
    auto* desc = resp->mutable_descriptor_();
    desc->set_artifact_id(out.artifact_id);
    desc->set_index_multihash(out.index_multihash);
    desc->set_data_multihash(out.data_multihash);
    desc->set_schema_version(out.schema_version);
    desc->set_encoding(out.encoding);
    desc->set_total_size(out.size_bytes);
    // Metrics: commit (coalesced)
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_commit_coalesced_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_commit_coalesced_total unavailable";
    }
    absl::MutexLock l(&reg_mu_);
    reg_meta_.erase(req->registration_id());
    return Status::OK;
  }
  return Status::OK;
}

Status StoreDaemonServiceImpl::AbortRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v1::AbortRegisteredArtifactRequest* req,
    v1::AbortRegisteredArtifactResponse* /*resp*/) {
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
  if (opts_.allow_high_card_attrs) {
    span->SetAttribute("tc.registration.id", req->registration_id());
  }
  auto st = engine_->abort_registered_artifact(req->registration_id());
  if (!st.ok())
    return to_grpc_status(st);
  // Metrics: abort
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_register_abort_total");
    counter->Add(1.0);
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_abort_total unavailable";
  }
  return Status::OK;
}

// Removed unary FeedRegisterArtifact; use streaming variant only

Status StoreDaemonServiceImpl::FeedRegisterArtifactStream(
    grpc::ServerContext* /*ctx*/,
    ::grpc::ServerReader<v1::FeedRegisterArtifactStreamRequest>* reader,
    v1::FeedRegisterArtifactStreamResponse* /*resp*/) {
  v1::FeedRegisterArtifactStreamRequest req;
  std::string reg_id;
  // Track final frame if needed in future; currently unused
  while (reader->Read(&req)) {
    if (reg_id.empty()) {
      reg_id = req.registration_id();
      // Validate registration exists
      absl::MutexLock l(&reg_mu_);
      if (!reg_meta_.contains(reg_id)) {
        return {StatusCode::NOT_FOUND, "registration_id not found"};
      }
      // Optional TTL fail-fast at first read
      auto it = reg_meta_.find(reg_id);
      if (it != reg_meta_.end() && it->second.expiry.time_since_epoch().count() > 0 &&
          std::chrono::steady_clock::now() > it->second.expiry) {
        reg_meta_.erase(reg_id);
        reg_leases_.erase(reg_id);
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto counter = meter->CreateDoubleCounter("tc_register_ttl_expired_feed_total");
          counter->Add(1.0);
        } catch (...) {
          VLOG(1) << "metrics counter tc_register_ttl_expired_feed_total unavailable";
        }
        return {StatusCode::DEADLINE_EXCEEDED, "registration expired (TTL)"};
      }
    } else if (req.registration_id() != reg_id) {
      return {StatusCode::INVALID_ARGUMENT, "registration_id changed in stream"};
    }

    // Refresh TTL on every frame when TTL was set at Begin (daemon meta + engine)
    {
      absl::MutexLock l(&reg_mu_);
      auto it = reg_meta_.find(reg_id);
      if (it != reg_meta_.end() && it->second.ttl_ms > 0) {
        it->second.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(it->second.ttl_ms);
      }
    }
    {
      absl::MutexLock l(&reg_mu_);
      auto it = reg_meta_.find(reg_id);
      if (it != reg_meta_.end() && it->second.ttl_ms > 0) {
        (void)engine_->keep_alive_registered_artifact(reg_id, it->second.ttl_ms);
      }
    }

    if (req.has_dvmp_chunk()) {
      const auto& ck = req.dvmp_chunk();
      auto st = engine_->feed_register_dvmp_chunk(reg_id, ck.offset(), ck.data().data(), ck.data().size());
      if (!st.ok())
        return to_grpc_status(st);
      // No-op on last frame: commit performed explicitly via CommitRegisteredArtifact
    } else if (req.has_lease_segments()) {
      absl::MutexLock l(&reg_mu_);
      auto& vec = reg_leases_[reg_id];
      for (const auto& s : req.lease_segments().segments()) {
        LeaseSegMeta m;
        m.device_id = s.device_id();
        m.handle_bytes = s.cuda_ipc_handle();
        m.base_offset = s.base_addr();
        m.length = s.length();
        m.dst_offset = s.dst_offset();
        vec.push_back(std::move(m));
      }
    } else {
      return {StatusCode::INVALID_ARGUMENT, "missing feed payload"};
    }
  }
  // Note: Do not auto-commit here. Commit is performed explicitly via
  // CommitRegisteredArtifact to keep lifecycle consistent and allow
  // callers to interleave KeepAlive calls after streaming finishes.
  return Status::OK;
}

grpc::Status StoreDaemonServiceImpl::feed_register_artifact_stream_vector(
    const std::vector<v1::FeedRegisterArtifactStreamRequest>& reqs) {
  std::string reg_id;
  bool saw_last = false;
  for (const auto& req : reqs) {
    if (reg_id.empty()) {
      reg_id = req.registration_id();
      absl::MutexLock l(&reg_mu_);
      if (!reg_meta_.contains(reg_id)) {
        return {StatusCode::NOT_FOUND, "registration_id not found"};
      }
      auto it = reg_meta_.find(reg_id);
      if (it != reg_meta_.end() && it->second.expiry.time_since_epoch().count() > 0 &&
          std::chrono::steady_clock::now() > it->second.expiry) {
        reg_meta_.erase(reg_id);
        reg_leases_.erase(reg_id);
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto counter = meter->CreateDoubleCounter("tc_register_ttl_expired_feed_total");
          counter->Add(1.0);
        } catch (...) {
          VLOG(1) << "metrics counter tc_register_ttl_expired_feed_total unavailable";
        }
        return {StatusCode::DEADLINE_EXCEEDED, "registration expired (TTL)"};
      }
    } else if (req.registration_id() != reg_id) {
      return {StatusCode::INVALID_ARGUMENT, "registration_id changed in stream"};
    }

    // Refresh TTL
    {
      absl::MutexLock l(&reg_mu_);
      auto it = reg_meta_.find(reg_id);
      if (it != reg_meta_.end() && it->second.ttl_ms > 0) {
        it->second.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(it->second.ttl_ms);
        (void)engine_->keep_alive_registered_artifact(reg_id, it->second.ttl_ms);
      }
    }

    if (req.has_dvmp_chunk()) {
      const auto& ck = req.dvmp_chunk();
      auto st = engine_->feed_register_dvmp_chunk(reg_id, ck.offset(), ck.data().data(), ck.data().size());
      if (!st.ok())
        return to_grpc_status(st);
      // No-op on last frame: commit performed explicitly via CommitRegisteredArtifact
    } else if (req.has_lease_segments()) {
      absl::MutexLock l(&reg_mu_);
      auto& vec = reg_leases_[reg_id];
      for (const auto& s : req.lease_segments().segments()) {
        LeaseSegMeta m;
        m.device_id = s.device_id();
        m.handle_bytes = s.cuda_ipc_handle();
        m.base_offset = s.base_addr();
        m.length = s.length();
        m.dst_offset = s.dst_offset();
        vec.push_back(std::move(m));
      }
    } else {
      return {StatusCode::INVALID_ARGUMENT, "missing feed payload"};
    }
  }
  (void)saw_last; // no auto-commit
  return Status::OK;
}

Status StoreDaemonServiceImpl::KeepAliveRegisterArtifact(
    grpc::ServerContext* /*ctx*/,
    const v1::KeepAliveRegisterArtifactRequest* req,
    v1::KeepAliveRegisterArtifactResponse* /*resp*/) {
  absl::MutexLock l(&reg_mu_);
  auto it = reg_meta_.find(req->registration_id());
  if (it == reg_meta_.end()) {
    return {StatusCode::NOT_FOUND, "registration_id not found"};
  }
  it->second.epoch = req->epoch();
  if (req->ttl_ms() > 0) {
    it->second.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(req->ttl_ms());
    it->second.ttl_ms = static_cast<uint32_t>(req->ttl_ms());
    // Propagate to engine so internal TTL check also extends
    (void)engine_->keep_alive_registered_artifact(req->registration_id(), it->second.ttl_ms);
  }
  // Metrics: keepalive
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_register_keepalive_total");
    counter->Add(1.0);
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_keepalive_total unavailable";
  }
  return Status::OK;
}

Status StoreDaemonServiceImpl::RevokeRegisteredArtifact(
    grpc::ServerContext* /*ctx*/,
    const v1::RevokeRegisteredArtifactRequest* req,
    v1::RevokeRegisteredArtifactResponse* /*resp*/) {
  // Best-effort abort + cleanup meta
  (void)engine_->abort_registered_artifact(req->registration_id());
  absl::MutexLock l(&reg_mu_);
  reg_meta_.erase(req->registration_id());
  reg_buffers_.erase(req->registration_id());
  reg_leases_.erase(req->registration_id());
  // Metrics: revoke
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_register_revoke_total");
    counter->Add(1.0);
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_revoke_total unavailable";
  }
  return Status::OK;
}

void StoreDaemonServiceImpl::start_sweepers() {
  stop_.store(false);
  sweep_sessions_th_ = std::thread([this]() {
    while (!stop_.load()) {
      for (const auto& k : sessions_.keys()) {
        sessions_.remove_if_expired(k);
      }
      std::this_thread::sleep_for(opts_.sessions_sweep_interval);
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
      std::this_thread::sleep_for(opts_.locks_sweep_interval);
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
          VLOG(1) << "descriptor parse error (ignored)";
        }
        auto st = engine_->register_replica_with_global_store(task.key, mi2_id);
        if (!st.ok()) {
          VLOG(1) << "Auto-register disk load failed: " << st;
        }
      }

      std::this_thread::sleep_for(opts_.verification_sweep_interval);
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
      std::this_thread::sleep_for(opts_.proc_check_interval);
    }
  });
  // Optional periodic eviction policy: unload least-recently-used GPU replicas
  // on devices where used memory exceeds the configured fraction. Disabled by default.
  // Controlled via environment variables:
  //  - TC_DAEMON_ENABLE_PERIODIC_EVICTION: truthy to enable (default: false)
  //  - TC_DAEMON_GPU_MEMORY_LIMIT_FRACTION: threshold fraction (default: 0.90)
  //  - TC_DAEMON_EVICTION_CHECK_INTERVAL_MS: check interval in ms (default: 1000)
  if (opts_.enable_periodic_eviction) {
    const double gpu_memory_limit_fraction = opts_.gpu_memory_limit_fraction;
    const auto eviction_check_interval = opts_.eviction_check_interval;
    eviction_th_ = std::thread([this, gpu_memory_limit_fraction, eviction_check_interval]() {
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
            const auto used2 = static_cast<double>(*tot_or - *f2);
            ratio = used2 / total;
          }
        }
        std::this_thread::sleep_for(eviction_check_interval);
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
        it = gpu_map.emplace(info.gpu_device_id, GpuAgg{.out = gpu, .mem_filled = false}).first;
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
  if (opts_.allow_high_card_attrs) {
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
    key.device = (device_id >= 0)
        ? tensorcast::store::DeviceRegistry::instance().gpu_key(device_id)
        : tensorcast::store::DeviceKey{.type = tensorcast::DeviceType::CPU, .ordinal = -1, .uuid = ""};
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
