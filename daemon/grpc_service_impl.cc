// Copyright (c) 2025, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <nlohmann/json.hpp>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "core/store/components/global_store_client.h"
#include "core/store/device_registry.h"
#include "core/store/device_types.h"
#include "core/store/loading/loading_spec.h"
#include "daemon/grpc_metrics.h"
#include "daemon/grpc_span.h"
#include "daemon/replica_listing.h"
#include "daemon/status_utils.h"
#include "daemon/sweep_tasks.h"
#include "daemon/types.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

absl::StatusOr<std::vector<uint8_t>> StoreDaemonServiceImpl::lip_copy_to_new_coalesced_int(
    int target_device_id,
    const std::string& canonical_index_json,
    uint64_t total_size,
    absl::Span<const LeaseSegMeta> segments) {
  return lip_mgr_->copy_to_new_coalesced(target_device_id, canonical_index_json, total_size, segments);
}

Status StoreDaemonServiceImpl::MaterializeReplica(
    grpc::ServerContext* ctx,
    const v1::MaterializeReplicaRequest* req,
    v1::MaterializeReplicaResponse* resp) {
  RpcContext rctx{"MaterializeReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->MaterializeReplica(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ConfirmReplica(
    grpc::ServerContext* ctx,
    const v1::ConfirmReplicaRequest* req,
    v1::ConfirmReplicaResponse* resp) {
  RpcContext rctx{"ConfirmReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->Confirm(rctx, *req, *resp);
}

// RFC-0014: Materialize by key using Global Store mapping
Status StoreDaemonServiceImpl::MaterializeByKey(
    grpc::ServerContext* ctx,
    const v1::MaterializeByKeyRequest* req,
    v1::MaterializeByKeyResponse* resp) {
  RpcContext rctx{"MaterializeByKey", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->MaterializeByKey(rctx, *req, *resp);
}

// RFC-0014: Publish key mapping – lightweight wrapper to Global Store
Status StoreDaemonServiceImpl::PublishReplicaKey(
    grpc::ServerContext* ctx,
    const v1::PublishReplicaKeyRequest* req,
    v1::PublishReplicaKeyResponse* resp) {
  RpcContext rctx{"PublishReplicaKey", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req->key());

  if (req->key().empty() || !req->has_artifact_descriptor() || req->artifact_descriptor().artifact_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "key and artifact_descriptor.artifact_id are required"};
  }
  if (is_shutting_down_.load()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  // Use engine's configured Global Store client for upsert.
  auto up = engine_->upsert_key_mapping(req->key(), req->artifact_descriptor().artifact_id(), req->disk_path());
  if (!up.ok()) {
    resp->set_ok(false);
    resp->set_conflict_reason(std::string(up.message()));
    return to_grpc_status(up);
  }
  resp->set_ok(true);
  rctx.mark_success();
  return grpc::Status::OK;
}

Status StoreDaemonServiceImpl::ResolveKeyMapping(
    grpc::ServerContext* ctx,
    const v1::ResolveKeyMappingRequest* req,
    v1::ResolveKeyMappingResponse* resp) {
  RpcContext rctx{"ResolveKeyMapping", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req->key());

  if (req->key().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "key is required"};
  }
  if (is_shutting_down_.load()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  auto mapping_or = engine_->resolve_key_mapping(req->key());
  if (!mapping_or.ok()) {
    return to_grpc_status(mapping_or.status());
  }
  const auto& m = *mapping_or;
  resp->set_artifact_id(m.artifact_id);
  resp->set_used_disk_path(m.disk_path);
  rctx.mark_success();
  return grpc::Status::OK;
}

Status StoreDaemonServiceImpl::GetArtifactIndexById(
    grpc::ServerContext* ctx,
    const v1::GetArtifactIndexByIdRequest* req,
    v1::GetArtifactIndexByIdResponse* resp) {
  RpcContext rctx{"GetArtifactIndexById", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->GetArtifactIndexById(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::UnloadReplica(
    grpc::ServerContext* ctx,
    const v1::UnloadReplicaRequest* req,
    v1::UnloadReplicaResponse* resp) {
  RpcContext rctx{"UnloadReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->Unload(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ClearMem(
    grpc::ServerContext* ctx,
    const v1::ClearMemRequest* /*req*/,
    v1::ClearMemResponse* /*resp*/) {
  metrics::RpcMethodMetricsTimer mt("ClearMem");
  GrpcSpan gspan("ClearMem", *ctx);
  const int rc = engine_->clear_mem();
  if (rc == 0) {
    mt.mark_success();
    return Status::OK;
  }
  return {StatusCode::INTERNAL, absl::StrFormat("clear_mem() returned %d", rc)};
}

Status StoreDaemonServiceImpl::GetServerConfig(
    grpc::ServerContext* ctx,
    const v1::GetServerConfigRequest* /*req*/,
    v1::GetServerConfigResponse* resp) {
  RpcContext rctx{"GetServerConfig", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_server_config(rctx, *resp);
}

// Destructor: stop sweepers
StoreDaemonServiceImpl::~StoreDaemonServiceImpl() {
  stop_sweepers();
}

// ──────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────

// Legacy device/key helpers have been removed in favor of DeviceResolver

Status StoreDaemonServiceImpl::WaitReplicaVerification(
    grpc::ServerContext* ctx,
    const v1::WaitReplicaVerificationRequest* req,
    v1::WaitReplicaVerificationResponse* resp) {
  RpcContext rctx{"WaitReplicaVerification", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->WaitVerification(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::LockTransportChunks(
    grpc::ServerContext* ctx,
    const v1::LockTransportChunksRequest* req,
    v1::LockTransportChunksResponse* resp) {
  RpcContext rctx{"LockTransportChunks", *ctx, opts_.allow_high_card_attrs};
  return transport_controller_->lock(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::UnlockTransportChunks(
    grpc::ServerContext* ctx,
    const v1::UnlockTransportChunksRequest* req,
    v1::UnlockTransportChunksResponse* /*resp*/) {
  RpcContext rctx{"UnlockTransportChunks", *ctx, opts_.allow_high_card_attrs};
  v1::UnlockTransportChunksResponse dummy;
  return transport_controller_->unlock(rctx, *req, dummy);
}

Status StoreDaemonServiceImpl::BeginRegisterArtifact(
    grpc::ServerContext* ctx,
    const v1::BeginRegisterArtifactRequest* req,
    v1::BeginRegisterArtifactResponse* resp) {
  RpcContext rctx{"BeginRegisterArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->Begin(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::CommitRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v1::CommitRegisteredArtifactRequest* req,
    v1::CommitRegisteredArtifactResponse* resp) {
  RpcContext rctx{"CommitRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->Commit(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::AbortRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v1::AbortRegisteredArtifactRequest* req,
    v1::AbortRegisteredArtifactResponse* resp) {
  RpcContext rctx{"AbortRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->Abort(rctx, *req, *resp);
}

// Removed unary FeedRegisterArtifact; use streaming variant only

Status StoreDaemonServiceImpl::FeedRegisterArtifactStream(
    grpc::ServerContext* ctx,
    ::grpc::ServerReader<v1::FeedRegisterArtifactStreamRequest>* reader,
    v1::FeedRegisterArtifactStreamResponse* resp) {
  RpcContext rctx{"FeedRegisterArtifactStream", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->FeedStream(rctx, *reader, *resp);
}

grpc::Status StoreDaemonServiceImpl::feed_register_artifact_stream_vector(
    const std::vector<v1::FeedRegisterArtifactStreamRequest>& reqs) {
  return registration_controller_->FeedVector(reqs);
}

Status StoreDaemonServiceImpl::KeepAliveRegisterArtifact(
    grpc::ServerContext* ctx,
    const v1::KeepAliveRegisterArtifactRequest* req,
    v1::KeepAliveRegisterArtifactResponse* resp) {
  RpcContext rctx{"KeepAliveRegisterArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->KeepAlive(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RevokeRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v1::RevokeRegisteredArtifactRequest* req,
    v1::RevokeRegisteredArtifactResponse* resp) {
  RpcContext rctx{"RevokeRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->Revoke(rctx, *req, *resp);
}

void StoreDaemonServiceImpl::start_sweepers() {
  scheduler_ = std::make_unique<BackgroundScheduler>();
  using std::chrono::milliseconds;
  // Session TTL
  {
    auto t = std::make_shared<SessionTtlTask>(sessions_);
    scheduler_->add_task(
        TaskKind::kSessionTTL, std::chrono::duration_cast<milliseconds>(opts_.sessions_sweep_interval), [t]() {
          t->run_once();
        });
  }
  // Lock TTL
  {
    auto t = std::make_shared<LockTtlTask>(locks_, *engine_);
    scheduler_->add_task(
        TaskKind::kLockTTL, std::chrono::duration_cast<milliseconds>(opts_.locks_sweep_interval), [t]() {
          t->run_once();
        });
  }
  // Verification + auto-registration
  {
    auto t = std::make_shared<VerificationTask>(*verif_tracker_, *engine_, &bg_tasks_mu_, &auto_reg_tasks_);
    scheduler_->add_task(
        TaskKind::kVerification, std::chrono::duration_cast<milliseconds>(opts_.verification_sweep_interval), [t]() {
          t->run_once();
        });
  }
  // PID watch
  {
    auto t = std::make_shared<PidWatchTask>(refs_, *lip_mgr_);
    scheduler_->add_task(
        TaskKind::kPidWatch, std::chrono::duration_cast<milliseconds>(opts_.proc_check_interval), [t]() {
          t->run_once();
        });
  }
  // Optional eviction
  if (opts_.enable_periodic_eviction) {
    const double limit = opts_.gpu_memory_limit_fraction;
    auto t = std::make_shared<EvictionTask>(*engine_, refs_, limit);
    scheduler_->add_task(
        TaskKind::kEviction, std::chrono::duration_cast<milliseconds>(opts_.eviction_check_interval), [t]() {
          t->run_once();
        });
  }
  scheduler_->start();
}

void StoreDaemonServiceImpl::stop_sweepers() {
  if (scheduler_) {
    scheduler_->stop();
    scheduler_.reset();
  }
}

Status StoreDaemonServiceImpl::GetWorkerStatus(
    grpc::ServerContext* ctx,
    const v1::GetWorkerStatusRequest* /*req*/,
    v1::GetWorkerStatusResponse* resp) {
  RpcContext rctx{"GetWorkerStatus", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_worker_status(rctx, *resp);
}

Status StoreDaemonServiceImpl::GetDetailedStatus(
    grpc::ServerContext* ctx,
    const v1::GetDetailedStatusRequest* /*req*/,
    v1::GetDetailedStatusResponse* resp) {
  RpcContext rctx{"GetDetailedStatus", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_detailed_status(rctx, *resp);
}

// verification tracking moved to VerificationTracker

// Legacy GetLoadedReplicas removed; use V2

Status StoreDaemonServiceImpl::GetLoadedReplicasV2(
    grpc::ServerContext* ctx,
    const v1::GetLoadedReplicasV2Request* req,
    v1::GetLoadedReplicasV2Response* resp) {
  RpcContext rctx{"GetLoadedReplicasV2", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_loaded_replicas_v2(rctx, *req, *resp, opts_.use_cursor_pagination);
}

} // namespace tensorcast::daemon
