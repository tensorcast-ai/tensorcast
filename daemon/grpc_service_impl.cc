// Copyright (c) 2025, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <nlohmann/json.hpp>
#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "core/store/components/global_store_client.h"
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
    absl::Span<const LeaseSegMeta> segments,
    absl::Span<const RegisterStorageMeta> storages) {
  return lip_mgr_->copy_to_new_coalesced(target_device_id, canonical_index_json, total_size, segments, storages);
}

Status StoreDaemonServiceImpl::MaterializeReplica(
    grpc::ServerContext* ctx,
    const v1::MaterializeReplicaRequest* req,
    v1::MaterializeReplicaResponse* resp) {
  RpcContext rctx{"MaterializeReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->materialize_replica(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ConfirmReplica(
    grpc::ServerContext* ctx,
    const v1::ConfirmReplicaRequest* req,
    v1::ConfirmReplicaResponse* resp) {
  RpcContext rctx{"ConfirmReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->confirm(rctx, *req, *resp);
}

// RFC-0014: Materialize by key using Global Store mapping
Status StoreDaemonServiceImpl::MaterializeByKey(
    grpc::ServerContext* ctx,
    const v1::MaterializeByKeyRequest* req,
    v1::MaterializeByKeyResponse* resp) {
  RpcContext rctx{"MaterializeByKey", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->materialize_by_key(rctx, *req, *resp);
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
    // For conflicts, return OK with ok=false for idempotency.
    if (absl::IsAlreadyExists(up)) {
      resp->set_ok(false);
      resp->set_conflict_reason(std::string(up.message()));
      rctx.mark_success();
      return grpc::Status::OK;
    }
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
  return materialization_controller_->get_artifact_index_by_id(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::StartPersistence(
    grpc::ServerContext* ctx,
    const v1::StartPersistenceRequest* req,
    v1::StartPersistenceResponse* resp) {
  RpcContext rctx{"StartPersistence", *ctx, opts_.allow_high_card_attrs};
  if (req->artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  if (req->placement_policy() == v1::PLACEMENT_POLICY_UNSPECIFIED) {
    return {StatusCode::INVALID_ARGUMENT, "placement_policy is required"};
  }
  if (is_shutting_down_.load()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!persistence_mgr_) {
    return {StatusCode::FAILED_PRECONDITION, "persistence manager unavailable"};
  }
  auto task_or =
      persistence_mgr_->start_task(req->artifact_id(), req->placement_policy(), req->persist_to_shared_disk());
  if (!task_or.ok()) {
    return to_grpc_status(task_or.status());
  }
  const auto& task = *task_or;

  resp->set_task_id(task.task_id);
  resp->set_plan_id(task.plan_id);
  resp->set_state(task.state);
  resp->set_progress(task.progress);
  if (!task.degraded_reason.empty()) {
    resp->set_degraded_reason(task.degraded_reason);
  }
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::QueryPersistenceStatus(
    grpc::ServerContext* ctx,
    const v1::QueryPersistenceStatusRequest* req,
    v1::QueryPersistenceStatusResponse* resp) {
  RpcContext rctx{"QueryPersistenceStatus", *ctx, opts_.allow_high_card_attrs};
  if (req->task_id().empty() && req->artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "task_id or artifact_id is required"};
  }
  std::optional<std::string> task_key;
  if (!req->task_id().empty()) {
    task_key = req->task_id();
  }
  if (!persistence_mgr_) {
    return {StatusCode::FAILED_PRECONDITION, "persistence manager unavailable"};
  }
  absl::optional<PersistenceTaskState> task;
  if (task_key.has_value()) {
    task = persistence_mgr_->get_by_task_id(*task_key);
  } else {
    task = persistence_mgr_->get_latest_for_artifact(req->artifact_id());
  }

  if (!task.has_value()) {
    return {StatusCode::NOT_FOUND, "persistence task not found"};
  }
  resp->set_task_id(task_key.value_or(task->task_id));
  resp->set_artifact_id(task->artifact_id);
  resp->set_plan_id(task->plan_id);
  resp->set_state(task->state);
  resp->set_progress(task->progress);
  if (!task->degraded_reason.empty()) {
    resp->set_degraded_reason(task->degraded_reason);
  }
  if (!task->last_error.empty()) {
    resp->set_last_error(task->last_error);
  }
  for (const auto& shard : task->shards) {
    auto* out = resp->add_shards();
    out->set_shard_id(shard.shard_id);
    out->set_shard_idx(shard.shard_idx);
    out->set_state(shard.state);
    out->set_progress(shard.progress);
    if (!shard.degraded_reason.empty()) {
      out->set_degraded_reason(shard.degraded_reason);
    }
    if (!shard.last_error.empty()) {
      out->set_last_error(shard.last_error);
    }
    out->mutable_target_nodes()->Reserve(static_cast<int>(shard.targets.size()));
    out->mutable_lease_ids()->Reserve(static_cast<int>(shard.targets.size()));
    for (const auto& target : shard.targets) {
      out->add_target_nodes(target.node_id);
      out->add_lease_ids(target.lease_id);
    }
  }
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::UnloadReplica(
    grpc::ServerContext* ctx,
    const v1::UnloadReplicaRequest* req,
    v1::UnloadReplicaResponse* resp) {
  RpcContext rctx{"UnloadReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->unload(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ClearMem(
    grpc::ServerContext* ctx,
    const v1::ClearMemRequest* /*req*/,
    v1::ClearMemResponse* /*resp*/) {
  RpcContext rctx{"ClearMem", *ctx, opts_.allow_high_card_attrs};
  const int rc = engine_->clear_mem();
  if (rc == 0) {
    rctx.mark_success();
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
  return materialization_controller_->wait_verification(rctx, *req, *resp);
}

Status StoreDaemonServiceV2Impl::MaterializeReplica(
    grpc::ServerContext* ctx,
    const v2::MaterializeReplicaRequest* req,
    v2::MaterializeReplicaResponse* resp) {
  RpcContext rctx{"MaterializeReplicaV2", *ctx, allow_high_card_attrs_};
  return materialization_controller_.materialize_replica_v2(rctx, *req, *resp);
}

Status StoreDaemonServiceV2Impl::MaterializeByKey(
    grpc::ServerContext* ctx,
    const v2::MaterializeByKeyRequest* req,
    v2::MaterializeByKeyResponse* resp) {
  RpcContext rctx{"MaterializeByKeyV2", *ctx, allow_high_card_attrs_};
  return materialization_controller_.materialize_by_key_v2(rctx, *req, *resp);
}

Status StoreDaemonServiceV2Impl::ResolveArtifactFromDisk(
    grpc::ServerContext* ctx,
    const v2::ResolveArtifactFromDiskRequest* req,
    v2::ResolveArtifactFromDiskResponse* resp) {
  RpcContext rctx{"ResolveArtifactFromDisk", *ctx, allow_high_card_attrs_};
  return materialization_controller_.resolve_artifact_from_disk(rctx, *req, *resp);
}

Status StoreDaemonServiceV2Impl::GetMaterializeCapabilities(
    grpc::ServerContext* ctx,
    const v2::GetMaterializeCapabilitiesRequest* /*req*/,
    v2::GetMaterializeCapabilitiesResponse* resp) {
  RpcContext rctx{"GetMaterializeCapabilities", *ctx, allow_high_card_attrs_};
  resp->set_supports_view_subset_hash(true);
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::LockTransportChunks(
    grpc::ServerContext* ctx,
    const v1::LockTransportChunksRequest* req,
    v1::LockTransportChunksResponse* resp) {
  RpcContext rctx{"LockTransportChunks", *ctx, opts_.allow_high_card_attrs};
  return transport_controller_->lock(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RegisterVramRegion(
    grpc::ServerContext* ctx,
    const v1::RegisterVramRegionRequest* req,
    v1::RegisterVramRegionResponse* resp) {
  RpcContext rctx{"RegisterVramRegion", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.device.id", static_cast<int64_t>(req->device_id()));
  span->SetAttribute("tc.region.size_bytes", static_cast<int64_t>(req->size_bytes()));
  span->SetAttribute("tc.region.ttl_ms", static_cast<int64_t>(req->ttl_ms()));

  if (is_shutting_down_.load()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req->owner_pid() <= 0) {
    return {StatusCode::INVALID_ARGUMENT, "owner_pid must be > 0"};
  }
  if (req->device_id() < 0) {
    return {StatusCode::INVALID_ARGUMENT, "device_id must be >= 0"};
  }
  if (req->size_bytes() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "size_bytes must be > 0"};
  }
  if (req->ttl_ms() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "ttl_ms must be > 0"};
  }
  if (req->cuda_ipc_handle().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "cuda_ipc_handle must not be empty"};
  }

  IpcRegionRegistry::RegisterParams params;
  params.device_id = req->device_id();
  params.owner_pid = req->owner_pid();
  params.size_bytes = req->size_bytes();
  params.ttl_ms = req->ttl_ms();
  if (req->has_session_id()) {
    params.session_id = req->session_id();
  }
  if (req->has_region_name()) {
    params.region_name = req->region_name();
  }
  params.handle_bytes = std::string(req->cuda_ipc_handle());

  auto desc_or = region_registry_->register_region(params);
  if (!desc_or.ok()) {
    return to_grpc_status(desc_or.status());
  }
  const auto& desc = *desc_or;
  resp->set_region_id(desc.region_id);
  resp->set_ttl_ms(desc.ttl_ms);
  if (desc.expires_at != absl::InfiniteFuture()) {
    const int64_t micros = absl::ToUnixMicros(desc.expires_at);
    auto* ts = resp->mutable_expires_at();
    ts->set_seconds(micros / 1'000'000);
    ts->set_nanos(static_cast<int32_t>((micros % 1'000'000) * 1'000));
  }
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::UnregisterVramRegion(
    grpc::ServerContext* ctx,
    const v1::UnregisterVramRegionRequest* req,
    v1::UnregisterVramRegionResponse* resp) {
  RpcContext rctx{"UnregisterVramRegion", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.region.id", req->region_id());

  if (req->region_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "region_id is required"};
  }
  if (req->owner_pid() <= 0) {
    return {StatusCode::INVALID_ARGUMENT, "owner_pid must be > 0"};
  }

  const bool force = req->has_force() ? req->force() : false;
  auto released_or = region_registry_->unregister_region(req->region_id(), req->owner_pid(), force);
  if (!released_or.ok()) {
    return to_grpc_status(released_or.status());
  }
  resp->set_released(*released_or);
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::DeregisterArtifact(
    grpc::ServerContext* ctx,
    const v1::DeregisterArtifactRequest* req,
    v1::DeregisterArtifactResponse* resp) {
  RpcContext rctx{"DeregisterArtifact", *ctx, opts_.allow_high_card_attrs};
  if (!req->artifact_id().empty()) {
    rctx.span()->SetAttribute("tc.artifact.id", req->artifact_id());
  }
  if (is_shutting_down_.load()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req->artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  const std::string artifact_id = req->artifact_id();
  auto lip_opt = lip_mgr_->find_active_by_artifact_id(artifact_id);
  if (!lip_opt.has_value()) {
    return {StatusCode::NOT_FOUND, "no active lease for artifact"};
  }
  const auto& lip = *lip_opt;
  if (req->has_owner_pid() && lip.owner_pid != req->owner_pid()) {
    return {StatusCode::PERMISSION_DENIED, "owner_pid mismatch for active lease"};
  }
  // Optional TTL extension before quiesce
  if (req->has_extend_ttl_ms() && req->extend_ttl_ms() > 0) {
    auto st = lip_mgr_->extend_ttl_for_artifact(artifact_id, req->extend_ttl_ms());
    if (!st.ok())
      return to_grpc_status(st);
  }
  // Quiesce new staged exports
  lip_mgr_->quiesce_artifact(artifact_id);
  bool drained = true;
  if (req->wait_for_drain()) {
    const uint32_t timeout_ms = req->has_drain_timeout_ms() ? req->drain_timeout_ms() : 30000U;
    absl::Time deadline = absl::Now() + absl::Milliseconds(timeout_ms);
    drained = lip_mgr_->wait_exports_drained(artifact_id, deadline);
    if (!drained) {
      return {StatusCode::DEADLINE_EXCEEDED, "drain timed out; artifact remains quiesced"};
    }
  }
  // Remove active lease if owner matches (or no owner provided)
  bool removed = lip_mgr_->revoke_commit_lease_if_owner_matches(artifact_id, lip.device_id, lip.owner_pid);
  resp->set_drained(drained);
  resp->set_removed(removed);
  if (removed) {
    // Best-effort: synchronize removal with Global Store
    absl::Status gs_st = engine_->unregister_replica_from_global_store(artifact_id, lip.device_id);
    if (!gs_st.ok()) {
      // Do not fail the RPC; attach message for observability
      resp->set_message(absl::StrCat("Global Store deregister failed: ", gs_st.message()));
    }
  }
  // Return referenced region ids for observability
  absl::flat_hash_set<std::string> unique_regions;
  for (const auto& s : lip.storages) {
    if (s.has_region())
      unique_regions.insert(s.region_id);
  }
  for (const auto& rid : unique_regions) {
    resp->add_released_region_ids(rid);
  }
  rctx.mark_success();
  return grpc::Status::OK;
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
  return registration_controller_->begin(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::CommitRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v1::CommitRegisteredArtifactRequest* req,
    v1::CommitRegisteredArtifactResponse* resp) {
  RpcContext rctx{"CommitRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->commit(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::AbortRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v1::AbortRegisteredArtifactRequest* req,
    v1::AbortRegisteredArtifactResponse* resp) {
  RpcContext rctx{"AbortRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->abort(rctx, *req, *resp);
}

// Removed unary FeedRegisterArtifact; use streaming variant only

Status StoreDaemonServiceImpl::FeedRegisterArtifactStream(
    grpc::ServerContext* ctx,
    ::grpc::ServerReader<v1::FeedRegisterArtifactStreamRequest>* reader,
    v1::FeedRegisterArtifactStreamResponse* resp) {
  RpcContext rctx{"FeedRegisterArtifactStream", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->feed_stream(rctx, *reader, *resp);
}

grpc::Status StoreDaemonServiceImpl::feed_register_artifact_stream_vector(
    const std::vector<v1::FeedRegisterArtifactStreamRequest>& reqs) {
  return registration_controller_->feed_vector(reqs);
}

Status StoreDaemonServiceImpl::KeepAliveRegisterArtifact(
    grpc::ServerContext* ctx,
    const v1::KeepAliveRegisterArtifactRequest* req,
    v1::KeepAliveRegisterArtifactResponse* resp) {
  RpcContext rctx{"KeepAliveRegisterArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->keep_alive(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RevokeRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v1::RevokeRegisteredArtifactRequest* req,
    v1::RevokeRegisteredArtifactResponse* resp) {
  RpcContext rctx{"RevokeRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->revoke(rctx, *req, *resp);
}

void StoreDaemonServiceImpl::start_sweepers() {
  scheduler_ = std::make_unique<BackgroundScheduler>();
  using std::chrono::milliseconds;
  // Session lifecycle: unified task for sessions TTL, PID liveness, join TTL
  {
    lifecycle_mgr_ = std::make_shared<SessionLifecycleManager>(sessions_, refs_, *lip_mgr_, *engine_);
    // Create PID monitor: event-driven liveness via pidfd with /proc fallback
    pid_monitor_ = std::make_unique<PidMonitor>(
        [this](pid_t pid) {
          if (this->lifecycle_mgr_) {
            this->lifecycle_mgr_->handle_pid_exit(pid);
          }
        },
        std::chrono::duration_cast<std::chrono::milliseconds>(opts_.proc_check_interval));
    pid_monitor_->start();
    // Note: pid monitor fallback metric removed for minimal dependency surface.
    lifecycle_mgr_->attach_pid_monitor(pid_monitor_.get());
    auto lifecycle_task = std::make_shared<SessionLifecycleTask>(*lifecycle_mgr_);
    auto* sched_ptr = scheduler_.get();
    // Tighten deadline scheduling: reschedule lifecycle task when earliest deadline changes
    lifecycle_mgr_->set_schedule_hook([this, sched_ptr](absl::Time when) {
      absl::Duration delta;
      if (when == absl::InfiniteFuture()) {
        delta = absl::Milliseconds(
            std::chrono::duration_cast<std::chrono::milliseconds>(this->opts_.sessions_sweep_interval).count());
      } else {
        delta = when - absl::Now();
      }
      if (delta < absl::Milliseconds(1))
        delta = absl::Milliseconds(1);
      auto next = BackgroundScheduler::Clock::now() + std::chrono::milliseconds(absl::ToInt64Milliseconds(delta));
      if (sched_ptr)
        sched_ptr->set_next_due(TaskKind::kSessionLifecycle, next);
    });
    scheduler_->add_task(
        TaskKind::kSessionLifecycle, std::chrono::milliseconds(0), [lifecycle_task, this, sched_ptr]() {
          lifecycle_task->poll();
          if (sched_ptr) {
            absl::Time next_deadline = this->lifecycle_mgr_->next_deadline();
            absl::Duration delta;
            if (next_deadline == absl::InfiniteFuture()) {
              // Fallback: sleep for configured poll minimum when no deadlines
              delta = absl::Milliseconds(
                  std::chrono::duration_cast<std::chrono::milliseconds>(this->opts_.sessions_sweep_interval).count());
            } else {
              delta = next_deadline - absl::Now();
            }
            if (delta < absl::Milliseconds(1))
              delta = absl::Milliseconds(1);
            auto when = BackgroundScheduler::Clock::now() + std::chrono::milliseconds(absl::ToInt64Milliseconds(delta));
            sched_ptr->set_next_due(TaskKind::kSessionLifecycle, when);
          }
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
  // Region registry sweep
  {
    auto t = std::make_shared<RegionRegistrySweepTask>(*region_registry_);
    scheduler_->add_task(
        TaskKind::kRegionRegistry, std::chrono::duration_cast<milliseconds>(opts_.region_sweep_interval), [t]() {
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
  // Registration join TTL and PID watch unified under SessionLifecycleTask
  // Optional eviction
  if (opts_.enable_periodic_eviction) {
    const double limit = opts_.gpu_memory_limit_fraction;
    auto t = std::make_shared<EvictionTask>(*engine_, refs_, lifecycle_mgr_.get(), limit);
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
  if (pid_monitor_) {
    pid_monitor_->stop();
    pid_monitor_.reset();
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
