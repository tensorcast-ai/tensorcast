// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/grpc_service_impl.h"

#include <nlohmann/json.hpp>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "core/store/components/global_store_client.h"
#include "core/store/device_registry.h"
#include "daemon/state/store_policy_resolver.h"
#include "daemon/util/path_utils.h"
#include "daemon/util/status_utils.h"
#include "folly/futures/Future.h"
#include "google/protobuf/util/time_util.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

std::string status_code_name(grpc::StatusCode code) {
  switch (code) {
    case grpc::StatusCode::OK:
      return "OK";
    case grpc::StatusCode::CANCELLED:
      return "CANCELLED";
    case grpc::StatusCode::UNKNOWN:
      return "UNKNOWN";
    case grpc::StatusCode::INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case grpc::StatusCode::DEADLINE_EXCEEDED:
      return "DEADLINE_EXCEEDED";
    case grpc::StatusCode::NOT_FOUND:
      return "NOT_FOUND";
    case grpc::StatusCode::ALREADY_EXISTS:
      return "ALREADY_EXISTS";
    case grpc::StatusCode::PERMISSION_DENIED:
      return "PERMISSION_DENIED";
    case grpc::StatusCode::RESOURCE_EXHAUSTED:
      return "RESOURCE_EXHAUSTED";
    case grpc::StatusCode::FAILED_PRECONDITION:
      return "FAILED_PRECONDITION";
    case grpc::StatusCode::ABORTED:
      return "ABORTED";
    case grpc::StatusCode::OUT_OF_RANGE:
      return "OUT_OF_RANGE";
    case grpc::StatusCode::UNIMPLEMENTED:
      return "UNIMPLEMENTED";
    case grpc::StatusCode::INTERNAL:
      return "INTERNAL";
    case grpc::StatusCode::UNAVAILABLE:
      return "UNAVAILABLE";
    case grpc::StatusCode::DATA_LOSS:
      return "DATA_LOSS";
    case grpc::StatusCode::UNAUTHENTICATED:
      return "UNAUTHENTICATED";
    default:
      return "UNKNOWN";
  }
}

bool is_retryable(grpc::StatusCode code) {
  switch (code) {
    case grpc::StatusCode::UNAVAILABLE:
    case grpc::StatusCode::DEADLINE_EXCEEDED:
    case grpc::StatusCode::RESOURCE_EXHAUSTED:
    case grpc::StatusCode::ABORTED:
      return true;
    default:
      return false;
  }
}

std::string replica_key_hash_bytes(const store::loading::ReplicaKey& key) {
  const uint64_t h = static_cast<uint64_t>(store::loading::ReplicaKeyHash{}(key));
  std::string out(sizeof(h), '\0');
  std::memcpy(out.data(), &h, sizeof(h));
  return out;
}

void fill_success_status(v2::ReplicaOperationStatus& out) {
  out.set_state(v2::ReplicaOperationState::REPLICA_OPERATION_STATE_SUCCESS);
  *out.mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
}

void fill_running_status(v2::ReplicaOperationStatus& out) {
  out.set_state(v2::ReplicaOperationState::REPLICA_OPERATION_STATE_RUNNING);
  *out.mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
}

void fill_failed_status(v2::ReplicaOperationStatus& out, const absl::Status& st) {
  grpc::Status grpc_st = status_utils::to_grpc_status(st);
  out.set_state(v2::ReplicaOperationState::REPLICA_OPERATION_STATE_FAILED);
  out.set_message(std::string(st.message()));
  *out.mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
  auto* err = out.mutable_error();
  err->set_status_code(status_code_name(grpc_st.error_code()));
  err->set_message(std::string(st.message()));
  err->set_retryable(is_retryable(grpc_st.error_code()));
}

void fill_degraded_timeout_status(v2::ReplicaOperationStatus& out, std::string_view message) {
  out.set_state(v2::ReplicaOperationState::REPLICA_OPERATION_STATE_DEGRADED);
  out.set_message(std::string(message));
  *out.mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
  auto* err = out.mutable_error();
  err->set_status_code("DEADLINE_EXCEEDED");
  err->set_message(std::string(message));
  err->set_retryable(true);
}

} // namespace

StoreDaemonServiceImpl::StoreDaemonServiceImpl(Deps deps, Options opts)
    : engine_(&deps.engine),
      materialization_controller_(&deps.materialization_controller),
      registration_controller_(&deps.registration_controller),
      transport_controller_(&deps.transport_controller),
      status_controller_(&deps.status_controller),
      region_registry_(&deps.region_registry),
      lip_manager_(&deps.lip_manager),
      persistence_manager_(deps.persistence_manager),
      sessions_service_(&deps.sessions_service),
      lifecycle_manager_(&deps.lifecycle_manager),
      placement_lease_tokens_(&deps.placement_lease_tokens),
      shutdown_signal_(&deps.shutdown_signal),
      opts_(std::move(opts)) {}

Status StoreDaemonServiceImpl::MaterializeReplica(
    grpc::ServerContext* ctx,
    const v2::MaterializeReplicaRequest* req,
    v2::MaterializeReplicaResponse* resp) {
  RpcContext rctx{"MaterializeReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->materialize_replica(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::MaterializeIntoTarget(
    grpc::ServerContext* ctx,
    const v2::MaterializeIntoTargetRequest* req,
    v2::MaterializeIntoTargetResponse* resp) {
  RpcContext rctx{"MaterializeIntoTarget", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->materialize_into_target(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ConfirmReplica(
    grpc::ServerContext* ctx,
    const v2::ConfirmReplicaRequest* req,
    v2::ConfirmReplicaResponse* resp) {
  RpcContext rctx{"ConfirmReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->confirm(rctx, *req, *resp);
}

// Key-based materialization via Global Store mapping.
Status StoreDaemonServiceImpl::MaterializeByKey(
    grpc::ServerContext* ctx,
    const v2::MaterializeByKeyRequest* req,
    v2::MaterializeByKeyResponse* resp) {
  RpcContext rctx{"MaterializeByKey", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->materialize_by_key(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ResolveArtifactFromDisk(
    grpc::ServerContext* ctx,
    const v2::ResolveArtifactFromDiskRequest* req,
    v2::ResolveArtifactFromDiskResponse* resp) {
  RpcContext rctx{"ResolveArtifactFromDisk", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->resolve_artifact_from_disk(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::QueryReplicaStatus(
    grpc::ServerContext* ctx,
    const v2::QueryReplicaStatusRequest* req,
    v2::QueryReplicaStatusResponse* resp) {
  RpcContext rctx{"QueryReplicaStatus", *ctx, opts_.allow_high_card_attrs};
  if (!req->has_ticket() || req->ticket().replica_uuid().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "ticket.replica_uuid is required"};
  }
  const std::string& replica_uuid = req->ticket().replica_uuid();
  auto entry = sessions_service_->get(replica_uuid);
  if (!entry.has_value()) {
    return {StatusCode::NOT_FOUND, "replica_uuid not found"};
  }

  resp->mutable_ticket()->set_replica_uuid(replica_uuid);
  resp->set_replica_key_hash(replica_key_hash_bytes(entry->key));

  auto* status = resp->mutable_status();
  if (!entry->ready_signal) {
    fill_success_status(*status);
  } else if (!entry->ready_signal->is_ready()) {
    fill_running_status(*status);
  } else {
    const absl::Status st = entry->wait_ready(std::chrono::milliseconds(0));
    if (st.ok()) {
      fill_success_status(*status);
    } else {
      fill_failed_status(*status, st);
    }
  }

  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::WaitReplicaStatus(
    grpc::ServerContext* ctx,
    const v2::WaitReplicaStatusRequest* req,
    v2::WaitReplicaStatusResponse* resp) {
  RpcContext rctx{"WaitReplicaStatus", *ctx, opts_.allow_high_card_attrs};
  if (!req->has_ticket() || req->ticket().replica_uuid().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "ticket.replica_uuid is required"};
  }
  const std::string& replica_uuid = req->ticket().replica_uuid();
  auto entry = sessions_service_->get(replica_uuid);
  if (!entry.has_value()) {
    return {StatusCode::NOT_FOUND, "replica_uuid not found"};
  }

  resp->mutable_ticket()->set_replica_uuid(replica_uuid);
  resp->set_replica_key_hash(replica_key_hash_bytes(entry->key));

  auto* status = resp->mutable_status();
  if (!entry->ready_signal) {
    fill_success_status(*status);
    rctx.mark_success();
    return Status::OK;
  }
  if (entry->ready_signal->is_ready()) {
    const absl::Status st = entry->wait_ready(std::chrono::milliseconds(0));
    if (st.ok()) {
      fill_success_status(*status);
    } else {
      fill_failed_status(*status, st);
    }
    rctx.mark_success();
    return Status::OK;
  }

  // timeout_ms is a server-side wait budget. When unset/0, the server waits until the replica reaches a terminal
  // state or until the RPC deadline expires.
  const std::chrono::milliseconds user_timeout(static_cast<int64_t>(req->timeout_ms()));
  const auto start = std::chrono::steady_clock::now();
  const std::optional<std::chrono::steady_clock::time_point> user_deadline = user_timeout.count() > 0
      ? std::optional<std::chrono::steady_clock::time_point>(start + user_timeout)
      : std::nullopt;

  constexpr std::chrono::milliseconds k_wait_slice(250);

  while (true) {
    std::optional<std::chrono::milliseconds> remaining;
    if (user_deadline.has_value()) {
      remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(*user_deadline - std::chrono::steady_clock::now());
    }

    using clock = std::chrono::system_clock;
    const auto grpc_deadline = ctx->deadline();
    if (grpc_deadline != clock::time_point::max()) {
      const auto now = clock::now();
      const auto d = grpc_deadline <= now ? std::chrono::milliseconds(0)
                                          : std::chrono::duration_cast<std::chrono::milliseconds>(grpc_deadline - now);
      remaining = remaining.has_value() ? std::min(*remaining, d) : d;
    }

    if (remaining.has_value() && remaining->count() <= 0) {
      fill_degraded_timeout_status(*status, "replica wait budget exhausted");
      rctx.mark_success();
      return Status::OK;
    }

    if (ctx->IsCancelled()) {
      return {StatusCode::CANCELLED, "request cancelled"};
    }

    const std::chrono::milliseconds slice = remaining.has_value() ? std::min(*remaining, k_wait_slice) : k_wait_slice;
    if (slice.count() <= 0) {
      fill_degraded_timeout_status(*status, "replica wait budget exhausted");
      rctx.mark_success();
      return Status::OK;
    }

    absl::Status wait_st;
    try {
      wait_st = std::move(entry->subscribe_ready()).get(slice);
    } catch (const folly::FutureTimeout&) {
      continue;
    } catch (const std::exception& ex) {
      wait_st = absl::InternalError(ex.what());
    } catch (...) {
      wait_st = absl::InternalError("replica wait_ready failed with unknown exception");
    }

    if (wait_st.ok()) {
      fill_success_status(*status);
    } else if (absl::IsDeadlineExceeded(wait_st)) {
      fill_degraded_timeout_status(*status, wait_st.message());
    } else {
      fill_failed_status(*status, wait_st);
    }
    rctx.mark_success();
    return Status::OK;
  }
}

Status StoreDaemonServiceImpl::ReleaseReplica(
    grpc::ServerContext* ctx,
    const v2::ReleaseReplicaRequest* req,
    v2::ReleaseReplicaResponse* resp) {
  RpcContext rctx{"ReleaseReplica", *ctx, opts_.allow_high_card_attrs};
  if (!req->has_ticket() || req->ticket().replica_uuid().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "ticket.replica_uuid is required"};
  }
  const std::string& replica_uuid = req->ticket().replica_uuid();
  const bool erased = sessions_service_->erase(replica_uuid);
  lifecycle_manager_->release_session(replica_uuid);
  resp->set_released(erased);
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::CreatePlacementLease(
    grpc::ServerContext* ctx,
    const v2::CreatePlacementLeaseRequest* req,
    v2::CreatePlacementLeaseResponse* resp) {
  RpcContext rctx{"CreatePlacementLease", *ctx, opts_.allow_high_card_attrs};
  if (req->artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  const int device_id = req->device_id();
  if (device_id < 0 || device_id >= engine_->get_num_gpus()) {
    return {StatusCode::INVALID_ARGUMENT, "invalid device_id"};
  }

  store::loading::ReplicaKey key;
  key.artifact_id = req->artifact_id();
  if (!req->view_id().empty()) {
    key.view_id = req->view_id();
  }
  key.device = store::DeviceRegistry::instance().gpu_key(device_id);
  key.replica = 0;

  const absl::Duration ttl =
      req->has_ttl_ms() ? absl::Milliseconds(static_cast<int64_t>(req->ttl_ms())) : absl::ZeroDuration();
  auto lease_or = lifecycle_manager_->create_placement_lease(key, ttl);
  if (!lease_or.ok()) {
    return to_grpc_status(lease_or.status());
  }
  const auto token_or = placement_lease_tokens_->mint(*lease_or, ttl);
  if (!token_or.ok()) {
    lifecycle_manager_->release_lease(*lease_or);
    return to_grpc_status(token_or.status());
  }

  resp->set_lease_id(*lease_or);
  resp->set_lease_token(*token_or);
  if (ttl > absl::ZeroDuration()) {
    const absl::Time expires_at = absl::Now() + ttl;
    google::protobuf::Timestamp ts;
    ts.set_seconds(absl::ToUnixSeconds(expires_at));
    ts.set_nanos(static_cast<int32_t>(absl::ToInt64Nanoseconds(expires_at - absl::FromUnixSeconds(ts.seconds()))));
    *resp->mutable_expires_at() = ts;
  }
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::RenewPlacementLease(
    grpc::ServerContext* ctx,
    const v2::RenewPlacementLeaseRequest* req,
    v2::RenewPlacementLeaseResponse* resp) {
  RpcContext rctx{"RenewPlacementLease", *ctx, opts_.allow_high_card_attrs};
  if (req->lease_token().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "lease_token is required"};
  }
  if (req->ttl_ms() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "ttl_ms is required"};
  }
  const absl::Duration ttl = absl::Milliseconds(static_cast<int64_t>(req->ttl_ms()));
  auto lease_or = placement_lease_tokens_->resolve(req->lease_token());
  if (!lease_or.ok()) {
    return to_grpc_status(lease_or.status());
  }
  auto st = lifecycle_manager_->renew_placement(*lease_or, ttl);
  if (!st.ok()) {
    return to_grpc_status(st);
  }
  st = placement_lease_tokens_->refresh(req->lease_token(), ttl);
  if (!st.ok()) {
    return to_grpc_status(st);
  }
  resp->set_lease_id(*lease_or);
  const absl::Time expires_at = absl::Now() + ttl;
  google::protobuf::Timestamp ts;
  ts.set_seconds(absl::ToUnixSeconds(expires_at));
  ts.set_nanos(static_cast<int32_t>(absl::ToInt64Nanoseconds(expires_at - absl::FromUnixSeconds(ts.seconds()))));
  *resp->mutable_expires_at() = ts;
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::ReleasePlacementLease(
    grpc::ServerContext* ctx,
    const v2::ReleasePlacementLeaseRequest* req,
    v2::ReleasePlacementLeaseResponse* resp) {
  RpcContext rctx{"ReleasePlacementLease", *ctx, opts_.allow_high_card_attrs};
  if (req->lease_token().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "lease_token is required"};
  }
  auto lease_or = placement_lease_tokens_->resolve(req->lease_token());
  if (!lease_or.ok()) {
    return to_grpc_status(lease_or.status());
  }
  lifecycle_manager_->release_lease(*lease_or);
  const bool erased = placement_lease_tokens_->erase(req->lease_token());
  resp->set_released(erased);
  rctx.mark_success();
  return Status::OK;
}

// Publish key mapping via Global Store.
Status StoreDaemonServiceImpl::PublishReplicaKey(
    grpc::ServerContext* ctx,
    const v2::PublishReplicaKeyRequest* req,
    v2::PublishReplicaKeyResponse* resp) {
  RpcContext rctx{"PublishReplicaKey", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req->key());

  if (req->key().empty() || !req->has_artifact_descriptor() || req->artifact_descriptor().artifact_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "key and artifact_descriptor.artifact_id are required"};
  }
  if (shutdown_signal_->is_shutting_down()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  std::string normalized_disk_path;
  if (!req->disk_path().empty()) {
    auto normalized_or = normalize_disk_path(req->disk_path(), opts_.storage_path);
    if (!normalized_or.ok()) {
      return to_grpc_status(normalized_or.status());
    }
    normalized_disk_path = normalized_or->string();
  }

  // Use engine's configured Global Store client for upsert.
  auto up = engine_->upsert_key_mapping(req->key(), req->artifact_descriptor().artifact_id(), normalized_disk_path);
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
    const v2::ResolveKeyMappingRequest* req,
    v2::ResolveKeyMappingResponse* resp) {
  RpcContext rctx{"ResolveKeyMapping", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req->key());

  if (req->key().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "key is required"};
  }
  if (shutdown_signal_->is_shutting_down()) {
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
    const v2::GetArtifactIndexByIdRequest* req,
    v2::GetArtifactIndexByIdResponse* resp) {
  RpcContext rctx{"GetArtifactIndexById", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->get_artifact_index_by_id(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::SealAssembly(
    grpc::ServerContext* ctx,
    const v2::SealAssemblyRequest* req,
    v2::SealAssemblyResponse* resp) {
  RpcContext rctx{"SealAssembly", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->seal_assembly(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::StartSealAssembly(
    grpc::ServerContext* ctx,
    const v2::StartSealAssemblyRequest* req,
    v2::StartSealAssemblyResponse* resp) {
  RpcContext rctx{"StartSealAssembly", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->start_seal_assembly(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::GetOperation(
    grpc::ServerContext* ctx,
    const tensorcast::operation::v1::GetOperationRequest* req,
    tensorcast::operation::v1::GetOperationResponse* resp) {
  RpcContext rctx{"GetOperation", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->get_operation(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::WaitOperation(
    grpc::ServerContext* ctx,
    const v2::WaitOperationRequest* req,
    v2::WaitOperationResponse* resp) {
  RpcContext rctx{"WaitOperation", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->wait_operation(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::StartPersistence(
    grpc::ServerContext* ctx,
    const v2::StartPersistenceRequest* req,
    v2::StartPersistenceResponse* resp) {
  RpcContext rctx{"StartPersistence", *ctx, opts_.allow_high_card_attrs};
  if (req->artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  if (shutdown_signal_->is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!persistence_manager_) {
    return {StatusCode::FAILED_PRECONDITION, "persistence manager unavailable"};
  }
  auto policy_or = resolve_store_policy(req->has_policy() ? &req->policy() : nullptr);
  if (!policy_or.ok()) {
    return to_grpc_status(policy_or.status());
  }
  auto task_or = persistence_manager_->start_task(req->artifact_id(), *policy_or);
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
    const v2::QueryPersistenceStatusRequest* req,
    v2::QueryPersistenceStatusResponse* resp) {
  RpcContext rctx{"QueryPersistenceStatus", *ctx, opts_.allow_high_card_attrs};
  if (req->task_id().empty() && req->artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "task_id or artifact_id is required"};
  }
  std::optional<std::string> task_key;
  if (!req->task_id().empty()) {
    task_key = req->task_id();
  }
  if (!persistence_manager_) {
    return {StatusCode::FAILED_PRECONDITION, "persistence manager unavailable"};
  }
  absl::optional<PersistenceTaskState> task;
  if (task_key.has_value()) {
    task = persistence_manager_->get_by_task_id(*task_key);
  } else {
    task = persistence_manager_->get_latest_for_artifact(req->artifact_id());
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
    const v2::UnloadReplicaRequest* req,
    v2::UnloadReplicaResponse* resp) {
  RpcContext rctx{"UnloadReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->unload(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ClearMem(
    grpc::ServerContext* ctx,
    const v2::ClearMemRequest* /*req*/,
    v2::ClearMemResponse* /*resp*/) {
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
    const v2::GetServerConfigRequest* /*req*/,
    v2::GetServerConfigResponse* resp) {
  RpcContext rctx{"GetServerConfig", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_server_config(rctx, *resp);
}

// ──────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────

// Legacy device/key helpers have been removed in favor of DeviceResolver

Status StoreDaemonServiceImpl::WaitReplicaVerification(
    grpc::ServerContext* ctx,
    const v2::WaitReplicaVerificationRequest* req,
    v2::WaitReplicaVerificationResponse* resp) {
  RpcContext rctx{"WaitReplicaVerification", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->wait_verification(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::LockTransportChunks(
    grpc::ServerContext* ctx,
    const v2::LockTransportChunksRequest* req,
    v2::LockTransportChunksResponse* resp) {
  RpcContext rctx{"LockTransportChunks", *ctx, opts_.allow_high_card_attrs};
  return transport_controller_->lock(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RegisterVramRegion(
    grpc::ServerContext* ctx,
    const v2::RegisterVramRegionRequest* req,
    v2::RegisterVramRegionResponse* resp) {
  RpcContext rctx{"RegisterVramRegion", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.device.id", static_cast<int64_t>(req->device_id()));
  span->SetAttribute("tc.region.size_bytes", static_cast<int64_t>(req->size_bytes()));
  span->SetAttribute("tc.region.ttl_ms", static_cast<int64_t>(req->ttl_ms()));

  if (shutdown_signal_->is_shutting_down()) {
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
    const v2::UnregisterVramRegionRequest* req,
    v2::UnregisterVramRegionResponse* resp) {
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
    const v2::DeregisterArtifactRequest* req,
    v2::DeregisterArtifactResponse* resp) {
  RpcContext rctx{"DeregisterArtifact", *ctx, opts_.allow_high_card_attrs};
  if (!req->artifact_id().empty()) {
    rctx.span()->SetAttribute("tc.artifact.id", req->artifact_id());
  }
  if (shutdown_signal_->is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req->artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  const std::string artifact_id = req->artifact_id();
  auto lip_opt = lip_manager_->find_active_by_artifact_id(artifact_id);
  if (!lip_opt.has_value()) {
    return {StatusCode::NOT_FOUND, "no active lease for artifact"};
  }
  const auto& lip = *lip_opt;
  if (req->has_owner_pid() && lip.owner_pid != req->owner_pid()) {
    return {StatusCode::PERMISSION_DENIED, "owner_pid mismatch for active lease"};
  }
  // Optional TTL extension before quiesce
  if (req->has_extend_ttl_ms() && req->extend_ttl_ms() > 0) {
    auto st = lip_manager_->extend_ttl_for_artifact(artifact_id, req->extend_ttl_ms());
    if (!st.ok())
      return to_grpc_status(st);
  }
  // Quiesce new staged exports
  lip_manager_->quiesce_artifact(artifact_id);
  bool drained = true;
  if (req->wait_for_drain()) {
    const uint32_t timeout_ms = req->has_drain_timeout_ms() ? req->drain_timeout_ms() : 30000U;
    absl::Time deadline = absl::Now() + absl::Milliseconds(timeout_ms);
    drained = lip_manager_->wait_exports_drained(artifact_id, deadline);
    if (!drained) {
      return {StatusCode::DEADLINE_EXCEEDED, "drain timed out; artifact remains quiesced"};
    }
  }
  // Remove active lease if owner matches (or no owner provided)
  bool removed = lip_manager_->revoke_commit_lease_if_owner_matches(artifact_id, lip.device_id, lip.owner_pid);
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
    const v2::UnlockTransportChunksRequest* req,
    v2::UnlockTransportChunksResponse* /*resp*/) {
  RpcContext rctx{"UnlockTransportChunks", *ctx, opts_.allow_high_card_attrs};
  v2::UnlockTransportChunksResponse dummy;
  return transport_controller_->unlock(rctx, *req, dummy);
}

Status StoreDaemonServiceImpl::BeginRegisterArtifact(
    grpc::ServerContext* ctx,
    const v2::BeginRegisterArtifactRequest* req,
    v2::BeginRegisterArtifactResponse* resp) {
  RpcContext rctx{"BeginRegisterArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->begin(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::CommitRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v2::CommitRegisteredArtifactRequest* req,
    v2::CommitRegisteredArtifactResponse* resp) {
  RpcContext rctx{"CommitRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->commit(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::AbortRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v2::AbortRegisteredArtifactRequest* req,
    v2::AbortRegisteredArtifactResponse* resp) {
  RpcContext rctx{"AbortRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->abort(rctx, *req, *resp);
}

// Removed unary FeedRegisterArtifact; use streaming variant only

Status StoreDaemonServiceImpl::FeedRegisterArtifactStream(
    grpc::ServerContext* ctx,
    ::grpc::ServerReader<v2::FeedRegisterArtifactStreamRequest>* reader,
    v2::FeedRegisterArtifactStreamResponse* resp) {
  RpcContext rctx{"FeedRegisterArtifactStream", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->feed_stream(rctx, *reader, *resp);
}

grpc::Status StoreDaemonServiceImpl::feed_register_artifact_stream_vector(
    const std::vector<v2::FeedRegisterArtifactStreamRequest>& reqs) {
  return registration_controller_->feed_vector(reqs);
}

Status StoreDaemonServiceImpl::KeepAliveRegisterArtifact(
    grpc::ServerContext* ctx,
    const v2::KeepAliveRegisterArtifactRequest* req,
    v2::KeepAliveRegisterArtifactResponse* resp) {
  RpcContext rctx{"KeepAliveRegisterArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->keep_alive(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RevokeRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v2::RevokeRegisteredArtifactRequest* req,
    v2::RevokeRegisteredArtifactResponse* resp) {
  RpcContext rctx{"RevokeRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->revoke(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::GetWorkerStatus(
    grpc::ServerContext* ctx,
    const v2::GetWorkerStatusRequest* /*req*/,
    v2::GetWorkerStatusResponse* resp) {
  RpcContext rctx{"GetWorkerStatus", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_worker_status(rctx, *resp);
}

Status StoreDaemonServiceImpl::GetDetailedStatus(
    grpc::ServerContext* ctx,
    const v2::GetDetailedStatusRequest* /*req*/,
    v2::GetDetailedStatusResponse* resp) {
  RpcContext rctx{"GetDetailedStatus", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_detailed_status(rctx, *resp);
}

// verification tracking moved to VerificationTracker

// Legacy GetLoadedReplicas removed; use V2

Status StoreDaemonServiceImpl::GetLoadedReplicasV2(
    grpc::ServerContext* ctx,
    const v2::GetLoadedReplicasV2Request* req,
    v2::GetLoadedReplicasV2Response* resp) {
  RpcContext rctx{"GetLoadedReplicasV2", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_loaded_replicas_v2(rctx, *req, *resp, opts_.use_cursor_pagination);
}

} // namespace tensorcast::daemon
