// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/persistence_rpc_controller.h"

#include <optional>

#include "daemon/state/store_policy_resolver.h"
#include "daemon/util/status_utils.h"

namespace tensorcast::daemon {

using ::grpc::StatusCode;
using status_utils::to_grpc_status;

grpc::Status PersistenceRpcController::start_persistence(
    RpcContext& rctx,
    const v2::StartPersistenceRequest& req,
    v2::StartPersistenceResponse& resp) {
  if (req.artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (d_.persistence_manager == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "persistence manager unavailable"};
  }
  auto policy_or = resolve_store_policy(req.has_policy() ? &req.policy() : nullptr);
  if (!policy_or.ok()) {
    return to_grpc_status(policy_or.status());
  }
  auto task_or = d_.persistence_manager->start_task(req.artifact_id(), *policy_or, req.key_hint());
  if (!task_or.ok()) {
    return to_grpc_status(task_or.status());
  }
  const auto& task = *task_or;

  resp.set_task_id(task.task_id);
  resp.set_plan_id(task.plan_id);
  resp.set_state(task.state);
  resp.set_progress(task.progress);
  if (!task.degraded_reason.empty()) {
    resp.set_degraded_reason(task.degraded_reason);
  }
  rctx.mark_success();
  return grpc::Status::OK;
}

grpc::Status PersistenceRpcController::query_persistence_status(
    RpcContext& rctx,
    const v2::QueryPersistenceStatusRequest& req,
    v2::QueryPersistenceStatusResponse& resp) {
  if (req.task_id().empty() && req.artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "task_id or artifact_id is required"};
  }
  std::optional<std::string> task_key;
  if (!req.task_id().empty()) {
    task_key = req.task_id();
  }
  if (d_.persistence_manager == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "persistence manager unavailable"};
  }
  absl::optional<PersistenceTaskState> task;
  if (task_key.has_value()) {
    task = d_.persistence_manager->get_by_task_id(*task_key);
  } else {
    task = d_.persistence_manager->get_latest_for_artifact(req.artifact_id());
  }

  if (!task.has_value()) {
    return {StatusCode::NOT_FOUND, "persistence task not found"};
  }
  resp.set_task_id(task_key.value_or(task->task_id));
  resp.set_artifact_id(task->artifact_id);
  resp.set_plan_id(task->plan_id);
  resp.set_state(task->state);
  resp.set_progress(task->progress);
  if (!task->degraded_reason.empty()) {
    resp.set_degraded_reason(task->degraded_reason);
  }
  if (!task->last_error.empty()) {
    resp.set_last_error(task->last_error);
  }
  for (const auto& shard : task->shards) {
    auto* out = resp.add_shards();
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
  return grpc::Status::OK;
}

} // namespace tensorcast::daemon
