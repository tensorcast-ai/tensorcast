// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/target_publish_service.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/statusor.h"

#include "daemon/util/status_utils.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

absl::StatusOr<tensorcast::common::v1::ByteSpaceRef> normalize_byte_space(
    const tensorcast::common::v1::ByteSpaceRef& space) {
  tensorcast::common::v1::ByteSpaceRef out;
  switch (space.kind()) {
    case tensorcast::common::v1::BYTE_SPACE_KIND_UNSPECIFIED:
    case tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL:
      out.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
      out.set_id("");
      return out;
    case tensorcast::common::v1::BYTE_SPACE_KIND_VIEW:
      if (space.id().empty()) {
        return absl::InvalidArgumentError("byte_space VIEW requires id");
      }
      out.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
      out.set_id(space.id());
      return out;
    default:
      return absl::InvalidArgumentError("unsupported byte_space kind");
  }
}

} // namespace

TargetPublishService::TargetPublishService(Dep d)
    : d_(std::move(d)),
      capability_tokens_(d_.capability_tokens),
      target_write_registry_(TargetWriteRegistry::Options{.ttl = target_write_token_ttl()}) {}

absl::Duration TargetPublishService::target_write_token_ttl() {
  return absl::Minutes(5);
}

TargetWriteRegistry::Record TargetPublishService::remember_target_write(TargetWriteRegistry::Record record) {
  return target_write_registry_.insert(std::move(record));
}

grpc::Status TargetPublishService::publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::PublishTargetReplicaResponse& resp) {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs() && req.has_operation_id()) {
    span->SetAttribute("tc.operation.id", req.operation_id());
  }
  if (req.target_write_token().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "target_write_token is required"};
  }
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return {StatusCode::FAILED_PRECONDITION, "capability tokens not configured"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "Global Store client unavailable"};
  }

  auto normalized_req_or = normalize_byte_space(req.byte_space());
  if (!normalized_req_or.ok()) {
    return to_grpc_status(normalized_req_or.status());
  }
  tensorcast::common::v1::ByteSpaceRef normalized_req = std::move(*normalized_req_or);

  auto env_or = capability_tokens_->verify(
      req.target_write_token(),
      tensorcast::common::v1::CAPABILITY_AUDIENCE_TARGET_WRITE,
      d_.identity.daemon_id(),
      absl::Now(),
      /*require_not_expired=*/true);
  if (!env_or.ok()) {
    return to_grpc_status(env_or.status());
  }

  tensorcast::common::v1::TargetWriteScope scope;
  if (!scope.ParseFromString(env_or->scope())) {
    return {StatusCode::INVALID_ARGUMENT, "target_write_token scope parse failed"};
  }
  if (req.has_owner_pid() && scope.owner_pid() != req.owner_pid()) {
    return {StatusCode::PERMISSION_DENIED, "owner_pid mismatch for target_write_token"};
  }

  auto normalized_scope_or = normalize_byte_space(scope.byte_space());
  if (!normalized_scope_or.ok()) {
    return to_grpc_status(normalized_scope_or.status());
  }
  tensorcast::common::v1::ByteSpaceRef normalized_scope = std::move(*normalized_scope_or);
  if (normalized_scope.kind() != normalized_req.kind() || normalized_scope.id() != normalized_req.id()) {
    return {StatusCode::INVALID_ARGUMENT, "byte_space does not match target_write_token"};
  }

  if (scope.write_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "target_write_token missing write_id"};
  }

  auto record_opt = target_write_registry_.lookup(scope.write_id(), absl::Now(), /*require_not_expired=*/true);
  if (!record_opt.has_value()) {
    return {StatusCode::NOT_FOUND, "target_write_token is no longer valid"};
  }
  auto record = std::move(*record_opt);
  if (!target_write_registry_.is_current_for_layout(record.layout_key, scope.write_id())) {
    return {StatusCode::FAILED_PRECONDITION, "target_write_token is stale for layout"};
  }
  if (record.device_uuid != scope.device_uuid()) {
    return {StatusCode::FAILED_PRECONDITION, "device_uuid mismatch for target_write_token"};
  }
  if (record.owner_pid != scope.owner_pid()) {
    return {StatusCode::FAILED_PRECONDITION, "owner_pid mismatch for target_write_token"};
  }
  if (record.target_layout_hash != scope.target_layout_hash()) {
    return {StatusCode::FAILED_PRECONDITION, "target_layout_hash mismatch for target_write_token"};
  }
  if (record.byte_space.kind() != normalized_scope.kind() || record.byte_space.id() != normalized_scope.id()) {
    return {StatusCode::FAILED_PRECONDITION, "byte_space mismatch for target_write_token"};
  }
  if (record.selection.artifact_id() != scope.selection().artifact_id() ||
      record.selection.view_id() != scope.selection().view_id() ||
      record.selection.logical_layout_hash() != scope.selection().logical_layout_hash() ||
      record.selection.selection_hash() != scope.selection().selection_hash() ||
      record.selection.view_subset_hash() != scope.selection().view_subset_hash() ||
      record.selection.tensor_names_size() != scope.selection().tensor_names_size()) {
    return {StatusCode::FAILED_PRECONDITION, "selection mismatch for target_write_token"};
  }
  for (int i = 0; i < record.selection.tensor_names_size(); ++i) {
    if (record.selection.tensor_names(i) != scope.selection().tensor_names(i)) {
      return {StatusCode::FAILED_PRECONDITION, "selection tensor_names mismatch for target_write_token"};
    }
  }

  if (!scope.selection().tensor_names().empty() || !scope.selection().view_subset_hash().empty()) {
    return {StatusCode::FAILED_PRECONDITION, "selection is not publishable (packed or subset)"};
  }
  if (scope.selection().artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id missing from target_write_token"};
  }

  const auto device = d_.devices.From(v2::DeviceType::DEVICE_TYPE_GPU, scope.device_uuid(), std::nullopt);
  if (device.type != DeviceType::GPU || device.ordinal < 0) {
    return {StatusCode::INVALID_ARGUMENT, "invalid device_uuid for target_write_token"};
  }

  const std::string view_id =
      normalized_scope.kind() == tensorcast::common::v1::BYTE_SPACE_KIND_VIEW ? normalized_scope.id() : "";
  ArtifactDeviceKey key{
      .artifact_id = scope.selection().artifact_id(), .view_id = view_id, .device_id = device.ordinal};

  if (auto active = d_.lip_manager.find_active_by_key(key); active.has_value()) {
    if (active->registration_id == scope.write_id()) {
      auto replica_id = d_.lip_manager.find_replica_id(key);
      if (!replica_id.has_value()) {
        return {StatusCode::FAILED_PRECONDITION, "target already published without replica_id"};
      }
      resp.set_lease_id(scope.write_id());
      resp.set_replica_id(*replica_id);
      rctx.mark_success();
      return Status::OK;
    }
    return {StatusCode::ALREADY_EXISTS, "another lease already exists for target"};
  }

  uint64_t total_size = 0;
  for (const auto& seg : record.segments) {
    if (seg.length == 0) {
      continue;
    }
    const uint64_t end = seg.artifact_offset + seg.length;
    if (end > total_size) {
      total_size = end;
    }
  }
  if (total_size == 0) {
    return {StatusCode::FAILED_PRECONDITION, "target_write_token has empty segments"};
  }

  struct LipRollback {
    LipManager* lip{nullptr};
    std::string registration_id;
    bool active{true};

    ~LipRollback() {
      if (!active || lip == nullptr) {
        return;
      }
      absl::Status st = lip->revoke_by_registration_id(registration_id);
      if (!st.ok()) {
        LOG(WARNING) << "PublishTargetReplica rollback: revoke failed for id=" << registration_id << ": " << st;
      }
    }

    void release() {
      active = false;
    }
  } lip_rollback{.lip = &d_.lip_manager, .registration_id = scope.write_id()};

  const uint32_t ttl_ms = req.has_ttl_ms() ? req.ttl_ms() : 0U;
  const uint64_t epoch = static_cast<uint64_t>(absl::ToUnixMillis(absl::Now()));
  auto lease_or = d_.lip_manager.commit_routable_view_lease_in_place(
      scope.write_id(),
      scope.selection().artifact_id(),
      view_id,
      device.ordinal,
      scope.owner_pid(),
      ttl_ms,
      epoch,
      total_size,
      std::move(record.segments),
      std::move(record.storages));
  if (!lease_or.ok()) {
    lip_rollback.release();
    return to_grpc_status(lease_or.status());
  }

  std::string worker_id = d_.identity.worker_id();
  if (worker_id.empty()) {
    LOG(WARNING) << "worker_id is empty while publishing target replica for artifact_id="
                 << scope.selection().artifact_id() << " view_id=" << view_id
                 << "; using fallback worker_id='local' (transport eligibility may lag until worker lifecycle sync)";
    worker_id = "local";
  }

  auto replica_id_or = d_.global_store_client->register_memory_replica(
      scope.selection().artifact_id(),
      worker_id,
      device,
      total_size,
      record.index_key_hex,
      lease_or->remote_memory_keys,
      lease_or->buffer_sizes,
      record.canonical_index_json,
      /*encoding=*/"json",
      /*schema_version=*/"v3",
      /*max_concurrency=*/std::max<uint32_t>(1, d_.max_concurrency),
      /*verification_json=*/std::nullopt,
      view_id.empty() ? std::nullopt : std::optional<std::string_view>(view_id));
  if (!replica_id_or.ok()) {
    return to_grpc_status(replica_id_or.status());
  }
  const std::string replica_id = *replica_id_or;
  d_.lip_manager.attach_replica_id(scope.write_id(), replica_id);

  lip_rollback.release();
  resp.set_lease_id(scope.write_id());
  resp.set_replica_id(replica_id);
  rctx.mark_success();
  return Status::OK;
}

} // namespace tensorcast::daemon
