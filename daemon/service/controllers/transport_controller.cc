// Copyright (c) 2025-2026, TensorCast Team.

// Implementation of TransportController

#include "daemon/service/controllers/transport_controller.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "core/store/device_registry.h"
#include "daemon/service/controllers/materialization_controller.h"
#include "daemon/state/distributed_security_kernel.h"
#include "daemon/state/routed_authority_wire.h"
#include "daemon/util/status_utils.h"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon {

using ::grpc::Status;
using status_utils::to_grpc_status;

namespace {

std::optional<std::string> serialize_resolved_source_capability_payload(const OwnerStageReply& owner_stage_reply) {
  if (owner_stage_reply.resolved_source_capability) {
    nlohmann::json json;
    json["artifact_id"] = owner_stage_reply.resolved_source_capability->selection_identity.artifact_id;
    json["logical_layout_hash_hex"] = absl::BytesToHexString(
        absl::string_view(owner_stage_reply.resolved_source_capability->selection_identity.logical_layout_hash));
    json["selection_hash_hex"] = absl::BytesToHexString(
        absl::string_view(owner_stage_reply.resolved_source_capability->selection_identity.selection_hash));
    json["size_bytes"] =
        owner_stage_reply.resolved_source_capability->verified_content_descriptor.content_identity.logical_size_bytes;
    json["digest_alg"] =
        owner_stage_reply.resolved_source_capability->verified_content_descriptor.content_identity.digest_alg;
    json["digest_hex"] = store::runtime::ingestion::content_digest_bytes_to_hex(
        owner_stage_reply.resolved_source_capability->verified_content_descriptor.content_identity.digest_bytes);
    json["capability_id_hex"] = absl::BytesToHexString(
        absl::string_view(owner_stage_reply.resolved_source_capability->serving_capability.capability_id));
    json["expires_at_ms"] =
        absl::ToUnixMillis(owner_stage_reply.resolved_source_capability->serving_capability.expires_at);
    json["mode"] = static_cast<int>(owner_stage_reply.resolved_source_capability->serving_capability.mode);
    json["local"] = owner_stage_reply.resolved_source_capability->serving_capability.local;
    json["subject_kind"] =
        static_cast<int>(owner_stage_reply.resolved_source_capability->serving_capability.subject_kind);
    json["owner_kind"] = static_cast<int>(
        owner_stage_reply.resolved_source_capability->serving_capability.lifecycle_owner_ref.owner_kind);
    json["owner_id"] = owner_stage_reply.resolved_source_capability->serving_capability.lifecycle_owner_ref.owner_id;
    json["source_kind"] = static_cast<int>(owner_stage_reply.resolved_source_capability->source_kind);
    json["payload_ref_hex"] =
        absl::BytesToHexString(absl::string_view(owner_stage_reply.resolved_source_capability->payload_ref));
    return json.dump();
  }
  return std::nullopt;
}

absl::Status populate_proto_owner_stage_reply(
    const OwnerStageReply& owner_stage_reply,
    v2::OwnerStageReply* proto_reply) {
  return routed_authority_wire::populate_proto_owner_stage_reply_shell(
      owner_stage_reply, serialize_resolved_source_capability_payload(owner_stage_reply), proto_reply);
}

} // namespace

grpc::Status TransportController::lock(
    RpcContext& rctx,
    const v2::LockTransportChunksRequest& req,
    v2::LockTransportChunksResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.artifact_id());
  if (req.has_device_id())
    span->SetAttribute("tc.device.id", static_cast<int64_t>(req.device_id()));

  std::optional<std::string> requested_view_id;
  if (req.has_byte_space()) {
    const auto& space = req.byte_space();
    switch (space.kind()) {
      case tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL:
      case tensorcast::common::v1::BYTE_SPACE_KIND_UNSPECIFIED:
        break;
      case tensorcast::common::v1::BYTE_SPACE_KIND_VIEW:
        if (space.id().empty()) {
          return {grpc::StatusCode::INVALID_ARGUMENT, "byte_space VIEW requires id"};
        }
        requested_view_id = space.id();
        break;
      default:
        return {grpc::StatusCode::INVALID_ARGUMENT, "unsupported byte_space kind"};
    }
  }

  store::loading::ReplicaKey key;
  key.artifact_id = req.artifact_id();
  if (requested_view_id.has_value()) {
    key.view_id = *requested_view_id;
  }
  key.replica = 0;
  // Resolve device lazily to avoid defaulting to GPU0 prematurely
  if (req.has_device_id()) {
    key.device = store::DeviceRegistry::instance().gpu_key(req.device_id());
  }

  // LIP staged export path
  if (!requested_view_id.has_value()) {
    if (auto lip_opt = d_.lip.find_active_by_artifact_id(req.artifact_id(), std::nullopt); lip_opt.has_value()) {
      const auto& lip = *lip_opt;
      if (req.has_extend_ttl_ms() && req.extend_ttl_ms() > 0) {
        auto st = d_.lip.extend_ttl_for_artifact(req.artifact_id(), req.extend_ttl_ms(), std::nullopt);
        if (!st.ok())
          return to_grpc_status(st);
      }
      std::vector<uint32_t> indices(req.chunk_indices().begin(), req.chunk_indices().end());
      auto tok_or = d_.lip.create_staged_export(lip, absl::MakeSpan(indices), d_.engine);
      if (!tok_or.ok())
        return to_grpc_status(tok_or.status());
      resp.set_lock_token(*tok_or);
      if (!lip.verification_json.empty())
        resp.set_verification_json(lip.verification_json);
      rctx.mark_success();
      return Status::OK;
    }
  }

  if (!req.has_device_id()) {
    auto dev_or = d_.engine.get_unique_gpu_residency(req.artifact_id(), requested_view_id);
    if (!dev_or.ok())
      return to_grpc_status(dev_or.status());
    if (*dev_or >= 0) {
      key.device = store::DeviceRegistry::instance().gpu_key(*dev_or);
    } else {
      // No GPU residency found; do not default to GPU0. Require explicit device_id or pre-load.
      return grpc::Status(grpc::StatusCode::NOT_FOUND, "artifact not resident on any GPU; specify device_id or load");
    }
  }

  // UMA V3: No engine-level transfer locks; UMA plan/commit supersedes any prior locking.
  // Maintain a daemon-local token workflow strictly for bookkeeping.
  std::vector<uint32_t> indices(req.chunk_indices().begin(), req.chunk_indices().end());
  std::string token = d_.locks.mint_token();
  d_.locks.put(token, key, std::move(indices));
  resp.set_lock_token(token);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status TransportController::unlock(
    RpcContext& rctx,
    const v2::UnlockTransportChunksRequest& req,
    v2::UnlockTransportChunksResponse& /*resp*/) {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs())
    span->SetAttribute("tc.lock.token", req.lock_token());
  auto entry = d_.locks.get(req.lock_token());
  if (!entry.has_value()) {
    // Try staged LIP export unlock; if not found there either, treat as idempotent success.
    auto st = d_.lip.release_staged_export(req.lock_token(), d_.engine);
    if (!st.ok()) {
      // If unknown token in LIP exports, do not fail; return OK for idempotency
      rctx.mark_success();
      return Status::OK;
    }
    rctx.mark_success();
    return Status::OK;
  }
  // UMA V3: No engine-level unlock; treat daemon unlock as idempotent bookkeeping: just erase the token.
  d_.locks.erase(req.lock_token());
  rctx.mark_success();
  return Status::OK;
}

grpc::Status TransportController::fetch_payload_ref_chunk(
    RpcContext& rctx,
    const v2::FetchPayloadRefChunkRequest& req,
    v2::FetchPayloadRefChunkResponse& resp) {
  if (d_.payload_transport_broker == nullptr) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "payload transport broker unavailable"};
  }
  if (req.payload_ref().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "payload_ref is required"};
  }
  if (req.artifact_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }

  const std::uint64_t max_bytes =
      req.max_bytes() == 0 ? d_.payload_transport_broker->max_chunk_bytes() : req.max_bytes();
  auto chunk_or = d_.payload_transport_broker->read_local_payload_ref_chunk(
      req.payload_ref(),
      req.artifact_id(),
      absl::Now(),
      req.offset(),
      max_bytes,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED,
      req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view(""));
  if (!chunk_or.ok()) {
    resp.set_status(v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION);
    resp.set_message(std::string(chunk_or.status().message()));
    rctx.mark_success();
    return Status::OK;
  }
  resp.set_status(v2::BATCH_ITEM_STATUS_OK);
  resp.set_total_size(chunk_or->metadata.payload_size);
  resp.set_eof(chunk_or->eof);
  resp.set_chunk(chunk_or->chunk);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status TransportController::fetch_batch_payload_ref_chunk(
    RpcContext& rctx,
    const v2::FetchBatchPayloadRefChunkRequest& req,
    v2::FetchBatchPayloadRefChunkResponse& resp) {
  if (d_.payload_transport_broker == nullptr) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "payload transport broker unavailable"};
  }
  if (req.batch_payload_ref().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "batch_payload_ref is required"};
  }

  const std::uint64_t max_bytes =
      req.max_bytes() == 0 ? d_.payload_transport_broker->max_chunk_bytes() : req.max_bytes();
  auto chunk_or = d_.payload_transport_broker->read_local_batch_payload_ref_chunk(
      req.batch_payload_ref(),
      absl::Now(),
      req.offset(),
      max_bytes,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED,
      req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view(""));
  if (!chunk_or.ok()) {
    resp.set_status(v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION);
    resp.set_message(std::string(chunk_or.status().message()));
    rctx.mark_success();
    return Status::OK;
  }
  resp.set_status(v2::BATCH_ITEM_STATUS_OK);
  resp.set_total_size(chunk_or->metadata.payload_size);
  resp.set_eof(chunk_or->eof);
  resp.set_chunk(chunk_or->chunk);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status TransportController::route_authority_stage(
    RpcContext& rctx,
    const v2::RouteAuthorityStageRequest& req,
    v2::RouteAuthorityStageResponse& resp) {
  if (!req.has_routed_request()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "routed_request is required"};
  }
  const auto& proto_routed_request = req.routed_request();
  auto routed_request_or = routed_authority_wire::routed_authority_request_from_proto(proto_routed_request);
  if (!routed_request_or.ok()) {
    resp.set_status(batch_item_status_from_absl_status(routed_request_or.status()));
    resp.set_message(std::string(routed_request_or.status().message()));
    rctx.mark_success();
    return Status::OK;
  }
  const auto& routed_request = *routed_request_or;

  auto& span = rctx.span();
  span->SetAttribute("tc.route.authority_id", routed_request.authority_ref.authority_id);
  span->SetAttribute("tc.route.path_family", routed_request.path_family);
  span->SetAttribute("tc.route.stage_ref", routed_request.stage_ref);

  const auto transport_security_context =
      DistributedSecurityKernel::transport_security_context_from_server_context(rctx.server_context());
  const auto authenticated_peer_identity =
      DistributedSecurityKernel::derive_authenticated_peer_identity(transport_security_context);
  const std::optional<std::string_view> incoming_edge_ref = proto_routed_request.has_forwarded_claims_envelope() &&
          proto_routed_request.forwarded_claims_envelope().has_bound_edge()
      ? std::optional<std::string_view>(proto_routed_request.forwarded_claims_envelope().bound_edge())
      : std::nullopt;
  const auto disclosure_policy =
      DistributedSecurityKernel::declared_stage_disclosure_policy(proto_routed_request, incoming_edge_ref);
  span->SetAttribute("tc.route.peer_auth_class", static_cast<int>(authenticated_peer_identity.auth_class));
  span->SetAttribute("tc.route.continuity_class", static_cast<int>(disclosure_policy.continuity_class));

  const auto hop_projection_status =
      DistributedSecurityKernel::validate_sender_hop_auth_projection(proto_routed_request, authenticated_peer_identity);
  if (!hop_projection_status.ok()) {
    resp.set_status(batch_item_status_from_absl_status(hop_projection_status));
    resp.set_message(std::string(hop_projection_status.message()));
    rctx.mark_success();
    return Status::OK;
  }

  if (d_.payload_transport_broker == nullptr) {
    resp.set_status(v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION);
    resp.set_message("payload transport broker unavailable");
    rctx.mark_success();
    return Status::OK;
  }

  auto binding_or = DistributedSecurityKernel::verify_local_authority_binding(
      proto_routed_request.authority_ref(),
      d_.payload_transport_broker->daemon_id(),
      authenticated_peer_identity,
      disclosure_policy.continuity_class);
  if (!binding_or.ok()) {
    span->SetAttribute("tc.route.binding_status", std::string(binding_or.status().message()));
    resp.set_status(batch_item_status_from_absl_status(binding_or.status()));
    resp.set_message(std::string(binding_or.status().message()));
    rctx.mark_success();
    return Status::OK;
  }
  const auto disclosure_status = DistributedSecurityKernel::enforce_pre_disclosure_policy(
      proto_routed_request, disclosure_policy, std::optional<AuthorityBindingProof>(*binding_or));
  if (!disclosure_status.ok()) {
    resp.set_status(batch_item_status_from_absl_status(disclosure_status));
    resp.set_message(std::string(disclosure_status.message()));
    span->SetAttribute("tc.route.sensitive_pre_disclosure_blocked", true);
    rctx.mark_success();
    return Status::OK;
  }
  span->SetAttribute("tc.route.sensitive_pre_disclosure_blocked", false);

  std::optional<OwnerStageReply> owner_stage_reply;
  if (d_.materialization_controller != nullptr) {
    auto maybe_reply_or = d_.materialization_controller->maybe_route_authority_stage(routed_request, absl::Now());
    if (!maybe_reply_or.ok()) {
      resp.set_status(batch_item_status_from_absl_status(maybe_reply_or.status()));
      resp.set_message(std::string(maybe_reply_or.status().message()));
      rctx.mark_success();
      return Status::OK;
    }
    owner_stage_reply = std::move(*maybe_reply_or);
  }
  if (!owner_stage_reply.has_value()) {
    auto maybe_reply_or = d_.payload_transport_broker->maybe_route_authority_stage(routed_request, absl::Now());
    if (!maybe_reply_or.ok()) {
      resp.set_status(batch_item_status_from_absl_status(maybe_reply_or.status()));
      resp.set_message(std::string(maybe_reply_or.status().message()));
      rctx.mark_success();
      return Status::OK;
    }
    owner_stage_reply = std::move(*maybe_reply_or);
  }
  if (!owner_stage_reply.has_value()) {
    const auto undeclared_status = absl::FailedPreconditionError(
        std::string("undeclared routed authority path/stage: ") + routed_request.path_family + "/" +
        routed_request.stage_ref);
    resp.set_status(batch_item_status_from_absl_status(undeclared_status));
    resp.set_message(std::string(undeclared_status.message()));
    rctx.mark_success();
    return Status::OK;
  }
  auto populate_status = populate_proto_owner_stage_reply(*owner_stage_reply, resp.mutable_owner_stage_reply());
  if (!populate_status.ok()) {
    resp.clear_owner_stage_reply();
    resp.set_status(batch_item_status_from_absl_status(populate_status));
    resp.set_message(std::string(populate_status.message()));
    rctx.mark_success();
    return Status::OK;
  }
  resp.set_status(v2::BATCH_ITEM_STATUS_OK);
  resp.clear_message();
  rctx.mark_success();
  return Status::OK;
}

} // namespace tensorcast::daemon
