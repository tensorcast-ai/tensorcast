// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/lease_controller.h"

#include <cstdint>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/store/device_registry.h"
#include "daemon/util/status_utils.h"
#include "tensorcast/common/v1/capability_token.pb.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

constexpr absl::Duration kPlacementLeaseDefaultTokenTtl = absl::Hours(24 * 30);

std::string placement_capability_id(uint64_t lease_id) {
  return absl::StrCat("placement:", lease_id);
}

std::string placement_subject_id(const store::loading::ReplicaKey& key) {
  return absl::StrCat(
      "placement:",
      key.artifact_id,
      "|",
      key.view_id.value_or(""),
      "|",
      static_cast<int>(key.device.type),
      "|",
      key.device.ordinal,
      "|",
      key.replica);
}

void fill_retention_handle(const RetentionRegistry::Handle& handle, v2::RetentionHandle* out) {
  out->set_handle_id(handle.handle_id);
  out->set_expires_at_ms(handle.expires_at_ms);
  out->set_capability_token(handle.capability_token);
  out->set_charged_bytes(handle.charged_bytes);
  if (!handle.diagnostics.empty()) {
    out->set_diagnostics(handle.diagnostics);
  }
}

void fill_timestamp(absl::Time when, google::protobuf::Timestamp* out) {
  out->set_seconds(absl::ToUnixSeconds(when));
  out->set_nanos(static_cast<int32_t>(absl::ToInt64Nanoseconds(when - absl::FromUnixSeconds(out->seconds()))));
}

absl::StatusOr<uint64_t> placement_lease_id_from_scope(
    const tensorcast::common::v1::CapabilityTokenEnvelope& envelope) {
  tensorcast::common::v1::PlacementLeaseScope scope;
  if (!scope.ParseFromString(envelope.scope())) {
    return absl::InvalidArgumentError("invalid placement lease scope");
  }
  if (scope.lease_id() == 0) {
    return absl::InvalidArgumentError("placement lease scope missing lease_id");
  }
  return scope.lease_id();
}

struct PlacementLeaseResolution {
  uint64_t lease_id{0};
  bool envelope_token{false};
};

absl::StatusOr<PlacementLeaseResolution> resolve_placement_lease_token(
    std::string_view lease_token,
    const LeaseController::Dep& d,
    bool require_not_expired) {
  if (d.capability_tokens != nullptr && d.capability_tokens->configured() &&
      tensorcast::common::CapabilityTokenManager::looks_like_envelope(lease_token)) {
    auto envelope_or = d.capability_tokens->verify(
        std::string(lease_token),
        tensorcast::common::v1::CAPABILITY_AUDIENCE_PLACEMENT_LEASE,
        d.daemon_id,
        absl::Now(),
        require_not_expired);
    if (!envelope_or.ok()) {
      return envelope_or.status();
    }
    auto lease_id_or = placement_lease_id_from_scope(*envelope_or);
    if (!lease_id_or.ok()) {
      return lease_id_or.status();
    }
    return PlacementLeaseResolution{
        .lease_id = *lease_id_or,
        .envelope_token = true,
    };
  }

  auto lease_or = d.placement_lease_tokens.resolve(std::string(lease_token));
  if (!lease_or.ok()) {
    return lease_or.status();
  }
  return PlacementLeaseResolution{
      .lease_id = *lease_or,
      .envelope_token = false,
  };
}

grpc::Status retention_acquire_status(const absl::Status& st) {
  if (absl::IsResourceExhausted(st)) {
    return {StatusCode::RESOURCE_EXHAUSTED, std::string(st.message())};
  }
  if (absl::IsUnavailable(st)) {
    return {StatusCode::UNAVAILABLE, std::string(st.message())};
  }
  if (absl::IsDeadlineExceeded(st)) {
    return {StatusCode::DEADLINE_EXCEEDED, std::string(st.message())};
  }
  if (absl::IsInvalidArgument(st) || absl::IsFailedPrecondition(st) || absl::IsNotFound(st) ||
      absl::IsPermissionDenied(st)) {
    return {StatusCode::FAILED_PRECONDITION, std::string(st.message())};
  }
  return to_grpc_status(st);
}

grpc::Status retention_renew_status(const absl::Status& st) {
  if (absl::IsUnavailable(st)) {
    return {StatusCode::UNAVAILABLE, std::string(st.message())};
  }
  if (absl::IsDeadlineExceeded(st)) {
    return {StatusCode::DEADLINE_EXCEEDED, std::string(st.message())};
  }
  if (absl::IsResourceExhausted(st)) {
    return {StatusCode::RESOURCE_EXHAUSTED, std::string(st.message())};
  }
  if (absl::IsInvalidArgument(st) || absl::IsFailedPrecondition(st) || absl::IsNotFound(st) ||
      absl::IsPermissionDenied(st)) {
    return {StatusCode::FAILED_PRECONDITION, std::string(st.message())};
  }
  return to_grpc_status(st);
}

grpc::Status retention_release_status(const absl::Status& st) {
  if (absl::IsUnavailable(st)) {
    return {StatusCode::UNAVAILABLE, std::string(st.message())};
  }
  if (absl::IsDeadlineExceeded(st)) {
    return {StatusCode::DEADLINE_EXCEEDED, std::string(st.message())};
  }
  if (absl::IsInvalidArgument(st) || absl::IsFailedPrecondition(st) || absl::IsNotFound(st) ||
      absl::IsPermissionDenied(st)) {
    return {StatusCode::FAILED_PRECONDITION, std::string(st.message())};
  }
  return to_grpc_status(st);
}

} // namespace

grpc::Status LeaseController::create_placement_lease(
    RpcContext& rctx,
    const v2::CreatePlacementLeaseRequest& req,
    v2::CreatePlacementLeaseResponse& resp) {
  if (req.artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  const int device_id = req.device_id();
  if (device_id < 0 || device_id >= d_.engine.get_num_gpus()) {
    return {StatusCode::INVALID_ARGUMENT, "invalid device_id"};
  }

  store::loading::ReplicaKey key;
  key.artifact_id = req.artifact_id();
  if (!req.view_id().empty()) {
    key.view_id = req.view_id();
  }
  key.device = store::DeviceRegistry::instance().gpu_key(device_id);
  key.replica = 0;

  const absl::Duration ttl =
      req.has_ttl_ms() ? absl::Milliseconds(static_cast<int64_t>(req.ttl_ms())) : absl::ZeroDuration();
  auto lease_or = d_.lifecycle.create_placement_lease(key, ttl);
  if (!lease_or.ok()) {
    return to_grpc_status(lease_or.status());
  }
  const absl::Duration token_ttl = ttl > absl::ZeroDuration() ? ttl : kPlacementLeaseDefaultTokenTtl;
  const uint64_t expires_at_ms = static_cast<uint64_t>(absl::ToUnixMillis(absl::Now() + token_ttl));
  std::string lease_token;
  const std::string capability_id = placement_capability_id(*lease_or);
  if (d_.capability_tokens != nullptr && d_.capability_tokens->configured()) {
    tensorcast::common::v1::PlacementLeaseScope scope;
    scope.set_lease_id(*lease_or);
    auto scope_or = tensorcast::common::CapabilityTokenManager::serialize_scope_deterministic(scope);
    if (!scope_or.ok()) {
      d_.lifecycle.release_lease(*lease_or);
      return to_grpc_status(scope_or.status());
    }
    auto token_or = d_.capability_tokens->mint(
        d_.daemon_id, tensorcast::common::v1::CAPABILITY_AUDIENCE_PLACEMENT_LEASE, *scope_or, expires_at_ms);
    if (!token_or.ok()) {
      d_.lifecycle.release_lease(*lease_or);
      return to_grpc_status(token_or.status());
    }
    lease_token = *token_or;
  } else {
    auto token_or = d_.placement_lease_tokens.mint(*lease_or, ttl);
    if (!token_or.ok()) {
      d_.lifecycle.release_lease(*lease_or);
      return to_grpc_status(token_or.status());
    }
    lease_token = *token_or;
  }

  auto finalizer_status = d_.lifecycle.add_finalizer(*lease_or, [kernel = &d_.lifecycle_kernel, capability_id]() {
    return kernel->release_capability(capability_id);
  });
  if (!finalizer_status.ok()) {
    d_.lifecycle.release_lease(*lease_or);
    return to_grpc_status(finalizer_status);
  }

  const absl::Time issued_at = absl::Now();
  LifecycleSubjectRecord subject;
  subject.subject_id = placement_subject_id(key);
  subject.epochs.subject_generation = *lease_or;
  subject.subject_kind = LifecycleSubjectKind::kPlacementTarget;
  subject.created_at = issued_at;
  subject.last_observed_at = issued_at;
  subject.artifact_id = key.artifact_id;
  subject.semantic_ref_id = subject.subject_id;
  auto capability_or = d_.lifecycle_kernel.mint_capability(
      MintCapabilityRequest{
          .subject = subject,
          .address =
              CapabilityBindingAddress{
                  .route_principal = make_issuer_route_principal(d_.lifecycle_kernel.issuer_daemon_id()),
                  .family = LifecycleCapabilityFamily::kPlacement,
                  .binding_space = LifecycleBindingSpace::kPlacementLease,
                  .binding_key_kind = d_.capability_tokens != nullptr && d_.capability_tokens->configured()
                      ? BindingKeyKind::kLeaseId
                      : BindingKeyKind::kOpaqueLocalToken,
                  .binding_key = d_.capability_tokens != nullptr && d_.capability_tokens->configured()
                      ? std::to_string(*lease_or)
                      : lease_token,
                  .epochs = subject.epochs,
                  .binding_id = d_.capability_tokens != nullptr && d_.capability_tokens->configured()
                      ? std::nullopt
                      : std::optional<std::string>(lease_token),
              },
          .front_door_kind = d_.capability_tokens != nullptr && d_.capability_tokens->configured()
              ? LifecycleFrontDoorKind::kPlacementLeaseEnvelope
              : LifecycleFrontDoorKind::kPlacementLeaseLocalToken,
          .capability_id = capability_id,
          .lease_id = *lease_or,
          .capability_expires_at = issued_at + token_ttl,
          .carriage_kind = d_.capability_tokens != nullptr && d_.capability_tokens->configured()
              ? CredentialCarriageKind::kSelfDescribing
              : CredentialCarriageKind::kOpaqueLocalCompat,
          .binding_mode = d_.capability_tokens != nullptr && d_.capability_tokens->configured()
              ? LifecycleBindingMode::kAddressDerived
              : LifecycleBindingMode::kBindingRecord,
          .constraint_claims =
              ConstraintClaims{
                  .artifact_id = key.artifact_id,
                  .local_only = d_.capability_tokens == nullptr || !d_.capability_tokens->configured(),
              },
          .credential_expires_at = d_.capability_tokens != nullptr && d_.capability_tokens->configured()
              ? std::nullopt
              : std::optional<absl::Time>(issued_at + token_ttl),
          .binding_id = d_.capability_tokens != nullptr && d_.capability_tokens->configured()
              ? std::nullopt
              : std::optional<std::string>(lease_token),
          .local_only = d_.capability_tokens == nullptr || !d_.capability_tokens->configured(),
          .workflow_gate = WorkflowGateKind::kNone,
      });
  if (!capability_or.ok()) {
    d_.lifecycle.release_lease(*lease_or);
    return to_grpc_status(capability_or.status());
  }

  resp.set_lease_id(*lease_or);
  resp.set_lease_token(lease_token);
  if (ttl > absl::ZeroDuration()) {
    fill_timestamp(absl::Now() + ttl, resp.mutable_expires_at());
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status LeaseController::renew_placement_lease(
    RpcContext& rctx,
    const v2::RenewPlacementLeaseRequest& req,
    v2::RenewPlacementLeaseResponse& resp) {
  if (req.lease_token().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "lease_token is required"};
  }
  if (req.ttl_ms() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "ttl_ms is required"};
  }

  const absl::Duration ttl = absl::Milliseconds(static_cast<int64_t>(req.ttl_ms()));
  auto resolution_or = resolve_placement_lease_token(req.lease_token(), d_, /*require_not_expired=*/true);
  if (!resolution_or.ok()) {
    return to_grpc_status(resolution_or.status());
  }
  const uint64_t lease_id = resolution_or->lease_id;
  const bool envelope_token = resolution_or->envelope_token;

  auto st = d_.lifecycle.renew_placement(lease_id, ttl);
  if (!st.ok()) {
    return to_grpc_status(st);
  }

  if (envelope_token) {
    tensorcast::common::v1::PlacementLeaseScope scope;
    scope.set_lease_id(lease_id);
    auto scope_or = tensorcast::common::CapabilityTokenManager::serialize_scope_deterministic(scope);
    if (!scope_or.ok()) {
      return to_grpc_status(scope_or.status());
    }
    const uint64_t expires_at_ms = static_cast<uint64_t>(absl::ToUnixMillis(absl::Now() + ttl));
    auto token_or = d_.capability_tokens->mint(
        d_.daemon_id, tensorcast::common::v1::CAPABILITY_AUDIENCE_PLACEMENT_LEASE, *scope_or, expires_at_ms);
    if (!token_or.ok()) {
      return to_grpc_status(token_or.status());
    }
    resp.set_lease_token(*token_or);
  } else {
    st = d_.placement_lease_tokens.refresh(req.lease_token(), ttl);
    if (!st.ok()) {
      return to_grpc_status(st);
    }
    resp.set_lease_token(req.lease_token());
  }

  resp.set_lease_id(lease_id);
  fill_timestamp(absl::Now() + ttl, resp.mutable_expires_at());
  auto capability_or = d_.lifecycle_kernel.renew_capability(
      RenewCapabilityRequest{
          .capability_id = placement_capability_id(lease_id),
          .capability_expires_at = absl::Now() + ttl,
          .credential_expires_at = envelope_token ? std::nullopt : std::optional<absl::Time>(absl::Now() + ttl),
          .binding_id = envelope_token ? std::nullopt : std::optional<std::string>(req.lease_token()),
      });
  if (!capability_or.ok()) {
    return to_grpc_status(capability_or.status());
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status LeaseController::release_placement_lease(
    RpcContext& rctx,
    const v2::ReleasePlacementLeaseRequest& req,
    v2::ReleasePlacementLeaseResponse& resp) {
  if (req.lease_token().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "lease_token is required"};
  }
  auto resolution_or = resolve_placement_lease_token(req.lease_token(), d_, /*require_not_expired=*/false);
  if (!resolution_or.ok()) {
    return to_grpc_status(resolution_or.status());
  }
  const uint64_t lease_id = resolution_or->lease_id;
  const bool envelope_token = resolution_or->envelope_token;

  d_.lifecycle.release_lease(lease_id);
  auto capability_status = d_.lifecycle_kernel.release_capability(placement_capability_id(lease_id));
  if (!capability_status.ok()) {
    return to_grpc_status(capability_status);
  }
  const bool erased = envelope_token ? true : d_.placement_lease_tokens.erase(req.lease_token());
  resp.set_released(erased);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status LeaseController::acquire_retention_handle(
    RpcContext& rctx,
    const v2::AcquireRetentionHandleRequest& req,
    v2::AcquireRetentionHandleResponse& resp) {
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (d_.retention_registry == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "retention handles are disabled"};
  }
  auto handle_or =
      d_.retention_registry->acquire(req.selection(), req.has_policy() ? &req.policy() : nullptr, req.ttl_ms());
  if (!handle_or.ok()) {
    return retention_acquire_status(handle_or.status());
  }
  fill_retention_handle(*handle_or, resp.mutable_handle());
  rctx.mark_success();
  return Status::OK;
}

grpc::Status LeaseController::renew_retention_handle(
    RpcContext& rctx,
    const v2::RenewRetentionHandleRequest& req,
    v2::RenewRetentionHandleResponse& resp) {
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (d_.retention_registry == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "retention handles are disabled"};
  }
  auto handle_or = d_.retention_registry->renew(req.handle_token(), req.extend_ttl_ms());
  if (!handle_or.ok()) {
    return retention_renew_status(handle_or.status());
  }
  fill_retention_handle(*handle_or, resp.mutable_handle());
  rctx.mark_success();
  return Status::OK;
}

grpc::Status LeaseController::release_retention_handle(
    RpcContext& rctx,
    const v2::ReleaseRetentionHandleRequest& req,
    v2::ReleaseRetentionHandleResponse& resp) {
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (d_.retention_registry == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "retention handles are disabled"};
  }
  auto released_or = d_.retention_registry->release(req.handle_token());
  if (!released_or.ok()) {
    return retention_release_status(released_or.status());
  }
  resp.set_released(*released_or);
  rctx.mark_success();
  return Status::OK;
}

} // namespace tensorcast::daemon
