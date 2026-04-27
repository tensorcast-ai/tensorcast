// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/payload_transport_broker.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/components/endpoint_id.h"
#include "core/store/materialization/dataplane/contracts/inline_buffer_loader.h"
#include "core/store/materialization/dataplane/sources/remote_key_source.h"
#include "daemon/service/serving_lifecycle.h"
#include "daemon/state/distributed_security_kernel.h"
#include "daemon/state/routed_authority_protocol.h"
#include "daemon/state/routed_authority_wire.h"
#include "daemon/state/worker_directory_cache.h"
#include "daemon/util/grpc_daemon_transport.h"
#include "daemon/util/grpc_peer_utils.h"
#include "grpcpp/grpcpp.h"
#include "nlohmann/json.hpp"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"
#include "tensorcast/common/v1/capability_token.pb.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

namespace {

constexpr std::string_view kImmediateLoweringPathFamily = "immediate_lowering";
constexpr std::string_view kIssuerValidateStageRef = "issuer_validate";

std::string to_lower_copy(std::string_view value) {
  std::string out(value);
  absl::AsciiStrToLower(&out);
  return out;
}

std::string compute_sha256_hex(std::string_view payload) {
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  std::string hex =
      absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));
  absl::AsciiStrToLower(&hex);
  return hex;
}

std::string compute_batch_manifest_digest_hex(const v2::BatchPayloadManifest& manifest) {
  auto manifest_bytes_or = common::CapabilityTokenManager::serialize_scope_deterministic(manifest);
  if (!manifest_bytes_or.ok()) {
    return "";
  }
  return compute_sha256_hex(*manifest_bytes_or);
}

std::string payload_ref_capability_id(std::string_view payload_id) {
  return absl::StrCat("payload-ref:", payload_id);
}

std::string payload_ref_subject_id_for_backing(const store::runtime::ingestion::BackingIdentity& backing_identity) {
  return absl::StrCat(
      "backing:",
      backing_identity.physical_artifact_id,
      "|",
      backing_identity.replica_key.artifact_id,
      "|",
      backing_identity.replica_key.view_id.value_or(""),
      "|",
      static_cast<int>(backing_identity.replica_key.device.type),
      "|",
      backing_identity.replica_key.device.ordinal,
      "|",
      backing_identity.replica_key.device.uuid,
      "|",
      backing_identity.replica_key.replica);
}

std::string payload_ref_subject_id_for_inline(std::string_view payload_id) {
  return absl::StrCat("inline-snapshot:", payload_id);
}

std::string payload_direction_label(tensorcast::common::v1::PayloadRefDirection direction) {
  switch (direction) {
    case tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET:
      return "get";
    case tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT:
      return "put";
    case tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED:
    default:
      return "unspecified";
  }
}

absl::Status validate_batch_payload_manifest(const v2::BatchPayloadManifest& manifest) {
  if (manifest.total_size() == 0) {
    return absl::InvalidArgumentError("batch payload manifest total_size must be > 0");
  }
  if (manifest.entries_size() == 0) {
    return absl::InvalidArgumentError("batch payload manifest requires at least one entry");
  }
  absl::flat_hash_set<std::string> seen_artifact_ids;
  std::uint64_t cursor = 0;
  for (const auto& entry : manifest.entries()) {
    if (entry.artifact_id().empty()) {
      return absl::InvalidArgumentError("batch payload manifest entry artifact_id is required");
    }
    if (!seen_artifact_ids.insert(entry.artifact_id()).second) {
      return absl::InvalidArgumentError("batch payload manifest contains duplicate artifact_id");
    }
    if (entry.length() == 0) {
      return absl::InvalidArgumentError("batch payload manifest entry length must be > 0");
    }
    if (entry.digest_alg().empty() != entry.digest_hex().empty()) {
      return absl::InvalidArgumentError("batch payload manifest entry digest_alg and digest_hex must both be set");
    }
    if (entry.offset() != cursor) {
      return absl::InvalidArgumentError("batch payload manifest entries must be densely packed and offset-ordered");
    }
    if (entry.offset() > manifest.total_size() || entry.length() > manifest.total_size() - entry.offset()) {
      return absl::InvalidArgumentError("batch payload manifest entry exceeds total_size");
    }
    cursor = entry.offset() + entry.length();
  }
  if (cursor != manifest.total_size()) {
    return absl::InvalidArgumentError("batch payload manifest total_size does not match packed entry lengths");
  }
  return absl::OkStatus();
}

absl::Status validate_batch_communicator_source(
    const v2::BatchPayloadTransport& transport,
    const PayloadTransportBroker::BatchRefMetadata& metadata) {
  if (!transport.has_communicator_source()) {
    return absl::InvalidArgumentError("batch transport communicator_source is required");
  }
  const auto& source = transport.communicator_source();
  if (source.batch_payload_ref().empty()) {
    return absl::InvalidArgumentError("communicator_source batch_payload_ref is required");
  }
  if (source.producer_daemon_id().empty()) {
    return absl::InvalidArgumentError("communicator_source producer_daemon_id is required");
  }
  if (source.producer_port() == 0) {
    return absl::InvalidArgumentError("communicator_source producer_port is required");
  }
  if (source.remote_memory_keys_size() == 0 || source.remote_memory_keys_size() != source.buffer_sizes_size()) {
    return absl::InvalidArgumentError("communicator_source remote_memory_keys and buffer_sizes must match");
  }
  if (source.total_payload_bytes() == 0) {
    return absl::InvalidArgumentError("communicator_source total_payload_bytes must be > 0");
  }
  if (metadata.payload_size != 0 && metadata.payload_size != source.total_payload_bytes()) {
    return absl::FailedPreconditionError("communicator_source total_payload_bytes mismatch");
  }
  if (!metadata.consumer_daemon_id.empty() && !source.consumer_daemon_id().empty() &&
      metadata.consumer_daemon_id != source.consumer_daemon_id()) {
    return absl::FailedPreconditionError("communicator_source consumer_daemon_id mismatch");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::uint64_t> export_registration_total_bytes(const store::ExportRegistration& registration) {
  if (registration.remote_memory_keys.empty() ||
      registration.remote_memory_keys.size() != registration.buffer_sizes.size()) {
    return absl::InvalidArgumentError("communicator export registration is incomplete");
  }
  std::uint64_t total_bytes = 0;
  for (const auto buffer_size : registration.buffer_sizes) {
    if (buffer_size > static_cast<size_t>(std::numeric_limits<std::uint64_t>::max() - total_bytes)) {
      return absl::OutOfRangeError("communicator export registration exceeds uint64 range");
    }
    total_bytes += static_cast<std::uint64_t>(buffer_size);
  }
  if (total_bytes == 0) {
    return absl::InvalidArgumentError("communicator export registration is empty");
  }
  return total_bytes;
}

absl::StatusOr<absl::Time> resolve_payload_ref_expiry(
    absl::Time now,
    absl::Duration default_ttl,
    absl::Time capability_expires_at) {
  absl::Time expires_at = now + default_ttl;
  if (capability_expires_at != absl::InfiniteFuture()) {
    expires_at = std::min(expires_at, capability_expires_at);
  }
  if (expires_at <= now) {
    return absl::FailedPreconditionError("payload_ref capability is already expired");
  }
  return expires_at;
}

const char* capability_mode_label(BodyCapabilityResolutionMode mode) {
  switch (mode) {
    case BodyCapabilityResolutionMode::kLocalBodyHandle:
      return "local_body_handle";
    case BodyCapabilityResolutionMode::kChunkRpcFallback:
      return "chunk_rpc_fallback";
    case BodyCapabilityResolutionMode::kLoader:
    default:
      return "loader";
  }
}

void record_payload_ref_resolution_metrics(BodyCapabilityResolutionMode mode, bool local) {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_body_capability_resolution_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("mode", opentelemetry::common::AttributeValue(std::string(capability_mode_label(mode))));
    attrs.emplace("local", opentelemetry::common::AttributeValue(local ? "true" : "false"));
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void record_routed_authority_path_metrics(
    std::string_view family,
    std::string_view route_path_kind,
    std::string_view authority_kind,
    std::string_view route_source,
    std::string_view reply_admission_result,
    std::string_view issuer_loss_outcome,
    std::string_view reply_kind) {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_routed_authority_path_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("family", opentelemetry::common::AttributeValue(std::string(family)));
    attrs.emplace("route_path_kind", opentelemetry::common::AttributeValue(std::string(route_path_kind)));
    attrs.emplace("authority_kind", opentelemetry::common::AttributeValue(std::string(authority_kind)));
    attrs.emplace("route_source", opentelemetry::common::AttributeValue(std::string(route_source)));
    attrs.emplace("reply_admission_result", opentelemetry::common::AttributeValue(std::string(reply_admission_result)));
    attrs.emplace("issuer_loss_outcome", opentelemetry::common::AttributeValue(std::string(issuer_loss_outcome)));
    attrs.emplace("reply_kind", opentelemetry::common::AttributeValue(std::string(reply_kind)));
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void record_payload_ref_issuer_route_metrics(
    std::string_view route_source,
    std::string_view reply_admission_result,
    std::string_view issuer_loss_outcome,
    std::string_view reply_kind = "none") {
  record_routed_authority_path_metrics(
      "payload_ref",
      "issuer_route",
      "issuer_daemon",
      route_source,
      reply_admission_result,
      issuer_loss_outcome,
      reply_kind);
}

tensorcast::common::SelectionIdentity make_byte_artifact_selection_identity(std::string_view artifact_id) {
  return tensorcast::common::SelectionIdentity{
      .artifact_id = std::string(artifact_id),
      .logical_layout_hash = tensorcast::common::compute_byte_artifact_logical_layout_hash_bytes(),
      .selection_hash = tensorcast::common::compute_byte_artifact_selection_hash_bytes(),
  };
}

store::runtime::ingestion::VerifiedContentDescriptor payload_metadata_to_verified_content_descriptor(
    const PayloadTransportBroker::RefMetadata& metadata,
    std::string_view layout_id) {
  return body_descriptor_to_verified_content_descriptor_with_layout(
      layout_id, metadata.payload_size, metadata.digest_alg, metadata.digest_hex);
}

absl::Status absl_status_from_batch_item_status(v2::BatchItemStatus status, std::string_view message) {
  switch (status) {
    case v2::BATCH_ITEM_STATUS_OK:
      return absl::OkStatus();
    case v2::BATCH_ITEM_STATUS_MISS:
      return absl::NotFoundError(std::string(message));
    case v2::BATCH_ITEM_STATUS_UNAVAILABLE:
      return absl::UnavailableError(std::string(message));
    case v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION:
      return absl::FailedPreconditionError(std::string(message));
    case v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT:
      return absl::InvalidArgumentError(std::string(message));
    case v2::BATCH_ITEM_STATUS_INTERNAL_ERROR:
    case v2::BATCH_ITEM_STATUS_UNSPECIFIED:
    default:
      return absl::InternalError(std::string(message));
  }
}

tensorcast::common::v1::PayloadRefDirection payload_ref_direction_from_label(std::string_view direction) {
  if (direction == "get") {
    return tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET;
  }
  if (direction == "put") {
    return tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT;
  }
  return tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED;
}

absl::StatusOr<ResolvedSourceCapability> deserialize_payload_ref_routed_resolution(std::string_view payload) {
  auto json = nlohmann::json::parse(payload, nullptr, false);
  if (json.is_discarded()) {
    return absl::InvalidArgumentError("owner-stage resolved_source_capability payload is not valid json");
  }
  ResolvedSourceCapability resolution;
  resolution.selection_identity.artifact_id = json.value("artifact_id", "");
  if (!absl::HexStringToBytes(
          json.value("logical_layout_hash_hex", ""), &resolution.selection_identity.logical_layout_hash)) {
    return absl::InvalidArgumentError("owner-stage resolved_source_capability logical_layout_hash is not valid hex");
  }
  if (!absl::HexStringToBytes(json.value("selection_hash_hex", ""), &resolution.selection_identity.selection_hash)) {
    return absl::InvalidArgumentError("owner-stage resolved_source_capability selection_hash is not valid hex");
  }
  resolution.verified_content_descriptor.content_identity.semantic_layout_identity.kind =
      store::runtime::ingestion::SemanticLayoutKind::kFixedProfileLayout;
  resolution.verified_content_descriptor.content_identity.logical_size_bytes = json.value("size_bytes", 0ULL);
  resolution.verified_content_descriptor.content_identity.digest_alg = json.value("digest_alg", "");
  if (!absl::HexStringToBytes(
          json.value("digest_hex", ""), &resolution.verified_content_descriptor.content_identity.digest_bytes)) {
    return absl::InvalidArgumentError("owner-stage resolved_source_capability digest_hex is not valid hex");
  }
  if (!absl::HexStringToBytes(json.value("capability_id_hex", ""), &resolution.serving_capability.capability_id)) {
    return absl::InvalidArgumentError("owner-stage resolved_source_capability capability_id is not valid hex");
  }
  resolution.serving_capability.expires_at = absl::FromUnixMillis(json.value("expires_at_ms", 0LL));
  resolution.serving_capability.mode = static_cast<BodyCapabilityResolutionMode>(json.value("mode", 0));
  resolution.serving_capability.local = json.value("local", false);
  resolution.serving_capability.subject_kind = static_cast<ServingCapabilitySubjectKind>(json.value("subject_kind", 1));
  resolution.serving_capability.lifecycle_owner_ref.owner_kind =
      static_cast<LifecycleOwnerKind>(json.value("owner_kind", 1));
  resolution.serving_capability.lifecycle_owner_ref.owner_id = json.value("owner_id", "");
  resolution.source_kind = static_cast<store::loading::MaterializationSource>(json.value("source_kind", 0));
  if (!absl::HexStringToBytes(json.value("payload_ref_hex", ""), &resolution.payload_ref)) {
    return absl::InvalidArgumentError("owner-stage resolved_source_capability payload_ref is not valid hex");
  }
  auto validation_status = validate_resolved_source_capability(resolution);
  if (!validation_status.ok()) {
    return validation_status;
  }
  return resolution;
}

bool authority_ref_equals(const AuthorityRef& lhs, const AuthorityRef& rhs) {
  return lhs == rhs;
}

bool portable_credential_matches(const PortableParsedCredential& lhs, const PortableParsedCredential& rhs) {
  return lhs.address.route_principal == rhs.address.route_principal && lhs.address.family == rhs.address.family &&
      lhs.address.binding_space == rhs.address.binding_space &&
      lhs.address.binding_key_kind == rhs.address.binding_key_kind &&
      lhs.address.binding_key == rhs.address.binding_key && lhs.front_door_kind == rhs.front_door_kind &&
      lhs.binding_mode == rhs.binding_mode && lhs.portable_constraint_claims == rhs.portable_constraint_claims;
}

bool has_prefix(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

DaemonHopAuthContext hop_auth_context_for_route_address(
    std::string_view route_address,
    const DaemonOptions::InterDaemonGrpcSecurity& security) {
  return DaemonHopAuthContext{
      .auth_class = inter_daemon_hop_auth_class(route_address, security),
  };
}

AuthorityRef ingress_forwarding_authority_ref(std::string_view daemon_id) {
  return AuthorityRef{
      .authority_kind = AuthorityKind::kInternalAuthority,
      .authority_id = std::string(daemon_id),
  };
}

std::string payload_ref_root_request_id(const PayloadTransportBroker::RefMetadata& metadata) {
  return absl::StrCat("payload-ref:", metadata.payload_id);
}

bool is_payload_ref_issuer_stage(std::string_view path_family, std::string_view stage_ref) {
  return path_family == kImmediateLoweringPathFamily && stage_ref == kIssuerValidateStageRef;
}

absl::Status undeclared_route_authority_stage_status(std::string_view path_family, std::string_view stage_ref) {
  return absl::FailedPreconditionError(
      absl::StrCat("undeclared routed authority path/stage: ", path_family, "/", stage_ref));
}

absl::StatusOr<std::string_view> require_payload_ref_raw_issuer_evidence(
    const ForwardableCredentialEvidence& evidence) {
  if (evidence.evidence_kind != CredentialEvidenceKind::kRawCredential || !evidence.raw_credential_bytes.has_value()) {
    return absl::FailedPreconditionError(
        "payload_ref issuer route currently supports only raw_credential issuer evidence; canonical projection is not "
        "supported");
  }
  return *evidence.raw_credential_bytes;
}

DelegationEnvelope make_delegation_envelope(
    const AuthorityRef& audience_authority_ref,
    std::string_view root_request_id,
    std::string_view path_family,
    DelegationPayloadKind payload_kind,
    DelegationClass delegation_class,
    std::optional<absl::Time> expires_at,
    std::optional<std::string> bound_edge = std::nullopt) {
  return DelegationEnvelope{
      .audience_authority_ref = audience_authority_ref,
      .bound_root_request_id = std::string(root_request_id),
      .bound_path_family = std::string(path_family),
      .bound_edge = std::move(bound_edge),
      .payload_kind = payload_kind,
      .delegation_class = delegation_class,
      .expires_at = expires_at,
  };
}

std::string owner_stage_reply_kind_label(OwnerStageReplyKind reply_kind) {
  switch (reply_kind) {
    case OwnerStageReplyKind::kReadyForLowering:
      return "ready_for_lowering";
    case OwnerStageReplyKind::kContinueWithAuthority:
      return "continue_with_authority";
    case OwnerStageReplyKind::kRetryLater:
      return "retry_later";
    case OwnerStageReplyKind::kAttachExisting:
      return "attach_existing";
    case OwnerStageReplyKind::kTerminal:
      return "terminal";
  }
  return "unknown";
}

std::optional<LocalObservationRoutingRule> find_local_observation_rule(
    absl::Span<const LocalObservationRoutingRule> local_observation_rules,
    std::string_view observation_kind) {
  for (const auto& local_observation_rule : local_observation_rules) {
    if (local_observation_rule.observation_kind == observation_kind) {
      return local_observation_rule;
    }
  }
  return std::nullopt;
}

absl::StatusOr<std::vector<ForwardedClaim>> sanitize_local_observations_for_routing(
    const LocalObservationSet& local_observations,
    absl::Span<const LocalObservationRoutingRule> local_observation_rules,
    const AuthorityRef& minted_by_authority_ref,
    const AuthorityRef& audience_authority_ref,
    std::string_view root_request_id,
    std::string_view path_family,
    const std::optional<std::string>& credential_binding_digest) {
  std::vector<ForwardedClaim> forwarded_claims;
  forwarded_claims.reserve(local_observations.observations.size());
  for (const auto& observation : local_observations.observations) {
    const auto local_observation_rule =
        find_local_observation_rule(local_observation_rules, observation.observation_kind);
    if (!local_observation_rule.has_value() ||
        local_observation_rule->action == LocalObservationRoutingAction::kReject) {
      return absl::FailedPreconditionError(
          absl::StrCat("local observation is not routable: ", observation.observation_kind));
    }
    if (local_observation_rule->action == LocalObservationRoutingAction::kConsume) {
      continue;
    }
    const std::string claim_kind = local_observation_rule->forwarded_claim_kind.value_or(observation.observation_kind);
    const std::string claim_payload =
        local_observation_rule->forwarded_claim_payload.value_or(observation.observation_payload);
    forwarded_claims.push_back(
        ForwardedClaim{
            .claim_kind = claim_kind,
            .provenance = ForwardedClaimProvenance::kIngressLocal,
            .claim_payload = claim_payload,
            .minted_by_authority_ref = minted_by_authority_ref,
            .audience_authority_ref = audience_authority_ref,
            .bound_root_request_id = std::string(root_request_id),
            .bound_credential_binding_digest = credential_binding_digest,
            .bound_path_family = std::string(path_family),
        });
  }
  return forwarded_claims;
}

absl::StatusOr<OwnerStageReply> owner_stage_reply_from_proto(const v2::OwnerStageReply& proto_reply) {
  auto reply_shell_or = routed_authority_wire::owner_stage_reply_shell_from_proto(proto_reply);
  if (!reply_shell_or.ok()) {
    return reply_shell_or.status();
  }
  OwnerStageReply owner_stage_reply = std::move(reply_shell_or->reply);
  if (reply_shell_or->resolved_source_capability_payload.has_value()) {
    auto resolved_source_capability_or =
        deserialize_payload_ref_routed_resolution(*reply_shell_or->resolved_source_capability_payload);
    if (!resolved_source_capability_or.ok()) {
      return resolved_source_capability_or.status();
    }
    owner_stage_reply.resolved_source_capability =
        std::make_shared<ResolvedSourceCapability>(std::move(*resolved_source_capability_or));
  }
  auto validation_status = routed_authority_wire::validate_owner_stage_reply_shape(owner_stage_reply);
  if (!validation_status.ok()) {
    return validation_status;
  }
  return owner_stage_reply;
}

std::optional<absl::StatusCode> absl_status_code_from_string(std::string_view status_code) {
  if (status_code == "cancelled") {
    return absl::StatusCode::kCancelled;
  }
  if (status_code == "unknown") {
    return absl::StatusCode::kUnknown;
  }
  if (status_code == "invalid_argument") {
    return absl::StatusCode::kInvalidArgument;
  }
  if (status_code == "deadline_exceeded") {
    return absl::StatusCode::kDeadlineExceeded;
  }
  if (status_code == "not_found") {
    return absl::StatusCode::kNotFound;
  }
  if (status_code == "already_exists") {
    return absl::StatusCode::kAlreadyExists;
  }
  if (status_code == "permission_denied") {
    return absl::StatusCode::kPermissionDenied;
  }
  if (status_code == "resource_exhausted") {
    return absl::StatusCode::kResourceExhausted;
  }
  if (status_code == "failed_precondition") {
    return absl::StatusCode::kFailedPrecondition;
  }
  if (status_code == "aborted") {
    return absl::StatusCode::kAborted;
  }
  if (status_code == "out_of_range") {
    return absl::StatusCode::kOutOfRange;
  }
  if (status_code == "unimplemented") {
    return absl::StatusCode::kUnimplemented;
  }
  if (status_code == "internal") {
    return absl::StatusCode::kInternal;
  }
  if (status_code == "unavailable") {
    return absl::StatusCode::kUnavailable;
  }
  if (status_code == "data_loss") {
    return absl::StatusCode::kDataLoss;
  }
  if (status_code == "unauthenticated") {
    return absl::StatusCode::kUnauthenticated;
  }
  return std::nullopt;
}

absl::Status terminal_projection_to_status(const TerminalProjection& terminal_projection, std::string_view message) {
  if (terminal_projection.status_code.has_value()) {
    if (const auto status_code = absl_status_code_from_string(*terminal_projection.status_code);
        status_code.has_value()) {
      return absl::Status(*status_code, std::string(message));
    }
  }
  switch (terminal_projection.projection_kind) {
    case TerminalProjectionKind::kSemanticReject:
      return absl::PermissionDeniedError(std::string(message));
    case TerminalProjectionKind::kStatusSnapshot:
      return absl::UnavailableError(std::string(message));
    case TerminalProjectionKind::kSemanticSuccess:
    case TerminalProjectionKind::kFamilyDefined:
      return absl::FailedPreconditionError(std::string(message));
  }
  return absl::FailedPreconditionError(std::string(message));
}

class SharedStringSource final : public store::loader::SeekableSource {
 public:
  explicit SharedStringSource(std::shared_ptr<const std::string> payload) : payload_(std::move(payload)) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return payload_ ? payload_->size() : 0;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(cursor_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or.status();
    }
    cursor_ += *read_or;
    return *read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (!payload_ || offset >= payload_->size() || bytes == 0) {
      return static_cast<size_t>(0);
    }
    const size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, payload_->size() - offset));
    std::memcpy(dst, payload_->data() + offset, to_copy);
    return to_copy;
  }

  [[nodiscard]] bool supports_direct_write_at() const override {
    return true;
  }

  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const store::DirectWriteGrant& grant) override {
    if (!payload_ || bytes == 0 || src_offset >= payload_->size()) {
      return static_cast<size_t>(0);
    }
    const size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, payload_->size() - src_offset));
    const store::DirectWriteGrant::Window* target = nullptr;
    for (const auto& window : grant.windows) {
      if (dest_va_offset >= window.va_offset && dest_va_offset + to_copy <= window.va_offset + window.length) {
        target = &window;
        break;
      }
    }
    if (target == nullptr) {
      return absl::InvalidArgumentError("No direct-write window covers requested payload range");
    }
    const uint64_t window_offset = dest_va_offset - target->va_offset;
    auto* dst = reinterpret_cast<void*>(target->local_addr + window_offset);
    std::memcpy(dst, payload_->data() + src_offset, to_copy);
    return to_copy;
  }

 private:
  std::shared_ptr<const std::string> payload_;
  uint64_t cursor_{0};
};

class SharedSegmentedMemorySource final : public store::loader::SeekableSource {
 public:
  struct Segment {
    const std::uint8_t* data{nullptr};
    std::uint64_t size_bytes{0};
  };

  SharedSegmentedMemorySource(std::vector<Segment> segments, std::vector<std::shared_ptr<void>> keepalives)
      : segments_(std::move(segments)), keepalives_(std::move(keepalives)) {
    for (const auto& segment : segments_) {
      if (segment.size_bytes > std::numeric_limits<std::uint64_t>::max() - total_bytes_) {
        total_bytes_ = std::numeric_limits<std::uint64_t>::max();
        break;
      }
      total_bytes_ += segment.size_bytes;
    }
  }

  [[nodiscard]] uint64_t total_bytes() const override {
    return total_bytes_;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(cursor_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or.status();
    }
    cursor_ += *read_or;
    return *read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (bytes == 0 || offset >= total_bytes_) {
      return static_cast<size_t>(0);
    }
    const size_t to_copy = static_cast<size_t>(std::min<std::uint64_t>(bytes, total_bytes_ - offset));
    return copy_range_into(offset, static_cast<std::uint8_t*>(dst), to_copy);
  }

  [[nodiscard]] bool supports_direct_write_at() const override {
    return true;
  }

  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const store::DirectWriteGrant& grant) override {
    if (bytes == 0 || src_offset >= total_bytes_) {
      return static_cast<size_t>(0);
    }
    const size_t to_copy = static_cast<size_t>(std::min<std::uint64_t>(bytes, total_bytes_ - src_offset));
    const auto target_end_or = checked_add_u64(dest_va_offset, static_cast<uint64_t>(to_copy));
    if (!target_end_or.ok()) {
      return target_end_or.status();
    }
    const store::DirectWriteGrant::Window* target = nullptr;
    for (const auto& window : grant.windows) {
      const auto window_end_or = checked_add_u64(window.va_offset, window.length);
      if (!window_end_or.ok()) {
        return window_end_or.status();
      }
      if (dest_va_offset >= window.va_offset && *target_end_or <= *window_end_or) {
        target = &window;
        break;
      }
    }
    if (target == nullptr) {
      return absl::InvalidArgumentError("No direct-write window covers requested segmented payload range");
    }
    const uint64_t window_offset = dest_va_offset - target->va_offset;
    const auto target_addr_or = checked_add_u64(target->local_addr, window_offset);
    if (!target_addr_or.ok()) {
      return target_addr_or.status();
    }
    auto* dst = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(*target_addr_or));
    return copy_range_into(src_offset, dst, to_copy);
  }

 private:
  static absl::StatusOr<uint64_t> checked_add_u64(uint64_t lhs, uint64_t rhs) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
      return absl::OutOfRangeError("segmented payload offset overflow");
    }
    return lhs + rhs;
  }

  absl::StatusOr<size_t> copy_range_into(uint64_t offset, std::uint8_t* dst, size_t bytes) const {
    if (bytes == 0) {
      return static_cast<size_t>(0);
    }
    if (dst == nullptr) {
      return absl::InvalidArgumentError("segmented payload copy requires destination buffer");
    }
    std::uint64_t remaining = bytes;
    std::uint64_t segment_base = 0;
    size_t copied = 0;
    for (const auto& segment : segments_) {
      if (remaining == 0) {
        break;
      }
      const auto segment_end_or = checked_add_u64(segment_base, segment.size_bytes);
      if (!segment_end_or.ok()) {
        return segment_end_or.status();
      }
      if (offset >= *segment_end_or) {
        segment_base = *segment_end_or;
        continue;
      }
      const std::uint64_t segment_offset = offset > segment_base ? offset - segment_base : 0;
      const std::uint64_t available = segment.size_bytes - segment_offset;
      const size_t to_copy = static_cast<size_t>(std::min<std::uint64_t>(remaining, available));
      std::memcpy(dst + copied, segment.data + segment_offset, to_copy);
      copied += to_copy;
      remaining -= to_copy;
      offset += to_copy;
      segment_base = *segment_end_or;
    }
    return copied;
  }

  std::vector<Segment> segments_;
  std::vector<std::shared_ptr<void>> keepalives_;
  std::uint64_t total_bytes_{0};
  std::uint64_t cursor_{0};
};

class RemotePayloadRefSource final : public store::loader::SeekableSource {
 public:
  struct Options {
    PayloadTransportBroker::RefMetadata metadata;
    std::string payload_ref;
    std::string artifact_id;
    std::string operation_id;
    tensorcast::common::v1::PayloadRefDirection direction{tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED};
    std::string address;
    std::shared_ptr<grpc::ChannelCredentials> channel_credentials;
    std::uint64_t max_chunk_bytes{1ULL << 20};
    absl::Duration fetch_deadline{absl::Seconds(5)};
  };

  explicit RemotePayloadRefSource(Options options)
      : options_(std::move(options)),
        channel_(create_inter_daemon_channel(options_.address, options_.channel_credentials)),
        stub_(v2::StoreDaemonService::NewStub(channel_)) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return options_.metadata.payload_size;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(cursor_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or.status();
    }
    cursor_ += *read_or;
    return *read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= options_.metadata.payload_size || bytes == 0) {
      return static_cast<size_t>(0);
    }

    const absl::Time read_started_at = absl::Now();
    size_t copied = 0;
    std::size_t rpc_count = 0;
    absl::Duration rpc_elapsed = absl::ZeroDuration();
    const size_t target_bytes = static_cast<size_t>(std::min<uint64_t>(bytes, options_.metadata.payload_size - offset));
    auto* out = static_cast<char*>(dst);

    while (copied < target_bytes) {
      grpc::ClientContext client_ctx;
      client_ctx.set_deadline(
          std::chrono::system_clock::now() +
          std::chrono::milliseconds(absl::ToInt64Milliseconds(options_.fetch_deadline)));
      v2::FetchPayloadRefChunkRequest request;
      request.set_payload_ref(options_.payload_ref);
      request.set_artifact_id(options_.artifact_id);
      request.set_offset(offset + copied);
      request.set_max_bytes(std::min<std::uint64_t>(options_.max_chunk_bytes, target_bytes - copied));
      if (!options_.operation_id.empty()) {
        request.set_operation_id(options_.operation_id);
      }

      v2::FetchPayloadRefChunkResponse response;
      const absl::Time rpc_started_at = absl::Now();
      const auto rpc_status = stub_->FetchPayloadRefChunk(&client_ctx, request, &response);
      rpc_elapsed += absl::Now() - rpc_started_at;
      ++rpc_count;
      if (!rpc_status.ok()) {
        return absl::UnavailableError(rpc_status.error_message());
      }
      auto item_status = absl_status_from_batch_item_status(response.status(), response.message());
      if (!item_status.ok()) {
        return item_status;
      }
      if (response.total_size() != options_.metadata.payload_size) {
        return absl::DataLossError("payload_ref total_size mismatch");
      }
      if (response.chunk().empty()) {
        if (response.eof()) {
          break;
        }
        return absl::DataLossError("payload_ref fetch returned empty non-terminal chunk");
      }
      const size_t chunk_bytes = std::min(target_bytes - copied, response.chunk().size());
      std::memcpy(out + copied, response.chunk().data(), chunk_bytes);
      copied += chunk_bytes;
      if (response.eof()) {
        break;
      }
    }
    VLOG(2) << "payload_ref.remote_fetch_summary"
            << " direction=" << payload_direction_label(options_.direction) << " operation_id=" << options_.operation_id
            << " artifact_id=" << options_.artifact_id << " issuer_address=" << options_.address
            << " payload_bytes=" << options_.metadata.payload_size << " requested_bytes=" << target_bytes
            << " chunk_rpcs=" << rpc_count << " max_chunk_bytes=" << options_.max_chunk_bytes
            << " rpc_ms=" << absl::ToDoubleMilliseconds(rpc_elapsed)
            << " total_read_ms=" << absl::ToDoubleMilliseconds(absl::Now() - read_started_at);
    return copied;
  }

 private:
  Options options_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<v2::StoreDaemonService::Stub> stub_;
  uint64_t cursor_{0};
};

class PayloadRefLoader final : public store::IArtifactLoader {
 public:
  struct RemoteOptions {
    RemotePayloadRefSource::Options source;
  };

  explicit PayloadRefLoader(RemoteOptions options) : remote_(std::move(options)) {}

  absl::Status initialize() override {
    initialized_ = true;
    return absl::OkStatus();
  }

  absl::StatusOr<uint64_t> get_artifact_size() override {
    if (!initialized_) {
      return absl::FailedPreconditionError("PayloadRefLoader not initialized");
    }
    return remote_.source.metadata.payload_size;
  }

  absl::StatusOr<std::unique_ptr<store::loader::SeekableSource>> open_source() override {
    if (!initialized_) {
      return absl::FailedPreconditionError("PayloadRefLoader not initialized");
    }
    return std::unique_ptr<store::loader::SeekableSource>(std::make_unique<RemotePayloadRefSource>(remote_.source));
  }

 private:
  bool initialized_{false};
  RemoteOptions remote_;
};

class RemoteBatchPayloadRefSource final : public store::loader::SeekableSource {
 public:
  struct Options {
    PayloadTransportBroker::BatchRefMetadata metadata;
    std::string batch_payload_ref;
    std::string operation_id;
    tensorcast::common::v1::PayloadRefDirection direction{tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED};
    std::string address;
    std::shared_ptr<grpc::ChannelCredentials> channel_credentials;
    std::uint64_t max_chunk_bytes{1ULL << 20};
    absl::Duration fetch_deadline{absl::Seconds(5)};
  };

  explicit RemoteBatchPayloadRefSource(Options options)
      : options_(std::move(options)),
        channel_(create_inter_daemon_channel(options_.address, options_.channel_credentials)),
        stub_(v2::StoreDaemonService::NewStub(channel_)) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return options_.metadata.payload_size;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(cursor_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or.status();
    }
    cursor_ += *read_or;
    return *read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= options_.metadata.payload_size || bytes == 0) {
      return static_cast<size_t>(0);
    }

    const absl::Time read_started_at = absl::Now();
    size_t copied = 0;
    std::size_t rpc_count = 0;
    absl::Duration rpc_elapsed = absl::ZeroDuration();
    const size_t target_bytes = static_cast<size_t>(std::min<uint64_t>(bytes, options_.metadata.payload_size - offset));
    auto* out = static_cast<char*>(dst);

    while (copied < target_bytes) {
      grpc::ClientContext client_ctx;
      client_ctx.set_deadline(
          std::chrono::system_clock::now() +
          std::chrono::milliseconds(absl::ToInt64Milliseconds(options_.fetch_deadline)));
      v2::FetchBatchPayloadRefChunkRequest request;
      request.set_batch_payload_ref(options_.batch_payload_ref);
      request.set_offset(offset + copied);
      request.set_max_bytes(std::min<std::uint64_t>(options_.max_chunk_bytes, target_bytes - copied));
      if (!options_.operation_id.empty()) {
        request.set_operation_id(options_.operation_id);
      }

      v2::FetchBatchPayloadRefChunkResponse response;
      const absl::Time rpc_started_at = absl::Now();
      const auto rpc_status = stub_->FetchBatchPayloadRefChunk(&client_ctx, request, &response);
      rpc_elapsed += absl::Now() - rpc_started_at;
      ++rpc_count;
      if (!rpc_status.ok()) {
        return absl::UnavailableError(rpc_status.error_message());
      }
      auto item_status = absl_status_from_batch_item_status(response.status(), response.message());
      if (!item_status.ok()) {
        return item_status;
      }
      if (response.total_size() != options_.metadata.payload_size) {
        return absl::DataLossError("batch_payload_ref total_size mismatch");
      }
      if (response.chunk().empty()) {
        if (response.eof()) {
          break;
        }
        return absl::DataLossError("batch_payload_ref fetch returned empty non-terminal chunk");
      }
      const size_t chunk_bytes = std::min(target_bytes - copied, response.chunk().size());
      std::memcpy(out + copied, response.chunk().data(), chunk_bytes);
      copied += chunk_bytes;
      if (response.eof()) {
        break;
      }
    }
    VLOG(2) << "batch_payload_ref.remote_fetch_summary"
            << " direction=" << payload_direction_label(options_.direction) << " operation_id=" << options_.operation_id
            << " transport_id=" << options_.metadata.transport_id << " issuer_address=" << options_.address
            << " payload_bytes=" << options_.metadata.payload_size << " requested_bytes=" << target_bytes
            << " chunk_rpcs=" << rpc_count << " max_chunk_bytes=" << options_.max_chunk_bytes
            << " rpc_ms=" << absl::ToDoubleMilliseconds(rpc_elapsed)
            << " total_read_ms=" << absl::ToDoubleMilliseconds(absl::Now() - read_started_at);
    return copied;
  }

 private:
  Options options_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<v2::StoreDaemonService::Stub> stub_;
  uint64_t cursor_{0};
};

absl::StatusOr<std::string> read_source_fully(store::loader::SeekableSource& source) {
  const uint64_t total_bytes = source.total_bytes();
  if (total_bytes == 0) {
    return absl::DataLossError("payload_ref fetch returned empty payload");
  }
  if (total_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return absl::OutOfRangeError("payload_ref size exceeds host limits");
  }
  std::string payload;
  payload.resize(static_cast<size_t>(total_bytes));
  size_t copied = 0;
  while (copied < payload.size()) {
    auto read_or = source.read_at(copied, payload.data() + copied, payload.size() - copied);
    if (!read_or.ok()) {
      return read_or.status();
    }
    if (*read_or == 0) {
      return absl::DataLossError("payload_ref fetch terminated before expected size");
    }
    copied += *read_or;
  }
  return payload;
}

} // namespace

PayloadTransportBroker::PayloadTransportBroker(
    std::string daemon_id,
    common::CapabilityTokenManager* capability_tokens,
    SessionLifecycleManager* lifecycle_manager,
    LifecycleKernel* lifecycle_kernel,
    Options options)
    : daemon_id_(std::move(daemon_id)),
      capability_tokens_(capability_tokens),
      lifecycle_manager_(lifecycle_manager),
      lifecycle_kernel_(lifecycle_kernel),
      options_(std::move(options)) {}

absl::StatusOr<std::string> PayloadTransportBroker::issue_payload_ref(
    std::string_view artifact_id,
    std::string payload,
    tensorcast::common::v1::PayloadRefDirection direction,
    std::string_view operation_id,
    absl::Time capability_expires_at) {
  return issue_payload_ref(
      artifact_id,
      std::make_shared<const std::string>(std::move(payload)),
      direction,
      operation_id,
      capability_expires_at);
}

absl::StatusOr<std::string> PayloadTransportBroker::issue_payload_ref(
    std::string_view artifact_id,
    std::shared_ptr<const std::string> payload,
    const BodyDescriptor& descriptor,
    tensorcast::common::v1::PayloadRefDirection direction,
    std::string_view operation_id,
    absl::Time capability_expires_at) {
  if (!payload || payload->empty()) {
    return absl::InvalidArgumentError("payload is required for payload_ref issuance");
  }
  const BodyDescriptor normalized_descriptor = normalized_body_descriptor(descriptor);
  if (normalized_descriptor.size_bytes != payload->size()) {
    return absl::FailedPreconditionError("descriptor size does not match payload size");
  }
  if (normalized_descriptor.payload_digest_alg.empty() || normalized_descriptor.payload_digest_hex.empty()) {
    return absl::InvalidArgumentError("descriptor digest is required for payload_ref issuance");
  }
  if (daemon_id_.empty()) {
    return absl::FailedPreconditionError("daemon_id is required for payload_ref issuance");
  }
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required for payload_ref issuance");
  }
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return absl::FailedPreconditionError("capability tokens are required for payload_ref issuance");
  }
  if (direction == tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED) {
    return absl::InvalidArgumentError("payload_ref direction is required");
  }
  if (lifecycle_manager_ == nullptr || lifecycle_kernel_ == nullptr) {
    return absl::FailedPreconditionError("payload_ref lifecycle kernel is unavailable");
  }

  const absl::Time now = absl::Now();
  auto expires_at_or = resolve_payload_ref_expiry(now, options_.ttl, capability_expires_at);
  if (!expires_at_or.ok()) {
    return expires_at_or.status();
  }
  const absl::Time expires_at = *expires_at_or;
  const std::string payload_id = [&]() {
    absl::MutexLock lock(&mu_);
    prune_locked(now);
    return mint_payload_id();
  }();

  RefMetadata metadata;
  metadata.issuer_daemon_id = daemon_id_;
  metadata.payload_id = payload_id;
  metadata.artifact_id = std::string(artifact_id);
  metadata.payload_size = payload->size();
  metadata.digest_alg = normalized_descriptor.payload_digest_alg;
  metadata.digest_hex = normalized_descriptor.payload_digest_hex;
  metadata.direction = direction;
  metadata.operation_id = std::string(operation_id);
  metadata.expires_at = expires_at;

  const std::string capability_id = payload_ref_capability_id(payload_id);
  auto lease_or = lifecycle_manager_->create_retention_lease(
      expires_at - now,
      std::vector<std::function<absl::Status()>>{
          [this, payload_id, capability_id]() -> absl::Status {
            {
              absl::MutexLock lock(&mu_);
              auto it = records_.find(payload_id);
              if (it != records_.end()) {
                if (!it->second.body_handle.empty() && it->second.body_handle.unique_owner()) {
                  (void)it->second.body_handle.retire();
                }
                records_.erase(it);
              }
            }
            return lifecycle_kernel_ ? lifecycle_kernel_->release_capability(capability_id) : absl::OkStatus();
          },
      });
  if (!lease_or.ok()) {
    return lease_or.status();
  }

  LifecycleSubjectRecord subject;
  subject.subject_id = payload_ref_subject_id_for_inline(payload_id);
  subject.epochs.subject_generation = 1;
  subject.subject_kind = LifecycleSubjectKind::kInlineSnapshot;
  subject.created_at = now;
  subject.last_observed_at = now;
  subject.artifact_id = std::string(artifact_id);
  subject.semantic_ref_id = payload_id;
  auto capability_or = lifecycle_kernel_->mint_capability(
      MintCapabilityRequest{
          .subject = subject,
          .address =
              CapabilityBindingAddress{
                  .route_principal = make_issuer_route_principal(daemon_id_),
                  .family = LifecycleCapabilityFamily::kServe,
                  .binding_space = LifecycleBindingSpace::kPayload,
                  .binding_key_kind = BindingKeyKind::kPayloadId,
                  .binding_key = payload_id,
                  .epochs = subject.epochs,
              },
          .front_door_kind = LifecycleFrontDoorKind::kPayloadRef,
          .capability_id = capability_id,
          .lease_id = *lease_or,
          .capability_expires_at = expires_at,
          .carriage_kind = CredentialCarriageKind::kSelfDescribing,
          .binding_mode = LifecycleBindingMode::kAddressDerived,
          .constraint_claims =
              ConstraintClaims{
                  .artifact_id = std::string(artifact_id),
                  .digest_alg = metadata.digest_alg,
                  .digest_hex = metadata.digest_hex,
                  .direction = payload_direction_label(direction),
                  .operation_id = std::string(operation_id),
              },
          .direction = payload_direction_label(direction),
          .workflow_gate = WorkflowGateKind::kNone,
      });
  if (!capability_or.ok()) {
    lifecycle_manager_->release_lease(*lease_or);
    return capability_or.status();
  }

  {
    absl::MutexLock lock(&mu_);
    records_[payload_id] = Record{
        .metadata = metadata,
        .payload = std::move(payload),
        .body_handle = BodyHandle(),
        .descriptor = normalized_descriptor,
        .backing_identity = std::nullopt,
        .backing_instance_generation = 0,
        .lease_id = *lease_or,
    };
  }

  tensorcast::common::v1::PayloadRefScope scope;
  scope.set_payload_id(metadata.payload_id);
  scope.set_artifact_id(metadata.artifact_id);
  scope.set_payload_size(metadata.payload_size);
  scope.set_digest_alg(metadata.digest_alg);
  scope.set_digest_hex(metadata.digest_hex);
  scope.set_direction(direction);
  if (!metadata.operation_id.empty()) {
    scope.set_operation_id(metadata.operation_id);
  }
  auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  if (!scope_or.ok()) {
    lifecycle_manager_->release_lease(*lease_or);
    return scope_or.status();
  }
  auto token_or = capability_tokens_->mint(
      daemon_id_,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PAYLOAD_REF,
      *scope_or,
      static_cast<std::uint64_t>(absl::ToUnixMillis(expires_at)));
  if (!token_or.ok()) {
    lifecycle_manager_->release_lease(*lease_or);
    return token_or.status();
  }
  return *token_or;
}

absl::StatusOr<std::string> PayloadTransportBroker::issue_payload_ref(
    std::string_view artifact_id,
    std::shared_ptr<const std::string> payload,
    tensorcast::common::v1::PayloadRefDirection direction,
    std::string_view operation_id,
    absl::Time capability_expires_at) {
  if (daemon_id_.empty()) {
    return absl::FailedPreconditionError("daemon_id is required for payload_ref issuance");
  }
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required for payload_ref issuance");
  }
  if (!payload || payload->empty()) {
    return absl::InvalidArgumentError("payload is required for payload_ref issuance");
  }
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return absl::FailedPreconditionError("capability tokens are required for payload_ref issuance");
  }
  if (direction == tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED) {
    return absl::InvalidArgumentError("payload_ref direction is required");
  }
  if (lifecycle_manager_ == nullptr || lifecycle_kernel_ == nullptr) {
    return absl::FailedPreconditionError("payload_ref lifecycle kernel is unavailable");
  }

  const absl::Time now = absl::Now();
  auto expires_at_or = resolve_payload_ref_expiry(now, options_.ttl, capability_expires_at);
  if (!expires_at_or.ok()) {
    return expires_at_or.status();
  }
  const absl::Time expires_at = *expires_at_or;
  const std::string payload_id = [&]() {
    absl::MutexLock lock(&mu_);
    prune_locked(now);
    return mint_payload_id();
  }();

  RefMetadata metadata;
  metadata.issuer_daemon_id = daemon_id_;
  metadata.payload_id = payload_id;
  metadata.artifact_id = std::string(artifact_id);
  metadata.payload_size = payload->size();
  metadata.digest_alg = "sha256";
  metadata.digest_hex = compute_sha256_hex(*payload);
  metadata.direction = direction;
  metadata.operation_id = std::string(operation_id);
  metadata.expires_at = expires_at;

  const std::string capability_id = payload_ref_capability_id(payload_id);
  auto lease_or = lifecycle_manager_->create_retention_lease(
      expires_at - now,
      std::vector<std::function<absl::Status()>>{
          [this, payload_id, capability_id]() -> absl::Status {
            {
              absl::MutexLock lock(&mu_);
              auto it = records_.find(payload_id);
              if (it != records_.end()) {
                if (!it->second.body_handle.empty() && it->second.body_handle.unique_owner()) {
                  (void)it->second.body_handle.retire();
                }
                records_.erase(it);
              }
            }
            return lifecycle_kernel_ ? lifecycle_kernel_->release_capability(capability_id) : absl::OkStatus();
          },
      });
  if (!lease_or.ok()) {
    return lease_or.status();
  }

  LifecycleSubjectRecord subject;
  subject.subject_id = payload_ref_subject_id_for_inline(payload_id);
  subject.epochs.subject_generation = 1;
  subject.subject_kind = LifecycleSubjectKind::kInlineSnapshot;
  subject.created_at = now;
  subject.last_observed_at = now;
  subject.artifact_id = std::string(artifact_id);
  subject.semantic_ref_id = payload_id;
  auto capability_or = lifecycle_kernel_->mint_capability(
      MintCapabilityRequest{
          .subject = subject,
          .address =
              CapabilityBindingAddress{
                  .route_principal = make_issuer_route_principal(daemon_id_),
                  .family = LifecycleCapabilityFamily::kServe,
                  .binding_space = LifecycleBindingSpace::kPayload,
                  .binding_key_kind = BindingKeyKind::kPayloadId,
                  .binding_key = payload_id,
                  .epochs = subject.epochs,
              },
          .front_door_kind = LifecycleFrontDoorKind::kPayloadRef,
          .capability_id = capability_id,
          .lease_id = *lease_or,
          .capability_expires_at = expires_at,
          .carriage_kind = CredentialCarriageKind::kSelfDescribing,
          .binding_mode = LifecycleBindingMode::kAddressDerived,
          .constraint_claims =
              ConstraintClaims{
                  .artifact_id = std::string(artifact_id),
                  .digest_alg = metadata.digest_alg,
                  .digest_hex = metadata.digest_hex,
                  .direction = payload_direction_label(direction),
                  .operation_id = std::string(operation_id),
              },
          .direction = payload_direction_label(direction),
          .workflow_gate = WorkflowGateKind::kNone,
      });
  if (!capability_or.ok()) {
    lifecycle_manager_->release_lease(*lease_or);
    return capability_or.status();
  }

  {
    absl::MutexLock lock(&mu_);
    records_[payload_id] = Record{
        .metadata = metadata,
        .payload = std::move(payload),
        .body_handle = BodyHandle(),
        .descriptor = BodyDescriptor(),
        .backing_identity = std::nullopt,
        .backing_instance_generation = 0,
        .lease_id = *lease_or,
    };
  }

  tensorcast::common::v1::PayloadRefScope scope;
  scope.set_payload_id(metadata.payload_id);
  scope.set_artifact_id(metadata.artifact_id);
  scope.set_payload_size(metadata.payload_size);
  scope.set_digest_alg(metadata.digest_alg);
  scope.set_digest_hex(metadata.digest_hex);
  scope.set_direction(direction);
  if (!metadata.operation_id.empty()) {
    scope.set_operation_id(metadata.operation_id);
  }
  auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  if (!scope_or.ok()) {
    lifecycle_manager_->release_lease(*lease_or);
    return scope_or.status();
  }
  auto token_or = capability_tokens_->mint(
      daemon_id_,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PAYLOAD_REF,
      *scope_or,
      static_cast<std::uint64_t>(absl::ToUnixMillis(expires_at)));
  if (!token_or.ok()) {
    lifecycle_manager_->release_lease(*lease_or);
    return token_or.status();
  }
  return *token_or;
}

absl::StatusOr<std::string> PayloadTransportBroker::issue_payload_ref(
    std::string_view artifact_id,
    const BodyHandle& body_handle,
    const BodyDescriptor& descriptor,
    std::optional<store::runtime::ingestion::BackingIdentity> backing_identity,
    std::uint64_t backing_instance_generation,
    tensorcast::common::v1::PayloadRefDirection direction,
    std::string_view operation_id,
    absl::Time capability_expires_at) {
  if (daemon_id_.empty()) {
    return absl::FailedPreconditionError("daemon_id is required for payload_ref issuance");
  }
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required for payload_ref issuance");
  }
  if (body_handle.empty()) {
    return absl::InvalidArgumentError("body_handle is required for payload_ref issuance");
  }
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return absl::FailedPreconditionError("capability tokens are required for payload_ref issuance");
  }
  if (direction == tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED) {
    return absl::InvalidArgumentError("payload_ref direction is required");
  }
  if (descriptor.payload_digest_alg.empty() || descriptor.payload_digest_hex.empty()) {
    return absl::InvalidArgumentError("descriptor digest is required for payload_ref issuance");
  }
  if (lifecycle_manager_ == nullptr || lifecycle_kernel_ == nullptr) {
    return absl::FailedPreconditionError("payload_ref lifecycle kernel is unavailable");
  }
  if (backing_identity.has_value() &&
      !store::runtime::ingestion::backing_identity_matches_replica_key(*backing_identity)) {
    return absl::InvalidArgumentError("backing_identity must match replica_key.artifact_id");
  }
  const BodyDescriptor normalized_descriptor = normalized_body_descriptor(descriptor);
  if (!backing_identity.has_value()) {
    backing_identity = body_descriptor_to_backing_identity(normalized_descriptor, body_handle);
  }
  if (!backing_identity.has_value() ||
      !store::runtime::ingestion::backing_identity_matches_replica_key(*backing_identity)) {
    return absl::InvalidArgumentError("descriptor and body_handle do not identify a valid backing");
  }
  if (backing_instance_generation == 0) {
    backing_instance_generation = body_handle.binding_generation();
  }
  if (backing_instance_generation == 0) {
    return absl::InvalidArgumentError("backing_instance_generation is required for live-backing payload_ref issuance");
  }
  std::shared_ptr<const std::string> payload_snapshot;
  if (direction == tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET) {
    auto payload_or = body_handle.read_all_bytes();
    if (!payload_or.ok()) {
      return payload_or.status();
    }
    payload_snapshot = std::make_shared<const std::string>(std::move(*payload_or));
  }

  const absl::Time now = absl::Now();
  auto expires_at_or = resolve_payload_ref_expiry(now, options_.ttl, capability_expires_at);
  if (!expires_at_or.ok()) {
    return expires_at_or.status();
  }
  const absl::Time expires_at = *expires_at_or;
  const std::string payload_id = [&]() {
    absl::MutexLock lock(&mu_);
    prune_locked(now);
    return mint_payload_id();
  }();

  RefMetadata metadata;
  metadata.issuer_daemon_id = daemon_id_;
  metadata.payload_id = payload_id;
  metadata.artifact_id = std::string(artifact_id);
  metadata.payload_size = body_handle.size_bytes();
  metadata.digest_alg = descriptor.payload_digest_alg;
  metadata.digest_hex = descriptor.payload_digest_hex;
  metadata.direction = direction;
  metadata.operation_id = std::string(operation_id);
  metadata.expires_at = expires_at;
  if (metadata.payload_size == 0 || metadata.digest_alg.empty() || metadata.digest_hex.empty()) {
    return absl::InvalidArgumentError("body_handle metadata is incomplete for payload_ref issuance");
  }

  const std::string capability_id = payload_ref_capability_id(payload_id);
  auto lease_or = lifecycle_manager_->create_retention_lease(
      expires_at - now,
      std::vector<std::function<absl::Status()>>{
          [this, payload_id, capability_id]() -> absl::Status {
            {
              absl::MutexLock lock(&mu_);
              auto it = records_.find(payload_id);
              if (it != records_.end()) {
                if (!it->second.body_handle.empty() && it->second.body_handle.unique_owner()) {
                  (void)it->second.body_handle.retire();
                }
                records_.erase(it);
              }
            }
            return lifecycle_kernel_ ? lifecycle_kernel_->release_capability(capability_id) : absl::OkStatus();
          },
      });
  if (!lease_or.ok()) {
    return lease_or.status();
  }

  LifecycleSubjectRecord subject;
  subject.subject_id = payload_ref_subject_id_for_backing(*backing_identity);
  subject.epochs.subject_generation = backing_instance_generation;
  subject.subject_kind = LifecycleSubjectKind::kBacking;
  subject.created_at = now;
  subject.last_observed_at = now;
  subject.artifact_id = std::string(artifact_id);
  subject.semantic_ref_id = subject.subject_id;
  subject.verified_content_descriptor = body_descriptor_to_verified_content_descriptor(normalized_descriptor);
  auto capability_or = lifecycle_kernel_->mint_capability(
      MintCapabilityRequest{
          .subject = subject,
          .address =
              CapabilityBindingAddress{
                  .route_principal = make_issuer_route_principal(daemon_id_),
                  .family = LifecycleCapabilityFamily::kServe,
                  .binding_space = LifecycleBindingSpace::kPayload,
                  .binding_key_kind = BindingKeyKind::kPayloadId,
                  .binding_key = payload_id,
                  .epochs = subject.epochs,
              },
          .front_door_kind = LifecycleFrontDoorKind::kPayloadRef,
          .capability_id = capability_id,
          .lease_id = *lease_or,
          .capability_expires_at = expires_at,
          .carriage_kind = CredentialCarriageKind::kSelfDescribing,
          .binding_mode = LifecycleBindingMode::kAddressDerived,
          .constraint_claims =
              ConstraintClaims{
                  .artifact_id = std::string(artifact_id),
                  .digest_alg = metadata.digest_alg,
                  .digest_hex = metadata.digest_hex,
                  .direction = payload_direction_label(direction),
                  .operation_id = std::string(operation_id),
              },
          .direction = payload_direction_label(direction),
          .workflow_gate = WorkflowGateKind::kNone,
      });
  if (!capability_or.ok()) {
    lifecycle_manager_->release_lease(*lease_or);
    return capability_or.status();
  }

  {
    absl::MutexLock lock(&mu_);
    records_[payload_id] = Record{
        .metadata = metadata,
        .payload = std::move(payload_snapshot),
        .body_handle = body_handle,
        .descriptor = normalized_descriptor,
        .backing_identity = std::move(backing_identity),
        .backing_instance_generation = backing_instance_generation,
        .lease_id = *lease_or,
    };
  }

  tensorcast::common::v1::PayloadRefScope scope;
  scope.set_payload_id(metadata.payload_id);
  scope.set_artifact_id(metadata.artifact_id);
  scope.set_payload_size(metadata.payload_size);
  scope.set_digest_alg(metadata.digest_alg);
  scope.set_digest_hex(metadata.digest_hex);
  scope.set_direction(direction);
  if (!metadata.operation_id.empty()) {
    scope.set_operation_id(metadata.operation_id);
  }
  auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  if (!scope_or.ok()) {
    lifecycle_manager_->release_lease(*lease_or);
    return scope_or.status();
  }
  auto token_or = capability_tokens_->mint(
      daemon_id_,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PAYLOAD_REF,
      *scope_or,
      static_cast<std::uint64_t>(absl::ToUnixMillis(expires_at)));
  if (!token_or.ok()) {
    lifecycle_manager_->release_lease(*lease_or);
    return token_or.status();
  }
  return *token_or;
}

absl::StatusOr<PayloadTransportBroker::BatchRefIssueResult> PayloadTransportBroker::issue_batch_payload_ref_record(
    const v2::BatchPayloadManifest& manifest,
    std::shared_ptr<const std::string> payload,
    tensorcast::common::v1::PayloadRefDirection direction,
    std::string_view operation_id,
    absl::Time capability_expires_at,
    std::string_view consumer_daemon_id) {
  if (!batch_transport_enabled()) {
    return absl::FailedPreconditionError("batch payload transport is disabled");
  }
  if (daemon_id_.empty()) {
    return absl::FailedPreconditionError("daemon_id is required for batch_payload_ref issuance");
  }
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return absl::FailedPreconditionError("capability tokens are required for batch_payload_ref transport");
  }
  if (lifecycle_manager_ == nullptr) {
    return absl::FailedPreconditionError("batch_payload_ref lifecycle manager is unavailable");
  }
  if (direction == tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED) {
    return absl::InvalidArgumentError("batch_payload_ref direction is required");
  }
  auto manifest_status = validate_batch_payload_manifest(manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }
  if (payload && manifest.total_size() != payload->size()) {
    return absl::FailedPreconditionError("batch payload manifest total_size does not match payload size");
  }

  const absl::Time now = absl::Now();
  auto expires_at_or = resolve_payload_ref_expiry(now, options_.ttl, capability_expires_at);
  if (!expires_at_or.ok()) {
    return expires_at_or.status();
  }
  const absl::Time expires_at = *expires_at_or;
  const std::string transport_id = [&]() {
    absl::MutexLock lock(&mu_);
    prune_locked(now);
    return mint_payload_id();
  }();

  BatchRefMetadata metadata;
  metadata.issuer_daemon_id = daemon_id_;
  metadata.transport_id = transport_id;
  metadata.manifest_digest_hex = compute_batch_manifest_digest_hex(manifest);
  metadata.consumer_daemon_id = std::string(consumer_daemon_id);
  metadata.payload_size = manifest.total_size();
  metadata.direction = direction;
  metadata.operation_id = std::string(operation_id);
  metadata.expires_at = expires_at;

  auto lease_or = lifecycle_manager_->create_retention_lease(
      expires_at - now,
      std::vector<std::function<absl::Status()>>{
          [this, transport_id]() -> absl::Status {
            std::optional<store::ExportRegistration> communicator_export;
            std::vector<std::shared_ptr<void>> communicator_export_keepalives;
            bool communicator_export_requires_unregister = false;
            {
              absl::MutexLock lock(&mu_);
              const auto it = batch_records_.find(transport_id);
              if (it != batch_records_.end()) {
                communicator_export = it->second.communicator_export;
                communicator_export_keepalives = std::move(it->second.communicator_export_keepalives);
                communicator_export_requires_unregister = it->second.communicator_export_requires_unregister;
                batch_records_.erase(it);
              }
            }
            (void)communicator_export_keepalives;
            if (communicator_export_requires_unregister && communicator_export.has_value() &&
                options_.comm_manager != nullptr) {
              for (const auto& tensor_key : communicator_export->remote_memory_keys) {
                (void)options_.comm_manager->get_engine().unregister_tensor(tensor_key);
              }
            }
            return absl::OkStatus();
          },
      });
  if (!lease_or.ok()) {
    return lease_or.status();
  }

  {
    absl::MutexLock lock(&mu_);
    auto& record = batch_records_[transport_id];
    record.metadata = metadata;
    record.manifest = manifest;
    record.payload = std::move(payload);
    record.communicator_export.reset();
    record.communicator_export_keepalives.clear();
    record.communicator_export_requires_unregister = false;
    record.lease_id = *lease_or;
  }

  tensorcast::common::v1::BatchPayloadRefScope scope;
  scope.set_transport_id(metadata.transport_id);
  scope.set_payload_size(metadata.payload_size);
  scope.set_direction(direction);
  if (!metadata.operation_id.empty()) {
    scope.set_operation_id(metadata.operation_id);
  }
  if (!metadata.manifest_digest_hex.empty()) {
    scope.set_manifest_digest_hex(metadata.manifest_digest_hex);
  }
  if (!metadata.consumer_daemon_id.empty()) {
    scope.set_consumer_daemon_id(metadata.consumer_daemon_id);
  }
  auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  if (!scope_or.ok()) {
    lifecycle_manager_->release_lease(*lease_or);
    return scope_or.status();
  }
  auto token_or = capability_tokens_->mint(
      daemon_id_,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_BATCH_PAYLOAD_REF,
      *scope_or,
      static_cast<std::uint64_t>(absl::ToUnixMillis(expires_at)));
  if (!token_or.ok()) {
    lifecycle_manager_->release_lease(*lease_or);
    return token_or.status();
  }
  return BatchRefIssueResult{
      .metadata = metadata,
      .batch_payload_ref = *token_or,
      .lease_id = *lease_or,
  };
}

absl::StatusOr<std::string> PayloadTransportBroker::issue_batch_payload_ref(
    const v2::BatchPayloadManifest& manifest,
    std::shared_ptr<const std::string> payload,
    tensorcast::common::v1::PayloadRefDirection direction,
    std::string_view operation_id,
    absl::Time capability_expires_at,
    std::string_view consumer_daemon_id) {
  if (!payload || payload->empty()) {
    return absl::InvalidArgumentError("payload is required for batch_payload_ref issuance");
  }
  auto issue_or = issue_batch_payload_ref_record(
      manifest, std::move(payload), direction, operation_id, capability_expires_at, consumer_daemon_id);
  if (!issue_or.ok()) {
    return issue_or.status();
  }
  return issue_or->batch_payload_ref;
}

absl::StatusOr<PayloadTransportBroker::BatchCommunicatorExport> PayloadTransportBroker::
    issue_batch_payload_communicator_export(
        const v2::BatchPayloadManifest& manifest,
        std::shared_ptr<const std::string> payload,
        tensorcast::common::v1::PayloadRefDirection direction,
        std::string_view operation_id,
        absl::Time capability_expires_at,
        std::string_view consumer_daemon_id) {
  if (!batch_transport_communicator_enabled()) {
    return absl::FailedPreconditionError("batch communicator transport is disabled");
  }
  if (!payload || payload->empty()) {
    return absl::InvalidArgumentError("payload is required for batch communicator transport");
  }
  auto manifest_status = validate_batch_payload_manifest(manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }
  if (manifest.total_size() != payload->size()) {
    return absl::FailedPreconditionError("batch payload manifest total_size does not match payload size");
  }
  const absl::Time now = absl::Now();
  auto expires_at_or = resolve_payload_ref_expiry(now, options_.ttl, capability_expires_at);
  if (!expires_at_or.ok()) {
    return expires_at_or.status();
  }
  const absl::Time expires_at = *expires_at_or;
  if (expires_at - now < options_.minimum_batch_transport_ttl) {
    return absl::FailedPreconditionError("batch communicator transport ttl below minimum");
  }
  const absl::Time export_started_at = absl::Now();
  std::vector<void*> buffer_addresses{const_cast<char*>(payload->data())};
  std::vector<size_t> buffer_sizes{payload->size()};
  const absl::Time register_started_at = absl::Now();
  auto registration_or = options_.comm_manager->register_memory(buffer_addresses, buffer_sizes, /*device_id=*/-1);
  const absl::Duration register_elapsed = absl::Now() - register_started_at;
  if (!registration_or.ok()) {
    return registration_or.status();
  }
  const absl::Time issue_ref_started_at = absl::Now();
  auto batch_payload_ref_or = issue_batch_payload_ref_record(
      manifest, payload, direction, operation_id, capability_expires_at, consumer_daemon_id);
  const absl::Duration issue_ref_elapsed = absl::Now() - issue_ref_started_at;
  if (!batch_payload_ref_or.ok()) {
    for (const auto& tensor_key : registration_or->remote_memory_keys) {
      (void)options_.comm_manager->get_engine().unregister_tensor(tensor_key);
    }
    return batch_payload_ref_or.status();
  }
  {
    absl::MutexLock lock(&mu_);
    const auto it = batch_records_.find(batch_payload_ref_or->metadata.transport_id);
    if (it == batch_records_.end()) {
      for (const auto& tensor_key : registration_or->remote_memory_keys) {
        (void)options_.comm_manager->get_engine().unregister_tensor(tensor_key);
      }
      lifecycle_manager_->release_lease(batch_payload_ref_or->lease_id);
      return absl::NotFoundError("batch communicator transport record is missing");
    }
    it->second.communicator_export = *registration_or;
    it->second.communicator_export_requires_unregister = true;
  }
  VLOG(2) << "batch_payload_ref.communicator_export_summary"
          << " direction=" << payload_direction_label(direction) << " operation_id=" << operation_id
          << " realization=staged_slab"
          << " transport_id=" << batch_payload_ref_or->metadata.transport_id
          << " payload_bytes=" << batch_payload_ref_or->metadata.payload_size
          << " remote_keys=" << registration_or->remote_memory_keys.size()
          << " register_ms=" << absl::ToDoubleMilliseconds(register_elapsed)
          << " issue_ref_ms=" << absl::ToDoubleMilliseconds(issue_ref_elapsed)
          << " total_ms=" << absl::ToDoubleMilliseconds(absl::Now() - export_started_at);
  return BatchCommunicatorExport{
      .metadata = batch_payload_ref_or->metadata,
      .batch_payload_ref = batch_payload_ref_or->batch_payload_ref,
      .export_registration = *registration_or,
  };
}

absl::StatusOr<PayloadTransportBroker::BatchCommunicatorExport> PayloadTransportBroker::
    issue_batch_payload_communicator_export(
        const v2::BatchPayloadManifest& manifest,
        absl::Span<const BatchCommunicatorSourceSegment> source_segments,
        tensorcast::common::v1::PayloadRefDirection direction,
        std::string_view operation_id,
        absl::Time capability_expires_at,
        std::string_view consumer_daemon_id) {
  if (!batch_transport_segmented_communicator_export_enabled()) {
    return absl::FailedPreconditionError("segmented batch communicator transport is disabled");
  }
  auto manifest_status = validate_batch_payload_manifest(manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }
  if (static_cast<int>(source_segments.size()) != manifest.entries_size()) {
    return absl::InvalidArgumentError("segmented batch communicator source count mismatch");
  }
  const absl::Time now = absl::Now();
  auto expires_at_or = resolve_payload_ref_expiry(now, options_.ttl, capability_expires_at);
  if (!expires_at_or.ok()) {
    return expires_at_or.status();
  }
  const absl::Time expires_at = *expires_at_or;
  if (expires_at - now < options_.minimum_batch_transport_ttl) {
    return absl::FailedPreconditionError("batch communicator transport ttl below minimum");
  }

  store::ExportRegistration registration;
  registration.artifact_size = manifest.total_size();
  registration.location = common::memory::MemoryLocation::CPU;
  registration.device_id = -1;
  std::vector<std::shared_ptr<void>> keepalives;
  keepalives.reserve(source_segments.size());
  std::uint64_t total_payload_bytes = 0;

  for (int entry_index = 0; entry_index < manifest.entries_size(); ++entry_index) {
    const auto& segment = source_segments[entry_index];
    if (!segment.export_view.communicator_export.has_value()) {
      return absl::FailedPreconditionError("segmented batch communicator source requires communicator export");
    }
    if (segment.export_view.keepalive == nullptr) {
      return absl::FailedPreconditionError("segmented batch communicator source requires keepalive");
    }
    if (segment.export_view.memory_location != common::memory::MemoryLocation::CPU ||
        segment.export_view.communicator_export->location != common::memory::MemoryLocation::CPU) {
      return absl::FailedPreconditionError("segmented batch communicator source currently requires CPU exports");
    }
    auto segment_bytes_or = export_registration_total_bytes(*segment.export_view.communicator_export);
    if (!segment_bytes_or.ok()) {
      return segment_bytes_or.status();
    }
    if (*segment_bytes_or != manifest.entries(entry_index).length()) {
      return absl::FailedPreconditionError("segmented batch communicator source length mismatch");
    }
    if (total_payload_bytes > std::numeric_limits<std::uint64_t>::max() - *segment_bytes_or) {
      return absl::OutOfRangeError("segmented batch communicator payload exceeds uint64 range");
    }
    total_payload_bytes += *segment_bytes_or;
    if (entry_index == 0) {
      registration.comm_dev_type = segment.export_view.communicator_export->comm_dev_type;
    } else if (registration.comm_dev_type != segment.export_view.communicator_export->comm_dev_type) {
      return absl::FailedPreconditionError("segmented batch communicator source requires one communicator device type");
    }
    registration.buffer_addresses.insert(
        registration.buffer_addresses.end(),
        segment.export_view.communicator_export->buffer_addresses.begin(),
        segment.export_view.communicator_export->buffer_addresses.end());
    registration.buffer_sizes.insert(
        registration.buffer_sizes.end(),
        segment.export_view.communicator_export->buffer_sizes.begin(),
        segment.export_view.communicator_export->buffer_sizes.end());
    registration.remote_memory_keys.insert(
        registration.remote_memory_keys.end(),
        segment.export_view.communicator_export->remote_memory_keys.begin(),
        segment.export_view.communicator_export->remote_memory_keys.end());
    keepalives.push_back(segment.export_view.keepalive);
  }
  if (total_payload_bytes != manifest.total_size()) {
    return absl::FailedPreconditionError("segmented batch communicator payload size mismatch");
  }

  const absl::Time issue_ref_started_at = absl::Now();
  auto batch_payload_ref_or = issue_batch_payload_ref_record(
      manifest,
      /*payload=*/nullptr,
      direction,
      operation_id,
      capability_expires_at,
      consumer_daemon_id);
  const absl::Duration issue_ref_elapsed = absl::Now() - issue_ref_started_at;
  if (!batch_payload_ref_or.ok()) {
    return batch_payload_ref_or.status();
  }
  {
    absl::MutexLock lock(&mu_);
    const auto it = batch_records_.find(batch_payload_ref_or->metadata.transport_id);
    if (it == batch_records_.end()) {
      lifecycle_manager_->release_lease(batch_payload_ref_or->lease_id);
      return absl::NotFoundError("segmented batch communicator transport record is missing");
    }
    it->second.communicator_export = registration;
    it->second.communicator_export_keepalives = std::move(keepalives);
    it->second.communicator_export_requires_unregister = false;
  }
  VLOG(2) << "batch_payload_ref.communicator_export_summary"
          << " direction=" << payload_direction_label(direction) << " operation_id=" << operation_id
          << " realization=segmented_source"
          << " transport_id=" << batch_payload_ref_or->metadata.transport_id
          << " payload_bytes=" << batch_payload_ref_or->metadata.payload_size
          << " remote_keys=" << registration.remote_memory_keys.size() << " source_segments=" << source_segments.size()
          << " issue_ref_ms=" << absl::ToDoubleMilliseconds(issue_ref_elapsed)
          << " total_ms=" << absl::ToDoubleMilliseconds(absl::Now() - issue_ref_started_at);
  return BatchCommunicatorExport{
      .metadata = batch_payload_ref_or->metadata,
      .batch_payload_ref = batch_payload_ref_or->batch_payload_ref,
      .export_registration = registration,
  };
}

absl::StatusOr<PayloadTransportBroker::BatchCommunicatorExport> PayloadTransportBroker::
    issue_batch_payload_communicator_export(
        const v2::BatchPayloadManifest& manifest,
        absl::Span<const BatchCommunicatorRegionSourceSegment> source_segments,
        tensorcast::common::v1::PayloadRefDirection direction,
        std::string_view operation_id,
        absl::Time capability_expires_at,
        std::string_view consumer_daemon_id) {
  if (!batch_transport_segmented_communicator_export_enabled()) {
    return absl::FailedPreconditionError("segmented batch communicator transport is disabled");
  }
  if (options_.comm_manager == nullptr || !options_.comm_manager->is_enabled()) {
    return absl::FailedPreconditionError("communication manager is unavailable for region source export");
  }
  auto manifest_status = validate_batch_payload_manifest(manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }
  if (static_cast<int>(source_segments.size()) != manifest.entries_size()) {
    return absl::InvalidArgumentError("segmented region source count mismatch");
  }
  const absl::Time now = absl::Now();
  auto expires_at_or = resolve_payload_ref_expiry(now, options_.ttl, capability_expires_at);
  if (!expires_at_or.ok()) {
    return expires_at_or.status();
  }
  const absl::Time expires_at = *expires_at_or;
  if (expires_at - now < options_.minimum_batch_transport_ttl) {
    return absl::FailedPreconditionError("batch communicator transport ttl below minimum");
  }

  std::vector<void*> buffer_addresses;
  std::vector<size_t> buffer_sizes;
  std::vector<std::shared_ptr<void>> keepalives;
  buffer_addresses.reserve(source_segments.size());
  buffer_sizes.reserve(source_segments.size());
  keepalives.reserve(source_segments.size());

  std::uint64_t total_payload_bytes = 0;
  for (int entry_index = 0; entry_index < manifest.entries_size(); ++entry_index) {
    const auto& segment = source_segments[entry_index];
    if (segment.data == nullptr) {
      return absl::FailedPreconditionError("segmented region source requires data pointer");
    }
    if (segment.size_bytes == 0) {
      return absl::FailedPreconditionError("segmented region source requires non-empty segment");
    }
    if (segment.keepalive == nullptr) {
      return absl::FailedPreconditionError("segmented region source requires keepalive");
    }
    if (segment.size_bytes != manifest.entries(entry_index).length()) {
      return absl::FailedPreconditionError("segmented region source length mismatch");
    }
    if (segment.size_bytes > std::numeric_limits<size_t>::max()) {
      return absl::OutOfRangeError("segmented region source exceeds host memory limits");
    }
    if (total_payload_bytes > std::numeric_limits<std::uint64_t>::max() - segment.size_bytes) {
      return absl::OutOfRangeError("segmented region source payload exceeds uint64 range");
    }
    total_payload_bytes += segment.size_bytes;
    buffer_addresses.push_back(const_cast<void*>(segment.data));
    buffer_sizes.push_back(static_cast<size_t>(segment.size_bytes));
    keepalives.push_back(segment.keepalive);
  }
  if (total_payload_bytes != manifest.total_size()) {
    return absl::FailedPreconditionError("segmented region source payload size mismatch");
  }

  const absl::Time export_started_at = absl::Now();
  const absl::Time register_started_at = absl::Now();
  auto registration_or = options_.comm_manager->register_memory(buffer_addresses, buffer_sizes, /*device_id=*/-1);
  const absl::Duration register_elapsed = absl::Now() - register_started_at;
  if (!registration_or.ok()) {
    return registration_or.status();
  }
  registration_or->location = common::memory::MemoryLocation::CPU;
  registration_or->device_id = -1;
  registration_or->artifact_size = manifest.total_size();

  const auto unregister_registered = [&]() {
    for (const auto& tensor_key : registration_or->remote_memory_keys) {
      (void)options_.comm_manager->get_engine().unregister_tensor(tensor_key);
    }
  };

  const absl::Time issue_ref_started_at = absl::Now();
  auto batch_payload_ref_or = issue_batch_payload_ref_record(
      manifest,
      /*payload=*/nullptr,
      direction,
      operation_id,
      capability_expires_at,
      consumer_daemon_id);
  const absl::Duration issue_ref_elapsed = absl::Now() - issue_ref_started_at;
  if (!batch_payload_ref_or.ok()) {
    unregister_registered();
    return batch_payload_ref_or.status();
  }
  {
    absl::MutexLock lock(&mu_);
    const auto it = batch_records_.find(batch_payload_ref_or->metadata.transport_id);
    if (it == batch_records_.end()) {
      unregister_registered();
      lifecycle_manager_->release_lease(batch_payload_ref_or->lease_id);
      return absl::NotFoundError("segmented region source transport record is missing");
    }
    it->second.communicator_export = *registration_or;
    it->second.communicator_export_keepalives = std::move(keepalives);
    it->second.communicator_export_requires_unregister = true;
  }
  VLOG(2) << "batch_payload_ref.communicator_export_summary"
          << " direction=" << payload_direction_label(direction) << " operation_id=" << operation_id
          << " realization=segmented_region_source"
          << " transport_id=" << batch_payload_ref_or->metadata.transport_id
          << " payload_bytes=" << batch_payload_ref_or->metadata.payload_size
          << " remote_keys=" << registration_or->remote_memory_keys.size()
          << " source_segments=" << source_segments.size()
          << " register_ms=" << absl::ToDoubleMilliseconds(register_elapsed)
          << " issue_ref_ms=" << absl::ToDoubleMilliseconds(issue_ref_elapsed)
          << " total_ms=" << absl::ToDoubleMilliseconds(absl::Now() - export_started_at);
  return BatchCommunicatorExport{
      .metadata = batch_payload_ref_or->metadata,
      .batch_payload_ref = batch_payload_ref_or->batch_payload_ref,
      .export_registration = *registration_or,
  };
}

absl::StatusOr<PayloadTransportBroker::RefMetadata> PayloadTransportBroker::inspect_payload_ref(
    std::string_view payload_ref,
    absl::Time now,
    bool require_not_expired) const {
  if (payload_ref.empty()) {
    return absl::InvalidArgumentError("payload_ref is required");
  }
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return absl::FailedPreconditionError("capability tokens are required for payload_ref transport");
  }
  if (!common::CapabilityTokenManager::looks_like_envelope(payload_ref)) {
    return absl::InvalidArgumentError("payload_ref format is invalid");
  }
  auto env_or = capability_tokens_->verify(
      payload_ref,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PAYLOAD_REF,
      /*expected_issuer=*/"",
      now,
      require_not_expired);
  if (!env_or.ok()) {
    return env_or.status();
  }

  tensorcast::common::v1::PayloadRefScope scope;
  if (!scope.ParseFromString(env_or->scope())) {
    return absl::InvalidArgumentError("payload_ref scope parse failed");
  }

  RefMetadata metadata;
  metadata.issuer_daemon_id = env_or->issuer_daemon_id();
  metadata.payload_id = scope.payload_id();
  metadata.artifact_id = scope.artifact_id();
  metadata.payload_size = scope.payload_size();
  metadata.digest_alg = to_lower_copy(scope.digest_alg());
  metadata.digest_hex = to_lower_copy(scope.digest_hex());
  metadata.direction = scope.direction();
  metadata.operation_id = scope.operation_id();
  metadata.expires_at = absl::FromUnixMillis(static_cast<std::int64_t>(env_or->expires_at_ms()));
  if (metadata.payload_id.empty() || metadata.artifact_id.empty() || metadata.payload_size == 0 ||
      metadata.digest_alg.empty() || metadata.digest_hex.empty() ||
      metadata.direction == tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED) {
    return absl::InvalidArgumentError("payload_ref scope missing required fields");
  }
  if (require_not_expired && metadata.expires_at <= now) {
    return absl::PermissionDeniedError("payload_ref expired");
  }
  return metadata;
}

absl::StatusOr<PayloadTransportBroker::BatchRefMetadata> PayloadTransportBroker::inspect_batch_payload_ref(
    std::string_view batch_payload_ref,
    absl::Time now,
    bool require_not_expired) const {
  if (batch_payload_ref.empty()) {
    return absl::InvalidArgumentError("batch_payload_ref is required");
  }
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return absl::FailedPreconditionError("capability tokens are required for batch_payload_ref transport");
  }
  if (!common::CapabilityTokenManager::looks_like_envelope(batch_payload_ref)) {
    return absl::InvalidArgumentError("batch_payload_ref format is invalid");
  }
  auto env_or = capability_tokens_->verify(
      batch_payload_ref,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_BATCH_PAYLOAD_REF,
      /*expected_issuer=*/"",
      now,
      require_not_expired);
  if (!env_or.ok()) {
    return env_or.status();
  }

  tensorcast::common::v1::BatchPayloadRefScope scope;
  if (!scope.ParseFromString(env_or->scope())) {
    return absl::InvalidArgumentError("batch_payload_ref scope parse failed");
  }

  BatchRefMetadata metadata;
  metadata.issuer_daemon_id = env_or->issuer_daemon_id();
  metadata.transport_id = scope.transport_id();
  metadata.manifest_digest_hex = to_lower_copy(scope.manifest_digest_hex());
  metadata.consumer_daemon_id = scope.consumer_daemon_id();
  metadata.payload_size = scope.payload_size();
  metadata.direction = scope.direction();
  metadata.operation_id = scope.operation_id();
  metadata.expires_at = absl::FromUnixMillis(static_cast<std::int64_t>(env_or->expires_at_ms()));
  if (metadata.transport_id.empty() || metadata.payload_size == 0 ||
      metadata.direction == tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED) {
    return absl::InvalidArgumentError("batch_payload_ref scope missing required fields");
  }
  if (require_not_expired && metadata.expires_at <= now) {
    return absl::PermissionDeniedError("batch_payload_ref expired");
  }
  return metadata;
}

FrontDoorCredentialContext PayloadTransportBroker::build_payload_ref_front_door_context(
    const RefMetadata& metadata,
    std::string_view payload_ref,
    std::uint64_t subject_generation) const {
  ParsedCredential parsed_credential{
      .address =
          CapabilityBindingAddress{
              .route_principal = make_issuer_route_principal(
                  metadata.issuer_daemon_id.empty() ? daemon_id_ : metadata.issuer_daemon_id),
              .family = LifecycleCapabilityFamily::kServe,
              .binding_space = LifecycleBindingSpace::kPayload,
              .binding_key_kind = BindingKeyKind::kPayloadId,
              .binding_key = metadata.payload_id,
              .epochs =
                  LifecycleEpochs{
                      .subject_generation = subject_generation == 0 ? 1 : subject_generation,
                  },
          },
      .front_door_kind = LifecycleFrontDoorKind::kPayloadRef,
      .credential_expires_at = metadata.expires_at,
      .carriage_kind = CredentialCarriageKind::kSelfDescribing,
      .binding_mode = LifecycleBindingMode::kAddressDerived,
      .constraint_claims =
          ConstraintClaims{
              .artifact_id = metadata.artifact_id,
              .digest_alg = metadata.digest_alg,
              .digest_hex = metadata.digest_hex,
              .direction = payload_direction_label(metadata.direction),
              .operation_id = metadata.operation_id,
          },
  };

  return FrontDoorCredentialContext{
      .parsed_credential = std::move(parsed_credential),
      .forwardable_evidence =
          ForwardableCredentialEvidence{
              .evidence_kind = CredentialEvidenceKind::kRawCredential,
              .raw_credential_bytes = std::string(payload_ref),
          },
      .local_observations = LocalObservationSet{},
  };
}

absl::StatusOr<AuthorityRef> PayloadTransportBroker::derive_issuer_authority_ref(
    const FrontDoorCredentialContext& front_door_context) const {
  const auto& route_principal = front_door_context.parsed_credential.address.route_principal;
  if (route_principal.principal_kind != LifecycleRoutePrincipalKind::kIssuerDaemon ||
      route_principal.principal_id.empty()) {
    return absl::FailedPreconditionError("payload_ref issuer authority must derive from issuer route_principal");
  }
  return AuthorityRef{
      .authority_kind = AuthorityKind::kIssuerDaemon,
      .authority_id = route_principal.principal_id,
  };
}

absl::StatusOr<PortableParsedCredential> PayloadTransportBroker::derive_payload_ref_portable_credential(
    const FrontDoorCredentialContext& front_door_context) const {
  const ParsedCredential& parsed_credential = front_door_context.parsed_credential;
  if (parsed_credential.carriage_kind != CredentialCarriageKind::kSelfDescribing) {
    return absl::FailedPreconditionError("payload_ref portable credential requires self-describing carriage");
  }
  return PortableParsedCredential{
      .address = parsed_credential.address,
      .front_door_kind = parsed_credential.front_door_kind,
      .credential_expires_at = parsed_credential.credential_expires_at,
      .binding_mode = parsed_credential.binding_mode,
      .portable_constraint_claims = parsed_credential.constraint_claims,
  };
}

absl::StatusOr<OwnerStageReply> PayloadTransportBroker::resolve_payload_ref_issuer_reply(
    std::string_view payload_ref,
    const LocalResolvedPayload& local_resolved_payload,
    bool remote_consumer) {
  if (lifecycle_kernel_ == nullptr) {
    return absl::FailedPreconditionError("payload_ref lifecycle kernel is unavailable");
  }
  auto issuer_authority_ref_or = derive_issuer_authority_ref(local_resolved_payload.front_door_context);
  if (!issuer_authority_ref_or.ok()) {
    return issuer_authority_ref_or.status();
  }
  auto admitted_or = lifecycle_kernel_->admit_redemption(local_resolved_payload.front_door_context.parsed_credential);
  if (!admitted_or.ok()) {
    return admitted_or.status();
  }
  const auto release_use_guard = [&]() {
    auto release_status = lifecycle_kernel_->release_use_guard(admitted_or->use_guard);
    LOG_IF(WARNING, !release_status.ok()) << "payload_ref: failed to release local use guard: " << release_status;
  };

  ResolvedSourceCapability resolution;
  resolution.selection_identity = make_byte_artifact_selection_identity(local_resolved_payload.metadata.artifact_id);
  resolution.verified_content_descriptor = local_resolved_payload.descriptor.layout_id.empty()
      ? payload_metadata_to_verified_content_descriptor(local_resolved_payload.metadata, /*layout_id=*/"")
      : body_descriptor_to_verified_content_descriptor(local_resolved_payload.descriptor);

  if (remote_consumer) {
    resolution.source_kind = store::loading::MaterializationSource::kP2P;
    resolution.payload_ref = std::string(payload_ref);
    auto capability_or = mint_serving_capability(
        MintServingCapabilityRequest{
            .capability_id = std::string(payload_ref),
            .expires_at = local_resolved_payload.metadata.expires_at,
            .mode = BodyCapabilityResolutionMode::kChunkRpcFallback,
            .local = false,
            .subject_kind = ServingCapabilitySubjectKind::kCopiedPayload,
            .lifecycle_owner_ref =
                LifecycleOwnerRef{
                    .owner_kind = LifecycleOwnerKind::kPayloadRefToken,
                    .owner_id = local_resolved_payload.metadata.payload_id,
                },
        });
    if (!capability_or.ok()) {
      release_use_guard();
      return capability_or.status();
    }
    resolution.serving_capability = std::move(*capability_or);
  } else {
    const bool prefer_payload_snapshot =
        local_resolved_payload.metadata.direction == tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET &&
        local_resolved_payload.payload;
    if (!local_resolved_payload.body_handle.empty() && !prefer_payload_snapshot) {
      resolution.source_kind = store::loading::MaterializationSource::kLocalReplica;
      resolution.body_capability = ResolvedBodyCapability{
          .mode = BodyCapabilityResolutionMode::kLocalBodyHandle,
          .local = true,
          .body_handle = local_resolved_payload.body_handle,
          .descriptor = local_resolved_payload.descriptor,
      };
      auto backing_identity = local_resolved_payload.backing_identity;
      if (!backing_identity.has_value()) {
        backing_identity =
            body_descriptor_to_backing_identity(local_resolved_payload.descriptor, local_resolved_payload.body_handle);
      }
      if (!backing_identity.has_value() ||
          !store::runtime::ingestion::backing_identity_matches_replica_key(*backing_identity)) {
        release_use_guard();
        return absl::FailedPreconditionError("payload_ref backing_identity is missing or inconsistent");
      }
      std::uint64_t backing_instance_generation = local_resolved_payload.backing_instance_generation;
      if (backing_instance_generation == 0) {
        backing_instance_generation = local_resolved_payload.body_handle.binding_generation();
      }
      if (backing_instance_generation == 0) {
        release_use_guard();
        return absl::FailedPreconditionError("payload_ref backing_instance_generation is missing");
      }
      auto capability_or = mint_serving_capability(
          MintServingCapabilityRequest{
              .capability_id = std::string(payload_ref),
              .expires_at = local_resolved_payload.metadata.expires_at,
              .mode = resolution.body_capability->mode,
              .local = true,
              .subject_kind = ServingCapabilitySubjectKind::kBacking,
              .lifecycle_owner_ref =
                  LifecycleOwnerRef{
                      .owner_kind = LifecycleOwnerKind::kPayloadRefToken,
                      .owner_id = local_resolved_payload.metadata.payload_id,
                  },
              .backing_identity = backing_identity,
              .backing_instance_generation = backing_instance_generation,
          });
      if (!capability_or.ok()) {
        release_use_guard();
        return capability_or.status();
      }
      resolution.backing_identity = std::move(backing_identity);
      resolution.serving_capability = std::move(*capability_or);
    } else {
      resolution.source_kind = store::loading::MaterializationSource::kLocalReplica;
      resolution.inline_payload = local_resolved_payload.payload;
      auto capability_or = mint_serving_capability(
          MintServingCapabilityRequest{
              .capability_id = std::string(payload_ref),
              .expires_at = local_resolved_payload.metadata.expires_at,
              .mode = BodyCapabilityResolutionMode::kLoader,
              .local = true,
              .subject_kind = ServingCapabilitySubjectKind::kCopiedPayload,
              .lifecycle_owner_ref =
                  LifecycleOwnerRef{
                      .owner_kind = LifecycleOwnerKind::kPayloadRefToken,
                      .owner_id = local_resolved_payload.metadata.payload_id,
                  },
          });
      if (!capability_or.ok()) {
        release_use_guard();
        return capability_or.status();
      }
      resolution.serving_capability = std::move(*capability_or);
    }
  }

  auto validation_status = validate_resolved_source_capability(resolution);
  if (!validation_status.ok()) {
    release_use_guard();
    return validation_status;
  }
  record_payload_ref_resolution_metrics(resolution.serving_capability.mode, /*local=*/!remote_consumer);
  release_use_guard();
  return OwnerStageReply{
      .answered_by = *issuer_authority_ref_or,
      .path_family = std::string(kImmediateLoweringPathFamily),
      .stage_ref = std::string(kIssuerValidateStageRef),
      .reply_kind = OwnerStageReplyKind::kReadyForLowering,
      .resolved_source_capability = std::make_shared<ResolvedSourceCapability>(std::move(resolution)),
  };
}

absl::StatusOr<ResolvedSourceCapability> PayloadTransportBroker::resolve_payload_ref_capability_from_reply(
    const OwnerStageReply& issuer_reply) const {
  switch (issuer_reply.reply_kind) {
    case OwnerStageReplyKind::kReadyForLowering: {
      if (!issuer_reply.resolved_source_capability) {
        return absl::DataLossError("payload_ref issuer reply omitted resolved_source_capability");
      }
      auto validation_status = validate_resolved_source_capability(*issuer_reply.resolved_source_capability);
      if (!validation_status.ok()) {
        return validation_status;
      }
      return *issuer_reply.resolved_source_capability;
    }
    case OwnerStageReplyKind::kRetryLater:
      return absl::UnavailableError("payload_ref issuer replied retry_later; immediate lowering is not available");
    case OwnerStageReplyKind::kAttachExisting:
      return absl::FailedPreconditionError(
          issuer_reply.attachment_ref.has_value()
              ? absl::StrCat(
                    "payload_ref issuer replied attach_existing via ",
                    issuer_reply.attachment_ref->attachment_kind,
                    ":",
                    issuer_reply.attachment_ref->attachment_id)
              : "payload_ref issuer replied attach_existing; immediate lowering is not available");
    case OwnerStageReplyKind::kTerminal:
      if (!issuer_reply.terminal_projection.has_value()) {
        return absl::DataLossError("payload_ref issuer terminal reply omitted terminal_projection");
      }
      return terminal_projection_to_status(
          *issuer_reply.terminal_projection,
          "payload_ref issuer replied terminal; immediate lowering is not available");
    case OwnerStageReplyKind::kContinueWithAuthority:
      return absl::FailedPreconditionError(
          "payload_ref issuer route must not continue_with_authority to a third authority");
  }
  return absl::FailedPreconditionError("payload_ref issuer reply kind is unsupported");
}

absl::StatusOr<PayloadTransportBroker::PayloadRefFrontDoorContext> PayloadTransportBroker::inspect_payload_ref_context(
    std::string_view payload_ref,
    std::string_view expected_artifact_id,
    absl::Time now,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto resolved_or = resolve_local_payload_ref_record(
      payload_ref, expected_artifact_id, now, expected_direction, expected_operation_id);
  if (!resolved_or.ok()) {
    return resolved_or.status();
  }
  return PayloadRefFrontDoorContext{
      .metadata = resolved_or->metadata,
      .front_door_context = resolved_or->front_door_context,
  };
}

absl::StatusOr<RoutedAuthorityRequest> PayloadTransportBroker::build_payload_ref_issuer_routed_request(
    const RefMetadata& metadata,
    const FrontDoorCredentialContext& front_door_context,
    std::string_view route_address,
    absl::Span<const LocalObservationRoutingRule> local_observation_rules) const {
  if (!front_door_context.forwardable_evidence.has_value()) {
    return absl::FailedPreconditionError("payload_ref issuer route requires forwardable_evidence");
  }
  auto raw_payload_ref_or = require_payload_ref_raw_issuer_evidence(*front_door_context.forwardable_evidence);
  if (!raw_payload_ref_or.ok()) {
    return raw_payload_ref_or.status();
  }
  auto issuer_authority_ref_or = derive_issuer_authority_ref(front_door_context);
  if (!issuer_authority_ref_or.ok()) {
    return issuer_authority_ref_or.status();
  }
  auto portable_credential_or = derive_payload_ref_portable_credential(front_door_context);
  if (!portable_credential_or.ok()) {
    return portable_credential_or.status();
  }
  const std::string root_request_id = payload_ref_root_request_id(metadata);
  auto forwarded_claims_or = sanitize_local_observations_for_routing(
      front_door_context.local_observations,
      local_observation_rules,
      ingress_forwarding_authority_ref(daemon_id_),
      *issuer_authority_ref_or,
      root_request_id,
      kImmediateLoweringPathFamily,
      std::nullopt);
  if (!forwarded_claims_or.ok()) {
    return forwarded_claims_or.status();
  }
  RoutedAuthorityRequest routed_request{
      .authority_ref = *issuer_authority_ref_or,
      .path_family = std::string(kImmediateLoweringPathFamily),
      .stage_ref = std::string(kIssuerValidateStageRef),
      .portable_credential = *portable_credential_or,
      .forwardable_evidence = front_door_context.forwardable_evidence,
      .portable_credential_envelope = make_delegation_envelope(
          *issuer_authority_ref_or,
          root_request_id,
          kImmediateLoweringPathFamily,
          DelegationPayloadKind::kPortableCredential,
          DelegationClass::kBootstrapSafe,
          front_door_context.parsed_credential.credential_expires_at),
      .forwardable_evidence_envelope = make_delegation_envelope(
          *issuer_authority_ref_or,
          root_request_id,
          kImmediateLoweringPathFamily,
          DelegationPayloadKind::kForwardableEvidence,
          DelegationClass::kOwnerScopedSensitive,
          front_door_context.parsed_credential.credential_expires_at),
      .hop_auth_context = hop_auth_context_for_route_address(route_address, options_.inter_daemon_grpc_security),
      .forwarded_claims = std::move(*forwarded_claims_or),
      .forwarded_claims_envelope = std::nullopt,
      .request_metadata =
          RoutedRequestMetadata{
              .root_request_id = std::move(root_request_id),
              .hop_budget_remaining = 1,
          },
  };
  if (!routed_request.forwarded_claims.empty()) {
    routed_request.forwarded_claims_envelope = make_delegation_envelope(
        *issuer_authority_ref_or,
        routed_request.request_metadata.root_request_id,
        routed_request.path_family,
        DelegationPayloadKind::kForwardedClaim,
        DelegationClass::kOwnerScopedSensitive,
        front_door_context.parsed_credential.credential_expires_at);
  }
  if (!metadata.operation_id.empty()) {
    routed_request.request_metadata.idempotency_key = metadata.operation_id;
  }
  auto validation_status = routed_authority_wire::validate_routed_authority_request_shape(routed_request);
  if (!validation_status.ok()) {
    return validation_status;
  }
  return routed_request;
}

absl::StatusOr<PayloadTransportBroker::ResolvedPayload> PayloadTransportBroker::resolve_local_payload_ref(
    std::string_view payload_ref,
    std::string_view expected_artifact_id,
    absl::Time now,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto resolved_or = resolve_local_payload_ref_record(
      payload_ref, expected_artifact_id, now, expected_direction, expected_operation_id);
  if (!resolved_or.ok()) {
    return resolved_or.status();
  }
  if (resolved_or->payload) {
    return ResolvedPayload{
        .metadata = resolved_or->metadata,
        .payload = *resolved_or->payload,
    };
  }
  auto payload_or = resolved_or->body_handle.read_all_bytes();
  if (!payload_or.ok()) {
    return payload_or.status();
  }
  return ResolvedPayload{
      .metadata = resolved_or->metadata,
      .payload = std::move(*payload_or),
  };
}

absl::StatusOr<ResolvedSourceCapability> PayloadTransportBroker::resolve_payload_ref_capability(
    WorkerDirectoryCache& worker_directory_cache,
    std::string_view payload_ref,
    std::string_view expected_artifact_id,
    absl::Time now,
    absl::Duration worker_directory_staleness_budget,
    std::string_view local_daemon_id,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto metadata_or = inspect_payload_ref(payload_ref, now, /*require_not_expired=*/true);
  if (!metadata_or.ok()) {
    return metadata_or.status();
  }
  if (expected_direction != tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED &&
      metadata_or->direction != expected_direction) {
    return absl::FailedPreconditionError("payload_ref direction mismatch");
  }
  if (!metadata_or->operation_id.empty()) {
    if (expected_operation_id.empty()) {
      return absl::InvalidArgumentError("operation_id is required for payload_ref");
    }
    if (metadata_or->operation_id != expected_operation_id) {
      return absl::FailedPreconditionError("payload_ref operation_id mismatch");
    }
  }

  if (metadata_or->issuer_daemon_id.empty() || metadata_or->issuer_daemon_id == local_daemon_id) {
    auto local_or = resolve_local_payload_ref_record(
        payload_ref, expected_artifact_id, now, expected_direction, expected_operation_id);
    if (!local_or.ok()) {
      return local_or.status();
    }
    auto issuer_reply_or = resolve_payload_ref_issuer_reply(payload_ref, *local_or, /*remote_consumer=*/false);
    if (!issuer_reply_or.ok()) {
      return issuer_reply_or.status();
    }
    return resolve_payload_ref_capability_from_reply(*issuer_reply_or);
  }
  const FrontDoorCredentialContext front_door_context =
      build_payload_ref_front_door_context(*metadata_or, payload_ref, 1);
  return resolve_remote_payload_ref_capability_via_issuer_route(
      worker_directory_cache, payload_ref, *metadata_or, front_door_context, now, worker_directory_staleness_budget);
}

absl::StatusOr<ResolvedSourceCapability> PayloadTransportBroker::resolve_remote_payload_ref_capability_via_issuer_route(
    WorkerDirectoryCache& worker_directory_cache,
    std::string_view payload_ref,
    const RefMetadata& metadata,
    const FrontDoorCredentialContext& front_door_context,
    absl::Time now,
    absl::Duration worker_directory_staleness_budget) {
  auto issuer_authority_ref_or = derive_issuer_authority_ref(front_door_context);
  if (!issuer_authority_ref_or.ok()) {
    return issuer_authority_ref_or.status();
  }
  auto address_or = worker_directory_cache.resolve_daemon_address(
      issuer_authority_ref_or->authority_id, now, worker_directory_staleness_budget);
  if (!address_or.ok()) {
    record_payload_ref_issuer_route_metrics("remote", "route_resolution_failed", "route_unavailable");
    LOG(WARNING) << "payload_ref issuer route resolution failed for authority_id="
                 << issuer_authority_ref_or->authority_id << ": " << address_or.status();
    return address_or.status();
  }
  const AuthorityLocatorResult initial_locator_result{
      .authority_ref = *issuer_authority_ref_or,
      .target_daemon_id = issuer_authority_ref_or->authority_id,
      .target_address = *address_or,
      .resolved_at = now,
      .staleness_budget = worker_directory_staleness_budget,
  };

  auto channel = create_inter_daemon_channel(*address_or, options_.inter_daemon_channel_credentials);
  auto stub = v2::StoreDaemonService::NewStub(channel);
  grpc::ClientContext client_ctx;
  client_ctx.set_deadline(
      std::chrono::system_clock::now() + std::chrono::milliseconds(absl::ToInt64Milliseconds(fetch_deadline())));

  auto routed_request_or = build_payload_ref_issuer_routed_request(metadata, front_door_context, *address_or);
  if (!routed_request_or.ok()) {
    record_payload_ref_issuer_route_metrics("remote", "request_build_failed", "none");
    return routed_request_or.status();
  }
  v2::RouteAuthorityStageRequest request;
  routed_authority_wire::populate_proto_routed_authority_request(*routed_request_or, request.mutable_routed_request());

  v2::RouteAuthorityStageResponse response;
  const auto rpc_status = stub->RouteAuthorityStage(&client_ctx, request, &response);
  if (!rpc_status.ok()) {
    record_payload_ref_issuer_route_metrics("remote", "rpc_transport_error", "route_unavailable");
    LOG(WARNING) << "payload_ref issuer route rpc failed for authority_id=" << issuer_authority_ref_or->authority_id
                 << " address=" << *address_or << ": " << rpc_status.error_message();
    return absl::UnavailableError(rpc_status.error_message());
  }
  auto item_status = absl_status_from_batch_item_status(response.status(), response.message());
  if (!item_status.ok()) {
    const bool issuer_loss = item_status.code() == absl::StatusCode::kNotFound ||
        item_status.code() == absl::StatusCode::kFailedPrecondition;
    record_payload_ref_issuer_route_metrics(
        "remote", "issuer_stage_error", issuer_loss ? "issuer_state_missing" : "none");
    if (issuer_loss) {
      LOG(WARNING) << "payload_ref issuer route failed closed for authority_id="
                   << issuer_authority_ref_or->authority_id << " due to issuer-stage state loss: " << item_status;
    }
    return item_status;
  }
  if (!response.has_owner_stage_reply()) {
    record_payload_ref_issuer_route_metrics("remote", "reply_missing", "none");
    return absl::DataLossError("payload_ref issuer route omitted owner_stage_reply");
  }
  const auto client_transport_security_context =
      DistributedSecurityKernel::transport_security_context_from_client_context(client_ctx);
  const auto authenticated_peer_identity =
      DistributedSecurityKernel::derive_authenticated_peer_identity(client_transport_security_context);
  std::optional<AuthorityLocatorResult> current_locator_result;
  auto current_address_or = worker_directory_cache.resolve_daemon_address(
      issuer_authority_ref_or->authority_id, absl::Now(), worker_directory_staleness_budget);
  if (current_address_or.ok()) {
    current_locator_result = AuthorityLocatorResult{
        .authority_ref = *issuer_authority_ref_or,
        .target_daemon_id = issuer_authority_ref_or->authority_id,
        .target_address = *current_address_or,
        .resolved_at = absl::Now(),
        .staleness_budget = worker_directory_staleness_budget,
    };
  }
  auto reply_admission_status = DistributedSecurityKernel::admit_reply(
      request.routed_request(),
      response.owner_stage_reply(),
      authenticated_peer_identity,
      initial_locator_result,
      current_locator_result,
      DistributedSecurityKernel::declared_stage_disclosure_policy(request.routed_request()).continuity_class);
  if (!reply_admission_status.ok()) {
    const std::string issuer_loss_outcome =
        reply_admission_status.code() == absl::StatusCode::kUnavailable ? "continuity_lost" : "binding_failed";
    record_payload_ref_issuer_route_metrics("remote", "reply_admission_failed", issuer_loss_outcome);
    LOG(WARNING) << "payload_ref issuer reply admission failed for authority_id="
                 << issuer_authority_ref_or->authority_id << ": " << reply_admission_status;
    return reply_admission_status;
  }
  auto owner_stage_reply_or = owner_stage_reply_from_proto(response.owner_stage_reply());
  if (!owner_stage_reply_or.ok()) {
    record_payload_ref_issuer_route_metrics("remote", "reply_payload_invalid", "none");
    return owner_stage_reply_or.status();
  }
  auto resolution_or = resolve_payload_ref_capability_from_reply(*owner_stage_reply_or);
  if (!resolution_or.ok()) {
    record_payload_ref_issuer_route_metrics(
        "remote",
        owner_stage_reply_or->reply_kind == OwnerStageReplyKind::kReadyForLowering ? "reply_projection_failed"
                                                                                   : "reply_non_lowering",
        "none",
        owner_stage_reply_kind_label(owner_stage_reply_or->reply_kind));
    return resolution_or.status();
  }
  if (resolution_or->payload_ref != payload_ref) {
    record_payload_ref_issuer_route_metrics(
        "remote", "reply_payload_mismatch", "none", owner_stage_reply_kind_label(owner_stage_reply_or->reply_kind));
    return absl::FailedPreconditionError("payload_ref issuer reply payload_ref mismatch");
  }
  record_payload_ref_issuer_route_metrics(
      "remote", "admitted", "none", owner_stage_reply_kind_label(owner_stage_reply_or->reply_kind));
  return *resolution_or;
}

absl::StatusOr<std::optional<OwnerStageReply>> PayloadTransportBroker::maybe_route_authority_stage(
    const RoutedAuthorityRequest& routed_request,
    absl::Time now) {
  if (is_payload_ref_issuer_stage(routed_request.path_family, routed_request.stage_ref)) {
    auto reply_or = route_payload_ref_issuer_stage(routed_request, now);
    if (!reply_or.ok()) {
      return reply_or.status();
    }
    return std::optional<OwnerStageReply>(std::move(*reply_or));
  }
  return std::optional<OwnerStageReply>();
}

absl::StatusOr<OwnerStageReply> PayloadTransportBroker::route_authority_stage(
    const RoutedAuthorityRequest& routed_request,
    absl::Time now) {
  auto maybe_reply_or = maybe_route_authority_stage(routed_request, now);
  if (!maybe_reply_or.ok()) {
    return maybe_reply_or.status();
  }
  if (!maybe_reply_or->has_value()) {
    return undeclared_route_authority_stage_status(routed_request.path_family, routed_request.stage_ref);
  }
  return std::move(**maybe_reply_or);
}

absl::StatusOr<OwnerStageReply> PayloadTransportBroker::route_payload_ref_issuer_stage(
    const RoutedAuthorityRequest& routed_request,
    absl::Time now) {
  if (!routed_request.forwardable_evidence.has_value()) {
    return absl::FailedPreconditionError("payload_ref issuer route requires forwardable_evidence");
  }
  if (!is_payload_ref_issuer_stage(routed_request.path_family, routed_request.stage_ref)) {
    return undeclared_route_authority_stage_status(routed_request.path_family, routed_request.stage_ref);
  }
  auto raw_payload_ref_or = require_payload_ref_raw_issuer_evidence(*routed_request.forwardable_evidence);
  if (!raw_payload_ref_or.ok()) {
    return raw_payload_ref_or.status();
  }
  const std::string_view raw_payload_ref = *raw_payload_ref_or;
  const auto expected_direction =
      payload_ref_direction_from_label(routed_request.portable_credential.portable_constraint_claims.direction);
  auto local_resolved_or = resolve_local_payload_ref_record(
      raw_payload_ref,
      routed_request.portable_credential.portable_constraint_claims.artifact_id,
      now,
      expected_direction,
      routed_request.portable_credential.portable_constraint_claims.operation_id);
  if (!local_resolved_or.ok()) {
    return local_resolved_or.status();
  }
  auto issuer_authority_ref_or = derive_issuer_authority_ref(local_resolved_or->front_door_context);
  if (!issuer_authority_ref_or.ok()) {
    return issuer_authority_ref_or.status();
  }
  if (!authority_ref_equals(*issuer_authority_ref_or, routed_request.authority_ref)) {
    return absl::FailedPreconditionError("payload_ref issuer authority mismatch");
  }
  auto portable_credential_or = derive_payload_ref_portable_credential(local_resolved_or->front_door_context);
  if (!portable_credential_or.ok()) {
    return portable_credential_or.status();
  }
  if (!portable_credential_matches(*portable_credential_or, routed_request.portable_credential)) {
    return absl::FailedPreconditionError("payload_ref portable credential mismatches issuer-validated state");
  }
  return resolve_payload_ref_issuer_reply(raw_payload_ref, *local_resolved_or, /*remote_consumer=*/true);
}

absl::StatusOr<PayloadTransportBroker::LocalResolvedPayload> PayloadTransportBroker::resolve_local_payload_ref_record(
    std::string_view payload_ref,
    std::string_view expected_artifact_id,
    absl::Time now,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto metadata_or = inspect_payload_ref(payload_ref, now, /*require_not_expired=*/true);
  if (!metadata_or.ok()) {
    return metadata_or.status();
  }
  if (!expected_artifact_id.empty() && metadata_or->artifact_id.size() > 0 &&
      metadata_or->artifact_id != expected_artifact_id) {
    return absl::FailedPreconditionError("payload_ref artifact_id mismatch");
  }
  if (expected_direction != tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED &&
      metadata_or->direction != expected_direction) {
    return absl::FailedPreconditionError("payload_ref direction mismatch");
  }
  if (!metadata_or->operation_id.empty()) {
    if (expected_operation_id.empty()) {
      return absl::InvalidArgumentError("operation_id is required for payload_ref");
    }
    if (metadata_or->operation_id != expected_operation_id) {
      return absl::FailedPreconditionError("payload_ref operation_id mismatch");
    }
  }

  absl::MutexLock lock(&mu_);
  prune_locked(now);
  auto it = records_.find(metadata_or->payload_id);
  if (it == records_.end()) {
    return absl::NotFoundError("payload_ref is no longer valid");
  }
  if (it->second.metadata.expires_at <= now) {
    if (!it->second.body_handle.empty() && it->second.body_handle.unique_owner()) {
      (void)it->second.body_handle.retire();
    }
    records_.erase(it);
    return absl::PermissionDeniedError("payload_ref expired");
  }
  if (!expected_artifact_id.empty() && it->second.metadata.artifact_id != expected_artifact_id) {
    return absl::FailedPreconditionError("payload_ref artifact_id mismatch");
  }
  if (metadata_or->artifact_id.size() > 0 && metadata_or->artifact_id != it->second.metadata.artifact_id) {
    return absl::FailedPreconditionError("payload_ref metadata mismatch");
  }
  if (metadata_or->payload_size != 0 && metadata_or->payload_size != it->second.metadata.payload_size) {
    return absl::FailedPreconditionError("payload_ref payload_size mismatch");
  }
  if (!metadata_or->digest_alg.empty() && metadata_or->digest_alg != it->second.metadata.digest_alg) {
    return absl::FailedPreconditionError("payload_ref digest_alg mismatch");
  }
  if (!metadata_or->digest_hex.empty() && metadata_or->digest_hex != it->second.metadata.digest_hex) {
    return absl::FailedPreconditionError("payload_ref digest_hex mismatch");
  }
  if (metadata_or->direction != it->second.metadata.direction) {
    return absl::FailedPreconditionError("payload_ref direction mismatch");
  }
  if (metadata_or->operation_id != it->second.metadata.operation_id) {
    return absl::FailedPreconditionError("payload_ref operation_id mismatch");
  }
  std::uint64_t subject_generation = it->second.backing_instance_generation;
  if (subject_generation == 0 && !it->second.body_handle.empty()) {
    subject_generation = it->second.body_handle.binding_generation();
  }
  return LocalResolvedPayload{
      .metadata = it->second.metadata,
      .front_door_context = build_payload_ref_front_door_context(it->second.metadata, payload_ref, subject_generation),
      .payload = it->second.payload,
      .body_handle = it->second.body_handle,
      .descriptor = it->second.descriptor,
      .backing_identity = it->second.backing_identity,
      .backing_instance_generation = it->second.backing_instance_generation,
  };
}

absl::StatusOr<PayloadTransportBroker::LocalResolvedBatchPayload> PayloadTransportBroker::
    resolve_local_batch_payload_ref_record(
        std::string_view batch_payload_ref,
        absl::Time now,
        tensorcast::common::v1::PayloadRefDirection expected_direction,
        std::string_view expected_operation_id) {
  auto metadata_or = inspect_batch_payload_ref(batch_payload_ref, now, /*require_not_expired=*/true);
  if (!metadata_or.ok()) {
    return metadata_or.status();
  }
  if (expected_direction != tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED &&
      metadata_or->direction != expected_direction) {
    return absl::FailedPreconditionError("batch_payload_ref direction mismatch");
  }
  if (!metadata_or->operation_id.empty()) {
    if (expected_operation_id.empty()) {
      return absl::InvalidArgumentError("operation_id is required for batch_payload_ref");
    }
    if (metadata_or->operation_id != expected_operation_id) {
      return absl::FailedPreconditionError("batch_payload_ref operation_id mismatch");
    }
  }

  absl::MutexLock lock(&mu_);
  prune_locked(now);
  auto it = batch_records_.find(metadata_or->transport_id);
  if (it == batch_records_.end()) {
    return absl::NotFoundError("batch_payload_ref is no longer valid");
  }
  if (it->second.metadata.expires_at <= now) {
    batch_records_.erase(it);
    return absl::PermissionDeniedError("batch_payload_ref expired");
  }
  if (metadata_or->payload_size != 0 && metadata_or->payload_size != it->second.metadata.payload_size) {
    return absl::FailedPreconditionError("batch_payload_ref payload_size mismatch");
  }
  if (!metadata_or->manifest_digest_hex.empty() &&
      metadata_or->manifest_digest_hex != it->second.metadata.manifest_digest_hex) {
    return absl::FailedPreconditionError("batch_payload_ref manifest_digest mismatch");
  }
  if (!metadata_or->consumer_daemon_id.empty() &&
      metadata_or->consumer_daemon_id != it->second.metadata.consumer_daemon_id) {
    return absl::FailedPreconditionError("batch_payload_ref consumer_daemon_id mismatch");
  }
  if (metadata_or->direction != it->second.metadata.direction) {
    return absl::FailedPreconditionError("batch_payload_ref direction mismatch");
  }
  if (metadata_or->operation_id != it->second.metadata.operation_id) {
    return absl::FailedPreconditionError("batch_payload_ref operation_id mismatch");
  }
  return LocalResolvedBatchPayload{
      .metadata = it->second.metadata,
      .payload = it->second.payload,
      .communicator_export = it->second.communicator_export,
      .communicator_export_keepalives = it->second.communicator_export_keepalives,
  };
}

absl::StatusOr<PayloadTransportBroker::PayloadChunk> PayloadTransportBroker::read_local_payload_ref_chunk(
    std::string_view payload_ref,
    std::string_view expected_artifact_id,
    absl::Time now,
    std::uint64_t offset,
    std::uint64_t max_bytes,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto resolved_or = resolve_local_payload_ref_record(
      payload_ref, expected_artifact_id, now, expected_direction, expected_operation_id);
  if (!resolved_or.ok()) {
    return resolved_or.status();
  }
  if (offset > resolved_or->metadata.payload_size) {
    return absl::OutOfRangeError("payload_ref offset exceeds payload size");
  }
  const std::uint64_t remaining = resolved_or->metadata.payload_size - offset;
  const std::uint64_t chunk_bytes = std::min(max_bytes, remaining);
  std::string chunk;
  if (resolved_or->payload) {
    chunk.assign(resolved_or->payload->data() + offset, static_cast<std::size_t>(chunk_bytes));
  } else {
    auto chunk_or = resolved_or->body_handle.read_range(offset, static_cast<std::size_t>(chunk_bytes));
    if (!chunk_or.ok()) {
      return chunk_or.status();
    }
    chunk = std::move(*chunk_or);
  }
  return PayloadChunk{
      .metadata = resolved_or->metadata,
      .chunk = std::move(chunk),
      .eof = (offset + chunk_bytes) >= resolved_or->metadata.payload_size,
  };
}

absl::StatusOr<PayloadTransportBroker::BatchPayloadChunk> PayloadTransportBroker::read_local_batch_payload_ref_chunk(
    std::string_view batch_payload_ref,
    absl::Time now,
    std::uint64_t offset,
    std::uint64_t max_bytes,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto resolved_or =
      resolve_local_batch_payload_ref_record(batch_payload_ref, now, expected_direction, expected_operation_id);
  if (!resolved_or.ok()) {
    return resolved_or.status();
  }
  if (!resolved_or->payload || resolved_or->payload->empty()) {
    return absl::FailedPreconditionError("batch_payload_ref payload is unavailable");
  }
  if (offset > resolved_or->metadata.payload_size) {
    return absl::OutOfRangeError("batch_payload_ref offset exceeds payload size");
  }
  const std::uint64_t remaining = resolved_or->metadata.payload_size - offset;
  const std::uint64_t chunk_bytes = std::min(max_bytes, remaining);
  std::string chunk;
  chunk.assign(resolved_or->payload->data() + offset, static_cast<std::size_t>(chunk_bytes));
  return BatchPayloadChunk{
      .metadata = resolved_or->metadata,
      .chunk = std::move(chunk),
      .eof = (offset + chunk_bytes) >= resolved_or->metadata.payload_size,
  };
}

absl::StatusOr<PayloadTransportBroker::ResolvedPayload> PayloadTransportBroker::fetch_payload_ref(
    WorkerDirectoryCache& worker_directory_cache,
    absl::Time now,
    absl::Duration worker_directory_staleness_budget,
    std::string_view local_daemon_id,
    std::string_view payload_ref,
    std::string_view expected_artifact_id,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto loader_or = open_payload_ref_loader(
      worker_directory_cache,
      now,
      worker_directory_staleness_budget,
      local_daemon_id,
      payload_ref,
      expected_artifact_id,
      expected_direction,
      expected_operation_id);
  if (!loader_or.ok()) {
    return loader_or.status();
  }
  auto init_status = loader_or->loader->initialize();
  if (!init_status.ok()) {
    return init_status;
  }
  auto source_or = loader_or->loader->open_source();
  if (!source_or.ok()) {
    return source_or.status();
  }
  auto payload_or = read_source_fully(**source_or);
  if (!payload_or.ok()) {
    return payload_or.status();
  }
  if (!loader_or->metadata.digest_alg.empty() && loader_or->metadata.digest_alg != "sha256") {
    return absl::FailedPreconditionError("payload_ref digest_alg mismatch");
  }
  if (!loader_or->metadata.digest_hex.empty() && loader_or->metadata.digest_hex != compute_sha256_hex(*payload_or)) {
    return absl::DataLossError("payload_ref digest_hex mismatch");
  }
  return ResolvedPayload{
      .metadata = loader_or->metadata,
      .payload = std::move(*payload_or),
  };
}

absl::StatusOr<PayloadTransportBroker::ResolvedBatchPayload> PayloadTransportBroker::fetch_batch_payload_ref(
    WorkerDirectoryCache& worker_directory_cache,
    absl::Time now,
    absl::Duration worker_directory_staleness_budget,
    std::string_view local_daemon_id,
    std::string_view batch_payload_ref,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto metadata_or = inspect_batch_payload_ref(batch_payload_ref, now, /*require_not_expired=*/true);
  if (!metadata_or.ok()) {
    return metadata_or.status();
  }
  if (expected_direction != tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED &&
      metadata_or->direction != expected_direction) {
    return absl::FailedPreconditionError("batch_payload_ref direction mismatch");
  }
  if (!metadata_or->operation_id.empty()) {
    if (expected_operation_id.empty()) {
      return absl::InvalidArgumentError("operation_id is required for batch_payload_ref");
    }
    if (metadata_or->operation_id != expected_operation_id) {
      return absl::FailedPreconditionError("batch_payload_ref operation_id mismatch");
    }
  }

  if (metadata_or->issuer_daemon_id.empty() || metadata_or->issuer_daemon_id == local_daemon_id) {
    auto local_or =
        resolve_local_batch_payload_ref_record(batch_payload_ref, now, expected_direction, expected_operation_id);
    if (!local_or.ok()) {
      return local_or.status();
    }
    if (!local_or->payload || local_or->payload->empty()) {
      return absl::FailedPreconditionError("batch_payload_ref payload is unavailable");
    }
    return ResolvedBatchPayload{
        .metadata = local_or->metadata,
        .payload = local_or->payload,
        .remote = false,
    };
  }

  auto address_or = worker_directory_cache.resolve_daemon_address(
      metadata_or->issuer_daemon_id, now, worker_directory_staleness_budget);
  if (!address_or.ok()) {
    return address_or.status();
  }
  RemoteBatchPayloadRefSource source(
      RemoteBatchPayloadRefSource::Options{
          .metadata = *metadata_or,
          .batch_payload_ref = std::string(batch_payload_ref),
          .operation_id = std::string(expected_operation_id),
          .direction = expected_direction,
          .address = *address_or,
          .channel_credentials = options_.inter_daemon_channel_credentials,
          .max_chunk_bytes = options_.max_chunk_bytes,
          .fetch_deadline = options_.fetch_deadline,
      });
  auto payload_or = read_source_fully(source);
  if (!payload_or.ok()) {
    return payload_or.status();
  }
  if (metadata_or->payload_size != payload_or->size()) {
    return absl::DataLossError("batch_payload_ref payload_size mismatch");
  }
  return ResolvedBatchPayload{
      .metadata = *metadata_or,
      .payload = std::make_shared<const std::string>(std::move(*payload_or)),
      .remote = true,
  };
}

absl::StatusOr<PayloadTransportBroker::BatchPayloadSource> PayloadTransportBroker::
    open_batch_payload_communicator_source(
        WorkerDirectoryCache& worker_directory_cache,
        absl::Time now,
        absl::Duration worker_directory_staleness_budget,
        std::string_view local_daemon_id,
        const v2::BatchPayloadTransport& transport,
        tensorcast::common::v1::PayloadRefDirection expected_direction,
        std::string_view expected_operation_id) {
  if (!transport.has_communicator_source()) {
    return absl::InvalidArgumentError("batch transport communicator_source is required");
  }
  const absl::Time open_started_at = absl::Now();
  absl::Duration endpoint_resolve_elapsed{absl::ZeroDuration()};
  const auto& source = transport.communicator_source();
  auto metadata_or = inspect_batch_payload_ref(source.batch_payload_ref(), now, /*require_not_expired=*/true);
  if (!metadata_or.ok()) {
    return metadata_or.status();
  }
  if (expected_direction != tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED &&
      metadata_or->direction != expected_direction) {
    return absl::FailedPreconditionError("batch_payload_ref direction mismatch");
  }
  if (!metadata_or->operation_id.empty()) {
    if (expected_operation_id.empty()) {
      return absl::InvalidArgumentError("operation_id is required for batch_payload_ref");
    }
    if (metadata_or->operation_id != expected_operation_id) {
      return absl::FailedPreconditionError("batch_payload_ref operation_id mismatch");
    }
  }
  if (!metadata_or->consumer_daemon_id.empty() && metadata_or->consumer_daemon_id != local_daemon_id) {
    return absl::PermissionDeniedError("batch_payload_ref consumer_daemon_id mismatch");
  }
  auto manifest_status = validate_batch_payload_manifest(transport.manifest());
  if (!manifest_status.ok()) {
    return manifest_status;
  }
  if (!metadata_or->manifest_digest_hex.empty() &&
      metadata_or->manifest_digest_hex != compute_batch_manifest_digest_hex(transport.manifest())) {
    return absl::DataLossError("batch transport manifest digest mismatch");
  }
  auto communicator_status = validate_batch_communicator_source(transport, *metadata_or);
  if (!communicator_status.ok()) {
    return communicator_status;
  }

  if (metadata_or->issuer_daemon_id.empty() || metadata_or->issuer_daemon_id == local_daemon_id) {
    auto local_or = resolve_local_batch_payload_ref_record(
        source.batch_payload_ref(), now, expected_direction, expected_operation_id);
    if (!local_or.ok()) {
      return local_or.status();
    }
    if (local_or->payload && !local_or->payload->empty()) {
      VLOG(2) << "batch_payload_ref.communicator_open_summary"
              << " direction=" << payload_direction_label(expected_direction)
              << " operation_id=" << expected_operation_id << " transport_id=" << metadata_or->transport_id
              << " remote=false"
              << " producer_daemon_id=" << source.producer_daemon_id() << " consumer_daemon_id=" << local_daemon_id
              << " payload_bytes=" << metadata_or->payload_size << " memory_keys=" << source.remote_memory_keys_size();
      return BatchPayloadSource{
          .metadata = local_or->metadata,
          .source =
              std::shared_ptr<store::loader::SeekableSource>(std::make_shared<SharedStringSource>(local_or->payload)),
          .remote = false,
      };
    }
    if (!local_or->communicator_export.has_value()) {
      return absl::FailedPreconditionError("batch_payload_ref payload is unavailable for local communicator source");
    }
    auto export_total_or = export_registration_total_bytes(*local_or->communicator_export);
    if (!export_total_or.ok()) {
      return export_total_or.status();
    }
    if (*export_total_or != metadata_or->payload_size) {
      return absl::FailedPreconditionError("local communicator export payload_size mismatch");
    }
    std::vector<SharedSegmentedMemorySource::Segment> segments;
    segments.reserve(local_or->communicator_export->buffer_sizes.size());
    for (std::size_t index = 0; index < local_or->communicator_export->buffer_sizes.size(); ++index) {
      const auto address = local_or->communicator_export->buffer_addresses[index];
      const auto size_bytes = local_or->communicator_export->buffer_sizes[index];
      if (address == 0) {
        return absl::FailedPreconditionError("local communicator export buffer address is unavailable");
      }
      if (size_bytes == 0) {
        return absl::FailedPreconditionError("local communicator export buffer size must be > 0");
      }
      segments.push_back(
          SharedSegmentedMemorySource::Segment{
              .data = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(address)),
              .size_bytes = static_cast<std::uint64_t>(size_bytes),
          });
    }
    VLOG(2) << "batch_payload_ref.communicator_open_summary"
            << " direction=" << payload_direction_label(expected_direction) << " operation_id=" << expected_operation_id
            << " transport_id=" << metadata_or->transport_id << " remote=false"
            << " producer_daemon_id=" << source.producer_daemon_id() << " consumer_daemon_id=" << local_daemon_id
            << " payload_bytes=" << metadata_or->payload_size << " memory_keys=" << source.remote_memory_keys_size()
            << " realization=communicator_export";
    return BatchPayloadSource{
        .metadata = local_or->metadata,
        .source = std::shared_ptr<store::loader::SeekableSource>(std::make_shared<SharedSegmentedMemorySource>(
            std::move(segments), local_or->communicator_export_keepalives)),
        .remote = false,
    };
  }
  if (options_.comm_manager == nullptr || !options_.comm_manager->is_enabled()) {
    return absl::FailedPreconditionError("communicator is unavailable for batch transport");
  }

  std::string producer_host = source.producer_host();
  uint16_t producer_port = static_cast<uint16_t>(source.producer_port());
  std::string remote_endpoint_id = source.remote_endpoint_id();
  if (producer_host.empty() || producer_port == 0 || remote_endpoint_id.empty()) {
    const absl::Time resolve_started_at = absl::Now();
    auto producer_entry_or = worker_directory_cache.resolve_daemon_entry(
        metadata_or->issuer_daemon_id, now, worker_directory_staleness_budget);
    endpoint_resolve_elapsed += absl::Now() - resolve_started_at;
    if (!producer_entry_or.ok()) {
      return producer_entry_or.status();
    }
    if (producer_host.empty()) {
      producer_host = producer_entry_or->node_address;
    }
    if (producer_port == 0) {
      producer_port = static_cast<uint16_t>(producer_entry_or->p2p_port);
    }
    if (remote_endpoint_id.empty() && !producer_entry_or->node_id.empty()) {
      remote_endpoint_id = store::components::derive_endpoint_id(
          producer_entry_or->node_id, common::memory::MemoryLocation::CPU, /*device_id=*/0);
    }
  }
  if (producer_host.empty() || producer_port == 0) {
    return absl::FailedPreconditionError("communicator_source producer endpoint is incomplete");
  }

  std::string local_endpoint_id = source.local_endpoint_id_hint();
  std::string local_endpoint_id_mode = local_endpoint_id.empty() ? "missing" : "transport_hint";
  if (local_endpoint_id.empty() && options_.local_cpu_endpoint_id_provider) {
    local_endpoint_id = options_.local_cpu_endpoint_id_provider();
    if (!local_endpoint_id.empty()) {
      local_endpoint_id_mode = "local_provider";
    }
  }
  if (local_endpoint_id.empty()) {
    const absl::Time resolve_started_at = absl::Now();
    auto local_entry_or =
        worker_directory_cache.resolve_daemon_entry(local_daemon_id, now, worker_directory_staleness_budget);
    endpoint_resolve_elapsed += absl::Now() - resolve_started_at;
    if (local_entry_or.ok() && !local_entry_or->node_id.empty()) {
      local_endpoint_id = store::components::derive_endpoint_id(
          local_entry_or->node_id, common::memory::MemoryLocation::CPU, /*device_id=*/0);
      local_endpoint_id_mode = "worker_directory";
    }
  }

  std::vector<size_t> buffer_sizes;
  buffer_sizes.reserve(source.buffer_sizes_size());
  std::uint64_t total_size = 0;
  for (const auto buf_size : source.buffer_sizes()) {
    buffer_sizes.push_back(static_cast<size_t>(buf_size));
    total_size += buf_size;
  }
  if (total_size != source.total_payload_bytes()) {
    return absl::FailedPreconditionError("communicator_source buffer_sizes do not match total_payload_bytes");
  }
  store::loader::RemoteKeySource::Options source_opts{
      .comm_engine =
          gsl::not_null<std::shared_ptr<tensorcast::communicator::engine::Communicator>>{
              options_.comm_manager->get_shared_engine()},
      .memory_keys = std::vector<std::string>(source.remote_memory_keys().begin(), source.remote_memory_keys().end()),
      .buffer_sizes = std::move(buffer_sizes),
      .ip = producer_host,
      .port = producer_port,
      .local_endpoint_id = std::move(local_endpoint_id),
      .remote_endpoint_id = std::move(remote_endpoint_id),
      .routing_context = options_.comm_manager->routing_context(),
      .total_size = source.total_payload_bytes(),
      .request_budget = std::chrono::milliseconds(absl::ToInt64Milliseconds(options_.fetch_deadline)),
      .artifact_id = metadata_or->transport_id,
  };
  VLOG(2) << "batch_payload_ref.communicator_open_summary"
          << " direction=" << payload_direction_label(expected_direction) << " operation_id=" << expected_operation_id
          << " transport_id=" << metadata_or->transport_id << " remote=true"
          << " producer_daemon_id=" << source.producer_daemon_id() << " consumer_daemon_id=" << local_daemon_id
          << " producer_host=" << producer_host << " producer_port=" << producer_port
          << " local_endpoint_id_mode=" << local_endpoint_id_mode
          << " local_endpoint_id_present=" << !local_endpoint_id.empty()
          << " remote_endpoint_id_present=" << !remote_endpoint_id.empty()
          << " payload_bytes=" << metadata_or->payload_size << " memory_keys=" << source.remote_memory_keys_size()
          << " endpoint_resolve_ms=" << absl::ToDoubleMilliseconds(endpoint_resolve_elapsed)
          << " total_open_ms=" << absl::ToDoubleMilliseconds(absl::Now() - open_started_at);
  return BatchPayloadSource{
      .metadata = *metadata_or,
      .source =
          std::shared_ptr<store::loader::SeekableSource>(std::make_shared<store::loader::RemoteKeySource>(source_opts)),
      .remote = true,
  };
}

absl::StatusOr<PayloadTransportBroker::PayloadLoader> PayloadTransportBroker::open_payload_ref_loader(
    WorkerDirectoryCache& worker_directory_cache,
    absl::Time now,
    absl::Duration worker_directory_staleness_budget,
    std::string_view local_daemon_id,
    std::string_view payload_ref,
    std::string_view expected_artifact_id,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto metadata_or = inspect_payload_ref(payload_ref, now, /*require_not_expired=*/true);
  if (!metadata_or.ok()) {
    return metadata_or.status();
  }
  auto resolution_or = resolve_payload_ref_capability(
      worker_directory_cache,
      payload_ref,
      expected_artifact_id,
      now,
      worker_directory_staleness_budget,
      local_daemon_id,
      expected_direction,
      expected_operation_id);
  if (!resolution_or.ok()) {
    return resolution_or.status();
  }
  if (resolution_or->serving_capability.local) {
    if (resolution_or->body_capability.has_value()) {
      auto loader_or = resolution_or->body_capability->body_handle.make_loader();
      if (!loader_or.ok()) {
        return loader_or.status();
      }
      return PayloadLoader{
          .metadata = *metadata_or,
          .loader = std::move(*loader_or),
          .remote = false,
      };
    }
    if (!resolution_or->inline_payload) {
      return absl::FailedPreconditionError("payload_ref local record has no body_handle or payload");
    }
    return PayloadLoader{
        .metadata = *metadata_or,
        .loader = std::make_unique<store::InlineBufferLoader>(store::loading::InlineBufferSource{
            .data = std::shared_ptr<const void>(
                resolution_or->inline_payload, static_cast<const void*>(resolution_or->inline_payload->data())),
            .size_bytes = resolution_or->inline_payload->size(),
        }),
        .remote = false,
    };
  }

  auto address_or = worker_directory_cache.resolve_daemon_address(
      metadata_or->issuer_daemon_id, now, worker_directory_staleness_budget);
  if (!address_or.ok()) {
    return address_or.status();
  }
  return PayloadLoader{
      .metadata = *metadata_or,
      .loader = std::make_unique<PayloadRefLoader>(PayloadRefLoader::RemoteOptions{
          .source =
              RemotePayloadRefSource::Options{
                  .metadata = *metadata_or,
                  .payload_ref = std::string(payload_ref),
                  .artifact_id = std::string(expected_artifact_id),
                  .operation_id = std::string(expected_operation_id),
                  .direction = expected_direction,
                  .address = *address_or,
                  .channel_credentials = options_.inter_daemon_channel_credentials,
                  .max_chunk_bytes = options_.max_chunk_bytes,
                  .fetch_deadline = options_.fetch_deadline,
              },
      }),
      .remote = true,
  };
}

void PayloadTransportBroker::prune(absl::Time now) {
  absl::MutexLock lock(&mu_);
  prune_locked(now);
}

void PayloadTransportBroker::prune_locked(absl::Time now) {
  std::vector<std::string> expired;
  expired.reserve(records_.size());
  for (const auto& [payload_id, record] : records_) {
    if (record.metadata.expires_at <= now) {
      expired.push_back(payload_id);
    }
  }
  for (const auto& payload_id : expired) {
    auto it = records_.find(payload_id);
    if (it == records_.end()) {
      continue;
    }
    if (!it->second.body_handle.empty() && it->second.body_handle.unique_owner()) {
      (void)it->second.body_handle.retire();
    }
    records_.erase(it);
  }

  struct ExpiredBatchRef {
    std::string transport_id;
    std::optional<store::ExportRegistration> communicator_export;
    std::vector<std::shared_ptr<void>> communicator_export_keepalives;
    bool communicator_export_requires_unregister{false};
  };

  std::vector<ExpiredBatchRef> expired_batch_refs;
  expired_batch_refs.reserve(batch_records_.size());
  for (const auto& [transport_id, record] : batch_records_) {
    if (record.metadata.expires_at <= now) {
      expired_batch_refs.push_back(
          ExpiredBatchRef{
              .transport_id = transport_id,
              .communicator_export = record.communicator_export,
              .communicator_export_keepalives = record.communicator_export_keepalives,
              .communicator_export_requires_unregister = record.communicator_export_requires_unregister,
          });
    }
  }
  for (auto& expired : expired_batch_refs) {
    const auto& transport_id = expired.transport_id;
    batch_records_.erase(transport_id);
    if (expired.communicator_export_requires_unregister && expired.communicator_export.has_value() &&
        options_.comm_manager != nullptr) {
      for (const auto& tensor_key : expired.communicator_export->remote_memory_keys) {
        (void)options_.comm_manager->get_engine().unregister_tensor(tensor_key);
      }
    }
  }
}

std::string PayloadTransportBroker::mint_payload_id() {
  for (;;) {
    std::string raw;
    raw.resize(16);
    for (char& byte : raw) {
      byte = static_cast<char>(absl::Uniform<std::uint32_t>(bitgen_, 0u, 256u));
    }
    const std::string payload_id = absl::BytesToHexString(raw);
    if (!records_.contains(payload_id) && !batch_records_.contains(payload_id)) {
      return payload_id;
    }
  }
}

} // namespace tensorcast::daemon
