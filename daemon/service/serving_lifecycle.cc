// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/serving_lifecycle.h"

#include <map>
#include <utility>

#include "absl/status/status.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

namespace {

const char* subject_kind_label(ServingCapabilitySubjectKind kind) {
  switch (kind) {
    case ServingCapabilitySubjectKind::kCopiedPayload:
      return "copied_payload";
    case ServingCapabilitySubjectKind::kPolicyBackedPath:
      return "policy_backed_path";
    case ServingCapabilitySubjectKind::kBacking:
    default:
      return "backing";
  }
}

const char* owner_kind_label(LifecycleOwnerKind kind) {
  switch (kind) {
    case LifecycleOwnerKind::kPayloadRefToken:
      return "payload_ref_token";
    case LifecycleOwnerKind::kRetentionHandle:
      return "retention_handle";
    case LifecycleOwnerKind::kPersistenceTask:
      return "persistence_task";
    case LifecycleOwnerKind::kInlineCopyWindow:
    default:
      return "inline_copy_window";
  }
}

void record_serving_capability_event(
    std::string_view operation,
    ServingCapabilitySubjectKind subject_kind,
    LifecycleOwnerKind owner_kind) {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_serving_capability_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("operation", opentelemetry::common::AttributeValue(std::string(operation)));
    attrs.emplace("subject_kind", opentelemetry::common::AttributeValue(std::string(subject_kind_label(subject_kind))));
    attrs.emplace("owner_kind", opentelemetry::common::AttributeValue(std::string(owner_kind_label(owner_kind))));
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

absl::Status validate_mint_request(const MintServingCapabilityRequest& request) {
  if (request.capability_id.empty()) {
    return absl::InvalidArgumentError("capability_id is required");
  }
  if (request.lifecycle_owner_ref.owner_id.empty()) {
    return absl::InvalidArgumentError("lifecycle_owner_ref.owner_id is required");
  }
  switch (request.subject_kind) {
    case ServingCapabilitySubjectKind::kBacking:
      if (!request.backing_identity.has_value()) {
        return absl::InvalidArgumentError("backing capability requires backing_identity");
      }
      if (!store::runtime::ingestion::backing_identity_matches_replica_key(*request.backing_identity)) {
        return absl::InvalidArgumentError("backing_identity must match replica_key.artifact_id");
      }
      if (request.backing_instance_generation == 0) {
        return absl::InvalidArgumentError("backing capability requires backing_instance_generation");
      }
      break;
    case ServingCapabilitySubjectKind::kPolicyBackedPath:
      if (!request.policy_visibility_ref.has_value()) {
        return absl::InvalidArgumentError("policy-backed capability requires policy_visibility_ref");
      }
      if (request.policy_visibility_ref->path_id.empty() || request.policy_visibility_ref->control_ref.empty()) {
        return absl::InvalidArgumentError("policy_visibility_ref is incomplete");
      }
      break;
    case ServingCapabilitySubjectKind::kCopiedPayload:
    default:
      break;
  }
  return absl::OkStatus();
}

} // namespace

absl::StatusOr<ServingCapability> mint_serving_capability(MintServingCapabilityRequest request) {
  auto request_status = validate_mint_request(request);
  if (!request_status.ok()) {
    return request_status;
  }
  ServingCapability capability;
  capability.capability_id = std::move(request.capability_id);
  capability.expires_at = request.expires_at;
  capability.mode = request.mode;
  capability.local = request.local;
  capability.subject_kind = request.subject_kind;
  capability.lifecycle_owner_ref = std::move(request.lifecycle_owner_ref);
  capability.backing_identity = std::move(request.backing_identity);
  capability.backing_instance_generation = request.backing_instance_generation;
  capability.policy_visibility_ref = std::move(request.policy_visibility_ref);
  record_serving_capability_event("mint", capability.subject_kind, capability.lifecycle_owner_ref.owner_kind);
  return capability;
}

absl::Status release_serving_capability(const ServingCapability& capability) {
  record_serving_capability_event("release", capability.subject_kind, capability.lifecycle_owner_ref.owner_kind);
  return absl::OkStatus();
}

} // namespace tensorcast::daemon
