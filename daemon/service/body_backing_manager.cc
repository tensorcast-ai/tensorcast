// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/body_backing_manager.h"

#include <map>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "core/store/device_registry.h"
#include "core/store/runtime/ingestion/artifact_lowering_plan.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

namespace {

using tensorcast::common::memory::MemoryLocation;

const char* access_class_label(BodyAccessClass access_class) {
  switch (access_class) {
    case BodyAccessClass::kLocalGpuHot:
      return "local_gpu_hot";
    case BodyAccessClass::kTransientForward:
      return "transient_forward";
    case BodyAccessClass::kSmallObject:
      return "small_object";
    case BodyAccessClass::kHomeDefault:
    default:
      return "home_default";
  }
}

const char* residency_label(BodyPreferredResidency residency) {
  return residency == BodyPreferredResidency::kGpu ? "gpu" : "cpu";
}

const char* retention_label(BodyRetentionIntent retention) {
  return retention == BodyRetentionIntent::kRetained ? "retained" : "ephemeral";
}

const char* stable_requirement_label(BodyStableRetentionRequirement requirement) {
  switch (requirement) {
    case BodyStableRetentionRequirement::kPreferStable:
      return "prefer";
    case BodyStableRetentionRequirement::kRequireStable:
      return "require";
    case BodyStableRetentionRequirement::kNone:
    default:
      return "none";
  }
}

void record_body_backing_metrics(
    std::string_view mode,
    BodyAccessClass access_class,
    const BodyBackingIntent& intent,
    std::string_view outcome) {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_body_backing_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("mode", opentelemetry::common::AttributeValue(std::string(mode)));
    attrs.emplace("access_class", opentelemetry::common::AttributeValue(std::string(access_class_label(access_class))));
    attrs.emplace(
        "preferred_residency",
        opentelemetry::common::AttributeValue(std::string(residency_label(intent.preferred_residency))));
    attrs.emplace(
        "retention_intent",
        opentelemetry::common::AttributeValue(std::string(retention_label(intent.retention_intent))));
    attrs.emplace(
        "stable_requirement",
        opentelemetry::common::AttributeValue(
            std::string(stable_requirement_label(intent.stable_retention_requirement))));
    attrs.emplace("outcome", opentelemetry::common::AttributeValue(std::string(outcome)));
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

std::string build_body_backing_artifact_id(std::string_view artifact_id, const v2::PutIfAbsentInvariant& invariant) {
  std::string digest_hex = invariant.payload_digest_hex();
  absl::AsciiStrToLower(&digest_hex);
  return absl::StrCat("__tc_byte_body__:", artifact_id, ":", invariant.layout_id(), ":", digest_hex);
}

store::loading::MaterializeHints build_lowering_hints(std::string_view artifact_id, std::string_view operation_id) {
  store::loading::MaterializeHints hints;
  hints.artifact_id = std::string(artifact_id);
  if (!operation_id.empty()) {
    hints.transport_request_id = std::string(operation_id);
  }
  return hints;
}

store::DeviceKey resolve_target_device(const BodyBackingIntent& intent) {
  if (intent.preferred_residency == BodyPreferredResidency::kGpu) {
    return store::DeviceRegistry::instance().gpu_key(0);
  }
  return store::DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
}

store::loading::ReplicaTarget build_replica_target(const BodyBackingIntent& intent) {
  store::loading::ReplicaTarget target;
  if (intent.preferred_residency == BodyPreferredResidency::kGpu) {
    target.location.type = MemoryLocation::GPU;
    target.location.device_id = 0;
    return target;
  }
  target.location.type = MemoryLocation::CPU;
  target.location.device_id = -1;
  return target;
}

bool stable_retention_requested(
    const ResolvedStorePolicy& resolved_policy,
    const BodyBackingIntent& intent,
    const store::loading::ReplicaHandle& replica_handle) {
  if (replica_handle.key().device.type != DeviceType::CPU ||
      intent.stable_retention_requirement == BodyStableRetentionRequirement::kNone) {
    return false;
  }
  return stable_cache_policy_from_resolved(resolved_policy).has_value();
}

absl::StatusOr<BodyStableRetentionState> maybe_admit_stable_retention(
    store::StoreEngine& engine,
    const ResolvedStorePolicy& resolved_policy,
    const BodyBackingIntent& intent,
    const store::loading::ReplicaHandle& replica_handle) {
  if (!stable_retention_requested(resolved_policy, intent, replica_handle)) {
    return BodyStableRetentionState::kNotRequested;
  }

  const auto stable_policy_opt = stable_cache_policy_from_resolved(resolved_policy);
  if (!stable_policy_opt.has_value()) {
    return BodyStableRetentionState::kNotRequested;
  }

  auto admit_or = engine.admit_stable_cache_policy(replica_handle.key(), *stable_policy_opt);
  if (!admit_or.ok()) {
    if (stable_policy_opt->required) {
      return admit_or.status();
    }
    LOG(WARNING) << "body_backing: best-effort stable admission skipped for key=" << replica_handle.key() << ": "
                 << admit_or.status();
    return BodyStableRetentionState::kSkipped;
  }
  return admit_or->admitted ? BodyStableRetentionState::kHeld : BodyStableRetentionState::kSkipped;
}

BodyDescriptor make_body_descriptor(
    std::string_view physical_artifact_id,
    std::string_view layout_id,
    const store::runtime::ingestion::VerifiedArtifactContent& verified_content) {
  BodyDescriptor descriptor;
  descriptor.physical_artifact_id = std::string(physical_artifact_id);
  descriptor.layout_id = std::string(layout_id);
  descriptor.size_bytes = verified_content.size_bytes;
  descriptor.payload_digest_alg = normalize_body_digest_value(verified_content.payload_digest_alg);
  descriptor.payload_digest_hex = normalize_body_digest_value(verified_content.payload_digest_hex);
  descriptor.created_at = verified_content.verified_at;
  descriptor.verified_at = verified_content.verified_at;
  return descriptor;
}

bool invariant_matches_descriptor(const v2::PutIfAbsentInvariant& invariant, const BodyDescriptor& descriptor) {
  return invariant.layout_id() == descriptor.layout_id && invariant.byte_length() == descriptor.size_bytes &&
      normalize_body_digest_value(invariant.payload_digest_alg()) == descriptor.payload_digest_alg &&
      normalize_body_digest_value(invariant.payload_digest_hex()) == descriptor.payload_digest_hex;
}

BodyBackingObservation make_observation(
    const BodyDescriptor& descriptor,
    const store::StoreEngine::ReplicaBackingObservation& core_observation,
    BodyStableRetentionState stable_retention_state,
    absl::Time now) {
  BodyBackingObservation observation;
  observation.physical_artifact_id = descriptor.physical_artifact_id;
  observation.memory_location = core_observation.memory_location;
  observation.size_bytes = core_observation.size_bytes;
  observation.cpu_memfd_available = core_observation.cpu_memfd_available;
  observation.cuda_ipc_available = core_observation.cuda_ipc_available;
  observation.communicator_export_state = core_observation.remote_access_enabled
      ? BodyCommunicatorExportState::kExported
      : BodyCommunicatorExportState::kNotExported;
  observation.stable_retention_state = stable_retention_state;
  observation.observed_at = now;
  return observation;
}

} // namespace

BodyBackingManager::BodyBackingManager(store::StoreEngine& engine) : engine_(engine) {}

absl::StatusOr<ResolvedStorePolicy> BodyBackingManager::resolve_body_store_policy(
    BodyAccessClass access_class,
    RouteRole route_role,
    const std::optional<ResolvedStorePolicy>& resolved_store_policy) const {
  ResolvedStorePolicy resolved;
  if (resolved_store_policy.has_value()) {
    resolved = *resolved_store_policy;
  } else {
    auto default_policy_or = resolve_store_policy(nullptr);
    if (!default_policy_or.ok()) {
      return default_policy_or.status();
    }
    resolved = *default_policy_or;
  }
  const auto clear_local_stable = [&resolved]() {
    resolved.local_requirement = RequirementLevel::kNone;
    resolved.local_retention = store::components::StableRetentionPolicy::kBestEffort;
    resolved.local_ttl.reset();
  };

  if (route_role == RouteRole::kTransientForwarder) {
    clear_local_stable();
  }
  switch (access_class) {
    case BodyAccessClass::kLocalGpuHot:
    case BodyAccessClass::kTransientForward:
    case BodyAccessClass::kSmallObject:
      clear_local_stable();
      break;
    case BodyAccessClass::kHomeDefault:
    default:
      break;
  }
  return resolved;
}

BodyBackingIntent BodyBackingManager::classify_intent(
    BodyAccessClass access_class,
    const ResolvedStorePolicy& resolved_policy) const {
  const bool stable_requested = stable_cache_policy_from_resolved(resolved_policy).has_value();
  const auto stable_requirement = [&]() {
    if (!stable_requested) {
      return BodyStableRetentionRequirement::kNone;
    }
    return resolved_policy.local_requirement == RequirementLevel::kMust ? BodyStableRetentionRequirement::kRequireStable
                                                                        : BodyStableRetentionRequirement::kPreferStable;
  }();

  switch (access_class) {
    case BodyAccessClass::kLocalGpuHot:
      return BodyBackingIntent{
          .preferred_residency = BodyPreferredResidency::kGpu,
          .retention_intent = BodyRetentionIntent::kRetained,
          .stable_retention_requirement = stable_requirement,
          .sharing_intent = BodySharingIntent::kLocalReadMostly,
      };
    case BodyAccessClass::kTransientForward:
      return BodyBackingIntent{
          .preferred_residency = BodyPreferredResidency::kCpu,
          .retention_intent = BodyRetentionIntent::kEphemeral,
          .stable_retention_requirement = BodyStableRetentionRequirement::kNone,
          .sharing_intent = BodySharingIntent::kRemoteShareable,
      };
    case BodyAccessClass::kSmallObject:
      return BodyBackingIntent{
          .preferred_residency = BodyPreferredResidency::kCpu,
          .retention_intent = BodyRetentionIntent::kEphemeral,
          .stable_retention_requirement = BodyStableRetentionRequirement::kNone,
          .sharing_intent = BodySharingIntent::kPrivateLocal,
      };
    case BodyAccessClass::kHomeDefault:
    default:
      return BodyBackingIntent{
          .preferred_residency = BodyPreferredResidency::kCpu,
          .retention_intent = BodyRetentionIntent::kRetained,
          .stable_retention_requirement = stable_requirement,
          .sharing_intent = BodySharingIntent::kRemoteShareable,
      };
  }
}

absl::StatusOr<BodyBackingManager::StageResult> BodyBackingManager::stage_body(StageRequest request) const {
  if (request.artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required for body staging");
  }
  if (request.invariant.layout_id().empty()) {
    return absl::InvalidArgumentError("invariant.layout_id is required for body staging");
  }
  if (request.loader == nullptr) {
    return absl::InvalidArgumentError("loader is required for body staging");
  }

  auto resolved_policy_or =
      resolve_body_store_policy(request.access_class, request.route_role, request.resolved_store_policy);
  if (!resolved_policy_or.ok()) {
    return resolved_policy_or.status();
  }
  const BodyBackingIntent intent = classify_intent(request.access_class, *resolved_policy_or);
  store::runtime::ingestion::ArtifactLoweringPlan plan;
  plan.identity.logical_artifact_id = request.artifact_id;
  plan.identity.physical_artifact_id = build_body_backing_artifact_id(request.artifact_id, request.invariant);
  plan.identity.request_id = request.operation_id;
  plan.target_device = resolve_target_device(intent);
  plan.source_loader = std::move(request.loader);
  plan.byte_range_map = store::loading::build_identity_byte_range_map(request.invariant.byte_length());
  plan.canonical_index_json =
      store::loading::build_synthetic_payload_canonical_index_json(request.invariant.byte_length());
  plan.generation = 1;
  plan.hints = build_lowering_hints(request.artifact_id, request.operation_id);
  plan.source_kind = request.source_kind;
  plan.replica_target = build_replica_target(intent);

  const std::string physical_artifact_id = plan.identity.physical_artifact_id;
  auto result_or = engine_.execute_artifact_lowering_plan(std::move(plan));
  if (!result_or.ok()) {
    record_body_backing_metrics("stage", request.access_class, intent, "error");
    return result_or.status();
  }
  if (!result_or->replica_handle.has_value()) {
    record_body_backing_metrics("stage", request.access_class, intent, "missing_replica");
    return absl::InternalError("body staging did not return a replica handle");
  }
  if (!result_or->verified_content.has_value()) {
    record_body_backing_metrics("stage", request.access_class, intent, "missing_descriptor");
    return absl::InternalError("body staging did not return verified content metadata");
  }

  store::loading::ReplicaHandle replica_handle = std::move(*result_or->replica_handle);
  const BodyDescriptor descriptor =
      make_body_descriptor(physical_artifact_id, request.invariant.layout_id(), *result_or->verified_content);

  auto stable_state_or = maybe_admit_stable_retention(engine_, *resolved_policy_or, intent, replica_handle);
  if (!stable_state_or.ok() && intent.stable_retention_requirement == BodyStableRetentionRequirement::kRequireStable) {
    const auto retire_status = engine_.retire_replica_status(replica_handle.key());
    if (!retire_status.ok()) {
      record_body_backing_metrics("stage", request.access_class, intent, "retire_error");
      return retire_status;
    }
    record_body_backing_metrics("stage", request.access_class, intent, "stable_required_error");
    return stable_state_or.status();
  }
  auto core_observation_or = engine_.inspect_replica_backing(replica_handle.key());
  if (!core_observation_or.ok()) {
    (void)engine_.retire_replica_status(replica_handle.key());
    record_body_backing_metrics("stage", request.access_class, intent, "observe_error");
    return core_observation_or.status();
  }
  const BodyBackingObservation observation = make_observation(
      descriptor,
      *core_observation_or,
      stable_state_or.ok() ? *stable_state_or : BodyStableRetentionState::kSkipped,
      absl::Now());
  const auto replica_key = replica_handle.key();
  auto body_handle_or = BodyHandle::create(engine_, std::move(replica_handle));
  if (!body_handle_or.ok()) {
    (void)engine_.retire_replica_status(replica_key);
    record_body_backing_metrics("stage", request.access_class, intent, "handle_error");
    return body_handle_or.status();
  }
  record_body_backing_metrics("stage", request.access_class, intent, "ok");
  return StageResult{
      .descriptor = descriptor,
      .observation = observation,
      .body_handle = std::move(*body_handle_or),
  };
}

absl::StatusOr<std::optional<BodyBackingManager::StageResult>> BodyBackingManager::try_reuse_body(
    ReuseRequest request) const {
  if (request.artifact_id.empty() || request.body_handle.empty()) {
    return std::nullopt;
  }

  auto resolved_policy_or =
      resolve_body_store_policy(request.access_class, request.route_role, request.resolved_store_policy);
  if (!resolved_policy_or.ok()) {
    return resolved_policy_or.status();
  }
  const BodyBackingIntent intent = classify_intent(request.access_class, *resolved_policy_or);
  if (intent.stable_retention_requirement == BodyStableRetentionRequirement::kRequireStable) {
    record_body_backing_metrics("reuse", request.access_class, intent, "skipped_required_stable");
    return std::nullopt;
  }

  const BodyDescriptor descriptor = normalized_body_descriptor(std::move(request.descriptor));
  if (!invariant_matches_descriptor(request.invariant, descriptor)) {
    record_body_backing_metrics("reuse", request.access_class, intent, "descriptor_mismatch");
    return absl::InvalidArgumentError("descriptor does not match requested invariant for body reuse");
  }
  if (descriptor.physical_artifact_id != build_body_backing_artifact_id(request.artifact_id, request.invariant)) {
    record_body_backing_metrics("reuse", request.access_class, intent, "skipped_identity_mismatch");
    return std::nullopt;
  }

  auto core_observation_or = engine_.inspect_replica_backing(request.body_handle.replica_handle().key());
  if (!core_observation_or.ok()) {
    record_body_backing_metrics("reuse", request.access_class, intent, "observe_error");
    return core_observation_or.status();
  }

  const BodyStableRetentionState stable_state =
      intent.stable_retention_requirement == BodyStableRetentionRequirement::kNone
      ? BodyStableRetentionState::kNotRequested
      : BodyStableRetentionState::kSkipped;
  record_body_backing_metrics("reuse", request.access_class, intent, "ok");
  return std::optional<StageResult>(StageResult{
      .descriptor = descriptor,
      .observation = make_observation(descriptor, *core_observation_or, stable_state, absl::Now()),
      .body_handle = request.body_handle,
  });
}

} // namespace tensorcast::daemon
