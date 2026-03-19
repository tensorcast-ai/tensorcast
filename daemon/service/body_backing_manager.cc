// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/body_backing_manager.h"

#include <algorithm>
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
    const store::runtime::ingestion::BackingIdentity& backing_identity,
    std::string_view layout_id,
    const store::runtime::ingestion::VerifiedContentDescriptor& verified_content_descriptor,
    const store::runtime::ingestion::VerificationRecord& verification_record) {
  BodyDescriptor descriptor;
  descriptor.physical_artifact_id = backing_identity.physical_artifact_id;
  descriptor.layout_id = std::string(layout_id);
  descriptor.size_bytes = verified_content_descriptor.content_identity.logical_size_bytes;
  descriptor.payload_digest_alg = normalize_body_digest_value(verified_content_descriptor.content_identity.digest_alg);
  descriptor.payload_digest_hex = normalize_body_digest_value(
      store::runtime::ingestion::content_digest_bytes_to_hex(
          verified_content_descriptor.content_identity.digest_bytes));
  descriptor.created_at = verification_record.verified_at;
  descriptor.verified_at = verification_record.verified_at;
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
    BodyRouteRole route_role,
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

  if (route_role == BodyRouteRole::kTransientForwarder) {
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

BodyPlacementContext BodyBackingManager::normalize_placement_context(
    BodyAccessClass access_class,
    BodyRouteRole route_role,
    std::uint64_t size_bytes) const {
  BodyPlacementContext context;
  context.route_role = route_role;
  context.size_bytes = size_bytes;
  context.expected_fanout = route_role == BodyRouteRole::kTransientForwarder ? 2 : 1;
  context.locality = route_role == BodyRouteRole::kTransientForwarder ? BodyConsumerLocality::kRemoteOrMixed
                                                                      : BodyConsumerLocality::kLocalOnly;
  switch (access_class) {
    case BodyAccessClass::kLocalGpuHot:
      context.access_pattern = BodyAccessPattern::kLocalGpuHot;
      context.locality = BodyConsumerLocality::kLocalOnly;
      break;
    case BodyAccessClass::kTransientForward:
      context.access_pattern = BodyAccessPattern::kTransientForward;
      context.locality = BodyConsumerLocality::kRemoteOrMixed;
      context.expected_fanout = std::max<std::uint32_t>(context.expected_fanout, 2);
      break;
    case BodyAccessClass::kSmallObject:
      context.access_pattern = BodyAccessPattern::kSmallObject;
      context.locality = BodyConsumerLocality::kLocalOnly;
      break;
    case BodyAccessClass::kHomeDefault:
    default:
      context.access_pattern = BodyAccessPattern::kDefault;
      break;
  }
  return context;
}

BodyBackingIntent BodyBackingManager::classify_intent(
    const BodyPlacementContext& context,
    const ResolvedStorePolicy& resolved_policy) const {
  const bool stable_requested = stable_cache_policy_from_resolved(resolved_policy).has_value();
  const auto stable_requirement = [&]() {
    if (!stable_requested) {
      return BodyStableRetentionRequirement::kNone;
    }
    return resolved_policy.local_requirement == RequirementLevel::kMust ? BodyStableRetentionRequirement::kRequireStable
                                                                        : BodyStableRetentionRequirement::kPreferStable;
  }();

  BodyBackingIntent intent;
  intent.preferred_residency = BodyPreferredResidency::kCpu;
  intent.retention_intent = BodyRetentionIntent::kRetained;
  intent.stable_retention_requirement = stable_requirement;
  intent.sharing_intent = context.locality == BodyConsumerLocality::kRemoteOrMixed
      ? BodySharingIntent::kRemoteShareable
      : BodySharingIntent::kLocalReadMostly;

  if (context.route_role == BodyRouteRole::kTransientForwarder ||
      context.access_pattern == BodyAccessPattern::kTransientForward) {
    intent.retention_intent = BodyRetentionIntent::kEphemeral;
    intent.stable_retention_requirement = BodyStableRetentionRequirement::kNone;
    intent.preferred_residency = BodyPreferredResidency::kCpu;
    intent.sharing_intent = BodySharingIntent::kRemoteShareable;
    return intent;
  }

  if (context.access_pattern == BodyAccessPattern::kLocalGpuHot) {
    intent.preferred_residency = BodyPreferredResidency::kGpu;
    intent.sharing_intent = BodySharingIntent::kLocalReadMostly;
    return intent;
  }

  if (context.access_pattern == BodyAccessPattern::kSmallObject) {
    intent.retention_intent = BodyRetentionIntent::kEphemeral;
    intent.stable_retention_requirement = BodyStableRetentionRequirement::kNone;
    intent.sharing_intent = BodySharingIntent::kPrivateLocal;
    return intent;
  }

  return intent;
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
  const BodyPlacementContext context =
      normalize_placement_context(request.access_class, request.route_role, request.invariant.byte_length());
  const BodyBackingIntent intent = classify_intent(context, *resolved_policy_or);
  auto plan_or = store::runtime::ingestion::lower_to_artifact_plan(
      store::runtime::ingestion::LowerToArtifactPlanRequest{
          .identity =
              store::runtime::ingestion::ArtifactLoweringIdentity{
                  .logical_artifact_id = request.artifact_id,
                  .physical_artifact_id = build_body_backing_artifact_id(request.artifact_id, request.invariant),
                  .request_id = request.operation_id,
              },
          .target_device = resolve_target_device(intent),
          .source_loader = std::move(request.loader),
          .selection_identity =
              tensorcast::common::SelectionIdentity{
                  .artifact_id = request.artifact_id,
                  .logical_layout_hash = tensorcast::common::compute_byte_artifact_logical_layout_hash_bytes(),
                  .selection_hash = tensorcast::common::compute_byte_artifact_selection_hash_bytes(),
              },
          .semantic_layout_identity =
              store::runtime::ingestion::SemanticLayoutIdentity{
                  .kind = store::runtime::ingestion::SemanticLayoutKind::kNamedLayoutId,
                  .value = request.invariant.layout_id(),
              },
          .expected_size_bytes = request.invariant.byte_length(),
          .generation = 1,
          .hints = build_lowering_hints(request.artifact_id, request.operation_id),
          .source_kind = request.source_kind,
          .replica_target = build_replica_target(intent),
      });
  if (!plan_or.ok()) {
    record_body_backing_metrics("stage", request.access_class, intent, "lowering_error");
    return plan_or.status();
  }
  store::runtime::ingestion::ArtifactLoweringPlan plan = std::move(*plan_or);

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
  if (!result_or->verified_content_descriptor.has_value() || !result_or->verification_record.has_value() ||
      !result_or->backing_identity.has_value()) {
    record_body_backing_metrics("stage", request.access_class, intent, "missing_descriptor");
    return absl::InternalError("body staging did not return shared truth metadata");
  }

  store::loading::ReplicaHandle replica_handle = std::move(*result_or->replica_handle);
  store::runtime::ingestion::BackingIdentity backing_identity = *result_or->backing_identity;
  if (backing_identity.physical_artifact_id.empty()) {
    backing_identity.physical_artifact_id = physical_artifact_id;
  }
  if (!store::runtime::ingestion::backing_identity_matches_replica_key(backing_identity)) {
    (void)engine_.retire_replica_status(replica_handle.key());
    record_body_backing_metrics("stage", request.access_class, intent, "backing_identity_mismatch");
    return absl::InternalError("body staging returned inconsistent backing identity");
  }
  const store::runtime::ingestion::VerifiedContentDescriptor verified_content_descriptor =
      *result_or->verified_content_descriptor;
  store::runtime::ingestion::VerifiedContentDescriptor staged_verified_content_descriptor = verified_content_descriptor;
  staged_verified_content_descriptor.content_identity.semantic_layout_identity.kind =
      store::runtime::ingestion::SemanticLayoutKind::kNamedLayoutId;
  staged_verified_content_descriptor.content_identity.semantic_layout_identity.value = request.invariant.layout_id();
  store::runtime::ingestion::VerificationRecord verification_record = *result_or->verification_record;
  verification_record.layout_proof_kind = store::runtime::ingestion::LayoutProofKind::kNamedLayoutId;
  verification_record.layout_proof_value = request.invariant.layout_id();
  const BodyDescriptor descriptor = make_body_descriptor(
      backing_identity, request.invariant.layout_id(), staged_verified_content_descriptor, verification_record);

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
      .verified_content_descriptor = std::move(staged_verified_content_descriptor),
      .verification_record = verification_record,
      .backing_identity = std::move(backing_identity),
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
  const BodyPlacementContext context =
      normalize_placement_context(request.access_class, request.route_role, request.descriptor.size_bytes);
  const BodyBackingIntent intent = classify_intent(context, *resolved_policy_or);
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
      .verified_content_descriptor = body_descriptor_to_verified_content_descriptor(descriptor),
      .verification_record = body_descriptor_to_verification_record(descriptor),
      .backing_identity = body_descriptor_to_backing_identity(descriptor, request.body_handle),
  });
}

} // namespace tensorcast::daemon
