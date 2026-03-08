// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "core/common/selection_identity.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/contracts/loader.h"
#include "core/store/runtime/ingestion/artifact_truth.h"

namespace tensorcast::store::runtime::ingestion {

struct ArtifactLoweringIdentity {
  std::string logical_artifact_id;
  std::string physical_artifact_id;
  std::string request_id;
  std::string publish_context_id;
};

struct ArtifactLoweringPlan {
  ArtifactLoweringIdentity identity;
  DeviceKey target_device;
  std::unique_ptr<IArtifactLoader> source_loader;
  loader::ByteRangeMap byte_range_map;
  std::optional<tensorcast::common::SelectionIdentity> selection_identity;
  std::optional<ResolvedSourceDescriptor> resolved_source_descriptor;
  std::optional<SemanticLayoutIdentity> semantic_layout_identity;
  std::string canonical_index_json;
  std::uint64_t generation{1};
  loading::MaterializeHints hints;
  loading::MaterializationSource source_kind{loading::MaterializationSource::kUnspecified};
  std::optional<loading::IntoTargetLayout> into_target;
  std::optional<loading::ReplicaTarget> replica_target;
};

struct LowerToArtifactPlanRequest {
  ArtifactLoweringIdentity identity;
  DeviceKey target_device;
  std::unique_ptr<IArtifactLoader> source_loader;
  std::optional<tensorcast::common::SelectionIdentity> selection_identity;
  std::optional<ResolvedSourceDescriptor> resolved_source_descriptor;
  std::optional<SemanticLayoutIdentity> semantic_layout_identity;
  std::uint64_t expected_size_bytes{0};
  std::string canonical_index_json;
  std::uint64_t generation{1};
  loading::MaterializeHints hints;
  loading::MaterializationSource source_kind{loading::MaterializationSource::kUnspecified};
  std::optional<loading::IntoTargetLayout> into_target;
  std::optional<loading::ReplicaTarget> replica_target;
};

struct ArtifactLoweringResult {
  std::optional<loading::MaterializeIntoTargetResult> into_target_result;
  std::optional<loading::ReplicaHandle> replica_handle;
  std::optional<tensorcast::common::SelectionIdentity> selection_identity;
  std::optional<ResolvedSourceDescriptor> resolved_source_descriptor;
  std::optional<VerifiedContentDescriptor> verified_content_descriptor;
  std::optional<VerificationRecord> verification_record;
  std::optional<BackingIdentity> backing_identity;
};

inline absl::Status validate_artifact_lowering_plan(const ArtifactLoweringPlan& plan) {
  if (plan.identity.logical_artifact_id.empty()) {
    return absl::InvalidArgumentError("ArtifactLoweringPlan requires logical_artifact_id");
  }
  if (plan.source_loader == nullptr) {
    return absl::InvalidArgumentError("ArtifactLoweringPlan requires source_loader");
  }
  if (plan.byte_range_map.num_sources != 1) {
    return absl::InvalidArgumentError("ArtifactLoweringPlan requires byte_range_map.num_sources == 1");
  }
  if (!plan.resolved_source_descriptor.has_value()) {
    return absl::InvalidArgumentError("ArtifactLoweringPlan requires resolved_source_descriptor");
  }
  if (!plan.resolved_source_descriptor->size_is_authoritative) {
    return absl::InvalidArgumentError("ArtifactLoweringPlan requires authoritative resolved source size");
  }
  const bool has_into_target = plan.into_target.has_value();
  const bool has_replica_target = plan.replica_target.has_value();
  if (has_into_target == has_replica_target) {
    return absl::InvalidArgumentError(
        "ArtifactLoweringPlan must specify exactly one target: into_target or replica_target");
  }
  if (has_into_target && plan.target_device.type != DeviceType::GPU) {
    return absl::InvalidArgumentError("ArtifactLoweringPlan into_target execution requires a GPU target_device");
  }
  if (has_replica_target) {
    const DeviceKey replica_device = plan.replica_target->location.to_device_key();
    if (replica_device.type != plan.target_device.type || replica_device.ordinal != plan.target_device.ordinal ||
        replica_device.uuid != plan.target_device.uuid) {
      return absl::InvalidArgumentError("ArtifactLoweringPlan target_device must match replica_target.location");
    }
    if (plan.identity.physical_artifact_id.empty()) {
      return absl::InvalidArgumentError("ArtifactLoweringPlan replica execution requires physical_artifact_id");
    }
  }
  return absl::OkStatus();
}

inline absl::StatusOr<ResolvedSourceDescriptor> build_resolved_source_descriptor(
    IArtifactLoader& loader,
    loading::MaterializationSource source_kind,
    std::string_view source_id) {
  auto init_status = loader.initialize();
  if (!init_status.ok()) {
    return init_status;
  }
  auto size_or = loader.get_artifact_size();
  if (!size_or.ok()) {
    return size_or.status();
  }
  ResolvedSourceDescriptor descriptor;
  descriptor.source_id = std::string(source_id);
  descriptor.exact_size_bytes = *size_or;
  descriptor.size_is_authoritative = true;
  descriptor.resolved_locally = source_kind != loading::MaterializationSource::kP2P;
  descriptor.resolved_remotely = source_kind == loading::MaterializationSource::kP2P;
  descriptor.already_verified = false;
  descriptor.source_kind = source_kind;
  return descriptor;
}

inline absl::StatusOr<ArtifactLoweringPlan> lower_to_artifact_plan(LowerToArtifactPlanRequest request) {
  if (request.source_loader == nullptr) {
    return absl::InvalidArgumentError("lower_to_artifact_plan requires source_loader");
  }
  const std::string source_id = !request.identity.physical_artifact_id.empty()
      ? request.identity.physical_artifact_id
      : (!request.identity.request_id.empty() ? request.identity.request_id : request.identity.logical_artifact_id);
  auto resolved_source_or = request.resolved_source_descriptor.has_value()
      ? absl::StatusOr<ResolvedSourceDescriptor>(*request.resolved_source_descriptor)
      : build_resolved_source_descriptor(*request.source_loader, request.source_kind, source_id);
  if (!resolved_source_or.ok()) {
    return resolved_source_or.status();
  }
  const auto& resolved_source = *resolved_source_or;
  if (!resolved_source.size_is_authoritative) {
    return absl::InvalidArgumentError("lower_to_artifact_plan requires authoritative resolved source size");
  }
  if (request.expected_size_bytes != 0 && request.expected_size_bytes != resolved_source.exact_size_bytes) {
    return absl::FailedPreconditionError("lower_to_artifact_plan exact source size does not match expected size");
  }
  if (resolved_source.exact_size_bytes == 0) {
    return absl::InvalidArgumentError("lower_to_artifact_plan requires exact_size_bytes > 0");
  }

  ArtifactLoweringPlan plan;
  plan.identity = std::move(request.identity);
  plan.target_device = request.target_device;
  plan.source_loader = std::move(request.source_loader);
  plan.selection_identity = std::move(request.selection_identity);
  plan.resolved_source_descriptor = *resolved_source_or;
  plan.semantic_layout_identity = std::move(request.semantic_layout_identity);
  plan.byte_range_map = loading::build_identity_byte_range_map(resolved_source.exact_size_bytes);
  plan.canonical_index_json = request.canonical_index_json.empty()
      ? loading::build_synthetic_payload_canonical_index_json(resolved_source.exact_size_bytes)
      : std::move(request.canonical_index_json);
  plan.generation = request.generation;
  plan.hints = std::move(request.hints);
  plan.source_kind = request.source_kind;
  plan.into_target = std::move(request.into_target);
  plan.replica_target = std::move(request.replica_target);

  const auto validation_status = validate_artifact_lowering_plan(plan);
  if (!validation_status.ok()) {
    return validation_status;
  }
  return plan;
}

} // namespace tensorcast::store::runtime::ingestion
