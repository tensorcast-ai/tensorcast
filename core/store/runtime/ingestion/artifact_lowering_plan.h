// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/contracts/loader.h"

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
  std::string canonical_index_json;
  std::uint64_t generation{1};
  loading::MaterializeHints hints;
  loading::MaterializationSource source_kind{loading::MaterializationSource::kUnspecified};
  std::optional<loading::IntoTargetLayout> into_target;
  std::optional<loading::ReplicaTarget> replica_target;
};

struct VerifiedArtifactContent {
  std::uint64_t size_bytes{0};
  std::string payload_digest_alg;
  std::string payload_digest_hex;
  absl::Time verified_at{absl::InfinitePast()};
};

struct ArtifactLoweringResult {
  std::optional<loading::MaterializeIntoTargetResult> into_target_result;
  std::optional<loading::ReplicaHandle> replica_handle;
  std::optional<VerifiedArtifactContent> verified_content;
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

} // namespace tensorcast::store::runtime::ingestion
