// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/store_engine.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::serving_artifact_manifest {

inline constexpr std::string_view kPhase1ServingManifestTensorName = "__tensorcast_meta__.manifest_json";
inline constexpr std::string_view kPhase1ServingManifestRef = "tensor:__tensorcast_meta__.manifest_json";
inline constexpr std::string_view kPhase1ServingBuildDigestVersion = "tensorcast.serving_build_digest.v1";

struct ServingArtifactManifestRecord {
  int64_t schema_version{0};
  std::string artifact_kind;
  std::string framework_name;
  std::string adapter_version;
  std::string serving_abi_version;
  std::string representation_contract_hash;
  std::string serving_build_digest;
  std::optional<std::string> serving_build_digest_version;
  std::string tensor_schema_hash;
  uint64_t canonical_tensor_count{0};
  std::optional<std::string> serving_manifest_ref;
  std::optional<std::string> source_artifact_ref;
  std::string builder_mode;
  std::string build_pipeline_version;
  std::optional<std::string> logical_topology_json;
};

struct ServingArtifactPreflightRequest {
  std::string artifact_id;
  std::string canonical_index_json;
  std::optional<store::loading::DiskSource> disk_source;
  std::optional<std::string> serving_manifest_ref;
  std::optional<std::string> expected_representation_contract_hash;
  std::optional<std::string> expected_serving_build_digest;
  std::optional<std::string> expected_serving_build_digest_version;
  bool require_manifest{false};
};

struct ServingArtifactPreflightResult {
  bool serving_manifest_present{false};
  std::string serving_manifest_ref;
  std::string representation_contract_hash;
  std::string serving_build_digest;
  std::string tensor_schema_hash;
  uint64_t canonical_tensor_count{0};
  ServingArtifactManifestRecord manifest;
};

struct ServingManifestPayloadPreflightRequest {
  std::string canonical_index_json;
  std::string manifest_payload;
  std::optional<std::string> serving_manifest_ref;
  std::optional<std::string> expected_representation_contract_hash;
  std::optional<std::string> expected_serving_build_digest;
  std::optional<std::string> expected_serving_build_digest_version;
  bool require_manifest{false};
};

absl::StatusOr<std::string> parse_tensor_manifest_ref(std::string_view serving_manifest_ref);

absl::StatusOr<ServingArtifactManifestRecord> parse_serving_manifest_payload(std::string_view payload);

ServingArtifactPreflightRequest build_preflight_request(
    std::string artifact_id,
    std::string canonical_index_json,
    std::optional<store::loading::DiskSource> disk_source = std::nullopt,
    const v2::ServingArtifactRuntimePolicy* runtime_policy = nullptr);

absl::StatusOr<ServingArtifactPreflightResult> preflight_serving_artifact(
    store::StoreEngine* engine,
    const ServingArtifactPreflightRequest& request);

absl::StatusOr<ServingArtifactPreflightResult> preflight_serving_manifest_payload(
    const ServingManifestPayloadPreflightRequest& request);

} // namespace tensorcast::daemon::serving_artifact_manifest
