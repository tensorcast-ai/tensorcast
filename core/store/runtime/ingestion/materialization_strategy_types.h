// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/contracts/representation_contract.h"
#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"

namespace tensorcast::store::runtime::ingestion::strategy {

// The selected source may expose canonical bytes or an already materialized
// view/mapped byte-space. This is independent from the request's view_id:
// mapped-target requests can carry a target byte-space identity while still
// reconstructing the result from canonical/disk fallback.
enum class SourceByteSpace : std::uint8_t {
  kCanonical = 0,
  kView = 1,
};

enum class SourceBoundPolicy : std::uint8_t {
  kRequirePureCollective = 0,
  kCollectiveFirst = 1,
  kDisableCollective = 2,
};

enum class SourceBoundExecutionMode : std::uint8_t {
  kPureCollective = 0,
  kCollectiveFirstMixed = 1,
  kGenericOnly = 2,
  kLocalTypedOnly = 3,
  kRejected = 4,
  kLocalMappedTyped = 5,
};

struct SourceBoundSourceFacts {
  bool disk_source_available{false};
  bool disk_source_is_safetensors{false};
};

inline std::string_view source_bound_execution_mode_name(SourceBoundExecutionMode mode) {
  switch (mode) {
    case SourceBoundExecutionMode::kPureCollective:
      return "pure_collective";
    case SourceBoundExecutionMode::kCollectiveFirstMixed:
      return "collective_first_mixed";
    case SourceBoundExecutionMode::kLocalMappedTyped:
      return "local_mapped_typed";
    case SourceBoundExecutionMode::kLocalTypedOnly:
      return "local_typed_only";
    case SourceBoundExecutionMode::kRejected:
      return "reject";
    case SourceBoundExecutionMode::kGenericOnly:
    default:
      return "generic_only";
  }
}

struct ResolvedSourceBinding {
  loading::MaterializationSource source{loading::MaterializationSource::kDisk};
  SourceByteSpace source_byte_space{SourceByteSpace::kCanonical};
  bool source_layout_available{false};
  bool direct_write_capable{false};
  bool collective_eligible{false};
};

struct SourceBoundExecutionPlanSummary {
  std::string execution_plan_kind;
  uint64_t planned_collective_candidate_bytes{0};
  uint64_t planned_collective_admitted_bytes{0};
  uint64_t planned_local_typed_bytes{0};
  uint64_t planned_non_admitted_typed_bytes{0};
  uint64_t planned_generic_residual_bytes{0};
  uint64_t compatibility_lowered_bytes{0};
  absl::flat_hash_map<std::string, uint64_t> planner_reject_reason_buckets;
  std::string planner_version;
  std::string plan_hash;
  uint64_t estimated_collective_peak_temporary_bytes{0};
  uint64_t estimated_collective_batch_bytes{0};
  uint64_t estimated_collective_dedup_saving_bytes{0};
  bool collective_lane_eligible{false};
  bool strict_pure_collective_eligible{false};
};

struct SourceBoundLoweringStats {
  uint64_t total_dst_tensors{0};
  uint64_t compatible_candidates{0};
  uint64_t compatible_bytes{0};
  uint64_t concat_candidates{0};
  uint64_t concat_bytes{0};
  uint64_t rejected_mixed_src_or_dim{0};
  uint64_t rejected_mixed_src_or_dim_bytes{0};
  uint64_t rejected_non_contiguous{0};
  uint64_t rejected_non_contiguous_bytes{0};
  uint64_t rejected_unsupported_distribution{0};
  uint64_t rejected_unsupported_distribution_bytes{0};
};

struct ResolvedMaterializationPlan {
  std::string artifact_id;
  uint64_t generation{0};
  std::optional<loading::VariantIdentity> variant;
  std::string canonical_index_json;
  loading::IntoTargetLayout target_layout;
  std::optional<materialization::contracts::RepresentationTransformContract> representation_transform_contract;
  std::optional<materialization::contracts::RepresentationWorkPlan> representation_work_plan;
};

struct SourceBoundLoweringArtifacts {
  std::optional<loader::ByteRangeMap> executor_generic_data_map;
  std::optional<loader::ByteRangeMap> collective_data_map;
  SourceBoundLoweringStats lowering_stats;
};

struct SourceBoundLanePlan {
  SourceBoundExecutionMode mode{SourceBoundExecutionMode::kGenericOnly};
  loader::ByteRangeMap collective_lane_map;
  loader::ByteRangeMap generic_backend_map;
  loader::ByteRangeMap true_residual_map;
  uint64_t local_typed_bytes{0};
  uint64_t local_pad_bytes{0};
  uint64_t local_fill_bytes{0};
  uint64_t deferred_typed_bytes{0};
  bool local_mapped_typed_selected{false};
  bool require_collective_success{false};
  std::string selection_reason;
  absl::flat_hash_map<std::string, uint64_t> reject_reason_buckets;
};

struct SourceBoundStrategyPlan {
  SourceBoundPolicy policy{SourceBoundPolicy::kCollectiveFirst};
  SourceBoundLanePlan lane_plan;
  SourceBoundExecutionPlanSummary summary;
};

struct PreparedSourceBoundExecutionPlan {
  ResolvedMaterializationPlan resolved_plan;
  std::optional<SourceBoundLoweringArtifacts> lowering_artifacts;
  std::optional<SourceBoundStrategyPlan> strategy_plan;
};

enum class ExecutionStrategyExecutor : std::uint8_t {
  kGenericByteRange = 0,
  kTensorBatchedLocal = 1,
  kOwnerFileCollective = 2,
};

inline std::string_view execution_strategy_executor_name(ExecutionStrategyExecutor executor) {
  switch (executor) {
    case ExecutionStrategyExecutor::kTensorBatchedLocal:
      return "TensorBatchedLocalExecutor";
    case ExecutionStrategyExecutor::kOwnerFileCollective:
      return "OwnerFileCollectiveExecutor";
    case ExecutionStrategyExecutor::kGenericByteRange:
    default:
      return "GenericByteRangeExecutor";
  }
}

struct ExecutionStrategyCostEstimate {
  uint64_t requested_source_bytes{0};
  uint64_t unique_source_bytes{0};
  uint64_t estimated_peak_temporary_bytes{0};
  uint64_t estimated_batch_bytes{0};
  double estimated_owner_skew_ratio{1.0};
  uint64_t estimated_dedup_saving_bytes{0};
};

struct ExecutionStrategyCandidate {
  ExecutionStrategyExecutor executor{ExecutionStrategyExecutor::kGenericByteRange};
  bool eligible{false};
  std::string reason;
  ExecutionStrategyCostEstimate estimate;
};

struct ExecutionEnvironmentFacts {
  loading::ExecutionTopologyContext execution_topology;
  loading::MaterializationSource source{loading::MaterializationSource::kUnspecified};
  bool target_is_gpu{false};
  bool source_layout_available{false};
  bool coordinator_available{false};
  bool has_complete_metadata{false};
  bool requires_server_transform{false};
  bool allow_mixed_execution{false};
  uint64_t requested_bytes{0};
  uint64_t committed_bytes{0};
  uint64_t residual_bytes{0};

  uint64_t owner_file_collective_peak_bytes_budget{0};
  uint64_t owner_file_collective_batch_bytes{0};
  uint64_t owner_file_collective_dim1_staging_bytes{0};
  uint32_t owner_file_collective_max_inflight_batches{0};
  bool owner_file_collective_shared_fs_only{true};
  double owner_file_collective_max_owner_skew_ratio{0.0};
  uint64_t owner_file_collective_min_dedup_saving_bytes{0};
  std::chrono::milliseconds owner_file_collective_group_assemble_timeout{0};
  bool owner_file_collective_allow_mixed_residual{false};
  uint32_t owner_file_collective_planner_cache_entries{0};
};

struct ExecutionStrategyPlan {
  ExecutionStrategyExecutor executor{ExecutionStrategyExecutor::kGenericByteRange};
  std::string selection_reason;
  ExecutionEnvironmentFacts environment;
  std::vector<ExecutionStrategyCandidate> candidates;
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  std::optional<materialization::contracts::RepresentationWorkPlan> representation_work_plan;
  std::optional<loading::CollectiveLoadGroupHint> collective_load_group;
  std::optional<SourceBoundLanePlan> source_bound_lane_plan;
};

struct CollectiveExecutionMetrics {
  uint64_t unique_source_bytes{0};
  uint64_t peer_transfer_bytes{0};
  uint64_t peak_temporary_bytes{0};
  uint64_t batch_count{0};
  uint64_t dedup_saving_bytes{0};
};

struct ExecutionCommitReport {
  loading::MaterializationSource source{loading::MaterializationSource::kDisk};
  uint64_t requested_bytes{0};
  uint64_t committed_bytes{0};
  uint64_t fallback_bytes{0};
  uint64_t residual_bytes{0};
  uint64_t actual_collective_committed_bytes{0};
  uint64_t actual_local_typed_bytes{0};
  uint64_t actual_generic_backend_bytes{0};
  CollectiveExecutionMetrics collective_metrics;
  std::string collective_skip_reason;
  bool collective_handled{false};
  bool direct_write_supported{false};
  bool source_ordered{false};
  std::string dominant_executor;
  std::string selection_reason;
};

} // namespace tensorcast::store::runtime::ingestion::strategy
