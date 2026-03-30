// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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

struct ResolvedSourceBinding {
  loading::MaterializationSource source{loading::MaterializationSource::kDisk};
  SourceByteSpace source_byte_space{SourceByteSpace::kCanonical};
  bool source_layout_available{false};
  bool direct_write_capable{false};
  bool collective_eligible{false};
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
};

struct ExecutionCommitReport {
  loading::MaterializationSource source{loading::MaterializationSource::kDisk};
  uint64_t requested_bytes{0};
  uint64_t committed_bytes{0};
  uint64_t fallback_bytes{0};
  uint64_t residual_bytes{0};
  bool collective_handled{false};
  bool direct_write_supported{false};
  bool source_ordered{false};
  std::string dominant_executor;
  std::string selection_reason;
};

} // namespace tensorcast::store::runtime::ingestion::strategy
