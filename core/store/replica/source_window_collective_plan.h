// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/contracts/representation_contract.h"
#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"
#include "core/store/runtime/ingestion/materialization_strategy_types.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::replica {

struct TargetStorageSpan {
  uint64_t base_offset{0};
  uint64_t length{0};
  std::uint8_t* base_ptr{nullptr};
};

struct SourceWindowCollectiveConfig {
  bool enabled{false};
  runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode selection_mode{
      runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kDryRun};
  uint64_t window_bytes{512ULL * 1024ULL * 1024ULL};
  uint64_t max_gap_bytes{256ULL * 1024ULL};
  uint64_t max_window_amplification_x1000{2000};
  uint64_t max_plan_read_amplification_x1000{1200};
  uint64_t max_scatter_ops_per_window{4096};
  uint64_t peak_bytes_budget{4ULL * 1024ULL * 1024ULL * 1024ULL};
  uint64_t min_rank_read_saving_bytes{512ULL * 1024ULL * 1024ULL};
  uint64_t max_peer_to_read_ratio_x1000{8000};
  uint64_t min_routed_peer_saving_bytes{64ULL * 1024ULL * 1024ULL};
  runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode distribution_mode{
      runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kAuto};
  bool allow_mixed_residual{false};
};

SourceWindowCollectiveConfig source_window_collective_config_from_strategy(
    const StoreEngineOptions::MaterializationStrategyConfig& strategy_config);

struct SourceWindowCollectiveMemberInput {
  uint32_t rank{0};
  int device_id{-1};
  std::shared_ptr<const materialization::contracts::RepresentationWorkPlan> work_plan_ref;
  materialization::contracts::RepresentationWorkPlan work_plan;
  std::shared_ptr<const loading::IntoTargetLayout> target_layout_ref;
  loading::IntoTargetLayout target_layout;
  std::vector<TargetStorageSpan> storage_spans;
  std::optional<loading::SourceWindowPreparedRealizationFacts> prepared_realization;
};

struct SourceWindowCollectiveGroupInput {
  loading::CollectiveLoadGroupHint group;
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  std::string source_index_digest;
  std::vector<SourceWindowCollectiveMemberInput> members;
  SourceWindowCollectiveConfig config;
};

struct SourceWindowCollectiveConsumerSpan {
  uint32_t rank{0};
  uint32_t source_index{0};
  uint32_t storage_index{0};
  uint64_t source_window_start{0};
  uint64_t source_window_end{0};
  uint64_t source_offset{0};
  uint64_t target_offset{0};
  uint64_t length{0};
  uint64_t row_count{1};
  uint64_t row_bytes{0};
  uint64_t source_stride_bytes{0};
  uint64_t target_stride_bytes{0};
};

struct SourceWindowCollectiveWindow {
  uint32_t source_index{0};
  uint32_t owner_rank{0};
  runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode distribution_mode{
      runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kAuto};
  uint64_t start{0};
  uint64_t end{0};
  uint64_t unique_payload_bytes{0};
  std::vector<SourceWindowCollectiveConsumerSpan> consumer_spans;
};

struct SourceWindowCollectivePlan {
  loading::CollectiveLoadGroupHint group;
  runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode distribution_mode{
      runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kAuto};
  std::vector<SourceWindowCollectiveWindow> windows;
  std::vector<uint64_t> rank_read_bytes;
  uint64_t residual_bytes{0};
  runtime::ingestion::strategy::SourceWindowCollectiveCandidateSummary summary;
  std::string plan_hash;
};

struct SourceWindowTensorStagedCopySummary {
  bool feasible{false};
  std::string reject_reason;
  uint64_t rank_count{0};
  uint64_t source_fragment_count{0};
  uint64_t destination_tensor_count{0};
  uint64_t source_tensor_count{0};
  uint64_t eligible_bytes{0};
  uint64_t ineligible_bytes{0};
  uint64_t raw_copy_ops{0};
  uint64_t tensor_staged_copy_ops{0};
  uint64_t linear_copy_ops{0};
  uint64_t copy_2d_ops{0};
  uint64_t max_tensor_staged_copy_ops_per_rank{0};
  uint64_t estimated_op_reduction_x1000{0};
};

struct SourceWindowBatchedScatterSummary {
  bool feasible{false};
  std::string reject_reason;
  uint64_t rank_count{0};
  uint64_t window_count{0};
  uint64_t runtime_chunk_bytes{0};
  uint64_t estimated_runtime_chunk_count{0};
  uint64_t consumer_span_count{0};
  uint64_t full_window_all_gather_windows{0};
  uint64_t consumer_routed_windows{0};
  uint64_t local_only_windows{0};
  uint64_t target_write_bytes{0};
  uint64_t estimated_current_copy_launches{0};
  uint64_t estimated_current_scatter_launches{0};
  uint64_t estimated_current_pack_launches{0};
  uint64_t estimated_current_linear_copy_launches{0};
  uint64_t estimated_current_copy_2d_launches{0};
  uint64_t batched_total_copy_launches{0};
  uint64_t batched_scatter_launches{0};
  uint64_t batched_pack_launches{0};
  uint64_t max_descriptors_per_batched_scatter{0};
  uint64_t max_descriptors_per_batched_pack{0};
  uint64_t estimated_copy_launch_reduction_x1000{0};
};

absl::StatusOr<SourceWindowCollectivePlan> build_source_window_collective_plan(
    const SourceWindowCollectiveGroupInput& input);

runtime::ingestion::strategy::SourceWindowCollectiveCandidateSummary summarize_source_window_collective(
    const SourceWindowCollectiveGroupInput& input);

absl::StatusOr<SourceWindowTensorStagedCopySummary> summarize_source_window_tensor_staged_copy(
    const SourceWindowCollectiveGroupInput& input);

absl::StatusOr<SourceWindowBatchedScatterSummary> summarize_source_window_batched_scatter(
    const SourceWindowCollectivePlan& plan,
    uint64_t runtime_chunk_bytes);

} // namespace tensorcast::store::replica
