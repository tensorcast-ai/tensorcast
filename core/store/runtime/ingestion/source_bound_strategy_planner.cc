// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/ingestion/source_bound_strategy_planner.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"

namespace tensorcast::store::runtime::ingestion::strategy {
namespace {

using RepresentationWorkItem = materialization::contracts::RepresentationWorkItem;
using RepresentationWorkItemKind = materialization::contracts::RepresentationWorkItemKind;
using RepresentationWorkPlan = materialization::contracts::RepresentationWorkPlan;
using WorkPartitionKind = materialization::contracts::WorkPartitionKind;
using MaterializationStrategyConfig = StoreEngineOptions::MaterializationStrategyConfig;

uint64_t byte_range_map_covered_bytes(const loader::ByteRangeMap& map) {
  uint64_t total = 0;
  for (const auto& segment : map.segments) {
    total += segment.length;
  }
  return total;
}

uint64_t missing_coverage_bytes(uint64_t required_bytes, uint64_t covered_bytes) {
  return covered_bytes >= required_bytes ? 0 : required_bytes - covered_bytes;
}

SourceWindowCollectiveSelectionMode source_window_selection_mode_from_config(
    MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode mode) {
  switch (mode) {
    case MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto:
      return SourceWindowCollectiveSelectionMode::kAuto;
    case MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kStrict:
      return SourceWindowCollectiveSelectionMode::kStrict;
    case MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kDryRun:
    default:
      return SourceWindowCollectiveSelectionMode::kDryRun;
  }
}

SourceWindowCollectiveDistributionMode source_window_distribution_mode_from_config(
    MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode mode) {
  switch (mode) {
    case MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kFullWindowAllGather:
      return SourceWindowCollectiveDistributionMode::kFullWindowAllGather;
    case MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kConsumerRouted:
      return SourceWindowCollectiveDistributionMode::kConsumerRouted;
    case MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kHybridWindow:
      return SourceWindowCollectiveDistributionMode::kHybridWindow;
    case MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kLocalOnly:
      return SourceWindowCollectiveDistributionMode::kLocalOnly;
    case MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kAuto:
    default:
      return SourceWindowCollectiveDistributionMode::kAuto;
  }
}

bool typed_work_item_has_collective_source_overlap(const RepresentationWorkItem& item) {
  return item.kind == RepresentationWorkItemKind::kTensorCopy && item.partition_kind == WorkPartitionKind::kReplicated;
}

void add_reject_reason_bytes(
    absl::flat_hash_map<std::string, uint64_t>* buckets,
    std::string_view reason,
    uint64_t bytes);

void add_deferred_typed_work(
    const RepresentationWorkItem& item,
    SourceBoundExecutionPlanSummary* summary,
    SourceBoundLanePlan* lane_plan) {
  summary->planned_non_admitted_typed_bytes += item.committed_bytes;
  lane_plan->deferred_typed_bytes += item.committed_bytes;
  add_reject_reason_bytes(
      &summary->planner_reject_reason_buckets, "typed_work_without_source_overlap", item.committed_bytes);
}

void add_reject_reason_bytes(
    absl::flat_hash_map<std::string, uint64_t>* buckets,
    std::string_view reason,
    uint64_t bytes) {
  if (buckets == nullptr || reason.empty() || bytes == 0) {
    return;
  }
  (*buckets)[std::string(reason)] += bytes;
}

absl::StatusOr<std::string> compute_source_bound_plan_hash(const SourceBoundExecutionPlanSummary& summary) {
  std::vector<std::pair<std::string, uint64_t>> buckets(
      summary.planner_reject_reason_buckets.begin(), summary.planner_reject_reason_buckets.end());
  std::sort(buckets.begin(), buckets.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  std::string payload = absl::StrCat(
      summary.execution_plan_kind,
      "\n",
      summary.planned_collective_candidate_bytes,
      "\n",
      summary.planned_collective_admitted_bytes,
      "\n",
      summary.planned_local_typed_bytes,
      "\n",
      summary.planned_non_admitted_typed_bytes,
      "\n",
      summary.planned_generic_residual_bytes,
      "\n",
      summary.collective_lowered_bytes,
      "\n",
      summary.estimated_collective_peak_temporary_bytes,
      "\n",
      summary.estimated_collective_batch_bytes,
      "\n",
      summary.estimated_collective_dedup_saving_bytes,
      "\n",
      summary.collective_lane_eligible ? "1" : "0",
      "\n",
      summary.strict_pure_collective_eligible ? "1" : "0",
      "\n",
      summary.source_window_collective_candidate ? "1" : "0",
      "\n",
      source_window_collective_selection_mode_name(summary.source_window_selection_mode),
      "\n",
      summary.source_window_group_reject_reason,
      "\n",
      summary.source_window_candidate_summary.source_window_group_disk_read_bytes,
      "\n",
      summary.source_window_candidate_summary.source_window_rank_read_bytes_max,
      "\n",
      summary.source_window_candidate_summary.source_window_local_rank_read_bytes_max,
      "\n",
      summary.source_window_candidate_summary.source_window_rank_read_saving_bytes,
      "\n",
      summary.source_window_candidate_summary.source_window_unique_payload_bytes,
      "\n",
      summary.source_window_candidate_summary.source_window_target_write_bytes,
      "\n",
      summary.source_window_candidate_summary.source_window_read_amplification_x1000);
  for (const auto& [reason, bytes] : buckets) {
    absl::StrAppend(&payload, "\n", reason, "=", bytes);
  }
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return common::multibase_multihash_sha256(digest);
}

void summarize_work_plan(
    const RepresentationWorkPlan& work_plan,
    SourceBoundExecutionPlanSummary* summary,
    SourceBoundLanePlan* lane_plan) {
  if (summary == nullptr || lane_plan == nullptr) {
    return;
  }
  for (const auto& item : work_plan.items) {
    switch (item.kind) {
      case RepresentationWorkItemKind::kTensorCopy:
        if (typed_work_item_has_collective_source_overlap(item)) {
          summary->planned_collective_candidate_bytes += item.committed_bytes;
        } else {
          add_deferred_typed_work(item, summary, lane_plan);
        }
        break;
      case RepresentationWorkItemKind::kExpertDim0Concat:
      case RepresentationWorkItemKind::kConcatAssemble: {
        add_deferred_typed_work(item, summary, lane_plan);
      } break;
      case RepresentationWorkItemKind::kConstFill:
      case RepresentationWorkItemKind::kScalarBroadcastFill:
        summary->planned_local_typed_bytes += item.committed_bytes;
        lane_plan->local_typed_bytes += item.committed_bytes;
        lane_plan->local_fill_bytes += item.committed_bytes;
        break;
      case RepresentationWorkItemKind::kPadFill:
        summary->planned_local_typed_bytes += item.committed_bytes;
        lane_plan->local_typed_bytes += item.committed_bytes;
        lane_plan->local_pad_bytes += item.committed_bytes;
        break;
      case RepresentationWorkItemKind::kResidualByteRange:
        summary->planned_generic_residual_bytes += item.committed_bytes;
        break;
    }
  }
}

uint64_t saturating_multiply_u64(uint64_t lhs, uint64_t rhs) {
  if (lhs == 0 || rhs == 0) {
    return 0;
  }
  if (lhs > std::numeric_limits<uint64_t>::max() / rhs) {
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs * rhs;
}

uint64_t estimate_collective_dedup_saving_bytes(const RepresentationWorkPlan& work_plan, uint32_t world_size) {
  if (world_size <= 1) {
    return 0;
  }
  uint64_t saving = 0;
  const uint64_t peer_count = static_cast<uint64_t>(world_size - 1);
  for (const auto& item : work_plan.items) {
    if (typed_work_item_has_collective_source_overlap(item)) {
      const uint64_t item_saving = saturating_multiply_u64(item.committed_bytes, peer_count);
      saving = std::numeric_limits<uint64_t>::max() - saving < item_saving ? std::numeric_limits<uint64_t>::max()
                                                                           : saving + item_saving;
    }
  }
  return saving;
}

} // namespace

absl::StatusOr<SourceBoundStrategyPlan> build_source_bound_execution_strategy_plan(
    const ResolvedMaterializationPlan& resolved_plan,
    const std::optional<SourceBoundLoweringArtifacts>& lowering_artifacts,
    SourceBoundPolicy policy,
    const StoreEngineOptions::MaterializationStrategyConfig& strategy_config,
    const loading::ExecutionTopologyContext& execution_topology,
    SourceBoundSourceFacts source_facts) {
  using ExecutorPreference = StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference;
  SourceBoundStrategyPlan strategy_plan;
  strategy_plan.policy = policy;
  strategy_plan.summary.planner_version = "source_bound_collective_first.v4";
  strategy_plan.lane_plan.require_collective_success = policy == SourceBoundPolicy::kRequirePureCollective;
  strategy_plan.summary.source_window_selection_mode =
      source_window_selection_mode_from_config(strategy_config.source_window_collective_selection_mode);
  strategy_plan.summary.source_window_candidate_summary.selection_mode =
      strategy_plan.summary.source_window_selection_mode;
  strategy_plan.summary.source_window_candidate_summary.distribution_mode =
      source_window_distribution_mode_from_config(strategy_config.source_window_collective_distribution_mode);
  strategy_plan.summary.source_window_candidate_summary.pre_admission_reason = "source_window_disabled";
  const bool disk_source_available = source_facts.disk_source_available;
  const bool safetensors_disk_source_available = disk_source_available && source_facts.disk_source_is_safetensors;

  if (lowering_artifacts.has_value()) {
    if (lowering_artifacts->collective_data_map.has_value()) {
      strategy_plan.summary.collective_lowered_bytes =
          byte_range_map_covered_bytes(*lowering_artifacts->collective_data_map);
      strategy_plan.lane_plan.collective_lane_map = *lowering_artifacts->collective_data_map;
    }
    if (lowering_artifacts->executor_generic_data_map.has_value()) {
      strategy_plan.lane_plan.generic_backend_map = *lowering_artifacts->executor_generic_data_map;
      strategy_plan.lane_plan.generic_backend_map_coverage_only =
          lowering_artifacts->executor_generic_data_map_coverage_only;
    }
  }

  if (resolved_plan.representation_work_plan.has_value()) {
    summarize_work_plan(*resolved_plan.representation_work_plan, &strategy_plan.summary, &strategy_plan.lane_plan);
    strategy_plan.lane_plan.true_residual_map = resolved_plan.representation_work_plan->residual_fallback_map;
  }

  const bool has_collective_group = execution_topology.collective_load_group.has_value();
  const uint32_t world_size = has_collective_group ? execution_topology.collective_load_group->world_size : 0;
  const bool shared_source_proven = execution_topology.source_locality == loading::SourceLocalityHint::kSharedSource ||
      execution_topology.source_sharing_domain.has_value();
  const bool collective_policy_enabled = policy != SourceBoundPolicy::kDisableCollective;
  strategy_plan.summary.estimated_collective_peak_temporary_bytes =
      strategy_plan.summary.planned_collective_candidate_bytes;
  strategy_plan.summary.estimated_collective_batch_bytes = strategy_config.owner_file_collective_batch_bytes;
  if (resolved_plan.representation_work_plan.has_value()) {
    strategy_plan.summary.estimated_collective_dedup_saving_bytes =
        estimate_collective_dedup_saving_bytes(*resolved_plan.representation_work_plan, world_size);
  }

  auto reject_source_window = [&](std::string reason) {
    strategy_plan.summary.source_window_collective_candidate = false;
    strategy_plan.summary.source_window_group_reject_reason = reason;
    strategy_plan.summary.source_window_candidate_summary.candidate = false;
    strategy_plan.summary.source_window_candidate_summary.group_final_admitted = false;
    strategy_plan.summary.source_window_candidate_summary.pre_admission_reason = reason;
    strategy_plan.summary.source_window_candidate_summary.group_reject_reason = reason;
  };
  auto admit_source_window_pre_candidate = [&]() {
    strategy_plan.summary.source_window_collective_candidate = true;
    strategy_plan.summary.source_window_group_reject_reason.clear();
    auto& source_window = strategy_plan.summary.source_window_candidate_summary;
    source_window.candidate = true;
    source_window.group_final_admitted = false;
    source_window.pre_admission_reason = "candidate_pending_group_final_admission";
    source_window.group_reject_reason.clear();
    source_window.source_window_target_write_bytes = resolved_plan.representation_work_plan.has_value()
        ? resolved_plan.representation_work_plan->committed_bytes
        : 0;
    source_window.source_window_residual_bytes = resolved_plan.representation_work_plan.has_value()
        ? resolved_plan.representation_work_plan->residual_fallback_map.total_bytes
        : 0;
  };
  const bool source_window_policy_enabled = policy != SourceBoundPolicy::kDisableCollective;
  if (!strategy_config.enable_source_window_collective) {
    reject_source_window("source_window_disabled");
  } else if (!source_window_policy_enabled) {
    reject_source_window("source_window_policy_disabled");
  } else if (!disk_source_available) {
    reject_source_window("disk_source_unavailable");
  } else if (!source_facts.disk_source_is_safetensors) {
    reject_source_window("non_safetensors_source");
  } else if (!has_collective_group || world_size <= 1) {
    reject_source_window("collective_group_missing");
  } else if (!resolved_plan.representation_work_plan.has_value()) {
    reject_source_window("representation_work_plan_missing");
  } else if (resolved_plan.target_layout.storages.empty()) {
    reject_source_window("target_layout_missing");
  } else if (
      resolved_plan.representation_work_plan->residual_fallback_map.total_bytes > 0 &&
      !strategy_config.source_window_collective_allow_mixed_residual) {
    reject_source_window("residual_requires_mixed_source_window");
  } else {
    admit_source_window_pre_candidate();
  }

  if (!collective_policy_enabled) {
    add_reject_reason_bytes(
        &strategy_plan.summary.planner_reject_reason_buckets,
        "collective_policy_disabled",
        strategy_plan.summary.planned_collective_candidate_bytes);
  } else if (!strategy_config.enable_owner_file_collective) {
    add_reject_reason_bytes(
        &strategy_plan.summary.planner_reject_reason_buckets,
        "collective_strategy_disabled",
        strategy_plan.summary.planned_collective_candidate_bytes);
  } else if (!disk_source_available) {
    add_reject_reason_bytes(
        &strategy_plan.summary.planner_reject_reason_buckets,
        "disk_source_unavailable",
        strategy_plan.summary.planned_collective_candidate_bytes);
  } else if (!source_facts.disk_source_is_safetensors) {
    add_reject_reason_bytes(
        &strategy_plan.summary.planner_reject_reason_buckets,
        "non_safetensors_source",
        strategy_plan.summary.planned_collective_candidate_bytes);
  } else if (!has_collective_group || world_size <= 1) {
    add_reject_reason_bytes(
        &strategy_plan.summary.planner_reject_reason_buckets,
        "collective_group_missing",
        strategy_plan.summary.planned_collective_candidate_bytes);
  } else if (
      strategy_config.owner_file_collective_shared_fs_only &&
      execution_topology.source_locality == loading::SourceLocalityHint::kHostLocal) {
    add_reject_reason_bytes(
        &strategy_plan.summary.planner_reject_reason_buckets,
        "source_locality_host_local",
        strategy_plan.summary.planned_collective_candidate_bytes);
  } else if (strategy_config.owner_file_collective_shared_fs_only && !shared_source_proven) {
    add_reject_reason_bytes(
        &strategy_plan.summary.planner_reject_reason_buckets,
        "shared_source_unproven",
        strategy_plan.summary.planned_collective_candidate_bytes);
  } else if (
      strategy_plan.summary.planned_collective_candidate_bytes > 0 &&
      strategy_plan.summary.estimated_collective_dedup_saving_bytes <
          strategy_config.owner_file_collective_min_dedup_saving_bytes) {
    add_reject_reason_bytes(
        &strategy_plan.summary.planner_reject_reason_buckets,
        "collective_source_overlap_below_threshold",
        strategy_plan.summary.planned_collective_candidate_bytes);
  }

  if (strategy_plan.summary.planned_non_admitted_typed_bytes > 0) {
    add_reject_reason_bytes(
        &strategy_plan.summary.planner_reject_reason_buckets,
        "typed_work_not_collective_admitted",
        strategy_plan.summary.planned_non_admitted_typed_bytes);
  }
  if (strategy_plan.summary.planned_generic_residual_bytes > 0 &&
      !strategy_config.owner_file_collective_allow_mixed_residual) {
    add_reject_reason_bytes(
        &strategy_plan.summary.planner_reject_reason_buckets,
        "mixed_generic_residual_policy_disabled",
        strategy_plan.summary.planned_generic_residual_bytes);
  }
  const uint64_t local_mapped_candidate_bytes = strategy_plan.summary.planned_non_admitted_typed_bytes;
  const bool typed_work_prefers_local_mapped =
      local_mapped_candidate_bytes > 0 && strategy_config.executor_preference != ExecutorPreference::kGenericByteRange;
  const bool local_mapped_typed_available = strategy_config.enable_tensor_aware_mapped_executor &&
      strategy_config.allow_mixed_execution && safetensors_disk_source_available;
  if (typed_work_prefers_local_mapped && !local_mapped_typed_available) {
    std::string_view reason = "local_mapped_typed_executor_unavailable";
    if (!disk_source_available) {
      reason = "disk_source_unavailable";
    } else if (!source_facts.disk_source_is_safetensors) {
      reason = "non_safetensors_source";
    }
    add_reject_reason_bytes(&strategy_plan.summary.planner_reject_reason_buckets, reason, local_mapped_candidate_bytes);
  }

  const bool base_collective_prereqs_ok = collective_policy_enabled && strategy_config.enable_owner_file_collective &&
      safetensors_disk_source_available && has_collective_group && world_size > 1 &&
      (!strategy_config.owner_file_collective_shared_fs_only || shared_source_proven) &&
      execution_topology.source_locality != loading::SourceLocalityHint::kHostLocal &&
      (strategy_plan.summary.planned_collective_candidate_bytes == 0 ||
       strategy_plan.summary.estimated_collective_dedup_saving_bytes >=
           strategy_config.owner_file_collective_min_dedup_saving_bytes);
  strategy_plan.summary.collective_lane_eligible =
      base_collective_prereqs_ok && strategy_plan.summary.planned_collective_candidate_bytes > 0;
  strategy_plan.summary.planned_collective_admitted_bytes =
      strategy_plan.summary.collective_lane_eligible ? strategy_plan.summary.planned_collective_candidate_bytes : 0;
  strategy_plan.summary.strict_pure_collective_eligible = strategy_plan.summary.collective_lane_eligible &&
      strategy_plan.summary.planned_non_admitted_typed_bytes == 0 &&
      strategy_plan.summary.planned_local_typed_bytes == 0 && strategy_plan.summary.planned_generic_residual_bytes == 0;
  const bool mixed_generic_residual_allowed = strategy_plan.summary.planned_generic_residual_bytes == 0 ||
      strategy_config.owner_file_collective_allow_mixed_residual;
  const bool can_execute_collective_plan = strategy_plan.summary.collective_lane_eligible &&
      mixed_generic_residual_allowed &&
      (strategy_config.allow_mixed_execution || strategy_plan.summary.strict_pure_collective_eligible);
  const bool source_window_can_select = strategy_plan.summary.source_window_collective_candidate &&
      (strategy_config.executor_preference == ExecutorPreference::kSourceWindowCollective ||
       strategy_plan.summary.source_window_selection_mode != SourceWindowCollectiveSelectionMode::kDryRun);
  const uint64_t required_data_bytes = strategy_plan.summary.planned_collective_candidate_bytes +
      strategy_plan.summary.planned_non_admitted_typed_bytes + strategy_plan.summary.planned_generic_residual_bytes;
  const uint64_t generic_backend_coverage_bytes =
      byte_range_map_covered_bytes(strategy_plan.lane_plan.generic_backend_map);
  const uint64_t collective_lane_coverage_bytes =
      byte_range_map_covered_bytes(strategy_plan.lane_plan.collective_lane_map);
  const uint64_t generic_coverage_gap = missing_coverage_bytes(required_data_bytes, generic_backend_coverage_bytes);
  if (generic_coverage_gap > 0) {
    add_reject_reason_bytes(
        &strategy_plan.summary.planner_reject_reason_buckets,
        "generic_backend_coverage_unproven",
        generic_coverage_gap);
  }
  const uint64_t collective_coverage_gap = strategy_plan.summary.collective_lane_eligible
      ? missing_coverage_bytes(strategy_plan.summary.planned_collective_admitted_bytes, collective_lane_coverage_bytes)
      : 0;
  if (collective_coverage_gap > 0) {
    add_reject_reason_bytes(
        &strategy_plan.summary.planner_reject_reason_buckets,
        "collective_lane_coverage_unproven",
        collective_coverage_gap);
  }
  const auto select_source_window_mode = [&]() {
    return (typed_work_prefers_local_mapped && local_mapped_typed_available)
        ? SourceBoundExecutionMode::kSourceWindowCollectiveMixed
        : SourceBoundExecutionMode::kSourceWindowCollective;
  };

  if (policy == SourceBoundPolicy::kRequirePureCollective) {
    if (strategy_plan.summary.strict_pure_collective_eligible && generic_coverage_gap == 0 &&
        collective_coverage_gap == 0) {
      strategy_plan.lane_plan.mode = SourceBoundExecutionMode::kPureCollective;
    } else if (source_window_can_select && generic_coverage_gap == 0) {
      strategy_plan.lane_plan.mode = select_source_window_mode();
    } else {
      strategy_plan.lane_plan.mode = SourceBoundExecutionMode::kRejected;
    }
  } else if (generic_coverage_gap > 0) {
    strategy_plan.lane_plan.mode = SourceBoundExecutionMode::kRejected;
  } else if (
      strategy_plan.summary.planned_collective_candidate_bytes == 0 &&
      strategy_plan.summary.planned_non_admitted_typed_bytes == 0 &&
      strategy_plan.summary.planned_generic_residual_bytes == 0 &&
      strategy_plan.summary.planned_local_typed_bytes > 0) {
    strategy_plan.lane_plan.mode = SourceBoundExecutionMode::kLocalTypedOnly;
  } else if (source_window_can_select) {
    strategy_plan.lane_plan.mode = select_source_window_mode();
  } else if (
      typed_work_prefers_local_mapped && local_mapped_typed_available &&
      strategy_plan.summary.planned_collective_candidate_bytes == 0 &&
      !strategy_plan.summary.collective_lane_eligible && strategy_plan.summary.planned_generic_residual_bytes == 0) {
    strategy_plan.lane_plan.mode = SourceBoundExecutionMode::kLocalMappedTyped;
  } else if (can_execute_collective_plan && collective_coverage_gap == 0) {
    strategy_plan.lane_plan.mode = SourceBoundExecutionMode::kCollectiveFirstMixed;
  } else {
    strategy_plan.lane_plan.mode = SourceBoundExecutionMode::kGenericOnly;
  }

  strategy_plan.summary.execution_plan_kind =
      std::string(source_bound_execution_mode_name(strategy_plan.lane_plan.mode));
  strategy_plan.lane_plan.selection_reason = strategy_plan.summary.execution_plan_kind;
  strategy_plan.lane_plan.local_mapped_typed_selected = typed_work_prefers_local_mapped &&
      local_mapped_typed_available && strategy_plan.lane_plan.mode != SourceBoundExecutionMode::kRejected;
  const bool coverage_only_can_use_local_mapped = local_mapped_typed_available &&
      strategy_plan.lane_plan.mode != SourceBoundExecutionMode::kRejected &&
      (strategy_plan.lane_plan.local_mapped_typed_selected ||
       strategy_config.executor_preference == ExecutorPreference::kTensorAwareLocal ||
       strategy_plan.lane_plan.mode == SourceBoundExecutionMode::kLocalMappedTyped);
  const bool coverage_only_can_use_strict_source_window =
      (strategy_plan.lane_plan.mode == SourceBoundExecutionMode::kSourceWindowCollective ||
       strategy_plan.lane_plan.mode == SourceBoundExecutionMode::kSourceWindowCollectiveMixed) &&
      strategy_plan.summary.source_window_selection_mode == SourceWindowCollectiveSelectionMode::kStrict;
  if (strategy_plan.lane_plan.generic_backend_map_coverage_only && !coverage_only_can_use_local_mapped &&
      !coverage_only_can_use_strict_source_window) {
    const uint64_t coverage_only_bytes = byte_range_map_covered_bytes(strategy_plan.lane_plan.generic_backend_map);
    add_reject_reason_bytes(
        &strategy_plan.summary.planner_reject_reason_buckets,
        "coverage_only_map_requires_local_mapped",
        coverage_only_bytes);
    strategy_plan.lane_plan.mode = SourceBoundExecutionMode::kRejected;
    strategy_plan.summary.execution_plan_kind =
        std::string(source_bound_execution_mode_name(strategy_plan.lane_plan.mode));
    strategy_plan.lane_plan.selection_reason = strategy_plan.summary.execution_plan_kind;
    strategy_plan.lane_plan.local_mapped_typed_selected = false;
  }
  strategy_plan.lane_plan.collective_executor = SourceBoundCollectiveExecutor::kNone;
  if (strategy_plan.lane_plan.mode == SourceBoundExecutionMode::kPureCollective ||
      strategy_plan.lane_plan.mode == SourceBoundExecutionMode::kCollectiveFirstMixed) {
    strategy_plan.lane_plan.collective_executor = SourceBoundCollectiveExecutor::kOwnerFile;
  } else if (
      strategy_plan.lane_plan.mode == SourceBoundExecutionMode::kSourceWindowCollective ||
      strategy_plan.lane_plan.mode == SourceBoundExecutionMode::kSourceWindowCollectiveMixed) {
    strategy_plan.lane_plan.collective_executor = SourceBoundCollectiveExecutor::kSourceWindow;
  }
  if (typed_work_prefers_local_mapped && local_mapped_typed_available &&
      strategy_plan.lane_plan.mode != SourceBoundExecutionMode::kRejected &&
      strategy_plan.lane_plan.mode != SourceBoundExecutionMode::kLocalMappedTyped) {
    absl::StrAppend(&strategy_plan.lane_plan.selection_reason, ";local_mapped_typed");
  }
  strategy_plan.lane_plan.reject_reason_buckets = strategy_plan.summary.planner_reject_reason_buckets;

  auto plan_hash_or = compute_source_bound_plan_hash(strategy_plan.summary);
  if (plan_hash_or.ok()) {
    strategy_plan.summary.plan_hash = *plan_hash_or;
  }
  return strategy_plan;
}

} // namespace tensorcast::store::runtime::ingestion::strategy
