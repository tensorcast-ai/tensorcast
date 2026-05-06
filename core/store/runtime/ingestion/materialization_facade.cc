// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/ingestion/materialization_facade.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "core/common/trace/trace_macros.h"
#include "core/cuda/cuda_ipc.h"
#include "core/store/components/endpoint_id.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/worker_identity.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/control/materialize_orchestrator.h"
#include "core/store/materialization/dataplane/loaders/disk_loader.h"
#include "core/store/materialization/dataplane/loaders/p2p_loader.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/runtime/pump.h"
#include "core/store/materialization/dataplane/runtime/streaming_buffer_adapter.h"
#include "core/store/materialization/dataplane/sinks/target_layout_gpu_sink.h"
#include "core/store/materialization/dataplane/sinks/target_layout_host_sink.h"
#include "core/store/materialization/dataplane/sources/byte_range_map_builder.h"
#include "core/store/materialization/dataplane/sources/byte_range_mapped_source.h"
#include "core/store/materialization/dataplane/sources/byte_range_program.h"
#include "core/store/materialization/dataplane/sources/memory_source.h"
#include "core/store/materialization/dataplane/sources/remote_key_source.h"
#include "core/store/materialization/dataplane/sources/source_window_scheduler.h"
#include "core/store/materialization/dataplane/view/view_ingest_executor.h"
#include "core/store/materialization/dataplane/view/view_plan_source.h"
#include "core/store/materialization/dataplane/view/view_transform_executor.h"
#include "core/store/replica/collective_disk_loader.h"
#include "core/store/replica/replica.h"
#include "core/store/runtime/ingestion/materialization_pump_options.h"
#include "core/store/view_utils.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::runtime::ingestion {

namespace {

using StrategyConfig = StoreEngineOptions::MaterializationStrategyConfig;
using RepresentationTensorSpec = materialization::contracts::RepresentationTensorSpec;
using RepresentationWorkItem = materialization::contracts::RepresentationWorkItem;
using RepresentationWorkItemKind = materialization::contracts::RepresentationWorkItemKind;
using RepresentationWorkPlan = materialization::contracts::RepresentationWorkPlan;
using TensorAxisRange = materialization::contracts::TensorAxisRange;
using TensorCoordinateSpec = materialization::contracts::TensorCoordinateSpec;
using WorkPartitionKind = materialization::contracts::WorkPartitionKind;

bool prefer_local_canonical_source_for_mapped(const StrategyConfig& strategy_config) {
  return strategy_config.prefer_local_canonical_for_mapped;
}

bool allow_source_ordered_for_mapped(const StrategyConfig& strategy_config) {
  return strategy_config.allow_source_ordered_for_mapped;
}

bool allow_collective_mapped_executor(const StrategyConfig& strategy_config) {
  if (!strategy_config.enable_tensor_aware_mapped_executor) {
    return false;
  }
  switch (strategy_config.executor_preference) {
    case StrategyConfig::ExecutorPreference::kGenericByteRange:
    case StrategyConfig::ExecutorPreference::kTensorAwareLocal:
      return false;
    case StrategyConfig::ExecutorPreference::kAuto:
    case StrategyConfig::ExecutorPreference::kOwnerFileCollective:
      return true;
  }
  return true;
}

bool allow_collective_tensor_executor(const StrategyConfig& strategy_config) {
  return allow_collective_mapped_executor(strategy_config);
}

bool strategy_diagnostics_basic_enabled(const StrategyConfig& strategy_config) {
  return strategy_config.diagnostics_verbosity != StrategyConfig::DiagnosticsVerbosity::kOff;
}

bool strategy_diagnostics_verbose_enabled(const StrategyConfig& strategy_config) {
  return strategy_config.diagnostics_verbosity == StrategyConfig::DiagnosticsVerbosity::kVerbose;
}

std::string_view source_locality_name(loading::SourceLocalityHint hint) {
  switch (hint) {
    case loading::SourceLocalityHint::kHostLocal:
      return "host_local";
    case loading::SourceLocalityHint::kSharedSource:
      return "shared_source";
    case loading::SourceLocalityHint::kAuto:
    default:
      return "auto";
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

uint64_t estimate_collective_dedup_saving_bytes(
    const std::optional<RepresentationWorkPlan>& work_plan,
    uint32_t world_size) {
  if (!work_plan.has_value() || world_size <= 1) {
    return 0;
  }
  uint64_t saving = 0;
  const uint64_t peer_count = static_cast<uint64_t>(world_size - 1);
  for (const auto& item : work_plan->items) {
    if (item.kind != RepresentationWorkItemKind::kTensorCopy || item.partition_kind != WorkPartitionKind::kReplicated) {
      continue;
    }
    const uint64_t item_saving = saturating_multiply_u64(item.committed_bytes, peer_count);
    saving = std::numeric_limits<uint64_t>::max() - saving < item_saving ? std::numeric_limits<uint64_t>::max()
                                                                         : saving + item_saving;
  }
  return saving;
}

std::string format_execution_strategy_candidates(const std::vector<strategy::ExecutionStrategyCandidate>& candidates) {
  std::string out;
  for (size_t index = 0; index < candidates.size(); ++index) {
    const auto& candidate = candidates[index];
    if (index > 0) {
      out.append(" | ");
    }
    absl::StrAppend(
        &out,
        strategy::execution_strategy_executor_name(candidate.executor),
        ":eligible=",
        candidate.eligible ? 1 : 0,
        ":reason=",
        candidate.reason,
        ":requested=",
        candidate.estimate.requested_source_bytes,
        ":unique=",
        candidate.estimate.unique_source_bytes,
        ":peak=",
        candidate.estimate.estimated_peak_temporary_bytes,
        ":batch=",
        candidate.estimate.estimated_batch_bytes,
        ":dedup=",
        candidate.estimate.estimated_dedup_saving_bytes,
        ":skew=",
        candidate.estimate.estimated_owner_skew_ratio);
  }
  return out;
}

std::optional<std::string> local_auto_reject_reason(
    const strategy::ExecutionStrategyCostEstimate& generic_estimate,
    const strategy::ExecutionStrategyCostEstimate& local_estimate,
    const std::optional<replica::LocalBatchedPlanSummary>& local_plan_summary) {
  if (!local_plan_summary.has_value() || !local_plan_summary->eligible) {
    return std::nullopt;
  }
  if (local_estimate.unique_source_bytes > generic_estimate.unique_source_bytes) {
    return std::string("local_source_amplification_exceeds_generic");
  }
  return std::nullopt;
}

std::string dominant_collective_reject_reason(const std::optional<strategy::SourceBoundExecutionPlanSummary>& summary) {
  if (!summary.has_value()) {
    return {};
  }
  if (summary->execution_plan_kind == "pure_collective" || summary->execution_plan_kind == "collective_first_mixed") {
    return {};
  }
  if (summary->planner_reject_reason_buckets.empty()) {
    return "planner_not_collective_eligible";
  }
  auto best = std::max_element(
      summary->planner_reject_reason_buckets.begin(),
      summary->planner_reject_reason_buckets.end(),
      [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) {
          return lhs.second < rhs.second;
        }
        return lhs.first > rhs.first;
      });
  return best == summary->planner_reject_reason_buckets.end() ? std::string{} : absl::StrCat("planner_", best->first);
}

absl::StatusOr<loading::MaterializeHints> build_effective_mapped_hints(
    const strategy::ResolvedMaterializationPlan& resolved_plan,
    const loading::MaterializeHints& hints) {
  if (resolved_plan.artifact_id.empty()) {
    return absl::InvalidArgumentError("materialize_mapped_into_target requires resolved_plan.artifact_id");
  }
  if (resolved_plan.canonical_index_json.empty()) {
    return absl::InvalidArgumentError("materialize_mapped_into_target requires resolved_plan.canonical_index_json");
  }
  if (!resolved_plan.representation_transform_contract.has_value()) {
    return absl::InvalidArgumentError(
        "materialize_mapped_into_target requires resolved_plan.representation_transform_contract");
  }
  if (!resolved_plan.representation_work_plan.has_value()) {
    return absl::InvalidArgumentError("materialize_mapped_into_target requires resolved_plan.representation_work_plan");
  }
  loading::MaterializeHints effective_hints = hints;
  if (effective_hints.artifact_id.empty()) {
    effective_hints.artifact_id = resolved_plan.artifact_id;
  } else if (effective_hints.artifact_id != resolved_plan.artifact_id) {
    return absl::InvalidArgumentError("materialize_mapped_into_target hints.artifact_id does not match resolved_plan");
  }

  if (resolved_plan.variant.has_value()) {
    if (effective_hints.variant.has_value()) {
      if (effective_hints.variant->canonical_artifact_id != resolved_plan.variant->canonical_artifact_id) {
        return absl::InvalidArgumentError(
            "materialize_mapped_into_target hints.variant canonical_artifact_id does not match resolved_plan");
      }
      if (effective_hints.variant->view_id.value_or("") != resolved_plan.variant->view_id.value_or("")) {
        return absl::InvalidArgumentError(
            "materialize_mapped_into_target hints.variant view_id does not match resolved_plan");
      }
    }
    effective_hints.variant = resolved_plan.variant;
  } else if (effective_hints.variant.has_value()) {
    return absl::InvalidArgumentError(
        "materialize_mapped_into_target requires resolved_plan.variant when hints.variant is provided");
  }

  return effective_hints;
}

bool byte_range_map_crosses_target_storage_boundaries(
    const loader::ByteRangeMap& map,
    const loading::IntoTargetLayout& target_layout) {
  if (target_layout.storages.size() <= 1) {
    return false;
  }

  struct StorageRange {
    uint64_t begin{0};
    uint64_t end{0};
  };

  std::vector<StorageRange> storage_ranges;
  storage_ranges.reserve(target_layout.storages.size());
  uint64_t cursor = 0;
  for (const auto& storage : target_layout.storages) {
    if (storage.length > std::numeric_limits<uint64_t>::max() - cursor) {
      return true;
    }
    const uint64_t begin = cursor;
    cursor += storage.length;
    storage_ranges.push_back(StorageRange{.begin = begin, .end = cursor});
  }

  for (const auto& segment : map.segments) {
    if (segment.length == 0) {
      continue;
    }
    if (segment.dst_offset > std::numeric_limits<uint64_t>::max() - segment.length) {
      return true;
    }
    const uint64_t segment_end = segment.dst_offset + segment.length;
    bool covered_by_single_storage = false;
    for (const auto& storage_range : storage_ranges) {
      if (segment.dst_offset >= storage_range.begin && segment_end <= storage_range.end) {
        covered_by_single_storage = true;
        break;
      }
    }
    if (!covered_by_single_storage) {
      return true;
    }
  }
  return false;
}

tensorcast::common::v1::ByteSpaceRef canonical_byte_space_ref() {
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  return byte_space;
}

struct ByteSpan {
  uint64_t offset{0};
  uint64_t length{0};
};

bool is_row_major_contiguous(const RepresentationTensorSpec& spec) {
  if (spec.shape.empty()) {
    return spec.stride.empty();
  }
  if (spec.shape.size() != spec.stride.size()) {
    return false;
  }
  int64_t expected = 1;
  for (int64_t index = static_cast<int64_t>(spec.shape.size()) - 1; index >= 0; --index) {
    if (spec.stride[static_cast<size_t>(index)] != expected) {
      return false;
    }
    expected *= spec.shape[static_cast<size_t>(index)];
  }
  return true;
}

std::optional<TensorAxisRange> single_axis_range(const TensorCoordinateSpec& spec) {
  if (spec.selects_scalar || spec.axes.size() != 1) {
    return std::nullopt;
  }
  return spec.axes.front();
}

bool coordinate_selects_single_element(const TensorCoordinateSpec& spec, const RepresentationTensorSpec& tensor) {
  if (spec.selects_scalar) {
    return tensor.shape.empty();
  }
  if (tensor.shape.empty()) {
    return spec.axes.empty();
  }
  if (spec.axes.size() != tensor.shape.size()) {
    return false;
  }
  std::vector<bool> seen(tensor.shape.size(), false);
  for (const auto& axis : spec.axes) {
    if (axis.dim < 0 || axis.dim >= static_cast<int32_t>(tensor.shape.size())) {
      return false;
    }
    if (seen[static_cast<size_t>(axis.dim)]) {
      return false;
    }
    seen[static_cast<size_t>(axis.dim)] = true;
    if (axis.end - axis.start != 1) {
      return false;
    }
  }
  return std::all_of(seen.begin(), seen.end(), [](bool value) { return value; });
}

bool work_plan_has_fill_items(const RepresentationWorkPlan& plan) {
  return std::any_of(plan.items.begin(), plan.items.end(), [](const RepresentationWorkItem& item) {
    return item.kind == RepresentationWorkItemKind::kConstFill ||
        item.kind == RepresentationWorkItemKind::kScalarBroadcastFill ||
        item.kind == RepresentationWorkItemKind::kPadFill;
  });
}

bool work_plan_has_scalar_fill_items(const RepresentationWorkPlan& plan) {
  return std::any_of(plan.items.begin(), plan.items.end(), [](const RepresentationWorkItem& item) {
    return item.kind == RepresentationWorkItemKind::kScalarBroadcastFill;
  });
}

uint64_t local_typed_work_bytes(const RepresentationWorkPlan& plan) {
  uint64_t total = 0;
  for (const auto& item : plan.items) {
    if (item.kind == RepresentationWorkItemKind::kConstFill ||
        item.kind == RepresentationWorkItemKind::kScalarBroadcastFill ||
        item.kind == RepresentationWorkItemKind::kPadFill) {
      total += item.committed_bytes;
    }
  }
  return total;
}

bool work_plan_requires_explicit_executor_data_map(const RepresentationWorkPlan& plan) {
  return std::any_of(plan.items.begin(), plan.items.end(), [](const RepresentationWorkItem& item) {
    return item.kind == RepresentationWorkItemKind::kTensorCopy ||
        item.kind == RepresentationWorkItemKind::kConcatAssemble ||
        item.kind == RepresentationWorkItemKind::kExpertDim0Concat ||
        item.kind == RepresentationWorkItemKind::kResidualByteRange;
  });
}

bool source_bound_execution_mode_uses_collective(strategy::SourceBoundExecutionMode mode) {
  return mode == strategy::SourceBoundExecutionMode::kPureCollective ||
      mode == strategy::SourceBoundExecutionMode::kCollectiveFirstMixed;
}

bool source_bound_execution_mode_uses_local_mapped(strategy::SourceBoundExecutionMode mode) {
  return mode == strategy::SourceBoundExecutionMode::kLocalMappedTyped;
}

bool source_bound_execution_mode_requires_executor_source_map(strategy::SourceBoundExecutionMode mode) {
  return mode == strategy::SourceBoundExecutionMode::kPureCollective ||
      mode == strategy::SourceBoundExecutionMode::kCollectiveFirstMixed ||
      mode == strategy::SourceBoundExecutionMode::kGenericOnly ||
      mode == strategy::SourceBoundExecutionMode::kLocalMappedTyped;
}

uint64_t planned_generic_backend_bytes_after_local_mapped(const strategy::SourceBoundExecutionPlanSummary& summary) {
  const uint64_t non_admitted_collective_bytes =
      summary.planned_collective_candidate_bytes > summary.planned_collective_admitted_bytes
      ? summary.planned_collective_candidate_bytes - summary.planned_collective_admitted_bytes
      : 0;
  if (summary.planned_generic_residual_bytes > std::numeric_limits<uint64_t>::max() - non_admitted_collective_bytes) {
    return std::numeric_limits<uint64_t>::max();
  }
  return non_admitted_collective_bytes + summary.planned_generic_residual_bytes;
}

std::string format_reject_reason_buckets(const absl::flat_hash_map<std::string, uint64_t>& buckets) {
  if (buckets.empty()) {
    return "{}";
  }
  std::vector<std::pair<std::string, uint64_t>> ordered(buckets.begin(), buckets.end());
  std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  std::string rendered = "{";
  for (size_t index = 0; index < ordered.size(); ++index) {
    if (index > 0) {
      rendered.append(",");
    }
    absl::StrAppend(&rendered, ordered[index].first, "=", ordered[index].second);
  }
  rendered.push_back('}');
  return rendered;
}

uint64_t byte_range_map_covered_bytes(const loader::ByteRangeMap& map) {
  uint64_t total = 0;
  for (const auto& segment : map.segments) {
    total += segment.length;
  }
  return total;
}

absl::StatusOr<loader::ByteRangeMap> filter_byte_range_map_by_kind(
    const loader::ByteRangeMap& map,
    loader::ByteRangeSegment::Kind kind) {
  loader::ByteRangeMap filtered;
  filtered.total_bytes = map.total_bytes;
  filtered.num_sources = map.num_sources;
  for (const auto& segment : map.segments) {
    if (segment.kind == kind) {
      filtered.segments.push_back(segment);
    }
  }
  std::sort(filtered.segments.begin(), filtered.segments.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.dst_offset < rhs.dst_offset;
  });
  for (size_t index = 1; index < filtered.segments.size(); ++index) {
    const auto& previous = filtered.segments[index - 1];
    const auto& current = filtered.segments[index];
    if (current.dst_offset < previous.dst_offset + previous.length) {
      return absl::InvalidArgumentError("filtered byte range map contains overlapping segments");
    }
  }
  return filtered;
}

absl::StatusOr<loader::ByteRangeMap> merge_byte_range_maps(
    const loader::ByteRangeMap& lhs,
    const loader::ByteRangeMap& rhs) {
  loader::ByteRangeMap merged;
  merged.total_bytes = std::max(lhs.total_bytes, rhs.total_bytes);
  merged.num_sources = std::max(lhs.num_sources, rhs.num_sources);
  merged.segments.reserve(lhs.segments.size() + rhs.segments.size());
  merged.segments.insert(merged.segments.end(), lhs.segments.begin(), lhs.segments.end());
  merged.segments.insert(merged.segments.end(), rhs.segments.begin(), rhs.segments.end());
  std::sort(merged.segments.begin(), merged.segments.end(), [](const auto& lhs_seg, const auto& rhs_seg) {
    return lhs_seg.dst_offset < rhs_seg.dst_offset;
  });
  for (size_t index = 1; index < merged.segments.size(); ++index) {
    const auto& previous = merged.segments[index - 1];
    const auto& current = merged.segments[index];
    if (current.dst_offset < previous.dst_offset + previous.length) {
      return absl::InvalidArgumentError("merged byte range map contains overlapping segments");
    }
  }
  return merged;
}

absl::StatusOr<std::vector<loader::ByteRangeSegment>> sorted_validated_segments(
    const loader::ByteRangeMap& map,
    bool data_only = false) {
  if (map.total_bytes == 0) {
    if (!map.segments.empty()) {
      return absl::InvalidArgumentError("byte range map total_bytes is zero with non-empty segments");
    }
    return std::vector<loader::ByteRangeSegment>{};
  }
  if (map.num_sources == 0) {
    return absl::InvalidArgumentError("byte range map num_sources must be > 0");
  }
  std::vector<loader::ByteRangeSegment> segments;
  segments.reserve(map.segments.size());
  for (const auto& segment : map.segments) {
    if (segment.length == 0) {
      continue;
    }
    if (segment.dst_offset >= map.total_bytes || segment.length > map.total_bytes - segment.dst_offset) {
      return absl::InvalidArgumentError("byte range map segment exceeds total_bytes");
    }
    if (data_only && segment.kind != loader::ByteRangeSegment::Kind::kData) {
      return absl::InvalidArgumentError("byte range map subtraction requires data-only segments");
    }
    if (segment.kind == loader::ByteRangeSegment::Kind::kData && segment.source_index >= map.num_sources) {
      return absl::InvalidArgumentError("byte range map segment source_index out of range");
    }
    segments.push_back(segment);
  }
  std::sort(segments.begin(), segments.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.dst_offset < rhs.dst_offset;
  });
  for (size_t index = 1; index < segments.size(); ++index) {
    const auto& previous = segments[index - 1];
    const auto& current = segments[index];
    if (current.dst_offset < previous.dst_offset + previous.length) {
      return absl::InvalidArgumentError("byte range map contains overlapping segments");
    }
  }
  return segments;
}

absl::StatusOr<loader::ByteRangeMap> densify_byte_range_map_with_pad(const loader::ByteRangeMap& map) {
  if (map.total_bytes == 0) {
    return loader::ByteRangeMap{};
  }
  auto segments_or = sorted_validated_segments(map);
  if (!segments_or.ok()) {
    return segments_or.status();
  }
  loader::ByteRangeMap dense;
  dense.total_bytes = map.total_bytes;
  dense.num_sources = map.num_sources;
  dense.segments.reserve(segments_or->size() * 2 + 1);
  uint64_t cursor = 0;
  for (const auto& segment : *segments_or) {
    if (segment.dst_offset > cursor) {
      dense.segments.push_back(
          loader::ByteRangeSegment{
              .kind = loader::ByteRangeSegment::Kind::kPad,
              .dst_offset = cursor,
              .length = segment.dst_offset - cursor,
              .src_offset = 0,
              .source_index = 0,
          });
    }
    dense.segments.push_back(segment);
    cursor = segment.dst_offset + segment.length;
  }
  if (cursor < dense.total_bytes) {
    dense.segments.push_back(
        loader::ByteRangeSegment{
            .kind = loader::ByteRangeSegment::Kind::kPad,
            .dst_offset = cursor,
            .length = dense.total_bytes - cursor,
            .src_offset = 0,
            .source_index = 0,
        });
  }
  return loader::normalize_byte_range_map(std::move(dense));
}

absl::StatusOr<loader::ByteRangeMap> subtract_byte_range_maps(
    const loader::ByteRangeMap& whole,
    const loader::ByteRangeMap& part) {
  if (whole.total_bytes != part.total_bytes) {
    return absl::InvalidArgumentError("byte range map subtraction requires matching total_bytes");
  }
  if (whole.num_sources != part.num_sources) {
    return absl::InvalidArgumentError("byte range map subtraction requires matching num_sources");
  }
  auto whole_segments_or = sorted_validated_segments(whole, /*data_only=*/true);
  if (!whole_segments_or.ok()) {
    return whole_segments_or.status();
  }
  auto part_segments_or = sorted_validated_segments(part, /*data_only=*/true);
  if (!part_segments_or.ok()) {
    return part_segments_or.status();
  }

  loader::ByteRangeMap residual;
  residual.total_bytes = whole.total_bytes;
  residual.num_sources = whole.num_sources;
  residual.segments.reserve(whole_segments_or->size());

  size_t part_index = 0;
  for (const auto& whole_segment : *whole_segments_or) {
    const uint64_t whole_end = whole_segment.dst_offset + whole_segment.length;
    uint64_t cursor = whole_segment.dst_offset;
    while (part_index < part_segments_or->size() &&
           ((*part_segments_or)[part_index].dst_offset + (*part_segments_or)[part_index].length) <= cursor) {
      ++part_index;
    }
    size_t current_part = part_index;
    while (current_part < part_segments_or->size()) {
      const auto& part_segment = (*part_segments_or)[current_part];
      if (part_segment.dst_offset >= whole_end) {
        break;
      }
      const uint64_t part_end = part_segment.dst_offset + part_segment.length;
      if (part_segment.dst_offset < whole_segment.dst_offset || part_end > whole_end) {
        return absl::InvalidArgumentError("collective effective data map is not a subset of executor data map");
      }
      const uint64_t source_delta = part_segment.dst_offset - whole_segment.dst_offset;
      if (part_segment.source_index != whole_segment.source_index ||
          part_segment.src_offset != whole_segment.src_offset + source_delta) {
        return absl::InvalidArgumentError(
            "collective effective data map source offsets do not match executor data map");
      }
      if (part_segment.dst_offset > cursor) {
        const uint64_t length = part_segment.dst_offset - cursor;
        residual.segments.push_back(
            loader::ByteRangeSegment{
                .kind = loader::ByteRangeSegment::Kind::kData,
                .dst_offset = cursor,
                .length = length,
                .src_offset = whole_segment.src_offset + (cursor - whole_segment.dst_offset),
                .source_index = whole_segment.source_index,
            });
      }
      cursor = part_end;
      current_part += 1;
    }
    if (cursor < whole_end) {
      residual.segments.push_back(
          loader::ByteRangeSegment{
              .kind = loader::ByteRangeSegment::Kind::kData,
              .dst_offset = cursor,
              .length = whole_end - cursor,
              .src_offset = whole_segment.src_offset + (cursor - whole_segment.dst_offset),
              .source_index = whole_segment.source_index,
          });
    }
    part_index = current_part;
  }
  if (part_index != part_segments_or->size()) {
    return absl::InvalidArgumentError("collective effective data map extends beyond executor data map");
  }
  return residual;
}

absl::StatusOr<loader::ByteRangeMap> subtract_byte_range_map_by_dst_ranges(
    const loader::ByteRangeMap& whole,
    const loader::ByteRangeMap& part) {
  if (whole.total_bytes != part.total_bytes) {
    return absl::InvalidArgumentError("byte range map subtraction requires matching total_bytes");
  }
  auto whole_segments_or = sorted_validated_segments(whole);
  if (!whole_segments_or.ok()) {
    return whole_segments_or.status();
  }
  auto part_segments_or = sorted_validated_segments(part);
  if (!part_segments_or.ok()) {
    return part_segments_or.status();
  }

  loader::ByteRangeMap residual;
  residual.total_bytes = whole.total_bytes;
  residual.num_sources = whole.num_sources;
  residual.segments.reserve(whole_segments_or->size());

  size_t part_index = 0;
  for (const auto& whole_segment : *whole_segments_or) {
    const uint64_t whole_end = whole_segment.dst_offset + whole_segment.length;
    uint64_t cursor = whole_segment.dst_offset;
    while (part_index < part_segments_or->size() &&
           ((*part_segments_or)[part_index].dst_offset + (*part_segments_or)[part_index].length) <= cursor) {
      ++part_index;
    }
    size_t current_part = part_index;
    while (current_part < part_segments_or->size()) {
      const auto& part_segment = (*part_segments_or)[current_part];
      if (part_segment.dst_offset >= whole_end) {
        break;
      }
      const uint64_t part_end = part_segment.dst_offset + part_segment.length;
      if (part_segment.dst_offset < whole_segment.dst_offset) {
        return absl::InvalidArgumentError("subtracted byte range map is not a destination-range subset");
      }
      if (part_end > whole_end) {
        return absl::InvalidArgumentError("subtracted byte range map crosses destination coverage boundaries");
      }
      if (part_segment.dst_offset > cursor) {
        const uint64_t length = part_segment.dst_offset - cursor;
        residual.segments.push_back(
            loader::ByteRangeSegment{
                .kind = whole_segment.kind,
                .dst_offset = cursor,
                .length = length,
                .src_offset = whole_segment.kind == loader::ByteRangeSegment::Kind::kPad
                    ? 0
                    : whole_segment.src_offset + (cursor - whole_segment.dst_offset),
                .source_index =
                    whole_segment.kind == loader::ByteRangeSegment::Kind::kPad ? 0 : whole_segment.source_index,
            });
      }
      cursor = part_end;
      current_part += 1;
    }
    if (cursor < whole_end) {
      residual.segments.push_back(
          loader::ByteRangeSegment{
              .kind = whole_segment.kind,
              .dst_offset = cursor,
              .length = whole_end - cursor,
              .src_offset = whole_segment.kind == loader::ByteRangeSegment::Kind::kPad
                  ? 0
                  : whole_segment.src_offset + (cursor - whole_segment.dst_offset),
              .source_index =
                  whole_segment.kind == loader::ByteRangeSegment::Kind::kPad ? 0 : whole_segment.source_index,
          });
    }
    part_index = current_part;
  }
  if (part_index != part_segments_or->size()) {
    return absl::InvalidArgumentError("subtracted byte range map extends beyond destination coverage");
  }
  return residual;
}

absl::StatusOr<loader::ByteRangeMap> finalize_runtime_source_map(
    const loader::ByteRangeMap& map,
    bool source_exposes_target_byte_space,
    const std::optional<loader::ViewPlan>& view_plan,
    bool source_is_view,
    bool use_source_layout,
    const std::optional<loader::ByteRangeMap>& source_layout_map) {
  auto dense_or = densify_byte_range_map_with_pad(map);
  if (!dense_or.ok()) {
    return dense_or.status();
  }
  loader::ByteRangeMap effective_map = std::move(*dense_or);

  if (source_exposes_target_byte_space) {
    for (auto& segment : effective_map.segments) {
      if (segment.kind == loader::ByteRangeSegment::Kind::kData) {
        segment.src_offset = segment.dst_offset;
        segment.source_index = 0;
      }
    }
    auto normalized_or = loader::normalize_byte_range_map(std::move(effective_map));
    if (!normalized_or.ok()) {
      return normalized_or.status();
    }
    effective_map = std::move(*normalized_or);
  }
  if (!source_is_view && view_plan.has_value() && !view_plan->is_identity) {
    auto composed_or = loader::compose_byte_range_maps(effective_map, view_plan->selection.map);
    if (!composed_or.ok()) {
      return composed_or.status();
    }
    effective_map = std::move(*composed_or);
  }
  if (use_source_layout) {
    if (!source_layout_map.has_value()) {
      return absl::InvalidArgumentError("runtime source finalization requires source layout map");
    }
    auto composed_or = loader::compose_byte_range_maps(effective_map, *source_layout_map);
    if (!composed_or.ok()) {
      return composed_or.status();
    }
    effective_map = std::move(*composed_or);
  }
  return effective_map;
}

struct EffectiveSourceMaps {
  loader::ByteRangeMap executor_effective_map;
  loader::ByteRangeMap executor_effective_data_map;
  loader::ByteRangeMap executor_effective_pad_map;
  loader::ByteRangeMap collective_effective_data_map;
  loader::ByteRangeMap generic_effective_data_map;
};

absl::StatusOr<EffectiveSourceMaps> derive_effective_source_maps(
    const loader::ByteRangeMap& executor_private_map,
    const loader::ByteRangeMap& collective_lane_map,
    const loader::ByteRangeMap& local_typed_pad_map,
    bool source_exposes_target_byte_space,
    const std::optional<loader::ViewPlan>& view_plan,
    bool source_is_view,
    bool use_source_layout,
    const std::optional<loader::ByteRangeMap>& source_layout_map) {
  loader::ByteRangeMap executor_input = executor_private_map;
  if (!local_typed_pad_map.segments.empty()) {
    auto merged_or = merge_byte_range_maps(executor_input, local_typed_pad_map);
    if (!merged_or.ok()) {
      return merged_or.status();
    }
    executor_input = std::move(*merged_or);
  }

  auto executor_effective_map_or = finalize_runtime_source_map(
      executor_input,
      source_exposes_target_byte_space,
      view_plan,
      source_is_view,
      use_source_layout,
      source_layout_map);
  if (!executor_effective_map_or.ok()) {
    return executor_effective_map_or.status();
  }
  auto executor_effective_data_map_or =
      filter_byte_range_map_by_kind(*executor_effective_map_or, loader::ByteRangeSegment::Kind::kData);
  if (!executor_effective_data_map_or.ok()) {
    return executor_effective_data_map_or.status();
  }
  auto executor_effective_pad_map_or =
      filter_byte_range_map_by_kind(*executor_effective_map_or, loader::ByteRangeSegment::Kind::kPad);
  if (!executor_effective_pad_map_or.ok()) {
    return executor_effective_pad_map_or.status();
  }

  auto collective_effective_map_or = finalize_runtime_source_map(
      collective_lane_map,
      source_exposes_target_byte_space,
      view_plan,
      source_is_view,
      use_source_layout,
      source_layout_map);
  if (!collective_effective_map_or.ok()) {
    return collective_effective_map_or.status();
  }
  auto collective_effective_data_map_or =
      filter_byte_range_map_by_kind(*collective_effective_map_or, loader::ByteRangeSegment::Kind::kData);
  if (!collective_effective_data_map_or.ok()) {
    return collective_effective_data_map_or.status();
  }
  if (collective_effective_data_map_or->total_bytes == 0 && collective_effective_data_map_or->segments.empty()) {
    collective_effective_data_map_or->total_bytes = executor_effective_data_map_or->total_bytes;
    collective_effective_data_map_or->num_sources = executor_effective_data_map_or->num_sources;
  }

  auto generic_effective_data_map_or =
      subtract_byte_range_maps(*executor_effective_data_map_or, *collective_effective_data_map_or);
  if (!generic_effective_data_map_or.ok()) {
    return generic_effective_data_map_or.status();
  }

  return EffectiveSourceMaps{
      .executor_effective_map = std::move(*executor_effective_map_or),
      .executor_effective_data_map = std::move(*executor_effective_data_map_or),
      .executor_effective_pad_map = std::move(*executor_effective_pad_map_or),
      .collective_effective_data_map = std::move(*collective_effective_data_map_or),
      .generic_effective_data_map = std::move(*generic_effective_data_map_or),
  };
}

absl::Status execute_sparse_generic_backend_map(
    loader::SeekableSource& source,
    const loader::ByteRangeMap& data_map,
    loader::TargetLayoutGpuSink& sink,
    size_t chunk_bytes) {
  if (chunk_bytes == 0) {
    return absl::InvalidArgumentError("sparse generic execution requires chunk_bytes > 0");
  }
  auto segments_or = sorted_validated_segments(data_map, /*data_only=*/true);
  if (!segments_or.ok()) {
    return segments_or.status();
  }
  std::vector<std::uint8_t> buffer(chunk_bytes);
  for (const auto& segment : *segments_or) {
    uint64_t copied = 0;
    while (copied < segment.length) {
      const size_t step =
          static_cast<size_t>(std::min<uint64_t>(segment.length - copied, static_cast<uint64_t>(buffer.size())));
      auto read_or = source.read_at(segment.src_offset + copied, buffer.data(), step);
      if (!read_or.ok()) {
        return read_or.status();
      }
      if (*read_or != step) {
        return absl::OutOfRangeError("short read while executing sparse generic backend map");
      }
      auto write_status = sink.write_at(segment.dst_offset + copied, buffer.data(), step);
      if (!write_status.ok()) {
        return write_status;
      }
      copied += static_cast<uint64_t>(step);
    }
  }
  return absl::OkStatus();
}

replica::CollectiveMappedTargetLoadResult execute_collective_mapped_target_load(
    const std::shared_ptr<const tensorcast::store::runtime::MaterializationHooks>& hooks,
    const replica::CollectiveMappedTargetLoadRequest& request,
    const std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const replica::CollectiveMappedTargetLoadOptions& options) {
  if (hooks != nullptr && hooks->collective_mapped_target_load_override) {
    return hooks->collective_mapped_target_load_override(request, pinned_pool, pinned_timeout, options);
  }
  return replica::try_collective_mapped_target_load(request, pinned_pool, pinned_timeout, options);
}

absl::StatusOr<std::vector<ByteSpan>> build_destination_byte_spans(
    const RepresentationTensorSpec& spec,
    const TensorCoordinateSpec& coordinate) {
  if (spec.element_size == 0 && spec.logical_length == 0) {
    return absl::InvalidArgumentError("destination tensor requires non-zero element_size or logical_length");
  }
  if (coordinate.selects_scalar || coordinate_selects_single_element(coordinate, spec)) {
    uint64_t element_offset = 0;
    for (const auto& axis : coordinate.axes) {
      element_offset +=
          static_cast<uint64_t>(axis.start) * static_cast<uint64_t>(spec.stride[static_cast<size_t>(axis.dim)]);
    }
    const uint64_t scalar_bytes = spec.element_size == 0 ? spec.logical_length : spec.element_size;
    return std::vector<ByteSpan>{
        {.offset = spec.logical_offset + element_offset * scalar_bytes, .length = scalar_bytes}};
  }
  if (coordinate.axes.empty()) {
    return std::vector<ByteSpan>{{.offset = spec.logical_offset, .length = spec.logical_length}};
  }
  if (!is_row_major_contiguous(spec)) {
    return absl::UnimplementedError("fill work requires row-major contiguous destination tensor");
  }
  const auto axis = single_axis_range(coordinate);
  if (!axis.has_value()) {
    return absl::UnimplementedError("fill work only supports full, scalar, dim0, or 2D dim1 destinations");
  }
  if (axis->dim == 0) {
    uint64_t tail = 1;
    for (size_t index = 1; index < spec.shape.size(); ++index) {
      tail *= static_cast<uint64_t>(spec.shape[index]);
    }
    return std::vector<ByteSpan>{{
        .offset = spec.logical_offset + static_cast<uint64_t>(axis->start) * tail * spec.element_size,
        .length = static_cast<uint64_t>(axis->end - axis->start) * tail * spec.element_size,
    }};
  }
  if (axis->dim == 1 && spec.shape.size() == 2) {
    std::vector<ByteSpan> spans;
    spans.reserve(static_cast<size_t>(spec.shape[0]));
    const uint64_t row_stride_bytes = static_cast<uint64_t>(spec.stride[0]) * spec.element_size;
    const uint64_t col_offset_bytes = static_cast<uint64_t>(axis->start) * spec.element_size;
    const uint64_t col_bytes = static_cast<uint64_t>(axis->end - axis->start) * spec.element_size;
    for (int64_t row = 0; row < spec.shape[0]; ++row) {
      spans.push_back(
          ByteSpan{
              .offset = spec.logical_offset + static_cast<uint64_t>(row) * row_stride_bytes + col_offset_bytes,
              .length = col_bytes,
          });
    }
    return spans;
  }
  return absl::UnimplementedError("fill work only supports dim0 or 2D dim1 destinations");
}

absl::StatusOr<std::vector<std::uint8_t>> read_scalar_source_value(
    loader::SeekableSource& source,
    const RepresentationWorkItem& item) {
  if (item.sources.size() != 1) {
    return absl::InvalidArgumentError("scalar-from-source work item requires exactly one source");
  }
  const auto& fragment = item.sources.front().fragment;
  const auto& source_spec = fragment.source_spec;
  if (!coordinate_selects_single_element(fragment.source_range, source_spec)) {
    return absl::InvalidArgumentError("scalar-from-source work item requires a single source element");
  }
  const uint64_t scalar_bytes = source_spec.element_size == 0 ? source_spec.logical_length : source_spec.element_size;
  uint64_t element_offset = 0;
  for (const auto& axis : fragment.source_range.axes) {
    element_offset +=
        static_cast<uint64_t>(axis.start) * static_cast<uint64_t>(source_spec.stride[static_cast<size_t>(axis.dim)]);
  }
  std::vector<std::uint8_t> value(scalar_bytes);
  const uint64_t byte_offset =
      source_spec.logical_offset + source_spec.storage_offset * scalar_bytes + element_offset * scalar_bytes;
  size_t got = 0;
  auto got_or = source.read_at(byte_offset, value.data(), value.size());
  if (!got_or.ok()) {
    return got_or.status();
  }
  got = *got_or;
  if (got != value.size()) {
    return absl::OutOfRangeError("short read while reading scalar source value");
  }
  return value;
}

absl::Status write_pattern_to_sink(
    loader::TargetLayoutGpuSink& sink,
    uint64_t offset,
    uint64_t length,
    absl::Span<const std::uint8_t> pattern,
    size_t chunk_bytes) {
  if (pattern.empty()) {
    return absl::InvalidArgumentError("fill pattern must not be empty");
  }
  const size_t buffer_bytes = std::max<size_t>(pattern.size(), std::min<uint64_t>(length, chunk_bytes));
  std::vector<std::uint8_t> buffer(buffer_bytes);
  for (size_t cursor = 0; cursor < buffer.size(); cursor += pattern.size()) {
    const size_t copy_bytes = std::min<size_t>(pattern.size(), buffer.size() - cursor);
    std::memcpy(buffer.data() + cursor, pattern.data(), copy_bytes);
  }
  uint64_t remaining = length;
  uint64_t dst_offset = offset;
  while (remaining > 0) {
    const size_t take = static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
    auto write_status = sink.write_at(dst_offset, buffer.data(), take);
    if (!write_status.ok()) {
      return write_status;
    }
    remaining -= take;
    dst_offset += take;
  }
  return absl::OkStatus();
}

absl::Status execute_fill_work_items(
    absl::Span<const RepresentationWorkItem> items,
    loader::TargetLayoutGpuSink& sink,
    loader::SeekableSource* source,
    size_t chunk_bytes) {
  for (const auto& item : items) {
    if (item.kind != RepresentationWorkItemKind::kConstFill &&
        item.kind != RepresentationWorkItemKind::kScalarBroadcastFill &&
        item.kind != RepresentationWorkItemKind::kPadFill) {
      continue;
    }
    std::vector<ByteSpan> spans;
    std::vector<std::uint8_t> pattern;
    if (item.kind == RepresentationWorkItemKind::kPadFill) {
      pattern = {0};
      for (const auto& segment : item.byte_range_map.segments) {
        if (segment.length == 0) {
          continue;
        }
        auto status = write_pattern_to_sink(sink, segment.dst_offset, segment.length, pattern, chunk_bytes);
        if (!status.ok()) {
          return status;
        }
      }
      continue;
    }
    if (item.kind == RepresentationWorkItemKind::kConstFill) {
      if (!item.fill_rule.has_value() || item.fill_rule->constant_value.empty()) {
        return absl::InvalidArgumentError("const fill work item requires a non-empty fill_rule");
      }
      pattern = item.fill_rule->constant_value;
      auto spans_or = build_destination_byte_spans(item.dst_spec, item.fill_rule->destination_range);
      if (!spans_or.ok()) {
        return spans_or.status();
      }
      spans = std::move(*spans_or);
    } else {
      if (source == nullptr) {
        return absl::FailedPreconditionError("scalar-from-source work item requires a source");
      }
      auto pattern_or = read_scalar_source_value(*source, item);
      if (!pattern_or.ok()) {
        return pattern_or.status();
      }
      pattern = std::move(*pattern_or);
      auto spans_or = build_destination_byte_spans(item.dst_spec, item.sources.front().fragment.destination_range);
      if (!spans_or.ok()) {
        return spans_or.status();
      }
      spans = std::move(*spans_or);
    }
    for (const auto& span : spans) {
      auto status = write_pattern_to_sink(sink, span.offset, span.length, pattern, chunk_bytes);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<loader::ByteRangeMap> build_local_typed_pad_map(
    const RepresentationWorkPlan& plan,
    uint64_t total_bytes) {
  loader::ByteRangeMap map;
  map.total_bytes = total_bytes;
  map.num_sources = 1;

  auto append_span = [&](uint64_t offset, uint64_t length) {
    if (length == 0) {
      return;
    }
    map.segments.push_back(
        loader::ByteRangeSegment{
            .kind = loader::ByteRangeSegment::Kind::kPad,
            .dst_offset = offset,
            .length = length,
            .src_offset = 0,
            .source_index = 0,
        });
  };

  auto append_spans = [&](const std::vector<ByteSpan>& spans) {
    for (const auto& span : spans) {
      append_span(span.offset, span.length);
    }
  };

  for (const auto& item : plan.items) {
    switch (item.kind) {
      case RepresentationWorkItemKind::kConstFill: {
        if (!item.fill_rule.has_value()) {
          return absl::InvalidArgumentError("const fill work item requires fill_rule");
        }
        auto spans_or = build_destination_byte_spans(item.dst_spec, item.fill_rule->destination_range);
        if (!spans_or.ok()) {
          return spans_or.status();
        }
        append_spans(*spans_or);
        break;
      }
      case RepresentationWorkItemKind::kScalarBroadcastFill: {
        if (item.sources.empty()) {
          return absl::InvalidArgumentError("scalar fill work item requires a source fragment");
        }
        auto spans_or = build_destination_byte_spans(item.dst_spec, item.sources.front().fragment.destination_range);
        if (!spans_or.ok()) {
          return spans_or.status();
        }
        append_spans(*spans_or);
        break;
      }
      case RepresentationWorkItemKind::kPadFill: {
        for (const auto& segment : item.byte_range_map.segments) {
          append_span(segment.dst_offset, segment.length);
        }
        break;
      }
      case RepresentationWorkItemKind::kTensorCopy:
      case RepresentationWorkItemKind::kConcatAssemble:
      case RepresentationWorkItemKind::kExpertDim0Concat:
      case RepresentationWorkItemKind::kResidualByteRange:
        break;
    }
  }

  std::sort(map.segments.begin(), map.segments.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.dst_offset < rhs.dst_offset;
  });
  for (size_t index = 1; index < map.segments.size(); ++index) {
    const auto& previous = map.segments[index - 1];
    const auto& current = map.segments[index];
    if (current.dst_offset < previous.dst_offset + previous.length) {
      return absl::InvalidArgumentError("local typed work map contains overlapping segments");
    }
  }
  return map;
}

} // namespace

namespace pipeline = tensorcast::store::materialization::runtime::pipeline;
using materialization::control::MaterializeOrchestrator;

namespace {

constexpr uint32_t kDefaultTransportWaitTimeoutMs = 30000;
constexpr uint32_t kViewTransportProbeTimeoutMs = 1000;

size_t resolve_streaming_buffer_chunks_for_transfer(
    uint64_t transfer_bytes,
    size_t slice_bytes,
    size_t configured_chunks) {
  const size_t max_chunks = std::max<size_t>(1, configured_chunks);
  if (transfer_bytes == 0 || slice_bytes == 0) {
    return 1;
  }
  const uint64_t required_chunks = (transfer_bytes + static_cast<uint64_t>(slice_bytes) - 1) / slice_bytes;
  return static_cast<size_t>(std::min<uint64_t>(required_chunks, max_chunks));
}

bool is_local_identity(const components::WorkerIdentity& local) {
  return !local.node_id.empty() || !local.node_address.empty();
}

bool is_local_replica(const components::RemoteReplicaInfo& remote, const components::WorkerIdentity& local) {
  if (!is_local_identity(local)) {
    return false;
  }
  if (!local.node_id.empty() && !remote.node_id.empty()) {
    return local.node_id == remote.node_id;
  }
  if (!local.node_address.empty() && !remote.node_address.empty() && local.node_address == remote.node_address) {
    if (local.p2p_port == 0 || remote.node_port == 0) {
      return true;
    }
    return local.p2p_port == remote.node_port;
  }
  return false;
}

uint32_t clamp_timeout_to_u32_ms(std::chrono::milliseconds timeout) {
  if (timeout.count() <= 0) {
    return 0;
  }
  if (timeout.count() > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return std::numeric_limits<uint32_t>::max();
  }
  return static_cast<uint32_t>(timeout.count());
}

uint32_t resolve_transport_wait_timeout_ms(
    std::chrono::milliseconds configured_timeout,
    std::chrono::milliseconds request_budget) {
  std::chrono::milliseconds timeout =
      configured_timeout.count() > 0 ? configured_timeout : std::chrono::milliseconds(kDefaultTransportWaitTimeoutMs);
  if (request_budget.count() > 0) {
    timeout = std::min(timeout, request_budget);
  }
  if (timeout.count() <= 0) {
    return 1;
  }
  return std::max<uint32_t>(1, clamp_timeout_to_u32_ms(timeout));
}

uint32_t resolve_transport_wait_timeout_ms(const loading::MaterializeHints& hints) {
  return resolve_transport_wait_timeout_ms(hints.transport_wait_timeout, hints.request_budget);
}

uint32_t resolve_view_transport_probe_timeout_ms(uint32_t wait_timeout_ms) {
  if (wait_timeout_ms == 0) {
    return 0;
  }
  return std::min(wait_timeout_ms, kViewTransportProbeTimeoutMs);
}

std::string make_materialize_into_target_pinned_wait_context(
    const loading::MaterializeHints& hints,
    int target_device_ordinal,
    size_t num_chunks,
    size_t slice_bytes) {
  std::string context = absl::StrCat(
      "op=materialize_into_target",
      " artifact_id=",
      hints.artifact_id,
      " device_id=",
      target_device_ordinal,
      " chunks=",
      num_chunks,
      " slice_bytes=",
      slice_bytes,
      " pipeline_concurrency=",
      hints.pipeline_concurrency);
  if (!hints.transport_request_id.empty()) {
    absl::StrAppend(&context, " request_id=", hints.transport_request_id);
  }
  if (!hints.transport_requester_worker_id.empty()) {
    absl::StrAppend(&context, " requester=", hints.transport_requester_worker_id);
  }
  if (hints.transport_scheduling_group.has_value()) {
    const auto& group = *hints.transport_scheduling_group;
    if (!group.group_id.empty()) {
      absl::StrAppend(&context, " group_id=", group.group_id);
    }
    if (!group.part_id.empty()) {
      absl::StrAppend(&context, " part_id=", group.part_id);
    }
    if (!group.group_kind.empty()) {
      absl::StrAppend(&context, " group_kind=", group.group_kind);
    }
  }
  return context;
}

absl::Status stale_local_route_status(std::string_view artifact_id) {
  return absl::UnavailableError(
      absl::StrCat("Global Store route stale for artifact_id=", artifact_id, "; retry or provide disk source"));
}

std::optional<components::TransportSchedulingGroupHint> to_transport_scheduling_group_hint(
    const loading::MaterializeHints& hints) {
  if (!hints.transport_scheduling_group.has_value()) {
    return std::nullopt;
  }
  const auto& group = *hints.transport_scheduling_group;
  if (group.group_id.empty() || group.group_kind.empty() || group.part_id.empty()) {
    return std::nullopt;
  }
  components::TransportSchedulingGroupHint out;
  out.group_id = group.group_id;
  out.group_kind = group.group_kind;
  out.total_parts = group.total_parts;
  out.part_id = group.part_id;
  out.priority = group.priority;
  out.epoch = group.epoch;
  return out;
}

loading::ReplicaHandle build_local_replica_handle(
    const loading::ReplicaKey& key,
    const std::shared_ptr<replica::Replica>& replica,
    common::memory::MemoryLocation target_location) {
  loading::ReplicaHandle handle;
  handle.replica_key = key;
  handle.ready_signal = replica->ready_signal_for(target_location);
  handle.cpu_state = replica->get_memory_state(common::memory::MemoryLocation::CPU);
  handle.gpu_state = replica->get_memory_state(common::memory::MemoryLocation::GPU);
  handle.source = loading::MaterializationSource::kLocalReplica;
  if (target_location == common::memory::MemoryLocation::GPU) {
    const auto gpu_ptrs = replica->get_data_pointer(common::memory::MemoryLocation::GPU);
    handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;
    auto ipc_or = replica->get_memory_manager().get_ipc_handle();
    if (ipc_or.ok()) {
      handle.cuda_ipc_handle = cuda::IpcHandleBytes::from_native(*ipc_or);
    }
    return handle;
  }

  auto uma = replica->get_memory_manager().memory_authority();
  if (uma != nullptr) {
    const loading::ReplicaKey& allocation_key = replica->replica_key();
    auto region_or = uma->get_cpu_memfd_region(allocation_key);
    if (region_or.ok()) {
      handle.cpu_memfd_region = loading::CpuMemfdRegion{
          .fd = region_or->fd,
          .size_bytes = region_or->size_bytes,
          .offset_bytes = region_or->offset_bytes,
      };
    }
  }
  return handle;
}

absl::Status validate_existing_replica_for_reuse(
    const std::shared_ptr<replica::Replica>& replica,
    common::memory::MemoryLocation target_location) {
  if (target_location == common::memory::MemoryLocation::GPU) {
    const auto gpu_state = replica->get_memory_state(common::memory::MemoryLocation::GPU);
    const auto cpu_state = replica->get_memory_state(common::memory::MemoryLocation::CPU);
    if (gpu_state == replica::MemoryState::ALLOCATED && cpu_state != replica::MemoryState::LOADED) {
      return absl::NotFoundError("reuse replica has no loaded CPU source; rebuilding");
    }
  }

  auto ready_signal = replica->ready_signal_for(target_location);
  if (ready_signal && ready_signal->is_ready()) {
    const absl::Status ready_status = std::move(ready_signal->subscribe()).get();
    if (!ready_status.ok()) {
      if (absl::IsFailedPrecondition(ready_status) || absl::IsNotFound(ready_status)) {
        return absl::NotFoundError(ready_status.message());
      }
      return ready_status;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> compute_required_source_bytes_for_map(const loader::ByteRangeMap& map) {
  uint64_t required_bytes = 0;
  for (const auto& segment : map.segments) {
    if (segment.kind != loader::ByteRangeSegment::Kind::kData || segment.length == 0) {
      continue;
    }
    if (segment.source_index != 0) {
      return absl::FailedPreconditionError(
          absl::StrCat("local view map references unexpected source_index=", segment.source_index));
    }
    if (segment.src_offset > std::numeric_limits<uint64_t>::max() - segment.length) {
      return absl::OutOfRangeError("local view map source range overflows uint64_t");
    }
    required_bytes = std::max(required_bytes, segment.src_offset + segment.length);
  }
  return required_bytes;
}

absl::Status load_assembled_ranges_into_replica(
    const std::shared_ptr<replica::Replica>& replica,
    const loader::ByteRangeMap& map,
    std::vector<std::shared_ptr<loader::SeekableSource>> piece_sources,
    const StoreEngineOptions::ByteMappingConfig& byte_mapping_config,
    common::memory::MemoryLocation target_location,
    int concurrency,
    const std::optional<loader::ViewPlan>& target_view_plan,
    loading::TransformPlacement transform_placement,
    int target_device_id) {
  absl::Status release_status = replica->release_memory(target_location);
  if (!release_status.ok() && !absl::IsNotFound(release_status)) {
    return release_status;
  }

  loader::ByteRangeCompiler compiler(byte_mapping_config, "assembly");
  auto program_or = compiler.Compile(map);
  if (!program_or.ok()) {
    return program_or.status();
  }

  loader::ByteRangeMappedSource::Options map_opts{
      .path = "assembly",
      .enable_direct_write_at = byte_mapping_config.enable_direct_write_at,
  };
  auto source_or =
      loader::ByteRangeMappedSource::Create(map, *program_or, std::move(piece_sources), std::move(map_opts));
  if (!source_or.ok()) {
    return source_or.status();
  }

  std::unique_ptr<loader::SeekableSource> source = std::move(*source_or);
  auto load_future = replica->get_memory_manager().load_async_from_source(
      std::move(source), target_location, concurrency, std::nullopt, std::function<absl::Status()>{});
  absl::Status load_status = std::move(load_future).get();
  if (!load_status.ok()) {
    return load_status;
  }

  if (target_view_plan.has_value() && !target_view_plan->is_identity &&
      target_view_plan->transform.requires_materialization &&
      transform_placement == loading::TransformPlacement::kServer) {
    auto ptrs = replica->get_data_pointer(target_location);
    if (ptrs.empty() || ptrs[0] == nullptr) {
      return absl::FailedPreconditionError("assembly view transform requires loaded memory");
    }
    const int dev = target_location == common::memory::MemoryLocation::GPU ? target_device_id : -1;
    absl::Status transform_status =
        loader::execute_transform(target_view_plan->transform, target_location, ptrs[0], dev);
    if (!transform_status.ok()) {
      return absl::DataLossError(absl::StrCat("assembly view transform failed: ", transform_status.message()));
    }
  }

  replica->set_ready_signal(target_location, absl::OkStatus());
  return absl::OkStatus();
}

void fill_runtime_p2p_bindings(
    const std::shared_ptr<components::CommunicationManager>& comm_manager,
    P2PSource* source) {
  if (source == nullptr || !comm_manager || !comm_manager->is_enabled()) {
    return;
  }
  source->comm_engine =
      gsl::not_null<std::shared_ptr<communicator::engine::Communicator>>{comm_manager->get_shared_engine()};
  source->routing_context = comm_manager->routing_context();
}

absl::StatusOr<std::pair<std::string, std::string>> parse_mi2_multihashes(std::string_view artifact_id) {
  constexpr std::string_view kPrefix = common::kMi2Prefix;
  if (!artifact_id.starts_with(kPrefix)) {
    return absl::InvalidArgumentError("sealed_artifact_id must start with \"mi2:\"");
  }
  const size_t index_begin = kPrefix.size();
  const size_t sep = artifact_id.find(':', index_begin);
  if (sep == std::string_view::npos) {
    return absl::InvalidArgumentError("sealed_artifact_id must be of form mi2:<index_multihash>:<data_multihash>");
  }
  const std::string_view index_mh = artifact_id.substr(index_begin, sep - index_begin);
  const std::string_view data_mh = artifact_id.substr(sep + 1);
  if (index_mh.empty() || data_mh.empty()) {
    return absl::InvalidArgumentError("sealed_artifact_id must include index and data multihash components");
  }
  return std::make_pair(std::string(index_mh), std::string(data_mh));
}

absl::Status publish_sealed_artifact_metadata(
    components::IGlobalStoreClient& global_store_client,
    std::string_view sealed_artifact_id,
    std::string_view schema_version,
    std::string_view encoding,
    uint64_t total_size,
    std::string_view canonical_index_json) {
  if (sealed_artifact_id.empty()) {
    return absl::InvalidArgumentError("publish_sealed_artifact_metadata requires sealed_artifact_id");
  }
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("publish_sealed_artifact_metadata requires canonical_index_json");
  }
  auto multihashes_or = parse_mi2_multihashes(sealed_artifact_id);
  if (!multihashes_or.ok()) {
    return multihashes_or.status();
  }

  common::v1::ArtifactDescriptor descriptor;
  descriptor.set_artifact_id(std::string(sealed_artifact_id));
  descriptor.set_index_multihash(multihashes_or->first);
  descriptor.set_data_multihash(multihashes_or->second);
  descriptor.set_schema_version(std::string(schema_version));
  descriptor.set_encoding(std::string(encoding));
  descriptor.set_total_size(total_size);
  descriptor.set_id_kind(tensorcast::common::v1::ArtifactIdKind::ARTIFACT_ID_KIND_MI2);
  return global_store_client.upsert_artifact_metadata(descriptor, canonical_index_json);
}

absl::StatusOr<DeviceKey> select_seal_target_device(components::DeviceManager& device_manager) {
  if (device_manager.get_num_gpus() <= 0) {
    return absl::FailedPreconditionError("seal_assembly requires at least one GPU device");
  }
  return DeviceRegistry::instance().gpu_key(0);
}

} // namespace

class LocalReplicaSource final : public loader::SeekableSource {
 public:
  static absl::StatusOr<std::shared_ptr<loader::SeekableSource>> Create(
      std::shared_ptr<replica::Replica> replica,
      common::memory::MemoryLocation location,
      int device_id,
      uint64_t total_size) {
    if (!replica) {
      return absl::InvalidArgumentError("local replica source requires replica");
    }
    if (total_size == 0) {
      return absl::InvalidArgumentError("local replica source requires non-zero size");
    }
    std::shared_ptr<loader::SeekableSource> source;
    if (location == common::memory::MemoryLocation::GPU) {
      const auto gpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
      if (gpu_ptrs.empty() || gpu_ptrs[0] == nullptr) {
        return absl::FailedPreconditionError("local replica GPU pointer unavailable");
      }
      source = std::make_shared<loader::GpuMemorySource>(gsl::not_null<void*>{gpu_ptrs[0]}, device_id, total_size);
    } else {
      const auto cpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::CPU);
      if (cpu_ptrs.empty() || cpu_ptrs[0] == nullptr) {
        return absl::FailedPreconditionError("local replica CPU pointer unavailable");
      }
      source = std::make_shared<loader::CpuMemorySource>(gsl::not_null<const void*>{cpu_ptrs[0]}, total_size);
    }
    return std::shared_ptr<loader::SeekableSource>(new LocalReplicaSource(std::move(replica), std::move(source)));
  }

  [[nodiscard]] uint64_t total_bytes() const override {
    return source_->total_bytes();
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    return source_->read(dst, max_bytes);
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    return source_->read_at(offset, dst, bytes);
  }

  [[nodiscard]] bool supports_direct_write_at() const override {
    return source_->supports_direct_write_at();
  }

  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const DirectWriteGrant& grant) override {
    return source_->read_into_at(src_offset, dest_va_offset, bytes, grant);
  }

  [[nodiscard]] const uint8_t* cpu_base_ptr() const override {
    return source_->cpu_base_ptr();
  }

 private:
  LocalReplicaSource(std::shared_ptr<replica::Replica> replica, std::shared_ptr<loader::SeekableSource> source)
      : replica_(std::move(replica)), source_(std::move(source)) {}

  std::shared_ptr<replica::Replica> replica_;
  std::shared_ptr<loader::SeekableSource> source_;
};

absl::StatusOr<std::string> read_source_fully(loader::SeekableSource& source) {
  const auto total_bytes = source.total_bytes();
  if (total_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return absl::OutOfRangeError("artifact size exceeds host limits");
  }
  std::string payload;
  payload.resize(static_cast<size_t>(total_bytes));
  size_t copied = 0;
  while (copied < payload.size()) {
    auto read_or = source.read_at(copied, payload.data() + copied, payload.size() - copied);
    if (!read_or.ok()) {
      return read_or.status();
    }
    if (*read_or == 0) {
      return absl::DataLossError("artifact verification source terminated before expected size");
    }
    copied += *read_or;
  }
  return payload;
}

std::string compute_sha256_bytes(std::string_view payload) {
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return std::string(reinterpret_cast<const char*>(digest.data()), digest.size());
}

SemanticLayoutIdentity compute_canonical_index_semantic_layout_identity(std::string_view canonical_index_json) {
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(
          reinterpret_cast<const uint8_t*>(canonical_index_json.data()), canonical_index_json.size()));
  SemanticLayoutIdentity identity;
  identity.kind = SemanticLayoutKind::kCanonicalIndexDigest;
  identity.value = absl::AsciiStrToLower(
      absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size())));
  return identity;
}

struct VerifiedContentProjection {
  VerifiedContentDescriptor descriptor;
  VerificationRecord verification_record;
};

VerifiedContentDescriptor build_sealed_artifact_verified_content_descriptor(
    std::string_view index_multihash,
    std::string_view data_multihash,
    uint64_t total_size) {
  VerifiedContentDescriptor descriptor;
  descriptor.content_identity.semantic_layout_identity.kind = SemanticLayoutKind::kCanonicalIndexDigest;
  descriptor.content_identity.semantic_layout_identity.value = std::string(index_multihash);
  descriptor.content_identity.logical_size_bytes = total_size;
  descriptor.content_identity.digest_alg = "multihash";
  descriptor.content_identity.digest_bytes = std::string(data_multihash);
  return descriptor;
}

VerificationRecord build_seal_verification_record(std::string_view index_multihash) {
  return VerificationRecord{
      .verification_method = VerificationMethod::kSealCommit,
      .verified_at = absl::Now(),
      .layout_proof_kind = LayoutProofKind::kCanonicalIndexDigest,
      .layout_proof_value = std::string(index_multihash),
  };
}

absl::StatusOr<VerifiedContentProjection> build_verified_content_projection_from_replica(
    const std::shared_ptr<replica::Replica>& replica,
    common::memory::MemoryLocation location,
    std::string_view canonical_index_json,
    const std::optional<SemanticLayoutIdentity>& semantic_layout_identity_override) {
  auto size_or = replica->get_artifact_size();
  if (!size_or.ok()) {
    return size_or.status();
  }
  const int device_id = location == common::memory::MemoryLocation::GPU ? replica->replica_key().device.ordinal : -1;
  auto source_or = LocalReplicaSource::Create(replica, location, device_id, *size_or);
  if (!source_or.ok()) {
    return source_or.status();
  }
  auto payload_or = read_source_fully(**source_or);
  if (!payload_or.ok()) {
    return payload_or.status();
  }

  VerifiedContentProjection projection;
  projection.descriptor.content_identity.semantic_layout_identity = semantic_layout_identity_override.value_or(
      compute_canonical_index_semantic_layout_identity(canonical_index_json));
  projection.descriptor.content_identity.logical_size_bytes = *size_or;
  projection.descriptor.content_identity.digest_alg = "sha256";
  projection.descriptor.content_identity.digest_bytes = compute_sha256_bytes(*payload_or);
  projection.verification_record.verification_method = VerificationMethod::kSharedExecutorFullReadDigest;
  projection.verification_record.verified_at = absl::Now();
  projection.verification_record.layout_proof_kind = semantic_layout_identity_override.has_value()
      ? LayoutProofKind::kNamedLayoutId
      : LayoutProofKind::kCanonicalIndexDigest;
  projection.verification_record.layout_proof_value =
      projection.descriptor.content_identity.semantic_layout_identity.value;
  return projection;
}

class SharedSeekableSource final : public loader::SeekableSource {
 public:
  explicit SharedSeekableSource(std::shared_ptr<loader::SeekableSource> source) : source_(std::move(source)) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return source_->total_bytes();
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    return source_->read(dst, max_bytes);
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    return source_->read_at(offset, dst, bytes);
  }

  [[nodiscard]] const uint8_t* cpu_base_ptr() const override {
    return source_->cpu_base_ptr();
  }

  [[nodiscard]] bool supports_direct_write_at() const override {
    return source_->supports_direct_write_at();
  }

  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const DirectWriteGrant& grant) override {
    return source_->read_into_at(src_offset, dest_va_offset, bytes, grant);
  }

 private:
  std::shared_ptr<loader::SeekableSource> source_;
};

class SharedSourceLoader final : public IArtifactLoader {
 public:
  explicit SharedSourceLoader(std::shared_ptr<loader::SeekableSource> source) : source_(std::move(source)) {}

  absl::Status initialize() override {
    if (!source_) {
      return absl::FailedPreconditionError("shared source is unavailable");
    }
    return absl::OkStatus();
  }

  absl::StatusOr<uint64_t> get_artifact_size() override {
    if (!source_) {
      return absl::FailedPreconditionError("shared source is unavailable");
    }
    return source_->total_bytes();
  }

  absl::StatusOr<std::unique_ptr<loader::SeekableSource>> open_source() override {
    if (!source_) {
      return absl::FailedPreconditionError("shared source is unavailable");
    }
    return std::make_unique<SharedSeekableSource>(source_);
  }

 private:
  std::shared_ptr<loader::SeekableSource> source_;
};

class BoundCanonicalSource final : public loader::SeekableSource {
 public:
  explicit BoundCanonicalSource(
      std::vector<MaterializationFacade::SealAssemblyCutInput::BoundCanonicalSpan> spans,
      uint64_t total_bytes)
      : spans_(std::move(spans)), total_bytes_(total_bytes) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return total_bytes_;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(cursor_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or;
    }
    cursor_ += *read_or;
    return read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= total_bytes_) {
      return static_cast<size_t>(0);
    }
    uint64_t cursor = offset;
    size_t remaining = static_cast<size_t>(std::min<uint64_t>(bytes, total_bytes_ - offset));
    uint8_t* out = static_cast<uint8_t*>(dst);
    while (remaining > 0) {
      const auto* span = find_span(cursor);
      if (span == nullptr) {
        return absl::FailedPreconditionError("bound canonical source is missing byte coverage");
      }
      const uint64_t local_offset = cursor - span->logical_offset;
      const size_t take = static_cast<size_t>(std::min<uint64_t>(remaining, span->logical_length - local_offset));
      cuda::CudaDeviceGuard guard(span->device_id);
      if (!guard.status().ok()) {
        return guard.status();
      }
      const auto src_addr =
          static_cast<uintptr_t>(span->base_ptr + span->mapping_base_offset + span->storage_offset + local_offset);
      auto memcpy_status = cuda::memcpy(out, reinterpret_cast<const void*>(src_addr), take, cudaMemcpyDeviceToHost);
      if (!memcpy_status.ok()) {
        LOG(ERROR) << "bound canonical source memcpy failed"
                   << " cursor=" << cursor << " local_offset=" << local_offset << " take=" << take
                   << " span.logical_offset=" << span->logical_offset << " span.logical_length=" << span->logical_length
                   << " span.base_ptr=0x" << std::hex << span->base_ptr << " span.mapping_base_offset=0x"
                   << span->mapping_base_offset << " span.storage_offset=0x" << span->storage_offset << " src_addr=0x"
                   << src_addr << std::dec;
        return memcpy_status;
      }
      if (auto sync_status = cuda::device_synchronize(); !sync_status.ok()) {
        return sync_status;
      }
      out += take;
      cursor += take;
      remaining -= take;
    }
    return static_cast<size_t>(out - static_cast<uint8_t*>(dst));
  }

 private:
  const MaterializationFacade::SealAssemblyCutInput::BoundCanonicalSpan* find_span(uint64_t logical_offset) const {
    for (const auto& span : spans_) {
      if (logical_offset >= span.logical_offset && logical_offset < span.logical_offset + span.logical_length) {
        return &span;
      }
    }
    return nullptr;
  }

  std::vector<MaterializationFacade::SealAssemblyCutInput::BoundCanonicalSpan> spans_;
  uint64_t total_bytes_{0};
  uint64_t cursor_{0};
};

struct ContiguousBoundCanonicalGpuRange {
  int device_id{-1};
  uintptr_t base_addr{0};
  uint64_t total_bytes{0};
};

absl::StatusOr<std::optional<ContiguousBoundCanonicalGpuRange>> resolve_contiguous_bound_canonical_gpu_range(
    const MaterializationFacade::SealAssemblyCutInput& cut_input,
    uint64_t canonical_total_size) {
  if (cut_input.bound_canonical_spans.empty()) {
    return std::nullopt;
  }
  const auto& first = cut_input.bound_canonical_spans.front();
  const uintptr_t first_addr =
      static_cast<uintptr_t>(first.base_ptr + first.mapping_base_offset + first.storage_offset);
  int device_id = first.device_id;
  uint64_t expected_logical_offset = 0;
  uintptr_t expected_addr = first_addr;
  uint64_t covered = 0;
  for (const auto& span : cut_input.bound_canonical_spans) {
    if (span.device_id != device_id) {
      return std::nullopt;
    }
    if (span.logical_offset != expected_logical_offset) {
      return std::nullopt;
    }
    const uintptr_t span_addr = static_cast<uintptr_t>(span.base_ptr + span.mapping_base_offset + span.storage_offset);
    if (span_addr != expected_addr) {
      return std::nullopt;
    }
    expected_logical_offset += span.logical_length;
    expected_addr += span.logical_length;
    covered += span.logical_length;
  }
  if (covered != canonical_total_size) {
    return absl::FailedPreconditionError("bound canonical gpu range size does not match canonical total size");
  }
  return ContiguousBoundCanonicalGpuRange{
      .device_id = device_id,
      .base_addr = first_addr,
      .total_bytes = canonical_total_size,
  };
}

absl::StatusOr<std::shared_ptr<loader::SeekableSource>> make_bound_canonical_source(
    const MaterializationFacade::SealAssemblyCutInput& cut_input,
    uint64_t canonical_total_size) {
  if (cut_input.bound_canonical_spans.empty()) {
    return absl::FailedPreconditionError("bound canonical source spans are missing");
  }
  return std::shared_ptr<loader::SeekableSource>(
      std::make_shared<BoundCanonicalSource>(cut_input.bound_canonical_spans, canonical_total_size));
}

absl::StatusOr<uint64_t> compute_logical_total_size(std::string_view canonical_index_json) {
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("canonical index JSON must not be empty");
  }
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(canonical_index_json, nullptr, true);
  } catch (const std::exception& ex) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON: ", ex.what()));
  }
  if (!j.is_object()) {
    return absl::InvalidArgumentError("canonical index JSON must be an object");
  }
  uint64_t total_size = 0;
  for (auto it = j.begin(); it != j.end(); ++it) {
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() < 2) {
      continue;
    }
    const uint64_t offset = arr[0].get<uint64_t>();
    const uint64_t size = arr[1].get<uint64_t>();
    total_size = std::max<uint64_t>(total_size, offset + size);
  }
  if (total_size == 0) {
    return absl::InvalidArgumentError("canonical index total_size is zero");
  }
  return total_size;
}

struct AssemblyTargetRange {
  enum class Kind : uint8_t { kData = 0, kPad = 1 };
  Kind kind{Kind::kData};
  uint64_t canonical_offset{0};
  uint64_t target_offset{0};
  uint64_t length{0};
};

struct AssemblySourceInfo {
  std::string view_id;
  components::TransportSession session;
  uint64_t view_size_bytes{0};
  components::TransportLease transport_lease;
  loader::TransformPlan inverse_transform;
};

struct AssemblyPlan {
  loader::ByteRangeMap map;
  std::vector<AssemblySourceInfo> sources;
  std::vector<view::CanonicalRange> missing_ranges;
};

struct PieceInterval {
  uint64_t canonical_offset{0};
  uint64_t length{0};
  size_t source_index{0};
  uint64_t view_offset{0};
};

std::vector<AssemblyTargetRange> build_target_ranges_from_view_plan(const loader::ViewPlan& plan) {
  std::vector<AssemblyTargetRange> ranges;
  ranges.reserve(plan.selection.map.segments.size());
  for (const auto& range : plan.selection.map.segments) {
    if (range.length == 0) {
      continue;
    }
    AssemblyTargetRange out;
    out.target_offset = range.dst_offset;
    out.length = range.length;
    if (range.kind == loader::ByteRangeSegment::Kind::kPad) {
      out.kind = AssemblyTargetRange::Kind::kPad;
    } else {
      out.kind = AssemblyTargetRange::Kind::kData;
      out.canonical_offset = range.src_offset;
    }
    ranges.push_back(out);
  }
  return ranges;
}

absl::StatusOr<std::vector<AssemblyTargetRange>> build_target_ranges_for_canonical(
    std::string_view canonical_index_json,
    uint64_t total_size) {
  std::vector<AssemblyTargetRange> ranges;
  auto map_or = loader::build_byte_range_map_from_canonical_index_json(canonical_index_json, total_size);
  if (!map_or.ok()) {
    return map_or.status();
  }
  ranges.reserve(map_or->segments.size());
  for (const auto& segment : map_or->segments) {
    if (segment.length == 0) {
      continue;
    }
    AssemblyTargetRange out;
    out.target_offset = segment.dst_offset;
    out.length = segment.length;
    if (segment.kind == loader::ByteRangeSegment::Kind::kPad) {
      out.kind = AssemblyTargetRange::Kind::kPad;
    } else {
      out.kind = AssemblyTargetRange::Kind::kData;
      out.canonical_offset = segment.dst_offset;
    }
    ranges.push_back(out);
  }
  return ranges;
}

void coalesce_missing_ranges(std::vector<view::CanonicalRange>* ranges) {
  if (!ranges) {
    return;
  }
  auto& vec = *ranges;
  if (vec.empty()) {
    return;
  }
  std::sort(vec.begin(), vec.end(), [](const auto& lhs, const auto& rhs) { return lhs.offset < rhs.offset; });
  std::vector<view::CanonicalRange> merged;
  merged.reserve(vec.size());
  for (const auto& range : vec) {
    if (range.length == 0) {
      continue;
    }
    if (merged.empty()) {
      merged.push_back(range);
      continue;
    }
    auto& last = merged.back();
    const uint64_t last_end = last.offset + last.length;
    if (range.offset <= last_end) {
      const uint64_t new_end = std::max(last_end, range.offset + range.length);
      last.length = new_end - last.offset;
    } else {
      merged.push_back(range);
    }
  }
  vec = std::move(merged);
}

std::string format_missing_ranges(absl::Span<const view::CanonicalRange> ranges, size_t limit = 5) {
  std::string out;
  size_t count = 0;
  for (const auto& range : ranges) {
    if (count++ > 0) {
      absl::StrAppend(&out, ", ");
    }
    absl::StrAppend(&out, "[", range.offset, "+", range.length, ")");
    if (count >= limit) {
      break;
    }
  }
  if (ranges.size() > limit) {
    absl::StrAppend(&out, ", ...");
  }
  return out;
}

absl::Status build_assembly_map(
    absl::Span<const AssemblyTargetRange> target_ranges,
    const std::vector<PieceInterval>& intervals,
    uint32_t num_sources,
    uint64_t target_total_size,
    loader::ByteRangeMap* out_map,
    std::vector<view::CanonicalRange>* missing_ranges) {
  if (out_map == nullptr) {
    return absl::InvalidArgumentError("assembly map output must not be null");
  }
  if (missing_ranges != nullptr) {
    missing_ranges->clear();
  }

  loader::ByteRangeMap map;
  map.total_bytes = target_total_size;
  map.num_sources = num_sources;
  map.segments.reserve(target_ranges.size() * 2 + intervals.size());

  std::vector<uint64_t> interval_starts;
  interval_starts.reserve(intervals.size());
  for (const auto& interval : intervals) {
    interval_starts.push_back(interval.canonical_offset);
  }

  auto find_interval_index = [&](uint64_t offset) -> size_t {
    if (intervals.empty()) {
      return 0;
    }
    auto it = std::upper_bound(interval_starts.begin(), interval_starts.end(), offset);
    if (it == interval_starts.begin()) {
      return 0;
    }
    return static_cast<size_t>(it - interval_starts.begin() - 1);
  };

  for (const auto& target : target_ranges) {
    if (target.length == 0) {
      continue;
    }
    if (target.kind == AssemblyTargetRange::Kind::kPad) {
      map.segments.push_back(
          loader::ByteRangeSegment{
              .kind = loader::ByteRangeSegment::Kind::kPad,
              .dst_offset = target.target_offset,
              .length = target.length,
              .src_offset = 0,
              .source_index = 0,
          });
      continue;
    }

    uint64_t cursor = target.canonical_offset;
    const uint64_t end = target.canonical_offset + target.length;
    size_t idx = find_interval_index(cursor);
    while (cursor < end) {
      while (idx < intervals.size() && intervals[idx].canonical_offset + intervals[idx].length <= cursor) {
        ++idx;
      }
      if (idx >= intervals.size() || intervals[idx].canonical_offset > cursor) {
        const uint64_t gap_end = (idx < intervals.size()) ? std::min(end, intervals[idx].canonical_offset) : end;
        if (missing_ranges != nullptr && gap_end > cursor) {
          missing_ranges->push_back(view::CanonicalRange{.offset = cursor, .length = gap_end - cursor});
        }
        cursor = gap_end;
        continue;
      }

      const auto& interval = intervals[idx];
      const uint64_t interval_end = interval.canonical_offset + interval.length;
      const uint64_t take_end = std::min(interval_end, end);
      const uint64_t take_len = take_end - cursor;
      map.segments.push_back(
          loader::ByteRangeSegment{
              .kind = loader::ByteRangeSegment::Kind::kData,
              .dst_offset = target.target_offset + (cursor - target.canonical_offset),
              .length = take_len,
              .src_offset = interval.view_offset + (cursor - interval.canonical_offset),
              .source_index = static_cast<uint32_t>(interval.source_index),
          });
      cursor = take_end;
    }
  }

  coalesce_missing_ranges(missing_ranges);
  if (missing_ranges != nullptr && !missing_ranges->empty()) {
    *out_map = std::move(map);
    return absl::OkStatus();
  }

  auto normalized_or = normalize_byte_range_map(std::move(map));
  if (!normalized_or.ok()) {
    return normalized_or.status();
  }
  *out_map = std::move(*normalized_or);
  return absl::OkStatus();
}

absl::StatusOr<AssemblyPlan> build_assembly_plan_from_views(
    components::IGlobalStoreClient& gs_client,
    std::string_view assembly_id,
    std::string_view canonical_index_json,
    uint64_t canonical_total_size,
    absl::Span<const AssemblyTargetRange> target_ranges,
    uint64_t target_total_size,
    const DeviceKey& target_device,
    const components::WorkerIdentity& local_identity,
    std::vector<components::ViewInfo> views) {
  if (views.empty()) {
    return absl::NotFoundError(absl::StrCat("no views found for assembly_id=", assembly_id));
  }
  std::sort(views.begin(), views.end(), [](const components::ViewInfo& a, const components::ViewInfo& b) {
    return a.view_id < b.view_id;
  });

  std::vector<AssemblySourceInfo> sources;
  std::vector<PieceInterval> intervals;

  for (const auto& view : views) {
    if (view.view_id.empty()) {
      continue;
    }
    if (view.canonical_ranges.empty()) {
      return absl::FailedPreconditionError(absl::StrCat("coverage metadata missing for view_id=", view.view_id));
    }
    if (view.canonical_size_bytes > 0 && canonical_total_size > 0 &&
        view.canonical_size_bytes != canonical_total_size) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "canonical size mismatch for view_id=",
              view.view_id,
              " expected=",
              canonical_total_size,
              " got=",
              view.canonical_size_bytes));
    }

    auto parsed_or = view::parse_view_selection_json(view.view_spec_json);
    if (!parsed_or.ok()) {
      return parsed_or.status();
    }
    auto plan_or = parsed_or->tensor_names.empty()
        ? loader::ViewPlanner::compute_bidirectional_view_plan(canonical_index_json, parsed_or->spec)
        : loader::ViewPlanner::compute_bidirectional_view_plan(
              canonical_index_json, parsed_or->spec, parsed_or->tensor_names);
    if (!plan_or.ok()) {
      return plan_or.status();
    }
    const auto& forward = plan_or->forward;
    loader::TransformPlan inverse_transform = plan_or->inverse_transform;
    if (forward.transform.requires_materialization) {
      std::unordered_map<std::string, uint64_t> view_offsets;
      view_offsets.reserve(forward.transform.tensors.size());
      for (const auto& tensor_plan : forward.transform.tensors) {
        view_offsets.emplace(tensor_plan.tensor_name, tensor_plan.dst_offset);
      }
      for (auto& tensor_plan : inverse_transform.tensors) {
        auto it = view_offsets.find(tensor_plan.tensor_name);
        if (it == view_offsets.end()) {
          return absl::FailedPreconditionError(
              absl::StrCat("missing transpose dst_offset for tensor ", tensor_plan.tensor_name));
        }
        tensor_plan.dst_offset = it->second;
        tensor_plan.storage_offset_elements = 0;
      }
    }

    auto session_or = gs_client.request_view_transport(
        assembly_id,
        view.view_id,
        local_identity.node_id,
        local_identity.node_address,
        local_identity.p2p_port,
        target_device,
        resolve_transport_wait_timeout_ms(
            std::chrono::milliseconds(kDefaultTransportWaitTimeoutMs), std::chrono::milliseconds(0)));
    if (!session_or.ok()) {
      if (absl::IsNotFound(session_or.status())) {
        continue;
      }
      return session_or.status();
    }
    auto session = std::move(*session_or);
    components::TransportLease transport_lease(&gs_client, session.transport_id);
    if (view.view_size_bytes > 0 && session.remote_replica.memory_size != view.view_size_bytes) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "view size mismatch for view_id=",
              view.view_id,
              " expected=",
              view.view_size_bytes,
              " got=",
              session.remote_replica.memory_size));
    }

    const size_t source_index = sources.size();
    sources.push_back(
        AssemblySourceInfo{
            .view_id = view.view_id,
            .session = std::move(session),
            .view_size_bytes = view.view_size_bytes,
            .transport_lease = std::move(transport_lease),
            .inverse_transform = std::move(inverse_transform),
        });

    for (const auto& range : forward.selection.map.segments) {
      if (range.kind != loader::ByteRangeSegment::Kind::kData || range.length == 0) {
        continue;
      }
      intervals.push_back(
          PieceInterval{
              .canonical_offset = range.src_offset,
              .length = range.length,
              .source_index = source_index,
              .view_offset = range.dst_offset});
    }
  }

  if (sources.empty()) {
    return absl::NotFoundError(absl::StrCat("no available piece replicas for assembly_id=", assembly_id));
  }

  std::sort(intervals.begin(), intervals.end(), [](const PieceInterval& a, const PieceInterval& b) {
    if (a.canonical_offset != b.canonical_offset) {
      return a.canonical_offset < b.canonical_offset;
    }
    return a.source_index < b.source_index;
  });
  std::vector<PieceInterval> resolved;
  resolved.reserve(intervals.size());
  for (const auto& interval : intervals) {
    if (interval.length == 0) {
      continue;
    }
    if (resolved.empty()) {
      resolved.push_back(interval);
      continue;
    }
    auto& prev = resolved.back();
    const uint64_t prev_end = prev.canonical_offset + prev.length;
    const uint64_t cur_end = interval.canonical_offset + interval.length;
    if (interval.canonical_offset >= prev_end) {
      resolved.push_back(interval);
      continue;
    }
    if (interval.source_index == prev.source_index) {
      prev.length = std::max(prev_end, cur_end) - prev.canonical_offset;
      continue;
    }
    // Overlaps are permitted only when Global Store has validated equality proofs
    // (v2 LayoutSpec REPLICATE_EQUAL). Choose deterministically: lower view_id
    // (lexicographic) wins because sources are ordered by view_id.
    if (cur_end <= prev_end) {
      continue;
    }
    PieceInterval trimmed = interval;
    trimmed.view_offset += prev_end - interval.canonical_offset;
    trimmed.canonical_offset = prev_end;
    trimmed.length = cur_end - prev_end;
    resolved.push_back(std::move(trimmed));
  }

  AssemblyPlan plan;
  plan.sources = std::move(sources);
  const uint32_t num_sources = static_cast<uint32_t>(plan.sources.size());
  absl::Status map_status =
      build_assembly_map(target_ranges, resolved, num_sources, target_total_size, &plan.map, &plan.missing_ranges);
  if (!map_status.ok()) {
    return map_status;
  }
  return plan;
}

absl::StatusOr<AssemblyPlan> build_assembly_plan(
    components::IGlobalStoreClient& gs_client,
    std::string_view assembly_id,
    std::string_view canonical_index_json,
    uint64_t canonical_total_size,
    absl::Span<const AssemblyTargetRange> target_ranges,
    uint64_t target_total_size,
    const DeviceKey& target_device,
    const components::WorkerIdentity& local_identity,
    const absl::flat_hash_set<absl::string_view>* allowed_view_ids) {
  auto views_or = gs_client.list_views(assembly_id);
  if (!views_or.ok()) {
    return views_or.status();
  }
  std::vector<components::ViewInfo> filtered_views;
  filtered_views.reserve(views_or->size());
  for (const auto& view : *views_or) {
    if (view.view_id.empty()) {
      continue;
    }
    if (allowed_view_ids != nullptr && !allowed_view_ids->contains(view.view_id)) {
      continue;
    }
    filtered_views.push_back(view);
  }
  return build_assembly_plan_from_views(
      gs_client,
      assembly_id,
      canonical_index_json,
      canonical_total_size,
      target_ranges,
      target_total_size,
      target_device,
      local_identity,
      std::move(filtered_views));
}

absl::StatusOr<std::shared_ptr<loader::SeekableSource>> make_local_piece_source(
    components::ReplicaRegistry& registry,
    std::string_view assembly_id,
    std::string_view view_id,
    const DeviceKey& device,
    common::memory::MemoryLocation location,
    int device_id,
    uint64_t view_size_bytes) {
  loading::ReplicaKey key;
  key.artifact_id = std::string(assembly_id);
  key.view_id = std::string(view_id);
  key.device = device;
  key.replica = 0;
  auto replica_or = registry.find(key);
  if (!replica_or.ok()) {
    if (!absl::IsNotFound(replica_or.status())) {
      return replica_or.status();
    }
    const auto candidates = registry.find_by_artifact(assembly_id);
    for (const auto& candidate : candidates) {
      if (!candidate.view_id.has_value()) {
        continue;
      }
      if (candidate.view_id.value() != view_id) {
        continue;
      }
      auto candidate_or = registry.find(candidate);
      if (candidate_or.ok()) {
        replica_or = std::move(candidate_or);
        break;
      }
    }
  }
  if (!replica_or.ok()) {
    return absl::NotFoundError(absl::StrCat("local replica missing for view_id=", view_id));
  }
  return LocalReplicaSource::Create(std::move(*replica_or), location, device_id, view_size_bytes);
}

absl::StatusOr<std::shared_ptr<loader::SeekableSource>> make_remote_piece_source(
    components::CommunicationManager& comm_manager,
    const components::RemoteReplicaInfo& remote,
    std::string_view view_id,
    std::string_view local_endpoint_id,
    std::chrono::milliseconds request_budget,
    std::string_view artifact_id_for_diagnostics) {
  if (remote.remote_memory_keys.empty()) {
    return absl::FailedPreconditionError(absl::StrCat("remote memory keys missing for view_id=", view_id));
  }
  if (remote.buffer_sizes.size() != remote.remote_memory_keys.size()) {
    return absl::FailedPreconditionError(absl::StrCat("buffer size mismatch for view_id=", view_id));
  }
  std::vector<size_t> buffer_sizes;
  buffer_sizes.reserve(remote.buffer_sizes.size());
  for (uint64_t size : remote.buffer_sizes) {
    buffer_sizes.push_back(static_cast<size_t>(size));
  }
  loader::RemoteKeySource::Options opts{
      .comm_engine =
          gsl::not_null<std::shared_ptr<tensorcast::communicator::engine::Communicator>>{
              comm_manager.get_shared_engine()},
      .memory_keys = remote.remote_memory_keys,
      .buffer_sizes = std::move(buffer_sizes),
      .ip = remote.node_address,
      .port = static_cast<uint16_t>(remote.node_port),
      .local_endpoint_id = std::string(local_endpoint_id),
      .remote_endpoint_id = remote.endpoint_id,
      .routing_context = comm_manager.routing_context(),
      .total_size = remote.memory_size,
      .request_budget = request_budget,
      .artifact_id = std::string(artifact_id_for_diagnostics),
  };
  return std::shared_ptr<loader::SeekableSource>(std::make_shared<loader::RemoteKeySource>(std::move(opts)));
}

absl::StatusOr<std::shared_ptr<loader::SeekableSource>> resolve_assembly_piece_source(
    components::ReplicaRegistry& registry,
    std::string_view assembly_id,
    const AssemblySourceInfo& source,
    const components::WorkerIdentity& local_identity,
    const DeviceKey& target_device,
    std::string_view local_endpoint_id,
    const std::shared_ptr<components::CommunicationManager>& comm_manager,
    std::chrono::milliseconds request_budget,
    std::string_view artifact_id_for_diagnostics) {
  const auto& remote = source.session.remote_replica;
  const bool comm_enabled = comm_manager != nullptr && comm_manager->is_enabled();
  if (is_local_replica(remote, local_identity)) {
    DeviceKey local_device;
    if (remote.memory_type == common::memory::MemoryLocation::CPU) {
      local_device = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
    } else {
      const int local_device_id = remote.device_id >= 0 ? remote.device_id : target_device.ordinal;
      local_device = DeviceRegistry::instance().gpu_key(local_device_id);
    }
    auto local_or = make_local_piece_source(
        registry,
        assembly_id,
        source.view_id,
        local_device,
        remote.memory_type,
        remote.device_id,
        source.view_size_bytes);
    if (local_or.ok()) {
      return local_or;
    }
    if (!absl::IsNotFound(local_or.status()) || !comm_enabled) {
      return local_or.status();
    }
    return make_remote_piece_source(
        *comm_manager, remote, source.view_id, local_endpoint_id, request_budget, artifact_id_for_diagnostics);
  }
  if (!comm_enabled) {
    return absl::FailedPreconditionError("Communication not enabled");
  }
  return make_remote_piece_source(
      *comm_manager, remote, source.view_id, local_endpoint_id, request_budget, artifact_id_for_diagnostics);
}

absl::StatusOr<std::shared_ptr<loader::SeekableSource>> make_local_canonical_source(
    components::ReplicaRegistry& registry,
    std::string_view artifact_id,
    const DeviceKey& target_device,
    uint64_t source_size) {
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("local canonical source requires artifact_id");
  }
  if (source_size == 0) {
    return absl::InvalidArgumentError("local canonical source requires non-zero source_size");
  }

  const auto candidates = registry.find_by_artifact(artifact_id);
  std::shared_ptr<replica::Replica> selected_replica;
  common::memory::MemoryLocation selected_location = common::memory::MemoryLocation::NONE;
  int selected_device_id = -1;
  int selected_score = -1;

  for (const auto& candidate : candidates) {
    if (candidate.view_id.has_value()) {
      continue;
    }

    auto replica_or = registry.find(candidate);
    if (!replica_or.ok()) {
      continue;
    }
    const auto& replica = *replica_or;
    auto candidate_size_or = replica->get_artifact_size();
    if (!candidate_size_or.ok() || source_size > *candidate_size_or) {
      continue;
    }

    if (candidate.device.type == DeviceType::GPU &&
        replica->get_memory_state(common::memory::MemoryLocation::GPU) == replica::MemoryState::LOADED) {
      const auto gpu_ptrs = replica->get_data_pointer(common::memory::MemoryLocation::GPU);
      if (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) {
        int score = 1;
        if (candidate.device.ordinal == target_device.ordinal) {
          score = 3;
        }
        if (score > selected_score) {
          selected_replica = replica;
          selected_location = common::memory::MemoryLocation::GPU;
          selected_device_id = candidate.device.ordinal;
          selected_score = score;
        }
      }
    }

    if (candidate.device.type == DeviceType::CPU &&
        replica->get_memory_state(common::memory::MemoryLocation::CPU) == replica::MemoryState::LOADED) {
      const auto cpu_ptrs = replica->get_data_pointer(common::memory::MemoryLocation::CPU);
      if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr) {
        constexpr int kCpuScore = 2;
        if (kCpuScore > selected_score) {
          selected_replica = replica;
          selected_location = common::memory::MemoryLocation::CPU;
          selected_device_id = -1;
          selected_score = kCpuScore;
        }
      }
    }
  }

  if (selected_replica == nullptr || selected_location == common::memory::MemoryLocation::NONE) {
    return absl::NotFoundError(absl::StrCat("local canonical replica missing for artifact_id=", artifact_id));
  }
  return LocalReplicaSource::Create(std::move(selected_replica), selected_location, selected_device_id, source_size);
}

absl::StatusOr<std::shared_ptr<loader::SeekableSource>> make_local_canonical_source_from_ready_handle(
    components::ReplicaRegistry& registry,
    const loading::ReplicaHandle& handle,
    const DeviceKey& target_device,
    uint64_t source_size) {
  const absl::Status ready_status = handle.wait_ready(std::chrono::milliseconds(0));
  if (!ready_status.ok()) {
    return ready_status;
  }

  auto replica_or = registry.find(handle.replica_key);
  if (replica_or.ok()) {
    const auto& replica = *replica_or;
    auto candidate_size_or = replica->get_artifact_size();
    if (candidate_size_or.ok() && source_size <= *candidate_size_or) {
      if (handle.replica_key.device.type == DeviceType::GPU &&
          replica->get_memory_state(common::memory::MemoryLocation::GPU) == replica::MemoryState::LOADED) {
        const auto gpu_ptrs = replica->get_data_pointer(common::memory::MemoryLocation::GPU);
        if (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) {
          return LocalReplicaSource::Create(
              replica, common::memory::MemoryLocation::GPU, handle.replica_key.device.ordinal, source_size);
        }
      }
      if (handle.replica_key.device.type == DeviceType::CPU &&
          replica->get_memory_state(common::memory::MemoryLocation::CPU) == replica::MemoryState::LOADED) {
        const auto cpu_ptrs = replica->get_data_pointer(common::memory::MemoryLocation::CPU);
        if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr) {
          return LocalReplicaSource::Create(replica, common::memory::MemoryLocation::CPU, -1, source_size);
        }
      }
    }
  }

  return make_local_canonical_source(registry, handle.replica_key.artifact_id, target_device, source_size);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::materialize_view_from_local_canonical(
    const loading::MaterializationRequest& request) {
  if (!request.requested_view_id().has_value()) {
    return absl::NotFoundError("no requested view id");
  }
  const auto& view_id = *request.requested_view_id();
  if (view_id.empty()) {
    return absl::InvalidArgumentError("requested view_id must be non-empty");
  }
  if (!request.hints().variant.has_value() || !request.hints().variant->cached_plan.has_value()) {
    return absl::NotFoundError("local view materialization requires cached view plan");
  }

  const auto& target_view_plan = *request.hints().variant->cached_plan;
  if (target_view_plan.is_identity) {
    return absl::NotFoundError("requested view plan is identity");
  }
  if (target_view_plan.selection.map.total_bytes == 0) {
    return absl::FailedPreconditionError("requested view plan has zero bytes");
  }
  loader::ByteRangeMap local_map = target_view_plan.selection.map;
  local_map.num_sources = 1;
  VLOG(1) << "materialize_view.local_canonical_map"
          << " artifact_id=" << request.canonical_artifact_id() << " view_id=" << view_id
          << " total_bytes=" << local_map.total_bytes << " segments=" << local_map.segments.size()
          << " source_count=" << local_map.num_sources << " target_is_gpu=" << request.target_is_gpu();
  auto required_source_bytes_or = compute_required_source_bytes_for_map(local_map);
  if (!required_source_bytes_or.ok()) {
    return required_source_bytes_or.status();
  }
  const uint64_t required_source_bytes = *required_source_bytes_or;

  auto& replica_registry = config_.replica_runtime->registry();
  const auto candidates = replica_registry.find_by_artifact(request.canonical_artifact_id());
  int canonical_candidates = 0;

  std::optional<loading::ReplicaKey> selected_key;
  std::shared_ptr<replica::Replica> selected_replica;
  common::memory::MemoryLocation selected_location = common::memory::MemoryLocation::NONE;
  int selected_device_id = -1;
  int selected_score = -1;
  uint64_t selected_source_size = 0;

  auto consider_candidate = [&](const loading::ReplicaKey& candidate,
                                const std::shared_ptr<replica::Replica>& replica,
                                common::memory::MemoryLocation location,
                                uint64_t source_size,
                                int score) {
    if (score <= selected_score) {
      return;
    }
    selected_key = candidate;
    selected_replica = replica;
    selected_location = location;
    selected_device_id = location == common::memory::MemoryLocation::GPU ? candidate.device.ordinal : -1;
    selected_source_size = source_size;
    selected_score = score;
  };

  for (const auto& candidate : candidates) {
    if (candidate.view_id.has_value()) {
      continue;
    }
    ++canonical_candidates;

    auto replica_or = replica_registry.find(candidate);
    if (!replica_or.ok()) {
      continue;
    }
    const auto& replica = *replica_or;
    auto source_size_or = replica->get_artifact_size();
    if (!source_size_or.ok()) {
      continue;
    }
    const uint64_t source_size = *source_size_or;
    if (required_source_bytes > source_size) {
      VLOG(1) << "materialize_view.local_canonical_skip artifact_id=" << request.canonical_artifact_id()
              << " view_id=" << view_id << " candidate=" << candidate
              << " required_source_bytes=" << required_source_bytes << " candidate_source_bytes=" << source_size;
      continue;
    }

    if (candidate.device.type == DeviceType::GPU &&
        replica->get_memory_state(common::memory::MemoryLocation::GPU) == replica::MemoryState::LOADED) {
      const auto gpu_ptrs = replica->get_data_pointer(common::memory::MemoryLocation::GPU);
      if (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) {
        int score = 1;
        if (request.target_is_gpu() && candidate.device.ordinal == request.target_device().ordinal) {
          score = 3;
        }
        consider_candidate(candidate, replica, common::memory::MemoryLocation::GPU, source_size, score);
      }
    }

    if (candidate.device.type == DeviceType::CPU &&
        replica->get_memory_state(common::memory::MemoryLocation::CPU) == replica::MemoryState::LOADED) {
      const auto cpu_ptrs = replica->get_data_pointer(common::memory::MemoryLocation::CPU);
      if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr) {
        const int score = request.target_is_gpu() ? 2 : 3;
        consider_candidate(candidate, replica, common::memory::MemoryLocation::CPU, source_size, score);
      }
    }
  }

  if (!selected_key.has_value() || selected_replica == nullptr) {
    return absl::NotFoundError(
        absl::StrCat(
            "no loaded canonical source for local view materialization: artifact_id=",
            request.canonical_artifact_id(),
            ", view_id=",
            view_id,
            ", canonical_candidates=",
            canonical_candidates));
  }

  VLOG(1) << "materialize_view.local_canonical_selected"
          << " artifact_id=" << request.canonical_artifact_id() << " view_id=" << view_id
          << " source_key=" << *selected_key << " selected_location=" << static_cast<int>(selected_location)
          << " selected_device_id=" << selected_device_id << " source_size=" << selected_source_size
          << " target_device=" << request.target_device().ordinal;

  const uint64_t source_size = selected_source_size;

  const int concurrency = loading::resolve_materialization_concurrency(config_.num_threads, request.hints());
  const loading::TransformPlacement transform_placement =
      request.hints().variant ? request.hints().variant->placement : loading::TransformPlacement::kServer;

  auto build_sources = [&]() -> absl::StatusOr<std::vector<std::shared_ptr<loader::SeekableSource>>> {
    auto source_or = LocalReplicaSource::Create(selected_replica, selected_location, selected_device_id, source_size);
    if (!source_or.ok()) {
      return source_or.status();
    }
    std::vector<std::shared_ptr<loader::SeekableSource>> sources;
    sources.push_back(*source_or);
    return sources;
  };

  loading::ReplicaKey key = request.replica_key();

  auto build_local_view_handle = [&](const std::shared_ptr<replica::Replica>& replica_instance) {
    loading::ReplicaHandle handle = build_local_replica_handle(key, replica_instance, request.target_location());
    const auto& replica_view_plan = replica_instance->view_plan();
    const loader::ViewPlan* effective_plan = nullptr;
    if (replica_view_plan.has_value() && !replica_view_plan->is_identity) {
      effective_plan = &*replica_view_plan;
    } else if (!target_view_plan.is_identity) {
      effective_plan = &target_view_plan;
    }
    if (effective_plan != nullptr) {
      handle.view_index_json = effective_plan->view_index_json;
      if (request.hints().need_view_data_hash && effective_plan->view_size_bytes > 0) {
        auto computer = config_.runtime_context->view_hash_computer();
        if (computer) {
          auto hash = computer->hash_replica_view(
              *replica_instance,
              request.target_location(),
              effective_plan->view_size_bytes,
              request.target_is_gpu() ? std::optional<int>(request.target_device().ordinal) : std::nullopt);
          if (hash.has_value()) {
            handle.view_data_hash = std::move(hash);
          }
        }
      }
    }
    return handle;
  };

  auto existing_or = replica_registry.find(key);
  if (existing_or.ok()) {
    const auto& existing = existing_or.value();
    absl::Status reuse_status = validate_existing_replica_for_reuse(existing, request.target_location());
    if (reuse_status.ok()) {
      LOG(INFO) << "materialize_view.local_canonical_reuse artifact_id=" << request.canonical_artifact_id()
                << " view_id=" << view_id << " source_key=" << *selected_key;
      return build_local_view_handle(existing);
    }
    if (!absl::IsNotFound(reuse_status)) {
      return reuse_status;
    }

    auto sources_or = build_sources();
    if (!sources_or.ok()) {
      return sources_or.status();
    }
    absl::Status rebuild_status = load_assembled_ranges_into_replica(
        existing,
        local_map,
        std::move(*sources_or),
        config_.options->byte_mapping,
        request.target_location(),
        concurrency,
        target_view_plan,
        transform_placement,
        request.target_device().ordinal);
    if (!rebuild_status.ok()) {
      return rebuild_status;
    }
    LOG(INFO) << "materialize_view.local_canonical_rebuild artifact_id=" << request.canonical_artifact_id()
              << " view_id=" << view_id << " source_key=" << *selected_key;
    return build_local_view_handle(existing);
  }
  if (!absl::IsNotFound(existing_or.status())) {
    return existing_or.status();
  }

  loading::InlineBufferSource inline_source{.data = nullptr, .size_bytes = local_map.total_bytes};
  replica::ReplicaConfig cfg{
      .source = inline_source,
      .artifact_identifier = key.artifact_id,
      .device_type = key.device.type,
      .local_device_id = key.device.type == DeviceType::GPU ? key.device.ordinal : -1,
      .pinned_buffer_pool = config_.runtime_context->pinned_buffer_pool(),
      .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{config_.runtime_context->async_runtime()},
      .artifact_chunk_bytes = config_.artifact_chunk_bytes,
      .expected_artifact_size = local_map.total_bytes,
      .view_plan = target_view_plan,
      .byte_mapping_config = config_.options->byte_mapping,
      .materialization_strategy = config_.options->materialization_strategy,
      .memory_tier_config = config_.options->memory_tier_config,
  };
  cfg.pinned_memory_timeout = config_.pinned_memory_timeout;
  cfg.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);
  cfg.view_id = request.requested_view_id();
  cfg.transform_placement = transform_placement;

  auto replica_or = replica::Replica::create(cfg);
  if (!replica_or.ok()) {
    return replica_or.status();
  }
  auto replica = std::shared_ptr<replica::Replica>(std::move(replica_or.value()));

  auto sources_or = build_sources();
  if (!sources_or.ok()) {
    return sources_or.status();
  }
  absl::Status load_status = load_assembled_ranges_into_replica(
      replica,
      local_map,
      std::move(*sources_or),
      config_.options->byte_mapping,
      request.target_location(),
      concurrency,
      target_view_plan,
      transform_placement,
      request.target_device().ordinal);
  if (!load_status.ok()) {
    return load_status;
  }

  absl::Status emplace_status = replica_registry.emplace(key, gsl::not_null{replica});
  if (absl::IsAlreadyExists(emplace_status)) {
    auto concurrent_or = replica_registry.find(key);
    if (!concurrent_or.ok()) {
      return concurrent_or.status();
    }
    const auto& concurrent_replica = concurrent_or.value();
    absl::Status reuse_status = validate_existing_replica_for_reuse(concurrent_replica, request.target_location());
    if (!reuse_status.ok()) {
      return reuse_status;
    }
    LOG(INFO) << "materialize_view.local_canonical_raced artifact_id=" << request.canonical_artifact_id()
              << " view_id=" << view_id << " source_key=" << *selected_key;
    return build_local_view_handle(concurrent_replica);
  }
  if (!emplace_status.ok()) {
    return emplace_status;
  }

  LOG(INFO) << "materialize_view.local_canonical_loaded artifact_id=" << request.canonical_artifact_id()
            << " view_id=" << view_id << " source_key=" << *selected_key
            << " source_location=" << static_cast<int>(selected_location) << " view_bytes=" << local_map.total_bytes;
  return build_local_view_handle(replica);
}

MaterializationFacade::MaterializationFacade(Config config)
    : config_(std::move(config)),
      hooks_(config_.hooks),
      ingestion_event_hub_(config_.runtime_context->ingestion_event_hub()) {
  ABSL_CHECK(config_.runtime_context != nullptr) << "RuntimeContext is required";
  ABSL_CHECK(config_.replica_runtime != nullptr) << "ReplicaRuntime is required";
  ABSL_CHECK(config_.metadata_gateway != nullptr) << "MetadataGateway is required";
  ABSL_CHECK(config_.options != nullptr) << "StoreEngineOptions must not be null";
  ABSL_CHECK(ingestion_event_hub_ != nullptr) << "RuntimeContext missing ingestion event hub";

  pipeline::IngestionPipeline::Config pipeline_config{
      .storage_path = config_.storage_path,
      .num_threads = config_.num_threads,
      .artifact_chunk_bytes = config_.artifact_chunk_bytes,
      .pinned_memory_timeout = config_.pinned_memory_timeout,
      .engine_options = config_.options,
      .replica_runtime = config_.replica_runtime,
      .runtime_context = config_.runtime_context.get(),
      .ordinary_disk_strategy_planner =
          [this](const pipeline::IngestionContext& ctx) { return build_ordinary_disk_execution_strategy_plan(ctx); },
  };
  if (hooks_ && hooks_->pipeline_factory) {
    pipeline_ = hooks_->pipeline_factory(pipeline_config);
  } else {
    pipeline_ = std::make_unique<pipeline::IngestionPipeline>(pipeline_config);
  }
  ABSL_CHECK(pipeline_ != nullptr) << "Ingestion pipeline factory returned null";

  auto& registry = config_.replica_runtime->registry();
  auto pinned_pool = config_.runtime_context->pinned_buffer_pool();
  MaterializationDeps deps(
      gsl::not_null<components::ReplicaRegistry*>{&registry},
      gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{pinned_pool});
  deps.async_runtime = config_.runtime_context->async_runtime();
  deps.artifact_chunk_bytes = config_.artifact_chunk_bytes;
  deps.pinned_memory_timeout = config_.pinned_memory_timeout;
  deps.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);
  deps.num_threads = config_.num_threads;
  deps.byte_mapping_config = config_.options->byte_mapping;
  deps.materialization_strategy = config_.options->materialization_strategy;
  deps.view_hash_computer = config_.runtime_context->view_hash_computer();
  deps.run_auto_handles_disk_fallback = true;
  deps.ingest_from_disk = [this](
                              const std::string& artifact_identifier,
                              const loading::DiskSource& source,
                              const loading::ReplicaTarget& target,
                              const loading::MaterializeHints& hints) {
    return run_disk_ingestion_internal(artifact_identifier, source, target, hints, /*publish_to_global_store=*/false);
  };
  deps.run_auto = [this](const loading::MaterializationRequest& request) -> absl::StatusOr<loading::ReplicaHandle> {
    auto client = config_.runtime_context->global_store_client();
    const bool gs_connected = client && client->is_connected();
    if (request.requested_view_id().has_value()) {
      auto local_view_or = materialize_view_from_local_canonical(request);
      if (local_view_or.ok()) {
        return *local_view_or;
      }
      if (!absl::IsNotFound(local_view_or.status())) {
        return local_view_or.status();
      }
      LOG(INFO) << "materialize_view.local_canonical_unavailable artifact_id=" << request.canonical_artifact_id()
                << " view_id=" << *request.requested_view_id() << " reason=" << local_view_or.status();
      if (!gs_connected && request.hints().allow_disk && request.has_disk_source()) {
        loading::ReplicaTarget target;
        target.location.type =
            request.target_is_gpu() ? common::memory::MemoryLocation::GPU : common::memory::MemoryLocation::CPU;
        target.location.device_id = request.target_device().ordinal;
        LOG(INFO) << "materialize_view.disk_fallback artifact_id=" << request.canonical_artifact_id()
                  << " view_id=" << *request.requested_view_id()
                  << " target_device=" << request.target_device().ordinal;
        return run_disk_ingestion_internal(
            request.canonical_artifact_id(),
            *request.disk_source(),
            target,
            request.hints(),
            /*publish_to_global_store=*/false);
      }
    }

    if (!gs_connected) {
      if (!request.hints().allow_disk || !request.has_disk_source()) {
        return absl::FailedPreconditionError("GlobalStoreClient not connected");
      }
      loading::ReplicaTarget target;
      target.location.type =
          request.target_is_gpu() ? common::memory::MemoryLocation::GPU : common::memory::MemoryLocation::CPU;
      target.location.device_id = request.target_device().ordinal;
      return run_disk_ingestion_internal(
          request.canonical_artifact_id(),
          *request.disk_source(),
          target,
          request.hints(),
          /*publish_to_global_store=*/false);
    }
    MaterializeOrchestrator orchestrator(
        gsl::not_null<materialization::control::MaterializationBackend*>{this},
        gsl::not_null<components::IGlobalStoreClient*>{client.get()},
        config_.runtime_context->worker_identity());
    auto orchestrated_or = orchestrator.run(
        request.canonical_artifact_id(), request.target_device(), request.hints(), request.disk_source());
    if (orchestrated_or.ok()) {
      return *orchestrated_or;
    }
    if (absl::IsNotFound(orchestrated_or.status())) {
      const auto id_kind = common::infer_artifact_id_kind(request.canonical_artifact_id());
      if (id_kind == common::ArtifactIdKind::kCgid) {
        auto assembled_or = assemble_from_pieces(request);
        if (assembled_or.ok()) {
          return *assembled_or;
        }
        return assembled_or.status();
      }
    }
    return orchestrated_or.status();
  };

  if (hooks_ && hooks_->materialization_service_factory) {
    materialization_service_ = hooks_->materialization_service_factory(std::move(deps));
  } else {
    materialization_service_ = std::make_unique<MaterializationService>(std::move(deps));
  }
  ABSL_CHECK(materialization_service_ != nullptr) << "Materialization service factory returned null";
}

absl::StatusOr<strategy::ExecutionStrategyPlan> MaterializationFacade::build_ordinary_disk_execution_strategy_plan(
    const pipeline::IngestionContext& ctx) const {
  strategy::ExecutionStrategyPlan plan;
  const auto& strategy_config = config_.options->materialization_strategy;
  auto& environment = plan.environment;
  environment.execution_topology = ctx.hints.execution_topology();
  environment.source = loading::MaterializationSource::kDisk;
  environment.target_is_gpu = ctx.target_is_gpu;
  environment.source_layout_available = ctx.disk.source_index_json.has_value();
  environment.allow_mixed_execution = strategy_config.allow_mixed_execution;
  environment.owner_file_collective_peak_bytes_budget = strategy_config.owner_file_collective_peak_bytes_budget;
  environment.owner_file_collective_batch_bytes = strategy_config.owner_file_collective_batch_bytes;
  environment.owner_file_collective_dim1_staging_bytes = strategy_config.owner_file_collective_dim1_staging_bytes;
  environment.owner_file_collective_max_inflight_batches = strategy_config.owner_file_collective_max_inflight_batches;
  environment.owner_file_collective_shared_fs_only = strategy_config.owner_file_collective_shared_fs_only;
  environment.owner_file_collective_max_owner_skew_ratio = strategy_config.owner_file_collective_max_owner_skew_ratio;
  environment.owner_file_collective_min_dedup_saving_bytes =
      strategy_config.owner_file_collective_min_dedup_saving_bytes;
  environment.owner_file_collective_group_assemble_timeout =
      strategy_config.owner_file_collective_group_assemble_timeout;
  environment.owner_file_collective_allow_mixed_residual = strategy_config.owner_file_collective_allow_mixed_residual;
  environment.owner_file_collective_planner_cache_entries = strategy_config.owner_file_collective_planner_cache_entries;

  if (ctx.resolved_view_plan.has_value()) {
    environment.requested_bytes = ctx.resolved_view_plan->view_size_bytes;
  } else if (ctx.verification.logical_total_size > 0) {
    environment.requested_bytes = ctx.verification.logical_total_size;
  } else if (ctx.disk.source.expected_size.has_value()) {
    environment.requested_bytes = *ctx.disk.source.expected_size;
  }

  strategy::ExecutionStrategyCostEstimate generic_estimate{
      .requested_source_bytes = environment.requested_bytes,
      .unique_source_bytes = environment.requested_bytes,
  };
  strategy::ExecutionStrategyCandidate generic_candidate{
      .executor = strategy::ExecutionStrategyExecutor::kGenericByteRange,
      .eligible = true,
      .reason = "exact_generic_fallback",
      .estimate = generic_estimate,
  };

  std::string local_reason = "target_not_gpu";
  std::string collective_reason = "target_not_gpu";
  strategy::ExecutionStrategyCostEstimate local_estimate = generic_estimate;
  strategy::ExecutionStrategyCostEstimate collective_estimate = generic_estimate;
  bool shared_source_proven = false;
  std::optional<replica::LocalBatchedPlanSummary> local_plan_summary;

  if (ctx.target_is_gpu) {
    local_reason = "strategy_disabled";
    collective_reason = "strategy_disabled";

    auto disk_context_or = loader::get_disk_artifact_context(ctx.disk.artifact_path);
    if (disk_context_or.ok()) {
      plan.disk_context = *disk_context_or;
      environment.coordinator_available = plan.disk_context != nullptr;
    } else if (strategy_diagnostics_verbose_enabled(strategy_config)) {
      LOG(WARNING) << "materialize_replica strategy planning could not acquire shared disk context"
                   << " artifact_id=" << ctx.artifact_identifier << " status=" << disk_context_or.status();
    }

    environment.requires_server_transform = ctx.resolved_view_plan.has_value() &&
        ctx.resolved_view_plan->transform.requires_materialization &&
        (ctx.hints.variant ? ctx.hints.variant->placement : loading::TransformPlacement::kServer) ==
            loading::TransformPlacement::kServer;

    const bool variant_required_for_representation = ctx.resolved_view_plan.has_value();
    std::string representation_reason =
        variant_required_for_representation ? "missing_variant_or_source_layout" : "missing_source_layout";
    const bool can_build_representation = ctx.disk.source_index_json.has_value() &&
        ctx.verification.canonical_index_json.has_value() &&
        (ctx.hints.variant.has_value() || !variant_required_for_representation);
    if (can_build_representation) {
      const std::string_view target_index_json = ctx.resolved_view_plan.has_value()
          ? std::string_view(ctx.resolved_view_plan->view_index_json)
          : std::string_view(*ctx.verification.canonical_index_json);
      if (!target_index_json.empty()) {
        auto representation_or = materialization::contracts::build_index_backed_representation_work(
            *ctx.disk.source_index_json,
            target_index_json,
            ctx.hints.variant,
            canonical_byte_space_ref(),
            "ephemeral_into_target");
        if (representation_or.ok()) {
          plan.representation_work_plan = std::move(representation_or->work_plan);
          if (plan.representation_work_plan->items.empty()) {
            plan.representation_work_plan.reset();
            representation_reason = "representation_work_plan_empty";
          } else {
            environment.committed_bytes = plan.representation_work_plan->committed_bytes;
            environment.residual_bytes = plan.representation_work_plan->residual_fallback_map.total_bytes;
            environment.has_complete_metadata = true;
            representation_reason = "ok";
          }
        } else {
          representation_reason = representation_or.status().message();
          if (strategy_diagnostics_verbose_enabled(strategy_config)) {
            LOG(INFO) << "materialize_replica representation_work unavailable"
                      << " artifact_id=" << ctx.artifact_identifier << " status=" << representation_or.status();
          }
        }
      } else {
        representation_reason = "missing_target_index_json";
      }
    }

    const bool has_fill_items =
        plan.representation_work_plan.has_value() && work_plan_has_fill_items(*plan.representation_work_plan);
    const bool has_residual_fallback = plan.representation_work_plan.has_value() &&
        plan.representation_work_plan->residual_fallback_map.total_bytes > 0;
    if (has_fill_items) {
      representation_reason = "fill_items_require_generic_fallback";
    } else if (has_residual_fallback) {
      representation_reason = "residual_fallback_requires_generic_executor";
    } else if (environment.requires_server_transform) {
      representation_reason = "server_transform_requires_generic_executor";
    }

    const uint64_t local_peak_bytes = std::min<uint64_t>(
        environment.requested_bytes,
        std::max<uint64_t>(config_.artifact_chunk_bytes, ctx.runtime_context->tx_slice_bytes()));
    if (plan.representation_work_plan.has_value() && !has_fill_items && !has_residual_fallback &&
        !environment.requires_server_transform) {
      auto local_summary_or =
          replica::summarize_local_batched_disk_load(*plan.representation_work_plan, strategy_config);
      if (local_summary_or.ok()) {
        local_plan_summary = std::move(*local_summary_or);
      } else if (strategy_diagnostics_verbose_enabled(strategy_config)) {
        LOG(INFO) << "materialize_replica local_batched summary unavailable"
                  << " artifact_id=" << ctx.artifact_identifier << " status=" << local_summary_or.status();
      }
    }

    local_estimate = strategy::ExecutionStrategyCostEstimate{
        .requested_source_bytes = local_plan_summary.has_value() && local_plan_summary->requested_source_bytes > 0
            ? local_plan_summary->requested_source_bytes
            : environment.requested_bytes,
        .unique_source_bytes = local_plan_summary.has_value() && local_plan_summary->unique_source_bytes > 0
            ? local_plan_summary->unique_source_bytes
            : environment.requested_bytes,
        .estimated_peak_temporary_bytes = local_plan_summary.has_value()
            ? std::max<uint64_t>(local_peak_bytes, local_plan_summary->peak_temporary_bytes)
            : local_peak_bytes,
        .estimated_batch_bytes = local_peak_bytes,
        .estimated_owner_skew_ratio = 1.0,
        .estimated_dedup_saving_bytes = local_plan_summary.has_value() ? local_plan_summary->dedup_saving_bytes : 0,
    };

    const bool local_prereqs_ready = strategy_config.enable_local_batched_disk_load && ctx.disk.is_safetensors &&
        plan.disk_context != nullptr && plan.representation_work_plan.has_value() && !has_fill_items &&
        !has_residual_fallback && !environment.requires_server_transform && local_plan_summary.has_value() &&
        local_plan_summary->eligible;
    if (local_prereqs_ready) {
      local_reason = local_plan_summary->reason;
    } else if (!strategy_config.enable_local_batched_disk_load) {
      local_reason = "strategy_disabled";
    } else if (!ctx.disk.is_safetensors) {
      local_reason = "non_safetensors_source";
    } else if (plan.disk_context == nullptr) {
      local_reason = "shared_disk_context_unavailable";
    } else if (local_plan_summary.has_value() && !local_plan_summary->eligible) {
      local_reason = local_plan_summary->reason;
    } else {
      local_reason = representation_reason;
    }

    const auto group = environment.execution_topology.collective_load_group;
    const uint32_t world_size = group.has_value() ? group->world_size : 0;
    const uint64_t dedup_saving_bytes =
        estimate_collective_dedup_saving_bytes(plan.representation_work_plan, world_size);
    shared_source_proven =
        environment.execution_topology.source_locality == loading::SourceLocalityHint::kSharedSource ||
        environment.execution_topology.source_sharing_domain.has_value();
    collective_estimate = strategy::ExecutionStrategyCostEstimate{
        .requested_source_bytes = environment.requested_bytes,
        .unique_source_bytes = world_size > 0 ? environment.requested_bytes : environment.requested_bytes,
        .estimated_peak_temporary_bytes = environment.requested_bytes,
        .estimated_batch_bytes = strategy_config.owner_file_collective_batch_bytes,
        .estimated_owner_skew_ratio = 1.0,
        .estimated_dedup_saving_bytes = dedup_saving_bytes,
    };

    const bool locality_blocks_collective = strategy_config.owner_file_collective_shared_fs_only &&
        (!shared_source_proven ||
         environment.execution_topology.source_locality == loading::SourceLocalityHint::kHostLocal);
    const bool collective_budget_ok = strategy_config.owner_file_collective_peak_bytes_budget == 0 ||
        collective_estimate.estimated_peak_temporary_bytes <= strategy_config.owner_file_collective_peak_bytes_budget;
    const bool collective_skew_ok = strategy_config.owner_file_collective_max_owner_skew_ratio <= 0.0 ||
        collective_estimate.estimated_owner_skew_ratio <= strategy_config.owner_file_collective_max_owner_skew_ratio;
    const bool collective_dedup_ok = collective_estimate.estimated_dedup_saving_bytes >=
        strategy_config.owner_file_collective_min_dedup_saving_bytes;
    const bool collective_prereqs_ready = strategy_config.enable_owner_file_collective && ctx.disk.is_safetensors &&
        group.has_value() && group->world_size > 1 && plan.disk_context != nullptr &&
        plan.representation_work_plan.has_value() && !has_fill_items && !has_residual_fallback &&
        !environment.requires_server_transform && !locality_blocks_collective && collective_budget_ok &&
        collective_skew_ok && collective_dedup_ok;
    if (collective_prereqs_ready) {
      collective_reason = "eligible";
      plan.collective_load_group = group;
    } else if (!strategy_config.enable_owner_file_collective) {
      collective_reason = "strategy_disabled";
    } else if (!ctx.disk.is_safetensors) {
      collective_reason = "non_safetensors_source";
    } else if (!group.has_value() || group->world_size <= 1) {
      collective_reason = "collective_group_missing";
    } else if (plan.disk_context == nullptr) {
      collective_reason = "shared_disk_context_unavailable";
    } else if (
        strategy_config.owner_file_collective_shared_fs_only &&
        environment.execution_topology.source_locality == loading::SourceLocalityHint::kHostLocal) {
      collective_reason = "source_locality_host_local";
    } else if (!shared_source_proven && strategy_config.owner_file_collective_shared_fs_only) {
      collective_reason = "shared_source_unproven";
    } else if (locality_blocks_collective) {
      collective_reason = "shared_source_unproven";
    } else if (!collective_budget_ok) {
      collective_reason = "peak_budget_exceeded";
    } else if (!collective_skew_ok) {
      collective_reason = "owner_skew_threshold_exceeded";
    } else if (!collective_dedup_ok) {
      collective_reason = "dedup_saving_below_threshold";
    } else {
      collective_reason = representation_reason;
    }
  }

  plan.candidates = {
      generic_candidate,
      strategy::ExecutionStrategyCandidate{
          .executor = strategy::ExecutionStrategyExecutor::kTensorBatchedLocal,
          .eligible = local_reason == "eligible",
          .reason = local_reason,
          .estimate = local_estimate,
      },
      strategy::ExecutionStrategyCandidate{
          .executor = strategy::ExecutionStrategyExecutor::kOwnerFileCollective,
          .eligible = collective_reason == "eligible",
          .reason = collective_reason,
          .estimate = collective_estimate,
      },
  };

  auto choose_generic = [&](std::string reason) {
    plan.executor = strategy::ExecutionStrategyExecutor::kGenericByteRange;
    plan.selection_reason = std::move(reason);
  };
  const bool local_eligible = plan.candidates[1].eligible;
  const bool collective_eligible = plan.candidates[2].eligible;
  const std::optional<std::string> local_auto_reject =
      local_eligible ? local_auto_reject_reason(generic_estimate, local_estimate, local_plan_summary) : std::nullopt;
  const bool auto_prefers_local = local_eligible && !local_auto_reject.has_value();
  switch (strategy_config.executor_preference) {
    case StrategyConfig::ExecutorPreference::kGenericByteRange:
      choose_generic("executor_preference_generic");
      break;
    case StrategyConfig::ExecutorPreference::kTensorAwareLocal:
      if (local_eligible) {
        plan.executor = strategy::ExecutionStrategyExecutor::kTensorBatchedLocal;
        plan.selection_reason = "executor_preference_tensor_aware_local";
      } else {
        choose_generic(absl::StrCat("tensor_aware_local_unavailable:", local_reason));
      }
      break;
    case StrategyConfig::ExecutorPreference::kOwnerFileCollective:
      if (collective_eligible) {
        plan.executor = strategy::ExecutionStrategyExecutor::kOwnerFileCollective;
        plan.selection_reason = "executor_preference_owner_file_collective";
      } else if (local_eligible) {
        plan.executor = strategy::ExecutionStrategyExecutor::kTensorBatchedLocal;
        plan.selection_reason =
            absl::StrCat("owner_file_collective_unavailable:", collective_reason, ";fallback=local");
      } else {
        choose_generic(absl::StrCat("owner_file_collective_unavailable:", collective_reason));
      }
      break;
    case StrategyConfig::ExecutorPreference::kAuto:
    default:
      if (collective_eligible && shared_source_proven) {
        plan.executor = strategy::ExecutionStrategyExecutor::kOwnerFileCollective;
        plan.selection_reason = "auto_prefers_owner_file_collective_shared_source";
      } else if (auto_prefers_local) {
        plan.executor = strategy::ExecutionStrategyExecutor::kTensorBatchedLocal;
        plan.selection_reason = "auto_prefers_local_batched";
      } else if (collective_eligible) {
        plan.executor = strategy::ExecutionStrategyExecutor::kOwnerFileCollective;
        plan.selection_reason = "auto_collective_after_local_rejection";
      } else {
        choose_generic(
            absl::StrCat(
                "auto_generic:",
                local_auto_reject.has_value() ? *local_auto_reject : local_reason,
                ",",
                collective_reason));
      }
      break;
  }

  if (strategy_diagnostics_basic_enabled(strategy_config)) {
    LOG(INFO) << "materialize_replica strategy_plan"
              << " artifact_id=" << ctx.artifact_identifier
              << " executor=" << strategy::execution_strategy_executor_name(plan.executor)
              << " selection_reason=" << plan.selection_reason << " requested_bytes=" << environment.requested_bytes
              << " committed_bytes=" << environment.committed_bytes << " residual_bytes=" << environment.residual_bytes
              << " source_layout_available=" << environment.source_layout_available
              << " coordinator_available=" << environment.coordinator_available
              << " source_locality=" << source_locality_name(environment.execution_topology.source_locality)
              << " collective_group=" << (environment.execution_topology.collective_load_group.has_value() ? 1 : 0)
              << " candidates=" << format_execution_strategy_candidates(plan.candidates);
  }

  return plan;
}

MaterializationFacade::~MaterializationFacade() = default;

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::materialize_replica(
    const DeviceKey& target_device,
    loading::MaterializeMode mode,
    const loading::MaterializeHints& hints,
    std::optional<loading::DiskSource> disk_source) {
  auto request_or = loading::MaterializationRequest::Create(
      target_device, mode, hints, config_.replica_runtime->device_manager(), std::move(disk_source));
  if (!request_or.ok()) {
    return request_or.status();
  }
  LOG(INFO) << "materialize_replica.request"
            << " artifact_id=" << request_or->canonical_artifact_id()
            << " target_device=" << request_or->target_device().ordinal
            << " mode=" << static_cast<int>(request_or->mode()) << " requested_view_id="
            << (request_or->requested_view_id().has_value() ? *request_or->requested_view_id() : std::string())
            << " has_variant=" << (hints.variant.has_value() ? 1 : 0)
            << " has_cached_plan=" << (hints.variant.has_value() && hints.variant->cached_plan.has_value() ? 1 : 0)
            << " has_disk_source=" << (request_or->has_disk_source() ? 1 : 0);
  return materialization_service_->execute(request_or.value());
}

absl::StatusOr<loading::MaterializeIntoTargetResult> MaterializationFacade::materialize_into_target(
    const DeviceKey& target_device,
    const loading::IntoTargetLayout& target_layout,
    std::string_view canonical_index_json,
    uint64_t generation,
    const loading::MaterializeHints& hints,
    std::optional<loading::DiskSource> disk_source) {
  if (target_device.type != DeviceType::GPU) {
    return absl::InvalidArgumentError("materialize_into_target requires GPU target device");
  }
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("materialize_into_target requires canonical index bytes");
  }
  if (hints.artifact_id.empty()) {
    return absl::InvalidArgumentError("materialize_into_target requires hints.artifact_id");
  }
  if (target_layout.storages.empty()) {
    return absl::InvalidArgumentError("materialize_into_target requires at least one target storage");
  }

  uint64_t total_size = target_layout.total_size;
  uint64_t computed_total = 0;
  for (const auto& storage : target_layout.storages) {
    if (storage.length == 0) {
      return absl::InvalidArgumentError("materialize_into_target requires non-empty storage length");
    }
    if (storage.length > std::numeric_limits<uint64_t>::max() - computed_total) {
      return absl::OutOfRangeError("materialize_into_target storage length overflow");
    }
    computed_total += storage.length;
  }
  if (total_size == 0) {
    total_size = computed_total;
  } else if (total_size != computed_total) {
    return absl::InvalidArgumentError("materialize_into_target total_size does not match storage lengths");
  }
  if (total_size == 0) {
    return absl::InvalidArgumentError("materialize_into_target requires total_size > 0");
  }

  auto canonical_total_or = compute_logical_total_size(canonical_index_json);
  if (!canonical_total_or.ok()) {
    return canonical_total_or.status();
  }
  const uint64_t canonical_total_size = *canonical_total_or;

  std::optional<loader::ViewPlan> view_plan;
  if (hints.variant && hints.variant->cached_plan.has_value()) {
    view_plan = *hints.variant->cached_plan;
  }
  const loading::TransformPlacement placement =
      hints.variant ? hints.variant->placement : loading::TransformPlacement::kServer;
  if (view_plan.has_value() && view_plan->transform.requires_materialization &&
      placement == loading::TransformPlacement::kServer && target_layout.storages.size() != 1) {
    return absl::InvalidArgumentError(
        "materialize_into_target requires single storage for server-side view transforms");
  }
  if (view_plan.has_value() && view_plan->view_size_bytes > 0 && view_plan->view_size_bytes != total_size) {
    return absl::InvalidArgumentError("materialize_into_target view size does not match target layout size");
  }

  auto index_hash = [](std::string_view index_json) -> std::string {
    auto mh_or = common::compute_index_multihash(std::optional<std::string>(index_json), "");
    if (mh_or.ok()) {
      return *mh_or;
    }
    const size_t fallback_hash = std::hash<std::string_view>{}(index_json);
    return absl::StrCat("raw:", fallback_hash);
  };

  const std::optional<std::string_view> source_index_json =
      (hints.disk_metadata && hints.disk_metadata->source_index_json.has_value())
      ? std::optional<std::string_view>(*hints.disk_metadata->source_index_json)
      : std::nullopt;

  std::optional<materialization::contracts::RepresentationWorkPlan> ordinary_representation_work_plan;
  if (!(view_plan.has_value() && view_plan->transform.requires_materialization)) {
    const std::string_view semantic_source_index_json =
        source_index_json.has_value() ? *source_index_json : canonical_index_json;
    const std::string_view semantic_target_index_json = (view_plan.has_value() && !view_plan->is_identity)
        ? std::string_view(view_plan->view_index_json)
        : canonical_index_json;
    auto representation_or = materialization::contracts::build_index_backed_representation_work(
        semantic_source_index_json,
        semantic_target_index_json,
        hints.variant,
        canonical_byte_space_ref(),
        "ephemeral_into_target");
    if (representation_or.ok()) {
      size_t work_item_count = 0;
      uint64_t committed_bytes = 0;
      auto work_plan_or =
          materialization::contracts::build_representation_work_plan(representation_or->transform_contract);
      if (!work_plan_or.ok()) {
        VLOG(1) << "materialize_into_target shared work-plan lowering unavailable: " << work_plan_or.status();
      } else {
        work_item_count = work_plan_or->items.size();
        committed_bytes = work_plan_or->committed_bytes;
        ordinary_representation_work_plan = std::move(*work_plan_or);
      }
      VLOG(1) << "materialize_into_target representation_work"
              << " items=" << work_item_count << " committed_bytes=" << committed_bytes
              << " representation_contract_hash="
              << representation_or->transform_contract.target_representation.representation_contract_hash;
    } else {
      VLOG(1) << "materialize_into_target shared work-plan lowering unavailable: " << representation_or.status();
    }
  }

  auto get_map_ptr = [&](bool use_source_layout) -> absl::StatusOr<std::shared_ptr<loader::ByteRangeMap>> {
    const std::string canonical_hash = index_hash(canonical_index_json);
    std::string plan_key = absl::StrCat(generation, ":canon:", canonical_hash);
    if (use_source_layout && source_index_json.has_value()) {
      plan_key = absl::StrCat(plan_key, ":src:", index_hash(*source_index_json));
    }

    std::shared_ptr<loader::ByteRangeMap> map_ptr;
    {
      absl::MutexLock lock(&byte_range_map_mu_);
      auto it = byte_range_map_cache_.find(plan_key);
      if (it != byte_range_map_cache_.end()) {
        return it->second;
      }
    }

    absl::StatusOr<loader::ByteRangeMap> map_or;
    if (use_source_layout && source_index_json.has_value()) {
      map_or = loader::build_byte_range_map_from_canonical_and_source_index_json(
          canonical_index_json, *source_index_json, canonical_total_size);
    } else {
      map_or = loader::build_byte_range_map_from_canonical_index_json(canonical_index_json, canonical_total_size);
    }
    if (!map_or.ok()) {
      return map_or.status();
    }
    map_ptr = std::make_shared<loader::ByteRangeMap>(std::move(*map_or));
    {
      absl::MutexLock lock(&byte_range_map_mu_);
      byte_range_map_cache_.emplace(plan_key, map_ptr);
    }
    return map_ptr;
  };

  auto run_source =
      [&](std::unique_ptr<IArtifactLoader> loader,
          loading::MaterializationSource source_kind) -> absl::StatusOr<loading::MaterializeIntoTargetResult> {
    const absl::Time total_started_at = absl::Now();
    absl::Duration streaming_buffer_init_elapsed = absl::ZeroDuration();
    absl::Duration pump_elapsed = absl::ZeroDuration();
    absl::Duration sink_close_elapsed = absl::ZeroDuration();
    bool sink_closed = false;
    auto init_status = loader->initialize();
    if (!init_status.ok()) {
      return init_status;
    }
    auto source_or = loader->open_source();
    if (!source_or.ok()) {
      return source_or.status();
    }

    const bool use_source_layout =
        (source_kind == loading::MaterializationSource::kDisk) && source_index_json.has_value();
    auto map_ptr_or = get_map_ptr(use_source_layout);
    if (!map_ptr_or.ok()) {
      return map_ptr_or.status();
    }
    auto map_ptr = *map_ptr_or;

    loader::ByteRangeMap effective_map = *map_ptr;
    bool composed_view = false;
    if (view_plan.has_value() && !view_plan->is_identity && use_source_layout) {
      auto composed_or = loader::compose_byte_range_maps(view_plan->selection.map, *map_ptr);
      if (!composed_or.ok()) {
        return composed_or.status();
      }
      effective_map = std::move(*composed_or);
      composed_view = true;
    }

    std::shared_ptr<loader::SeekableSource> base_source = std::move(*source_or);
    std::vector<std::shared_ptr<loader::SeekableSource>> sources;
    sources.emplace_back(base_source);

    const bool map_crosses_storage_boundaries =
        byte_range_map_crosses_target_storage_boundaries(effective_map, target_layout);
    const bool source_ordered = use_source_layout && config_.options->byte_mapping.disk_source_ordered_read &&
        allow_source_ordered_for_mapped(config_.options->materialization_strategy) && !map_crosses_storage_boundaries;
    const bool collective_eligible = source_kind == loading::MaterializationSource::kDisk &&
        hints.collective_load_group.has_value() && ordinary_representation_work_plan.has_value() &&
        allow_collective_tensor_executor(config_.options->materialization_strategy) &&
        !work_plan_has_fill_items(*ordinary_representation_work_plan) &&
        ordinary_representation_work_plan->residual_fallback_map.total_bytes == 0;
    if (use_source_layout && config_.options->byte_mapping.disk_source_ordered_read && map_crosses_storage_boundaries) {
      LOG(INFO) << "materialize_into_target disabling source-ordered fast path because byte-range segments "
                   "cross target storage boundaries"
                << " map_segments=" << effective_map.segments.size()
                << " target_storages=" << target_layout.storages.size();
    }
    std::unique_ptr<loader::SeekableSource> plan_source;
    if (!source_ordered) {
      loader::ByteRangeCompiler compiler(config_.options->byte_mapping, "materialize_into_target");
      auto program_or = compiler.Compile(effective_map);
      if (!program_or.ok()) {
        return program_or.status();
      }
      loader::ByteRangeMappedSource::Options map_opts{
          .path = "materialize_into_target",
          .transport_request_id = std::string(hints.transport_request_id),
          .enable_direct_write_at = config_.options->byte_mapping.enable_direct_write_at,
      };
      auto mapped_or =
          loader::ByteRangeMappedSource::Create(effective_map, *program_or, std::move(sources), std::move(map_opts));
      if (!mapped_or.ok()) {
        return mapped_or.status();
      }
      plan_source = std::move(*mapped_or);
      if (view_plan.has_value() && !view_plan->is_identity && !composed_view) {
        plan_source =
            loader::make_view_plan_source(std::move(plan_source), view_plan->selection, config_.options->byte_mapping);
      }
      if (!plan_source) {
        return absl::InternalError("materialize_into_target failed to build view plan source");
      }
    }

    const size_t slice_bytes = config_.runtime_context->tx_slice_bytes();
    if (slice_bytes == 0 || config_.artifact_chunk_bytes == 0) {
      return absl::FailedPreconditionError("tx_slice_bytes or artifact_chunk_bytes is zero");
    }
    const std::chrono::milliseconds timeout =
        hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : config_.pinned_memory_timeout;

    std::vector<loader::TargetStorage> storages;
    storages.reserve(target_layout.storages.size());
    std::vector<loader::Range> ranges;
    ranges.reserve(target_layout.storages.size());
    uint64_t range_cursor = 0;
    for (const auto& storage : target_layout.storages) {
      if (storage.length == 0) {
        return absl::InvalidArgumentError("materialize_into_target requires non-empty storage length");
      }
      if (storage.length > std::numeric_limits<size_t>::max()) {
        return absl::OutOfRangeError("materialize_into_target storage length exceeds host limits");
      }
      storages.push_back(loader::TargetStorage{storage.base_ptr, storage.length});
      ranges.emplace_back(range_cursor, static_cast<size_t>(storage.length));
      range_cursor += storage.length;
    }
    if (range_cursor != total_size) {
      return absl::InvalidArgumentError("materialize_into_target storage ranges do not span total_size");
    }

    std::string collective_skip_reason;
    if (collective_eligible) {
      if (auto* disk_loader = dynamic_cast<DiskLoader*>(loader.get()); disk_loader != nullptr) {
        auto shared_or = disk_loader->shared_context();
        if (!shared_or.ok()) {
          LOG(WARNING) << "materialize_into_target collective path unavailable: shared_context error="
                       << shared_or.status();
          collective_skip_reason = "disk_shared_context_unavailable";
        } else {
          const replica::CollectiveMappedTargetLoadOptions collective_options{
              .chunk_bytes = std::min<uint64_t>(slice_bytes, total_size),
              .merge_max_gap_bytes = config_.options->byte_mapping.disk_source_merge_max_gap_bytes,
              .merge_max_amplification = config_.options->byte_mapping.disk_source_merge_max_amplification,
              .strategy_config = config_.options->materialization_strategy,
          };
          auto collective_result = execute_collective_mapped_target_load(
              config_.hooks,
              replica::CollectiveMappedTargetLoadRequest{
                  .artifact_id = hints.artifact_id,
                  .group = *hints.collective_load_group,
                  .disk_context = *shared_or,
                  .representation_work_plan = *ordinary_representation_work_plan,
                  .target_layout = target_layout,
                  .device_id = target_device.ordinal,
              },
              config_.runtime_context->pinned_buffer_pool(),
              timeout,
              collective_options);
          if (collective_result.handled) {
            if (!collective_result.status.ok()) {
              return absl::DataLossError(
                  absl::StrCat(
                      "materialize_into_target collective execution failed: ", collective_result.status.message()));
            }
            return loading::MaterializeIntoTargetResult{
                .source = source_kind,
                .requested_bytes = effective_map.total_bytes,
                .committed_bytes = effective_map.total_bytes,
                .fallback_bytes = 0,
                .residual_bytes = 0,
                .actual_collective_committed_bytes = effective_map.total_bytes,
                .actual_local_typed_bytes = 0,
                .actual_generic_backend_bytes = 0,
                .collective_unique_source_bytes = collective_result.metrics.unique_source_bytes,
                .collective_peer_transfer_bytes = collective_result.metrics.peer_transfer_bytes,
                .collective_peak_temporary_bytes = collective_result.metrics.peak_temporary_bytes,
                .collective_batch_count = collective_result.metrics.batch_count,
                .collective_dedup_saving_bytes = collective_result.metrics.dedup_saving_bytes,
                .collective_skip_reason = {},
                .collective_handled = true,
                .direct_write_supported = false,
                .source_ordered = false,
                .dominant_executor = "OwnerFileCollectiveExecutor",
                .selection_reason = "collective_mapped_target",
            };
          }
          collective_skip_reason =
              collective_result.skip_reason.empty() ? "collective_executor_unhandled" : collective_result.skip_reason;
        }
      } else {
        collective_skip_reason = "disk_loader_unavailable";
      }
    } else if (hints.collective_load_group.has_value()) {
      collective_skip_reason = "collective_not_eligible";
    }
    if (hints.require_collective_execution) {
      return absl::FailedPreconditionError(
          collective_skip_reason.empty()
              ? "collective requested but the selected source could not be handled without generic fallback"
              : absl::StrCat(
                    "collective requested but the selected source could not be handled without generic fallback: ",
                    collective_skip_reason));
    }

    loader::TargetLayoutGpuSink::Options sink_opts{
        .storages = std::move(storages),
        .chunk_size = config_.artifact_chunk_bytes,
        .device_id = target_device.ordinal,
    };
    loader::TargetLayoutGpuSink sink(std::move(sink_opts));

    const int concurrency = loading::resolve_materialization_concurrency(config_.num_threads, hints);
    if (source_ordered) {
      uint64_t max_dst_end = 0;
      uint64_t max_src_end = 0;
      size_t max_dst_index = 0;
      for (size_t idx = 0; idx < effective_map.segments.size(); ++idx) {
        const auto& segment = effective_map.segments[idx];
        max_dst_end = std::max<uint64_t>(max_dst_end, segment.dst_offset + segment.length);
        max_src_end = std::max<uint64_t>(max_src_end, segment.src_offset + segment.length);
        if (max_dst_end == segment.dst_offset + segment.length) {
          max_dst_index = idx;
        }
      }
      if (max_dst_end > total_size) {
        return absl::InvalidArgumentError(
            absl::StrCat(
                "materialize_into_target source-ordered map exceeds target size: total_size=",
                total_size,
                " max_dst_end=",
                max_dst_end,
                " max_src_end=",
                max_src_end,
                " segment_index=",
                max_dst_index,
                " segment_count=",
                effective_map.segments.size()));
      }
      const uint64_t max_window_bytes = std::min<uint64_t>(hints.max_buffer_bytes, 64ULL * 1024 * 1024);
      const uint64_t window_cap_bytes = std::min<uint64_t>(max_window_bytes, slice_bytes);
      loader::SourceWindowScheduler::Options sched_opts{
          .merge_max_gap_bytes = config_.options->byte_mapping.disk_source_merge_max_gap_bytes,
          .merge_max_amplification = config_.options->byte_mapping.disk_source_merge_max_amplification,
          .prefetch_depth = config_.options->byte_mapping.disk_source_prefetch_depth,
          .window_cap_bytes = window_cap_bytes,
          .path = "materialize_into_target",
      };
      loader::SourceWindowScheduler scheduler(std::move(sched_opts));
      auto exec_status = scheduler.Execute(
          effective_map,
          sources,
          sink,
          config_.runtime_context->pinned_buffer_pool(),
          timeout,
          /*use_pinned_buffers=*/true);
      if (!exec_status.ok()) {
        return absl::DataLossError(
            absl::StrCat("materialize_into_target source-ordered execution failed: ", exec_status.message()));
      }
    } else {
      const size_t num_chunks = resolve_streaming_buffer_chunks_for_transfer(
          effective_map.total_bytes, slice_bytes, config_.runtime_context->options().streaming_buffer_chunks);
      auto session_spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
          /*num_chunks=*/num_chunks, slice_bytes, config_.runtime_context->pinned_buffer_pool());
      const absl::Time streaming_buffer_init_started_at = absl::Now();
      auto init_spb_status = session_spb->initialize(
          timeout,
          make_materialize_into_target_pinned_wait_context(hints, target_device.ordinal, num_chunks, slice_bytes));
      streaming_buffer_init_elapsed = absl::Now() - streaming_buffer_init_started_at;
      if (!init_spb_status.ok()) {
        return init_spb_status;
      }
      loader::StreamingBufferAdapter adapter(session_spb);
      const absl::Time pump_started_at = absl::Now();
      auto pump_status = loader::pump_ranges(
          *plan_source,
          sink,
          adapter,
          absl::MakeSpan(ranges),
          concurrency,
          config_.runtime_context->async_runtime()->blocking_executor(),
          nullptr,
          make_pump_direct_write_options(config_.options->materialization_strategy));
      pump_elapsed = absl::Now() - pump_started_at;
      if (!pump_status.ok()) {
        return absl::DataLossError(absl::StrCat("materialize_into_target pump failed: ", pump_status.message()));
      }
      const absl::Time sink_close_started_at = absl::Now();
      auto close_status = sink.close();
      sink_close_elapsed = absl::Now() - sink_close_started_at;
      if (!close_status.ok()) {
        return absl::DataLossError(absl::StrCat("materialize_into_target sink close failed: ", close_status.message()));
      }
      sink_closed = true;
      const absl::Duration total_elapsed = absl::Now() - total_started_at;
      if (!hints.transport_request_id.empty()) {
        LOG(INFO) << "materialize_into_target.summary"
                  << " request_id=" << (hints.transport_request_id.empty() ? "<unset>" : hints.transport_request_id)
                  << " artifact_id=" << hints.artifact_id << " source_ordered=false"
                  << " direct_write_supported=" << plan_source->supports_direct_write_at()
                  << " storage_count=" << target_layout.storages.size()
                  << " mapping_segments=" << effective_map.segments.size()
                  << " total_bytes=" << effective_map.total_bytes << " num_chunks=" << num_chunks
                  << " slice_bytes=" << slice_bytes
                  << " streaming_buffer_init_ms=" << absl::ToDoubleMilliseconds(streaming_buffer_init_elapsed)
                  << " pump_ms=" << absl::ToDoubleMilliseconds(pump_elapsed)
                  << " sink_close_ms=" << absl::ToDoubleMilliseconds(sink_close_elapsed)
                  << " total_ms=" << absl::ToDoubleMilliseconds(total_elapsed);
      } else if (absl::ToDoubleMilliseconds(total_elapsed) >= 25.0) {
        VLOG(1) << "materialize_into_target.summary"
                << " request_id=<unset>"
                << " artifact_id=" << hints.artifact_id << " source_ordered=false"
                << " direct_write_supported=" << plan_source->supports_direct_write_at()
                << " storage_count=" << target_layout.storages.size()
                << " mapping_segments=" << effective_map.segments.size() << " total_bytes=" << effective_map.total_bytes
                << " num_chunks=" << num_chunks << " slice_bytes=" << slice_bytes
                << " streaming_buffer_init_ms=" << absl::ToDoubleMilliseconds(streaming_buffer_init_elapsed)
                << " pump_ms=" << absl::ToDoubleMilliseconds(pump_elapsed)
                << " sink_close_ms=" << absl::ToDoubleMilliseconds(sink_close_elapsed)
                << " total_ms=" << absl::ToDoubleMilliseconds(total_elapsed);
      }
    }
    if (!sink_closed) {
      const absl::Time sink_close_started_at = absl::Now();
      auto close_status = sink.close();
      sink_close_elapsed = absl::Now() - sink_close_started_at;
      if (!close_status.ok()) {
        return absl::DataLossError(absl::StrCat("materialize_into_target sink close failed: ", close_status.message()));
      }
    }
    if (view_plan.has_value() && view_plan->transform.requires_materialization &&
        placement == loading::TransformPlacement::kServer) {
      auto transform_status = loader::execute_transform(
          view_plan->transform,
          common::memory::MemoryLocation::GPU,
          target_layout.storages.front().base_ptr.get(),
          target_device.ordinal);
      if (!transform_status.ok()) {
        return absl::DataLossError(
            absl::StrCat("materialize_into_target view transform failed: ", transform_status.message()));
      }
    }
    return loading::MaterializeIntoTargetResult{
        .source = source_kind,
        .requested_bytes = effective_map.total_bytes,
        .committed_bytes = effective_map.total_bytes,
        .fallback_bytes = effective_map.total_bytes,
        .residual_bytes = 0,
        .collective_skip_reason = collective_skip_reason,
        .collective_handled = false,
        .direct_write_supported = false,
        .source_ordered = source_ordered,
        .dominant_executor = source_ordered ? "SourceOrderedDirectTargetExecutor" : "DirectTargetStreamingExecutor",
        .selection_reason = source_ordered ? "source_ordered_direct_target" : "direct_target_streaming",
    };
  };

  const bool requires_source_layout_remap = source_index_json.has_value();
  if (!requires_source_layout_remap) {
    auto local_source_or = make_local_canonical_source(
        config_.replica_runtime->registry(), hints.artifact_id, target_device, canonical_total_size);
    if (local_source_or.ok()) {
      return run_source(
          std::make_unique<SharedSourceLoader>(*local_source_or), loading::MaterializationSource::kLocalReplica);
    }
    if (!absl::IsNotFound(local_source_or.status())) {
      return local_source_or.status();
    }
  }

  auto gs_client = config_.runtime_context->global_store_client();
  auto comm_manager = config_.runtime_context->communication_manager();
  const bool gs_connected = gs_client && gs_client->is_connected();
  const bool comm_enabled = comm_manager && comm_manager->is_enabled();
  const auto retrieval_policy = hints.retrieval_policy();
  if (auto policy_status = loading::validate_retrieval_policy(retrieval_policy); !policy_status.ok()) {
    return policy_status;
  }
  const bool prefer_disk = retrieval_policy.preference == loading::SourcePreference::kPreferDisk;
  const bool prefer_p2p = retrieval_policy.preference == loading::SourcePreference::kPreferP2P;
  const bool allow_p2p = retrieval_policy.allow_p2p;
  const bool allow_disk = retrieval_policy.allow_disk;
  const bool has_disk_source = disk_source.has_value();
  components::WorkerIdentity local_identity = config_.runtime_context->worker_identity();
  if (!is_local_identity(local_identity)) {
    const auto& options = config_.runtime_context->options();
    if (!options.p2p_listen_host.empty()) {
      local_identity.node_address = options.p2p_listen_host;
    }
    local_identity.p2p_port = options.p2p_port;
  }

  if (prefer_disk && has_disk_source && allow_disk) {
    loading::DiskSource disk_src = *disk_source;
    disk_src.require_descriptor = tensorcast::common::is_mi2_artifact_id(hints.artifact_id);
    auto disk_or = run_source(std::make_unique<DiskLoader>(disk_src), loading::MaterializationSource::kDisk);
    if (disk_or.ok()) {
      return disk_or;
    }
    if (!gs_connected || !allow_p2p) {
      return disk_or.status();
    }
  }

  if (!gs_connected && (!has_disk_source || !allow_disk)) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (allow_p2p && !comm_enabled && (!allow_disk || !has_disk_source || prefer_p2p)) {
    return absl::FailedPreconditionError("Communication not enabled");
  }

  const auto scheduling_group_hint = to_transport_scheduling_group_hint(hints);
  const std::string_view requester_worker_id = hints.transport_requester_worker_id.empty()
      ? std::string_view(local_identity.worker_id)
      : std::string_view(hints.transport_requester_worker_id);
  const std::string_view transport_request_id = hints.transport_request_id;

  if (allow_p2p && gs_connected && !hints.artifact_id.empty()) {
    auto transport_or = gs_client->request_replica_transport(
        hints.artifact_id,
        local_identity.node_id,
        local_identity.node_address,
        local_identity.p2p_port,
        target_device,
        resolve_transport_wait_timeout_ms(hints),
        scheduling_group_hint,
        requester_worker_id,
        transport_request_id);
    if (transport_or.ok()) {
      const auto& session = *transport_or;
      const auto& remote = session.remote_replica;
      P2PSource p2p_src;
      p2p_src.comm_engine = gsl::not_null<std::shared_ptr<tensorcast::communicator::engine::Communicator>>{
          comm_manager->get_shared_engine()};
      p2p_src.size_bytes = remote.memory_size;
      p2p_src.ip = remote.node_address;
      p2p_src.port = static_cast<uint16_t>(remote.node_port);
      p2p_src.local_endpoint_id = components::derive_endpoint_id(local_identity, target_device);
      p2p_src.remote_endpoint_id = remote.endpoint_id;
      p2p_src.memory_keys = remote.remote_memory_keys;
      p2p_src.buf_sizes = remote.buffer_sizes;
      p2p_src.verification_json = remote.verification_json;
      p2p_src.enable_checksum = false;
      p2p_src.location.type = remote.memory_type;
      p2p_src.location.device_id = remote.device_id;
      fill_runtime_p2p_bindings(comm_manager, &p2p_src);
      p2p_src.request_budget = hints.request_budget;
      p2p_src.artifact_id = hints.artifact_id;
      p2p_src.transport_request_id = std::string(transport_request_id);
      if (has_disk_source && allow_disk && !prefer_p2p) {
        p2p_src.fallback_disk_dir = disk_source->path.string();
      }
      auto p2p_or = run_source(std::make_unique<P2PLoader>(p2p_src), loading::MaterializationSource::kP2P);
      auto complete_status = gs_client->complete_replica_transport(
          session.transport_id,
          p2p_or.ok() ? components::TransportCompletionOutcome::kSuccess
                      : components::TransportCompletionOutcome::kFailed,
          p2p_or.ok() ? std::string_view{} : std::string_view(p2p_or.status().ToString()));
      if (!complete_status.ok()) {
        LOG(WARNING) << "complete_replica_transport returned error: " << complete_status;
      }
      if (p2p_or.ok()) {
        return p2p_or;
      }
      if (!allow_disk || !has_disk_source || prefer_p2p) {
        return p2p_or.status();
      }
    } else if (!allow_disk || !has_disk_source) {
      return transport_or.status();
    }
  }

  if (allow_disk && has_disk_source) {
    loading::DiskSource disk_src = *disk_source;
    disk_src.require_descriptor = tensorcast::common::is_mi2_artifact_id(hints.artifact_id);
    return run_source(std::make_unique<DiskLoader>(disk_src), loading::MaterializationSource::kDisk);
  }

  if (!allow_p2p && !allow_disk) {
    return absl::FailedPreconditionError("source_policy disallows P2P and disk for materialize_into_target");
  }

  return absl::FailedPreconditionError("materialize_into_target requires disk source or Global Store connectivity");
}

absl::StatusOr<loading::MaterializeIntoTargetResult> MaterializationFacade::materialize_mapped_into_target(
    const DeviceKey& target_device,
    const strategy::PreparedSourceBoundExecutionPlan& prepared_execution,
    const loading::MaterializeHints& hints,
    std::optional<loading::DiskSource> disk_source) {
  const auto& resolved_plan = prepared_execution.resolved_plan;
  const auto* source_bound_strategy =
      prepared_execution.strategy_plan.has_value() ? &*prepared_execution.strategy_plan : nullptr;
  auto effective_hints_or = build_effective_mapped_hints(resolved_plan, hints);
  if (!effective_hints_or.ok()) {
    return effective_hints_or.status();
  }
  const loading::MaterializeHints& request_hints = *effective_hints_or;
  if (source_bound_strategy == nullptr) {
    return absl::FailedPreconditionError("materialize_mapped_into_target requires explicit source_bound_strategy_plan");
  }
  const strategy::SourceBoundLanePlan& source_bound_lane_plan = source_bound_strategy->lane_plan;
  if (source_bound_lane_plan.mode == strategy::SourceBoundExecutionMode::kRejected) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            "materialize_mapped_into_target source_bound_strategy_plan rejected request before execution setup "
            "(planner_reject_reason_buckets=",
            format_reject_reason_buckets(source_bound_strategy->summary.planner_reject_reason_buckets),
            ")"));
  }
  const loading::IntoTargetLayout& target_layout = resolved_plan.target_layout;
  const std::string_view canonical_index_json = resolved_plan.canonical_index_json;
  const uint64_t generation = resolved_plan.generation;
  const auto& representation_work_plan = *resolved_plan.representation_work_plan;
  const bool strategy_uses_collective_lane = source_bound_execution_mode_uses_collective(source_bound_lane_plan.mode);
  const bool require_explicit_executor_map =
      source_bound_execution_mode_requires_executor_source_map(source_bound_lane_plan.mode) ||
      work_plan_requires_explicit_executor_data_map(representation_work_plan);
  if (require_explicit_executor_map && source_bound_lane_plan.generic_backend_map.segments.empty()) {
    return absl::FailedPreconditionError(
        "materialize_mapped_into_target explicit source_bound_strategy_plan is missing executor fallback map for "
        "source-bound data lanes");
  }
  if (strategy_uses_collective_lane && source_bound_lane_plan.collective_lane_map.segments.empty()) {
    return absl::FailedPreconditionError(
        "materialize_mapped_into_target explicit source_bound_strategy_plan is missing collective compatibility map "
        "for collective-admitted source-bound lanes");
  }
  if (strategy_uses_collective_lane && !request_hints.collective_load_group.has_value()) {
    return absl::FailedPreconditionError(
        "materialize_mapped_into_target explicit source_bound_strategy_plan selected a collective lane but "
        "collective_load_group is missing");
  }
  if (strategy_uses_collective_lane && !disk_source.has_value()) {
    return absl::FailedPreconditionError(
        "materialize_mapped_into_target explicit source_bound_strategy_plan selected a collective lane but no disk "
        "source was provided");
  }
  const bool has_fill_items = work_plan_has_fill_items(representation_work_plan);
  const loader::ByteRangeMap semantic_residual_map = representation_work_plan.residual_fallback_map;
  const loader::ByteRangeMap& executor_private_map = source_bound_lane_plan.generic_backend_map;
  const loader::ByteRangeMap& collective_data_map = source_bound_lane_plan.collective_lane_map;
  const uint64_t semantic_residual_bytes = byte_range_map_covered_bytes(semantic_residual_map);
  const uint64_t local_typed_bytes = local_typed_work_bytes(representation_work_plan);

  if (target_device.type != DeviceType::GPU) {
    return absl::InvalidArgumentError("materialize_mapped_into_target requires GPU target device");
  }
  if (target_layout.storages.empty()) {
    return absl::InvalidArgumentError("materialize_mapped_into_target requires at least one target storage");
  }
  uint64_t total_size = target_layout.total_size;
  uint64_t computed_total = 0;
  for (const auto& storage : target_layout.storages) {
    if (storage.length == 0) {
      return absl::InvalidArgumentError("materialize_mapped_into_target requires non-empty storage length");
    }
    if (storage.length > std::numeric_limits<uint64_t>::max() - computed_total) {
      return absl::OutOfRangeError("materialize_mapped_into_target storage length overflow");
    }
    computed_total += storage.length;
  }
  if (total_size == 0) {
    total_size = computed_total;
  } else if (total_size != computed_total) {
    return absl::InvalidArgumentError("materialize_mapped_into_target total_size does not match storage lengths");
  }
  if (total_size == 0) {
    return absl::InvalidArgumentError("materialize_mapped_into_target requires total_size > 0");
  }
  if (executor_private_map.total_bytes > total_size) {
    return absl::InvalidArgumentError("materialize_mapped_into_target mapping total_bytes exceeds target size");
  }
  auto local_typed_pad_map_or = build_local_typed_pad_map(representation_work_plan, total_size);
  if (!local_typed_pad_map_or.ok()) {
    return local_typed_pad_map_or.status();
  }
  const loader::ByteRangeMap local_typed_pad_map = *local_typed_pad_map_or;

  auto canonical_total_or = compute_logical_total_size(canonical_index_json);
  if (!canonical_total_or.ok()) {
    return canonical_total_or.status();
  }
  const uint64_t canonical_total_size = *canonical_total_or;

  std::optional<loader::ViewPlan> view_plan;
  if (request_hints.variant && request_hints.variant->cached_plan.has_value()) {
    view_plan = *request_hints.variant->cached_plan;
  }
  if (view_plan.has_value() && view_plan->transform.requires_materialization) {
    return absl::InvalidArgumentError("materialize_mapped_into_target does not support view transforms");
  }

  auto index_hash = [](std::string_view index_json) -> std::string {
    auto mh_or = common::compute_index_multihash(std::optional<std::string>(index_json), "");
    if (mh_or.ok()) {
      return *mh_or;
    }
    const size_t fallback_hash = std::hash<std::string_view>{}(index_json);
    return absl::StrCat("raw:", fallback_hash);
  };

  const std::optional<std::string_view> source_index_json =
      (request_hints.disk_metadata && request_hints.disk_metadata->source_index_json.has_value())
      ? std::optional<std::string_view>(*request_hints.disk_metadata->source_index_json)
      : std::nullopt;
  const auto& strategy_config = config_.options->materialization_strategy;
  const bool source_bound_plan_uses_local_mapped =
      source_bound_execution_mode_uses_local_mapped(source_bound_lane_plan.mode) ||
      source_bound_lane_plan.local_mapped_typed_selected;

  if (executor_private_map.segments.empty() && has_fill_items &&
      !work_plan_has_scalar_fill_items(representation_work_plan)) {
    std::vector<loader::TargetStorage> storages;
    storages.reserve(target_layout.storages.size());
    uint64_t range_cursor = 0;
    for (const auto& storage : target_layout.storages) {
      if (storage.length == 0) {
        return absl::InvalidArgumentError("materialize_mapped_into_target requires non-empty storage length");
      }
      if (storage.length > std::numeric_limits<size_t>::max()) {
        return absl::OutOfRangeError("materialize_mapped_into_target storage length exceeds host limits");
      }
      storages.push_back(loader::TargetStorage{storage.base_ptr, storage.length});
      range_cursor += storage.length;
    }
    if (range_cursor != total_size) {
      return absl::InvalidArgumentError("materialize_mapped_into_target storage ranges do not span total_size");
    }
    loader::TargetLayoutGpuSink sink(
        loader::TargetLayoutGpuSink::Options{
            .storages = std::move(storages),
            .chunk_size = config_.artifact_chunk_bytes,
            .device_id = target_device.ordinal,
        });
    auto fill_status = execute_fill_work_items(
        absl::MakeSpan(representation_work_plan.items), sink, /*source=*/nullptr, config_.artifact_chunk_bytes);
    if (!fill_status.ok()) {
      return absl::DataLossError(
          absl::StrCat("materialize_mapped_into_target fill execution failed: ", fill_status.message()));
    }
    if (representation_work_plan.committed_bytes >= total_size) {
      auto close_status = sink.close();
      if (!close_status.ok()) {
        return absl::DataLossError(
            absl::StrCat("materialize_mapped_into_target fill sink close failed: ", close_status.message()));
      }
    }
    return loading::MaterializeIntoTargetResult{
        .source = loading::MaterializationSource::kUnspecified,
        .requested_bytes = total_size,
        .committed_bytes = local_typed_bytes,
        .fallback_bytes = 0,
        .residual_bytes = semantic_residual_bytes,
        .actual_collective_committed_bytes = 0,
        .actual_local_typed_bytes = local_typed_bytes,
        .actual_generic_backend_bytes = 0,
        .collective_handled = false,
        .direct_write_supported = false,
        .source_ordered = false,
        .dominant_executor = "BindingRealizationPlanExecutor",
        .selection_reason = "binding_realization_plan",
    };
  }

  auto get_map_ptr = [&](bool use_source_layout) -> absl::StatusOr<std::shared_ptr<loader::ByteRangeMap>> {
    const std::string canonical_hash = index_hash(canonical_index_json);
    std::string plan_key = absl::StrCat(generation, ":canon:", canonical_hash);
    if (use_source_layout && source_index_json.has_value()) {
      plan_key = absl::StrCat(plan_key, ":src:", index_hash(*source_index_json));
    }

    std::shared_ptr<loader::ByteRangeMap> map_ptr;
    {
      absl::MutexLock lock(&byte_range_map_mu_);
      auto it = byte_range_map_cache_.find(plan_key);
      if (it != byte_range_map_cache_.end()) {
        return it->second;
      }
    }

    absl::StatusOr<loader::ByteRangeMap> map_or;
    if (use_source_layout && source_index_json.has_value()) {
      map_or = loader::build_byte_range_map_from_canonical_and_source_index_json(
          canonical_index_json, *source_index_json, canonical_total_size);
    } else {
      map_or = loader::build_byte_range_map_from_canonical_index_json(canonical_index_json, canonical_total_size);
    }
    if (!map_or.ok()) {
      return map_or.status();
    }
    map_ptr = std::make_shared<loader::ByteRangeMap>(std::move(*map_or));
    {
      absl::MutexLock lock(&byte_range_map_mu_);
      byte_range_map_cache_.emplace(plan_key, map_ptr);
    }
    return map_ptr;
  };

  auto run_source =
      [&](std::unique_ptr<IArtifactLoader> loader,
          loading::MaterializationSource source_kind,
          strategy::SourceByteSpace source_byte_space) -> absl::StatusOr<loading::MaterializeIntoTargetResult> {
    const auto source_total_start = std::chrono::steady_clock::now();
    auto init_status = loader->initialize();
    const auto init_done = std::chrono::steady_clock::now();
    if (!init_status.ok()) {
      return init_status;
    }
    auto source_or = loader->open_source();
    const auto source_open_done = std::chrono::steady_clock::now();
    if (!source_or.ok()) {
      return source_or.status();
    }

    const bool source_is_view = source_byte_space == strategy::SourceByteSpace::kView;
    const bool use_source_layout =
        !source_is_view && (source_kind == loading::MaterializationSource::kDisk) && source_index_json.has_value();
    const bool source_exposes_target_byte_space =
        source_is_view && request_hints.variant && request_hints.variant->view_id.has_value() && !view_plan.has_value();
    std::optional<loader::ByteRangeMap> source_layout_map;
    if (use_source_layout) {
      auto map_ptr_or = get_map_ptr(true);
      if (!map_ptr_or.ok()) {
        return map_ptr_or.status();
      }
      source_layout_map = **map_ptr_or;
    }
    auto effective_maps_or = derive_effective_source_maps(
        executor_private_map,
        collective_data_map,
        local_typed_pad_map,
        source_exposes_target_byte_space,
        view_plan,
        source_is_view,
        use_source_layout,
        source_layout_map);
    if (!effective_maps_or.ok()) {
      return effective_maps_or.status();
    }
    const EffectiveSourceMaps& effective_maps = *effective_maps_or;
    const loader::ByteRangeMap& effective_map = effective_maps.executor_effective_map;
    const loader::ByteRangeMap& effective_data_map = effective_maps.executor_effective_data_map;
    const loader::ByteRangeMap& collective_effective_data_map = effective_maps.collective_effective_data_map;
    const loader::ByteRangeMap& generic_effective_data_map = effective_maps.generic_effective_data_map;
    auto additional_pad_map_or =
        subtract_byte_range_map_by_dst_ranges(effective_maps.executor_effective_pad_map, local_typed_pad_map);
    if (!additional_pad_map_or.ok()) {
      return additional_pad_map_or.status();
    }
    const loader::ByteRangeMap additional_executor_pad_map = *additional_pad_map_or;
    const uint64_t effective_local_typed_bytes =
        local_typed_bytes + byte_range_map_covered_bytes(additional_executor_pad_map);

    std::shared_ptr<loader::SeekableSource> base_source = std::move(*source_or);
    std::vector<std::shared_ptr<loader::SeekableSource>> sources;
    sources.emplace_back(base_source);
    const uint32_t required_source_count =
        std::max(effective_map.num_sources, collective_effective_data_map.num_sources);
    if (required_source_count > sources.size()) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "materialize_mapped_into_target resolved ",
              sources.size(),
              " source(s) but execution maps require ",
              required_source_count,
              "; composite source resolution is not wired for source_kind=",
              static_cast<int>(source_kind),
              " source_byte_space=",
              static_cast<int>(source_byte_space)));
    }

    const bool map_crosses_storage_boundaries =
        byte_range_map_crosses_target_storage_boundaries(effective_map, target_layout);
    const bool source_ordered = use_source_layout && config_.options->byte_mapping.disk_source_ordered_read &&
        allow_source_ordered_for_mapped(strategy_config) && !map_crosses_storage_boundaries;
    if (use_source_layout && config_.options->byte_mapping.disk_source_ordered_read && map_crosses_storage_boundaries) {
      LOG(INFO) << "materialize_mapped_into_target disabling source-ordered fast path because byte-range segments "
                   "cross target storage boundaries"
                << " map_segments=" << effective_map.segments.size()
                << " target_storages=" << target_layout.storages.size();
    }
    std::unique_ptr<loader::SeekableSource> plan_source;
    bool direct_write_supported = false;
    double map_build_sec = 0.0;
    if (!source_ordered) {
      const auto map_build_start = std::chrono::steady_clock::now();
      loader::ByteRangeCompiler compiler(config_.options->byte_mapping, "materialize_mapped_into_target");
      auto program_or = compiler.Compile(effective_map);
      if (!program_or.ok()) {
        return program_or.status();
      }
      loader::ByteRangeMappedSource::Options map_opts{
          .path = "materialize_mapped_into_target",
          .transport_request_id = std::string(request_hints.transport_request_id),
          .enable_direct_write_at = config_.options->byte_mapping.enable_direct_write_at,
      };
      auto mapped_or =
          loader::ByteRangeMappedSource::Create(effective_map, *program_or, std::move(sources), std::move(map_opts));
      if (!mapped_or.ok()) {
        return mapped_or.status();
      }
      plan_source = std::move(*mapped_or);
      if (!plan_source) {
        return absl::InternalError("materialize_mapped_into_target failed to build mapped source");
      }
      direct_write_supported = plan_source->supports_direct_write_at();
      map_build_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - map_build_start).count();
    }
    const strategy::ResolvedSourceBinding source_binding{
        .source = source_kind,
        .source_byte_space = source_byte_space,
        .source_layout_available = use_source_layout,
        .direct_write_capable = direct_write_supported,
        .collective_eligible = source_kind == loading::MaterializationSource::kDisk &&
            request_hints.collective_load_group.has_value() && allow_collective_mapped_executor(strategy_config) &&
            strategy_uses_collective_lane,
    };

    const size_t slice_bytes = config_.runtime_context->tx_slice_bytes();
    if (slice_bytes == 0 || config_.artifact_chunk_bytes == 0) {
      return absl::FailedPreconditionError("tx_slice_bytes or artifact_chunk_bytes is zero");
    }
    const std::chrono::milliseconds timeout =
        request_hints.pinned_timeout.count() > 0 ? request_hints.pinned_timeout : config_.pinned_memory_timeout;

    std::vector<loader::TargetStorage> storages;
    storages.reserve(target_layout.storages.size());
    std::vector<loader::Range> ranges;
    ranges.reserve(target_layout.storages.size());
    uint64_t range_cursor = 0;
    for (const auto& storage : target_layout.storages) {
      if (storage.length == 0) {
        return absl::InvalidArgumentError("materialize_mapped_into_target requires non-empty storage length");
      }
      if (storage.length > std::numeric_limits<size_t>::max()) {
        return absl::OutOfRangeError("materialize_mapped_into_target storage length exceeds host limits");
      }
      storages.push_back(loader::TargetStorage{storage.base_ptr, storage.length});
      ranges.emplace_back(range_cursor, static_cast<size_t>(storage.length));
      range_cursor += storage.length;
    }
    if (range_cursor != total_size) {
      return absl::InvalidArgumentError("materialize_mapped_into_target storage ranges do not span total_size");
    }

    std::string collective_skip_reason = source_binding.collective_eligible
        ? std::string{}
        : dominant_collective_reject_reason(source_bound_strategy->summary);

    auto execute_local_typed_work = [&](loader::TargetLayoutGpuSink& sink) -> absl::Status {
      if (has_fill_items) {
        auto fill_status = execute_fill_work_items(
            absl::MakeSpan(representation_work_plan.items), sink, base_source.get(), config_.artifact_chunk_bytes);
        if (!fill_status.ok()) {
          return fill_status;
        }
      }
      if (!additional_executor_pad_map.segments.empty()) {
        RepresentationWorkItem pad_item;
        pad_item.kind = RepresentationWorkItemKind::kPadFill;
        pad_item.byte_range_map = additional_executor_pad_map;
        pad_item.committed_bytes = byte_range_map_covered_bytes(additional_executor_pad_map);
        auto pad_status = execute_fill_work_items(
            absl::MakeSpan(&pad_item, 1), sink, /*source=*/nullptr, config_.artifact_chunk_bytes);
        if (!pad_status.ok()) {
          return pad_status;
        }
      }
      return absl::OkStatus();
    };

    const bool local_mapped_required_by_plan = source_bound_plan_uses_local_mapped;
    const bool local_mapped_executor_available =
        strategy_config.enable_tensor_aware_mapped_executor && strategy_config.allow_mixed_execution;
    if (local_mapped_required_by_plan && !local_mapped_executor_available) {
      return absl::FailedPreconditionError(
          "materialize_mapped_into_target source_bound_strategy_plan requires local mapped typed execution but the "
          "tensor-aware mapped executor is unavailable");
    }
    const bool local_mapped_selected = local_mapped_executor_available &&
        (strategy_config.executor_preference == StrategyConfig::ExecutorPreference::kTensorAwareLocal ||
         local_mapped_required_by_plan);
    const uint64_t planned_generic_after_local_mapped =
        planned_generic_backend_bytes_after_local_mapped(source_bound_strategy->summary);
    const auto setup_done = std::chrono::steady_clock::now();
    LOG(INFO) << "tc_profile materialize_mapped_into_target source_setup"
              << " artifact_id=" << request_hints.artifact_id << " target_device=" << target_device.ordinal
              << " source_kind=" << static_cast<int>(source_kind)
              << " source_byte_space=" << static_cast<int>(source_byte_space) << " total_size=" << total_size
              << " effective_map_bytes=" << effective_map.total_bytes
              << " effective_data_bytes=" << effective_data_map.total_bytes
              << " collective_data_bytes=" << collective_effective_data_map.total_bytes
              << " generic_data_bytes=" << generic_effective_data_map.total_bytes
              << " effective_local_typed_bytes=" << effective_local_typed_bytes
              << " source_bound_plan_uses_local_mapped=" << source_bound_plan_uses_local_mapped
              << " local_mapped_selected=" << local_mapped_selected
              << " collective_eligible=" << source_binding.collective_eligible
              << " init_sec=" << std::chrono::duration<double>(init_done - source_total_start).count()
              << " open_sec=" << std::chrono::duration<double>(source_open_done - init_done).count()
              << " setup_after_open_sec=" << std::chrono::duration<double>(setup_done - source_open_done).count()
              << " total_setup_sec=" << std::chrono::duration<double>(setup_done - source_total_start).count();

    if (source_binding.collective_eligible) {
      if (auto* disk_loader = dynamic_cast<DiskLoader*>(loader.get()); disk_loader != nullptr) {
        auto shared_or = disk_loader->shared_context();
        if (!shared_or.ok()) {
          LOG(WARNING) << "materialize_mapped_into_target collective path unavailable: shared_context error="
                       << shared_or.status();
          collective_skip_reason = "disk_shared_context_unavailable";
        } else {
          const replica::CollectiveMappedTargetLoadOptions collective_options{
              .chunk_bytes = std::min<uint64_t>(slice_bytes, total_size),
              .merge_max_gap_bytes = config_.options->byte_mapping.disk_source_merge_max_gap_bytes,
              .merge_max_amplification = config_.options->byte_mapping.disk_source_merge_max_amplification,
              .strategy_config = strategy_config,
          };
          const auto collective_start = std::chrono::steady_clock::now();
          auto collective_result = execute_collective_mapped_target_load(
              config_.hooks,
              replica::CollectiveMappedTargetLoadRequest{
                  .artifact_id = request_hints.artifact_id,
                  .group = *request_hints.collective_load_group,
                  .disk_context = *shared_or,
                  .representation_work_plan = representation_work_plan,
                  .collective_lane_map = collective_effective_data_map,
                  .target_layout = target_layout,
                  .device_id = target_device.ordinal,
              },
              config_.runtime_context->pinned_buffer_pool(),
              timeout,
              collective_options);
          const auto collective_done = std::chrono::steady_clock::now();
          LOG(INFO) << "tc_profile materialize_mapped_into_target collective_call"
                    << " artifact_id=" << request_hints.artifact_id << " target_device=" << target_device.ordinal
                    << " handled=" << collective_result.handled << " status_ok=" << collective_result.status.ok()
                    << " unique_source_bytes=" << collective_result.metrics.unique_source_bytes
                    << " peer_transfer_bytes=" << collective_result.metrics.peer_transfer_bytes
                    << " batch_count=" << collective_result.metrics.batch_count
                    << " skip_reason=" << collective_result.skip_reason
                    << " sec=" << std::chrono::duration<double>(collective_done - collective_start).count();
          if (collective_result.handled) {
            if (!collective_result.status.ok()) {
              return absl::DataLossError(
                  absl::StrCat(
                      "materialize_mapped_into_target collective execution failed: ",
                      collective_result.status.message()));
            }
            const uint64_t generic_residual_bytes = byte_range_map_covered_bytes(generic_effective_data_map);
            uint64_t local_mapped_bytes = 0;
            uint64_t generic_backend_bytes = generic_residual_bytes;
            loader::ByteRangeMap generic_backend_data_map = generic_effective_data_map;
            double local_mapped_sec = 0.0;
            if (generic_residual_bytes > 0 && local_mapped_selected) {
              const auto local_start = std::chrono::steady_clock::now();
              auto local_result = replica::try_local_mapped_target_load(
                  replica::LocalMappedTargetLoadRequest{
                      .artifact_id = request_hints.artifact_id,
                      .disk_context = *shared_or,
                      .representation_work_plan = representation_work_plan,
                      .data_lane_map = generic_effective_data_map,
                      .target_layout = target_layout,
                      .strategy_config = strategy_config,
                      .device_id = target_device.ordinal,
                  },
                  config_.runtime_context->pinned_buffer_pool(),
                  timeout,
                  collective_options);
              const auto local_done = std::chrono::steady_clock::now();
              local_mapped_sec = std::chrono::duration<double>(local_done - local_start).count();
              LOG(INFO) << "tc_profile materialize_mapped_into_target local_mapped_continuation_call"
                        << " artifact_id=" << request_hints.artifact_id << " target_device=" << target_device.ordinal
                        << " handled=" << local_result.handled << " status_ok=" << local_result.status.ok()
                        << " handled_bytes=" << local_result.handled_bytes
                        << " residual_bytes=" << local_result.residual_data_map.total_bytes
                        << " skip_reason=" << local_result.skip_reason << " sec=" << local_mapped_sec;
              if (local_result.handled) {
                if (!local_result.status.ok()) {
                  return absl::Status(
                      local_result.status.code(),
                      absl::StrCat(
                          "materialize_mapped_into_target local mapped continuation failed: ",
                          local_result.status.message()));
                }
                local_mapped_bytes = local_result.handled_bytes;
                generic_backend_data_map = std::move(local_result.residual_data_map);
                generic_backend_bytes = byte_range_map_covered_bytes(generic_backend_data_map);
                if (generic_backend_bytes > planned_generic_after_local_mapped) {
                  return absl::FailedPreconditionError(
                      absl::StrCat(
                          "materialize_mapped_into_target local mapped continuation produced unplanned generic "
                          "residual bytes: actual=",
                          generic_backend_bytes,
                          " planned=",
                          planned_generic_after_local_mapped));
                }
                if (generic_backend_bytes > 0 && !strategy_config.allow_mixed_execution) {
                  return absl::FailedPreconditionError(
                      "materialize_mapped_into_target local mapped continuation produced generic residual but mixed "
                      "execution is disabled");
                }
              } else {
                return absl::FailedPreconditionError(
                    absl::StrCat(
                        "materialize_mapped_into_target selected local mapped continuation but admission returned "
                        "handled=false: ",
                        local_result.skip_reason.empty() ? "unavailable" : local_result.skip_reason));
              }
            }
            double continuation_sec = 0.0;
            if (generic_backend_bytes > 0 || effective_local_typed_bytes > 0) {
              const auto continuation_start = std::chrono::steady_clock::now();
              loader::TargetLayoutGpuSink continuation_sink(
                  loader::TargetLayoutGpuSink::Options{
                      .storages = storages,
                      .chunk_size = config_.artifact_chunk_bytes,
                      .device_id = target_device.ordinal,
                  });
              if (generic_backend_bytes > 0) {
                auto generic_status = execute_sparse_generic_backend_map(
                    *base_source, generic_backend_data_map, continuation_sink, slice_bytes);
                if (!generic_status.ok()) {
                  return absl::DataLossError(
                      absl::StrCat(
                          "materialize_mapped_into_target residual generic execution failed: ",
                          generic_status.message()));
                }
              }
              if (effective_local_typed_bytes > 0) {
                auto fill_status = execute_local_typed_work(continuation_sink);
                if (!fill_status.ok()) {
                  return absl::DataLossError(
                      absl::StrCat(
                          "materialize_mapped_into_target local typed execution failed: ", fill_status.message()));
                }
              }
              continuation_sec =
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - continuation_start).count();
            }
            const uint64_t collective_bytes = byte_range_map_covered_bytes(collective_effective_data_map);
            const strategy::ExecutionCommitReport collective_report{
                .source = source_kind,
                .requested_bytes = total_size,
                .committed_bytes = collective_bytes + generic_residual_bytes + effective_local_typed_bytes,
                .fallback_bytes = generic_backend_bytes,
                .residual_bytes = semantic_residual_bytes,
                .actual_collective_committed_bytes = collective_bytes,
                .actual_local_typed_bytes = local_mapped_bytes + effective_local_typed_bytes,
                .actual_generic_backend_bytes = generic_backend_bytes,
                .collective_metrics = collective_result.metrics,
                .collective_skip_reason = {},
                .collective_handled = true,
                .direct_write_supported = false,
                .source_ordered = false,
                .dominant_executor = local_mapped_bytes > 0
                    ? (generic_backend_bytes > 0
                           ? "OwnerFileCollectiveExecutor+TensorMappedLocalExecutor+GenericByteRangeExecutor"
                           : "OwnerFileCollectiveExecutor+TensorMappedLocalExecutor")
                    : "OwnerFileCollectiveExecutor",
                .selection_reason = !source_bound_lane_plan.selection_reason.empty()
                    ? (local_mapped_bytes > 0
                           ? absl::StrCat(source_bound_lane_plan.selection_reason, ";local_mapped_tensor_aware")
                           : source_bound_lane_plan.selection_reason)
                    : (local_mapped_bytes > 0 ? "collective_first_mixed;local_mapped_tensor_aware"
                                              : "collective_first_mixed"),
            };
            LOG(INFO) << "materialize_mapped_into_target execution_commit"
                      << " source_kind=" << static_cast<int>(collective_report.source)
                      << " committed_bytes=" << collective_report.committed_bytes
                      << " actual_collective_committed_bytes=" << collective_report.actual_collective_committed_bytes
                      << " actual_local_typed_bytes=" << collective_report.actual_local_typed_bytes
                      << " actual_generic_backend_bytes=" << collective_report.actual_generic_backend_bytes
                      << " fallback_bytes=" << collective_report.fallback_bytes
                      << " collective_handled=" << collective_report.collective_handled
                      << " dominant_executor=" << collective_report.dominant_executor;
            LOG(INFO) << "tc_profile materialize_mapped_into_target collective_path timings"
                      << " artifact_id=" << request_hints.artifact_id << " target_device=" << target_device.ordinal
                      << " setup_sec=" << std::chrono::duration<double>(setup_done - source_total_start).count()
                      << " collective_sec=" << std::chrono::duration<double>(collective_done - collective_start).count()
                      << " local_mapped_sec=" << local_mapped_sec << " continuation_sec=" << continuation_sec
                      << " total_sec="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - source_total_start).count()
                      << " committed_bytes=" << collective_report.committed_bytes
                      << " collective_bytes=" << collective_report.actual_collective_committed_bytes
                      << " local_mapped_bytes=" << local_mapped_bytes
                      << " generic_backend_bytes=" << generic_backend_bytes
                      << " effective_local_typed_bytes=" << effective_local_typed_bytes;
            return loading::MaterializeIntoTargetResult{
                .source = source_kind,
                .requested_bytes = collective_report.requested_bytes,
                .committed_bytes = collective_report.committed_bytes,
                .fallback_bytes = collective_report.fallback_bytes,
                .residual_bytes = collective_report.residual_bytes,
                .actual_collective_committed_bytes = collective_report.actual_collective_committed_bytes,
                .actual_local_typed_bytes = collective_report.actual_local_typed_bytes,
                .actual_generic_backend_bytes = collective_report.actual_generic_backend_bytes,
                .collective_unique_source_bytes = collective_report.collective_metrics.unique_source_bytes,
                .collective_peer_transfer_bytes = collective_report.collective_metrics.peer_transfer_bytes,
                .collective_peak_temporary_bytes = collective_report.collective_metrics.peak_temporary_bytes,
                .collective_batch_count = collective_report.collective_metrics.batch_count,
                .collective_dedup_saving_bytes = collective_report.collective_metrics.dedup_saving_bytes,
                .collective_skip_reason = collective_report.collective_skip_reason,
                .collective_handled = collective_report.collective_handled,
                .direct_write_supported = collective_report.direct_write_supported,
                .source_ordered = collective_report.source_ordered,
                .dominant_executor = collective_report.dominant_executor,
                .selection_reason = collective_report.selection_reason,
            };
          }
          collective_skip_reason =
              collective_result.skip_reason.empty() ? "collective_executor_unhandled" : collective_result.skip_reason;
          if (strategy_uses_collective_lane) {
            return absl::FailedPreconditionError(
                absl::StrCat(
                    "collective lane selected by explicit source-bound strategy plan but collective executor returned "
                    "handled=false: ",
                    collective_skip_reason));
          }
        }
      } else {
        collective_skip_reason = "disk_loader_unavailable";
      }
    }
    if (request_hints.require_collective_execution || source_bound_lane_plan.require_collective_success) {
      return absl::FailedPreconditionError(
          collective_skip_reason.empty()
              ? "collective requested but the selected source could not be handled without generic fallback"
              : absl::StrCat(
                    "collective requested but the selected source could not be handled without generic fallback: ",
                    collective_skip_reason));
    }

    if (!strategy_uses_collective_lane && source_kind == loading::MaterializationSource::kDisk &&
        local_mapped_selected && effective_data_map.total_bytes > 0) {
      if (auto* disk_loader = dynamic_cast<DiskLoader*>(loader.get()); disk_loader != nullptr) {
        auto shared_or = disk_loader->shared_context();
        if (shared_or.ok()) {
          const replica::CollectiveMappedTargetLoadOptions local_options{
              .chunk_bytes = std::min<uint64_t>(slice_bytes, total_size),
              .merge_max_gap_bytes = config_.options->byte_mapping.disk_source_merge_max_gap_bytes,
              .merge_max_amplification = config_.options->byte_mapping.disk_source_merge_max_amplification,
              .strategy_config = strategy_config,
          };
          const auto local_start = std::chrono::steady_clock::now();
          auto local_result = replica::try_local_mapped_target_load(
              replica::LocalMappedTargetLoadRequest{
                  .artifact_id = request_hints.artifact_id,
                  .disk_context = *shared_or,
                  .representation_work_plan = representation_work_plan,
                  .data_lane_map = effective_data_map,
                  .target_layout = target_layout,
                  .strategy_config = strategy_config,
                  .device_id = target_device.ordinal,
              },
              config_.runtime_context->pinned_buffer_pool(),
              timeout,
              local_options);
          const auto local_done = std::chrono::steady_clock::now();
          const double local_mapped_sec = std::chrono::duration<double>(local_done - local_start).count();
          LOG(INFO) << "tc_profile materialize_mapped_into_target local_mapped_call"
                    << " artifact_id=" << request_hints.artifact_id << " target_device=" << target_device.ordinal
                    << " handled=" << local_result.handled << " status_ok=" << local_result.status.ok()
                    << " handled_bytes=" << local_result.handled_bytes
                    << " residual_bytes=" << local_result.residual_data_map.total_bytes
                    << " skip_reason=" << local_result.skip_reason << " sec=" << local_mapped_sec;
          if (local_result.handled) {
            if (!local_result.status.ok()) {
              return absl::Status(
                  local_result.status.code(),
                  absl::StrCat(
                      "materialize_mapped_into_target local mapped execution failed: ", local_result.status.message()));
            }
            const uint64_t local_mapped_bytes = local_result.handled_bytes;
            const loader::ByteRangeMap local_residual_data_map = std::move(local_result.residual_data_map);
            const uint64_t generic_backend_bytes = byte_range_map_covered_bytes(local_residual_data_map);
            if (generic_backend_bytes > planned_generic_after_local_mapped) {
              return absl::FailedPreconditionError(
                  absl::StrCat(
                      "materialize_mapped_into_target local mapped execution produced unplanned generic residual "
                      "bytes: actual=",
                      generic_backend_bytes,
                      " planned=",
                      planned_generic_after_local_mapped));
            }
            if (generic_backend_bytes > 0 && !strategy_config.allow_mixed_execution) {
              return absl::FailedPreconditionError(
                  "materialize_mapped_into_target local mapped execution produced generic residual but mixed execution "
                  "is disabled");
            }
            double continuation_sec = 0.0;
            if (generic_backend_bytes > 0 || effective_local_typed_bytes > 0) {
              const auto continuation_start = std::chrono::steady_clock::now();
              loader::TargetLayoutGpuSink continuation_sink(
                  loader::TargetLayoutGpuSink::Options{
                      .storages = storages,
                      .chunk_size = config_.artifact_chunk_bytes,
                      .device_id = target_device.ordinal,
                  });
              if (generic_backend_bytes > 0) {
                auto generic_status = execute_sparse_generic_backend_map(
                    *base_source, local_residual_data_map, continuation_sink, slice_bytes);
                if (!generic_status.ok()) {
                  return absl::DataLossError(
                      absl::StrCat(
                          "materialize_mapped_into_target local mapped residual generic execution failed: ",
                          generic_status.message()));
                }
              }
              auto fill_status = execute_local_typed_work(continuation_sink);
              if (!fill_status.ok()) {
                return absl::DataLossError(
                    absl::StrCat(
                        "materialize_mapped_into_target local typed execution failed: ", fill_status.message()));
              }
              continuation_sec =
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - continuation_start).count();
            }
            const strategy::ExecutionCommitReport local_report{
                .source = source_kind,
                .requested_bytes = total_size,
                .committed_bytes = local_mapped_bytes + generic_backend_bytes + effective_local_typed_bytes,
                .fallback_bytes = generic_backend_bytes,
                .residual_bytes = semantic_residual_bytes,
                .actual_collective_committed_bytes = 0,
                .actual_local_typed_bytes = local_mapped_bytes + effective_local_typed_bytes,
                .actual_generic_backend_bytes = generic_backend_bytes,
                .collective_metrics = local_result.metrics,
                .collective_skip_reason = collective_skip_reason,
                .collective_handled = false,
                .direct_write_supported = false,
                .source_ordered = false,
                .dominant_executor = generic_backend_bytes > 0 ? "TensorMappedLocalExecutor+GenericByteRangeExecutor"
                                                               : "TensorMappedLocalExecutor",
                .selection_reason = !source_bound_lane_plan.selection_reason.empty()
                    ? absl::StrCat(source_bound_lane_plan.selection_reason, ";local_mapped_tensor_aware")
                    : "local_mapped_tensor_aware",
            };
            LOG(INFO) << "materialize_mapped_into_target execution_commit"
                      << " source_kind=" << static_cast<int>(local_report.source)
                      << " committed_bytes=" << local_report.committed_bytes
                      << " actual_local_typed_bytes=" << local_report.actual_local_typed_bytes
                      << " collective_handled=" << local_report.collective_handled
                      << " collective_skip_reason=" << local_report.collective_skip_reason
                      << " dominant_executor=" << local_report.dominant_executor;
            LOG(INFO) << "tc_profile materialize_mapped_into_target local_path timings"
                      << " artifact_id=" << request_hints.artifact_id << " target_device=" << target_device.ordinal
                      << " setup_sec=" << std::chrono::duration<double>(setup_done - source_total_start).count()
                      << " local_mapped_sec=" << local_mapped_sec << " continuation_sec=" << continuation_sec
                      << " total_sec="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - source_total_start).count()
                      << " committed_bytes=" << local_report.committed_bytes
                      << " local_mapped_bytes=" << local_mapped_bytes
                      << " generic_backend_bytes=" << generic_backend_bytes
                      << " effective_local_typed_bytes=" << effective_local_typed_bytes;
            return loading::MaterializeIntoTargetResult{
                .source = source_kind,
                .requested_bytes = local_report.requested_bytes,
                .committed_bytes = local_report.committed_bytes,
                .fallback_bytes = local_report.fallback_bytes,
                .residual_bytes = local_report.residual_bytes,
                .actual_collective_committed_bytes = local_report.actual_collective_committed_bytes,
                .actual_local_typed_bytes = local_report.actual_local_typed_bytes,
                .actual_generic_backend_bytes = local_report.actual_generic_backend_bytes,
                .collective_unique_source_bytes = local_report.collective_metrics.unique_source_bytes,
                .collective_peer_transfer_bytes = local_report.collective_metrics.peer_transfer_bytes,
                .collective_peak_temporary_bytes = local_report.collective_metrics.peak_temporary_bytes,
                .collective_batch_count = local_report.collective_metrics.batch_count,
                .collective_dedup_saving_bytes = local_report.collective_metrics.dedup_saving_bytes,
                .collective_skip_reason = local_report.collective_skip_reason,
                .collective_handled = local_report.collective_handled,
                .direct_write_supported = local_report.direct_write_supported,
                .source_ordered = local_report.source_ordered,
                .dominant_executor = local_report.dominant_executor,
                .selection_reason = local_report.selection_reason,
            };
          }
          return absl::FailedPreconditionError(
              absl::StrCat(
                  "materialize_mapped_into_target selected local mapped execution but admission returned "
                  "handled=false: ",
                  local_result.skip_reason.empty() ? "unavailable" : local_result.skip_reason));
        } else {
          return absl::FailedPreconditionError(
              absl::StrCat(
                  "materialize_mapped_into_target selected local mapped execution but disk shared context is "
                  "unavailable: ",
                  shared_or.status().message()));
        }
      } else {
        return absl::FailedPreconditionError(
            "materialize_mapped_into_target selected local mapped execution but disk loader is unavailable");
      }
    }

    loader::TargetLayoutGpuSink::Options sink_opts{
        .storages = std::move(storages),
        .chunk_size = config_.artifact_chunk_bytes,
        .device_id = target_device.ordinal,
    };
    loader::TargetLayoutGpuSink sink(std::move(sink_opts));

    const int concurrency = loading::resolve_materialization_concurrency(config_.num_threads, request_hints);
    if (source_ordered && effective_map.total_bytes > 0 && !effective_map.segments.empty()) {
      const uint64_t max_window_bytes = std::min<uint64_t>(request_hints.max_buffer_bytes, 64ULL * 1024 * 1024);
      const uint64_t window_cap_bytes = std::min<uint64_t>(max_window_bytes, slice_bytes);
      loader::SourceWindowScheduler::Options sched_opts{
          .merge_max_gap_bytes = config_.options->byte_mapping.disk_source_merge_max_gap_bytes,
          .merge_max_amplification = config_.options->byte_mapping.disk_source_merge_max_amplification,
          .prefetch_depth = config_.options->byte_mapping.disk_source_prefetch_depth,
          .window_cap_bytes = window_cap_bytes,
          .path = "materialize_mapped_into_target",
      };
      loader::SourceWindowScheduler scheduler(std::move(sched_opts));
      auto exec_status = scheduler.Execute(
          effective_map,
          sources,
          sink,
          config_.runtime_context->pinned_buffer_pool(),
          timeout,
          /*use_pinned_buffers=*/true);
      if (!exec_status.ok()) {
        LOG(ERROR) << "materialize_mapped_into_target source-ordered execution failed"
                   << " source_kind=" << static_cast<int>(source_kind)
                   << " source_byte_space=" << static_cast<int>(source_binding.source_byte_space)
                   << " map_total_bytes=" << effective_map.total_bytes
                   << " map_segments=" << effective_map.segments.size()
                   << " target_storages=" << target_layout.storages.size() << " status=" << exec_status;
        return absl::DataLossError(
            absl::StrCat("materialize_mapped_into_target source-ordered execution failed: ", exec_status.message()));
      }
    } else if (effective_map.total_bytes > 0 && !effective_map.segments.empty()) {
      const size_t num_chunks = resolve_streaming_buffer_chunks_for_transfer(
          effective_map.total_bytes, slice_bytes, config_.runtime_context->options().streaming_buffer_chunks);
      auto session_spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
          /*num_chunks=*/num_chunks, slice_bytes, config_.runtime_context->pinned_buffer_pool());
      auto init_spb_status = session_spb->initialize(timeout);
      if (!init_spb_status.ok()) {
        return init_spb_status;
      }
      loader::StreamingBufferAdapter adapter(session_spb);
      auto pump_status = loader::pump_ranges(
          *plan_source,
          sink,
          adapter,
          absl::MakeSpan(ranges),
          concurrency,
          config_.runtime_context->async_runtime()->blocking_executor(),
          nullptr,
          make_pump_direct_write_options(config_.options->materialization_strategy));
      if (!pump_status.ok()) {
        LOG(ERROR) << "materialize_mapped_into_target pump failed"
                   << " source_kind=" << static_cast<int>(source_kind)
                   << " source_byte_space=" << static_cast<int>(source_binding.source_byte_space)
                   << " map_total_bytes=" << effective_map.total_bytes
                   << " map_segments=" << effective_map.segments.size()
                   << " target_storages=" << target_layout.storages.size() << " status=" << pump_status;
        return absl::DataLossError(absl::StrCat("materialize_mapped_into_target pump failed: ", pump_status.message()));
      }
    }
    if (effective_local_typed_bytes > 0) {
      auto fill_status = execute_local_typed_work(sink);
      if (!fill_status.ok()) {
        return absl::DataLossError(
            absl::StrCat("materialize_mapped_into_target local typed execution failed: ", fill_status.message()));
      }
    }
    const auto exec_done = std::chrono::steady_clock::now();
    auto close_status = sink.close();
    const auto sink_close_done = std::chrono::steady_clock::now();
    if (!close_status.ok()) {
      LOG(ERROR) << "materialize_mapped_into_target sink close failed"
                 << " source_kind=" << static_cast<int>(source_kind)
                 << " source_byte_space=" << static_cast<int>(source_binding.source_byte_space)
                 << " map_total_bytes=" << effective_map.total_bytes
                 << " map_segments=" << effective_map.segments.size()
                 << " target_storages=" << target_layout.storages.size() << " status=" << close_status;
      return absl::DataLossError(
          absl::StrCat("materialize_mapped_into_target sink close failed: ", close_status.message()));
    }
    const strategy::ExecutionCommitReport commit_report{
        .source = source_binding.source,
        .requested_bytes = total_size,
        .committed_bytes = byte_range_map_covered_bytes(effective_data_map) + effective_local_typed_bytes,
        .fallback_bytes = byte_range_map_covered_bytes(effective_data_map),
        .residual_bytes = semantic_residual_bytes,
        .actual_collective_committed_bytes = 0,
        .actual_local_typed_bytes = effective_local_typed_bytes,
        .actual_generic_backend_bytes = byte_range_map_covered_bytes(effective_data_map),
        .collective_metrics = {},
        .collective_skip_reason = collective_skip_reason,
        .collective_handled = false,
        .direct_write_supported = source_binding.direct_write_capable,
        .source_ordered = source_ordered,
        .dominant_executor = source_ordered ? "SourceOrderedMappedTargetExecutor" : "MappedTargetStreamingExecutor",
        .selection_reason = !source_bound_lane_plan.selection_reason.empty() ? source_bound_lane_plan.selection_reason
            : source_ordered                                                 ? "source_ordered_mapped_target"
                                                                             : "mapped_target_streaming",
    };
    LOG(INFO) << "materialize_mapped_into_target source execution"
              << " source_kind=" << static_cast<int>(source_kind)
              << " source_byte_space=" << static_cast<int>(source_binding.source_byte_space)
              << " source_ordered=" << source_ordered << " use_source_layout=" << use_source_layout
              << " direct_write_supported=" << direct_write_supported
              << " map_total_bytes=" << effective_map.total_bytes << " map_segments=" << effective_map.segments.size()
              << " target_storages=" << target_layout.storages.size() << " concurrency=" << concurrency
              << " init_sec=" << std::chrono::duration<double>(init_done - source_total_start).count()
              << " open_sec=" << std::chrono::duration<double>(source_open_done - init_done).count()
              << " map_build_sec=" << map_build_sec
              << " exec_sec=" << std::chrono::duration<double>(exec_done - source_open_done).count()
              << " close_sec=" << std::chrono::duration<double>(sink_close_done - exec_done).count()
              << " total_sec=" << std::chrono::duration<double>(sink_close_done - source_total_start).count();
    LOG(INFO) << "materialize_mapped_into_target execution_commit"
              << " source_kind=" << static_cast<int>(commit_report.source)
              << " committed_bytes=" << commit_report.committed_bytes
              << " fallback_bytes=" << commit_report.fallback_bytes
              << " actual_local_typed_bytes=" << commit_report.actual_local_typed_bytes
              << " actual_generic_backend_bytes=" << commit_report.actual_generic_backend_bytes
              << " collective_handled=" << commit_report.collective_handled
              << " collective_skip_reason=" << commit_report.collective_skip_reason
              << " direct_write_supported=" << commit_report.direct_write_supported
              << " source_ordered=" << commit_report.source_ordered
              << " dominant_executor=" << commit_report.dominant_executor;
    return loading::MaterializeIntoTargetResult{
        .source = source_kind,
        .requested_bytes = commit_report.requested_bytes,
        .committed_bytes = commit_report.committed_bytes,
        .fallback_bytes = commit_report.fallback_bytes,
        .residual_bytes = commit_report.residual_bytes,
        .actual_collective_committed_bytes = commit_report.actual_collective_committed_bytes,
        .actual_local_typed_bytes = commit_report.actual_local_typed_bytes,
        .actual_generic_backend_bytes = commit_report.actual_generic_backend_bytes,
        .collective_unique_source_bytes = commit_report.collective_metrics.unique_source_bytes,
        .collective_peer_transfer_bytes = commit_report.collective_metrics.peer_transfer_bytes,
        .collective_peak_temporary_bytes = commit_report.collective_metrics.peak_temporary_bytes,
        .collective_batch_count = commit_report.collective_metrics.batch_count,
        .collective_dedup_saving_bytes = commit_report.collective_metrics.dedup_saving_bytes,
        .collective_skip_reason = commit_report.collective_skip_reason,
        .collective_handled = commit_report.collective_handled,
        .direct_write_supported = commit_report.direct_write_supported,
        .source_ordered = commit_report.source_ordered,
        .dominant_executor = commit_report.dominant_executor,
        .selection_reason = commit_report.selection_reason,
    };
  };

  const bool local_canonical_override = prefer_local_canonical_source_for_mapped(strategy_config);
  const bool try_local_canonical_source = !strategy_uses_collective_lane && !source_bound_plan_uses_local_mapped &&
      (!source_index_json.has_value() || local_canonical_override);
  if (try_local_canonical_source) {
    auto local_source_or = make_local_canonical_source(
        config_.replica_runtime->registry(), request_hints.artifact_id, target_device, canonical_total_size);
    if (local_source_or.ok()) {
      LOG(INFO) << "materialize_mapped_into_target selecting local canonical source"
                << " artifact_id=" << request_hints.artifact_id
                << " source_index_json_present=" << source_index_json.has_value()
                << " local_override=" << local_canonical_override;
      return run_source(
          std::make_unique<SharedSourceLoader>(*local_source_or),
          loading::MaterializationSource::kLocalReplica,
          strategy::SourceByteSpace::kCanonical);
    }
    if (!absl::IsNotFound(local_source_or.status())) {
      return local_source_or.status();
    }
  }

  auto gs_client = config_.runtime_context->global_store_client();
  auto comm_manager = config_.runtime_context->communication_manager();
  const bool gs_connected = gs_client && gs_client->is_connected();
  const bool comm_enabled = comm_manager && comm_manager->is_enabled();
  const auto retrieval_policy = request_hints.retrieval_policy();
  if (auto policy_status = loading::validate_retrieval_policy(retrieval_policy); !policy_status.ok()) {
    return policy_status;
  }
  const bool prefer_disk = retrieval_policy.preference == loading::SourcePreference::kPreferDisk;
  const bool prefer_p2p = retrieval_policy.preference == loading::SourcePreference::kPreferP2P;
  const bool allow_p2p = retrieval_policy.allow_p2p;
  const bool allow_disk = retrieval_policy.allow_disk;
  const bool has_disk_source = disk_source.has_value();
  if (source_bound_plan_uses_local_mapped && (!allow_disk || !has_disk_source)) {
    return absl::FailedPreconditionError(
        "materialize_mapped_into_target source_bound_strategy_plan requires local mapped typed execution but disk "
        "execution is unavailable");
  }
  if (strategy_uses_collective_lane && (!allow_disk || !has_disk_source)) {
    return absl::FailedPreconditionError(
        "materialize_mapped_into_target explicit source_bound_strategy_plan selected a collective lane but disk "
        "execution is unavailable");
  }
  components::WorkerIdentity local_identity = config_.runtime_context->worker_identity();
  if (!is_local_identity(local_identity)) {
    const auto& options = config_.runtime_context->options();
    if (!options.p2p_listen_host.empty()) {
      local_identity.node_address = options.p2p_listen_host;
    }
    local_identity.p2p_port = options.p2p_port;
  }

  if ((prefer_disk || source_bound_plan_uses_local_mapped) && has_disk_source && allow_disk) {
    loading::DiskSource disk_src = *disk_source;
    disk_src.require_descriptor = tensorcast::common::is_mi2_artifact_id(request_hints.artifact_id);
    auto disk_or = run_source(
        std::make_unique<DiskLoader>(disk_src),
        loading::MaterializationSource::kDisk,
        strategy::SourceByteSpace::kCanonical);
    if (disk_or.ok()) {
      return disk_or;
    }
    if (!gs_connected || !allow_p2p || source_bound_plan_uses_local_mapped) {
      return disk_or.status();
    }
  }

  if (!gs_connected && (!has_disk_source || !allow_disk)) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (allow_p2p && !comm_enabled && (!allow_disk || !has_disk_source || prefer_p2p)) {
    return absl::FailedPreconditionError("Communication not enabled");
  }

  std::optional<std::string> requested_view_id;
  if (request_hints.variant && request_hints.variant->view_id.has_value() && !request_hints.variant->view_id->empty()) {
    requested_view_id = *request_hints.variant->view_id;
  }
  const auto scheduling_group_hint = to_transport_scheduling_group_hint(request_hints);
  const std::string_view requester_worker_id = request_hints.transport_requester_worker_id.empty()
      ? std::string_view(local_identity.worker_id)
      : std::string_view(request_hints.transport_requester_worker_id);
  const std::string_view transport_request_id = request_hints.transport_request_id;

  if (!strategy_uses_collective_lane && allow_p2p && gs_connected && !request_hints.artifact_id.empty()) {
    bool used_canonical_transport_fallback = false;
    auto request_transport = [&]() -> absl::StatusOr<components::TransportSession> {
      const uint32_t wait_timeout_ms = resolve_transport_wait_timeout_ms(request_hints);
      const uint32_t view_probe_timeout_ms = resolve_view_transport_probe_timeout_ms(wait_timeout_ms);
      used_canonical_transport_fallback = false;
      if (requested_view_id.has_value()) {
        auto view_transport_or = gs_client->request_view_transport(
            request_hints.artifact_id,
            *requested_view_id,
            local_identity.node_id,
            local_identity.node_address,
            local_identity.p2p_port,
            target_device,
            view_probe_timeout_ms,
            scheduling_group_hint,
            requester_worker_id,
            transport_request_id);
        if (!view_transport_or.ok() &&
            (absl::IsNotFound(view_transport_or.status()) || absl::IsUnimplemented(view_transport_or.status()) ||
             absl::IsDeadlineExceeded(view_transport_or.status()))) {
          if (has_disk_source && allow_disk && !prefer_p2p) {
            LOG(INFO) << "request_view_transport unavailable for artifact_id=" << request_hints.artifact_id
                      << " view_id=" << *requested_view_id << " within probe_timeout_ms=" << view_probe_timeout_ms
                      << "; bypassing canonical transport route and falling back to disk";
            return absl::AbortedError(
                absl::StrCat(
                    "view transport unavailable for artifact_id=",
                    request_hints.artifact_id,
                    "; disk fallback available"));
          }
          LOG(INFO) << "request_view_transport unavailable for artifact_id=" << request_hints.artifact_id
                    << " view_id=" << *requested_view_id << " within probe_timeout_ms=" << view_probe_timeout_ms
                    << "; retrying canonical transport route with wait_timeout_ms=" << wait_timeout_ms;
          used_canonical_transport_fallback = true;
          return gs_client->request_replica_transport(
              request_hints.artifact_id,
              local_identity.node_id,
              local_identity.node_address,
              local_identity.p2p_port,
              target_device,
              wait_timeout_ms,
              scheduling_group_hint,
              requester_worker_id,
              transport_request_id);
        }
        return view_transport_or;
      }
      return gs_client->request_replica_transport(
          request_hints.artifact_id,
          local_identity.node_id,
          local_identity.node_address,
          local_identity.p2p_port,
          target_device,
          wait_timeout_ms,
          scheduling_group_hint,
          requester_worker_id,
          transport_request_id);
    };

    auto transport_or = request_transport();
    if (transport_or.ok()) {
      const auto& session = *transport_or;
      const auto& remote = session.remote_replica;
      P2PSource p2p_src;
      const bool source_is_view = requested_view_id.has_value() && !used_canonical_transport_fallback;
      p2p_src.comm_engine = gsl::not_null<std::shared_ptr<tensorcast::communicator::engine::Communicator>>{
          comm_manager->get_shared_engine()};
      p2p_src.size_bytes = remote.memory_size;
      p2p_src.ip = remote.node_address;
      p2p_src.port = static_cast<uint16_t>(remote.node_port);
      p2p_src.local_endpoint_id = components::derive_endpoint_id(local_identity, target_device);
      p2p_src.remote_endpoint_id = remote.endpoint_id;
      p2p_src.memory_keys = remote.remote_memory_keys;
      p2p_src.buf_sizes = remote.buffer_sizes;
      p2p_src.verification_json = remote.verification_json;
      p2p_src.enable_checksum = false;
      p2p_src.location.type = remote.memory_type;
      p2p_src.location.device_id = remote.device_id;
      fill_runtime_p2p_bindings(comm_manager, &p2p_src);
      p2p_src.request_budget = request_hints.request_budget;
      p2p_src.artifact_id = request_hints.artifact_id;
      p2p_src.transport_request_id = std::string(transport_request_id);
      if (has_disk_source && allow_disk && !prefer_p2p) {
        p2p_src.fallback_disk_dir = disk_source->path.string();
      }
      auto p2p_or = run_source(
          std::make_unique<P2PLoader>(p2p_src),
          loading::MaterializationSource::kP2P,
          source_is_view ? strategy::SourceByteSpace::kView : strategy::SourceByteSpace::kCanonical);
      auto complete_status = gs_client->complete_replica_transport(
          session.transport_id,
          p2p_or.ok() ? components::TransportCompletionOutcome::kSuccess
                      : components::TransportCompletionOutcome::kFailed,
          p2p_or.ok() ? std::string_view{} : std::string_view(p2p_or.status().ToString()));
      if (!complete_status.ok()) {
        LOG(WARNING) << "complete_replica_transport returned error: " << complete_status;
      }
      if (p2p_or.ok()) {
        return p2p_or;
      }
      if (!allow_disk || !has_disk_source || prefer_p2p) {
        return p2p_or.status();
      }
    } else if (!allow_disk || !has_disk_source) {
      return transport_or.status();
    }
  }

  if (allow_disk && has_disk_source) {
    loading::DiskSource disk_src = *disk_source;
    disk_src.require_descriptor = tensorcast::common::is_mi2_artifact_id(request_hints.artifact_id);
    return run_source(
        std::make_unique<DiskLoader>(disk_src),
        loading::MaterializationSource::kDisk,
        strategy::SourceByteSpace::kCanonical);
  }

  if (!allow_p2p && !allow_disk) {
    return absl::FailedPreconditionError("source_policy disallows P2P and disk for materialize_mapped_into_target");
  }

  return absl::FailedPreconditionError(
      "materialize_mapped_into_target requires disk source or Global Store connectivity");
}

absl::StatusOr<loading::MaterializeIntoTargetResult> MaterializationFacade::materialize_mapped_into_target(
    const DeviceKey& target_device,
    const strategy::PreparedSourceBoundExecutionPlan& prepared_execution,
    const loading::MaterializeHints& hints) {
  return materialize_mapped_into_target(target_device, prepared_execution, hints, std::nullopt);
}

absl::StatusOr<loading::MaterializeIntoTargetResult> MaterializationFacade::materialize_mapped_sources_into_target(
    const DeviceKey& target_device,
    const loading::IntoTargetLayout& target_layout,
    std::vector<std::shared_ptr<loader::SeekableSource>> sources,
    const loader::ByteRangeMap& mapping,
    const loading::MaterializeHints& hints,
    loading::MaterializationSource source_kind) {
  const absl::Time total_started_at = absl::Now();
  absl::Duration compile_elapsed = absl::ZeroDuration();
  absl::Duration mapped_source_create_elapsed = absl::ZeroDuration();
  absl::Duration sink_setup_elapsed = absl::ZeroDuration();
  absl::Duration streaming_buffer_init_elapsed = absl::ZeroDuration();
  absl::Duration pump_elapsed = absl::ZeroDuration();
  absl::Duration sink_close_elapsed = absl::ZeroDuration();
  bool streaming_buffer_used = false;
  if (target_device.type != DeviceType::GPU && target_device.type != DeviceType::CPU) {
    return absl::InvalidArgumentError("materialize_mapped_sources_into_target requires GPU or CPU target device");
  }
  if (hints.artifact_id.empty()) {
    return absl::InvalidArgumentError("materialize_mapped_sources_into_target requires hints.artifact_id");
  }
  if (target_layout.storages.empty()) {
    return absl::InvalidArgumentError("materialize_mapped_sources_into_target requires at least one target storage");
  }
  if (sources.size() < mapping.num_sources) {
    return absl::InvalidArgumentError("materialize_mapped_sources_into_target sources do not satisfy map.num_sources");
  }
  if (hints.variant && hints.variant->cached_plan.has_value() && !hints.variant->cached_plan->is_identity) {
    return absl::InvalidArgumentError("materialize_mapped_sources_into_target does not support view transforms");
  }

  uint64_t total_size = target_layout.total_size;
  uint64_t computed_total = 0;
  for (const auto& storage : target_layout.storages) {
    if (storage.length == 0) {
      return absl::InvalidArgumentError("materialize_mapped_sources_into_target requires non-empty storage length");
    }
    if (storage.length > std::numeric_limits<uint64_t>::max() - computed_total) {
      return absl::OutOfRangeError("materialize_mapped_sources_into_target storage length overflow");
    }
    computed_total += storage.length;
  }
  if (total_size == 0) {
    total_size = computed_total;
  } else if (total_size != computed_total) {
    return absl::InvalidArgumentError(
        "materialize_mapped_sources_into_target total_size does not match storage lengths");
  }
  if (total_size == 0) {
    return absl::InvalidArgumentError("materialize_mapped_sources_into_target requires total_size > 0");
  }
  if (mapping.total_bytes != total_size) {
    return absl::InvalidArgumentError("materialize_mapped_sources_into_target mapping total_bytes mismatch");
  }

  loader::ByteRangeCompiler compiler(config_.options->byte_mapping, "materialize_mapped_sources_into_target");
  const absl::Time compile_started_at = absl::Now();
  auto program_or = compiler.Compile(mapping);
  compile_elapsed = absl::Now() - compile_started_at;
  if (!program_or.ok()) {
    return program_or.status();
  }
  loader::ByteRangeMappedSource::Options map_opts{
      .path = "materialize_mapped_sources_into_target",
      .transport_request_id = std::string(hints.transport_request_id),
      .enable_direct_write_at = config_.options->byte_mapping.enable_direct_write_at,
  };
  const absl::Time mapped_source_create_started_at = absl::Now();
  auto mapped_or = loader::ByteRangeMappedSource::Create(mapping, *program_or, std::move(sources), std::move(map_opts));
  mapped_source_create_elapsed = absl::Now() - mapped_source_create_started_at;
  if (!mapped_or.ok()) {
    return mapped_or.status();
  }
  const bool direct_write_supported = (*mapped_or)->supports_direct_write_at();

  const size_t slice_bytes = config_.runtime_context->tx_slice_bytes();
  if (slice_bytes == 0 || config_.artifact_chunk_bytes == 0) {
    return absl::FailedPreconditionError("tx_slice_bytes or artifact_chunk_bytes is zero");
  }
  const std::chrono::milliseconds timeout =
      hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : config_.pinned_memory_timeout;

  std::vector<loader::TargetStorage> gpu_storages;
  std::vector<loader::HostTargetStorage> host_storages;
  gpu_storages.reserve(target_layout.storages.size());
  host_storages.reserve(target_layout.storages.size());
  std::vector<loader::Range> ranges;
  const bool collapse_ranges_for_single_source_direct_write = direct_write_supported && mapping.num_sources == 1;
  ranges.reserve(collapse_ranges_for_single_source_direct_write ? 1 : target_layout.storages.size());
  std::unique_ptr<loader::PositionedSink> sink;
  {
    const absl::Time sink_setup_started_at = absl::Now();
    uint64_t range_cursor = 0;
    for (const auto& storage : target_layout.storages) {
      if (storage.length > std::numeric_limits<size_t>::max()) {
        return absl::OutOfRangeError("materialize_mapped_sources_into_target storage length exceeds host limits");
      }
      if (target_device.type == DeviceType::GPU) {
        gpu_storages.push_back(loader::TargetStorage{storage.base_ptr, storage.length});
      } else {
        host_storages.push_back(
            loader::HostTargetStorage{
                .base_ptr = storage.base_ptr,
                .length = storage.length,
                .stable_backing = storage.stable_backing,
                .keepalive = storage.keepalive,
            });
      }
      range_cursor += storage.length;
    }
    if (range_cursor != total_size) {
      return absl::InvalidArgumentError("materialize_mapped_sources_into_target storage ranges do not span total_size");
    }
    if (collapse_ranges_for_single_source_direct_write) {
      ranges.emplace_back(/*offset=*/0, static_cast<size_t>(total_size));
    } else {
      uint64_t range_offset = 0;
      for (const auto& storage : target_layout.storages) {
        ranges.emplace_back(range_offset, static_cast<size_t>(storage.length));
        range_offset += storage.length;
      }
    }

    if (target_device.type == DeviceType::GPU) {
      loader::TargetLayoutGpuSink::Options sink_opts{
          .storages = std::move(gpu_storages),
          .chunk_size = config_.artifact_chunk_bytes,
          .device_id = target_device.ordinal,
      };
      sink = std::make_unique<loader::TargetLayoutGpuSink>(std::move(sink_opts));
    } else {
      loader::TargetLayoutHostSink::Options sink_opts{
          .storages = std::move(host_storages),
      };
      sink = std::make_unique<loader::TargetLayoutHostSink>(std::move(sink_opts));
    }
    sink_setup_elapsed = absl::Now() - sink_setup_started_at;
  }

  const int concurrency = loading::resolve_materialization_concurrency(config_.num_threads, hints);
  const bool use_direct_write_path = direct_write_supported && target_device.type == DeviceType::CPU &&
      dynamic_cast<loader::DirectWriteCapable*>(sink.get()) != nullptr;
  const size_t num_chunks = use_direct_write_path
      ? 0
      : resolve_streaming_buffer_chunks_for_transfer(
            total_size, slice_bytes, config_.runtime_context->options().streaming_buffer_chunks);
  loader::PumpDebugStats pump_debug_stats;
  const absl::Time pump_started_at = absl::Now();
  absl::Status pump_status;
  if (use_direct_write_path) {
    pump_status = loader::pump_ranges_direct_write(
        **mapped_or,
        *sink,
        absl::MakeSpan(ranges),
        slice_bytes,
        concurrency,
        &pump_debug_stats,
        make_pump_direct_write_options(config_.options->materialization_strategy));
  } else {
    streaming_buffer_used = true;
    auto session_spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
        /*num_chunks=*/num_chunks, slice_bytes, config_.runtime_context->pinned_buffer_pool());
    const absl::Time streaming_buffer_init_started_at = absl::Now();
    auto init_spb_status = session_spb->initialize(
        timeout,
        make_materialize_into_target_pinned_wait_context(hints, target_device.ordinal, num_chunks, slice_bytes));
    streaming_buffer_init_elapsed = absl::Now() - streaming_buffer_init_started_at;
    if (!init_spb_status.ok()) {
      return init_spb_status;
    }
    loader::StreamingBufferAdapter adapter(session_spb);
    pump_status = loader::pump_ranges(
        **mapped_or,
        *sink,
        adapter,
        absl::MakeSpan(ranges),
        concurrency,
        config_.runtime_context->async_runtime()->blocking_executor(),
        &pump_debug_stats,
        make_pump_direct_write_options(config_.options->materialization_strategy));
  }
  pump_elapsed = absl::Now() - pump_started_at;
  if (!pump_status.ok()) {
    return absl::DataLossError(
        absl::StrCat("materialize_mapped_sources_into_target pump failed: ", pump_status.message()));
  }
  const absl::Time sink_close_started_at = absl::Now();
  auto close_status = sink->close();
  sink_close_elapsed = absl::Now() - sink_close_started_at;
  if (!close_status.ok()) {
    return absl::DataLossError(
        absl::StrCat("materialize_mapped_sources_into_target sink close failed: ", close_status.message()));
  }
  if (target_device.type == DeviceType::CPU) {
    const absl::Duration total_elapsed = absl::Now() - total_started_at;
    if (!hints.transport_request_id.empty()) {
      LOG(INFO) << "materialize_mapped_sources_into_target.summary"
                << " request_id=" << (hints.transport_request_id.empty() ? "<unset>" : hints.transport_request_id)
                << " artifact_id=" << hints.artifact_id << " direct_write_supported=" << direct_write_supported
                << " streaming_buffer_used=" << (streaming_buffer_used ? "true" : "false")
                << " storage_count=" << target_layout.storages.size() << " mapping_segments=" << mapping.segments.size()
                << " total_bytes=" << total_size << " num_chunks=" << num_chunks << " slice_bytes=" << slice_bytes
                << " compile_ms=" << absl::ToDoubleMilliseconds(compile_elapsed)
                << " mapped_source_create_ms=" << absl::ToDoubleMilliseconds(mapped_source_create_elapsed)
                << " sink_setup_ms=" << absl::ToDoubleMilliseconds(sink_setup_elapsed)
                << " streaming_buffer_init_ms=" << absl::ToDoubleMilliseconds(streaming_buffer_init_elapsed)
                << " pump_ms=" << absl::ToDoubleMilliseconds(pump_elapsed)
                << " sink_close_ms=" << absl::ToDoubleMilliseconds(sink_close_elapsed)
                << " total_ms=" << absl::ToDoubleMilliseconds(total_elapsed);
    } else if (absl::ToDoubleMilliseconds(total_elapsed) >= 25.0) {
      VLOG(1) << "materialize_mapped_sources_into_target.summary"
              << " request_id=<unset>"
              << " artifact_id=" << hints.artifact_id << " direct_write_supported=" << direct_write_supported
              << " streaming_buffer_used=" << (streaming_buffer_used ? "true" : "false")
              << " storage_count=" << target_layout.storages.size() << " mapping_segments=" << mapping.segments.size()
              << " total_bytes=" << total_size << " num_chunks=" << num_chunks << " slice_bytes=" << slice_bytes
              << " compile_ms=" << absl::ToDoubleMilliseconds(compile_elapsed)
              << " mapped_source_create_ms=" << absl::ToDoubleMilliseconds(mapped_source_create_elapsed)
              << " sink_setup_ms=" << absl::ToDoubleMilliseconds(sink_setup_elapsed)
              << " streaming_buffer_init_ms=" << absl::ToDoubleMilliseconds(streaming_buffer_init_elapsed)
              << " pump_ms=" << absl::ToDoubleMilliseconds(pump_elapsed)
              << " sink_close_ms=" << absl::ToDoubleMilliseconds(sink_close_elapsed)
              << " total_ms=" << absl::ToDoubleMilliseconds(total_elapsed);
    } else {
      VLOG(2) << "materialize_mapped_sources_into_target.summary"
              << " artifact_id=" << hints.artifact_id << " direct_write_supported=" << direct_write_supported
              << " streaming_buffer_used=" << (streaming_buffer_used ? "true" : "false")
              << " storage_count=" << target_layout.storages.size() << " mapping_segments=" << mapping.segments.size()
              << " total_bytes=" << total_size << " num_chunks=" << num_chunks << " slice_bytes=" << slice_bytes
              << " compile_ms=" << absl::ToDoubleMilliseconds(compile_elapsed)
              << " mapped_source_create_ms=" << absl::ToDoubleMilliseconds(mapped_source_create_elapsed)
              << " sink_setup_ms=" << absl::ToDoubleMilliseconds(sink_setup_elapsed)
              << " streaming_buffer_init_ms=" << absl::ToDoubleMilliseconds(streaming_buffer_init_elapsed)
              << " pump_ms=" << absl::ToDoubleMilliseconds(pump_elapsed)
              << " sink_close_ms=" << absl::ToDoubleMilliseconds(sink_close_elapsed)
              << " total_ms=" << absl::ToDoubleMilliseconds(total_elapsed);
    }
  }
  return loading::MaterializeIntoTargetResult{
      .source = source_kind,
      .requested_bytes = total_size,
      .committed_bytes = total_size,
      .fallback_bytes = total_size,
      .residual_bytes = 0,
      .collective_handled = false,
      .direct_write_supported = direct_write_supported,
      .source_ordered = false,
      .dominant_executor = "MappedSourcesTargetExecutor",
      .selection_reason = "mapped_sources_target",
      .debug_stats =
          loading::MaterializeIntoTargetResult::DebugStats{
              .produced_chunks = pump_debug_stats.produced_chunks,
              .produced_bytes = pump_debug_stats.produced_bytes,
              .source_read_at_us_total = pump_debug_stats.source_read_at_us_total,
              .gpu_write_wait_us_total = pump_debug_stats.gpu_write_wait_us_total,
              .gpu_write_bytes_total = pump_debug_stats.gpu_write_bytes_total,
          },
  };
}

absl::StatusOr<loading::MaterializeIntoTargetResult> MaterializationFacade::materialize_mapped_loader_into_target(
    const DeviceKey& target_device,
    const loading::IntoTargetLayout& target_layout,
    std::unique_ptr<IArtifactLoader> loader,
    const loader::ByteRangeMap& mapping,
    const loading::MaterializeHints& hints,
    loading::MaterializationSource source_kind) {
  // Public compatibility wrapper: callers still provide one loader even
  // though internal execution can now target multiple resolved sources.
  if (target_device.type != DeviceType::GPU && target_device.type != DeviceType::CPU) {
    return absl::InvalidArgumentError("materialize_mapped_loader_into_target requires GPU or CPU target device");
  }
  if (loader == nullptr) {
    return absl::InvalidArgumentError("materialize_mapped_loader_into_target requires a source loader");
  }
  if (hints.artifact_id.empty()) {
    return absl::InvalidArgumentError("materialize_mapped_loader_into_target requires hints.artifact_id");
  }
  if (target_layout.storages.empty()) {
    return absl::InvalidArgumentError("materialize_mapped_loader_into_target requires at least one target storage");
  }
  if (mapping.num_sources != 1) {
    return absl::InvalidArgumentError("materialize_mapped_loader_into_target requires mapping.num_sources == 1");
  }
  if (hints.variant && hints.variant->cached_plan.has_value() && !hints.variant->cached_plan->is_identity) {
    return absl::InvalidArgumentError("materialize_mapped_loader_into_target does not support view transforms");
  }

  uint64_t total_size = target_layout.total_size;
  uint64_t computed_total = 0;
  for (const auto& storage : target_layout.storages) {
    if (storage.length == 0) {
      return absl::InvalidArgumentError("materialize_mapped_loader_into_target requires non-empty storage length");
    }
    if (storage.length > std::numeric_limits<uint64_t>::max() - computed_total) {
      return absl::OutOfRangeError("materialize_mapped_loader_into_target storage length overflow");
    }
    computed_total += storage.length;
  }
  if (total_size == 0) {
    total_size = computed_total;
  } else if (total_size != computed_total) {
    return absl::InvalidArgumentError(
        "materialize_mapped_loader_into_target total_size does not match storage lengths");
  }
  if (total_size == 0) {
    return absl::InvalidArgumentError("materialize_mapped_loader_into_target requires total_size > 0");
  }
  if (mapping.total_bytes != total_size) {
    return absl::InvalidArgumentError("materialize_mapped_loader_into_target mapping total_bytes mismatch");
  }

  auto init_status = loader->initialize();
  if (!init_status.ok()) {
    return init_status;
  }
  auto source_size_or = loader->get_artifact_size();
  if (!source_size_or.ok()) {
    return source_size_or.status();
  }
  if (*source_size_or < total_size) {
    return absl::FailedPreconditionError("materialize_mapped_loader_into_target source is smaller than target mapping");
  }
  auto source_or = loader->open_source();
  if (!source_or.ok()) {
    return source_or.status();
  }

  std::vector<std::shared_ptr<loader::SeekableSource>> sources;
  sources.emplace_back(std::move(*source_or));
  return materialize_mapped_sources_into_target(
      target_device, target_layout, std::move(sources), mapping, hints, source_kind);
}

absl::StatusOr<MaterializationFacade::IngestMappedSourcesIntoReplicasResult> MaterializationFacade::
    ingest_mapped_sources_into_replicas(
        std::vector<MappedReplicaTarget> targets,
        std::vector<std::shared_ptr<loader::SeekableSource>> sources,
        const loader::ByteRangeMap& mapping,
        const loading::MaterializeHints& hints,
        loading::MaterializationSource source_kind) {
  if (targets.empty()) {
    return absl::InvalidArgumentError("ingest_mapped_sources_into_replicas requires at least one target");
  }
  if (sources.empty()) {
    return absl::InvalidArgumentError("ingest_mapped_sources_into_replicas requires at least one source");
  }
  if (sources.size() < mapping.num_sources) {
    return absl::InvalidArgumentError("ingest_mapped_sources_into_replicas sources do not satisfy map.num_sources");
  }

  const DeviceKey target_device = targets.front().target_device;
  const common::memory::MemoryLocation target_location = targets.front().target.location.type;
  if (target_location != common::memory::MemoryLocation::CPU &&
      target_location != common::memory::MemoryLocation::GPU) {
    return absl::InvalidArgumentError("ingest_mapped_sources_into_replicas requires CPU or GPU targets");
  }

  struct PreparedReplicaTarget {
    std::string logical_artifact_id;
    loading::ReplicaKey key;
    std::shared_ptr<replica::Replica> replica;
    common::memory::MemoryLocation target_location{common::memory::MemoryLocation::NONE};
    std::uint64_t size_bytes{0};
  };

  auto registry = &config_.replica_runtime->registry();
  std::vector<PreparedReplicaTarget> prepared;
  prepared.reserve(targets.size());
  loading::IntoTargetLayout target_layout;
  target_layout.storages.reserve(targets.size());
  std::uint64_t total_bytes = 0;

  for (const auto& target : targets) {
    if (target.logical_artifact_id.empty()) {
      return absl::InvalidArgumentError("ingest_mapped_sources_into_replicas requires logical_artifact_id");
    }
    if (target.physical_artifact_id.empty()) {
      return absl::InvalidArgumentError("ingest_mapped_sources_into_replicas requires physical_artifact_id");
    }
    if (target.size_bytes == 0) {
      return absl::InvalidArgumentError("ingest_mapped_sources_into_replicas requires non-empty target size");
    }
    if (target.target.location.type != target_location) {
      return absl::InvalidArgumentError("ingest_mapped_sources_into_replicas target locations must match");
    }
    const DeviceKey target_location_device = target.target.location.to_device_key();
    if (target.target_device.type != target_device.type || target.target_device.ordinal != target_device.ordinal ||
        target.target_device.uuid != target_device.uuid || target_location_device.type != target.target_device.type ||
        target_location_device.ordinal != target.target_device.ordinal ||
        target_location_device.uuid != target.target_device.uuid) {
      return absl::InvalidArgumentError("ingest_mapped_sources_into_replicas target device mismatch");
    }
    if (total_bytes > std::numeric_limits<std::uint64_t>::max() - target.size_bytes) {
      return absl::OutOfRangeError("ingest_mapped_sources_into_replicas target size overflow");
    }

    loading::ReplicaKey key{
        .artifact_id = target.physical_artifact_id,
        .view_id = std::nullopt,
        .device = target.target_device,
        .replica = 0,
    };
    auto existing_or = registry->find(key);
    if (existing_or.ok()) {
      return absl::AlreadyExistsError("ingest_mapped_sources_into_replicas target replica already exists");
    }
    if (!absl::IsNotFound(existing_or.status())) {
      return existing_or.status();
    }

    loading::InlineBufferSource inline_source{.data = nullptr, .size_bytes = target.size_bytes};
    replica::ReplicaConfig cfg{
        .source = inline_source,
        .artifact_identifier = key.artifact_id,
        .device_type = target.target_device.type,
        .local_device_id = target.target_device.type == DeviceType::GPU ? target.target_device.ordinal : -1,
        .pinned_buffer_pool = config_.runtime_context->pinned_buffer_pool(),
        .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{config_.runtime_context->async_runtime()},
        .artifact_chunk_bytes = config_.artifact_chunk_bytes,
        .expected_artifact_size = target.size_bytes,
        .byte_mapping_config = config_.options->byte_mapping,
        .materialization_strategy = config_.options->materialization_strategy,
        .memory_tier_config = config_.options->memory_tier_config,
    };
    cfg.pinned_memory_timeout = hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : config_.pinned_memory_timeout;
    cfg.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);
    cfg.cpu_shared_memory_enabled = config_.runtime_context->options().cpu_shared_memory_enabled;

    auto replica_or = replica::Replica::create(cfg);
    if (!replica_or.ok()) {
      return replica_or.status();
    }
    auto replica = std::shared_ptr<replica::Replica>(std::move(replica_or.value()));
    auto allocate_status = replica->get_memory_manager().allocate_memory(target_location);
    if (!allocate_status.ok()) {
      return allocate_status;
    }
    auto ptrs = replica->get_data_pointer(target_location);
    if (ptrs.empty() || ptrs.front() == nullptr) {
      return absl::FailedPreconditionError("ingest_mapped_sources_into_replicas target pointer unavailable");
    }

    target_layout.storages.push_back(
        loading::IntoTargetStorage{
            .base_ptr = gsl::not_null<void*>{ptrs.front()},
            .length = target.size_bytes,
            .keepalive = replica,
        });
    total_bytes += target.size_bytes;
    prepared.push_back(
        PreparedReplicaTarget{
            .logical_artifact_id = target.logical_artifact_id,
            .key = std::move(key),
            .replica = std::move(replica),
            .target_location = target_location,
            .size_bytes = target.size_bytes,
        });
  }

  if (mapping.total_bytes != total_bytes) {
    return absl::InvalidArgumentError("ingest_mapped_sources_into_replicas mapping total_bytes mismatch");
  }
  target_layout.total_size = total_bytes;

  auto materialize_or = materialize_mapped_sources_into_target(
      target_device, target_layout, std::move(sources), mapping, hints, source_kind);
  if (!materialize_or.ok()) {
    return materialize_or.status();
  }

  std::vector<loading::ReplicaKey> inserted_keys;
  inserted_keys.reserve(prepared.size());
  IngestMappedSourcesIntoReplicasResult result;
  result.materialize_result = std::move(*materialize_or);
  result.replica_handles.reserve(prepared.size());
  for (const auto& entry : prepared) {
    auto mark_status = entry.replica->mark_loaded(entry.target_location);
    if (!mark_status.ok()) {
      for (const auto& inserted_key : inserted_keys) {
        (void)registry->erase(inserted_key);
      }
      return mark_status;
    }
    entry.replica->set_ready_signal(entry.target_location, absl::OkStatus());
    auto emplace_status = registry->emplace(entry.key, gsl::not_null{entry.replica});
    if (!emplace_status.ok()) {
      for (const auto& inserted_key : inserted_keys) {
        (void)registry->erase(inserted_key);
      }
      return emplace_status;
    }
    inserted_keys.push_back(entry.key);
    loading::ReplicaHandle handle = build_local_replica_handle(entry.key, entry.replica, entry.target_location);
    handle.source = source_kind;
    result.replica_handles.push_back(std::move(handle));
  }
  return result;
}

absl::StatusOr<ArtifactLoweringResult> MaterializationFacade::execute_artifact_lowering_plan(
    ArtifactLoweringPlan plan) {
  auto validation_status = validate_artifact_lowering_plan(plan);
  if (!validation_status.ok()) {
    return validation_status;
  }

  const auto build_resolved_source_descriptor = [&](const ArtifactLoweringPlan& lowering_plan) {
    if (lowering_plan.resolved_source_descriptor.has_value()) {
      return *lowering_plan.resolved_source_descriptor;
    }
    ResolvedSourceDescriptor descriptor;
    if (!lowering_plan.identity.physical_artifact_id.empty()) {
      descriptor.source_id = lowering_plan.identity.physical_artifact_id;
    } else if (!lowering_plan.identity.request_id.empty()) {
      descriptor.source_id = lowering_plan.identity.request_id;
    } else {
      descriptor.source_id = lowering_plan.identity.logical_artifact_id;
    }
    auto source_size_or = compute_logical_total_size(lowering_plan.canonical_index_json);
    if (!source_size_or.ok()) {
      return ResolvedSourceDescriptor{
          .source_id = descriptor.source_id,
          .exact_size_bytes = 0,
          .size_is_authoritative = false,
          .resolved_locally = lowering_plan.source_kind != loading::MaterializationSource::kP2P,
          .resolved_remotely = lowering_plan.source_kind == loading::MaterializationSource::kP2P,
          .already_verified = false,
          .source_kind = lowering_plan.source_kind,
      };
    }
    descriptor.exact_size_bytes = *source_size_or;
    descriptor.size_is_authoritative = true;
    descriptor.resolved_locally = lowering_plan.source_kind != loading::MaterializationSource::kP2P;
    descriptor.resolved_remotely = lowering_plan.source_kind == loading::MaterializationSource::kP2P;
    descriptor.already_verified = false;
    descriptor.source_kind = lowering_plan.source_kind;
    return descriptor;
  };

  if (plan.into_target.has_value()) {
    auto result_or = materialize_mapped_loader_into_target(
        plan.target_device,
        *plan.into_target,
        std::move(plan.source_loader),
        plan.byte_range_map,
        plan.hints,
        plan.source_kind);
    if (!result_or.ok()) {
      return result_or.status();
    }
    ArtifactLoweringResult result;
    result.into_target_result = std::move(*result_or);
    result.selection_identity = plan.selection_identity;
    result.resolved_source_descriptor = build_resolved_source_descriptor(plan);
    return result;
  }

  auto replica_handle_or = ingest_mapped_loader_into_replica(
      plan.identity.logical_artifact_id,
      plan.identity.physical_artifact_id,
      plan.target_device,
      *plan.replica_target,
      std::move(plan.source_loader),
      plan.byte_range_map,
      plan.hints,
      plan.source_kind);
  if (!replica_handle_or.ok()) {
    return replica_handle_or.status();
  }

  auto replica_or = config_.replica_runtime->registry().find(replica_handle_or->key());
  if (!replica_or.ok()) {
    return replica_or.status();
  }
  const auto verified_projection_or = build_verified_content_projection_from_replica(
      *replica_or, plan.replica_target->location.type, plan.canonical_index_json, plan.semantic_layout_identity);
  if (!verified_projection_or.ok()) {
    return verified_projection_or.status();
  }

  ArtifactLoweringResult result;
  result.replica_handle = std::move(*replica_handle_or);
  result.selection_identity = plan.selection_identity;
  result.resolved_source_descriptor = build_resolved_source_descriptor(plan);
  result.verified_content_descriptor = verified_projection_or->descriptor;
  result.verification_record = verified_projection_or->verification_record;
  result.backing_identity = BackingIdentity{
      .physical_artifact_id = plan.identity.physical_artifact_id,
      .replica_key = result.replica_handle->key(),
  };
  return result;
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::materialize_view_from_assembly(
    std::string_view assembly_id,
    std::string_view target_artifact_id,
    std::string_view view_id,
    std::string_view view_spec_json,
    const DeviceKey& target_device,
    loading::TransformPlacement placement,
    const std::vector<std::string>* allowed_view_ids) {
  if (assembly_id.empty() || target_artifact_id.empty()) {
    return absl::InvalidArgumentError("materialize_view_from_assembly requires assembly_id and target_artifact_id");
  }
  if (view_id.empty()) {
    return absl::InvalidArgumentError("materialize_view_from_assembly requires view_id");
  }
  if (target_device.type != DeviceType::GPU) {
    return absl::FailedPreconditionError("materialize_view_from_assembly requires GPU target device");
  }

  auto gs_client = config_.runtime_context->global_store_client();
  if (!gs_client || !gs_client->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  auto index_or = gs_client->get_artifact_index_by_id(assembly_id);
  if (!index_or.ok()) {
    return index_or.status();
  }
  std::string canonical_index_json = std::move(*index_or);

  auto canonical_total_or = compute_logical_total_size(canonical_index_json);
  if (!canonical_total_or.ok()) {
    return canonical_total_or.status();
  }
  const uint64_t canonical_total_size = *canonical_total_or;

  auto parsed_or = view::parse_view_selection_json(view_spec_json);
  if (!parsed_or.ok()) {
    return parsed_or.status();
  }
  auto plan_or = parsed_or->tensor_names.empty()
      ? loader::ViewPlanner::compute_view_plan(canonical_index_json, parsed_or->spec)
      : loader::ViewPlanner::compute_view_plan(canonical_index_json, parsed_or->spec, parsed_or->tensor_names);
  if (!plan_or.ok()) {
    return plan_or.status();
  }
  const loader::ViewPlan target_view_plan = *plan_or;
  std::vector<AssemblyTargetRange> target_ranges = build_target_ranges_from_view_plan(target_view_plan);
  const uint64_t target_total_size = target_view_plan.view_size_bytes;
  if (target_total_size == 0) {
    return absl::InvalidArgumentError("materialize_view_from_assembly view size must be > 0");
  }

  const auto target_id_kind = common::infer_artifact_id_kind(target_artifact_id);
  if (target_id_kind == common::ArtifactIdKind::kMi2) {
    auto mh_or = common::compute_index_multihash(std::optional<std::string>(canonical_index_json), "");
    if (!mh_or.ok()) {
      return mh_or.status();
    }
    auto parse_or = parse_mi2_multihashes(target_artifact_id);
    if (!parse_or.ok()) {
      return parse_or.status();
    }
    if (*mh_or != parse_or->first) {
      return absl::FailedPreconditionError("index multihash does not match target mi2 id");
    }
  }

  absl::flat_hash_set<absl::string_view> allowed_set;
  const absl::flat_hash_set<absl::string_view>* allowed_ptr = nullptr;
  if (allowed_view_ids != nullptr) {
    allowed_set.reserve(allowed_view_ids->size());
    for (const auto& id : *allowed_view_ids) {
      if (!id.empty()) {
        allowed_set.insert(id);
      }
    }
    allowed_ptr = &allowed_set;
  }

  auto plan = build_assembly_plan(
      *gs_client,
      assembly_id,
      canonical_index_json,
      canonical_total_size,
      absl::MakeSpan(target_ranges),
      target_total_size,
      target_device,
      config_.runtime_context->worker_identity(),
      allowed_ptr);
  if (!plan.ok()) {
    return plan.status();
  }
  AssemblyPlan assembly_plan = std::move(*plan);
  if (!assembly_plan.missing_ranges.empty()) {
    return absl::UnavailableError(
        absl::StrCat("assembly missing canonical ranges: ", format_missing_ranges(assembly_plan.missing_ranges)));
  }

  components::WorkerIdentity local_identity = config_.runtime_context->worker_identity();
  if (!is_local_identity(local_identity)) {
    const auto& options = config_.runtime_context->options();
    if (!options.p2p_listen_host.empty()) {
      local_identity.node_address = options.p2p_listen_host;
    }
    local_identity.p2p_port = options.p2p_port;
  }
  auto comm_manager = config_.runtime_context->communication_manager();
  const std::string local_endpoint_id = components::derive_endpoint_id(local_identity, target_device);
  auto& replica_registry = config_.replica_runtime->registry();

  std::vector<std::shared_ptr<loader::SeekableSource>> piece_sources;
  piece_sources.reserve(assembly_plan.sources.size());
  std::vector<std::shared_ptr<std::vector<std::uint8_t>>> canonicalized_sources;
  canonicalized_sources.reserve(assembly_plan.sources.size());
  for (auto& source : assembly_plan.sources) {
    auto source_or = resolve_assembly_piece_source(
        replica_registry,
        assembly_id,
        source,
        local_identity,
        target_device,
        local_endpoint_id,
        comm_manager,
        std::chrono::milliseconds(0),
        assembly_id);
    if (!source_or.ok()) {
      return source_or.status();
    }
    const auto& remote = source.session.remote_replica;
    std::shared_ptr<loader::SeekableSource> source_ptr = *source_or;

    if (source.inverse_transform.requires_materialization && !source.inverse_transform.tensors.empty()) {
      const uint64_t total_bytes = source.view_size_bytes > 0 ? source.view_size_bytes : remote.memory_size;
      if (total_bytes == 0) {
        return absl::FailedPreconditionError(absl::StrCat("view size missing for view_id=", source.view_id));
      }
      if (total_bytes > std::numeric_limits<size_t>::max()) {
        return absl::OutOfRangeError("view bytes exceed host memory limits");
      }
      std::vector<std::uint8_t> view_bytes(static_cast<size_t>(total_bytes));
      auto got_or = source_ptr->read_at(0, view_bytes.data(), static_cast<size_t>(total_bytes));
      if (!got_or.ok()) {
        return got_or.status();
      }
      if (*got_or != static_cast<size_t>(total_bytes)) {
        return absl::DataLossError(
            absl::StrCat("short read while canonicalizing transpose piece: got=", *got_or, " expected=", total_bytes));
      }
      auto canonicalized = std::make_shared<std::vector<std::uint8_t>>(static_cast<size_t>(total_bytes));

      loader::ViewWritePlan write_plan;
      write_plan.chunks.push_back(
          loader::ViewWritePlan::Chunk{
              .canonical_offset = 0,
              .view_offset = 0,
              .length = total_bytes,
              .segment_aligned = false,
          });

      loader::ViewIngestExecutor executor(
          std::move(write_plan),
          std::move(source.inverse_transform),
          loader::ViewIngestExecutor::IngestTarget::kCanonical);
      absl::Status ingest_status = executor.ingest_chunk(
          /*view_offset=*/0,
          absl::Span<const std::byte>(reinterpret_cast<const std::byte*>(view_bytes.data()), view_bytes.size()),
          common::memory::MemoryLocation::CPU,
          canonicalized->data(),
          /*device_id=*/-1);
      if (!ingest_status.ok()) {
        return ingest_status;
      }
      absl::Status finalize_status =
          executor.finalize(common::memory::MemoryLocation::CPU, canonicalized->data(), /*device_id=*/-1);
      if (!finalize_status.ok()) {
        return finalize_status;
      }

      canonicalized_sources.push_back(canonicalized);
      source_ptr =
          std::make_shared<loader::CpuMemorySource>(gsl::not_null<const void*>{canonicalized->data()}, total_bytes);
    }

    piece_sources.push_back(std::move(source_ptr));
  }

  loading::ReplicaKey key;
  key.artifact_id = std::string(target_artifact_id);
  key.view_id = std::string(view_id);
  key.device = target_device;
  key.replica = 0;
  const int concurrency = std::max(1, config_.num_threads);

  auto existing_or = replica_registry.find(key);
  if (existing_or.ok()) {
    const auto& existing = existing_or.value();
    absl::Status reuse_status = validate_existing_replica_for_reuse(existing, common::memory::MemoryLocation::GPU);
    if (reuse_status.ok()) {
      return build_local_replica_handle(key, existing, common::memory::MemoryLocation::GPU);
    }
    if (!absl::IsNotFound(reuse_status)) {
      return reuse_status;
    }
    absl::Status rebuild_status = load_assembled_ranges_into_replica(
        existing,
        assembly_plan.map,
        std::move(piece_sources),
        config_.options->byte_mapping,
        common::memory::MemoryLocation::GPU,
        concurrency,
        target_view_plan,
        placement,
        target_device.ordinal);
    if (!rebuild_status.ok()) {
      return rebuild_status;
    }
    return build_local_replica_handle(key, existing, common::memory::MemoryLocation::GPU);
  }
  if (!absl::IsNotFound(existing_or.status())) {
    return existing_or.status();
  }

  const uint64_t plan_total_bytes = assembly_plan.map.total_bytes;
  loading::InlineBufferSource inline_source{.data = nullptr, .size_bytes = plan_total_bytes};
  replica::ReplicaConfig cfg{
      .source = inline_source,
      .artifact_identifier = std::string(target_artifact_id),
      .device_type = DeviceType::GPU,
      .local_device_id = key.device.ordinal,
      .pinned_buffer_pool = config_.runtime_context->pinned_buffer_pool(),
      .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{config_.runtime_context->async_runtime()},
      .artifact_chunk_bytes = config_.artifact_chunk_bytes,
      .expected_artifact_size = plan_total_bytes,
      .view_plan = target_view_plan,
      .byte_mapping_config = config_.options->byte_mapping,
      .materialization_strategy = config_.options->materialization_strategy,
      .memory_tier_config = config_.options->memory_tier_config,
  };
  cfg.pinned_memory_timeout = config_.pinned_memory_timeout;
  cfg.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);
  cfg.view_id = std::string(view_id);
  cfg.transform_placement = placement;

  auto replica_or = replica::Replica::create(cfg);
  if (!replica_or.ok()) {
    return replica_or.status();
  }
  auto replica = std::shared_ptr<replica::Replica>(std::move(replica_or.value()));

  absl::Status load_status = load_assembled_ranges_into_replica(
      replica,
      assembly_plan.map,
      std::move(piece_sources),
      config_.options->byte_mapping,
      common::memory::MemoryLocation::GPU,
      concurrency,
      target_view_plan,
      cfg.transform_placement,
      target_device.ordinal);
  if (!load_status.ok()) {
    return load_status;
  }

  absl::Status emplace_status = replica_registry.emplace(key, gsl::not_null{replica});
  if (absl::IsAlreadyExists(emplace_status)) {
    auto existing = replica_registry.find(key);
    if (!existing.ok()) {
      return existing.status();
    }
    const auto& reuse = existing.value();
    absl::Status reuse_status = validate_existing_replica_for_reuse(reuse, common::memory::MemoryLocation::GPU);
    if (!reuse_status.ok()) {
      return reuse_status;
    }
    return build_local_replica_handle(key, reuse, common::memory::MemoryLocation::GPU);
  }
  if (!emplace_status.ok()) {
    return emplace_status;
  }

  loading::ReplicaHandle handle;
  handle.replica_key = key;
  handle.ready_signal = replica->ready_signal_for(common::memory::MemoryLocation::GPU);
  handle.cpu_state = replica->get_memory_state(common::memory::MemoryLocation::CPU);
  handle.gpu_state = replica->get_memory_state(common::memory::MemoryLocation::GPU);
  handle.source = loading::MaterializationSource::kP2P;
  const auto gpu_ptrs = replica->get_data_pointer(common::memory::MemoryLocation::GPU);
  handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;
  auto ipc_or = replica->get_memory_manager().get_ipc_handle();
  if (ipc_or.ok()) {
    handle.cuda_ipc_handle = cuda::IpcHandleBytes::from_native(*ipc_or);
  }
  if (!target_view_plan.is_identity) {
    handle.view_index_json = target_view_plan.view_index_json;
    const uint64_t view_size = target_view_plan.view_size_bytes;
    if (view_size > 0) {
      auto computer = config_.runtime_context->view_hash_computer();
      if (computer) {
        auto hash = computer->hash_replica_view(
            *replica, common::memory::MemoryLocation::GPU, view_size, std::optional<int>(target_device.ordinal));
        if (hash.has_value()) {
          handle.view_data_hash = std::move(hash);
        }
      }
    }
  }

  return handle;
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::assemble_from_pieces(
    const loading::MaterializationRequest& request) {
  auto gs_client = config_.runtime_context->global_store_client();
  if (!gs_client || !gs_client->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (request.canonical_artifact_id().empty()) {
    return absl::InvalidArgumentError("assemble_from_pieces requires artifact_id");
  }
  if (!request.target_is_gpu()) {
    return absl::FailedPreconditionError("assemble_from_pieces requires GPU target device");
  }

  std::string canonical_index_json;
  if (request.hints().variant && request.hints().variant->canonical_index_json.has_value()) {
    canonical_index_json = *request.hints().variant->canonical_index_json;
  } else {
    auto index_or = gs_client->get_artifact_index_by_id(request.canonical_artifact_id());
    if (!index_or.ok()) {
      return index_or.status();
    }
    canonical_index_json = std::move(*index_or);
  }

  auto canonical_total_or = compute_logical_total_size(canonical_index_json);
  if (!canonical_total_or.ok()) {
    return canonical_total_or.status();
  }
  const uint64_t canonical_total_size = *canonical_total_or;

  std::optional<loader::ViewPlan> target_view_plan;
  std::vector<AssemblyTargetRange> target_ranges;
  uint64_t target_total_size = 0;

  if (request.requested_view_id().has_value()) {
    const auto& view_id = *request.requested_view_id();
    if (view_id.empty()) {
      return absl::InvalidArgumentError("requested view_id must be non-empty for assembly");
    }
    if (request.hints().variant && request.hints().variant->cached_plan.has_value()) {
      target_view_plan = *request.hints().variant->cached_plan;
    } else {
      std::optional<loader::ViewSpec> view_spec;
      if (request.hints().variant && request.hints().variant->view_spec.has_value()) {
        view_spec = *request.hints().variant->view_spec;
      } else {
        auto meta_or = gs_client->get_view_metadata(request.canonical_artifact_id(), view_id);
        if (!meta_or.ok()) {
          return meta_or.status();
        }
        auto parsed_or = view::parse_view_selection_json(meta_or->view_spec_json);
        if (!parsed_or.ok()) {
          return parsed_or.status();
        }
        view_spec = std::move(parsed_or->spec);
        if (!parsed_or->tensor_names.empty()) {
          auto plan_or =
              loader::ViewPlanner::compute_view_plan(canonical_index_json, *view_spec, parsed_or->tensor_names);
          if (!plan_or.ok()) {
            return plan_or.status();
          }
          target_view_plan = std::move(*plan_or);
        }
      }
      if (target_view_plan.has_value()) {
        // Metadata-backed subset views already resolved the target plan.
      } else if (!view_spec.has_value()) {
        return absl::InvalidArgumentError("view_spec is required to assemble requested view_id");
      } else {
        auto plan_or = loader::ViewPlanner::compute_view_plan(canonical_index_json, *view_spec);
        if (!plan_or.ok()) {
          return plan_or.status();
        }
        target_view_plan = std::move(*plan_or);
      }
    }

    target_ranges = build_target_ranges_from_view_plan(*target_view_plan);
    target_total_size = target_view_plan->view_size_bytes;
  } else {
    auto ranges_or = build_target_ranges_for_canonical(canonical_index_json, canonical_total_size);
    if (!ranges_or.ok()) {
      return ranges_or.status();
    }
    target_ranges = std::move(*ranges_or);
    target_total_size = canonical_total_size;
  }

  if (target_total_size == 0) {
    return absl::InvalidArgumentError("assembly target size must be > 0");
  }

  auto plan_or = build_assembly_plan(
      *gs_client,
      request.canonical_artifact_id(),
      canonical_index_json,
      canonical_total_size,
      absl::MakeSpan(target_ranges),
      target_total_size,
      request.target_device(),
      config_.runtime_context->worker_identity(),
      /*allowed_view_ids=*/nullptr);
  if (!plan_or.ok()) {
    return plan_or.status();
  }
  AssemblyPlan plan = std::move(*plan_or);
  if (!plan.missing_ranges.empty()) {
    return absl::UnavailableError(
        absl::StrCat("assembly missing canonical ranges: ", format_missing_ranges(plan.missing_ranges)));
  }

  components::WorkerIdentity local_identity = config_.runtime_context->worker_identity();
  if (!is_local_identity(local_identity)) {
    const auto& options = config_.runtime_context->options();
    if (!options.p2p_listen_host.empty()) {
      local_identity.node_address = options.p2p_listen_host;
    }
    local_identity.p2p_port = options.p2p_port;
  }
  auto comm_manager = config_.runtime_context->communication_manager();
  const std::string local_endpoint_id = components::derive_endpoint_id(local_identity, request.target_device());
  auto& replica_registry = config_.replica_runtime->registry();
  std::vector<std::shared_ptr<loader::SeekableSource>> piece_sources;
  piece_sources.reserve(plan.sources.size());
  std::vector<std::shared_ptr<std::vector<std::uint8_t>>> canonicalized_sources;
  canonicalized_sources.reserve(plan.sources.size());
  for (auto& source : plan.sources) {
    auto source_or = resolve_assembly_piece_source(
        replica_registry,
        request.canonical_artifact_id(),
        source,
        local_identity,
        request.target_device(),
        local_endpoint_id,
        comm_manager,
        request.hints().request_budget,
        request.canonical_artifact_id());
    if (!source_or.ok()) {
      return source_or.status();
    }
    const auto& remote = source.session.remote_replica;
    std::shared_ptr<loader::SeekableSource> source_ptr = *source_or;

    if (source.inverse_transform.requires_materialization && !source.inverse_transform.tensors.empty()) {
      const uint64_t total_bytes = source.view_size_bytes > 0 ? source.view_size_bytes : remote.memory_size;
      if (total_bytes == 0) {
        return absl::FailedPreconditionError(absl::StrCat("view size missing for view_id=", source.view_id));
      }
      if (total_bytes > std::numeric_limits<size_t>::max()) {
        return absl::OutOfRangeError("view bytes exceed host memory limits");
      }
      std::vector<std::uint8_t> view_bytes(static_cast<size_t>(total_bytes));
      auto got_or = source_ptr->read_at(0, view_bytes.data(), static_cast<size_t>(total_bytes));
      if (!got_or.ok()) {
        return got_or.status();
      }
      if (*got_or != static_cast<size_t>(total_bytes)) {
        return absl::DataLossError(
            absl::StrCat("short read while canonicalizing transpose piece: got=", *got_or, " expected=", total_bytes));
      }
      auto canonicalized = std::make_shared<std::vector<std::uint8_t>>(static_cast<size_t>(total_bytes));

      loader::ViewWritePlan write_plan;
      write_plan.chunks.push_back(
          loader::ViewWritePlan::Chunk{
              .canonical_offset = 0,
              .view_offset = 0,
              .length = total_bytes,
              .segment_aligned = false,
          });

      loader::ViewIngestExecutor executor(
          std::move(write_plan),
          std::move(source.inverse_transform),
          loader::ViewIngestExecutor::IngestTarget::kCanonical);
      absl::Status ingest_status = executor.ingest_chunk(
          /*view_offset=*/0,
          absl::Span<const std::byte>(reinterpret_cast<const std::byte*>(view_bytes.data()), view_bytes.size()),
          common::memory::MemoryLocation::CPU,
          canonicalized->data(),
          /*device_id=*/-1);
      if (!ingest_status.ok()) {
        return ingest_status;
      }
      absl::Status finalize_status =
          executor.finalize(common::memory::MemoryLocation::CPU, canonicalized->data(), /*device_id=*/-1);
      if (!finalize_status.ok()) {
        return finalize_status;
      }

      canonicalized_sources.push_back(canonicalized);
      source_ptr =
          std::make_shared<loader::CpuMemorySource>(gsl::not_null<const void*>{canonicalized->data()}, total_bytes);
    }
    piece_sources.push_back(std::move(source_ptr));
  }

  loading::ReplicaKey key;
  key.artifact_id = request.canonical_artifact_id();
  key.view_id = request.requested_view_id();
  key.device = request.target_device();
  key.replica = 0;
  const loading::TransformPlacement transform_placement =
      request.hints().variant ? request.hints().variant->placement : loading::TransformPlacement::kServer;
  const int concurrency = loading::resolve_materialization_concurrency(config_.num_threads, request.hints());

  auto existing_or = replica_registry.find(key);
  if (existing_or.ok()) {
    const auto& existing = existing_or.value();
    absl::Status reuse_status = validate_existing_replica_for_reuse(existing, request.target_location());
    if (reuse_status.ok()) {
      return build_local_replica_handle(key, existing, request.target_location());
    }
    if (!absl::IsNotFound(reuse_status)) {
      return reuse_status;
    }
    absl::Status rebuild_status = load_assembled_ranges_into_replica(
        existing,
        plan.map,
        std::move(piece_sources),
        config_.options->byte_mapping,
        request.target_location(),
        concurrency,
        target_view_plan,
        transform_placement,
        request.target_device().ordinal);
    if (!rebuild_status.ok()) {
      return rebuild_status;
    }
    return build_local_replica_handle(key, existing, request.target_location());
  }
  if (!absl::IsNotFound(existing_or.status())) {
    return existing_or.status();
  }

  const uint64_t plan_total_bytes = plan.map.total_bytes;
  loading::InlineBufferSource inline_source{.data = nullptr, .size_bytes = plan_total_bytes};
  replica::ReplicaConfig cfg{
      .source = inline_source,
      .artifact_identifier = key.artifact_id,
      .device_type = DeviceType::GPU,
      .local_device_id = key.device.ordinal,
      .pinned_buffer_pool = config_.runtime_context->pinned_buffer_pool(),
      .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{config_.runtime_context->async_runtime()},
      .artifact_chunk_bytes = config_.artifact_chunk_bytes,
      .expected_artifact_size = plan_total_bytes,
      .view_plan = target_view_plan,
      .byte_mapping_config = config_.options->byte_mapping,
      .materialization_strategy = config_.options->materialization_strategy,
      .memory_tier_config = config_.options->memory_tier_config,
  };
  cfg.pinned_memory_timeout = config_.pinned_memory_timeout;
  cfg.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);
  cfg.view_id = request.requested_view_id();
  cfg.transform_placement = transform_placement;

  auto replica_or = replica::Replica::create(cfg);
  if (!replica_or.ok()) {
    return replica_or.status();
  }
  auto replica = std::shared_ptr<replica::Replica>(std::move(replica_or.value()));

  absl::Status load_status = load_assembled_ranges_into_replica(
      replica,
      plan.map,
      std::move(piece_sources),
      config_.options->byte_mapping,
      request.target_location(),
      concurrency,
      target_view_plan,
      cfg.transform_placement,
      request.target_device().ordinal);
  if (!load_status.ok()) {
    return load_status;
  }

  absl::Status emplace_status = replica_registry.emplace(key, gsl::not_null{replica});
  if (absl::IsAlreadyExists(emplace_status)) {
    auto existing_or = replica_registry.find(key);
    if (!existing_or.ok()) {
      return existing_or.status();
    }
    const auto& existing = existing_or.value();
    absl::Status reuse_status = validate_existing_replica_for_reuse(existing, request.target_location());
    if (!reuse_status.ok()) {
      return reuse_status;
    }
    return build_local_replica_handle(key, existing, request.target_location());
  }
  if (!emplace_status.ok()) {
    return emplace_status;
  }

  loading::ReplicaHandle handle = build_local_replica_handle(key, replica, request.target_location());
  handle.source = loading::MaterializationSource::kP2P;
  const auto& view_plan = replica->view_plan();
  if (view_plan.has_value() && !view_plan->is_identity) {
    handle.view_index_json = view_plan->view_index_json;
    const uint64_t view_size = view_plan->view_size_bytes;
    if (request.hints().need_view_data_hash && view_size > 0) {
      auto computer = config_.runtime_context->view_hash_computer();
      if (computer) {
        auto hash = computer->hash_replica_view(
            *replica,
            request.target_location(),
            view_size,
            request.target_is_gpu() ? std::optional<int>(request.target_device().ordinal) : std::nullopt);
        if (hash.has_value()) {
          handle.view_data_hash = std::move(hash);
        }
      }
    }
  }

  return handle;
}

absl::StatusOr<store::SealAssemblyResult> MaterializationFacade::seal_assembly(
    std::string_view assembly_id,
    bool publish_canonical,
    SealProgressCallback progress_cb,
    const std::vector<std::string>* allowed_view_ids) {
  if (assembly_id.empty()) {
    return absl::InvalidArgumentError("seal_assembly requires non-empty assembly_id");
  }

  auto gs_client = config_.runtime_context->global_store_client();
  if (!gs_client || !gs_client->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  store::SealAssemblyResult result;
  result.assembly_id = std::string(assembly_id);
  result.schema_version = "v3";
  result.encoding = "json";

  const auto id_kind = common::infer_artifact_id_kind(assembly_id);
  if (id_kind == common::ArtifactIdKind::kMi2) {
    result.sealed_artifact_id = std::string(assembly_id);
    result.already_sealed = true;
    auto parse_or = parse_mi2_multihashes(assembly_id);
    if (!parse_or.ok()) {
      return parse_or.status();
    }
    result.index_multihash = parse_or->first;
    result.data_multihash = parse_or->second;
  }

  std::string sealed_artifact_id;
  if (result.sealed_artifact_id.empty()) {
    auto binding_or = gs_client->get_artifact_binding(assembly_id);
    if (binding_or.ok()) {
      result.sealed_artifact_id = binding_or->to_artifact_id;
      result.already_sealed = true;
      auto parse_or = parse_mi2_multihashes(result.sealed_artifact_id);
      if (!parse_or.ok()) {
        return parse_or.status();
      }
      result.index_multihash = parse_or->first;
      result.data_multihash = parse_or->second;
    } else if (!absl::IsNotFound(binding_or.status())) {
      return binding_or.status();
    }
  }

  if (id_kind == common::ArtifactIdKind::kUnspecified) {
    return absl::InvalidArgumentError(R"(seal_assembly requires "cgid:" or "mi2:" artifact id)");
  }

  auto index_or = gs_client->get_artifact_index_by_id(assembly_id);
  if (!index_or.ok()) {
    return index_or.status();
  }
  std::string canonical_index_json = std::move(*index_or);

  auto canonical_total_or = compute_logical_total_size(canonical_index_json);
  if (!canonical_total_or.ok()) {
    return canonical_total_or.status();
  }
  const uint64_t canonical_total_size = *canonical_total_or;
  result.total_size = canonical_total_size;

  auto index_mh_or =
      common::compute_index_multihash(std::optional<std::string>(canonical_index_json), std::string_view());
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }
  if (!result.index_multihash.empty() && result.index_multihash != *index_mh_or) {
    return absl::FailedPreconditionError("index multihash does not match sealed artifact id");
  }
  result.index_multihash = *index_mh_or;

  auto target_device_or = select_seal_target_device(config_.runtime_context->device_manager());
  if (!target_device_or.ok()) {
    return target_device_or.status();
  }
  const DeviceKey target_device = *target_device_or;
  auto registry = &config_.replica_runtime->registry();

  auto target_ranges_or = build_target_ranges_for_canonical(canonical_index_json, canonical_total_size);
  if (!target_ranges_or.ok()) {
    return target_ranges_or.status();
  }
  std::vector<AssemblyTargetRange> target_ranges = std::move(*target_ranges_or);

  absl::flat_hash_set<absl::string_view> allowed_set;
  const absl::flat_hash_set<absl::string_view>* allowed_ptr = nullptr;
  if (allowed_view_ids != nullptr) {
    allowed_set.reserve(allowed_view_ids->size());
    for (const auto& view_id : *allowed_view_ids) {
      if (!view_id.empty()) {
        allowed_set.insert(view_id);
      }
    }
    allowed_ptr = &allowed_set;
  }

  auto plan_or = build_assembly_plan(
      *gs_client,
      assembly_id,
      canonical_index_json,
      canonical_total_size,
      absl::MakeSpan(target_ranges),
      canonical_total_size,
      target_device,
      config_.runtime_context->worker_identity(),
      allowed_ptr);
  bool use_canonical_fallback = false;
  std::shared_ptr<loader::SeekableSource> canonical_fallback_source;
  AssemblyPlan plan;
  if (!plan_or.ok()) {
    auto local_source_or = make_local_canonical_source(*registry, assembly_id, target_device, canonical_total_size);
    if (!local_source_or.ok() && absl::IsNotFound(local_source_or.status())) {
      loading::MaterializeHints canonical_hints;
      canonical_hints.artifact_id = std::string(assembly_id);
      auto canonical_handle_or = materialize_replica(
          DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
          loading::MaterializeMode::AUTO,
          canonical_hints);
      if (canonical_handle_or.ok()) {
        local_source_or = make_local_canonical_source_from_ready_handle(
            *registry, *canonical_handle_or, target_device, canonical_total_size);
      }
    }
    if (!local_source_or.ok()) {
      return plan_or.status();
    }
    use_canonical_fallback = true;
    canonical_fallback_source = *local_source_or;
  } else {
    plan = std::move(*plan_or);
  }
  if (!use_canonical_fallback && !plan.missing_ranges.empty()) {
    return absl::UnavailableError(
        absl::StrCat("seal_assembly missing canonical ranges: ", format_missing_ranges(plan.missing_ranges)));
  }

  auto comm_manager = config_.runtime_context->communication_manager();
  if (!comm_manager || !comm_manager->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }
  const components::WorkerIdentity local_identity = config_.runtime_context->worker_identity();
  const std::string local_endpoint_id = components::derive_endpoint_id(local_identity, target_device);

  std::vector<std::shared_ptr<loader::SeekableSource>> piece_sources;
  piece_sources.reserve(plan.sources.size());
  std::vector<std::shared_ptr<std::vector<std::uint8_t>>> canonicalized_sources;
  canonicalized_sources.reserve(plan.sources.size());
  for (auto& source : plan.sources) {
    auto source_or = resolve_assembly_piece_source(
        *registry,
        assembly_id,
        source,
        local_identity,
        target_device,
        local_endpoint_id,
        comm_manager,
        std::chrono::milliseconds(0),
        assembly_id);
    if (!source_or.ok()) {
      return source_or.status();
    }
    const auto& remote = source.session.remote_replica;
    std::shared_ptr<loader::SeekableSource> source_ptr = *source_or;
    if (source.inverse_transform.requires_materialization && !source.inverse_transform.tensors.empty()) {
      const uint64_t total_bytes = source.view_size_bytes > 0 ? source.view_size_bytes : remote.memory_size;
      if (total_bytes == 0) {
        return absl::FailedPreconditionError(absl::StrCat("view size missing for view_id=", source.view_id));
      }
      if (total_bytes > std::numeric_limits<size_t>::max()) {
        return absl::OutOfRangeError("view bytes exceed host memory limits");
      }
      std::vector<std::uint8_t> view_bytes(static_cast<size_t>(total_bytes));
      auto got_or = source_ptr->read_at(0, view_bytes.data(), static_cast<size_t>(total_bytes));
      if (!got_or.ok()) {
        return got_or.status();
      }
      if (*got_or != static_cast<size_t>(total_bytes)) {
        return absl::DataLossError(
            absl::StrCat("short read while canonicalizing transpose piece: got=", *got_or, " expected=", total_bytes));
      }
      auto canonicalized = std::make_shared<std::vector<std::uint8_t>>(static_cast<size_t>(total_bytes));

      loader::ViewWritePlan write_plan;
      write_plan.chunks.push_back(
          loader::ViewWritePlan::Chunk{
              .canonical_offset = 0,
              .view_offset = 0,
              .length = total_bytes,
              .segment_aligned = false,
          });

      loader::ViewIngestExecutor executor(
          std::move(write_plan),
          std::move(source.inverse_transform),
          loader::ViewIngestExecutor::IngestTarget::kCanonical);
      absl::Status ingest_status = executor.ingest_chunk(
          /*view_offset=*/0,
          absl::Span<const std::byte>(reinterpret_cast<const std::byte*>(view_bytes.data()), view_bytes.size()),
          common::memory::MemoryLocation::CPU,
          canonicalized->data(),
          /*device_id=*/-1);
      if (!ingest_status.ok()) {
        return ingest_status;
      }
      absl::Status finalize_status =
          executor.finalize(common::memory::MemoryLocation::CPU, canonicalized->data(), /*device_id=*/-1);
      if (!finalize_status.ok()) {
        return finalize_status;
      }

      canonicalized_sources.push_back(canonicalized);
      source_ptr =
          std::make_shared<loader::CpuMemorySource>(gsl::not_null<const void*>{canonicalized->data()}, total_bytes);
    }
    piece_sources.push_back(std::move(source_ptr));
  }

  const uint64_t plan_total_bytes = use_canonical_fallback ? canonical_total_size : plan.map.total_bytes;
  loader::ByteRangeCompiler compiler(config_.options->byte_mapping, "seal_assembly");
  auto program_or = compiler.Compile(plan.map);
  if (!use_canonical_fallback && !program_or.ok()) {
    return program_or.status();
  }

  const size_t leaf_chunk_bytes =
      config_.artifact_chunk_bytes == 0 ? static_cast<size_t>(4ULL * 1024 * 1024) : config_.artifact_chunk_bytes;
  if (result.sealed_artifact_id.empty()) {
    absl::StatusOr<std::string> data_mh_or = [&]() -> absl::StatusOr<std::string> {
      if (use_canonical_fallback) {
        return loader::compute_data_multihash_from_seekable_source(
            *canonical_fallback_source, canonical_total_size, leaf_chunk_bytes, std::move(progress_cb));
      }
      loader::ByteRangeMappedSource::Options map_opts{
          .path = "seal_assembly",
          .enable_direct_write_at = config_.options->byte_mapping.enable_direct_write_at,
      };
      auto hash_source_or =
          loader::ByteRangeMappedSource::Create(plan.map, *program_or, piece_sources, std::move(map_opts));
      if (!hash_source_or.ok()) {
        return hash_source_or.status();
      }
      return loader::compute_data_multihash_from_seekable_source(
          *hash_source_or.value(), plan_total_bytes, leaf_chunk_bytes, std::move(progress_cb));
    }();
    if (!data_mh_or.ok()) {
      return data_mh_or.status();
    }
    result.data_multihash = *data_mh_or;
    result.sealed_artifact_id = absl::StrCat("mi2:", result.index_multihash, ":", result.data_multihash);

    components::ArtifactBinding binding;
    binding.from_artifact_id = std::string(assembly_id);
    binding.to_artifact_id = result.sealed_artifact_id;
    binding.kind = tensorcast::global_store::v1::ARTIFACT_BINDING_KIND_SEAL;
    auto upsert_or = gs_client->upsert_artifact_binding(binding);
    if (!upsert_or.ok()) {
      return upsert_or.status();
    }
    if (!upsert_or->created) {
      result.already_sealed = true;
      result.sealed_artifact_id = upsert_or->binding.to_artifact_id;
      auto parse_or = parse_mi2_multihashes(result.sealed_artifact_id);
      if (!parse_or.ok()) {
        return parse_or.status();
      }
      result.index_multihash = parse_or->first;
      result.data_multihash = parse_or->second;
    }
  }

  result.verified_content_descriptor = build_sealed_artifact_verified_content_descriptor(
      result.index_multihash, result.data_multihash, result.total_size);
  result.verification_record = build_seal_verification_record(result.index_multihash);

  // Seal creates a new canonical mi2 identity. Publish the canonical index for
  // that sealed identity immediately so follow-on layout/proof flows can
  // resolve GetArtifactIndexById(sealed_artifact_id) without depending on a
  // later replica registration side effect.
  absl::Status metadata_status = publish_sealed_artifact_metadata(
      *gs_client,
      result.sealed_artifact_id,
      result.schema_version,
      result.encoding,
      result.total_size,
      canonical_index_json);
  if (!metadata_status.ok()) {
    return metadata_status;
  }

  if (!publish_canonical) {
    return result;
  }

  loading::ReplicaKey key;
  key.artifact_id = result.sealed_artifact_id;
  key.device = target_device;
  key.replica = 0;

  auto existing_or = registry->find(key);
  if (!existing_or.ok() && !absl::IsNotFound(existing_or.status())) {
    return existing_or.status();
  }
  auto build_seal_source = [&]() -> absl::StatusOr<std::unique_ptr<loader::SeekableSource>> {
    if (use_canonical_fallback) {
      return SharedSourceLoader(canonical_fallback_source).open_source();
    }
    loader::ByteRangeMappedSource::Options map_opts{
        .path = "seal_assembly",
        .enable_direct_write_at = config_.options->byte_mapping.enable_direct_write_at,
    };
    return loader::ByteRangeMappedSource::Create(plan.map, *program_or, piece_sources, std::move(map_opts));
  };
  if (!existing_or.ok()) {
    loading::InlineBufferSource inline_source{.data = nullptr, .size_bytes = plan_total_bytes};
    replica::ReplicaConfig cfg{
        .source = inline_source,
        .artifact_identifier = key.artifact_id,
        .device_type = DeviceType::GPU,
        .local_device_id = key.device.ordinal,
        .pinned_buffer_pool = config_.runtime_context->pinned_buffer_pool(),
        .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{config_.runtime_context->async_runtime()},
        .artifact_chunk_bytes = config_.artifact_chunk_bytes,
        .expected_artifact_size = plan_total_bytes,
        .byte_mapping_config = config_.options->byte_mapping,
        .materialization_strategy = config_.options->materialization_strategy,
        .memory_tier_config = config_.options->memory_tier_config,
    };
    cfg.pinned_memory_timeout = config_.pinned_memory_timeout;
    cfg.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);

    auto replica_or = replica::Replica::create(cfg);
    if (!replica_or.ok()) {
      return replica_or.status();
    }
    auto replica = std::shared_ptr<replica::Replica>(std::move(replica_or.value()));

    absl::StatusOr<std::unique_ptr<loader::SeekableSource>> source_or = build_seal_source();
    if (!source_or.ok()) {
      return source_or.status();
    }
    std::unique_ptr<loader::SeekableSource> source = std::move(*source_or);
    const int concurrency = std::max(1, config_.num_threads);
    auto load_future = replica->get_memory_manager().load_async_from_source(
        std::move(source),
        common::memory::MemoryLocation::GPU,
        concurrency,
        std::nullopt,
        std::function<absl::Status()>{});
    absl::Status load_status = std::move(load_future).get();
    if (!load_status.ok()) {
      return load_status;
    }
    replica->set_ready_signal(common::memory::MemoryLocation::GPU, absl::OkStatus());

    absl::Status emplace_status = registry->emplace(key, gsl::not_null{replica});
    if (!emplace_status.ok() && !absl::IsAlreadyExists(emplace_status)) {
      return emplace_status;
    }
  }

  loading::ReplicaKey cpu_key;
  cpu_key.artifact_id = result.sealed_artifact_id;
  cpu_key.device = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  cpu_key.replica = 0;
  auto cpu_existing_or = registry->find(cpu_key);
  if (!cpu_existing_or.ok() && !absl::IsNotFound(cpu_existing_or.status())) {
    return cpu_existing_or.status();
  }
  if (!cpu_existing_or.ok()) {
    loading::InlineBufferSource inline_source{.data = nullptr, .size_bytes = plan_total_bytes};
    replica::ReplicaConfig cpu_cfg{
        .source = inline_source,
        .artifact_identifier = cpu_key.artifact_id,
        .device_type = DeviceType::CPU,
        .local_device_id = -1,
        .pinned_buffer_pool = config_.runtime_context->pinned_buffer_pool(),
        .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{config_.runtime_context->async_runtime()},
        .artifact_chunk_bytes = config_.artifact_chunk_bytes,
        .expected_artifact_size = plan_total_bytes,
        .byte_mapping_config = config_.options->byte_mapping,
        .materialization_strategy = config_.options->materialization_strategy,
        .memory_tier_config = config_.options->memory_tier_config,
    };
    cpu_cfg.pinned_memory_timeout = config_.pinned_memory_timeout;
    cpu_cfg.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);

    auto cpu_replica_or = replica::Replica::create(cpu_cfg);
    if (!cpu_replica_or.ok()) {
      return cpu_replica_or.status();
    }
    auto cpu_replica = std::shared_ptr<replica::Replica>(std::move(cpu_replica_or.value()));
    auto cpu_source_or = build_seal_source();
    if (!cpu_source_or.ok()) {
      return cpu_source_or.status();
    }
    const int concurrency = std::max(1, config_.num_threads);
    auto cpu_load_future = cpu_replica->get_memory_manager().load_async_from_source(
        std::move(*cpu_source_or),
        common::memory::MemoryLocation::CPU,
        concurrency,
        std::nullopt,
        std::function<absl::Status()>{});
    absl::Status cpu_load_status = std::move(cpu_load_future).get();
    if (!cpu_load_status.ok()) {
      return cpu_load_status;
    }
    cpu_replica->set_ready_signal(common::memory::MemoryLocation::CPU, absl::OkStatus());

    absl::Status cpu_emplace_status = registry->emplace(cpu_key, gsl::not_null{cpu_replica});
    if (!cpu_emplace_status.ok() && !absl::IsAlreadyExists(cpu_emplace_status)) {
      return cpu_emplace_status;
    }
  }

  absl::Status publish_status = register_replica_with_global_store(key, {});
  if (!publish_status.ok()) {
    return publish_status;
  }

  return result;
}

absl::StatusOr<store::SealAssemblyResult> MaterializationFacade::seal_assembly_from_cut(
    std::string_view assembly_id,
    const SealAssemblyCutInput& cut_input,
    bool publish_canonical,
    SealProgressCallback progress_cb) {
  SC_TRACE_SCOPE("seal_from_cut.total");
  if (assembly_id.empty()) {
    return absl::InvalidArgumentError("seal_assembly_from_cut requires non-empty assembly_id");
  }
  if (cut_input.canonical_full && !cut_input.structural_views.empty()) {
    return absl::InvalidArgumentError("cut-driven seal does not allow structural_views with canonical_full");
  }
  if (!cut_input.canonical_full && cut_input.structural_views.empty()) {
    return absl::InvalidArgumentError("cut-driven seal requires structural_views or canonical_full");
  }

  auto gs_client = config_.runtime_context->global_store_client();
  if (!gs_client || !gs_client->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  store::SealAssemblyResult result;
  result.assembly_id = std::string(assembly_id);
  result.schema_version = "v3";
  result.encoding = "json";

  const std::string source_artifact_id =
      !cut_input.canonical_artifact_id.empty() ? cut_input.canonical_artifact_id : std::string(assembly_id);
  const auto id_kind = common::infer_artifact_id_kind(source_artifact_id);
  if (id_kind == common::ArtifactIdKind::kMi2) {
    result.sealed_artifact_id = source_artifact_id;
    result.already_sealed = true;
    auto parse_or = parse_mi2_multihashes(source_artifact_id);
    if (!parse_or.ok()) {
      return parse_or.status();
    }
    result.index_multihash = parse_or->first;
    result.data_multihash = parse_or->second;
  }

  if (result.sealed_artifact_id.empty()) {
    absl::StatusOr<components::ArtifactBinding> binding_or = absl::UnknownError("uninitialized");
    {
      SC_TRACE_SCOPE("seal_from_cut.get_artifact_binding");
      binding_or = gs_client->get_artifact_binding(source_artifact_id);
    }
    if (binding_or.ok()) {
      result.sealed_artifact_id = binding_or->to_artifact_id;
      result.already_sealed = true;
      auto parse_or = parse_mi2_multihashes(result.sealed_artifact_id);
      if (!parse_or.ok()) {
        return parse_or.status();
      }
      result.index_multihash = parse_or->first;
      result.data_multihash = parse_or->second;
    } else if (!absl::IsNotFound(binding_or.status())) {
      return binding_or.status();
    }
  }

  if (id_kind == common::ArtifactIdKind::kUnspecified) {
    return absl::InvalidArgumentError(R"(seal_assembly_from_cut requires "cgid:" or "mi2:" artifact id)");
  }

  std::string canonical_index_json;
  if (!cut_input.canonical_index_json.empty()) {
    canonical_index_json = cut_input.canonical_index_json;
  } else {
    absl::StatusOr<std::string> index_or = absl::UnknownError("uninitialized");
    {
      SC_TRACE_SCOPE("seal_from_cut.get_artifact_index");
      index_or = gs_client->get_artifact_index_by_id(source_artifact_id);
    }
    if (!index_or.ok()) {
      return index_or.status();
    }
    canonical_index_json = std::move(*index_or);
  }

  auto canonical_total_or = compute_logical_total_size(canonical_index_json);
  if (!canonical_total_or.ok()) {
    return canonical_total_or.status();
  }
  const uint64_t canonical_total_size = *canonical_total_or;
  result.total_size = canonical_total_size;

  absl::StatusOr<std::string> index_mh_or = absl::UnknownError("uninitialized");
  {
    SC_TRACE_SCOPE("seal_from_cut.compute_index_multihash");
    index_mh_or = common::compute_index_multihash(std::optional<std::string>(canonical_index_json), std::string_view());
  }
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }
  if (!result.index_multihash.empty() && result.index_multihash != *index_mh_or) {
    return absl::FailedPreconditionError("index multihash does not match sealed artifact id");
  }
  result.index_multihash = *index_mh_or;

  auto target_device_or = select_seal_target_device(config_.runtime_context->device_manager());
  if (!target_device_or.ok()) {
    return target_device_or.status();
  }
  const DeviceKey target_device = *target_device_or;
  auto registry = &config_.replica_runtime->registry();

  std::vector<AssemblyTargetRange> target_ranges;
  if (!(cut_input.canonical_full && !publish_canonical)) {
    absl::StatusOr<std::vector<AssemblyTargetRange>> target_ranges_or = absl::UnknownError("uninitialized");
    {
      SC_TRACE_SCOPE("seal_from_cut.build_target_ranges");
      target_ranges_or = build_target_ranges_for_canonical(canonical_index_json, canonical_total_size);
    }
    if (!target_ranges_or.ok()) {
      return target_ranges_or.status();
    }
    target_ranges = std::move(*target_ranges_or);
  }

  AssemblyPlan plan;
  std::shared_ptr<loader::SeekableSource> canonical_source;
  if (cut_input.canonical_full) {
    if (!cut_input.bound_canonical_spans.empty()) {
      {
        SC_TRACE_SCOPE("seal_from_cut.resolve_bound_canonical_source");
        auto source_or = make_bound_canonical_source(cut_input, canonical_total_size);
        if (!source_or.ok()) {
          return source_or.status();
        }
        canonical_source = *source_or;
      }
    } else {
      absl::StatusOr<std::shared_ptr<loader::SeekableSource>> local_source_or = absl::UnknownError("uninitialized");
      {
        SC_TRACE_SCOPE("seal_from_cut.resolve_local_canonical_source");
        local_source_or =
            make_local_canonical_source(*registry, source_artifact_id, target_device, canonical_total_size);
      }
      if (!local_source_or.ok()) {
        if (absl::IsNotFound(local_source_or.status())) {
          return absl::FailedPreconditionError(
              absl::StrCat(
                  "canonical_full seal invariant violated: local canonical source not found for ", source_artifact_id));
        }
        return local_source_or.status();
      }
      canonical_source = *local_source_or;
    }
  } else {
    absl::StatusOr<AssemblyPlan> plan_or = absl::UnknownError("uninitialized");
    {
      SC_TRACE_SCOPE("seal_from_cut.build_plan_from_views");
      plan_or = build_assembly_plan_from_views(
          *gs_client,
          assembly_id,
          canonical_index_json,
          canonical_total_size,
          absl::MakeSpan(target_ranges),
          canonical_total_size,
          target_device,
          config_.runtime_context->worker_identity(),
          cut_input.structural_views);
    }
    if (!plan_or.ok()) {
      return plan_or.status();
    }
    plan = std::move(*plan_or);
    if (!plan.missing_ranges.empty()) {
      return absl::UnavailableError(
          absl::StrCat(
              "seal_assembly_from_cut missing canonical ranges: ", format_missing_ranges(plan.missing_ranges)));
    }
  }

  auto comm_manager = config_.runtime_context->communication_manager();
  if (!comm_manager || !comm_manager->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }
  const components::WorkerIdentity local_identity = config_.runtime_context->worker_identity();
  const std::string local_endpoint_id = components::derive_endpoint_id(local_identity, target_device);

  std::vector<std::shared_ptr<loader::SeekableSource>> piece_sources;
  std::vector<std::shared_ptr<std::vector<std::uint8_t>>> canonicalized_sources;
  if (!cut_input.canonical_full) {
    piece_sources.reserve(plan.sources.size());
    canonicalized_sources.reserve(plan.sources.size());
    for (auto& source : plan.sources) {
      absl::StatusOr<std::shared_ptr<loader::SeekableSource>> source_or = absl::UnknownError("uninitialized");
      {
        SC_TRACE_SCOPE("seal_from_cut.resolve_piece_source");
        source_or = resolve_assembly_piece_source(
            *registry,
            assembly_id,
            source,
            local_identity,
            target_device,
            local_endpoint_id,
            comm_manager,
            std::chrono::milliseconds(0),
            assembly_id);
      }
      if (!source_or.ok()) {
        return source_or.status();
      }
      const auto& remote = source.session.remote_replica;
      std::shared_ptr<loader::SeekableSource> source_ptr = *source_or;
      if (source.inverse_transform.requires_materialization && !source.inverse_transform.tensors.empty()) {
        SC_TRACE_SCOPE("seal_from_cut.canonicalize_piece");
        const uint64_t total_bytes = source.view_size_bytes > 0 ? source.view_size_bytes : remote.memory_size;
        if (total_bytes == 0) {
          return absl::FailedPreconditionError(absl::StrCat("view size missing for view_id=", source.view_id));
        }
        if (total_bytes > std::numeric_limits<size_t>::max()) {
          return absl::OutOfRangeError("view bytes exceed host memory limits");
        }
        std::vector<std::uint8_t> view_bytes(static_cast<size_t>(total_bytes));
        auto got_or = source_ptr->read_at(0, view_bytes.data(), static_cast<size_t>(total_bytes));
        if (!got_or.ok()) {
          return got_or.status();
        }
        if (*got_or != static_cast<size_t>(total_bytes)) {
          return absl::DataLossError(
              absl::StrCat(
                  "short read while canonicalizing transpose piece: got=", *got_or, " expected=", total_bytes));
        }
        auto canonicalized = std::make_shared<std::vector<std::uint8_t>>(static_cast<size_t>(total_bytes));

        loader::ViewWritePlan write_plan;
        write_plan.chunks.push_back(
            loader::ViewWritePlan::Chunk{
                .canonical_offset = 0,
                .view_offset = 0,
                .length = total_bytes,
                .segment_aligned = false,
            });

        loader::ViewIngestExecutor executor(
            std::move(write_plan),
            std::move(source.inverse_transform),
            loader::ViewIngestExecutor::IngestTarget::kCanonical);
        absl::Status ingest_status = executor.ingest_chunk(
            /*view_offset=*/0,
            absl::Span<const std::byte>(reinterpret_cast<const std::byte*>(view_bytes.data()), view_bytes.size()),
            common::memory::MemoryLocation::CPU,
            canonicalized->data(),
            /*device_id=*/-1);
        if (!ingest_status.ok()) {
          return ingest_status;
        }
        absl::Status finalize_status =
            executor.finalize(common::memory::MemoryLocation::CPU, canonicalized->data(), /*device_id=*/-1);
        if (!finalize_status.ok()) {
          return finalize_status;
        }

        canonicalized_sources.push_back(canonicalized);
        source_ptr =
            std::make_shared<loader::CpuMemorySource>(gsl::not_null<const void*>{canonicalized->data()}, total_bytes);
      }
      piece_sources.push_back(std::move(source_ptr));
    }
  }

  const uint64_t plan_total_bytes = cut_input.canonical_full ? canonical_total_size : plan.map.total_bytes;
  loader::ByteRangeCompiler compiler(config_.options->byte_mapping, "seal_assembly_from_cut");
  absl::StatusOr<std::shared_ptr<const loader::ByteRangeProgram>> program_or = absl::InvalidArgumentError("not used");
  if (!cut_input.canonical_full) {
    {
      SC_TRACE_SCOPE("seal_from_cut.compile_byte_range_program");
      program_or = compiler.Compile(plan.map);
    }
    if (!program_or.ok()) {
      return program_or.status();
    }
  }

  const size_t leaf_chunk_bytes =
      config_.artifact_chunk_bytes == 0 ? static_cast<size_t>(4ULL * 1024 * 1024) : config_.artifact_chunk_bytes;
  if (result.sealed_artifact_id.empty()) {
    absl::StatusOr<std::string> data_mh_or = absl::UnknownError("uninitialized");
    {
      SC_TRACE_SCOPE("seal_from_cut.compute_data_multihash");
      data_mh_or = [&]() -> absl::StatusOr<std::string> {
        if (cut_input.canonical_full) {
          auto contiguous_gpu_or = resolve_contiguous_bound_canonical_gpu_range(cut_input, canonical_total_size);
          if (!contiguous_gpu_or.ok()) {
            return contiguous_gpu_or.status();
          }
          if (contiguous_gpu_or->has_value()) {
            const auto& contiguous_gpu = **contiguous_gpu_or;
            const size_t gpu_leaf_chunk_bytes = std::min<size_t>(leaf_chunk_bytes, 64ULL * 1024ULL * 1024ULL);
            return loader::compute_data_multihash_from_gpu_memory(
                gsl::not_null<void*>{reinterpret_cast<void*>(contiguous_gpu.base_addr)},
                contiguous_gpu.total_bytes,
                contiguous_gpu.device_id,
                gpu_leaf_chunk_bytes);
          }
          return loader::compute_data_multihash_from_seekable_source(
              *canonical_source, canonical_total_size, leaf_chunk_bytes, std::move(progress_cb));
        }
        loader::ByteRangeMappedSource::Options map_opts{
            .path = "seal_assembly_from_cut",
            .enable_direct_write_at = config_.options->byte_mapping.enable_direct_write_at,
        };
        auto hash_source_or =
            loader::ByteRangeMappedSource::Create(plan.map, *program_or, piece_sources, std::move(map_opts));
        if (!hash_source_or.ok()) {
          return hash_source_or.status();
        }
        return loader::compute_data_multihash_from_seekable_source(
            *hash_source_or.value(), plan_total_bytes, leaf_chunk_bytes, std::move(progress_cb));
      }();
    }
    if (!data_mh_or.ok()) {
      return data_mh_or.status();
    }
    result.data_multihash = *data_mh_or;
    result.sealed_artifact_id = absl::StrCat("mi2:", result.index_multihash, ":", result.data_multihash);

    components::ArtifactBinding binding;
    binding.from_artifact_id = std::string(assembly_id);
    binding.to_artifact_id = result.sealed_artifact_id;
    binding.kind = tensorcast::global_store::v1::ARTIFACT_BINDING_KIND_SEAL;
    absl::StatusOr<components::ArtifactBindingResult> upsert_or = absl::UnknownError("uninitialized");
    {
      SC_TRACE_SCOPE("seal_from_cut.upsert_artifact_binding");
      upsert_or = gs_client->upsert_artifact_binding(binding);
    }
    if (!upsert_or.ok()) {
      return upsert_or.status();
    }
    if (!upsert_or->created) {
      result.already_sealed = true;
      result.sealed_artifact_id = upsert_or->binding.to_artifact_id;
      auto parse_or = parse_mi2_multihashes(result.sealed_artifact_id);
      if (!parse_or.ok()) {
        return parse_or.status();
      }
      result.index_multihash = parse_or->first;
      result.data_multihash = parse_or->second;
    }
  }

  result.verified_content_descriptor = build_sealed_artifact_verified_content_descriptor(
      result.index_multihash, result.data_multihash, result.total_size);
  result.verification_record = build_seal_verification_record(result.index_multihash);

  absl::Status metadata_status = publish_sealed_artifact_metadata(
      *gs_client,
      result.sealed_artifact_id,
      result.schema_version,
      result.encoding,
      result.total_size,
      canonical_index_json);
  if (!metadata_status.ok()) {
    return metadata_status;
  }

  if (!publish_canonical) {
    return result;
  }

  loading::ReplicaKey key;
  key.artifact_id = result.sealed_artifact_id;
  key.device = target_device;
  key.replica = 0;

  auto existing_or = registry->find(key);
  if (!existing_or.ok() && !absl::IsNotFound(existing_or.status())) {
    return existing_or.status();
  }
  auto build_seal_source = [&]() -> absl::StatusOr<std::unique_ptr<loader::SeekableSource>> {
    if (cut_input.canonical_full) {
      return SharedSourceLoader(canonical_source).open_source();
    }
    loader::ByteRangeMappedSource::Options map_opts{
        .path = "seal_assembly_from_cut",
        .enable_direct_write_at = config_.options->byte_mapping.enable_direct_write_at,
    };
    return loader::ByteRangeMappedSource::Create(plan.map, *program_or, piece_sources, std::move(map_opts));
  };
  if (!existing_or.ok()) {
    SC_TRACE_SCOPE("seal_from_cut.load_gpu_replica");
    loading::InlineBufferSource inline_source{.data = nullptr, .size_bytes = plan_total_bytes};
    replica::ReplicaConfig cfg{
        .source = inline_source,
        .artifact_identifier = key.artifact_id,
        .device_type = DeviceType::GPU,
        .local_device_id = key.device.ordinal,
        .pinned_buffer_pool = config_.runtime_context->pinned_buffer_pool(),
        .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{config_.runtime_context->async_runtime()},
        .artifact_chunk_bytes = config_.artifact_chunk_bytes,
        .expected_artifact_size = plan_total_bytes,
        .byte_mapping_config = config_.options->byte_mapping,
        .materialization_strategy = config_.options->materialization_strategy,
        .memory_tier_config = config_.options->memory_tier_config,
    };
    cfg.pinned_memory_timeout = config_.pinned_memory_timeout;
    cfg.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);

    auto replica_or = replica::Replica::create(cfg);
    if (!replica_or.ok()) {
      return replica_or.status();
    }
    auto replica = std::shared_ptr<replica::Replica>(std::move(replica_or.value()));

    absl::StatusOr<std::unique_ptr<loader::SeekableSource>> source_or = build_seal_source();
    if (!source_or.ok()) {
      return source_or.status();
    }
    std::unique_ptr<loader::SeekableSource> source = std::move(*source_or);
    const int concurrency = std::max(1, config_.num_threads);
    auto load_future = replica->get_memory_manager().load_async_from_source(
        std::move(source),
        common::memory::MemoryLocation::GPU,
        concurrency,
        std::nullopt,
        std::function<absl::Status()>{});
    absl::Status load_status = std::move(load_future).get();
    if (!load_status.ok()) {
      return load_status;
    }
    replica->set_ready_signal(common::memory::MemoryLocation::GPU, absl::OkStatus());

    absl::Status emplace_status = registry->emplace(key, gsl::not_null{replica});
    if (!emplace_status.ok() && !absl::IsAlreadyExists(emplace_status)) {
      return emplace_status;
    }
  }

  loading::ReplicaKey cpu_key;
  cpu_key.artifact_id = result.sealed_artifact_id;
  cpu_key.device = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  cpu_key.replica = 0;
  auto cpu_existing_or = registry->find(cpu_key);
  if (!cpu_existing_or.ok() && !absl::IsNotFound(cpu_existing_or.status())) {
    return cpu_existing_or.status();
  }
  if (!cpu_existing_or.ok()) {
    SC_TRACE_SCOPE("seal_from_cut.load_cpu_replica");
    loading::InlineBufferSource inline_source{.data = nullptr, .size_bytes = plan_total_bytes};
    replica::ReplicaConfig cpu_cfg{
        .source = inline_source,
        .artifact_identifier = cpu_key.artifact_id,
        .device_type = DeviceType::CPU,
        .local_device_id = -1,
        .pinned_buffer_pool = config_.runtime_context->pinned_buffer_pool(),
        .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{config_.runtime_context->async_runtime()},
        .artifact_chunk_bytes = config_.artifact_chunk_bytes,
        .expected_artifact_size = plan_total_bytes,
        .byte_mapping_config = config_.options->byte_mapping,
        .materialization_strategy = config_.options->materialization_strategy,
        .memory_tier_config = config_.options->memory_tier_config,
    };
    cpu_cfg.pinned_memory_timeout = config_.pinned_memory_timeout;
    cpu_cfg.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);

    auto cpu_replica_or = replica::Replica::create(cpu_cfg);
    if (!cpu_replica_or.ok()) {
      return cpu_replica_or.status();
    }
    auto cpu_replica = std::shared_ptr<replica::Replica>(std::move(cpu_replica_or.value()));
    auto cpu_source_or = build_seal_source();
    if (!cpu_source_or.ok()) {
      return cpu_source_or.status();
    }
    const int concurrency = std::max(1, config_.num_threads);
    auto cpu_load_future = cpu_replica->get_memory_manager().load_async_from_source(
        std::move(*cpu_source_or),
        common::memory::MemoryLocation::CPU,
        concurrency,
        std::nullopt,
        std::function<absl::Status()>{});
    absl::Status cpu_load_status = std::move(cpu_load_future).get();
    if (!cpu_load_status.ok()) {
      return cpu_load_status;
    }
    cpu_replica->set_ready_signal(common::memory::MemoryLocation::CPU, absl::OkStatus());

    absl::Status cpu_emplace_status = registry->emplace(cpu_key, gsl::not_null{cpu_replica});
    if (!cpu_emplace_status.ok() && !absl::IsAlreadyExists(cpu_emplace_status)) {
      return cpu_emplace_status;
    }
  }

  absl::Status publish_status = absl::UnknownError("uninitialized");
  {
    SC_TRACE_SCOPE("seal_from_cut.register_replica_with_global_store");
    publish_status = register_replica_with_global_store(key, {});
  }
  if (!publish_status.ok()) {
    return publish_status;
  }

  return result;
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::ingest_mapped_loader_into_replica(
    std::string_view logical_artifact_id,
    std::string_view physical_artifact_id,
    const DeviceKey& target_device,
    const loading::ReplicaTarget& target,
    std::unique_ptr<IArtifactLoader> loader,
    const loader::ByteRangeMap& mapping,
    const loading::MaterializeHints& hints,
    loading::MaterializationSource source_kind) {
  // Public compatibility wrapper: logical/physical ids plus the single-loader
  // contract remain part of the facade shape until composite sources become a
  // first-class request concept.
  if (logical_artifact_id.empty()) {
    return absl::InvalidArgumentError("ingest_mapped_loader_into_replica requires logical_artifact_id");
  }
  if (physical_artifact_id.empty()) {
    return absl::InvalidArgumentError("ingest_mapped_loader_into_replica requires physical_artifact_id");
  }
  if (loader == nullptr) {
    return absl::InvalidArgumentError("ingest_mapped_loader_into_replica requires source loader");
  }
  if (mapping.num_sources != 1) {
    return absl::InvalidArgumentError("ingest_mapped_loader_into_replica requires mapping.num_sources == 1");
  }

  const common::memory::MemoryLocation target_location = target.location.type;
  if (target_location != common::memory::MemoryLocation::CPU &&
      target_location != common::memory::MemoryLocation::GPU) {
    return absl::InvalidArgumentError("ingest_mapped_loader_into_replica requires a CPU or GPU target");
  }

  auto runner = [&](const std::string& request_id,
                    const std::string& publish_context_id,
                    IngestionResultEvent* event_out) -> absl::StatusOr<loading::ReplicaHandle> {
    const absl::Time started_at = absl::Now();

    auto init_status = loader->initialize();
    if (!init_status.ok()) {
      return init_status;
    }
    auto source_size_or = loader->get_artifact_size();
    if (!source_size_or.ok()) {
      return source_size_or.status();
    }
    auto required_bytes_or = compute_required_source_bytes_for_map(mapping);
    if (!required_bytes_or.ok()) {
      return required_bytes_or.status();
    }
    if (*source_size_or < *required_bytes_or) {
      return absl::FailedPreconditionError(
          "ingest_mapped_loader_into_replica source is smaller than required byte-range map");
    }
    auto source_or = loader->open_source();
    if (!source_or.ok()) {
      return source_or.status();
    }

    std::vector<std::shared_ptr<loader::SeekableSource>> sources;
    sources.emplace_back(std::move(*source_or));

    loader::ByteRangeCompiler compiler(config_.options->byte_mapping, "ingest_mapped_loader_into_replica");
    auto program_or = compiler.Compile(mapping);
    if (!program_or.ok()) {
      return program_or.status();
    }
    loader::ByteRangeMappedSource::Options map_opts{
        .path = "ingest_mapped_loader_into_replica",
        .enable_direct_write_at = config_.options->byte_mapping.enable_direct_write_at,
    };
    auto mapped_or =
        loader::ByteRangeMappedSource::Create(mapping, *program_or, std::move(sources), std::move(map_opts));
    if (!mapped_or.ok()) {
      return mapped_or.status();
    }

    auto registry = &config_.replica_runtime->registry();
    loading::ReplicaKey key{
        .artifact_id = std::string(physical_artifact_id),
        .view_id = std::nullopt,
        .device = target_device,
        .replica = 0,
    };

    auto existing_or = registry->find(key);
    if (existing_or.ok()) {
      auto reuse_status = validate_existing_replica_for_reuse(*existing_or, target_location);
      if (reuse_status.ok()) {
        loading::ReplicaHandle handle = build_local_replica_handle(key, *existing_or, target_location);
        handle.source = source_kind;
        if (event_out != nullptr) {
          event_out->request_id = request_id;
          event_out->artifact_id = std::string(logical_artifact_id);
          event_out->target_device = target_device;
          event_out->target_location = target_location;
          event_out->bytes_transferred = mapping.total_bytes;
          event_out->duration_seconds = absl::ToDoubleSeconds(absl::Now() - started_at);
          event_out->publish_context_id = publish_context_id;
          event_out->replica_key = handle.key();
        }
        return handle;
      }
    } else if (!absl::IsNotFound(existing_or.status())) {
      return existing_or.status();
    }

    loading::InlineBufferSource inline_source{.data = nullptr, .size_bytes = mapping.total_bytes};
    replica::ReplicaConfig cfg{
        .source = inline_source,
        .artifact_identifier = key.artifact_id,
        .device_type = target_device.type,
        .local_device_id = target_device.type == DeviceType::GPU ? target_device.ordinal : -1,
        .pinned_buffer_pool = config_.runtime_context->pinned_buffer_pool(),
        .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{config_.runtime_context->async_runtime()},
        .artifact_chunk_bytes = config_.artifact_chunk_bytes,
        .expected_artifact_size = mapping.total_bytes,
        .byte_mapping_config = config_.options->byte_mapping,
        .materialization_strategy = config_.options->materialization_strategy,
        .memory_tier_config = config_.options->memory_tier_config,
    };
    cfg.pinned_memory_timeout = hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : config_.pinned_memory_timeout;
    cfg.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);
    cfg.cpu_shared_memory_enabled = config_.runtime_context->options().cpu_shared_memory_enabled;

    auto replica_or = replica::Replica::create(cfg);
    if (!replica_or.ok()) {
      return replica_or.status();
    }
    auto replica = std::shared_ptr<replica::Replica>(std::move(replica_or.value()));
    const int concurrency = hints.pipeline_concurrency > 0 ? static_cast<int>(hints.pipeline_concurrency)
                                                           : std::max(1, config_.num_threads);
    auto load_future = replica->get_memory_manager().load_async_from_source(
        std::move(*mapped_or), target_location, concurrency, std::nullopt, std::function<absl::Status()>{});
    absl::Status load_status = std::move(load_future).get();
    if (!load_status.ok()) {
      return load_status;
    }
    replica->set_ready_signal(target_location, absl::OkStatus());

    absl::Status emplace_status = registry->emplace(key, gsl::not_null{replica});
    if (!emplace_status.ok() && !absl::IsAlreadyExists(emplace_status)) {
      return emplace_status;
    }
    if (absl::IsAlreadyExists(emplace_status)) {
      auto registered_or = registry->find(key);
      if (!registered_or.ok()) {
        return registered_or.status();
      }
      replica = *registered_or;
    }

    loading::ReplicaHandle handle = build_local_replica_handle(key, replica, target_location);
    handle.source = source_kind;
    if (event_out != nullptr) {
      event_out->request_id = request_id;
      event_out->artifact_id = std::string(logical_artifact_id);
      event_out->target_device = target_device;
      event_out->target_location = target_location;
      event_out->bytes_transferred = mapping.total_bytes;
      event_out->duration_seconds = absl::ToDoubleSeconds(absl::Now() - started_at);
      event_out->publish_context_id = publish_context_id;
      event_out->replica_key = handle.key();
    }
    return handle;
  };

  return run_pipeline_ingestion(
      IngestionSource::kMemory,
      std::string(logical_artifact_id),
      mapping,
      target,
      hints,
      /*publish_to_global_store=*/false,
      runner);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return run_disk_ingestion_internal(artifact_identifier, source, target, hints, /*publish_to_global_store=*/false);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  return run_disk_ingestion_internal(artifact_identifier, source, target, hints, publish_to_global_store);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return run_p2p_ingestion_internal(artifact_identifier, source, target, hints, /*publish_to_global_store=*/false);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  return run_p2p_ingestion_internal(artifact_identifier, source, target, hints, publish_to_global_store);
}

void MaterializationFacade::prepare_p2p_source(P2PSource* source) const {
  fill_runtime_p2p_bindings(config_.runtime_context->communication_manager(), source);
}

absl::Status MaterializationFacade::register_replica_with_global_store(
    const loading::ReplicaKey& key,
    std::string_view artifact_id_override,
    std::string_view publish_context_id) {
  std::string context = publish_context_id.empty() ? "" : std::string(publish_context_id);
  if (context.empty()) {
    auto stored_context = lookup_publish_context_for_replica(key);
    if (stored_context.has_value()) {
      context = *stored_context;
    } else {
      context = config_.runtime_context->mint_publish_context_id();
      record_publish_context_for_replica(key, context);
    }
  } else {
    record_publish_context_for_replica(key, context);
  }

  if (hooks_ && hooks_->register_replica_override) {
    return hooks_->register_replica_override(key, artifact_id_override, context);
  }
  return config_.metadata_gateway->register_replica(key, artifact_id_override, context);
}

template <typename SourceT, typename RunnerFn>
absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::run_pipeline_ingestion(
    IngestionSource source_type,
    const std::string& artifact_identifier,
    const SourceT& /*source*/,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store,
    RunnerFn&& runner) {
  const std::string request_id = [&]() {
    switch (source_type) {
      case IngestionSource::kDisk:
        return make_request_id("disk");
      case IngestionSource::kP2P:
        return make_request_id("p2p");
      case IngestionSource::kMemory:
        return make_request_id("memory");
    }
    return make_request_id("ingest");
  }();
  const std::string publish_context_id =
      publish_to_global_store ? config_.runtime_context->mint_publish_context_id() : std::string();
  const loading::MaterializeMode mode =
      source_type == IngestionSource::kDisk ? loading::MaterializeMode::LOAD_ONLY : loading::MaterializeMode::COPY_ONLY;
  IngestionRequestMetadata metadata{
      .request_id = request_id,
      .artifact_identifier = artifact_identifier,
      .source = source_type,
      .target = target,
      .publish_context_id = publish_context_id,
      .publish_to_global_store = publish_to_global_store,
      .materialize_mode = mode,
      .hints = hints,
  };
  maybe_invoke_before_pipeline_start(metadata);

  const IngestionStartedEvent started_event = make_started_event(
      request_id, artifact_identifier, source_type, target, publish_context_id, publish_to_global_store, mode, hints);
  publish_started_event(started_event);

  IngestionResultEvent defaults = make_ingestion_event_seed(
      request_id, artifact_identifier, source_type, target, publish_to_global_store, publish_context_id, mode, hints);

  if (auto override_result = maybe_override_result(); override_result.has_value()) {
    IngestionResultEvent event = defaults;
    if (!override_result->ok()) {
      event.status = override_result->status();
      maybe_mutate_completion_event(event);
      publish_completed_event(std::move(event));
      return override_result->status();
    }
    auto handle = std::move(override_result->value());
    event.replica_key = handle.key();
    maybe_mutate_completion_event(event);
    if (publish_to_global_store && !publish_context_id.empty()) {
      record_publish_context_for_replica(handle.key(), publish_context_id);
    }
    publish_completed_event(event);
    return handle;
  }

  IngestionResultEvent pipeline_event;
  auto handle_or = runner(request_id, publish_context_id, &pipeline_event);
  if (!handle_or.ok()) {
    IngestionResultEvent failure_event = pipeline_event.request_id.empty() ? defaults : pipeline_event;
    apply_event_defaults(failure_event, defaults);
    failure_event.status = handle_or.status();
    maybe_mutate_completion_event(failure_event);
    publish_completed_event(std::move(failure_event));
    return handle_or.status();
  }

  auto handle = std::move(handle_or.value());
  apply_event_defaults(pipeline_event, defaults);
  if (!pipeline_event.replica_key.has_value()) {
    pipeline_event.replica_key = handle.key();
  }
  maybe_mutate_completion_event(pipeline_event);
  if (publish_to_global_store && !pipeline_event.publish_context_id.empty()) {
    record_publish_context_for_replica(handle.key(), pipeline_event.publish_context_id);
  }
  publish_completed_event(pipeline_event);
  return handle;
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::run_disk_ingestion_internal(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  auto runner = [&](const std::string& request_id,
                    const std::string& publish_context_id,
                    IngestionResultEvent* event_out) {
    return pipeline_->ingest_from_disk(
        artifact_identifier, source, target, hints, publish_to_global_store, event_out, request_id, publish_context_id);
  };

  return run_pipeline_ingestion(
      IngestionSource::kDisk, artifact_identifier, source, target, hints, publish_to_global_store, runner);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::run_p2p_ingestion_internal(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  auto runner = [&](const std::string& request_id,
                    const std::string& publish_context_id,
                    IngestionResultEvent* event_out) {
    return pipeline_->ingest_from_p2p(
        artifact_identifier, source, target, hints, publish_to_global_store, event_out, request_id, publish_context_id);
  };

  return run_pipeline_ingestion(
      IngestionSource::kP2P, artifact_identifier, source, target, hints, publish_to_global_store, runner);
}

std::string MaterializationFacade::make_request_id(std::string_view prefix) {
  const uint64_t sequence = request_counter_.fetch_add(1, std::memory_order_relaxed);
  const int64_t timestamp = absl::ToUnixNanos(absl::Now());
  return absl::StrCat(prefix, "_", timestamp, "_", sequence);
}

IngestionResultEvent MaterializationFacade::make_ingestion_event_seed(
    const std::string& request_id,
    std::string_view artifact_identifier,
    IngestionSource source,
    const loading::ReplicaTarget& target,
    bool publish_to_global_store,
    const std::string& publish_context_id,
    loading::MaterializeMode mode,
    const loading::MaterializeHints& hints) const {
  IngestionResultEvent event;
  event.request_id = request_id;
  event.source = source;
  event.materialize_mode = mode;
  event.export_policy = hints.export_policy;
  event.artifact_id = std::string(artifact_identifier);
  event.target_device = target.location.to_device_key();
  event.target_location = target.location.type;
  event.publish_to_global_store = publish_to_global_store;
  event.publish_context_id = publish_context_id;
  event.status = absl::OkStatus();
  if (hints.variant && hints.variant->view_id.has_value()) {
    event.view_id = hints.variant->view_id;
  }
  return event;
}

IngestionStartedEvent MaterializationFacade::make_started_event(
    const std::string& request_id,
    std::string_view artifact_identifier,
    IngestionSource source,
    const loading::ReplicaTarget& target,
    const std::string& publish_context_id,
    bool publish_to_global_store,
    loading::MaterializeMode mode,
    const loading::MaterializeHints& hints) const {
  IngestionStartedEvent started;
  started.request_id = request_id;
  started.artifact_id = std::string(artifact_identifier);
  started.source = source;
  started.target = target;
  started.publish_context_id = publish_context_id;
  started.publish_to_global_store = publish_to_global_store;
  started.materialize_mode = mode;
  started.export_policy = hints.export_policy;
  if (hints.variant && hints.variant->view_id.has_value()) {
    started.view_id = hints.variant->view_id;
  }
  return started;
}

void MaterializationFacade::publish_started_event(const IngestionStartedEvent& event) const {
  if (ingestion_event_hub_ != nullptr) {
    ingestion_event_hub_->publish_started(event);
  }
}

void MaterializationFacade::publish_completed_event(IngestionCompletedEvent event) const {
  if (ingestion_event_hub_ != nullptr) {
    ingestion_event_hub_->publish_completed(event);
  }
}

void MaterializationFacade::apply_event_defaults(IngestionResultEvent& event, const IngestionResultEvent& defaults)
    const {
  if (event.request_id.empty()) {
    event.request_id = defaults.request_id;
  }
  if (event.artifact_id.empty()) {
    event.artifact_id = defaults.artifact_id;
  }
  event.source = defaults.source;
  event.materialize_mode = defaults.materialize_mode;
  event.export_policy = defaults.export_policy;
  event.target_device = defaults.target_device;
  event.target_location = defaults.target_location;
  if (!event.view_id.has_value() && defaults.view_id.has_value()) {
    event.view_id = defaults.view_id;
  }
  event.publish_to_global_store = defaults.publish_to_global_store;
  if (event.publish_context_id.empty()) {
    event.publish_context_id = defaults.publish_context_id;
  }
}

std::optional<absl::StatusOr<loading::ReplicaHandle>> MaterializationFacade::maybe_override_result() const {
  if (!hooks_ || !hooks_->override_result) {
    return std::nullopt;
  }
  return hooks_->override_result();
}

void MaterializationFacade::maybe_invoke_before_pipeline_start(const IngestionRequestMetadata& metadata) const {
  if (!hooks_ || !hooks_->before_pipeline_start) {
    return;
  }
  hooks_->before_pipeline_start(metadata);
}

void MaterializationFacade::maybe_mutate_completion_event(IngestionResultEvent& event) const {
  if (!hooks_ || !hooks_->mutate_completion_event) {
    return;
  }
  hooks_->mutate_completion_event(event);
}

void MaterializationFacade::record_publish_context_for_replica(
    const loading::ReplicaKey& key,
    std::string_view publish_context_id) {
  if (publish_context_id.empty()) {
    return;
  }
  absl::MutexLock lock(&publish_context_mu_);
  publish_context_by_replica_[key] = std::string(publish_context_id);
}

std::optional<std::string> MaterializationFacade::lookup_publish_context_for_replica(
    const loading::ReplicaKey& key) const {
  absl::MutexLock lock(&publish_context_mu_);
  auto it = publish_context_by_replica_.find(key);
  if (it == publish_context_by_replica_.end()) {
    return std::nullopt;
  }
  return it->second;
}

} // namespace tensorcast::store::runtime::ingestion
