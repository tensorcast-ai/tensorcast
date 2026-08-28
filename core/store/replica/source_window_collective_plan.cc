// Copyright (c) 2026, TensorCast Team.

#include "core/store/replica/source_window_collective_plan.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "glog/logging.h"

namespace tensorcast::store::replica {
namespace {

namespace contracts = materialization::contracts;
namespace strategy = runtime::ingestion::strategy;

using RepresentationWorkItem = contracts::RepresentationWorkItem;
using RepresentationWorkItemKind = contracts::RepresentationWorkItemKind;
using SourceWindowDistributionMode = strategy::SourceWindowCollectiveDistributionMode;
using SourceWindowSelectionMode = strategy::SourceWindowCollectiveSelectionMode;

struct CandidateSpan {
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

struct Interval {
  uint64_t begin{0};
  uint64_t end{0};
};

struct SourceWindowPlanBuildStats {
  uint64_t work_items{0};
  uint64_t compressed_2d_spans{0};
  uint64_t expanded_work_items{0};
  uint64_t expanded_spans{0};
  uint64_t residual_work_items{0};
  uint64_t guarded_residual_work_items{0};
  uint64_t residual_sample_count{0};
  uint64_t local_typed_work_items{0};
  uint64_t local_typed_bytes{0};
  std::map<int, uint64_t> residual_items_by_kind;
  std::map<int, uint64_t> residual_bytes_by_kind;
  std::map<std::string, uint64_t> residual_items_by_reason;
};

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t* out);
bool checked_mul(uint64_t lhs, uint64_t rhs, uint64_t* out);
uint64_t consumer_span_useful_bytes(const SourceWindowCollectiveConsumerSpan& span);

SourceWindowSelectionMode selection_mode_from_strategy(
    StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode mode) {
  switch (mode) {
    case StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto:
      return SourceWindowSelectionMode::kAuto;
    case StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kStrict:
      return SourceWindowSelectionMode::kStrict;
    case StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kDryRun:
    default:
      return SourceWindowSelectionMode::kDryRun;
  }
}

SourceWindowDistributionMode distribution_mode_from_strategy(
    StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode mode) {
  switch (mode) {
    case StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::
        kFullWindowAllGather:
      return SourceWindowDistributionMode::kFullWindowAllGather;
    case StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kConsumerRouted:
      return SourceWindowDistributionMode::kConsumerRouted;
    case StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kHybridWindow:
      return SourceWindowDistributionMode::kHybridWindow;
    case StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kLocalOnly:
      return SourceWindowDistributionMode::kLocalOnly;
    case StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kAuto:
    default:
      return SourceWindowDistributionMode::kAuto;
  }
}

SourceWindowDistributionMode resolve_distribution_mode(SourceWindowDistributionMode mode) {
  if (mode == SourceWindowDistributionMode::kAuto) {
    return SourceWindowDistributionMode::kHybridWindow;
  }
  return mode;
}

const contracts::RepresentationWorkPlan& member_work_plan(const SourceWindowCollectiveMemberInput& member) {
  if (member.work_plan_ref != nullptr) {
    return *member.work_plan_ref;
  }
  return member.work_plan;
}

const loading::IntoTargetLayout& member_target_layout(const SourceWindowCollectiveMemberInput& member) {
  if (member.target_layout_ref != nullptr) {
    return *member.target_layout_ref;
  }
  return member.target_layout;
}

bool window_has_only_one_consumer_rank(const SourceWindowCollectiveWindow& window) {
  if (window.consumer_spans.empty()) {
    return false;
  }
  const uint32_t rank = window.consumer_spans.front().rank;
  return std::all_of(
      window.consumer_spans.begin(),
      window.consumer_spans.end(),
      [rank](const SourceWindowCollectiveConsumerSpan& span) { return span.rank == rank; });
}

std::optional<uint64_t> consumer_routed_peer_saving_bytes(
    const SourceWindowCollectiveWindow& window,
    uint32_t world_size) {
  if (world_size <= 1 || window_has_only_one_consumer_rank(window)) {
    return std::nullopt;
  }
  const uint64_t window_bytes = window.end > window.start ? window.end - window.start : 0;
  uint64_t all_gather_peer_bytes = std::numeric_limits<uint64_t>::max();
  if (window_bytes <= std::numeric_limits<uint64_t>::max() / (world_size - 1)) {
    all_gather_peer_bytes = window_bytes * static_cast<uint64_t>(world_size - 1);
  }
  uint64_t routed_peer_bytes = 0;
  for (const auto& span : window.consumer_spans) {
    uint64_t span_peer_bytes = 0;
    const uint64_t useful_bytes = consumer_span_useful_bytes(span);
    if (!checked_mul(useful_bytes, world_size - 1, &span_peer_bytes)) {
      return std::nullopt;
    }
    span_peer_bytes /= world_size;
    if (!checked_add(routed_peer_bytes, span_peer_bytes, &routed_peer_bytes)) {
      return std::nullopt;
    }
  }
  if (routed_peer_bytes >= all_gather_peer_bytes) {
    return std::nullopt;
  }
  return all_gather_peer_bytes - routed_peer_bytes;
}

bool window_prefers_consumer_routed(
    const SourceWindowCollectiveWindow& window,
    uint32_t world_size,
    const SourceWindowCollectiveConfig& config) {
  const std::optional<uint64_t> saving = consumer_routed_peer_saving_bytes(window, world_size);
  return saving.has_value() && *saving >= config.min_routed_peer_saving_bytes;
}

SourceWindowDistributionMode resolve_distribution_mode_for_windows(
    SourceWindowDistributionMode requested_mode,
    absl::Span<const SourceWindowCollectiveWindow> windows,
    uint32_t world_size,
    const SourceWindowCollectiveConfig& config) {
  if (requested_mode == SourceWindowDistributionMode::kHybridWindow) {
    return SourceWindowDistributionMode::kHybridWindow;
  }
  if (requested_mode != SourceWindowDistributionMode::kAuto) {
    return requested_mode;
  }
  if (!windows.empty() && std::all_of(windows.begin(), windows.end(), window_has_only_one_consumer_rank)) {
    return SourceWindowDistributionMode::kLocalOnly;
  }
  bool has_consumer_routed = false;
  bool has_full_window = false;
  bool has_local_only = false;
  for (const auto& window : windows) {
    if (window_has_only_one_consumer_rank(window)) {
      has_local_only = true;
      continue;
    }
    if (window_prefers_consumer_routed(window, world_size, config)) {
      has_consumer_routed = true;
      continue;
    }
    has_full_window = true;
  }
  if (has_consumer_routed && !has_full_window && !has_local_only) {
    return SourceWindowDistributionMode::kConsumerRouted;
  }
  if (has_consumer_routed || has_local_only) {
    return SourceWindowDistributionMode::kHybridWindow;
  }
  return SourceWindowDistributionMode::kFullWindowAllGather;
}

SourceWindowDistributionMode resolve_window_distribution_mode(
    SourceWindowDistributionMode plan_mode,
    const SourceWindowCollectiveWindow& window,
    uint32_t world_size,
    const SourceWindowCollectiveConfig& config) {
  if (plan_mode == SourceWindowDistributionMode::kHybridWindow) {
    if (window_has_only_one_consumer_rank(window)) {
      return SourceWindowDistributionMode::kLocalOnly;
    }
    if (window_prefers_consumer_routed(window, world_size, config)) {
      return SourceWindowDistributionMode::kConsumerRouted;
    }
    return SourceWindowDistributionMode::kFullWindowAllGather;
  }
  return plan_mode;
}

void assign_window_distribution_modes(
    SourceWindowDistributionMode plan_mode,
    uint32_t world_size,
    const SourceWindowCollectiveConfig& config,
    std::vector<SourceWindowCollectiveWindow>* windows) {
  if (windows == nullptr) {
    return;
  }
  for (auto& window : *windows) {
    window.distribution_mode = resolve_window_distribution_mode(plan_mode, window, world_size, config);
  }
}

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t* out) {
  if (out == nullptr || rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    return false;
  }
  *out = lhs + rhs;
  return true;
}

bool checked_mul(uint64_t lhs, uint64_t rhs, uint64_t* out) {
  if (out == nullptr || (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
    return false;
  }
  *out = lhs * rhs;
  return true;
}

uint64_t ceil_ratio_x1000(uint64_t numerator, uint64_t denominator) {
  if (denominator == 0) {
    return 0;
  }
  if (numerator > (std::numeric_limits<uint64_t>::max() - denominator + 1) / 1000ULL) {
    return std::numeric_limits<uint64_t>::max();
  }
  return (numerator * 1000ULL + denominator - 1) / denominator;
}

bool amplification_within_limit(uint64_t window_bytes, uint64_t payload_bytes, uint64_t max_x1000) {
  if (payload_bytes == 0 || max_x1000 == 0) {
    return true;
  }
  return ceil_ratio_x1000(window_bytes, payload_bytes) <= max_x1000;
}

bool is_row_major_contiguous(const contracts::RepresentationTensorSpec& spec) {
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

std::optional<contracts::TensorAxisRange> single_axis_range(const contracts::TensorCoordinateSpec& range) {
  if (range.selects_scalar || range.axes.size() != 1) {
    return std::nullopt;
  }
  return range.axes.front();
}

std::optional<contracts::TensorAxisRange> axis_range_for_dim(
    const contracts::TensorCoordinateSpec& range,
    int32_t dim) {
  if (range.selects_scalar) {
    return std::nullopt;
  }
  for (const auto& axis : range.axes) {
    if (axis.dim == dim) {
      return axis;
    }
  }
  return std::nullopt;
}

bool range_is_full_or_dim1_full(
    const contracts::TensorCoordinateSpec& range,
    const contracts::RepresentationTensorSpec& spec) {
  if (range.axes.empty() && !range.selects_scalar) {
    return true;
  }
  const auto axis = single_axis_range(range);
  return axis.has_value() && axis->dim == 1 && spec.shape.size() == 2 && axis->start == 0 && axis->end == spec.shape[1];
}

struct CompressedTargetRows {
  uint64_t target_offset{0};
  uint64_t row_count{0};
  uint64_t row_bytes{0};
  uint64_t target_stride_bytes{0};
};

absl::StatusOr<CompressedTargetRows> resolve_compressed_target_rows(
    const contracts::RepresentationTensorSpec& dst_spec,
    const contracts::TensorCoordinateSpec& destination_range) {
  if (dst_spec.shape.size() == 2) {
    if (!range_is_full_or_dim1_full(destination_range, dst_spec)) {
      return absl::FailedPreconditionError("source-window compressed 2d requires full dim1 destination");
    }
    return CompressedTargetRows{
        .target_offset = dst_spec.logical_offset,
        .row_count = static_cast<uint64_t>(dst_spec.shape[0]),
        .row_bytes = static_cast<uint64_t>(dst_spec.shape[1]) * dst_spec.element_size,
        .target_stride_bytes = static_cast<uint64_t>(dst_spec.stride[0]) * dst_spec.element_size,
    };
  }
  if (dst_spec.shape.size() != 3 || destination_range.selects_scalar) {
    return absl::FailedPreconditionError("source-window compressed target requires 2d or 3d destination");
  }
  const auto expert_axis = axis_range_for_dim(destination_range, 0);
  if (!expert_axis.has_value() || expert_axis->start < 0 || expert_axis->end != expert_axis->start + 1 ||
      expert_axis->end > dst_spec.shape[0]) {
    return absl::FailedPreconditionError("source-window compressed 3d target requires one expert dim0 slice");
  }
  const auto row_axis = axis_range_for_dim(destination_range, 1);
  const auto col_axis = axis_range_for_dim(destination_range, 2);
  const int64_t row_start = row_axis.has_value() ? row_axis->start : 0;
  const int64_t row_end = row_axis.has_value() ? row_axis->end : dst_spec.shape[1];
  const int64_t col_start = col_axis.has_value() ? col_axis->start : 0;
  const int64_t col_end = col_axis.has_value() ? col_axis->end : dst_spec.shape[2];
  if (row_start < 0 || row_end <= row_start || row_end > dst_spec.shape[1] || col_start < 0 || col_end <= col_start ||
      col_end > dst_spec.shape[2]) {
    return absl::FailedPreconditionError("source-window compressed 3d target range mismatch");
  }
  const uint64_t element_bytes = dst_spec.element_size;
  const uint64_t target_offset = dst_spec.logical_offset +
      static_cast<uint64_t>(expert_axis->start) * static_cast<uint64_t>(dst_spec.stride[0]) * element_bytes +
      static_cast<uint64_t>(row_start) * static_cast<uint64_t>(dst_spec.stride[1]) * element_bytes +
      static_cast<uint64_t>(col_start) * static_cast<uint64_t>(dst_spec.stride[2]) * element_bytes;
  return CompressedTargetRows{
      .target_offset = target_offset,
      .row_count = static_cast<uint64_t>(row_end - row_start),
      .row_bytes = static_cast<uint64_t>(col_end - col_start) * element_bytes,
      .target_stride_bytes = static_cast<uint64_t>(dst_spec.stride[1]) * element_bytes,
  };
}

std::optional<uint64_t> estimate_coordinate_span_count(
    const contracts::RepresentationTensorSpec& spec,
    const contracts::TensorCoordinateSpec& range) {
  if (range.selects_scalar || spec.shape.empty() || range.axes.empty()) {
    return 1;
  }
  if (!is_row_major_contiguous(spec)) {
    return std::nullopt;
  }
  std::vector<int64_t> starts(spec.shape.size(), 0);
  std::vector<int64_t> ends = spec.shape;
  for (const auto& axis : range.axes) {
    if (axis.dim < 0 || axis.dim >= static_cast<int32_t>(spec.shape.size()) || axis.end <= axis.start) {
      return std::nullopt;
    }
    starts[static_cast<size_t>(axis.dim)] = axis.start;
    ends[static_cast<size_t>(axis.dim)] = axis.end;
  }
  uint64_t count = 1;
  for (size_t dim = 0; dim < spec.shape.size(); ++dim) {
    bool later_full = true;
    for (size_t next = dim + 1; next < spec.shape.size(); ++next) {
      if (starts[next] != 0 || ends[next] != spec.shape[next]) {
        later_full = false;
        break;
      }
    }
    if (later_full) {
      return count;
    }
    const uint64_t extent = static_cast<uint64_t>(ends[dim] - starts[dim]);
    if (!checked_mul(count, extent, &count)) {
      return std::nullopt;
    }
  }
  return count;
}

bool coordinate_expansion_is_too_large(
    const contracts::RepresentationTensorSpec& source_spec,
    const contracts::TensorCoordinateSpec& source_range,
    const contracts::RepresentationTensorSpec& dst_spec,
    const contracts::TensorCoordinateSpec& destination_range) {
  constexpr uint64_t kMaxFallbackCoordinateSpanExpansion = 128;
  const auto source_count = estimate_coordinate_span_count(source_spec, source_range);
  const auto destination_count = estimate_coordinate_span_count(dst_spec, destination_range);
  return !source_count.has_value() || !destination_count.has_value() ||
      *source_count > kMaxFallbackCoordinateSpanExpansion || *destination_count > kMaxFallbackCoordinateSpanExpansion;
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
  return absl::StrCat("[", absl::StrJoin(shape, ","), "]");
}

std::string range_to_string(const contracts::TensorCoordinateSpec& range) {
  std::vector<std::string> axes;
  axes.reserve(range.axes.size());
  for (const auto& axis : range.axes) {
    axes.push_back(absl::StrCat(axis.dim, ":", axis.start, "-", axis.end));
  }
  return absl::StrCat("{scalar=", range.selects_scalar ? "1" : "0", ",axes=[", absl::StrJoin(axes, ","), "]}");
}

uint64_t span_window_bytes(const CandidateSpan& span) {
  return span.source_window_end > span.source_window_start ? span.source_window_end - span.source_window_start : 0;
}

uint64_t consumer_span_useful_bytes(const SourceWindowCollectiveConsumerSpan& span) {
  if (span.row_count > 1 && span.row_bytes > 0) {
    return span.row_count * span.row_bytes;
  }
  return span.length;
}

bool source_window_copy_kind(RepresentationWorkItemKind kind) {
  return kind == RepresentationWorkItemKind::kTensorCopy || kind == RepresentationWorkItemKind::kExpertDim0Concat ||
      kind == RepresentationWorkItemKind::kConcatAssemble;
}

bool source_window_local_typed_kind(RepresentationWorkItemKind kind) {
  return kind == RepresentationWorkItemKind::kConstFill || kind == RepresentationWorkItemKind::kScalarBroadcastFill ||
      kind == RepresentationWorkItemKind::kPadFill;
}

std::string residual_map_to_string(const std::map<int, uint64_t>& values) {
  std::vector<std::string> parts;
  parts.reserve(values.size());
  for (const auto& [key, value] : values) {
    parts.push_back(absl::StrCat(key, ":", value));
  }
  return absl::StrJoin(parts, ",");
}

std::string residual_reason_map_to_string(const std::map<std::string, uint64_t>& values) {
  std::vector<std::string> parts;
  parts.reserve(values.size());
  for (const auto& [key, value] : values) {
    parts.push_back(absl::StrCat(key, ":", value));
  }
  return absl::StrJoin(parts, ",");
}

void record_residual_work_item(
    SourceWindowPlanBuildStats* stats,
    const SourceWindowCollectiveMemberInput& member,
    const RepresentationWorkItem& item,
    std::string_view reason,
    uint64_t bytes) {
  if (stats == nullptr) {
    return;
  }
  stats->residual_work_items += 1;
  const int kind = static_cast<int>(item.kind);
  stats->residual_items_by_kind[kind] += 1;
  stats->residual_bytes_by_kind[kind] += bytes;
  stats->residual_items_by_reason[std::string(reason)] += 1;
  if (stats->residual_sample_count < 32) {
    LOG(INFO) << "tc_profile source_window_collective_plan_builder residual_sample"
              << " rank=" << member.rank << " reason=" << reason << " dst_name=" << item.dst_name
              << " item_kind=" << kind << " partition_kind=" << static_cast<int>(item.partition_kind)
              << " source_count=" << item.sources.size() << " committed_bytes=" << item.committed_bytes
              << " dst_shape=" << shape_to_string(item.dst_spec.shape);
    stats->residual_sample_count += 1;
  }
}

CandidateSpan make_contiguous_candidate_span(
    const SourceWindowCollectiveMemberInput& member,
    uint32_t storage_index,
    uint64_t source_offset,
    uint64_t target_offset,
    uint64_t length) {
  return CandidateSpan{
      .rank = member.rank,
      .source_index = 0,
      .storage_index = storage_index,
      .source_window_start = source_offset,
      .source_window_end = source_offset + length,
      .source_offset = source_offset,
      .target_offset = target_offset,
      .length = length,
      .row_count = 1,
      .row_bytes = length,
      .source_stride_bytes = length,
      .target_stride_bytes = length,
  };
}

std::vector<TargetStorageSpan> storage_spans_for_member(const SourceWindowCollectiveMemberInput& member) {
  if (!member.storage_spans.empty()) {
    return member.storage_spans;
  }
  const auto& target_layout = member_target_layout(member);
  std::vector<TargetStorageSpan> spans;
  spans.reserve(target_layout.storages.size());
  uint64_t cursor = 0;
  for (const auto& storage : target_layout.storages) {
    spans.push_back(
        TargetStorageSpan{
            .base_offset = cursor,
            .length = storage.length,
            .base_ptr = static_cast<std::uint8_t*>(storage.base_ptr.get()),
        });
    cursor += storage.length;
  }
  return spans;
}

absl::Status append_storage_split_spans(
    const SourceWindowCollectiveMemberInput& member,
    absl::Span<const TargetStorageSpan> storage_spans,
    uint64_t source_offset,
    uint64_t target_offset,
    uint64_t length,
    std::vector<CandidateSpan>* spans) {
  if (spans == nullptr) {
    return absl::InternalError("source-window span output is null");
  }
  uint64_t remaining = length;
  uint64_t src_cursor = source_offset;
  uint64_t dst_cursor = target_offset;
  while (remaining > 0) {
    std::optional<uint32_t> storage_index;
    uint64_t storage_end = 0;
    for (uint32_t index = 0; index < storage_spans.size(); ++index) {
      const auto& storage = storage_spans[index];
      storage_end = storage.base_offset + storage.length;
      if (dst_cursor >= storage.base_offset && dst_cursor < storage_end) {
        storage_index = index;
        break;
      }
    }
    if (!storage_index.has_value()) {
      return absl::InvalidArgumentError(
          absl::StrCat("source-window target span is outside target storage: rank=", member.rank));
    }
    const uint64_t chunk = std::min<uint64_t>(remaining, storage_end - dst_cursor);
    spans->push_back(make_contiguous_candidate_span(member, *storage_index, src_cursor, dst_cursor, chunk));
    remaining -= chunk;
    src_cursor += chunk;
    dst_cursor += chunk;
  }
  return absl::OkStatus();
}

absl::Status append_paired_source_destination_spans(
    const SourceWindowCollectiveMemberInput& member,
    absl::Span<const TargetStorageSpan> storage_spans,
    absl::Span<const contracts::TensorByteSpan> source_spans,
    absl::Span<const contracts::TensorByteSpan> destination_spans,
    std::vector<CandidateSpan>* spans) {
  size_t source_index = 0;
  size_t destination_index = 0;
  uint64_t source_consumed = 0;
  uint64_t destination_consumed = 0;
  while (source_index < source_spans.size() && destination_index < destination_spans.size()) {
    const auto& source = source_spans[source_index];
    const auto& destination = destination_spans[destination_index];
    const uint64_t source_remaining = source.length - source_consumed;
    const uint64_t destination_remaining = destination.length - destination_consumed;
    const uint64_t chunk = std::min(source_remaining, destination_remaining);
    if (chunk > 0) {
      auto status = append_storage_split_spans(
          member,
          storage_spans,
          source.offset + source_consumed,
          destination.offset + destination_consumed,
          chunk,
          spans);
      if (!status.ok()) {
        return status;
      }
    }
    source_consumed += chunk;
    destination_consumed += chunk;
    if (source_consumed == source.length) {
      source_index += 1;
      source_consumed = 0;
    }
    if (destination_consumed == destination.length) {
      destination_index += 1;
      destination_consumed = 0;
    }
  }
  if (source_index != source_spans.size() || destination_index != destination_spans.size()) {
    return absl::InvalidArgumentError("source-window source/destination byte spans have different total sizes");
  }
  return absl::OkStatus();
}

std::optional<uint32_t> storage_index_for_target_offset(
    absl::Span<const TargetStorageSpan> storage_spans,
    uint64_t target_offset,
    uint64_t target_end) {
  for (uint32_t index = 0; index < storage_spans.size(); ++index) {
    const auto& storage = storage_spans[index];
    const uint64_t storage_end = storage.base_offset + storage.length;
    if (target_offset >= storage.base_offset && target_end <= storage_end) {
      return index;
    }
  }
  return std::nullopt;
}

absl::Status append_compressed_2d_source_span(
    const SourceWindowCollectiveMemberInput& member,
    absl::Span<const TargetStorageSpan> storage_spans,
    const contracts::RepresentationWorkSourceFragment& source,
    const contracts::RepresentationWorkItem& item,
    std::vector<CandidateSpan>* spans,
    SourceWindowPlanBuildStats* stats) {
  const auto& source_spec = source.fragment.source_spec;
  const auto& dst_spec = item.dst_spec;
  if (source_spec.shape.size() != 2 || (dst_spec.shape.size() != 2 && dst_spec.shape.size() != 3) ||
      source_spec.element_size == 0 || dst_spec.element_size == 0 ||
      source_spec.element_size != dst_spec.element_size || !is_row_major_contiguous(source_spec) ||
      !is_row_major_contiguous(dst_spec)) {
    return absl::FailedPreconditionError("source-window compressed requires row-major source/target tensors");
  }
  if (source_spec.shape[1] <= 0) {
    return absl::FailedPreconditionError("source-window compressed source shape mismatch");
  }
  const auto src_row_axis = axis_range_for_dim(source.fragment.source_range, 0);
  const auto src_col_axis = axis_range_for_dim(source.fragment.source_range, 1);
  const int64_t row_start = src_row_axis.has_value() ? src_row_axis->start : 0;
  const int64_t row_end = src_row_axis.has_value() ? src_row_axis->end : source_spec.shape[0];
  const int64_t col_start = src_col_axis.has_value() ? src_col_axis->start : 0;
  const int64_t col_end = src_col_axis.has_value() ? src_col_axis->end : source_spec.shape[1];
  if (row_start < 0 || row_end <= row_start || row_end > source_spec.shape[0] || col_start < 0 ||
      col_end <= col_start || col_end > source_spec.shape[1]) {
    return absl::FailedPreconditionError("source-window compressed 2d source range mismatch");
  }
  auto target_rows_or = resolve_compressed_target_rows(dst_spec, source.fragment.destination_range);
  if (!target_rows_or.ok()) {
    return target_rows_or.status();
  }
  const auto& target_rows = *target_rows_or;

  const uint64_t element_bytes = source_spec.element_size;
  const uint64_t row_count = static_cast<uint64_t>(row_end - row_start);
  const uint64_t source_stride_bytes = static_cast<uint64_t>(source_spec.shape[1]) * element_bytes;
  const uint64_t row_bytes = static_cast<uint64_t>(col_end - col_start) * element_bytes;
  if (row_count != target_rows.row_count || row_bytes != target_rows.row_bytes) {
    return absl::FailedPreconditionError("source-window compressed 2d destination width mismatch");
  }
  const bool full_row_slice = !src_row_axis.has_value() && row_start == 0 && row_end == source_spec.shape[0];
  const uint64_t source_offset = source_spec.logical_offset + static_cast<uint64_t>(row_start) * source_stride_bytes +
      static_cast<uint64_t>(col_start) * element_bytes;
  const uint64_t source_window_start = full_row_slice ? source_spec.logical_offset : source_offset;
  const uint64_t source_window_end = full_row_slice
      ? source_spec.logical_offset + static_cast<uint64_t>(source_spec.shape[0]) * source_stride_bytes
      : source_spec.logical_offset + static_cast<uint64_t>(row_end - 1) * source_stride_bytes +
          static_cast<uint64_t>(col_end) * element_bytes;
  const uint64_t target_offset = target_rows.target_offset;
  const uint64_t target_stride_bytes = target_rows.target_stride_bytes;
  const uint64_t target_end = target_offset + (row_count - 1) * target_stride_bytes + row_bytes;
  const auto storage_index = storage_index_for_target_offset(storage_spans, target_offset, target_end);
  if (!storage_index.has_value()) {
    return absl::FailedPreconditionError("source-window compressed 2d target crosses storage boundary");
  }
  spans->push_back(
      CandidateSpan{
          .rank = member.rank,
          .source_index = 0,
          .storage_index = *storage_index,
          .source_window_start = source_window_start,
          .source_window_end = source_window_end,
          .source_offset = source_offset,
          .target_offset = target_offset,
          .length = row_count * row_bytes,
          .row_count = row_count,
          .row_bytes = row_bytes,
          .source_stride_bytes = source_stride_bytes,
          .target_stride_bytes = target_stride_bytes,
      });
  if (stats != nullptr) {
    stats->compressed_2d_spans += 1;
  }
  return absl::OkStatus();
}

absl::Status append_work_item_spans(
    const SourceWindowCollectiveMemberInput& member,
    absl::Span<const TargetStorageSpan> storage_spans,
    const RepresentationWorkItem& item,
    uint64_t* residual_bytes,
    std::vector<CandidateSpan>* spans,
    SourceWindowPlanBuildStats* stats) {
  if (stats != nullptr) {
    stats->work_items += 1;
  }
  if (source_window_local_typed_kind(item.kind)) {
    if (stats != nullptr) {
      stats->local_typed_work_items += 1;
      stats->local_typed_bytes += item.committed_bytes;
    }
    return absl::OkStatus();
  }
  if (!source_window_copy_kind(item.kind) || item.sources.empty()) {
    *residual_bytes += item.committed_bytes;
    record_residual_work_item(stats, member, item, "unsupported_item_kind_or_empty_sources", item.committed_bytes);
    return absl::OkStatus();
  }
  for (const auto& source : item.sources) {
    if (source.prefix_count != 1) {
      *residual_bytes += item.committed_bytes;
      record_residual_work_item(stats, member, item, "prefix_count_not_one", item.committed_bytes);
      return absl::OkStatus();
    }
    if (source.fragment.source_spec.shape.size() == 2 &&
        (item.dst_spec.shape.size() == 2 || item.dst_spec.shape.size() == 3)) {
      auto compressed_status = append_compressed_2d_source_span(member, storage_spans, source, item, spans, stats);
      if (compressed_status.ok()) {
        continue;
      }
    }
    if (coordinate_expansion_is_too_large(
            source.fragment.source_spec,
            source.fragment.source_range,
            item.dst_spec,
            source.fragment.destination_range)) {
      if (stats != nullptr && stats->guarded_residual_work_items < 16) {
        LOG(INFO) << "tc_profile source_window_collective_plan_builder guarded_sample"
                  << " rank=" << member.rank << " dst_name=" << item.dst_name
                  << " item_kind=" << static_cast<int>(item.kind)
                  << " partition_kind=" << static_cast<int>(item.partition_kind)
                  << " source_shape=" << shape_to_string(source.fragment.source_spec.shape)
                  << " dst_shape=" << shape_to_string(item.dst_spec.shape)
                  << " source_range=" << range_to_string(source.fragment.source_range)
                  << " destination_range=" << range_to_string(source.fragment.destination_range);
      }
      *residual_bytes += item.committed_bytes;
      if (stats != nullptr) {
        stats->guarded_residual_work_items += 1;
      }
      record_residual_work_item(stats, member, item, "coordinate_expansion_too_large", item.committed_bytes);
      return absl::OkStatus();
    }
    auto source_spans_or =
        contracts::build_coordinate_byte_spans(source.fragment.source_spec, source.fragment.source_range);
    if (!source_spans_or.ok()) {
      return source_spans_or.status();
    }
    auto destination_spans_or =
        contracts::build_coordinate_byte_spans(item.dst_spec, source.fragment.destination_range);
    if (!destination_spans_or.ok()) {
      return destination_spans_or.status();
    }
    auto status =
        append_paired_source_destination_spans(member, storage_spans, *source_spans_or, *destination_spans_or, spans);
    if (!status.ok()) {
      return status;
    }
    if (stats != nullptr) {
      stats->expanded_work_items += 1;
      stats->expanded_spans += source_spans_or->size();
    }
  }
  return absl::OkStatus();
}

absl::Status validate_group_input(const SourceWindowCollectiveGroupInput& input) {
  if (!input.config.enabled) {
    return absl::FailedPreconditionError("source_window_disabled");
  }
  if (input.disk_context == nullptr) {
    return absl::FailedPreconditionError("disk_context_missing");
  }
  if (input.group.world_size <= 1) {
    return absl::FailedPreconditionError("collective_group_missing");
  }
  if (input.members.size() != input.group.world_size) {
    return absl::FailedPreconditionError("member_count_mismatch");
  }
  std::vector<bool> seen(input.group.world_size, false);
  for (const auto& member : input.members) {
    const auto& target_layout = member_target_layout(member);
    if (member.rank >= input.group.world_size) {
      return absl::InvalidArgumentError("member_rank_out_of_bounds");
    }
    if (seen[member.rank]) {
      return absl::InvalidArgumentError("duplicate_member_rank");
    }
    seen[member.rank] = true;
    uint64_t target_total = 0;
    for (const auto& storage : target_layout.storages) {
      if (!checked_add(target_total, storage.length, &target_total)) {
        return absl::OutOfRangeError("target layout size overflows");
      }
    }
    if (target_total != 0 && target_layout.total_size != 0 && target_total != target_layout.total_size) {
      return absl::InvalidArgumentError("target layout storage sizes do not match total_size");
    }
  }
  return absl::OkStatus();
}

uint64_t unique_payload_bytes_for_spans(absl::Span<const CandidateSpan> spans) {
  std::map<uint32_t, std::vector<Interval>> by_source;
  std::map<std::tuple<uint32_t, uint64_t, uint64_t, uint64_t>, std::vector<Interval>> by_2d_source;
  for (const auto& span : spans) {
    if (span.length == 0) {
      continue;
    }
    if (span.row_count > 1 && span.row_bytes > 0 && span.source_stride_bytes > 0) {
      const auto key =
          std::make_tuple(span.source_index, span.source_window_start, span.row_count, span.source_stride_bytes);
      by_2d_source[key].push_back(
          Interval{
              .begin = span.source_offset - span.source_window_start,
              .end = span.source_offset - span.source_window_start + span.row_bytes,
          });
      continue;
    }
    by_source[span.source_index].push_back(
        Interval{.begin = span.source_offset, .end = span.source_offset + span.length});
  }
  uint64_t total = 0;
  for (auto& [_, intervals] : by_source) {
    std::sort(intervals.begin(), intervals.end(), [](const Interval& lhs, const Interval& rhs) {
      if (lhs.begin != rhs.begin) {
        return lhs.begin < rhs.begin;
      }
      return lhs.end < rhs.end;
    });
    uint64_t current_begin = 0;
    uint64_t current_end = 0;
    bool has_current = false;
    for (const auto& interval : intervals) {
      if (!has_current) {
        current_begin = interval.begin;
        current_end = interval.end;
        has_current = true;
        continue;
      }
      if (interval.begin <= current_end) {
        current_end = std::max(current_end, interval.end);
      } else {
        total += current_end - current_begin;
        current_begin = interval.begin;
        current_end = interval.end;
      }
    }
    if (has_current) {
      total += current_end - current_begin;
    }
  }
  for (auto& [key, intervals] : by_2d_source) {
    std::sort(intervals.begin(), intervals.end(), [](const Interval& lhs, const Interval& rhs) {
      if (lhs.begin != rhs.begin) {
        return lhs.begin < rhs.begin;
      }
      return lhs.end < rhs.end;
    });
    uint64_t merged_width = 0;
    uint64_t current_begin = 0;
    uint64_t current_end = 0;
    bool has_current = false;
    for (const auto& interval : intervals) {
      if (!has_current) {
        current_begin = interval.begin;
        current_end = interval.end;
        has_current = true;
        continue;
      }
      if (interval.begin <= current_end) {
        current_end = std::max(current_end, interval.end);
      } else {
        merged_width += current_end - current_begin;
        current_begin = interval.begin;
        current_end = interval.end;
      }
    }
    if (has_current) {
      merged_width += current_end - current_begin;
    }
    total += merged_width * std::get<2>(key);
  }
  return total;
}

uint64_t merged_interval_bytes(std::vector<Interval>* intervals) {
  if (intervals == nullptr || intervals->empty()) {
    return 0;
  }
  std::sort(intervals->begin(), intervals->end(), [](const Interval& lhs, const Interval& rhs) {
    if (lhs.begin != rhs.begin) {
      return lhs.begin < rhs.begin;
    }
    return lhs.end < rhs.end;
  });
  uint64_t total = 0;
  uint64_t current_begin = intervals->front().begin;
  uint64_t current_end = intervals->front().end;
  for (size_t index = 1; index < intervals->size(); ++index) {
    const auto& interval = (*intervals)[index];
    if (interval.begin <= current_end) {
      current_end = std::max(current_end, interval.end);
      continue;
    }
    total += current_end - current_begin;
    current_begin = interval.begin;
    current_end = interval.end;
  }
  total += current_end - current_begin;
  return total;
}

using LocalMappedReadIntervalMap = std::map<std::pair<uint32_t, uint32_t>, std::vector<Interval>>;

void append_local_mapped_read_interval(
    LocalMappedReadIntervalMap* intervals_by_rank_source,
    const SourceWindowCollectiveConsumerSpan& span,
    uint32_t world_size) {
  if (intervals_by_rank_source == nullptr || span.rank >= world_size || span.length == 0) {
    return;
  }
  const uint64_t begin =
      span.source_window_end > span.source_window_start ? span.source_window_start : span.source_offset;
  const uint64_t end =
      span.source_window_end > span.source_window_start ? span.source_window_end : span.source_offset + span.length;
  (*intervals_by_rank_source)[{span.rank, span.source_index}].push_back(Interval{.begin = begin, .end = end});
}

std::vector<uint64_t> finish_local_mapped_physical_read_bytes_by_rank(
    LocalMappedReadIntervalMap* intervals_by_rank_source,
    uint32_t world_size) {
  std::vector<uint64_t> bytes_by_rank(world_size, 0);
  if (intervals_by_rank_source == nullptr) {
    return bytes_by_rank;
  }
  for (auto& [key, intervals] : *intervals_by_rank_source) {
    const uint32_t rank = key.first;
    if (rank < world_size) {
      bytes_by_rank[rank] += merged_interval_bytes(&intervals);
    }
  }
  return bytes_by_rank;
}

std::vector<CandidateSpan> split_spans_by_window_cap(absl::Span<const CandidateSpan> spans, uint64_t window_bytes) {
  if (window_bytes == 0) {
    return std::vector<CandidateSpan>(spans.begin(), spans.end());
  }
  std::vector<CandidateSpan> out;
  for (const auto& span : spans) {
    if (span.row_count > 1 || span_window_bytes(span) > span.length) {
      out.push_back(span);
      continue;
    }
    uint64_t remaining = span.length;
    uint64_t source_cursor = span.source_offset;
    uint64_t target_cursor = span.target_offset;
    while (remaining > 0) {
      const uint64_t chunk = std::min<uint64_t>(remaining, window_bytes);
      CandidateSpan split = span;
      split.source_window_start = source_cursor;
      split.source_window_end = source_cursor + chunk;
      split.source_offset = source_cursor;
      split.target_offset = target_cursor;
      split.length = chunk;
      split.row_count = 1;
      split.row_bytes = chunk;
      split.source_stride_bytes = chunk;
      split.target_stride_bytes = chunk;
      out.push_back(split);
      remaining -= chunk;
      source_cursor += chunk;
      target_cursor += chunk;
    }
  }
  return out;
}

uint64_t unique_payload_bytes_for_same_window_group(absl::Span<const CandidateSpan> spans) {
  if (spans.empty()) {
    return 0;
  }
  const bool all_2d = std::all_of(spans.begin(), spans.end(), [](const CandidateSpan& span) {
    return span.row_count > 1 && span.row_bytes > 0 && span.source_stride_bytes > 0;
  });
  if (!all_2d) {
    return unique_payload_bytes_for_spans(spans);
  }

  std::map<std::tuple<uint32_t, uint64_t, uint64_t, uint64_t>, std::vector<Interval>> intervals_by_2d_base;
  uint64_t total = 0;
  for (const auto& span : spans) {
    const auto key =
        std::make_tuple(span.source_index, span.source_window_start, span.row_count, span.source_stride_bytes);
    intervals_by_2d_base[key].push_back(
        Interval{
            .begin = span.source_offset - span.source_window_start,
            .end = span.source_offset - span.source_window_start + span.row_bytes,
        });
  }
  for (auto& [key, intervals] : intervals_by_2d_base) {
    std::sort(intervals.begin(), intervals.end(), [](const Interval& lhs, const Interval& rhs) {
      if (lhs.begin != rhs.begin) {
        return lhs.begin < rhs.begin;
      }
      return lhs.end < rhs.end;
    });
    uint64_t merged_width = 0;
    bool has_current = false;
    uint64_t current_begin = 0;
    uint64_t current_end = 0;
    for (const auto& interval : intervals) {
      if (!has_current) {
        current_begin = interval.begin;
        current_end = interval.end;
        has_current = true;
        continue;
      }
      if (interval.begin <= current_end) {
        current_end = std::max(current_end, interval.end);
      } else {
        merged_width += current_end - current_begin;
        current_begin = interval.begin;
        current_end = interval.end;
      }
    }
    if (has_current) {
      merged_width += current_end - current_begin;
    }
    total += merged_width * std::get<2>(key);
  }
  return total;
}

SourceWindowCollectiveConsumerSpan to_consumer_span(const CandidateSpan& span) {
  return SourceWindowCollectiveConsumerSpan{
      .rank = span.rank,
      .source_index = span.source_index,
      .storage_index = span.storage_index,
      .source_window_start = span.source_window_start,
      .source_window_end = span.source_window_end,
      .source_offset = span.source_offset,
      .target_offset = span.target_offset,
      .length = span.length,
      .row_count = span.row_count,
      .row_bytes = span.row_bytes == 0 ? span.length : span.row_bytes,
      .source_stride_bytes = span.source_stride_bytes == 0 ? span.length : span.source_stride_bytes,
      .target_stride_bytes = span.target_stride_bytes == 0 ? span.length : span.target_stride_bytes,
  };
}

bool is_linear_consumer_span(const SourceWindowCollectiveConsumerSpan& span) {
  return span.row_count <= 1;
}

bool can_merge_common_consumer_span_prefix(
    const SourceWindowCollectiveConsumerSpan& lhs,
    const SourceWindowCollectiveConsumerSpan& rhs) {
  return lhs.rank == rhs.rank && lhs.source_index == rhs.source_index && lhs.storage_index == rhs.storage_index;
}

bool try_merge_linear_consumer_span(
    SourceWindowCollectiveConsumerSpan* lhs,
    const SourceWindowCollectiveConsumerSpan& rhs) {
  uint64_t lhs_source_end = 0;
  uint64_t lhs_target_end = 0;
  if (lhs == nullptr || !is_linear_consumer_span(*lhs) || !is_linear_consumer_span(rhs) ||
      !can_merge_common_consumer_span_prefix(*lhs, rhs) ||
      !checked_add(lhs->source_offset, lhs->length, &lhs_source_end) ||
      !checked_add(lhs->target_offset, lhs->length, &lhs_target_end) || lhs_source_end != rhs.source_offset ||
      lhs_target_end != rhs.target_offset) {
    return false;
  }
  uint64_t merged_length = 0;
  if (!checked_add(lhs->length, rhs.length, &merged_length)) {
    return false;
  }
  lhs->source_window_start = std::min(lhs->source_window_start, rhs.source_window_start);
  lhs->source_window_end = std::max(lhs->source_window_end, rhs.source_window_end);
  lhs->length = merged_length;
  lhs->row_count = 1;
  lhs->row_bytes = merged_length;
  lhs->source_stride_bytes = merged_length;
  lhs->target_stride_bytes = merged_length;
  return true;
}

bool try_merge_horizontal_2d_consumer_span(
    SourceWindowCollectiveConsumerSpan* lhs,
    const SourceWindowCollectiveConsumerSpan& rhs) {
  uint64_t lhs_source_row_end = 0;
  uint64_t lhs_target_row_end = 0;
  if (lhs == nullptr || lhs->row_count <= 1 || rhs.row_count <= 1 ||
      !can_merge_common_consumer_span_prefix(*lhs, rhs) || lhs->row_count != rhs.row_count ||
      lhs->source_stride_bytes != rhs.source_stride_bytes || lhs->target_stride_bytes != rhs.target_stride_bytes ||
      !checked_add(lhs->source_offset, lhs->row_bytes, &lhs_source_row_end) ||
      !checked_add(lhs->target_offset, lhs->row_bytes, &lhs_target_row_end) ||
      lhs_source_row_end != rhs.source_offset || lhs_target_row_end != rhs.target_offset) {
    return false;
  }
  uint64_t merged_row_bytes = 0;
  uint64_t merged_length = 0;
  if (!checked_add(lhs->row_bytes, rhs.row_bytes, &merged_row_bytes) || merged_row_bytes > lhs->source_stride_bytes ||
      merged_row_bytes > lhs->target_stride_bytes || !checked_mul(lhs->row_count, merged_row_bytes, &merged_length)) {
    return false;
  }
  lhs->source_window_start = std::min(lhs->source_window_start, rhs.source_window_start);
  lhs->source_window_end = std::max(lhs->source_window_end, rhs.source_window_end);
  lhs->row_bytes = merged_row_bytes;
  lhs->length = merged_length;
  return true;
}

bool try_merge_vertical_2d_consumer_span(
    SourceWindowCollectiveConsumerSpan* lhs,
    const SourceWindowCollectiveConsumerSpan& rhs) {
  uint64_t lhs_source_span = 0;
  uint64_t lhs_target_span = 0;
  uint64_t expected_rhs_source_offset = 0;
  uint64_t expected_rhs_target_offset = 0;
  if (lhs == nullptr || lhs->row_count <= 1 || rhs.row_count <= 1 ||
      !can_merge_common_consumer_span_prefix(*lhs, rhs) || lhs->row_bytes != rhs.row_bytes ||
      lhs->source_stride_bytes != rhs.source_stride_bytes || lhs->target_stride_bytes != rhs.target_stride_bytes ||
      !checked_mul(lhs->row_count, lhs->source_stride_bytes, &lhs_source_span) ||
      !checked_mul(lhs->row_count, lhs->target_stride_bytes, &lhs_target_span) ||
      !checked_add(lhs->source_offset, lhs_source_span, &expected_rhs_source_offset) ||
      !checked_add(lhs->target_offset, lhs_target_span, &expected_rhs_target_offset) ||
      expected_rhs_source_offset != rhs.source_offset || expected_rhs_target_offset != rhs.target_offset) {
    return false;
  }
  uint64_t merged_row_count = 0;
  uint64_t merged_length = 0;
  if (!checked_add(lhs->row_count, rhs.row_count, &merged_row_count) ||
      !checked_mul(merged_row_count, lhs->row_bytes, &merged_length)) {
    return false;
  }
  lhs->source_window_start = std::min(lhs->source_window_start, rhs.source_window_start);
  lhs->source_window_end = std::max(lhs->source_window_end, rhs.source_window_end);
  lhs->row_count = merged_row_count;
  lhs->length = merged_length;
  return true;
}

std::vector<SourceWindowCollectiveConsumerSpan> coalesce_consumer_spans_for_scatter(
    std::vector<SourceWindowCollectiveConsumerSpan> spans) {
  if (spans.size() < 2) {
    return spans;
  }
  std::sort(
      spans.begin(),
      spans.end(),
      [](const SourceWindowCollectiveConsumerSpan& lhs, const SourceWindowCollectiveConsumerSpan& rhs) {
        return std::tie(
                   lhs.rank,
                   lhs.storage_index,
                   lhs.source_index,
                   lhs.source_offset,
                   lhs.target_offset,
                   lhs.row_count,
                   lhs.source_stride_bytes,
                   lhs.target_stride_bytes) <
            std::tie(
                   rhs.rank,
                   rhs.storage_index,
                   rhs.source_index,
                   rhs.source_offset,
                   rhs.target_offset,
                   rhs.row_count,
                   rhs.source_stride_bytes,
                   rhs.target_stride_bytes);
      });
  std::vector<SourceWindowCollectiveConsumerSpan> merged;
  merged.reserve(spans.size());
  for (const auto& span : spans) {
    if (!merged.empty() &&
        (try_merge_linear_consumer_span(&merged.back(), span) ||
         try_merge_horizontal_2d_consumer_span(&merged.back(), span) ||
         try_merge_vertical_2d_consumer_span(&merged.back(), span))) {
      continue;
    }
    merged.push_back(span);
  }
  return merged;
}

struct TensorStageCopyOp {
  uint32_t rank{0};
  uint32_t source_index{0};
  uint32_t storage_index{0};
  std::string dst_name;
  std::string source_name;
  uint64_t source_offset{0};
  uint64_t target_offset{0};
  uint64_t length{0};
  uint64_t row_count{1};
  uint64_t row_bytes{0};
  uint64_t source_stride_bytes{0};
  uint64_t target_stride_bytes{0};
};

bool is_linear_tensor_stage_op(const TensorStageCopyOp& op) {
  return op.row_count <= 1;
}

bool can_merge_common_tensor_stage_prefix(const TensorStageCopyOp& lhs, const TensorStageCopyOp& rhs) {
  return lhs.rank == rhs.rank && lhs.source_index == rhs.source_index && lhs.storage_index == rhs.storage_index &&
      lhs.dst_name == rhs.dst_name && lhs.source_name == rhs.source_name;
}

bool try_merge_linear_tensor_stage_op(TensorStageCopyOp* lhs, const TensorStageCopyOp& rhs) {
  uint64_t lhs_source_end = 0;
  uint64_t lhs_target_end = 0;
  if (lhs == nullptr || !is_linear_tensor_stage_op(*lhs) || !is_linear_tensor_stage_op(rhs) ||
      !can_merge_common_tensor_stage_prefix(*lhs, rhs) ||
      !checked_add(lhs->source_offset, lhs->length, &lhs_source_end) ||
      !checked_add(lhs->target_offset, lhs->length, &lhs_target_end) || lhs_source_end != rhs.source_offset ||
      lhs_target_end != rhs.target_offset) {
    return false;
  }
  uint64_t merged_length = 0;
  if (!checked_add(lhs->length, rhs.length, &merged_length)) {
    return false;
  }
  lhs->length = merged_length;
  lhs->row_count = 1;
  lhs->row_bytes = merged_length;
  lhs->source_stride_bytes = merged_length;
  lhs->target_stride_bytes = merged_length;
  return true;
}

bool try_merge_horizontal_2d_tensor_stage_op(TensorStageCopyOp* lhs, const TensorStageCopyOp& rhs) {
  uint64_t lhs_source_row_end = 0;
  uint64_t lhs_target_row_end = 0;
  if (lhs == nullptr || lhs->row_count <= 1 || rhs.row_count <= 1 || !can_merge_common_tensor_stage_prefix(*lhs, rhs) ||
      lhs->row_count != rhs.row_count || lhs->source_stride_bytes != rhs.source_stride_bytes ||
      lhs->target_stride_bytes != rhs.target_stride_bytes ||
      !checked_add(lhs->source_offset, lhs->row_bytes, &lhs_source_row_end) ||
      !checked_add(lhs->target_offset, lhs->row_bytes, &lhs_target_row_end) ||
      lhs_source_row_end != rhs.source_offset || lhs_target_row_end != rhs.target_offset) {
    return false;
  }
  uint64_t merged_row_bytes = 0;
  uint64_t merged_length = 0;
  if (!checked_add(lhs->row_bytes, rhs.row_bytes, &merged_row_bytes) || merged_row_bytes > lhs->source_stride_bytes ||
      merged_row_bytes > lhs->target_stride_bytes || !checked_mul(lhs->row_count, merged_row_bytes, &merged_length)) {
    return false;
  }
  lhs->row_bytes = merged_row_bytes;
  lhs->length = merged_length;
  return true;
}

bool try_merge_vertical_2d_tensor_stage_op(TensorStageCopyOp* lhs, const TensorStageCopyOp& rhs) {
  uint64_t lhs_source_span = 0;
  uint64_t lhs_target_span = 0;
  uint64_t expected_rhs_source_offset = 0;
  uint64_t expected_rhs_target_offset = 0;
  if (lhs == nullptr || lhs->row_count <= 1 || rhs.row_count <= 1 || !can_merge_common_tensor_stage_prefix(*lhs, rhs) ||
      lhs->row_bytes != rhs.row_bytes || lhs->source_stride_bytes != rhs.source_stride_bytes ||
      lhs->target_stride_bytes != rhs.target_stride_bytes ||
      !checked_mul(lhs->row_count, lhs->source_stride_bytes, &lhs_source_span) ||
      !checked_mul(lhs->row_count, lhs->target_stride_bytes, &lhs_target_span) ||
      !checked_add(lhs->source_offset, lhs_source_span, &expected_rhs_source_offset) ||
      !checked_add(lhs->target_offset, lhs_target_span, &expected_rhs_target_offset) ||
      expected_rhs_source_offset != rhs.source_offset || expected_rhs_target_offset != rhs.target_offset) {
    return false;
  }
  uint64_t merged_row_count = 0;
  uint64_t merged_length = 0;
  if (!checked_add(lhs->row_count, rhs.row_count, &merged_row_count) ||
      !checked_mul(merged_row_count, lhs->row_bytes, &merged_length)) {
    return false;
  }
  lhs->row_count = merged_row_count;
  lhs->length = merged_length;
  return true;
}

std::vector<TensorStageCopyOp> coalesce_tensor_stage_copy_ops(std::vector<TensorStageCopyOp> ops) {
  if (ops.size() < 2) {
    return ops;
  }
  std::sort(ops.begin(), ops.end(), [](const TensorStageCopyOp& lhs, const TensorStageCopyOp& rhs) {
    return std::tie(
               lhs.rank,
               lhs.dst_name,
               lhs.source_name,
               lhs.storage_index,
               lhs.source_index,
               lhs.source_offset,
               lhs.target_offset,
               lhs.row_count,
               lhs.source_stride_bytes,
               lhs.target_stride_bytes) <
        std::tie(
               rhs.rank,
               rhs.dst_name,
               rhs.source_name,
               rhs.storage_index,
               rhs.source_index,
               rhs.source_offset,
               rhs.target_offset,
               rhs.row_count,
               rhs.source_stride_bytes,
               rhs.target_stride_bytes);
  });
  std::vector<TensorStageCopyOp> merged;
  merged.reserve(ops.size());
  for (const auto& op : ops) {
    if (!merged.empty() &&
        (try_merge_linear_tensor_stage_op(&merged.back(), op) ||
         try_merge_horizontal_2d_tensor_stage_op(&merged.back(), op) ||
         try_merge_vertical_2d_tensor_stage_op(&merged.back(), op))) {
      continue;
    }
    merged.push_back(op);
  }
  return merged;
}

absl::Status add_summary_bytes(uint64_t value, uint64_t* total, std::string_view field_name) {
  if (total == nullptr || !checked_add(*total, value, total)) {
    return absl::OutOfRangeError(absl::StrCat("source-window tensor-stage summary ", field_name, " overflows"));
  }
  return absl::OkStatus();
}

absl::Status append_tensor_stage_storage_split_ops(
    const SourceWindowCollectiveMemberInput& member,
    absl::Span<const TargetStorageSpan> storage_spans,
    std::string_view source_name,
    std::string_view dst_name,
    uint64_t source_offset,
    uint64_t target_offset,
    uint64_t length,
    std::vector<TensorStageCopyOp>* ops) {
  if (ops == nullptr) {
    return absl::InternalError("source-window tensor-stage op output is null");
  }
  uint64_t remaining = length;
  uint64_t src_cursor = source_offset;
  uint64_t dst_cursor = target_offset;
  while (remaining > 0) {
    std::optional<uint32_t> storage_index;
    uint64_t storage_end = 0;
    for (uint32_t index = 0; index < storage_spans.size(); ++index) {
      const auto& storage = storage_spans[index];
      storage_end = storage.base_offset + storage.length;
      if (dst_cursor >= storage.base_offset && dst_cursor < storage_end) {
        storage_index = index;
        break;
      }
    }
    if (!storage_index.has_value()) {
      return absl::InvalidArgumentError(
          absl::StrCat("source-window tensor-stage target span is outside target storage: rank=", member.rank));
    }
    const uint64_t chunk = std::min<uint64_t>(remaining, storage_end - dst_cursor);
    ops->push_back(
        TensorStageCopyOp{
            .rank = member.rank,
            .source_index = 0,
            .storage_index = *storage_index,
            .dst_name = std::string(dst_name),
            .source_name = std::string(source_name),
            .source_offset = src_cursor,
            .target_offset = dst_cursor,
            .length = chunk,
            .row_count = 1,
            .row_bytes = chunk,
            .source_stride_bytes = chunk,
            .target_stride_bytes = chunk,
        });
    remaining -= chunk;
    src_cursor += chunk;
    dst_cursor += chunk;
  }
  return absl::OkStatus();
}

absl::Status append_tensor_stage_paired_ops(
    const SourceWindowCollectiveMemberInput& member,
    absl::Span<const TargetStorageSpan> storage_spans,
    std::string_view source_name,
    std::string_view dst_name,
    absl::Span<const contracts::TensorByteSpan> source_spans,
    absl::Span<const contracts::TensorByteSpan> destination_spans,
    std::vector<TensorStageCopyOp>* ops) {
  size_t source_index = 0;
  size_t destination_index = 0;
  uint64_t source_consumed = 0;
  uint64_t destination_consumed = 0;
  while (source_index < source_spans.size() && destination_index < destination_spans.size()) {
    const auto& source = source_spans[source_index];
    const auto& destination = destination_spans[destination_index];
    const uint64_t source_remaining = source.length - source_consumed;
    const uint64_t destination_remaining = destination.length - destination_consumed;
    const uint64_t chunk = std::min(source_remaining, destination_remaining);
    if (chunk > 0) {
      auto status = append_tensor_stage_storage_split_ops(
          member,
          storage_spans,
          source_name,
          dst_name,
          source.offset + source_consumed,
          destination.offset + destination_consumed,
          chunk,
          ops);
      if (!status.ok()) {
        return status;
      }
    }
    source_consumed += chunk;
    destination_consumed += chunk;
    if (source_consumed == source.length) {
      source_index += 1;
      source_consumed = 0;
    }
    if (destination_consumed == destination.length) {
      destination_index += 1;
      destination_consumed = 0;
    }
  }
  if (source_index != source_spans.size() || destination_index != destination_spans.size()) {
    return absl::InvalidArgumentError("source-window tensor-stage source/destination spans differ in total size");
  }
  return absl::OkStatus();
}

absl::Status append_tensor_stage_compressed_2d_op(
    const SourceWindowCollectiveMemberInput& member,
    absl::Span<const TargetStorageSpan> storage_spans,
    const contracts::RepresentationWorkSourceFragment& source,
    const contracts::RepresentationWorkItem& item,
    std::vector<TensorStageCopyOp>* ops) {
  const auto& source_spec = source.fragment.source_spec;
  const auto& dst_spec = item.dst_spec;
  if (source_spec.shape.size() != 2 || (dst_spec.shape.size() != 2 && dst_spec.shape.size() != 3) ||
      source_spec.element_size == 0 || dst_spec.element_size == 0 ||
      source_spec.element_size != dst_spec.element_size || !is_row_major_contiguous(source_spec) ||
      !is_row_major_contiguous(dst_spec) || source_spec.shape[1] <= 0) {
    return absl::FailedPreconditionError("source-window tensor-stage compressed requires row-major 2d source");
  }
  const auto src_row_axis = axis_range_for_dim(source.fragment.source_range, 0);
  const auto src_col_axis = axis_range_for_dim(source.fragment.source_range, 1);
  const int64_t row_start = src_row_axis.has_value() ? src_row_axis->start : 0;
  const int64_t row_end = src_row_axis.has_value() ? src_row_axis->end : source_spec.shape[0];
  const int64_t col_start = src_col_axis.has_value() ? src_col_axis->start : 0;
  const int64_t col_end = src_col_axis.has_value() ? src_col_axis->end : source_spec.shape[1];
  if (row_start < 0 || row_end <= row_start || row_end > source_spec.shape[0] || col_start < 0 ||
      col_end <= col_start || col_end > source_spec.shape[1]) {
    return absl::FailedPreconditionError("source-window tensor-stage compressed source range mismatch");
  }
  auto target_rows_or = resolve_compressed_target_rows(dst_spec, source.fragment.destination_range);
  if (!target_rows_or.ok()) {
    return target_rows_or.status();
  }
  const auto& target_rows = *target_rows_or;
  const uint64_t element_bytes = source_spec.element_size;
  const uint64_t row_count = static_cast<uint64_t>(row_end - row_start);
  const uint64_t source_stride_bytes = static_cast<uint64_t>(source_spec.shape[1]) * element_bytes;
  const uint64_t row_bytes = static_cast<uint64_t>(col_end - col_start) * element_bytes;
  if (row_count != target_rows.row_count || row_bytes != target_rows.row_bytes) {
    return absl::FailedPreconditionError("source-window tensor-stage compressed destination width mismatch");
  }
  const uint64_t source_offset = source_spec.logical_offset + static_cast<uint64_t>(row_start) * source_stride_bytes +
      static_cast<uint64_t>(col_start) * element_bytes;
  const uint64_t target_offset = target_rows.target_offset;
  const uint64_t target_end = target_offset + (row_count - 1) * target_rows.target_stride_bytes + row_bytes;
  const auto storage_index = storage_index_for_target_offset(storage_spans, target_offset, target_end);
  if (!storage_index.has_value()) {
    return absl::FailedPreconditionError("source-window tensor-stage compressed target crosses storage boundary");
  }
  uint64_t length = 0;
  if (!checked_mul(row_count, row_bytes, &length)) {
    return absl::OutOfRangeError("source-window tensor-stage compressed length overflows");
  }
  ops->push_back(
      TensorStageCopyOp{
          .rank = member.rank,
          .source_index = 0,
          .storage_index = *storage_index,
          .dst_name = item.dst_name,
          .source_name = source_spec.name,
          .source_offset = source_offset,
          .target_offset = target_offset,
          .length = length,
          .row_count = row_count,
          .row_bytes = row_bytes,
          .source_stride_bytes = source_stride_bytes,
          .target_stride_bytes = target_rows.target_stride_bytes,
      });
  return absl::OkStatus();
}

absl::Status append_tensor_stage_work_item_ops(
    const SourceWindowCollectiveMemberInput& member,
    absl::Span<const TargetStorageSpan> storage_spans,
    const RepresentationWorkItem& item,
    std::vector<TensorStageCopyOp>* ops,
    SourceWindowTensorStagedCopySummary* summary,
    std::map<std::string, bool>* source_tensors,
    std::map<std::string, bool>* destination_tensors) {
  if (source_window_local_typed_kind(item.kind)) {
    return absl::OkStatus();
  }
  if (!source_window_copy_kind(item.kind) || item.sources.empty()) {
    auto status = add_summary_bytes(item.committed_bytes, &summary->ineligible_bytes, "ineligible_bytes");
    return status;
  }
  (*destination_tensors)[item.dst_name] = true;
  for (const auto& source : item.sources) {
    summary->source_fragment_count += 1;
    (*source_tensors)[source.fragment.source_spec.name] = true;
    if (source.prefix_count != 1) {
      auto status = add_summary_bytes(item.committed_bytes, &summary->ineligible_bytes, "ineligible_bytes");
      return status;
    }

    const size_t before = ops->size();
    auto compressed_status = append_tensor_stage_compressed_2d_op(member, storage_spans, source, item, ops);
    if (compressed_status.ok()) {
      auto status = add_summary_bytes(ops->back().length, &summary->eligible_bytes, "eligible_bytes");
      if (!status.ok()) {
        return status;
      }
      continue;
    }
    if (coordinate_expansion_is_too_large(
            source.fragment.source_spec,
            source.fragment.source_range,
            item.dst_spec,
            source.fragment.destination_range)) {
      auto status = add_summary_bytes(item.committed_bytes, &summary->ineligible_bytes, "ineligible_bytes");
      return status;
    }
    auto source_spans_or =
        contracts::build_coordinate_byte_spans(source.fragment.source_spec, source.fragment.source_range);
    if (!source_spans_or.ok()) {
      return source_spans_or.status();
    }
    auto destination_spans_or =
        contracts::build_coordinate_byte_spans(item.dst_spec, source.fragment.destination_range);
    if (!destination_spans_or.ok()) {
      return destination_spans_or.status();
    }
    auto status = append_tensor_stage_paired_ops(
        member,
        storage_spans,
        source.fragment.source_spec.name,
        item.dst_name,
        *source_spans_or,
        *destination_spans_or,
        ops);
    if (!status.ok()) {
      return status;
    }
    for (size_t index = before; index < ops->size(); ++index) {
      status = add_summary_bytes((*ops)[index].length, &summary->eligible_bytes, "eligible_bytes");
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

std::vector<SourceWindowCollectiveWindow> build_windows_from_spans(
    absl::Span<const CandidateSpan> raw_spans,
    const SourceWindowCollectiveConfig& config) {
  std::vector<CandidateSpan> spans = split_spans_by_window_cap(raw_spans, config.window_bytes);
  std::sort(spans.begin(), spans.end(), [](const CandidateSpan& lhs, const CandidateSpan& rhs) {
    if (lhs.source_index != rhs.source_index) {
      return lhs.source_index < rhs.source_index;
    }
    if (lhs.source_window_start != rhs.source_window_start) {
      return lhs.source_window_start < rhs.source_window_start;
    }
    if (lhs.source_window_end != rhs.source_window_end) {
      return lhs.source_window_end < rhs.source_window_end;
    }
    return lhs.rank < rhs.rank;
  });

  std::vector<SourceWindowCollectiveWindow> windows;
  SourceWindowCollectiveWindow current;
  bool has_current = false;

  auto flush_current = [&]() {
    if (!has_current) {
      return;
    }
    current.consumer_spans = coalesce_consumer_spans_for_scatter(std::move(current.consumer_spans));
    windows.push_back(std::move(current));
    current = SourceWindowCollectiveWindow{};
    has_current = false;
  };
  auto append_group_consumer_spans = [&](absl::Span<const CandidateSpan> group_spans) {
    const size_t needed = current.consumer_spans.size() + group_spans.size();
    if (needed > current.consumer_spans.capacity()) {
      const size_t doubled = std::max<size_t>(current.consumer_spans.capacity() * 2, 16);
      current.consumer_spans.reserve(std::max(needed, doubled));
    }
    for (const auto& span : group_spans) {
      current.consumer_spans.push_back(to_consumer_span(span));
    }
  };

  size_t group_begin = 0;
  while (group_begin < spans.size()) {
    const auto& first = spans[group_begin];
    const uint32_t group_source_index = first.source_index;
    const uint64_t group_start = first.source_window_start;
    const uint64_t group_end_offset = first.source_window_end;
    size_t group_end = group_begin + 1;
    while (group_end < spans.size() && spans[group_end].source_index == group_source_index &&
           spans[group_end].source_window_start == group_start &&
           spans[group_end].source_window_end == group_end_offset) {
      ++group_end;
    }
    const auto group_spans = absl::MakeConstSpan(spans).subspan(group_begin, group_end - group_begin);
    group_begin = group_end;
    if (group_end_offset <= group_start) {
      continue;
    }
    const uint64_t group_unique_payload_bytes = unique_payload_bytes_for_same_window_group(group_spans);
    if (!has_current) {
      has_current = true;
      current.source_index = group_source_index;
      current.start = group_start;
      current.end = group_end_offset;
      current.unique_payload_bytes = group_unique_payload_bytes;
      append_group_consumer_spans(group_spans);
      continue;
    }

    const uint64_t gap = group_start > current.end ? group_start - current.end : 0;
    const uint64_t candidate_end = std::max(current.end, group_end_offset);
    const uint64_t candidate_payload = current.unique_payload_bytes + group_unique_payload_bytes;
    const uint64_t candidate_window_bytes = candidate_end - current.start;
    const bool source_matches = group_source_index == current.source_index;
    const bool gap_ok = gap <= config.max_gap_bytes;
    const bool amp_ok =
        amplification_within_limit(candidate_window_bytes, candidate_payload, config.max_window_amplification_x1000);
    const bool cap_ok = config.window_bytes == 0 || candidate_window_bytes <= config.window_bytes;
    if (!source_matches || !gap_ok || !amp_ok || !cap_ok) {
      flush_current();
      has_current = true;
      current.source_index = group_source_index;
      current.start = group_start;
      current.end = group_end_offset;
      current.unique_payload_bytes = group_unique_payload_bytes;
    } else {
      current.end = candidate_end;
      current.unique_payload_bytes = candidate_payload;
    }
    append_group_consumer_spans(group_spans);
  }
  flush_current();
  return windows;
}

std::vector<uint32_t> sorted_member_ranks(const SourceWindowCollectiveGroupInput& input) {
  std::vector<uint32_t> ranks;
  ranks.reserve(input.members.size());
  for (const auto& member : input.members) {
    ranks.push_back(member.rank);
  }
  std::sort(ranks.begin(), ranks.end());
  return ranks;
}

void assign_window_owners(const std::vector<uint32_t>& ranks, std::vector<SourceWindowCollectiveWindow>* windows) {
  if (windows == nullptr || ranks.empty()) {
    return;
  }
  for (size_t index = 0; index < windows->size(); ++index) {
    auto& window = (*windows)[index];
    if (window.distribution_mode == SourceWindowDistributionMode::kLocalOnly && !window.consumer_spans.empty()) {
      window.owner_rank = window.consumer_spans.front().rank;
    } else {
      window.owner_rank = ranks[index % ranks.size()];
    }
  }
}

absl::Status validate_local_only_windows(absl::Span<const SourceWindowCollectiveWindow> windows) {
  for (const auto& window : windows) {
    if (window.distribution_mode != SourceWindowDistributionMode::kLocalOnly) {
      continue;
    }
    for (const auto& span : window.consumer_spans) {
      if (span.rank != window.owner_rank) {
        return absl::FailedPreconditionError("local_only_window_has_remote_consumers");
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> compute_plan_hash(
    const SourceWindowCollectiveGroupInput& input,
    const SourceWindowCollectivePlan& plan) {
  size_t consumer_span_count = 0;
  for (const auto& window : plan.windows) {
    consumer_span_count += window.consumer_spans.size();
  }

  std::string payload;
  payload.reserve(
      512 + input.source_index_digest.size() + input.members.size() * 128 + plan.rank_read_bytes.size() * 32 +
      plan.windows.size() * 96 + consumer_span_count * 128);
  absl::StrAppend(
      &payload,
      "source_window_collective_plan_v2|world=",
      input.group.world_size,
      "|source_index_digest=",
      input.source_index_digest,
      "|distribution=",
      strategy::source_window_collective_distribution_mode_name(plan.distribution_mode),
      "|");
  for (const auto& member : input.members) {
    const auto& target_layout = member_target_layout(member);
    absl::StrAppend(&payload, "member=", member.rank, "/", member.device_id, "/total=", target_layout.total_size);
    for (const auto& storage : target_layout.storages) {
      absl::StrAppend(&payload, "/storage=", storage.length);
    }
    absl::StrAppend(&payload, "|");
  }
  absl::StrAppend(
      &payload,
      "summary=",
      plan.residual_bytes,
      ":",
      plan.summary.source_window_group_disk_read_bytes,
      ":",
      plan.summary.source_window_rank_read_bytes_max,
      ":",
      plan.summary.source_window_local_rank_read_bytes_max,
      ":",
      plan.summary.source_window_unique_payload_bytes,
      ":",
      plan.summary.source_window_target_write_bytes,
      ":",
      plan.summary.source_window_peer_transfer_bytes,
      ":",
      plan.summary.source_window_peer_useful_bytes,
      ":",
      plan.summary.source_window_peer_waste_bytes,
      ":",
      plan.summary.source_window_read_amplification_x1000,
      ":",
      plan.summary.source_window_scatter_op_count,
      ":",
      plan.summary.source_window_window_count,
      "|rank_reads=");
  for (const auto rank_read_bytes : plan.rank_read_bytes) {
    absl::StrAppend(&payload, rank_read_bytes, ",");
  }
  absl::StrAppend(&payload, "|");
  for (const auto& window : plan.windows) {
    absl::StrAppend(
        &payload,
        "window=",
        window.source_index,
        ":",
        strategy::source_window_collective_distribution_mode_name(window.distribution_mode),
        ":",
        window.owner_rank,
        ":",
        window.start,
        ":",
        window.end,
        ":",
        window.unique_payload_bytes);
    for (const auto& span : window.consumer_spans) {
      absl::StrAppend(
          &payload,
          "/consumer=",
          span.rank,
          ":",
          span.storage_index,
          ":",
          span.source_window_start,
          ":",
          span.source_window_end,
          ":",
          span.source_offset,
          ":",
          span.target_offset,
          ":",
          span.length,
          ":",
          span.row_count,
          ":",
          span.row_bytes,
          ":",
          span.source_stride_bytes,
          ":",
          span.target_stride_bytes);
    }
    absl::StrAppend(&payload, "|");
  }
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return common::multibase_multihash_sha256(digest);
}

void append_source_window_config_identity(std::string* payload, const SourceWindowCollectiveConfig& config) {
  absl::StrAppend(
      payload,
      "{enabled=",
      config.enabled,
      ",selection=",
      strategy::source_window_collective_selection_mode_name(config.selection_mode),
      ",window=",
      config.window_bytes,
      ",max_gap=",
      config.max_gap_bytes,
      ",max_window_amp=",
      config.max_window_amplification_x1000,
      ",max_plan_read_amp=",
      config.max_plan_read_amplification_x1000,
      ",max_scatter_ops=",
      config.max_scatter_ops_per_window,
      ",peak=",
      config.peak_bytes_budget,
      ",min_rank_saving=",
      config.min_rank_read_saving_bytes,
      ",max_peer_ratio=",
      config.max_peer_to_read_ratio_x1000,
      ",min_routed_peer_saving=",
      config.min_routed_peer_saving_bytes,
      ",distribution=",
      strategy::source_window_collective_distribution_mode_name(config.distribution_mode),
      ",allow_mixed_residual=",
      config.allow_mixed_residual,
      "}");
}

std::optional<std::string> compute_prepared_identity_plan_hash(
    const SourceWindowCollectiveGroupInput& input,
    const SourceWindowCollectivePlan& plan) {
  if (input.members.empty()) {
    return std::nullopt;
  }

  struct PreparedMemberIdentity {
    uint32_t rank{0};
    int device_id{-1};
    std::string group_key;
    std::string member_key;
    std::string realization_plan_hash;
    std::string target_layout_template_hash;
    std::string target_index_hash;
    uint64_t target_total_size{0};
    std::vector<uint64_t> storage_lengths;
    std::vector<std::pair<uint64_t, uint64_t>> storage_spans;
  };

  std::vector<PreparedMemberIdentity> members;
  members.reserve(input.members.size());
  for (const auto& member : input.members) {
    if (!member.prepared_realization.has_value()) {
      return std::nullopt;
    }
    const auto& facts = *member.prepared_realization;
    if (facts.group_key.empty() || facts.member_key.empty() || facts.realization_plan_hash.empty() ||
        facts.target_layout_template_hash.empty() || facts.target_index_hash.empty()) {
      return std::nullopt;
    }

    const auto& target_layout = member_target_layout(member);
    PreparedMemberIdentity identity{
        .rank = member.rank,
        .device_id = member.device_id,
        .group_key = facts.group_key,
        .member_key = facts.member_key,
        .realization_plan_hash = facts.realization_plan_hash,
        .target_layout_template_hash = facts.target_layout_template_hash,
        .target_index_hash = facts.target_index_hash,
        .target_total_size = target_layout.total_size,
    };
    identity.storage_lengths.reserve(target_layout.storages.size());
    for (const auto& storage : target_layout.storages) {
      identity.storage_lengths.push_back(storage.length);
    }
    identity.storage_spans.reserve(member.storage_spans.size());
    for (const auto& storage_span : member.storage_spans) {
      identity.storage_spans.emplace_back(storage_span.base_offset, storage_span.length);
    }
    members.push_back(std::move(identity));
  }

  std::sort(members.begin(), members.end(), [](const PreparedMemberIdentity& lhs, const PreparedMemberIdentity& rhs) {
    return std::tie(lhs.rank, lhs.device_id, lhs.member_key) < std::tie(rhs.rank, rhs.device_id, rhs.member_key);
  });

  std::string payload;
  payload.reserve(1024 + members.size() * 512);
  absl::StrAppend(&payload, "source_window_collective_prepared_plan_hash_v1|artifact_path=");
  if (input.disk_context != nullptr) {
    absl::StrAppend(&payload, input.disk_context->artifact_path().generic_string());
  }
  absl::StrAppend(
      &payload,
      "|source_index_digest=",
      input.source_index_digest,
      "|world_size=",
      input.group.world_size,
      "|resolved_distribution=",
      strategy::source_window_collective_distribution_mode_name(plan.distribution_mode),
      "|config=");
  append_source_window_config_identity(&payload, input.config);
  absl::StrAppend(
      &payload,
      "|summary=",
      plan.residual_bytes,
      ":",
      plan.summary.source_window_group_disk_read_bytes,
      ":",
      plan.summary.source_window_rank_read_bytes_max,
      ":",
      plan.summary.source_window_local_rank_read_bytes_max,
      ":",
      plan.summary.source_window_unique_payload_bytes,
      ":",
      plan.summary.source_window_target_write_bytes,
      ":",
      plan.summary.source_window_peer_transfer_bytes,
      ":",
      plan.summary.source_window_peer_useful_bytes,
      ":",
      plan.summary.source_window_peer_waste_bytes,
      ":",
      plan.summary.source_window_read_amplification_x1000,
      ":",
      plan.summary.source_window_scatter_op_count,
      ":",
      plan.summary.source_window_window_count,
      "|members=",
      members.size());
  for (const auto& member : members) {
    absl::StrAppend(
        &payload,
        "|member=",
        member.rank,
        ",device=",
        member.device_id,
        ",group_key=",
        member.group_key,
        ",member_key=",
        member.member_key,
        ",realization_plan_hash=",
        member.realization_plan_hash,
        ",target_layout_template_hash=",
        member.target_layout_template_hash,
        ",target_index_hash=",
        member.target_index_hash,
        ",target_total=",
        member.target_total_size,
        ",storages=",
        member.storage_lengths.size());
    for (const auto length : member.storage_lengths) {
      absl::StrAppend(&payload, "/storage_length=", length);
    }
    absl::StrAppend(&payload, ",storage_spans=", member.storage_spans.size());
    for (const auto& [base_offset, length] : member.storage_spans) {
      absl::StrAppend(&payload, "/span=", base_offset, ":", length);
    }
  }
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return common::multibase_multihash_sha256(digest);
}

} // namespace

SourceWindowCollectiveConfig source_window_collective_config_from_strategy(
    const StoreEngineOptions::MaterializationStrategyConfig& strategy_config) {
  return SourceWindowCollectiveConfig{
      .enabled = strategy_config.enable_source_window_collective,
      .selection_mode = selection_mode_from_strategy(strategy_config.source_window_collective_selection_mode),
      .window_bytes = strategy_config.source_window_collective_window_bytes,
      .max_gap_bytes = strategy_config.source_window_collective_max_gap_bytes,
      .max_window_amplification_x1000 = strategy_config.source_window_collective_max_window_amplification_x1000,
      .max_plan_read_amplification_x1000 = strategy_config.source_window_collective_max_plan_read_amplification_x1000,
      .max_scatter_ops_per_window = strategy_config.source_window_collective_max_scatter_ops_per_window,
      .peak_bytes_budget = strategy_config.source_window_collective_peak_bytes_budget,
      .min_rank_read_saving_bytes = strategy_config.source_window_collective_min_rank_read_saving_bytes,
      .max_peer_to_read_ratio_x1000 = strategy_config.source_window_collective_max_peer_to_read_ratio_x1000,
      .min_routed_peer_saving_bytes = strategy_config.source_window_collective_min_routed_peer_saving_bytes,
      .distribution_mode = distribution_mode_from_strategy(strategy_config.source_window_collective_distribution_mode),
      .allow_mixed_residual = strategy_config.source_window_collective_allow_mixed_residual,
  };
}

absl::StatusOr<SourceWindowCollectivePlan> build_source_window_collective_plan(
    const SourceWindowCollectiveGroupInput& input) {
  const auto total_start = std::chrono::steady_clock::now();
  auto validation_status = validate_group_input(input);
  if (!validation_status.ok()) {
    return validation_status;
  }

  const SourceWindowDistributionMode requested_distribution_mode = input.config.distribution_mode;
  std::vector<CandidateSpan> spans;
  uint64_t residual_bytes = 0;
  SourceWindowPlanBuildStats stats;
  const auto append_start = std::chrono::steady_clock::now();
  for (const auto& member : input.members) {
    const auto member_start = std::chrono::steady_clock::now();
    std::vector<TargetStorageSpan> fallback_storage_spans;
    absl::Span<const TargetStorageSpan> member_storage_spans = absl::MakeConstSpan(member.storage_spans);
    if (member_storage_spans.empty()) {
      fallback_storage_spans = storage_spans_for_member(member);
      member_storage_spans = absl::MakeConstSpan(fallback_storage_spans);
    }
    const size_t spans_before = spans.size();
    const uint64_t residual_before = residual_bytes;
    const uint64_t work_items_before = stats.work_items;
    const uint64_t compressed_before = stats.compressed_2d_spans;
    const uint64_t expanded_before = stats.expanded_work_items;
    const uint64_t guarded_before = stats.guarded_residual_work_items;
    const auto& work_plan = member_work_plan(member);
    residual_bytes += work_plan.residual_fallback_map.total_bytes;
    for (const auto& item : work_plan.items) {
      auto status = append_work_item_spans(member, member_storage_spans, item, &residual_bytes, &spans, &stats);
      if (!status.ok()) {
        return status;
      }
    }
    const double member_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - member_start).count();
    LOG(INFO) << "tc_profile source_window_collective_plan_builder member"
              << " group_id=" << input.group.group_id << " rank=" << member.rank
              << " work_items=" << (stats.work_items - work_items_before)
              << " compressed_2d_spans=" << (stats.compressed_2d_spans - compressed_before)
              << " expanded_work_items=" << (stats.expanded_work_items - expanded_before)
              << " guarded_residual_work_items=" << (stats.guarded_residual_work_items - guarded_before)
              << " spans_added=" << (spans.size() - spans_before)
              << " residual_added=" << (residual_bytes - residual_before) << " sec=" << member_sec;
  }
  const double append_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - append_start).count();
  if (spans.empty()) {
    return absl::FailedPreconditionError("no_source_window_spans");
  }
  if (residual_bytes > 0 && !input.config.allow_mixed_residual) {
    return absl::FailedPreconditionError("residual_requires_mixed_source_window");
  }

  SourceWindowCollectivePlan plan;
  plan.group = input.group;
  plan.residual_bytes = residual_bytes;
  const auto windows_start = std::chrono::steady_clock::now();
  plan.windows = build_windows_from_spans(spans, input.config);
  const double windows_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - windows_start).count();
  if (plan.windows.empty()) {
    return absl::FailedPreconditionError("no_source_window_windows");
  }
  const SourceWindowDistributionMode distribution_mode = resolve_distribution_mode_for_windows(
      requested_distribution_mode, absl::MakeSpan(plan.windows), input.group.world_size, input.config);
  plan.distribution_mode = distribution_mode;
  assign_window_distribution_modes(distribution_mode, input.group.world_size, input.config, &plan.windows);
  assign_window_owners(sorted_member_ranks(input), &plan.windows);
  auto local_only_status = validate_local_only_windows(absl::MakeSpan(plan.windows));
  if (!local_only_status.ok()) {
    return local_only_status;
  }

  const auto metrics_start = std::chrono::steady_clock::now();
  plan.rank_read_bytes.assign(input.group.world_size, 0);
  uint64_t group_disk_read_bytes = 0;
  uint64_t target_write_bytes = 0;
  uint64_t peer_transfer_bytes = 0;
  uint64_t peer_useful_bytes = 0;
  uint64_t max_window_bytes = 0;
  uint64_t max_scatter_ops_per_window = 0;
  uint64_t scatter_op_count = 0;
  uint64_t full_window_all_gather_windows = 0;
  uint64_t consumer_routed_windows = 0;
  uint64_t local_only_windows = 0;
  LocalMappedReadIntervalMap local_read_intervals_by_rank_source;
  for (const auto& window : plan.windows) {
    const auto window_distribution_mode = window.distribution_mode;
    switch (window_distribution_mode) {
      case SourceWindowDistributionMode::kFullWindowAllGather:
        full_window_all_gather_windows += 1;
        break;
      case SourceWindowDistributionMode::kConsumerRouted:
        consumer_routed_windows += 1;
        break;
      case SourceWindowDistributionMode::kLocalOnly:
        local_only_windows += 1;
        break;
      case SourceWindowDistributionMode::kAuto:
      case SourceWindowDistributionMode::kHybridWindow:
        break;
    }
    const uint64_t window_bytes = window.end - window.start;
    group_disk_read_bytes += window_bytes;
    max_window_bytes = std::max(max_window_bytes, window_bytes);
    max_scatter_ops_per_window = std::max<uint64_t>(max_scatter_ops_per_window, window.consumer_spans.size());
    scatter_op_count += window.consumer_spans.size();
    if (window.owner_rank >= plan.rank_read_bytes.size()) {
      return absl::InvalidArgumentError("source-window owner rank out of bounds");
    }
    plan.rank_read_bytes[window.owner_rank] += window_bytes;
    if (window_distribution_mode == SourceWindowDistributionMode::kFullWindowAllGather) {
      uint64_t window_peer_bytes = 0;
      if (!checked_mul(window_bytes, input.group.world_size - 1, &window_peer_bytes)) {
        return absl::OutOfRangeError("source-window peer bytes overflow");
      }
      peer_transfer_bytes += window_peer_bytes;
    }
    for (const auto& span : window.consumer_spans) {
      append_local_mapped_read_interval(&local_read_intervals_by_rank_source, span, input.group.world_size);
      const uint64_t useful_bytes = consumer_span_useful_bytes(span);
      target_write_bytes += useful_bytes;
      if (window_distribution_mode == SourceWindowDistributionMode::kFullWindowAllGather &&
          span.rank != window.owner_rank) {
        peer_useful_bytes += useful_bytes;
      }
      if (window_distribution_mode == SourceWindowDistributionMode::kConsumerRouted) {
        uint64_t routed_peer_bytes = 0;
        if (!checked_mul(useful_bytes, input.group.world_size - 1, &routed_peer_bytes)) {
          return absl::OutOfRangeError("source-window routed peer bytes overflow");
        }
        routed_peer_bytes /= input.group.world_size;
        peer_transfer_bytes += routed_peer_bytes;
        peer_useful_bytes += routed_peer_bytes;
      }
    }
  }

  const uint64_t unique_payload_bytes = unique_payload_bytes_for_spans(absl::MakeSpan(spans));
  const std::vector<uint64_t> local_rank_physical_read_bytes =
      finish_local_mapped_physical_read_bytes_by_rank(&local_read_intervals_by_rank_source, input.group.world_size);
  const uint64_t local_rank_read_max =
      *std::max_element(local_rank_physical_read_bytes.begin(), local_rank_physical_read_bytes.end());
  const uint64_t rank_read_max = *std::max_element(plan.rank_read_bytes.begin(), plan.rank_read_bytes.end());
  const uint64_t rank_read_saving = local_rank_read_max > rank_read_max ? local_rank_read_max - rank_read_max : 0;
  const uint64_t read_amplification_x1000 = ceil_ratio_x1000(group_disk_read_bytes, unique_payload_bytes);
  const uint64_t peer_to_read_ratio_x1000 = ceil_ratio_x1000(peer_transfer_bytes, group_disk_read_bytes);
  const uint64_t peer_waste_bytes =
      peer_transfer_bytes > peer_useful_bytes ? peer_transfer_bytes - peer_useful_bytes : 0;

  std::string reject_reason;
  if (input.config.peak_bytes_budget > 0 && max_window_bytes > input.config.peak_bytes_budget) {
    reject_reason = "peak_budget_exceeded";
  } else if (
      input.config.max_scatter_ops_per_window > 0 &&
      max_scatter_ops_per_window > input.config.max_scatter_ops_per_window) {
    reject_reason = "scatter_ops_per_window_exceeded";
  } else if (
      input.config.max_plan_read_amplification_x1000 > 0 &&
      read_amplification_x1000 > input.config.max_plan_read_amplification_x1000) {
    reject_reason = "plan_read_amplification_exceeded";
  } else if (
      input.config.max_peer_to_read_ratio_x1000 > 0 &&
      peer_to_read_ratio_x1000 > input.config.max_peer_to_read_ratio_x1000) {
    reject_reason = "peer_to_read_ratio_exceeded";
  } else if (rank_read_saving < input.config.min_rank_read_saving_bytes) {
    reject_reason = "rank_read_saving_below_threshold";
  }

  plan.summary = strategy::SourceWindowCollectiveCandidateSummary{
      .candidate = true,
      .group_final_admitted = reject_reason.empty(),
      .selection_mode = input.config.selection_mode,
      .distribution_mode = distribution_mode,
      .pre_admission_reason = "eligible",
      .group_reject_reason = reject_reason,
      .source_window_group_disk_read_bytes = group_disk_read_bytes,
      .source_window_rank_read_bytes_max = rank_read_max,
      .source_window_local_rank_read_bytes_max = local_rank_read_max,
      .source_window_rank_read_saving_bytes = rank_read_saving,
      .source_window_unique_payload_bytes = unique_payload_bytes,
      .source_window_target_write_bytes = target_write_bytes,
      .source_window_peer_transfer_bytes = peer_transfer_bytes,
      .source_window_peer_useful_bytes = peer_useful_bytes,
      .source_window_peer_waste_bytes = peer_waste_bytes,
      .source_window_read_amplification_x1000 = read_amplification_x1000,
      .source_window_scatter_op_count = scatter_op_count,
      .source_window_window_count = plan.windows.size(),
      .source_window_residual_bytes = residual_bytes,
  };
  const double metrics_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - metrics_start).count();
  const auto hash_start = std::chrono::steady_clock::now();
  const auto prepared_identity_hash = compute_prepared_identity_plan_hash(input, plan);
  bool prepared_identity_hash_used = prepared_identity_hash.has_value();
  if (prepared_identity_hash_used) {
    plan.plan_hash = *prepared_identity_hash;
  } else {
    auto hash_or = compute_plan_hash(input, plan);
    if (!hash_or.ok()) {
      return hash_or.status();
    }
    plan.plan_hash = *hash_or;
  }
  const double hash_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - hash_start).count();
  const double total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  LOG(INFO) << "tc_profile source_window_collective_plan_builder total"
            << " group_id=" << input.group.group_id << " members=" << input.members.size()
            << " work_items=" << stats.work_items << " compressed_2d_spans=" << stats.compressed_2d_spans
            << " expanded_work_items=" << stats.expanded_work_items << " expanded_spans=" << stats.expanded_spans
            << " local_typed_work_items=" << stats.local_typed_work_items
            << " local_typed_bytes=" << stats.local_typed_bytes << " residual_work_items=" << stats.residual_work_items
            << " guarded_residual_work_items=" << stats.guarded_residual_work_items
            << " residual_items_by_kind=" << residual_map_to_string(stats.residual_items_by_kind)
            << " residual_bytes_by_kind=" << residual_map_to_string(stats.residual_bytes_by_kind)
            << " residual_items_by_reason=" << residual_reason_map_to_string(stats.residual_items_by_reason)
            << " candidate_spans=" << spans.size() << " windows=" << plan.windows.size()
            << " full_window_all_gather_windows=" << full_window_all_gather_windows
            << " consumer_routed_windows=" << consumer_routed_windows << " local_only_windows=" << local_only_windows
            << " residual_bytes=" << residual_bytes << " group_disk_read_bytes=" << group_disk_read_bytes
            << " rank_read_bytes_max=" << rank_read_max << " local_rank_read_bytes_max=" << local_rank_read_max
            << " rank_read_saving_bytes=" << rank_read_saving
            << " read_amplification_x1000=" << read_amplification_x1000 << " append_sec=" << append_sec
            << " windows_sec=" << windows_sec << " metrics_sec=" << metrics_sec << " hash_sec=" << hash_sec
            << " prepared_identity_hash=" << (prepared_identity_hash_used ? 1 : 0) << " total_sec=" << total_sec;
  return plan;
}

strategy::SourceWindowCollectiveCandidateSummary summarize_source_window_collective(
    const SourceWindowCollectiveGroupInput& input) {
  auto plan_or = build_source_window_collective_plan(input);
  if (plan_or.ok()) {
    return plan_or->summary;
  }
  strategy::SourceWindowCollectiveCandidateSummary summary;
  summary.candidate = false;
  summary.group_final_admitted = false;
  summary.selection_mode = input.config.selection_mode;
  summary.distribution_mode = resolve_distribution_mode(input.config.distribution_mode);
  summary.pre_admission_reason = plan_or.status().message();
  summary.group_reject_reason = plan_or.status().message();
  return summary;
}

absl::StatusOr<SourceWindowTensorStagedCopySummary> summarize_source_window_tensor_staged_copy(
    const SourceWindowCollectiveGroupInput& input) {
  auto status = validate_group_input(input);
  if (!status.ok()) {
    return status;
  }

  SourceWindowTensorStagedCopySummary summary;
  summary.rank_count = input.members.size();
  std::vector<TensorStageCopyOp> raw_ops;
  std::map<std::string, bool> source_tensors;
  std::map<std::string, bool> destination_tensors;
  raw_ops.reserve(1024);
  for (const auto& member : input.members) {
    std::vector<TargetStorageSpan> fallback_storage_spans;
    absl::Span<const TargetStorageSpan> member_storage_spans = absl::MakeConstSpan(member.storage_spans);
    if (member_storage_spans.empty()) {
      fallback_storage_spans = storage_spans_for_member(member);
      member_storage_spans = absl::MakeConstSpan(fallback_storage_spans);
    }
    const auto& work_plan = member_work_plan(member);
    for (const auto& item : work_plan.items) {
      status = append_tensor_stage_work_item_ops(
          member, member_storage_spans, item, &raw_ops, &summary, &source_tensors, &destination_tensors);
      if (!status.ok()) {
        return status;
      }
    }
  }

  summary.raw_copy_ops = raw_ops.size();
  summary.destination_tensor_count = destination_tensors.size();
  summary.source_tensor_count = source_tensors.size();
  std::vector<TensorStageCopyOp> coalesced = coalesce_tensor_stage_copy_ops(std::move(raw_ops));
  summary.tensor_staged_copy_ops = coalesced.size();
  std::map<uint32_t, uint64_t> ops_by_rank;
  for (const auto& op : coalesced) {
    ops_by_rank[op.rank] += 1;
    if (is_linear_tensor_stage_op(op)) {
      summary.linear_copy_ops += 1;
    } else {
      summary.copy_2d_ops += 1;
    }
  }
  for (const auto& [_, count] : ops_by_rank) {
    summary.max_tensor_staged_copy_ops_per_rank =
        std::max<uint64_t>(summary.max_tensor_staged_copy_ops_per_rank, count);
  }
  if (summary.raw_copy_ops > 0 && summary.raw_copy_ops > summary.tensor_staged_copy_ops) {
    const uint64_t removed_ops = summary.raw_copy_ops - summary.tensor_staged_copy_ops;
    summary.estimated_op_reduction_x1000 = ceil_ratio_x1000(removed_ops, summary.raw_copy_ops);
  }
  summary.feasible = summary.raw_copy_ops > 0 && summary.ineligible_bytes == 0;
  if (!summary.feasible) {
    summary.reject_reason = summary.ineligible_bytes > 0 ? "ineligible_work_items" : "no_copy_ops";
  }
  LOG(INFO) << "tc_profile source_window_tensor_staged_copy_summary"
            << " group_id=" << input.group.group_id << " feasible=" << (summary.feasible ? 1 : 0)
            << " reject_reason=" << summary.reject_reason << " ranks=" << summary.rank_count
            << " source_fragments=" << summary.source_fragment_count
            << " destination_tensors=" << summary.destination_tensor_count
            << " source_tensors=" << summary.source_tensor_count << " eligible_bytes=" << summary.eligible_bytes
            << " ineligible_bytes=" << summary.ineligible_bytes << " raw_copy_ops=" << summary.raw_copy_ops
            << " tensor_staged_copy_ops=" << summary.tensor_staged_copy_ops
            << " linear_copy_ops=" << summary.linear_copy_ops << " copy_2d_ops=" << summary.copy_2d_ops
            << " max_ops_per_rank=" << summary.max_tensor_staged_copy_ops_per_rank
            << " estimated_op_reduction_x1000=" << summary.estimated_op_reduction_x1000;
  return summary;
}

absl::StatusOr<SourceWindowBatchedScatterSummary> summarize_source_window_batched_scatter(
    const SourceWindowCollectivePlan& plan,
    uint64_t runtime_chunk_bytes) {
  SourceWindowBatchedScatterSummary summary;
  summary.rank_count = plan.group.world_size;
  summary.window_count = plan.windows.size();
  summary.runtime_chunk_bytes = runtime_chunk_bytes;
  summary.target_write_bytes = plan.summary.source_window_target_write_bytes;
  if (plan.group.world_size == 0) {
    summary.reject_reason = "world_size_zero";
    return summary;
  }
  if (plan.windows.empty()) {
    summary.reject_reason = "no_windows";
    return summary;
  }

  const uint64_t world_size = plan.group.world_size;
  for (const auto& window : plan.windows) {
    switch (window.distribution_mode) {
      case SourceWindowDistributionMode::kFullWindowAllGather:
        summary.full_window_all_gather_windows += 1;
        break;
      case SourceWindowDistributionMode::kConsumerRouted:
        summary.consumer_routed_windows += 1;
        break;
      case SourceWindowDistributionMode::kLocalOnly:
        summary.local_only_windows += 1;
        break;
      case SourceWindowDistributionMode::kAuto:
      case SourceWindowDistributionMode::kHybridWindow:
        break;
    }
    const uint64_t window_bytes = window.end > window.start ? window.end - window.start : 0;
    if (window_bytes == 0) {
      continue;
    }
    summary.consumer_span_count += window.consumer_spans.size();
    uint64_t chunk_bytes = runtime_chunk_bytes == 0 ? window_bytes : runtime_chunk_bytes;
    if (window.distribution_mode != SourceWindowDistributionMode::kLocalOnly) {
      chunk_bytes = (chunk_bytes / world_size) * world_size;
    }
    if (chunk_bytes == 0) {
      summary.reject_reason = "runtime_chunk_too_small";
      return summary;
    }
    const uint64_t stripe_bytes =
        window.distribution_mode == SourceWindowDistributionMode::kConsumerRouted ? chunk_bytes / world_size : 0;
    if (window.distribution_mode == SourceWindowDistributionMode::kConsumerRouted && stripe_bytes == 0) {
      summary.reject_reason = "routed_stripe_too_small";
      return summary;
    }

    for (uint64_t chunk_start = window.start; chunk_start < window.end;) {
      const uint64_t chunk_end = std::min<uint64_t>(window.end, chunk_start + chunk_bytes);
      if (chunk_end <= chunk_start) {
        summary.reject_reason = "invalid_runtime_chunk";
        return summary;
      }
      summary.estimated_runtime_chunk_count += 1;
      std::map<uint32_t, uint64_t> scatter_descriptors_by_rank;
      std::map<std::pair<uint32_t, uint32_t>, uint64_t> pack_descriptors_by_pair;
      for (const auto& span : window.consumer_spans) {
        const uint64_t source_begin =
            span.source_window_end > span.source_window_start ? span.source_window_start : span.source_offset;
        const uint64_t source_end = span.source_window_end > span.source_window_start
            ? span.source_window_end
            : span.source_offset + span.length;
        const uint64_t overlap_begin = std::max<uint64_t>(source_begin, chunk_start);
        const uint64_t overlap_end = std::min<uint64_t>(source_end, chunk_end);
        if (overlap_end <= overlap_begin) {
          continue;
        }
        summary.estimated_current_scatter_launches += 1;
        if (span.row_count > 1 && span.row_bytes > 0) {
          summary.estimated_current_copy_2d_launches += 1;
        } else {
          summary.estimated_current_linear_copy_launches += 1;
        }
        scatter_descriptors_by_rank[span.rank] += 1;

        if (window.distribution_mode == SourceWindowDistributionMode::kConsumerRouted) {
          const uint64_t first_chunk_offset = overlap_begin - chunk_start;
          const uint64_t last_chunk_offset = overlap_end - 1 - chunk_start;
          uint64_t first_producer = first_chunk_offset / stripe_bytes;
          uint64_t last_producer = last_chunk_offset / stripe_bytes;
          first_producer = std::min<uint64_t>(first_producer, world_size - 1);
          last_producer = std::min<uint64_t>(last_producer, world_size - 1);
          for (uint64_t producer = first_producer; producer <= last_producer; ++producer) {
            if (producer == span.rank) {
              continue;
            }
            summary.estimated_current_pack_launches += 1;
            pack_descriptors_by_pair[{static_cast<uint32_t>(producer), span.rank}] += 1;
          }
        }
      }
      summary.batched_scatter_launches += scatter_descriptors_by_rank.size();
      for (const auto& [_, descriptors] : scatter_descriptors_by_rank) {
        summary.max_descriptors_per_batched_scatter =
            std::max<uint64_t>(summary.max_descriptors_per_batched_scatter, descriptors);
      }
      summary.batched_pack_launches += pack_descriptors_by_pair.size();
      for (const auto& [_, descriptors] : pack_descriptors_by_pair) {
        summary.max_descriptors_per_batched_pack =
            std::max<uint64_t>(summary.max_descriptors_per_batched_pack, descriptors);
      }
      chunk_start = chunk_end;
    }
  }

  summary.estimated_current_copy_launches =
      summary.estimated_current_scatter_launches + summary.estimated_current_pack_launches;
  summary.batched_total_copy_launches = summary.batched_scatter_launches + summary.batched_pack_launches;
  if (summary.estimated_current_copy_launches > 0 &&
      summary.estimated_current_copy_launches > summary.batched_total_copy_launches) {
    const uint64_t removed_launches = summary.estimated_current_copy_launches - summary.batched_total_copy_launches;
    summary.estimated_copy_launch_reduction_x1000 =
        ceil_ratio_x1000(removed_launches, summary.estimated_current_copy_launches);
  }
  summary.feasible = summary.estimated_current_copy_launches > 0;
  if (!summary.feasible && summary.reject_reason.empty()) {
    summary.reject_reason = "no_copy_launches";
  }
  LOG(INFO) << "tc_profile source_window_batched_scatter_summary"
            << " group_id=" << plan.group.group_id << " feasible=" << (summary.feasible ? 1 : 0)
            << " reject_reason=" << summary.reject_reason << " ranks=" << summary.rank_count
            << " windows=" << summary.window_count << " runtime_chunk_bytes=" << summary.runtime_chunk_bytes
            << " estimated_runtime_chunks=" << summary.estimated_runtime_chunk_count
            << " consumer_spans=" << summary.consumer_span_count
            << " full_window_all_gather_windows=" << summary.full_window_all_gather_windows
            << " consumer_routed_windows=" << summary.consumer_routed_windows
            << " local_only_windows=" << summary.local_only_windows
            << " target_write_bytes=" << summary.target_write_bytes
            << " estimated_current_copy_launches=" << summary.estimated_current_copy_launches
            << " estimated_current_scatter_launches=" << summary.estimated_current_scatter_launches
            << " estimated_current_pack_launches=" << summary.estimated_current_pack_launches
            << " estimated_current_linear_copy_launches=" << summary.estimated_current_linear_copy_launches
            << " estimated_current_copy_2d_launches=" << summary.estimated_current_copy_2d_launches
            << " batched_total_copy_launches=" << summary.batched_total_copy_launches
            << " batched_scatter_launches=" << summary.batched_scatter_launches
            << " batched_pack_launches=" << summary.batched_pack_launches
            << " max_descriptors_per_batched_scatter=" << summary.max_descriptors_per_batched_scatter
            << " max_descriptors_per_batched_pack=" << summary.max_descriptors_per_batched_pack
            << " estimated_copy_launch_reduction_x1000=" << summary.estimated_copy_launch_reduction_x1000;
  return summary;
}

} // namespace tensorcast::store::replica
