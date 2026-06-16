// Copyright (c) 2026, TensorCast Team.

#include "core/store/replica/source_window_collective_plan.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/contracts/representation_contract.h"
#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"
#include "gsl/pointers"

namespace tensorcast::store::replica {
namespace {

namespace contracts = materialization::contracts;
namespace strategy = runtime::ingestion::strategy;

contracts::RepresentationTensorSpec make_u8_spec(
    std::string name,
    uint64_t logical_offset,
    std::vector<int64_t> shape,
    std::vector<int64_t> stride) {
  uint64_t elements = 1;
  for (const auto dim : shape) {
    elements *= static_cast<uint64_t>(dim);
  }
  return contracts::RepresentationTensorSpec{
      .name = std::move(name),
      .shape = std::move(shape),
      .stride = std::move(stride),
      .dtype = "torch.uint8",
      .logical_offset = logical_offset,
      .logical_length = elements,
      .storage_offset = 0,
      .element_size = 1,
  };
}

contracts::TensorCoordinateSpec dim_range(int32_t dim, int64_t start, int64_t end) {
  return contracts::TensorCoordinateSpec{
      .axes = {contracts::TensorAxisRange{.dim = dim, .start = start, .end = end}},
  };
}

contracts::TensorCoordinateSpec rect_range(int64_t row_start, int64_t row_end, int64_t col_start, int64_t col_end) {
  return contracts::TensorCoordinateSpec{
      .axes =
          {
              contracts::TensorAxisRange{.dim = 0, .start = row_start, .end = row_end},
              contracts::TensorAxisRange{.dim = 1, .start = col_start, .end = col_end},
          },
  };
}

contracts::TensorCoordinateSpec expert_col_range(
    int64_t expert_start,
    int64_t expert_end,
    int64_t col_start,
    int64_t col_end) {
  return contracts::TensorCoordinateSpec{
      .axes =
          {
              contracts::TensorAxisRange{.dim = 0, .start = expert_start, .end = expert_end},
              contracts::TensorAxisRange{.dim = 2, .start = col_start, .end = col_end},
          },
  };
}

contracts::TensorCoordinateSpec expert_rect_range(
    int64_t expert_start,
    int64_t expert_end,
    int64_t row_start,
    int64_t row_end,
    int64_t col_start,
    int64_t col_end) {
  return contracts::TensorCoordinateSpec{
      .axes =
          {
              contracts::TensorAxisRange{.dim = 0, .start = expert_start, .end = expert_end},
              contracts::TensorAxisRange{.dim = 1, .start = row_start, .end = row_end},
              contracts::TensorAxisRange{.dim = 2, .start = col_start, .end = col_end},
          },
  };
}

contracts::RepresentationWorkItem make_copy_item(
    const contracts::RepresentationTensorSpec& source_spec,
    contracts::TensorCoordinateSpec source_range,
    const contracts::RepresentationTensorSpec& destination_spec,
    contracts::TensorCoordinateSpec destination_range,
    uint64_t committed_bytes) {
  contracts::RepresentationWorkItem item;
  item.kind = contracts::RepresentationWorkItemKind::kTensorCopy;
  item.partition_kind = contracts::WorkPartitionKind::kUnknown;
  item.dst_name = destination_spec.name;
  item.dst_spec = destination_spec;
  item.committed_bytes = committed_bytes;
  item.sources.push_back(
      contracts::RepresentationWorkSourceFragment{
          .fragment =
              contracts::SourceFragment{
                  .source_spec = source_spec,
                  .source_range = std::move(source_range),
                  .destination_range = std::move(destination_range),
              },
      });
  return item;
}

std::shared_ptr<loader::DiskArtifactContext> make_disk_context() {
  return std::make_shared<loader::DiskArtifactContext>(
      std::filesystem::path("/tmp/source_window_collective_plan_test"),
      std::vector<std::filesystem::path>{},
      std::vector<size_t>{},
      0,
      true,
      false,
      false,
      false,
      std::vector<loader::SharedSafetensorsSegment>{});
}

SourceWindowCollectiveConfig base_config() {
  SourceWindowCollectiveConfig config;
  config.enabled = true;
  config.selection_mode = strategy::SourceWindowCollectiveSelectionMode::kDryRun;
  config.window_bytes = 8;
  config.max_gap_bytes = 0;
  config.max_window_amplification_x1000 = 1000;
  config.max_plan_read_amplification_x1000 = 1000;
  config.max_scatter_ops_per_window = 16;
  config.peak_bytes_budget = 1024;
  config.min_rank_read_saving_bytes = 0;
  config.max_peer_to_read_ratio_x1000 = 2000;
  config.distribution_mode = strategy::SourceWindowCollectiveDistributionMode::kFullWindowAllGather;
  return config;
}

loading::IntoTargetLayout target_layout(void* ptr, uint64_t bytes) {
  return loading::IntoTargetLayout{
      .storages =
          {
              loading::IntoTargetStorage{
                  .base_ptr = gsl::not_null<void*>{ptr},
                  .length = bytes,
              },
          },
      .total_size = bytes,
  };
}

SourceWindowCollectiveGroupInput make_two_rank_dim1_group(SourceWindowCollectiveConfig config = base_config()) {
  static std::array<std::uint8_t, 16> rank0_target{};
  static std::array<std::uint8_t, 16> rank1_target{};
  const auto source = make_u8_spec("src", 0, {4, 8}, {8, 1});
  const auto destination = make_u8_spec("dst", 0, {4, 4}, {4, 1});
  contracts::RepresentationWorkPlan rank0_plan{
      .items = {make_copy_item(source, dim_range(1, 0, 4), destination, contracts::TensorCoordinateSpec{}, 16)},
      .committed_bytes = 16,
  };
  contracts::RepresentationWorkPlan rank1_plan{
      .items = {make_copy_item(source, dim_range(1, 4, 8), destination, contracts::TensorCoordinateSpec{}, 16)},
      .committed_bytes = 16,
  };
  return SourceWindowCollectiveGroupInput{
      .group = loading::CollectiveLoadGroupHint{.group_id = "tp-test", .world_size = 2, .rank = 0},
      .disk_context = make_disk_context(),
      .source_index_digest = "source-index-a",
      .members =
          {
              SourceWindowCollectiveMemberInput{
                  .rank = 0,
                  .device_id = 0,
                  .work_plan = rank0_plan,
                  .target_layout = target_layout(rank0_target.data(), rank0_target.size()),
              },
              SourceWindowCollectiveMemberInput{
                  .rank = 1,
                  .device_id = 1,
                  .work_plan = rank1_plan,
                  .target_layout = target_layout(rank1_target.data(), rank1_target.size()),
              },
          },
      .config = config,
  };
}

SourceWindowCollectiveGroupInput make_local_only_single_consumer_group() {
  static std::array<std::uint8_t, 8> rank0_target{};
  static std::array<std::uint8_t, 8> rank1_target{};
  auto config = base_config();
  config.window_bytes = 16;
  config.distribution_mode = strategy::SourceWindowCollectiveDistributionMode::kLocalOnly;

  const auto source = make_u8_spec("src", 0, {8}, {1});
  const auto destination = make_u8_spec("dst", 0, {8}, {1});
  contracts::RepresentationWorkPlan rank0_plan{
      .items = {make_copy_item(
          source, contracts::TensorCoordinateSpec{}, destination, contracts::TensorCoordinateSpec{}, 8)},
      .committed_bytes = 8,
  };
  contracts::RepresentationWorkPlan rank1_plan{
      .items = {},
      .committed_bytes = 0,
  };
  return SourceWindowCollectiveGroupInput{
      .group = loading::CollectiveLoadGroupHint{.group_id = "tp-local-only", .world_size = 2, .rank = 0},
      .disk_context = make_disk_context(),
      .source_index_digest = "source-index-local-only",
      .members =
          {
              SourceWindowCollectiveMemberInput{
                  .rank = 0,
                  .device_id = 0,
                  .work_plan = rank0_plan,
                  .target_layout = target_layout(rank0_target.data(), rank0_target.size()),
              },
              SourceWindowCollectiveMemberInput{
                  .rank = 1,
                  .device_id = 1,
                  .work_plan = rank1_plan,
                  .target_layout = target_layout(rank1_target.data(), rank1_target.size()),
              },
          },
      .config = config,
  };
}

SourceWindowCollectiveGroupInput make_adjacent_linear_spans_group(SourceWindowCollectiveConfig config) {
  static std::array<std::uint8_t, 16> rank0_target{};
  static std::array<std::uint8_t, 16> rank1_target{};
  const auto source = make_u8_spec("linear_src", 0, {16}, {1});
  const auto destination = make_u8_spec("linear_dst", 0, {16}, {1});
  contracts::RepresentationWorkPlan rank0_plan{
      .items =
          {
              make_copy_item(source, dim_range(0, 0, 4), destination, dim_range(0, 0, 4), 4),
              make_copy_item(source, dim_range(0, 4, 8), destination, dim_range(0, 4, 8), 4),
          },
      .committed_bytes = 8,
  };
  contracts::RepresentationWorkPlan rank1_plan{
      .items = {},
      .committed_bytes = 0,
  };
  return SourceWindowCollectiveGroupInput{
      .group = loading::CollectiveLoadGroupHint{.group_id = "tp-linear-coalesce", .world_size = 2, .rank = 0},
      .disk_context = make_disk_context(),
      .source_index_digest = "source-index-linear-coalesce",
      .members =
          {
              SourceWindowCollectiveMemberInput{
                  .rank = 0,
                  .device_id = 0,
                  .work_plan = rank0_plan,
                  .target_layout = target_layout(rank0_target.data(), rank0_target.size()),
              },
              SourceWindowCollectiveMemberInput{
                  .rank = 1,
                  .device_id = 1,
                  .work_plan = rank1_plan,
                  .target_layout = target_layout(rank1_target.data(), rank1_target.size()),
              },
          },
      .config = config,
  };
}

SourceWindowCollectiveGroupInput make_adjacent_2d_column_spans_group(SourceWindowCollectiveConfig config) {
  static std::array<std::uint8_t, 32> rank0_target{};
  static std::array<std::uint8_t, 32> rank1_target{};
  const auto source = make_u8_spec("cols_src", 0, {4, 8}, {8, 1});
  const auto destination = make_u8_spec("cols_dst", 0, {1, 4, 8}, {32, 8, 1});
  contracts::RepresentationWorkPlan rank0_plan{
      .items =
          {
              make_copy_item(source, dim_range(1, 0, 4), destination, expert_col_range(0, 1, 0, 4), 16),
              make_copy_item(source, dim_range(1, 4, 8), destination, expert_col_range(0, 1, 4, 8), 16),
          },
      .committed_bytes = 32,
  };
  contracts::RepresentationWorkPlan rank1_plan{
      .items = {},
      .committed_bytes = 0,
  };
  return SourceWindowCollectiveGroupInput{
      .group = loading::CollectiveLoadGroupHint{.group_id = "tp-2d-coalesce", .world_size = 2, .rank = 0},
      .disk_context = make_disk_context(),
      .source_index_digest = "source-index-2d-coalesce",
      .members =
          {
              SourceWindowCollectiveMemberInput{
                  .rank = 0,
                  .device_id = 0,
                  .work_plan = rank0_plan,
                  .target_layout = target_layout(rank0_target.data(), rank0_target.size()),
              },
              SourceWindowCollectiveMemberInput{
                  .rank = 1,
                  .device_id = 1,
                  .work_plan = rank1_plan,
                  .target_layout = target_layout(rank1_target.data(), rank1_target.size()),
              },
          },
      .config = config,
  };
}

SourceWindowCollectiveGroupInput make_adjacent_2d_row_spans_group(SourceWindowCollectiveConfig config) {
  static std::array<std::uint8_t, 32> rank0_target{};
  static std::array<std::uint8_t, 32> rank1_target{};
  const auto source = make_u8_spec("rows_src", 0, {4, 8}, {8, 1});
  const auto destination = make_u8_spec("rows_dst", 0, {1, 4, 8}, {32, 8, 1});
  contracts::RepresentationWorkPlan rank0_plan{
      .items =
          {
              make_copy_item(source, rect_range(0, 2, 0, 4), destination, expert_rect_range(0, 1, 0, 2, 0, 4), 8),
              make_copy_item(source, rect_range(2, 4, 0, 4), destination, expert_rect_range(0, 1, 2, 4, 0, 4), 8),
          },
      .committed_bytes = 16,
  };
  contracts::RepresentationWorkPlan rank1_plan{
      .items = {},
      .committed_bytes = 0,
  };
  return SourceWindowCollectiveGroupInput{
      .group = loading::CollectiveLoadGroupHint{.group_id = "tp-2d-row-coalesce", .world_size = 2, .rank = 0},
      .disk_context = make_disk_context(),
      .source_index_digest = "source-index-2d-row-coalesce",
      .members =
          {
              SourceWindowCollectiveMemberInput{
                  .rank = 0,
                  .device_id = 0,
                  .work_plan = rank0_plan,
                  .target_layout = target_layout(rank0_target.data(), rank0_target.size()),
              },
              SourceWindowCollectiveMemberInput{
                  .rank = 1,
                  .device_id = 1,
                  .work_plan = rank1_plan,
                  .target_layout = target_layout(rank1_target.data(), rank1_target.size()),
              },
          },
      .config = config,
  };
}

SourceWindowCollectiveGroupInput make_hybrid_owner_and_shared_windows_group(SourceWindowCollectiveConfig config) {
  static std::array<std::uint8_t, 16> rank0_target{};
  static std::array<std::uint8_t, 16> rank1_target{};
  const auto source = make_u8_spec("hybrid_src", 0, {32}, {1});
  const auto destination = make_u8_spec("hybrid_dst", 0, {16}, {1});
  contracts::RepresentationWorkPlan rank0_plan{
      .items =
          {
              make_copy_item(source, dim_range(0, 0, 4), destination, dim_range(0, 0, 4), 4),
              make_copy_item(source, dim_range(0, 16, 20), destination, dim_range(0, 4, 8), 4),
          },
      .committed_bytes = 8,
  };
  contracts::RepresentationWorkPlan rank1_plan{
      .items = {make_copy_item(source, dim_range(0, 20, 24), destination, dim_range(0, 0, 4), 4)},
      .committed_bytes = 4,
  };
  return SourceWindowCollectiveGroupInput{
      .group = loading::CollectiveLoadGroupHint{.group_id = "tp-hybrid", .world_size = 2, .rank = 0},
      .disk_context = make_disk_context(),
      .source_index_digest = "source-index-hybrid",
      .members =
          {
              SourceWindowCollectiveMemberInput{
                  .rank = 0,
                  .device_id = 0,
                  .work_plan = rank0_plan,
                  .target_layout = target_layout(rank0_target.data(), rank0_target.size()),
              },
              SourceWindowCollectiveMemberInput{
                  .rank = 1,
                  .device_id = 1,
                  .work_plan = rank1_plan,
                  .target_layout = target_layout(rank1_target.data(), rank1_target.size()),
              },
          },
      .config = config,
  };
}

SourceWindowCollectiveGroupInput make_hybrid_dense_shared_window_group(SourceWindowCollectiveConfig config) {
  static std::array<std::uint8_t, 8> rank0_target{};
  static std::array<std::uint8_t, 8> rank1_target{};
  const auto source = make_u8_spec("hybrid_dense_src", 0, {8}, {1});
  const auto destination = make_u8_spec("hybrid_dense_dst", 0, {8}, {1});
  contracts::RepresentationWorkPlan rank0_plan{
      .items = {make_copy_item(source, dim_range(0, 0, 8), destination, dim_range(0, 0, 8), 8)},
      .committed_bytes = 8,
  };
  contracts::RepresentationWorkPlan rank1_plan{
      .items = {make_copy_item(source, dim_range(0, 0, 8), destination, dim_range(0, 0, 8), 8)},
      .committed_bytes = 8,
  };
  return SourceWindowCollectiveGroupInput{
      .group = loading::CollectiveLoadGroupHint{.group_id = "tp-hybrid-dense", .world_size = 2, .rank = 0},
      .disk_context = make_disk_context(),
      .source_index_digest = "source-index-hybrid-dense",
      .members =
          {
              SourceWindowCollectiveMemberInput{
                  .rank = 0,
                  .device_id = 0,
                  .work_plan = rank0_plan,
                  .target_layout = target_layout(rank0_target.data(), rank0_target.size()),
              },
              SourceWindowCollectiveMemberInput{
                  .rank = 1,
                  .device_id = 1,
                  .work_plan = rank1_plan,
                  .target_layout = target_layout(rank1_target.data(), rank1_target.size()),
              },
          },
      .config = config,
  };
}

TEST_CASE("source-window group planner estimates TP dim1 windows without per-rank full reads", "[source_window]") {
  auto input = make_two_rank_dim1_group();

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  REQUIRE(plan.windows.size() == 1);
  CHECK(plan.summary.source_window_group_disk_read_bytes == 32);
  CHECK(plan.summary.source_window_unique_payload_bytes == 32);
  CHECK(plan.summary.source_window_target_write_bytes == 32);
  CHECK(plan.summary.source_window_rank_read_bytes_max == 32);
  REQUIRE(plan.rank_read_bytes.size() == 2);
  CHECK(plan.rank_read_bytes[0] == 32);
  CHECK(plan.rank_read_bytes[1] == 0);
  CHECK(plan.summary.source_window_read_amplification_x1000 == 1000);
  CHECK(plan.summary.source_window_scatter_op_count == 2);
  REQUIRE(plan.windows.front().consumer_spans.size() == 2);
  CHECK(plan.windows.front().consumer_spans.front().row_count == 4);
  CHECK(plan.windows.front().consumer_spans.front().row_bytes == 4);
  CHECK(plan.windows.front().consumer_spans.front().source_stride_bytes == 8);
  CHECK(plan.windows.front().consumer_spans.front().target_stride_bytes == 4);
  CHECK(plan.summary.group_final_admitted);

  auto batched_summary_or = summarize_source_window_batched_scatter(plan, /*runtime_chunk_bytes=*/16);
  REQUIRE(batched_summary_or.ok());
  CHECK(batched_summary_or->feasible);
  CHECK(batched_summary_or->estimated_runtime_chunk_count == 2);
  CHECK(batched_summary_or->target_write_bytes == 32);
  CHECK(batched_summary_or->estimated_current_copy_2d_launches == 4);
  CHECK(batched_summary_or->batched_scatter_launches == 4);
}

TEST_CASE("source-window group planner coalesces adjacent linear scatter spans before admission", "[source_window]") {
  auto config = base_config();
  config.max_scatter_ops_per_window = 1;
  auto input = make_adjacent_linear_spans_group(config);

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  REQUIRE(plan.windows.size() == 1);
  REQUIRE(plan.windows.front().consumer_spans.size() == 1);
  const auto& span = plan.windows.front().consumer_spans.front();
  CHECK(span.rank == 0);
  CHECK(span.storage_index == 0);
  CHECK(span.source_offset == 0);
  CHECK(span.target_offset == 0);
  CHECK(span.length == 8);
  CHECK(span.row_count == 1);
  CHECK(plan.summary.source_window_group_disk_read_bytes == 8);
  CHECK(plan.summary.source_window_target_write_bytes == 8);
  CHECK(plan.summary.source_window_scatter_op_count == 1);
  CHECK(plan.summary.group_final_admitted);
}

TEST_CASE(
    "source-window group planner coalesces adjacent 2d column scatter spans before admission",
    "[source_window]") {
  auto config = base_config();
  config.window_bytes = 64;
  config.max_scatter_ops_per_window = 1;
  auto input = make_adjacent_2d_column_spans_group(config);

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  REQUIRE(plan.windows.size() == 1);
  REQUIRE(plan.windows.front().consumer_spans.size() == 1);
  const auto& span = plan.windows.front().consumer_spans.front();
  CHECK(span.rank == 0);
  CHECK(span.storage_index == 0);
  CHECK(span.source_offset == 0);
  CHECK(span.target_offset == 0);
  CHECK(span.row_count == 4);
  CHECK(span.row_bytes == 8);
  CHECK(span.source_stride_bytes == 8);
  CHECK(span.target_stride_bytes == 8);
  CHECK(span.length == 32);
  CHECK(plan.summary.source_window_group_disk_read_bytes == 32);
  CHECK(plan.summary.source_window_unique_payload_bytes == 32);
  CHECK(plan.summary.source_window_target_write_bytes == 32);
  CHECK(plan.summary.source_window_scatter_op_count == 1);
  CHECK(plan.summary.group_final_admitted);
}

TEST_CASE("source-window group planner coalesces adjacent 2d row scatter spans before admission", "[source_window]") {
  auto config = base_config();
  config.window_bytes = 64;
  config.max_gap_bytes = 8;
  config.max_window_amplification_x1000 = 2000;
  config.max_plan_read_amplification_x1000 = 2000;
  config.max_scatter_ops_per_window = 1;
  auto input = make_adjacent_2d_row_spans_group(config);

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  REQUIRE(plan.windows.size() == 1);
  REQUIRE(plan.windows.front().consumer_spans.size() == 1);
  const auto& span = plan.windows.front().consumer_spans.front();
  CHECK(span.rank == 0);
  CHECK(span.storage_index == 0);
  CHECK(span.source_offset == 0);
  CHECK(span.target_offset == 0);
  CHECK(span.row_count == 4);
  CHECK(span.row_bytes == 4);
  CHECK(span.source_stride_bytes == 8);
  CHECK(span.target_stride_bytes == 8);
  CHECK(span.length == 16);
  CHECK(plan.summary.source_window_group_disk_read_bytes == 28);
  CHECK(plan.summary.source_window_unique_payload_bytes == 16);
  CHECK(plan.summary.source_window_target_write_bytes == 16);
  CHECK(plan.summary.source_window_scatter_op_count == 1);
  CHECK(plan.summary.group_final_admitted);
}

TEST_CASE("source-window group planner admits consumer-routed distribution with useful peer bytes", "[source_window]") {
  auto input = make_two_rank_dim1_group();
  input.config.distribution_mode = strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted;

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  REQUIRE(plan.windows.size() == 1);
  CHECK(plan.distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted);
  CHECK(plan.summary.source_window_group_disk_read_bytes == 32);
  CHECK(plan.summary.source_window_target_write_bytes == 32);
  CHECK(plan.summary.source_window_peer_transfer_bytes == 16);
  CHECK(plan.summary.source_window_peer_useful_bytes == 16);
  CHECK(plan.summary.source_window_peer_waste_bytes == 0);
  CHECK(plan.summary.group_final_admitted);
}

TEST_CASE("source-window group planner admits local-only windows with owner-only consumers", "[source_window]") {
  auto input = make_local_only_single_consumer_group();

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  CHECK(plan.distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kLocalOnly);
  REQUIRE(plan.windows.size() == 1);
  CHECK(plan.windows.front().owner_rank == 0);
  REQUIRE(plan.windows.front().consumer_spans.size() == 1);
  CHECK(plan.windows.front().consumer_spans.front().rank == 0);
  REQUIRE(plan.rank_read_bytes.size() == 2);
  CHECK(plan.rank_read_bytes[0] == 8);
  CHECK(plan.rank_read_bytes[1] == 0);
  CHECK(plan.summary.source_window_group_disk_read_bytes == 8);
  CHECK(plan.summary.source_window_target_write_bytes == 8);
  CHECK(plan.summary.source_window_peer_transfer_bytes == 0);
  CHECK(plan.summary.source_window_peer_useful_bytes == 0);
  CHECK(plan.summary.source_window_peer_waste_bytes == 0);
  CHECK(plan.summary.group_final_admitted);
}

TEST_CASE("source-window auto distribution selects local-only for owner-only windows", "[source_window]") {
  auto input = make_local_only_single_consumer_group();
  input.config.distribution_mode = strategy::SourceWindowCollectiveDistributionMode::kAuto;

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  CHECK(plan.distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kLocalOnly);
  CHECK(plan.summary.distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kLocalOnly);
  REQUIRE(plan.windows.size() == 1);
  CHECK(plan.windows.front().owner_rank == 0);
  REQUIRE(plan.rank_read_bytes.size() == 2);
  CHECK(plan.rank_read_bytes[0] == 8);
  CHECK(plan.rank_read_bytes[1] == 0);
  CHECK(plan.summary.source_window_peer_transfer_bytes == 0);
  CHECK(plan.summary.source_window_peer_waste_bytes == 0);
  CHECK(plan.summary.group_final_admitted);
}

TEST_CASE("source-window auto distribution keeps all-gather for multi-rank consumers", "[source_window]") {
  auto input = make_two_rank_dim1_group();
  input.config.distribution_mode = strategy::SourceWindowCollectiveDistributionMode::kAuto;

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  CHECK(plan.distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kFullWindowAllGather);
  CHECK(plan.summary.distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kFullWindowAllGather);
  CHECK(plan.summary.source_window_peer_transfer_bytes == 32);
  CHECK(plan.summary.source_window_peer_waste_bytes == 16);
  CHECK(plan.summary.group_final_admitted);
}

TEST_CASE("source-window hybrid distribution mixes local-only and routed windows", "[source_window]") {
  auto config = base_config();
  config.window_bytes = 64;
  config.max_gap_bytes = 0;
  config.distribution_mode = strategy::SourceWindowCollectiveDistributionMode::kHybridWindow;
  config.min_routed_peer_saving_bytes = 0;
  auto input = make_hybrid_owner_and_shared_windows_group(config);

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  CHECK(plan.distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kHybridWindow);
  CHECK(plan.summary.distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kHybridWindow);
  REQUIRE(plan.windows.size() == 2);
  CHECK(plan.windows[0].distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kLocalOnly);
  CHECK(plan.windows[0].owner_rank == 0);
  REQUIRE(plan.windows[0].consumer_spans.size() == 1);
  CHECK(plan.windows[0].consumer_spans.front().rank == 0);
  CHECK(plan.windows[1].distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted);
  REQUIRE(plan.windows[1].consumer_spans.size() == 2);
  CHECK(plan.summary.source_window_group_disk_read_bytes == 12);
  CHECK(plan.summary.source_window_target_write_bytes == 12);
  CHECK(plan.summary.source_window_peer_transfer_bytes == 4);
  CHECK(plan.summary.source_window_peer_useful_bytes == 4);
  CHECK(plan.summary.source_window_peer_waste_bytes == 0);
  CHECK(plan.summary.source_window_scatter_op_count == 3);
  REQUIRE(plan.rank_read_bytes.size() == 2);
  CHECK(plan.rank_read_bytes[0] == 4);
  CHECK(plan.rank_read_bytes[1] == 8);
  CHECK(plan.summary.group_final_admitted);

  auto batched_summary_or = summarize_source_window_batched_scatter(plan, /*runtime_chunk_bytes=*/64);
  REQUIRE(batched_summary_or.ok());
  CHECK(batched_summary_or->feasible);
  CHECK(batched_summary_or->estimated_current_scatter_launches == 3);
  CHECK(batched_summary_or->estimated_current_pack_launches == 1);
  CHECK(batched_summary_or->estimated_current_copy_launches == 4);
  CHECK(batched_summary_or->batched_scatter_launches == 3);
  CHECK(batched_summary_or->batched_pack_launches == 1);
  CHECK(batched_summary_or->estimated_copy_launch_reduction_x1000 == 0);
}

TEST_CASE("source-window hybrid distribution keeps dense multi-rank windows on all-gather", "[source_window]") {
  auto config = base_config();
  config.window_bytes = 64;
  config.max_gap_bytes = 0;
  config.distribution_mode = strategy::SourceWindowCollectiveDistributionMode::kHybridWindow;
  auto input = make_hybrid_dense_shared_window_group(config);

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  CHECK(plan.distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kHybridWindow);
  REQUIRE(plan.windows.size() == 1);
  CHECK(
      plan.windows.front().distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kFullWindowAllGather);
  CHECK(plan.summary.source_window_group_disk_read_bytes == 8);
  CHECK(plan.summary.source_window_target_write_bytes == 16);
  CHECK(plan.summary.source_window_peer_transfer_bytes == 8);
  CHECK(plan.summary.source_window_peer_useful_bytes == 8);
  CHECK(plan.summary.source_window_peer_waste_bytes == 0);
  CHECK(plan.summary.group_final_admitted);
}

TEST_CASE("source-window hybrid distribution keeps strided 2d windows on all-gather", "[source_window]") {
  auto input = make_two_rank_dim1_group();
  input.config.distribution_mode = strategy::SourceWindowCollectiveDistributionMode::kHybridWindow;

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  CHECK(plan.distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kHybridWindow);
  REQUIRE(plan.windows.size() == 1);
  CHECK(
      plan.windows.front().distribution_mode == strategy::SourceWindowCollectiveDistributionMode::kFullWindowAllGather);
  CHECK(plan.summary.source_window_group_disk_read_bytes == 32);
  CHECK(plan.summary.source_window_target_write_bytes == 32);
  CHECK(plan.summary.source_window_peer_transfer_bytes == 32);
  CHECK(plan.summary.source_window_peer_useful_bytes == 16);
  CHECK(plan.summary.source_window_peer_waste_bytes == 16);
  CHECK(plan.summary.group_final_admitted);
}

TEST_CASE("source-window default hybrid routed threshold is reachable for default windows", "[source_window]") {
  StoreEngineOptions::MaterializationStrategyConfig strategy_config;
  const auto config = source_window_collective_config_from_strategy(strategy_config);
  const uint64_t max_tp8_window_peer_bytes = config.window_bytes * 7;

  CHECK(config.window_bytes == 512ULL * 1024ULL * 1024ULL);
  CHECK(config.min_routed_peer_saving_bytes < max_tp8_window_peer_bytes);
}

TEST_CASE("source-window group planner rejects local-only windows with remote consumers", "[source_window]") {
  auto input = make_two_rank_dim1_group();
  input.config.distribution_mode = strategy::SourceWindowCollectiveDistributionMode::kLocalOnly;

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE_FALSE(plan_or.ok());
  CHECK(std::string(plan_or.status().message()).find("local_only_window_has_remote_consumers") != std::string::npos);
}

TEST_CASE("source-window admission uses local mapped physical read baseline", "[source_window]") {
  static std::array<std::uint8_t, 32> rank0_target{};
  static std::array<std::uint8_t, 32> rank1_target{};
  auto config = base_config();
  config.min_rank_read_saving_bytes = 16;

  const auto source0 = make_u8_spec("src0", 0, {4, 8}, {8, 1});
  const auto source1 = make_u8_spec("src1", 64, {4, 8}, {8, 1});
  const auto destination0 = make_u8_spec("dst0", 0, {4, 4}, {4, 1});
  const auto destination1 = make_u8_spec("dst1", 16, {4, 4}, {4, 1});
  contracts::RepresentationWorkPlan rank0_plan{
      .items =
          {
              make_copy_item(source0, dim_range(1, 0, 4), destination0, contracts::TensorCoordinateSpec{}, 16),
              make_copy_item(source1, dim_range(1, 0, 4), destination1, contracts::TensorCoordinateSpec{}, 16),
          },
      .committed_bytes = 32,
  };
  contracts::RepresentationWorkPlan rank1_plan{
      .items =
          {
              make_copy_item(source0, dim_range(1, 4, 8), destination0, contracts::TensorCoordinateSpec{}, 16),
              make_copy_item(source1, dim_range(1, 4, 8), destination1, contracts::TensorCoordinateSpec{}, 16),
          },
      .committed_bytes = 32,
  };
  SourceWindowCollectiveGroupInput input{
      .group = loading::CollectiveLoadGroupHint{.group_id = "tp-physical-baseline", .world_size = 2, .rank = 0},
      .disk_context = make_disk_context(),
      .source_index_digest = "source-index-physical-baseline",
      .members =
          {
              SourceWindowCollectiveMemberInput{
                  .rank = 0,
                  .device_id = 0,
                  .work_plan = rank0_plan,
                  .target_layout = target_layout(rank0_target.data(), rank0_target.size()),
              },
              SourceWindowCollectiveMemberInput{
                  .rank = 1,
                  .device_id = 1,
                  .work_plan = rank1_plan,
                  .target_layout = target_layout(rank1_target.data(), rank1_target.size()),
              },
          },
      .config = config,
  };

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  REQUIRE(plan.windows.size() == 2);
  CHECK(plan.summary.source_window_group_disk_read_bytes == 64);
  CHECK(plan.summary.source_window_rank_read_bytes_max == 32);
  CHECK(plan.summary.source_window_local_rank_read_bytes_max == 64);
  CHECK(plan.summary.source_window_rank_read_saving_bytes == 32);
  CHECK(plan.summary.group_final_admitted);
}

TEST_CASE("source-window group planner coalesces rect2d source windows", "[source_window]") {
  static std::array<std::uint8_t, 16> rank0_target{};
  static std::array<std::uint8_t, 16> rank1_target{};
  auto config = base_config();
  config.window_bytes = 64;
  config.max_gap_bytes = 4;
  config.max_window_amplification_x1000 = 2000;
  config.max_plan_read_amplification_x1000 = 2000;

  const auto source = make_u8_spec("src", 0, {4, 8}, {8, 1});
  const auto destination = make_u8_spec("dst", 0, {4, 4}, {4, 1});
  const auto item = make_copy_item(source, rect_range(0, 4, 2, 6), destination, contracts::TensorCoordinateSpec{}, 16);
  const contracts::RepresentationWorkPlan work_plan{.items = {item}, .committed_bytes = 16};
  SourceWindowCollectiveGroupInput input{
      .group = loading::CollectiveLoadGroupHint{.group_id = "tp-rect", .world_size = 2, .rank = 0},
      .disk_context = make_disk_context(),
      .source_index_digest = "source-index-rect",
      .members =
          {
              SourceWindowCollectiveMemberInput{
                  .rank = 0,
                  .device_id = 0,
                  .work_plan = work_plan,
                  .target_layout = target_layout(rank0_target.data(), rank0_target.size()),
              },
              SourceWindowCollectiveMemberInput{
                  .rank = 1,
                  .device_id = 1,
                  .work_plan = work_plan,
                  .target_layout = target_layout(rank1_target.data(), rank1_target.size()),
              },
          },
      .config = config,
  };

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  REQUIRE(plan.windows.size() == 1);
  CHECK(plan.windows.front().start == 2);
  CHECK(plan.windows.front().end == 30);
  CHECK(plan.summary.source_window_group_disk_read_bytes == 28);
  CHECK(plan.summary.source_window_unique_payload_bytes == 16);
  CHECK(plan.summary.source_window_read_amplification_x1000 == 1750);
  CHECK(plan.summary.source_window_scatter_op_count == 2);
  REQUIRE(plan.windows.front().consumer_spans.size() == 2);
  CHECK(plan.windows.front().consumer_spans.front().row_count == 4);
  CHECK(plan.windows.front().consumer_spans.front().row_bytes == 4);
}

TEST_CASE("source-window group planner compresses 2d source into 3d expert target slots", "[source_window]") {
  static std::array<std::uint8_t, 32> rank0_target{};
  static std::array<std::uint8_t, 32> rank1_target{};
  auto config = base_config();
  config.window_bytes = 64;

  const auto source = make_u8_spec("expert_src", 0, {4, 8}, {8, 1});
  const auto destination = make_u8_spec("expert_dst", 0, {2, 4, 4}, {16, 4, 1});
  contracts::RepresentationWorkPlan rank0_plan{
      .items = {make_copy_item(source, dim_range(1, 0, 4), destination, dim_range(0, 0, 1), 16)},
      .committed_bytes = 16,
  };
  contracts::RepresentationWorkPlan rank1_plan{
      .items = {make_copy_item(source, dim_range(1, 4, 8), destination, dim_range(0, 1, 2), 16)},
      .committed_bytes = 16,
  };
  SourceWindowCollectiveGroupInput input{
      .group = loading::CollectiveLoadGroupHint{.group_id = "tp-expert3d", .world_size = 2, .rank = 0},
      .disk_context = make_disk_context(),
      .source_index_digest = "source-index-expert3d",
      .members =
          {
              SourceWindowCollectiveMemberInput{
                  .rank = 0,
                  .device_id = 0,
                  .work_plan = rank0_plan,
                  .target_layout = target_layout(rank0_target.data(), rank0_target.size()),
              },
              SourceWindowCollectiveMemberInput{
                  .rank = 1,
                  .device_id = 1,
                  .work_plan = rank1_plan,
                  .target_layout = target_layout(rank1_target.data(), rank1_target.size()),
              },
          },
      .config = config,
  };

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  REQUIRE(plan_or->windows.size() == 1);
  CHECK(plan_or->summary.source_window_group_disk_read_bytes == 32);
  CHECK(plan_or->summary.source_window_unique_payload_bytes == 32);
  CHECK(plan_or->summary.source_window_target_write_bytes == 32);
  REQUIRE(plan_or->windows.front().consumer_spans.size() == 2);
  CHECK(plan_or->windows.front().consumer_spans[0].row_count == 4);
  CHECK(plan_or->windows.front().consumer_spans[0].row_bytes == 4);
  CHECK(plan_or->windows.front().consumer_spans[0].target_offset == 0);
  CHECK(plan_or->windows.front().consumer_spans[1].target_offset == 16);
}

TEST_CASE("source-window group planner splits consumer spans at target storage boundaries", "[source_window]") {
  static std::array<std::uint8_t, 4> rank0_target{};
  static std::array<std::uint8_t, 4> rank1_target{};
  auto config = base_config();
  config.window_bytes = 16;

  const auto source = make_u8_spec("src", 0, {4}, {1});
  const auto destination = make_u8_spec("dst", 0, {4}, {1});
  const auto item =
      make_copy_item(source, contracts::TensorCoordinateSpec{}, destination, contracts::TensorCoordinateSpec{}, 4);
  const contracts::RepresentationWorkPlan work_plan{.items = {item}, .committed_bytes = 4};
  SourceWindowCollectiveGroupInput input{
      .group = loading::CollectiveLoadGroupHint{.group_id = "tp-storage", .world_size = 2, .rank = 0},
      .disk_context = make_disk_context(),
      .source_index_digest = "source-index-storage",
      .members =
          {
              SourceWindowCollectiveMemberInput{
                  .rank = 0,
                  .device_id = 0,
                  .work_plan = work_plan,
                  .target_layout = target_layout(rank0_target.data(), rank0_target.size()),
                  .storage_spans =
                      {
                          TargetStorageSpan{.base_offset = 0, .length = 2, .base_ptr = rank0_target.data()},
                          TargetStorageSpan{.base_offset = 2, .length = 2, .base_ptr = rank0_target.data() + 2},
                      },
              },
              SourceWindowCollectiveMemberInput{
                  .rank = 1,
                  .device_id = 1,
                  .work_plan = work_plan,
                  .target_layout = target_layout(rank1_target.data(), rank1_target.size()),
              },
          },
      .config = config,
  };

  auto plan_or = build_source_window_collective_plan(input);
  REQUIRE(plan_or.ok());
  const auto& spans = plan_or->windows.front().consumer_spans;
  REQUIRE(spans.size() == 3);
  bool saw_rank0_storage0 = false;
  bool saw_rank0_storage1 = false;
  for (const auto& span : spans) {
    if (span.rank == 0 && span.storage_index == 0 && span.length == 2) {
      saw_rank0_storage0 = true;
    }
    if (span.rank == 0 && span.storage_index == 1 && span.length == 2) {
      saw_rank0_storage1 = true;
    }
  }
  CHECK(saw_rank0_storage0);
  CHECK(saw_rank0_storage1);
  CHECK(plan_or->summary.source_window_scatter_op_count == 3);

  auto batched_summary_or = summarize_source_window_batched_scatter(*plan_or, /*runtime_chunk_bytes=*/16);
  REQUIRE(batched_summary_or.ok());
  CHECK(batched_summary_or->feasible);
  CHECK(batched_summary_or->consumer_span_count == 3);
  CHECK(batched_summary_or->estimated_current_scatter_launches == 3);
  CHECK(batched_summary_or->batched_scatter_launches == 2);
  CHECK(batched_summary_or->max_descriptors_per_batched_scatter == 2);
  CHECK(batched_summary_or->estimated_copy_launch_reduction_x1000 == 334);
}

TEST_CASE("source-window tensor-staged summary coalesces adjacent tensor fragments", "[source_window][tensor_stage]") {
  static std::array<std::uint8_t, 16> rank0_target{};
  static std::array<std::uint8_t, 16> rank1_target{};
  auto config = base_config();
  config.window_bytes = 32;

  const auto source = make_u8_spec("src", 0, {16}, {1});
  const auto destination = make_u8_spec("dst", 0, {16}, {1});
  contracts::RepresentationWorkPlan rank0_plan{
      .items =
          {
              make_copy_item(source, dim_range(0, 0, 4), destination, dim_range(0, 0, 4), 4),
              make_copy_item(source, dim_range(0, 4, 8), destination, dim_range(0, 4, 8), 4),
          },
      .committed_bytes = 8,
  };
  contracts::RepresentationWorkPlan rank1_plan{
      .items =
          {
              make_copy_item(source, dim_range(0, 8, 12), destination, dim_range(0, 8, 12), 4),
              make_copy_item(source, dim_range(0, 12, 16), destination, dim_range(0, 12, 16), 4),
          },
      .committed_bytes = 8,
  };
  SourceWindowCollectiveGroupInput input{
      .group = loading::CollectiveLoadGroupHint{.group_id = "tp-tensor-stage", .world_size = 2, .rank = 0},
      .disk_context = make_disk_context(),
      .source_index_digest = "source-index-tensor-stage",
      .members =
          {
              SourceWindowCollectiveMemberInput{
                  .rank = 0,
                  .device_id = 0,
                  .work_plan = rank0_plan,
                  .target_layout = target_layout(rank0_target.data(), rank0_target.size()),
              },
              SourceWindowCollectiveMemberInput{
                  .rank = 1,
                  .device_id = 1,
                  .work_plan = rank1_plan,
                  .target_layout = target_layout(rank1_target.data(), rank1_target.size()),
              },
          },
      .config = config,
  };

  auto summary_or = summarize_source_window_tensor_staged_copy(input);
  REQUIRE(summary_or.ok());
  CHECK(summary_or->feasible);
  CHECK(summary_or->eligible_bytes == 16);
  CHECK(summary_or->ineligible_bytes == 0);
  CHECK(summary_or->source_fragment_count == 4);
  CHECK(summary_or->destination_tensor_count == 1);
  CHECK(summary_or->source_tensor_count == 1);
  CHECK(summary_or->raw_copy_ops == 4);
  CHECK(summary_or->tensor_staged_copy_ops == 2);
  CHECK(summary_or->linear_copy_ops == 2);
  CHECK(summary_or->copy_2d_ops == 0);
  CHECK(summary_or->max_tensor_staged_copy_ops_per_rank == 1);
  CHECK(summary_or->estimated_op_reduction_x1000 == 500);
}

TEST_CASE("source-window tensor-staged summary rejects unsupported source fragments", "[source_window][tensor_stage]") {
  static std::array<std::uint8_t, 8> rank0_target{};
  static std::array<std::uint8_t, 8> rank1_target{};
  auto config = base_config();
  config.window_bytes = 16;

  const auto source = make_u8_spec("src", 0, {8}, {1});
  const auto destination = make_u8_spec("dst", 0, {8}, {1});
  auto item =
      make_copy_item(source, contracts::TensorCoordinateSpec{}, destination, contracts::TensorCoordinateSpec{}, 8);
  item.sources[0].prefix_count = 2;
  contracts::RepresentationWorkPlan rank0_plan{
      .items = {item},
      .committed_bytes = 8,
  };
  contracts::RepresentationWorkPlan rank1_plan{.items = {}, .committed_bytes = 0};
  SourceWindowCollectiveGroupInput input{
      .group = loading::CollectiveLoadGroupHint{.group_id = "tp-tensor-stage-reject", .world_size = 2, .rank = 0},
      .disk_context = make_disk_context(),
      .source_index_digest = "source-index-tensor-stage-reject",
      .members =
          {
              SourceWindowCollectiveMemberInput{
                  .rank = 0,
                  .device_id = 0,
                  .work_plan = rank0_plan,
                  .target_layout = target_layout(rank0_target.data(), rank0_target.size()),
              },
              SourceWindowCollectiveMemberInput{
                  .rank = 1,
                  .device_id = 1,
                  .work_plan = rank1_plan,
                  .target_layout = target_layout(rank1_target.data(), rank1_target.size()),
              },
          },
      .config = config,
  };

  auto summary_or = summarize_source_window_tensor_staged_copy(input);
  REQUIRE(summary_or.ok());
  CHECK_FALSE(summary_or->feasible);
  CHECK(summary_or->reject_reason == "ineligible_work_items");
  CHECK(summary_or->eligible_bytes == 0);
  CHECK(summary_or->ineligible_bytes == 8);
  CHECK(summary_or->raw_copy_ops == 0);
  CHECK(summary_or->tensor_staged_copy_ops == 0);
}

TEST_CASE("source-window group planner hash is sensitive to source, members, layout, and mode", "[source_window]") {
  auto base = make_two_rank_dim1_group();
  auto base_plan_or = build_source_window_collective_plan(base);
  REQUIRE(base_plan_or.ok());

  auto source_changed = base;
  source_changed.source_index_digest = "source-index-b";
  auto source_plan_or = build_source_window_collective_plan(source_changed);
  REQUIRE(source_plan_or.ok());
  CHECK(source_plan_or->plan_hash != base_plan_or->plan_hash);

  auto order_changed = base;
  std::swap(order_changed.members[0], order_changed.members[1]);
  auto order_plan_or = build_source_window_collective_plan(order_changed);
  REQUIRE(order_plan_or.ok());
  CHECK(order_plan_or->plan_hash != base_plan_or->plan_hash);

  auto layout_changed = base;
  layout_changed.members[0].target_layout.storages[0].length = 32;
  layout_changed.members[0].target_layout.total_size = 32;
  auto layout_plan_or = build_source_window_collective_plan(layout_changed);
  REQUIRE(layout_plan_or.ok());
  CHECK(layout_plan_or->plan_hash != base_plan_or->plan_hash);

  auto mode_changed = base;
  mode_changed.config.distribution_mode = strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted;
  auto mode_plan_or = build_source_window_collective_plan(mode_changed);
  REQUIRE(mode_plan_or.ok());
  CHECK(mode_plan_or->plan_hash != base_plan_or->plan_hash);

  auto span_changed = base;
  span_changed.members[0].work_plan.items[0].sources[0].fragment.source_range = dim_range(1, 1, 5);
  auto span_plan_or = build_source_window_collective_plan(span_changed);
  REQUIRE(span_plan_or.ok());
  CHECK(span_plan_or->plan_hash != base_plan_or->plan_hash);

  auto group_changed = base;
  group_changed.group.group_id = "source-window-different-runtime-group";
  auto group_plan_or = build_source_window_collective_plan(group_changed);
  REQUIRE(group_plan_or.ok());
  CHECK(group_plan_or->plan_hash == base_plan_or->plan_hash);
}

TEST_CASE("source-window prepared realization facts are non-semantic plan inputs", "[source_window]") {
  auto base = make_two_rank_dim1_group();
  auto base_plan_or = build_source_window_collective_plan(base);
  REQUIRE(base_plan_or.ok());

  auto with_facts = base;
  with_facts.members[0].prepared_realization = loading::SourceWindowPreparedRealizationFacts{
      .group_key = "rank0-group-key",
      .member_key = "rank0-member-key",
      .realization_plan_hash = "rank0-realization-plan",
      .target_layout_template_hash = "shared-target-layout-template",
      .target_index_hash = "shared-target-index",
  };
  with_facts.members[1].prepared_realization = loading::SourceWindowPreparedRealizationFacts{
      .group_key = "rank1-group-key",
      .member_key = "rank1-member-key",
      .realization_plan_hash = "rank1-realization-plan",
      .target_layout_template_hash = "shared-target-layout-template",
      .target_index_hash = "shared-target-index",
  };

  auto plan_or = build_source_window_collective_plan(with_facts);
  REQUIRE(plan_or.ok());
  CHECK(plan_or->plan_hash == base_plan_or->plan_hash);
}

} // namespace
} // namespace tensorcast::store::replica
