// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/ingestion/source_bound_strategy_planner.h"

#include <cstdint>

#include "catch2/catch_test_macros.hpp"

namespace tensorcast::store::runtime::ingestion::strategy {
namespace {

using RepresentationWorkItem = materialization::contracts::RepresentationWorkItem;
using RepresentationWorkItemKind = materialization::contracts::RepresentationWorkItemKind;
using RepresentationWorkPlan = materialization::contracts::RepresentationWorkPlan;
using WorkPartitionKind = materialization::contracts::WorkPartitionKind;

loader::ByteRangeMap make_data_map(uint64_t total_bytes) {
  loader::ByteRangeMap map;
  map.total_bytes = total_bytes;
  map.num_sources = total_bytes > 0 ? 1 : 0;
  if (total_bytes > 0) {
    map.segments.push_back(
        loader::ByteRangeSegment{
            .kind = loader::ByteRangeSegment::Kind::kData,
            .dst_offset = 0,
            .length = total_bytes,
            .src_offset = 0,
            .source_index = 0,
        });
  }
  return map;
}

ResolvedMaterializationPlan make_plan(RepresentationWorkPlan work_plan) {
  ResolvedMaterializationPlan plan;
  plan.artifact_id = "cgid:source-bound-plan";
  plan.generation = 1;
  plan.canonical_index_json = R"({"tensor":[0,8,[8],[1],"torch.uint8",0]})";
  plan.representation_work_plan = std::move(work_plan);
  return plan;
}

loading::ExecutionTopologyContext make_collective_topology() {
  loading::ExecutionTopologyContext execution_topology;
  execution_topology.collective_load_group =
      loading::CollectiveLoadGroupHint{.group_id = "group-a", .world_size = 4, .rank = 1};
  execution_topology.source_locality = loading::SourceLocalityHint::kSharedSource;
  return execution_topology;
}

StoreEngineOptions::MaterializationStrategyConfig make_strategy_config() {
  StoreEngineOptions::MaterializationStrategyConfig strategy_config;
  strategy_config.enable_owner_file_collective = true;
  strategy_config.allow_mixed_execution = true;
  strategy_config.owner_file_collective_batch_bytes = 512;
  return strategy_config;
}

TEST_CASE("Source-bound strategy planner emits pure_collective mode", "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kDim0Partitioned,
          .committed_bytes = 8,
      });
  auto plan = make_plan(std::move(work_plan));

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.collective_data_map = make_data_map(8);
  lowering_artifacts.executor_generic_data_map = make_data_map(8);

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kRequirePureCollective,
      make_strategy_config(),
      make_collective_topology(),
      /*disk_source_available=*/true);
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kPureCollective);
  CHECK(strategy_plan_or->lane_plan.require_collective_success);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "pure_collective");
  CHECK(strategy_plan_or->summary.strict_pure_collective_eligible);
  CHECK(strategy_plan_or->summary.planned_collective_admitted_bytes == 8);
}

TEST_CASE("Source-bound strategy planner emits collective_first_mixed mode", "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kDim0Partitioned,
          .committed_bytes = 8,
      });
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kPadFill,
          .committed_bytes = 2,
      });
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kResidualByteRange,
          .committed_bytes = 4,
      });
  auto plan = make_plan(std::move(work_plan));

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.collective_data_map = make_data_map(8);
  lowering_artifacts.executor_generic_data_map = make_data_map(12);

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      make_strategy_config(),
      make_collective_topology(),
      /*disk_source_available=*/true);
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kCollectiveFirstMixed);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "collective_first_mixed");
  CHECK(strategy_plan_or->lane_plan.local_pad_bytes == 2);
  CHECK(strategy_plan_or->summary.planned_generic_residual_bytes == 4);
  CHECK(strategy_plan_or->summary.planned_collective_admitted_bytes == 8);
  CHECK_FALSE(strategy_plan_or->lane_plan.reject_reason_buckets.contains("generic_backend_coverage_unproven"));
}

TEST_CASE(
    "Source-bound strategy planner emits generic_only mode when collective is disabled",
    "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kDim0Partitioned,
          .committed_bytes = 8,
      });
  auto plan = make_plan(std::move(work_plan));

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.executor_generic_data_map = make_data_map(8);

  auto strategy_config = make_strategy_config();
  strategy_config.enable_owner_file_collective = false;

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      strategy_config,
      make_collective_topology(),
      /*disk_source_available=*/true);
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kGenericOnly);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "generic_only");
  REQUIRE(strategy_plan_or->summary.planner_reject_reason_buckets.contains("collective_strategy_disabled"));
  CHECK(strategy_plan_or->summary.planner_reject_reason_buckets.at("collective_strategy_disabled") == 8);
}

TEST_CASE("Source-bound strategy planner emits local_typed_only mode", "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kConstFill,
          .committed_bytes = 3,
      });
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kPadFill,
          .committed_bytes = 5,
      });
  auto plan = make_plan(std::move(work_plan));

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      std::nullopt,
      SourceBoundPolicy::kDisableCollective,
      make_strategy_config(),
      loading::ExecutionTopologyContext{},
      /*disk_source_available=*/false);
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kLocalTypedOnly);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "local_typed_only");
  CHECK(strategy_plan_or->lane_plan.local_fill_bytes == 3);
  CHECK(strategy_plan_or->lane_plan.local_pad_bytes == 5);
  CHECK(strategy_plan_or->summary.planned_local_typed_bytes == 8);
}

TEST_CASE("Source-bound strategy planner tracks deferred typed bytes", "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kConcatAssemble,
          .committed_bytes = 4,
      });
  auto plan = make_plan(std::move(work_plan));

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.executor_generic_data_map = make_data_map(4);

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      make_strategy_config(),
      make_collective_topology(),
      /*disk_source_available=*/true);
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kGenericOnly);
  CHECK(strategy_plan_or->lane_plan.deferred_typed_bytes == 4);
  REQUIRE(strategy_plan_or->lane_plan.reject_reason_buckets.contains("concat_not_admitted"));
  CHECK(strategy_plan_or->lane_plan.reject_reason_buckets.at("concat_not_admitted") == 4);
}

TEST_CASE(
    "Source-bound strategy planner emits reject mode for impossible strict collective plan",
    "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kDim0Partitioned,
          .committed_bytes = 8,
      });
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kPadFill,
          .committed_bytes = 2,
      });
  auto plan = make_plan(std::move(work_plan));

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.collective_data_map = make_data_map(8);
  lowering_artifacts.executor_generic_data_map = make_data_map(8);

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kRequirePureCollective,
      make_strategy_config(),
      make_collective_topology(),
      /*disk_source_available=*/true);
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kRejected);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "reject");
  CHECK_FALSE(strategy_plan_or->summary.strict_pure_collective_eligible);
  CHECK(strategy_plan_or->lane_plan.require_collective_success);
}

TEST_CASE(
    "Source-bound strategy planner rejects when generic backend coverage is unproven",
    "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kDim0Partitioned,
          .committed_bytes = 8,
      });
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kResidualByteRange,
          .committed_bytes = 4,
      });
  auto plan = make_plan(std::move(work_plan));

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.collective_data_map = make_data_map(8);
  lowering_artifacts.executor_generic_data_map = make_data_map(8);

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      make_strategy_config(),
      make_collective_topology(),
      /*disk_source_available=*/true);
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kRejected);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "reject");
  REQUIRE(strategy_plan_or->summary.planner_reject_reason_buckets.contains("generic_backend_coverage_unproven"));
  CHECK(strategy_plan_or->summary.planner_reject_reason_buckets.at("generic_backend_coverage_unproven") == 4);
}

} // namespace
} // namespace tensorcast::store::runtime::ingestion::strategy
