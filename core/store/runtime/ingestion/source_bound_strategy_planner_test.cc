// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/ingestion/source_bound_strategy_planner.h"

#include <array>
#include <cstdint>

#include "catch2/catch_test_macros.hpp"
#include "gsl/pointers"

namespace tensorcast::store::runtime::ingestion::strategy {
namespace {

using RepresentationWorkItem = materialization::contracts::RepresentationWorkItem;
using RepresentationWorkItemKind = materialization::contracts::RepresentationWorkItemKind;
using RepresentationWorkPlan = materialization::contracts::RepresentationWorkPlan;
using RepresentationWorkSourceFragment = materialization::contracts::RepresentationWorkSourceFragment;
using RepresentationTensorSpec = materialization::contracts::RepresentationTensorSpec;
using SourceFragment = materialization::contracts::SourceFragment;
using TensorAxisRange = materialization::contracts::TensorAxisRange;
using TensorCoordinateSpec = materialization::contracts::TensorCoordinateSpec;
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

void attach_target_layout(ResolvedMaterializationPlan* plan, void* ptr, uint64_t bytes) {
  plan->target_layout = loading::IntoTargetLayout{
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
  strategy_config.owner_file_collective_min_dedup_saving_bytes = 0;
  return strategy_config;
}

SourceBoundSourceFacts safetensors_disk_source() {
  return SourceBoundSourceFacts{.disk_source_available = true, .disk_source_is_safetensors = true};
}

SourceBoundSourceFacts partitioned_disk_source() {
  return SourceBoundSourceFacts{.disk_source_available = true, .disk_source_is_safetensors = false};
}

SourceBoundSourceFacts no_disk_source() {
  return SourceBoundSourceFacts{};
}

TEST_CASE("Source-bound strategy planner emits pure_collective mode", "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kReplicated,
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
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kPureCollective);
  CHECK(strategy_plan_or->lane_plan.require_collective_success);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "pure_collective");
  CHECK(strategy_plan_or->summary.strict_pure_collective_eligible);
  CHECK(strategy_plan_or->summary.planned_collective_admitted_bytes == 8);
}

TEST_CASE(
    "Source-bound strategy planner keeps source-window disabled as diagnostics only",
    "[source_bound_strategy_planner][source_window]") {
  std::array<std::uint8_t, 8> target{};
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kReplicated,
          .committed_bytes = 8,
      });
  work_plan.committed_bytes = 8;
  auto plan = make_plan(std::move(work_plan));
  attach_target_layout(&plan, target.data(), target.size());

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.collective_data_map = make_data_map(8);
  lowering_artifacts.executor_generic_data_map = make_data_map(8);

  auto strategy_config = make_strategy_config();
  strategy_config.enable_source_window_collective = false;

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      strategy_config,
      make_collective_topology(),
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK_FALSE(strategy_plan_or->summary.source_window_collective_candidate);
  CHECK(strategy_plan_or->summary.source_window_candidate_summary.pre_admission_reason == "source_window_disabled");
  CHECK(strategy_plan_or->lane_plan.collective_executor == SourceBoundCollectiveExecutor::kOwnerFile);
}

TEST_CASE(
    "Source-bound strategy planner selects explicit source-window independent of owner-file locality gate",
    "[source_bound_strategy_planner][source_window]") {
  std::array<std::uint8_t, 8> target{};
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kReplicated,
          .committed_bytes = 8,
      });
  work_plan.committed_bytes = 8;
  auto plan = make_plan(std::move(work_plan));
  attach_target_layout(&plan, target.data(), target.size());

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.executor_generic_data_map = make_data_map(8);

  auto strategy_config = make_strategy_config();
  strategy_config.enable_source_window_collective = true;
  strategy_config.executor_preference =
      StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kSourceWindowCollective;
  strategy_config.owner_file_collective_shared_fs_only = true;
  auto topology = make_collective_topology();
  topology.source_locality = loading::SourceLocalityHint::kHostLocal;
  topology.source_sharing_domain.reset();

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      strategy_config,
      topology,
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->summary.source_window_collective_candidate);
  CHECK(strategy_plan_or->summary.source_window_candidate_summary.candidate);
  CHECK_FALSE(strategy_plan_or->summary.source_window_candidate_summary.group_final_admitted);
  CHECK(
      strategy_plan_or->summary.source_window_candidate_summary.pre_admission_reason ==
      "candidate_pending_group_final_admission");
  CHECK(strategy_plan_or->summary.source_window_candidate_summary.source_window_target_write_bytes == 8);
  CHECK(strategy_plan_or->summary.source_window_candidate_summary.source_window_residual_bytes == 0);
  CHECK_FALSE(strategy_plan_or->summary.collective_lane_eligible);
  CHECK(strategy_plan_or->lane_plan.collective_executor == SourceBoundCollectiveExecutor::kSourceWindow);
}

TEST_CASE(
    "Source-bound strategy planner selects source-window when explicitly preferred",
    "[source_bound_strategy_planner][source_window]") {
  std::array<std::uint8_t, 8> target{};
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kDim1Partitioned,
          .committed_bytes = 8,
      });
  work_plan.committed_bytes = 8;
  auto plan = make_plan(std::move(work_plan));
  attach_target_layout(&plan, target.data(), target.size());

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.executor_generic_data_map = make_data_map(8);
  lowering_artifacts.executor_generic_data_map_coverage_only = true;

  auto strategy_config = make_strategy_config();
  strategy_config.enable_source_window_collective = true;
  strategy_config.executor_preference =
      StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kSourceWindowCollective;

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      strategy_config,
      make_collective_topology(),
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->summary.source_window_collective_candidate);
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kSourceWindowCollectiveMixed);
  CHECK(strategy_plan_or->lane_plan.collective_executor == SourceBoundCollectiveExecutor::kSourceWindow);
  CHECK(strategy_plan_or->lane_plan.local_mapped_typed_selected);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "source_window_collective_mixed");
}

TEST_CASE(
    "Source-bound strategy planner selects source-window in enabled auto mode",
    "[source_bound_strategy_planner][source_window]") {
  std::array<std::uint8_t, 8> target{};
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kDim1Partitioned,
          .committed_bytes = 8,
      });
  work_plan.committed_bytes = 8;
  auto plan = make_plan(std::move(work_plan));
  attach_target_layout(&plan, target.data(), target.size());

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.executor_generic_data_map = make_data_map(8);
  lowering_artifacts.executor_generic_data_map_coverage_only = true;

  auto strategy_config = make_strategy_config();
  strategy_config.enable_source_window_collective = true;
  strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto;

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      strategy_config,
      make_collective_topology(),
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->summary.source_window_collective_candidate);
  CHECK(strategy_plan_or->summary.source_window_selection_mode == SourceWindowCollectiveSelectionMode::kAuto);
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kSourceWindowCollectiveMixed);
  CHECK(strategy_plan_or->lane_plan.collective_executor == SourceBoundCollectiveExecutor::kSourceWindow);
  CHECK(strategy_plan_or->lane_plan.local_mapped_typed_selected);
}

TEST_CASE(
    "Source-bound strategy planner keeps host-local auto mode on local mapped typed lane",
    "[source_bound_strategy_planner][source_window]") {
  std::array<std::uint8_t, 8> target{};
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kDim1Partitioned,
          .committed_bytes = 8,
      });
  work_plan.committed_bytes = 8;
  auto plan = make_plan(std::move(work_plan));
  attach_target_layout(&plan, target.data(), target.size());

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.executor_generic_data_map = make_data_map(8);
  lowering_artifacts.executor_generic_data_map_coverage_only = true;

  auto strategy_config = make_strategy_config();
  strategy_config.enable_source_window_collective = true;
  strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto;
  auto topology = make_collective_topology();
  topology.source_locality = loading::SourceLocalityHint::kHostLocal;
  topology.source_sharing_domain.reset();

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      strategy_config,
      topology,
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->summary.source_window_collective_candidate);
  CHECK(strategy_plan_or->summary.source_window_selection_mode == SourceWindowCollectiveSelectionMode::kAuto);
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kLocalMappedTyped);
  CHECK(strategy_plan_or->lane_plan.collective_executor == SourceBoundCollectiveExecutor::kNone);
  CHECK(strategy_plan_or->lane_plan.local_mapped_typed_selected);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "local_mapped_typed");
}

TEST_CASE(
    "Source-bound strategy planner treats source-window as collective under required policy",
    "[source_bound_strategy_planner][source_window]") {
  std::array<std::uint8_t, 8> target{};
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kDim1Partitioned,
          .committed_bytes = 8,
      });
  work_plan.committed_bytes = 8;
  auto plan = make_plan(std::move(work_plan));
  attach_target_layout(&plan, target.data(), target.size());

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.executor_generic_data_map = make_data_map(8);
  lowering_artifacts.executor_generic_data_map_coverage_only = true;

  auto strategy_config = make_strategy_config();
  strategy_config.enable_source_window_collective = true;
  strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kStrict;

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kRequirePureCollective,
      strategy_config,
      make_collective_topology(),
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->summary.source_window_collective_candidate);
  CHECK(strategy_plan_or->summary.source_window_selection_mode == SourceWindowCollectiveSelectionMode::kStrict);
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kSourceWindowCollectiveMixed);
  CHECK(strategy_plan_or->lane_plan.collective_executor == SourceBoundCollectiveExecutor::kSourceWindow);
  CHECK(strategy_plan_or->lane_plan.require_collective_success);
  CHECK_FALSE(strategy_plan_or->summary.strict_pure_collective_eligible);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "source_window_collective_mixed");
}

TEST_CASE("Source-bound strategy planner emits collective_first_mixed mode", "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kReplicated,
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

  auto strategy_config = make_strategy_config();
  strategy_config.owner_file_collective_allow_mixed_residual = true;

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      strategy_config,
      make_collective_topology(),
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kCollectiveFirstMixed);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "collective_first_mixed");
  CHECK(strategy_plan_or->lane_plan.local_pad_bytes == 2);
  CHECK(strategy_plan_or->summary.planned_generic_residual_bytes == 4);
  CHECK(strategy_plan_or->summary.planned_collective_admitted_bytes == 8);
  CHECK_FALSE(strategy_plan_or->lane_plan.reject_reason_buckets.contains("generic_backend_coverage_unproven"));
}

TEST_CASE(
    "Source-bound strategy planner keeps zero-residual-only owner-file collective policy by default",
    "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kReplicated,
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
  lowering_artifacts.executor_generic_data_map = make_data_map(12);

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      make_strategy_config(),
      make_collective_topology(),
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kGenericOnly);
  REQUIRE(strategy_plan_or->lane_plan.reject_reason_buckets.contains("mixed_generic_residual_policy_disabled"));
  CHECK(strategy_plan_or->lane_plan.reject_reason_buckets.at("mixed_generic_residual_policy_disabled") == 4);
}

TEST_CASE(
    "Source-bound strategy planner allows mixed residual when owner-file collective policy opts in",
    "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kReplicated,
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
  lowering_artifacts.executor_generic_data_map = make_data_map(12);

  auto strategy_config = make_strategy_config();
  strategy_config.owner_file_collective_allow_mixed_residual = true;

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      strategy_config,
      make_collective_topology(),
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kCollectiveFirstMixed);
  CHECK_FALSE(strategy_plan_or->lane_plan.reject_reason_buckets.contains("mixed_generic_residual_policy_disabled"));
}

TEST_CASE(
    "Source-bound strategy planner emits generic_only mode when collective is disabled",
    "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kReplicated,
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
      safetensors_disk_source());
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
      no_disk_source());
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
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kLocalMappedTyped);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "local_mapped_typed");
  CHECK(strategy_plan_or->lane_plan.deferred_typed_bytes == 4);
  REQUIRE(strategy_plan_or->lane_plan.reject_reason_buckets.contains("typed_work_without_source_overlap"));
  CHECK(strategy_plan_or->lane_plan.reject_reason_buckets.at("typed_work_without_source_overlap") == 4);
}

TEST_CASE(
    "Source-bound strategy planner uses generic lane for deferred typed work on non-safetensors disk source",
    "[source_bound_strategy_planner]") {
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
      partitioned_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kGenericOnly);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "generic_only");
  CHECK_FALSE(strategy_plan_or->lane_plan.local_mapped_typed_selected);
  REQUIRE(strategy_plan_or->lane_plan.reject_reason_buckets.contains("non_safetensors_source"));
  CHECK(strategy_plan_or->lane_plan.reject_reason_buckets.at("non_safetensors_source") == 4);
}

TEST_CASE(
    "Source-bound strategy planner uses generic lane for deferred typed work only by explicit preference",
    "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kConcatAssemble,
          .committed_bytes = 4,
      });
  auto plan = make_plan(std::move(work_plan));

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.executor_generic_data_map = make_data_map(4);
  auto strategy_config = make_strategy_config();
  strategy_config.executor_preference =
      StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kGenericByteRange;

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      strategy_config,
      make_collective_topology(),
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kGenericOnly);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "generic_only");
  CHECK(strategy_plan_or->lane_plan.deferred_typed_bytes == 4);
}

TEST_CASE(
    "Source-bound strategy planner leaves dim0 concat work on executor lane without source overlap",
    "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  RepresentationWorkItem concat_item;
  concat_item.kind = RepresentationWorkItemKind::kConcatAssemble;
  concat_item.committed_bytes = 8;
  concat_item.sources = {
      RepresentationWorkSourceFragment{
          .fragment =
              SourceFragment{
                  .source_spec =
                      RepresentationTensorSpec{
                          .name = "left",
                          .shape = {4},
                          .stride = {1},
                          .dtype = "torch.float16",
                          .logical_length = 8,
                          .element_size = 2,
                      },
                  .source_range =
                      TensorCoordinateSpec{
                          .axes = {TensorAxisRange{.dim = 0, .start = 0, .end = 4}},
                      },
                  .destination_range =
                      TensorCoordinateSpec{
                          .axes = {TensorAxisRange{.dim = 0, .start = 0, .end = 4}},
                      },
              },
      },
      RepresentationWorkSourceFragment{
          .fragment =
              SourceFragment{
                  .source_spec =
                      RepresentationTensorSpec{
                          .name = "right",
                          .shape = {4},
                          .stride = {1},
                          .dtype = "torch.float16",
                          .logical_length = 8,
                          .element_size = 2,
                      },
                  .source_range =
                      TensorCoordinateSpec{
                          .axes = {TensorAxisRange{.dim = 0, .start = 0, .end = 4}},
                      },
                  .destination_range =
                      TensorCoordinateSpec{
                          .axes = {TensorAxisRange{.dim = 0, .start = 4, .end = 8}},
                      },
              },
      },
  };
  work_plan.items.push_back(std::move(concat_item));
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
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->summary.planned_collective_candidate_bytes == 0);
  CHECK(strategy_plan_or->summary.planned_non_admitted_typed_bytes == 8);
  CHECK_FALSE(strategy_plan_or->summary.planner_reject_reason_buckets.contains("concat_not_admitted"));
  REQUIRE(strategy_plan_or->summary.planner_reject_reason_buckets.contains("typed_work_without_source_overlap"));
  CHECK(strategy_plan_or->summary.planner_reject_reason_buckets.at("typed_work_without_source_overlap") == 8);
}

TEST_CASE(
    "Source-bound strategy planner keeps concat work local even when collective lane covers its target range",
    "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  RepresentationWorkItem concat_item;
  concat_item.kind = RepresentationWorkItemKind::kConcatAssemble;
  concat_item.committed_bytes = 8;
  concat_item.dst_name = "concat";
  concat_item.dst_spec = RepresentationTensorSpec{
      .name = "concat",
      .shape = {4},
      .stride = {1},
      .dtype = "torch.float16",
      .logical_offset = 0,
      .logical_length = 8,
      .element_size = 2,
  };
  concat_item.sources = {
      RepresentationWorkSourceFragment{
          .fragment =
              SourceFragment{
                  .source_spec =
                      RepresentationTensorSpec{
                          .name = "left",
                          .shape = {2},
                          .stride = {1},
                          .dtype = "torch.float16",
                          .logical_length = 4,
                          .element_size = 2,
                      },
                  .source_range =
                      TensorCoordinateSpec{
                          .axes = {TensorAxisRange{.dim = 0, .start = 0, .end = 2}},
                      },
                  .destination_range =
                      TensorCoordinateSpec{
                          .axes = {TensorAxisRange{.dim = 0, .start = 0, .end = 2}},
                      },
              },
      },
      RepresentationWorkSourceFragment{
          .fragment =
              SourceFragment{
                  .source_spec =
                      RepresentationTensorSpec{
                          .name = "right",
                          .shape = {2},
                          .stride = {1},
                          .dtype = "torch.float16",
                          .logical_length = 4,
                          .element_size = 2,
                      },
                  .source_range =
                      TensorCoordinateSpec{
                          .axes = {TensorAxisRange{.dim = 0, .start = 0, .end = 2}},
                      },
                  .destination_range =
                      TensorCoordinateSpec{
                          .axes = {TensorAxisRange{.dim = 0, .start = 2, .end = 4}},
                      },
              },
      },
  };
  work_plan.items.push_back(std::move(concat_item));
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
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kLocalMappedTyped);
  CHECK(strategy_plan_or->summary.planned_collective_candidate_bytes == 0);
  CHECK(strategy_plan_or->summary.planned_non_admitted_typed_bytes == 8);
  CHECK(strategy_plan_or->summary.planned_collective_admitted_bytes == 0);
  CHECK(strategy_plan_or->summary.estimated_collective_dedup_saving_bytes == 0);
  REQUIRE(strategy_plan_or->summary.planner_reject_reason_buckets.contains("typed_work_without_source_overlap"));
  CHECK(strategy_plan_or->summary.planner_reject_reason_buckets.at("typed_work_without_source_overlap") == 8);
}

TEST_CASE(
    "Source-bound strategy planner keeps expert concat work local even when collective lane covers destination spans",
    "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  RepresentationWorkItem expert_item;
  expert_item.kind = RepresentationWorkItemKind::kExpertDim0Concat;
  expert_item.partition_kind = WorkPartitionKind::kUnknown;
  expert_item.committed_bytes = 4;
  expert_item.dst_name = "expert";
  expert_item.dst_spec = RepresentationTensorSpec{
      .name = "expert",
      .shape = {2, 2},
      .stride = {2, 1},
      .dtype = "torch.float16",
      .logical_offset = 0,
      .logical_length = 8,
      .element_size = 2,
  };
  expert_item.sources = {
      RepresentationWorkSourceFragment{
          .fragment =
              SourceFragment{
                  .source_spec =
                      RepresentationTensorSpec{
                          .name = "expert_src",
                          .shape = {1, 2},
                          .stride = {2, 1},
                          .dtype = "torch.float16",
                          .logical_length = 4,
                          .element_size = 2,
                      },
                  .source_range =
                      TensorCoordinateSpec{
                          .axes = {TensorAxisRange{.dim = 0, .start = 0, .end = 1}},
                      },
                  .destination_range =
                      TensorCoordinateSpec{
                          .axes =
                              {
                                  TensorAxisRange{.dim = 0, .start = 0, .end = 1},
                                  TensorAxisRange{.dim = 1, .start = 0, .end = 2},
                              },
                      },
              },
      },
  };
  work_plan.items.push_back(std::move(expert_item));
  auto plan = make_plan(std::move(work_plan));

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.collective_data_map = make_data_map(4);
  lowering_artifacts.executor_generic_data_map = make_data_map(4);

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      make_strategy_config(),
      make_collective_topology(),
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kLocalMappedTyped);
  CHECK(strategy_plan_or->summary.planned_collective_candidate_bytes == 0);
  CHECK(strategy_plan_or->summary.planned_non_admitted_typed_bytes == 4);
  CHECK(strategy_plan_or->summary.planned_collective_admitted_bytes == 0);
  CHECK(strategy_plan_or->summary.estimated_collective_dedup_saving_bytes == 0);
  REQUIRE(strategy_plan_or->summary.planner_reject_reason_buckets.contains("typed_work_without_source_overlap"));
  CHECK(strategy_plan_or->summary.planner_reject_reason_buckets.at("typed_work_without_source_overlap") == 4);
}

TEST_CASE(
    "Source-bound strategy planner keeps expert dim0 concat on executor lane without source overlap",
    "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kExpertDim0Concat,
          .partition_kind = WorkPartitionKind::kUnknown,
          .committed_bytes = 128,
      });
  auto plan = make_plan(std::move(work_plan));

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.executor_generic_data_map = make_data_map(128);

  auto strategy_config = make_strategy_config();
  strategy_config.owner_file_collective_min_dedup_saving_bytes = 1;

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      strategy_config,
      make_collective_topology(),
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->summary.planned_collective_candidate_bytes == 0);
  CHECK(strategy_plan_or->summary.planned_non_admitted_typed_bytes == 128);
  CHECK(strategy_plan_or->summary.estimated_collective_dedup_saving_bytes == 0);
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kLocalMappedTyped);
  REQUIRE(strategy_plan_or->summary.planner_reject_reason_buckets.contains("typed_work_without_source_overlap"));
  CHECK(strategy_plan_or->summary.planner_reject_reason_buckets.at("typed_work_without_source_overlap") == 128);
}

TEST_CASE(
    "Source-bound strategy planner does not admit below-threshold owner-file overlap into collective",
    "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kReplicated,
          .committed_bytes = 8,
      });
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kDim1Partitioned,
          .committed_bytes = 128,
      });
  auto plan = make_plan(std::move(work_plan));

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.collective_data_map = make_data_map(8);
  lowering_artifacts.executor_generic_data_map = make_data_map(136);

  auto strategy_config = make_strategy_config();
  strategy_config.owner_file_collective_min_dedup_saving_bytes = 1024;

  auto strategy_plan_or = build_source_bound_execution_strategy_plan(
      plan,
      lowering_artifacts,
      SourceBoundPolicy::kCollectiveFirst,
      strategy_config,
      make_collective_topology(),
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK_FALSE(strategy_plan_or->summary.collective_lane_eligible);
  CHECK(strategy_plan_or->summary.planned_collective_candidate_bytes == 8);
  CHECK(strategy_plan_or->summary.planned_collective_admitted_bytes == 0);
  CHECK(strategy_plan_or->lane_plan.deferred_typed_bytes == 128);
  CHECK(strategy_plan_or->lane_plan.selection_reason == "generic_only;local_mapped_typed");
  REQUIRE(
      strategy_plan_or->summary.planner_reject_reason_buckets.contains("collective_source_overlap_below_threshold"));
  CHECK(strategy_plan_or->summary.planner_reject_reason_buckets.at("collective_source_overlap_below_threshold") == 8);
  REQUIRE(strategy_plan_or->summary.planner_reject_reason_buckets.contains("typed_work_without_source_overlap"));
  CHECK(strategy_plan_or->summary.planner_reject_reason_buckets.at("typed_work_without_source_overlap") == 128);
}

TEST_CASE(
    "Source-bound strategy planner emits reject mode for impossible strict collective plan",
    "[source_bound_strategy_planner]") {
  RepresentationWorkPlan work_plan;
  work_plan.items.push_back(
      RepresentationWorkItem{
          .kind = RepresentationWorkItemKind::kTensorCopy,
          .partition_kind = WorkPartitionKind::kReplicated,
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
      safetensors_disk_source());
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
          .partition_kind = WorkPartitionKind::kReplicated,
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
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(strategy_plan_or->lane_plan.mode == SourceBoundExecutionMode::kRejected);
  CHECK(strategy_plan_or->summary.execution_plan_kind == "reject");
  REQUIRE(strategy_plan_or->summary.planner_reject_reason_buckets.contains("generic_backend_coverage_unproven"));
  CHECK(strategy_plan_or->summary.planner_reject_reason_buckets.at("generic_backend_coverage_unproven") == 4);
}

} // namespace
} // namespace tensorcast::store::runtime::ingestion::strategy
