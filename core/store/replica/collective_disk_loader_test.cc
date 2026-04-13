// Copyright (c) 2026, TensorCast Team.

#include "core/store/replica/collective_disk_loader.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

#include "catch2/catch_test_macros.hpp"

namespace tensorcast::store::replica {

namespace {

std::shared_ptr<const loader::DiskArtifactContext> make_disk_context() {
  return std::make_shared<loader::DiskArtifactContext>(
      std::filesystem::path("artifact"),
      std::vector<std::filesystem::path>{},
      std::vector<size_t>{},
      /*total_size=*/16,
      /*is_safetensors=*/true,
      /*descriptor_present=*/false,
      /*tensor_index_json_present=*/false,
      /*tensor_index_cbor_present=*/false,
      std::vector<loader::SharedSafetensorsSegment>{});
}

} // namespace

TEST_CASE(
    "try_local_batched_disk_load falls back when representation work is unavailable",
    "[collective_disk_loader][local_batched][fallback]") {
  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_local_batched_disk_load = true;

  LocalBatchedDiskLoadRequest request{
      .replica_key = loading::ReplicaKey{.artifact_id = "artifact_local_batched"},
      .disk_context = make_disk_context(),
      .strategy_config = strategy,
      .gpu_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(1)),
      .device_id = 0,
  };

  std::shared_ptr<common::memory::PinnedBufferPool> pinned_pool;
  const auto result = try_local_batched_disk_load(request, pinned_pool, std::chrono::milliseconds(0));

  REQUIRE_FALSE(result.handled);
  REQUIRE(result.status.ok());
  REQUIRE(result.skip_reason == "missing_prerequisites");
}

TEST_CASE(
    "summarize_local_batched_disk_load reports direct dedup savings for identical source slices",
    "[collective_disk_loader][local_batched][summary]") {
  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_local_batched_disk_load = true;

  materialization::contracts::RepresentationWorkPlan work_plan;
  auto make_tensor_copy = [](std::string_view dst_name,
                             uint64_t dst_offset) -> materialization::contracts::RepresentationWorkItem {
    materialization::contracts::RepresentationWorkItem item;
    item.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
    item.partition_kind = materialization::contracts::WorkPartitionKind::kReplicated;
    item.dst_name = std::string(dst_name);
    item.dst_spec = materialization::contracts::RepresentationTensorSpec{
        .name = std::string(dst_name),
        .shape = {8},
        .stride = {1},
        .dtype = "torch.uint8",
        .logical_offset = dst_offset,
        .logical_length = 8,
        .storage_offset = 0,
        .element_size = 1,
    };
    item.committed_bytes = 8;
    item.sources.push_back(
        materialization::contracts::RepresentationWorkSourceFragment{
            .fragment =
                materialization::contracts::SourceFragment{
                    .source_spec =
                        materialization::contracts::RepresentationTensorSpec{
                            .name = "source",
                            .shape = {8},
                            .stride = {1},
                            .dtype = "torch.uint8",
                            .logical_offset = 16,
                            .logical_length = 8,
                            .storage_offset = 0,
                            .element_size = 1,
                        },
                },
        });
    return item;
  };
  work_plan.items.push_back(make_tensor_copy("alpha", 0));
  work_plan.items.push_back(make_tensor_copy("beta", 32));

  auto summary_or = summarize_local_batched_disk_load(work_plan, strategy);
  REQUIRE(summary_or.ok());
  CHECK(summary_or->eligible);
  CHECK(summary_or->reason == "eligible");
  CHECK(summary_or->requested_source_bytes == 16);
  CHECK(summary_or->unique_source_bytes == 8);
  CHECK(summary_or->dedup_saving_bytes == 8);
  CHECK(summary_or->direct_dedup_copy_bytes == 8);
  CHECK(summary_or->batch_count == 1);
}

TEST_CASE(
    "summarize_local_batched_disk_load rejects unsupported tensor jobs deterministically",
    "[collective_disk_loader][local_batched][summary]") {
  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_local_batched_disk_load = true;

  materialization::contracts::RepresentationWorkPlan work_plan;
  materialization::contracts::RepresentationWorkItem item;
  item.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
  item.partition_kind = materialization::contracts::WorkPartitionKind::kUnknown;
  item.dst_name = "gamma";
  item.dst_spec = materialization::contracts::RepresentationTensorSpec{
      .name = "gamma",
      .shape = {8},
      .stride = {1},
      .dtype = "torch.uint8",
      .logical_offset = 0,
      .logical_length = 8,
      .storage_offset = 0,
      .element_size = 1,
  };
  item.committed_bytes = 8;
  item.sources.push_back(
      materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              materialization::contracts::SourceFragment{
                  .source_spec =
                      materialization::contracts::RepresentationTensorSpec{
                          .name = "gamma",
                          .shape = {8},
                          .stride = {1},
                          .dtype = "torch.uint8",
                          .logical_offset = 0,
                          .logical_length = 8,
                          .storage_offset = 0,
                          .element_size = 1,
                      },
              },
      });
  work_plan.items.push_back(std::move(item));

  auto summary_or = summarize_local_batched_disk_load(work_plan, strategy);
  REQUIRE(summary_or.ok());
  CHECK_FALSE(summary_or->eligible);
  CHECK(summary_or->reason == "unsupported_tensor_jobs");
}

TEST_CASE(
    "try_collective_mapped_target_load accepts local pad work when the collective lane map is data-only",
    "[collective_disk_loader][collective_mapped][fallback]") {
  CollectiveMappedTargetLoadRequest request{
      .artifact_id = "artifact_collective_mapped",
      .group =
          loading::CollectiveLoadGroupHint{
              .group_id = "mapped-group",
              .world_size = 8,
              .rank = 0,
          },
      .disk_context = make_disk_context(),
      .representation_work_plan =
          materialization::contracts::RepresentationWorkPlan{
              .items =
                  {
                      materialization::contracts::RepresentationWorkItem{
                          .kind = materialization::contracts::RepresentationWorkItemKind::kPadFill,
                          .byte_range_map =
                              loader::ByteRangeMap{
                                  .total_bytes = 16,
                                  .num_sources = 1,
                                  .segments =
                                      {
                                          loader::ByteRangeSegment{
                                              .kind = loader::ByteRangeSegment::Kind::kPad,
                                              .dst_offset = 8,
                                              .length = 8,
                                              .src_offset = 0,
                                              .source_index = 0,
                                          },
                                      },
                              },
                          .committed_bytes = 8,
                      },
                  },
          },
      .collective_lane_map =
          loader::ByteRangeMap{
              .total_bytes = 16,
              .num_sources = 1,
              .segments =
                  {
                      loader::ByteRangeSegment{
                          .kind = loader::ByteRangeSegment::Kind::kData,
                          .dst_offset = 0,
                          .length = 8,
                          .src_offset = 0,
                          .source_index = 0,
                      },
                  },
          },
      .target_layout =
          loading::IntoTargetLayout{
              .storages =
                  {
                      loading::IntoTargetStorage{
                          .base_ptr = gsl::not_null<void*>{reinterpret_cast<void*>(0x1)},
                          .length = 16,
                      },
                  },
              .total_size = 16,
          },
      .device_id = 0,
  };

  std::shared_ptr<common::memory::PinnedBufferPool> pinned_pool;
  const auto result = try_collective_mapped_target_load(
      request, pinned_pool, std::chrono::milliseconds(0), CollectiveMappedTargetLoadOptions{});

  REQUIRE_FALSE(result.handled);
  REQUIRE(result.status.ok());
}

TEST_CASE(
    "try_collective_mapped_target_load accepts local pad work when the collective lane map is data-only",
    "[collective_disk_loader][collective_mapped][fallback]") {
  CollectiveMappedTargetLoadRequest request{
      .artifact_id = "artifact_collective_mapped",
      .group =
          loading::CollectiveLoadGroupHint{
              .group_id = "mapped-group",
              .world_size = 8,
              .rank = 0,
          },
      .disk_context = make_disk_context(),
      .representation_work_plan =
          materialization::contracts::RepresentationWorkPlan{
              .items =
                  {
                      materialization::contracts::RepresentationWorkItem{
                          .kind = materialization::contracts::RepresentationWorkItemKind::kPadFill,
                          .byte_range_map =
                              loader::ByteRangeMap{
                                  .total_bytes = 16,
                                  .num_sources = 1,
                                  .segments =
                                      {
                                          loader::ByteRangeSegment{
                                              .kind = loader::ByteRangeSegment::Kind::kPad,
                                              .dst_offset = 8,
                                              .length = 8,
                                              .src_offset = 0,
                                              .source_index = 0,
                                          },
                                      },
                              },
                          .committed_bytes = 8,
                      },
                  },
          },
      .collective_lane_map =
          loader::ByteRangeMap{
              .total_bytes = 16,
              .num_sources = 1,
              .segments =
                  {
                      loader::ByteRangeSegment{
                          .kind = loader::ByteRangeSegment::Kind::kData,
                          .dst_offset = 0,
                          .length = 8,
                          .src_offset = 0,
                          .source_index = 0,
                      },
                  },
          },
      .target_layout =
          loading::IntoTargetLayout{
              .storages =
                  {
                      loading::IntoTargetStorage{
                          .base_ptr = gsl::not_null<void*>{reinterpret_cast<void*>(0x1)},
                          .length = 16,
                      },
                  },
              .total_size = 16,
          },
      .device_id = 0,
  };

  std::shared_ptr<common::memory::PinnedBufferPool> pinned_pool;
  const auto result = try_collective_mapped_target_load(
      request, pinned_pool, std::chrono::milliseconds(0), CollectiveMappedTargetLoadOptions{});

  REQUIRE_FALSE(result.handled);
  REQUIRE(result.status.ok());
}

} // namespace tensorcast::store::replica
