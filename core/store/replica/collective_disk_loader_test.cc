// Copyright (c) 2026, TensorCast Team.

#include "core/store/replica/collective_disk_loader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/cuda/cuda_api.h"
#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"
#include "core/testing/test_helpers.h"

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

std::filesystem::path make_temp_dir(const std::string& prefix) {
  auto dir = std::filesystem::temp_directory_path() /
      (prefix + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(dir);
  return dir;
}

void write_u64_le(std::ofstream& out, uint64_t value) {
  std::array<unsigned char, 8> buffer{};
  for (int index = 0; index < 8; ++index) {
    buffer[static_cast<size_t>(index)] = static_cast<unsigned char>((value >> (8 * index)) & 0xFF);
  }
  out.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
}

void create_safetensors_file(
    const std::filesystem::path& path,
    const std::string& header_json,
    const std::vector<unsigned char>& payload) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  REQUIRE(out.is_open());
  write_u64_le(out, static_cast<uint64_t>(header_json.size()));
  out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
  out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
}

materialization::contracts::RepresentationTensorSpec make_tensor_spec(
    std::string_view name,
    uint64_t offset,
    std::vector<int64_t> shape,
    std::vector<int64_t> stride,
    std::string_view dtype,
    uint64_t element_size) {
  uint64_t elements = 1;
  for (const int64_t dim : shape) {
    elements *= static_cast<uint64_t>(dim);
  }
  return materialization::contracts::RepresentationTensorSpec{
      .name = std::string(name),
      .shape = std::move(shape),
      .stride = std::move(stride),
      .dtype = std::string(dtype),
      .logical_offset = offset,
      .logical_length = elements * element_size,
      .storage_offset = 0,
      .element_size = element_size,
  };
}

materialization::contracts::RepresentationTensorSpec make_u8_tensor_spec(
    std::string_view name,
    uint64_t offset,
    std::vector<int64_t> shape,
    std::vector<int64_t> stride) {
  return make_tensor_spec(name, offset, std::move(shape), std::move(stride), "torch.uint8", 1);
}

materialization::contracts::TensorCoordinateSpec rect_range(
    int64_t row_begin,
    int64_t row_end,
    int64_t col_begin,
    int64_t col_end) {
  return materialization::contracts::TensorCoordinateSpec{
      .axes =
          {
              materialization::contracts::TensorAxisRange{.dim = 0, .start = row_begin, .end = row_end},
              materialization::contracts::TensorAxisRange{.dim = 1, .start = col_begin, .end = col_end},
          },
  };
}

std::string torch_dtype_for_element_size(uint64_t element_size) {
  if (element_size == 1) {
    return "torch.uint8";
  }
  if (element_size == 2) {
    return "torch.int16";
  }
  return "torch.float32";
}

std::string safetensors_dtype_for_element_size(uint64_t element_size) {
  if (element_size == 1) {
    return "U8";
  }
  if (element_size == 2) {
    return "I16";
  }
  return "F32";
}

std::vector<unsigned char> make_sequential_payload(uint64_t bytes) {
  std::vector<unsigned char> payload(bytes);
  for (uint64_t index = 0; index < bytes; ++index) {
    payload[static_cast<size_t>(index)] = static_cast<unsigned char>((0x20 + index) & 0xFF);
  }
  return payload;
}

struct RectCopyCase {
  int64_t src_row_begin;
  int64_t src_row_end;
  int64_t src_col_begin;
  int64_t src_col_end;
  int64_t dst_row_begin;
  int64_t dst_row_end;
  int64_t dst_col_begin;
  int64_t dst_col_end;
};

void append_rect_segments(
    const RectCopyCase& copy,
    uint64_t dst_tensor_offset,
    uint64_t dst_cols,
    uint64_t element_size,
    std::vector<loader::ByteRangeSegment>* segments) {
  const uint64_t col_bytes = static_cast<uint64_t>(copy.dst_col_end - copy.dst_col_begin) * element_size;
  for (int64_t row = copy.dst_row_begin; row < copy.dst_row_end; ++row) {
    const uint64_t dst_offset = dst_tensor_offset +
        (static_cast<uint64_t>(row) * dst_cols + static_cast<uint64_t>(copy.dst_col_begin)) * element_size;
    segments->push_back(
        loader::ByteRangeSegment{
            .kind = loader::ByteRangeSegment::Kind::kData,
            .dst_offset = dst_offset,
            .length = col_bytes,
            .src_offset = 0,
            .source_index = 0,
        });
  }
}

void apply_rect_copy_to_expected(
    const RectCopyCase& copy,
    uint64_t src_cols,
    uint64_t dst_cols,
    uint64_t dst_tensor_offset,
    uint64_t element_size,
    const std::vector<unsigned char>& payload,
    std::vector<uint8_t>* expected) {
  const uint64_t row_bytes = static_cast<uint64_t>(copy.src_col_end - copy.src_col_begin) * element_size;
  for (int64_t row = 0; row < copy.src_row_end - copy.src_row_begin; ++row) {
    const uint64_t src_offset =
        (static_cast<uint64_t>(copy.src_row_begin + row) * src_cols + static_cast<uint64_t>(copy.src_col_begin)) *
        element_size;
    const uint64_t dst_offset = dst_tensor_offset +
        (static_cast<uint64_t>(copy.dst_row_begin + row) * dst_cols + static_cast<uint64_t>(copy.dst_col_begin)) *
            element_size;
    std::copy_n(
        payload.begin() + static_cast<std::ptrdiff_t>(src_offset),
        static_cast<std::ptrdiff_t>(row_bytes),
        expected->begin() + static_cast<std::ptrdiff_t>(dst_offset));
  }
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
    "try_local_mapped_target_load rejects wide dim1 rows before execution",
    "[collective_disk_loader][local_mapped][admission]") {
  SKIP_IF_NO_CUDA();

  constexpr uint64_t kRows = 2;
  constexpr uint64_t kCols = 8;
  constexpr uint64_t kDstCols = 4;
  constexpr uint64_t kSourceBytes = kRows * kCols;
  constexpr uint64_t kTargetBytes = kRows * kDstCols;
  auto temp_root = make_temp_dir("collective-disk-loader-wide-dim1");
  const auto safetensors_path = temp_root / "weights.safetensors";
  std::vector<unsigned char> payload(kSourceBytes, 0xAB);
  create_safetensors_file(
      safetensors_path, "{\"src\":{\"dtype\":\"U8\",\"shape\":[2,8],\"data_offsets\":[0,16]}}", payload);

  auto disk_context_or = loader::get_disk_artifact_context(temp_root);
  REQUIRE(disk_context_or.ok());

  materialization::contracts::RepresentationWorkItem item;
  item.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
  item.partition_kind = materialization::contracts::WorkPartitionKind::kDim1Partitioned;
  item.dst_name = "dst";
  item.dst_spec = make_u8_tensor_spec("dst", 0, {2, 4}, {4, 1});
  item.committed_bytes = kTargetBytes;
  item.sources.push_back(
      materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              materialization::contracts::SourceFragment{
                  .source_spec = make_u8_tensor_spec("src", 0, {2, 8}, {8, 1}),
                  .source_range =
                      materialization::contracts::TensorCoordinateSpec{
                          .axes = {materialization::contracts::TensorAxisRange{.dim = 1, .start = 0, .end = 4}},
                      },
                  .destination_range =
                      materialization::contracts::TensorCoordinateSpec{
                          .axes = {materialization::contracts::TensorAxisRange{.dim = 1, .start = 0, .end = 4}},
                      },
              },
      });

  loader::ByteRangeMap data_lane_map;
  data_lane_map.total_bytes = kTargetBytes;
  data_lane_map.num_sources = 1;
  data_lane_map.segments = {
      loader::ByteRangeSegment{
          .kind = loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = kTargetBytes,
          .src_offset = 0,
          .source_index = 0,
      },
  };

  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_tensor_aware_mapped_executor = true;
  strategy.executor_preference =
      StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kTensorAwareLocal;

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(4, 4);
  const auto result = try_local_mapped_target_load(
      LocalMappedTargetLoadRequest{
          .artifact_id = "artifact_wide_dim1",
          .disk_context = *disk_context_or,
          .representation_work_plan = materialization::contracts::RepresentationWorkPlan{.items = {item}},
          .data_lane_map = data_lane_map,
          .target_layout =
              loading::IntoTargetLayout{
                  .storages =
                      {
                          loading::IntoTargetStorage{
                              .base_ptr = gsl::not_null<void*>{reinterpret_cast<void*>(0x1)},
                              .length = kTargetBytes,
                          },
                      },
                  .total_size = kTargetBytes,
              },
          .strategy_config = strategy,
          .device_id = 0,
      },
      pinned_pool,
      std::chrono::milliseconds(1000),
      CollectiveMappedTargetLoadOptions{.chunk_bytes = 4, .strategy_config = strategy});

  REQUIRE(result.handled);
  REQUIRE_FALSE(result.status.ok());
  CHECK(absl::IsFailedPrecondition(result.status));
  CHECK(result.status.message().find("row exceeds pinned buffer size") != std::string_view::npos);

  loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "try_local_mapped_target_load copies multi-range concat blocks with small buffer",
    "[collective_disk_loader][local_mapped][concat][correctness]") {
  SKIP_IF_NO_CUDA();

  constexpr uint64_t kSourceBytes = 8;
  constexpr uint64_t kTargetBytes = 8;
  auto temp_root = make_temp_dir("collective-disk-loader-multirange-concat");
  const auto safetensors_path = temp_root / "weights.safetensors";
  std::vector<unsigned char> payload(kSourceBytes);
  for (uint64_t index = 0; index < kSourceBytes; ++index) {
    payload[static_cast<size_t>(index)] = static_cast<unsigned char>(0x20 + index);
  }
  create_safetensors_file(
      safetensors_path, "{\"src\":{\"dtype\":\"U8\",\"shape\":[4,2],\"data_offsets\":[0,8]}}", payload);

  auto disk_context_or = loader::get_disk_artifact_context(temp_root);
  REQUIRE(disk_context_or.ok());

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, kTargetBytes).ok());
  std::array<uint8_t, kTargetBytes> initial{};
  initial.fill(0xEE);
  REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  materialization::contracts::RepresentationWorkItem item;
  item.kind = materialization::contracts::RepresentationWorkItemKind::kConcatAssemble;
  item.dst_name = "dst";
  item.dst_spec = make_u8_tensor_spec("dst", 0, {8}, {1});
  item.committed_bytes = kTargetBytes;
  item.sources.push_back(
      materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              materialization::contracts::SourceFragment{
                  .source_spec = make_u8_tensor_spec("src", 0, {4, 2}, {2, 1}),
                  .source_range =
                      materialization::contracts::TensorCoordinateSpec{
                          .axes = {materialization::contracts::TensorAxisRange{.dim = 0, .start = 0, .end = 2}},
                      },
                  .destination_range =
                      materialization::contracts::TensorCoordinateSpec{
                          .axes = {materialization::contracts::TensorAxisRange{.dim = 0, .start = 0, .end = 8}},
                      },
              },
          .prefix_count = 2,
          .dst_block_offset_bytes = 0,
          .dst_block_stride_bytes = 4,
          .dst_block_bytes = 4,
      });

  loader::ByteRangeMap data_lane_map;
  data_lane_map.total_bytes = kTargetBytes;
  data_lane_map.num_sources = 1;
  data_lane_map.segments = {
      loader::ByteRangeSegment{
          .kind = loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = kTargetBytes,
          .src_offset = 0,
          .source_index = 0,
      },
  };

  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_tensor_aware_mapped_executor = true;
  strategy.enable_mapped_concat_jobs = true;
  strategy.enable_mapped_concat_execution = true;
  strategy.executor_preference =
      StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kTensorAwareLocal;

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(4, 4);
  const auto result = try_local_mapped_target_load(
      LocalMappedTargetLoadRequest{
          .artifact_id = "artifact_multirange_concat",
          .disk_context = *disk_context_or,
          .representation_work_plan = materialization::contracts::RepresentationWorkPlan{.items = {item}},
          .data_lane_map = data_lane_map,
          .target_layout =
              loading::IntoTargetLayout{
                  .storages =
                      {
                          loading::IntoTargetStorage{
                              .base_ptr = gsl::not_null<void*>{gpu_buffer},
                              .length = kTargetBytes,
                          },
                      },
                  .total_size = kTargetBytes,
              },
          .strategy_config = strategy,
          .device_id = 0,
      },
      pinned_pool,
      std::chrono::milliseconds(1000),
      CollectiveMappedTargetLoadOptions{.chunk_bytes = 4, .strategy_config = strategy});

  REQUIRE(result.handled);
  REQUIRE(result.status.ok());
  CHECK(result.handled_bytes == kTargetBytes);
  CHECK(result.residual_data_map.segments.empty());

  std::array<uint8_t, kTargetBytes> actual{};
  REQUIRE(tensorcast::cuda::memcpy(actual.data(), gpu_buffer, actual.size(), cudaMemcpyDeviceToHost).ok());
  std::array<uint8_t, kTargetBytes> expected{};
  for (uint64_t index = 0; index < kTargetBytes; ++index) {
    expected[static_cast<size_t>(index)] = static_cast<uint8_t>(0x20 + index);
  }
  CHECK(actual == expected);

  REQUIRE(tensorcast::cuda::free(gpu_buffer).ok());
  loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "try_local_mapped_target_load copies rect2d partial tensor bytes exactly",
    "[collective_disk_loader][local_mapped][rect2d][correctness]") {
  SKIP_IF_NO_CUDA();
  const char* cuda_backend = std::getenv("TENSORCAST_CUDA_BACKEND");
  if (cuda_backend != nullptr && std::string_view(cuda_backend) == "fake") {
    SKIP("rect2d byte-equivalence requires real CUDA because the executor uses cudaMemcpy2DAsync");
  }

  constexpr uint64_t kRows = 4;
  constexpr uint64_t kCols = 6;
  constexpr uint64_t kTotalBytes = kRows * kCols;
  auto temp_root = make_temp_dir("collective-disk-loader-rect2d");
  const auto safetensors_path = temp_root / "weights.safetensors";
  std::vector<unsigned char> payload(kTotalBytes);
  for (uint64_t index = 0; index < kTotalBytes; ++index) {
    payload[static_cast<size_t>(index)] = static_cast<unsigned char>(index);
  }
  create_safetensors_file(
      safetensors_path, "{\"src\":{\"dtype\":\"U8\",\"shape\":[4,6],\"data_offsets\":[0,24]}}", payload);

  auto disk_context_or = loader::get_disk_artifact_context(temp_root);
  REQUIRE(disk_context_or.ok());
  REQUIRE((*disk_context_or)->is_safetensors());

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, kTotalBytes).ok());
  std::array<uint8_t, kTotalBytes> initial{};
  initial.fill(0xEE);
  REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  materialization::contracts::RepresentationWorkItem item;
  item.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
  item.partition_kind = materialization::contracts::WorkPartitionKind::kUnknown;
  item.dst_name = "dst";
  item.dst_spec = make_u8_tensor_spec("dst", 0, {4, 6}, {6, 1});
  item.committed_bytes = 6;
  item.sources.push_back(
      materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              materialization::contracts::SourceFragment{
                  .source_spec = make_u8_tensor_spec("src", 0, {4, 6}, {6, 1}),
                  .source_range = rect_range(1, 3, 2, 5),
                  .destination_range = rect_range(0, 2, 1, 4),
              },
      });

  loader::ByteRangeMap data_lane_map;
  data_lane_map.total_bytes = kTotalBytes;
  data_lane_map.num_sources = 1;
  data_lane_map.segments = {
      loader::ByteRangeSegment{
          .kind = loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 1,
          .length = 3,
          .src_offset = 0,
          .source_index = 0,
      },
      loader::ByteRangeSegment{
          .kind = loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 7,
          .length = 3,
          .src_offset = 0,
          .source_index = 0,
      },
  };

  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_tensor_aware_mapped_executor = true;
  strategy.allow_mixed_execution = true;
  strategy.enable_mapped_concat_jobs = true;
  strategy.enable_mapped_concat_execution = true;

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(64, 64);
  const auto result = try_local_mapped_target_load(
      LocalMappedTargetLoadRequest{
          .artifact_id = "artifact_rect2d",
          .disk_context = *disk_context_or,
          .representation_work_plan = materialization::contracts::RepresentationWorkPlan{.items = {item}},
          .data_lane_map = data_lane_map,
          .target_layout =
              loading::IntoTargetLayout{
                  .storages =
                      {
                          loading::IntoTargetStorage{
                              .base_ptr = gsl::not_null<void*>{gpu_buffer},
                              .length = kTotalBytes,
                          },
                      },
                  .total_size = kTotalBytes,
              },
          .strategy_config = strategy,
          .device_id = 0,
      },
      pinned_pool,
      std::chrono::milliseconds(1000),
      CollectiveMappedTargetLoadOptions{.chunk_bytes = 64, .strategy_config = strategy});

  REQUIRE(result.handled);
  REQUIRE(result.status.ok());
  CHECK(result.handled_bytes == 6);
  CHECK(result.residual_data_map.segments.empty());
  CHECK(result.residual_data_map.total_bytes == kTotalBytes);

  std::array<uint8_t, kTotalBytes> actual{};
  REQUIRE(tensorcast::cuda::memcpy(actual.data(), gpu_buffer, actual.size(), cudaMemcpyDeviceToHost).ok());
  std::array<uint8_t, kTotalBytes> expected = initial;
  expected[1] = payload[8];
  expected[2] = payload[9];
  expected[3] = payload[10];
  expected[7] = payload[14];
  expected[8] = payload[15];
  expected[9] = payload[16];
  CHECK(actual == expected);

  REQUIRE(tensorcast::cuda::free(gpu_buffer).ok());
  loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "try_local_mapped_target_load covers rect2d geometry and dtype matrix without residual",
    "[collective_disk_loader][local_mapped][rect2d][correctness]") {
  SKIP_IF_NO_CUDA();
  const char* cuda_backend = std::getenv("TENSORCAST_CUDA_BACKEND");
  if (cuda_backend != nullptr && std::string_view(cuda_backend) == "fake") {
    SKIP("rect2d byte-equivalence requires real CUDA because the executor uses cudaMemcpy2DAsync");
  }

  constexpr uint64_t kSrcRows = 6;
  constexpr uint64_t kSrcCols = 8;
  constexpr uint64_t kDstRows = 6;
  constexpr uint64_t kDstCols = 9;
  const std::vector<RectCopyCase> copies = {
      RectCopyCase{
          .src_row_begin = 1,
          .src_row_end = 3,
          .src_col_begin = 2,
          .src_col_end = 6,
          .dst_row_begin = 0,
          .dst_row_end = 2,
          .dst_col_begin = 1,
          .dst_col_end = 5},
      RectCopyCase{
          .src_row_begin = 3,
          .src_row_end = 6,
          .src_col_begin = 0,
          .src_col_end = 3,
          .dst_row_begin = 3,
          .dst_row_end = 6,
          .dst_col_begin = 5,
          .dst_col_end = 8},
  };

  for (const uint64_t element_size : {uint64_t{1}, uint64_t{2}, uint64_t{4}}) {
    INFO("element_size=" << element_size);
    const uint64_t source_bytes = kSrcRows * kSrcCols * element_size;
    const uint64_t dst_tensor_offset = 5 * element_size;
    const uint64_t dst_tensor_bytes = kDstRows * kDstCols * element_size;
    const uint64_t target_bytes = dst_tensor_offset + dst_tensor_bytes + 7 * element_size;
    const std::string torch_dtype = torch_dtype_for_element_size(element_size);
    const std::string safetensors_dtype = safetensors_dtype_for_element_size(element_size);

    auto temp_root = make_temp_dir("collective-disk-loader-rect2d-matrix");
    const auto safetensors_path = temp_root / "weights.safetensors";
    std::vector<unsigned char> payload = make_sequential_payload(source_bytes);
    const std::string header = "{\"src\":{\"dtype\":\"" + safetensors_dtype +
        "\",\"shape\":[6,8],\"data_offsets\":[0," + std::to_string(source_bytes) + "]}}";
    create_safetensors_file(safetensors_path, header, payload);

    auto disk_context_or = loader::get_disk_artifact_context(temp_root);
    REQUIRE(disk_context_or.ok());

    void* gpu_buffer = nullptr;
    REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, target_bytes).ok());
    std::vector<uint8_t> initial(static_cast<size_t>(target_bytes), 0xEE);
    REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

    std::vector<materialization::contracts::RepresentationWorkItem> items;
    items.reserve(copies.size());
    uint64_t selected_bytes = 0;
    const auto src_spec = make_tensor_spec("src", 0, {6, 8}, {8, 1}, torch_dtype, element_size);
    const auto dst_spec = make_tensor_spec("dst", dst_tensor_offset, {6, 9}, {9, 1}, torch_dtype, element_size);
    for (const RectCopyCase& copy : copies) {
      materialization::contracts::RepresentationWorkItem item;
      item.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
      item.partition_kind = materialization::contracts::WorkPartitionKind::kUnknown;
      item.dst_name = "dst";
      item.dst_spec = dst_spec;
      item.committed_bytes = static_cast<uint64_t>(copy.src_row_end - copy.src_row_begin) *
          static_cast<uint64_t>(copy.src_col_end - copy.src_col_begin) * element_size;
      selected_bytes += item.committed_bytes;
      item.sources.push_back(
          materialization::contracts::RepresentationWorkSourceFragment{
              .fragment =
                  materialization::contracts::SourceFragment{
                      .source_spec = src_spec,
                      .source_range =
                          rect_range(copy.src_row_begin, copy.src_row_end, copy.src_col_begin, copy.src_col_end),
                      .destination_range =
                          rect_range(copy.dst_row_begin, copy.dst_row_end, copy.dst_col_begin, copy.dst_col_end),
                  },
          });
      items.push_back(std::move(item));
    }

    loader::ByteRangeMap data_lane_map;
    data_lane_map.total_bytes = target_bytes;
    data_lane_map.num_sources = 1;
    for (const RectCopyCase& copy : copies) {
      append_rect_segments(copy, dst_tensor_offset, kDstCols, element_size, &data_lane_map.segments);
    }

    StoreEngineOptions::MaterializationStrategyConfig strategy;
    strategy.enable_tensor_aware_mapped_executor = true;
    strategy.allow_mixed_execution = true;
    strategy.executor_preference =
        StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kTensorAwareLocal;

    auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(128, 128);
    const auto result = try_local_mapped_target_load(
        LocalMappedTargetLoadRequest{
            .artifact_id = "artifact_rect2d_matrix",
            .disk_context = *disk_context_or,
            .representation_work_plan = materialization::contracts::RepresentationWorkPlan{.items = std::move(items)},
            .data_lane_map = data_lane_map,
            .target_layout =
                loading::IntoTargetLayout{
                    .storages =
                        {
                            loading::IntoTargetStorage{
                                .base_ptr = gsl::not_null<void*>{gpu_buffer},
                                .length = target_bytes,
                            },
                        },
                    .total_size = target_bytes,
                },
            .strategy_config = strategy,
            .device_id = 0,
        },
        pinned_pool,
        std::chrono::milliseconds(1000),
        CollectiveMappedTargetLoadOptions{.chunk_bytes = 128, .strategy_config = strategy});

    REQUIRE(result.handled);
    REQUIRE(result.status.ok());
    CHECK(result.handled_bytes == selected_bytes);
    CHECK(result.residual_data_map.segments.empty());

    std::vector<uint8_t> actual(static_cast<size_t>(target_bytes));
    REQUIRE(tensorcast::cuda::memcpy(actual.data(), gpu_buffer, actual.size(), cudaMemcpyDeviceToHost).ok());
    std::vector<uint8_t> expected = initial;
    for (const RectCopyCase& copy : copies) {
      apply_rect_copy_to_expected(copy, kSrcCols, kDstCols, dst_tensor_offset, element_size, payload, &expected);
    }
    CHECK(actual == expected);

    REQUIRE(tensorcast::cuda::free(gpu_buffer).ok());
    loader::reset_disk_artifact_context_cache_for_testing();
    std::error_code cleanup_ec;
    std::filesystem::remove_all(temp_root, cleanup_ec);
  }
}

TEST_CASE(
    "try_local_mapped_target_load rejects unsupported typed work without claiming handled bytes",
    "[collective_disk_loader][local_mapped][rect2d][admission]") {
  SKIP_IF_NO_CUDA();

  constexpr uint64_t kRows = 4;
  constexpr uint64_t kCols = 6;
  constexpr uint64_t kTargetBytes = kRows * kCols;
  auto temp_root = make_temp_dir("collective-disk-loader-unsupported-rect2d");
  const auto safetensors_path = temp_root / "weights.safetensors";
  std::vector<unsigned char> payload = make_sequential_payload(kTargetBytes);
  create_safetensors_file(
      safetensors_path, "{\"src\":{\"dtype\":\"U8\",\"shape\":[4,6],\"data_offsets\":[0,24]}}", payload);

  auto disk_context_or = loader::get_disk_artifact_context(temp_root);
  REQUIRE(disk_context_or.ok());

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, kTargetBytes).ok());

  const auto src_spec = make_u8_tensor_spec("src", 0, {4, 6}, {6, 1});
  const auto dst_spec = make_u8_tensor_spec("dst", 0, {4, 6}, {6, 1});
  materialization::contracts::RepresentationWorkItem supported;
  supported.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
  supported.partition_kind = materialization::contracts::WorkPartitionKind::kUnknown;
  supported.dst_name = "dst";
  supported.dst_spec = dst_spec;
  supported.committed_bytes = 6;
  supported.sources.push_back(
      materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              materialization::contracts::SourceFragment{
                  .source_spec = src_spec,
                  .source_range = rect_range(0, 2, 0, 3),
                  .destination_range = rect_range(0, 2, 0, 3),
              },
      });

  materialization::contracts::RepresentationWorkItem unsupported = supported;
  unsupported.committed_bytes = 4;
  unsupported.sources.front().fragment.source_spec = make_u8_tensor_spec("src", 0, {2, 2, 1}, {2, 1, 1});
  unsupported.sources.front().fragment.source_range = materialization::contracts::TensorCoordinateSpec{
      .axes =
          {
              materialization::contracts::TensorAxisRange{.dim = 0, .start = 0, .end = 2},
              materialization::contracts::TensorAxisRange{.dim = 1, .start = 0, .end = 2},
              materialization::contracts::TensorAxisRange{.dim = 2, .start = 0, .end = 1},
          },
  };
  unsupported.sources.front().fragment.destination_range = rect_range(2, 4, 0, 2);

  loader::ByteRangeMap data_lane_map;
  data_lane_map.total_bytes = kTargetBytes;
  data_lane_map.num_sources = 1;
  append_rect_segments(
      RectCopyCase{
          .src_row_begin = 0,
          .src_row_end = 2,
          .src_col_begin = 0,
          .src_col_end = 3,
          .dst_row_begin = 0,
          .dst_row_end = 2,
          .dst_col_begin = 0,
          .dst_col_end = 3},
      0,
      kCols,
      1,
      &data_lane_map.segments);
  append_rect_segments(
      RectCopyCase{
          .src_row_begin = 0,
          .src_row_end = 2,
          .src_col_begin = 0,
          .src_col_end = 2,
          .dst_row_begin = 2,
          .dst_row_end = 4,
          .dst_col_begin = 0,
          .dst_col_end = 2},
      0,
      kCols,
      1,
      &data_lane_map.segments);

  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_tensor_aware_mapped_executor = true;
  strategy.allow_mixed_execution = true;
  strategy.executor_preference =
      StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kTensorAwareLocal;

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(64, 64);
  const auto result = try_local_mapped_target_load(
      LocalMappedTargetLoadRequest{
          .artifact_id = "artifact_unsupported_rect2d",
          .disk_context = *disk_context_or,
          .representation_work_plan = materialization::contracts::RepresentationWorkPlan{.items = {unsupported}},
          .data_lane_map = data_lane_map,
          .target_layout =
              loading::IntoTargetLayout{
                  .storages =
                      {
                          loading::IntoTargetStorage{
                              .base_ptr = gsl::not_null<void*>{gpu_buffer},
                              .length = kTargetBytes,
                          },
                      },
                  .total_size = kTargetBytes,
              },
          .strategy_config = strategy,
          .device_id = 0,
      },
      pinned_pool,
      std::chrono::milliseconds(1000),
      CollectiveMappedTargetLoadOptions{.chunk_bytes = 64, .strategy_config = strategy});

  REQUIRE(result.handled);
  REQUIRE_FALSE(result.status.ok());
  CHECK(absl::IsFailedPrecondition(result.status));
  CHECK(result.status.message().find("no tensor-aware jobs") != std::string_view::npos);

  REQUIRE(tensorcast::cuda::free(gpu_buffer).ok());
  loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

} // namespace tensorcast::store::replica
