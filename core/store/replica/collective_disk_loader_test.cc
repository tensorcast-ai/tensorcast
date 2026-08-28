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
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "catch2/catch_test_macros.hpp"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/cuda/cuda_api.h"
#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"
#include "core/store/replica/source_window_collective_plan.h"
#include "core/testing/test_helpers.h"
#include "gsl/pointers"

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

loader::SharedSafetensorsSegment create_sparse_safetensors_segment(
    const std::filesystem::path& path,
    uint64_t payload_bytes) {
  const std::string header_json = "{\"big\":{\"dtype\":\"U8\",\"shape\":[" + std::to_string(payload_bytes) +
      "],\"data_offsets\":[0," + std::to_string(payload_bytes) + "]}}";
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  REQUIRE(out.is_open());
  write_u64_le(out, static_cast<uint64_t>(header_json.size()));
  out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
  if (payload_bytes > 0) {
    out.seekp(static_cast<std::streamoff>(8 + header_json.size() + payload_bytes - 1));
    const char zero = 0;
    out.write(&zero, 1);
  }
  out.close();
  REQUIRE(out.good());
  return loader::SharedSafetensorsSegment{
      .path = path,
      .file = nullptr,
      .data_start = 8 + header_json.size(),
      .data_size = payload_bytes,
      .base_offset = 0,
  };
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

materialization::contracts::RepresentationWorkItem make_u8_copy_item(
    const materialization::contracts::RepresentationTensorSpec& source,
    const materialization::contracts::RepresentationTensorSpec& destination,
    uint64_t bytes) {
  materialization::contracts::RepresentationWorkItem item;
  item.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
  item.partition_kind = materialization::contracts::WorkPartitionKind::kReplicated;
  item.dst_name = destination.name;
  item.dst_spec = destination;
  item.committed_bytes = bytes;
  item.sources.push_back(
      materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              materialization::contracts::SourceFragment{
                  .source_spec = source,
                  .source_range = materialization::contracts::TensorCoordinateSpec{},
                  .destination_range = materialization::contracts::TensorCoordinateSpec{},
              },
      });
  return item;
}

runtime::ingestion::strategy::SourceWindowCollectiveCandidateSummary make_source_window_candidate_summary(
    runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode mode,
    runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode distribution_mode =
        runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kFullWindowAllGather) {
  return runtime::ingestion::strategy::SourceWindowCollectiveCandidateSummary{
      .candidate = true,
      .group_final_admitted = false,
      .selection_mode = mode,
      .distribution_mode = distribution_mode,
      .pre_admission_reason = "candidate_pending_group_final_admission",
  };
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

class ScopedEnvVar {
 public:
  explicit ScopedEnvVar(const char* name) : name_(name) {
    const char* value = std::getenv(name_);
    if (value != nullptr) {
      old_value_ = value;
    }
  }

  ~ScopedEnvVar() {
    if (old_value_.has_value()) {
      (void)setenv(name_, old_value_->c_str(), /*overwrite=*/1);
    } else {
      (void)unsetenv(name_);
    }
  }

  void set(const char* value) {
    (void)setenv(name_, value, /*overwrite=*/1);
  }

  void unset() {
    (void)unsetenv(name_);
  }

 private:
  const char* name_;
  std::optional<std::string> old_value_;
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
    "auto local mapped safetensors io keeps small sources buffered",
    "[collective_disk_loader][local_mapped][io_policy]") {
  const loader::SharedSafetensorsSegment segment{
      .path = std::filesystem::path{},
      .file = nullptr,
      .data_start = 0,
      .data_size = 4096,
      .base_offset = 0,
  };

  const auto decision_or = choose_auto_local_mapped_safetensors_io_for_testing(absl::MakeSpan(&segment, 1));
  REQUIRE(decision_or.ok());
  CHECK_FALSE(decision_or->use_direct_aligned_edges);
  CHECK(decision_or->reason == "source_below_direct_threshold");
}

TEST_CASE(
    "auto local mapped safetensors io chooses direct for cold large local sources",
    "[collective_disk_loader][local_mapped][io_policy]") {
  constexpr uint64_t kPayloadBytes = (1ULL << 30) + 4096;
  auto temp_root = make_temp_dir("collective-disk-loader-auto-io-policy");
  const auto safetensors_path = temp_root / "weights.safetensors";
  const auto segment = create_sparse_safetensors_segment(safetensors_path, kPayloadBytes);

  const auto decision_or = choose_auto_local_mapped_safetensors_io_for_testing(absl::MakeSpan(&segment, 1));
  REQUIRE(decision_or.ok());
  if (!decision_or->use_direct_aligned_edges) {
    std::error_code cleanup_ec;
    std::filesystem::remove_all(temp_root, cleanup_ec);
    SKIP("test filesystem does not support direct auto path: " << decision_or->reason);
  }

  CHECK(decision_or->use_direct_aligned_edges);
  const bool cold_direct_reason = decision_or->reason == "page_cache_cold_or_partial_direct" ||
      decision_or->reason == "direct_probe_cold_or_partial_direct";
  CHECK(cold_direct_reason);

  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "auto local mapped safetensors io keeps tmpfs sources buffered",
    "[collective_disk_loader][local_mapped][io_policy]") {
  constexpr uint64_t kPayloadBytes = (1ULL << 30) + 4096;
  const std::filesystem::path tmpfs_root = "/dev/shm";
  if (!std::filesystem::is_directory(tmpfs_root)) {
    SKIP("/dev/shm is unavailable");
  }

  auto temp_root = tmpfs_root /
      ("collective-disk-loader-auto-io-policy-tmpfs-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(temp_root);
  const auto safetensors_path = temp_root / "weights.safetensors";
  const auto segment = create_sparse_safetensors_segment(safetensors_path, kPayloadBytes);

  const auto decision_or = choose_auto_local_mapped_safetensors_io_for_testing(absl::MakeSpan(&segment, 1));
  REQUIRE(decision_or.ok());
  CHECK_FALSE(decision_or->use_direct_aligned_edges);
  CHECK(decision_or->reason == "filesystem_memory_buffered");

  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "try_source_window_collective_mapped_target_load performs group final admission before runtime fallback",
    "[collective_disk_loader][source_window][admission]") {
  std::array<std::uint8_t, 8> rank0_target{};
  std::array<std::uint8_t, 8> rank1_target{};
  const auto source = make_u8_tensor_spec("src", 0, {8}, {1});
  const auto destination = make_u8_tensor_spec("dst", 0, {8}, {1});
  const materialization::contracts::RepresentationWorkPlan work_plan{
      .items = {make_u8_copy_item(source, destination, 8)},
      .committed_bytes = 8,
  };

  CollectiveMappedTargetLoadOptions options;
  options.strategy_config.enable_source_window_collective = true;
  options.strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto;
  options.strategy_config.source_window_collective_min_rank_read_saving_bytes = 0;
  options.strategy_config.source_window_collective_window_bytes = 16;
  options.strategy_config.source_window_collective_peak_bytes_budget = 64;
  options.strategy_config.source_window_collective_max_peer_to_read_ratio_x1000 = 2000;
  options.strategy_config.diagnostics_verbosity =
      StoreEngineOptions::MaterializationStrategyConfig::DiagnosticsVerbosity::kVerbose;
  const auto shared_work_plan = std::make_shared<const materialization::contracts::RepresentationWorkPlan>(work_plan);

  auto make_request = [&](uint32_t rank, void* target) {
    auto target_layout = std::make_shared<const loading::IntoTargetLayout>(loading::IntoTargetLayout{
        .storages =
            {
                loading::IntoTargetStorage{
                    .base_ptr = gsl::not_null<void*>{target},
                    .length = 8,
                },
            },
        .total_size = 8,
    });
    return SourceWindowCollectiveMappedTargetLoadRequest{
        .artifact_id = "artifact_source_window_admission",
        .group =
            loading::CollectiveLoadGroupHint{
                .group_id = "source-window-admission-auto",
                .world_size = 2,
                .rank = rank,
            },
        .disk_context = make_disk_context(),
        .representation_work_plan_ref = shared_work_plan,
        .target_layout_ref = std::move(target_layout),
        .candidate_summary = make_source_window_candidate_summary(
            runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kAuto),
        .source_index_digest = "source-index-a",
        .device_id = static_cast<int>(rank),
    };
  };

  std::array<SourceWindowCollectiveMappedTargetLoadResult, 2> results;
  std::thread rank0([&]() {
    results[0] = try_source_window_collective_mapped_target_load(
        make_request(0, rank0_target.data()), nullptr, std::chrono::milliseconds(1000), options);
  });
  std::thread rank1([&]() {
    results[1] = try_source_window_collective_mapped_target_load(
        make_request(1, rank1_target.data()), nullptr, std::chrono::milliseconds(1000), options);
  });
  rank0.join();
  rank1.join();

  for (const auto& result : results) {
    CHECK_FALSE(result.handled);
    CHECK(result.status.ok());
    CHECK(result.skip_reason == "source_window_runtime_unavailable:pinned_pool_missing");
    CHECK_FALSE(result.plan_hash.empty());
    CHECK(result.metrics.source_window_group_disk_read_bytes == 8);
    CHECK(result.metrics.source_window_unique_payload_bytes == 8);
    CHECK(result.metrics.source_window_target_write_bytes == 16);
    CHECK(result.metrics.source_window_window_count == 1);
    CHECK(result.metrics.source_window_read_amplification_x1000 == 1000);
  }
  CHECK(results[0].plan_hash == results[1].plan_hash);
}

TEST_CASE(
    "source-window group plan cache reuses plans across runtime group ids",
    "[collective_disk_loader][source_window][admission][cache]") {
  clear_source_window_collective_plan_cache_for_testing();

  const auto source = make_u8_tensor_spec("src", 0, {8}, {1});
  const auto destination = make_u8_tensor_spec("dst", 0, {8}, {1});
  const materialization::contracts::RepresentationWorkPlan work_plan{
      .items = {make_u8_copy_item(source, destination, 8)},
      .committed_bytes = 8,
  };
  const auto shared_work_plan = std::make_shared<const materialization::contracts::RepresentationWorkPlan>(work_plan);

  CollectiveMappedTargetLoadOptions options;
  options.strategy_config.enable_source_window_collective = true;
  options.strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto;
  options.strategy_config.source_window_collective_min_rank_read_saving_bytes = 0;
  options.strategy_config.source_window_collective_window_bytes = 16;
  options.strategy_config.source_window_collective_peak_bytes_budget = 64;
  options.strategy_config.source_window_collective_max_peer_to_read_ratio_x1000 = 2000;
  options.enable_source_window_plan_cache = true;

  auto run_group = [&](std::string_view group_id) {
    std::array<std::array<std::uint8_t, 8>, 2> targets{};
    auto make_request = [&](uint32_t rank) {
      auto target_layout = std::make_shared<const loading::IntoTargetLayout>(loading::IntoTargetLayout{
          .storages =
              {
                  loading::IntoTargetStorage{
                      .base_ptr = gsl::not_null<void*>{targets[rank].data()},
                      .length = 8,
                  },
              },
          .total_size = 8,
      });
      return SourceWindowCollectiveMappedTargetLoadRequest{
          .artifact_id = "artifact_source_window_plan_cache",
          .group =
              loading::CollectiveLoadGroupHint{
                  .group_id = std::string(group_id),
                  .world_size = 2,
                  .rank = rank,
              },
          .disk_context = make_disk_context(),
          .representation_work_plan_ref = shared_work_plan,
          .target_layout_ref = std::move(target_layout),
          .candidate_summary = make_source_window_candidate_summary(
              runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kAuto),
          .source_index_digest = "source-index-plan-cache",
          .device_id = static_cast<int>(rank),
      };
    };

    std::array<SourceWindowCollectiveMappedTargetLoadResult, 2> results;
    std::thread rank0([&]() {
      results[0] = try_source_window_collective_mapped_target_load(
          make_request(0), nullptr, std::chrono::milliseconds(1000), options);
    });
    std::thread rank1([&]() {
      results[1] = try_source_window_collective_mapped_target_load(
          make_request(1), nullptr, std::chrono::milliseconds(1000), options);
    });
    rank0.join();
    rank1.join();
    return results;
  };

  const auto first = run_group("source-window-plan-cache-a");
  const auto second = run_group("source-window-plan-cache-b");

  for (const auto& result : first) {
    CHECK_FALSE(result.handled);
    CHECK(result.status.ok());
    CHECK(result.skip_reason == "source_window_runtime_unavailable:pinned_pool_missing");
    CHECK_FALSE(result.plan_hash.empty());
    CHECK_FALSE(result.plan_cache_hit);
  }
  for (const auto& result : second) {
    CHECK_FALSE(result.handled);
    CHECK(result.status.ok());
    CHECK(result.skip_reason == "source_window_runtime_unavailable:pinned_pool_missing");
    CHECK_FALSE(result.plan_hash.empty());
    CHECK(result.plan_cache_hit);
  }
  CHECK(first[0].plan_hash == first[1].plan_hash);
  CHECK(second[0].plan_hash == second[1].plan_hash);
  CHECK(first[0].plan_hash == second[0].plan_hash);

  const auto stats = source_window_collective_plan_cache_stats_for_testing();
  CHECK(stats.misses == 1);
  CHECK(stats.hits == 1);
  CHECK(stats.entries == 1);

  clear_source_window_collective_plan_cache_for_testing();
}

TEST_CASE(
    "source-window group plan cache keys prepared realization facts",
    "[collective_disk_loader][source_window][admission][cache]") {
  clear_source_window_collective_plan_cache_for_testing();

  const auto source = make_u8_tensor_spec("src", 0, {8}, {1});
  const auto destination = make_u8_tensor_spec("dst", 0, {8}, {1});
  const materialization::contracts::RepresentationWorkPlan work_plan{
      .items = {make_u8_copy_item(source, destination, 8)},
      .committed_bytes = 8,
  };
  const auto shared_work_plan = std::make_shared<const materialization::contracts::RepresentationWorkPlan>(work_plan);

  CollectiveMappedTargetLoadOptions options;
  options.strategy_config.enable_source_window_collective = true;
  options.strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto;
  options.strategy_config.source_window_collective_min_rank_read_saving_bytes = 0;
  options.strategy_config.source_window_collective_window_bytes = 16;
  options.strategy_config.source_window_collective_peak_bytes_budget = 64;
  options.strategy_config.source_window_collective_max_peer_to_read_ratio_x1000 = 2000;
  options.enable_source_window_plan_cache = true;

  auto facts_for = [](uint32_t rank, std::string_view version) {
    return loading::SourceWindowPreparedRealizationFacts{
        .group_key = absl::StrCat("prepared-group-template-", version, "-", rank),
        .member_key = absl::StrCat("prepared-member-", version, "-", rank),
        .realization_plan_hash = absl::StrCat("prepared-plan-", version, "-", rank),
        .target_layout_template_hash = "prepared-layout-template",
        .target_index_hash = "prepared-target-index",
    };
  };

  auto run_group = [&](std::string_view group_id, std::string_view prepared_version) {
    std::array<std::array<std::uint8_t, 8>, 2> targets{};
    auto make_request = [&](uint32_t rank) {
      auto target_layout = std::make_shared<const loading::IntoTargetLayout>(loading::IntoTargetLayout{
          .storages =
              {
                  loading::IntoTargetStorage{
                      .base_ptr = gsl::not_null<void*>{targets[rank].data()},
                      .length = 8,
                  },
              },
          .total_size = 8,
      });
      return SourceWindowCollectiveMappedTargetLoadRequest{
          .artifact_id = "artifact_source_window_prepared_plan_cache",
          .group =
              loading::CollectiveLoadGroupHint{
                  .group_id = std::string(group_id),
                  .world_size = 2,
                  .rank = rank,
              },
          .disk_context = make_disk_context(),
          .representation_work_plan_ref = shared_work_plan,
          .target_layout_ref = std::move(target_layout),
          .candidate_summary = make_source_window_candidate_summary(
              runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kAuto),
          .source_index_digest = "source-index-prepared-plan-cache",
          .prepared_realization = facts_for(rank, prepared_version),
          .device_id = static_cast<int>(rank),
      };
    };

    std::array<SourceWindowCollectiveMappedTargetLoadResult, 2> results;
    std::thread rank0([&]() {
      results[0] = try_source_window_collective_mapped_target_load(
          make_request(0), nullptr, std::chrono::milliseconds(1000), options);
    });
    std::thread rank1([&]() {
      results[1] = try_source_window_collective_mapped_target_load(
          make_request(1), nullptr, std::chrono::milliseconds(1000), options);
    });
    rank0.join();
    rank1.join();
    return results;
  };

  const auto first = run_group("source-window-prepared-plan-cache-a", "v1");
  const auto second = run_group("source-window-prepared-plan-cache-b", "v1");
  const auto facts_changed = run_group("source-window-prepared-plan-cache-c", "v2");

  for (const auto& result : first) {
    CHECK_FALSE(result.handled);
    CHECK(result.status.ok());
    CHECK(result.skip_reason == "source_window_runtime_unavailable:pinned_pool_missing");
    CHECK_FALSE(result.plan_cache_hit);
  }
  for (const auto& result : second) {
    CHECK_FALSE(result.handled);
    CHECK(result.status.ok());
    CHECK(result.skip_reason == "source_window_runtime_unavailable:pinned_pool_missing");
    CHECK(result.plan_cache_hit);
  }
  for (const auto& result : facts_changed) {
    CHECK_FALSE(result.handled);
    CHECK(result.status.ok());
    CHECK(result.skip_reason == "source_window_runtime_unavailable:pinned_pool_missing");
    CHECK_FALSE(result.plan_cache_hit);
  }
  CHECK(first[0].plan_hash == second[0].plan_hash);
  CHECK(first[0].plan_hash != facts_changed[0].plan_hash);

  const auto stats = source_window_collective_plan_cache_stats_for_testing();
  CHECK(stats.misses == 2);
  CHECK(stats.hits == 1);
  CHECK(stats.entries == 2);

  clear_source_window_collective_plan_cache_for_testing();
}

TEST_CASE(
    "source-window routed program cache key is target-relative and geometry-sensitive",
    "[collective_disk_loader][source_window][routed_program][cache]") {
  SourceWindowCollectivePlan plan{
      .group =
          loading::CollectiveLoadGroupHint{
              .group_id = "routed-program-cache-key-group-a",
              .world_size = 2,
              .rank = 0,
          },
      .distribution_mode = runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted,
      .windows =
          {
              SourceWindowCollectiveWindow{
                  .source_index = 0,
                  .owner_rank = 0,
                  .distribution_mode =
                      runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted,
                  .start = 0,
                  .end = 16,
                  .unique_payload_bytes = 16,
              },
          },
      .plan_hash = "plan-hash-a",
  };

  std::array<std::uint8_t, 16> rank0_a{};
  std::array<std::uint8_t, 16> rank0_b{};
  std::array<std::uint8_t, 16> rank1_a{};
  std::array<std::uint8_t, 16> rank1_b{};

  auto make_target_layout = [](void* ptr, uint64_t length) {
    return std::make_shared<const loading::IntoTargetLayout>(loading::IntoTargetLayout{
        .storages =
            {
                loading::IntoTargetStorage{
                    .base_ptr = gsl::not_null<void*>{ptr},
                    .length = length,
                },
            },
        .total_size = length,
    });
  };

  auto make_requests = [&](void* rank0_ptr,
                           void* rank1_ptr,
                           uint64_t rank0_length,
                           uint64_t rank1_length,
                           std::string_view source_index_digest) {
    std::vector<SourceWindowCollectiveMappedTargetLoadRequest> requests;
    requests.reserve(2);
    for (uint32_t rank = 0; rank < 2; ++rank) {
      requests.push_back(
          SourceWindowCollectiveMappedTargetLoadRequest{
              .artifact_id = "artifact-routed-program-cache-key",
              .group =
                  loading::CollectiveLoadGroupHint{
                      .group_id = "routed-program-cache-key-group-a",
                      .world_size = 2,
                      .rank = rank,
                  },
              .disk_context = make_disk_context(),
              .target_layout_ref =
                  make_target_layout(rank == 0 ? rank0_ptr : rank1_ptr, rank == 0 ? rank0_length : rank1_length),
              .source_index_digest = std::string(source_index_digest),
              .device_id = static_cast<int>(rank),
          });
    }
    return requests;
  };

  auto base_requests = make_requests(rank0_a.data(), rank1_a.data(), 8, 8, "source-index-a");
  auto same_geometry_new_pointers = make_requests(rank0_b.data(), rank1_b.data(), 8, 8, "source-index-a");
  auto source_digest_changed = make_requests(rank0_b.data(), rank1_b.data(), 8, 8, "source-index-b");
  auto target_geometry_changed = make_requests(rank0_b.data(), rank1_b.data(), 16, 8, "source-index-a");
  auto reordered_requests = base_requests;
  std::reverse(reordered_requests.begin(), reordered_requests.end());

  constexpr size_t kChunkBytes = 64;
  constexpr size_t kMaxCollectiveChunkBytes = 64;
  constexpr size_t kMaxStripeBytes = 32;
  auto base_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-key",
      plan,
      absl::MakeConstSpan(base_requests),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(base_key.ok());

  auto same_geometry_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-key",
      plan,
      absl::MakeConstSpan(same_geometry_new_pointers),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(same_geometry_key.ok());
  CHECK(*base_key == *same_geometry_key);

  auto reordered_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-key",
      plan,
      absl::MakeConstSpan(reordered_requests),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(reordered_key.ok());
  CHECK(*base_key == *reordered_key);

  auto source_changed_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-key",
      plan,
      absl::MakeConstSpan(source_digest_changed),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(source_changed_key.ok());
  CHECK(*base_key != *source_changed_key);

  auto target_changed_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-key",
      plan,
      absl::MakeConstSpan(target_geometry_changed),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(target_changed_key.ok());
  CHECK(*base_key != *target_changed_key);

  auto chunk_changed_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-key",
      plan,
      absl::MakeConstSpan(base_requests),
      /*configured_chunk_bytes=*/128,
      /*max_collective_chunk_bytes=*/128,
      /*max_stripe_bytes=*/64);
  REQUIRE(chunk_changed_key.ok());
  CHECK(*base_key != *chunk_changed_key);

  auto plan_changed = plan;
  plan_changed.plan_hash = "plan-hash-b";
  auto plan_changed_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-key",
      plan_changed,
      absl::MakeConstSpan(base_requests),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(plan_changed_key.ok());
  CHECK(*base_key != *plan_changed_key);
}

TEST_CASE(
    "source-window compiled routed program auto parallelizes medium chunk counts",
    "[collective_disk_loader][source_window][routed_program][compiled_program]") {
  CHECK(source_window_compiled_routed_program_build_thread_count_for_testing(1, 0) == 1);
  CHECK(source_window_compiled_routed_program_build_thread_count_for_testing(31, 0) == 1);
  CHECK(source_window_compiled_routed_program_build_thread_count_for_testing(32, 0) > 1);
  CHECK(source_window_compiled_routed_program_build_thread_count_for_testing(115, 0) > 1);
  CHECK(source_window_compiled_routed_program_build_thread_count_for_testing(885, 0) <= 32);
  CHECK(source_window_compiled_routed_program_build_thread_count_for_testing(115, 1) == 1);
  CHECK(source_window_compiled_routed_program_build_thread_count_for_testing(115, 16) == 16);
  CHECK(source_window_compiled_routed_program_build_thread_count_for_testing(115, 32) == 32);
}

TEST_CASE(
    "source-window pipeline slot cap is configurable for controlled profiling",
    "[collective_disk_loader][source_window][pipeline]") {
  ScopedEnvVar env("TENSORCAST_SOURCE_WINDOW_MAX_PIPELINE_SLOTS");

  env.unset();
  CHECK(source_window_max_pipeline_slots_cap_for_testing() == 16);
  CHECK(source_window_effective_max_pipeline_slots_for_testing(0) == 1);
  CHECK(source_window_effective_max_pipeline_slots_for_testing(8) == 8);
  CHECK(source_window_effective_max_pipeline_slots_for_testing(32) == 16);

  env.set("32");
  CHECK(source_window_max_pipeline_slots_cap_for_testing() == 32);
  CHECK(source_window_effective_max_pipeline_slots_for_testing(16) == 16);
  CHECK(source_window_effective_max_pipeline_slots_for_testing(64) == 32);

  env.set("0");
  CHECK(source_window_max_pipeline_slots_cap_for_testing() == 16);

  env.set("not-a-number");
  CHECK(source_window_max_pipeline_slots_cap_for_testing() == 16);

  env.set("256");
  CHECK(source_window_max_pipeline_slots_cap_for_testing() == 128);
}

TEST_CASE(
    "source-window read-ahead slot count can be limited independently",
    "[collective_disk_loader][source_window][pipeline]") {
  ScopedEnvVar env("TENSORCAST_SOURCE_WINDOW_READ_AHEAD_SLOTS");

  env.unset();
  CHECK(!source_window_requested_read_ahead_slots_for_testing().has_value());
  CHECK(source_window_effective_read_ahead_slots_for_testing(0) == 1);
  CHECK(source_window_effective_read_ahead_slots_for_testing(16) == 16);

  env.set("8");
  REQUIRE(source_window_requested_read_ahead_slots_for_testing().has_value());
  CHECK(*source_window_requested_read_ahead_slots_for_testing() == 8);
  CHECK(source_window_effective_read_ahead_slots_for_testing(4) == 4);
  CHECK(source_window_effective_read_ahead_slots_for_testing(16) == 8);

  env.set("0");
  CHECK(!source_window_requested_read_ahead_slots_for_testing().has_value());
  CHECK(source_window_effective_read_ahead_slots_for_testing(16) == 16);

  env.set("not-a-number");
  CHECK(!source_window_requested_read_ahead_slots_for_testing().has_value());
  CHECK(source_window_effective_read_ahead_slots_for_testing(16) == 16);

  env.set("256");
  REQUIRE(source_window_requested_read_ahead_slots_for_testing().has_value());
  CHECK(*source_window_requested_read_ahead_slots_for_testing() == 128);
  CHECK(source_window_effective_read_ahead_slots_for_testing(64) == 64);
  CHECK(source_window_effective_read_ahead_slots_for_testing(256) == 128);
}

TEST_CASE(
    "source-window clique completion eviction is configurable for controlled profiling",
    "[collective_disk_loader][source_window][clique]") {
  ScopedEnvVar env("TENSORCAST_SOURCE_WINDOW_EVICT_CLIQUE_ON_COMPLETE");

  env.unset();
  CHECK(source_window_evict_clique_on_complete_for_testing());

  env.set("0");
  CHECK_FALSE(source_window_evict_clique_on_complete_for_testing());

  env.set("false");
  CHECK_FALSE(source_window_evict_clique_on_complete_for_testing());

  env.set("off");
  CHECK_FALSE(source_window_evict_clique_on_complete_for_testing());

  env.set("no");
  CHECK_FALSE(source_window_evict_clique_on_complete_for_testing());

  env.set("1");
  CHECK(source_window_evict_clique_on_complete_for_testing());

  env.set("true");
  CHECK(source_window_evict_clique_on_complete_for_testing());

  env.set("on");
  CHECK(source_window_evict_clique_on_complete_for_testing());

  env.set("yes");
  CHECK(source_window_evict_clique_on_complete_for_testing());

  env.set("not-a-bool");
  CHECK(source_window_evict_clique_on_complete_for_testing());
}

TEST_CASE(
    "source-window async clique destroy is enabled by default and configurable",
    "[collective_disk_loader][source_window][clique]") {
  ScopedEnvVar env("TENSORCAST_SOURCE_WINDOW_ASYNC_CLIQUE_DESTROY_ON_COMPLETE");

  env.unset();
  CHECK(source_window_async_clique_destroy_on_complete_for_testing());

  env.set("1");
  CHECK(source_window_async_clique_destroy_on_complete_for_testing());

  env.set("true");
  CHECK(source_window_async_clique_destroy_on_complete_for_testing());

  env.set("on");
  CHECK(source_window_async_clique_destroy_on_complete_for_testing());

  env.set("yes");
  CHECK(source_window_async_clique_destroy_on_complete_for_testing());

  env.set("0");
  CHECK_FALSE(source_window_async_clique_destroy_on_complete_for_testing());

  env.set("false");
  CHECK_FALSE(source_window_async_clique_destroy_on_complete_for_testing());

  env.set("off");
  CHECK_FALSE(source_window_async_clique_destroy_on_complete_for_testing());

  env.set("no");
  CHECK_FALSE(source_window_async_clique_destroy_on_complete_for_testing());

  env.set("not-a-bool");
  CHECK(source_window_async_clique_destroy_on_complete_for_testing());
}

TEST_CASE(
    "NCCL clique destroy mode is configurable for controlled profiling",
    "[collective_disk_loader][source_window][clique]") {
  ScopedEnvVar env("TENSORCAST_NCCL_CLIQUE_DESTROY_MODE");

  env.unset();
  CHECK(nccl_clique_destroy_mode_for_testing() == "serial");

  env.set("finalize");
  CHECK(nccl_clique_destroy_mode_for_testing() == "finalize");

  env.set("finalize_poll");
  CHECK(nccl_clique_destroy_mode_for_testing() == "finalize");

  env.set("true");
  CHECK(nccl_clique_destroy_mode_for_testing() == "finalize");

  env.set("serial");
  CHECK(nccl_clique_destroy_mode_for_testing() == "serial");

  env.set("destroy");
  CHECK(nccl_clique_destroy_mode_for_testing() == "serial");

  env.set("false");
  CHECK(nccl_clique_destroy_mode_for_testing() == "serial");

  env.set("not-a-mode");
  CHECK(nccl_clique_destroy_mode_for_testing() == "serial");
}

TEST_CASE(
    "source-window routed program cache key prefers prepared realization identity",
    "[collective_disk_loader][source_window][routed_program][cache]") {
  SourceWindowCollectivePlan plan{
      .group =
          loading::CollectiveLoadGroupHint{
              .group_id = "routed-program-cache-prepared-key-group-a",
              .world_size = 2,
              .rank = 0,
          },
      .distribution_mode = runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted,
      .windows =
          {
              SourceWindowCollectiveWindow{
                  .source_index = 0,
                  .owner_rank = 0,
                  .distribution_mode =
                      runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted,
                  .start = 0,
                  .end = 16,
                  .unique_payload_bytes = 16,
              },
          },
      .plan_hash = "prepared-plan-hash-a",
  };

  std::array<std::uint8_t, 16> rank0_a{};
  std::array<std::uint8_t, 16> rank0_b{};
  std::array<std::uint8_t, 16> rank1_a{};
  std::array<std::uint8_t, 16> rank1_b{};

  auto make_target_layout = [](void* ptr, uint64_t length) {
    return std::make_shared<const loading::IntoTargetLayout>(loading::IntoTargetLayout{
        .storages =
            {
                loading::IntoTargetStorage{
                    .base_ptr = gsl::not_null<void*>{ptr},
                    .length = length,
                },
            },
        .total_size = length,
    });
  };
  auto make_facts = [](std::string_view group_key,
                       std::string_view member_key,
                       std::string_view realization_plan_hash = "realization-plan-hash-a",
                       std::string_view target_layout_template_hash = "target-layout-template-hash-a",
                       std::string_view target_index_hash = "target-index-hash-a") {
    return loading::SourceWindowPreparedRealizationFacts{
        .group_key = std::string(group_key),
        .member_key = std::string(member_key),
        .realization_plan_hash = std::string(realization_plan_hash),
        .target_layout_template_hash = std::string(target_layout_template_hash),
        .target_index_hash = std::string(target_index_hash),
    };
  };
  auto make_requests = [&](void* rank0_ptr,
                           void* rank1_ptr,
                           uint64_t rank0_length,
                           uint64_t rank1_length,
                           std::string_view source_index_digest,
                           std::string_view group_key,
                           std::string_view rank0_member_key,
                           std::string_view rank1_member_key,
                           std::string_view target_layout_template_hash = "target-layout-template-hash-a") {
    std::vector<SourceWindowCollectiveMappedTargetLoadRequest> requests;
    requests.reserve(2);
    for (uint32_t rank = 0; rank < 2; ++rank) {
      requests.push_back(
          SourceWindowCollectiveMappedTargetLoadRequest{
              .artifact_id = "artifact-routed-program-cache-prepared-key",
              .group =
                  loading::CollectiveLoadGroupHint{
                      .group_id = "routed-program-cache-prepared-key-group-a",
                      .world_size = 2,
                      .rank = rank,
                  },
              .disk_context = make_disk_context(),
              .target_layout_ref =
                  make_target_layout(rank == 0 ? rank0_ptr : rank1_ptr, rank == 0 ? rank0_length : rank1_length),
              .source_index_digest = std::string(source_index_digest),
              .prepared_realization = make_facts(
                  group_key,
                  rank == 0 ? rank0_member_key : rank1_member_key,
                  "realization-plan-hash-a",
                  target_layout_template_hash,
                  "target-index-hash-a"),
              .device_id = static_cast<int>(rank),
          });
    }
    return requests;
  };

  constexpr size_t kChunkBytes = 64;
  constexpr size_t kMaxCollectiveChunkBytes = 64;
  constexpr size_t kMaxStripeBytes = 32;
  auto base_requests =
      make_requests(rank0_a.data(), rank1_a.data(), 8, 8, "source-index-a", "group-a", "member-0", "member-1");
  auto same_geometry_new_pointers =
      make_requests(rank0_b.data(), rank1_b.data(), 8, 8, "source-index-a", "group-a", "member-0", "member-1");
  auto group_changed =
      make_requests(rank0_b.data(), rank1_b.data(), 8, 8, "source-index-a", "group-b", "member-0", "member-1");
  auto member_changed =
      make_requests(rank0_b.data(), rank1_b.data(), 8, 8, "source-index-a", "group-a", "member-0b", "member-1");
  auto target_template_changed = make_requests(
      rank0_b.data(),
      rank1_b.data(),
      8,
      8,
      "source-index-a",
      "group-a",
      "member-0",
      "member-1",
      "target-layout-template-hash-b");
  auto source_digest_changed =
      make_requests(rank0_b.data(), rank1_b.data(), 8, 8, "source-index-b", "group-a", "member-0", "member-1");
  auto target_geometry_changed =
      make_requests(rank0_b.data(), rank1_b.data(), 16, 8, "source-index-a", "group-a", "member-0", "member-1");
  auto incomplete_prepared = base_requests;
  incomplete_prepared[1].prepared_realization->member_key.clear();

  auto base_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-prepared-key",
      plan,
      absl::MakeConstSpan(base_requests),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(base_key.ok());

  auto same_geometry_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-prepared-key",
      plan,
      absl::MakeConstSpan(same_geometry_new_pointers),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(same_geometry_key.ok());
  CHECK(*base_key == *same_geometry_key);

  auto group_changed_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-prepared-key",
      plan,
      absl::MakeConstSpan(group_changed),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(group_changed_key.ok());
  CHECK(*base_key != *group_changed_key);

  auto member_changed_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-prepared-key",
      plan,
      absl::MakeConstSpan(member_changed),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(member_changed_key.ok());
  CHECK(*base_key != *member_changed_key);

  auto target_template_changed_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-prepared-key",
      plan,
      absl::MakeConstSpan(target_template_changed),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(target_template_changed_key.ok());
  CHECK(*base_key != *target_template_changed_key);

  auto source_digest_changed_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-prepared-key",
      plan,
      absl::MakeConstSpan(source_digest_changed),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(source_digest_changed_key.ok());
  CHECK(*base_key != *source_digest_changed_key);

  auto target_geometry_changed_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-prepared-key",
      plan,
      absl::MakeConstSpan(target_geometry_changed),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(target_geometry_changed_key.ok());
  CHECK(*base_key != *target_geometry_changed_key);

  auto incomplete_prepared_key = source_window_routed_program_cache_key_for_testing(
      "artifact-routed-program-cache-prepared-key",
      plan,
      absl::MakeConstSpan(incomplete_prepared),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(incomplete_prepared_key.ok());
  CHECK(*base_key != *incomplete_prepared_key);
}

TEST_CASE(
    "source-window routed program cache can be prepared before runtime execution",
    "[collective_disk_loader][source_window][routed_program][cache][prepare]") {
  clear_source_window_routed_program_cache_for_testing();

  SourceWindowCollectivePlan plan{
      .group =
          loading::CollectiveLoadGroupHint{
              .group_id = "routed-program-cache-prepare-group-a",
              .world_size = 2,
              .rank = 0,
          },
      .distribution_mode = runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted,
      .windows =
          {
              SourceWindowCollectiveWindow{
                  .source_index = 0,
                  .owner_rank = 0,
                  .distribution_mode =
                      runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted,
                  .start = 0,
                  .end = 16,
                  .unique_payload_bytes = 16,
                  .consumer_spans =
                      {
                          SourceWindowCollectiveConsumerSpan{
                              .rank = 0,
                              .storage_index = 0,
                              .source_window_start = 0,
                              .source_window_end = 8,
                              .source_offset = 0,
                              .target_offset = 0,
                              .length = 8,
                          },
                          SourceWindowCollectiveConsumerSpan{
                              .rank = 1,
                              .storage_index = 0,
                              .source_window_start = 8,
                              .source_window_end = 16,
                              .source_offset = 8,
                              .target_offset = 0,
                              .length = 8,
                          },
                      },
              },
          },
      .plan_hash = "prepare-plan-hash-a",
  };

  std::array<std::uint8_t, 8> rank0_a{};
  std::array<std::uint8_t, 8> rank1_a{};
  std::array<std::uint8_t, 8> rank0_b{};
  std::array<std::uint8_t, 8> rank1_b{};

  auto make_target_layout = [](void* ptr) {
    return std::make_shared<const loading::IntoTargetLayout>(loading::IntoTargetLayout{
        .storages =
            {
                loading::IntoTargetStorage{
                    .base_ptr = gsl::not_null<void*>{ptr},
                    .length = 8,
                },
            },
        .total_size = 8,
    });
  };

  auto make_requests = [&](void* rank0_ptr, void* rank1_ptr) {
    std::vector<SourceWindowCollectiveMappedTargetLoadRequest> requests;
    requests.reserve(2);
    for (uint32_t rank = 0; rank < 2; ++rank) {
      requests.push_back(
          SourceWindowCollectiveMappedTargetLoadRequest{
              .artifact_id = "artifact-routed-program-cache-prepare",
              .group =
                  loading::CollectiveLoadGroupHint{
                      .group_id = "routed-program-cache-prepare-group-a",
                      .world_size = 2,
                      .rank = rank,
                  },
              .disk_context = make_disk_context(),
              .target_layout_ref = make_target_layout(rank == 0 ? rank0_ptr : rank1_ptr),
              .source_index_digest = "source-index-prepare",
              .prepared_realization =
                  loading::SourceWindowPreparedRealizationFacts{
                      .group_key = "prepared-group-prepare",
                      .member_key = absl::StrCat("prepared-member-", rank),
                      .realization_plan_hash = "realization-plan-prepare",
                      .target_layout_template_hash = "target-layout-template-prepare",
                      .target_index_hash = "target-index-prepare",
                  },
              .device_id = static_cast<int>(rank),
          });
    }
    return requests;
  };

  constexpr size_t kChunkBytes = 16;
  constexpr size_t kMaxCollectiveChunkBytes = 16;
  constexpr size_t kMaxStripeBytes = 8;
  auto first_requests = make_requests(rank0_a.data(), rank1_a.data());
  const auto first = prepare_source_window_routed_program_cache_for_testing(
      "artifact-routed-program-cache-prepare",
      plan,
      absl::MakeConstSpan(first_requests),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(first.status.ok());
  CHECK(first.prepared);
  CHECK_FALSE(first.cache_hit);
  CHECK(first.plan_hash == "prepare-plan-hash-a");
  CHECK(first.runtime_chunk_count == 1);
  CHECK(first.compiled_chunk_count == 1);
  auto stats = source_window_routed_program_cache_stats_for_testing();
  CHECK(stats.misses == 1);
  CHECK(stats.hits == 0);
  CHECK(stats.entries == 1);

  auto second_requests = make_requests(rank0_b.data(), rank1_b.data());
  const auto second = prepare_source_window_routed_program_cache_for_testing(
      "artifact-routed-program-cache-prepare",
      plan,
      absl::MakeConstSpan(second_requests),
      kChunkBytes,
      kMaxCollectiveChunkBytes,
      kMaxStripeBytes);
  REQUIRE(second.status.ok());
  CHECK(second.prepared);
  CHECK(second.cache_hit);
  CHECK(second.runtime_chunk_count == 1);
  CHECK(second.compiled_chunk_count == 1);
  stats = source_window_routed_program_cache_stats_for_testing();
  CHECK(stats.misses == 1);
  CHECK(stats.hits == 1);
  CHECK(stats.entries == 1);

  clear_source_window_routed_program_cache_for_testing();
}

TEST_CASE(
    "source-window collective routed program cache can be prepared from group requests",
    "[collective_disk_loader][source_window][routed_program][cache][prepare]") {
  clear_source_window_collective_plan_cache_for_testing();
  clear_source_window_routed_program_cache_for_testing();

  const auto source = make_u8_tensor_spec("src", 0, {4, 8}, {8, 1});
  const auto destination = make_u8_tensor_spec("dst", 0, {4, 4}, {4, 1});
  auto make_dim1_item = [&](int64_t col_begin, int64_t col_end) {
    materialization::contracts::RepresentationWorkItem item;
    item.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
    item.partition_kind = materialization::contracts::WorkPartitionKind::kUnknown;
    item.dst_name = "dst";
    item.dst_spec = destination;
    item.committed_bytes = 16;
    item.sources.push_back(
        materialization::contracts::RepresentationWorkSourceFragment{
            .fragment =
                materialization::contracts::SourceFragment{
                    .source_spec = source,
                    .source_range =
                        materialization::contracts::TensorCoordinateSpec{
                            .axes =
                                {
                                    materialization::contracts::TensorAxisRange{
                                        .dim = 1,
                                        .start = col_begin,
                                        .end = col_end,
                                    },
                                },
                        },
                    .destination_range = materialization::contracts::TensorCoordinateSpec{},
                },
        });
    return item;
  };
  const materialization::contracts::RepresentationWorkPlan rank0_plan{
      .items = {make_dim1_item(0, 4)},
      .committed_bytes = 16,
  };
  const materialization::contracts::RepresentationWorkPlan rank1_plan{
      .items = {make_dim1_item(4, 8)},
      .committed_bytes = 16,
  };
  const auto shared_rank0_plan = std::make_shared<const materialization::contracts::RepresentationWorkPlan>(rank0_plan);
  const auto shared_rank1_plan = std::make_shared<const materialization::contracts::RepresentationWorkPlan>(rank1_plan);

  CollectiveMappedTargetLoadOptions options;
  options.chunk_bytes = 16;
  options.enable_source_window_plan_cache = true;
  options.strategy_config.enable_source_window_collective = true;
  options.strategy_config.enable_source_window_compiled_routed_program = true;
  options.strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto;
  options.strategy_config.source_window_collective_distribution_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kConsumerRouted;
  options.strategy_config.source_window_collective_min_rank_read_saving_bytes = 0;
  options.strategy_config.source_window_collective_min_routed_peer_saving_bytes = 0;
  options.strategy_config.source_window_collective_window_bytes = 64;
  options.strategy_config.source_window_collective_peak_bytes_budget = 64;
  options.strategy_config.source_window_collective_max_peer_to_read_ratio_x1000 = 2000;

  std::array<std::uint8_t, 16> rank0_a{};
  std::array<std::uint8_t, 16> rank1_a{};
  std::array<std::uint8_t, 16> rank0_b{};
  std::array<std::uint8_t, 16> rank1_b{};
  auto make_target_layout = [](void* ptr) {
    return std::make_shared<const loading::IntoTargetLayout>(loading::IntoTargetLayout{
        .storages =
            {
                loading::IntoTargetStorage{
                    .base_ptr = gsl::not_null<void*>{ptr},
                    .length = 16,
                },
            },
        .total_size = 16,
    });
  };
  auto prepared_facts_for = [](uint32_t rank) {
    return loading::SourceWindowPreparedRealizationFacts{
        .group_key = "prepared-group-from-requests",
        .member_key = absl::StrCat("prepared-member-from-requests-", rank),
        .realization_plan_hash = "prepared-realization-plan-from-requests",
        .target_layout_template_hash = "prepared-target-layout-template-from-requests",
        .target_index_hash = "prepared-target-index-from-requests",
    };
  };
  auto make_requests = [&](void* rank0_ptr, void* rank1_ptr) {
    std::vector<SourceWindowCollectiveMappedTargetLoadRequest> requests;
    requests.reserve(2);
    for (uint32_t rank = 0; rank < 2; ++rank) {
      requests.push_back(
          SourceWindowCollectiveMappedTargetLoadRequest{
              .artifact_id = "artifact-routed-program-cache-prepare-from-requests",
              .group =
                  loading::CollectiveLoadGroupHint{
                      .group_id = "source-window-prepare-from-requests",
                      .world_size = 2,
                      .rank = rank,
                  },
              .disk_context = make_disk_context(),
              .representation_work_plan_ref = rank == 0 ? shared_rank0_plan : shared_rank1_plan,
              .target_layout_ref = make_target_layout(rank == 0 ? rank0_ptr : rank1_ptr),
              .candidate_summary = make_source_window_candidate_summary(
                  runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kAuto,
                  runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted),
              .source_index_digest = "source-index-prepare-from-requests",
              .prepared_realization = prepared_facts_for(rank),
              .device_id = static_cast<int>(rank),
          });
    }
    return requests;
  };

  auto first_requests = make_requests(rank0_a.data(), rank1_a.data());
  const auto first_plan = prepare_source_window_collective_plan_cache(absl::MakeConstSpan(first_requests), options);
  REQUIRE(first_plan.status.ok());
  CHECK(first_plan.prepared);
  CHECK_FALSE(first_plan.cache_hit);
  CHECK_FALSE(first_plan.plan_hash.empty());

  auto plan_stats = source_window_collective_plan_cache_stats_for_testing();
  CHECK(plan_stats.misses == 1);
  CHECK(plan_stats.hits == 0);
  CHECK(plan_stats.entries == 1);
  auto program_stats = source_window_routed_program_cache_stats_for_testing();
  CHECK(program_stats.misses == 0);
  CHECK(program_stats.hits == 0);
  CHECK(program_stats.entries == 0);

  auto second_plan_requests = make_requests(rank0_b.data(), rank1_b.data());
  const auto second_plan =
      prepare_source_window_collective_plan_cache(absl::MakeConstSpan(second_plan_requests), options);
  REQUIRE(second_plan.status.ok());
  CHECK(second_plan.prepared);
  CHECK(second_plan.cache_hit);
  CHECK(second_plan.plan_hash == first_plan.plan_hash);

  plan_stats = source_window_collective_plan_cache_stats_for_testing();
  CHECK(plan_stats.misses == 1);
  CHECK(plan_stats.hits == 1);
  CHECK(plan_stats.entries == 1);
  program_stats = source_window_routed_program_cache_stats_for_testing();
  CHECK(program_stats.misses == 0);
  CHECK(program_stats.hits == 0);
  CHECK(program_stats.entries == 0);

  const auto first =
      prepare_source_window_collective_routed_program_cache(absl::MakeConstSpan(first_requests), options);
  REQUIRE(first.status.ok());
  CHECK(first.prepared);
  CHECK_FALSE(first.cache_hit);
  CHECK(first.plan_hash == first_plan.plan_hash);
  CHECK(first.runtime_chunk_count == 1);
  CHECK(first.compiled_chunk_count == 1);

  plan_stats = source_window_collective_plan_cache_stats_for_testing();
  CHECK(plan_stats.misses == 1);
  CHECK(plan_stats.hits == 2);
  CHECK(plan_stats.entries == 1);
  program_stats = source_window_routed_program_cache_stats_for_testing();
  CHECK(program_stats.misses == 1);
  CHECK(program_stats.hits == 0);
  CHECK(program_stats.entries == 1);

  auto second_requests = make_requests(rank0_b.data(), rank1_b.data());
  const auto second =
      prepare_source_window_collective_routed_program_cache(absl::MakeConstSpan(second_requests), options);
  REQUIRE(second.status.ok());
  CHECK(second.prepared);
  CHECK(second.cache_hit);
  CHECK(second.plan_hash == first.plan_hash);
  CHECK(second.runtime_chunk_count == 1);
  CHECK(second.compiled_chunk_count == 1);

  plan_stats = source_window_collective_plan_cache_stats_for_testing();
  CHECK(plan_stats.misses == 1);
  CHECK(plan_stats.hits == 3);
  CHECK(plan_stats.entries == 1);
  program_stats = source_window_routed_program_cache_stats_for_testing();
  CHECK(program_stats.misses == 1);
  CHECK(program_stats.hits == 1);
  CHECK(program_stats.entries == 1);

  clear_source_window_collective_plan_cache_for_testing();
  clear_source_window_routed_program_cache_for_testing();
}

TEST_CASE(
    "try_source_window_collective_mapped_target_load admits hybrid-window plans before runtime fallback",
    "[collective_disk_loader][source_window][admission][hybrid]") {
  std::array<std::uint8_t, 8> rank0_target{};
  std::array<std::uint8_t, 8> rank1_target{};
  const auto source = make_u8_tensor_spec("src", 0, {8}, {1});
  const auto destination = make_u8_tensor_spec("dst", 0, {8}, {1});
  const materialization::contracts::RepresentationWorkPlan work_plan{
      .items = {make_u8_copy_item(source, destination, 8)},
      .committed_bytes = 8,
  };

  CollectiveMappedTargetLoadOptions options;
  options.strategy_config.enable_source_window_collective = true;
  options.strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto;
  options.strategy_config.source_window_collective_distribution_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kHybridWindow;
  options.strategy_config.source_window_collective_min_rank_read_saving_bytes = 0;
  options.strategy_config.source_window_collective_min_routed_peer_saving_bytes = 0;
  options.strategy_config.source_window_collective_window_bytes = 16;
  options.strategy_config.source_window_collective_peak_bytes_budget = 64;
  options.strategy_config.source_window_collective_max_peer_to_read_ratio_x1000 = 2000;

  auto make_request = [&](uint32_t rank, void* target) {
    return SourceWindowCollectiveMappedTargetLoadRequest{
        .artifact_id = "artifact_source_window_hybrid_admission",
        .group =
            loading::CollectiveLoadGroupHint{
                .group_id = "source-window-admission-hybrid",
                .world_size = 2,
                .rank = rank,
            },
        .disk_context = make_disk_context(),
        .representation_work_plan = work_plan,
        .target_layout =
            loading::IntoTargetLayout{
                .storages =
                    {
                        loading::IntoTargetStorage{
                            .base_ptr = gsl::not_null<void*>{target},
                            .length = 8,
                        },
                    },
                .total_size = 8,
            },
        .candidate_summary = make_source_window_candidate_summary(
            runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kAuto,
            runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kHybridWindow),
        .source_index_digest = "source-index-hybrid",
        .device_id = static_cast<int>(rank),
    };
  };

  std::array<SourceWindowCollectiveMappedTargetLoadResult, 2> results;
  std::thread rank0([&]() {
    results[0] = try_source_window_collective_mapped_target_load(
        make_request(0, rank0_target.data()), nullptr, std::chrono::milliseconds(1000), options);
  });
  std::thread rank1([&]() {
    results[1] = try_source_window_collective_mapped_target_load(
        make_request(1, rank1_target.data()), nullptr, std::chrono::milliseconds(1000), options);
  });
  rank0.join();
  rank1.join();

  for (const auto& result : results) {
    CHECK_FALSE(result.handled);
    CHECK(result.status.ok());
    CHECK(result.skip_reason == "source_window_runtime_unavailable:pinned_pool_missing");
    CHECK(result.metrics.source_window_distribution_mode == "hybrid_window");
    CHECK(result.metrics.source_window_group_disk_read_bytes == 8);
    CHECK(result.metrics.source_window_target_write_bytes == 16);
    CHECK(result.metrics.source_window_peer_transfer_bytes == 8);
    CHECK(result.metrics.source_window_peer_waste_bytes == 0);
  }
  CHECK(results[0].plan_hash == results[1].plan_hash);
}

TEST_CASE(
    "try_source_window_collective_mapped_target_load admits hybrid routed windows before runtime fallback",
    "[collective_disk_loader][source_window][admission][hybrid]") {
  constexpr uint64_t kRankBytes = 128ULL * 1024ULL * 1024ULL;
  std::array<std::uint8_t, 1> rank0_target{};
  std::array<std::uint8_t, 1> rank1_target{};
  const auto rank0_source = make_u8_tensor_spec("src", 0, {static_cast<int64_t>(kRankBytes)}, {1});
  const auto rank1_source = make_u8_tensor_spec("src", kRankBytes, {static_cast<int64_t>(kRankBytes)}, {1});
  const auto destination = make_u8_tensor_spec("dst", 0, {static_cast<int64_t>(kRankBytes)}, {1});
  const materialization::contracts::RepresentationWorkPlan rank0_work_plan{
      .items = {make_u8_copy_item(rank0_source, destination, kRankBytes)},
      .committed_bytes = kRankBytes,
  };
  const materialization::contracts::RepresentationWorkPlan rank1_work_plan{
      .items = {make_u8_copy_item(rank1_source, destination, kRankBytes)},
      .committed_bytes = kRankBytes,
  };

  CollectiveMappedTargetLoadOptions options;
  options.strategy_config.enable_source_window_collective = true;
  options.strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto;
  options.strategy_config.source_window_collective_distribution_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kHybridWindow;
  options.strategy_config.source_window_collective_min_rank_read_saving_bytes = 0;
  options.strategy_config.source_window_collective_min_routed_peer_saving_bytes = 0;
  options.strategy_config.source_window_collective_window_bytes = 2 * kRankBytes;
  options.strategy_config.source_window_collective_peak_bytes_budget = 4ULL * 1024ULL * 1024ULL * 1024ULL;
  options.strategy_config.source_window_collective_max_peer_to_read_ratio_x1000 = 2000;

  auto make_request = [&](uint32_t rank, void* target) {
    return SourceWindowCollectiveMappedTargetLoadRequest{
        .artifact_id = "artifact_source_window_hybrid_routed_admission",
        .group =
            loading::CollectiveLoadGroupHint{
                .group_id = "source-window-admission-hybrid-routed",
                .world_size = 2,
                .rank = rank,
            },
        .disk_context = make_disk_context(),
        .representation_work_plan = rank == 0 ? rank0_work_plan : rank1_work_plan,
        .target_layout =
            loading::IntoTargetLayout{
                .storages =
                    {
                        loading::IntoTargetStorage{
                            .base_ptr = gsl::not_null<void*>{target},
                            .length = kRankBytes,
                        },
                    },
                .total_size = kRankBytes,
            },
        .candidate_summary = make_source_window_candidate_summary(
            runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kAuto,
            runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kHybridWindow),
        .source_index_digest = "source-index-hybrid-routed",
        .device_id = static_cast<int>(rank),
    };
  };

  std::array<SourceWindowCollectiveMappedTargetLoadResult, 2> results;
  std::thread rank0([&]() {
    results[0] = try_source_window_collective_mapped_target_load(
        make_request(0, rank0_target.data()), nullptr, std::chrono::milliseconds(1000), options);
  });
  std::thread rank1([&]() {
    results[1] = try_source_window_collective_mapped_target_load(
        make_request(1, rank1_target.data()), nullptr, std::chrono::milliseconds(1000), options);
  });
  rank0.join();
  rank1.join();

  for (const auto& result : results) {
    CHECK_FALSE(result.handled);
    CHECK(result.status.ok());
    CHECK(result.skip_reason == "source_window_runtime_unavailable:pinned_pool_missing");
    CHECK(result.metrics.source_window_distribution_mode == "hybrid_window");
    CHECK(result.metrics.source_window_group_disk_read_bytes == 2 * kRankBytes);
    CHECK(result.metrics.source_window_target_write_bytes == 2 * kRankBytes);
    CHECK(result.metrics.source_window_peer_transfer_bytes == kRankBytes);
    CHECK(result.metrics.source_window_peer_waste_bytes == 0);
  }
  CHECK(results[0].plan_hash == results[1].plan_hash);
}

TEST_CASE(
    "try_source_window_collective_mapped_target_load strict mode fails before target mutation when runtime is absent",
    "[collective_disk_loader][source_window][admission]") {
  std::array<std::uint8_t, 8> rank0_target{};
  std::array<std::uint8_t, 8> rank1_target{};
  const auto source = make_u8_tensor_spec("src", 0, {8}, {1});
  const auto destination = make_u8_tensor_spec("dst", 0, {8}, {1});
  const materialization::contracts::RepresentationWorkPlan work_plan{
      .items = {make_u8_copy_item(source, destination, 8)},
      .committed_bytes = 8,
  };

  CollectiveMappedTargetLoadOptions options;
  options.strategy_config.enable_source_window_collective = true;
  options.strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kStrict;
  options.strategy_config.source_window_collective_min_rank_read_saving_bytes = 0;
  options.strategy_config.source_window_collective_window_bytes = 16;
  options.strategy_config.source_window_collective_peak_bytes_budget = 64;

  auto make_request = [&](uint32_t rank, void* target) {
    return SourceWindowCollectiveMappedTargetLoadRequest{
        .artifact_id = "artifact_source_window_strict",
        .group =
            loading::CollectiveLoadGroupHint{
                .group_id = "source-window-admission-strict",
                .world_size = 2,
                .rank = rank,
            },
        .disk_context = make_disk_context(),
        .representation_work_plan = work_plan,
        .target_layout =
            loading::IntoTargetLayout{
                .storages =
                    {
                        loading::IntoTargetStorage{
                            .base_ptr = gsl::not_null<void*>{target},
                            .length = 8,
                        },
                    },
                .total_size = 8,
            },
        .candidate_summary = make_source_window_candidate_summary(
            runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kStrict),
        .source_index_digest = "source-index-a",
        .device_id = static_cast<int>(rank),
    };
  };

  std::array<SourceWindowCollectiveMappedTargetLoadResult, 2> results;
  std::thread rank0([&]() {
    results[0] = try_source_window_collective_mapped_target_load(
        make_request(0, rank0_target.data()), nullptr, std::chrono::milliseconds(1000), options);
  });
  std::thread rank1([&]() {
    results[1] = try_source_window_collective_mapped_target_load(
        make_request(1, rank1_target.data()), nullptr, std::chrono::milliseconds(1000), options);
  });
  rank0.join();
  rank1.join();

  for (const auto& result : results) {
    CHECK(result.handled);
    CHECK(absl::IsFailedPrecondition(result.status));
    CHECK(result.skip_reason == "source_window_runtime_unavailable:pinned_pool_missing");
    CHECK_FALSE(result.plan_hash.empty());
  }
  CHECK(results[0].plan_hash == results[1].plan_hash);
  CHECK(rank0_target == std::array<std::uint8_t, 8>{});
  CHECK(rank1_target == std::array<std::uint8_t, 8>{});
}

TEST_CASE(
    "try_source_window_collective_mapped_target_load broadcasts source windows into rank targets",
    "[collective_disk_loader][source_window][correctness]") {
  SKIP_IF_NO_CUDA();
  const char* cuda_backend = std::getenv("TENSORCAST_CUDA_BACKEND");
  if (cuda_backend != nullptr && std::string_view(cuda_backend) == "fake") {
    SKIP("source-window collective correctness requires real NCCL devices");
  }
  int device_count = 0;
  REQUIRE(tensorcast::cuda::get_device_count(&device_count).ok());
  if (device_count < 2) {
    SKIP("source-window collective correctness requires at least two CUDA devices");
  }

  constexpr uint64_t kTotalBytes = 8;
  auto temp_root = make_temp_dir("collective-disk-loader-source-window");
  const auto safetensors_path = temp_root / "weights.safetensors";
  const std::vector<unsigned char> payload = {0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38};
  create_safetensors_file(
      safetensors_path, "{\"src\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}", payload);

  auto disk_context_or = loader::get_disk_artifact_context(temp_root);
  REQUIRE(disk_context_or.ok());
  REQUIRE((*disk_context_or)->is_safetensors());

  void* rank0_buffer = nullptr;
  void* rank1_buffer = nullptr;
  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::malloc(&rank0_buffer, kTotalBytes).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::malloc(&rank1_buffer, kTotalBytes).ok());
  std::array<uint8_t, kTotalBytes> initial{};
  initial.fill(0xEE);
  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::memcpy(rank0_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::memcpy(rank1_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  const auto source = make_u8_tensor_spec("src", 0, {8}, {1});
  const auto destination = make_u8_tensor_spec("dst", 0, {8}, {1});
  const materialization::contracts::RepresentationWorkPlan work_plan{
      .items = {make_u8_copy_item(source, destination, kTotalBytes)},
      .committed_bytes = kTotalBytes,
  };

  CollectiveMappedTargetLoadOptions options;
  options.chunk_bytes = 4;
  options.streaming_buffer_chunks = 2;
  options.strategy_config.enable_source_window_collective = true;
  options.strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto;
  options.strategy_config.source_window_collective_min_rank_read_saving_bytes = 0;
  options.strategy_config.source_window_collective_window_bytes = 16;
  options.strategy_config.source_window_collective_peak_bytes_budget = 64;
  options.strategy_config.source_window_collective_max_peer_to_read_ratio_x1000 = 2000;

  auto make_request = [&](uint32_t rank, void* target) {
    return SourceWindowCollectiveMappedTargetLoadRequest{
        .artifact_id = "artifact_source_window_correctness",
        .group =
            loading::CollectiveLoadGroupHint{
                .group_id = "source-window-correctness",
                .world_size = 2,
                .rank = rank,
            },
        .disk_context = *disk_context_or,
        .representation_work_plan = work_plan,
        .target_layout =
            loading::IntoTargetLayout{
                .storages =
                    {
                        loading::IntoTargetStorage{
                            .base_ptr = gsl::not_null<void*>{target},
                            .length = kTotalBytes,
                        },
                    },
                .total_size = kTotalBytes,
            },
        .candidate_summary = make_source_window_candidate_summary(
            runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kAuto),
        .source_index_digest = "source-index-correctness",
        .device_id = static_cast<int>(rank),
    };
  };

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(16, 4);
  std::array<SourceWindowCollectiveMappedTargetLoadResult, 2> results;
  std::thread rank0([&]() {
    results[0] = try_source_window_collective_mapped_target_load(
        make_request(0, rank0_buffer), pinned_pool, std::chrono::milliseconds(1000), options);
  });
  std::thread rank1([&]() {
    results[1] = try_source_window_collective_mapped_target_load(
        make_request(1, rank1_buffer), pinned_pool, std::chrono::milliseconds(1000), options);
  });
  rank0.join();
  rank1.join();

  for (const auto& result : results) {
    REQUIRE(result.handled);
    REQUIRE(result.status.ok());
    CHECK(result.skip_reason.empty());
    CHECK_FALSE(result.plan_hash.empty());
    CHECK(result.metrics.source_window_group_disk_read_bytes == kTotalBytes);
    CHECK(result.metrics.source_window_target_write_bytes == 2 * kTotalBytes);
    CHECK(result.metrics.source_window_window_count == 1);
    CHECK(result.metrics.batch_count == 2);
  }
  CHECK(results[0].plan_hash == results[1].plan_hash);

  std::array<uint8_t, kTotalBytes> rank0_actual{};
  std::array<uint8_t, kTotalBytes> rank1_actual{};
  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(
      tensorcast::cuda::memcpy(rank0_actual.data(), rank0_buffer, rank0_actual.size(), cudaMemcpyDeviceToHost).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(
      tensorcast::cuda::memcpy(rank1_actual.data(), rank1_buffer, rank1_actual.size(), cudaMemcpyDeviceToHost).ok());
  const std::array<uint8_t, kTotalBytes> expected = {0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38};
  CHECK(rank0_actual == expected);
  CHECK(rank1_actual == expected);

  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::free(rank0_buffer).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::free(rank1_buffer).ok());
  loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "try_source_window_collective_mapped_target_load keeps local-only windows on owner rank",
    "[collective_disk_loader][source_window][local_only][correctness]") {
  SKIP_IF_NO_CUDA();
  const char* cuda_backend = std::getenv("TENSORCAST_CUDA_BACKEND");
  if (cuda_backend != nullptr && std::string_view(cuda_backend) == "fake") {
    SKIP("source-window local-only correctness requires real NCCL devices");
  }
  int device_count = 0;
  REQUIRE(tensorcast::cuda::get_device_count(&device_count).ok());
  if (device_count < 2) {
    SKIP("source-window local-only correctness requires at least two CUDA devices");
  }

  constexpr uint64_t kTotalBytes = 8;
  auto temp_root = make_temp_dir("collective-disk-loader-source-window-local-only");
  const auto safetensors_path = temp_root / "weights.safetensors";
  const std::vector<unsigned char> payload = {0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48};
  create_safetensors_file(
      safetensors_path, "{\"src\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}", payload);

  auto disk_context_or = loader::get_disk_artifact_context(temp_root);
  REQUIRE(disk_context_or.ok());
  REQUIRE((*disk_context_or)->is_safetensors());

  void* rank0_buffer = nullptr;
  void* rank1_buffer = nullptr;
  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::malloc(&rank0_buffer, kTotalBytes).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::malloc(&rank1_buffer, kTotalBytes).ok());
  std::array<uint8_t, kTotalBytes> initial{};
  initial.fill(0xEE);
  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::memcpy(rank0_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::memcpy(rank1_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  const auto source = make_u8_tensor_spec("src", 0, {8}, {1});
  const auto destination = make_u8_tensor_spec("dst", 0, {8}, {1});
  const materialization::contracts::RepresentationWorkPlan rank0_plan{
      .items = {make_u8_copy_item(source, destination, kTotalBytes)},
      .committed_bytes = kTotalBytes,
  };
  materialization::contracts::RepresentationWorkItem rank1_local_item;
  rank1_local_item.kind = materialization::contracts::RepresentationWorkItemKind::kConstFill;
  rank1_local_item.dst_name = destination.name;
  rank1_local_item.dst_spec = destination;
  rank1_local_item.committed_bytes = 0;
  const materialization::contracts::RepresentationWorkPlan rank1_plan{
      .items = {rank1_local_item},
      .committed_bytes = 0,
  };

  CollectiveMappedTargetLoadOptions options;
  options.chunk_bytes = 4;
  options.streaming_buffer_chunks = 2;
  options.strategy_config.enable_source_window_collective = true;
  options.strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto;
  options.strategy_config.source_window_collective_distribution_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kLocalOnly;
  options.strategy_config.source_window_collective_min_rank_read_saving_bytes = 0;
  options.strategy_config.source_window_collective_window_bytes = 16;
  options.strategy_config.source_window_collective_peak_bytes_budget = 64;
  options.strategy_config.source_window_collective_max_peer_to_read_ratio_x1000 = 2000;

  auto make_request = [&](uint32_t rank, void* target) {
    return SourceWindowCollectiveMappedTargetLoadRequest{
        .artifact_id = "artifact_source_window_local_only",
        .group =
            loading::CollectiveLoadGroupHint{
                .group_id = "source-window-local-only-correctness",
                .world_size = 2,
                .rank = rank,
            },
        .disk_context = *disk_context_or,
        .representation_work_plan = rank == 0 ? rank0_plan : rank1_plan,
        .target_layout =
            loading::IntoTargetLayout{
                .storages =
                    {
                        loading::IntoTargetStorage{
                            .base_ptr = gsl::not_null<void*>{target},
                            .length = kTotalBytes,
                        },
                    },
                .total_size = kTotalBytes,
            },
        .candidate_summary = make_source_window_candidate_summary(
            runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kAuto,
            runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kLocalOnly),
        .source_index_digest = "source-index-local-only-correctness",
        .device_id = static_cast<int>(rank),
    };
  };

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(16, 4);
  std::array<SourceWindowCollectiveMappedTargetLoadResult, 2> results;
  std::thread rank0([&]() {
    results[0] = try_source_window_collective_mapped_target_load(
        make_request(0, rank0_buffer), pinned_pool, std::chrono::milliseconds(1000), options);
  });
  std::thread rank1([&]() {
    results[1] = try_source_window_collective_mapped_target_load(
        make_request(1, rank1_buffer), pinned_pool, std::chrono::milliseconds(1000), options);
  });
  rank0.join();
  rank1.join();

  for (const auto& result : results) {
    REQUIRE(result.handled);
    REQUIRE(result.status.ok());
    CHECK(result.skip_reason.empty());
    CHECK_FALSE(result.plan_hash.empty());
    CHECK(result.metrics.source_window_group_disk_read_bytes == kTotalBytes);
    CHECK(result.metrics.source_window_target_write_bytes == kTotalBytes);
    CHECK(result.metrics.source_window_peer_transfer_bytes == 0);
    CHECK(result.metrics.source_window_window_count == 1);
    CHECK(result.metrics.batch_count == 2);
  }
  CHECK(results[0].plan_hash == results[1].plan_hash);

  std::array<uint8_t, kTotalBytes> rank0_actual{};
  std::array<uint8_t, kTotalBytes> rank1_actual{};
  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(
      tensorcast::cuda::memcpy(rank0_actual.data(), rank0_buffer, rank0_actual.size(), cudaMemcpyDeviceToHost).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(
      tensorcast::cuda::memcpy(rank1_actual.data(), rank1_buffer, rank1_actual.size(), cudaMemcpyDeviceToHost).ok());
  const std::array<uint8_t, kTotalBytes> expected = {0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48};
  CHECK(rank0_actual == expected);
  CHECK(rank1_actual == initial);

  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::free(rank0_buffer).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::free(rank1_buffer).ok());
  loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "try_source_window_collective_mapped_target_load routes dim1 source windows only to consumers",
    "[collective_disk_loader][source_window][consumer_routed][correctness]") {
  SKIP_IF_NO_CUDA();
  const char* cuda_backend = std::getenv("TENSORCAST_CUDA_BACKEND");
  if (cuda_backend != nullptr && std::string_view(cuda_backend) == "fake") {
    SKIP("source-window consumer-routed correctness requires real NCCL devices");
  }
  int device_count = 0;
  REQUIRE(tensorcast::cuda::get_device_count(&device_count).ok());
  if (device_count < 2) {
    SKIP("source-window consumer-routed correctness requires at least two CUDA devices");
  }

  constexpr uint64_t kSourceBytes = 32;
  constexpr uint64_t kTargetBytes = 16;
  auto temp_root = make_temp_dir("collective-disk-loader-source-window-routed");
  const auto safetensors_path = temp_root / "weights.safetensors";
  std::vector<unsigned char> payload(kSourceBytes);
  for (uint64_t index = 0; index < kSourceBytes; ++index) {
    payload[static_cast<size_t>(index)] = static_cast<unsigned char>(0x10 + index);
  }
  create_safetensors_file(
      safetensors_path, "{\"src\":{\"dtype\":\"U8\",\"shape\":[4,8],\"data_offsets\":[0,32]}}", payload);

  auto disk_context_or = loader::get_disk_artifact_context(temp_root);
  REQUIRE(disk_context_or.ok());
  REQUIRE((*disk_context_or)->is_safetensors());

  void* rank0_buffer = nullptr;
  void* rank1_buffer = nullptr;
  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::malloc(&rank0_buffer, kTargetBytes).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::malloc(&rank1_buffer, kTargetBytes).ok());
  std::array<uint8_t, kTargetBytes> initial{};
  initial.fill(0xEE);
  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::memcpy(rank0_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::memcpy(rank1_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  const auto source = make_u8_tensor_spec("src", 0, {4, 8}, {8, 1});
  const auto destination = make_u8_tensor_spec("dst", 0, {4, 4}, {4, 1});
  auto make_dim1_item = [&](int64_t col_begin, int64_t col_end) {
    materialization::contracts::RepresentationWorkItem item;
    item.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
    item.partition_kind = materialization::contracts::WorkPartitionKind::kUnknown;
    item.dst_name = "dst";
    item.dst_spec = destination;
    item.committed_bytes = kTargetBytes;
    item.sources.push_back(
        materialization::contracts::RepresentationWorkSourceFragment{
            .fragment =
                materialization::contracts::SourceFragment{
                    .source_spec = source,
                    .source_range =
                        materialization::contracts::TensorCoordinateSpec{
                            .axes =
                                {
                                    materialization::contracts::TensorAxisRange{
                                        .dim = 1,
                                        .start = col_begin,
                                        .end = col_end,
                                    },
                                },
                        },
                    .destination_range = materialization::contracts::TensorCoordinateSpec{},
                },
        });
    return item;
  };
  const materialization::contracts::RepresentationWorkPlan rank0_plan{
      .items = {make_dim1_item(0, 4)},
      .committed_bytes = kTargetBytes,
  };
  const materialization::contracts::RepresentationWorkPlan rank1_plan{
      .items = {make_dim1_item(4, 8)},
      .committed_bytes = kTargetBytes,
  };

  CollectiveMappedTargetLoadOptions options;
  options.chunk_bytes = 16;
  options.streaming_buffer_chunks = 2;
  options.strategy_config.enable_source_window_collective = true;
  options.strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto;
  options.strategy_config.source_window_collective_distribution_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kConsumerRouted;
  options.strategy_config.source_window_collective_min_rank_read_saving_bytes = 0;
  options.strategy_config.source_window_collective_window_bytes = 64;
  options.strategy_config.source_window_collective_peak_bytes_budget = 64;
  options.strategy_config.source_window_collective_max_peer_to_read_ratio_x1000 = 2000;

  auto make_request = [&](uint32_t rank, void* target) {
    return SourceWindowCollectiveMappedTargetLoadRequest{
        .artifact_id = "artifact_source_window_routed",
        .group =
            loading::CollectiveLoadGroupHint{
                .group_id = "source-window-routed-correctness",
                .world_size = 2,
                .rank = rank,
            },
        .disk_context = *disk_context_or,
        .representation_work_plan = rank == 0 ? rank0_plan : rank1_plan,
        .target_layout =
            loading::IntoTargetLayout{
                .storages =
                    {
                        loading::IntoTargetStorage{
                            .base_ptr = gsl::not_null<void*>{target},
                            .length = kTargetBytes,
                        },
                    },
                .total_size = kTargetBytes,
            },
        .candidate_summary = make_source_window_candidate_summary(
            runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kAuto,
            runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted),
        .source_index_digest = "source-index-routed-correctness",
        .device_id = static_cast<int>(rank),
    };
  };

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(16, 4);
  std::array<SourceWindowCollectiveMappedTargetLoadResult, 2> results;
  std::thread rank0([&]() {
    results[0] = try_source_window_collective_mapped_target_load(
        make_request(0, rank0_buffer), pinned_pool, std::chrono::milliseconds(1000), options);
  });
  std::thread rank1([&]() {
    results[1] = try_source_window_collective_mapped_target_load(
        make_request(1, rank1_buffer), pinned_pool, std::chrono::milliseconds(1000), options);
  });
  rank0.join();
  rank1.join();

  for (const auto& result : results) {
    REQUIRE(result.handled);
    REQUIRE(result.status.ok());
    CHECK(result.skip_reason.empty());
    CHECK(result.metrics.source_window_group_disk_read_bytes == kSourceBytes);
    CHECK(result.metrics.source_window_target_write_bytes == 2 * kTargetBytes);
    CHECK(result.metrics.source_window_peer_transfer_bytes == kTargetBytes);
    CHECK(result.metrics.batch_count == 1);
  }
  CHECK(results[0].plan_hash == results[1].plan_hash);

  std::array<uint8_t, kTargetBytes> rank0_actual{};
  std::array<uint8_t, kTargetBytes> rank1_actual{};
  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(
      tensorcast::cuda::memcpy(rank0_actual.data(), rank0_buffer, rank0_actual.size(), cudaMemcpyDeviceToHost).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(
      tensorcast::cuda::memcpy(rank1_actual.data(), rank1_buffer, rank1_actual.size(), cudaMemcpyDeviceToHost).ok());
  std::array<uint8_t, kTargetBytes> rank0_expected{};
  std::array<uint8_t, kTargetBytes> rank1_expected{};
  for (uint64_t row = 0; row < 4; ++row) {
    for (uint64_t col = 0; col < 4; ++col) {
      rank0_expected[static_cast<size_t>(row * 4 + col)] = payload[static_cast<size_t>(row * 8 + col)];
      rank1_expected[static_cast<size_t>(row * 4 + col)] = payload[static_cast<size_t>(row * 8 + 4 + col)];
    }
  }
  CHECK(rank0_actual == rank0_expected);
  CHECK(rank1_actual == rank1_expected);

  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::free(rank0_buffer).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::free(rank1_buffer).ok());
  loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "source-window compiled routed program cache rebinds fresh target allocations",
    "[collective_disk_loader][source_window][consumer_routed][compiled_program][cache][correctness]") {
  SKIP_IF_NO_CUDA();
  const char* cuda_backend = std::getenv("TENSORCAST_CUDA_BACKEND");
  if (cuda_backend != nullptr && std::string_view(cuda_backend) == "fake") {
    SKIP("source-window compiled routed program cache correctness requires real NCCL devices");
  }
  int device_count = 0;
  REQUIRE(tensorcast::cuda::get_device_count(&device_count).ok());
  if (device_count < 2) {
    SKIP("source-window compiled routed program cache correctness requires at least two CUDA devices");
  }

  clear_source_window_collective_plan_cache_for_testing();
  clear_source_window_routed_program_cache_for_testing();

  constexpr uint64_t kSourceBytes = 32;
  constexpr uint64_t kTargetBytes = 16;
  auto temp_root = make_temp_dir("collective-disk-loader-source-window-routed-cache");
  const auto safetensors_path = temp_root / "weights.safetensors";
  std::vector<unsigned char> payload(kSourceBytes);
  for (uint64_t index = 0; index < kSourceBytes; ++index) {
    payload[static_cast<size_t>(index)] = static_cast<unsigned char>(0x40 + index);
  }
  create_safetensors_file(
      safetensors_path, "{\"src\":{\"dtype\":\"U8\",\"shape\":[4,8],\"data_offsets\":[0,32]}}", payload);

  auto disk_context_or = loader::get_disk_artifact_context(temp_root);
  REQUIRE(disk_context_or.ok());
  REQUIRE((*disk_context_or)->is_safetensors());

  const auto source = make_u8_tensor_spec("src", 0, {4, 8}, {8, 1});
  const auto destination = make_u8_tensor_spec("dst", 0, {4, 4}, {4, 1});
  auto make_dim1_item = [&](int64_t col_begin, int64_t col_end) {
    materialization::contracts::RepresentationWorkItem item;
    item.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
    item.partition_kind = materialization::contracts::WorkPartitionKind::kUnknown;
    item.dst_name = "dst";
    item.dst_spec = destination;
    item.committed_bytes = kTargetBytes;
    item.sources.push_back(
        materialization::contracts::RepresentationWorkSourceFragment{
            .fragment =
                materialization::contracts::SourceFragment{
                    .source_spec = source,
                    .source_range =
                        materialization::contracts::TensorCoordinateSpec{
                            .axes =
                                {
                                    materialization::contracts::TensorAxisRange{
                                        .dim = 1,
                                        .start = col_begin,
                                        .end = col_end,
                                    },
                                },
                        },
                    .destination_range = materialization::contracts::TensorCoordinateSpec{},
                },
        });
    return item;
  };
  const materialization::contracts::RepresentationWorkPlan rank0_plan{
      .items = {make_dim1_item(0, 4)},
      .committed_bytes = kTargetBytes,
  };
  const materialization::contracts::RepresentationWorkPlan rank1_plan{
      .items = {make_dim1_item(4, 8)},
      .committed_bytes = kTargetBytes,
  };

  CollectiveMappedTargetLoadOptions options;
  options.chunk_bytes = 16;
  options.streaming_buffer_chunks = 2;
  options.enable_source_window_plan_cache = true;
  options.strategy_config.enable_source_window_collective = true;
  options.strategy_config.enable_source_window_batched_scatter_kernel = true;
  options.strategy_config.enable_source_window_compiled_routed_program = true;
  options.strategy_config.source_window_collective_selection_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveSelectionMode::kAuto;
  options.strategy_config.source_window_collective_distribution_mode =
      StoreEngineOptions::MaterializationStrategyConfig::SourceWindowCollectiveDistributionMode::kConsumerRouted;
  options.strategy_config.source_window_collective_min_rank_read_saving_bytes = 0;
  options.strategy_config.source_window_collective_window_bytes = 64;
  options.strategy_config.source_window_collective_peak_bytes_budget = 64;
  options.strategy_config.source_window_collective_max_peer_to_read_ratio_x1000 = 2000;

  auto make_request = [&](std::string_view group_id, uint32_t rank, void* target) {
    return SourceWindowCollectiveMappedTargetLoadRequest{
        .artifact_id = "artifact_source_window_routed_compiled_program_cache",
        .group =
            loading::CollectiveLoadGroupHint{
                .group_id = std::string(group_id),
                .world_size = 2,
                .rank = rank,
            },
        .disk_context = *disk_context_or,
        .representation_work_plan = rank == 0 ? rank0_plan : rank1_plan,
        .target_layout =
            loading::IntoTargetLayout{
                .storages =
                    {
                        loading::IntoTargetStorage{
                            .base_ptr = gsl::not_null<void*>{target},
                            .length = kTargetBytes,
                        },
                    },
                .total_size = kTargetBytes,
            },
        .candidate_summary = make_source_window_candidate_summary(
            runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kAuto,
            runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted),
        .source_index_digest = "source-index-routed-compiled-program-cache",
        .device_id = static_cast<int>(rank),
    };
  };

  auto init_device_buffer = [](int device, void* buffer, uint8_t value) {
    std::array<uint8_t, kTargetBytes> initial{};
    initial.fill(value);
    REQUIRE(tensorcast::cuda::set_device(device).ok());
    REQUIRE(tensorcast::cuda::memcpy(buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());
  };
  auto read_device_buffer = [](int device, void* buffer) {
    std::array<uint8_t, kTargetBytes> actual{};
    REQUIRE(tensorcast::cuda::set_device(device).ok());
    REQUIRE(tensorcast::cuda::memcpy(actual.data(), buffer, actual.size(), cudaMemcpyDeviceToHost).ok());
    return actual;
  };

  auto malloc_target = [](int device) {
    void* buffer = nullptr;
    REQUIRE(tensorcast::cuda::set_device(device).ok());
    REQUIRE(tensorcast::cuda::malloc(&buffer, kTargetBytes).ok());
    return buffer;
  };

  void* run1_rank0_buffer = malloc_target(0);
  void* run1_rank1_buffer = malloc_target(1);
  init_device_buffer(0, run1_rank0_buffer, 0xEE);
  init_device_buffer(1, run1_rank1_buffer, 0xEE);

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(16, 4);
  auto run_group = [&](std::string_view group_id, void* rank0_buffer, void* rank1_buffer) {
    std::array<SourceWindowCollectiveMappedTargetLoadResult, 2> results;
    std::thread rank0([&]() {
      results[0] = try_source_window_collective_mapped_target_load(
          make_request(group_id, 0, rank0_buffer), pinned_pool, std::chrono::milliseconds(1000), options);
    });
    std::thread rank1([&]() {
      results[1] = try_source_window_collective_mapped_target_load(
          make_request(group_id, 1, rank1_buffer), pinned_pool, std::chrono::milliseconds(1000), options);
    });
    rank0.join();
    rank1.join();
    return results;
  };

  std::array<uint8_t, kTargetBytes> rank0_expected{};
  std::array<uint8_t, kTargetBytes> rank1_expected{};
  for (uint64_t row = 0; row < 4; ++row) {
    for (uint64_t col = 0; col < 4; ++col) {
      rank0_expected[static_cast<size_t>(row * 4 + col)] = payload[static_cast<size_t>(row * 8 + col)];
      rank1_expected[static_cast<size_t>(row * 4 + col)] = payload[static_cast<size_t>(row * 8 + 4 + col)];
    }
  }
  std::array<uint8_t, kTargetBytes> stale_guard{};
  stale_guard.fill(0xA5);

  const auto first_results =
      run_group("source-window-routed-compiled-program-cache-a", run1_rank0_buffer, run1_rank1_buffer);
  for (const auto& result : first_results) {
    REQUIRE(result.handled);
    REQUIRE(result.status.ok());
    CHECK(result.skip_reason.empty());
    CHECK(result.metrics.source_window_group_disk_read_bytes == kSourceBytes);
    CHECK(result.metrics.source_window_target_write_bytes == 2 * kTargetBytes);
    CHECK(result.metrics.source_window_peer_transfer_bytes == kTargetBytes);
  }
  CHECK(read_device_buffer(0, run1_rank0_buffer) == rank0_expected);
  CHECK(read_device_buffer(1, run1_rank1_buffer) == rank1_expected);
  const auto first_routed_stats = source_window_routed_program_cache_stats_for_testing();
  CHECK(first_routed_stats.misses == 1);
  CHECK(first_routed_stats.hits == 0);
  CHECK(first_routed_stats.entries == 1);

  init_device_buffer(0, run1_rank0_buffer, 0xA5);
  init_device_buffer(1, run1_rank1_buffer, 0xA5);
  void* run2_rank0_buffer = malloc_target(0);
  void* run2_rank1_buffer = malloc_target(1);
  init_device_buffer(0, run2_rank0_buffer, 0xCC);
  init_device_buffer(1, run2_rank1_buffer, 0xCC);

  const auto second_results =
      run_group("source-window-routed-compiled-program-cache-b", run2_rank0_buffer, run2_rank1_buffer);
  for (const auto& result : second_results) {
    REQUIRE(result.handled);
    REQUIRE(result.status.ok());
    CHECK(result.skip_reason.empty());
    CHECK(result.plan_cache_hit);
  }
  CHECK(first_results[0].plan_hash == second_results[0].plan_hash);
  CHECK(read_device_buffer(0, run2_rank0_buffer) == rank0_expected);
  CHECK(read_device_buffer(1, run2_rank1_buffer) == rank1_expected);
  CHECK(read_device_buffer(0, run1_rank0_buffer) == stale_guard);
  CHECK(read_device_buffer(1, run1_rank1_buffer) == stale_guard);

  const auto final_routed_stats = source_window_routed_program_cache_stats_for_testing();
  CHECK(final_routed_stats.misses == 1);
  CHECK(final_routed_stats.hits == 1);
  CHECK(final_routed_stats.entries == 1);
  const auto final_plan_stats = source_window_collective_plan_cache_stats_for_testing();
  CHECK(final_plan_stats.misses == 1);
  CHECK(final_plan_stats.hits == 1);
  CHECK(final_plan_stats.entries == 1);

  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::free(run1_rank0_buffer).ok());
  REQUIRE(tensorcast::cuda::free(run2_rank0_buffer).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::free(run1_rank1_buffer).ok());
  REQUIRE(tensorcast::cuda::free(run2_rank1_buffer).ok());
  clear_source_window_routed_program_cache_for_testing();
  clear_source_window_collective_plan_cache_for_testing();
  loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

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
  constexpr uint64_t kCols = 1024;
  constexpr uint64_t kDstCols = 512;
  constexpr uint64_t kSourceBytes = kRows * kCols;
  constexpr uint64_t kTargetBytes = kRows * kDstCols;
  auto temp_root = make_temp_dir("collective-disk-loader-wide-dim1");
  const auto safetensors_path = temp_root / "weights.safetensors";
  std::vector<unsigned char> payload(kSourceBytes, 0xAB);
  create_safetensors_file(
      safetensors_path, "{\"src\":{\"dtype\":\"U8\",\"shape\":[2,1024],\"data_offsets\":[0,2048]}}", payload);

  auto disk_context_or = loader::get_disk_artifact_context(temp_root);
  REQUIRE(disk_context_or.ok());

  materialization::contracts::RepresentationWorkItem item;
  item.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
  item.partition_kind = materialization::contracts::WorkPartitionKind::kDim1Partitioned;
  item.dst_name = "dst";
  item.dst_spec = make_u8_tensor_spec("dst", 0, {2, 512}, {512, 1});
  item.committed_bytes = kTargetBytes;
  item.sources.push_back(
      materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              materialization::contracts::SourceFragment{
                  .source_spec = make_u8_tensor_spec("src", 0, {2, 1024}, {1024, 1}),
                  .source_range =
                      materialization::contracts::TensorCoordinateSpec{
                          .axes = {materialization::contracts::TensorAxisRange{.dim = 1, .start = 0, .end = 512}},
                      },
                  .destination_range =
                      materialization::contracts::TensorCoordinateSpec{
                          .axes = {materialization::contracts::TensorAxisRange{.dim = 1, .start = 0, .end = 512}},
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

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(512, 512);
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
      CollectiveMappedTargetLoadOptions{.chunk_bytes = 512, .strategy_config = strategy});

  REQUIRE(result.handled);
  REQUIRE_FALSE(result.status.ok());
  CHECK(absl::IsFailedPrecondition(result.status));
  CHECK(result.status.message().find("row exceeds pinned buffer size") != std::string_view::npos);

  loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "try_local_mapped_target_load copies replicated tensor bytes without generic residual",
    "[collective_disk_loader][local_mapped][replicated][correctness]") {
  SKIP_IF_NO_CUDA();

  constexpr uint64_t kTotalBytes = 16;
  auto temp_root = make_temp_dir("collective-disk-loader-replicated");
  const auto safetensors_path = temp_root / "weights.safetensors";
  std::vector<unsigned char> payload(kTotalBytes);
  for (uint64_t index = 0; index < kTotalBytes; ++index) {
    payload[static_cast<size_t>(index)] = static_cast<unsigned char>(0x40 + index);
  }
  create_safetensors_file(
      safetensors_path, "{\"src\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":[0,16]}}", payload);

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
  item.partition_kind = materialization::contracts::WorkPartitionKind::kReplicated;
  item.dst_name = "dst";
  item.dst_spec = make_u8_tensor_spec("dst", 0, {16}, {1});
  item.committed_bytes = kTotalBytes;
  item.sources.push_back(
      materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              materialization::contracts::SourceFragment{
                  .source_spec = make_u8_tensor_spec("src", 0, {16}, {1}),
                  .source_range = materialization::contracts::TensorCoordinateSpec{},
                  .destination_range = materialization::contracts::TensorCoordinateSpec{},
              },
      });

  loader::ByteRangeMap data_lane_map;
  data_lane_map.total_bytes = kTotalBytes;
  data_lane_map.num_sources = 1;
  data_lane_map.segments = {
      loader::ByteRangeSegment{
          .kind = loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = kTotalBytes,
          .src_offset = 0,
          .source_index = 0,
      },
  };

  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_tensor_aware_mapped_executor = true;
  strategy.allow_mixed_execution = true;

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(16, 4);
  const auto result = try_local_mapped_target_load(
      LocalMappedTargetLoadRequest{
          .artifact_id = "artifact_replicated",
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
      CollectiveMappedTargetLoadOptions{.chunk_bytes = 4, .strategy_config = strategy});

  REQUIRE(result.handled);
  REQUIRE(result.status.ok());
  CHECK(result.handled_bytes == kTotalBytes);
  CHECK(result.residual_data_map.segments.empty());
  CHECK(result.residual_data_map.total_bytes == kTotalBytes);

  std::array<uint8_t, kTotalBytes> actual{};
  REQUIRE(tensorcast::cuda::memcpy(actual.data(), gpu_buffer, actual.size(), cudaMemcpyDeviceToHost).ok());
  std::array<uint8_t, kTotalBytes> expected{};
  std::copy(payload.begin(), payload.end(), expected.begin());
  CHECK(actual == expected);

  REQUIRE(tensorcast::cuda::free(gpu_buffer).ok());
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
  strategy.enable_packed_rect2d_row_reads = true;

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
  CHECK(result.metrics.unique_source_bytes == 6);
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
    "try_local_mapped_target_load admits packed rect2d reads by actual host row width",
    "[collective_disk_loader][local_mapped][rect2d][admission]") {
  SKIP_IF_NO_CUDA();
  const char* cuda_backend = std::getenv("TENSORCAST_CUDA_BACKEND");
  const bool fake_cuda_backend = cuda_backend != nullptr && std::string_view(cuda_backend) == "fake";

  constexpr uint64_t kRows = 2;
  constexpr uint64_t kSrcCols = 1024;
  constexpr uint64_t kDstCols = 512;
  constexpr uint64_t kSourceBytes = kRows * kSrcCols;
  constexpr uint64_t kTargetBytes = kRows * kDstCols;
  auto temp_root = make_temp_dir("collective-disk-loader-packed-rect2d-admission");
  const auto safetensors_path = temp_root / "weights.safetensors";
  std::vector<unsigned char> payload = make_sequential_payload(kSourceBytes);
  const std::string header =
      "{\"src\":{\"dtype\":\"U8\",\"shape\":[2,1024],\"data_offsets\":[0," + std::to_string(kSourceBytes) + "]}}";
  create_safetensors_file(safetensors_path, header, payload);

  auto disk_context_or = loader::get_disk_artifact_context(temp_root);
  REQUIRE(disk_context_or.ok());

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, kTargetBytes).ok());
  std::array<uint8_t, kTargetBytes> initial{};
  initial.fill(0xEE);
  REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  materialization::contracts::RepresentationWorkItem item;
  item.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
  item.partition_kind = materialization::contracts::WorkPartitionKind::kUnknown;
  item.dst_name = "dst";
  item.dst_spec = make_u8_tensor_spec("dst", 0, {2, 512}, {512, 1});
  item.committed_bytes = kTargetBytes;
  item.sources.push_back(
      materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              materialization::contracts::SourceFragment{
                  .source_spec = make_u8_tensor_spec("src", 0, {2, 1024}, {1024, 1}),
                  .source_range = rect_range(0, 2, 128, 640),
                  .destination_range = rect_range(0, 2, 0, 512),
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
  strategy.allow_mixed_execution = true;
  strategy.enable_mapped_concat_jobs = true;
  strategy.enable_mapped_concat_execution = true;
  strategy.executor_preference =
      StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kTensorAwareLocal;

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(512, 512);
  const auto unpacked_result = try_local_mapped_target_load(
      LocalMappedTargetLoadRequest{
          .artifact_id = "artifact_packed_rect2d_admission",
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
      CollectiveMappedTargetLoadOptions{.chunk_bytes = 512, .strategy_config = strategy});

  REQUIRE(unpacked_result.handled);
  REQUIRE_FALSE(unpacked_result.status.ok());
  CHECK(absl::IsFailedPrecondition(unpacked_result.status));
  CHECK(unpacked_result.status.message().find("row exceeds pinned buffer size") != std::string_view::npos);

  if (fake_cuda_backend) {
    REQUIRE(tensorcast::cuda::free(gpu_buffer).ok());
    loader::reset_disk_artifact_context_cache_for_testing();
    std::error_code cleanup_ec;
    std::filesystem::remove_all(temp_root, cleanup_ec);
    return;
  }

  strategy.enable_packed_rect2d_row_reads = true;
  const auto packed_result = try_local_mapped_target_load(
      LocalMappedTargetLoadRequest{
          .artifact_id = "artifact_packed_rect2d_admission",
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
      CollectiveMappedTargetLoadOptions{.chunk_bytes = 512, .strategy_config = strategy});

  REQUIRE(packed_result.handled);
  REQUIRE(packed_result.status.ok());
  CHECK(packed_result.handled_bytes == kTargetBytes);
  CHECK(packed_result.metrics.unique_source_bytes == kTargetBytes);
  CHECK(packed_result.residual_data_map.segments.empty());

  std::array<uint8_t, kTargetBytes> actual{};
  REQUIRE(tensorcast::cuda::memcpy(actual.data(), gpu_buffer, actual.size(), cudaMemcpyDeviceToHost).ok());
  std::array<uint8_t, kTargetBytes> expected{};
  std::copy_n(payload.begin() + 128, 512, expected.begin());
  std::copy_n(payload.begin() + 1152, 512, expected.begin() + 512);
  CHECK(actual == expected);

  REQUIRE(tensorcast::cuda::free(gpu_buffer).ok());
  loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "try_local_mapped_target_load copies unknown contiguous source into packed expert slot",
    "[collective_disk_loader][local_mapped][expert][correctness]") {
  SKIP_IF_NO_CUDA();

  constexpr uint64_t kSourceBytes = 8;
  constexpr uint64_t kTargetBytes = 24;
  auto temp_root = make_temp_dir("collective-disk-loader-packed-expert-slot");
  const auto safetensors_path = temp_root / "weights.safetensors";
  std::vector<unsigned char> payload(kSourceBytes);
  for (uint64_t index = 0; index < kSourceBytes; ++index) {
    payload[static_cast<size_t>(index)] = static_cast<unsigned char>(index);
  }
  create_safetensors_file(
      safetensors_path, "{\"src\":{\"dtype\":\"U8\",\"shape\":[4,2],\"data_offsets\":[0,8]}}", payload);

  auto disk_context_or = loader::get_disk_artifact_context(temp_root);
  REQUIRE(disk_context_or.ok());
  REQUIRE((*disk_context_or)->is_safetensors());

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, kTargetBytes).ok());
  std::array<uint8_t, kTargetBytes> initial{};
  initial.fill(0xEE);
  REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  materialization::contracts::RepresentationWorkItem item;
  item.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
  item.partition_kind = materialization::contracts::WorkPartitionKind::kUnknown;
  item.dst_name = "packed";
  item.dst_spec = make_u8_tensor_spec("packed", 0, {3, 4, 2}, {8, 2, 1});
  item.committed_bytes = kSourceBytes;
  item.sources.push_back(
      materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              materialization::contracts::SourceFragment{
                  .source_spec = make_u8_tensor_spec("src", 0, {4, 2}, {2, 1}),
                  .source_range =
                      materialization::contracts::TensorCoordinateSpec{
                          .axes = {materialization::contracts::TensorAxisRange{.dim = 1, .start = 0, .end = 2}},
                      },
                  .destination_range =
                      materialization::contracts::TensorCoordinateSpec{
                          .axes = {materialization::contracts::TensorAxisRange{.dim = 0, .start = 1, .end = 2}},
                      },
              },
      });

  loader::ByteRangeMap data_lane_map;
  data_lane_map.total_bytes = kTargetBytes;
  data_lane_map.num_sources = 1;
  data_lane_map.segments = {
      loader::ByteRangeSegment{
          .kind = loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 8,
          .length = kSourceBytes,
          .src_offset = 0,
          .source_index = 0,
      },
  };

  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_tensor_aware_mapped_executor = true;
  strategy.allow_mixed_execution = true;

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(64, 64);
  const auto result = try_local_mapped_target_load(
      LocalMappedTargetLoadRequest{
          .artifact_id = "artifact_packed_expert_slot",
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
      CollectiveMappedTargetLoadOptions{.chunk_bytes = 64, .strategy_config = strategy});

  REQUIRE(result.handled);
  REQUIRE(result.status.ok());
  CHECK(result.handled_bytes == kSourceBytes);
  CHECK(result.residual_data_map.segments.empty());

  std::array<uint8_t, kTargetBytes> actual{};
  REQUIRE(tensorcast::cuda::memcpy(actual.data(), gpu_buffer, actual.size(), cudaMemcpyDeviceToHost).ok());
  std::array<uint8_t, kTargetBytes> expected = initial;
  for (uint64_t index = 0; index < kSourceBytes; ++index) {
    expected[static_cast<size_t>(8 + index)] = payload[static_cast<size_t>(index)];
  }
  CHECK(actual == expected);

  REQUIRE(tensorcast::cuda::free(gpu_buffer).ok());
  loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "try_local_mapped_target_load copies unknown dim1 source slice into packed expert slot",
    "[collective_disk_loader][local_mapped][expert][correctness]") {
  SKIP_IF_NO_CUDA();
  const char* cuda_backend = std::getenv("TENSORCAST_CUDA_BACKEND");
  if (cuda_backend != nullptr && std::string_view(cuda_backend) == "fake") {
    SKIP("dim1 compact byte-equivalence requires real CUDA because the executor uses cudaMemcpy2DAsync");
  }

  constexpr uint64_t kSourceRows = 4;
  constexpr uint64_t kSourceCols = 6;
  constexpr uint64_t kSelectedCols = 2;
  constexpr uint64_t kSourceBytes = kSourceRows * kSourceCols;
  constexpr uint64_t kSelectedBytes = kSourceRows * kSelectedCols;
  constexpr uint64_t kTargetBytes = 3 * kSelectedBytes;
  constexpr uint64_t kDestinationOffset = kSelectedBytes;
  auto temp_root = make_temp_dir("collective-disk-loader-packed-expert-dim1-slice");
  const auto safetensors_path = temp_root / "weights.safetensors";
  std::vector<unsigned char> payload(kSourceBytes);
  for (uint64_t index = 0; index < kSourceBytes; ++index) {
    payload[static_cast<size_t>(index)] = static_cast<unsigned char>(index);
  }
  create_safetensors_file(
      safetensors_path, "{\"src\":{\"dtype\":\"U8\",\"shape\":[4,6],\"data_offsets\":[0,24]}}", payload);

  auto disk_context_or = loader::get_disk_artifact_context(temp_root);
  REQUIRE(disk_context_or.ok());
  REQUIRE((*disk_context_or)->is_safetensors());

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, kTargetBytes).ok());
  std::array<uint8_t, kTargetBytes> initial{};
  initial.fill(0xEE);
  REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  materialization::contracts::RepresentationWorkItem item;
  item.kind = materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
  item.partition_kind = materialization::contracts::WorkPartitionKind::kUnknown;
  item.dst_name = "packed";
  item.dst_spec = make_u8_tensor_spec("packed", 0, {3, 4, 2}, {8, 2, 1});
  item.committed_bytes = kSelectedBytes;
  item.sources.push_back(
      materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              materialization::contracts::SourceFragment{
                  .source_spec = make_u8_tensor_spec("src", 0, {4, 6}, {6, 1}),
                  .source_range =
                      materialization::contracts::TensorCoordinateSpec{
                          .axes = {materialization::contracts::TensorAxisRange{.dim = 1, .start = 2, .end = 4}},
                      },
                  .destination_range =
                      materialization::contracts::TensorCoordinateSpec{
                          .axes = {materialization::contracts::TensorAxisRange{.dim = 0, .start = 1, .end = 2}},
                      },
              },
      });

  loader::ByteRangeMap data_lane_map;
  data_lane_map.total_bytes = kTargetBytes;
  data_lane_map.num_sources = 1;
  data_lane_map.segments = {
      loader::ByteRangeSegment{
          .kind = loader::ByteRangeSegment::Kind::kData,
          .dst_offset = kDestinationOffset,
          .length = kSelectedBytes,
          .src_offset = 0,
          .source_index = 0,
      },
  };

  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_tensor_aware_mapped_executor = true;
  strategy.allow_mixed_execution = true;

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(64, 64);
  const auto result = try_local_mapped_target_load(
      LocalMappedTargetLoadRequest{
          .artifact_id = "artifact_packed_expert_dim1_slice",
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
      CollectiveMappedTargetLoadOptions{.chunk_bytes = 64, .strategy_config = strategy});

  REQUIRE(result.handled);
  REQUIRE(result.status.ok());
  CHECK(result.handled_bytes == kSelectedBytes);
  CHECK(result.residual_data_map.segments.empty());

  std::array<uint8_t, kTargetBytes> actual{};
  REQUIRE(tensorcast::cuda::memcpy(actual.data(), gpu_buffer, actual.size(), cudaMemcpyDeviceToHost).ok());
  std::array<uint8_t, kTargetBytes> expected = initial;
  for (uint64_t row = 0; row < kSourceRows; ++row) {
    for (uint64_t col = 0; col < kSelectedCols; ++col) {
      expected[static_cast<size_t>(kDestinationOffset + row * kSelectedCols + col)] =
          payload[static_cast<size_t>(row * kSourceCols + 2 + col)];
    }
  }
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
