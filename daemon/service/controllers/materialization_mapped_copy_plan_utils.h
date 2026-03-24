// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "daemon/service/controllers/materialization_layout_utils.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::materialization_mapped_copy_plan {

struct MappedTensorSpec {
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::string dtype;
  uint64_t storage_offset{0};
  uint64_t logical_length{0};
  uint64_t element_size{0};
};

struct ViewNarrowSpec {
  int32_t dim{0};
  int64_t start{0};
  int64_t end{0};
};

enum class TensorJobDistribution : uint8_t {
  kUnknown = 0,
  kReplicated = 1,
  kDim0Partitioned = 2,
  kDim1Partitioned = 3,
};

struct TensorJobCandidate {
  std::string src_name;
  std::string dst_name;
  TensorJobDistribution distribution{TensorJobDistribution::kUnknown};
  std::vector<int64_t> src_shape;
  std::vector<int64_t> src_stride;
  std::vector<int64_t> dst_shape;
  std::vector<int64_t> dst_stride;
  std::string dtype;
  uint64_t src_logical_offset{0};
  uint64_t src_storage_offset{0};
  uint64_t src_size_bytes{0};
  uint64_t dst_base_offset{0};
  uint64_t dst_size_bytes{0};
  uint64_t element_size{0};
  int32_t dim{-1};
  int64_t src_start{0};
  int64_t src_end{0};
  int64_t dst_start{0};
  int64_t dst_end{0};
};

struct ConcatSourceFragmentCandidate {
  std::string src_name;
  std::vector<int64_t> src_shape;
  std::vector<int64_t> src_stride;
  std::string dtype;
  uint64_t src_logical_offset{0};
  uint64_t src_storage_offset{0};
  uint64_t src_size_bytes{0};
  uint64_t element_size{0};
  int32_t dim{0};
  int64_t src_start{0};
  int64_t src_end{0};
  uint64_t prefix_count{0};
  uint64_t dst_block_offset_bytes{0};
  uint64_t dst_block_stride_bytes{0};
  uint64_t dst_block_bytes{0};
};

struct ConcatJobCandidate {
  std::string dst_name;
  std::vector<int64_t> dst_shape;
  std::vector<int64_t> dst_stride;
  std::string dtype;
  uint64_t dst_base_offset{0};
  uint64_t dst_size_bytes{0};
  uint64_t element_size{0};
  uint64_t prefix_count{0};
  uint64_t dst_block_stride_bytes{0};
  std::vector<ConcatSourceFragmentCandidate> sources;
};

struct TensorJobCompatibilityStats {
  uint64_t total_dst_tensors{0};
  uint64_t compatible_candidates{0};
  uint64_t compatible_bytes{0};
  uint64_t concat_candidates{0};
  uint64_t concat_bytes{0};
  uint64_t rejected_mixed_src_or_dim{0};
  uint64_t rejected_mixed_src_or_dim_bytes{0};
  uint64_t rejected_non_contiguous{0};
  uint64_t rejected_non_contiguous_bytes{0};
  uint64_t rejected_unsupported_distribution{0};
  uint64_t rejected_unsupported_distribution_bytes{0};
};

struct BuildCopyPlanResult {
  store::loader::ByteRangeMap map;
  uint64_t total_bytes_copied{0};
  std::vector<TensorJobCandidate> tensor_job_candidates;
  std::vector<ConcatJobCandidate> concat_job_candidates;
  TensorJobCompatibilityStats compatibility_stats;
};

bool is_contiguous(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride);

absl::StatusOr<BuildCopyPlanResult> build_copy_plan(
    const v2::CopyPlan& copy_plan,
    const materialization_layout::CanonicalIndexTable& source_table,
    const materialization_layout::CanonicalIndexTable& canonical_source_table,
    const absl::flat_hash_map<std::string, MappedTensorSpec>& dst_specs,
    const absl::flat_hash_map<std::string, uint64_t>& dst_base_offsets,
    const absl::flat_hash_map<std::string, ViewNarrowSpec>& view_narrows);

} // namespace tensorcast::daemon::materialization_mapped_copy_plan
