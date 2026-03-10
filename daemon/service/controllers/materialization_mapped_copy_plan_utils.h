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

struct BuildCopyPlanResult {
  store::loader::ByteRangeMap map;
  uint64_t total_bytes_copied{0};
};

bool is_contiguous(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride);

absl::StatusOr<BuildCopyPlanResult> build_copy_plan(
    const v2::CopyPlan& copy_plan,
    const materialization_layout::CanonicalIndexTable& source_table,
    const absl::flat_hash_map<std::string, MappedTensorSpec>& dst_specs,
    const absl::flat_hash_map<std::string, uint64_t>& dst_base_offsets,
    const absl::flat_hash_map<std::string, ViewNarrowSpec>& view_narrows);

} // namespace tensorcast::daemon::materialization_mapped_copy_plan
