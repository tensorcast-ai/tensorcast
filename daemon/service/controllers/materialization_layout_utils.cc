// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_layout_utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon::materialization_layout {

absl::StatusOr<CanonicalIndexTable> parse_canonical_index(std::string_view index_json) {
  if (index_json.empty()) {
    return absl::InvalidArgumentError("canonical index JSON is empty");
  }
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(index_json, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON: ", e.what()));
  }
  CanonicalIndexTable table;
  for (auto it = j.begin(); it != j.end(); ++it) {
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() != 6) {
      return absl::InvalidArgumentError("Invalid canonical index entry");
    }
    CanonicalIndexEntry entry;
    entry.logical_offset = arr[0].get<uint64_t>();
    entry.logical_length = arr[1].get<uint64_t>();
    entry.shape.reserve(arr[2].size());
    for (const auto& dim : arr[2]) {
      entry.shape.push_back(dim.get<int64_t>());
    }
    entry.stride.reserve(arr[3].size());
    for (const auto& dim : arr[3]) {
      entry.stride.push_back(dim.get<int64_t>());
    }
    entry.dtype = arr[4].get<std::string>();
    entry.storage_offset = arr[5].get<uint64_t>();
    table.logical_total_size =
        std::max<uint64_t>(table.logical_total_size, entry.logical_offset + entry.logical_length);
    table.entries.emplace(it.key(), std::move(entry));
  }
  return table;
}

absl::StatusOr<uint64_t> dtype_element_size(std::string_view dtype) {
  static const absl::flat_hash_map<std::string_view, uint64_t> kSizeMap = {
      {"torch.float16", 2},
      {"torch.bfloat16", 2},
      {"torch.float32", 4},
      {"torch.float64", 8},
      {"torch.int8", 1},
      {"torch.uint8", 1},
      {"torch.int16", 2},
      {"torch.int32", 4},
      {"torch.int64", 8},
      {"torch.bool", 1},
      {"torch.float", 4},
      {"torch.double", 8},
  };
  auto it = kSizeMap.find(dtype);
  if (it == kSizeMap.end()) {
    return absl::InvalidArgumentError(absl::StrCat("unsupported dtype: ", dtype));
  }
  return it->second;
}

absl::StatusOr<uint64_t> product_dims(absl::Span<const int64_t> dims) {
  uint64_t acc = 1;
  for (int64_t dim : dims) {
    if (dim <= 0) {
      return absl::InvalidArgumentError("shape dims must be positive");
    }
    if (acc > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
      return absl::OutOfRangeError("shape size overflow");
    }
    acc *= static_cast<uint64_t>(dim);
  }
  return acc;
}

absl::StatusOr<std::vector<TargetOffsetEntry>> resolve_target_offsets(const v2::TargetLayout& layout) {
  std::vector<TargetOffsetEntry> offsets;
  offsets.reserve(layout.offsets_size() + layout.aliases_size());
  if (layout.tensor_spec_kind() == v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS) {
    for (const auto& entry : layout.offsets()) {
      TargetOffsetEntry resolved;
      resolved.name = entry.name();
      resolved.storage_id = entry.storage_id();
      resolved.storage_offset = entry.storage_offset();
      resolved.logical_length = entry.logical_length();
      offsets.push_back(std::move(resolved));
    }
    return offsets;
  }
  if (layout.tensor_spec_kind() == v2::TargetLayout::TENSOR_SPEC_KIND_ALIAS_UNSPECIFIED) {
    for (const auto& entry : layout.aliases()) {
      TargetOffsetEntry resolved;
      resolved.name = entry.name();
      resolved.storage_id = entry.storage_id();
      resolved.storage_offset = entry.storage_offset();
      resolved.logical_length = entry.logical_length();
      offsets.push_back(std::move(resolved));
    }
    return offsets;
  }
  return absl::InvalidArgumentError("Unsupported tensor_spec_kind for target layout");
}

} // namespace tensorcast::daemon::materialization_layout
