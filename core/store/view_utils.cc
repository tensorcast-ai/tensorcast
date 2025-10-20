// Copyright (c) 2025, TensorCast Team.

#include "core/store/view_utils.h"

#include <algorithm>

#include "absl/container/flat_hash_set.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::view {

std::string build_view_spec_json(const loader::ViewSpec& spec) {
  nlohmann::json tensors = nlohmann::json::object();
  for (const auto& [tensor_name, ops] : spec.tensors) {
    nlohmann::json tensor_json;
    nlohmann::json ops_array = nlohmann::json::array();
    for (const auto& op : ops.ops) {
      nlohmann::json op_json;
      switch (op.kind) {
        case loader::ViewOp::Kind::kNarrow:
          op_json["type"] = "narrow";
          op_json["dim"] = op.narrow.dim;
          op_json["start"] = op.narrow.start;
          op_json["length"] = op.narrow.length;
          break;
        case loader::ViewOp::Kind::kTranspose:
          op_json["type"] = "transpose";
          op_json["dim0"] = op.transpose.dim0;
          op_json["dim1"] = op.transpose.dim1;
          break;
      }
      ops_array.push_back(std::move(op_json));
    }
    tensor_json["ops"] = std::move(ops_array);
    tensors[tensor_name] = std::move(tensor_json);
  }
  nlohmann::json root;
  root["tensors"] = std::move(tensors);
  return root.dump();
}

uint64_t align_up(uint64_t value, uint64_t align) {
  if (align == 0) {
    return value;
  }
  const uint64_t remainder = value % align;
  return remainder == 0 ? value : value + (align - remainder);
}

uint64_t align_down(uint64_t value, uint64_t align) {
  if (align == 0) {
    return value;
  }
  return value - (value % align);
}

std::vector<uint64_t> compute_fully_covered_canonical_leaf_indices(
    absl::Span<const CanonicalRange> ranges,
    uint64_t chunk_bytes) {
  if (chunk_bytes == 0) {
    return {};
  }
  absl::flat_hash_set<uint64_t> indices;
  for (const auto& range : ranges) {
    if (range.length == 0) {
      continue;
    }
    const uint64_t range_start = range.offset;
    const uint64_t range_end = range.offset + range.length;
    const uint64_t first_full = align_up(range_start, chunk_bytes);
    const uint64_t last_full = align_down(range_end, chunk_bytes);
    if (first_full >= last_full) {
      continue;
    }
    for (uint64_t pos = first_full; pos < last_full; pos += chunk_bytes) {
      indices.insert(pos / chunk_bytes);
    }
  }
  std::vector<uint64_t> sorted(indices.begin(), indices.end());
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}

} // namespace tensorcast::store::view
