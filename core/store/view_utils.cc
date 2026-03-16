// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/view_utils.h"

#include <algorithm>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::view {

std::string build_view_spec_json(const loader::ViewSpec& spec, absl::Span<const std::string> tensor_names) {
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
  if (!tensor_names.empty()) {
    nlohmann::json names = nlohmann::json::array();
    for (const auto& name : tensor_names) {
      names.push_back(name);
    }
    root["tensor_names"] = std::move(names);
  }
  return root.dump();
}

absl::StatusOr<ParsedViewSelection> parse_view_selection_json(std::string_view view_spec_json) {
  if (view_spec_json.empty()) {
    return absl::InvalidArgumentError("view_spec_json must not be empty");
  }
  nlohmann::json root;
  try {
    root = nlohmann::json::parse(view_spec_json, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse view_spec_json: ", e.what()));
  }
  if (!root.is_object()) {
    return absl::InvalidArgumentError("view_spec_json must be an object");
  }
  const auto tensors_it = root.find("tensors");
  if (tensors_it == root.end() || !tensors_it->is_object()) {
    return absl::InvalidArgumentError("view_spec_json.tensors must be an object");
  }

  ParsedViewSelection parsed;
  const auto tensor_names_it = root.find("tensor_names");
  if (tensor_names_it != root.end()) {
    if (!tensor_names_it->is_array()) {
      return absl::InvalidArgumentError("view_spec_json.tensor_names must be an array");
    }
    absl::flat_hash_set<std::string> seen_names;
    for (const auto& name_json : *tensor_names_it) {
      if (!name_json.is_string()) {
        return absl::InvalidArgumentError("view_spec_json.tensor_names entries must be strings");
      }
      const std::string name = name_json.get<std::string>();
      if (!seen_names.insert(name).second) {
        return absl::InvalidArgumentError("view_spec_json.tensor_names must not contain duplicates");
      }
      parsed.tensor_names.push_back(name);
    }
  }

  for (auto it = tensors_it->begin(); it != tensors_it->end(); ++it) {
    if (!it.value().is_object()) {
      return absl::InvalidArgumentError(absl::StrCat("view_spec_json tensor entry must be object for ", it.key()));
    }
    const auto ops_it = it.value().find("ops");
    if (ops_it == it.value().end() || !ops_it->is_array()) {
      return absl::InvalidArgumentError(absl::StrCat("view_spec_json tensor ops must be array for ", it.key()));
    }
    loader::TensorViewOps ops;
    ops.ops.reserve(ops_it->size());
    for (const auto& op_json : *ops_it) {
      if (!op_json.is_object()) {
        return absl::InvalidArgumentError(absl::StrCat("view_spec_json op must be object for ", it.key()));
      }
      const auto type_it = op_json.find("type");
      if (type_it == op_json.end() || !type_it->is_string()) {
        return absl::InvalidArgumentError(absl::StrCat("view_spec_json op type must be string for ", it.key()));
      }
      const std::string type = type_it->get<std::string>();
      if (type == "narrow") {
        if (!op_json.contains("dim") || !op_json.contains("start") || !op_json.contains("length")) {
          return absl::InvalidArgumentError(absl::StrCat("view_spec_json narrow op missing fields for ", it.key()));
        }
        loader::NarrowOp narrow{
            .dim = static_cast<int32_t>(op_json.at("dim").get<int64_t>()),
            .start = op_json.at("start").get<int64_t>(),
            .length = static_cast<uint64_t>(op_json.at("length").get<int64_t>()),
        };
        ops.ops.push_back(loader::ViewOp::Narrow(narrow));
      } else if (type == "transpose") {
        if (!op_json.contains("dim0") || !op_json.contains("dim1")) {
          return absl::InvalidArgumentError(absl::StrCat("view_spec_json transpose op missing fields for ", it.key()));
        }
        loader::TransposeOp transpose{
            .dim0 = static_cast<int32_t>(op_json.at("dim0").get<int64_t>()),
            .dim1 = static_cast<int32_t>(op_json.at("dim1").get<int64_t>()),
        };
        ops.ops.push_back(loader::ViewOp::Transpose(transpose));
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("view_spec_json op type unsupported for ", it.key(), ": ", type));
      }
    }
    parsed.spec.tensors.emplace(it.key(), std::move(ops));
  }
  return parsed;
}

absl::StatusOr<loader::ViewSpec> parse_view_spec_json(std::string_view view_spec_json) {
  auto parsed_or = parse_view_selection_json(view_spec_json);
  if (!parsed_or.ok()) {
    return parsed_or.status();
  }
  return std::move(parsed_or->spec);
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
