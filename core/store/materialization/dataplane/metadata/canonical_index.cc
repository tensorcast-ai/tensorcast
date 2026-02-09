// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/metadata/canonical_index.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::store::loader {

namespace {

// Lowercase helper for dtype normalization in ordering code
std::string to_lower(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

} // namespace

int torch_dtype_code(std::string_view dtype) {
  // Deterministic ordering bucket. Expand as needed; unknowns bucket to the end.
  // Use lowercase match to be tolerant of minor variant strings.
  const std::string d = to_lower(dtype);
  // Floating
  if (d == "torch.float64" || d == "torch.double") {
    return 100;
  }
  if (d == "torch.float32" || d == "torch.float") {
    return 101;
  }
  if (d == "torch.float16") {
    return 102;
  }
  if (d == "torch.bfloat16") {
    return 103;
  }
  // Complex
  if (d == "torch.complex128") {
    return 110;
  }
  if (d == "torch.complex64") {
    return 111;
  }
  // Int
  if (d == "torch.int64" || d == "torch.long") {
    return 200;
  }
  if (d == "torch.int32" || d == "torch.int") {
    return 201;
  }
  if (d == "torch.int16" || d == "torch.short") {
    return 202;
  }
  if (d == "torch.int8") {
    return 203;
  }
  // Unsigned / bool
  if (d == "torch.uint8") {
    return 300;
  }
  return 9999; // Unknowns ordered last
}

absl::StatusOr<std::string> rebuild_stable_canonical_index(const std::string& index_json, int /*default_device_id*/) {
  try {
    using nlohmann::json;
    json j = json::parse(index_json);
    if (!j.is_object()) {
      return absl::InvalidArgumentError("tensor_index must be a JSON object");
    }
    // Re-emit as stable: keys ascending, field order fixed
    json out = json::object();
    std::map<std::string, json> items;
    for (auto it = j.begin(); it != j.end(); ++it) {
      items.emplace(it.key(), it.value());
    }
    for (const auto& [name, meta] : items) {
      if (!meta.is_array() || meta.size() < 6) {
        return absl::InvalidArgumentError(
            "tensor_index entry must be array [offset,size,shape,stride,dtype,storage_offset]");
      }
      uint64_t offset = meta[0].get<uint64_t>();
      uint64_t size = meta[1].get<uint64_t>();
      json shape = meta[2];
      json stride = meta[3];
      std::string dtype = meta[4].get<std::string>();
      uint64_t storage_offset = meta[5].get<uint64_t>();
      json arr = json::array();
      arr.push_back(offset);
      arr.push_back(size);
      arr.push_back(shape);
      arr.push_back(stride);
      arr.push_back(dtype);
      arr.push_back(storage_offset);
      out[name] = std::move(arr);
    }
    return out.dump();
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to rebuild canonical index: ", e.what()));
  }
}

absl::StatusOr<std::string> build_canonical_index_json(
    const std::vector<std::string>& ordered_names,
    const std::unordered_map<std::string, uint64_t>& offsets,
    const std::unordered_map<std::string, uint64_t>& sizes,
    const std::unordered_map<std::string, CanonicalTensorMeta>& metas) {
  using nlohmann::json;
  json out = json::object();
  for (const auto& name : ordered_names) {
    auto it_off = offsets.find(name);
    auto it_sz = sizes.find(name);
    auto it_meta = metas.find(name);
    if (it_off == offsets.end() || it_sz == sizes.end() || it_meta == metas.end()) {
      continue; // skip incomplete entries
    }
    const uint64_t off = it_off->second;
    const uint64_t sz = it_sz->second;
    const auto& meta = it_meta->second;
    json j_shape = json::array();
    for (auto v : meta.shape) {
      j_shape.push_back(static_cast<uint64_t>(v));
    }
    json j_stride = json::array();
    for (auto v : meta.stride) {
      j_stride.push_back(static_cast<uint64_t>(v));
    }
    json arr = json::array();
    arr.push_back(off);
    arr.push_back(sz);
    arr.push_back(j_shape);
    arr.push_back(j_stride);
    arr.push_back(meta.dtype);
    arr.push_back(meta.storage_offset);
    out[name] = std::move(arr);
  }
  return out.dump();
}

absl::StatusOr<std::string> build_coalesced_canonical_index_from_source_index_json(
    std::string_view source_index_json,
    uint64_t align_bytes) {
  if (source_index_json.empty()) {
    return absl::InvalidArgumentError("source_index_json must not be empty");
  }
  if (align_bytes == 0) {
    return absl::InvalidArgumentError("align_bytes must be > 0");
  }

  nlohmann::json j;
  try {
    j = nlohmann::json::parse(source_index_json, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse source index JSON: ", e.what()));
  }
  if (!j.is_object()) {
    return absl::InvalidArgumentError("source index JSON must be an object");
  }

  std::vector<std::string> ordered_names;
  ordered_names.reserve(j.size());
  std::unordered_map<std::string, uint64_t> sizes;
  sizes.reserve(j.size());
  std::unordered_map<std::string, CanonicalTensorMeta> metas;
  metas.reserve(j.size());

  for (auto it = j.begin(); it != j.end(); ++it) {
    const std::string& name = it.key();
    const auto& meta = it.value();
    if (!meta.is_array() || meta.size() < 6) {
      return absl::InvalidArgumentError(
          "source index entry must be array [offset,size,shape,stride,dtype,storage_offset]");
    }
    uint64_t size = meta[1].get<uint64_t>();
    CanonicalTensorMeta out_meta;
    out_meta.shape.clear();
    for (const auto& dim : meta[2]) {
      out_meta.shape.push_back(dim.get<int64_t>());
    }
    out_meta.stride.clear();
    for (const auto& stride : meta[3]) {
      out_meta.stride.push_back(stride.get<int64_t>());
    }
    out_meta.dtype = meta[4].get<std::string>();
    out_meta.storage_offset = meta[5].get<uint64_t>();
    ordered_names.push_back(name);
    sizes.emplace(name, size);
    metas.emplace(name, std::move(out_meta));
  }

  std::sort(ordered_names.begin(), ordered_names.end());

  std::unordered_map<std::string, uint64_t> offsets;
  offsets.reserve(ordered_names.size());
  uint64_t cursor = 0;
  for (const auto& name : ordered_names) {
    if (cursor % align_bytes != 0) {
      cursor = ((cursor + align_bytes - 1) / align_bytes) * align_bytes;
    }
    offsets.emplace(name, cursor);
    const uint64_t size = sizes.at(name);
    cursor += size;
  }

  return build_canonical_index_json(ordered_names, offsets, sizes, metas);
}

} // namespace tensorcast::store::loader
