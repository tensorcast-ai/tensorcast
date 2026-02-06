// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/metadata/disk_dir_hash.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "absl/status/status.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/sources/file_partition_source.h"
#include "core/store/materialization/dataplane/sources/multi_safetensors_source.h"

namespace tensorcast::store::loader {
namespace {

absl::StatusOr<std::optional<uint64_t>> maybe_read_logical_total_size(const std::filesystem::path& index_path) {
  namespace fs = std::filesystem;
  if (!fs::exists(index_path)) {
    return std::nullopt;
  }

  std::ifstream in(index_path);
  if (!in.is_open()) {
    return absl::InternalError("Failed to open tensor_index.json");
  }
  std::string index_json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  in.close();

  uint64_t total_size = 0;
  try {
    nlohmann::json j = nlohmann::json::parse(index_json);
    for (auto it = j.begin(); it != j.end(); ++it) {
      const auto& arr = it.value();
      if (!arr.is_array() || arr.size() < 2) {
        continue;
      }
      const uint64_t off = arr[0].get<uint64_t>();
      const uint64_t sz = arr[1].get<uint64_t>();
      total_size = std::max<uint64_t>(total_size, off + sz);
    }
  } catch (const std::exception&) {
    return absl::InvalidArgumentError("Failed to parse tensor_index.json");
  }
  return total_size;
}

std::optional<uint64_t> parse_partition_index(std::string_view name) {
  constexpr std::string_view kPrefix = "tensor.data_";
  if (!name.starts_with(kPrefix)) {
    return std::nullopt;
  }
  const std::string_view suffix = name.substr(kPrefix.size());
  if (suffix.empty()) {
    return std::nullopt;
  }
  uint64_t value = 0;
  for (char c : suffix) {
    if (!std::isdigit(static_cast<unsigned char>(c))) {
      return std::nullopt;
    }
    value = value * 10 + static_cast<uint64_t>(c - '0');
  }
  return value;
}

} // namespace

absl::StatusOr<std::string> compute_data_multihash_from_disk_dir(const std::string& artifact_dir) {
  namespace fs = std::filesystem;
  fs::path dir(artifact_dir);
  const fs::path index_path = dir / "tensor_index.json";
  auto logical_total_size_or = maybe_read_logical_total_size(index_path);
  if (!logical_total_size_or.ok()) {
    return logical_total_size_or.status();
  }
  const std::optional<uint64_t> logical_total_size = *logical_total_size_or;

  // Collect partitions in deterministic order
  std::vector<std::pair<uint64_t, fs::path>> parts;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto name = entry.path().filename().string();
    if (auto idx = parse_partition_index(name)) {
      parts.emplace_back(*idx, entry.path());
    }
  }
  std::ranges::sort(parts, [](const auto& a, const auto& b) {
    if (a.first != b.first) {
      return a.first < b.first;
    }
    return a.second.filename() < b.second.filename();
  });
  if (parts.empty()) {
    fs::path single = dir / "tensor.data";
    if (fs::exists(single)) {
      parts.emplace_back(0u, single);
    }
  }
  if (!parts.empty()) {
    FilePartitionSource::Options opts;
    uint64_t fallback_total = 0;
    for (const auto& [_, path] : parts) {
      const auto part_size = static_cast<size_t>(fs::file_size(path));
      opts.partition_paths.push_back(path);
      opts.partition_sizes.push_back(part_size);
      fallback_total += static_cast<uint64_t>(part_size);
    }
    opts.total_size = logical_total_size.value_or(fallback_total);
    opts.io_batch_bytes = 128 * 1024 * 1024;
    FilePartitionSource src(std::move(opts));
    return compute_data_multihash_from_seekable_source(src, opts.total_size);
  }

  std::vector<fs::path> safetensors_files;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string name = entry.path().filename().string();
    if (name.ends_with(".safetensors")) {
      safetensors_files.push_back(entry.path());
    }
  }
  if (!safetensors_files.empty()) {
    MultiSafetensorsSource source(std::move(safetensors_files));
    const uint64_t source_total_size = source.total_size();
    if (logical_total_size.has_value() && source_total_size > 0 && *logical_total_size > source_total_size) {
      return absl::FailedPreconditionError("tensor_index.json total size exceeds safetensors payload bytes");
    }
    const uint64_t effective_total_size = logical_total_size.value_or(source_total_size);
    return compute_data_multihash_from_seekable_source(source, effective_total_size);
  }

  return absl::NotFoundError("No tensor.data partitions or .safetensors files found");
}

} // namespace tensorcast::store::loader
