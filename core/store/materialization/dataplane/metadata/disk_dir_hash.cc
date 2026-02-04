// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/metadata/disk_dir_hash.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "absl/status/status.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/sources/file_partition_source.h"

namespace tensorcast::store::loader {

absl::StatusOr<std::string> compute_data_multihash_from_disk_dir(const std::string& artifact_dir) {
  namespace fs = std::filesystem;
  fs::path dir(artifact_dir);
  const fs::path index_path = dir / "tensor_index.json";
  if (!fs::exists(index_path)) {
    return absl::NotFoundError("tensor_index.json not found");
  }

  // Determine logical total size by parsing canonical JSON
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
      uint64_t off = arr[0].get<uint64_t>();
      uint64_t sz = arr[1].get<uint64_t>();
      total_size = std::max<uint64_t>(total_size, off + sz);
    }
  } catch (const std::exception&) {
    return absl::InvalidArgumentError("Failed to parse tensor_index.json");
  }

  // Collect partitions in deterministic order
  std::vector<std::pair<uint64_t, fs::path>> parts;
  auto parse_partition_index = [](std::string_view name) -> std::optional<uint64_t> {
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
  };
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
  if (parts.empty()) {
    return absl::NotFoundError("No tensor.data partitions found");
  }

  FilePartitionSource::Options opts;
  for (const auto& [_, path] : parts) {
    opts.partition_paths.push_back(path);
    opts.partition_sizes.push_back(static_cast<size_t>(fs::file_size(path)));
  }
  opts.total_size = total_size;
  opts.io_batch_bytes = 128 * 1024 * 1024;
  FilePartitionSource src(std::move(opts));
  return compute_data_multihash_from_seekable_source(src, total_size);
}

} // namespace tensorcast::store::loader
