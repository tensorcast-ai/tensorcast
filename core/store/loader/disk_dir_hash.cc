// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/disk_dir_hash.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "absl/status/status.h"
#include "core/store/loader/file_partition_source.h"
#include "core/store/loader/source_hash.h"

namespace stepcast::store::loader {

absl::StatusOr<std::string> compute_data_multihash_from_disk_dir(const std::string& model_dir) {
  namespace fs = std::filesystem;
  fs::path dir(model_dir);
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
  std::vector<fs::path> parts;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto name = entry.path().filename().string();
    if (name.rfind("tensor.data_", 0) == 0) {
      parts.push_back(entry.path());
    }
  }
  std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
  if (parts.empty()) {
    fs::path single = dir / "tensor.data";
    if (fs::exists(single)) {
      parts.push_back(single);
    }
  }
  if (parts.empty()) {
    return absl::NotFoundError("No tensor.data partitions found");
  }

  FilePartitionSource::Options opts;
  for (const auto& p : parts) {
    opts.partition_paths.push_back(p);
    opts.partition_sizes.push_back(static_cast<size_t>(fs::file_size(p)));
  }
  opts.total_size = total_size;
  opts.chunk_size = 128 * 1024 * 1024;
  opts.use_direct_io = (total_size > 5ULL * 1024 * 1024 * 1024);
  FilePartitionSource src(std::move(opts));
  return compute_data_multihash_from_seekable_source(src, total_size);
}

} // namespace stepcast::store::loader
