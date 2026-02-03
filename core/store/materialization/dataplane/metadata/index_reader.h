// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"

namespace tensorcast::store::loader {

struct IndexInfo {
  std::string canonical_index_json;
  std::string index_multihash;
  uint64_t total_size_bytes{0};
  bool is_safetensors{false};
  std::optional<std::string> source_index_json;
  uint64_t source_total_size_bytes{0};
};

absl::StatusOr<IndexInfo> read_from_artifact_dir(const std::filesystem::path& artifact_path, int target_device_id);

absl::StatusOr<IndexInfo> canonicalize_from_raw_json(std::string raw_json, int target_device_id);

absl::StatusOr<IndexInfo> build_from_safetensors(
    const std::vector<std::filesystem::path>& safetensor_files,
    std::optional<std::string_view> existing_index_multihash = std::nullopt);

} // namespace tensorcast::store::loader
