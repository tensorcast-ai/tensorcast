// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"

namespace tensorcast::daemon::materialization_disk_resolve {

void record_disk_path_denied();
void record_disk_resolution_outcome(std::string_view outcome);

struct ResolveArtifactFromDiskResult {
  std::filesystem::path normalized_disk_path;
  std::string artifact_id;
  std::string canonical_index_json;
  std::string index_multihash;
  std::string data_multihash;
  uint64_t generation{0};
  bool descriptor_present{false};
};

absl::StatusOr<ResolveArtifactFromDiskResult> resolve_artifact_from_disk(
    std::string_view disk_path,
    const std::filesystem::path& storage_path,
    bool verify_checksums);

} // namespace tensorcast::daemon::materialization_disk_resolve
