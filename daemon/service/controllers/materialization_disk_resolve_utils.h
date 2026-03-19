// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tensorcast::daemon::materialization_disk_resolve {

void record_disk_path_denied();
void record_disk_import_outcome(std::string_view outcome);

enum class ImportArtifactPhase : std::uint8_t {
  kPrepare = 0,
  kScanSource = 1,
  kReadHeaders = 2,
  kBuildCanonicalIndex = 3,
  kHashData = 4,
  kWriteRegistry = 5,
  kDone = 6,
  kError = 7,
};

enum class ImportArtifactErrorCode : std::uint8_t {
  kUnspecified = 0,
  kSourceNotFound = 1,
  kSourcePermissionDenied = 2,
  kSourceFormatInvalid = 3,
  kSourceMutated = 4,
  kRegistryIoFailure = 5,
  kPolicyDeniedNonLocalPeer = 6,
};

struct SourceFileFingerprint {
  std::uint64_t inode{0};
  std::uint64_t size{0};
  std::int64_t mtime_ns{0};

  bool operator==(const SourceFileFingerprint&) const = default;
};

using SourceFingerprintMap = absl::flat_hash_map<std::string, SourceFileFingerprint>;

struct ImportProgressUpdate {
  ImportArtifactPhase phase{ImportArtifactPhase::kPrepare};
  std::uint64_t processed_bytes{0};
  std::uint64_t total_bytes{0};
  bool done{false};
  bool error{false};
  ImportArtifactErrorCode error_code{ImportArtifactErrorCode::kUnspecified};
  std::string message;
};

using ImportProgressCallback = std::function<void(const ImportProgressUpdate&)>;

struct ImportArtifactFromPathResult {
  std::filesystem::path normalized_path;
  std::string artifact_id;
  std::string canonical_index_json;
  std::optional<std::string> source_index_json;
  std::string index_multihash;
  std::string data_multihash;
  std::uint64_t generation{0};
  bool descriptor_present{false};
  SourceFingerprintMap file_fingerprints;
};

[[nodiscard]] ImportArtifactErrorCode classify_import_error(const absl::Status& status);

absl::StatusOr<ImportArtifactFromPathResult> import_artifact_from_path(
    std::string_view path,
    const std::filesystem::path& storage_path,
    bool verify_checksums,
    ImportProgressCallback progress_cb = {});

absl::Status validate_source_fingerprints(
    const std::filesystem::path& source_root,
    const SourceFingerprintMap& expected_fingerprints);

} // namespace tensorcast::daemon::materialization_disk_resolve
