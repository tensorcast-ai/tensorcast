// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/util/path_utils.h"

#include <system_error>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::daemon {
namespace {

bool path_has_prefix(const std::filesystem::path& path, const std::filesystem::path& prefix) {
  auto path_it = path.begin();
  for (auto prefix_it = prefix.begin(); prefix_it != prefix.end(); ++prefix_it, ++path_it) {
    if (path_it == path.end() || *path_it != *prefix_it) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<std::filesystem::path> canonicalize_path(
    const std::filesystem::path& candidate,
    const std::filesystem::path& normalized_root,
    bool enforce_root) {
  std::error_code ec;
  auto normalized = std::filesystem::weakly_canonical(candidate, ec);
  if (!ec) {
    if (enforce_root && !path_has_prefix(normalized, normalized_root)) {
      return absl::InvalidArgumentError(
          absl::StrCat(
              "disk_path must resolve under storage_path: ",
              normalized.string(),
              " (root=",
              normalized_root.string(),
              ")"));
    }
    return normalized;
  }
  normalized = candidate.lexically_normal();
  if (normalized.empty()) {
    return absl::ErrnoToStatus(ec.value(), "Failed to canonicalize disk_path");
  }
  if (enforce_root && !path_has_prefix(normalized, normalized_root)) {
    return absl::InvalidArgumentError(
        absl::StrCat(
            "disk_path must resolve under storage_path: ",
            normalized.string(),
            " (root=",
            normalized_root.string(),
            ")"));
  }
  return normalized;
}

} // namespace

absl::StatusOr<std::filesystem::path> normalize_disk_path(
    std::string_view disk_path,
    const std::filesystem::path& storage_root) {
  std::error_code ec;
  std::filesystem::path candidate(disk_path);
  const bool has_storage_root = !storage_root.empty();
  const bool is_absolute = candidate.is_absolute();
  if (!has_storage_root && !is_absolute) {
    return absl::InvalidArgumentError("storage_path is empty; disk_path must be absolute");
  }
  if (has_storage_root && !is_absolute) {
    candidate = storage_root / candidate;
  }
  std::filesystem::path normalized_root;
  if (has_storage_root) {
    std::error_code root_ec;
    normalized_root = std::filesystem::weakly_canonical(storage_root, root_ec);
    if (root_ec) {
      normalized_root = storage_root.lexically_normal();
    }
  }
  const bool enforce_root = has_storage_root;
  return canonicalize_path(candidate, normalized_root, enforce_root);
}

absl::StatusOr<std::filesystem::path> normalize_disk_import_path(
    std::string_view disk_path,
    const std::filesystem::path& storage_root) {
  std::filesystem::path candidate(disk_path);
  const bool has_storage_root = !storage_root.empty();
  const bool is_absolute = candidate.is_absolute();
  if (!has_storage_root && !is_absolute) {
    return absl::InvalidArgumentError("storage_path is empty; disk_path must be absolute");
  }
  if (has_storage_root && !is_absolute) {
    candidate = storage_root / candidate;
  }
  std::filesystem::path normalized_root;
  if (has_storage_root) {
    std::error_code root_ec;
    normalized_root = std::filesystem::weakly_canonical(storage_root, root_ec);
    if (root_ec) {
      normalized_root = storage_root.lexically_normal();
    }
  }
  const bool enforce_root = has_storage_root && !is_absolute;
  return canonicalize_path(candidate, normalized_root, enforce_root);
}

} // namespace tensorcast::daemon
