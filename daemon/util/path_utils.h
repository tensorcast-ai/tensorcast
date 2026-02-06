// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <filesystem>
#include <string_view>

#include "absl/status/statusor.h"

namespace tensorcast::daemon {

absl::StatusOr<std::filesystem::path> normalize_disk_path(
    std::string_view disk_path,
    const std::filesystem::path& storage_root);

absl::StatusOr<std::filesystem::path> normalize_disk_import_path(
    std::string_view disk_path,
    const std::filesystem::path& storage_root);

} // namespace tensorcast::daemon
