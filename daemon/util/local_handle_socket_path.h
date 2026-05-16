// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tensorcast::daemon {

size_t local_handle_socket_path_limit_bytes();

bool local_handle_socket_path_fits(std::string_view socket_path);

std::filesystem::path default_short_local_handle_dir();

std::string short_local_handle_socket_name(std::string_view seed);

absl::Status ensure_local_handle_socket_parent_dir(const std::filesystem::path& dir);

absl::Status validate_local_handle_socket_path(std::string_view socket_path);

absl::StatusOr<std::string> select_local_handle_socket_path(
    const std::filesystem::path& preferred,
    std::string_view seed,
    std::vector<std::filesystem::path> fallback_dirs);

} // namespace tensorcast::daemon
