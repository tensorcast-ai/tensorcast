// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <filesystem>
#include <string_view>

#include "absl/status/status.h"

namespace tensorcast::daemon::atomic_file_utils {

// Persist payload via temp file + fsync + rename + parent dir fsync.
absl::Status write_file_atomically(const std::filesystem::path& final_path, std::string_view payload);

} // namespace tensorcast::daemon::atomic_file_utils
