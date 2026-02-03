// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/util/path_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "absl/status/status.h"

namespace {

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_path_utils_test";
}

std::filesystem::path create_file(std::filesystem::path path) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  REQUIRE(out.is_open());
  return path;
}

} // namespace

TEST_CASE("normalize_disk_path enforces storage root for absolute and relative paths", "[daemon][util]") {
  const auto root = test_tmpdir() / "root";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "subdir");
  const auto file_path = create_file(root / "subdir" / "file");

  auto absolute = tensorcast::daemon::normalize_disk_path(file_path.string(), root);
  REQUIRE(absolute.ok());
  REQUIRE(*absolute == std::filesystem::weakly_canonical(file_path));

  auto relative = tensorcast::daemon::normalize_disk_path("subdir/file", root);
  REQUIRE(relative.ok());
  REQUIRE(*relative == std::filesystem::weakly_canonical(file_path));

  const auto outside = test_tmpdir() / "outside";
  std::filesystem::create_directories(outside);
  auto outside_status = tensorcast::daemon::normalize_disk_path(outside.string(), root);
  REQUIRE_FALSE(outside_status.ok());
  REQUIRE(outside_status.status().code() == absl::StatusCode::kInvalidArgument);

  auto escape = tensorcast::daemon::normalize_disk_path("../outside", root);
  REQUIRE_FALSE(escape.ok());
  REQUIRE(escape.status().code() == absl::StatusCode::kInvalidArgument);
}

TEST_CASE("normalize_disk_path allows absolute when storage root is empty", "[daemon][util]") {
  const auto root = test_tmpdir() / "root_empty";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const auto file_path = create_file(root / "tensor.data");

  auto absolute = tensorcast::daemon::normalize_disk_path(file_path.string(), std::filesystem::path{});
  REQUIRE(absolute.ok());
  REQUIRE(*absolute == std::filesystem::weakly_canonical(file_path));

  auto relative = tensorcast::daemon::normalize_disk_path("relative/path", std::filesystem::path{});
  REQUIRE_FALSE(relative.ok());
  REQUIRE(relative.status().code() == absl::StatusCode::kInvalidArgument);
}
