// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/util/local_handle_socket_path.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>

#include <unistd.h>

namespace {

std::filesystem::path test_tmpdir() {
  return std::filesystem::temp_directory_path() / "tensorcast_local_handle_socket_path_test";
}

std::filesystem::path unique_test_root(std::string_view name) {
  return test_tmpdir() / std::format("{}_{}", name, static_cast<unsigned long>(::getpid()));
}

} // namespace

TEST_CASE("local handle socket path selector keeps a valid short preferred path", "[daemon][util]") {
  const auto root = unique_test_root("short_preferred");
  std::filesystem::remove_all(root);
  const auto preferred = root / "local_handle.sock";

  auto selected = tensorcast::daemon::select_local_handle_socket_path(preferred, preferred.string(), {});
  REQUIRE(selected.ok());
  REQUIRE(*selected == preferred.string());
  REQUIRE(tensorcast::daemon::local_handle_socket_path_fits(*selected));
  REQUIRE(std::filesystem::is_directory(root));

  std::filesystem::remove_all(root);
}

TEST_CASE("local handle socket path selector shortens long preferred paths before returning", "[daemon][util]") {
  const auto root = unique_test_root("long_preferred");
  std::filesystem::remove_all(root);
  const auto long_dir = root / std::string(120, 'a');
  const auto preferred = long_dir / "local_handle.sock";
  const auto fallback_dir = root / "uds";

  auto selected = tensorcast::daemon::select_local_handle_socket_path(preferred, preferred.string(), {fallback_dir});
  REQUIRE(selected.ok());
  REQUIRE(tensorcast::daemon::local_handle_socket_path_fits(*selected));
  REQUIRE(std::filesystem::path(*selected).parent_path() == fallback_dir);
  REQUIRE(
      std::filesystem::path(*selected).filename().string() ==
      tensorcast::daemon::short_local_handle_socket_name(preferred.string()));
  REQUIRE_FALSE(std::filesystem::exists(long_dir));
  REQUIRE(std::filesystem::is_directory(fallback_dir));

  std::filesystem::remove_all(root);
}

TEST_CASE("local handle socket path selector has a bounded final fallback", "[daemon][util]") {
  const auto root = unique_test_root("bounded_fallback");
  std::filesystem::remove_all(root);
  const auto long_dir = root / std::string(120, 'b');
  const auto preferred = long_dir / "local_handle.sock";
  const auto long_fallback = root / std::string(120, 'c');

  auto selected = tensorcast::daemon::select_local_handle_socket_path(preferred, "bounded-fallback", {long_fallback});
  REQUIRE(selected.ok());
  REQUIRE(tensorcast::daemon::local_handle_socket_path_fits(*selected));
  REQUIRE(std::filesystem::path(*selected).parent_path() == tensorcast::daemon::default_short_local_handle_dir());
  REQUIRE(
      std::filesystem::path(*selected).filename().string() ==
      tensorcast::daemon::short_local_handle_socket_name("bounded-fallback"));
  REQUIRE_FALSE(std::filesystem::exists(long_dir));
  REQUIRE_FALSE(std::filesystem::exists(long_fallback));

  std::filesystem::remove_all(root);
}
