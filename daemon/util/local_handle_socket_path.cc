// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/util/local_handle_socket_path.h"

#include <cerrno>
#include <cstdint>
#include <format>
#include <string>
#include <utility>

#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::daemon {
namespace {

uint64_t fnv1a_hash_64(std::string_view value) {
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char c : value) {
    hash ^= c;
    hash *= 1099511628211ULL;
  }
  return hash;
}

absl::StatusOr<std::string> maybe_select_socket_path(const std::filesystem::path& dir, std::string_view socket_name) {
  std::filesystem::path candidate = dir / socket_name;
  const std::string candidate_str = candidate.string();
  if (!local_handle_socket_path_fits(candidate_str)) {
    return absl::InvalidArgumentError(
        absl::StrCat(
            "local_handle_socket_path is too long for AF_UNIX (len=",
            candidate_str.size(),
            ", max=",
            local_handle_socket_path_limit_bytes(),
            "): ",
            candidate_str));
  }
  absl::Status st = ensure_local_handle_socket_parent_dir(dir);
  if (!st.ok()) {
    return st;
  }
  return candidate_str;
}

} // namespace

size_t local_handle_socket_path_limit_bytes() {
  sockaddr_un addr{};
  return sizeof(addr.sun_path) - 1;
}

bool local_handle_socket_path_fits(std::string_view socket_path) {
  return !socket_path.empty() && socket_path.size() <= local_handle_socket_path_limit_bytes();
}

std::filesystem::path default_short_local_handle_dir() {
  return std::filesystem::path("/tmp") / std::format("tensorcast_lh_{}", static_cast<unsigned long>(::geteuid()));
}

std::string short_local_handle_socket_name(std::string_view seed) {
  const uint64_t hash = fnv1a_hash_64(seed);
  return std::format("lh-{:016x}.sock", hash);
}

absl::Status ensure_local_handle_socket_parent_dir(const std::filesystem::path& dir) {
  std::error_code ec;
  const bool exists = std::filesystem::exists(dir, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("stat failed for ", dir.string()));
  }
  if (!exists) {
    std::filesystem::create_directories(dir, ec);
    if (ec) {
      return absl::ErrnoToStatus(ec.value(), absl::StrCat("create_directories failed for ", dir.string()));
    }
    if (::chmod(dir.c_str(), 0700) < 0) {
      return absl::ErrnoToStatus(errno, absl::StrCat("chmod(0700) failed for ", dir.string()));
    }
  }
  if (!std::filesystem::is_directory(dir, ec)) {
    if (ec) {
      return absl::ErrnoToStatus(ec.value(), absl::StrCat("stat failed for ", dir.string()));
    }
    return absl::InvalidArgumentError(absl::StrCat("local handle parent is not a directory: ", dir.string()));
  }
  struct stat st{};
  if (::stat(dir.c_str(), &st) < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("stat failed for ", dir.string()));
  }
  const uid_t uid = ::geteuid();
  if (st.st_uid != uid) {
    return absl::PermissionDeniedError(absl::StrCat("local handle parent dir not owned by daemon uid: ", dir.string()));
  }
  if ((st.st_mode & S_IWOTH) != 0 && (st.st_mode & S_ISVTX) == 0) {
    return absl::PermissionDeniedError(
        absl::StrCat("local handle parent dir is world-writable without sticky bit: ", dir.string()));
  }
  return absl::OkStatus();
}

absl::Status validate_local_handle_socket_path(std::string_view socket_path) {
  if (socket_path.empty()) {
    return absl::InvalidArgumentError("local_handle_socket_path is empty");
  }
  if (!local_handle_socket_path_fits(socket_path)) {
    return absl::InvalidArgumentError(
        absl::StrCat(
            "local_handle_socket_path is too long for AF_UNIX (len=",
            socket_path.size(),
            ", max=",
            local_handle_socket_path_limit_bytes(),
            "): ",
            socket_path));
  }
  const std::filesystem::path parent = std::filesystem::path(std::string(socket_path)).parent_path();
  if (parent.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("local_handle_socket_path must include a parent directory: ", socket_path));
  }
  return ensure_local_handle_socket_parent_dir(parent);
}

absl::StatusOr<std::string> select_local_handle_socket_path(
    const std::filesystem::path& preferred,
    std::string_view seed,
    std::vector<std::filesystem::path> fallback_dirs) {
  const std::string preferred_str = preferred.string();
  if (local_handle_socket_path_fits(preferred_str)) {
    absl::Status st = validate_local_handle_socket_path(preferred_str);
    if (!st.ok()) {
      return st;
    }
    return preferred_str;
  }

  const std::string socket_name = short_local_handle_socket_name(seed.empty() ? preferred_str : seed);
  fallback_dirs.push_back(default_short_local_handle_dir());

  absl::Status last_status = absl::InvalidArgumentError(
      absl::StrCat(
          "preferred local_handle_socket_path is too long for AF_UNIX (len=",
          preferred_str.size(),
          ", max=",
          local_handle_socket_path_limit_bytes(),
          "): ",
          preferred_str));
  for (const std::filesystem::path& dir : fallback_dirs) {
    if (dir.empty()) {
      continue;
    }
    absl::StatusOr<std::string> candidate = maybe_select_socket_path(dir, socket_name);
    if (candidate.ok()) {
      return *candidate;
    }
    last_status = candidate.status();
  }
  return absl::InvalidArgumentError(
      absl::StrCat(
          "local_handle_socket_path too long or unavailable after bounded fallback selection: ",
          last_status.message()));
}

} // namespace tensorcast::daemon
