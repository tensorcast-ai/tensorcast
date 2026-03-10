// Copyright (c) 2026, TensorCast Team.

#include "daemon/util/atomic_file_utils.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "absl/strings/str_cat.h"

namespace tensorcast::daemon::atomic_file_utils {

namespace {

std::atomic<uint64_t> g_atomic_write_nonce{0};

std::filesystem::path build_temp_path(const std::filesystem::path& final_path) {
  const uint64_t nonce = g_atomic_write_nonce.fetch_add(1, std::memory_order_relaxed);
  const pid_t pid = ::getpid();
  std::filesystem::path tmp = final_path;
  tmp += absl::StrCat(".tmp.", static_cast<int>(pid), ".", static_cast<unsigned long long>(nonce)); // NOLINT
  return tmp;
}

} // namespace

absl::Status write_file_atomically(const std::filesystem::path& final_path, std::string_view payload) {
  const std::filesystem::path tmp_path = build_temp_path(final_path);
  const std::string tmp_path_str = tmp_path.string();
  const std::string final_path_str = final_path.string();

  int fd = ::open(tmp_path_str.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("Failed to open temporary file for atomic write: ", tmp_path_str));
  }

  const char* data = payload.data();
  size_t remaining = payload.size();
  while (remaining > 0) {
    const ssize_t written = ::write(fd, data, remaining);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      const int err = errno;
      (void)::close(fd);
      (void)::unlink(tmp_path_str.c_str());
      return absl::ErrnoToStatus(err, absl::StrCat("Failed to write temporary file: ", tmp_path_str));
    }
    data += written;
    remaining -= static_cast<size_t>(written);
  }

  if (::fsync(fd) != 0) {
    const int err = errno;
    (void)::close(fd);
    (void)::unlink(tmp_path_str.c_str());
    return absl::ErrnoToStatus(err, absl::StrCat("Failed to fsync temporary file: ", tmp_path_str));
  }

  if (::close(fd) != 0) {
    const int err = errno;
    (void)::unlink(tmp_path_str.c_str());
    return absl::ErrnoToStatus(err, absl::StrCat("Failed to close temporary file: ", tmp_path_str));
  }

  if (::rename(tmp_path_str.c_str(), final_path_str.c_str()) != 0) {
    const int err = errno;
    (void)::unlink(tmp_path_str.c_str());
    return absl::ErrnoToStatus(err, absl::StrCat("Failed to atomically rename file to ", final_path_str));
  }

  const std::filesystem::path parent_dir = final_path.parent_path();
  if (!parent_dir.empty()) {
    const std::string parent_dir_str = parent_dir.string();
    int dir_fd = ::open(parent_dir_str.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir_fd < 0) {
      return absl::ErrnoToStatus(errno, absl::StrCat("Failed to open parent directory for fsync: ", parent_dir_str));
    }
    if (::fsync(dir_fd) != 0) {
      const int err = errno;
      (void)::close(dir_fd);
      return absl::ErrnoToStatus(err, absl::StrCat("Failed to fsync parent directory: ", parent_dir_str));
    }
    (void)::close(dir_fd);
  }

  return absl::OkStatus();
}

} // namespace tensorcast::daemon::atomic_file_utils
