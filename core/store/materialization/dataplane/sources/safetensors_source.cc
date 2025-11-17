// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/dataplane/sources/safetensors_source.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <endian.h>
#include <cerrno>
#include <cstring>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "core/store/materialization/dataplane/metadata/safetensors_util.h"

namespace tensorcast::store::loader {

namespace {
// Helper to pread fully in a loop
absl::StatusOr<size_t> pread_fully(int fd, uint64_t off, void* dst, size_t bytes) {
  size_t total = 0;
  char* ptr = static_cast<char*>(dst);
  while (total < bytes) {
    ssize_t got = ::pread(fd, ptr + total, bytes - total, static_cast<off_t>(off + total));
    if (got < 0) {
      if (errno == EINTR)
        continue;
      return absl::ErrnoToStatus(errno, "pread failed");
    }
    if (got == 0)
      break; // EOF
    total += static_cast<size_t>(got);
  }
  return total;
}
} // namespace

SafetensorsSource::SafetensorsSource(std::filesystem::path file_path) : file_path_(std::move(file_path)) {}

SafetensorsSource::~SafetensorsSource() {
  absl::MutexLock lock(&init_mutex_);
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

absl::Status SafetensorsSource::OpenFile() {
  absl::MutexLock lock(&init_mutex_);
  if (initialized_)
    return absl::OkStatus();
  fd_ = ::open(file_path_.c_str(), O_RDONLY);
  if (fd_ < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("Failed to open ", file_path_.string()));
  }
  auto st = ParseHeaderLocked();
  if (!st.ok()) {
    ::close(fd_);
    fd_ = -1;
    return st;
  }
  initialized_ = true;
  return absl::OkStatus();
}

absl::Status SafetensorsSource::ParseHeaderLocked() {
  // Use the shared utility function to parse the header
  auto header_info = ParseSafetensorsHeader(fd_);
  if (!header_info.ok()) {
    return header_info.status();
  }

  // Validate the JSON header content
  std::string header;
  header.resize(static_cast<size_t>(header_info->header_length));
  auto got = pread_fully(fd_, sizeof(uint64_t), header.data(), header.size());
  if (!got.ok())
    return got.status();
  if (*got != header.size()) {
    return absl::InvalidArgumentError("Truncated safetensors header");
  }
  if (header.empty() || header.front() != '{') {
    return absl::InvalidArgumentError("Malformed safetensors header: must start with '{'");
  }

  // Store the parsed information
  data_start_ = header_info->data_start;
  data_size_ = header_info->data_size;
  return absl::OkStatus();
}

absl::StatusOr<size_t> SafetensorsSource::read(void* dst, size_t max_bytes) {
  auto st = OpenFile();
  if (!st.ok())
    return st;
  absl::MutexLock lock(&offset_mutex_);
  if (current_offset_ >= data_size_) {
    return static_cast<size_t>(0);
  }
  size_t to_read = std::min(max_bytes, static_cast<size_t>(data_size_ - current_offset_));
  auto got = pread_fully(fd_, data_start_ + current_offset_, dst, to_read);
  if (!got.ok())
    return got.status();
  current_offset_ += *got;
  return got;
}

absl::StatusOr<size_t> SafetensorsSource::read_at(uint64_t offset, void* dst, size_t bytes) {
  auto st = OpenFile();
  if (!st.ok())
    return st;
  if (offset >= data_size_) {
    return static_cast<size_t>(0);
  }
  size_t to_read = std::min(bytes, static_cast<size_t>(data_size_ - offset));
  return pread_fully(fd_, data_start_ + offset, dst, to_read);
}

} // namespace tensorcast::store::loader
