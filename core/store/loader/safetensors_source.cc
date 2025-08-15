// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/safetensors_source.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"

namespace stepcast::store::loader {

namespace {
// Helper to pread fully in a loop
static absl::StatusOr<size_t> pread_fully(int fd, uint64_t off, void* dst, size_t bytes) {
  size_t total = 0;
  char* ptr = static_cast<char*>(dst);
  while (total < bytes) {
    ssize_t got = ::pread(fd, ptr + total, bytes - total, static_cast<off_t>(off + total));
    if (got < 0) {
      if (errno == EINTR)
        continue;
      return absl::InternalError(absl::StrFormat("pread failed: %s", std::strerror(errno)));
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
    return absl::NotFoundError(absl::StrFormat("Failed to open %s: %s", file_path_.string(), std::strerror(errno)));
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
  // Layout: [8-byte u64 header_len][header_json][data...]
  uint64_t header_len_le = 0;
  ssize_t n = ::pread(fd_, &header_len_le, sizeof(header_len_le), 0);
  if (n != static_cast<ssize_t>(sizeof(header_len_le))) {
    return absl::InvalidArgumentError("Invalid safetensors file: cannot read header length");
  }
  // Assume little-endian host
  uint64_t header_len = header_len_le;
  if (header_len > (1ULL << 30)) {
    return absl::InvalidArgumentError("Safetensors header too large");
  }
  std::string header;
  header.resize(static_cast<size_t>(header_len));
  auto got = pread_fully(fd_, sizeof(uint64_t), header.data(), header.size());
  if (!got.ok())
    return got.status();
  if (*got != header.size()) {
    return absl::InvalidArgumentError("Truncated safetensors header");
  }
  if (header.empty() || header.front() != '{') {
    return absl::InvalidArgumentError("Malformed safetensors header: must start with '{'");
  }

  // Compute file size
  struct stat stbuf{};
  if (::fstat(fd_, &stbuf) != 0) {
    return absl::InternalError(absl::StrFormat("fstat failed: %s", std::strerror(errno)));
  }
  uint64_t file_size = static_cast<uint64_t>(stbuf.st_size);
  data_start_ = sizeof(uint64_t) + header_len;
  if (data_start_ > file_size) {
    return absl::InvalidArgumentError("Invalid safetensors file: data starts beyond EOF");
  }
  data_size_ = file_size - data_start_;
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

} // namespace stepcast::store::loader
