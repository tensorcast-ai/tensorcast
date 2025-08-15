// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/multi_safetensors_source.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"

namespace stepcast::store::loader {

namespace {
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
      break;
    total += static_cast<size_t>(got);
  }
  return total;
}
} // namespace

MultiSafetensorsSource::MultiSafetensorsSource(std::vector<std::filesystem::path> file_paths)
    : file_paths_(std::move(file_paths)) {}

MultiSafetensorsSource::~MultiSafetensorsSource() {
  absl::MutexLock lock(&init_mutex_);
  for (auto& s : segments_) {
    if (s.fd >= 0) {
      ::close(s.fd);
      s.fd = -1;
    }
  }
}

absl::Status MultiSafetensorsSource::OpenFiles() {
  absl::MutexLock lock(&init_mutex_);
  if (initialized_)
    return absl::OkStatus();
  if (file_paths_.empty()) {
    return absl::InvalidArgumentError("No safetensors files provided");
  }
  // Sort by filename for deterministic order
  std::sort(
      file_paths_.begin(), file_paths_.end(), [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
  segments_.clear();
  segments_.reserve(file_paths_.size());
  for (const auto& p : file_paths_) {
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) {
      return absl::NotFoundError(absl::StrFormat("Failed to open %s: %s", p.string(), std::strerror(errno)));
    }
    segments_.push_back(Segment{.fd = fd});
  }
  auto st = ParseAllHeadersLocked();
  if (!st.ok()) {
    for (auto& s : segments_) {
      if (s.fd >= 0)
        ::close(s.fd);
      s.fd = -1;
    }
    return st;
  }
  initialized_ = true;
  return absl::OkStatus();
}

absl::Status MultiSafetensorsSource::ParseAllHeadersLocked() {
  total_size_ = 0;
  uint64_t running_base = 0;
  for (auto& s : segments_) {
    uint64_t header_len_le = 0;
    ssize_t n = ::pread(s.fd, &header_len_le, sizeof(header_len_le), 0);
    if (n != static_cast<ssize_t>(sizeof(header_len_le))) {
      return absl::InvalidArgumentError("Invalid safetensors file: cannot read header length");
    }
    uint64_t header_len = header_len_le; // little-endian host assumption
    if (header_len > (1ULL << 30)) {
      return absl::InvalidArgumentError("Safetensors header too large");
    }
    std::string header;
    header.resize(static_cast<size_t>(header_len));
    auto got = pread_fully(s.fd, sizeof(uint64_t), header.data(), header.size());
    if (!got.ok())
      return got.status();
    if (*got != header.size()) {
      return absl::InvalidArgumentError("Truncated safetensors header");
    }
    if (header.empty() || header.front() != '{') {
      return absl::InvalidArgumentError("Malformed safetensors header: must start with '{'");
    }

    struct stat stbuf{};
    if (::fstat(s.fd, &stbuf) != 0) {
      return absl::InternalError(absl::StrFormat("fstat failed: %s", std::strerror(errno)));
    }
    uint64_t file_size = static_cast<uint64_t>(stbuf.st_size);
    s.data_start = sizeof(uint64_t) + header_len;
    if (s.data_start > file_size) {
      return absl::InvalidArgumentError("Invalid safetensors file: data starts beyond EOF");
    }
    s.data_size = file_size - s.data_start;
    s.base_offset = running_base;
    running_base += s.data_size;
  }
  total_size_ = running_base;
  return absl::OkStatus();
}

absl::StatusOr<size_t> MultiSafetensorsSource::read(void* dst, size_t max_bytes) {
  auto st = OpenFiles();
  if (!st.ok())
    return st;
  absl::MutexLock lock(&offset_mutex_);
  if (current_offset_ >= total_size_) {
    return static_cast<size_t>(0);
  }
  size_t to_read = std::min(max_bytes, static_cast<size_t>(total_size_ - current_offset_));
  size_t total = 0;
  char* ptr = static_cast<char*>(dst);
  uint64_t off = current_offset_;
  while (total < to_read) {
    // find segment
    // find segment using binary search for better performance with many files
    auto it = std::upper_bound(segments_.begin(), segments_.end(), off,
        [](uint64_t offset, const Segment& s) { return offset < s.base_offset + s.data_size; });
    if (it == segments_.begin()) break;
    --it;
    const auto& s = *it;
    if (idx >= segments_.size())
      break;
    const auto& s = segments_[idx];
    uint64_t within = off - s.base_offset;
    size_t seg_remaining = static_cast<size_t>(s.data_size - within);
    size_t want = std::min(to_read - total, seg_remaining);
    auto got = pread_fully(s.fd, s.data_start + within, ptr + total, want);
    if (!got.ok())
      return got.status();
    if (*got == 0)
      break;
    total += *got;
    off += *got;
  }
  current_offset_ = off;
  return total;
}

absl::StatusOr<size_t> MultiSafetensorsSource::read_at(uint64_t offset, void* dst, size_t bytes) {
  auto st = OpenFiles();
  if (!st.ok())
    return st;
  if (offset >= total_size_) {
    return static_cast<size_t>(0);
  }
  size_t to_read = std::min(bytes, static_cast<size_t>(total_size_ - offset));
  size_t total = 0;
  char* ptr = static_cast<char*>(dst);
  uint64_t off = offset;
  while (total < to_read) {
    size_t idx = 0;
    for (; idx < segments_.size(); ++idx) {
      const auto& s = segments_[idx];
      if (off < s.base_offset + s.data_size)
        break;
    }
    if (idx >= segments_.size())
      break;
    const auto& s = segments_[idx];
    uint64_t within = off - s.base_offset;
    size_t seg_remaining = static_cast<size_t>(s.data_size - within);
    size_t want = std::min(to_read - total, seg_remaining);
    auto got = pread_fully(s.fd, s.data_start + within, ptr + total, want);
    if (!got.ok())
      return got.status();
    if (*got == 0)
      break;
    total += *got;
    off += *got;
  }
  return total;
}

} // namespace stepcast::store::loader
