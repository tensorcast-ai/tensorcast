// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/multi_safetensors_source.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <endian.h>
#include <algorithm>
#include <cerrno>
#include <cstring>

#include "absl/strings/str_format.h"
#include "core/store/loader/safetensors_util.h"

namespace stepcast::store::loader {

namespace {
static absl::StatusOr<size_t> pread_fully(int fd, uint64_t off, void* dst, size_t bytes) {
  size_t total = 0;
  char* ptr = static_cast<char*>(dst);
  while (total < bytes) {
    ssize_t got = ::pread(fd, ptr + total, bytes - total, static_cast<off_t>(off + total));
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      return absl::InternalError(absl::StrFormat("pread failed: %s", std::strerror(errno)));
    }
    if (got == 0) {
      break;
    }
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
  if (initialized_) {
    return absl::OkStatus();
  }
  if (file_paths_.empty()) {
    return absl::InvalidArgumentError("No safetensors files provided");
  }
  // Sort by filename for deterministic order
  std::ranges::sort(file_paths_, [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
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
      if (s.fd >= 0) {
        ::close(s.fd);
      }
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
    // Use the shared utility function to parse the header
    auto header_info = ParseSafetensorsHeader(s.fd);
    if (!header_info.ok()) {
      return header_info.status();
    }
    
    // Validate the JSON header content
    std::string header;
    header.resize(static_cast<size_t>(header_info->header_length));
    auto got = pread_fully(s.fd, sizeof(uint64_t), header.data(), header.size());
    if (!got.ok()) {
      return got.status();
    }
    if (*got != header.size()) {
      return absl::InvalidArgumentError("Truncated safetensors header");
    }
    if (header.empty() || header.front() != '{') {
      return absl::InvalidArgumentError("Malformed safetensors header: must start with '{'");
    }

    // Store the parsed information
    s.data_start = header_info->data_start;
    s.data_size = header_info->data_size;
    s.base_offset = running_base;
    running_base += s.data_size;
  }
  total_size_ = running_base;
  return absl::OkStatus();
}

absl::StatusOr<size_t> MultiSafetensorsSource::read(void* dst, size_t max_bytes) {
  auto st = OpenFiles();
  if (!st.ok()) {
    return st;
  }
  absl::MutexLock lock(&offset_mutex_);
  if (current_offset_ >= total_size_) {
    return static_cast<size_t>(0);
  }
  size_t to_read = std::min(max_bytes, static_cast<size_t>(total_size_ - current_offset_));
  size_t total = 0;
  char* ptr = static_cast<char*>(dst);
  uint64_t off = current_offset_;
  while (total < to_read) {
    // Use binary search for better performance with many files
    auto it = std::upper_bound(segments_.begin(), segments_.end(), off,
        [](uint64_t offset, const Segment& s) { 
            return offset < s.base_offset + s.data_size; 
        });
    if (it == segments_.begin() && off < it->base_offset) {
      break;  // offset is before the first segment
    }
    if (it != segments_.begin()) {
      --it;  // upper_bound returns iterator to first element > off, we want <=
    }
    const auto& s = *it;
    uint64_t within = off - s.base_offset;
    auto seg_remaining = static_cast<size_t>(s.data_size - within);
    size_t want = std::min(to_read - total, seg_remaining);
    auto got = pread_fully(s.fd, s.data_start + within, ptr + total, want);
    if (!got.ok()) {
      return got.status();
    }
    if (*got == 0) {
      break;
    }
    total += *got;
    off += *got;
  }
  current_offset_ = off;
  return total;
}

absl::StatusOr<size_t> MultiSafetensorsSource::read_at(uint64_t offset, void* dst, size_t bytes) {
  auto st = OpenFiles();
  if (!st.ok()) {
    return st;
  }
  if (offset >= total_size_) {
    return static_cast<size_t>(0);
  }
  size_t to_read = std::min(bytes, static_cast<size_t>(total_size_ - offset));
  size_t total = 0;
  char* ptr = static_cast<char*>(dst);
  uint64_t off = offset;
  while (total < to_read) {
    // Use binary search for better performance with many files
    auto it = std::upper_bound(segments_.begin(), segments_.end(), off,
        [](uint64_t offset, const Segment& s) { 
            return offset < s.base_offset + s.data_size; 
        });
    if (it == segments_.begin() && off < it->base_offset) {
      break;  // offset is before the first segment
    }
    if (it != segments_.begin()) {
      --it;  // upper_bound returns iterator to first element > off, we want <=
    }
    const auto& s = *it;
    uint64_t within = off - s.base_offset;
    size_t seg_remaining = static_cast<size_t>(s.data_size - within);
    size_t want = std::min(to_read - total, seg_remaining);
    auto got = pread_fully(s.fd, s.data_start + within, ptr + total, want);
    if (!got.ok()) {
      return got.status();
    }
    if (*got == 0) {
      break;
    }
    total += *got;
    off += *got;
  }
  return total;
}

} // namespace stepcast::store::loader
