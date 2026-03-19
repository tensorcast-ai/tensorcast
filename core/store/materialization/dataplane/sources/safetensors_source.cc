// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/safetensors_source.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <endian.h>
#include <cerrno>
#include <cstring>
#include <string>

#include "absl/log/log.h"
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

SafetensorsSource::SafetensorsSource(SharedSafetensorsSegment segment)
    : file_path_(segment.path), shared_segment_(std::move(segment)) {}

SafetensorsSource::~SafetensorsSource() {
  absl::MutexLock lock(&init_mutex_);
  if (!shared_segment_.has_value() && fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

int SafetensorsSource::active_fd() const {
  if (shared_segment_.has_value()) {
    return shared_segment_->file ? shared_segment_->file->fd() : -1;
  }
  return fd_;
}

uint64_t SafetensorsSource::total_bytes() const {
  auto* self = const_cast<SafetensorsSource*>(this);
  if (auto st = self->OpenFile(); !st.ok()) {
    LOG(WARNING) << "SafetensorsSource: OpenFile failed in total_bytes: " << st;
    return 0;
  }
  return data_size_;
}

absl::Status SafetensorsSource::OpenFile() {
  absl::MutexLock lock(&init_mutex_);
  if (initialized_)
    return absl::OkStatus();
  if (shared_segment_.has_value()) {
    data_start_ = shared_segment_->data_start;
    data_size_ = shared_segment_->data_size;
    initialized_ = true;
    return absl::OkStatus();
  }
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
  auto header_info = ParseSafetensorsHeader(active_fd());
  if (!header_info.ok()) {
    return header_info.status();
  }

  // Validate the JSON header content
  std::string header;
  header.resize(static_cast<size_t>(header_info->header_length));
  auto got = pread_fully(active_fd(), sizeof(uint64_t), header.data(), header.size());
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
  if (shared_segment_.has_value() && shared_segment_->file) {
    const uint8_t* base = shared_segment_->file->mapped_base();
    if (base != nullptr) {
      std::memcpy(dst, base + data_start_ + current_offset_, to_read);
      current_offset_ += to_read;
      return to_read;
    }
  }
  auto got = pread_fully(active_fd(), data_start_ + current_offset_, dst, to_read);
  if (!got.ok())
    return got.status();
  if (*got != to_read) {
    return absl::DataLossError("SafetensorsSource short read before expected EOF");
  }
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
  if (shared_segment_.has_value() && shared_segment_->file) {
    const uint8_t* base = shared_segment_->file->mapped_base();
    if (base != nullptr) {
      std::memcpy(dst, base + data_start_ + offset, to_read);
      return to_read;
    }
  }
  auto got = pread_fully(active_fd(), data_start_ + offset, dst, to_read);
  if (!got.ok()) {
    return got.status();
  }
  if (*got != to_read) {
    return absl::DataLossError("SafetensorsSource short read before expected EOF");
  }
  return got;
}

const uint8_t* SafetensorsSource::cpu_base_ptr() const {
  auto* self = const_cast<SafetensorsSource*>(this);
  if (auto st = self->OpenFile(); !st.ok()) {
    return nullptr;
  }
  if (!shared_segment_.has_value() || !shared_segment_->file) {
    return nullptr;
  }
  const uint8_t* base = shared_segment_->file->mapped_base();
  if (base == nullptr) {
    return nullptr;
  }
  return base + data_start_;
}

} // namespace tensorcast::store::loader
