// Copyright (c) 2025, TensorCast Team.

#include "core/store/loader/file_partition_source.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>

#include <algorithm>
#include <cstring>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::store::loader {

FilePartitionSource::FilePartitionSource(Options options) : options_(std::move(options)) {
  // Use the provided use_direct_io flag, or auto-detect based on artifact size
  using_direct_io_ = options_.use_direct_io || (options_.total_size > 5ULL * 1024 * 1024 * 1024); // 5GB threshold

  // Allocate aligned buffer for O_DIRECT if needed
  if (using_direct_io_) {
    aligned_buffer_size_ = options_.io_batch_bytes + kDirectIOAlignment;
    aligned_buffer_.reset(new char[aligned_buffer_size_]);
  }
}

FilePartitionSource::~FilePartitionSource() {
  CloseFiles();
}

absl::Status FilePartitionSource::OpenFiles() {
  absl::MutexLock init_lock(&init_mutex_);
  if (initialized_) {
    return absl::OkStatus();
  }

  LOG(INFO) << "FilePartitionSource::OpenFiles opening " << options_.partition_paths.size()
            << " partition files, total_size=" << options_.total_size;

  if (options_.partition_paths.empty()) {
    // If there are no partitions and total_size is zero, treat as empty source (EOF)
    if (options_.total_size == 0) {
      initialized_ = true;
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError("No partition paths provided");
  }

  file_handles_.reserve(options_.partition_paths.size());

  // Attempt to open with current direct I/O setting. If O_DIRECT is unsupported,
  // fall back to regular I/O and retry once.
  bool attempted_fallback = false;
  while (true) {
    file_handles_.clear();
    uint64_t offset = 0;

    bool open_failed = false;
    int open_errno = 0;

    for (size_t i = 0; i < options_.partition_paths.size(); ++i) {
      const auto& path = options_.partition_paths[i];
      const auto size = options_.partition_sizes[i];

      int flags = O_RDONLY;
      if (using_direct_io_) {
        flags |= O_DIRECT;
      }

      int fd = ::open(path.c_str(), flags);
      if (fd < 0) {
        open_failed = true;
        open_errno = errno;
        break;
      }

      FileHandle handle;
      handle.fd = fd;
      handle.start_offset = offset;
      handle.end_offset = offset + size;
      file_handles_.push_back(handle);

      offset += size;
    }

    if (!open_failed) {
      initialized_ = true;
      return absl::OkStatus();
    }

    // Handle fallback from O_DIRECT if unsupported
    if (using_direct_io_ && !attempted_fallback &&
        (open_errno == EINVAL || open_errno == EOPNOTSUPP || open_errno == EPERM)) {
      LOG(INFO) << "FilePartitionSource::OpenFiles falling back from O_DIRECT due to errno=" << open_errno;
      // Close any partially opened file descriptors before retrying
      CloseFilesNoLock();
      using_direct_io_ = false;
      attempted_fallback = true;
      // Retry opening without O_DIRECT
      continue;
    }

    // If we reach here, opening failed and fallback (if any) either not applicable or already attempted
    CloseFilesNoLock();
    return absl::ErrnoToStatus(open_errno, "Failed to open partition(s)");
  }
}

void FilePartitionSource::CloseFiles() {
  absl::MutexLock init_lock(&init_mutex_);
  CloseFilesNoLock();
}

void FilePartitionSource::CloseFilesNoLock() {
  for (auto& handle : file_handles_) {
    if (handle.fd >= 0) {
      ::close(handle.fd);
      handle.fd = -1;
    }
  }
  file_handles_.clear();
  initialized_ = false;
}

absl::StatusOr<size_t> FilePartitionSource::read(void* dst, size_t max_bytes) {
  auto status = OpenFiles();
  if (!status.ok()) {
    return status;
  }

  // Simple approach: lock for the entire read operation
  absl::MutexLock lock(&offset_mutex_);

  if (current_offset_ >= options_.total_size) {
    LOG(INFO) << "FilePartitionSource::read returning EOF, current_offset=" << current_offset_
              << ", total_size=" << options_.total_size;
    return 0; // EOF
  }

  size_t bytes_to_read = std::min(max_bytes, static_cast<size_t>(options_.total_size - current_offset_));

  LOG(INFO) << "FilePartitionSource::read reading from offset " << current_offset_ << ", size " << bytes_to_read
            << ", total_size=" << options_.total_size;

  auto result = read_at(current_offset_, dst, bytes_to_read);
  if (result.ok()) {
    size_t bytes_read = *result;
    current_offset_ += bytes_read;
    LOG(INFO) << "FilePartitionSource::read successfully read " << bytes_read << " bytes, new offset "
              << current_offset_;
    return bytes_read;
  }
  LOG(ERROR) << "FilePartitionSource::read failed: " << result.status();

  return result;
}

absl::StatusOr<size_t> FilePartitionSource::read_at(uint64_t offset, void* dst, size_t bytes) {
  auto status = OpenFiles();
  if (!status.ok()) {
    return status;
  }

  if (offset >= options_.total_size) {
    return 0; // EOF
  }

  size_t bytes_to_read = std::min(bytes, static_cast<size_t>(options_.total_size - offset));
  size_t total_read = 0;
  char* dst_ptr = static_cast<char*>(dst);

  // Find starting partition
  size_t partition_idx = 0;
  for (; partition_idx < file_handles_.size(); ++partition_idx) {
    if (offset < file_handles_[partition_idx].end_offset) {
      break;
    }
  }

  if (partition_idx >= file_handles_.size()) {
    return absl::InternalError("Invalid offset");
  }

  // Read from potentially multiple partitions
  while (total_read < bytes_to_read && partition_idx < file_handles_.size()) {
    const auto& handle = file_handles_[partition_idx];
    uint64_t partition_offset = offset - handle.start_offset;
    size_t partition_remaining = handle.end_offset - offset;
    size_t to_read = std::min(bytes_to_read - total_read, partition_remaining);

    auto read_result = ReadFromPartition(partition_idx, partition_offset, dst_ptr + total_read, to_read);
    if (!read_result.ok()) {
      return read_result;
    }

    size_t bytes_read = *read_result;
    if (bytes_read == 0) {
      break; // Unexpected EOF
    }

    total_read += bytes_read;
    offset += bytes_read;

    if (offset >= handle.end_offset) {
      partition_idx++;
    }
  }

  return total_read;
}

absl::StatusOr<size_t> FilePartitionSource::ReadFromPartition(
    size_t partition_idx,
    uint64_t partition_offset,
    void* dst,
    size_t bytes) {
  const auto& handle = file_handles_[partition_idx];

  if (using_direct_io_) {
    // Perform chunked aligned reads using the bounce buffer to avoid exceeding its capacity.
    size_t total_copied = 0;
    char* dst_bytes = static_cast<char*>(dst);

    // Compute aligned buffer pointer and its usable capacity once.
    char* base = aligned_buffer_.get();
    char* aligned_ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(base) + kDirectIOAlignment - 1) & ~(kDirectIOAlignment - 1));
    size_t max_aligned_bytes = aligned_buffer_size_ - static_cast<size_t>(aligned_ptr - base);

    while (total_copied < bytes) {
      uint64_t cur_off = partition_offset + total_copied;
      size_t remaining = bytes - total_copied;

      uint64_t aligned_offset = (cur_off / kDirectIOAlignment) * kDirectIOAlignment;
      auto offset_diff = static_cast<size_t>(cur_off - aligned_offset);

      // Limit each iteration's payload to io_batch_bytes
      size_t payload = std::min(remaining, options_.io_batch_bytes);

      // Round up to alignment, then cap to bounce buffer capacity if needed
      size_t aligned_size =
          ((payload + offset_diff + kDirectIOAlignment - 1) / kDirectIOAlignment) * kDirectIOAlignment;
      if (aligned_size > max_aligned_bytes) {
        if (max_aligned_bytes <= offset_diff) {
          return absl::InternalError("Aligned buffer too small");
        }
        size_t max_payload = max_aligned_bytes - offset_diff;
        payload = std::min(payload, max_payload);
        aligned_size = ((payload + offset_diff + kDirectIOAlignment - 1) / kDirectIOAlignment) * kDirectIOAlignment;
      }

      ssize_t n = ::pread(handle.fd, aligned_ptr, aligned_size, aligned_offset);
      if (n < 0) {
        return absl::ErrnoToStatus(errno, "Failed to read partition");
      }
      if (std::cmp_less_equal(n, static_cast<ssize_t>(offset_diff))) {
        // Nothing useful read
        break;
      }

      size_t actual = std::min(static_cast<size_t>(n) - offset_diff, payload);
      std::memcpy(dst_bytes + total_copied, aligned_ptr + offset_diff, actual);
      total_copied += actual;

      if (actual == 0) {
        break;
      }
    }
    return total_copied;
  }
  // Regular read
  ssize_t n = ::pread(handle.fd, dst, bytes, partition_offset);
  if (n < 0) {
    return absl::ErrnoToStatus(errno, "Failed to read partition");
  }
  return static_cast<size_t>(n);
}

} // namespace tensorcast::store::loader
