// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/file_partition_source.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"

namespace stepcast::store::loader {

FilePartitionSource::FilePartitionSource(Options options) : options_(std::move(options)) {
  // Use the provided use_direct_io flag, or auto-detect based on model size
  using_direct_io_ = options_.use_direct_io || (options_.total_size > 5ULL * 1024 * 1024 * 1024); // 5GB threshold

  // Allocate aligned buffer for O_DIRECT if needed
  if (using_direct_io_) {
    aligned_buffer_size_ = options_.chunk_size + kDirectIOAlignment;
    aligned_buffer_.reset(new char[aligned_buffer_size_]);
  }
}

FilePartitionSource::~FilePartitionSource() {
  CloseFiles();
}

absl::Status FilePartitionSource::OpenFiles() {
  if (initialized_) {
    return absl::OkStatus();
  }

  LOG(INFO) << "FilePartitionSource::OpenFiles opening " << options_.partition_paths.size()
            << " partition files, total_size=" << options_.total_size;

  if (options_.partition_paths.empty()) {
    return absl::InvalidArgumentError("No partition paths provided");
  }

  file_handles_.reserve(options_.partition_paths.size());
  uint64_t offset = 0;

  for (size_t i = 0; i < options_.partition_paths.size(); ++i) {
    const auto& path = options_.partition_paths[i];
    const auto size = options_.partition_sizes[i];

    int flags = O_RDONLY;
    if (using_direct_io_) {
      flags |= O_DIRECT;
    }

    int fd = ::open(path.c_str(), flags);
    if (fd < 0) {
      CloseFiles();
      return absl::InternalError(absl::StrFormat("Failed to open partition %s: %s", path.string(), strerror(errno)));
    }

    FileHandle handle;
    handle.fd = fd;
    handle.start_offset = offset;
    handle.end_offset = offset + size;
    file_handles_.push_back(handle);

    offset += size;
  }

  initialized_ = true;
  return absl::OkStatus();
}

void FilePartitionSource::CloseFiles() {
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
  } else {
    LOG(ERROR) << "FilePartitionSource::read failed: " << result.status();
  }

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
    // O_DIRECT requires aligned reads
    uint64_t aligned_offset = (partition_offset / kDirectIOAlignment) * kDirectIOAlignment;
    size_t offset_diff = partition_offset - aligned_offset;
    size_t aligned_size = ((bytes + offset_diff + kDirectIOAlignment - 1) / kDirectIOAlignment) * kDirectIOAlignment;

    // Ensure aligned buffer is large enough
    if (aligned_size > aligned_buffer_size_) {
      return absl::InternalError("Aligned read size exceeds buffer");
    }

    // Align buffer pointer
    char* aligned_ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(aligned_buffer_.get()) + kDirectIOAlignment - 1) & ~(kDirectIOAlignment - 1));

    ssize_t n = ::pread(handle.fd, aligned_ptr, aligned_size, aligned_offset);
    if (n < 0) {
      return absl::InternalError(absl::StrFormat("Failed to read partition: %s", strerror(errno)));
    }

    size_t actual_bytes = std::min(static_cast<size_t>(n) - offset_diff, bytes);
    std::memcpy(dst, aligned_ptr + offset_diff, actual_bytes);
    return actual_bytes;

  } else {
    // Regular read
    ssize_t n = ::pread(handle.fd, dst, bytes, partition_offset);
    if (n < 0) {
      return absl::InternalError(absl::StrFormat("Failed to read partition: %s", strerror(errno)));
    }
    return static_cast<size_t>(n);
  }
}

} // namespace stepcast::store::loader