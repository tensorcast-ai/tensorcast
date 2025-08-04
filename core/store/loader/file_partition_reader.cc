// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/file_partition_reader.h"

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cctype> // for tolower
#include <cstdlib> // for posix_memalign, free
#include <cstring>
#include <memory> // for std::unique_ptr
#include <numeric>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"

namespace stepcast::store {

namespace {
// Helper function to get direct I/O mode from environment variable
enum class DirectIOMode {
  kDisabled, // Never use direct I/O
  kAuto, // Use direct I/O for large models
  kEnabled // Always use direct I/O
};

DirectIOMode get_direct_io_mode() {
  const char* env_value = std::getenv("SCSTORE_DIRECT_IO_MODE");
  if (env_value == nullptr) {
    return DirectIOMode::kDisabled; // Default to disabled
  }

  std::string mode(env_value);
  std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);

  if (mode == "true" || mode == "enabled" || mode == "1") {
    return DirectIOMode::kEnabled;
  } else if (mode == "auto") {
    return DirectIOMode::kAuto;
  } else {
    return DirectIOMode::kDisabled;
  }
}
} // namespace

FilePartitionReader::~FilePartitionReader() {
  close_all();
}

absl::Status FilePartitionReader::open_partitions(
    const std::vector<std::filesystem::path>& paths,
    const std::vector<size_t>& sizes) {
  if (is_open_) {
    return absl::FailedPreconditionError("Partitions already open");
  }
  if (paths.size() != sizes.size()) {
    return absl::InvalidArgumentError("Paths and sizes vectors must have the same size");
  }
  if (paths.empty()) {
    return absl::InvalidArgumentError("No partitions provided");
  }
  fds_.reserve(paths.size());
  partition_sizes_ = sizes;
  partition_paths_ = paths; // Keep a copy for potential fallback reads
  total_size_ = std::accumulate(sizes.begin(), sizes.end(), size_t(0));

  // Determine if we should use DIRECT_IO based on environment variable and model size
  DirectIOMode direct_io_mode = get_direct_io_mode();
  switch (direct_io_mode) {
    case DirectIOMode::kEnabled:
      use_direct_io_ = true;
      LOG(INFO) << "FilePartitionReader: DIRECT_IO enabled by environment variable";
      break;
    case DirectIOMode::kAuto:
      use_direct_io_ = (total_size_ > kLargeModelThreshold);
      if (use_direct_io_) {
        LOG(INFO) << "FilePartitionReader: Model size " << total_size_ / (1024.0 * 1024 * 1024)
                  << " GB exceeds threshold, attempting to enable DIRECT_IO (auto mode)";
      } else {
        LOG(INFO) << "FilePartitionReader: Model size " << total_size_ / (1024.0 * 1024 * 1024)
                  << " GB below threshold, using regular I/O (auto mode)";
      }
      break;
    case DirectIOMode::kDisabled:
    default:
      use_direct_io_ = false;
      VLOG(4) << "FilePartitionReader: DIRECT_IO disabled by environment variable";
      break;
  }

  for (const auto& path : paths) {
    int flags = O_RDONLY;

    // Try DIRECT_IO for large models
    if (use_direct_io_) {
#ifdef O_DIRECT
      flags |= O_DIRECT;
#else
      LOG(WARNING) << "O_DIRECT not supported on this platform, falling back to regular I/O";
      use_direct_io_ = false;
#endif
    }

    int fd = open(path.c_str(), flags);
    if (fd < 0 && use_direct_io_) {
      // DIRECT_IO failed, try again without it
      LOG(WARNING) << "Failed to open " << path.string() << " with O_DIRECT (errno=" << errno << ": "
                   << std::strerror(errno) << "), falling back to regular I/O";
      use_direct_io_ = false;
      flags = O_RDONLY;
      fd = open(path.c_str(), flags);
    }

    if (fd < 0) {
      std::string err_msg = absl::StrFormat("Failed to open %s: %s", path.string(), std::strerror(errno));
      LOG(ERROR) << "FilePartitionReader: " << err_msg;
      // Close already opened FDs
      close_all();
      return absl::InternalError(err_msg);
    }
    fds_.push_back(fd);
  }

  is_open_ = true;
  VLOG(1) << "FilePartitionReader: Opened " << fds_.size() << " partition files"
          << (use_direct_io_ ? " with DIRECT_IO" : " with regular I/O");
  return absl::OkStatus();
}

void FilePartitionReader::close_all() {
  for (int fd : fds_) {
    if (fd >= 0) {
      close(fd);
    }
  }
  fds_.clear();
  partition_sizes_.clear();
  partition_paths_.clear();
  total_size_ = 0;
  is_open_ = false;
}

absl::Status FilePartitionReader::read_at_offset(size_t global_offset, char* buffer, size_t bytes_to_read) {
  if (!is_open_) {
    return absl::FailedPreconditionError("Partitions not open");
  }
  if (global_offset + bytes_to_read > total_size_) {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "Read request (offset=%zu, size=%zu) exceeds total size %zu", global_offset, bytes_to_read, total_size_));
  }

  // Check alignment requirements for DIRECT_IO
  if (use_direct_io_) {
    // Check buffer alignment
    if (reinterpret_cast<uintptr_t>(buffer) % kDirectIOAlignment != 0) {
      LOG(WARNING) << "Buffer not aligned for DIRECT_IO (addr=" << static_cast<void*>(buffer)
                   << "), this may cause performance issues";
    }

    // For DIRECT_IO, we handle alignment at the partition level
    // Each partition read will be aligned independently
  }

  // Find the starting partition and offset within it
  size_t current_partition = 0;
  size_t partition_offset = global_offset;

  while (current_partition < partition_sizes_.size() && partition_offset >= partition_sizes_[current_partition]) {
    partition_offset -= partition_sizes_[current_partition];
    current_partition++;
  }

  // Read data, potentially spanning multiple partitions
  size_t bytes_read = 0;
  while (bytes_read < bytes_to_read && current_partition < fds_.size()) {
    size_t partition_remaining = partition_sizes_[current_partition] - partition_offset;

    // Skip over empty partitions to avoid busy-looping / deadlock
    if (partition_remaining == 0) {
      current_partition++;
      partition_offset = 0;
      continue;
    }

    size_t to_read = std::min(bytes_to_read - bytes_read, partition_remaining);

    // If DIRECT_IO is enabled but the remaining bytes to read are smaller than the required
    // alignment size we cannot satisfy the kernel's alignment constraints. In this (rare)
    // situation fall back to a temporary file descriptor opened without O_DIRECT so that
    // we can finish reading the trailing bytes (this typically happens at EOF).
    if (use_direct_io_ && to_read < kDirectIOAlignment) {
      // Open a temporary FD without O_DIRECT for this partition.
      int fd_tmp = open(partition_paths_[current_partition].c_str(), O_RDONLY);
      if (fd_tmp < 0) {
        return absl::InternalError(
            absl::StrFormat(
                "Failed to open %s for fallback read: %s",
                partition_paths_[current_partition].string(),
                std::strerror(errno)));
      }

      size_t local_bytes_read = 0;
      while (local_bytes_read < to_read) {
        ssize_t ret = pread(
            fd_tmp,
            buffer + bytes_read + local_bytes_read,
            to_read - local_bytes_read,
            partition_offset + local_bytes_read);

        if (ret < 0) {
          close(fd_tmp);
          return absl::InternalError(
              absl::StrFormat(
                  "Fallback pread() failed for partition %zu (offset=%zu, size=%zu): %s",
                  current_partition,
                  partition_offset + local_bytes_read,
                  to_read - local_bytes_read,
                  std::strerror(errno)));
        }

        if (ret == 0) {
          close(fd_tmp);
          return absl::InternalError(
              absl::StrFormat(
                  "Unexpected EOF during fallback read in partition %zu at offset %zu (wanted %zu more bytes)",
                  current_partition,
                  partition_offset + local_bytes_read,
                  to_read - local_bytes_read));
        }

        local_bytes_read += static_cast<size_t>(ret);
      }

      close(fd_tmp);

      bytes_read += to_read;
      partition_offset += to_read;

      // If we've reached the end of the current partition, move to the next one
      if (partition_offset >= partition_sizes_[current_partition]) {
        current_partition++;
        partition_offset = 0;
      }
      continue; // Proceed with the main while-loop
    }

    // Handle DIRECT_IO alignment for this partition read
    if (use_direct_io_ && (partition_offset % kDirectIOAlignment != 0 || to_read % kDirectIOAlignment != 0)) {
      // For unaligned reads with DIRECT_IO, we need to read aligned blocks
      // and extract the requested data. Ensure the request never crosses
      // the current partition boundary because we can only read from a
      // single fd at a time.

      size_t aligned_offset = (partition_offset / kDirectIOAlignment) * kDirectIOAlignment;
      size_t offset_adjustment = partition_offset - aligned_offset;

      if (offset_adjustment + to_read > partition_remaining) {
        to_read = partition_remaining - offset_adjustment;
      }

      // If nothing remains to read in this partition, move to next partition.
      if (to_read == 0) {
        current_partition++;
        partition_offset = 0;
        continue;
      }

      // After a potential shrink, compute aligned_size so that:
      //   1. It is a multiple of kDirectIOAlignment
      //   2. It does NOT exceed partition_remaining (cannot read past EOF)
      //   3. It is large enough to contain offset_adjustment + to_read bytes

      size_t aligned_size =
          ((offset_adjustment + to_read + kDirectIOAlignment - 1) / kDirectIOAlignment) * kDirectIOAlignment;

      const size_t partition_remaining_aligned = (partition_remaining / kDirectIOAlignment) * kDirectIOAlignment;
      if (aligned_size > partition_remaining_aligned) {
        aligned_size = partition_remaining_aligned;

        // Re-adjust to_read so the requested payload fits within aligned_size
        if (aligned_size <= offset_adjustment) {
          // Nothing we can read from this partition in an aligned way
          current_partition++;
          partition_offset = 0;
          continue;
        }
        to_read = aligned_size - offset_adjustment;
      }

      // Allocate aligned temporary buffer using posix_memalign so that the memory address
      // satisfies DIRECT_IO requirements (kDirectIOAlignment alignment).
      void* raw_aligned_ptr = nullptr;
      int mem_rc = posix_memalign(&raw_aligned_ptr, kDirectIOAlignment, aligned_size);
      if (mem_rc != 0 || raw_aligned_ptr == nullptr) {
        return absl::InternalError(
            absl::StrFormat(
                "posix_memalign failed (aligned_size=%zu, alignment=%zu, rc=%d)",
                aligned_size,
                kDirectIOAlignment,
                mem_rc));
      }

      // Use a unique_ptr with custom deleter to ensure the buffer is freed automatically.
      std::unique_ptr<char, decltype(&std::free)> temp_buffer(static_cast<char*>(raw_aligned_ptr), &std::free);

      ssize_t ret = pread(fds_[current_partition], temp_buffer.get(), aligned_size, aligned_offset);
      if (ret < 0) {
        return absl::InternalError(
            absl::StrFormat(
                "pread() failed for partition %zu with DIRECT_IO (aligned_offset=%zu, aligned_size=%zu): %s",
                current_partition,
                aligned_offset,
                aligned_size,
                std::strerror(errno)));
      }

      if (ret < static_cast<ssize_t>(offset_adjustment + to_read)) {
        return absl::InternalError(
            absl::StrFormat(
                "Unexpected short read in partition %zu with DIRECT_IO (got %zd, needed at least %zu)",
                current_partition,
                ret,
                offset_adjustment + to_read));
      }

      // Copy the requested portion to the output buffer
      std::memcpy(buffer + bytes_read, temp_buffer.get() + offset_adjustment, to_read);
      bytes_read += to_read;
      partition_offset += to_read;
    } else {
      // Regular read or aligned DIRECT_IO read
      size_t local_bytes_read = 0;
      while (local_bytes_read < to_read) {
        ssize_t ret = pread(
            fds_[current_partition],
            buffer + bytes_read + local_bytes_read,
            to_read - local_bytes_read,
            partition_offset + local_bytes_read);

        if (ret < 0) {
          return absl::InternalError(
              absl::StrFormat(
                  "pread() failed for partition %zu (offset=%zu, size=%zu): %s",
                  current_partition,
                  partition_offset + local_bytes_read,
                  to_read - local_bytes_read,
                  std::strerror(errno)));
        }

        if (ret == 0) {
          return absl::InternalError(
              absl::StrFormat(
                  "Unexpected EOF in partition %zu at offset %zu (wanted %zu more bytes)",
                  current_partition,
                  partition_offset + local_bytes_read,
                  to_read - local_bytes_read));
        }

        local_bytes_read += static_cast<size_t>(ret);
      }

      bytes_read += local_bytes_read;
      partition_offset += local_bytes_read;
    }

    // If we've reached the end of the current partition, move to the next one
    if (partition_offset >= partition_sizes_[current_partition]) {
      current_partition++;
      partition_offset = 0;
    }
  }

  if (bytes_read < bytes_to_read) {
    return absl::InternalError(
        absl::StrFormat("Could not read all requested bytes (read %zu of %zu)", bytes_read, bytes_to_read));
  }

  return absl::OkStatus();
}

} // namespace stepcast::store