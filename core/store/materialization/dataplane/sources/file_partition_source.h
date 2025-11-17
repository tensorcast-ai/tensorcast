// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/store/materialization/dataplane/contracts/source.h"

namespace tensorcast::store::loader {

class FilePartitionSource : public SeekableSource {
 public:
  struct Options {
    std::vector<std::filesystem::path> partition_paths;
    std::vector<size_t> partition_sizes;
    uint64_t total_size = 0;
    // Rename: io_batch_bytes (cap per read iteration)
    size_t io_batch_bytes = 128 * 1024 * 1024; // 128MB default
    bool use_direct_io = false; // Auto-detected if not set
  };

  explicit FilePartitionSource(Options options);
  ~FilePartitionSource() override;

  // Sequential read using internal offset (thread-safe via mutex).
  // Multiple threads can call this method concurrently; the internal
  // offset is protected by offset_mutex_ to ensure thread safety.
  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;

  // Stateless read at specific offset (inherently thread-safe).
  // Multiple threads can call this method concurrently without any
  // synchronization as it doesn't modify any internal state.
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override;

  uint64_t total_size() const {
    return options_.total_size;
  }

  bool is_using_direct_io() const {
    return using_direct_io_;
  }

 private:
  struct FileHandle {
    int fd = -1;
    uint64_t start_offset = 0;
    uint64_t end_offset = 0;
  };

  absl::Status OpenFiles();
  void CloseFiles();
  // Internal helper that assumes init_mutex_ is already held by caller.
  void CloseFilesNoLock();

  absl::StatusOr<size_t> ReadFromPartition(size_t partition_idx, uint64_t partition_offset, void* dst, size_t bytes);

  Options options_;
  // Guards initialization and all mutations to file_handles_, initialized_, and
  // using_direct_io_ during OpenFiles()/CloseFiles(). Ensures read_at() sees a
  // consistent snapshot after OpenFiles() returns.
  absl::Mutex init_mutex_;
  std::vector<FileHandle> file_handles_;

  // Internal offset for sequential read() calls.
  // Protected by offset_mutex_ for thread-safe access.
  // Note: read_at() does not use or modify this offset.
  mutable absl::Mutex offset_mutex_;
  uint64_t current_offset_ = 0;

  bool using_direct_io_ = false;
  bool initialized_ = false;

  // For O_DIRECT alignment
  static constexpr size_t kDirectIOAlignment = 512;
  std::unique_ptr<char[]> aligned_buffer_;
  size_t aligned_buffer_size_ = 0;
};

} // namespace tensorcast::store::loader
