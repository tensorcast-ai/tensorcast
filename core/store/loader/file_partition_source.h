// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/store/loader/source.h"

namespace stepcast::store::loader {

class FilePartitionSource : public SeekableSource {
 public:
  struct Options {
    std::vector<std::filesystem::path> partition_paths;
    std::vector<size_t> partition_sizes;
    uint64_t total_size = 0;
    size_t chunk_size = 128 * 1024 * 1024; // 128MB default
    bool use_direct_io = false; // Auto-detected if not set
  };

  explicit FilePartitionSource(Options options);
  ~FilePartitionSource() override;

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;

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

  absl::StatusOr<size_t> ReadFromPartition(size_t partition_idx, uint64_t partition_offset, void* dst, size_t bytes);

  Options options_;
  std::vector<FileHandle> file_handles_;

  // Protected by mutex for thread-safe access
  mutable absl::Mutex offset_mutex_;
  uint64_t current_offset_ = 0;

  bool using_direct_io_ = false;
  bool initialized_ = false;

  // For O_DIRECT alignment
  static constexpr size_t kDirectIOAlignment = 512;
  std::unique_ptr<char[]> aligned_buffer_;
  size_t aligned_buffer_size_ = 0;
};

} // namespace stepcast::store::loader