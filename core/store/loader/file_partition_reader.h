// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <filesystem>
#include <vector>

#include "absl/status/status.h"

namespace stepcast::store {

/**
 * @brief Helper class for reading data from partitioned model files.
 *
 * This class encapsulates the logic for opening, reading, and closing
 * multiple partition files, providing a unified interface for reading
 * data at any global offset across all partitions.
 *
 * Supports DIRECT_IO for large models (> 5GB) to bypass kernel page cache
 * for better performance and reduced memory pressure.
 *
 * DIRECT_IO behavior can be controlled via the SCSTORE_DIRECT_IO_MODE environment variable:
 * - "false" or unset (default): Never use DIRECT_IO
 * - "auto": Use DIRECT_IO for models larger than 5GB
 * - "true" or "enabled": Always attempt to use DIRECT_IO
 */
class FilePartitionReader {
 public:
  FilePartitionReader() = default;
  ~FilePartitionReader();

  // Disable copy
  FilePartitionReader(const FilePartitionReader&) = delete;
  FilePartitionReader& operator=(const FilePartitionReader&) = delete;

  /**
   * @brief Opens all partition files for reading.
   *
   * DIRECT_IO behavior is controlled by SCSTORE_DIRECT_IO_MODE environment variable.
   * Falls back to regular I/O if DIRECT_IO fails.
   *
   * @param paths Vector of paths to partition files.
   * @param sizes Vector of sizes for each partition file.
   * @return absl::Status indicating success or failure.
   */
  absl::Status open_partitions(const std::vector<std::filesystem::path>& paths, const std::vector<size_t>& sizes);

  /**
   * @brief Closes all open partition files.
   */
  void close_all();

  /**
   * @brief Reads data from the partitions at a specific global offset.
   *
   * This method handles reading across partition boundaries automatically.
   * When using DIRECT_IO, ensures proper alignment of read operations.
   *
   * @param global_offset Offset from the beginning of the entire model.
   * @param buffer Buffer to read data into (must be aligned for DIRECT_IO).
   * @param bytes_to_read Number of bytes to read.
   * @return absl::Status indicating success or failure.
   */
  absl::Status read_at_offset(size_t global_offset, char* buffer, size_t bytes_to_read);

  /**
   * @brief Gets the total size across all partitions.
   * @return Total size in bytes.
   */
  size_t get_total_size() const {
    return total_size_;
  }

  /**
   * @brief Checks if DIRECT_IO is enabled.
   * @return true if using DIRECT_IO, false otherwise.
   */
  bool is_direct_io_enabled() const {
    return use_direct_io_;
  }

 private:
  std::vector<int> fds_;
  std::vector<size_t> partition_sizes_;
  size_t total_size_ = 0;
  bool is_open_ = false;
  bool use_direct_io_ = false;

  // Constants for DIRECT_IO alignment
  static constexpr size_t kDirectIOAlignment = 512; // Common block size for DIRECT_IO
  static constexpr size_t kLargeModelThreshold = 5ULL * 1024 * 1024 * 1024; // 5GB

  // Added to keep track of partition file paths (needed for fallback non-DIRECT_IO reads)
  std::vector<std::filesystem::path> partition_paths_;
};

} // namespace stepcast::store