// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"

namespace tensorcast::store::loader {

class SharedFileHandle {
 public:
  SharedFileHandle(std::filesystem::path path, int fd, uint64_t file_size);
  ~SharedFileHandle();

  SharedFileHandle(const SharedFileHandle&) = delete;
  SharedFileHandle& operator=(const SharedFileHandle&) = delete;

  [[nodiscard]] int fd() const {
    return fd_;
  }

  [[nodiscard]] uint64_t file_size() const {
    return file_size_;
  }

  [[nodiscard]] const std::filesystem::path& path() const {
    return path_;
  }

  [[nodiscard]] const uint8_t* mapped_base() const;

 private:
  std::filesystem::path path_;
  int fd_{-1};
  uint64_t file_size_{0};

  mutable absl::Mutex mutex_;
  mutable const uint8_t* mapped_base_ ABSL_GUARDED_BY(mutex_) = nullptr;
  mutable bool mmap_attempted_ ABSL_GUARDED_BY(mutex_) = false;
};

struct SharedSafetensorsSegment {
  std::filesystem::path path;
  std::shared_ptr<SharedFileHandle> file;
  uint64_t data_start{0};
  uint64_t data_size{0};
  uint64_t base_offset{0};
};

struct DiskArtifactContextCacheStats {
  uint64_t context_hits{0};
  uint64_t context_misses{0};
  uint64_t index_hits{0};
  uint64_t index_misses{0};
};

class DiskArtifactContext {
 public:
  DiskArtifactContext(
      std::filesystem::path artifact_path,
      std::vector<std::filesystem::path> partition_paths,
      std::vector<size_t> partition_sizes,
      uint64_t total_size,
      bool is_safetensors,
      bool descriptor_present,
      bool tensor_index_json_present,
      bool tensor_index_cbor_present,
      std::vector<SharedSafetensorsSegment> safetensors_segments);

  [[nodiscard]] const std::filesystem::path& artifact_path() const {
    return artifact_path_;
  }

  [[nodiscard]] const std::vector<std::filesystem::path>& partition_paths() const {
    return partition_paths_;
  }

  [[nodiscard]] const std::vector<size_t>& partition_sizes() const {
    return partition_sizes_;
  }

  [[nodiscard]] uint64_t total_size() const {
    return total_size_;
  }

  [[nodiscard]] bool is_safetensors() const {
    return is_safetensors_;
  }

  [[nodiscard]] bool descriptor_present() const {
    return descriptor_present_;
  }

  [[nodiscard]] bool tensor_index_json_present() const {
    return tensor_index_json_present_;
  }

  [[nodiscard]] bool tensor_index_cbor_present() const {
    return tensor_index_cbor_present_;
  }

  [[nodiscard]] const std::vector<SharedSafetensorsSegment>& safetensors_segments() const {
    return safetensors_segments_;
  }

  absl::StatusOr<IndexInfo> get_index_info(int target_device_id) const;

 private:
  std::filesystem::path artifact_path_;
  std::vector<std::filesystem::path> partition_paths_;
  std::vector<size_t> partition_sizes_;
  uint64_t total_size_{0};
  bool is_safetensors_{false};
  bool descriptor_present_{false};
  bool tensor_index_json_present_{false};
  bool tensor_index_cbor_present_{false};
  std::vector<SharedSafetensorsSegment> safetensors_segments_;

  mutable absl::Mutex mutex_;
  mutable std::vector<std::pair<int, IndexInfo>> index_cache_ ABSL_GUARDED_BY(mutex_);
};

absl::StatusOr<std::shared_ptr<const DiskArtifactContext>> get_disk_artifact_context(
    const std::filesystem::path& artifact_path);

DiskArtifactContextCacheStats get_disk_artifact_context_cache_stats();

void reset_disk_artifact_context_cache_for_testing();

} // namespace tensorcast::store::loader
