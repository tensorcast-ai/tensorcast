// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"

namespace tensorcast::store::loader {

// Concatenates the data buffers of multiple .safetensors files into one logical
// byte space. Headers are hidden and not exposed to readers.
class MultiSafetensorsSource : public SeekableSource {
 public:
  explicit MultiSafetensorsSource(std::vector<std::filesystem::path> file_paths);
  explicit MultiSafetensorsSource(std::vector<SharedSafetensorsSegment> shared_segments);
  ~MultiSafetensorsSource() override;

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override;

  [[nodiscard]] uint64_t total_bytes() const override;

  [[nodiscard]] uint64_t total_size() const {
    return total_bytes();
  }

 private:
  struct Segment {
    int fd = -1;
    bool owns_fd = true;
    std::shared_ptr<SharedFileHandle> shared_file;
    uint64_t data_start = 0; // file offset where payload starts
    uint64_t data_size = 0; // payload size
    uint64_t base_offset = 0; // global logical offset of segment start
  };

  absl::Status OpenFiles();
  absl::Status ParseAllHeadersLocked();

  std::vector<std::filesystem::path> file_paths_;
  std::vector<SharedSafetensorsSegment> shared_segments_;
  std::vector<Segment> segments_;
  uint64_t total_size_ = 0;
  bool initialized_ = false;

  absl::Mutex init_mutex_;
  absl::Mutex offset_mutex_;
  uint64_t current_offset_ = 0;
};

} // namespace tensorcast::store::loader
