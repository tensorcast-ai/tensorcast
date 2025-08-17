// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/store/loader/source.h"

namespace stepcast::store::loader {

// Presents the data buffer of a single .safetensors file as a contiguous byte
// space starting at logical offset 0. The file header is hidden from readers.
class SafetensorsSource : public SeekableSource {
 public:
  explicit SafetensorsSource(std::filesystem::path file_path);
  ~SafetensorsSource() override;

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override;

  uint64_t total_size() const {
    return data_size_;
  }

 private:
  absl::Status OpenFile();
  absl::Status ParseHeaderLocked();

  std::filesystem::path file_path_;
  int fd_ = -1;
  uint64_t data_start_ = 0;
  uint64_t data_size_ = 0;
  bool initialized_ = false;

  absl::Mutex init_mutex_;
  absl::Mutex offset_mutex_;
  uint64_t current_offset_ = 0;
};

} // namespace stepcast::store::loader
