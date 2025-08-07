// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <filesystem>
#include <vector>

#include "absl/status/status.h"
#include "core/store/loader/sink.h"

namespace stepcast::store::loader {

class DVMPMappedSink : public Sink {
 public:
  struct Options {
    void* base_addr = nullptr; // Base address from DVMP allocation
    uint64_t total_size = 0;
    std::vector<std::filesystem::path> partition_paths;
    std::vector<size_t> partition_sizes;
    bool populate_pages = true; // Use MAP_POPULATE
  };

  explicit DVMPMappedSink(Options options);
  ~DVMPMappedSink() override;

  // For streaming interface compatibility
  absl::Status write(const void* src, size_t bytes) override;

  // Direct mapping method for fast-path
  absl::Status map_partitions();

  absl::Status close() override;

 private:
  absl::Status map_partition(size_t partition_idx);
  void unmap_partitions();

  Options options_;
  std::vector<void*> mapped_regions_;
  uint64_t current_offset_ = 0;
  bool mappings_done_ = false;
};

} // namespace stepcast::store::loader