// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"

#include "core/common/memory/pinned_buffer_pool.h"

namespace tensorcast::common::memory {

class PinnedMemoryAuthority {
 public:
  struct ClassConfig {
    std::string name;
    uint64_t slice_bytes = 0;
    // Phase 1 fixed-allocation: fully preallocated capacity of this class pool.
    uint64_t pool_bytes = 0;
    bool rdma_preregister = false;
    int numa_node = -1;
    bool numa_prefault = false;
  };

  struct Config {
    // Total pinned reservation (derived in Phase 1 as sum(classes[].pool_bytes)).
    uint64_t total_bytes = 0;
    absl::Duration allocation_timeout = absl::Seconds(30);
    std::vector<ClassConfig> classes;
    bool defer_host_registration = false;
  };

  static absl::StatusOr<std::shared_ptr<PinnedMemoryAuthority>> create(Config cfg);

  absl::StatusOr<std::shared_ptr<PinnedBufferPool>> get_class_pool(std::string_view name) const;
  absl::StatusOr<ClassConfig> get_class_config(std::string_view name) const;
  absl::Status register_all_pools();

  uint64_t total_bytes() const {
    return cfg_.total_bytes;
  }

  uint64_t committed_bytes() const;

  const Config& config() const {
    return cfg_;
  }

  std::vector<std::string> class_names() const;

 private:
  explicit PinnedMemoryAuthority(Config cfg);
  absl::Status validate_and_build_pools();

  Config cfg_;
  std::vector<ClassConfig> classes_;
  std::vector<std::shared_ptr<PinnedBufferPool>> pools_;
};

} // namespace tensorcast::common::memory
