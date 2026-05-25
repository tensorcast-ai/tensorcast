// Copyright (c) 2025-2026, TensorCast Team.

//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//  Modified by TensorCast Team, 2025-2026.
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gsl/pointers"

namespace tensorcast::common::memory {

class PinnedBufferPool {
 public:
  // Alignment requirements for DIRECT_IO support
  static constexpr size_t kMemoryAlignment = 4096; // Page size for optimal alignment
  static constexpr size_t kDirectIOAlignment = 512; // Minimum alignment for DIRECT_IO

  struct Slab {
    gsl::not_null<char*> base;
    size_t bytes;
  };

  struct Options {
    std::string name;
    // If >=0, best-effort bind the backing slab to this NUMA node before pinning.
    // Default (-1) leaves placement to the OS (first-touch / default policy).
    int numa_node = -1;
    // If true and numa_node is set, pre-fault the slab (touch each page) before
    // cudaHostRegister to make the NUMA placement deterministic.
    bool prefault = false;
    // When false, the pool only allocates/owns the slabs during construction.
    // Call register_host_memory() before first use.
    bool register_on_create = true;
  };

  explicit PinnedBufferPool(size_t total_size, size_t chunk_size);
  PinnedBufferPool(size_t total_size, size_t chunk_size, std::string name);
  PinnedBufferPool(size_t total_size, size_t chunk_size, Options options);
  ~PinnedBufferPool();

  int allocate(
      size_t size,
      std::vector<char*>& buffers,
      const std::chrono::milliseconds& timeout = std::chrono::milliseconds::zero(),
      std::string_view request_context = {});
  int deallocate(std::vector<char*>& buffers);

  // Canonical accessor: transfer slice/window size in bytes.
  size_t slice_bytes() const {
    return chunk_size_;
  }

  std::string_view name() const {
    return name_;
  }

  size_t get_available_size() const;
  size_t capacity_slices() const;
  size_t free_slices() const;
  size_t in_use_slices() const;
  size_t waiters() const;

  uint64_t acquire_timeouts_total() const;
  uint64_t budget_exhausted_total() const;
  [[nodiscard]] bool is_host_registered() const;
  absl::Status register_host_memory();

  // Expose current pool buffers for registration/warmup purposes.
  // Returns a snapshot copy of buffer base pointers.
  std::vector<gsl::not_null<char*>> list_buffers() const;

  // Expose the backing allocation slabs. When slabs are used, a single MR can
  // be registered per slab and reused for any slice pointer within the slab.
  std::vector<Slab> list_slabs() const;

  std::optional<Slab> slab_for_ptr(gsl::not_null<void*> ptr) const;

  // Forbid copy and assignment
  PinnedBufferPool(const PinnedBufferPool&) = delete;
  PinnedBufferPool& operator=(const PinnedBufferPool&) = delete;

 private:
  std::vector<Slab> slabs_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::unordered_set<char*> free_list_;
  std::unordered_set<char*> pool_;
  size_t waiters_ = 0;
  uint64_t acquire_timeouts_total_ = 0;
  uint64_t budget_exhausted_total_ = 0;
  size_t chunk_size_; // May be adjusted in constructor for alignment
  std::string name_;
  bool host_registered_ = false;
};
} // namespace tensorcast::common::memory
