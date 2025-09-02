// Copyright (c) 2025, TensorCast Team.

//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
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
#include <mutex>
#include <unordered_set>
#include <vector>

#include "gsl/pointers"

namespace tensorcast::common::memory {

class PinnedMemoryPool {
 public:
  // Alignment requirements for DIRECT_IO support
  static constexpr size_t kMemoryAlignment = 4096; // Page size for optimal alignment
  static constexpr size_t kDirectIOAlignment = 512; // Minimum alignment for DIRECT_IO

  PinnedMemoryPool(size_t total_size, size_t chunk_size);
  ~PinnedMemoryPool();

  int allocate(
      size_t size,
      std::vector<char*>& buffers,
      const std::chrono::milliseconds& timeout = std::chrono::milliseconds::zero());
  int deallocate(std::vector<char*>& buffers);
  size_t chunk_size() const {
    return chunk_size_;
  }

  size_t get_available_size() const;

  // Expose current pool buffers for registration/warmup purposes.
  // Returns a snapshot copy of buffer base pointers.
  std::vector<gsl::not_null<char*>> list_buffers() const;

  // Forbid copy and assignment
  PinnedMemoryPool(const PinnedMemoryPool&) = delete;
  PinnedMemoryPool& operator=(const PinnedMemoryPool&) = delete;

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::unordered_set<char*> free_list_;
  std::unordered_set<char*> pool_;
  size_t chunk_size_; // May be adjusted in constructor for alignment
};
} // namespace tensorcast::common::memory
