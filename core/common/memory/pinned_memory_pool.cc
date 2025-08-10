// Copyright (c) 2025, StepCast Team. All rights reserved.

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
#include "pinned_memory_pool.h"

#include "absl/log/log.h"
#include "core/common/cuda_api.h"

namespace stepcast::store {

PinnedMemoryPool::PinnedMemoryPool(size_t total_size, size_t chunk_size) : chunk_size_(chunk_size) {
  // Ensure chunk_size is aligned for DIRECT_IO support
  if (chunk_size_ % kDirectIOAlignment != 0) {
    size_t aligned_chunk_size = ((chunk_size_ + kDirectIOAlignment - 1) / kDirectIOAlignment) * kDirectIOAlignment;
    LOG(WARNING) << "PinnedMemoryPool: chunk_size " << chunk_size_ << " is not aligned to " << kDirectIOAlignment
                 << " bytes, rounding up to " << aligned_chunk_size;
    chunk_size_ = aligned_chunk_size;
  }

  const size_t num_buffers = (total_size + chunk_size_ - 1) / chunk_size_;
  if (num_buffers * chunk_size_ != total_size) {
    LOG(WARNING) << "PinnedMemoryPool: total_size " << total_size << " is not a multiple of aligned chunk_size "
                 << chunk_size_ << ", actual pool size will be " << (num_buffers * chunk_size_);
  }
  VLOG(1) << "Creating PinnedMemoryPool with " << num_buffers << " buffers of " << chunk_size_ << " bytes"
          << " (aligned for DIRECT_IO)";

  for (size_t i = 0; i < num_buffers; ++i) {
    // Allocate with page alignment for optimal performance
    char* buffer = static_cast<char*>(aligned_alloc(kMemoryAlignment, chunk_size_));
    if (buffer == nullptr) {
      LOG(FATAL) << "aligned_alloc failed for size " << chunk_size_;
    }

    auto register_status = cuda::host_register(buffer, chunk_size_, cudaHostRegisterDefault);
    if (!register_status.ok()) {
      LOG(FATAL) << "PinnedMemoryPool: cudaHostRegister failed: " << register_status.message();
    }
    pool_.insert(buffer);
    free_list_.insert(buffer);
  }
}

PinnedMemoryPool::~PinnedMemoryPool() {
  for (char* buffer : pool_) {
    auto unregister_status = cuda::host_unregister(buffer);
    if (!unregister_status.ok()) {
      LOG(ERROR) << "Failed to unregister CUDA host memory: " << unregister_status.message();
    }
    free(buffer);
  }
}

int PinnedMemoryPool::allocate(size_t size, std::vector<char*>& buffers, const std::chrono::milliseconds& timeout) {
  if (size == 0) {
    LOG(ERROR) << "PinnedMemoryPool Allocate size is zero";
    return -1;
  }

  const int num_buffers_needed = (size + chunk_size_ - 1) / chunk_size_;

  // Use timeout logic if timeout is specified (non-zero), otherwise use immediate check
  if (timeout.count() > 0) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait with timeout for enough memory to become available
    auto end_time = std::chrono::steady_clock::now() + timeout;
    bool memory_available = cv_.wait_until(lock, end_time, [this, num_buffers_needed]() {
      return num_buffers_needed <= static_cast<int>(free_list_.size());
    });

    if (!memory_available) {
      LOG(WARNING) << "PinnedMemoryPool allocation timed out after " << timeout.count() << "ms (" << free_list_.size()
                   << " buffers available, " << num_buffers_needed << " buffers needed)";
      return num_buffers_needed - free_list_.size();
    }

    // Perform the allocation
    buffers.clear();
    buffers.resize(num_buffers_needed);
    auto it = free_list_.begin();
    for (size_t i = 0; i < static_cast<size_t>(num_buffers_needed); ++i) {
      buffers[i] = *it;
      it = free_list_.erase(it);
    }
  } else {
    // Original immediate allocation logic for backward compatibility
    const std::lock_guard<std::mutex> lock(mutex_);

    if (num_buffers_needed > static_cast<int>(free_list_.size())) {
      LOG(ERROR) << "PinnedMemoryPool out of memory (" << free_list_.size() << " buffers available, "
                 << num_buffers_needed << " buffers needed)";
      return num_buffers_needed - free_list_.size();
    }

    buffers.clear();
    buffers.resize(num_buffers_needed);
    auto it = free_list_.begin();
    for (size_t i = 0; i < static_cast<size_t>(num_buffers_needed); ++i) {
      buffers[i] = *it;
      it = free_list_.erase(it);
    }
  }

  VLOG(1) << "PinnedMemoryPool Allocate " << buffers.size() << " buffers"
          << " free buffers " << free_list_.size() << " total buffers " << pool_.size()
          << (timeout.count() > 0 ? " (with timeout)" : "");

  return 0; // Success
}

int PinnedMemoryPool::deallocate(std::vector<char*>& buffers) {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (char* buffer : buffers) {
    if (pool_.find(buffer) == pool_.end()) {
      LOG(ERROR) << "Buffer not found in pool";
      return -1;
    }
    if (free_list_.find(buffer) != free_list_.end()) {
      LOG(ERROR) << "Buffer already in free list";
      return -1;
    }
    free_list_.insert(buffer);
  }

  // Notify waiting threads that memory has become available
  cv_.notify_all();

  VLOG(1) << "Deallocated " << buffers.size() << " buffers";
  return 0; // Success
}

size_t PinnedMemoryPool::get_available_size() const {
  return free_list_.size() * chunk_size_;
}

} // namespace stepcast::store
