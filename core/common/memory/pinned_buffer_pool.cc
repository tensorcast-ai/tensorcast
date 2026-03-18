// Copyright (c) 2025-2026, TensorCast Team.

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
#include "pinned_buffer_pool.h"

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::common::memory {

namespace {

std::vector<unsigned long> build_nodemask_single(int node) {
  constexpr size_t kBitsPerWord = sizeof(unsigned long) * 8;
  if (node < 0) {
    return {};
  }
  const size_t words = (static_cast<size_t>(node) / kBitsPerWord) + 1;
  std::vector<unsigned long> mask(words, 0);
  mask[static_cast<size_t>(node) / kBitsPerWord] |= (1ul << (static_cast<size_t>(node) % kBitsPerWord));
  return mask;
}

void prefault_pages(volatile char* base, size_t bytes) {
  // Touch one byte per page to ensure physical pages are mapped under the current policy.
  constexpr size_t kPage = 4096;
  for (size_t off = 0; off < bytes; off += kPage) {
    base[off] = 0;
  }
  if (bytes > 0) {
    base[bytes - 1] = 0;
  }
}

void maybe_bind_to_numa_node(char* base, size_t bytes, int numa_node, bool prefault) {
  if (numa_node < 0 || bytes == 0 || base == nullptr) {
    return;
  }
  auto mask = build_nodemask_single(numa_node);
  if (mask.empty()) {
    return;
  }
  const unsigned long maxnode = static_cast<unsigned long>(mask.size() * sizeof(unsigned long) * 8);
  const long rc = ::syscall(SYS_mbind, base, bytes, MPOL_BIND, mask.data(), maxnode, static_cast<unsigned long>(0));
  if (rc != 0) {
    PLOG(WARNING) << "PinnedBufferPool: mbind(MPOL_BIND) failed (numa_node=" << numa_node << ")";
    return;
  }
  if (prefault) {
    prefault_pages(base, bytes);
  }
}

} // namespace

PinnedBufferPool::PinnedBufferPool(size_t total_size, size_t chunk_size)
    : PinnedBufferPool(total_size, chunk_size, /*name=*/std::string()) {}

PinnedBufferPool::PinnedBufferPool(size_t total_size, size_t chunk_size, std::string name)
    : PinnedBufferPool(total_size, chunk_size, Options{.name = std::move(name)}) {}

PinnedBufferPool::PinnedBufferPool(size_t total_size, size_t chunk_size, Options options)
    : chunk_size_(chunk_size), name_(std::move(options.name)) {
  if (chunk_size_ == 0) {
    LOG(FATAL) << "PinnedBufferPool: chunk_size must be > 0";
  }
  if (total_size == 0) {
    LOG(FATAL) << "PinnedBufferPool: total_size must be > 0";
  }
  // Ensure chunk_size is aligned for DIRECT_IO support
  if (chunk_size_ % kDirectIOAlignment != 0) {
    size_t aligned_chunk_size = ((chunk_size_ + kDirectIOAlignment - 1) / kDirectIOAlignment) * kDirectIOAlignment;
    LOG(WARNING) << "PinnedBufferPool" << (name_.empty() ? "" : absl::StrCat("[name=", name_, "]")) << ": chunk_size "
                 << chunk_size_ << " is not aligned to " << kDirectIOAlignment << " bytes, rounding up to "
                 << aligned_chunk_size;
    chunk_size_ = aligned_chunk_size;
  }

  const size_t num_buffers = (total_size + chunk_size_ - 1) / chunk_size_;
  if (num_buffers * chunk_size_ != total_size) {
    LOG(WARNING) << "PinnedBufferPool" << (name_.empty() ? "" : absl::StrCat("[name=", name_, "]")) << ": total_size "
                 << total_size << " is not a multiple of aligned chunk_size " << chunk_size_
                 << ", actual pool size will be " << (num_buffers * chunk_size_);
  }
  VLOG(1) << "Creating PinnedBufferPool with " << num_buffers << " buffers of " << chunk_size_ << " bytes"
          << " (aligned for DIRECT_IO)";

  const size_t slab_bytes = num_buffers * chunk_size_;
  char* slab = static_cast<char*>(aligned_alloc(kMemoryAlignment, slab_bytes));
  if (slab == nullptr) {
    LOG(FATAL) << "PinnedBufferPool: aligned_alloc failed for slab size " << slab_bytes;
  }

  // Best-effort NUMA placement. Must happen before cudaHostRegister, otherwise
  // the driver may fault/lock pages under the default policy.
  maybe_bind_to_numa_node(slab, slab_bytes, options.numa_node, options.prefault);

  if (options.register_on_create) {
    auto register_status = cuda::host_register(slab, slab_bytes, cudaHostRegisterDefault);
    if (!register_status.ok()) {
      LOG(FATAL) << "PinnedBufferPool: cudaHostRegister failed: " << register_status.message();
    }
  }

  {
    const std::lock_guard<std::mutex> lock(mutex_);
    slabs_.push_back(Slab{gsl::not_null<char*>{slab}, slab_bytes});
    for (size_t i = 0; i < num_buffers; ++i) {
      char* buffer = slab + (i * chunk_size_);
      pool_.insert(buffer);
      free_list_.insert(buffer);
    }
    host_registered_ = options.register_on_create;
  }
}

PinnedBufferPool::~PinnedBufferPool() {
  std::vector<Slab> slabs_copy;
  bool host_registered = false;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    slabs_copy = slabs_;
    host_registered = host_registered_;
    host_registered_ = false;
  }
  if (host_registered) {
    for (const auto& slab : slabs_copy) {
      auto unregister_status = cuda::host_unregister(slab.base.get());
      if (!unregister_status.ok()) {
        LOG(ERROR) << "Failed to unregister CUDA host memory: " << unregister_status.message();
      }
    }
  }
  for (const auto& slab : slabs_copy) {
    free(slab.base.get());
  }
}

int PinnedBufferPool::allocate(
    size_t size,
    std::vector<char*>& buffers,
    const std::chrono::milliseconds& timeout,
    std::string_view request_context) {
  const std::string_view context = request_context.empty() ? std::string_view("<unspecified>") : request_context;
  if (size == 0) {
    LOG(ERROR) << "PinnedBufferPool" << (name_.empty() ? "" : absl::StrCat("[name=", name_, "]"))
               << " allocate: size is zero"
               << " request_context=" << context;
    return -1;
  }
  if (!is_host_registered()) {
    LOG(ERROR) << "PinnedBufferPool" << (name_.empty() ? "" : absl::StrCat("[name=", name_, "]"))
               << " allocate rejected before host registration completed"
               << " request_context=" << context;
    return -1;
  }

  const int num_buffers_needed = (size + chunk_size_ - 1) / chunk_size_;

  // Use timeout logic if timeout is specified (non-zero), otherwise use immediate check
  if (timeout.count() > 0) {
    std::unique_lock<std::mutex> lock(mutex_);
    ++waiters_;

    // Wait with timeout for enough memory to become available. Wake periodically so we can emit
    // diagnosable "still waiting" warnings (no-progress) without relying on notifications.
    const auto wait_start = std::chrono::steady_clock::now();
    auto next_log = wait_start + std::chrono::seconds(5);
    const auto end_time = wait_start + timeout;

    bool memory_available = false;
    while (true) {
      const auto now = std::chrono::steady_clock::now();
      memory_available = num_buffers_needed <= static_cast<int>(free_list_.size());
      if (memory_available) {
        break;
      }
      if (now >= end_time) {
        break;
      }
      if (now >= next_log) {
        const auto waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - wait_start).count();
        LOG(WARNING) << "PinnedBufferPool" << (name_.empty() ? "" : absl::StrCat("[name=", name_, "]"))
                     << " allocate still waiting: request_context=" << context << " waited_ms=" << waited_ms
                     << " requested_bytes=" << size << " need_slices=" << num_buffers_needed
                     << " free_slices=" << free_list_.size() << " total_slices=" << pool_.size()
                     << " in_use_slices=" << (pool_.size() - free_list_.size()) << " waiters=" << waiters_;
        next_log = now + std::chrono::seconds(5);
      }
      const auto wake_deadline = std::min(end_time, now + std::chrono::seconds(1));
      cv_.wait_until(lock, wake_deadline);
    }
    --waiters_;

    if (!memory_available) {
      ++acquire_timeouts_total_;
      ++budget_exhausted_total_;
      const auto waited_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - wait_start).count();
      LOG(WARNING) << "PinnedBufferPool" << (name_.empty() ? "" : absl::StrCat("[name=", name_, "]"))
                   << " allocate timed out: request_context=" << context << " waited_ms=" << waited_ms
                   << " timeout_ms=" << timeout.count() << " requested_bytes=" << size
                   << " need_slices=" << num_buffers_needed << " free_slices=" << free_list_.size()
                   << " total_slices=" << pool_.size() << " in_use_slices=" << (pool_.size() - free_list_.size());
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
      ++budget_exhausted_total_;
      LOG(ERROR) << "PinnedBufferPool" << (name_.empty() ? "" : absl::StrCat("[name=", name_, "]"))
                 << " out of memory: request_context=" << context << " requested_bytes=" << size
                 << " need_slices=" << num_buffers_needed << " free_slices=" << free_list_.size()
                 << " total_slices=" << pool_.size() << " in_use_slices=" << (pool_.size() - free_list_.size());
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

  VLOG(1) << "PinnedBufferPool Allocate " << buffers.size() << " buffers"
          << " free buffers " << free_list_.size() << " total buffers " << pool_.size()
          << " request_context=" << context << (timeout.count() > 0 ? " (with timeout)" : "");

  return 0; // Success
}

int PinnedBufferPool::deallocate(std::vector<char*>& buffers) {
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

size_t PinnedBufferPool::get_available_size() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return free_list_.size() * chunk_size_;
}

size_t PinnedBufferPool::capacity_slices() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return pool_.size();
}

size_t PinnedBufferPool::free_slices() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return free_list_.size();
}

size_t PinnedBufferPool::in_use_slices() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return pool_.size() - free_list_.size();
}

size_t PinnedBufferPool::waiters() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return waiters_;
}

uint64_t PinnedBufferPool::acquire_timeouts_total() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return acquire_timeouts_total_;
}

uint64_t PinnedBufferPool::budget_exhausted_total() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return budget_exhausted_total_;
}

bool PinnedBufferPool::is_host_registered() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return host_registered_;
}

absl::Status PinnedBufferPool::register_host_memory() {
  std::vector<Slab> slabs_copy;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (host_registered_) {
      return absl::OkStatus();
    }
    slabs_copy = slabs_;
  }

  std::vector<gsl::not_null<char*>> newly_registered;
  newly_registered.reserve(slabs_copy.size());
  for (const auto& slab : slabs_copy) {
    auto register_status = cuda::host_register(slab.base.get(), slab.bytes, cudaHostRegisterDefault);
    if (!register_status.ok()) {
      for (auto ptr : newly_registered) {
        auto unregister_status = cuda::host_unregister(ptr.get());
        if (!unregister_status.ok()) {
          LOG(ERROR) << "PinnedBufferPool rollback host_unregister failed: " << unregister_status.message();
        }
      }
      return register_status;
    }
    newly_registered.push_back(slab.base);
  }

  {
    const std::lock_guard<std::mutex> lock(mutex_);
    host_registered_ = true;
  }
  return absl::OkStatus();
}

std::vector<gsl::not_null<char*>> PinnedBufferPool::list_buffers() const {
  std::vector<gsl::not_null<char*>> out;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    out.reserve(pool_.size());
    for (auto* ptr : pool_) {
      out.push_back(gsl::not_null<char*>{ptr});
    }
  }
  return out;
}

std::vector<PinnedBufferPool::Slab> PinnedBufferPool::list_slabs() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return slabs_;
}

std::optional<PinnedBufferPool::Slab> PinnedBufferPool::slab_for_ptr(gsl::not_null<void*> ptr) const {
  const auto address = reinterpret_cast<uintptr_t>(ptr.get());
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& slab : slabs_) {
    const auto base = reinterpret_cast<uintptr_t>(slab.base.get());
    const auto end = base + slab.bytes;
    if (address >= base && address < end) {
      return slab;
    }
  }
  return std::nullopt;
}

} // namespace tensorcast::common::memory
