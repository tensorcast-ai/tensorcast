// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <algorithm>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gsl/pointers"

#include "core/common/cuda_api.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/communicator/engine/memory_stager.h"
#include "core/communicator/transport/partition_tensor.h"

namespace tensorcast::communicator::engine {

// GPU MemoryStager implementation using a StreamingPinnedBuffer for D2H staging.
class GpuNetStager : public MemoryStager {
 public:
  GpuNetStager(size_t chunk_size, size_t num_buffers, std::shared_ptr<common::memory::PinnedMemoryPool> pool)
      : chunk_size_(chunk_size), num_buffers_(num_buffers), pool_(std::move(pool)) {}

  absl::StatusOr<void*> stage(
      const std::shared_ptr<communicator::transport::PartitionTensor>& tensor,
      uint64_t offset,
      uint64_t bytes) override {
    if (bytes == 0)
      return absl::InvalidArgumentError("GpuNetStager: zero bytes");
    if (offset + bytes > tensor->get_bytes()) {
      return absl::InvalidArgumentError("GpuNetStager: staging beyond tensor bounds");
    }
    if (!streaming_) {
      if (!pool_)
        return absl::FailedPreconditionError("GpuNetStager: no pinned pool");
      streaming_ = std::make_unique<common::memory::StreamingPinnedBuffer>(num_buffers_, chunk_size_, pool_);
      auto st = streaming_->initialize();
      if (!st.ok())
        return st;
    }
    auto slot_or = streaming_->get_free_chunk();
    if (!slot_or.ok())
      return slot_or.status();
    int slot = *slot_or;
    char* dst = streaming_->get_chunk_ptr(slot);
    if (!dst)
      return absl::InternalError("GpuNetStager: invalid slot pointer");

    // Perform synchronous D2H copy into pinned buffer
    int device_id = tensor->get_device_id();
    device_id = std::max(device_id, 0);
    auto guard = tensorcast::cuda::set_device(device_id);
    if (!guard.ok()) {
      // Return the slot on failure
      absl::Status _ret = streaming_->return_chunk(slot);
      if (!_ret.ok()) {
        LOG(WARNING) << "GpuNetStager: return_chunk failed after set_device error: " << _ret;
      }
      return guard;
    }
    void* src = reinterpret_cast<void*>(tensor->get_uint64_addr() + offset);
    auto copy_st = tensorcast::cuda::memcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (!copy_st.ok()) {
      absl::Status _ret = streaming_->return_chunk(slot);
      if (!_ret.ok()) {
        LOG(WARNING) << "GpuNetStager: return_chunk failed after memcpy error: " << _ret;
      }
      return copy_st;
    }

    void* host_ptr = static_cast<void*>(dst);
    {
      absl::MutexLock lk(&mu_);
      ptr_to_slot_[host_ptr] = slot;
    }
    return host_ptr;
  }

  absl::Status release_staged_buffer(gsl::not_null<void*> host_ptr) override {
    int slot = -1;
    {
      absl::MutexLock lk(&mu_);
      auto it = ptr_to_slot_.find(host_ptr.get());
      if (it == ptr_to_slot_.end()) {
        return absl::NotFoundError("GpuNetStager: host ptr not found");
      }
      slot = it->second;
      ptr_to_slot_.erase(it);
    }
    return streaming_->return_chunk(slot);
  }

  size_t get_chunk_size() const override {
    return chunk_size_;
  }

  size_t get_num_buffers() const override {
    return num_buffers_;
  }

 private:
  size_t chunk_size_;
  size_t num_buffers_;
  std::shared_ptr<common::memory::PinnedMemoryPool> pool_;
  std::unique_ptr<common::memory::StreamingPinnedBuffer> streaming_;
  mutable absl::Mutex mu_;
  std::unordered_map<void*, int> ptr_to_slot_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::communicator::engine
