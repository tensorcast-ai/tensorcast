// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gsl/pointers"

#include "core/common/cuda_api.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/common/memory/streaming_chunk_guard.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/communicator/engine/memory_stager.h"
#include "core/communicator/transport/partition_tensor.h"

namespace tensorcast::communicator::engine {

// Host-pinned GPU stager using a StreamingPinnedBuffer for D2H staging.
class HostPinnedGpuStager : public MemoryStager {
 public:
  HostPinnedGpuStager(size_t chunk_size, size_t num_buffers, std::shared_ptr<common::memory::PinnedBufferPool> pool)
      : chunk_size_(chunk_size), num_buffers_(num_buffers), pool_(std::move(pool)) {
    if (chunk_size_ == 0) {
      LOG(FATAL) << "HostPinnedGpuStager chunk_size must be > 0";
    }
    if (num_buffers_ == 0) {
      LOG(FATAL) << "HostPinnedGpuStager num_buffers must be > 0";
    }
    if (!pool_) {
      LOG(FATAL) << "HostPinnedGpuStager pool must not be null";
    }
  }

  absl::StatusOr<void*> stage(
      const std::shared_ptr<communicator::transport::PartitionTensor>& tensor,
      uint64_t offset,
      uint64_t bytes,
      StageMode mode) override {
    if (bytes == 0 || bytes > chunk_size_) {
      return absl::InvalidArgumentError("HostPinnedGpuStager: bytes must be within (0, chunk_size]");
    }
    if (offset + bytes > tensor->get_bytes()) {
      return absl::InvalidArgumentError("HostPinnedGpuStager: staging beyond tensor bounds");
    }
    if (!streaming_) {
      if (!pool_)
        return absl::FailedPreconditionError("HostPinnedGpuStager: no pinned pool");
      streaming_ = std::make_unique<common::memory::StreamingPinnedBuffer>(num_buffers_, chunk_size_, pool_);
      auto st = streaming_->initialize();
      if (!st.ok())
        return st;
    }
    common::memory::StreamingChunkGuard slot_guard(streaming_.get());
    absl::StatusOr<char*> dst_or = mode == StageMode::kBlocking ? slot_guard.acquire() : slot_guard.try_acquire();
    if (!dst_or.ok()) {
      return dst_or.status();
    }
    char* dst = *dst_or;

    // Perform synchronous D2H copy into pinned buffer
    int device_id = tensor->get_device_id();
    device_id = std::max(device_id, 0);
    auto guard = cuda::set_device(device_id);
    if (!guard.ok()) {
      return guard;
    }
    void* src = reinterpret_cast<void*>(tensor->get_uint64_addr() + offset);
    auto copy_st = cuda::memcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (!copy_st.ok()) {
      return copy_st;
    }

    // Derive a stable chunk index for diagnostics and slot tracking.
    const size_t chunk_index = chunk_size_ > 0 ? static_cast<size_t>(offset / chunk_size_) : 0;
    auto promote_status = slot_guard.promote_to_consumer(chunk_index, bytes);
    if (!promote_status.ok()) {
      return promote_status;
    }
    const int promoted_slot = slot_guard.release_for_async();

    void* exposed_ptr = static_cast<void*>(dst);
    {
      absl::MutexLock lk(&mu_);
      staged_slots_[exposed_ptr] = SlotMetadata{
          .slot_id = promoted_slot,
          .bytes_in_chunk = static_cast<size_t>(bytes),
          .tensor_offset = offset,
          .chunk_index = chunk_index,
      };
    }
    return exposed_ptr;
  }

  absl::Status release_staged_buffer(gsl::not_null<void*> exposed_ptr) override {
    SlotMetadata metadata;
    {
      absl::MutexLock lk(&mu_);
      auto it = staged_slots_.find(exposed_ptr.get());
      if (it == staged_slots_.end()) {
        return absl::NotFoundError("HostPinnedGpuStager: exposed ptr not found");
      }
      metadata = it->second;
      staged_slots_.erase(it);
    }
    return streaming_->return_chunk(metadata.slot_id);
  }

  std::optional<MrSlab> mr_slab_for_ptr(gsl::not_null<void*> exposed_ptr) const override {
    auto slab = pool_->slab_for_ptr(exposed_ptr);
    if (!slab.has_value()) {
      return std::nullopt;
    }
    return MrSlab{slab->base.get(), slab->bytes};
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
  std::shared_ptr<common::memory::PinnedBufferPool> pool_;
  std::unique_ptr<common::memory::StreamingPinnedBuffer> streaming_;
  mutable absl::Mutex mu_;

  struct SlotMetadata {
    int slot_id;
    size_t bytes_in_chunk;
    uint64_t tensor_offset;
    size_t chunk_index;
  };

  std::unordered_map<void*, SlotMetadata> staged_slots_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::communicator::engine
