// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/replica/transfer_helpers.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "absl/log/absl_check.h"
#include "core/common/memory/memory_location.h"
#include "core/store/replica/chunk_meta.h"

namespace stepcast::store {

absl::Status perform_copy_cpu_to_gpu_streaming(
    const std::string& artifact_id,
    uint32_t device_id,
    const std::shared_ptr<StreamingPinnedBuffer>& streaming_buf,
    void* gpu_ptr,
    size_t total_size,
    cudaStream_t stream,
    void* dvmp_base,
    const std::shared_ptr<memory::DistributedVirtualMemoryPool>& dvmp,
    const std::shared_ptr<ReplicaMemoryCoordinator>& uma,
    const ReplicaKey& ikey) {
  // Required components must be present – enforce via CHECKKs
  ABSL_CHECK(streaming_buf) << "StreamingPinnedBuffer must not be null";
  ABSL_CHECK(gpu_ptr) << "GPU destination pointer must not be null";
  ABSL_CHECK_GT(total_size, 0) << "Total size must be positive";

  const size_t dvmp_chunk = memory::DistributedVirtualMemoryPool::kDefaultChunkSize;
  const size_t copy_chunk = streaming_buf->chunk_size();

  auto device_status = cuda::set_device(device_id);
  if (!device_status.ok()) {
    return device_status;
  }

  // Chunk-aware copy: iterate DVMP chunk ranges so that UMA can lock/unlock
  // per-chunk and update states correctly.
  const size_t num_dvmp_chunks = (total_size + dvmp_chunk - 1) / dvmp_chunk;
  for (size_t dvmp_idx = 0; dvmp_idx < num_dvmp_chunks; ++dvmp_idx) {
    const size_t dvmp_off = dvmp_idx * dvmp_chunk;
    const size_t this_len = std::min(dvmp_chunk, total_size - dvmp_off);

    // UMA lock this DVMP chunk for transfer (CPU -> GPU)
    if (uma) {
      std::vector<uint32_t> one{static_cast<uint32_t>(dvmp_idx)};
      auto st = uma->lock_chunks_for_transfer(ikey, MemoryLocation::PAGEABLE_CPU, MemoryLocation::GPU, one);
      if (!st.ok()) {
        return st;
      }
    }

    size_t copied = 0;
    while (copied < this_len) {
      size_t step = std::min(copy_chunk, this_len - copied);
      // Acquire a free chunk slot
      auto slot_or = streaming_buf->get_free_chunk();
      if (!slot_or.ok()) {
        return slot_or.status();
      }
      int slot_id = *slot_or;
      char* host_ptr = streaming_buf->get_chunk_ptr(slot_id);
      if (host_ptr == nullptr) {
        ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
        return absl::InternalError("Failed to get chunk pointer from streaming buffer");
      }

      // Copy from DVMP region to pinned chunk
      void* src_host = static_cast<char*>(dvmp_base) + dvmp_off + copied;
      std::memcpy(host_ptr, src_host, step);

      // Async copy H2D
      void* dst_device = static_cast<char*>(gpu_ptr) + dvmp_off + copied;
      auto memcpy_status = cuda::memcpy_async(dst_device, host_ptr, step, cudaMemcpyHostToDevice, stream);
      if (!memcpy_status.ok()) {
        ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
        return memcpy_status;
      }

      // Synchronize to ensure chunk can be reused safely
      auto sync_status = cuda::stream_synchronize(stream);
      if (!sync_status.ok()) {
        ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
        return sync_status;
      }

      // Return chunk to buffer
      ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
      copied += step;
    }

    // UMA update: mark DVMP chunk as COPIED_GPU which triggers DVMP unlock
    if (uma) {
      std::vector<uint32_t> one{static_cast<uint32_t>(dvmp_idx)};
      auto st = uma->update_chunk_states(ikey, MemoryLocation::GPU, one, ChunkState::COPIED_GPU, device_id);
      if (!st.ok()) {
        // Best-effort unlock on failure to avoid holding LOCKED_TX
        (void)dvmp->unlock_chunks(artifact_id, one, /*copied_gpu=*/true);
        return st;
      }
    }
  }

  return absl::OkStatus();
}

absl::Status perform_copy_gpu_to_cpu_streaming(
    const std::string& artifact_id,
    uint32_t device_id,
    const std::shared_ptr<StreamingPinnedBuffer>& streaming_buf,
    void* gpu_ptr,
    size_t total_size,
    cudaStream_t stream,
    void* dvmp_base,
    const std::shared_ptr<memory::DistributedVirtualMemoryPool>& dvmp) {
  ABSL_CHECK(streaming_buf) << "StreamingPinnedBuffer must not be null";
  ABSL_CHECK(gpu_ptr) << "GPU source pointer must not be null";
  ABSL_CHECK_GT(total_size, 0) << "Total size must be positive";
  ABSL_CHECK(dvmp_base) << "DVMP base pointer must not be null";

  size_t chunk_size = streaming_buf->chunk_size();
  size_t offset = 0;

  auto device_status = cuda::set_device(device_id);
  if (!device_status.ok()) {
    return device_status;
  }

  while (offset < total_size) {
    size_t current_chunk_size = std::min(chunk_size, total_size - offset);

    // Acquire a free chunk slot
    auto slot_or = streaming_buf->get_free_chunk();
    if (!slot_or.ok()) {
      return slot_or.status();
    }
    int slot_id = *slot_or;
    char* host_ptr = streaming_buf->get_chunk_ptr(slot_id);
    if (host_ptr == nullptr) {
      ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
      return absl::InternalError("Failed to get chunk pointer from streaming buffer");
    }

    // Async copy from GPU to pinned host chunk
    void* src_device = static_cast<char*>(gpu_ptr) + offset;
    auto memcpy_status = cuda::memcpy_async(host_ptr, src_device, current_chunk_size, cudaMemcpyDeviceToHost, stream);
    if (!memcpy_status.ok()) {
      ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
      return memcpy_status;
    }

    // Synchronize to ensure chunk hosts valid data before copying to DVMP
    auto sync_status = cuda::stream_synchronize(stream);
    if (!sync_status.ok()) {
      ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
      return sync_status;
    }

    // Write to DVMP using write_at to preserve metadata and residency
    // DVMP write_at updates CPU metadata visibility (at least HOT) and last_touch_s
    auto write_status = dvmp->write_at(artifact_id, offset, host_ptr, current_chunk_size);
    if (!write_status.ok()) {
      ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
      return write_status;
    }

    // Return chunk to buffer
    ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));

    offset += current_chunk_size;
  }

  return absl::OkStatus();
}

} // namespace stepcast::store