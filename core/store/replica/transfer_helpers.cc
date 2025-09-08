// Copyright (c) 2025, TensorCast Team.

#include "core/store/replica/transfer_helpers.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "core/common/async_copy_manager.h"
#include "core/common/memory/memory_location.h"
#include "core/store/replica/chunk_meta.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"

namespace tensorcast::store::replica {

absl::Status perform_copy_cpu_to_gpu_streaming(
    const std::string& artifact_id,
    uint32_t device_id,
    const std::shared_ptr<common::memory::StreamingPinnedBuffer>& streaming_buf,
    void* gpu_ptr,
    size_t total_size,
    cudaStream_t stream,
    void* dvmp_base,
    const std::shared_ptr<common::memory::DistributedVirtualMemoryPool>& dvmp,
    const std::shared_ptr<ReplicaMemoryCoordinator>& uma,
    const loading::ReplicaKey& ikey) {
  // Required components must be present – enforce via CHECKKs
  ABSL_CHECK(streaming_buf) << "StreamingPinnedBuffer must not be null";
  ABSL_CHECK(gpu_ptr) << "GPU destination pointer must not be null";
  ABSL_CHECK_GT(total_size, 0) << "Total size must be positive";

  const size_t dvmp_chunk = common::memory::DistributedVirtualMemoryPool::kDefaultChunkSize;
  const size_t copy_chunk = streaming_buf->chunk_size();

  // Collect the first error encountered in async UMA progression callbacks
  auto first_error = std::make_shared<absl::Status>(absl::OkStatus());
  auto first_error_mu = std::make_shared<absl::Mutex>();

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
      auto st = uma->lock_chunks_for_transfer(
          ikey, common::memory::MemoryLocation::PAGEABLE_CPU, common::memory::MemoryLocation::GPU, one);
      if (!st.ok()) {
        return st;
      }
    }

    // Aggregate UMA advancement per DVMP block via callbacks
    const int num_subchunks = static_cast<int>((this_len + copy_chunk - 1) / copy_chunk);
    auto remaining = std::make_shared<std::atomic<int>>(num_subchunks);
    size_t copied = 0;
    common::CopyHandle last_hdl;
    bool last_set = false;
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

      // Async copy H2D via ACM; return SPB slot in host callback
      void* dst_device = static_cast<char*>(gpu_ptr) + dvmp_off + copied;
      common::HostRegion h{.base = host_ptr, .length = step, .pinned = true};
      common::DeviceRegion d{.device_id = static_cast<int>(device_id), .dev_ptr = dst_device, .length = step};
      common::CopyOptions opts{
          .tracing_stage = "H2D/Copy",
          .callbacks = {
              .on_copy_done = [streaming_buf,
                               slot_id,
                               remaining,
                               uma,
                               dvmp,
                               artifact_id,
                               dvmp_idx,
                               device_id,
                               ikey,
                               first_error,
                               first_error_mu]() {
                // Return chunk when GPU DMA completes
                (void)streaming_buf->return_chunk(slot_id);
                // When all sub-chunks of this DVMP block complete, advance UMA state
                if (remaining->fetch_sub(1) == 1) {
                  std::vector<uint32_t> one{static_cast<uint32_t>(dvmp_idx)};
                  auto st = uma ? uma->update_chunk_states(
                                      ikey, common::memory::MemoryLocation::GPU, one, ChunkState::COPIED_GPU, device_id)
                                : absl::OkStatus();
                  if (!st.ok()) {
                    // Best-effort DVMP unlock to avoid holding LOCKED_TX
                    (void)dvmp->unlock_chunks(artifact_id, one, /*copied_gpu=*/true);
                    absl::MutexLock lk(first_error_mu.get());
                    if (first_error->ok()) {
                      *first_error = st;
                    }
                  }
                }
              }}};
      auto hdl_or = common::AsyncCopyManager::instance().submit_h2d(h, d, stream, opts);
      if (!hdl_or.ok()) {
        // Return the slot immediately on submission failure
        ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
        return hdl_or.status();
      }
      last_hdl = std::move(*hdl_or);
      last_set = true;

      copied += step;
    }
    // Ensure callbacks for the last submitted chunk are executed
    if (last_set) {
      auto wst = last_hdl.wait();
      if (!wst.ok()) {
        return wst;
      }
      // Propagate any UMA callback errors
      absl::MutexLock lk(first_error_mu.get());
      if (!first_error->ok()) {
        return *first_error;
      }
    }
  }

  return absl::OkStatus();
}

absl::Status perform_copy_gpu_to_cpu_streaming(
    const std::string& artifact_id,
    uint32_t device_id,
    const std::shared_ptr<common::memory::StreamingPinnedBuffer>& streaming_buf,
    void* gpu_ptr,
    size_t total_size,
    cudaStream_t stream,
    void* dvmp_base,
    const std::shared_ptr<common::memory::DistributedVirtualMemoryPool>& dvmp) {
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

    // Async D2H via ACM; upon completion, write into DVMP and return the slot
    void* src_device = static_cast<char*>(gpu_ptr) + offset;
    common::DeviceRegion s{
        .device_id = static_cast<int>(device_id), .dev_ptr = src_device, .length = current_chunk_size};
    common::HostRegion h{.base = host_ptr, .length = current_chunk_size, .pinned = true};
    const uint64_t chunk_offset = offset;
    common::CopyOptions opts{
        .tracing_stage = "D2H/Copy",
        .callbacks = {
            .on_copy_done = [dvmp, artifact_id, streaming_buf, slot_id, host_ptr, chunk_offset, current_chunk_size]() {
              // Write to DVMP then return slot
              (void)dvmp->write_at(artifact_id, chunk_offset, host_ptr, current_chunk_size);
              (void)streaming_buf->return_chunk(slot_id);
            }}};
    auto hdl_or = common::AsyncCopyManager::instance().submit_d2h(s, h, stream, opts);
    if (!hdl_or.ok()) {
      ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
      return hdl_or.status();
    }

    offset += current_chunk_size;
  }

  return absl::OkStatus();
}

} // namespace tensorcast::store::replica
