// Copyright (c) 2025, TensorCast Team.

#include "core/store/replica/transfer_helpers.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "core/common/async_copy_manager.h"
#include "core/common/const/granularity.h"
#include "core/common/memory/memory_location.h"
#include "core/common/memory/streaming_chunk_guard.h"
#include "core/store/replica/chunk_state.h"

namespace tensorcast::store::replica {

absl::Status perform_copy_cpu_to_gpu_streaming(
    const std::string& artifact_id,
    uint32_t device_id,
    const std::shared_ptr<common::memory::StreamingPinnedBuffer>& streaming_buf,
    gsl::not_null<void*> gpu_ptr,
    size_t total_size,
    gsl::not_null<void*> vs_base,
    const std::shared_ptr<UnifiedMemoryAuthority>& uma,
    const loading::ReplicaKey& ikey) {
  // Required components must be present – enforce via CHECKKs
  ABSL_CHECK(streaming_buf) << "StreamingPinnedBuffer must not be null";
  ABSL_CHECK_GT(total_size, 0) << "Total size must be positive";

  size_t va_chunk = 0;
  if (uma) {
    auto layout_or = uma->get_layout(ikey);
    if (layout_or.ok() && layout_or->artifact_chunk_bytes > 0) {
      va_chunk = layout_or->artifact_chunk_bytes;
    }
  }
  if (va_chunk == 0) {
    va_chunk = common::consts::kArtifactChunkDefault;
  }
  const size_t copy_chunk = streaming_buf->chunk_size();

  // No UMA per-chunk updates here; UMA ledger is handled via plan/commit at higher layers.

  auto device_status = cuda::set_device(device_id);
  if (!device_status.ok()) {
    return device_status;
  }

  // Chunk-aware copy: iterate VS chunk ranges; UMA is authoritative and
  // we do not rely on VS locks in final cutover.
  // Obtain an optional direct-write grant to pin CPU VA during transfer.
  uint64_t grant_base_addr = 0;
  std::shared_ptr<void> grant_keepalive;
  if (uma) {
    auto grant_or = uma->grant_direct_write(ikey, {store::VaRange{0, static_cast<uint64_t>(total_size)}});
    if (grant_or.ok() && !grant_or->windows.empty()) {
      grant_base_addr = grant_or->windows[0].local_addr;
      grant_keepalive = grant_or->keepalive;
    }
  }
  const size_t num_va_chunks = (total_size + va_chunk - 1) / va_chunk;
  const size_t inflight_limit = streaming_buf->capacity();
  ABSL_CHECK_GT(inflight_limit, 0u) << "StreamingPinnedBuffer must have positive capacity";

  struct InFlightCopy {
    common::CopyHandle handle;
    int slot_id;
  };

  std::deque<InFlightCopy> inflight;
  absl::Status first_error = absl::OkStatus();

  auto complete_front = [&]() {
    InFlightCopy entry = std::move(inflight.front());
    inflight.pop_front();
    absl::Status st = entry.handle.wait();
    absl::Status rc = streaming_buf->return_chunk(entry.slot_id);
    if (!rc.ok()) {
      LOG(WARNING) << "StreamingPinnedBuffer::return_chunk failed slot=" << entry.slot_id << ": " << rc;
      if (first_error.ok()) {
        first_error = rc;
      }
    }
    if (!st.ok()) {
      LOG(ERROR) << "AsyncCopyManager::submit_h2d failed slot=" << entry.slot_id << ": " << st;
      if (first_error.ok()) {
        first_error = st;
      }
    }
  };

  for (size_t va_idx = 0; va_idx < num_va_chunks; ++va_idx) {
    if (!first_error.ok()) {
      break;
    }
    const size_t va_off = va_idx * va_chunk;
    const size_t this_len = std::min(va_chunk, total_size - va_off);

    // No UMA/VS locking in final cutover. UMA is authoritative; residency is
    // guaranteed by direct-write grant when available (kept by grant_keepalive).

    size_t copied = 0;
    while (copied < this_len) {
      if (!first_error.ok()) {
        break;
      }
      size_t step = std::min(copy_chunk, this_len - copied);
      common::memory::StreamingChunkGuard guard(streaming_buf);
      auto host_ptr_or = guard.acquire();
      if (!host_ptr_or.ok()) {
        first_error = host_ptr_or.status();
        break;
      }
      char* host_ptr = *host_ptr_or;

      // Copy from CPU VA (prefer UMA grant base if available) to pinned chunk
      void* src_host = grant_base_addr != 0 ? reinterpret_cast<void*>(grant_base_addr + va_off + copied)
                                            : static_cast<char*>(vs_base.get()) + va_off + copied;
      std::memcpy(host_ptr, src_host, step);

      const size_t chunk_index = (va_off + copied) / copy_chunk;
      auto promote_status = guard.promote_to_consumer(chunk_index, step);
      if (!promote_status.ok()) {
        first_error = promote_status;
        break;
      }
      const int slot_id = guard.release_for_async();

      // Async copy H2D via ACM; return the SPB slot only after the copy handle completes.
      void* dst_device = static_cast<char*>(gpu_ptr.get()) + va_off + copied;
      common::HostRegion h{.base = host_ptr, .length = step, .pinned = true};
      common::DeviceRegion d{.device_id = static_cast<int>(device_id), .dev_ptr = dst_device, .length = step};
      common::CopyOptions opts{.tracing_stage = "H2D/Copy"};
      auto hdl_or = common::AsyncCopyManager::instance().submit_h2d(h, d, opts);
      if (!hdl_or.ok()) {
        absl::Status rc = streaming_buf->return_chunk(slot_id);
        if (!rc.ok()) {
          LOG(WARNING) << "StreamingPinnedBuffer::return_chunk failed after submit_h2d error slot=" << slot_id << ": "
                       << rc;
        }
        first_error = hdl_or.status();
        break;
      }
      inflight.push_back(InFlightCopy{.handle = std::move(*hdl_or), .slot_id = slot_id});
      if (inflight.size() >= inflight_limit) {
        complete_front();
      }

      copied += step;
    }
  }

  while (!inflight.empty()) {
    complete_front();
  }

  return first_error;
}

absl::Status perform_copy_gpu_to_cpu_streaming(
    const std::string& artifact_id,
    uint32_t device_id,
    const std::shared_ptr<common::memory::StreamingPinnedBuffer>& streaming_buf,
    gsl::not_null<void*> gpu_ptr,
    size_t total_size,
    gsl::not_null<void*> /*va_space_base*/,
    const std::shared_ptr<UnifiedMemoryAuthority>& uma,
    const loading::ReplicaKey& ikey) {
  ABSL_CHECK(streaming_buf) << "StreamingPinnedBuffer must not be null";
  ABSL_CHECK_GT(total_size, 0) << "Total size must be positive";

  size_t chunk_size = streaming_buf->chunk_size();
  ABSL_CHECK_GT(chunk_size, 0) << "StreamingPinnedBuffer chunk size must be positive";
  size_t offset = 0;

  // Track first error observed in host callbacks (e.g., VS write failures)
  auto first_error = std::make_shared<absl::Status>(absl::OkStatus());
  auto first_error_mu = std::make_shared<absl::Mutex>();

  auto device_status = cuda::set_device(device_id);
  if (!device_status.ok()) {
    return device_status;
  }

  const size_t total_chunks = (total_size + chunk_size - 1) / chunk_size;
  auto pending_callbacks = std::make_shared<absl::BlockingCounter>(static_cast<ptrdiff_t>(total_chunks));

  std::vector<common::CopyHandle> handles;
  while (offset < total_size) {
    size_t current_chunk_size = std::min(chunk_size, total_size - offset);

    common::memory::StreamingChunkGuard guard(streaming_buf);
    auto host_ptr_or = guard.acquire();
    if (!host_ptr_or.ok()) {
      return host_ptr_or.status();
    }
    char* host_ptr = *host_ptr_or;
    const size_t chunk_index = chunk_size > 0 ? offset / chunk_size : 0;

    auto promote_status = guard.promote_to_consumer(chunk_index, current_chunk_size);
    if (!promote_status.ok()) {
      return promote_status;
    }
    const int slot_id = guard.release_for_async();

    // Async D2H via ACM; upon completion, write into VS and return the slot
    void* src_device = static_cast<char*>(gpu_ptr.get()) + offset;
    common::DeviceRegion s{
        .device_id = static_cast<int>(device_id), .dev_ptr = src_device, .length = current_chunk_size};
    common::HostRegion h{.base = host_ptr, .length = current_chunk_size, .pinned = true};
    const uint64_t chunk_offset = offset;
    common::CopyOptions opts{
        .tracing_stage = "D2H/Copy",
        .callbacks = {
            .on_copy_done = [uma,
                             artifact_id,
                             ikey,
                             streaming_buf,
                             slot_id,
                             host_ptr,
                             chunk_offset,
                             current_chunk_size,
                             first_error,
                             first_error_mu,
                             pending_callbacks](absl::Status copy_status) {
              if (copy_status.ok()) {
                absl::Status st = uma ? uma->write_cpu_span(ikey, chunk_offset, host_ptr, current_chunk_size)
                                      : absl::FailedPreconditionError("UMA unavailable for GPU→CPU copy");
                if (!st.ok()) {
                  absl::MutexLock lk(first_error_mu.get());
                  if (first_error->ok()) {
                    *first_error = st;
                  }
                }
              } else {
                absl::MutexLock lk(first_error_mu.get());
                if (first_error->ok()) {
                  *first_error = copy_status;
                }
              }
              absl::Status rc = streaming_buf->return_chunk(slot_id);
              if (!rc.ok()) {
                LOG(WARNING) << "StreamingPinnedBuffer::return_chunk failed slot=" << slot_id << ": " << rc;
              }
              pending_callbacks->DecrementCount();
            }}};
    auto hdl_or = common::AsyncCopyManager::instance().submit_d2h(s, h, opts);
    if (!hdl_or.ok()) {
      absl::Status rc = streaming_buf->return_chunk(slot_id);
      if (!rc.ok()) {
        LOG(WARNING) << "StreamingPinnedBuffer::return_chunk failed after submit_d2h error slot=" << slot_id << ": "
                     << rc;
      }
      return hdl_or.status();
    }
    handles.emplace_back(std::move(*hdl_or));

    offset += current_chunk_size;
  }

  // Ensure all DMA operations and host callbacks have completed
  for (auto& h : handles) {
    auto wst = h.wait();
    if (!wst.ok()) {
      return wst;
    }
  }

  pending_callbacks->Wait();

  {
    absl::MutexLock lk(first_error_mu.get());
    if (!first_error->ok()) {
      return *first_error;
    }
  }

  return absl::OkStatus();
}

} // namespace tensorcast::store::replica
