// Copyright (c) 2025, TensorCast Team.

#include "core/store/replica/transfer_helpers.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "core/common/async_copy_manager.h"
#include "core/common/memory/memory_location.h"
#include "core/store/replica/chunk_meta.h"

namespace tensorcast::store::replica {

absl::Status perform_copy_cpu_to_gpu_streaming(
    const std::string& artifact_id,
    uint32_t device_id,
    const std::shared_ptr<common::memory::StreamingPinnedBuffer>& streaming_buf,
    gsl::not_null<void*> gpu_ptr,
    size_t total_size,
    gsl::not_null<void*> vs_base,
    const std::shared_ptr<common::memory::VirtualAddressSpace>& virtual_addr_space,
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
    va_chunk =
        virtual_addr_space ? virtual_addr_space->chunk_size() : common::memory::VirtualAddressSpace::kDefaultChunkSize;
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
  for (size_t va_idx = 0; va_idx < num_va_chunks; ++va_idx) {
    const size_t va_off = va_idx * va_chunk;
    const size_t this_len = std::min(va_chunk, total_size - va_off);

    // No UMA/VS locking in final cutover. UMA is authoritative; residency is
    // guaranteed by direct-write grant when available (kept by grant_keepalive).

    // Aggregate UMA advancement per CPU block via callbacks
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

      // Copy from CPU VA (prefer UMA grant base if available) to pinned chunk
      void* src_host = grant_base_addr != 0 ? reinterpret_cast<void*>(grant_base_addr + va_off + copied)
                                            : static_cast<char*>(vs_base.get()) + va_off + copied;
      std::memcpy(host_ptr, src_host, step);

      // Async copy H2D via ACM; return SPB slot in host callback
      void* dst_device = static_cast<char*>(gpu_ptr.get()) + va_off + copied;
      common::HostRegion h{.base = host_ptr, .length = step, .pinned = true};
      common::DeviceRegion d{.device_id = static_cast<int>(device_id), .dev_ptr = dst_device, .length = step};
      common::CopyOptions opts{
          .tracing_stage = "H2D/Copy", .callbacks = {.on_copy_done = [streaming_buf, slot_id, remaining]() {
                                         absl::Status rc = streaming_buf->return_chunk(slot_id);
                                         if (!rc.ok()) {
                                           LOG(WARNING) << "StreamingPinnedBuffer::return_chunk failed slot=" << slot_id
                                                        << ": " << rc;
                                         }
                                         (void)remaining->fetch_sub(1, std::memory_order_acq_rel);
                                       }}};
      auto hdl_or = common::AsyncCopyManager::instance().submit_h2d(h, d, opts);
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
    }
  }

  return absl::OkStatus();
}

absl::Status perform_copy_gpu_to_cpu_streaming(
    const std::string& artifact_id,
    uint32_t device_id,
    const std::shared_ptr<common::memory::StreamingPinnedBuffer>& streaming_buf,
    gsl::not_null<void*> gpu_ptr,
    size_t total_size,
    gsl::not_null<void*> va_space_base,
    const std::shared_ptr<common::memory::VirtualAddressSpace>& virtual_addr_space) {
  ABSL_CHECK(streaming_buf) << "StreamingPinnedBuffer must not be null";
  ABSL_CHECK_GT(total_size, 0) << "Total size must be positive";

  size_t chunk_size = streaming_buf->chunk_size();
  size_t offset = 0;

  // Track first error observed in host callbacks (e.g., VS write failures)
  auto first_error = std::make_shared<absl::Status>(absl::OkStatus());
  auto first_error_mu = std::make_shared<absl::Mutex>();

  auto device_status = cuda::set_device(device_id);
  if (!device_status.ok()) {
    return device_status;
  }

  std::vector<common::CopyHandle> handles;
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

    // Async D2H via ACM; upon completion, write into VS and return the slot
    void* src_device = static_cast<char*>(gpu_ptr.get()) + offset;
    common::DeviceRegion s{
        .device_id = static_cast<int>(device_id), .dev_ptr = src_device, .length = current_chunk_size};
    common::HostRegion h{.base = host_ptr, .length = current_chunk_size, .pinned = true};
    const uint64_t chunk_offset = offset;
    common::CopyOptions opts{
        .tracing_stage = "D2H/Copy",
        .callbacks = {
            .on_copy_done = [virtual_addr_space,
                             artifact_id,
                             streaming_buf,
                             slot_id,
                             host_ptr,
                             chunk_offset,
                             current_chunk_size,
                             first_error,
                             first_error_mu]() {
              // Write to VS then return slot; record first VS error if any
              absl::Status st = virtual_addr_space->write_at(artifact_id, chunk_offset, host_ptr, current_chunk_size);
              if (!st.ok()) {
                absl::MutexLock lk(first_error_mu.get());
                if (first_error->ok()) {
                  *first_error = st;
                }
              }
              {
                absl::Status rc = streaming_buf->return_chunk(slot_id);
                if (!rc.ok()) {
                  LOG(WARNING) << "StreamingPinnedBuffer::return_chunk failed slot=" << slot_id << ": " << rc;
                }
              }
            }}};
    auto hdl_or = common::AsyncCopyManager::instance().submit_d2h(s, h, opts);
    if (!hdl_or.ok()) {
      ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
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
  {
    absl::MutexLock lk(first_error_mu.get());
    if (!first_error->ok()) {
      return *first_error;
    }
  }

  return absl::OkStatus();
}

} // namespace tensorcast::store::replica
