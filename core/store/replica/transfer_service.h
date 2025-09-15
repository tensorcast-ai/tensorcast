// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "gsl/pointers"

#include "core/common/memory/memory_location.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/common/memory/virtual_address_space.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/replica/transfer_constants.h"
#include "core/store/replica/unified_memory_authority.h"
// Prefer explicit includes over forward declarations
#include "core/store/loader/sink.h"
#include "core/store/loader/source.h"

namespace tensorcast::store::replica {

class TransferService {
 public:
  struct Config {
    size_t max_buffer_bytes{kDefaultMaxBufferBytes};
    std::chrono::milliseconds pinned_memory_timeout{std::chrono::milliseconds::zero()};
  };

  TransferService(
      const gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>& pinned_pool,
      const gsl::not_null<std::shared_ptr<common::memory::VirtualAddressSpace>>& virtual_addr_space,
      const gsl::not_null<std::shared_ptr<UnifiedMemoryAuthority>>& uma,
      loading::ReplicaKey replica_key,
      Config cfg);

  ~TransferService() = default;

  // Staging buffer accessor (injected; always non-null)
  [[nodiscard]] std::shared_ptr<common::memory::StreamingPinnedBuffer> get_streaming_buffer() const {
    return spb_;
  }

  [[nodiscard]] size_t get_pool_chunk_size() const;

  // Copies
  absl::Status copy_cpu_to_gpu_streaming(uint32_t device_id, gsl::not_null<void*> gpu_ptr, size_t total_bytes);
  absl::Status copy_gpu_to_cpu_streaming(uint32_t device_id, gsl::not_null<void*> gpu_ptr, size_t total_bytes);

  // High-level load from source → CPU/GPU
  absl::Status load_from_source(
      std::unique_ptr<loader::SeekableSource>& source,
      common::memory::MemoryLocation target_location,
      int concurrency,
      std::optional<absl::Span<const uint32_t>> chunk_indices,
      void* gpu_ptr_or_null,
      int device_id);

  // Execute a pre-built UMA plan: data-plane only, no state changes
  absl::Status execute(
      const struct UnifiedMemoryAuthority::TransferPlan& plan,
      common::memory::MemoryLocation target_location,
      loader::SeekableSource& source,
      int concurrency,
      void* gpu_ptr_or_null,
      int device_id);

 private:
  std::unique_ptr<loader::PositionedSink> build_sink_(
      common::memory::MemoryLocation target_location,
      void* gpu_ptr,
      int device_id);

  static std::vector<std::pair<uint64_t, size_t>> build_ranges_(
      std::optional<absl::Span<const uint32_t>> chunk_indices,
      size_t chunk_size,
      uint64_t total_bytes);

  gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> pinned_pool_;
  gsl::not_null<std::shared_ptr<common::memory::VirtualAddressSpace>> va_space_;
  gsl::not_null<std::shared_ptr<UnifiedMemoryAuthority>> uma_;
  loading::ReplicaKey replica_key_;
  Config cfg_;

  gsl::not_null<std::shared_ptr<common::memory::StreamingPinnedBuffer>> spb_;
};

} // namespace tensorcast::store::replica
