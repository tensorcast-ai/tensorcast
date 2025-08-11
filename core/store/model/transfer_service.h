// Copyright (c) 2025, StepCast Team. All rights reserved.

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

#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/model/model_location.h"
#include "core/store/model/model_memory_coordinator.h"
#include "core/store/model/transfer_constants.h"

namespace stepcast::store {

// Forward decls
namespace loader {
class SeekableSource;
class PositionedSink;
} // namespace loader

class TransferService {
 public:
  struct Config {
    size_t max_buffer_bytes{kDefaultMaxBufferBytes};
    std::chrono::milliseconds pinned_memory_timeout{std::chrono::milliseconds::zero()};
  };

  TransferService(
      const gsl::not_null<std::shared_ptr<PinnedMemoryPool>>& pinned_pool,
      const gsl::not_null<std::shared_ptr<memory::DistributedVirtualMemoryPool>>& dvmp,
      const std::shared_ptr<ModelMemoryCoordinator>& uma,
      InstanceKey instance_key,
      Config cfg,
      const gsl::not_null<std::shared_ptr<StreamingPinnedBuffer>>& streaming_buffer);

  ~TransferService() = default;

  // Staging buffer accessor (injected; always non-null)
  [[nodiscard]] std::shared_ptr<StreamingPinnedBuffer> get_streaming_buffer() const {
    return spb_;
  }
  [[nodiscard]] size_t get_pool_chunk_size() const;

  // Copies
  absl::Status copy_cpu_to_gpu_streaming(uint32_t device_id, cudaStream_t stream, void* gpu_ptr, size_t total_bytes);
  absl::Status copy_gpu_to_cpu_streaming(uint32_t device_id, cudaStream_t stream, void* gpu_ptr, size_t total_bytes);

  // High-level load from source → CPU/GPU
  absl::Status load_from_source(
      std::unique_ptr<loader::SeekableSource>& source,
      ModelLocation target_location,
      int concurrency,
      std::optional<absl::Span<const uint32_t>> chunk_indices,
      void* gpu_ptr_or_null,
      int device_id);

 private:
  std::unique_ptr<loader::PositionedSink> build_sink_(ModelLocation target_location, void* gpu_ptr, int device_id);

  static std::vector<std::pair<uint64_t, size_t>> build_ranges_(
      std::optional<absl::Span<const uint32_t>> chunk_indices,
      size_t chunk_size,
      uint64_t total_bytes);

  gsl::not_null<std::shared_ptr<PinnedMemoryPool>> pinned_pool_;
  gsl::not_null<std::shared_ptr<memory::DistributedVirtualMemoryPool>> dvmp_;
  std::shared_ptr<ModelMemoryCoordinator> uma_;
  InstanceKey instance_key_;
  Config cfg_;

  gsl::not_null<std::shared_ptr<StreamingPinnedBuffer>> spb_;
};

} // namespace stepcast::store