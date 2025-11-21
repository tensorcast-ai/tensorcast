// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/store/device_types.h"
#include "core/store/replica/unified_memory_authority.h"
#include "gsl/pointers"

namespace tensorcast::store::replica {

/**
 * @brief Performs staged CPU to GPU copy using streaming pinned buffer.
 *
 * This helper function implements chunk-aware copy from CPU (VS) to GPU memory,
 * using a streaming pinned buffer for staging. UMA plan/commit at higher layers
 * handles ledger updates; this helper is copy-only.
 *
 * @param artifact_id Replica identifier for CPU operations
 * @param device_id GPU device ID
 * @param streaming_buf Streaming pinned buffer for staging
 * @param gpu_ptr Destination GPU memory pointer
 * @param total_size Total bytes to copy
 * @param stream CUDA stream for async operations
 * @param va_space_base Base pointer to VS CPU memory
 * @param uma UMA coordinator for chunk state management
 * @param ikey Instance key for UMA operations
 * @return Status indicating success or failure
 */
absl::Status perform_copy_cpu_to_gpu_streaming(
    const std::string& artifact_id,
    uint32_t device_id,
    const std::shared_ptr<common::memory::StreamingPinnedBuffer>& streaming_buf,
    gsl::not_null<void*> gpu_ptr,
    size_t total_size,
    gsl::not_null<void*> vs_base,
    const std::shared_ptr<UnifiedMemoryAuthority>& uma,
    const loading::ReplicaKey& ikey);

/**
 * @brief Performs staged GPU to CPU copy using streaming pinned buffer.
 *
 * This helper function implements copy from GPU to CPU (VS) memory,
 * using a streaming pinned buffer for staging. It uses VS write_at
 * to preserve metadata and ensure CPU visibility.
 *
 * @param artifact_id Replica identifier for CPU operations
 * @param device_id GPU device ID
 * @param streaming_buf Streaming pinned buffer for staging
 * @param gpu_ptr Source GPU memory pointer
 * @param total_size Total bytes to copy
 * @param stream CUDA stream for async operations
 * @param va_space_base Base pointer to VS CPU memory
 * @param uma UMA coordinator for CPU write operations
 * @return Status indicating success or failure
 */
absl::Status perform_copy_gpu_to_cpu_streaming(
    const std::string& artifact_id,
    uint32_t device_id,
    const std::shared_ptr<common::memory::StreamingPinnedBuffer>& streaming_buf,
    gsl::not_null<void*> gpu_ptr,
    size_t total_size,
    gsl::not_null<void*> vs_base,
    const std::shared_ptr<UnifiedMemoryAuthority>& uma,
    const loading::ReplicaKey& ikey);

} // namespace tensorcast::store::replica
