// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/store/device_types.h"
#include "core/store/replica/replica_memory_coordinator.h"

namespace tensorcast::store::replica {

/**
 * @brief Performs staged CPU to GPU copy using streaming pinned buffer.
 *
 * This helper function implements chunk-aware copy from CPU (DVMP) to GPU memory,
 * using a streaming pinned buffer for staging. It coordinates with UMA for chunk
 * locking and state updates.
 *
 * @param artifact_id Replica identifier for DVMP operations
 * @param device_id GPU device ID
 * @param streaming_buf Streaming pinned buffer for staging
 * @param gpu_ptr Destination GPU memory pointer
 * @param total_size Total bytes to copy
 * @param stream CUDA stream for async operations
 * @param dvmp_base Base pointer to DVMP CPU memory
 * @param dvmp DVMP instance for chunk operations
 * @param uma UMA coordinator for chunk state management
 * @param ikey Instance key for UMA operations
 * @return Status indicating success or failure
 */
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
    const loading::ReplicaKey& ikey);

/**
 * @brief Performs staged GPU to CPU copy using streaming pinned buffer.
 *
 * This helper function implements copy from GPU to CPU (DVMP) memory,
 * using a streaming pinned buffer for staging. It uses DVMP's write_at
 * to preserve metadata and ensure CPU visibility.
 *
 * @param artifact_id Replica identifier for DVMP operations
 * @param device_id GPU device ID
 * @param streaming_buf Streaming pinned buffer for staging
 * @param gpu_ptr Source GPU memory pointer
 * @param total_size Total bytes to copy
 * @param stream CUDA stream for async operations
 * @param dvmp_base Base pointer to DVMP CPU memory
 * @param dvmp DVMP instance for write operations
 * @return Status indicating success or failure
 */
absl::Status perform_copy_gpu_to_cpu_streaming(
    const std::string& artifact_id,
    uint32_t device_id,
    const std::shared_ptr<common::memory::StreamingPinnedBuffer>& streaming_buf,
    void* gpu_ptr,
    size_t total_size,
    cudaStream_t stream,
    void* dvmp_base,
    const std::shared_ptr<common::memory::DistributedVirtualMemoryPool>& dvmp);

} // namespace tensorcast::store::replica
