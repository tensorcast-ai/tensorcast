// Copyright (c) 2025, TensorCast Team.

#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/common/cuda_api.h"

namespace tensorcast::common::memory {

/**
 * @brief Represents a block of CUDA device memory.
 *
 * Manages the lifetime of the memory, supporting direct allocation,
 * or adoption of an external pointer.
 */
class GpuDeviceMemory {
 public:
  /**
   * @brief Default constructor, creates an uninitialized GpuDeviceMemory object.
   */
  GpuDeviceMemory() = default;

  ~GpuDeviceMemory();

  // Disable copy and move semantics to prevent double frees/mismanagement.
  GpuDeviceMemory(const GpuDeviceMemory&) = delete;
  GpuDeviceMemory& operator=(const GpuDeviceMemory&) = delete;
  GpuDeviceMemory(GpuDeviceMemory&&) = delete; // Could potentially implement move, but disable for simplicity/safety
  GpuDeviceMemory& operator=(GpuDeviceMemory&&) = delete;

  /**
   * @brief Allocates memory directly using cudaMalloc.
   * The GpuDeviceMemory object owns this memory and will call cudaFree on destruction.
   * @param size Number of bytes to allocate.
   * @param device_id The target GPU device ID.
   * @return absl::Status OK on success or detailed error status.
   */
  absl::Status allocate(size_t size, int device_id);

  /**
   * @brief Adopts an external CUDA IPC memory handle. The handle is opened via
   *        cudaIpcOpenMemHandle and will be closed automatically in the
   *        destructor / release_resources().
   *
   * @param ipc_handle  The CUDA IPC memory handle obtained from another
   *                    process/device.
   * @param size        Size of the memory region, in bytes.
   * @param device_id   Local CUDA device id that the memory belongs to.
   * @return absl::Status OK on success or detailed error.
   */
  absl::Status adopt_ipc_handle(const cudaIpcMemHandle_t& ipc_handle, size_t size, int device_id);

  /**
   * @brief Returns the raw CUDA device pointer.
   * @return Pointer to device memory, or nullptr if not allocated/adopted.
   */
  void* get() const {
    return data_;
  }

  /**
   * @brief Returns the size of the allocated/adopted memory block.
   * @return Size in bytes.
   */
  size_t size() const {
    return size_;
  }

  /**
   * @brief Returns the associated device ID.
   * @return Device ID, or -1 if uninitialized.
   */
  int device_id() const {
    return device_id_;
  }

  /**
   * @brief Returns the CUDA IPC memory handle (if available).
   * May be invalid if memory wasn't allocated with IPC flag or came from pool without it.
   * @return cudaIpcMemHandle_t.
   */
  [[nodiscard]] cudaIpcMemHandle_t get_handle() const {
    return handle_;
  }

 private:
  /**
   * @brief Frees or deallocates memory based on the allocation type.
   */
  void release_resources();

  enum class AllocationType : std::uint8_t {
    UNINITIALIZED,
    DIRECT, // Allocated via cudaMalloc
    IPC_EXTERNAL // Borrowed via CUDA IPC handle
  };

  void* data_ = nullptr;
  size_t size_ = 0;
  int device_id_ = -1;
  cudaIpcMemHandle_t handle_{}; // Initialize to zero/default
  AllocationType allocation_type_ = AllocationType::UNINITIALIZED;
};
} // namespace tensorcast::common::memory