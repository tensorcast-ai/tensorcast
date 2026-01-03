// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tensorcast::cuda {

// Memory allocation and management
absl::Status set_device(int device_id);
absl::Status get_device(int* device_id);
absl::Status malloc(void** ptr, size_t bytes);
absl::Status malloc_host(void** ptr, size_t bytes); // For pinned memory
absl::Status free(void* ptr);
absl::Status free_host(void* ptr);
absl::Status memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind);
absl::Status memcpy_async(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream = nullptr);
absl::Status memset(void* ptr, int value, size_t bytes);
absl::Status memset_async(void* ptr, int value, size_t bytes, cudaStream_t stream = nullptr);

// Host memory registration
absl::Status host_register(void* ptr, size_t size, unsigned int flags);
absl::Status host_unregister(void* ptr);

// Device information
absl::Status get_device_count(int* count);
absl::Status get_memory_info(size_t* free_bytes, size_t* total_bytes, int device_id);
absl::StatusOr<std::string> get_device_name(int device_id);
absl::Status get_device_properties(int device_id, void* prop); // cudaDeviceProp*
absl::Status pointer_get_attributes(void* ptr, int* device, void** device_ptr);
absl::Status pointer_get_attributes_full(const void* ptr, cudaPointerAttributes* attrs);
absl::Status mem_get_address_range(void** base, size_t* range_bytes, const void* ptr);

// IPC handle operations (string-based for compatibility)
absl::Status get_ipc_handle(const void* ptr, std::string* handle);
absl::Status open_ipc_handle(const std::string& handle, void** ptr);
absl::Status close_ipc_handle(void* ptr);

// IPC handle operations (native CUDA handle type)
absl::Status get_ipc_mem_handle(cudaIpcMemHandle_t* handle, void* dev_ptr);
absl::Status open_ipc_mem_handle(void** dev_ptr, cudaIpcMemHandle_t handle, unsigned int flags = 0);
absl::Status close_ipc_mem_handle(void* dev_ptr);

// Stream management
absl::Status stream_create(cudaStream_t* stream);
absl::Status stream_create_with_flags(cudaStream_t* stream, unsigned int flags);
absl::Status stream_destroy(cudaStream_t stream);
absl::Status stream_synchronize(cudaStream_t stream);
absl::Status stream_wait_event(cudaStream_t stream, cudaEvent_t event);

// Stream callback operations
absl::Status stream_add_callback(
    cudaStream_t stream,
    void (*callback)(cudaStream_t, cudaError_t, void*),
    void* user_data,
    unsigned int flags);
// Enqueue a host callback on a CUDA stream. The callback must never invoke CUDA
// Runtime/Driver APIs or block on stream work; callers should bounce any
// follow-up CUDA interactions onto a different worker thread to comply with
// cudaLaunchHostFunc requirements.
absl::Status launch_host_func(cudaStream_t stream, void (*func)(void*), void* user_data);

// Event management
absl::Status event_create(cudaEvent_t* event);
absl::Status event_create_with_flags(cudaEvent_t* event, unsigned int flags);
absl::Status event_destroy(cudaEvent_t event);
absl::Status event_record(cudaEvent_t event, cudaStream_t stream = nullptr);
absl::Status event_query(cudaEvent_t event, bool* ready);
absl::Status event_synchronize(cudaEvent_t event);
absl::Status event_elapsed_time(float* ms, cudaEvent_t start, cudaEvent_t end);

// Synchronization
absl::Status device_synchronize();

// Error handling
absl::Status get_last_error();
absl::Status peek_last_error();

// Runtime checks
bool is_fake();
bool is_available();

// Debug/testing controls
// Enable or disable same-process CUDA IPC fallback behavior in the real backend.
// When enabled, if an IPC handle exported and opened in the same process fails
// via the CUDA path, we will return the original pointer as a fallback.
// Default is disabled. No-op in the fake backend (always behaves like same-process).
void configure_same_process_ipc_fallback(bool enabled);

// Peer access and peer copy (for cross-device D2D)
absl::Status device_can_access_peer(int* can_access, int device, int peer_device);
absl::Status enable_peer_access(int current_device, int peer_device);
absl::Status memcpy_peer_async(
    void* dst,
    int dst_device,
    const void* src,
    int src_device,
    size_t bytes,
    cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// CUDA VMM (Driver API) wrappers.
// ---------------------------------------------------------------------------
absl::Status cu_init(unsigned int flags = 0);
absl::Status cu_device_get(CUdevice* device, int ordinal);
absl::Status cu_device_get_attribute(int* value, CUdevice_attribute attribute, CUdevice device);
absl::Status cu_mem_get_allocation_granularity(
    size_t* granularity,
    const CUmemAllocationProp* prop,
    CUmemAllocationGranularity_flags option);
absl::Status cu_mem_address_reserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, uint64_t flags);
absl::Status cu_mem_create(
    CUmemGenericAllocationHandle* handle,
    size_t size,
    const CUmemAllocationProp* prop,
    uint64_t flags);
absl::Status cu_mem_map(
    CUdeviceptr ptr,
    size_t size,
    size_t offset,
    CUmemGenericAllocationHandle handle,
    uint64_t flags);
absl::Status cu_mem_set_access(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count);
absl::Status cu_mem_unmap(CUdeviceptr ptr, size_t size);
absl::Status cu_mem_release(CUmemGenericAllocationHandle handle);
absl::Status cu_mem_address_free(CUdeviceptr ptr, size_t size);
absl::Status cu_mem_get_handle_for_address_range(
    void* handle,
    CUdeviceptr dptr,
    size_t size,
    CUmemRangeHandleType handle_type,
    unsigned long long flags);

} // namespace tensorcast::cuda
