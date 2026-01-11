// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::cuda {

#define TENSORCAST_CUDA_BACKEND_FUNCTIONS(X)                                                                          \
  X(absl::Status, set_device, (int device_id), device_id)                                                             \
  X(absl::Status, get_device, (int* device_id), device_id)                                                            \
  X(absl::Status, malloc, (void** ptr, size_t bytes), ptr, bytes)                                                     \
  X(absl::Status, malloc_host, (void** ptr, size_t bytes), ptr, bytes)                                                \
  X(absl::Status, free, (void* ptr), ptr)                                                                             \
  X(absl::Status, free_host, (void* ptr), ptr)                                                                        \
  X(absl::Status, memcpy, (void* dst, const void* src, size_t bytes, cudaMemcpyKind kind), dst, src, bytes, kind)     \
  X(absl::Status,                                                                                                     \
    memcpy_async,                                                                                                     \
    (void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream),                             \
    dst,                                                                                                              \
    src,                                                                                                              \
    bytes,                                                                                                            \
    kind,                                                                                                             \
    stream)                                                                                                           \
  X(absl::Status, memset, (void* ptr, int value, size_t bytes), ptr, value, bytes)                                    \
  X(absl::Status, memset_async, (void* ptr, int value, size_t bytes, cudaStream_t stream), ptr, value, bytes, stream) \
  X(absl::Status, host_register, (void* ptr, size_t size, unsigned int flags), ptr, size, flags)                      \
  X(absl::Status, host_unregister, (void* ptr), ptr)                                                                  \
  X(absl::Status, get_device_count, (int* count), count)                                                              \
  X(absl::Status,                                                                                                     \
    get_memory_info,                                                                                                  \
    (size_t* free_bytes, size_t* total_bytes, int device_id),                                                         \
    free_bytes,                                                                                                       \
    total_bytes,                                                                                                      \
    device_id)                                                                                                        \
  X(absl::StatusOr<std::string>, get_device_name, (int device_id), device_id)                                         \
  X(absl::Status, get_device_properties, (int device_id, void* prop), device_id, prop)                                \
  X(absl::Status, pointer_get_attributes, (void* ptr, int* device, void** device_ptr), ptr, device, device_ptr)       \
  X(absl::Status, pointer_get_attributes_full, (const void* ptr, cudaPointerAttributes* attrs), ptr, attrs)           \
  X(absl::Status, mem_get_address_range, (void** base, size_t* range_bytes, const void* ptr), base, range_bytes, ptr) \
  X(absl::Status, get_ipc_handle, (const void* ptr, std::string* handle), ptr, handle)                                \
  X(absl::Status, open_ipc_handle, (const std::string& handle, void** ptr), handle, ptr)                              \
  X(absl::Status, close_ipc_handle, (void* ptr), ptr)                                                                 \
  X(absl::Status, get_ipc_mem_handle, (cudaIpcMemHandle_t * handle, void* dev_ptr), handle, dev_ptr)                  \
  X(absl::Status,                                                                                                     \
    open_ipc_mem_handle,                                                                                              \
    (void** dev_ptr, cudaIpcMemHandle_t handle, unsigned int flags),                                                  \
    dev_ptr,                                                                                                          \
    handle,                                                                                                           \
    flags)                                                                                                            \
  X(absl::Status, close_ipc_mem_handle, (void* dev_ptr), dev_ptr)                                                     \
  X(absl::Status, stream_create, (cudaStream_t * stream), stream)                                                     \
  X(absl::Status, stream_create_with_flags, (cudaStream_t * stream, unsigned int flags), stream, flags)               \
  X(absl::Status,                                                                                                     \
    stream_create_with_priority,                                                                                      \
    (cudaStream_t * stream, unsigned int flags, int priority),                                                        \
    stream,                                                                                                           \
    flags,                                                                                                            \
    priority)                                                                                                         \
  X(absl::Status, stream_destroy, (cudaStream_t stream), stream)                                                      \
  X(absl::Status, stream_synchronize, (cudaStream_t stream), stream)                                                  \
  X(absl::Status, stream_wait_event, (cudaStream_t stream, cudaEvent_t event), stream, event)                         \
  X(absl::Status,                                                                                                     \
    stream_add_callback,                                                                                              \
    (cudaStream_t stream, void (*callback)(cudaStream_t, cudaError_t, void*), void* user_data, unsigned int flags),   \
    stream,                                                                                                           \
    callback,                                                                                                         \
    user_data,                                                                                                        \
    flags)                                                                                                            \
  X(absl::Status,                                                                                                     \
    launch_host_func,                                                                                                 \
    (cudaStream_t stream, void (*func)(void*), void* user_data),                                                      \
    stream,                                                                                                           \
    func,                                                                                                             \
    user_data)                                                                                                        \
  X(absl::Status, event_create, (cudaEvent_t * event), event)                                                         \
  X(absl::Status, event_create_with_flags, (cudaEvent_t * event, unsigned int flags), event, flags)                   \
  X(absl::Status, event_destroy, (cudaEvent_t event), event)                                                          \
  X(absl::Status, event_record, (cudaEvent_t event, cudaStream_t stream), event, stream)                              \
  X(absl::Status, event_query, (cudaEvent_t event, bool* ready), event, ready)                                        \
  X(absl::Status, event_synchronize, (cudaEvent_t event), event)                                                      \
  X(absl::Status, event_elapsed_time, (float* ms, cudaEvent_t start, cudaEvent_t end), ms, start, end)                \
  X(absl::Status, device_synchronize, (), )                                                                           \
  X(absl::Status, get_last_error, (), )                                                                               \
  X(absl::Status, peek_last_error, (), )                                                                              \
  X(bool, is_fake, (), )                                                                                              \
  X(bool, is_available, (), )                                                                                         \
  X(void, configure_same_process_ipc_fallback, (bool enabled), enabled)                                               \
  X(absl::Status,                                                                                                     \
    device_can_access_peer,                                                                                           \
    (int* can_access, int device, int peer_device),                                                                   \
    can_access,                                                                                                       \
    device,                                                                                                           \
    peer_device)                                                                                                      \
  X(absl::Status, enable_peer_access, (int current_device, int peer_device), current_device, peer_device)             \
  X(absl::Status,                                                                                                     \
    memcpy_peer_async,                                                                                                \
    (void* dst, int dst_device, const void* src, int src_device, size_t bytes, cudaStream_t stream),                  \
    dst,                                                                                                              \
    dst_device,                                                                                                       \
    src,                                                                                                              \
    src_device,                                                                                                       \
    bytes,                                                                                                            \
    stream)                                                                                                           \
  X(absl::Status, cu_init, (unsigned int flags), flags)                                                               \
  X(absl::Status, cu_device_get, (CUdevice * device, int ordinal), device, ordinal)                                   \
  X(absl::Status,                                                                                                     \
    cu_device_get_attribute,                                                                                          \
    (int* value, CUdevice_attribute attribute, CUdevice device),                                                      \
    value,                                                                                                            \
    attribute,                                                                                                        \
    device)                                                                                                           \
  X(absl::Status,                                                                                                     \
    cu_mem_get_allocation_granularity,                                                                                \
    (size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option),                  \
    granularity,                                                                                                      \
    prop,                                                                                                             \
    option)                                                                                                           \
  X(absl::Status,                                                                                                     \
    cu_mem_address_reserve,                                                                                           \
    (CUdeviceptr * ptr, size_t size, size_t alignment, CUdeviceptr addr, uint64_t flags),                             \
    ptr,                                                                                                              \
    size,                                                                                                             \
    alignment,                                                                                                        \
    addr,                                                                                                             \
    flags)                                                                                                            \
  X(absl::Status,                                                                                                     \
    cu_mem_create,                                                                                                    \
    (CUmemGenericAllocationHandle * handle, size_t size, const CUmemAllocationProp* prop, uint64_t flags),            \
    handle,                                                                                                           \
    size,                                                                                                             \
    prop,                                                                                                             \
    flags)                                                                                                            \
  X(absl::Status,                                                                                                     \
    cu_mem_map,                                                                                                       \
    (CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, uint64_t flags),               \
    ptr,                                                                                                              \
    size,                                                                                                             \
    offset,                                                                                                           \
    handle,                                                                                                           \
    flags)                                                                                                            \
  X(absl::Status,                                                                                                     \
    cu_mem_set_access,                                                                                                \
    (CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count),                                        \
    ptr,                                                                                                              \
    size,                                                                                                             \
    desc,                                                                                                             \
    count)                                                                                                            \
  X(absl::Status, cu_mem_unmap, (CUdeviceptr ptr, size_t size), ptr, size)                                            \
  X(absl::Status, cu_mem_release, (CUmemGenericAllocationHandle handle), handle)                                      \
  X(absl::Status, cu_mem_address_free, (CUdeviceptr ptr, size_t size), ptr, size)                                     \
  X(absl::Status,                                                                                                     \
    cu_mem_get_handle_for_address_range,                                                                              \
    (void* handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handle_type, unsigned long long flags),        \
    handle,                                                                                                           \
    dptr,                                                                                                             \
    size,                                                                                                             \
    handle_type,                                                                                                      \
    flags)

class CudaBackend {
 public:
  virtual ~CudaBackend() = default;

#define TENSORCAST_DECLARE_BACKEND_METHOD(return_type, name, args, ...) virtual return_type name args = 0;
  TENSORCAST_CUDA_BACKEND_FUNCTIONS(TENSORCAST_DECLARE_BACKEND_METHOD)
#undef TENSORCAST_DECLARE_BACKEND_METHOD
};

class RealCudaBackend final : public CudaBackend {
 public:
#define TENSORCAST_DECLARE_BACKEND_METHOD(return_type, name, args, ...) return_type name args override;
  TENSORCAST_CUDA_BACKEND_FUNCTIONS(TENSORCAST_DECLARE_BACKEND_METHOD)
#undef TENSORCAST_DECLARE_BACKEND_METHOD
};

class FakeCudaBackend final : public CudaBackend {
 public:
#define TENSORCAST_DECLARE_BACKEND_METHOD(return_type, name, args, ...) return_type name args override;
  TENSORCAST_CUDA_BACKEND_FUNCTIONS(TENSORCAST_DECLARE_BACKEND_METHOD)
#undef TENSORCAST_DECLARE_BACKEND_METHOD
};

} // namespace tensorcast::cuda
