// Copyright (c) 2025, TensorCast Team.

#ifndef TENSORCAST_CORE_COMMON_CUDA_API_H_
#define TENSORCAST_CORE_COMMON_CUDA_API_H_

#include <cstddef>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

// CUDA runtime API types that we need to replicate
#ifndef USE_FAKE_CUDA
#include <cuda_runtime_api.h>
#else
// Define minimal CUDA types for fake backend
enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};

// Define opaque types for fake backend
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

struct CUevent_st;
typedef struct CUevent_st* cudaEvent_t;

// CUDA error enum for fake backend
typedef enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorLaunchFailure = 4,
  cudaErrorInvalidConfiguration = 9,
  cudaErrorUnknown = 999
} cudaError_t;

// Define cudaIpcMemHandle_t for fake backend
typedef struct cudaIpcMemHandle_st {
  char reserved[64]; // CUDA IPC handles are 64 bytes
} cudaIpcMemHandle_t;

// Define cudaUUID_t for fake backend
typedef struct CUuuid_st {
  char bytes[16]; // CUDA definition of UUID
} cudaUUID_t;

// Define cudaDeviceProp for fake backend (minimal version)
typedef struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  int major;
  int minor;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  cudaUUID_t uuid; // Add UUID field
  // Add more fields as needed
} cudaDeviceProp;

// Stream creation flags
#define cudaStreamDefault 0x00
#define cudaStreamNonBlocking 0x01

// Event creation flags
#define cudaEventDefault 0x00
#define cudaEventBlockingSync 0x01
#define cudaEventDisableTiming 0x02
#define cudaEventInterprocess 0x04

// IPC flags
#define cudaIpcMemLazyEnablePeerAccess 0x01

// Host memory registration flags
#define cudaHostRegisterDefault 0x00
#define cudaHostRegisterPortable 0x01
#define cudaHostRegisterMapped 0x02
#define cudaHostRegisterIoMemory 0x04

// Memory types
enum cudaMemoryType {
  cudaMemoryTypeHost = 1,
  cudaMemoryTypeDevice = 2,
  cudaMemoryTypeArray = 3,
  cudaMemoryTypeUnified = 4
};

// Pointer attributes structure
typedef struct cudaPointerAttributes {
  void* devicePointer;
  void* hostPointer;
  int device;
  cudaMemoryType type; // CUDA 10.0+
  cudaMemoryType memoryType; // Pre-CUDA 10.0
  int isManaged;
} cudaPointerAttributes;
#endif

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
absl::Status pointer_get_attributes_full(void* ptr, cudaPointerAttributes* attrs);

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
absl::Status launch_host_func(cudaStream_t stream, void (*func)(void*), void* user_data);

// Event management
absl::Status event_create(cudaEvent_t* event);
absl::Status event_create_with_flags(cudaEvent_t* event, unsigned int flags);
absl::Status event_destroy(cudaEvent_t event);
absl::Status event_record(cudaEvent_t event, cudaStream_t stream = nullptr);
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

// Helper macro for operations not supported by fake backend
#ifdef USE_FAKE_CUDA
#define SC_RETURN_IF_FAKE_CUDA_UNSUPPORTED(operation)                                                   \
  do {                                                                                                  \
    return absl::UnimplementedError(absl::StrCat("Operation not supported in FakeCuda: ", #operation)); \
  } while (0)
#else
#define SC_RETURN_IF_FAKE_CUDA_UNSUPPORTED(operation) ((void)0)
#endif

} // namespace tensorcast::cuda

#endif // TENSORCAST_CORE_COMMON_CUDA_API_H_
