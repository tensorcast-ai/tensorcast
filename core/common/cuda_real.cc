// Copyright (c) 2025, TensorCast Team.

#include "core/common/cuda_api.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <iomanip>
#include <sstream>

#include <unistd.h>
#include <cstdlib>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "core/common/error_handling.h"

namespace tensorcast::cuda {

namespace {
// Best-effort same-process fallback for CUDA IPC handles during unit tests.
// We remember exported handles and their original pointers so that if a test
// runs export+open in the same process (which real CUDA does not support), we
// can return the original pointer instead of failing.
struct ExportedIpcInfo {
  void* ptr{nullptr};
  int device_id{-1};
  pid_t pid{0};
};

absl::Mutex g_ipc_map_mu;
absl::flat_hash_map<std::string, ExportedIpcInfo> g_exported_ipc_map ABSL_GUARDED_BY(g_ipc_map_mu);
absl::flat_hash_set<void*> g_exported_ptrs ABSL_GUARDED_BY(g_ipc_map_mu);
std::atomic<bool> g_enable_same_process_ipc_fallback{false};
} // namespace

void configure_same_process_ipc_fallback(bool enabled) {
  g_enable_same_process_ipc_fallback.store(enabled, std::memory_order_relaxed);
}

absl::Status set_device(int device_id) {
  SC_RETURN_IF_CUDA_ERROR(cudaSetDevice(device_id));
  return absl::OkStatus();
}

absl::Status get_device(int* device_id) {
  SC_RETURN_IF_CUDA_ERROR(cudaGetDevice(device_id));
  return absl::OkStatus();
}

absl::Status malloc(void** ptr, size_t bytes) {
  SC_RETURN_IF_CUDA_ERROR(cudaMalloc(ptr, bytes));
  return absl::OkStatus();
}

absl::Status malloc_host(void** ptr, size_t bytes) {
  SC_RETURN_IF_CUDA_ERROR(cudaMallocHost(ptr, bytes));
  return absl::OkStatus();
}

absl::Status free(void* ptr) {
  if (ptr == nullptr) {
    return absl::OkStatus();
  }
  // Remove any IPC fallback mapping for this pointer before freeing
  {
    absl::MutexLock lock(&g_ipc_map_mu);
    if (g_exported_ptrs.erase(ptr) > 0) {
      for (auto it = g_exported_ipc_map.begin(); it != g_exported_ipc_map.end();) {
        if (it->second.ptr == ptr) {
          g_exported_ipc_map.erase(it++);
        } else {
          ++it;
        }
      }
    }
  }
  SC_RETURN_IF_CUDA_ERROR(cudaFree(ptr));
  return absl::OkStatus();
}

absl::Status free_host(void* ptr) {
  if (ptr == nullptr) {
    return absl::OkStatus();
  }
  SC_RETURN_IF_CUDA_ERROR(cudaFreeHost(ptr));
  return absl::OkStatus();
}

absl::Status memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
  SC_RETURN_IF_CUDA_ERROR(cudaMemcpy(dst, src, bytes, kind));
  return absl::OkStatus();
}

absl::Status memcpy_async(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
  SC_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(dst, src, bytes, kind, stream));
  return absl::OkStatus();
}

absl::Status memset(void* ptr, int value, size_t bytes) {
  SC_RETURN_IF_CUDA_ERROR(cudaMemset(ptr, value, bytes));
  return absl::OkStatus();
}

absl::Status get_device_count(int* count) {
  SC_RETURN_IF_CUDA_ERROR(cudaGetDeviceCount(count));
  return absl::OkStatus();
}

absl::Status get_memory_info(size_t* free_bytes, size_t* total_bytes, int device_id) {
  int current_device;
  SC_RETURN_IF_CUDA_ERROR(cudaGetDevice(&current_device));

  SC_RETURN_IF_CUDA_ERROR(cudaSetDevice(device_id));
  SC_RETURN_IF_CUDA_ERROR(cudaMemGetInfo(free_bytes, total_bytes));
  SC_RETURN_IF_CUDA_ERROR(cudaSetDevice(current_device));

  return absl::OkStatus();
}

absl::StatusOr<std::string> get_device_name(int device_id) {
  cudaDeviceProp prop;
  SC_RETURN_IF_CUDA_ERROR(cudaGetDeviceProperties(&prop, device_id));
  return std::string(prop.name);
}

absl::Status get_ipc_handle(const void* ptr, std::string* handle) {
  cudaIpcMemHandle_t cuda_handle;
  // CUDA API requires non-const pointer here; we do not modify the memory.
  SC_RETURN_IF_CUDA_ERROR(
      cudaIpcGetMemHandle(&cuda_handle, const_cast<void*>(ptr))); // NOLINT(cppcoreguidelines-pro-type-const-cast)

  // Convert handle to string representation
  std::stringstream ss;
  for (int i = 0; i < sizeof(cudaIpcMemHandle_t); ++i) {
    ss << std::hex << std::setw(2) << std::setfill('0')
       << static_cast<int>(reinterpret_cast<const unsigned char*>(&cuda_handle)[i]);
  }
  *handle = ss.str();

  // Record mapping for same-process fallback when enabled
  if (g_enable_same_process_ipc_fallback.load(std::memory_order_relaxed)) {
    absl::MutexLock lock(&g_ipc_map_mu);
    int current_device = -1;
    ABSL_CHECK_OK(get_device(&current_device));
    g_exported_ipc_map[*handle] = ExportedIpcInfo{
        .ptr = const_cast<void*>(ptr), // NOLINT(cppcoreguidelines-pro-type-const-cast)
        .device_id = current_device,
        .pid = getpid()};
    g_exported_ptrs.insert(const_cast<void*>(ptr)); // NOLINT(cppcoreguidelines-pro-type-const-cast)
  }
  return absl::OkStatus();
}

absl::Status open_ipc_handle(const std::string& handle, void** ptr) {
  if (handle.size() != sizeof(cudaIpcMemHandle_t) * 2) {
    return absl::InvalidArgumentError("Invalid IPC handle size");
  }

  cudaIpcMemHandle_t cuda_handle;
  // Convert string back to handle
  for (size_t i = 0; i < sizeof(cudaIpcMemHandle_t); ++i) {
    std::string byte_str = handle.substr(i * 2, 2);
    reinterpret_cast<unsigned char*>(&cuda_handle)[i] = static_cast<unsigned char>(std::stoi(byte_str, nullptr, 16));
  }

  // Try the real CUDA path first
  cudaError_t err = cudaIpcOpenMemHandle(ptr, cuda_handle, cudaIpcMemLazyEnablePeerAccess);
  if (err == cudaSuccess) {
    return absl::OkStatus();
  }

  // If not enabled, return the CUDA error
  if (!g_enable_same_process_ipc_fallback.load(std::memory_order_relaxed)) {
    return common::cuda_as_status(err, "cudaIpcOpenMemHandle");
  }

  // Fallback: if export and open happen in the same process (unit tests),
  // return the original pointer that created the handle.
  {
    absl::MutexLock lock(&g_ipc_map_mu);
    auto it = g_exported_ipc_map.find(handle);
    if (it == g_exported_ipc_map.end()) {
      return common::cuda_as_status(err, "cudaIpcOpenMemHandle");
    }
    const ExportedIpcInfo& info = it->second;
    // Ensure same process
    if (info.pid != getpid()) {
      return common::cuda_as_status(err, "cudaIpcOpenMemHandle");
    }
    // Ensure device consistency when possible
    int cur_dev = -1;
    ABSL_CHECK_OK(get_device(&cur_dev));
    cudaPointerAttributes attrs;
    if (cudaPointerGetAttributes(&attrs, info.ptr) == cudaSuccess) {
      if (cur_dev >= 0 && attrs.device != cur_dev) {
        return absl::FailedPreconditionError("IPC fallback rejected: device mismatch with current device");
      }
    } else {
      // If we cannot inspect attributes, require device match with recorded export device
      if (cur_dev >= 0 && info.device_id >= 0 && info.device_id != cur_dev) {
        return absl::FailedPreconditionError("IPC fallback rejected: recorded device mismatch");
      }
    }
    *ptr = info.ptr;
    return absl::OkStatus();
  }
}

absl::Status close_ipc_handle(void* ptr) {
  // Try to close as a real IPC-opened pointer first
  cudaError_t err = cudaIpcCloseMemHandle(ptr);
  if (err == cudaSuccess) {
    return absl::OkStatus();
  }
  // If enabled and this pointer corresponds to an exported original (same-process fallback),
  // treat close as a no-op.
  if (g_enable_same_process_ipc_fallback.load(std::memory_order_relaxed)) {
    absl::MutexLock lock(&g_ipc_map_mu);
    if (g_exported_ptrs.contains(ptr)) {
      return absl::OkStatus();
    }
  }
  return common::cuda_as_status(err, "cudaIpcCloseMemHandle");
}

absl::Status device_synchronize() {
  SC_RETURN_IF_CUDA_ERROR(cudaDeviceSynchronize());
  return absl::OkStatus();
}

absl::Status memset_async(void* ptr, int value, size_t bytes, cudaStream_t stream) {
  SC_RETURN_IF_CUDA_ERROR(cudaMemsetAsync(ptr, value, bytes, stream));
  return absl::OkStatus();
}

absl::Status get_device_properties(int device_id, void* prop) {
  SC_RETURN_IF_CUDA_ERROR(cudaGetDeviceProperties(static_cast<cudaDeviceProp*>(prop), device_id));
  return absl::OkStatus();
}

absl::Status pointer_get_attributes(void* ptr, int* device, void** device_ptr) {
  cudaPointerAttributes attrs;
  SC_RETURN_IF_CUDA_ERROR(cudaPointerGetAttributes(&attrs, ptr));
  if (device) {
    *device = attrs.device;
  }
  if (device_ptr) {
    *device_ptr = attrs.devicePointer;
  }
  return absl::OkStatus();
}

absl::Status pointer_get_attributes_full(const void* ptr, cudaPointerAttributes* attrs) {
  SC_RETURN_IF_CUDA_ERROR(cudaPointerGetAttributes(attrs, const_cast<void*>(ptr))); // NOLINT
  return absl::OkStatus();
}

// Stream management
absl::Status stream_create(cudaStream_t* stream) {
  SC_RETURN_IF_CUDA_ERROR(cudaStreamCreate(stream));
  return absl::OkStatus();
}

absl::Status stream_create_with_flags(cudaStream_t* stream, unsigned int flags) {
  SC_RETURN_IF_CUDA_ERROR(cudaStreamCreateWithFlags(stream, flags));
  return absl::OkStatus();
}

absl::Status stream_destroy(cudaStream_t stream) {
  SC_RETURN_IF_CUDA_ERROR(cudaStreamDestroy(stream));
  return absl::OkStatus();
}

absl::Status stream_synchronize(cudaStream_t stream) {
  SC_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(stream));
  return absl::OkStatus();
}

absl::Status stream_wait_event(cudaStream_t stream, cudaEvent_t event) {
  SC_RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(stream, event, 0));
  return absl::OkStatus();
}

absl::Status stream_add_callback(
    cudaStream_t stream,
    void (*callback)(cudaStream_t, cudaError_t, void*),
    void* user_data,
    unsigned int flags) {
  SC_RETURN_IF_CUDA_ERROR(cudaStreamAddCallback(stream, callback, user_data, flags));
  return absl::OkStatus();
}

absl::Status launch_host_func(cudaStream_t stream, void (*func)(void*), void* user_data) {
  SC_RETURN_IF_CUDA_ERROR(cudaLaunchHostFunc(stream, func, user_data));
  return absl::OkStatus();
}

// Event management
absl::Status event_create(cudaEvent_t* event) {
  SC_RETURN_IF_CUDA_ERROR(cudaEventCreate(event));
  return absl::OkStatus();
}

absl::Status event_create_with_flags(cudaEvent_t* event, unsigned int flags) {
  SC_RETURN_IF_CUDA_ERROR(cudaEventCreateWithFlags(event, flags));
  return absl::OkStatus();
}

absl::Status event_destroy(cudaEvent_t event) {
  SC_RETURN_IF_CUDA_ERROR(cudaEventDestroy(event));
  return absl::OkStatus();
}

absl::Status event_record(cudaEvent_t event, cudaStream_t stream) {
  SC_RETURN_IF_CUDA_ERROR(cudaEventRecord(event, stream));
  return absl::OkStatus();
}

absl::Status event_synchronize(cudaEvent_t event) {
  SC_RETURN_IF_CUDA_ERROR(cudaEventSynchronize(event));
  return absl::OkStatus();
}

absl::Status event_elapsed_time(float* ms, cudaEvent_t start, cudaEvent_t end) {
  SC_RETURN_IF_CUDA_ERROR(cudaEventElapsedTime(ms, start, end));
  return absl::OkStatus();
}

// Error handling
absl::Status get_last_error() {
  SC_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  return absl::OkStatus();
}

absl::Status peek_last_error() {
  SC_RETURN_IF_CUDA_ERROR(cudaPeekAtLastError());
  return absl::OkStatus();
}

bool is_fake() {
  return false;
}

bool is_available() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}

// IPC handle operations (native CUDA handle type)
absl::Status get_ipc_mem_handle(cudaIpcMemHandle_t* handle, void* dev_ptr) {
  SC_RETURN_IF_CUDA_ERROR(cudaIpcGetMemHandle(handle, dev_ptr));
  return absl::OkStatus();
}

absl::Status open_ipc_mem_handle(void** dev_ptr, cudaIpcMemHandle_t handle, unsigned int flags) {
  SC_RETURN_IF_CUDA_ERROR(cudaIpcOpenMemHandle(dev_ptr, handle, flags));
  return absl::OkStatus();
}

// Peer access and memcpyPeer wrappers
absl::Status device_can_access_peer(int* can_access, int device, int peer_device) {
  if (can_access == nullptr) {
    return absl::InvalidArgumentError("can_access pointer is null");
  }
  int value = 0;
  cudaError_t err = cudaDeviceCanAccessPeer(&value, device, peer_device);
  if (err != cudaSuccess) {
    return common::cuda_as_status(err, "cudaDeviceCanAccessPeer");
  }
  *can_access = value;
  return absl::OkStatus();
}

absl::Status enable_peer_access(int current_device, int peer_device) {
  SC_RETURN_IF_CUDA_ERROR(cudaSetDevice(current_device));
  cudaError_t err = cudaDeviceEnablePeerAccess(peer_device, 0);
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    {
      cudaError_t _clear = cudaGetLastError();
      if (_clear != cudaSuccess) {
        LOG(ERROR) << "cudaGetLastError after peer-access-already-enabled: " << static_cast<int>(_clear);
      }
    }
    return absl::OkStatus();
  }
  if (err != cudaSuccess) {
    return common::cuda_as_status(err, "cudaDeviceEnablePeerAccess");
  }
  return absl::OkStatus();
}

absl::Status memcpy_peer_async(
    void* dst,
    int dst_device,
    const void* src,
    int src_device,
    size_t bytes,
    cudaStream_t stream) {
  SC_RETURN_IF_CUDA_ERROR(cudaSetDevice(dst_device));
  cudaError_t err = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, stream);
  if (err != cudaSuccess) {
    return common::cuda_as_status(err, "cudaMemcpyPeerAsync");
  }
  return absl::OkStatus();
}

absl::Status close_ipc_mem_handle(void* dev_ptr) {
  SC_RETURN_IF_CUDA_ERROR(cudaIpcCloseMemHandle(dev_ptr));
  return absl::OkStatus();
}

absl::Status host_register(void* ptr, size_t size, unsigned int flags) {
  SC_RETURN_IF_CUDA_ERROR(cudaHostRegister(ptr, size, flags));
  return absl::OkStatus();
}

absl::Status host_unregister(void* ptr) {
  SC_RETURN_IF_CUDA_ERROR(cudaHostUnregister(ptr));
  return absl::OkStatus();
}

} // namespace tensorcast::cuda
