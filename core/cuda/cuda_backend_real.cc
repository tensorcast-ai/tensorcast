// Copyright (c) 2025-2026, TensorCast Team.

#include "core/cuda/cuda_backend.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <sstream>

#include <unistd.h>
#include <cstdlib>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "core/cuda/cuda_driver_api.h"
#include "core/cuda/error_handling.h"

namespace tensorcast::cuda::real_backend {

namespace {
absl::Status ensure_cuda_driver_loaded() {
  return DriverApi::ensure_loaded();
}

absl::Status cu_result_as_status(CUresult result, absl::string_view context) {
  return DriverApi::get().to_status(result, context);
}

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

// Convert native cudaIpcMemHandle_t to lower-hex string (2 chars per byte),
// matching the representation used by get_ipc_handle().
std::string to_hex_string(const cudaIpcMemHandle_t& h) {
  std::stringstream ss;
  for (int i = 0; i < static_cast<int>(sizeof(cudaIpcMemHandle_t)); ++i) {
    ss << std::hex << std::setw(2) << std::setfill('0')
       << static_cast<int>(reinterpret_cast<const unsigned char*>(&h)[i]);
  }
  return ss.str();
}

struct DeviceMemorySnapshot {
  bool ok{false};
  int device_id{-1};
  size_t free_bytes{0};
  size_t total_bytes{0};
  cudaError_t error{cudaSuccess};
};

DeviceMemorySnapshot best_effort_current_device_memory_snapshot() {
  DeviceMemorySnapshot snapshot;
  snapshot.error = cudaGetDevice(&snapshot.device_id);
  if (snapshot.error != cudaSuccess) {
    return snapshot;
  }
  snapshot.error = cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes);
  snapshot.ok = snapshot.error == cudaSuccess;
  return snapshot;
}

long long signed_delta_bytes(size_t after, size_t before) {
  return static_cast<long long>(after) - static_cast<long long>(before);
}
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
    return cuda_as_status(err, "cudaIpcOpenMemHandle");
  }

  // Fallback: if export and open happen in the same process (unit tests),
  // return the original pointer that created the handle.
  {
    absl::MutexLock lock(&g_ipc_map_mu);
    auto it = g_exported_ipc_map.find(handle);
    if (it == g_exported_ipc_map.end()) {
      return cuda_as_status(err, "cudaIpcOpenMemHandle");
    }
    const ExportedIpcInfo& info = it->second;
    // Ensure same process
    if (info.pid != getpid()) {
      return cuda_as_status(err, "cudaIpcOpenMemHandle");
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
  return cuda_as_status(err, "cudaIpcCloseMemHandle");
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

absl::Status mem_get_address_range(void** base, size_t* range_bytes, const void* ptr) {
  if (base == nullptr || range_bytes == nullptr || ptr == nullptr) {
    return absl::InvalidArgumentError("base/range_bytes/ptr must be non-null");
  }

  auto status = ensure_cuda_driver_loaded();
  if (!status.ok()) {
    return status;
  }

  const DriverApi& driver = DriverApi::get();
  CUdeviceptr base_ptr = 0;
  status = cu_result_as_status(
      driver.cuMemGetAddressRange(&base_ptr, range_bytes, reinterpret_cast<CUdeviceptr>(ptr)), "cuMemGetAddressRange");
  if (!status.ok()) {
    return status;
  }

  *base = reinterpret_cast<void*>(base_ptr);
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

absl::Status stream_create_with_priority(cudaStream_t* stream, unsigned int flags, int priority) {
  SC_RETURN_IF_CUDA_ERROR(cudaStreamCreateWithPriority(stream, flags, priority));
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

absl::Status event_query(cudaEvent_t event, bool* ready) {
  if (ready == nullptr) {
    return absl::InvalidArgumentError("ready pointer is null");
  }
  const cudaError_t rc = cudaEventQuery(event);
  if (rc == cudaSuccess) {
    *ready = true;
    return absl::OkStatus();
  }
  if (rc == cudaErrorNotReady) {
    *ready = false;
    return absl::OkStatus();
  }
  return cuda_as_status(rc, "cudaEventQuery");
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
  // Record mapping unconditionally to enable precise same-process open
  // detection (and optional fallback) for native IPC handles.
  {
    absl::MutexLock lock(&g_ipc_map_mu);
    int current_device = -1;
    ABSL_CHECK_OK(get_device(&current_device));
    const std::string key = to_hex_string(*handle);
    g_exported_ipc_map[key] = ExportedIpcInfo{.ptr = dev_ptr, .device_id = current_device, .pid = getpid()};
    g_exported_ptrs.insert(dev_ptr);
  }
  return absl::OkStatus();
}

absl::Status open_ipc_mem_handle(void** dev_ptr, cudaIpcMemHandle_t handle, unsigned int flags) {
  const auto profile_start = std::chrono::steady_clock::now();
  const std::string handle_hex = to_hex_string(handle);
  const auto after_hex = std::chrono::steady_clock::now();
  // For same-process open, optionally fall back to the original exported pointer
  // instead of surfacing the CUDA runtime error. This is required for local
  // daemon flows that route through IPC-shaped metadata even when export and
  // open happen inside the same daemon process.
  {
    absl::MutexLock lock(&g_ipc_map_mu);
    auto it = g_exported_ipc_map.find(handle_hex);
    if (it != g_exported_ipc_map.end() && it->second.pid == getpid()) {
      if (!g_enable_same_process_ipc_fallback.load(std::memory_order_relaxed)) {
        LOG(ERROR) << "cudaIpcOpenMemHandle called on a handle exported in the same process; this is invalid. "
                   << "Use the original device pointer within the exporting process, or a non-IPC path.";
        return absl::FailedPreconditionError(
            "cudaIpcOpenMemHandle called on a handle exported in the same process; this is invalid. "
            "Use the original device pointer within the exporting process, or a non-IPC path.");
      }
      int cur_dev = -1;
      ABSL_CHECK_OK(get_device(&cur_dev));
      cudaPointerAttributes attrs;
      if (cudaPointerGetAttributes(&attrs, it->second.ptr) == cudaSuccess) {
        if (cur_dev >= 0 && attrs.device != cur_dev) {
          return absl::FailedPreconditionError("IPC fallback rejected: device mismatch with current device");
        }
      } else if (cur_dev >= 0 && it->second.device_id >= 0 && it->second.device_id != cur_dev) {
        return absl::FailedPreconditionError("IPC fallback rejected: recorded device mismatch");
      }
      *dev_ptr = it->second.ptr;
      const auto after_fallback = std::chrono::steady_clock::now();
      LOG(INFO) << "tc_profile_cpp cuda.open_ipc_mem_handle timings path=same_process_fallback"
                << " hex_sec=" << std::chrono::duration<double>(after_hex - profile_start).count()
                << " lookup_and_fallback_sec=" << std::chrono::duration<double>(after_fallback - after_hex).count()
                << " total_sec=" << std::chrono::duration<double>(after_fallback - profile_start).count();
      return absl::OkStatus();
    }
  }
  const auto after_lookup = std::chrono::steady_clock::now();
  const DeviceMemorySnapshot before_open_mem = best_effort_current_device_memory_snapshot();
  SC_RETURN_IF_CUDA_ERROR(cudaIpcOpenMemHandle(dev_ptr, handle, flags));
  const auto after_open = std::chrono::steady_clock::now();
  const DeviceMemorySnapshot after_open_mem = best_effort_current_device_memory_snapshot();
  LOG(INFO) << "tc_profile_cpp cuda.open_ipc_mem_handle timings path=cuda_runtime"
            << " hex_sec=" << std::chrono::duration<double>(after_hex - profile_start).count()
            << " lookup_sec=" << std::chrono::duration<double>(after_lookup - after_hex).count()
            << " cuda_ipc_open_sec=" << std::chrono::duration<double>(after_open - after_lookup).count()
            << " total_sec=" << std::chrono::duration<double>(after_open - profile_start).count() << " flags=" << flags
            << " mem_before_ok=" << before_open_mem.ok << " mem_after_ok=" << after_open_mem.ok
            << " device_before=" << before_open_mem.device_id << " device_after=" << after_open_mem.device_id
            << " free_before_bytes=" << before_open_mem.free_bytes
            << " free_after_bytes=" << after_open_mem.free_bytes
            << " free_delta_bytes="
            << (before_open_mem.ok && after_open_mem.ok
                    ? signed_delta_bytes(after_open_mem.free_bytes, before_open_mem.free_bytes)
                    : 0)
            << " total_before_bytes=" << before_open_mem.total_bytes
            << " total_after_bytes=" << after_open_mem.total_bytes
            << " mem_before_error=" << cudaGetErrorName(before_open_mem.error)
            << " mem_after_error=" << cudaGetErrorName(after_open_mem.error);
  VLOG(1) << "open_ipc_mem_handle: handle=0x" << handle_hex << " ptr=" << *dev_ptr << " flags=" << flags;
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
    return cuda_as_status(err, "cudaDeviceCanAccessPeer");
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
    return cuda_as_status(err, "cudaDeviceEnablePeerAccess");
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
    return cuda_as_status(err, "cudaMemcpyPeerAsync");
  }
  return absl::OkStatus();
}

absl::Status cu_init(unsigned int flags) {
  auto status = ensure_cuda_driver_loaded();
  if (!status.ok()) {
    return status;
  }
  const DriverApi& driver = DriverApi::get();
  return cu_result_as_status(driver.cuInit(flags), "cuInit");
}

absl::Status cu_device_get(CUdevice* device, int ordinal) {
  if (device == nullptr) {
    return absl::InvalidArgumentError("device pointer is null");
  }
  auto status = ensure_cuda_driver_loaded();
  if (!status.ok()) {
    return status;
  }
  const DriverApi& driver = DriverApi::get();
  return cu_result_as_status(driver.cuDeviceGet(device, ordinal), "cuDeviceGet");
}

absl::Status cu_device_get_attribute(int* value, CUdevice_attribute attribute, CUdevice device) {
  if (value == nullptr) {
    return absl::InvalidArgumentError("value pointer is null");
  }
  auto status = ensure_cuda_driver_loaded();
  if (!status.ok()) {
    return status;
  }
  const DriverApi& driver = DriverApi::get();
  return cu_result_as_status(driver.cuDeviceGetAttribute(value, attribute, device), "cuDeviceGetAttribute");
}

absl::Status cu_mem_get_allocation_granularity(
    size_t* granularity,
    const CUmemAllocationProp* prop,
    CUmemAllocationGranularity_flags option) {
  if (granularity == nullptr) {
    return absl::InvalidArgumentError("granularity pointer is null");
  }
  if (prop == nullptr) {
    return absl::InvalidArgumentError("prop pointer is null");
  }
  auto status = ensure_cuda_driver_loaded();
  if (!status.ok()) {
    return status;
  }
  const DriverApi& driver = DriverApi::get();
  return cu_result_as_status(
      driver.cuMemGetAllocationGranularity(granularity, prop, option), "cuMemGetAllocationGranularity");
}

absl::Status cu_mem_address_reserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, uint64_t flags) {
  if (ptr == nullptr) {
    return absl::InvalidArgumentError("ptr is null");
  }
  auto status = ensure_cuda_driver_loaded();
  if (!status.ok()) {
    return status;
  }
  const DriverApi& driver = DriverApi::get();
  return cu_result_as_status(driver.cuMemAddressReserve(ptr, size, alignment, addr, flags), "cuMemAddressReserve");
}

absl::Status cu_mem_create(
    CUmemGenericAllocationHandle* handle,
    size_t size,
    const CUmemAllocationProp* prop,
    uint64_t flags) {
  if (handle == nullptr) {
    return absl::InvalidArgumentError("handle is null");
  }
  if (prop == nullptr) {
    return absl::InvalidArgumentError("prop is null");
  }
  auto status = ensure_cuda_driver_loaded();
  if (!status.ok()) {
    return status;
  }
  const DriverApi& driver = DriverApi::get();
  return cu_result_as_status(driver.cuMemCreate(handle, size, prop, flags), "cuMemCreate");
}

absl::Status cu_mem_map(
    CUdeviceptr ptr,
    size_t size,
    size_t offset,
    CUmemGenericAllocationHandle handle,
    uint64_t flags) {
  auto status = ensure_cuda_driver_loaded();
  if (!status.ok()) {
    return status;
  }
  const DriverApi& driver = DriverApi::get();
  return cu_result_as_status(driver.cuMemMap(ptr, size, offset, handle, flags), "cuMemMap");
}

absl::Status cu_mem_set_access(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) {
  if (desc == nullptr && count != 0) {
    return absl::InvalidArgumentError("desc is null but count is non-zero");
  }
  auto status = ensure_cuda_driver_loaded();
  if (!status.ok()) {
    return status;
  }
  const DriverApi& driver = DriverApi::get();
  return cu_result_as_status(driver.cuMemSetAccess(ptr, size, desc, count), "cuMemSetAccess");
}

absl::Status cu_mem_unmap(CUdeviceptr ptr, size_t size) {
  auto status = ensure_cuda_driver_loaded();
  if (!status.ok()) {
    return status;
  }
  const DriverApi& driver = DriverApi::get();
  return cu_result_as_status(driver.cuMemUnmap(ptr, size), "cuMemUnmap");
}

absl::Status cu_mem_release(CUmemGenericAllocationHandle handle) {
  auto status = ensure_cuda_driver_loaded();
  if (!status.ok()) {
    return status;
  }
  const DriverApi& driver = DriverApi::get();
  return cu_result_as_status(driver.cuMemRelease(handle), "cuMemRelease");
}

absl::Status cu_mem_address_free(CUdeviceptr ptr, size_t size) {
  auto status = ensure_cuda_driver_loaded();
  if (!status.ok()) {
    return status;
  }
  const DriverApi& driver = DriverApi::get();
  return cu_result_as_status(driver.cuMemAddressFree(ptr, size), "cuMemAddressFree");
}

absl::Status cu_mem_get_handle_for_address_range(
    void* handle,
    CUdeviceptr dptr,
    size_t size,
    CUmemRangeHandleType handle_type,
    unsigned long long flags) {
  if (handle == nullptr) {
    return absl::InvalidArgumentError("handle is null");
  }
  if (dptr == 0 || size == 0) {
    return absl::InvalidArgumentError("dptr/size invalid");
  }

  auto status = ensure_cuda_driver_loaded();
  if (!status.ok()) {
    return status;
  }
  const DriverApi& driver = DriverApi::get();
  return cu_result_as_status(
      driver.cuMemGetHandleForAddressRange(handle, dptr, size, handle_type, flags), "cuMemGetHandleForAddressRange");
}

absl::Status close_ipc_mem_handle(void* dev_ptr) {
  const cudaError_t err = cudaIpcCloseMemHandle(dev_ptr);
  if (err != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return cuda_as_status(err, "cudaIpcCloseMemHandle");
  }
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

} // namespace tensorcast::cuda::real_backend

namespace tensorcast::cuda {

#define TENSORCAST_DEFINE_REAL_BACKEND(return_type, name, args, ...) \
  return_type RealCudaBackend::name args {                           \
    return real_backend::name(__VA_ARGS__);                          \
  }

TENSORCAST_CUDA_BACKEND_FUNCTIONS(TENSORCAST_DEFINE_REAL_BACKEND)

#undef TENSORCAST_DEFINE_REAL_BACKEND

} // namespace tensorcast::cuda
