// Copyright (c) 2025, TensorCast Team.

#include "core/common/cuda_api.h"

#include <cstring>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"

// NOLINTBEGIN

namespace tensorcast::cuda {
namespace {

// Fake GPU memory allocation structure
struct FakeAllocation {
  std::unique_ptr<uint8_t[]> buffer;
  size_t size;
  bool is_pinned;
  int device_id; // device where the allocation was performed
};

// Global state for fake CUDA runtime
struct FakeCudaState {
  absl::Mutex mutex;

  // Memory allocations indexed by pointer
  absl::flat_hash_map<void*, FakeAllocation> allocations ABSL_GUARDED_BY(mutex);

  // IPC handle mapping (handle -> pointer)
  absl::flat_hash_map<std::string, void*> ipc_handles ABSL_GUARDED_BY(mutex);

  // Current device
  int current_device ABSL_GUARDED_BY(mutex) = 0;

  // Simulated device properties
  static constexpr int kNumDevices = 4;
  static constexpr size_t kDeviceMemorySize = 2ULL * 1024 * 1024 * 1024; // 2GB
  size_t allocated_bytes[kNumDevices] ABSL_GUARDED_BY(mutex) = {0};
};

FakeCudaState& get_state() {
  static FakeCudaState state;
  return state;
}

std::string generate_random_handle() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 255);

  std::stringstream ss;
  for (int i = 0; i < 64; ++i) { // 64 bytes for CUDA IPC handle
    ss << std::hex << std::setw(2) << std::setfill('0') << dis(gen);
  }
  return ss.str();
}

} // namespace

// Thread-local current device to avoid cross-thread interference
static thread_local int tls_current_device = 0;

absl::Status set_device(int device_id) {
  if (device_id < 0 || device_id >= FakeCudaState::kNumDevices) {
    return absl::InvalidArgumentError(absl::StrCat("Invalid device ID: ", device_id));
  }
  tls_current_device = device_id;
  return absl::OkStatus();
}

absl::Status get_device(int* device_id) {
  if (device_id == nullptr) {
    return absl::InvalidArgumentError("device_id is null");
  }
  *device_id = tls_current_device;
  return absl::OkStatus();
}

absl::Status malloc(void** ptr, size_t bytes) {
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  const int device_id = tls_current_device;

  // Check if we have enough "GPU memory"
  if (state.allocated_bytes[device_id] + bytes > FakeCudaState::kDeviceMemorySize) {
    return absl::ResourceExhaustedError("Out of GPU memory");
  }

  // Allocate CPU memory to simulate GPU memory
  auto buffer = std::make_unique<uint8_t[]>(bytes);
  void* raw_ptr = buffer.get();

  state.allocations[raw_ptr] =
      FakeAllocation{.buffer = std::move(buffer), .size = bytes, .is_pinned = false, .device_id = device_id};
  state.allocated_bytes[device_id] += bytes;

  *ptr = raw_ptr;
  return absl::OkStatus();
}

absl::Status malloc_host(void** ptr, size_t bytes) {
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  // Allocate pinned memory (just regular memory in fake backend)
  auto buffer = std::make_unique<uint8_t[]>(bytes);
  void* raw_ptr = buffer.get();

  state.allocations[raw_ptr] =
      FakeAllocation{.buffer = std::move(buffer), .size = bytes, .is_pinned = true, .device_id = tls_current_device};

  *ptr = raw_ptr;
  return absl::OkStatus();
}

absl::Status free(void* ptr) {
  if (ptr == nullptr) {
    return absl::OkStatus();
  }

  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  auto it = state.allocations.find(ptr);
  if (it == state.allocations.end()) {
    return absl::InvalidArgumentError("Pointer not found in allocations");
  }

  if (!it->second.is_pinned) {
    const int dev = it->second.device_id;
    size_t& used = state.allocated_bytes[dev];
    used = (used >= it->second.size) ? (used - it->second.size) : 0;
  }

  state.allocations.erase(it);
  return absl::OkStatus();
}

absl::Status free_host(void* ptr) {
  // Same as free() in fake backend
  return free(ptr);
}

absl::Status memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
  // In fake backend, all memory is CPU memory, so just use std::memcpy
  std::memcpy(dst, src, bytes);
  return absl::OkStatus();
}

absl::Status memcpy_async(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
  // In fake backend, async is the same as sync
  return memcpy(dst, src, bytes, kind);
}

absl::Status memset(void* ptr, int value, size_t bytes) {
  std::memset(ptr, value, bytes);
  return absl::OkStatus();
}

absl::Status get_device_count(int* count) {
  *count = FakeCudaState::kNumDevices;
  return absl::OkStatus();
}

absl::Status get_memory_info(size_t* free_bytes, size_t* total_bytes, int device_id) {
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  if (device_id < 0 || device_id >= FakeCudaState::kNumDevices) {
    return absl::InvalidArgumentError(absl::StrCat("Invalid device ID: ", device_id));
  }

  *total_bytes = FakeCudaState::kDeviceMemorySize;
  *free_bytes = FakeCudaState::kDeviceMemorySize - state.allocated_bytes[device_id];
  return absl::OkStatus();
}

absl::StatusOr<std::string> get_device_name(int device_id) {
  if (device_id < 0 || device_id >= FakeCudaState::kNumDevices) {
    return absl::InvalidArgumentError(absl::StrCat("Invalid device ID: ", device_id));
  }
  return "FakeCuda GPU 0";
}

absl::Status get_ipc_handle(const void* ptr, std::string* handle) {
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  // Check if pointer is a valid allocation
  auto it = state.allocations.find(const_cast<void*>(ptr));
  if (it == state.allocations.end()) {
    return absl::InvalidArgumentError("Pointer not found in allocations");
  }

  // Generate a random handle and store the mapping
  std::string new_handle = generate_random_handle();
  state.ipc_handles[new_handle] = const_cast<void*>(ptr);
  *handle = new_handle;

  return absl::OkStatus();
}

absl::Status open_ipc_handle(const std::string& handle, void** ptr) {
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  auto it = state.ipc_handles.find(handle);
  if (it == state.ipc_handles.end()) {
    return absl::NotFoundError("IPC handle not found");
  }

  *ptr = it->second;
  return absl::OkStatus();
}

absl::Status close_ipc_handle(void* ptr) {
  // In fake backend, nothing to do
  return absl::OkStatus();
}

absl::Status device_synchronize() {
  // In fake backend, all operations are synchronous
  return absl::OkStatus();
}

absl::Status memset_async(void* ptr, int value, size_t bytes, cudaStream_t stream) {
  // In fake backend, async is same as sync
  return memset(ptr, value, bytes);
}

absl::Status get_device_properties(int device_id, void* prop) {
  if (prop == nullptr) {
    return absl::InvalidArgumentError("prop is null");
  }

  if (device_id < 0 || device_id >= FakeCudaState::kNumDevices) {
    return absl::InvalidArgumentError(absl::StrCat("Invalid device ID: ", device_id));
  }

  // Populate a minimal set of device properties for the fake backend.
  cudaDeviceProp* dev_prop = static_cast<cudaDeviceProp*>(prop);
  std::memset(dev_prop, 0, sizeof(cudaDeviceProp));

  std::string name = absl::StrCat("FakeCuda GPU ", device_id);
  std::strncpy(dev_prop->name, name.c_str(), sizeof(dev_prop->name) - 1);
  dev_prop->name[sizeof(dev_prop->name) - 1] = '\0';

  dev_prop->totalGlobalMem = FakeCudaState::kDeviceMemorySize;
  dev_prop->major = 7; // Arbitrary compute capability
  dev_prop->minor = 5;

  // Fake but deterministic PCI information so that Communicator code can derive path.
  dev_prop->pciBusID = 0x01 + device_id; // 0x01, 0x02, ...
  dev_prop->pciDeviceID = 0x00 + device_id; // 0x00, 0x01, ...
  dev_prop->pciDomainID = 0x0000; // Single domain

  // Generate a fake but deterministic UUID for the device
  std::memset(&dev_prop->uuid, 0, sizeof(dev_prop->uuid));
  // Use a simple pattern: GPU-<device_id> with zeros padding
  const char* prefix = "GPU-";
  std::memcpy(dev_prop->uuid.bytes, prefix, 4);
  dev_prop->uuid.bytes[4] = static_cast<char>(device_id);
  // Rest of the bytes remain zero

  return absl::OkStatus();
}

absl::Status pointer_get_attributes(void* ptr, int* device, void** device_ptr) {
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  auto it = state.allocations.find(ptr);
  if (it == state.allocations.end()) {
    return absl::InvalidArgumentError("Pointer not found in allocations");
  }

  if (device)
    *device = it->second.device_id;
  if (device_ptr)
    *device_ptr = ptr; // In fake backend, host and device pointers are the same
  return absl::OkStatus();
}

absl::Status pointer_get_attributes_full(void* ptr, cudaPointerAttributes* attrs) {
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  auto it = state.allocations.find(ptr);
  if (it == state.allocations.end()) {
    return absl::InvalidArgumentError("Pointer not found in allocations");
  }

  // Fill in attributes for fake backend
  attrs->devicePointer = ptr;
  attrs->hostPointer = ptr;
  attrs->device = it->second.device_id;
  attrs->type = cudaMemoryTypeDevice;
#ifdef USE_FAKE_CUDA
  // These legacy fields are only present in the fake CUDA runtime we define
  attrs->memoryType = cudaMemoryTypeDevice;
  attrs->isManaged = 0;
#endif

  return absl::OkStatus();
}

// Stream management
absl::Status stream_create(cudaStream_t* stream) {
  // Create a fake stream - just use a unique pointer value
  static int stream_counter = 1;
  *stream = reinterpret_cast<cudaStream_t>(static_cast<intptr_t>(stream_counter++));
  return absl::OkStatus();
}

absl::Status stream_create_with_flags(cudaStream_t* stream, unsigned int flags) {
  // Create a fake stream - just use a unique pointer value
  // Ignore flags for fake implementation
  static int stream_counter = 1000; // Start from different value to distinguish
  *stream = reinterpret_cast<cudaStream_t>(static_cast<intptr_t>(stream_counter++));
  return absl::OkStatus();
}

absl::Status stream_destroy(cudaStream_t stream) {
  // Nothing to do in fake backend
  return absl::OkStatus();
}

absl::Status stream_synchronize(cudaStream_t stream) {
  // In fake backend, all operations are synchronous
  return absl::OkStatus();
}

absl::Status stream_wait_event(cudaStream_t stream, cudaEvent_t event) {
  // In fake backend, all operations are synchronous
  return absl::OkStatus();
}

absl::Status stream_add_callback(
    cudaStream_t stream,
    void (*callback)(cudaStream_t, cudaError_t, void*),
    void* user_data,
    unsigned int flags) {
  // In fake backend, execute callback immediately
  if (callback) {
    callback(stream, cudaSuccess, user_data);
  }
  return absl::OkStatus();
}

absl::Status launch_host_func(cudaStream_t stream, void (*func)(void*), void* user_data) {
  // In fake backend, execute function immediately
  if (func) {
    func(user_data);
  }
  return absl::OkStatus();
}

// Event management
absl::Status event_create(cudaEvent_t* event) {
  // Create a fake event - just use a unique pointer value
  static int event_counter = 1000; // Start from 1000 to differentiate from streams
  *event = reinterpret_cast<cudaEvent_t>(static_cast<intptr_t>(event_counter++));
  return absl::OkStatus();
}

absl::Status event_create_with_flags(cudaEvent_t* event, unsigned int flags) {
  // In fake backend, flags are ignored
  return event_create(event);
}

absl::Status event_destroy(cudaEvent_t event) {
  // Nothing to do in fake backend
  return absl::OkStatus();
}

absl::Status event_record(cudaEvent_t event, cudaStream_t stream) {
  // In fake backend, events are instantly recorded
  return absl::OkStatus();
}

absl::Status event_synchronize(cudaEvent_t event) {
  // In fake backend, all operations are synchronous
  return absl::OkStatus();
}

absl::Status event_elapsed_time(float* ms, cudaEvent_t start, cudaEvent_t end) {
  // Return a fake elapsed time
  *ms = 0.1f; // Always report 0.1ms
  return absl::OkStatus();
}

// Error handling
absl::Status get_last_error() {
  // No errors in fake backend
  return absl::OkStatus();
}

absl::Status peek_last_error() {
  // No errors in fake backend
  return absl::OkStatus();
}

bool is_fake() {
  return true;
}

bool is_available() {
  return true; // Fake backend is always available
}

void configure_same_process_ipc_fallback(bool /*enabled*/) {
  // No-op for fake backend; semantics already allow same-process opens
}

// IPC handle operations (native CUDA handle type)
absl::Status get_ipc_mem_handle(cudaIpcMemHandle_t* handle, void* dev_ptr) {
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  // Check if pointer is a valid allocation
  auto it = state.allocations.find(dev_ptr);
  if (it == state.allocations.end()) {
    return absl::InvalidArgumentError("Pointer not found in allocations");
  }

  // Generate a unique handle by using the pointer address
  std::memset(handle, 0, sizeof(cudaIpcMemHandle_t));
  // Store pointer address in the first 8 bytes of the handle
  std::memcpy(handle->reserved, &dev_ptr, sizeof(void*));

  return absl::OkStatus();
}

absl::Status open_ipc_mem_handle(void** dev_ptr, cudaIpcMemHandle_t handle, unsigned int flags) {
  // In fake backend, we can only open handles within the same process
  void* original_ptr = nullptr;
  std::memcpy(&original_ptr, handle.reserved, sizeof(void*));

  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  // Check if the original pointer is still valid
  auto it = state.allocations.find(original_ptr);
  if (it == state.allocations.end()) {
    return absl::NotFoundError("IPC handle refers to invalid memory");
  }

  *dev_ptr = original_ptr;
  return absl::OkStatus();
}

absl::Status close_ipc_mem_handle(void* dev_ptr) {
  // In fake backend, nothing to do
  return absl::OkStatus();
}

absl::Status host_register(void* ptr, size_t size, unsigned int flags) {
  // In fake backend, host memory registration is a no-op
  return absl::OkStatus();
}

absl::Status host_unregister(void* ptr) {
  // In fake backend, host memory unregistration is a no-op
  return absl::OkStatus();
}

} // namespace tensorcast::cuda

// NOLINTEND