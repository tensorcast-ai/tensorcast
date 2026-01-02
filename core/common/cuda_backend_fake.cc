// Copyright (c) 2025-2026, TensorCast Team.

#include "core/common/cuda_backend.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

// NOLINTBEGIN

namespace tensorcast::cuda::fake_backend {
namespace {

struct StreamTask {
  uint64_t id = 0;
  std::function<void()> work;
  absl::Duration delay = absl::ZeroDuration();
  const char* label = nullptr;
};

class FakeStream {
 public:
  FakeStream();
  ~FakeStream();

  absl::StatusOr<uint64_t> Enqueue(StreamTask task);
  void WaitUntilIdle();
  void Stop();

 private:
  void RunLoop();

  absl::Mutex mu_;
  absl::CondVar cv_;
  absl::CondVar idle_cv_;
  std::deque<StreamTask> tasks_ ABSL_GUARDED_BY(mu_);
  bool stop_ ABSL_GUARDED_BY(mu_) = false;
  uint64_t next_task_id_ ABSL_GUARDED_BY(mu_) = 1;
  uint64_t last_completed_id_ ABSL_GUARDED_BY(mu_) = 0;
  std::thread worker_;
};

struct FakeEvent {
  void ResetForRecord();
  void MarkCompleted();
  void MarkDestroyed();
  absl::Status Wait();

  absl::Mutex mu;
  absl::CondVar cv;
  bool recorded ABSL_GUARDED_BY(mu) = false;
  bool completed ABSL_GUARDED_BY(mu) = false;
  bool destroyed ABSL_GUARDED_BY(mu) = false;
  absl::Time record_time ABSL_GUARDED_BY(mu);
  absl::Time complete_time ABSL_GUARDED_BY(mu);
};

// Fake GPU memory allocation structure
struct FakeAllocation {
  void* ptr = nullptr;
  size_t size = 0;
  bool is_pinned = false;
  int device_id = 0; // device where the allocation was performed
  std::unique_ptr<uint8_t[]> owned_buffer; // used for host/pinned allocations
  std::string shm_name; // used for device allocations
  bool uses_shm = false;
};

struct IpcMapping {
  void* mapped = nullptr;
  size_t size = 0;
  std::string shm_name;
  int fd = -1;
};

struct FakeVmmAllocation {
  std::string shm_name;
  int fd = -1;
  size_t size = 0;
};

// Global state for fake CUDA runtime
struct FakeCudaState {
  absl::Mutex mutex;

  // Memory allocations indexed by pointer
  absl::flat_hash_map<void*, FakeAllocation> allocations ABSL_GUARDED_BY(mutex);

  // IPC handle mapping (handle -> pointer)
  absl::flat_hash_map<std::string, void*> ipc_handles ABSL_GUARDED_BY(mutex);

  // Cross-process IPC mappings opened in the current process (ptr -> metadata)
  absl::flat_hash_map<void*, IpcMapping> ipc_mappings ABSL_GUARDED_BY(mutex);

  // Fake CUDA VMM allocations: handle -> backing store
  absl::flat_hash_map<CUmemGenericAllocationHandle, FakeVmmAllocation> vmm_allocations ABSL_GUARDED_BY(mutex);
  CUmemGenericAllocationHandle next_vmm_handle ABSL_GUARDED_BY(mutex) = 1;

  // Streams and events for async simulation
  std::shared_ptr<FakeStream> default_stream ABSL_GUARDED_BY(mutex);
  absl::flat_hash_map<cudaStream_t, std::shared_ptr<FakeStream>> streams ABSL_GUARDED_BY(mutex);
  absl::flat_hash_map<cudaEvent_t, std::shared_ptr<FakeEvent>> events ABSL_GUARDED_BY(mutex);

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

constexpr size_t kHandleSizeBytes = sizeof(cudaIpcMemHandle_t);
constexpr size_t kHandleSizeFieldBytes = sizeof(std::uint64_t);
constexpr size_t kHandleNameBytes = kHandleSizeBytes - kHandleSizeFieldBytes;

std::string make_shm_name() {
  // Limit the name to what fits in cudaIpcMemHandle_t after the size prefix.
  std::string suffix = generate_random_handle().substr(0, 24); // 24 hex chars
  return absl::StrCat("/tcfake_", suffix);
}

absl::Status encode_shm_handle(cudaIpcMemHandle_t* handle, absl::string_view shm_name, size_t size) {
  if (handle == nullptr) {
    return absl::InvalidArgumentError("handle is null");
  }
  if (shm_name.empty()) {
    return absl::InvalidArgumentError("shared memory name is empty");
  }
  if (shm_name.size() >= kHandleNameBytes) {
    return absl::InvalidArgumentError("shared memory name too long for cudaIpcMemHandle_t");
  }

  std::memset(handle, 0, sizeof(cudaIpcMemHandle_t));
  std::memcpy(handle->reserved, &size, sizeof(std::uint64_t));
  std::memcpy(handle->reserved + sizeof(std::uint64_t), shm_name.data(), shm_name.size());
  return absl::OkStatus();
}

absl::StatusOr<std::pair<std::string, size_t>> decode_shm_handle(const cudaIpcMemHandle_t& handle) {
  size_t size = 0;
  std::memcpy(&size, handle.reserved, sizeof(std::uint64_t));
  if (size == 0) {
    return absl::InvalidArgumentError("decoded IPC size is zero");
  }

  char name_buf[kHandleNameBytes + 1];
  std::memset(name_buf, 0, sizeof(name_buf));
  std::memcpy(name_buf, handle.reserved + sizeof(std::uint64_t), kHandleNameBytes);
  std::string shm_name(name_buf);
  if (shm_name.empty()) {
    return absl::InvalidArgumentError("decoded shared memory name is empty");
  }

  return std::make_pair(shm_name, size);
}

FakeStream::FakeStream() {
  worker_ = std::thread(&FakeStream::RunLoop, this);
}

FakeStream::~FakeStream() {
  Stop();
}

absl::StatusOr<uint64_t> FakeStream::Enqueue(StreamTask task) {
  absl::MutexLock lock(&mu_);
  if (stop_) {
    return absl::FailedPreconditionError("FakeStream already stopped");
  }
  task.id = next_task_id_++;
  tasks_.push_back(std::move(task));
  cv_.Signal();
  return task.id;
}

void FakeStream::WaitUntilIdle() {
  absl::MutexLock lock(&mu_);
  const uint64_t target = next_task_id_ - 1;
  while ((last_completed_id_ < target) || !tasks_.empty()) {
    idle_cv_.Wait(&mu_);
  }
}

void FakeStream::Stop() {
  {
    absl::MutexLock lock(&mu_);
    if (stop_) {
      return;
    }
    stop_ = true;
    cv_.SignalAll();
  }
  if (worker_.joinable()) {
    worker_.join();
  }
}

void FakeStream::RunLoop() {
  mu_.Lock();
  while (true) {
    while (!stop_ && tasks_.empty()) {
      idle_cv_.SignalAll();
      cv_.Wait(&mu_);
    }
    if (stop_ && tasks_.empty()) {
      idle_cv_.SignalAll();
      mu_.Unlock();
      break;
    }
    StreamTask task = std::move(tasks_.front());
    tasks_.pop_front();
    mu_.Unlock();

    if (task.delay > absl::ZeroDuration()) {
      absl::SleepFor(task.delay);
    }
    if (task.label) {
      VLOG(3) << "FakeCuda stream executing task '" << task.label << "'";
    }
    if (task.work) {
      task.work();
    }

    mu_.Lock();
    last_completed_id_ = task.id;
    idle_cv_.SignalAll();
  }
}

void FakeEvent::ResetForRecord() {
  absl::MutexLock lock(&mu);
  recorded = true;
  destroyed = false;
  completed = false;
  record_time = absl::Now();
}

void FakeEvent::MarkCompleted() {
  absl::MutexLock lock(&mu);
  completed = true;
  complete_time = absl::Now();
  cv.SignalAll();
}

void FakeEvent::MarkDestroyed() {
  absl::MutexLock lock(&mu);
  destroyed = true;
  cv.SignalAll();
}

absl::Status FakeEvent::Wait() {
  absl::MutexLock lock(&mu);
  while (!completed && !destroyed) {
    cv.Wait(&mu);
  }
  if (destroyed && !completed) {
    return absl::FailedPreconditionError("fake CUDA event destroyed before completion");
  }
  return absl::OkStatus();
}

absl::Duration simulate_async_delay(size_t bytes) {
  if (bytes == 0) {
    return absl::Microseconds(10);
  }
  constexpr double kEffectiveBandwidthBytesPerSec = 12.0 * 1024 * 1024 * 1024; // 12 GiB/s
  const double seconds = static_cast<double>(bytes) / kEffectiveBandwidthBytesPerSec;
  absl::Duration delay = absl::Seconds(seconds);
  if (delay < absl::Microseconds(10)) {
    delay = absl::Microseconds(10);
  }
  if (delay > absl::Milliseconds(2)) {
    delay = absl::Milliseconds(2);
  }
  return delay;
}

std::shared_ptr<FakeStream> ensure_default_stream_locked(FakeCudaState& state)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mutex) {
  if (!state.default_stream) {
    state.default_stream = std::make_shared<FakeStream>();
  }
  return state.default_stream;
}

absl::StatusOr<std::shared_ptr<FakeStream>> get_stream_state(cudaStream_t stream) {
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);
  if (stream == nullptr) {
    return ensure_default_stream_locked(state);
  }
  auto it = state.streams.find(stream);
  if (it == state.streams.end()) {
    return absl::InvalidArgumentError("invalid fake CUDA stream handle");
  }
  return it->second;
}

absl::StatusOr<std::shared_ptr<FakeEvent>> get_event_state(cudaEvent_t event) {
  if (event == nullptr) {
    return absl::InvalidArgumentError("invalid fake CUDA event handle");
  }
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);
  auto it = state.events.find(event);
  if (it == state.events.end()) {
    return absl::InvalidArgumentError("invalid fake CUDA event handle");
  }
  return it->second;
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
  if (bytes == 0) {
    if (ptr != nullptr) {
      *ptr = nullptr;
    }
    return absl::OkStatus();
  }

  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  const int device_id = tls_current_device;

  // Check if we have enough "GPU memory"
  if (state.allocated_bytes[device_id] + bytes > FakeCudaState::kDeviceMemorySize) {
    return absl::ResourceExhaustedError("Out of GPU memory");
  }

  const std::string shm_name = make_shm_name();
  int fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, "shm_open failed in fake cuda malloc");
  }

  if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    const absl::Status err = absl::ErrnoToStatus(errno, "ftruncate failed in fake cuda malloc");
    close(fd);
    shm_unlink(shm_name.c_str());
    return err;
  }

  void* mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) {
    const absl::Status err = absl::ErrnoToStatus(errno, "mmap failed in fake cuda malloc");
    close(fd);
    shm_unlink(shm_name.c_str());
    return err;
  }

  if (close(fd) != 0) {
    const absl::Status err = absl::ErrnoToStatus(errno, "close failed in fake cuda malloc");
    munmap(mapped, bytes);
    shm_unlink(shm_name.c_str());
    return err;
  }

  std::memset(mapped, 0, bytes);

  FakeAllocation alloc;
  alloc.ptr = mapped;
  alloc.size = bytes;
  alloc.is_pinned = false;
  alloc.device_id = device_id;
  alloc.shm_name = shm_name;
  alloc.uses_shm = true;

  state.allocations[mapped] = std::move(alloc);
  state.allocated_bytes[device_id] += bytes;

  *ptr = mapped;
  return absl::OkStatus();
}

absl::Status malloc_host(void** ptr, size_t bytes) {
  if (bytes == 0) {
    if (ptr != nullptr) {
      *ptr = nullptr;
    }
    return absl::OkStatus();
  }

  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  // Allocate pinned memory (just regular memory in fake backend)
  auto buffer = std::make_unique<uint8_t[]>(bytes);
  void* raw_ptr = buffer.get();

  state.allocations[raw_ptr] = FakeAllocation{
      .ptr = raw_ptr,
      .size = bytes,
      .is_pinned = true,
      .device_id = tls_current_device,
      .owned_buffer = std::move(buffer),
      .uses_shm = false};

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

  absl::Status status = absl::OkStatus();
  if (it->second.uses_shm) {
    if (munmap(it->second.ptr, it->second.size) != 0) {
      status = absl::ErrnoToStatus(errno, "munmap failed in fake cuda free");
    }
    if (!it->second.shm_name.empty()) {
      if (shm_unlink(it->second.shm_name.c_str()) != 0 && status.ok()) {
        status = absl::ErrnoToStatus(errno, "shm_unlink failed in fake cuda free");
      }
    }
  }

  state.allocations.erase(it);
  return status;
}

absl::Status free_host(void* ptr) {
  // Same as free() in fake backend
  return free(ptr);
}

absl::Status memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
  static_cast<void>(kind);
  if (bytes == 0) {
    return absl::OkStatus();
  }
  if (bytes > 0 && (dst == nullptr || src == nullptr)) {
    return absl::InvalidArgumentError("null pointer passed to fake cudaMemcpy");
  }
  // In fake backend, all memory is CPU memory, so just use std::memcpy
  std::memcpy(dst, src, bytes);
  return absl::OkStatus();
}

absl::Status memcpy_async(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
  static_cast<void>(kind);
  if (bytes == 0) {
    auto stream_or = get_stream_state(stream);
    if (!stream_or.ok()) {
      return stream_or.status();
    }
    return absl::OkStatus();
  }
  if (bytes > 0 && (dst == nullptr || src == nullptr)) {
    return absl::InvalidArgumentError("null pointer passed to fake cudaMemcpyAsync");
  }
  auto stream_or = get_stream_state(stream);
  if (!stream_or.ok()) {
    return stream_or.status();
  }
  auto stream_ptr = *stream_or;

  StreamTask task;
  task.label = "memcpy_async";
  task.delay = simulate_async_delay(bytes);
  task.work = [dst, src, bytes]() { std::memcpy(dst, src, bytes); };

  auto enqueue_or = stream_ptr->Enqueue(std::move(task));
  if (!enqueue_or.ok()) {
    return enqueue_or.status();
  }

  return absl::OkStatus();
}

absl::Status memset(void* ptr, int value, size_t bytes) {
  if (bytes == 0) {
    return absl::OkStatus();
  }
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
  auto& state = get_state();
  std::vector<std::shared_ptr<FakeStream>> streams;
  {
    absl::MutexLock lock(&state.mutex);
    if (state.default_stream) {
      streams.push_back(state.default_stream);
    }
    streams.reserve(streams.size() + state.streams.size());
    for (auto& entry : state.streams) {
      streams.push_back(entry.second);
    }
  }

  for (auto& stream : streams) {
    if (stream) {
      stream->WaitUntilIdle();
    }
  }

  return absl::OkStatus();
}

absl::Status memset_async(void* ptr, int value, size_t bytes, cudaStream_t stream) {
  if (bytes > 0 && ptr == nullptr) {
    return absl::InvalidArgumentError("null pointer passed to fake cudaMemsetAsync");
  }
  auto stream_or = get_stream_state(stream);
  if (!stream_or.ok()) {
    return stream_or.status();
  }
  auto stream_ptr = *stream_or;

  StreamTask task;
  task.label = "memset_async";
  task.delay = simulate_async_delay(bytes);
  task.work = [ptr, value, bytes]() {
    VLOG(2) << "FakeCuda memset_async executing memset of " << bytes << " bytes";
    std::memset(ptr, value, bytes);
  };

  auto enqueue_or = stream_ptr->Enqueue(std::move(task));
  if (!enqueue_or.ok()) {
    return enqueue_or.status();
  }

  return absl::OkStatus();
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

absl::Status pointer_get_attributes_full(const void* ptr, cudaPointerAttributes* attrs) {
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  auto it = state.allocations.find(const_cast<void*>(ptr));
  if (it == state.allocations.end()) {
    return absl::InvalidArgumentError("Pointer not found in allocations");
  }

  // Fill in attributes for fake backend
  attrs->devicePointer = const_cast<void*>(ptr);
  attrs->hostPointer = const_cast<void*>(ptr);
  attrs->device = it->second.device_id;
  attrs->type = cudaMemoryTypeDevice;

  return absl::OkStatus();
}

// Stream management
absl::Status stream_create(cudaStream_t* stream) {
  if (stream == nullptr) {
    return absl::InvalidArgumentError("stream pointer is null");
  }
  auto new_stream = std::make_shared<FakeStream>();
  cudaStream_t handle = reinterpret_cast<cudaStream_t>(new_stream.get());
  auto& state = get_state();
  {
    absl::MutexLock lock(&state.mutex);
    if (!state.streams.emplace(handle, new_stream).second) {
      return absl::InternalError("duplicate fake CUDA stream handle");
    }
  }
  *stream = handle;
  VLOG(2) << "FakeCuda created stream " << reinterpret_cast<const void*>(handle);
  return absl::OkStatus();
}

absl::Status stream_create_with_flags(cudaStream_t* stream, unsigned int /*flags*/) {
  return stream_create(stream);
}

absl::Status stream_destroy(cudaStream_t stream) {
  if (stream == nullptr) {
    return absl::InvalidArgumentError("cannot destroy default fake CUDA stream");
  }

  std::shared_ptr<FakeStream> to_destroy;
  auto& state = get_state();
  {
    absl::MutexLock lock(&state.mutex);
    auto it = state.streams.find(stream);
    if (it == state.streams.end()) {
      return absl::InvalidArgumentError("unknown fake CUDA stream handle");
    }
    to_destroy = std::move(it->second);
    state.streams.erase(it);
  }

  if (to_destroy) {
    to_destroy->Stop();
  }
  return absl::OkStatus();
}

absl::Status stream_synchronize(cudaStream_t stream) {
  auto stream_or = get_stream_state(stream);
  if (!stream_or.ok()) {
    return stream_or.status();
  }
  (*stream_or)->WaitUntilIdle();
  return absl::OkStatus();
}

absl::Status stream_wait_event(cudaStream_t stream, cudaEvent_t event) {
  auto stream_or = get_stream_state(stream);
  if (!stream_or.ok()) {
    return stream_or.status();
  }
  auto event_or = get_event_state(event);
  if (!event_or.ok()) {
    return event_or.status();
  }

  StreamTask task;
  task.label = "stream_wait_event";
  auto event_ptr = *event_or;
  task.work = [event_ptr]() {
    absl::Status st = event_ptr->Wait();
    if (!st.ok()) {
      LOG(WARNING) << "Fake CUDA stream_wait_event wait failed: " << st;
    }
  };

  auto enqueue_or = (*stream_or)->Enqueue(std::move(task));
  if (!enqueue_or.ok()) {
    return enqueue_or.status();
  }
  return absl::OkStatus();
}

absl::Status stream_add_callback(
    cudaStream_t stream,
    void (*callback)(cudaStream_t, cudaError_t, void*),
    void* user_data,
    unsigned int /*flags*/) {
  auto stream_or = get_stream_state(stream);
  if (!stream_or.ok()) {
    return stream_or.status();
  }

  StreamTask task;
  task.label = "stream_add_callback";
  task.work = [callback, stream, user_data]() {
    VLOG(2) << "FakeCuda stream_add_callback running";
    if (callback) {
      callback(stream, cudaSuccess, user_data);
    }
  };

  auto enqueue_or = (*stream_or)->Enqueue(std::move(task));
  if (!enqueue_or.ok()) {
    return enqueue_or.status();
  }

  return absl::OkStatus();
}

absl::Status launch_host_func(cudaStream_t stream, void (*func)(void*), void* user_data) {
  auto stream_or = get_stream_state(stream);
  if (!stream_or.ok()) {
    return stream_or.status();
  }

  StreamTask task;
  task.label = "launch_host_func";
  task.work = [func, user_data]() {
    VLOG(2) << "FakeCuda launch_host_func executing";
    if (func) {
      func(user_data);
    }
  };

  auto enqueue_or = (*stream_or)->Enqueue(std::move(task));
  if (!enqueue_or.ok()) {
    return enqueue_or.status();
  }
  return absl::OkStatus();
}

// Event management
absl::Status event_create(cudaEvent_t* event) {
  if (event == nullptr) {
    return absl::InvalidArgumentError("event pointer is null");
  }
  auto fake_event = std::make_shared<FakeEvent>();
  cudaEvent_t handle = reinterpret_cast<cudaEvent_t>(fake_event.get());
  auto& state = get_state();
  {
    absl::MutexLock lock(&state.mutex);
    if (!state.events.emplace(handle, fake_event).second) {
      return absl::InternalError("duplicate fake CUDA event handle");
    }
  }
  *event = handle;
  return absl::OkStatus();
}

absl::Status event_create_with_flags(cudaEvent_t* event, unsigned int /*flags*/) {
  return event_create(event);
}

absl::Status event_destroy(cudaEvent_t event) {
  if (event == nullptr) {
    return absl::InvalidArgumentError("invalid fake CUDA event handle");
  }
  std::shared_ptr<FakeEvent> to_destroy;
  auto& state = get_state();
  {
    absl::MutexLock lock(&state.mutex);
    auto it = state.events.find(event);
    if (it == state.events.end()) {
      return absl::InvalidArgumentError("invalid fake CUDA event handle");
    }
    to_destroy = std::move(it->second);
    state.events.erase(it);
  }
  if (to_destroy) {
    to_destroy->MarkDestroyed();
  }
  return absl::OkStatus();
}

absl::Status event_record(cudaEvent_t event, cudaStream_t stream) {
  auto stream_or = get_stream_state(stream);
  if (!stream_or.ok()) {
    return stream_or.status();
  }
  auto event_or = get_event_state(event);
  if (!event_or.ok()) {
    return event_or.status();
  }

  auto fake_event = *event_or;
  fake_event->ResetForRecord();

  StreamTask task;
  task.label = "event_record";
  task.work = [fake_event]() { fake_event->MarkCompleted(); };

  auto enqueue_or = (*stream_or)->Enqueue(std::move(task));
  if (!enqueue_or.ok()) {
    return enqueue_or.status();
  }
  return absl::OkStatus();
}

absl::Status event_query(cudaEvent_t event, bool* ready) {
  if (ready == nullptr) {
    return absl::InvalidArgumentError("ready pointer is null");
  }
  auto event_or = get_event_state(event);
  if (!event_or.ok()) {
    return event_or.status();
  }
  auto fake_event = *event_or;
  absl::MutexLock lock(&fake_event->mu);
  if (fake_event->destroyed && !fake_event->completed) {
    return absl::FailedPreconditionError("fake CUDA event destroyed before completion");
  }
  *ready = fake_event->completed;
  return absl::OkStatus();
}

absl::Status event_synchronize(cudaEvent_t event) {
  auto event_or = get_event_state(event);
  if (!event_or.ok()) {
    return event_or.status();
  }
  auto fake_event = *event_or;
  return fake_event->Wait();
}

absl::Status event_elapsed_time(float* ms, cudaEvent_t start, cudaEvent_t end) {
  if (ms == nullptr) {
    return absl::InvalidArgumentError("ms pointer is null");
  }
  auto start_or = get_event_state(start);
  if (!start_or.ok()) {
    return start_or.status();
  }
  auto end_or = get_event_state(end);
  if (!end_or.ok()) {
    return end_or.status();
  }

  auto start_event = *start_or;
  auto end_event = *end_or;

  absl::Time start_time;
  {
    absl::MutexLock lock(&start_event->mu);
    if (!start_event->completed) {
      return absl::FailedPreconditionError("start event not completed");
    }
    start_time = start_event->complete_time;
  }

  absl::Time end_time;
  {
    absl::MutexLock lock(&end_event->mu);
    if (!end_event->completed) {
      return absl::FailedPreconditionError("end event not completed");
    }
    end_time = end_event->complete_time;
  }

  *ms = static_cast<float>(absl::ToDoubleMilliseconds(end_time - start_time));
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

absl::Status device_can_access_peer(int* can_access, int device, int peer_device) {
  static_cast<void>(device);
  static_cast<void>(peer_device);
  if (can_access == nullptr) {
    return absl::InvalidArgumentError("can_access pointer is null");
  }
  *can_access = 1;
  return absl::OkStatus();
}

absl::Status enable_peer_access(int current_device, int peer_device) {
  static_cast<void>(current_device);
  static_cast<void>(peer_device);
  return absl::OkStatus();
}

absl::Status memcpy_peer_async(
    void* dst,
    int dst_device,
    const void* src,
    int src_device,
    size_t bytes,
    cudaStream_t stream) {
  static_cast<void>(dst_device);
  static_cast<void>(src_device);
  return memcpy_async(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
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

  const FakeAllocation& alloc = it->second;
  if (!alloc.uses_shm || alloc.shm_name.empty()) {
    return absl::FailedPreconditionError("fake cuda IPC requested for non-shareable allocation");
  }

  const size_t size = alloc.size;
  auto encode_status = encode_shm_handle(handle, alloc.shm_name, size);
  if (!encode_status.ok()) {
    return encode_status;
  }

  return absl::OkStatus();
}

absl::Status open_ipc_mem_handle(void** dev_ptr, cudaIpcMemHandle_t handle, unsigned int flags) {
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);
  static_cast<void>(flags);

  auto decoded_or = decode_shm_handle(handle);
  if (!decoded_or.ok()) {
    return decoded_or.status();
  }
  const std::string& shm_name = decoded_or->first;
  const size_t size = decoded_or->second;

  int fd = shm_open(shm_name.c_str(), O_RDWR, 0600);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, "shm_open failed in fake open_ipc_mem_handle");
  }

  void* mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) {
    const absl::Status err = absl::ErrnoToStatus(errno, "mmap failed in fake open_ipc_mem_handle");
    close(fd);
    return err;
  }

  IpcMapping mapping;
  mapping.mapped = mapped;
  mapping.size = size;
  mapping.shm_name = shm_name;
  mapping.fd = fd;
  state.ipc_mappings[mapped] = std::move(mapping);

  *dev_ptr = mapped;
  return absl::OkStatus();
}

absl::Status close_ipc_mem_handle(void* dev_ptr) {
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  auto it = state.ipc_mappings.find(dev_ptr);
  if (it == state.ipc_mappings.end()) {
    // In fake backend, unknown mappings are treated as already-closed.
    return absl::OkStatus();
  }

  IpcMapping mapping = std::move(it->second);
  state.ipc_mappings.erase(it);

  absl::Status status = absl::OkStatus();
  if (munmap(mapping.mapped, mapping.size) != 0) {
    status = absl::ErrnoToStatus(errno, "munmap failed in fake close_ipc_mem_handle");
  }

  if (mapping.fd >= 0 && close(mapping.fd) != 0) {
    if (status.ok()) {
      status = absl::ErrnoToStatus(errno, "close failed in fake close_ipc_mem_handle");
    }
  }

  return status;
}

absl::Status host_register(void* ptr, size_t size, unsigned int flags) {
  if (size == 0) {
    return absl::InvalidArgumentError("fake cuda host_register: size must be > 0");
  }
  // In fake backend, host memory registration is a no-op
  return absl::OkStatus();
}

absl::Status host_unregister(void* ptr) {
  // In fake backend, host memory unregistration is a no-op
  return absl::OkStatus();
}

absl::Status cu_init(unsigned int flags) {
  static_cast<void>(flags);
  return absl::OkStatus();
}

absl::Status cu_device_get(CUdevice* device, int ordinal) {
  static_cast<void>(ordinal);
  if (device == nullptr) {
    return absl::InvalidArgumentError("device pointer is null");
  }
  return absl::UnimplementedError("cuDeviceGet not supported in FakeCuda");
}

absl::Status cu_device_get_attribute(int* value, CUdevice_attribute attribute, CUdevice device) {
  static_cast<void>(attribute);
  static_cast<void>(device);
  if (value == nullptr) {
    return absl::InvalidArgumentError("value pointer is null");
  }
  return absl::UnimplementedError("cuDeviceGetAttribute not supported in FakeCuda");
}

absl::Status cu_mem_get_allocation_granularity(
    size_t* granularity,
    const CUmemAllocationProp* prop,
    CUmemAllocationGranularity_flags option) {
  static_cast<void>(prop);
  static_cast<void>(option);
  if (granularity == nullptr) {
    return absl::InvalidArgumentError("granularity pointer is null");
  }
  // Use a conservative alignment that matches common CUDA device granularity.
  *granularity = 64 * 1024;
  return absl::OkStatus();
}

absl::Status cu_mem_address_reserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, uint64_t flags) {
  static_cast<void>(flags);
  if (ptr == nullptr) {
    return absl::InvalidArgumentError("ptr is null");
  }
  if (size == 0) {
    return absl::InvalidArgumentError("size must be > 0");
  }
  // Best-effort: FakeCuda treats "reserve" as reserving a PROT_NONE VMA.
  // alignment is advisory in the fake backend; mmap alignment is page-based.
  static_cast<void>(alignment);

  int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
  void* requested = nullptr;
  if (addr != 0) {
    requested = reinterpret_cast<void*>(addr);
    mmap_flags |= MAP_FIXED;
  }

  void* mapped = mmap(requested, size, PROT_NONE, mmap_flags, -1, 0);
  if (mapped == MAP_FAILED) {
    return absl::ErrnoToStatus(errno, "mmap failed in fake cu_mem_address_reserve");
  }
  *ptr = reinterpret_cast<CUdeviceptr>(mapped);
  return absl::OkStatus();
}

absl::Status cu_mem_create(
    CUmemGenericAllocationHandle* handle,
    size_t size,
    const CUmemAllocationProp* prop,
    uint64_t flags) {
  static_cast<void>(prop);
  static_cast<void>(flags);
  if (handle == nullptr) {
    return absl::InvalidArgumentError("handle is null");
  }
  if (size == 0) {
    return absl::InvalidArgumentError("size must be > 0");
  }

  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  std::string shm_name = make_shm_name();
  int fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, "shm_open failed in fake cu_mem_create");
  }
  if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
    const absl::Status err = absl::ErrnoToStatus(errno, "ftruncate failed in fake cu_mem_create");
    close(fd);
    shm_unlink(shm_name.c_str());
    return err;
  }

  const CUmemGenericAllocationHandle new_handle = state.next_vmm_handle++;
  state.vmm_allocations[new_handle] = FakeVmmAllocation{
      .shm_name = shm_name,
      .fd = fd,
      .size = size,
  };
  *handle = new_handle;
  return absl::OkStatus();
}

absl::Status cu_mem_map(
    CUdeviceptr ptr,
    size_t size,
    size_t offset,
    CUmemGenericAllocationHandle handle,
    uint64_t flags) {
  static_cast<void>(flags);
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  auto it = state.vmm_allocations.find(handle);
  if (it == state.vmm_allocations.end()) {
    return absl::InvalidArgumentError("unknown VMM allocation handle");
  }
  const FakeVmmAllocation& alloc = it->second;

  if (offset != 0) {
    return absl::InvalidArgumentError("fake cu_mem_map only supports offset=0");
  }
  if (size == 0 || size > alloc.size) {
    return absl::InvalidArgumentError("invalid mapping size");
  }
  void* addr = reinterpret_cast<void*>(ptr);
  if (addr == nullptr) {
    return absl::InvalidArgumentError("ptr is null");
  }

  void* mapped = mmap(addr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, alloc.fd, 0);
  if (mapped == MAP_FAILED) {
    return absl::ErrnoToStatus(errno, "mmap failed in fake cu_mem_map");
  }
  return absl::OkStatus();
}

absl::Status cu_mem_set_access(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) {
  static_cast<void>(ptr);
  static_cast<void>(size);
  static_cast<void>(desc);
  static_cast<void>(count);
  // Fake backend does not enforce per-device page permissions.
  return absl::OkStatus();
}

absl::Status cu_mem_unmap(CUdeviceptr ptr, size_t size) {
  if (ptr == 0 || size == 0) {
    return absl::InvalidArgumentError("ptr/size invalid");
  }
  void* addr = reinterpret_cast<void*>(ptr);
  // Preserve the "reserved VA" semantics by replacing the mapping with a
  // PROT_NONE anonymous mapping at the same address.
  void* mapped = mmap(addr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  if (mapped == MAP_FAILED) {
    return absl::ErrnoToStatus(errno, "mmap(PROT_NONE) failed in fake cu_mem_unmap");
  }
  return absl::OkStatus();
}

absl::Status cu_mem_release(CUmemGenericAllocationHandle handle) {
  auto& state = get_state();
  absl::MutexLock lock(&state.mutex);

  auto it = state.vmm_allocations.find(handle);
  if (it == state.vmm_allocations.end()) {
    return absl::OkStatus();
  }

  FakeVmmAllocation alloc = std::move(it->second);
  state.vmm_allocations.erase(it);

  absl::Status status = absl::OkStatus();
  if (alloc.fd >= 0 && close(alloc.fd) != 0) {
    status = absl::ErrnoToStatus(errno, "close failed in fake cu_mem_release");
  }
  if (shm_unlink(alloc.shm_name.c_str()) != 0) {
    if (status.ok()) {
      status = absl::ErrnoToStatus(errno, "shm_unlink failed in fake cu_mem_release");
    }
  }
  return status;
}

absl::Status cu_mem_address_free(CUdeviceptr ptr, size_t size) {
  if (ptr == 0 || size == 0) {
    return absl::InvalidArgumentError("ptr/size invalid");
  }
  void* addr = reinterpret_cast<void*>(ptr);
  if (munmap(addr, size) != 0) {
    return absl::ErrnoToStatus(errno, "munmap failed in fake cu_mem_address_free");
  }
  return absl::OkStatus();
}

absl::Status cu_mem_get_handle_for_address_range(
    void* handle,
    CUdeviceptr dptr,
    size_t size,
    CUmemRangeHandleType handle_type,
    unsigned long long flags) {
  static_cast<void>(handle);
  static_cast<void>(dptr);
  static_cast<void>(size);
  static_cast<void>(handle_type);
  static_cast<void>(flags);
  return absl::UnimplementedError("cuMemGetHandleForAddressRange not supported in FakeCuda");
}

} // namespace tensorcast::cuda::fake_backend

namespace tensorcast::cuda {

#define TENSORCAST_DEFINE_FAKE_BACKEND(return_type, name, args, ...) \
  return_type FakeCudaBackend::name args {                           \
    return fake_backend::name(__VA_ARGS__);                          \
  }

TENSORCAST_CUDA_BACKEND_FUNCTIONS(TENSORCAST_DEFINE_FAKE_BACKEND)

#undef TENSORCAST_DEFINE_FAKE_BACKEND

} // namespace tensorcast::cuda

// NOLINTEND
