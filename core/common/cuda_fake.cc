// Copyright (c) 2025, TensorCast Team.

#include "core/common/cuda_api.h"

#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

// NOLINTBEGIN

namespace tensorcast::cuda {
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
  static_cast<void>(kind);
  // In fake backend, all memory is CPU memory, so just use std::memcpy
  std::memcpy(dst, src, bytes);
  return absl::OkStatus();
}

absl::Status memcpy_async(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
  static_cast<void>(kind);
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
  task.work = [dst, src, bytes]() {
    VLOG(2) << "FakeCuda memcpy_async executing copy of " << bytes << " bytes";
    std::memcpy(dst, src, bytes);
  };

  auto enqueue_or = stream_ptr->Enqueue(std::move(task));
  if (!enqueue_or.ok()) {
    return enqueue_or.status();
  }

  return absl::OkStatus();
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
#ifdef USE_FAKE_CUDA
  // These legacy fields are only present in the fake CUDA runtime we define
  attrs->memoryType = cudaMemoryTypeDevice;
  attrs->isManaged = 0;
#endif

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

} // namespace tensorcast::cuda

// NOLINTEND
