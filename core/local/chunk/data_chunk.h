// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <sys/types.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"

// #include "core/local/chunk/chunk.h"
#include "core/common/memory/cuda_memory.h"
#include "core/local/loader/chunk_loader.h"
#include "core/store/device_types.h"

namespace tensorcast::local::meta {
class Chunk;
} // namespace tensorcast::local::meta

namespace tensorcast::local::data {

// A chunk of data in the address space of a device.
class DataChunk {
 public:
  virtual ~DataChunk() = default;

  [[nodiscard]] int lock_refcnt() const;

  [[nodiscard]] bool is_locked() const;

  enum class LoaderPriority : uint8_t { High, Low };
  // Register a loader with a priority marker (high or low).
  void register_loader(std::shared_ptr<ChunkLoader> loader, LoaderPriority priority);

  size_t get_size() const;

  virtual absl::Status load() = 0;
  virtual std::future<absl::Status> load_async() = 0;
  virtual absl::Status drop() = 0;

  void* get_base_addr() const;
  bool is_loaded() const;
  int get_preempt_level() const;

  explicit DataChunk(meta::Chunk* chunk, store::DeviceKey device_key)
      : chunk_(chunk), device_key_(std::move(device_key)) {};

 protected:
  friend class ChunkPinLease;

  struct LockState {
    mutable std::mutex mutex;
    int lock_refcnt{0};
    bool locked{false};
  };

  meta::Chunk* chunk_;
  LockState lock_state_;
  store::DeviceKey device_key_;
  bool loaded_{false};
  int preempt_level_{0};

  void* base_addr_{nullptr};

  std::vector<std::shared_ptr<ChunkLoader>> high_priority_loaders_;
  std::vector<std::shared_ptr<ChunkLoader>> low_priority_loaders_;
};

class CPUDataChunk final : public DataChunk {
 public:
  explicit CPUDataChunk(meta::Chunk* chunk, store::DeviceKey device_key);

  absl::Status load() override;
  std::future<absl::Status> load_async() override;
  absl::Status drop() override;
};

class GPUDataChunk final : public DataChunk {
 public:
  explicit GPUDataChunk(
      meta::Chunk* chunk,
      store::DeviceKey device_key,
      common::memory::GpuDeviceMemory* gpu_memory = nullptr,
      off_t offset = 0);
  absl::Status bind_gpu_memory(common::memory::GpuDeviceMemory* gpu_memory, off_t offset);
  absl::Status detach_gpu_memory();
  absl::Status load() override;
  std::future<absl::Status> load_async() override;
  absl::Status drop() override;

 private:
  common::memory::GpuDeviceMemory* gpu_memory_{nullptr};
  off_t offset_{0};
};

// ===== Pin data_chunks in its device memory =====
class ChunkPinLease {
 public:
  // ChunkPinLease() = default;
  ~ChunkPinLease();
  ChunkPinLease(ChunkPinLease&&) noexcept;
  ChunkPinLease& operator=(ChunkPinLease&&) noexcept;
  ChunkPinLease(const ChunkPinLease&) = delete;
  ChunkPinLease& operator=(const ChunkPinLease&) = delete;

  explicit ChunkPinLease(
      const std::vector<DataChunk*>& data_chunks,
      std::optional<std::chrono::steady_clock::time_point> expiry_time = std::nullopt);

  explicit ChunkPinLease(std::optional<std::chrono::steady_clock::time_point> expiry_time = std::nullopt);
  // static ChunkPinLease pin_chunks(
  //     std::vector<DataChunk*>&& data_chunks,
  //     std::optional<std::chrono::steady_clock::time_point> expiry_time = std::nullopt);

  absl::Status pin(DataChunk* data_chunk);

  // Check if lease has expired (returns false if no timeout was set)
  [[nodiscard]] bool is_expired() const;

  // Release the currently held lease, if any. Safe to call multiple times.
  void release() noexcept;

 private:
  friend class VirtualAddressSpace;

  struct Impl {
    std::vector<DataChunk*> data_chunks;
    std::optional<std::chrono::steady_clock::time_point> expiry_time;
  };

  explicit ChunkPinLease(Impl impl);

  std::shared_ptr<Impl> impl_;
};

} // namespace tensorcast::local::data
