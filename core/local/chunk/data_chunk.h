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
#include <vector>

#include "absl/status/status.h"

#include "core/local/chunk/chunk.h"
#include "core/store/replica/replica.h"

namespace tensorcast::local::loader {
class ChunkLoader;
}

namespace tensorcast::local::chunk {

// A chunk of data in the address space of a device.
class DataChunk {
 public:
  virtual ~DataChunk() = default;

  [[nodiscard]] int lock_refcnt() const;

  [[nodiscard]] bool is_locked() const;

  enum class LoaderPriority : uint8_t { High, Low };
  // Register a loader with a priority marker (high or low).
  void register_loader(loader::ChunkLoader* loader, LoaderPriority priority);

  size_t get_size() const {
    return chunk_->get_size();
  };

  virtual absl::Status load() = 0;
  virtual std::future<absl::Status> load_async() = 0;
  virtual absl::Status drop() = 0;

  void* base_addr{nullptr};

  bool loaded{false};
  int preempt_level{0};

  explicit DataChunk(Chunk* chunk) : chunk_(chunk) {};

 protected:
  friend class ChunkPinLease;

  struct LockState {
    mutable std::mutex mutex;
    int lock_refcnt{0};
    bool locked{false};
  };

  Chunk* chunk_;
  LockState lock_state_;

  std::vector<loader::ChunkLoader*> high_priority_loaders_;
  std::vector<loader::ChunkLoader*> low_priority_loaders_;
};

class CPUDataChunk final : public DataChunk {
 public:
  explicit CPUDataChunk(Chunk* chunk);

  absl::Status load() override;
  std::future<absl::Status> load_async() override;
  absl::Status drop() override;
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

  static ChunkPinLease pin_chunks(
      std::vector<std::shared_ptr<DataChunk>>&& data_chunks,
      std::optional<std::chrono::steady_clock::time_point> expiry_time = std::nullopt);

  // Check if lease has expired (returns false if no timeout was set)
  [[nodiscard]] bool is_expired() const;

 private:
  friend class VirtualAddressSpace;
  // Release the currently held lease, if any. Safe to call multiple times.
  void release() noexcept;

  struct Impl {
    std::vector<std::shared_ptr<DataChunk>> data_chunks;
    std::optional<std::chrono::steady_clock::time_point> expiry_time;
  };

  explicit ChunkPinLease(Impl impl);

  std::shared_ptr<Impl> impl_;
};

} // namespace tensorcast::local::chunk
