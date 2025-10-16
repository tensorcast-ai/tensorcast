// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <sys/types.h>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "absl/status/status.h"

#include "core/store/replica/replica.h"

namespace tensorcast::local::loader {
class ChunkLoader;
}

namespace tensorcast::local::chunk {

// A chunk of data in the virtual address space.
class DataChunk {
 public:
  size_t size{0};

  void* gpu_base{nullptr};
  void* cpu_base{nullptr};

  std::shared_ptr<store::replica::Replica> replica;
  off_t r_offset{0};

  // flags
  bool in_dram{false};
  bool in_gpu{false};
  int preempt_level{0};

  [[nodiscard]] int lock_refcnt() const;

  [[nodiscard]] bool is_locked() const;

  absl::Status load();
  std::future<absl::Status> load_async();
  absl::Status drop();

  enum class LoaderPriority { High, Low };

  // Register a loader with a priority marker (high or low).
  void register_loader(loader::ChunkLoader* loader, LoaderPriority priority);

  // store::StoreEngine::ReplicaInfo replica_info;

  DataChunk() = default;
  DataChunk(
      std::shared_ptr<store::replica::Replica> replica_ptr,
      off_t replica_offset, // chunk position in the replica in bytes
      size_t size // size of the chunk in bytes
  );

 private:
  friend class ChunkPinLease;

  struct LockState {
    mutable std::mutex mutex;
    int lock_refcnt{0};
    bool locked{false};
  };

  LockState lock_state_;

  std::vector<loader::ChunkLoader*> high_priority_loaders_;
  std::vector<loader::ChunkLoader*> low_priority_loaders_;
};

// ===== Pin chunks in CPU DRAM =====
class ChunkPinLease {
 public:
  // ChunkPinLease() = default;
  ~ChunkPinLease();
  ChunkPinLease(ChunkPinLease&&) noexcept;
  ChunkPinLease& operator=(ChunkPinLease&&) noexcept;
  ChunkPinLease(const ChunkPinLease&) = delete;
  ChunkPinLease& operator=(const ChunkPinLease&) = delete;

  static ChunkPinLease pin_chunks(
      std::vector<std::shared_ptr<DataChunk>>&& chunks,
      std::optional<std::chrono::steady_clock::time_point> expiry_time = std::nullopt);

  // Check if lease has expired (returns false if no timeout was set)
  [[nodiscard]] bool is_expired() const;

 private:
  friend class VirtualAddressSpace;
  // Release the currently held lease, if any. Safe to call multiple times.
  void release() noexcept;

  struct Impl {
    std::vector<std::shared_ptr<DataChunk>> chunks;
    std::optional<std::chrono::steady_clock::time_point> expiry_time;
  };

  explicit ChunkPinLease(Impl impl);

  std::shared_ptr<Impl> impl_;
};

} // namespace tensorcast::local::chunk
