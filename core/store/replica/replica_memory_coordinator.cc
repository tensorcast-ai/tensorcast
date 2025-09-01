// Copyright (c) 2025, TensorCast Team.

#include "core/store/replica/replica_memory_coordinator.h"

#include <algorithm>
#include <cmath>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "core/store/device_registry.h"
#include "core/store/replica/chunk_meta.h" // For chunk_state_to_string

namespace tensorcast::store {

ReplicaMemoryCoordinator::ReplicaMemoryCoordinator(
    gsl::not_null<std::shared_ptr<tensorcast::memory::DistributedVirtualMemoryPool>> dvmp)
    : dvmp_(std::move(dvmp)) {}

absl::Status ReplicaMemoryCoordinator::allocate(const ReplicaKey& key, size_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if already allocated
  if (allocations_.find(key) != allocations_.end()) {
    return absl::AlreadyExistsError(absl::StrFormat("Replica %s already has unified allocation", key.artifact_id));
  }

  // Allocate DRAM via DVMP
  auto region_or = dvmp_->allocate(key.artifact_id, bytes);
  ReplicaAllocation alloc;
  if (region_or.ok()) {
    alloc.dram_region = *region_or;
  } else if (region_or.status().code() == absl::StatusCode::kAlreadyExists) {
    // Region already reserved elsewhere. Query region info to populate base pointer.
    auto info_or = dvmp_->region_info(key.artifact_id);
    if (!info_or.ok()) {
      return info_or.status();
    }
    alloc.dram_region = *info_or;
  } else {
    return region_or.status();
  }
  alloc.total_bytes = bytes;
  alloc.num_chunks = (bytes + tensorcast::memory::DistributedVirtualMemoryPool::kDefaultChunkSize - 1) /
      tensorcast::memory::DistributedVirtualMemoryPool::kDefaultChunkSize;
  // Pre-initialise loaded chunk counters to 0 for all devices – counters grow
  // lazily on first GPU allocation.

  alloc.loaded_chunk_counts = {};

  // Initialize chunk mappings
  alloc.chunk_mappings.reserve(alloc.num_chunks);
  for (size_t i = 0; i < alloc.num_chunks; ++i) {
    ChunkMapping mapping;
    mapping.chunk_idx = i;
    mapping.cpu_state = ChunkState::COLD; // Initial state from DVMP
    mapping.last_access_ns = 0;
    alloc.chunk_mappings.push_back(std::move(mapping));
  }

  allocations_[key] = std::move(alloc);

  VLOG(1) << "ReplicaMemoryCoordinator: allocated " << bytes << " bytes for replica " << key.artifact_id << " with "
          << alloc.num_chunks << " chunks";

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<CudaMemory>> ReplicaMemoryCoordinator::get_or_create_gpu_allocation(
    const ReplicaKey& key,
    int device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }

  DeviceKey dev_key = DeviceRegistry::instance().gpu_key(device_id);
  auto& gpu_allocs = it->second.gpu_allocations;
  auto& loaded_counts = it->second.loaded_chunk_counts;

  // Check if already allocated
  auto gpu_it = gpu_allocs.find(dev_key);
  if (gpu_it != gpu_allocs.end()) {
    return gpu_it->second;
  }

  // Lazy allocation of GPU memory
  auto cuda_mem = std::make_shared<CudaMemory>();
  auto status = cuda_mem->allocate(it->second.total_bytes, device_id);
  if (!status.ok()) {
    return status;
  }

  gpu_allocs[dev_key] = cuda_mem;
  loaded_counts[dev_key] = 0; // Initialise counter

  VLOG(1) << "ReplicaMemoryCoordinator: allocated GPU memory for replica " << key.artifact_id << " on device "
          << device_id;

  return cuda_mem;
}

absl::Span<const ReplicaMemoryCoordinator::ChunkMapping> ReplicaMemoryCoordinator::get_chunk_mappings(
    const ReplicaKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return {};
  }

  // Sync CPU states from DVMP before returning
  const_cast<ReplicaMemoryCoordinator*>(this)->sync_cpu_chunk_states(key);

  return absl::MakeConstSpan(it->second.chunk_mappings);
}

std::vector<uint32_t> ReplicaMemoryCoordinator::get_missing_chunks(
    const ReplicaKey& key,
    MemoryLocation target,
    std::optional<int> device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return {};
  }

  // Sync CPU states from DVMP
  const_cast<ReplicaMemoryCoordinator*>(this)->sync_cpu_chunk_states(key);

  std::vector<uint32_t> missing;
  const auto& mappings = it->second.chunk_mappings;

  for (size_t i = 0; i < mappings.size(); ++i) {
    const auto& mapping = mappings[i];
    bool is_missing = false;

    if (target == MemoryLocation::GPU) {
      if (!device_id.has_value()) {
        LOG(ERROR) << "Device ID required for GPU target";
        continue;
      }

      DeviceKey dev_key = DeviceRegistry::instance().gpu_key(*device_id);
      auto gpu_it = mapping.gpu_state.find(dev_key);
      if (gpu_it == mapping.gpu_state.end() ||
          (gpu_it->second != ChunkState::HOT && gpu_it->second != ChunkState::COPIED_GPU)) {
        is_missing = true;
      }
    } else if (target == MemoryLocation::PAGEABLE_CPU) {
      if (mapping.cpu_state != ChunkState::HOT && mapping.cpu_state != ChunkState::COLD &&
          mapping.cpu_state != ChunkState::PREEMPTIBLE) {
        is_missing = true;
      }
    }

    if (is_missing) {
      missing.push_back(i);
    }
  }

  return missing;
}

absl::Status ReplicaMemoryCoordinator::lock_chunks_for_transfer(
    const ReplicaKey& key,
    MemoryLocation source,
    MemoryLocation target,
    const std::vector<uint32_t>& chunks) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }

  // If source is CPU, delegate to DVMP for locking
  if (source == MemoryLocation::PAGEABLE_CPU) {
    auto status = dvmp_->lock_chunks(key.artifact_id, chunks);
    if (!status.ok()) {
      return status;
    }
  }

  // Update our tracking
  auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  for (uint32_t chunk_idx : chunks) {
    if (chunk_idx >= it->second.chunk_mappings.size()) {
      return absl::OutOfRangeError("Chunk index out of range");
    }

    auto& mapping = it->second.chunk_mappings[chunk_idx];
    mapping.last_access_ns = now;
  }

  return absl::OkStatus();
}

absl::Status ReplicaMemoryCoordinator::update_chunk_states(
    const ReplicaKey& key,
    MemoryLocation location,
    const std::vector<uint32_t>& chunks,
    ChunkState new_state,
    std::optional<int> device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }

  auto now = std::chrono::steady_clock::now().time_since_epoch().count();

  for (uint32_t chunk_idx : chunks) {
    if (chunk_idx >= it->second.chunk_mappings.size()) {
      return absl::OutOfRangeError("Chunk index out of range");
    }

    auto& mapping = it->second.chunk_mappings[chunk_idx];
    mapping.last_access_ns = now;

    // Phase 5: remove legacy per-transition counter (store_daemon_*); unified metrics capture latency/bytes elsewhere.

    if (location == MemoryLocation::GPU) {
      if (!device_id.has_value()) {
        return absl::InvalidArgumentError("Device ID required for GPU location");
      }

      DeviceKey dev_key = DeviceRegistry::instance().gpu_key(*device_id);
      // Capture previous state before we overwrite, needed for counter update.
      auto prev_it = mapping.gpu_state.find(dev_key);
      ChunkState prev_state = (prev_it != mapping.gpu_state.end()) ? prev_it->second : ChunkState::EVICTED;

      mapping.gpu_state[dev_key] = new_state;

      auto& loaded_counts = it->second.loaded_chunk_counts;
      auto is_loaded_state = [](ChunkState s) { return s == ChunkState::HOT || s == ChunkState::COPIED_GPU; };

      if (is_loaded_state(new_state) && !is_loaded_state(prev_state)) {
        loaded_counts[dev_key] += 1;
      } else if (!is_loaded_state(new_state) && is_loaded_state(prev_state)) {
        if (loaded_counts[dev_key] > 0) {
          loaded_counts[dev_key] -= 1;
        }
      }

      // If completing a transfer from CPU, unlock in DVMP
      if (new_state == ChunkState::COPIED_GPU) {
        auto status = dvmp_->unlock_chunks(key.artifact_id, {chunk_idx}, true);
        if (!status.ok()) {
          LOG(WARNING) << "Failed to unlock chunk " << chunk_idx << ": " << status;
        }
      }
    } else if (location == MemoryLocation::PAGEABLE_CPU) {
      // CPU state is managed by DVMP, but we track it here for queries
      mapping.cpu_state = new_state;
      // Inform DVMP about recent activity so that LRU-based policies (e.g. mark_cpu_preemptible)
      // can rely on up-to-date timestamps. We only need to touch the chunk we just updated.
      dvmp_->refresh_chunks(key.artifact_id, {chunk_idx});
    }
  }

  return absl::OkStatus();
}

ReplicaMemoryCoordinator::ChunkSource ReplicaMemoryCoordinator::get_best_source_for_chunk(
    const ReplicaKey& key,
    uint32_t chunk_idx,
    MemoryLocation target) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end() || chunk_idx >= it->second.chunk_mappings.size()) {
    return {ChunkSource::DISK, -1, ""};
  }

  const auto& mapping = it->second.chunk_mappings[chunk_idx];

  // Priority 1: Local copy (CPU→GPU or GPU→CPU)
  if (target == MemoryLocation::GPU) {
    if (mapping.cpu_state == ChunkState::HOT || mapping.cpu_state == ChunkState::COLD ||
        mapping.cpu_state == ChunkState::PREEMPTIBLE) {
      return {ChunkSource::LOCAL_CPU, -1, ""};
    }
  } else if (target == MemoryLocation::PAGEABLE_CPU) {
    // Check if available on any GPU
    for (const auto& [dev_key, state] : mapping.gpu_state) {
      if (state == ChunkState::HOT || state == ChunkState::COPIED_GPU) {
        return {ChunkSource::LOCAL_GPU, dev_key.ordinal, ""};
      }
    }
  }

  // Priority 2: Remote P2P (would need GlobalStore integration)
  // TODO: Query GlobalStore for remote locations

  // Priority 3: Disk
  return {ChunkSource::DISK, -1, ""};
}

std::unordered_map<MemoryLocation, size_t> ReplicaMemoryCoordinator::get_memory_stats(const ReplicaKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::unordered_map<MemoryLocation, size_t> stats;

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return stats;
  }

  // Check DRAM allocation
  if (it->second.dram_region.cpu_base != nullptr) {
    stats[MemoryLocation::PAGEABLE_CPU] = it->second.total_bytes;
  }

  // Check GPU allocations
  if (!it->second.gpu_allocations.empty()) {
    stats[MemoryLocation::GPU] = it->second.total_bytes * it->second.gpu_allocations.size();
  }

  return stats;
}

bool ReplicaMemoryCoordinator::has_allocation(const ReplicaKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocations_.find(key) != allocations_.end();
}

absl::StatusOr<size_t> ReplicaMemoryCoordinator::get_artifact_size(const ReplicaKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }

  return it->second.total_bytes;
}

void* ReplicaMemoryCoordinator::get_cpu_base_ptr(const ReplicaKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return nullptr;
  }

  return it->second.dram_region.cpu_base;
}

void* ReplicaMemoryCoordinator::get_gpu_base_ptr(const ReplicaKey& key, int device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return nullptr;
  }

  DeviceKey dev_key = DeviceRegistry::instance().gpu_key(device_id);
  auto gpu_it = it->second.gpu_allocations.find(dev_key);
  if (gpu_it == it->second.gpu_allocations.end()) {
    return nullptr;
  }

  return gpu_it->second->get();
}

absl::Status ReplicaMemoryCoordinator::release(const ReplicaKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }

  // GPU memory will be released automatically when shared_ptrs are destroyed
  allocations_.erase(it);

  return absl::OkStatus();
}

size_t ReplicaMemoryCoordinator::get_chunk_size() const {
  return tensorcast::memory::DistributedVirtualMemoryPool::kDefaultChunkSize;
}

absl::Status ReplicaMemoryCoordinator::mark_cpu_chunks_preemptible(const ReplicaKey& key, float ratio) {
  if (ratio < 0.0F || ratio > 1.0F) {
    return absl::InvalidArgumentError("ratio must be between 0.0 and 1.0");
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }

  const size_t total_chunks = it->second.num_chunks;
  const size_t chunks_to_mark = static_cast<size_t>(std::floor(ratio * total_chunks));

  if (chunks_to_mark == 0) {
    return absl::OkStatus(); // Nothing to do
  }

  std::vector<uint32_t> indices;
  indices.reserve(chunks_to_mark);
  for (uint32_t i = 0; i < chunks_to_mark; ++i) {
    indices.push_back(i);
  }

  // Delegate to DVMP
  auto status = dvmp_->mark_preemptible(key.artifact_id, indices);
  if (!status.ok()) {
    return status;
  }

  // Update our tracking states
  for (uint32_t idx : indices) {
    if (idx < it->second.chunk_mappings.size()) {
      it->second.chunk_mappings[idx].cpu_state = ChunkState::PREEMPTIBLE;
    }
  }

  return absl::OkStatus();
}

void ReplicaMemoryCoordinator::sync_cpu_chunk_states(const ReplicaKey& key) {
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return;
  }

  // Get current chunk states from DVMP
  auto chunk_snapshot = dvmp_->chunk_snapshot(key.artifact_id);

  // Update our mappings with DVMP states
  for (size_t i = 0; i < std::min(chunk_snapshot.size(), it->second.chunk_mappings.size()); ++i) {
    it->second.chunk_mappings[i].cpu_state = chunk_snapshot[i].state.load();
  }
}

void ReplicaMemoryCoordinator::sync_cpu_chunk_states(
    const ReplicaKey& key,
    absl::Span<const std::pair<uint32_t, uint32_t>> ranges) {
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return;
  }

  // Get current chunk states from DVMP
  auto chunk_snapshot = dvmp_->chunk_snapshot(key.artifact_id);

  // Update only the specified ranges
  for (const auto& [start_idx, end_idx] : ranges) {
    for (uint32_t i = start_idx; i <= end_idx && i < chunk_snapshot.size() && i < it->second.chunk_mappings.size();
         ++i) {
      it->second.chunk_mappings[i].cpu_state = chunk_snapshot[i].state.load();
    }
  }
}

bool ReplicaMemoryCoordinator::is_gpu_loading_complete(const ReplicaKey& key, int device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return false;
  }

  // Quick path using counter.
  DeviceKey dev_key = DeviceRegistry::instance().gpu_key(device_id);
  const auto& loaded_counts = it->second.loaded_chunk_counts;
  auto counter_it = loaded_counts.find(dev_key);
  if (counter_it != loaded_counts.end() && counter_it->second == it->second.num_chunks) {
    return true;
  }

  // Fallback to scan (first time or after state transitions we missed).
  const auto& mappings = it->second.chunk_mappings;
  size_t loaded = 0;
  for (const auto& mapping : mappings) {
    auto state_it = mapping.gpu_state.find(dev_key);
    if (state_it != mapping.gpu_state.end() &&
        (state_it->second == ChunkState::HOT || state_it->second == ChunkState::COPIED_GPU)) {
      ++loaded;
    }
  }

  // Update counter for next fast-path check.
  if (loaded == it->second.num_chunks) {
    const_cast<ReplicaMemoryCoordinator::ReplicaAllocation&>(it->second).loaded_chunk_counts[dev_key] = loaded;
    return true;
  }
  const_cast<ReplicaMemoryCoordinator::ReplicaAllocation&>(it->second).loaded_chunk_counts[dev_key] = loaded;
  return false;
}

void ReplicaMemoryCoordinator::record_gpu_touch(
    const ReplicaKey& key,
    int device_id,
    absl::Span<const uint32_t> chunks) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return;
  }

  uint64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
  for (uint32_t idx : chunks) {
    if (idx < it->second.chunk_mappings.size()) {
      it->second.chunk_mappings[idx].last_access_ns = now;
    }
  }
}

absl::StatusOr<DirectWriteToken> ReplicaMemoryCoordinator::create_direct_write_token(
    const ReplicaKey& key,
    absl::Span<const VaRange> ranges) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::FailedPreconditionError("UMA allocation not found for replica");
  }
  const auto& alloc = it->second;
  if (alloc.dram_region.cpu_base == nullptr) {
    return absl::FailedPreconditionError("CPU base not available for direct write");
  }

  struct DwKeep {
    std::vector<memory::DistributedVirtualMemoryPool::ChunkResidencyLease> leases;
  };
  auto keep = std::make_shared<DwKeep>();

  DirectWriteToken token;
  token.segments.reserve(ranges.size());

  for (const auto& r : ranges) {
    if (r.offset + r.length > alloc.total_bytes) {
      return absl::OutOfRangeError("Direct write range exceeds replica bounds");
    }
    auto lease_or = dvmp_->pin_range(key.artifact_id, r.offset, r.length, "DirectWrite");
    if (!lease_or.ok()) {
      return lease_or.status();
    }
    keep->leases.emplace_back(std::move(*lease_or));
    token.segments.push_back(
        DirectWriteToken::Segment{
            .va_offset = r.offset,
            .local_addr = reinterpret_cast<uint64_t>(static_cast<char*>(alloc.dram_region.cpu_base) + r.offset),
            .length = r.length});
  }
  token.keepalive = keep;
  return token;
}

absl::Status ReplicaMemoryCoordinator::post_gpu_load_policy(
    const ReplicaKey& key,
    size_t bytes,
    PostGpuLoadPolicy policy) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::FailedPreconditionError("UMA allocation not found for replica");
  }
  switch (policy) {
    case PostGpuLoadPolicy::EvictCPU: {
      (void)dvmp_->evict_tail_bytes(key.artifact_id, bytes);
      return absl::OkStatus();
    }
    case PostGpuLoadPolicy::MarkPreemptible: {
      // Mark all chunks preemptible as a conservative default
      const size_t chunks = it->second.num_chunks;
      std::vector<uint32_t> all(chunks);
      for (uint32_t i = 0; i < chunks; ++i)
        all[i] = i;
      return dvmp_->mark_preemptible(key.artifact_id, all);
    }
    case PostGpuLoadPolicy::Keep:
    default:
      return absl::OkStatus();
  }
}

} // namespace tensorcast::store
