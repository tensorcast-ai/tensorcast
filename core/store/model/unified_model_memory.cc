// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/model/unified_model_memory.h"

#include <algorithm>
#include <cmath>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "core/common/metrics/metric_objects.h"
#include "core/store/device_registry.h"
#include "core/store/model/chunk_meta.h" // For chunk_state_to_string

namespace stepcast::store {

UnifiedModelMemory::UnifiedModelMemory(gsl::not_null<std::shared_ptr<stepcast::memory::DistributedMemoryPool>> dvmp)
    : dvmp_(std::move(dvmp)) {}

absl::Status UnifiedModelMemory::allocate(const InstanceKey& key, size_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if already allocated
  if (allocations_.find(key) != allocations_.end()) {
    return absl::AlreadyExistsError(absl::StrFormat("Model %s already has unified allocation", key.model_id));
  }

  // Allocate DRAM via DVMP
  auto region_or = dvmp_->allocate(key.model_id, bytes);
  ModelAllocation alloc;
  if (region_or.ok()) {
    alloc.dram_region = *region_or;
  } else if (region_or.status().code() == absl::StatusCode::kAlreadyExists) {
    // Region already reserved elsewhere. Proceed to initialise UMA bookkeeping
    // without duplicating the reservation. We may not know the base pointer.
    alloc.dram_region = stepcast::memory::DistributedMemoryPool::VirtualRegion{
        .cpu_base = nullptr, .gpu_base = nullptr, .bytes = bytes};
  } else {
    return region_or.status();
  }
  alloc.total_bytes = bytes;
  alloc.num_chunks =
      (bytes + stepcast::memory::DistributedMemoryPool::kChunk - 1) / stepcast::memory::DistributedMemoryPool::kChunk;
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

  VLOG(1) << "UnifiedModelMemory: allocated " << bytes << " bytes for model " << key.model_id << " with "
          << alloc.num_chunks << " chunks";

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<CudaMemory>> UnifiedModelMemory::get_or_create_gpu_allocation(
    const InstanceKey& key,
    int device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Model %s not found in unified memory", key.model_id));
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

  VLOG(1) << "UnifiedModelMemory: allocated GPU memory for model " << key.model_id << " on device " << device_id;

  return cuda_mem;
}

absl::Span<const UnifiedModelMemory::ChunkMapping> UnifiedModelMemory::get_chunk_mappings(
    const InstanceKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return {};
  }

  // Sync CPU states from DVMP before returning
  const_cast<UnifiedModelMemory*>(this)->sync_cpu_chunk_states(key);

  return absl::MakeConstSpan(it->second.chunk_mappings);
}

std::vector<uint32_t> UnifiedModelMemory::get_missing_chunks(
    const InstanceKey& key,
    ModelLocation target,
    std::optional<int> device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return {};
  }

  // Sync CPU states from DVMP
  const_cast<UnifiedModelMemory*>(this)->sync_cpu_chunk_states(key);

  std::vector<uint32_t> missing;
  const auto& mappings = it->second.chunk_mappings;

  for (size_t i = 0; i < mappings.size(); ++i) {
    const auto& mapping = mappings[i];
    bool is_missing = false;

    if (target == ModelLocation::GPU) {
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
    } else if (target == ModelLocation::PAGEABLE_CPU) {
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

absl::Status UnifiedModelMemory::lock_chunks_for_transfer(
    const InstanceKey& key,
    ModelLocation source,
    ModelLocation target,
    const std::vector<uint32_t>& chunks) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Model %s not found in unified memory", key.model_id));
  }

  // If source is CPU, delegate to DVMP for locking
  if (source == ModelLocation::PAGEABLE_CPU) {
    auto status = dvmp_->lock_chunks(key.model_id, chunks);
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

absl::Status UnifiedModelMemory::update_chunk_states(
    const InstanceKey& key,
    ModelLocation location,
    const std::vector<uint32_t>& chunks,
    ChunkState new_state,
    std::optional<int> device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Model %s not found in unified memory", key.model_id));
  }

  auto now = std::chrono::steady_clock::now().time_since_epoch().count();

  for (uint32_t chunk_idx : chunks) {
    if (chunk_idx >= it->second.chunk_mappings.size()) {
      return absl::OutOfRangeError("Chunk index out of range");
    }

    auto& mapping = it->second.chunk_mappings[chunk_idx];
    mapping.last_access_ns = now;

    // Record Prometheus counter for this transition (one per chunk event).
    static const stepcast::metrics::Counter kTransitionsCounter("model_chunk_state_transitions_total");
    kTransitionsCounter
        .with_labels(
            {{"location", (location == ModelLocation::GPU ? "GPU" : "CPU")},
             {"state", chunk_state_to_string(new_state)}})
        .inc();

    if (location == ModelLocation::GPU) {
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
        auto status = dvmp_->unlock_chunks(key.model_id, {chunk_idx}, true);
        if (!status.ok()) {
          LOG(WARNING) << "Failed to unlock chunk " << chunk_idx << ": " << status;
        }
      }
    } else if (location == ModelLocation::PAGEABLE_CPU) {
      // CPU state is managed by DVMP, but we track it here for queries
      mapping.cpu_state = new_state;
      // Inform DVMP about recent activity so that LRU-based policies (e.g. mark_cpu_preemptible)
      // can rely on up-to-date timestamps. We only need to touch the chunk we just updated.
      dvmp_->refresh_chunks(key.model_id, {chunk_idx});
    }
  }

  return absl::OkStatus();
}

UnifiedModelMemory::ChunkSource UnifiedModelMemory::get_best_source_for_chunk(
    const InstanceKey& key,
    uint32_t chunk_idx,
    ModelLocation target) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end() || chunk_idx >= it->second.chunk_mappings.size()) {
    return {ChunkSource::DISK, -1, ""};
  }

  const auto& mapping = it->second.chunk_mappings[chunk_idx];

  // Priority 1: Local copy (CPU→GPU or GPU→CPU)
  if (target == ModelLocation::GPU) {
    if (mapping.cpu_state == ChunkState::HOT || mapping.cpu_state == ChunkState::COLD ||
        mapping.cpu_state == ChunkState::PREEMPTIBLE) {
      return {ChunkSource::LOCAL_CPU, -1, ""};
    }
  } else if (target == ModelLocation::PAGEABLE_CPU) {
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

std::unordered_map<ModelLocation, size_t> UnifiedModelMemory::get_memory_stats(const InstanceKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::unordered_map<ModelLocation, size_t> stats;

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return stats;
  }

  // Check DRAM allocation
  if (it->second.dram_region.cpu_base != nullptr) {
    stats[ModelLocation::PAGEABLE_CPU] = it->second.total_bytes;
  }

  // Check GPU allocations
  if (!it->second.gpu_allocations.empty()) {
    stats[ModelLocation::GPU] = it->second.total_bytes * it->second.gpu_allocations.size();
  }

  return stats;
}

bool UnifiedModelMemory::has_allocation(const InstanceKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocations_.find(key) != allocations_.end();
}

absl::StatusOr<size_t> UnifiedModelMemory::get_model_size(const InstanceKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Model %s not found in unified memory", key.model_id));
  }

  return it->second.total_bytes;
}

void* UnifiedModelMemory::get_cpu_base_ptr(const InstanceKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return nullptr;
  }

  return it->second.dram_region.cpu_base;
}

void* UnifiedModelMemory::get_gpu_base_ptr(const InstanceKey& key, int device_id) const {
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

absl::Status UnifiedModelMemory::release(const InstanceKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Model %s not found in unified memory", key.model_id));
  }

  // GPU memory will be released automatically when shared_ptrs are destroyed
  allocations_.erase(it);

  return absl::OkStatus();
}

size_t UnifiedModelMemory::get_chunk_size() const {
  return stepcast::memory::DistributedMemoryPool::kChunk;
}

absl::Status UnifiedModelMemory::mark_cpu_chunks_preemptible(const InstanceKey& key, float ratio) {
  if (ratio < 0.0F || ratio > 1.0F) {
    return absl::InvalidArgumentError("ratio must be between 0.0 and 1.0");
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Model %s not found in unified memory", key.model_id));
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
  auto status = dvmp_->mark_preemptible(key.model_id, indices);
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

void UnifiedModelMemory::sync_cpu_chunk_states(const InstanceKey& key) {
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return;
  }

  // Get current chunk states from DVMP
  auto chunk_snapshot = dvmp_->chunk_snapshot(key.model_id);

  // Update our mappings with DVMP states
  for (size_t i = 0; i < std::min(chunk_snapshot.size(), it->second.chunk_mappings.size()); ++i) {
    it->second.chunk_mappings[i].cpu_state = chunk_snapshot[i].state.load();
  }
}

bool UnifiedModelMemory::is_gpu_loading_complete(const InstanceKey& key, int device_id) const {
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
    const_cast<UnifiedModelMemory::ModelAllocation&>(it->second).loaded_chunk_counts[dev_key] = loaded;
    return true;
  }
  const_cast<UnifiedModelMemory::ModelAllocation&>(it->second).loaded_chunk_counts[dev_key] = loaded;
  return false;
}

void UnifiedModelMemory::record_gpu_touch(const InstanceKey& key, int device_id, absl::Span<const uint32_t> chunks) {
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

} // namespace stepcast::store