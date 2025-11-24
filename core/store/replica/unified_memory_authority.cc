// Copyright (c) 2025, TensorCast Team.

#include "core/store/replica/unified_memory_authority.h"
#include "gsl/pointers"

#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "core/common/const/granularity.h"
#include "core/common/system_capabilities.h"
#include "core/store/device_registry.h"

// OpenTelemetry metrics (optional)
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::replica {
using tensorcast::common::SystemCapabilities;

UnifiedMemoryAuthority::UnifiedMemoryAuthority(size_t artifact_chunk_bytes)
    : chunk_size_bytes_(
          artifact_chunk_bytes == 0 ? tensorcast::common::consts::kArtifactChunkDefault : artifact_chunk_bytes),
      cpu_arena_(chunk_size_bytes_) {}

absl::Status UnifiedMemoryAuthority::allocate(const loading::ReplicaKey& key, size_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if already allocated
  if (allocations_.find(key) != allocations_.end()) {
    return absl::AlreadyExistsError(absl::StrFormat("Replica %s already has unified allocation", key.artifact_id));
  }

  ReplicaAllocation alloc;
  auto region_status = cpu_arena_.allocate_region(alloc, bytes);
  if (!region_status.ok()) {
    return region_status;
  }
  alloc.total_bytes = bytes;
  alloc.num_chunks = (bytes + chunk_size_bytes_ - 1) / chunk_size_bytes_;
  // Pre-initialise loaded chunk counters to 0 for all devices – counters grow
  // lazily on first GPU allocation.

  alloc.loaded_chunk_counts = {};

  // Initialize chunk records (orthogonal model). Mapping view will be built lazily.
  alloc.chunk_records.reserve(alloc.num_chunks);
  for (size_t i = 0; i < alloc.num_chunks; ++i) {
    // Initialize orthogonal ChunkRecord (internal only)
    ReplicaAllocation::ChunkRecord rec;
    rec.chunk_idx = static_cast<uint32_t>(i);
    rec.cpu = ChunkState::COLD;
    rec.last_access_ns = 0;
    rec.exported_cpu = false;
    rec.pin_refcnt = 0;
    rec.version = 0;
    alloc.chunk_records.push_back(std::move(rec));
  }
  alloc.mlock_refcnt.assign(alloc.num_chunks, 0);

  allocations_[key] = std::move(alloc);

  VLOG(1) << "UnifiedMemoryAuthority: allocated " << bytes << " bytes for replica " << key.artifact_id << " with "
          << alloc.num_chunks << " chunks";

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<common::memory::GpuDeviceMemory>> UnifiedMemoryAuthority::get_or_create_gpu_allocation(
    const loading::ReplicaKey& key,
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
  auto cuda_mem = std::make_shared<common::memory::GpuDeviceMemory>();
  auto status = cuda_mem->allocate(it->second.total_bytes, device_id);
  if (!status.ok()) {
    return status;
  }

  gpu_allocs[dev_key] = cuda_mem;
  loaded_counts[dev_key] = 0; // Initialise counter

  VLOG(1) << "UnifiedMemoryAuthority: allocated GPU memory for replica " << key.artifact_id << " on device "
          << device_id;

  return cuda_mem;
}

std::vector<uint32_t> UnifiedMemoryAuthority::get_missing_chunks(
    const loading::ReplicaKey& key,
    common::memory::MemoryLocation target,
    std::optional<int> device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return {};
  }

  std::vector<uint32_t> missing;
  const auto& records = it->second.chunk_records;
  bool preemptible_changed = false;

  for (size_t i = 0; i < records.size(); ++i) {
    const auto& rec = records[i];
    bool is_missing = false;

    if (target == common::memory::MemoryLocation::GPU) {
      if (!device_id.has_value()) {
        LOG(ERROR) << "Device ID required for GPU target";
        continue;
      }
      DeviceKey dev_key = DeviceRegistry::instance().gpu_key(*device_id);
      auto itg = rec.gpu.find(dev_key);
      if (itg == rec.gpu.end() || (itg->second != ChunkState::HOT && itg->second != ChunkState::COPIED_GPU)) {
        is_missing = true;
      }
    } else if (target == common::memory::MemoryLocation::CPU) {
      bool resident = is_cpu_resident_state_(rec.cpu);
      if (rec.cpu == ChunkState::PREEMPTIBLE && resident) {
        resident = is_preemptible_resident_locked_(it->second, static_cast<uint32_t>(i));
        if (!resident) {
          record_preemptible_fault_locked_(key, static_cast<uint32_t>(i));
          preemptible_changed = true;
        }
      }
      if (!resident) {
        is_missing = true;
      }
    }
    if (is_missing)
      missing.push_back(static_cast<uint32_t>(i));
  }

  if (preemptible_changed) {
    update_preemptible_budget_locked_();
  }
  return missing;
}

UnifiedMemoryAuthority::ChunkSource UnifiedMemoryAuthority::get_best_source_for_chunk(
    const loading::ReplicaKey& key,
    uint32_t chunk_idx,
    common::memory::MemoryLocation target) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end() || chunk_idx >= it->second.chunk_records.size()) {
    return {ChunkSource::DISK, -1, ""};
  }

  const auto& rec = it->second.chunk_records[chunk_idx];

  // Priority 1: Local copy (CPU→GPU or GPU→CPU)
  if (target == common::memory::MemoryLocation::GPU) {
    bool cpu_available = is_cpu_resident_state_(rec.cpu);
    if (rec.cpu == ChunkState::PREEMPTIBLE && cpu_available) {
      cpu_available = is_preemptible_resident_locked_(it->second, chunk_idx);
    }
    if (cpu_available) {
      return {ChunkSource::LOCAL_CPU, -1, ""};
    }
  } else if (target == common::memory::MemoryLocation::CPU) {
    // Check if available on any GPU
    for (const auto& [dev_key, state] : rec.gpu) {
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

std::unordered_map<common::memory::MemoryLocation, size_t> UnifiedMemoryAuthority::get_memory_stats(
    const loading::ReplicaKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::unordered_map<common::memory::MemoryLocation, size_t> stats;

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return stats;
  }

  // Check DRAM allocation
  if (it->second.cpu_region.base != nullptr) {
    stats[common::memory::MemoryLocation::CPU] = it->second.total_bytes;
  }

  // Check GPU allocations
  if (!it->second.gpu_allocations.empty()) {
    stats[common::memory::MemoryLocation::GPU] = it->second.total_bytes * it->second.gpu_allocations.size();
  }

  return stats;
}

std::vector<UnifiedMemoryAuthority::ChunkRecordView> UnifiedMemoryAuthority::snapshot_cpu_chunks(
    const loading::ReplicaKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<ChunkRecordView> out;
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return out;
  }
  out.reserve(it->second.chunk_records.size());
  for (const auto& rec : it->second.chunk_records) {
    out.push_back(
        ChunkRecordView{
            .chunk_idx = rec.chunk_idx,
            .cpu = rec.cpu,
            .exported_cpu = rec.exported_cpu,
            .pin_refcnt = rec.pin_refcnt,
            .stable_lease_count = rec.stable_lease_count,
            .last_access_ns = rec.last_access_ns,
            .version = rec.version});
  }
  return out;
}

bool UnifiedMemoryAuthority::has_allocation(const loading::ReplicaKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocations_.find(key) != allocations_.end();
}

absl::StatusOr<size_t> UnifiedMemoryAuthority::get_artifact_size(const loading::ReplicaKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }

  return it->second.total_bytes;
}

void* UnifiedMemoryAuthority::get_cpu_base_ptr(const loading::ReplicaKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return nullptr;
  }

  return it->second.cpu_region.base;
}

void* UnifiedMemoryAuthority::get_gpu_base_ptr(const loading::ReplicaKey& key, int device_id) const {
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

absl::Status UnifiedMemoryAuthority::release(const loading::ReplicaKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }

  clear_rehydrate_records_for_key_(key);
  cpu_arena_.release_region(it->second);
  // GPU memory will be released automatically when shared_ptrs are destroyed
  allocations_.erase(it);

  return absl::OkStatus();
}

absl::Status UnifiedMemoryAuthority::release_gpu_device(
    const loading::ReplicaKey& key,
    int device_id,
    bool drop_allocation) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }

  DeviceKey dev_key = DeviceRegistry::instance().gpu_key(device_id);

  // Reset per-chunk GPU residency for this device and update counters
  size_t& counter = it->second.loaded_chunk_counts[dev_key];
  counter = 0;
  bool mutated = false;
  for (auto& rec : it->second.chunk_records) {
    auto gr = rec.gpu.find(dev_key);
    if (gr != rec.gpu.end()) {
      gr->second = ChunkState::EVICTED;
      rec.version += 1;
      mutated = true;
    }
  }

  if (drop_allocation) {
    auto& gpu_allocs = it->second.gpu_allocations;
    auto ga_it = gpu_allocs.find(dev_key);
    if (ga_it != gpu_allocs.end()) {
      // Drop UMA-owned shared_ptr to allow VRAM to be reclaimed
      gpu_allocs.erase(ga_it);
    }
  }

  if (mutated) {
    it->second.ledger_version += 1;
  }

  VLOG(1) << "UnifiedMemoryAuthority: released GPU device state for replica " << key.artifact_id << " on device "
          << device_id << (drop_allocation ? " (allocation dropped)" : "");
  return absl::OkStatus();
}

size_t UnifiedMemoryAuthority::get_artifact_chunk_bytes() const {
  return chunk_size_bytes_;
}

absl::StatusOr<ChunkState> UnifiedMemoryAuthority::get_cpu_chunk_state(
    const loading::ReplicaKey& key,
    uint32_t chunk_idx) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }
  if (chunk_idx >= it->second.chunk_records.size()) {
    return absl::OutOfRangeError("Chunk index out of range");
  }
  return it->second.chunk_records[chunk_idx].cpu;
}

absl::StatusOr<ChunkState> UnifiedMemoryAuthority::get_gpu_chunk_state(
    const loading::ReplicaKey& key,
    int device_id,
    uint32_t chunk_idx) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }
  if (chunk_idx >= it->second.chunk_records.size()) {
    return absl::OutOfRangeError("Chunk index out of range");
  }
  DeviceKey dev_key = DeviceRegistry::instance().gpu_key(device_id);
  const auto& rec = it->second.chunk_records[chunk_idx];
  auto itg = rec.gpu.find(dev_key);
  if (itg == rec.gpu.end()) {
    return absl::NotFoundError("GPU state not present for device");
  }
  return itg->second;
}

absl::Status UnifiedMemoryAuthority::mark_chunks_preemptible_locked_(
    ReplicaAllocation& alloc,
    absl::Span<const uint32_t> indices) {
  if (indices.empty()) {
    return absl::OkStatus();
  }
  std::vector<uint32_t> eligible;
  eligible.reserve(indices.size());
  for (uint32_t idx : indices) {
    if (idx >= alloc.chunk_records.size()) {
      return absl::OutOfRangeError("Chunk index out of range");
    }
    const auto& rec = alloc.chunk_records[idx];
    if (rec.stable_lease_count > 0 || rec.cpu == ChunkState::STABLE) {
      continue; // protected by stable lease
    }
    eligible.push_back(idx);
  }
  if (eligible.empty()) {
    return absl::OkStatus();
  }
  auto status = cpu_arena_.mark_preemptible(alloc, absl::MakeSpan(eligible));
  if (!status.ok()) {
    return status;
  }
  bool mutated = false;
  for (uint32_t idx : eligible) {
    auto& rec = alloc.chunk_records[idx];
    rec.cpu = ChunkState::PREEMPTIBLE;
    rec.version += 1;
    mutated = true;
  }
  if (mutated) {
    alloc.ledger_version += 1;
    update_preemptible_budget_locked_();
  }
  return absl::OkStatus();
}

bool UnifiedMemoryAuthority::is_preemptible_resident_locked_(const ReplicaAllocation& alloc, uint32_t chunk_idx) const {
  if (alloc.cpu_region.base == nullptr || chunk_idx >= alloc.num_chunks) {
    return false;
  }
  const size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
  const size_t chunk_bytes = cpu_arena_.chunk_bytes();
  const size_t offset = static_cast<size_t>(chunk_idx) * chunk_bytes;
  if (offset >= alloc.cpu_region.bytes) {
    return false;
  }
  const size_t len = std::min(chunk_bytes, alloc.cpu_region.bytes - offset);
  const size_t pages = (len + page_size - 1) / page_size;
  std::vector<unsigned char> vec(pages, 0);
  void* addr = static_cast<char*>(alloc.cpu_region.base) + offset;
  if (::mincore(addr, len, vec.data()) != 0) {
    // Assume resident when mincore is unavailable to avoid false positives.
    return true;
  }
  for (unsigned char b : vec) {
    if ((b & 1U) == 0U) {
      return false;
    }
  }
  return true;
}

void UnifiedMemoryAuthority::record_preemptible_fault_locked_(const loading::ReplicaKey& key, uint32_t chunk_idx)
    const {
  const auto now = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lk(telemetry_mu_);
    pending_rehydrate_[PendingRehydrateKey{.key = key, .chunk_idx = chunk_idx}] = now;
  }
  if (memory_tier_budget_) {
    memory_tier_budget_->record_fault();
  }
}

void UnifiedMemoryAuthority::maybe_record_rehydrate_latency_locked_(
    const loading::ReplicaKey& key,
    uint32_t chunk_idx) {
  std::optional<std::chrono::steady_clock::time_point> start;
  {
    std::lock_guard<std::mutex> lk(telemetry_mu_);
    auto it = pending_rehydrate_.find(PendingRehydrateKey{.key = key, .chunk_idx = chunk_idx});
    if (it != pending_rehydrate_.end()) {
      start = it->second;
      pending_rehydrate_.erase(it);
    }
  }
  if (start.has_value() && memory_tier_budget_) {
    const auto delta = std::chrono::steady_clock::now() - *start;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count();
    if (ns > 0) {
      memory_tier_budget_->record_rehydrate_latency(static_cast<uint64_t>(ns));
    }
  }
}

void UnifiedMemoryAuthority::clear_rehydrate_records_for_key_(const loading::ReplicaKey& key) {
  std::lock_guard<std::mutex> lk(telemetry_mu_);
  for (auto it = pending_rehydrate_.begin(); it != pending_rehydrate_.end();) {
    if (it->first.key == key) {
      it = pending_rehydrate_.erase(it);
    } else {
      ++it;
    }
  }
}

absl::Status UnifiedMemoryAuthority::mark_cpu_chunks_preemptible(const loading::ReplicaKey& key, float ratio) {
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

  return mark_chunks_preemptible_locked_(it->second, indices);
}

bool UnifiedMemoryAuthority::is_gpu_loading_complete(const loading::ReplicaKey& key, int device_id) const {
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

  // Fallback to scan (first time or after state transitions we missed) using records.
  const auto& records = it->second.chunk_records;
  size_t loaded = 0;
  for (const auto& rec : records) {
    auto gr = rec.gpu.find(dev_key);
    if (gr != rec.gpu.end() && (gr->second == ChunkState::HOT || gr->second == ChunkState::COPIED_GPU)) {
      ++loaded;
    }
  }

  // Update counter for next fast-path check.
  if (loaded == it->second.num_chunks) {
    const_cast<UnifiedMemoryAuthority::ReplicaAllocation&>(it->second).loaded_chunk_counts[dev_key] = loaded;
    return true;
  }
  const_cast<UnifiedMemoryAuthority::ReplicaAllocation&>(it->second).loaded_chunk_counts[dev_key] = loaded;
  return false;
}

void UnifiedMemoryAuthority::record_gpu_touch(
    const loading::ReplicaKey& key,
    int /*device_id*/,
    absl::Span<const uint32_t> chunks) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return;
  }

  uint64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
  for (uint32_t idx : chunks) {
    if (idx < it->second.chunk_records.size()) {
      it->second.chunk_records[idx].last_access_ns = now;
    }
  }
}

void UnifiedMemoryAuthority::record_cpu_write(const loading::ReplicaKey& key, uint64_t va_offset, uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return;
  }
  record_cpu_write_locked_(it->second, va_offset, bytes);
}

void UnifiedMemoryAuthority::record_cpu_write_locked_(ReplicaAllocation& alloc, uint64_t va_offset, uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  const size_t chunk_size = chunk_size_bytes_;
  if (chunk_size == 0) {
    return;
  }
  const uint64_t first = va_offset / chunk_size;
  const uint64_t last = (va_offset + bytes - 1) / chunk_size;
  uint64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
  for (uint64_t i = first; i <= last && i < alloc.chunk_records.size(); ++i) {
    alloc.chunk_records[static_cast<size_t>(i)].last_access_ns = now;
  }
}

absl::StatusOr<UnifiedMemoryAuthority::ArtifactLayout> UnifiedMemoryAuthority::get_layout(
    const loading::ReplicaKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }
  ArtifactLayout layout;
  layout.artifact_bytes = it->second.total_bytes;
  layout.artifact_chunk_bytes = chunk_size_bytes_;
  // UMA does not own tx_slice_bytes; report 0 and let callers use pool slice size.
  layout.transfer_slice_bytes = 0;
  return layout;
}

std::vector<std::pair<uint32_t, uint32_t>> UnifiedMemoryAuthority::coalesce_runs_(absl::Span<const uint32_t> sorted) {
  std::vector<std::pair<uint32_t, uint32_t>> runs;
  if (sorted.empty())
    return runs;
  uint32_t run_start = sorted.front();
  uint32_t prev = run_start;
  for (size_t i = 1; i < sorted.size(); ++i) {
    const uint32_t idx = sorted[i];
    if (idx == prev + 1) {
      prev = idx;
      continue;
    }
    runs.emplace_back(run_start, prev);
    run_start = prev = idx;
  }
  runs.emplace_back(run_start, prev);
  return runs;
}

bool UnifiedMemoryAuthority::is_cpu_resident_state_(ChunkState state) {
  switch (state) {
    case ChunkState::HOT:
    case ChunkState::STABLE:
    case ChunkState::LOCKED_TX:
    case ChunkState::COLD:
    case ChunkState::PREEMPTIBLE:
      return true;
    default:
      return false;
  }
}

std::vector<uint32_t> UnifiedMemoryAuthority::normalize_chunk_indices_(
    absl::Span<const uint32_t> indices,
    size_t num_chunks) {
  std::vector<uint32_t> normalized;
  if (indices.empty()) {
    normalized.resize(num_chunks);
    std::iota(normalized.begin(), normalized.end(), 0U);
    return normalized;
  }
  normalized.assign(indices.begin(), indices.end());
  std::sort(normalized.begin(), normalized.end());
  normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
  return normalized;
}

uint64_t UnifiedMemoryAuthority::chunk_bytes_for_index_(
    const ReplicaAllocation& alloc,
    uint64_t chunk_size_bytes,
    uint32_t idx) {
  if (chunk_size_bytes == 0) {
    return 0;
  }
  const uint64_t offset = static_cast<uint64_t>(idx) * chunk_size_bytes;
  if (offset >= alloc.total_bytes) {
    return 0;
  }
  const uint64_t remaining = alloc.total_bytes - offset;
  return std::min<uint64_t>(chunk_size_bytes, remaining);
}

absl::StatusOr<UnifiedMemoryAuthority::TransferPlan> UnifiedMemoryAuthority::plan_load(
    const loading::ReplicaKey& key,
    common::memory::MemoryLocation target,
    std::optional<int> device_id,
    std::optional<absl::Span<const uint32_t>> chunk_indices) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }

  // Determine chunk indices: use missing chunks for target if not provided
  std::vector<uint32_t> chunks;
  if (chunk_indices.has_value() && !chunk_indices->empty()) {
    chunks.assign(chunk_indices->begin(), chunk_indices->end());
  } else {
    // Compute missing using internal helper that assumes mutex_ is held to
    // avoid re-entrant locking. This preserves semantics of public
    // get_missing_chunks() without double-locking.
    chunks = get_missing_chunks_locked_(key, target, device_id);
  }
  if (chunks.empty()) {
    // Under LEDGER authority, freshly allocated CPU chunks may be marked COLD in UMA
    // and considered non-missing. For CPU targets, plan the full artifact to ensure
    // first materialization proceeds. GPU path still requires explicit missing chunks.
    if (target == common::memory::MemoryLocation::CPU) {
      chunks.resize(it->second.num_chunks);
      std::iota(chunks.begin(), chunks.end(), 0U);
    } else {
      return absl::FailedPreconditionError("No chunks to load for the requested plan");
    }
  }
  std::sort(chunks.begin(), chunks.end());
  chunks.erase(std::unique(chunks.begin(), chunks.end()), chunks.end());

  if (chunk_indices.has_value() && !chunk_indices->empty()) {
    bool preemptible_changed = false;
    for (uint32_t idx : chunks) {
      if (idx >= it->second.chunk_records.size()) {
        continue;
      }
      const auto& rec = it->second.chunk_records[idx];
      if (rec.cpu == ChunkState::PREEMPTIBLE && !is_preemptible_resident_locked_(it->second, idx)) {
        record_preemptible_fault_locked_(key, idx);
        preemptible_changed = true;
      }
    }
    if (preemptible_changed) {
      update_preemptible_budget_locked_();
    }
  }

  // Build byte ranges (chunk-aligned)
  const size_t chunk_size = chunk_size_bytes_;
  const uint64_t total_bytes = it->second.total_bytes;
  std::vector<std::pair<uint64_t, size_t>> ranges;
  for (const auto& [start_idx, end_idx] : coalesce_runs_(absl::MakeSpan(chunks))) {
    const uint64_t off = static_cast<uint64_t>(start_idx) * chunk_size;
    const uint64_t end_off = static_cast<uint64_t>(end_idx + 1) * chunk_size;
    const uint64_t len64 = (end_off > total_bytes) ? (total_bytes - off) : (end_off - off);
    ranges.emplace_back(off, static_cast<size_t>(len64));
  }

  // Create session record
  const uint64_t sid = next_session_id_++;
  sessions_[sid] =
      SessionRecord{.key = key, .target = target, .device_id = device_id, .chunks = chunks, .cpu_keepalives = {}};
  // Final: do not acquire plan-time CPU pin leases.
  // Sliding-window direct-write leases are obtained per-window by the pump when needed.

  TransferPlan plan;
  plan.session_id = sid;
  plan.ranges = std::move(ranges);
  plan.chunk_indices = chunks;

  // Final: do not pre-issue plan-level DirectWriteGrant.
  // Pump will negotiate per-window grants and auto-fallback if unsupported.

  VLOG(1) << "UMA plan_load: session=" << sid << " chunks=" << plan.chunk_indices.size()
          << " target=" << static_cast<int>(target);
  return plan;
}

// Locked variant: expects mutex_ held by caller
std::vector<uint32_t> UnifiedMemoryAuthority::get_missing_chunks_locked_(
    const loading::ReplicaKey& key,
    common::memory::MemoryLocation target,
    std::optional<int> device_id) const {
  // Preconditions: mutex_ is held
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return {};
  }

  // UMA authoritative; evaluate from records
  std::vector<uint32_t> missing;
  const auto& records = it->second.chunk_records;
  for (size_t i = 0; i < records.size(); ++i) {
    const auto& rec = records[i];
    bool is_missing = false;
    if (target == common::memory::MemoryLocation::GPU) {
      if (!device_id.has_value()) {
        LOG(ERROR) << "Device ID required for GPU target";
        continue;
      }
      DeviceKey dev_key = DeviceRegistry::instance().gpu_key(*device_id);
      auto itg = rec.gpu.find(dev_key);
      if (itg == rec.gpu.end() || (itg->second != ChunkState::HOT && itg->second != ChunkState::COPIED_GPU)) {
        is_missing = true;
      }
    } else if (target == common::memory::MemoryLocation::CPU) {
      if (!is_cpu_resident_state_(rec.cpu)) {
        is_missing = true;
      }
    }
    if (is_missing)
      missing.push_back(static_cast<uint32_t>(i));
  }
  return missing;
}

absl::Status UnifiedMemoryAuthority::commit(
    uint64_t session_id,
    common::memory::MemoryLocation target,
    absl::Span<const uint32_t> committed_chunks,
    std::optional<int> device_id) {
  const auto t0 = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  auto sit = sessions_.find(session_id);
  if (sit == sessions_.end()) {
    return absl::OkStatus(); // idempotent
  }
  const auto& rec = sit->second;
  if (rec.target != target) {
    LOG(WARNING) << "UMA commit: target mismatch for session " << session_id;
  }

  std::vector<uint32_t> chunks(committed_chunks.begin(), committed_chunks.end());
  if (chunks.empty()) {
    chunks = rec.chunks; // default to full plan
  }

  auto it = allocations_.find(rec.key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", rec.key.artifact_id));
  }
  auto& alloc = it->second;
  bool mutated = false;
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();

  if (target == common::memory::MemoryLocation::GPU) {
    if (!device_id.has_value()) {
      device_id = rec.device_id;
    }
    if (!device_id.has_value()) {
      return absl::InvalidArgumentError("UMA commit(GPU): device_id is required");
    }
    DeviceKey dev_key = DeviceRegistry::instance().gpu_key(*device_id);
    for (uint32_t chunk_idx : chunks) {
      if (chunk_idx >= alloc.chunk_records.size()) {
        return absl::OutOfRangeError("Chunk index out of range");
      }
      auto& r = alloc.chunk_records[chunk_idx];
      auto prev_it = r.gpu.find(dev_key);
      ChunkState prev_state = (prev_it != r.gpu.end()) ? prev_it->second : ChunkState::EVICTED;
      r.gpu[dev_key] = ChunkState::COPIED_GPU;
      auto& loaded_counts = alloc.loaded_chunk_counts;
      auto is_loaded = [](ChunkState s) { return s == ChunkState::HOT || s == ChunkState::COPIED_GPU; };
      if (is_loaded(ChunkState::COPIED_GPU) && !is_loaded(prev_state)) {
        loaded_counts[dev_key] += 1;
      } else if (!is_loaded(ChunkState::COPIED_GPU) && is_loaded(prev_state)) {
        if (loaded_counts[dev_key] > 0)
          loaded_counts[dev_key] -= 1;
      }
      r.last_access_ns = now;
      r.version += 1;
      mutated = true;
    }
  } else if (target == common::memory::MemoryLocation::CPU) {
    // UMA is authoritative: mark CPU chunks HOT/STABLE in UMA ledger
    for (uint32_t chunk_idx : chunks) {
      if (chunk_idx >= alloc.chunk_records.size()) {
        return absl::OutOfRangeError("Chunk index out of range");
      }
      auto& r = alloc.chunk_records[chunk_idx];
      const ChunkState target_state = (r.stable_lease_count > 0) ? ChunkState::STABLE : ChunkState::HOT;
      r.cpu = target_state;
      r.last_access_ns = now;
      r.version += 1;
      mutated = true;
      maybe_record_rehydrate_latency_locked_(rec.key, chunk_idx);
    }
  }

  if (mutated) {
    alloc.ledger_version += 1;
    update_preemptible_budget_locked_();
  }

  sessions_.erase(sit); // RAII: releasing session drops CPU pin leases

  // Metrics: commit duration and chunks
  try {
    namespace otel = opentelemetry;
    auto meter = otel::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto commit_hist = meter->CreateDoubleHistogram("tc_um_commit_duration_ms");
    static auto commit_chunks = meter->CreateDoubleCounter("tc_um_commit_chunks_total");
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::map<std::string, otel::common::AttributeValue> attrs;
    const char* target_str = (target == common::memory::MemoryLocation::GPU) ? "GPU" : "CPU";
    attrs.emplace("target", otel::common::AttributeValue(std::string(target_str)));
    commit_hist->Record(ms, otel::common::KeyValueIterableView(attrs), otel::context::Context{});
    commit_chunks->Add(
        static_cast<double>(chunks.size()), otel::common::KeyValueIterableView(attrs), otel::context::Context{});
  } catch (...) {
    VLOG(1) << "UMA commit: metrics unavailable";
  }
  return absl::OkStatus();
}

absl::Status UnifiedMemoryAuthority::abort(uint64_t session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto sit = sessions_.find(session_id);
  if (sit == sessions_.end()) {
    return absl::OkStatus(); // idempotent
  }
  // No VS unlocks in final cutover (pin leases are RAII via grants/keepalive).

  // Metrics: abort counter (capture target before erasing session)
  const char* target_str = (sit->second.target == common::memory::MemoryLocation::GPU) ? "GPU" : "CPU";
  sessions_.erase(sit);

  // Metrics: abort counter
  try {
    namespace otel = opentelemetry;
    auto meter = otel::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto aborts = meter->CreateDoubleCounter("tc_um_abort_total");
    std::map<std::string, otel::common::AttributeValue> attrs;
    attrs.emplace("target", otel::common::AttributeValue(std::string(target_str)));
    aborts->Add(1.0, otel::common::KeyValueIterableView(attrs), otel::context::Context{});
  } catch (...) {
    VLOG(1) << "UMA abort: metrics unavailable";
  }
  return absl::OkStatus();
}

absl::StatusOr<UnifiedMemoryAuthority::StableLease> UnifiedMemoryAuthority::acquire_stable_lease(
    const loading::ReplicaKey& key,
    absl::Span<const uint32_t> chunk_indices) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }
  if (it->second.num_chunks == 0) {
    return absl::FailedPreconditionError("Stable lease requested for empty replica");
  }

  std::vector<uint32_t> normalized = normalize_chunk_indices_(chunk_indices, it->second.num_chunks);
  if (normalized.empty()) {
    return absl::InvalidArgumentError("No chunk indices provided for stable lease");
  }

  const uint64_t pre_version = it->second.ledger_version;
  uint64_t bytes = 0;
  bool mutated = false;
  for (uint32_t idx : normalized) {
    if (idx >= it->second.num_chunks) {
      return absl::OutOfRangeError("Chunk index out of range in stable lease");
    }
    auto& rec = it->second.chunk_records[idx];
    rec.stable_lease_count += 1;
    rec.cpu = ChunkState::STABLE;
    rec.version += 1;
    mutated = true;
    bytes += chunk_bytes_for_index_(it->second, chunk_size_bytes_, idx);
  }
  if (mutated) {
    it->second.ledger_version += 1;
  }

  StableLease lease;
  lease.key = key;
  lease.chunk_indices = std::move(normalized);
  lease.bytes = bytes;
  lease.ledger_version = it->second.ledger_version;

  if (memory_tier_budget_) {
    // Best-effort admission: roll back stable marks if budget is exhausted.
    auto budget_status = memory_tier_budget_->try_acquire_stable(bytes);
    if (!budget_status.ok()) {
      // Undo ledger changes before returning
      for (uint32_t idx : lease.chunk_indices) {
        auto& rec = it->second.chunk_records[idx];
        if (rec.stable_lease_count > 0) {
          rec.stable_lease_count -= 1;
          if (rec.stable_lease_count == 0 && rec.cpu == ChunkState::STABLE) {
            rec.cpu = ChunkState::HOT;
          }
          rec.version += 1;
        }
      }
      it->second.ledger_version = pre_version;
      return budget_status;
    }
  }

  return lease;
}

absl::Status UnifiedMemoryAuthority::release_stable_lease(const StableLease& lease) {
  return release_stable_lease(lease.key, absl::MakeSpan(lease.chunk_indices));
}

absl::Status UnifiedMemoryAuthority::release_stable_lease(
    const loading::ReplicaKey& key,
    absl::Span<const uint32_t> chunk_indices) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }
  if (chunk_indices.empty() && it->second.num_chunks == 0) {
    return absl::OkStatus();
  }
  std::vector<uint32_t> normalized = normalize_chunk_indices_(chunk_indices, it->second.num_chunks);
  if (normalized.empty()) {
    return absl::InvalidArgumentError("No chunk indices provided for stable lease release");
  }

  bool mutated = false;
  for (uint32_t idx : normalized) {
    if (idx >= it->second.num_chunks) {
      return absl::OutOfRangeError("Chunk index out of range in stable lease release");
    }
    auto& rec = it->second.chunk_records[idx];
    if (rec.stable_lease_count == 0) {
      return absl::FailedPreconditionError("Stable lease not held for specified chunk");
    }
    rec.stable_lease_count -= 1;
    if (rec.stable_lease_count == 0 && rec.cpu == ChunkState::STABLE) {
      rec.cpu = ChunkState::HOT;
    }
    rec.version += 1;
    mutated = true;
  }

  if (mutated) {
    it->second.ledger_version += 1;
  }
  if (memory_tier_budget_) {
    const uint64_t released_bytes = [&]() -> uint64_t {
      uint64_t total = 0;
      for (uint32_t idx : normalized) {
        total += chunk_bytes_for_index_(it->second, chunk_size_bytes_, idx);
      }
      return total;
    }();
    memory_tier_budget_->release_stable(released_bytes);
  }
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> UnifiedMemoryAuthority::get_ledger_version(const loading::ReplicaKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }
  return it->second.ledger_version;
}

absl::StatusOr<UnifiedMemoryAuthority::ExportRegistration> UnifiedMemoryAuthority::set_exported(
    const loading::ReplicaKey& key,
    common::memory::MemoryLocation location,
    absl::Span<const uint32_t> chunks,
    bool on) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }
  if (location != common::memory::MemoryLocation::CPU && location != common::memory::MemoryLocation::GPU) {
    return absl::InvalidArgumentError("set_exported: invalid location");
  }

  // Coalesce chunk indices into contiguous ranges
  std::vector<uint32_t> idx(chunks.begin(), chunks.end());
  std::sort(idx.begin(), idx.end());
  idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
  auto ranges = coalesce_runs_(absl::MakeSpan(idx));

  // Update UMA ledger exported flag
  bool mutated = false;
  for (uint32_t c : idx) {
    if (c < it->second.chunk_records.size()) {
      if (location == common::memory::MemoryLocation::CPU) {
        auto& rec = it->second.chunk_records[c];
        if (rec.exported_cpu != on) {
          rec.exported_cpu = on;
          rec.version += 1;
          mutated = true;
        }
      }
    }
  }

  ExportRegistration reg;
  reg.chunk_ranges = ranges;

  // For CPU export enable: acquire UMA pin leases over the VA ranges and attach keepalive
  if (location == common::memory::MemoryLocation::CPU && on) {
    auto keep_or = cpu_arena_.pin_chunks(it->second, absl::MakeSpan(idx), "Export", std::nullopt);
    if (!keep_or.ok()) {
      return keep_or.status();
    }
    reg.keepalive = *keep_or; // RAII: caller holds to keep leases alive

    // Metrics: tc_va_pin_leases_total{reason=Export}
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto leases_total = meter->CreateDoubleCounter("tc_va_pin_leases_total");
    if (!idx.empty()) {
      std::map<std::string, opentelemetry::common::AttributeValue> attrs;
      attrs.emplace("reason", opentelemetry::common::AttributeValue("Export"));
      leases_total->Add(
          static_cast<double>(idx.size()),
          opentelemetry::common::KeyValueIterableView(attrs),
          opentelemetry::context::Context{});
    }
  }

  // For GPU export: update per-device exported flag in record
  if (location == common::memory::MemoryLocation::GPU) {
    DeviceKey dev_key = DeviceRegistry::instance().gpu_key(key.device.ordinal);
    for (uint32_t c : idx) {
      if (c < it->second.chunk_records.size()) {
        auto& rec = it->second.chunk_records[c];
        auto prev = rec.exported_gpu[dev_key];
        if (prev != on) {
          rec.exported_gpu[dev_key] = on;
          rec.version += 1;
          mutated = true;
        }
      }
    }
  }

  if (mutated) {
    it->second.ledger_version += 1;
  }

  // For CPU export disable: UMA drops ledger; pin leases are released when keepalive is dropped by caller.
  return reg;
}

absl::StatusOr<DirectWriteGrant> UnifiedMemoryAuthority::grant_direct_write(
    const loading::ReplicaKey& key,
    absl::Span<const VaRange> ranges) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::FailedPreconditionError("UMA allocation not found for replica");
  }
  const auto& alloc = it->second;
  if (alloc.cpu_region.base == nullptr) {
    return absl::FailedPreconditionError("CPU base not available for direct write");
  }

  DirectWriteGrant grant;
  grant.windows.reserve(ranges.size());

  std::vector<uint32_t> chunk_indices;
  chunk_indices.reserve(ranges.size() * 2);
  for (const auto& r : ranges) {
    if (r.length == 0) {
      continue;
    }
    if (r.offset + r.length > alloc.total_bytes) {
      return absl::OutOfRangeError("Direct write range exceeds replica bounds");
    }
    const uint64_t first_chunk = r.offset / chunk_size_bytes_;
    const uint64_t last_chunk = (r.offset + r.length - 1) / chunk_size_bytes_;
    for (uint64_t c = first_chunk; c <= last_chunk && c < alloc.num_chunks; ++c) {
      chunk_indices.push_back(static_cast<uint32_t>(c));
    }
  }
  std::sort(chunk_indices.begin(), chunk_indices.end());
  chunk_indices.erase(std::unique(chunk_indices.begin(), chunk_indices.end()), chunk_indices.end());

  absl::StatusOr<std::shared_ptr<void>> keep_or =
      cpu_arena_.pin_chunks(it->second, absl::MakeSpan(chunk_indices), "DirectWrite", std::nullopt);
  if (!keep_or.ok()) {
    return keep_or.status();
  }

  for (const auto& r : ranges) {
    gsl::not_null<void*> base{alloc.cpu_region.base};
    grant.windows.push_back(
        DirectWriteGrant::Window{
            .va_offset = r.offset,
            .local_addr = reinterpret_cast<uint64_t>(static_cast<char*>(base.get()) + r.offset),
            .length = r.length});
  }
  grant.keepalive = *keep_or;
  return grant;
}

absl::Status UnifiedMemoryAuthority::post_gpu_load_policy(
    const loading::ReplicaKey& key,
    size_t bytes,
    PostGpuLoadPolicy policy) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::FailedPreconditionError("UMA allocation not found for replica");
  }
  auto policy_name = [](PostGpuLoadPolicy p) -> const char* {
    switch (p) {
      case PostGpuLoadPolicy::EvictCPU:
        return "EvictCPU";
      case PostGpuLoadPolicy::MarkPreemptible:
        return "MarkPreemptible";
      case PostGpuLoadPolicy::Keep:
      default:
        return "Keep";
    }
  };
  VLOG(1) << "UMA post_gpu_load_policy: policy=" << policy_name(policy) << " bytes=" << bytes
          << " artifact=" << key.artifact_id;
  switch (policy) {
    case PostGpuLoadPolicy::EvictCPU: {
      size_t freed = cpu_arena_.evict_tail_bytes(it->second, bytes);
      VLOG(2) << "EvictCPU policy: requested=" << bytes << " freed=" << freed
              << " bytes for artifact=" << key.artifact_id;
      return absl::OkStatus();
    }
    case PostGpuLoadPolicy::MarkPreemptible: {
      // Mark all chunks preemptible as a conservative default
      const size_t chunks = it->second.num_chunks;
      std::vector<uint32_t> all(chunks);
      for (uint32_t i = 0; i < chunks; ++i) {
        all[i] = i;
      }
      return mark_chunks_preemptible_locked_(it->second, all);
    }
    case PostGpuLoadPolicy::Keep:
    default:
      return absl::OkStatus();
  }
}

absl::Status UnifiedMemoryAuthority::write_cpu_span(
    const loading::ReplicaKey& key,
    uint64_t va_offset,
    const void* src,
    size_t bytes) {
  if (bytes == 0) {
    return absl::OkStatus();
  }
  if (src == nullptr) {
    return absl::InvalidArgumentError("write_cpu_span: src is null");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(key);
  if (it == allocations_.end()) {
    return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", key.artifact_id));
  }
  auto status = cpu_arena_.write_span(it->second, va_offset, src, bytes);
  if (!status.ok()) {
    return status;
  }
  record_cpu_write_locked_(it->second, va_offset, bytes);
  return absl::OkStatus();
}

UnifiedMemoryAuthority::CpuArena::CpuArena(size_t chunk_bytes) : chunk_bytes_(chunk_bytes) {}

absl::Status UnifiedMemoryAuthority::CpuArena::allocate_region(ReplicaAllocation& alloc, size_t bytes) const {
  if (alloc.cpu_region.base != nullptr) {
    return absl::AlreadyExistsError("CPU region already allocated");
  }
  if (bytes == 0) {
    alloc.cpu_region.base = nullptr;
    alloc.cpu_region.bytes = 0;
    return absl::OkStatus();
  }
  void* addr = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED) {
    return absl::ErrnoToStatus(errno, "mmap failed while reserving UMA CPU arena");
  }
  alloc.cpu_region.base = addr;
  alloc.cpu_region.bytes = bytes;
  return absl::OkStatus();
}

void UnifiedMemoryAuthority::CpuArena::release_region(ReplicaAllocation& alloc) const {
  if (alloc.cpu_region.base != nullptr && alloc.cpu_region.bytes > 0) {
    if (::munmap(alloc.cpu_region.base, alloc.cpu_region.bytes) != 0) {
      PLOG(WARNING) << "munmap failed while releasing UMA CPU arena";
    }
  }
  alloc.cpu_region.base = nullptr;
  alloc.cpu_region.bytes = 0;
}

absl::Status UnifiedMemoryAuthority::CpuArena::ensure_bounds(
    const ReplicaAllocation& alloc,
    uint64_t va_offset,
    size_t bytes) const {
  if (alloc.cpu_region.base == nullptr || alloc.cpu_region.bytes == 0) {
    return absl::FailedPreconditionError("CPU region not allocated");
  }
  if (va_offset >= alloc.cpu_region.bytes) {
    return absl::OutOfRangeError("offset beyond CPU arena bounds");
  }
  if (va_offset + bytes > alloc.cpu_region.bytes) {
    return absl::InvalidArgumentError("write would exceed CPU arena bounds");
  }
  return absl::OkStatus();
}

absl::Status UnifiedMemoryAuthority::CpuArena::write_span(
    ReplicaAllocation& alloc,
    uint64_t va_offset,
    const void* src,
    size_t bytes) const {
  if (bytes == 0) {
    return absl::OkStatus();
  }
  if (auto st = ensure_bounds(alloc, va_offset, bytes); !st.ok()) {
    return st;
  }

  const long page = sysconf(_SC_PAGESIZE);
  uint64_t page_aligned_off = (va_offset / page) * page;
  uint64_t end_off = va_offset + bytes;
  uint64_t page_aligned_end = ((end_off + page - 1) / page) * page;
  size_t aligned_len = static_cast<size_t>(page_aligned_end - page_aligned_off);
  void* aligned_addr = static_cast<char*>(alloc.cpu_region.base) + page_aligned_off;
  if (::mprotect(aligned_addr, aligned_len, PROT_READ | PROT_WRITE) != 0) {
    void* mapped =
        ::mmap(aligned_addr, aligned_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (mapped == MAP_FAILED || mapped != aligned_addr) {
      return absl::ErrnoToStatus(errno, "UMA write: failed to ensure writable mapping");
    }
  }

  if (SystemCapabilities::instance().madv_willneed_available()) {
    int rc = ::madvise(aligned_addr, aligned_len, MADV_WILLNEED);
    if (rc != 0 && errno != EINVAL) {
      PLOG(WARNING) << "madvise WILLNEED failed in UMA write";
    }
  }

  void* dst = static_cast<char*>(alloc.cpu_region.base) + va_offset;
  std::memcpy(dst, src, bytes);
  return absl::OkStatus();
}

absl::Status UnifiedMemoryAuthority::CpuArena::mark_preemptible(
    ReplicaAllocation& alloc,
    absl::Span<const uint32_t> idx) const {
  if (alloc.cpu_region.base == nullptr) {
    return absl::FailedPreconditionError("CPU region not allocated");
  }
  for (uint32_t i : idx) {
    if (i >= alloc.num_chunks) {
      return absl::OutOfRangeError("Chunk index out of range");
    }
    auto& rec = alloc.chunk_records[i];
    if (rec.pin_refcnt > 0 || rec.stable_lease_count > 0 || rec.cpu == ChunkState::STABLE) {
      continue;
    }
    if (rec.cpu == ChunkState::LOCKED_TX || rec.cpu == ChunkState::EVICTED) {
      continue;
    }
    void* addr = static_cast<char*>(alloc.cpu_region.base) + static_cast<size_t>(i) * chunk_bytes_;
    int rc = 0;
    if (SystemCapabilities::instance().madv_free_available()) {
      rc = ::madvise(addr, chunk_bytes_, MADV_FREE);
      if (rc != 0 && errno == EINVAL) {
        rc = ::madvise(addr, chunk_bytes_, MADV_DONTNEED);
      }
    } else {
      rc = ::madvise(addr, chunk_bytes_, MADV_DONTNEED);
    }
    if (rc != 0) {
      PLOG(WARNING) << "madvise FREE/DONTNEED failed";
    }
  }
  return absl::OkStatus();
}

size_t UnifiedMemoryAuthority::CpuArena::evict_tail_bytes(ReplicaAllocation& alloc, size_t bytes) const {
  if (alloc.cpu_region.base == nullptr || bytes == 0) {
    return 0;
  }
  size_t freed = 0;
  for (ssize_t idx = static_cast<ssize_t>(alloc.num_chunks) - 1; idx >= 0 && freed < bytes; --idx) {
    auto& rec = alloc.chunk_records[static_cast<size_t>(idx)];
    if (rec.pin_refcnt > 0 || rec.stable_lease_count > 0 || rec.cpu == ChunkState::STABLE) {
      continue;
    }
    if (rec.cpu == ChunkState::LOCKED_TX || rec.cpu == ChunkState::EVICTED) {
      continue;
    }
    void* addr = static_cast<char*>(alloc.cpu_region.base) + static_cast<size_t>(idx) * chunk_bytes_;
    size_t advise_len = chunk_bytes_;
    int madv_flag = MADV_DONTNEED;
#ifdef MADV_PAGEOUT
    if (SystemCapabilities::instance().madv_pageout_available()) {
      madv_flag = MADV_PAGEOUT;
    }
#endif
    int rc = ::madvise(addr, advise_len, madv_flag);
    if (rc != 0) {
      PLOG(WARNING) << "madvise PAGEOUT failed";
    }
    freed += chunk_bytes_;
  }
  return freed;
}

absl::StatusOr<std::shared_ptr<void>> UnifiedMemoryAuthority::CpuArena::pin_chunks(
    ReplicaAllocation& alloc,
    absl::Span<const uint32_t> chunk_indices,
    std::string_view reason,
    std::optional<std::chrono::milliseconds> /*timeout*/) const {
  if (chunk_indices.empty()) {
    return std::shared_ptr<void>{};
  }
  if (alloc.cpu_region.base == nullptr) {
    return absl::FailedPreconditionError("CPU region not allocated");
  }

  std::vector<uint32_t> unique_chunks(chunk_indices.begin(), chunk_indices.end());
  std::sort(unique_chunks.begin(), unique_chunks.end());
  unique_chunks.erase(std::unique(unique_chunks.begin(), unique_chunks.end()), unique_chunks.end());

  for (uint32_t idx : unique_chunks) {
    if (idx >= alloc.num_chunks) {
      return absl::OutOfRangeError("Chunk index out of range");
    }
    alloc.chunk_records[idx].pin_refcnt += 1;
    if (SystemCapabilities::instance().mlock_enabled()) {
      void* addr = static_cast<char*>(alloc.cpu_region.base) + static_cast<size_t>(idx) * chunk_bytes_;
      if (::mlock(addr, chunk_bytes_) == 0) {
        if (idx >= alloc.mlock_refcnt.size()) {
          alloc.mlock_refcnt.resize(alloc.num_chunks);
        }
        alloc.mlock_refcnt[idx] += 1;
      } else {
        const int err = errno;
        if (err == ENOMEM || err == EPERM) {
          static std::atomic<bool> warned{false};
          if (!warned.exchange(true, std::memory_order_acq_rel)) {
            PLOG(WARNING) << "mlock failed (" << err << ") — disabling UMA page pinning";
          }
          SystemCapabilities::instance().set_mlock_enabled(false);
        }
      }
    }
  }

  auto handle = std::make_shared<PinHandle>();
  handle->arena = this;
  handle->alloc = &alloc;
  handle->chunks = std::move(unique_chunks);
  std::shared_ptr<void> keepalive(handle, handle.get());

  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_va_pin_leases_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("reason", opentelemetry::common::AttributeValue(std::string(reason)));
    counter->Add(
        static_cast<double>(handle->chunks.size()),
        opentelemetry::common::KeyValueIterableView(attrs),
        opentelemetry::context::Context{});
  } catch (...) {
    VLOG(1) << "metrics counter tc_va_pin_leases_total unavailable";
  }

  return keepalive;
}

UnifiedMemoryAuthority::CpuArena::PinHandle::~PinHandle() {
  if (arena != nullptr && alloc != nullptr) {
    arena->release_pins(*alloc, absl::MakeSpan(chunks));
  }
}

void UnifiedMemoryAuthority::CpuArena::release_pins(ReplicaAllocation& alloc, absl::Span<const uint32_t> chunk_indices)
    const {
  if (alloc.cpu_region.base == nullptr) {
    return;
  }
  for (uint32_t idx : chunk_indices) {
    if (idx >= alloc.num_chunks) {
      continue;
    }
    auto& rec = alloc.chunk_records[idx];
    if (rec.pin_refcnt > 0) {
      rec.pin_refcnt -= 1;
    }
    if (idx < alloc.mlock_refcnt.size()) {
      auto& mlock_count = alloc.mlock_refcnt[idx];
      if (mlock_count > rec.pin_refcnt && SystemCapabilities::instance().mlock_enabled()) {
        if (mlock_count > 0) {
          mlock_count -= 1;
        }
        if (mlock_count == 0) {
          void* addr = static_cast<char*>(alloc.cpu_region.base) + static_cast<size_t>(idx) * chunk_bytes_;
          if (::munlock(addr, chunk_bytes_) != 0) {
            PLOG(WARNING) << "munlock failed during UMA pin release";
          }
        }
      }
    }
  }
}

uint64_t UnifiedMemoryAuthority::compute_preemptible_bytes_locked_() const {
  uint64_t total = 0;
  for (const auto& kv : allocations_) {
    const auto& alloc = kv.second;
    for (size_t idx = 0; idx < alloc.chunk_records.size(); ++idx) {
      if (alloc.chunk_records[idx].cpu == ChunkState::PREEMPTIBLE &&
          is_preemptible_resident_locked_(alloc, static_cast<uint32_t>(idx))) {
        total += chunk_bytes_for_index_(alloc, chunk_size_bytes_, static_cast<uint32_t>(idx));
      }
    }
  }
  return total;
}

void UnifiedMemoryAuthority::update_preemptible_budget_locked_() const {
  if (!memory_tier_budget_) {
    return;
  }
  memory_tier_budget_->set_preemptible_marked_bytes(compute_preemptible_bytes_locked_());
}

} // namespace tensorcast::store::replica
