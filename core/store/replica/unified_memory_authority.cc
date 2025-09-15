// Copyright (c) 2025, TensorCast Team.

#include "core/store/replica/unified_memory_authority.h"
#include "gsl/pointers"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "core/store/device_registry.h"
#include "core/store/replica/chunk_meta.h" // For chunk_state_to_string

// OpenTelemetry metrics (optional)
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::replica {

UnifiedMemoryAuthority::UnifiedMemoryAuthority(
    gsl::not_null<std::shared_ptr<tensorcast::common::memory::VirtualAddressSpace>> virtual_addr_space)
    : va_space_(std::move(virtual_addr_space)) {}

absl::Status UnifiedMemoryAuthority::allocate(const loading::ReplicaKey& key, size_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if already allocated
  if (allocations_.find(key) != allocations_.end()) {
    return absl::AlreadyExistsError(absl::StrFormat("Replica %s already has unified allocation", key.artifact_id));
  }

  // Allocate DRAM via VirtualAddressSpace (VS)
  auto region_or = va_space_->allocate(key.artifact_id, bytes);
  ReplicaAllocation alloc;
  if (region_or.ok()) {
    alloc.dram_region = *region_or;
  } else if (region_or.status().code() == absl::StatusCode::kAlreadyExists) {
    // Region already reserved elsewhere. Query region info to populate base pointer.
    auto info_or = va_space_->region_info(key.artifact_id);
    if (!info_or.ok()) {
      return info_or.status();
    }
    alloc.dram_region = *info_or;
  } else {
    return region_or.status();
  }
  alloc.total_bytes = bytes;
  // Derive number of chunks from the actual VA chunk size to avoid drift
  const size_t va_chunk = va_space_->chunk_size();
  alloc.num_chunks = (bytes + va_chunk - 1) / va_chunk;
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

  allocations_[key] = std::move(alloc);

  // Register VS write hook to update UMA last_access telemetry
  va_space_->register_write_hook(
      key.artifact_id, [this, key](uint64_t off, uint64_t len) { this->record_cpu_write(key, off, len); });

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
      if (rec.cpu != ChunkState::HOT && rec.cpu != ChunkState::COLD && rec.cpu != ChunkState::PREEMPTIBLE) {
        is_missing = true;
      }
    }
    if (is_missing)
      missing.push_back(static_cast<uint32_t>(i));
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
    if (rec.cpu == ChunkState::HOT || rec.cpu == ChunkState::COLD || rec.cpu == ChunkState::PREEMPTIBLE) {
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
  if (it->second.dram_region.cpu_base != nullptr) {
    stats[common::memory::MemoryLocation::CPU] = it->second.total_bytes;
  }

  // Check GPU allocations
  if (!it->second.gpu_allocations.empty()) {
    stats[common::memory::MemoryLocation::GPU] = it->second.total_bytes * it->second.gpu_allocations.size();
  }

  return stats;
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

  return it->second.dram_region.cpu_base;
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
  for (auto& rec : it->second.chunk_records) {
    auto gr = rec.gpu.find(dev_key);
    if (gr != rec.gpu.end()) {
      gr->second = ChunkState::EVICTED;
      rec.version += 1;
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

  VLOG(1) << "UnifiedMemoryAuthority: released GPU device state for replica " << key.artifact_id << " on device "
          << device_id << (drop_allocation ? " (allocation dropped)" : "");
  return absl::OkStatus();
}

size_t UnifiedMemoryAuthority::get_chunk_size() const {
  return va_space_->chunk_size();
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

  // Delegate to VS
  auto status = va_space_->mark_preemptible(key.artifact_id, indices);
  if (!status.ok()) {
    return status;
  }

  // Update our tracking states
  for (uint32_t idx : indices) {
    if (idx < it->second.chunk_records.size()) {
      it->second.chunk_records[idx].cpu = ChunkState::PREEMPTIBLE;
      it->second.chunk_records[idx].version += 1;
    }
  }

  return absl::OkStatus();
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
  const size_t chunk_size = va_space_->chunk_size();
  const uint64_t first = va_offset / chunk_size;
  const uint64_t last = (va_offset + bytes - 1) / chunk_size;
  uint64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
  for (uint64_t i = first; i <= last && i < it->second.chunk_records.size(); ++i) {
    it->second.chunk_records[static_cast<size_t>(i)].last_access_ns = now;
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
  layout.artifact_chunk_bytes = va_space_->chunk_size();
  // Optional: transfer slice bytes from env
  size_t slice = 0;
  if (const char* env = std::getenv("TCAST_TX_SLICE_BYTES")) {
    // Simple decimal parse; ignore invalid
    char* endp = nullptr;
    uint64_t v = std::strtoull(env, &endp, 10);
    if (endp != env && v > 0ULL) {
      slice = static_cast<size_t>(v);
    }
  }
  layout.transfer_slice_bytes = slice; // 0 indicates caller default
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

  // Build byte ranges (chunk-aligned)
  const size_t chunk_size = va_space_->chunk_size();
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
      SessionRecord{.key = key, .target = target, .device_id = device_id, .chunks = chunks, .cpu_leases = {}};
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
      if (rec.cpu != ChunkState::HOT && rec.cpu != ChunkState::COLD && rec.cpu != ChunkState::PREEMPTIBLE) {
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

  if (target == common::memory::MemoryLocation::GPU) {
    if (!device_id.has_value()) {
      device_id = rec.device_id;
    }
    if (!device_id.has_value()) {
      return absl::InvalidArgumentError("UMA commit(GPU): device_id is required");
    }
    // Update UMA ledger directly
    auto it = allocations_.find(rec.key);
    if (it == allocations_.end()) {
      return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", rec.key.artifact_id));
    }
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    DeviceKey dev_key = DeviceRegistry::instance().gpu_key(*device_id);
    for (uint32_t chunk_idx : chunks) {
      if (chunk_idx >= it->second.chunk_records.size()) {
        return absl::OutOfRangeError("Chunk index out of range");
      }
      auto& r = it->second.chunk_records[chunk_idx];
      auto prev_it = r.gpu.find(dev_key);
      ChunkState prev_state = (prev_it != r.gpu.end()) ? prev_it->second : ChunkState::EVICTED;
      r.gpu[dev_key] = ChunkState::COPIED_GPU;
      auto& loaded_counts = it->second.loaded_chunk_counts;
      auto is_loaded = [](ChunkState s) { return s == ChunkState::HOT || s == ChunkState::COPIED_GPU; };
      if (is_loaded(ChunkState::COPIED_GPU) && !is_loaded(prev_state)) {
        loaded_counts[dev_key] += 1;
      } else if (!is_loaded(ChunkState::COPIED_GPU) && is_loaded(prev_state)) {
        if (loaded_counts[dev_key] > 0)
          loaded_counts[dev_key] -= 1;
      }
      r.last_access_ns = now;
      r.version += 1;
    }
  } else if (target == common::memory::MemoryLocation::CPU) {
    // UMA is authoritative: mark CPU chunks HOT in UMA ledger
    auto it = allocations_.find(rec.key);
    if (it == allocations_.end()) {
      return absl::NotFoundError(absl::StrFormat("Replica %s not found in unified memory", rec.key.artifact_id));
    }
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    for (uint32_t chunk_idx : chunks) {
      if (chunk_idx >= it->second.chunk_records.size()) {
        return absl::OutOfRangeError("Chunk index out of range");
      }
      auto& r = it->second.chunk_records[chunk_idx];
      r.cpu = ChunkState::HOT;
      r.last_access_ns = now;
      r.version += 1;
    }
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
  for (uint32_t c : idx) {
    if (c < it->second.chunk_records.size()) {
      if (location == common::memory::MemoryLocation::CPU) {
        it->second.chunk_records[c].exported_cpu = on;
        it->second.chunk_records[c].version += 1;
      }
    }
  }

  ExportRegistration reg;
  reg.chunk_ranges = ranges;

  // For CPU export enable: acquire VS pin leases over the VA ranges and attach keepalive
  if (location == common::memory::MemoryLocation::CPU && on) {
    struct Keep {
      std::vector<common::memory::VirtualAddressSpace::CpuPinLease> leases;
    };

    auto keep = std::make_shared<Keep>();
    const uint64_t total_bytes = it->second.total_bytes;
    const uint64_t chunk_sz = static_cast<uint64_t>(va_space_->chunk_size());
    size_t lease_count = 0;
    for (const auto& [start_idx, end_idx] : ranges) {
      const uint64_t off = static_cast<uint64_t>(start_idx) * chunk_sz;
      const uint64_t end_off = static_cast<uint64_t>(end_idx + 1) * chunk_sz;
      const uint64_t len64 = (end_off > total_bytes) ? (total_bytes - off) : (end_off - off);
      if (len64 == 0)
        continue;
      auto lease_or = va_space_->pin_range(key.artifact_id, off, len64, "Export");
      if (!lease_or.ok()) {
        return lease_or.status();
      }
      keep->leases.emplace_back(std::move(*lease_or));
      lease_count += 1;
    }
    reg.keepalive = keep; // RAII: caller holds to keep leases alive

    // Metrics: tc_va_pin_leases_total{reason=Export}
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto leases_total = meter->CreateDoubleCounter("tc_va_pin_leases_total");
    if (lease_count > 0) {
      std::map<std::string, opentelemetry::common::AttributeValue> attrs;
      attrs.emplace("reason", opentelemetry::common::AttributeValue("Export"));
      leases_total->Add(
          static_cast<double>(lease_count),
          opentelemetry::common::KeyValueIterableView(attrs),
          opentelemetry::context::Context{});
    }
  }

  // For GPU export: update per-device exported flag in record
  if (location == common::memory::MemoryLocation::GPU) {
    DeviceKey dev_key = DeviceRegistry::instance().gpu_key(key.device.ordinal);
    for (uint32_t c : idx) {
      if (c < it->second.chunk_records.size()) {
        it->second.chunk_records[c].exported_gpu[dev_key] = on;
        it->second.chunk_records[c].version += 1;
      }
    }
  }

  // For CPU export disable: UMA drops ledger; VS pin leases are released when keepalive is dropped by caller.
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
  if (alloc.dram_region.cpu_base == nullptr) {
    return absl::FailedPreconditionError("CPU base not available for direct write");
  }

  struct DwKeep {
    std::vector<common::memory::VirtualAddressSpace::CpuPinLease> leases;
  };

  auto keep = std::make_shared<DwKeep>();

  DirectWriteGrant grant;
  grant.windows.reserve(ranges.size());

  for (const auto& r : ranges) {
    if (r.offset + r.length > alloc.total_bytes) {
      return absl::OutOfRangeError("Direct write range exceeds replica bounds");
    }
    auto lease_or = va_space_->pin_range(key.artifact_id, r.offset, r.length, "DirectWrite");
    if (!lease_or.ok()) {
      return lease_or.status();
    }
    keep->leases.emplace_back(std::move(*lease_or));
    gsl::not_null<void*> base{alloc.dram_region.cpu_base};
    grant.windows.push_back(
        DirectWriteGrant::Window{
            .va_offset = r.offset,
            .local_addr = reinterpret_cast<uint64_t>(static_cast<char*>(base.get()) + r.offset),
            .length = r.length});
  }
  grant.keepalive = keep;
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
  switch (policy) {
    case PostGpuLoadPolicy::EvictCPU: {
      size_t freed = va_space_->evict_tail_bytes(key.artifact_id, bytes);
      VLOG(2) << "EvictCPU policy: requested=" << bytes << " freed=" << freed
              << " bytes for artifact=" << key.artifact_id;
      return absl::OkStatus();
    }
    case PostGpuLoadPolicy::MarkPreemptible: {
      // Mark all chunks preemptible as a conservative default
      const size_t chunks = it->second.num_chunks;
      std::vector<uint32_t> all(chunks);
      for (uint32_t i = 0; i < chunks; ++i)
        all[i] = i;
      return va_space_->mark_preemptible(key.artifact_id, all);
    }
    case PostGpuLoadPolicy::Keep:
    default:
      return absl::OkStatus();
  }
}

} // namespace tensorcast::store::replica
