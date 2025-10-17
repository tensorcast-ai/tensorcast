// Copyright (c) 2025, TensorCast Team.

#include "store_engine.h"

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <memory>
#include <random>
#include <sstream>
// For reading/writing descriptor and canonical index files
#include <fstream>
#include <limits>
#include <system_error>
#include <unordered_map>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_verification.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/common/trace/trace_macros.h"
#include "core/communicator/misc/common.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/replica_config.h"
// RFC-0007 helpers for safetensors canonical index
#include <nlohmann/json.hpp>
#include "core/store/loader/canonical_index.h"
#include "core/store/loader/safetensors_util.h"
#include "core/store/loader/view_ingest_executor.h"
#include "core/store/loader/view_plan_source.h"
#include "core/store/loader/view_planner.h"
// Unified hashing over SeekableSource for CPU/GPU/P2P
#include "core/store/loader/source_hash.h"
// SegmentPlan linearization (PAD=0 hashing)
#include "core/store/loader/segment_plan_source.h"
#include "core/store/loading/materialize_orchestrator.h"
#include "core/store/loading/replica_registration_helper.h"
#include "gsl/pointers"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::store {

using common::memory::MemoryLocation;
using components::DeviceManager;
using components::MetricsCollector;
using components::ReplicaRegistry;
// using loading::DiskSource; // unused
using loading::InlineBufferSource;
using loading::MaterializeHints;
using loading::ReplicaHandle;
using loading::ReplicaKey;
using loading::ReplicaRegistrationHelper;
using replica::MemoryState;
using replica::Replica;

namespace {

std::string sanitize_view_id_for_filename(const std::string& view_id) {
  if (view_id.empty()) {
    return std::string("view");
  }
  std::string sanitized;
  sanitized.reserve(view_id.size());
  for (char ch : view_id) {
    const unsigned char u = static_cast<unsigned char>(ch);
    if ((u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') || (u >= '0' && u <= '9') || ch == '_' || ch == '-' ||
        ch == '.') {
      sanitized.push_back(ch);
    } else {
      sanitized.push_back('_');
    }
  }
  // Truncate extremely long identifiers from the front to keep filenames manageable.
  constexpr size_t kMaxSuffixLength = 160;
  if (sanitized.size() > kMaxSuffixLength) {
    sanitized.erase(0, sanitized.size() - kMaxSuffixLength);
  }
  return sanitized;
}

std::filesystem::path verification_path_for_view(
    const std::filesystem::path& artifact_path,
    const std::optional<std::string>& view_id) {
  if (!view_id.has_value()) {
    return artifact_path / "verification.json";
  }
  return artifact_path / absl::StrCat("verification.view_", sanitize_view_id_for_filename(*view_id), ".json");
}

std::optional<std::string> ComputeViewDataHash(
    replica::Replica& replica,
    MemoryLocation location,
    uint64_t view_size_bytes,
    std::optional<int> gpu_device_id) {
  if (view_size_bytes == 0) {
    return std::nullopt;
  }
  if (location == MemoryLocation::GPU) {
    if (!gpu_device_id.has_value()) {
      return std::nullopt;
    }
    auto view_or = replica.get_memory_manager().get_gpu_allocation_view();
    if (!view_or.ok() || view_or->base_ptr == nullptr) {
      VLOG(1) << "ComputeViewDataHash: GPU allocation view unavailable";
      return std::nullopt;
    }
    auto hash_or = loader::compute_data_multihash_from_gpu_memory(
        gsl::not_null<void*>{view_or->base_ptr}, view_size_bytes, *gpu_device_id);
    if (!hash_or.ok()) {
      LOG(WARNING) << "compute_data_multihash_from_gpu_memory (view) failed: " << hash_or.status();
      return std::nullopt;
    }
    return *hash_or;
  }

  const auto cpu_ptrs = replica.get_memory_manager().get_pointer(MemoryLocation::CPU);
  if (cpu_ptrs.empty() || cpu_ptrs[0] == nullptr) {
    VLOG(1) << "ComputeViewDataHash: CPU memory unavailable for view hash";
    return std::nullopt;
  }
  auto hash_or =
      loader::compute_data_multihash_from_cpu_memory(gsl::not_null<const void*>{cpu_ptrs[0]}, view_size_bytes);
  if (!hash_or.ok()) {
    LOG(WARNING) << "compute_data_multihash_from_cpu_memory (view) failed: " << hash_or.status();
    return std::nullopt;
  }
  return *hash_or;
}

struct TreeHashWithLeaves {
  std::string multihash;
  std::vector<std::vector<uint8_t>> leaf_digests;
};

absl::StatusOr<TreeHashWithLeaves> compute_tree_hash_and_leaves(
    loader::SeekableSource& source,
    uint64_t total_size,
    size_t leaf_chunk_bytes) {
  if (total_size == 0) {
    return absl::InvalidArgumentError("total_size must be > 0");
  }
  if (leaf_chunk_bytes == 0) {
    return absl::InvalidArgumentError("leaf_chunk_bytes must be > 0");
  }
  std::vector<std::vector<uint8_t>> leaves;
  leaves.reserve(static_cast<size_t>((total_size + leaf_chunk_bytes - 1) / leaf_chunk_bytes));
  std::vector<uint8_t> buffer(leaf_chunk_bytes);
  uint64_t processed = 0;
  while (processed < total_size) {
    const size_t to_read = static_cast<size_t>(std::min<uint64_t>(leaf_chunk_bytes, total_size - processed));
    auto read_or = source.read_at(processed, buffer.data(), to_read);
    if (!read_or.ok()) {
      return read_or.status();
    }
    const size_t got = read_or.value();
    if (got == 0) {
      return absl::DataLossError("short read while computing tree hash");
    }
    leaves.push_back(common::sha256_digest_bytes(absl::Span<const uint8_t>(buffer.data(), got)));
    processed += got;
  }
  TreeHashWithLeaves out;
  out.multihash = common::multibase_multihash_sha256(common::compute_tree_hash_root_sha256(leaves));
  out.leaf_digests = std::move(leaves);
  return out;
}

uint64_t align_up(uint64_t value, uint64_t align) {
  if (align == 0) {
    return value;
  }
  const uint64_t remainder = value % align;
  return remainder == 0 ? value : value + (align - remainder);
}

uint64_t align_down(uint64_t value, uint64_t align) {
  if (align == 0) {
    return value;
  }
  return value - (value % align);
}

std::vector<uint64_t> compute_fully_covered_canonical_leaf_indices(
    absl::Span<const StoreEngine::CanonicalRange> ranges,
    uint64_t chunk_bytes) {
  if (chunk_bytes == 0) {
    return {};
  }
  absl::flat_hash_set<uint64_t> indices;
  for (const auto& range : ranges) {
    if (range.length == 0) {
      continue;
    }
    const uint64_t range_start = range.offset;
    const uint64_t range_end = range.offset + range.length;
    const uint64_t first_full = align_up(range_start, chunk_bytes);
    const uint64_t last_full = align_down(range_end, chunk_bytes);
    if (first_full >= last_full) {
      continue;
    }
    for (uint64_t pos = first_full; pos < last_full; pos += chunk_bytes) {
      indices.insert(pos / chunk_bytes);
    }
  }
  std::vector<uint64_t> sorted(indices.begin(), indices.end());
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}

absl::StatusOr<std::vector<std::vector<uint8_t>>> compute_canonical_leaf_digests(
    const replica::Replica& replica,
    bool use_cpu_memory,
    void* gpu_ptr_hint,
    int device_id,
    uint64_t total_size_bytes,
    absl::Span<const uint64_t> leaf_indices,
    size_t chunk_bytes) {
  std::vector<std::vector<uint8_t>> digests;
  if (leaf_indices.empty()) {
    return digests;
  }
  if (use_cpu_memory) {
    const auto cpu_ptrs = replica.get_memory_manager().get_pointer(MemoryLocation::CPU);
    if (cpu_ptrs.empty() || cpu_ptrs[0] == nullptr) {
      return absl::FailedPreconditionError("CPU pointer unavailable for canonical leaf digests");
    }
    const auto* base = static_cast<const uint8_t*>(cpu_ptrs[0]);
    digests.reserve(leaf_indices.size());
    for (uint64_t idx : leaf_indices) {
      const uint64_t offset = idx * chunk_bytes;
      if (offset + chunk_bytes > total_size_bytes) {
        continue;
      }
      digests.push_back(common::sha256_digest_bytes(absl::Span<const uint8_t>(base + offset, chunk_bytes)));
    }
    return digests;
  }
  void* gpu_ptr = gpu_ptr_hint;
  if (gpu_ptr == nullptr) {
    const auto gpu_ptrs = replica.get_memory_manager().get_pointer(MemoryLocation::GPU);
    if (!gpu_ptrs.empty()) {
      gpu_ptr = gpu_ptrs[0];
    }
  }
  if (gpu_ptr == nullptr) {
    return absl::FailedPreconditionError("GPU pointer unavailable for canonical leaf digests");
  }
  std::vector<uint8_t> buffer(chunk_bytes);
  digests.reserve(leaf_indices.size());
  for (uint64_t idx : leaf_indices) {
    const uint64_t offset = idx * chunk_bytes;
    if (offset + chunk_bytes > total_size_bytes) {
      continue;
    }
    if (auto st = tensorcast::cuda::set_device(device_id); !st.ok()) {
      return st;
    }
    auto copy_status = tensorcast::cuda::memcpy(
        buffer.data(), static_cast<uint8_t*>(gpu_ptr) + offset, chunk_bytes, cudaMemcpyDeviceToHost);
    if (!copy_status.ok()) {
      return copy_status;
    }
    if (auto sync_status = tensorcast::cuda::device_synchronize(); !sync_status.ok()) {
      return sync_status;
    }
    digests.push_back(common::sha256_digest_bytes(absl::Span<const uint8_t>(buffer.data(), chunk_bytes)));
  }
  return digests;
}

uint64_t sum_view_write_bytes(const loader::ViewWritePlan& write_plan) {
  uint64_t total = 0;
  for (const auto& chunk : write_plan.chunks) {
    total += chunk.length;
  }
  return total;
}

std::vector<StoreEngine::CanonicalRange> canonical_ranges_from_write_plan(const loader::ViewWritePlan& write_plan) {
  std::vector<StoreEngine::CanonicalRange> ranges;
  ranges.reserve(write_plan.chunks.size());
  for (const auto& chunk : write_plan.chunks) {
    StoreEngine::CanonicalRange range;
    range.offset = chunk.canonical_offset;
    range.length = chunk.length;
    ranges.push_back(range);
  }
  std::sort(
      ranges.begin(), ranges.end(), [](const StoreEngine::CanonicalRange& a, const StoreEngine::CanonicalRange& b) {
        return a.offset < b.offset;
      });
  std::vector<StoreEngine::CanonicalRange> merged;
  merged.reserve(ranges.size());
  for (const auto& range : ranges) {
    if (range.length == 0) {
      continue;
    }
    if (merged.empty()) {
      merged.push_back(range);
      continue;
    }
    auto& last = merged.back();
    const uint64_t last_end = last.offset + last.length;
    if (range.offset <= last_end) {
      const uint64_t new_end = std::max(last_end, range.offset + range.length);
      last.length = new_end - last.offset;
    } else {
      merged.push_back(range);
    }
  }
  return merged;
}

std::string build_view_spec_json(const loader::ViewSpec& spec) {
  nlohmann::json tensors = nlohmann::json::object();
  for (const auto& [tensor_name, ops] : spec.tensors) {
    nlohmann::json tensor_json;
    nlohmann::json ops_array = nlohmann::json::array();
    for (const auto& op : ops.ops) {
      nlohmann::json op_json;
      switch (op.kind) {
        case loader::ViewOp::Kind::kNarrow:
          op_json["type"] = "narrow";
          op_json["dim"] = op.narrow.dim;
          op_json["start"] = op.narrow.start;
          op_json["length"] = op.narrow.length;
          break;
        case loader::ViewOp::Kind::kTranspose:
          op_json["type"] = "transpose";
          op_json["dim0"] = op.transpose.dim0;
          op_json["dim1"] = op.transpose.dim1;
          break;
      }
      ops_array.push_back(std::move(op_json));
    }
    tensor_json["ops"] = std::move(ops_array);
    tensors[tensor_name] = std::move(tensor_json);
  }
  nlohmann::json root;
  root["tensors"] = std::move(tensors);
  return root.dump();
}

} // namespace

// (hashing utilities moved to core/common/artifact_hash.*)
// GPU eviction helper kept internal to this translation unit.
absl::Status try_evict_gpu_memory_impl(
    ReplicaRegistry& registry,
    DeviceManager& device_manager,
    MetricsCollector& metrics,
    int device_id,
    size_t required_bytes) {
  // Query initial free memory so we can track progress.
  auto free_before_or = device_manager.get_free_memory(device_id);
  if (!free_before_or.ok()) {
    return free_before_or.status();
  }
  size_t free_before = free_before_or.value();

  // Helper to check whether we have reclaimed enough GPU memory.
  auto has_freed_enough = [&](size_t free_now) { return (free_now - free_before) >= required_bytes; };

  // Iterate LRU instances – GPU only.
  auto lru_instances = registry.get_lru_instances();
  for (const auto& key : lru_instances) {
    if (key.device.type != DeviceType::GPU || key.device.ordinal != device_id) {
      continue; // Different device or CPU instance.
    }

    auto replica_or = registry.find(key);
    if (!replica_or.ok()) {
      continue;
    }
    const auto& replica = replica_or.value();

    if (replica->get_memory_state(MemoryLocation::GPU) != MemoryState::LOADED) {
      continue; // Nothing to free.
    }

    // Attempt to release GPU memory (safe mode to avoid mid-transfer memory).
    auto st = replica->release_memory(MemoryLocation::GPU);
    if (!st.ok()) {
      continue; // Couldn't free – maybe busy.
    }

    metrics.record_memory_eviction();

    // Update free memory reading.
    auto free_now_or = device_manager.get_free_memory(device_id);
    if (!free_now_or.ok()) {
      // Non-fatal – continue trying.
      continue;
    }
    size_t free_now = free_now_or.value();

    if (has_freed_enough(free_now)) {
      return absl::OkStatus();
    }
  }

  return absl::ResourceExhaustedError("Could not free enough GPU memory for device " + std::to_string(device_id));
}

// ═══════════════════════════════════════════════════════════════════════════
// Construction and Destruction
// ═══════════════════════════════════════════════════════════════════════════

// New unified constructor based on StoreEngineOptions (Phase-3+)
StoreEngine::StoreEngine(const StoreEngineOptions& opts)
    : options_(opts),
      storage_path_(opts.storage_path),
      memory_pool_size_(opts.memory_pool_size),
      num_thread_(opts.num_thread),
      tx_slice_bytes_(opts.tx_slice_bytes),
      pinned_memory_timeout_(opts.pinned_memory_timeout),
      device_manager_(
          gsl::not_null<std::unique_ptr<components::DeviceManager>>(std::make_unique<components::DeviceManager>())),
      replica_registry_(
          gsl::not_null<std::unique_ptr<components::ReplicaRegistry>>(std::make_unique<components::ReplicaRegistry>())),
      metrics_collector_(gsl::not_null<std::unique_ptr<components::MetricsCollector>>(
          std::make_unique<components::MetricsCollector>())),
      comm_manager_(gsl::not_null<std::shared_ptr<components::CommunicationManager>>(
          std::make_shared<components::CommunicationManager>())),
      memory_pool_(gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>(
          std::make_shared<common::memory::PinnedBufferPool>(memory_pool_size_, tx_slice_bytes_))),
      va_space_(gsl::not_null<std::shared_ptr<common::memory::VirtualAddressSpace>>(
          std::make_shared<common::memory::VirtualAddressSpace>(opts.artifact_chunk_bytes))) {
  LOG(INFO) << "Initializing StoreEngine with unified Options constructor";
  LOG(INFO) << "Storage path: "
            << (storage_path_.empty() ? "<empty - artifact_identifier will be full path>" : storage_path_.string());
  LOG(INFO) << "Memory pool size: " << memory_pool_size_ / communicator::misc::GB << "GB";
  LOG(INFO) << "I/O threads: " << num_thread_ << ", tx_slice_bytes: " << tx_slice_bytes_ / communicator::misc::MB
            << "MB";

  // Enforce invariants:
  // 1) Transfer slice (tx_slice_bytes) must divide artifact chunk (artifact_chunk_bytes)
  ABSL_CHECK_EQ(opts.artifact_chunk_bytes % tx_slice_bytes_, 0)
      << "StoreEngine: artifact_chunk_bytes=" << opts.artifact_chunk_bytes
      << " must be a multiple of transfer slice (tx_slice_bytes)=" << tx_slice_bytes_ << " to avoid cross-chunk slices";

  // 2) Pinned pool block size must be aligned to DIRECT_IO and page size
  const size_t pool_block = memory_pool_->slice_bytes();
  ABSL_CHECK_EQ(pool_block % common::memory::PinnedBufferPool::kDirectIOAlignment, 0)
      << "Pinned buffer block size (" << pool_block << ") not aligned to DIRECT_IO ("
      << common::memory::PinnedBufferPool::kDirectIOAlignment << ")";
  ABSL_CHECK_EQ(pool_block % common::memory::PinnedBufferPool::kMemoryAlignment, 0)
      << "Pinned buffer block size (" << pool_block << ") not aligned to page size ("
      << common::memory::PinnedBufferPool::kMemoryAlignment << ")";

  initialize_components();
  initialize_global_store(opts);
  initialize_communication_manager(opts);

  metrics_collector_->update_all_metrics(*memory_pool_, *replica_registry_, *device_manager_);
}

void StoreEngine::initialize_components() {
  // Initialize core components
  absl::Status status = device_manager_->initialize();
  CHECK(status.ok()) << "Failed to initialize DeviceManager: " << status.message();
}

void StoreEngine::initialize_global_store(const StoreEngineOptions& opts) {
  // Global Store client (remote coordination).  If a non-empty
  // global_store_address is provided via StoreEngineOptions, attempt to
  // connect immediately so that MaterializeOrchestrator can leverage it for remote
  // replica discovery.
  if (!opts.global_store_address.empty()) {
    components::GlobalStoreClientConfig gs_cfg;
    gs_cfg.global_store_address = opts.global_store_address;

    global_store_client_ = std::make_shared<components::GlobalStoreClient>(gs_cfg);
    absl::Status st = global_store_client_->initialize();
    if (!st.ok()) {
      LOG(WARNING) << "StoreEngine: GlobalStoreClient init failed: " << st;
    } else {
      LOG(INFO) << "StoreEngine: connected to Global Store at " << gs_cfg.global_store_address;
    }
  }
}

void StoreEngine::initialize_communication_manager(const StoreEngineOptions& opts) {
  // Initialize CommunicationManager handling (P2P, RDMA, etc.)
  if (opts.comm_manager) {
    // Use externally supplied manager (already initialised by caller)
    comm_manager_ = gsl::not_null<std::shared_ptr<components::CommunicationManager>>(opts.comm_manager);
  } else {
    LOG(INFO) << "CommunicationManager not provided; P2P disabled until explicitly initialized";
  }
}

StoreEngine::~StoreEngine() {
  LOG(INFO) << "Shutting down StoreEngine";
  clear_mem();
}

void StoreEngine::set_global_store_client_for_testing(std::shared_ptr<components::IGlobalStoreClient> client) {
  global_store_client_ = std::move(client);
}

// ═══════════════════════════════════════════════════════════════════════════
// Status Queries
// ═══════════════════════════════════════════════════════════════════════════

size_t StoreEngine::get_available_memory() const {
  return memory_pool_->get_available_size();
}

void StoreEngine::update_memory_pool_metrics() {
  metrics_collector_->update_all_metrics(*memory_pool_, *replica_registry_, *device_manager_);
}

std::vector<StoreEngine::ReplicaInfo> StoreEngine::get_all_replicas_info() const {
  std::vector<ReplicaInfo> result;

  // Use LRU list to retrieve all known ReplicaKeys. This covers every entry
  // in the registry without exposing internal storage.
  const auto replica_keys = replica_registry_->get_lru_instances();

  for (const auto& key : replica_keys) {
    auto replica_or = replica_registry_->find(key);
    if (!replica_or.ok()) {
      continue; // Instance may have been removed concurrently.
    }

    const auto& replica = replica_or.value();

    ReplicaInfo info;
    info.artifact_id = key.artifact_id;

    auto size_result = replica->get_artifact_size();
    info.size_bytes = size_result.ok() ? size_result.value() : 0;

    auto cpu_state = replica->get_memory_state(common::memory::MemoryLocation::CPU);
    auto gpu_state = replica->get_memory_state(common::memory::MemoryLocation::GPU);

    auto is_present = [](replica::MemoryState st) {
      return st == replica::MemoryState::ALLOCATED || st == replica::MemoryState::LOADING ||
          st == replica::MemoryState::LOADED;
    };

    info.cpu_state = is_present(cpu_state) ? common::memory::MemoryLocation::CPU : common::memory::MemoryLocation::NONE;
    info.gpu_state = is_present(gpu_state) ? common::memory::MemoryLocation::GPU : common::memory::MemoryLocation::NONE;

    info.gpu_device_id = -1;
    info.gpu_device_uuid.clear();

    if (key.device.type == DeviceType::GPU && is_present(gpu_state)) {
      info.gpu_device_id = key.device.ordinal;
      if (!key.device.uuid.empty()) {
        info.gpu_device_uuid = key.device.uuid;
      } else {
        // Fallback: query via CUDA API if uuid not stored.
        const auto gpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
        if (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) {
          cudaPointerAttributes attrs;
          auto attr_status = cuda::pointer_get_attributes_full(gpu_ptrs[0], &attrs);
          if (attr_status.ok() && attrs.type == cudaMemoryTypeDevice) {
            auto gpu_info_result = device_manager_->get_gpu_info(attrs.device);
            if (gpu_info_result.ok()) {
              info.gpu_device_uuid = (*gpu_info_result)->uuid;
            }
          }
        }
      }
    }

    info.is_registered_for_comm =
        comm_manager_->is_enabled() && (cpu_state == MemoryState::LOADED || gpu_state == MemoryState::LOADED);

    // Precise access / load timestamps are not tracked at this layer after the
    // registry refactor.  We set them to the current time as a placeholder.
    auto now = std::chrono::system_clock::now();
    info.last_access_time = now;
    info.load_time = now;

    result.push_back(info);
  }

  return result;
}

absl::StatusOr<int> StoreEngine::get_unique_gpu_residency(std::string_view artifact_id) const {
  int unique_gpu_device = -2; // -2: unknown, -1: none, >=0: unique device
  for (const auto& info : get_all_replicas_info()) {
    if (info.artifact_id == artifact_id && info.gpu_state == common::memory::MemoryLocation::GPU) {
      if (unique_gpu_device == -2) {
        unique_gpu_device = info.gpu_device_id;
      } else if (unique_gpu_device != info.gpu_device_id) {
        return absl::InvalidArgumentError("ambiguous artifact residency across multiple GPUs; device_id required");
      }
    }
  }
  if (unique_gpu_device == -2)
    return -1;
  return unique_gpu_device;
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Implementation - using new unified types
// ═══════════════════════════════════════════════════════════════════════════

absl::StatusOr<loading::ReplicaHandle> StoreEngine::ingest_from_disk_internal(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  const auto start_time = std::chrono::steady_clock::now();
  const std::string request_id = absl::StrCat("disk_", absl::ToUnixNanos(absl::Now()));
  SC_TRACE_INIT_GUARD(request_id, artifact_identifier, "ingest_from_disk_internal");

  // Convert Location to legacy MemoryLocation
  common::memory::MemoryLocation target_location = common::memory::MemoryLocation::CPU;
  if (target.location.type == common::memory::MemoryLocation::GPU) {
    target_location = common::memory::MemoryLocation::GPU;
  }

  // Resolve device ID if GPU target
  int target_device_id = target.location.device_id;
  if (target_location == common::memory::MemoryLocation::GPU && !target.location.device_uuid.empty()) {
    auto device_result = device_manager_->find_device_by_uuid(target.location.device_uuid);
    if (!device_result.ok()) {
      return device_result.status();
    }
    target_device_id = device_result.value();
  }

  // Defensive check: ensure GPU ordinal is valid before proceeding.
  if (target_location == common::memory::MemoryLocation::GPU) {
    const int num_gpus = device_manager_->get_num_gpus();
    if (target_device_id < 0 || target_device_id >= num_gpus) {
      return absl::InvalidArgumentError(std::string("Invalid GPU device ordinal: ") + std::to_string(target_device_id));
    }
  }

  // If StoreEngine was initialised with a non-empty storage_path_ and the
  // incoming DiskSource path is *not* absolute, we interpret it as a
  // sub-directory under the configured storage root (the behaviour expected by
  // unit-tests).  This mirrors the semantics of the legacy Python
  // implementation and avoids surprises when the current working directory is
  // different from storage_path_.
  loading::DiskSource resolved_source = source;
  if (!storage_path_.empty() && !source.path.is_absolute()) {
    resolved_source.path = storage_path_ / source.path;
  }

  std::filesystem::path artifact_path = resolved_source.path;
  std::error_code fs_error;
  const bool artifact_exists = std::filesystem::exists(artifact_path, fs_error);
  if (fs_error) {
    return absl::ErrnoToStatus(
        fs_error.value(), absl::StrCat("Failed to access artifact directory '", artifact_path.string(), "'"));
  }
  if (!artifact_exists) {
    return absl::NotFoundError(absl::StrCat("Artifact directory not found: ", artifact_path.string()));
  }
  const bool is_artifact_dir = std::filesystem::is_directory(artifact_path, fs_error);
  if (fs_error) {
    return absl::ErrnoToStatus(
        fs_error.value(), absl::StrCat("Failed to stat artifact directory '", artifact_path.string(), "'"));
  }
  if (!is_artifact_dir) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected artifact path to be a directory: ", artifact_path.string()));
  }
  std::optional<std::string> computed_data_mh;
  std::optional<std::string> computed_index_mh;
  std::optional<std::string> existing_index_mh;
  std::optional<std::string> existing_data_mh;
  std::optional<std::string> descriptor_schema_version;
  std::optional<std::string> canonical_index_json;
  std::optional<loader::ViewPlan> resolved_view_plan;
  std::optional<std::string> view_byte_space_hash;
  uint64_t logical_total_size = 0;

  bool is_safetensors = false;
  for (const auto& entry : std::filesystem::directory_iterator(artifact_path)) {
    if (entry.is_regular_file()) {
      const auto name = entry.path().filename().string();
      if (name.ends_with(".safetensors")) {
        is_safetensors = true;
        break;
      }
    }
  }

  const auto descriptor_path = artifact_path / "artifact_descriptor.json";
  if (std::filesystem::exists(descriptor_path)) {
    std::ifstream f(descriptor_path);
    if (f.is_open()) {
      try {
        nlohmann::json j;
        f >> j;
        if (j.contains("schema_version") && j["schema_version"].is_string()) {
          descriptor_schema_version = j["schema_version"].get<std::string>();
        }
        if (j.contains("index_multihash") && j["index_multihash"].is_string()) {
          existing_index_mh = j["index_multihash"].get<std::string>();
        }
        if (j.contains("data_multihash") && j["data_multihash"].is_string()) {
          existing_data_mh = j["data_multihash"].get<std::string>();
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "Ignoring malformed artifact_descriptor.json: " << e.what();
      }
    }
  }

  const bool descriptor_present = std::filesystem::exists(descriptor_path);
  if (descriptor_present && !descriptor_schema_version.has_value()) {
    return absl::FailedPreconditionError(
        "artifact_descriptor.json missing schema_version; canonical index v3 required");
  }
  if (descriptor_schema_version.has_value() && *descriptor_schema_version != "v3") {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unsupported artifact descriptor schema_version='",
        *descriptor_schema_version,
        "'; canonical index v3 is required"));
  }

  if (!canonical_index_json.has_value()) {
    if (is_safetensors) {
      if (existing_index_mh.has_value()) {
        computed_index_mh = existing_index_mh;
      }
      std::vector<std::filesystem::path> st_files;
      for (const auto& entry : std::filesystem::directory_iterator(artifact_path)) {
        if (entry.is_regular_file()) {
          const auto name = entry.path().filename().string();
          if (name.ends_with(".safetensors")) {
            st_files.push_back(entry.path());
          }
        }
      }
      auto index_bytes_or = loader::BuildCanonicalIndexFromSafetensors(st_files);
      if (index_bytes_or.ok()) {
        canonical_index_json = index_bytes_or.value();
        if (!computed_index_mh.has_value()) {
          auto index_mh_or = common::compute_index_multihash(std::optional<std::string>(*canonical_index_json), "");
          if (index_mh_or.ok()) {
            computed_index_mh = *index_mh_or;
          } else {
            LOG(WARNING) << "Index multihash computation failed: " << index_mh_or.status();
          }
        }
        try {
          nlohmann::json idx_json = nlohmann::json::parse(*canonical_index_json, nullptr, true);
          for (auto it = idx_json.begin(); it != idx_json.end(); ++it) {
            const auto& arr = it.value();
            if (!arr.is_array() || arr.size() < 2) {
              continue;
            }
            uint64_t off = arr[0].get<uint64_t>();
            uint64_t sz = arr[1].get<uint64_t>();
            logical_total_size = std::max<uint64_t>(logical_total_size, off + sz);
          }
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to parse canonical index for total_size: " << e.what();
        }
      } else {
        LOG(WARNING) << "Failed to build canonical index from safetensors: " << index_bytes_or.status();
      }
    } else {
      const auto index_json_path = artifact_path / "tensor_index.json";
      try {
        std::string raw_json;
        if (std::filesystem::exists(index_json_path)) {
          std::ifstream f(index_json_path);
          nlohmann::json j;
          f >> j;
          raw_json = j.dump();
        }
        if (!raw_json.empty()) {
          auto rebuilt_or = loader::rebuild_stable_canonical_index(raw_json, target_device_id);
          const std::string& canonical_json = rebuilt_or.ok() ? *rebuilt_or : raw_json;
          canonical_index_json = canonical_json;
          auto index_mh_or = common::compute_index_multihash(std::optional<std::string>(canonical_json), "");
          if (index_mh_or.ok()) {
            computed_index_mh = *index_mh_or;
          } else {
            LOG(WARNING) << "Index multihash computation failed: " << index_mh_or.status();
          }
          try {
            nlohmann::json idx_json = nlohmann::json::parse(canonical_json);
            for (auto it = idx_json.begin(); it != idx_json.end(); ++it) {
              const auto& arr = it.value();
              if (!arr.is_array() || arr.size() < 2) {
                continue;
              }
              uint64_t off = arr[0].get<uint64_t>();
              uint64_t sz = arr[1].get<uint64_t>();
              logical_total_size = std::max<uint64_t>(logical_total_size, off + sz);
            }
          } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to parse canonical index JSON for total_size: " << e.what();
          }
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to read/parse tensor_index.json: " << e.what();
      }
    }
  }

  if (!resolved_view_plan.has_value() && hints.variant.has_value()) {
    if (hints.variant->cached_plan.has_value()) {
      resolved_view_plan = *hints.variant->cached_plan;
    } else if (hints.variant->view_spec.has_value()) {
      if (!canonical_index_json.has_value()) {
        return absl::FailedPreconditionError(
            "Variant view requested but canonical index bytes are unavailable for planning");
      }
      auto plan_or = compute_view_plan(*canonical_index_json, *hints.variant->view_spec);
      if (!plan_or.ok()) {
        return plan_or.status();
      }
      resolved_view_plan = std::move(*plan_or);
    }
  }

  if (resolved_view_plan.has_value() && resolved_view_plan->view_size_bytes > 0) {
    logical_total_size = resolved_view_plan->view_size_bytes;
  }

  // Get or create replica
  replica::ReplicaConfig config{
      .source = resolved_source,
      .artifact_identifier = artifact_identifier,
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool = memory_pool_,
      .virtual_addr_space = va_space_,
      .expected_artifact_size =
          resolved_view_plan.has_value() ? std::optional<uint64_t>(resolved_view_plan->view_size_bytes) : std::nullopt};
  config.pinned_memory_timeout = hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : pinned_memory_timeout_;
  config.max_buffer_bytes = hints.max_buffer_bytes;
  config.view_id = hints.variant ? hints.variant->view_id : std::nullopt;
  config.view_plan = resolved_view_plan;
  config.transform_placement = hints.variant ? hints.variant->placement : loading::TransformPlacement::kServer;
  if (target_location == common::memory::MemoryLocation::GPU) {
    config.local_device_id = target_device_id;
  }
  config.device_type = (target_location == common::memory::MemoryLocation::GPU) ? DeviceType::GPU : DeviceType::CPU;

  auto replica = get_or_create_replica(artifact_identifier, config);
  if (!replica) {
    return absl::InternalError("Failed to create replica");
  }

  // Start async loading
  std::optional<int> opt_dev;
  if (target_location == common::memory::MemoryLocation::GPU) {
    opt_dev = target_device_id;
  }

  auto load_future = replica->ensure_loaded_async(target_location, num_thread_, opt_dev);

  // Wait for allocation
  const auto allocation_timeout = hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : pinned_memory_timeout_;
  // Treat a timeout of 0 (the default used by many tests) as "wait indefinitely" rather than
  // returning immediately.  Mapping to `absl::InfiniteDuration()` avoids spurious
  // `DEADLINE_EXCEEDED` errors during unit-tests that intentionally rely on the default.
  absl::Duration wait_duration =
      (allocation_timeout.count() > 0) ? absl::Milliseconds(allocation_timeout.count()) : absl::InfiniteDuration();

  auto wait_status =
      replica->get_memory_manager().wait_for_state(target_location, replica::MemoryState::LOADED, wait_duration);

  // ------------------------------------------------------------------
  // NEW (Phase 3.2-3): On GPU allocation failure attempt eviction + retry
  // ------------------------------------------------------------------
  if (!wait_status.ok() && target_location == common::memory::MemoryLocation::GPU) {
    // Approximate bytes we need = artifact size (may be 0 if unknown)
    size_t required_bytes = 0;
    if (auto sz_or = replica->get_artifact_size(); sz_or.ok()) {
      required_bytes = *sz_or;
    }

    LOG(WARNING) << "ingest_from_disk_internal(): initial GPU allocation failed (" << wait_status
                 << "). Attempting GPU eviction on device " << target_device_id << " for ~" << required_bytes
                 << " bytes.";

    auto evict_st = try_evict_gpu_memory_impl(
        *replica_registry_, *device_manager_, *metrics_collector_, target_device_id, required_bytes);

    if (evict_st.ok()) {
      // Reset replica GPU memory state then retry loading.
      {
        absl::Status _st = replica->release_memory(common::memory::MemoryLocation::GPU);
        if (!_st.ok()) {
          LOG(WARNING) << "release_memory(GPU) failed during retry after eviction: " << _st;
        }
      }

      // Trigger load again.
      load_future = replica->ensure_loaded_async(target_location, num_thread_, opt_dev);

      wait_status =
          replica->get_memory_manager().wait_for_state(target_location, replica::MemoryState::LOADED, wait_duration);

      if (!wait_status.ok()) {
        LOG(WARNING) << "ingest_from_disk_internal(): Retry after eviction still failed: " << wait_status;
      }
    } else {
      LOG(WARNING) << "ingest_from_disk_internal(): GPU eviction did not free enough memory: " << evict_st;
    }
  }

  if (!wait_status.ok()) {
    return wait_status;
  }

  // Ensure async load (TransferService + pump) fully completed before accessing GPU memory for verification.
  if (load_future.valid()) {
    load_future.wait();
    const absl::Status& load_status = load_future.get();
    if (!load_status.ok()) {
      return load_status;
    }
  }

  if (resolved_view_plan.has_value() && !resolved_view_plan->is_identity) {
    std::optional<int> gpu_device =
        (target_location == MemoryLocation::GPU) ? std::optional<int>(target_device_id) : std::nullopt;
    view_byte_space_hash =
        ComputeViewDataHash(*replica, target_location, resolved_view_plan->view_size_bytes, gpu_device);
  }

  // RFC-0007: After loading, compute/verify content-addressed identity when possible.
  // Only perform strong verification when the replica is resident in GPU memory (fast path).

  // Compute data multihash from GPU memory when requested by hints
  uint64_t verify_size = 0;
  const bool force_full_digest = options_.force_full_digest_on_load;
  if (hints.verify == MaterializeHints::Verify::FULL_DIGEST || force_full_digest) {
    if (auto sz_or = replica->get_artifact_size(); sz_or.ok()) {
      verify_size = *sz_or;
    }
    if (target_location == MemoryLocation::GPU && verify_size > 0) {
      auto view_or = replica->get_memory_manager().get_gpu_allocation_view();
      if (view_or.ok()) {
        [[maybe_unused]] auto keep_gpu_allocation = view_or->allocation;
        auto data_mh_or = loader::compute_data_multihash_from_gpu_memory(
            gsl::not_null<void*>{view_or->base_ptr}, verify_size, target_device_id);
        if (data_mh_or.ok()) {
          computed_data_mh = *data_mh_or;
        } else {
          LOG(WARNING) << "Data multihash computation (GPU) failed: " << data_mh_or.status();
        }
      }
    } else if (target_location == MemoryLocation::CPU) {
      const auto cpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::CPU);
      if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr && verify_size > 0) {
        auto data_mh_or =
            loader::compute_data_multihash_from_cpu_memory(gsl::not_null<const void*>{cpu_ptrs[0]}, verify_size);
        if (data_mh_or.ok()) {
          computed_data_mh = *data_mh_or;
        } else {
          LOG(WARNING) << "Data multihash computation (CPU) failed: " << data_mh_or.status();
        }
      }
    }
  }

  // Compute or obtain index multihash
  if (is_safetensors) {
    if (existing_index_mh.has_value()) {
      computed_index_mh = existing_index_mh; // trust descriptor when present
    }
    std::vector<std::filesystem::path> st_files;
    for (const auto& entry : std::filesystem::directory_iterator(artifact_path)) {
      if (entry.is_regular_file()) {
        const auto name = entry.path().filename().string();
        if (name.ends_with(".safetensors")) {
          st_files.push_back(entry.path());
        }
      }
    }
    auto index_bytes_or = loader::BuildCanonicalIndexFromSafetensors(st_files);
    if (index_bytes_or.ok()) {
      canonical_index_json = index_bytes_or.value();
      if (!computed_index_mh.has_value()) {
        auto index_mh_or = common::compute_index_multihash(std::optional<std::string>(*canonical_index_json), "");
        if (index_mh_or.ok()) {
          computed_index_mh = *index_mh_or;
        } else {
          LOG(WARNING) << "Index multihash computation failed: " << index_mh_or.status();
        }
      }
      try {
        nlohmann::json idx_json = nlohmann::json::parse(*canonical_index_json, nullptr, true);
        for (auto it = idx_json.begin(); it != idx_json.end(); ++it) {
          const auto& arr = it.value();
          if (!arr.is_array() || arr.size() < 2) {
            continue;
          }
          uint64_t off = arr[0].get<uint64_t>();
          uint64_t sz = arr[1].get<uint64_t>();
          logical_total_size = std::max<uint64_t>(logical_total_size, off + sz);
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to parse canonical index for total_size: " << e.what();
      }
    } else {
      LOG(WARNING) << "Failed to build canonical index from safetensors: " << index_bytes_or.status();
    }
  } else {
    // Standard partition format – read tensor_index.json and canonicalize bytes via nlohmann::json
    const auto index_json_path = artifact_path / "tensor_index.json";
    try {
      // Read canonical index (JSON), then rebuild with stable grouping
      std::string raw_json;
      if (std::filesystem::exists(index_json_path)) {
        std::ifstream f(index_json_path);
        nlohmann::json j;
        f >> j;
        raw_json = j.dump();
      }
      if (!raw_json.empty()) {
        // Apply stable canonicalization using C++ authority
        auto rebuilt_or = loader::rebuild_stable_canonical_index(raw_json, target_device_id);
        const std::string& canonical_json = rebuilt_or.ok() ? *rebuilt_or : raw_json;
        canonical_index_json = canonical_json;
        auto index_mh_or = common::compute_index_multihash(std::optional<std::string>(canonical_json), "");
        if (index_mh_or.ok()) {
          computed_index_mh = *index_mh_or;
        } else {
          LOG(WARNING) << "Index multihash computation failed: " << index_mh_or.status();
        }
        // Determine logical total size from canonical index JSON
        try {
          nlohmann::json idx_json = nlohmann::json::parse(canonical_json);
          for (auto it = idx_json.begin(); it != idx_json.end(); ++it) {
            const auto& arr = it.value();
            if (!arr.is_array() || arr.size() < 2) {
              continue;
            }
            uint64_t off = arr[0].get<uint64_t>();
            uint64_t sz = arr[1].get<uint64_t>();
            logical_total_size = std::max<uint64_t>(logical_total_size, off + sz);
          }
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to parse canonical index JSON for total_size: " << e.what();
        }
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to read/parse tensor_index.json: " << e.what();
    }
  }

  if (!canonical_index_json.has_value()) {
    if (is_safetensors) {
      if (existing_index_mh.has_value()) {
        computed_index_mh = existing_index_mh;
      }
      std::vector<std::filesystem::path> st_files;
      for (const auto& entry : std::filesystem::directory_iterator(artifact_path)) {
        if (entry.is_regular_file()) {
          const auto name = entry.path().filename().string();
          if (name.ends_with(".safetensors")) {
            st_files.push_back(entry.path());
          }
        }
      }
      auto index_bytes_or = loader::BuildCanonicalIndexFromSafetensors(st_files);
      if (index_bytes_or.ok()) {
        canonical_index_json = index_bytes_or.value();
        if (!computed_index_mh.has_value()) {
          auto index_mh_or = common::compute_index_multihash(std::optional<std::string>(*canonical_index_json), "");
          if (index_mh_or.ok()) {
            computed_index_mh = *index_mh_or;
          } else {
            LOG(WARNING) << "Index multihash computation failed: " << index_mh_or.status();
          }
        }
        try {
          nlohmann::json idx_json = nlohmann::json::parse(*canonical_index_json, nullptr, true);
          for (auto it = idx_json.begin(); it != idx_json.end(); ++it) {
            const auto& arr = it.value();
            if (!arr.is_array() || arr.size() < 2) {
              continue;
            }
            uint64_t off = arr[0].get<uint64_t>();
            uint64_t sz = arr[1].get<uint64_t>();
            logical_total_size = std::max<uint64_t>(logical_total_size, off + sz);
          }
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to parse canonical index for total_size: " << e.what();
        }
      } else {
        LOG(WARNING) << "Failed to build canonical index from safetensors: " << index_bytes_or.status();
      }
    } else {
      const auto index_json_path = artifact_path / "tensor_index.json";
      try {
        std::string raw_json;
        if (std::filesystem::exists(index_json_path)) {
          std::ifstream f(index_json_path);
          nlohmann::json j;
          f >> j;
          raw_json = j.dump();
        }
        if (!raw_json.empty()) {
          auto rebuilt_or = loader::rebuild_stable_canonical_index(raw_json, target_device_id);
          const std::string& canonical_json = rebuilt_or.ok() ? *rebuilt_or : raw_json;
          canonical_index_json = canonical_json;
          auto index_mh_or = common::compute_index_multihash(std::optional<std::string>(canonical_json), "");
          if (index_mh_or.ok()) {
            computed_index_mh = *index_mh_or;
          } else {
            LOG(WARNING) << "Index multihash computation failed: " << index_mh_or.status();
          }
          try {
            nlohmann::json idx_json = nlohmann::json::parse(canonical_json);
            for (auto it = idx_json.begin(); it != idx_json.end(); ++it) {
              const auto& arr = it.value();
              if (!arr.is_array() || arr.size() < 2) {
                continue;
              }
              uint64_t off = arr[0].get<uint64_t>();
              uint64_t sz = arr[1].get<uint64_t>();
              logical_total_size = std::max<uint64_t>(logical_total_size, off + sz);
            }
          } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to parse canonical index JSON for total_size: " << e.what();
          }
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to read/parse tensor_index.json: " << e.what();
      }
    }
  }

  if (!resolved_view_plan.has_value() && hints.variant.has_value()) {
    if (hints.variant->cached_plan.has_value()) {
      resolved_view_plan = *hints.variant->cached_plan;
    } else if (hints.variant->view_spec.has_value()) {
      if (!canonical_index_json.has_value()) {
        return absl::FailedPreconditionError(
            "Variant view requested but canonical index bytes are unavailable for planning");
      }
      auto plan_or = compute_view_plan(*canonical_index_json, *hints.variant->view_spec);
      if (!plan_or.ok()) {
        return plan_or.status();
      }
      resolved_view_plan = std::move(*plan_or);
    }
  }
  if (resolved_view_plan.has_value() && resolved_view_plan->view_size_bytes > 0) {
    logical_total_size = resolved_view_plan->view_size_bytes;
  }

  if (resolved_view_plan.has_value() && !resolved_view_plan->is_identity && !view_byte_space_hash.has_value()) {
    std::optional<int> gpu_device =
        (target_location == MemoryLocation::GPU) ? std::optional<int>(target_device_id) : std::nullopt;
    view_byte_space_hash =
        ComputeViewDataHash(*replica, target_location, resolved_view_plan->view_size_bytes, gpu_device);
  }

  // If descriptor exists, verify data_multihash matches when we computed it
  if (existing_data_mh.has_value() && computed_data_mh.has_value()) {
    if (*existing_data_mh != *computed_data_mh) {
      return absl::DataLossError("ARTIFACT_ID_MISMATCH: data_multihash does not match loaded data");
    }
  }

  const bool variant_requested = hints.variant.has_value();
  const std::optional<std::string> requested_view_id =
      (variant_requested && hints.variant->view_id.has_value()) ? hints.variant->view_id : std::optional<std::string>{};
  const bool allow_verification_metadata = !variant_requested || requested_view_id.has_value();
  const std::string expected_byte_space_id = requested_view_id.value_or("");

  // Default post-load lightweight verification when descriptor is present
  if (std::filesystem::exists(descriptor_path) && allow_verification_metadata) {
    try {
      struct VerificationBuffers {
        std::vector<void*> ptrs;
        std::vector<size_t> sizes;
        int device_id{-1};
        uint64_t total_size{0};
        std::shared_ptr<common::memory::GpuDeviceMemory> gpu_guard;
      };

      auto prepare_buffers = [&]() -> VerificationBuffers {
        VerificationBuffers buffers;
        uint64_t resolved_size = 0;
        if (auto sz_or = replica->get_artifact_size(); sz_or.ok()) {
          resolved_size = *sz_or;
        } else {
          resolved_size = logical_total_size;
        }
        buffers.total_size = resolved_size;
        if (target_location == MemoryLocation::GPU && buffers.total_size > 0) {
          auto view_or = replica->get_memory_manager().get_gpu_allocation_view();
          if (view_or.ok()) {
            buffers.ptrs.push_back(view_or->base_ptr);
            buffers.sizes.push_back(static_cast<size_t>(buffers.total_size));
            buffers.device_id = target_device_id;
            buffers.gpu_guard = view_or->allocation;
          }
        } else {
          const auto cpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::CPU);
          if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr && buffers.total_size > 0) {
            buffers.ptrs.push_back(cpu_ptrs[0]);
            buffers.sizes.push_back(static_cast<size_t>(buffers.total_size));
          }
        }
        return buffers;
      };

      const auto verification_path = verification_path_for_view(artifact_path, requested_view_id);
      bool regenerate_metadata = !std::filesystem::exists(verification_path);
      bool metadata_verified = false;

      if (!regenerate_metadata) {
        std::ifstream vf(verification_path);
        if (vf.is_open()) {
          std::stringstream vbuf;
          vbuf << vf.rdbuf();
          vf.close();
          auto ver_or = common::ArtifactVerificationInfo::from_json(vbuf.str());
          if (ver_or.ok()) {
            const common::ArtifactVerificationInfo& info = *ver_or;
            if ((!expected_byte_space_id.empty() || !info.byte_space_id.empty()) &&
                info.byte_space_id != expected_byte_space_id) {
              LOG(WARNING) << "Verification metadata at '" << verification_path.string()
                           << "' was recorded for byte_space_id='" << info.byte_space_id << "', expected '"
                           << expected_byte_space_id << "'; regenerating.";
              regenerate_metadata = true;
            } else {
              auto buffers = prepare_buffers();
              if (!buffers.ptrs.empty()) {
                absl::Status vstatus = common::ArtifactVerifier::verify_artifact_data(
                    buffers.ptrs, buffers.sizes, info, common::VerificationLevel::SEGMENT_HASHES, buffers.device_id);
                if (!vstatus.ok()) {
                  return absl::DataLossError(
                      absl::StrCat("ARTIFACT_ID_MISMATCH: verification failed: ", vstatus.message()));
                }
                metadata_verified = true;
              } else {
                metadata_verified = true; // Nothing to verify (empty replica)
              }
            }
          } else {
            LOG(WARNING) << "Failed to parse verification metadata at '" << verification_path.string()
                         << "': " << ver_or.status();
            regenerate_metadata = true;
          }
        } else {
          regenerate_metadata = true;
        }
      }

      if (!metadata_verified && regenerate_metadata) {
        auto buffers = prepare_buffers();
        if (!buffers.ptrs.empty()) {
          auto gen_or = common::ArtifactVerifier::generate_verification_info(
              buffers.ptrs, buffers.sizes, buffers.device_id, common::VerificationLevel::SEGMENT_HASHES);
          if (gen_or.ok()) {
            auto info = *gen_or;
            info.byte_space_id = expected_byte_space_id;
            try {
              std::ofstream vf(verification_path, std::ios::trunc);
              if (vf.is_open()) {
                vf << info.to_json();
                vf.close();
              }
            } catch (const std::exception& e) {
              LOG(WARNING) << "Failed to persist verification metadata at '" << verification_path.string()
                           << "': " << e.what();
            }
          }
        }
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Post-load verification skipped due to error: " << e.what();
    }
  } else if (std::filesystem::exists(descriptor_path) && !allow_verification_metadata) {
    VLOG(1) << "Skipping verification metadata reuse for unnamed view variant (no view_id provided).";
  }

  // If safetensors path lacks descriptor, write it back computing hashes as needed
  if (is_safetensors && !std::filesystem::exists(descriptor_path)) {
    // Ensure data multihash is computed even when FULL_DIGEST is not requested
    if (!computed_data_mh.has_value()) {
      uint64_t total = logical_total_size;
      if (total == 0) {
        if (auto sz_or = replica->get_artifact_size(); sz_or.ok()) {
          total = *sz_or;
        }
      }
      if (total > 0) {
        if (target_location == MemoryLocation::GPU) {
          auto view_or = replica->get_memory_manager().get_gpu_allocation_view();
          if (view_or.ok()) {
            [[maybe_unused]] auto keep_gpu_allocation = view_or->allocation;
            auto mh_or = loader::compute_data_multihash_from_gpu_memory(
                gsl::not_null<void*>{view_or->base_ptr}, total, target_device_id);
            if (mh_or.ok()) {
              computed_data_mh = *mh_or;
            }
          }
        } else {
          const auto cpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::CPU);
          if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr) {
            auto mh_or = loader::compute_data_multihash_from_cpu_memory(gsl::not_null<const void*>{cpu_ptrs[0]}, total);
            if (mh_or.ok()) {
              computed_data_mh = *mh_or;
            }
          }
        }
      }
    }

    if (computed_index_mh.has_value() && computed_data_mh.has_value()) {
      try {
        // 1) Persist artifact_descriptor.json
        nlohmann::json j;
        j["artifact_id"] = std::string("mi2:") + *computed_index_mh + ":" + *computed_data_mh;
        j["index_multihash"] = *computed_index_mh;
        j["data_multihash"] = *computed_data_mh;
        j["schema_version"] = "v3";
        // Encoding is JSON only per project decision
        const auto index_json_path = artifact_path / "tensor_index.json";
        j["encoding"] = "json";
        uint64_t total_for_desc = logical_total_size;
        if (total_for_desc == 0) {
          if (auto sz_or = replica->get_artifact_size(); sz_or.ok()) {
            total_for_desc = *sz_or;
          }
        }
        j["total_size"] = total_for_desc;
        nlohmann::json hp;
        hp["chunk_size"] = 4 * 1024 * 1024;
        hp["fanout"] = 2;
        hp["algorithm"] = "sha2-256";
        j["hash_params"] = hp;
        std::ofstream of(descriptor_path);
        if (!of.is_open()) {
          return absl::PermissionDeniedError("DESCRIPTOR_NOT_WRITABLE: cannot write artifact_descriptor.json");
        }
        of << j.dump(2);

        // 2) Optionally persist canonical index (JSON) if not already present
        if (!std::filesystem::exists(index_json_path)) {
          // Rebuild canonical index bytes from safetensors headers
          std::vector<std::filesystem::path> st_files;
          for (const auto& entry : std::filesystem::directory_iterator(artifact_path)) {
            if (entry.is_regular_file()) {
              const auto name = entry.path().filename().string();
              if (name.ends_with(".safetensors")) {
                st_files.push_back(entry.path());
              }
            }
          }
          auto index_bytes_or = loader::BuildCanonicalIndexFromSafetensors(st_files);
          if (index_bytes_or.ok()) {
            // Write JSON index directly
            std::ofstream oj(index_json_path);
            if (!oj.is_open()) {
              return absl::PermissionDeniedError("DESCRIPTOR_NOT_WRITABLE: cannot write tensor_index.json");
            }
            oj << index_bytes_or.value();
            oj.close();
          }
        }
      } catch (const std::exception& e) {
        return absl::PermissionDeniedError(std::string("DESCRIPTOR_NOT_WRITABLE: ") + e.what());
      }
    }
  }

  // Build result using new ReplicaHandle structure
  loading::ReplicaHandle handle;

  // Compose ReplicaKey
  DeviceKey dev_key;
  if (target_location == common::memory::MemoryLocation::GPU) {
    dev_key = DeviceKey{.type = DeviceType::GPU, .ordinal = target_device_id, .uuid = ""};
  } else {
    dev_key = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  }
  handle.replica_key = loading::ReplicaKey{
      .artifact_id = artifact_identifier, .view_id = config.view_id, .device = dev_key, .replica = 0};

  // Loading future and states
  handle.ready_future = load_future;
  handle.cpu_state = replica->get_memory_state(common::memory::MemoryLocation::CPU);
  handle.gpu_state = replica->get_memory_state(common::memory::MemoryLocation::GPU);

  if (target_location == common::memory::MemoryLocation::GPU) {
    const auto gpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
    handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

    // Attempt to obtain CUDA IPC handle bytes for the allocated GPU buffer.
    auto ipc_or = replica->get_memory_manager().get_ipc_handle();
    if (ipc_or.ok()) {
      std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
    }
  }

  if (resolved_view_plan.has_value() && !resolved_view_plan->is_identity) {
    handle.view_index_json = resolved_view_plan->view_index_json;
  }
  if (view_byte_space_hash.has_value()) {
    handle.view_data_hash = view_byte_space_hash;
  }

  // Update metrics
  const auto duration = std::chrono::steady_clock::now() - start_time;
  const double duration_s = std::chrono::duration<double>(duration).count();
  metrics_collector_->record_operation("load_from_disk", duration_s); // no-op after Phase 5
  // Unified artifact load metric with labels
  std::optional<std::string_view> view_scope =
      (config.view_id.has_value() ? std::optional<std::string_view>(*config.view_id) : std::nullopt);
  metrics_collector_->record_artifact_load(
      /*source=*/"disk",
      /*device=*/(target_location == common::memory::MemoryLocation::GPU ? std::string("gpu") : std::string("cpu")),
      /*phase=*/"finalize",
      duration_s,
      view_scope);
  metrics_collector_->update_all_metrics(*memory_pool_, *replica_registry_, *device_manager_);

  return handle;
}

absl::StatusOr<loading::ReplicaHandle> StoreEngine::ingest_from_p2p_internal(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  const auto start_time = std::chrono::steady_clock::now();
  const std::string request_id = absl::StrCat("p2p_", absl::ToUnixNanos(absl::Now()));
  SC_TRACE_INIT_GUARD(request_id, artifact_identifier, "ingest_from_p2p_internal");

  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.store");
  otel::trace::StartSpanOptions span_opts;
  // Treat this as an internal span within the daemon process
  span_opts.kind = otel::trace::SpanKind::kInternal;
  // Parent-child relationship is inferred from the current context already.
  auto p2p_span = tracer->StartSpan("StoreEngine/P2PIngest", span_opts);
  otel::trace::Scope p2p_scope(p2p_span);
  // Set standard attributes and business attributes per RFC schema
  p2p_span->SetAttribute("component", "StoreEngine");
  p2p_span->SetAttribute("tc.source.type", "remote");
  p2p_span->SetAttribute("tc.source.address", source.ip);
  p2p_span->SetAttribute("tc.p2p.port", static_cast<int64_t>(source.port));
  p2p_span->SetAttribute("tc.size.bytes", static_cast<int64_t>(source.size_bytes));
  p2p_span->SetAttribute("tc.location", target.location.type == common::memory::MemoryLocation::GPU ? "gpu" : "cpu");
  if (hints.variant && hints.variant->view_id.has_value()) {
    p2p_span->SetAttribute("tc.view.id", *hints.variant->view_id);
  }

  if (!comm_manager_->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }

  std::optional<loader::ViewPlan> resolved_view_plan;
  std::optional<std::string> view_byte_space_hash;
  if (hints.variant.has_value()) {
    if (hints.variant->cached_plan.has_value()) {
      resolved_view_plan = *hints.variant->cached_plan;
    } else if (hints.variant->view_spec.has_value()) {
      std::optional<std::string> canonical_index_json = hints.variant->canonical_index_json;
      if (!canonical_index_json.has_value()) {
        // Fetch canonical index bytes from Global Store.
        auto idx_or = get_canonical_index_by_id(artifact_identifier);
        if (!idx_or.ok()) {
          return idx_or.status();
        }
        canonical_index_json = std::move(idx_or).value();
      }
      if (canonical_index_json->empty()) {
        return absl::FailedPreconditionError("Canonical index bytes required for view planning are empty");
      }
      auto plan_or = compute_view_plan(*canonical_index_json, *hints.variant->view_spec);
      if (!plan_or.ok()) {
        return plan_or.status();
      }
      resolved_view_plan = std::move(*plan_or);
    }
  }

  common::memory::MemoryLocation target_location = common::memory::MemoryLocation::CPU;
  if (target.location.type == common::memory::MemoryLocation::GPU) {
    target_location = common::memory::MemoryLocation::GPU;
  }

  // Create replica with P2P source
  auto p2p_source = source;
  p2p_source.comm_engine =
      gsl::not_null<std::shared_ptr<communicator::engine::Communicator>>{comm_manager_->get_shared_engine()};
  // Provide optional disk fallback directory from engine options
  p2p_source.fallback_disk_dir = options_.p2p_fallback_disk_dir;
  replica::ReplicaConfig config{
      .source = p2p_source,
      .artifact_identifier = artifact_identifier,
      .device_type = (target_location == common::memory::MemoryLocation::GPU ? DeviceType::GPU : DeviceType::CPU),
      .local_device_id = target.location.device_id,
      .pinned_buffer_pool = memory_pool_,
      .virtual_addr_space = va_space_,
      .expected_artifact_size =
          resolved_view_plan.has_value() ? std::optional<uint64_t>(resolved_view_plan->view_size_bytes) : std::nullopt};
  config.pinned_memory_timeout = hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : pinned_memory_timeout_;
  config.local_device_id = target.location.device_id;
  config.max_buffer_bytes = hints.max_buffer_bytes;
  config.p2p_comm_enabled = true;
  config.device_type = (target_location == common::memory::MemoryLocation::GPU) ? DeviceType::GPU : DeviceType::CPU;
  config.view_id = hints.variant ? hints.variant->view_id : std::nullopt;
  config.view_plan = resolved_view_plan;
  config.transform_placement = hints.variant ? hints.variant->placement : loading::TransformPlacement::kServer;

  auto replica = get_or_create_replica(artifact_identifier, config);
  if (!replica) {
    return absl::InternalError("Failed to create replica");
  }

  // Load synchronously for remote (maintain existing behavior)
  auto load_future =
      replica->ensure_loaded_async(target_location, num_thread_, std::optional<int>(target.location.device_id));
  auto status = load_future.get();

  if (!status.ok()) {
    p2p_span->SetAttribute("error", true);
    p2p_span->AddEvent("p2p_ingest_error", {{"message", std::string(status.message())}});
    if (absl::IsResourceExhausted(status)) {
      // Try to evict memory
      LOG(WARNING) << "Resource exhausted, attempting memory eviction";
      auto evict_status = try_evict_memory_for_replica(source.size_bytes);
      if (evict_status.ok()) {
        // Retry
        p2p_span->AddEvent("p2p_ingest_retry_after_eviction");
        load_future =
            replica->ensure_loaded_async(target_location, num_thread_, std::optional<int>(target.location.device_id));
        status = load_future.get();
      }
    }

    if (!status.ok()) {
      metrics_collector_->record_p2p_transfer(0, false);
      p2p_span->End();
      return status;
    }
  }

  // Build result using new ReplicaHandle structure
  loading::ReplicaHandle handle;

  DeviceKey dev_key;
  if (target_location == common::memory::MemoryLocation::GPU) {
    dev_key = DeviceKey{.type = DeviceType::GPU, .ordinal = target.location.device_id, .uuid = ""};
  } else {
    dev_key = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  }
  handle.replica_key = loading::ReplicaKey{
      .artifact_id = artifact_identifier, .view_id = config.view_id, .device = dev_key, .replica = 0};

  // Receiver-side verification: if the P2P source provides verification
  // metadata (JSON), verify the loaded replica before returning.
  if (!p2p_source.verification_json.empty()) {
    auto info_or = common::ArtifactVerificationInfo::from_json(p2p_source.verification_json);
    if (!info_or.ok()) {
      LOG(WARNING) << "P2P verification_json parse failed: " << info_or.status();
      return absl::DataLossError("verification_json parse failed");
    }
    const auto& info = *info_or;
    const auto verify_loc = target_location;
    auto vst = replica->verify_key_points(verify_loc, info);
    if (!vst.ok()) {
      LOG(ERROR) << "Receiver-side verification failed for artifact '" << artifact_identifier << "': " << vst;
      return absl::DataLossError(std::string(vst.message()));
    }
  }

  if (resolved_view_plan.has_value() && !resolved_view_plan->is_identity) {
    std::optional<int> gpu_device = (target_location == common::memory::MemoryLocation::GPU)
        ? std::optional<int>(target.location.device_id)
        : std::nullopt;
    view_byte_space_hash =
        ComputeViewDataHash(*replica, target_location, resolved_view_plan->view_size_bytes, gpu_device);
  }

  // Ready future is already resolved (synchronous path)
  std::promise<absl::Status> promise;
  promise.set_value(absl::OkStatus());
  handle.ready_future = promise.get_future().share();

  handle.cpu_state = replica->get_memory_state(common::memory::MemoryLocation::CPU);
  handle.gpu_state = replica->get_memory_state(common::memory::MemoryLocation::GPU);

  if (target_location == common::memory::MemoryLocation::GPU) {
    const auto gpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
    handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

    auto ipc_or = replica->get_memory_manager().get_ipc_handle();
    if (ipc_or.ok()) {
      std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
    }
  }

  if (resolved_view_plan.has_value() && !resolved_view_plan->is_identity) {
    handle.view_index_json = resolved_view_plan->view_index_json;
  }
  if (view_byte_space_hash.has_value()) {
    handle.view_data_hash = view_byte_space_hash;
  }

  // Update metrics
  metrics_collector_->record_p2p_transfer(source.size_bytes, true);
  const auto duration = std::chrono::steady_clock::now() - start_time;
  const double duration_s = std::chrono::duration<double>(duration).count();
  metrics_collector_->record_operation("load_from_p2p", duration_s);
  // Unified artifact load metric with labels
  std::optional<std::string_view> view_scope =
      (config.view_id.has_value() ? std::optional<std::string_view>(*config.view_id) : std::nullopt);
  metrics_collector_->record_artifact_load(
      /*source=*/"remote",
      /*device=*/(target_location == common::memory::MemoryLocation::GPU ? std::string("gpu") : std::string("cpu")),
      /*phase=*/"finalize",
      duration_s,
      view_scope);
  metrics_collector_->update_all_metrics(*memory_pool_, *replica_registry_, *device_manager_);

  p2p_span->AddEvent("p2p_ingest_complete", {{"bytes", static_cast<int64_t>(source.size_bytes)}});
  p2p_span->End();
  return handle;
}

absl::StatusOr<loading::ReplicaHandle> StoreEngine::ingest_from_buffer_internal(
    const std::string& /*artifact_identifier*/,
    const loading::InlineBufferSource& /*source*/,
    const loading::ReplicaTarget& /*target*/,
    const loading::MaterializeHints& /*hints*/) {
  // InlineBufferSource is a newly added type, temporarily returning unimplemented error
  // Future: implement direct replica loading from memory buffer
  return absl::UnimplementedError("InlineBufferSource loading not yet implemented");
}

std::shared_ptr<replica::Replica> StoreEngine::get_or_create_replica(
    const std::string& artifact_identifier,
    const replica::ReplicaConfig& config) {
  // Build ReplicaKey for the requested device (CPU when local_device_id < 0)
  DeviceKey dev_key;
  if (config.device_type == DeviceType::GPU) {
    dev_key = DeviceKey{
        .type = DeviceType::GPU, .ordinal = (config.local_device_id >= 0 ? config.local_device_id : 0), .uuid = ""};
  } else {
    dev_key = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  }
  loading::ReplicaKey inst_key{
      .artifact_id = artifact_identifier, .view_id = config.view_id, .device = dev_key, .replica = 0};

  // Fast-path: already present in registry
  if (auto existing_or = replica_registry_->find(inst_key); existing_or.ok()) {
    return existing_or.value();
  }

  // Create new Replica
  auto replica_create_or = replica::Replica::create(config);
  if (!replica_create_or.ok()) {
    LOG(ERROR) << "Failed to create replica: " << replica_create_or.status().message();
    return nullptr;
  }
  auto replica = std::shared_ptr<replica::Replica>(std::move(replica_create_or.value()));

  // Register in multi-device registry (best effort)
  absl::Status emplace_status =
      replica_registry_->emplace(inst_key, gsl::not_null<std::shared_ptr<replica::Replica>>{replica});

  if (absl::IsAlreadyExists(emplace_status)) {
    // Another thread inserted the instance concurrently. Reuse the existing
    // entry to avoid duplicate replica objects and double-loading.
    if (auto existing_or = replica_registry_->find(inst_key); existing_or.ok()) {
      return existing_or.value();
    }
    // Fall through on error – treat as internal failure.
  } else if (!emplace_status.ok()) {
    // Unexpected error while registering – propagate as failure.
    LOG(ERROR) << "Failed to register replica: " << emplace_status.message();
    return nullptr;
  }

  return replica;
}

absl::Status StoreEngine::try_evict_memory_for_replica(size_t required_size) {
  // Prefer the new multi-device LRU ordering.
  auto lru_instances = replica_registry_->get_lru_instances();

  for (const auto& inst_key : lru_instances) {
    auto replica_or = replica_registry_->find(inst_key);
    if (!replica_or.ok()) {
      continue;
    }

    const auto& replica = replica_or.value();

    // Only attempt to free CPU memory for now – GPU eviction will be handled
    // in a future iteration.
    auto free_status = replica->release_memory(common::memory::MemoryLocation::CPU);
    if (free_status.ok()) {
      metrics_collector_->record_memory_eviction();
      LOG(INFO) << "Evicted replica " << inst_key.artifact_id
                << " (device=" << (inst_key.device.type == DeviceType::CPU ? "CPU" : "GPU") << ":"
                << inst_key.device.ordinal << ") from CPU memory";

      // Check if we have freed enough memory.
      if (memory_pool_->get_available_size() >= required_size) {
        return absl::OkStatus();
      }
    }
  }

  // No further legacy fallbacks – the new multi-device registry covers all
  // cases.  If eviction above fails to free enough memory, report resource
  // exhaustion.

  return absl::ResourceExhaustedError("Could not free enough memory");
}

size_t StoreEngine::get_num_chunk_from_tensor_size(size_t tensor_size) const {
  return (tensor_size + tx_slice_bytes_ - 1) / tx_slice_bytes_;
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-Device Binding – Public API Implementation (Phase-1 bridge)
// ═══════════════════════════════════════════════════════════════════════════

// ------------ ReplicaHandle helpers -------------------------

MemoryState ReplicaHandle::state(DeviceType type) const {
  return type == DeviceType::CPU ? cpu_state : gpu_state;
}

absl::Status ReplicaHandle::wait_ready(std::chrono::milliseconds timeout) {
  if (!ready_future.valid()) {
    // Nothing to wait for – treat as OK.
    return absl::OkStatus();
  }
  const auto status = ready_future.wait_for(timeout);
  if (status == std::future_status::timeout) {
    return absl::DeadlineExceededError("Timeout while waiting for replica to become ready");
  }
  // Future is ready – propagate underlying status value.
  return ready_future.get();
}

// ---------------------------------------------------------------------------
// Bridge implementation of materialize_replica() that internally maps to the legacy load()
// interface.  The new implementation first checks if a Replica already
// exists on the requested device.  If not, it tries COPY_ONLY (GPU peer copy),
// LOAD_ONLY (disk), or AUTO (orchestrator → disk) according to the requested
// mode.  All duplicate fallback branches and GPU-eviction heuristics have been
// removed to keep the codebase lean – MaterializeOrchestrator now owns almost all
// decision complexity.
// ---------------------------------------------------------------------------
absl::StatusOr<ReplicaHandle> StoreEngine::materialize_replica(
    const DeviceKey& target_device,
    MaterializeMode mode,
    const MaterializeHints& hints) {
  const std::optional<loading::VariantIdentity>& variant = hints.variant;

  std::string canonical_artifact_id;
  if (variant.has_value()) {
    if (variant->canonical_artifact_id.empty()) {
      return absl::InvalidArgumentError("VariantIdentity must provide canonical_artifact_id when present");
    }
    if (!hints.artifact_id.empty() && hints.artifact_id != variant->canonical_artifact_id) {
      return absl::InvalidArgumentError(
          "MaterializeHints.artifact_id must match VariantIdentity.canonical_artifact_id");
    }
    canonical_artifact_id = variant->canonical_artifact_id;
  } else {
    canonical_artifact_id = !hints.artifact_id.empty() ? hints.artifact_id : hints.disk_path;
  }

  if (canonical_artifact_id.empty() && mode != MaterializeMode::COPY_ONLY) {
    return absl::InvalidArgumentError("MaterializeHints must provide artifact_id or disk_path for LOAD modes");
  }

  const std::optional<std::string> requested_view_id =
      variant.has_value() ? variant->view_id : std::optional<std::string>{};

  // ────────────────────────────────────────────────────────────────────
  // Validate target device early to avoid entering CUDA paths with
  // invalid ordinals or unsupported device types.
  // ────────────────────────────────────────────────────────────────────
  if (target_device.type == DeviceType::GPU) {
    const int num_gpus = device_manager_->get_num_gpus();
    if (target_device.ordinal < 0 || target_device.ordinal >= num_gpus) {
      return absl::InvalidArgumentError(
          std::string("Invalid GPU device ordinal: ") + std::to_string(target_device.ordinal));
    }
  } else if (target_device.type == DeviceType::CPU) {
    // CPU is supported – no additional validation required.
  } else {
    // For REMOTE/NONE/DISK etc. reject in this implementation.
    return absl::InvalidArgumentError("Unsupported target device type for materialize_replica()");
  }

  // ────────────────────────────────────────────────────────────────────
  // Fast-path: instance already present on the requested device.
  // ────────────────────────────────────────────────────────────────────
  const ReplicaKey dst_key{
      .artifact_id = canonical_artifact_id,
      .view_id = requested_view_id,
      .device = target_device,
      /*replica=*/.replica = 0};
  if (auto existing_or = replica_registry_->find(dst_key); existing_or.ok()) {
    const auto& replica = existing_or.value();

    MemoryLocation dst_loc = (target_device.type == DeviceType::GPU) ? MemoryLocation::GPU : MemoryLocation::CPU;
    std::optional<int> opt_dev;
    if (dst_loc == MemoryLocation::GPU) {
      opt_dev = target_device.ordinal;
    }

    auto fut = replica->ensure_loaded_async(dst_loc, num_thread_, opt_dev);

    ReplicaHandle handle;
    handle.replica_key = dst_key;
    handle.ready_future = fut;
    handle.cpu_state = replica->get_memory_state(MemoryLocation::CPU);
    handle.gpu_state = replica->get_memory_state(MemoryLocation::GPU);

    if (dst_loc == MemoryLocation::GPU) {
      const auto gpu_ptrs = replica->get_data_pointer(MemoryLocation::GPU);
      handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

      auto ipc_or = replica->get_memory_manager().get_ipc_handle();
      if (ipc_or.ok()) {
        std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
      }
    }
    const auto& replica_view_plan = replica->view_plan();
    if (replica_view_plan.has_value() && !replica_view_plan->is_identity) {
      handle.view_index_json = replica_view_plan->view_index_json;
      const uint64_t view_size = replica_view_plan->view_size_bytes;
      if (view_size > 0) {
        const bool target_is_gpu = (dst_loc == MemoryLocation::GPU);
        const bool target_loaded = target_is_gpu ? handle.gpu_state == replica::MemoryState::LOADED
                                                 : handle.cpu_state == replica::MemoryState::LOADED;
        std::optional<int> gpu_device = target_is_gpu ? std::optional<int>(target_device.ordinal) : std::nullopt;
        if (target_loaded) {
          auto hash = ComputeViewDataHash(*replica, dst_loc, view_size, gpu_device);
          if (hash.has_value()) {
            handle.view_data_hash = std::move(hash);
          }
        }
      }
    }
    return handle;
  }

  // Helper lambda: minimal disk-loading path.
  auto load_from_disk = [&](const DeviceKey& dev_key) -> absl::StatusOr<loading::ReplicaHandle> {
    // Guard: content-addressed IDs (mi2:...) are not paths.
    if (absl::StartsWith(canonical_artifact_id, "mi2:")) {
      return absl::FailedPreconditionError(
          "LOAD_ONLY/disk fallback disabled for content-addressed artifact_id; Global Store routing required");
    }
    loading::DiskSource disk_src;
    if (!hints.disk_path.empty()) {
      disk_src.path = std::filesystem::path(hints.disk_path);
    } else {
      disk_src.path = std::filesystem::path(canonical_artifact_id);
    }

    loading::ReplicaTarget target;
    target.location.type =
        (dev_key.type == DeviceType::GPU) ? common::memory::MemoryLocation::GPU : common::memory::MemoryLocation::CPU;
    target.location.device_id = dev_key.ordinal;

    return ingest_from_disk_internal(canonical_artifact_id, disk_src, target, hints);
  };

  // ────────────────────────────────────────────────────────────────────
  // Mode-specific handling
  // ────────────────────────────────────────────────────────────────────
  switch (mode) {
    case MaterializeMode::COPY_ONLY: {
      // COPY_ONLY is only meaningful for GPU targets – perform a local GPU→GPU copy.
      if (target_device.type != DeviceType::GPU) {
        return absl::InvalidArgumentError("COPY_ONLY mode requires a GPU target device");
      }

      // Require an explicit artifact identifier so we can locate a source instance.
      // This avoids implicit coupling to disk_path or other hints and keeps COPY_ONLY semantics clear.
      if (canonical_artifact_id.empty()) {
        return absl::InvalidArgumentError("COPY_ONLY requires a canonical artifact identifier");
      }

      const auto candidates = replica_registry_->find_by_artifact(canonical_artifact_id);
      for (const auto& cand_key : candidates) {
        if (cand_key.device.type != DeviceType::GPU) {
          continue;
        }
        if (cand_key.view_id != requested_view_id) {
          continue;
        }

        auto src_or = replica_registry_->find(cand_key);
        if (!src_or.ok()) {
          continue;
        }
        const auto& src_replica = src_or.value();
        if (src_replica->get_memory_state(common::memory::MemoryLocation::GPU) != replica::MemoryState::LOADED) {
          continue;
        }

        // Create destination replica configuration using an inline buffer source to
        // avoid any dependency on on-disk paths when performing GPU→GPU copy.
        // The inline buffer loader requires a known total size.
        uint64_t expected_size = 0;
        if (auto sz_or = src_replica->get_artifact_size(); sz_or.ok()) {
          expected_size = *sz_or;
        } else {
          return sz_or.status();
        }
        loading::InlineBufferSource ib_source{.data = nullptr, .size_bytes = expected_size};
        replica::ReplicaConfig cfg{
            .source = ib_source,
            .artifact_identifier = canonical_artifact_id,
            .device_type = DeviceType::GPU,
            .local_device_id = target_device.ordinal,
            .pinned_buffer_pool = memory_pool_,
            .virtual_addr_space = va_space_,
            .expected_artifact_size = expected_size};
        cfg.pinned_memory_timeout = pinned_memory_timeout_;
        cfg.view_id = requested_view_id;
        cfg.view_plan = src_replica->view_plan();
        cfg.transform_placement = hints.variant ? hints.variant->placement : loading::TransformPlacement::kServer;

        auto dst_or = replica::Replica::create(cfg);
        if (!dst_or.ok()) {
          return dst_or.status();
        }
        auto dst_replica = std::shared_ptr<replica::Replica>(std::move(dst_or.value()));
        {
          absl::Status emplace_status =
              replica_registry_->emplace(dst_key, gsl::not_null<std::shared_ptr<replica::Replica>>{dst_replica});
          if (absl::IsAlreadyExists(emplace_status)) {
            VLOG(1) << "Replica already present for COPY_ONLY dst_key (will reuse existing instance): " << dst_key;
          } else if (!emplace_status.ok()) {
            return emplace_status;
          }
        }

        absl::Status copy_st = dst_replica->copy_from(*src_replica);

        std::promise<absl::Status> p;
        p.set_value(copy_st);

        ReplicaHandle handle;
        handle.replica_key = dst_key;
        handle.ready_future = p.get_future().share();
        handle.cpu_state = dst_replica->get_memory_state(MemoryLocation::CPU);
        handle.gpu_state = dst_replica->get_memory_state(MemoryLocation::GPU);
        if (handle.gpu_state == MemoryState::LOADED) {
          const auto gpu_ptrs = dst_replica->get_data_pointer(MemoryLocation::GPU);
          handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

          auto ipc_or = dst_replica->get_memory_manager().get_ipc_handle();
          if (ipc_or.ok()) {
            std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
          }
        }
        const auto& dst_view_plan = dst_replica->view_plan();
        if (dst_view_plan.has_value() && !dst_view_plan->is_identity) {
          handle.view_index_json = dst_view_plan->view_index_json;
          auto hash = ComputeViewDataHash(
              *dst_replica,
              MemoryLocation::GPU,
              dst_view_plan->view_size_bytes,
              std::optional<int>(target_device.ordinal));
          if (hash.has_value()) {
            handle.view_data_hash = std::move(hash);
          }
        }
        return handle;
      }
      return absl::FailedPreconditionError(absl::StrCat(
          "No suitable source instance for COPY_ONLY mode (artifact_id=",
          canonical_artifact_id,
          ", view_id=",
          requested_view_id.has_value() ? *requested_view_id : std::string("<none>"),
          ")"));
    }

    case MaterializeMode::LOAD_ONLY: {
      auto handle_or = load_from_disk(target_device);
      if (!handle_or.ok()) {
        VLOG(1) << "StoreEngine::materialize_replica LOAD_ONLY artifact=" << canonical_artifact_id
                << " device=" << target_device.ordinal << " status=" << handle_or.status();
      }
      return handle_or;
    }

    case MaterializeMode::AUTO: {
      if (global_store_client_ && global_store_client_->is_connected() && !canonical_artifact_id.empty()) {
        loading::MaterializeOrchestrator orchestrator(
            gsl::not_null<StoreEngine*>{this},
            gsl::not_null<components::IGlobalStoreClient*>{global_store_client_.get()});
        auto orchestrated_or = orchestrator.run(canonical_artifact_id, target_device, hints);
        if (orchestrated_or.ok()) {
          return *orchestrated_or;
        }
        LOG(WARNING) << "MaterializeOrchestrator failed: " << orchestrated_or.status() << "; falling back to disk load";
      }
      return absl::FailedPreconditionError(
          "AUTO materialize_replica requires a canonical artifact identifier (mi2:...) with Global Store routing or an explicit hints.disk_path");
    }
  }

  // Should be unreachable.
  return absl::InternalError("Invalid MaterializeMode");
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------
std::vector<DeviceKey> StoreEngine::get_resident_devices(std::string_view artifact_id) const {
  // Implementation: leverage the modern multi-device registry exclusively. Replicas loaded via
  // ReplicaRegistry::emplace are visible to this helper; no backward-compatibility fallbacks remain.
  absl::flat_hash_set<DeviceKey, DeviceKeyHash> unique_devices;
  std::vector<DeviceKey> devices;

  // ──────────────────────────────────────────────────────────────────
  // 1. Multi-device path – gather all instances whose artifact_id matches
  // ──────────────────────────────────────────────────────────────────
  const auto replica_keys = replica_registry_->find_by_artifact(artifact_id);
  for (const auto& key : replica_keys) {
    auto replica_or = replica_registry_->find(key);
    if (!replica_or.ok()) {
      continue;
    }
    const auto& replica = replica_or.value();

    auto is_present = [](MemoryState st) {
      return st == MemoryState::ALLOCATED || st == MemoryState::LOADING || st == MemoryState::LOADED;
    };

    if (key.device.type == DeviceType::CPU) {
      if (is_present(replica->get_memory_state(MemoryLocation::CPU))) {
        unique_devices.insert(DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""});
      }
    } else if (key.device.type == DeviceType::GPU) {
      if (is_present(replica->get_memory_state(MemoryLocation::GPU))) {
        unique_devices.insert(
            DeviceKey{.type = DeviceType::GPU, .ordinal = key.device.ordinal, .uuid = key.device.uuid});
      }
    }
  }

  // Legacy fallback removed – we no longer support the deprecated single-map
  // registry.  All look-ups are served exclusively through the multi-device
  // registry interfaces above.

  // Convert set → vector for return value.
  devices.assign(unique_devices.begin(), unique_devices.end());
  return devices;
}

std::vector<ReplicaKey> StoreEngine::list_device_replicas(const DeviceKey& device) const {
  // Implementation that relies solely on the new multi-index registry (ReplicaKey-based). No
  // legacy fallback remains.
  std::vector<ReplicaKey> list;

  // ------------------------------------------------------------------
  // 1. Multi-device registry path
  // ------------------------------------------------------------------
  const auto inst_keys = replica_registry_->find_by_device(device);
  for (const auto& key : inst_keys) {
    auto replica_or = replica_registry_->find(key);
    if (!replica_or.ok()) {
      continue;
    }
    const auto& replica = replica_or.value();

    auto is_present = [](MemoryState st) {
      return st == MemoryState::ALLOCATED || st == MemoryState::LOADING || st == MemoryState::LOADED;
    };

    if (device.type == DeviceType::CPU) {
      if (is_present(replica->get_memory_state(MemoryLocation::CPU))) {
        list.push_back(key);
      }
    } else if (device.type == DeviceType::GPU) {
      if (is_present(replica->get_memory_state(MemoryLocation::GPU))) {
        list.push_back(key);
      }
    }
  }

  // Legacy fallback path removed – only multi-device registry results are
  // returned.  If no instances match, the list will be empty.

  return list;
}

// ---------------------------------------------------------------------------
// Multi-Device Binding – GPU-aware memory eviction (NEW in Phase 3.2)
// ---------------------------------------------------------------------------

// ═══════════════════════════════════════════════════════════════════════════
// New ReplicaKey-centric API wrappers
// ═══════════════════════════════════════════════════════════════════════════

int StoreEngine::wait_replica_ready(const ReplicaKey& key) {
  auto replica_or = replica_registry_->find(key);
  if (!replica_or.ok()) {
    return 1; // Not found
  }
  const auto& replica = replica_or.value();
  MemoryLocation loc = (key.device.type == DeviceType::CPU) ? MemoryLocation::CPU : MemoryLocation::GPU;
  absl::Status st = replica->wait_until_loaded(loc, absl::InfiniteDuration());
  return st.ok() ? 0 : 1;
}

int StoreEngine::unload_replica(const ReplicaKey& key) {
  auto replica_or = replica_registry_->find(key);
  if (!replica_or.ok()) {
    VLOG(1) << "StoreEngine::unload_replica artifact=" << key.artifact_id << " device=" << key.device.ordinal
            << " not found in registry: " << replica_or.status();
    return 1; // Instance not found.
  }

  const auto& replica = replica_or.value();
  MemoryLocation loc = (key.device.type == DeviceType::CPU) ? MemoryLocation::CPU : MemoryLocation::GPU;

  // Inspect current state *before* attempting the release so we can tell if
  // there was anything to unload.  This avoids treating a no-op release as a
  // success – a scenario that would allow multiple threads to report success
  // when only the first one actually freed memory.
  store::MemoryState before_state = replica->get_memory_state(loc);

  if (before_state <= MemoryState::UNALLOCATED) {
    // The materialize call may have just scheduled the load and not yet
    // transitioned the replica into ALLOCATED/LOADING. Give it a brief window
    // to advance before concluding that there is truly nothing to unload.
    constexpr absl::Duration kLoadProgressProbe = absl::Milliseconds(250);
    constexpr absl::Duration kProbeInterval = absl::Milliseconds(5);
    const absl::Time probe_deadline = absl::Now() + kLoadProgressProbe;

    MemoryState observed_state = before_state;
    while (observed_state <= MemoryState::UNALLOCATED && absl::Now() < probe_deadline) {
      absl::SleepFor(kProbeInterval);
      observed_state = replica->get_memory_state(loc);
    }

    if (observed_state <= MemoryState::UNALLOCATED) {
      // Still nothing allocated and no load in-flight; treat as a no-op unload.
      VLOG(1) << "StoreEngine::unload_replica artifact=" << key.artifact_id << " device=" << key.device.ordinal
              << ": no allocation observed after probe window; treating as no-op unload.";
      return -1;
    }

    before_state = observed_state;
  }

  absl::Status release_status = replica->release_memory(loc);

  if (absl::IsFailedPrecondition(release_status)) {
    // A release requested while a load is still in-flight. Wait for the load
    // to settle (either succeed or fail) and retry once before giving up so
    // callers see the expected best-effort unload semantics.
    constexpr absl::Duration kUnloadRetryTimeout = absl::Seconds(30);
    absl::Status wait_status = replica->wait_until_loaded(loc, kUnloadRetryTimeout);

    if (!wait_status.ok() && !absl::IsFailedPrecondition(wait_status)) {
      // Deadline exceeded or another unexpected error – surface the failure so
      // higher layers can decide whether to retry.
      VLOG(1) << "StoreEngine::unload_replica artifact=" << key.artifact_id << " device=" << key.device.ordinal
              << ": wait for load completion returned " << wait_status;
      return -1;
    }

    release_status = replica->release_memory(loc);
  }

  if (!release_status.ok()) {
    VLOG(1) << "StoreEngine::unload_replica artifact=" << key.artifact_id << " device=" << key.device.ordinal
            << ": unload failed with " << release_status;
  }

  return release_status.ok() ? 0 : -1;
}

MemoryState StoreEngine::get_replica_state(const ReplicaKey& key, DeviceType memory_type) const {
  auto replica_or = replica_registry_->find(key);
  if (!replica_or.ok()) {
    return MemoryState::UNINITIALIZED;
  }
  MemoryLocation loc = (memory_type == DeviceType::CPU) ? MemoryLocation::CPU : MemoryLocation::GPU;
  return replica_or.value()->get_memory_state(loc);
}

absl::StatusOr<uint64_t> StoreEngine::get_replica_gpu_ptr(const ReplicaKey& key) {
  if (key.device.type != DeviceType::GPU) {
    return absl::InvalidArgumentError("ReplicaKey does not reference a GPU device");
  }
  auto replica_or = replica_registry_->find(key);
  if (!replica_or.ok()) {
    return absl::NotFoundError("Replica not found");
  }
  const auto ptrs = replica_or.value()->get_memory_manager().get_pointer(MemoryLocation::GPU);
  if (ptrs.empty() || ptrs[0] == nullptr) {
    return absl::FailedPreconditionError("GPU memory not available");
  }
  return reinterpret_cast<uint64_t>(ptrs[0]);
}

absl::StatusOr<uint64_t> StoreEngine::get_replica_size(const ReplicaKey& key) {
  auto replica_or = replica_registry_->find(key);
  if (!replica_or.ok()) {
    return absl::NotFoundError("Replica not found");
  }
  auto size_or = replica_or.value()->get_artifact_size();
  if (!size_or.ok()) {
    return size_or.status();
  }
  return *size_or;
}

absl::StatusOr<ExportRegistration> StoreEngine::enable_remote_replica_access(
    const ReplicaKey& key,
    MemoryLocation location) {
  if (!comm_manager_->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }
  auto replica_or = replica_registry_->find(key);
  if (!replica_or.ok()) {
    return absl::NotFoundError("Replica not found");
  }
  return replica_or.value()->enable_remote_memory_access(location, comm_manager_->get_engine());
}

absl::Status StoreEngine::disable_remote_replica_access(const ReplicaKey& key, MemoryLocation location) {
  if (!comm_manager_->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }
  auto replica_or = replica_registry_->find(key);
  if (!replica_or.ok()) {
    return absl::NotFoundError("Replica not found");
  }
  return replica_or.value()->disable_remote_memory_access(location, comm_manager_->get_engine());
}

// ═══════════════════════════════════════════════════════════════════════════
// Memory cleanup
// ═══════════════════════════════════════════════════════════════════════════

int StoreEngine::clear_mem() {
  auto replicas = replica_registry_->clear_all();
  std::vector<absl::Status> errors;

  for (const auto& [inst_key, replica] : replicas) {
    // Release CPU memory with proper error tracking
    auto cpu_status = replica->release_memory(MemoryLocation::CPU);
    if (!cpu_status.ok()) {
      LOG(WARNING) << "Failed to release CPU memory for " << inst_key << ": " << cpu_status.message();
      errors.push_back(cpu_status);
    }

    // Release GPU memory, ignoring NotFound errors (expected when no GPU memory allocated)
    auto gpu_status = replica->release_memory(MemoryLocation::GPU);
    if (!gpu_status.ok() && !absl::IsNotFound(gpu_status)) {
      LOG(WARNING) << "Failed to release GPU memory for " << inst_key << ": " << gpu_status.message();
      errors.push_back(gpu_status);
    }
  }

  // Update metrics even if some releases failed
  metrics_collector_->update_all_metrics(*memory_pool_, *replica_registry_, *device_manager_);

  // Log aggregated error summary if failures occurred
  if (!errors.empty()) {
    LOG(ERROR) << "Failed to release memory for " << errors.size() << " replica(s) during shutdown";
    return -1;
  }

  return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// Global Store registration helper for already-loaded replicas
// ═══════════════════════════════════════════════════════════════════════════

absl::Status StoreEngine::register_replica_with_global_store(
    const ReplicaKey& key,
    std::string_view artifact_id_override) {
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (key.view_id.has_value()) {
    VLOG(1) << "register_replica_with_global_store variant view_id=" << *key.view_id
            << " (canonical_artifact_id=" << key.artifact_id << ")";
  }
  // Determine memory location based on device type
  MemoryLocation loc = (key.device.type == DeviceType::GPU) ? MemoryLocation::GPU : MemoryLocation::CPU;

  // Fetch total size for registration
  uint64_t size = 0;
  if (auto sz_or = get_replica_size(key); sz_or.ok()) {
    size = *sz_or;
  } else {
    return sz_or.status();
  }

  const std::string artifact_id = artifact_id_override.empty() ? key.artifact_id : std::string(artifact_id_override);
  const std::string wid = worker_id_.empty() ? std::string("local") : worker_id_;
  absl::Status register_status = ReplicaRegistrationHelper::register_local_replica(
      gsl::not_null<components::IGlobalStoreClient*>{global_store_client_.get()},
      wid,
      artifact_id,
      key.device,
      loc,
      size);
  if (!register_status.ok()) {
    return register_status;
  }

  if (key.view_id.has_value()) {
    auto variant_status = global_store_client_->record_variant_residency(key.artifact_id, *key.view_id, size);
    if (!variant_status.ok()) {
      if (absl::IsUnimplemented(variant_status)) {
        VLOG(1) << "Global Store does not yet accept variant residency updates: " << variant_status.message();
      } else {
        LOG(WARNING) << "record_variant_residency failed for view_id=" << *key.view_id << ": " << variant_status;
      }
    }
  }
  return register_status;
}

// ═══════════════════════════════════════════════════════════════════════════
// RFC-0014: Key-mapping wrappers
// ═══════════════════════════════════════════════════════════════════════════

absl::StatusOr<components::KeyMapping> StoreEngine::resolve_key_mapping(std::string_view key) {
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  return global_store_client_->resolve_key_mapping(key);
}

absl::StatusOr<std::string> StoreEngine::get_canonical_index_by_id(std::string_view artifact_id) {
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  return global_store_client_->get_artifact_index_by_id(artifact_id);
}

absl::Status StoreEngine::upsert_key_mapping(
    std::string_view key,
    std::string_view artifact_id,
    std::string_view disk_path,
    absl::Duration ttl) {
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  return global_store_client_->upsert_key_mapping(key, artifact_id, disk_path, ttl);
}

absl::Status StoreEngine::revoke_key_mapping(std::string_view key) {
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  return global_store_client_->revoke_key_mapping(key);
}

// ═══════════════════════════════════════════════════════════════════════════
// RFC-0006 – Memory Artifact Registration (coalesced)
// ═══════════════════════════════════════════════════════════════════════════

absl::StatusOr<StoreEngine::RegistrationBeginResult> StoreEngine::begin_register_artifact(
    const ArtifactRegistration& reg) {
  if (reg.total_size_bytes == 0) {
    return absl::InvalidArgumentError("total_size_bytes must be > 0");
  }
  if (reg.device_id < 0) {
    return absl::InvalidArgumentError("device_id must be >= 0");
  }
  if (!reg.schema_version.empty() && reg.schema_version != "v3") {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported ArtifactRegistration.schema_version='", reg.schema_version, "'; expected 'v3'"));
  }
  // Accept either a pre-existing index key or inline index data.  Only error
  // when both are absent.
  if (reg.tensor_index_key.empty() && !reg.tensor_index_data.has_value()) {
    return absl::InvalidArgumentError("tensor index key or data must be provided");
  }
  const uint64_t canonical_size = (reg.view.has_value() && reg.view->canonical_size_bytes != 0)
      ? reg.view->canonical_size_bytes
      : reg.total_size_bytes;

  std::optional<loader::BidirectionalViewPlan> view_plan;
  uint64_t expected_view_bytes = 0;
  std::vector<CanonicalRange> canonical_ranges;
  if (reg.view.has_value()) {
    const auto& view_opts = *reg.view;
    if (!reg.tensor_index_data.has_value() || reg.tensor_index_data->empty()) {
      return absl::InvalidArgumentError("view registration requires inline canonical index data");
    }
    if (view_opts.placement == ViewPlacement::kUnspecified) {
      return absl::InvalidArgumentError("view registration requires explicit placement");
    }
    if (view_opts.canonical_size_bytes != 0 && view_opts.canonical_size_bytes != reg.total_size_bytes) {
      return absl::InvalidArgumentError("view.canonical_size_bytes must match total_size_bytes");
    }
    auto plan_or = loader::ViewPlanner::compute_bidirectional_view_plan(*reg.tensor_index_data, view_opts.spec);
    if (!plan_or.ok()) {
      return plan_or.status();
    }
    view_plan = std::move(*plan_or);
    expected_view_bytes = sum_view_write_bytes(view_plan->write);
    canonical_ranges = canonical_ranges_from_write_plan(view_plan->write);
    uint64_t covered_bytes = 0;
    for (const auto& range : canonical_ranges) {
      covered_bytes += range.length;
    }
    if (!view_opts.allow_partial && covered_bytes != canonical_size) {
      return absl::InvalidArgumentError(
          "view registration does not fully cover canonical bytes; set allow_partial=true to permit partial coverage");
    }
    if (covered_bytes > canonical_size) {
      return absl::InvalidArgumentError("view registration exceeds canonical byte space");
    }
    if (view_opts.placement == ViewPlacement::kServer && view_plan->inverse_transform.requires_materialization) {
      auto info_or = device_manager_->get_gpu_info(reg.device_id);
      if (!info_or.ok()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "SERVER placement for view registration requires GPU transpose support on device ",
            reg.device_id,
            "; retry with placement=CLIENT (",
            info_or.status().message(),
            ")"));
      }
    }
  }

  // Prepare a memory-only Replica bound to target GPU to own the allocation.
  // Use InlineBufferSource with the known total size so Replica::create() can
  // construct ReplicaLoadController without requiring any on-disk layout.
  InlineBufferSource ib_source{.data = nullptr, .size_bytes = reg.total_size_bytes};
  replica::ReplicaConfig cfg{
      .source = ib_source,
      .artifact_identifier = reg.artifact_id,
      .device_type = DeviceType::GPU,
      .local_device_id = reg.device_id,
      .pinned_buffer_pool = memory_pool_,
      .virtual_addr_space = va_space_,
      .expected_artifact_size = reg.total_size_bytes};
  cfg.pinned_memory_timeout = pinned_memory_timeout_;

  // Check memory pressure before allocation to avoid unexpected evictions
  // This helps prevent large registrations from causing issues with existing replicas
  {
    auto free_or = device_manager_->get_free_memory(reg.device_id);
    if (free_or.ok()) {
      size_t free_bytes = free_or.value();
      if (reg.total_size_bytes > free_bytes) {
        // Try to evict unused GPU memory first on the specific device
        auto evict_status = try_evict_gpu_memory_impl(
            *replica_registry_,
            *device_manager_,
            *metrics_collector_,
            reg.device_id,
            reg.total_size_bytes - free_bytes);
        if (!evict_status.ok()) {
          return absl::ResourceExhaustedError(absl::StrCat(
              "Insufficient GPU memory available. Requested: ",
              reg.total_size_bytes,
              " bytes, Free: ",
              free_bytes,
              ". ",
              evict_status.message()));
        }
      }
    }
  }

  // Try to create the replica with DiskSource first. If the loader fails due to
  // missing directory or missing partitions, fall back to a memory-only path
  // that uses InlineBufferSource to construct a size-known Artifact and allocate
  // GPU memory without touching disk.
  auto create_or = Replica::create(cfg);
  if (!create_or.ok()) {
    return create_or.status();
  }
  auto replica = std::shared_ptr<Replica>(std::move(create_or.value()));

  // Allocate GPU memory only. No loading/copying here.
  absl::Status st = replica->get_memory_manager().allocate_memory(MemoryLocation::GPU);
  if (!st.ok()) {
    return st;
  }

  // Obtain CUDA IPC handle to return to caller.
  auto ipc_or = replica->get_memory_manager().get_ipc_handle();
  if (!ipc_or.ok()) {
    return ipc_or.status();
  }

  const auto gpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::GPU);
  void* base_ptr = (!gpu_ptrs.empty() ? gpu_ptrs[0] : nullptr);

  // Emplace into registry to ensure lifecycle is tracked (ReplicaKey via config).
  DeviceKey dev_key{.type = DeviceType::GPU, .ordinal = reg.device_id, .uuid = ""};
  ReplicaKey inst_key{.artifact_id = reg.artifact_id, .device = dev_key, /*replica=*/.replica = 0};
  {
    absl::Status emplace_status =
        replica_registry_->emplace(inst_key, gsl::not_null<std::shared_ptr<replica::Replica>>{replica});
    if (absl::IsAlreadyExists(emplace_status)) {
      VLOG(1) << "Pending registry already had instance for key=" << inst_key.artifact_id;
    } else if (!emplace_status.ok()) {
      return emplace_status;
    }
  }

  // Create pending entry with cryptographically secure random ID
  // Use a combination of timestamp, process ID, and random bytes for uniqueness
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;
  auto reg_id = absl::StrCat("reg_", absl::ToUnixNanos(absl::Now()), "_", getpid(), "_", dis(gen));

  auto entry = std::make_shared<PendingRegistrationEntry>();
  entry->registration_id = reg_id;
  entry->artifact_id = reg.artifact_id;
  entry->device_id = reg.device_id;
  entry->size_bytes = reg.total_size_bytes;
  entry->tensor_index_key = reg.tensor_index_key;
  entry->tensor_index_data = reg.tensor_index_data;
  entry->schema_version = reg.schema_version;
  entry->encoding = reg.encoding;
  entry->enable_p2p = reg.enable_p2p;
  if (reg.ttl_ms > 0) {
    entry->expiry_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(reg.ttl_ms);
  }
  entry->replica = replica;
  entry->gpu_ptr = base_ptr;
  entry->ipc_handle = *ipc_or;
  entry->plan = PendingRegistrationEntry::Plan::COALESCED;
  if (view_plan.has_value()) {
    entry->view_state = std::make_unique<PendingRegistrationEntry::ViewState>();
    entry->view_state->options = *reg.view;
    entry->view_state->options.canonical_size_bytes = canonical_size;
    entry->view_state->options.canonical_ranges = canonical_ranges;
    entry->view_state->plan = *view_plan;
    entry->view_state->expected_view_bytes = expected_view_bytes;
    entry->view_state->ingested_bytes = 0;
    if (entry->view_state->options.placement == ViewPlacement::kServer) {
      entry->view_state->executor = std::make_unique<loader::ViewIngestExecutor>(
          entry->view_state->plan.write, entry->view_state->plan.inverse_transform);
    }
  }

  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_regs_.emplace(reg_id, entry);
  }

  RegistrationBeginResult out;
  out.registration_id = reg_id;
  out.device_id = reg.device_id;
  out.size_bytes = reg.total_size_bytes;
  std::memcpy(out.cuda_ipc_handle_bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
  return out;
}

// CPU registration path removed

absl::StatusOr<StoreEngine::RegistrationCommitResult> StoreEngine::commit_registered_artifact(
    std::string_view registration_id) {
  std::shared_ptr<PendingRegistrationEntry> entry;
  std::shared_ptr<Replica> expired_replica;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_regs_.find(std::string(registration_id));
    if (it == pending_regs_.end()) {
      return absl::NotFoundError("registration_id not found");
    }
    if (it->second->expiry_time.time_since_epoch().count() > 0 &&
        std::chrono::steady_clock::now() > it->second->expiry_time) {
      expired_replica = it->second->replica;
      pending_regs_.erase(it);
    } else {
      entry = it->second;
    }
  }

  // Release memory outside the critical section if TTL expired
  if (expired_replica) {
    {
      absl::Status _st = expired_replica->release_memory(MemoryLocation::GPU);
      if (!_st.ok()) {
        VLOG(1) << "release_memory(GPU) failed during TTL cleanup: " << _st;
      }
    }
    return absl::DeadlineExceededError("registration expired (TTL)");
  }

  if (!entry) {
    return absl::InternalError("pending registration entry missing after TTL check");
  }

  if (entry->view_state && entry->view_state->options.placement == ViewPlacement::kServer) {
    auto& view_state = *entry->view_state;
    if (!view_state.executor) {
      return absl::FailedPreconditionError("view executor missing for server placement registration");
    }
    if (!view_state.executor->is_complete()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "view bytes incomplete; expected ",
          view_state.expected_view_bytes,
          " received ",
          view_state.executor->ingested_bytes()));
    }
    auto finalize_status = view_state.executor->finalize(MemoryLocation::GPU, entry->gpu_ptr, entry->device_id);
    if (!finalize_status.ok()) {
      return finalize_status;
    }
    view_state.finalized = true;
  }

  // Zero-fill any canonical regions not covered by the view when allow_partial=true.
  // This ensures deterministic hashing and read semantics: uncovered bytes are canonical zeros.
  if (entry->view_state && entry->view_state->options.allow_partial) {
    const auto& ranges = entry->view_state->options.canonical_ranges;
    uint64_t covered_bytes = 0;
    for (const auto& r : ranges) {
      covered_bytes += r.length;
    }
    if (covered_bytes < entry->size_bytes) {
      void* gpu_ptr = entry->gpu_ptr;
      if (gpu_ptr == nullptr) {
        const auto ptrs = entry->replica->get_memory_manager().get_pointer(MemoryLocation::GPU);
        gpu_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
      }
      if (!gpu_ptr) {
        return absl::FailedPreconditionError("GPU pointer is null; cannot zero uncovered regions");
      }

      auto dev_status = cuda::set_device(entry->device_id);
      if (!dev_status.ok()) {
        return dev_status;
      }

      uint64_t cursor = 0;
      for (const auto& r : ranges) {
        if (r.offset > cursor) {
          auto ms = cuda::memset(static_cast<uint8_t*>(gpu_ptr) + cursor, 0, static_cast<size_t>(r.offset - cursor));
          if (!ms.ok()) {
            return ms;
          }
        }
        const uint64_t r_end = r.offset + r.length;
        cursor = std::max(r_end, cursor);
      }
      if (cursor < entry->size_bytes) {
        auto ms =
            cuda::memset(static_cast<uint8_t*>(gpu_ptr) + cursor, 0, static_cast<size_t>(entry->size_bytes - cursor));
        if (!ms.ok()) {
          return ms;
        }
      }
      auto sync = cuda::device_synchronize();
      if (!sync.ok()) {
        return sync;
      }
    }
  }

  // Compute content-addressed artifact_id per RFC-0007: "mi2:<index_multihash>:<data_multihash>"
  // 1) index_multihash from canonical index bytes when provided, otherwise from key (sha256 hex)
  absl::StatusOr<std::string> index_mh_or =
      common::compute_index_multihash(entry->tensor_index_data, entry->tensor_index_key);
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }

  if (!entry->schema_version.empty() && entry->schema_version != "v3") {
    return absl::FailedPreconditionError(
        absl::StrCat("Pending registration schema_version must be 'v3'; found '", entry->schema_version, "'"));
  }

  std::vector<loader::SegmentPiece> segment_plan;
  bool segment_plan_ready = false;

  // 2) data_multihash via SegmentPlan linearization (PAD=0).
  absl::StatusOr<std::string> data_mh_or;
  if (entry->plan == PendingRegistrationEntry::Plan::CPU) {
    // Hash directly from VS CPU memory (zero-initialized PAD regions).
    auto region_or = va_space_->region_info(entry->artifact_id);
    if (!region_or.ok()) {
      return region_or.status();
    }
    auto mh_or = loader::compute_data_multihash_from_cpu_memory(
        gsl::not_null<const void*>{region_or->cpu_base}, entry->size_bytes);
    if (!mh_or.ok()) {
      return mh_or.status();
    }
    data_mh_or = *mh_or;
  } else {
    void* gpu_ptr = entry->gpu_ptr;
    if (gpu_ptr == nullptr) {
      const auto ptrs = entry->replica->get_memory_manager().get_pointer(MemoryLocation::GPU);
      gpu_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
    }
    if (!gpu_ptr) {
      return absl::FailedPreconditionError("GPU pointer is null; cannot hash GPU data");
    }
    if (entry->tensor_index_data.has_value() && !entry->tensor_index_data->empty() && entry->encoding == "json") {
      auto plan_or = loader::build_segment_plan_from_canonical_index_json(
          *entry->tensor_index_data, entry->size_bytes, /*align_bytes=*/8);
      if (!plan_or.ok()) {
        return plan_or.status();
      }
      segment_plan = std::move(*plan_or);
      segment_plan_ready = true;
      auto mh_or = loader::compute_data_multihash_from_gpu_plan(
          gsl::not_null<void*>{gpu_ptr}, entry->device_id, absl::MakeSpan(segment_plan), entry->size_bytes);
      if (!mh_or.ok()) {
        return mh_or.status();
      }
      data_mh_or = *mh_or;
    } else {
      auto mh_or = common::compute_data_multihash_from_gpu(gpu_ptr, entry->size_bytes, entry->device_id);
      if (!mh_or.ok()) {
        return mh_or.status();
      }
      data_mh_or = *mh_or;
    }
  }
  if (!data_mh_or.ok()) {
    return data_mh_or.status();
  }

  entry->artifact_id = absl::StrCat("mi2:", *index_mh_or, ":", *data_mh_or);

  // Idempotent success: if the same content-addressed artifact already has a
  // replica on the target device, reclaim this allocation and return OK with
  // existed=true and the computed descriptor. No new registry mapping is
  // created and no Global Store upsert is attempted.
  {
    DeviceKey dev_key{
        .type = (entry->plan == PendingRegistrationEntry::Plan::CPU ? DeviceType::CPU : DeviceType::GPU),
        .ordinal = (entry->plan == PendingRegistrationEntry::Plan::CPU ? -1 : entry->device_id),
        .uuid = ""};
    loading::ReplicaKey check_key{.artifact_id = entry->artifact_id, .device = dev_key, .replica = 0};
    auto existing_or = replica_registry_->find(check_key);
    if (existing_or.ok()) {
      {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_regs_.erase(std::string(registration_id));
      }
      if (entry->plan == PendingRegistrationEntry::Plan::CPU) {
        absl::Status _st = entry->replica->release_memory(MemoryLocation::CPU);
        if (!_st.ok()) {
          VLOG(1) << "release_memory(CPU) failed during idempotent success cleanup: " << _st;
        }
      } else {
        absl::Status _st = entry->replica->release_memory(MemoryLocation::GPU);
        if (!_st.ok()) {
          VLOG(1) << "release_memory(GPU) failed during idempotent success cleanup: " << _st;
        }
      }
      RegistrationCommitResult result;
      result.registration_id = std::string(registration_id);
      result.artifact_id = entry->artifact_id;
      result.device_id = entry->device_id;
      result.size_bytes = entry->size_bytes;
      result.existed = true;
      result.index_multihash = *index_mh_or;
      result.data_multihash = *data_mh_or;
      result.schema_version = entry->schema_version;
      result.encoding = entry->encoding;
      if (entry->view_state) {
        if (!entry->view_state->options.view_id.empty()) {
          result.view_id = entry->view_state->options.view_id;
        }
        result.canonical_ranges = entry->view_state->options.canonical_ranges;
        result.allow_partial = entry->view_state->options.allow_partial;
        if (!entry->view_state->plan.forward.view_index_json.empty()) {
          result.view_index_json = entry->view_state->plan.forward.view_index_json;
        }
      }
      return result;
    }
  }

  // Also add a registry mapping for the content-addressed artifact_id so callers can
  // subsequently reference the instance by its mi2: identifier (in addition to the
  // original logical artifact_id used during Begin/Commit).
  {
    DeviceKey dev_key{
        .type = (entry->plan == PendingRegistrationEntry::Plan::CPU ? DeviceType::CPU : DeviceType::GPU),
        .ordinal = (entry->plan == PendingRegistrationEntry::Plan::CPU ? -1 : entry->device_id),
        .uuid = ""};
    ReplicaKey mi2_key{.artifact_id = entry->artifact_id, .device = dev_key, .replica = 0};
    absl::Status emplace_status =
        replica_registry_->emplace(mi2_key, gsl::not_null<std::shared_ptr<replica::Replica>>{entry->replica});
    if (absl::IsAlreadyExists(emplace_status)) {
      VLOG(1) << "mi2 mapping already present for artifact_id=" << entry->artifact_id;
    } else if (!emplace_status.ok()) {
      return emplace_status;
    }
  }

  // Export remote memory keys if communication is enabled (GPU location).
  std::vector<std::string> remote_keys;
  std::vector<uint64_t> buffer_sizes;
  if (entry->plan != PendingRegistrationEntry::Plan::CPU && entry->enable_p2p && comm_manager_->is_enabled()) {
    auto reg_info_or = entry->replica->enable_remote_memory_access(MemoryLocation::GPU, comm_manager_->get_engine());
    if (!reg_info_or.ok()) {
      return reg_info_or.status();
    }
    remote_keys = reg_info_or->remote_memory_keys;
    buffer_sizes.reserve(reg_info_or->buffer_sizes.size());
    for (const auto& sz : reg_info_or->buffer_sizes) {
      buffer_sizes.push_back(static_cast<uint64_t>(sz));
    }
  }

  // Register with Global Store (memory replica). Provide tensor_index_key and optional UPSERT data.
  if (global_store_client_ && global_store_client_->is_connected()) {
    DeviceKey device{
        .type = (entry->plan == PendingRegistrationEntry::Plan::CPU ? DeviceType::CPU : DeviceType::GPU),
        .ordinal = (entry->plan == PendingRegistrationEntry::Plan::CPU ? -1 : entry->device_id),
        .uuid = ""};
    const std::string wid = worker_id_.empty() ? std::string("local") : worker_id_;
    std::optional<std::string> verification_json;
    if (!remote_keys.empty()) {
      std::vector<void*> data_ptrs = entry->replica->get_memory_manager().get_pointer(MemoryLocation::GPU);
      if (!data_ptrs.empty() && data_ptrs[0] != nullptr) {
        std::vector<size_t> data_sizes{static_cast<size_t>(entry->size_bytes)};
        auto info_or = common::ArtifactVerifier::generate_verification_info(
            data_ptrs, data_sizes, entry->device_id, common::VerificationLevel::KEY_POINTS);
        if (info_or.ok()) {
          verification_json = info_or->to_json();
        }
      }
    }

    auto reg_or = global_store_client_->register_memory_replica(
        entry->artifact_id,
        /*worker_id=*/wid,
        device,
        entry->size_bytes,
        entry->tensor_index_key,
        remote_keys,
        buffer_sizes,
        entry->tensor_index_data,
        entry->encoding,
        entry->schema_version,
        /*max_concurrency=*/1,
        verification_json);
    if (!reg_or.ok()) {
      return reg_or.status();
    }
  }

  // Compute view data hash and collect leaf digests once canonical data commit succeeds.
  std::optional<std::string> view_data_hash;
  std::optional<std::string> view_spec_json;
  std::vector<global_store::v1::LeafWrite> leaf_writes;
  std::vector<uint64_t> canonical_leaf_indices;
  std::optional<size_t> leaf_chunk_bytes;
  if (entry->view_state) {
    if (!segment_plan_ready && entry->tensor_index_data.has_value() && !entry->tensor_index_data->empty() &&
        entry->encoding == "json") {
      auto plan_or = loader::build_segment_plan_from_canonical_index_json(
          *entry->tensor_index_data, entry->size_bytes, /*align_bytes=*/8);
      if (plan_or.ok()) {
        segment_plan = std::move(*plan_or);
        segment_plan_ready = true;
      } else {
        LOG(WARNING) << "Failed to rebuild segment plan for view hash: " << plan_or.status();
      }
    }
    view_spec_json = build_view_spec_json(entry->view_state->options.spec);
    leaf_chunk_bytes =
        options_.artifact_chunk_bytes == 0 ? static_cast<size_t>(4ULL * 1024 * 1024) : options_.artifact_chunk_bytes;

    if (segment_plan_ready && entry->view_state->plan.forward.selection.total_bytes > 0 &&
        leaf_chunk_bytes.has_value()) {
      void* gpu_ptr = entry->gpu_ptr;
      if (gpu_ptr == nullptr) {
        const auto ptrs = entry->replica->get_memory_manager().get_pointer(MemoryLocation::GPU);
        gpu_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
      }
      if (gpu_ptr != nullptr) {
        loader::LinearizedGpuPlanSource canonical_source(
            gsl::not_null<void*>{gpu_ptr}, entry->device_id, absl::MakeSpan(segment_plan), entry->size_bytes);
        loader::ViewPlanSource view_source(
            gsl::not_null<loader::SeekableSource*>{&canonical_source}, entry->view_state->plan.forward.selection);
        auto view_hash_or = compute_tree_hash_and_leaves(
            view_source, entry->view_state->plan.forward.selection.total_bytes, *leaf_chunk_bytes);
        if (view_hash_or.ok()) {
          view_data_hash = view_hash_or->multihash;
          const auto& digests = view_hash_or->leaf_digests;
          leaf_writes.reserve(leaf_writes.size() + digests.size());
          for (size_t idx = 0; idx < digests.size(); ++idx) {
            global_store::v1::LeafWrite leaf;
            leaf.set_space_kind(tensorcast::global_store::v1::BYTE_SPACE_KIND_VARIANT);
            leaf.set_space_id(entry->view_state->options.view_id);
            leaf.set_leaf_idx(static_cast<uint64_t>(idx));
            const auto& digest = digests[idx];
            leaf.set_digest(digest.data(), static_cast<int>(digest.size()));
            leaf_writes.push_back(std::move(leaf));
          }
        } else {
          LOG(WARNING) << "ComputeTreeHashAndLeaves (view) failed: " << view_hash_or.status();
        }
      }
    }
    canonical_leaf_indices = compute_fully_covered_canonical_leaf_indices(
        entry->view_state->options.canonical_ranges, leaf_chunk_bytes.value_or(0));
  }

  if (entry->view_state && leaf_chunk_bytes.has_value() && !canonical_leaf_indices.empty()) {
    auto canonical_digest_or = compute_canonical_leaf_digests(
        *entry->replica,
        entry->plan == PendingRegistrationEntry::Plan::CPU,
        entry->gpu_ptr,
        entry->device_id,
        entry->size_bytes,
        canonical_leaf_indices,
        *leaf_chunk_bytes);
    if (!canonical_digest_or.ok()) {
      LOG(WARNING) << "Failed to compute canonical leaf digests for view registration: "
                   << canonical_digest_or.status();
    } else if (canonical_digest_or->size() != canonical_leaf_indices.size()) {
      LOG(WARNING) << "Canonical leaf digest count mismatch: expected " << canonical_leaf_indices.size() << " got "
                   << canonical_digest_or->size();
    } else {
      leaf_writes.reserve(leaf_writes.size() + canonical_leaf_indices.size());
      for (size_t i = 0; i < canonical_leaf_indices.size(); ++i) {
        global_store::v1::LeafWrite leaf;
        leaf.set_space_kind(tensorcast::global_store::v1::BYTE_SPACE_KIND_CANONICAL);
        leaf.set_space_id(*index_mh_or);
        leaf.set_leaf_idx(canonical_leaf_indices[i]);
        const auto& digest = (*canonical_digest_or)[i];
        leaf.set_digest(digest.data(), static_cast<int>(digest.size()));
        leaf_writes.push_back(std::move(leaf));
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_regs_.erase(std::string(registration_id));
  }

  RegistrationCommitResult result;
  result.registration_id = std::string(registration_id);
  result.artifact_id = entry->artifact_id;
  result.device_id = entry->device_id;
  result.size_bytes = entry->size_bytes;
  result.existed = false;
  result.index_multihash = *index_mh_or;
  result.data_multihash = *data_mh_or;
  result.schema_version = entry->schema_version;
  result.encoding = entry->encoding;
  uint64_t covered_bytes = 0;
  if (entry->view_state) {
    if (!entry->view_state->options.view_id.empty()) {
      result.view_id = entry->view_state->options.view_id;
    }
    if (view_data_hash.has_value()) {
      result.view_data_multihash = view_data_hash;
    }
    if (!entry->view_state->plan.forward.view_index_json.empty()) {
      result.view_index_json = entry->view_state->plan.forward.view_index_json;
    }
    result.canonical_ranges = entry->view_state->options.canonical_ranges;
    result.allow_partial = entry->view_state->options.allow_partial;
    covered_bytes = 0;
    for (const auto& range : result.canonical_ranges) {
      covered_bytes += range.length;
    }
  }

  if (entry->view_state && global_store_client_ && global_store_client_->is_connected() &&
      !entry->view_state->options.view_id.empty()) {
    components::VariantViewUpdate update;
    update.artifact_id = entry->artifact_id;
    update.view_id = entry->view_state->options.view_id;
    update.view_spec_json = view_spec_json.value_or(build_view_spec_json(entry->view_state->options.spec));
    update.view_size_bytes = entry->view_state->plan.forward.view_size_bytes;
    update.view_data_hash = view_data_hash;
    update.mark_verified = true;
    update.canonical_size_bytes = entry->size_bytes;
    update.canonical_bytes_covered = covered_bytes;
    update.leaf_writes = std::move(leaf_writes);
    absl::Status update_status = global_store_client_->update_artifact_view_state(update);
    if (!update_status.ok()) {
      LOG(WARNING) << "UpdateArtifactViewState failed for artifact " << entry->artifact_id
                   << " view_id=" << entry->view_state->options.view_id << ": " << update_status;
    }
  }
  return result;
}

absl::Status StoreEngine::ingest_view_registration_chunk(
    std::string_view registration_id,
    uint64_t view_offset,
    absl::Span<const std::byte> data) {
  if (data.empty()) {
    return absl::OkStatus();
  }
  std::shared_ptr<PendingRegistrationEntry> entry;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_regs_.find(std::string(registration_id));
    if (it == pending_regs_.end()) {
      return absl::NotFoundError("registration_id not found");
    }
    entry = it->second;
  }
  if (!entry || !entry->view_state) {
    return absl::FailedPreconditionError("registration is not view-enabled");
  }
  auto& view_state = *entry->view_state;
  if (view_state.options.placement != ViewPlacement::kServer) {
    return absl::FailedPreconditionError("view chunk ingestion is only valid for SERVER placement");
  }
  if (!view_state.executor) {
    return absl::FailedPreconditionError("view executor missing for SERVER placement ingestion");
  }
  void* canonical_base = entry->gpu_ptr;
  if (canonical_base == nullptr) {
    const auto ptrs = entry->replica->get_memory_manager().get_pointer(MemoryLocation::GPU);
    canonical_base = (!ptrs.empty() ? ptrs[0] : nullptr);
    if (canonical_base == nullptr) {
      return absl::FailedPreconditionError("GPU pointer unavailable for view ingestion");
    }
    entry->gpu_ptr = canonical_base;
  }
  auto status =
      view_state.executor->ingest_chunk(view_offset, data, MemoryLocation::GPU, canonical_base, entry->device_id);
  if (!status.ok()) {
    return status;
  }
  view_state.ingested_bytes = view_state.executor->ingested_bytes();
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> StoreEngine::get_view_registration_ingested_bytes(std::string_view registration_id) {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  auto it = pending_regs_.find(std::string(registration_id));
  if (it == pending_regs_.end()) {
    return absl::NotFoundError("registration_id not found");
  }
  const std::shared_ptr<PendingRegistrationEntry>& entry = it->second;
  if (!entry || !entry->view_state) {
    return absl::FailedPreconditionError("registration has no view state");
  }
  if (entry->view_state->executor) {
    return entry->view_state->executor->ingested_bytes();
  }
  return entry->view_state->expected_view_bytes;
}

absl::Status StoreEngine::keep_alive_registered_artifact(std::string_view registration_id, uint32_t ttl_ms) {
  if (ttl_ms == 0) {
    return absl::OkStatus();
  }
  std::lock_guard<std::mutex> lock(pending_mutex_);
  auto it = pending_regs_.find(std::string(registration_id));
  if (it == pending_regs_.end()) {
    return absl::NotFoundError("registration_id not found");
  }
  it->second->expiry_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(ttl_ms);
  return absl::OkStatus();
}

absl::StatusOr<loader::ViewPlan> StoreEngine::compute_view_plan(
    std::string_view canonical_index_json,
    const loader::ViewSpec& spec) {
  loader::ViewPlanner planner;
  return planner.compute_view_plan(canonical_index_json, spec);
}

bool StoreEngine::view_plan_allows_alias(const loader::ViewPlan& plan) {
  return plan.selection.is_segment_aligned && plan.selection.total_bytes > 0;
}

absl::StatusOr<std::string> StoreEngine::compute_view_data_hash_from_source(
    loader::SeekableSource& base_source,
    const loader::ViewPlan& plan,
    size_t leaf_chunk_bytes) {
  if (plan.selection.total_bytes == 0) {
    return absl::InvalidArgumentError("view plan selection has zero length");
  }
  loader::ViewPlanSource view_source(gsl::not_null<loader::SeekableSource*>{&base_source}, plan.selection);
  return loader::compute_data_multihash_from_seekable_source(view_source, plan.selection.total_bytes, leaf_chunk_bytes);
}

absl::Status StoreEngine::abort_registered_artifact(std::string_view registration_id) {
  std::shared_ptr<replica::Replica> replica;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_regs_.find(std::string(registration_id));
    if (it == pending_regs_.end()) {
      return absl::NotFoundError("registration_id not found");
    }
    replica = it->second->replica;
    pending_regs_.erase(it);
  }
  if (replica) {
    // Release GPU memory; best-effort cleanup but log failures.
    absl::Status _st = replica->release_memory(MemoryLocation::GPU);
    if (!_st.ok()) {
      VLOG(1) << "abort_registered_artifact: release_memory(GPU) failed: " << _st;
    }
  }
  return absl::OkStatus();
}

std::vector<replica::ChunkState> StoreEngine::get_chunk_states_telemetry(std::string_view artifact_id) const {
  std::vector<replica::ChunkState> out;
  auto span = va_space_->chunk_telemetry_snapshot(artifact_id);
  out.reserve(span.size());
  for (const auto& meta : span) {
    out.push_back(meta.state.load(std::memory_order_acquire));
  }
  return out;
}

std::vector<replica::ChunkState> StoreEngine::get_chunk_states_for_device(std::string_view artifact_id, int device_id)
    const {
  std::vector<replica::ChunkState> out;
  // Resolve the replica bound to the provided GPU device for this artifact.
  auto keys = replica_registry_->find_by_artifact(artifact_id);
  for (const auto& key : keys) {
    if (key.device.type == DeviceType::GPU && key.device.ordinal == device_id) {
      auto rep_or = replica_registry_->find(key);
      if (!rep_or.ok() || !*rep_or) {
        return out;
      }
      auto& rep = *rep_or;
      auto& mm = rep->get_memory_manager();
      return mm.get_chunk_states_uma(common::memory::MemoryLocation::GPU, device_id);
    }
  }
  // No GPU replica matched; return empty.
  return out;
}

// GPU device queries (exposed for status/health reporting)
absl::StatusOr<size_t> StoreEngine::get_device_total_memory(int device_id) const {
  auto info_or = device_manager_->get_gpu_info(device_id);
  if (!info_or.ok())
    return info_or.status();
  return static_cast<size_t>((*info_or)->total_memory);
}

absl::StatusOr<size_t> StoreEngine::get_device_free_memory(int device_id) const {
  return device_manager_->get_free_memory(device_id);
}

std::vector<replica::ChunkState> StoreEngine::get_chunk_states_cpu_uma(std::string_view artifact_id) const {
  std::vector<replica::ChunkState> out;
  // Gather all instances for this artifact
  auto keys = replica_registry_->find_by_artifact(artifact_id);
  if (keys.empty()) {
    return out;
  }
  // Choose primary: prefer CPU instance; otherwise GPU with smallest ordinal
  std::optional<loading::ReplicaKey> chosen;
  int min_gpu_ord = std::numeric_limits<int>::max();
  for (const auto& k : keys) {
    if (k.device.type == DeviceType::CPU) {
      chosen = k;
      break;
    }
    if (k.device.type == DeviceType::GPU) {
      if (!chosen.has_value() || (chosen->device.type != DeviceType::CPU && k.device.ordinal < min_gpu_ord)) {
        chosen = k;
        min_gpu_ord = k.device.ordinal;
      }
    }
  }
  if (!chosen.has_value()) {
    return out;
  }

  auto rep_or = replica_registry_->find(*chosen);
  if (!rep_or.ok() || !*rep_or) {
    return out;
  }
  auto& rep = *rep_or;
  auto& mm = rep->get_memory_manager();
  return mm.get_chunk_states_uma(common::memory::MemoryLocation::CPU);
}

} // namespace tensorcast::store
