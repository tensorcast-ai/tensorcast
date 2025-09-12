// Copyright (c) 2025, TensorCast Team.

#include "store_engine.h"

#include <unistd.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <memory>
#include <random>
// For reading/writing descriptor and canonical index files
#include <fstream>
#include <unordered_map>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_verification.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/common/trace/trace_macros.h"
#include "core/communicator/misc/common.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/replica_config.h"
// RFC-0007 helpers for safetensors canonical index
#include <nlohmann/json.hpp>
#include "core/store/loader/canonical_index.h"
#include "core/store/loader/safetensors_util.h"
// Unified hashing over SeekableSource for CPU/GPU/P2P
#include "core/store/loader/source_hash.h"
// SegmentPlan linearization (PAD=0 hashing)
#include "core/store/loader/segment_plan_source.h"
#include "core/store/loading/materialize_orchestrator.h"
#include "core/store/loading/replica_registration_helper.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"

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
    auto st = replica->release_memory(MemoryLocation::GPU, /*safe_release=*/true);
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
      chunk_size_(opts.chunk_size),
      pinned_memory_timeout_(opts.pinned_memory_timeout),
      device_manager_(
          gsl::not_null<std::unique_ptr<components::DeviceManager>>(std::make_unique<components::DeviceManager>())),
      replica_registry_(
          gsl::not_null<std::unique_ptr<components::ReplicaRegistry>>(std::make_unique<components::ReplicaRegistry>())),
      metrics_collector_(
          gsl::not_null<std::unique_ptr<components::MetricsCollector>>(
              std::make_unique<components::MetricsCollector>())),
      comm_manager_(
          gsl::not_null<std::shared_ptr<components::CommunicationManager>>(
              std::make_shared<components::CommunicationManager>())),
      memory_pool_(
          gsl::not_null<std::shared_ptr<common::memory::PinnedMemoryPool>>(
              std::make_shared<common::memory::PinnedMemoryPool>(memory_pool_size_, chunk_size_))),
      dvmp_(
          gsl::not_null<std::shared_ptr<common::memory::DistributedVirtualMemoryPool>>(
              std::make_shared<common::memory::DistributedVirtualMemoryPool>(opts.dvmp_chunk_size))) {
  VLOG(1) << "Initializing StoreEngine with unified Options constructor";
  VLOG(1) << "Storage path: "
          << (storage_path_.empty() ? "<empty - artifact_identifier will be full path>" : storage_path_.string());
  VLOG(1) << "Memory pool size: " << memory_pool_size_ / communicator::misc::GB << "GB";
  VLOG(1) << "I/O threads: " << num_thread_ << ", chunk size: " << chunk_size_ / communicator::misc::MB << "MB";

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

    global_store_client_ = std::make_unique<components::GlobalStoreClient>(gs_cfg);
    absl::Status st = global_store_client_->initialize();
    if (!st.ok()) {
      LOG(WARNING) << "StoreEngine: GlobalStoreClient init failed: " << st;
    } else {
      LOG(INFO) << "StoreEngine: connected to Global Store at " << gs_cfg.global_store_address;
    }
  }
}

void StoreEngine::initialize_communication_manager(const StoreEngineOptions& opts) {
  // Initialize system-wide DVMP instance and CommunicationManager handling
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

    auto cpu_state = replica->get_memory_state(common::memory::MemoryLocation::PAGEABLE_CPU);
    auto gpu_state = replica->get_memory_state(common::memory::MemoryLocation::GPU);

    auto is_present = [](replica::MemoryState st) {
      return st == replica::MemoryState::ALLOCATED || st == replica::MemoryState::LOADING ||
          st == replica::MemoryState::LOADED;
    };

    info.cpu_state =
        is_present(cpu_state) ? common::memory::MemoryLocation::PAGEABLE_CPU : common::memory::MemoryLocation::NONE;
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
  common::memory::MemoryLocation target_location = common::memory::MemoryLocation::PAGEABLE_CPU;
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

  // Get or create replica
  replica::ReplicaConfig config{
      .source = resolved_source,
      .artifact_identifier = artifact_identifier,
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_memory_pool = memory_pool_,
      .dvmp = dvmp_,
      .expected_artifact_size = std::nullopt};
  config.pinned_memory_timeout = hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : pinned_memory_timeout_;
  config.max_buffer_bytes = hints.max_buffer_bytes;
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
      (void)replica->release_memory(common::memory::MemoryLocation::GPU, /*safe_release=*/true);

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

  // RFC-0007: After loading, compute/verify content-addressed identity when possible.
  // Only perform strong verification when the replica is resident in GPU memory (fast path).
  std::optional<std::string> computed_data_mh;
  std::optional<std::string> computed_index_mh;
  std::optional<std::string> existing_index_mh;
  std::optional<std::string> existing_data_mh;
  uint64_t logical_total_size = 0;
  std::filesystem::path artifact_path = resolved_source.path;
  bool is_safetensors = false;
  {
    // Probe directory for .safetensors to differentiate formats
    for (const auto& entry : std::filesystem::directory_iterator(artifact_path)) {
      if (entry.is_regular_file()) {
        const auto name = entry.path().filename().string();
        if (name.ends_with(".safetensors")) {
          is_safetensors = true;
          break;
        }
      }
    }
  }

  // Try to read descriptor if present
  const auto descriptor_path = artifact_path / "artifact_descriptor.json";
  if (std::filesystem::exists(descriptor_path)) {
    std::ifstream f(descriptor_path);
    if (f.is_open()) {
      try {
        nlohmann::json j;
        f >> j;
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

  // Compute data multihash from GPU memory when requested by hints
  uint64_t verify_size = 0;
  const bool force_full_digest = options_.force_full_digest_on_load;
  if (hints.verify == MaterializeHints::Verify::FULL_DIGEST || force_full_digest) {
    if (auto sz_or = replica->get_artifact_size(); sz_or.ok()) {
      verify_size = *sz_or;
    }
    if (target_location == MemoryLocation::GPU) {
      const auto gpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::GPU);
      if (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr && verify_size > 0) {
        auto data_mh_or = loader::compute_data_multihash_from_gpu_memory(
            gsl::not_null<void*>{gpu_ptrs[0]}, verify_size, target_device_id);
        if (data_mh_or.ok()) {
          computed_data_mh = *data_mh_or;
        } else {
          LOG(WARNING) << "Data multihash computation (GPU) failed: " << data_mh_or.status();
        }
      }
    } else if (target_location == MemoryLocation::PAGEABLE_CPU) {
      const auto cpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::PAGEABLE_CPU);
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
    } else {
      // Build canonical index from safetensors headers
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
        // compute_index_multihash prefers inline data
        auto index_mh_or = common::compute_index_multihash(std::optional<std::string>(index_bytes_or.value()), "");
        if (index_mh_or.ok()) {
          computed_index_mh = *index_mh_or;
          // Determine logical total size from canonical index bytes
          try {
            nlohmann::json idx_json = nlohmann::json::parse(index_bytes_or.value(), nullptr, true);
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
          LOG(WARNING) << "Index multihash computation failed: " << index_mh_or.status();
        }
      } else {
        LOG(WARNING) << "Failed to build canonical index from safetensors: " << index_bytes_or.status();
      }
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
        auto index_mh_or = common::compute_index_multihash(std::optional<std::string>(canonical_json), "");
        if (index_mh_or.ok()) {
          computed_index_mh = *index_mh_or;
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
        } else {
          LOG(WARNING) << "Index multihash computation failed: " << index_mh_or.status();
        }
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to read/parse tensor_index.json: " << e.what();
    }
  }

  // If descriptor exists, verify data_multihash matches when we computed it
  if (existing_data_mh.has_value() && computed_data_mh.has_value()) {
    if (*existing_data_mh != *computed_data_mh) {
      return absl::DataLossError("ARTIFACT_ID_MISMATCH: data_multihash does not match loaded data");
    }
  }

  // Default post-load lightweight verification when descriptor is present
  if (std::filesystem::exists(descriptor_path)) {
    try {
      // Attempt to load optional verification.json and verify at SEGMENT_HASHES level
      const auto verification_path = artifact_path / "verification.json";
      if (std::filesystem::exists(verification_path)) {
        std::ifstream vf(verification_path);
        if (vf.is_open()) {
          std::stringstream vbuf;
          vbuf << vf.rdbuf();
          vf.close();
          auto ver_or = common::ArtifactVerificationInfo::from_json(vbuf.str());
          if (ver_or.ok()) {
            std::vector<void*> ptrs;
            std::vector<size_t> sizes;
            int dev_id = -1;
            uint64_t sz_for_verify = logical_total_size;
            if (sz_for_verify == 0) {
              if (auto sz_or = replica->get_artifact_size(); sz_or.ok()) {
                sz_for_verify = *sz_or;
              }
            }
            if (target_location == MemoryLocation::GPU) {
              const auto gpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::GPU);
              if (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr && sz_for_verify > 0) {
                ptrs.push_back(gpu_ptrs[0]);
                sizes.push_back(static_cast<size_t>(sz_for_verify));
                dev_id = target_device_id;
              }
            } else {
              const auto cpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::PAGEABLE_CPU);
              if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr && sz_for_verify > 0) {
                ptrs.push_back(cpu_ptrs[0]);
                sizes.push_back(static_cast<size_t>(sz_for_verify));
                dev_id = -1;
              }
            }
            if (!ptrs.empty()) {
              absl::Status vstatus = common::ArtifactVerifier::verify_artifact_data(
                  ptrs, sizes, *ver_or, common::VerificationLevel::SEGMENT_HASHES, dev_id);
              if (!vstatus.ok()) {
                return absl::DataLossError(
                    absl::StrCat("ARTIFACT_ID_MISMATCH: verification failed: ", vstatus.message()));
              }
            }
          }
        }
      } else {
        // No verification info persisted yet: generate at SEGMENT_HASHES level and persist for future fast checks
        std::vector<void*> ptrs;
        std::vector<size_t> sizes;
        int dev_id = -1;
        uint64_t sz_for_verify = logical_total_size;
        if (sz_for_verify == 0) {
          if (auto sz_or = replica->get_artifact_size(); sz_or.ok()) {
            sz_for_verify = *sz_or;
          }
        }
        if (target_location == MemoryLocation::GPU) {
          const auto gpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::GPU);
          if (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr && sz_for_verify > 0) {
            ptrs.push_back(gpu_ptrs[0]);
            sizes.push_back(static_cast<size_t>(sz_for_verify));
            dev_id = target_device_id;
          }
        } else {
          const auto cpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::PAGEABLE_CPU);
          if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr && sz_for_verify > 0) {
            ptrs.push_back(cpu_ptrs[0]);
            sizes.push_back(static_cast<size_t>(sz_for_verify));
            dev_id = -1;
          }
        }
        if (!ptrs.empty()) {
          auto gen_or = common::ArtifactVerifier::generate_verification_info(
              ptrs, sizes, dev_id, common::VerificationLevel::SEGMENT_HASHES);
          if (gen_or.ok()) {
            try {
              std::ofstream vf(verification_path);
              if (vf.is_open()) {
                vf << gen_or->to_json();
                vf.close();
              }
            } catch (const std::exception& e) {
              LOG(WARNING) << "Failed to persist verification.json: " << e.what();
            }
          }
        }
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Post-load verification skipped due to error: " << e.what();
    }
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
          const auto gpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::GPU);
          if (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) {
            auto mh_or = loader::compute_data_multihash_from_gpu_memory(
                gsl::not_null<void*>{gpu_ptrs[0]}, total, target_device_id);
            if (mh_or.ok()) {
              computed_data_mh = *mh_or;
            }
          }
        } else {
          const auto cpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::PAGEABLE_CPU);
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
        j["schema_version"] = "v2";
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
  handle.replica_key = loading::ReplicaKey{.artifact_id = artifact_identifier, .device = dev_key, .replica = 0};

  // Loading future and states
  handle.ready_future = load_future;
  handle.cpu_state = replica->get_memory_state(common::memory::MemoryLocation::PAGEABLE_CPU);
  handle.gpu_state = replica->get_memory_state(common::memory::MemoryLocation::GPU);

  if (target_location == common::memory::MemoryLocation::GPU) {
    const auto gpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
    handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

    // Attempt to obtain CUDA IPC handle bytes for the allocated GPU buffer.
    auto ipc_or = replica->get_memory_manager().get_cuda_ipc_handle();
    if (ipc_or.ok()) {
      std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
    }
  }

  // Update metrics
  const auto duration = std::chrono::steady_clock::now() - start_time;
  const double duration_s = std::chrono::duration<double>(duration).count();
  metrics_collector_->record_operation("load_from_disk", duration_s); // no-op after Phase 5
  // Unified artifact load metric with labels
  metrics_collector_->record_artifact_load(
      /*source=*/"disk",
      /*device=*/(target_location == common::memory::MemoryLocation::GPU ? std::string("gpu") : std::string("cpu")),
      /*phase=*/"finalize",
      duration_s);
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

  if (!comm_manager_->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }

  common::memory::MemoryLocation target_location = common::memory::MemoryLocation::PAGEABLE_CPU;
  if (target.location.type == common::memory::MemoryLocation::GPU) {
    target_location = common::memory::MemoryLocation::GPU;
  }

  // Create replica with P2P source
  auto p2p_source = source;
  p2p_source.comm_engine =
      gsl::not_null<std::shared_ptr<communicator::engine::CommunicateEngine>>{comm_manager_->get_shared_engine()};
  // Provide optional disk fallback directory from engine options
  p2p_source.fallback_disk_dir = options_.p2p_fallback_disk_dir;
  replica::ReplicaConfig config{
      .source = p2p_source,
      .artifact_identifier = artifact_identifier,
      .device_type = (target_location == common::memory::MemoryLocation::GPU ? DeviceType::GPU : DeviceType::CPU),
      .local_device_id = target.location.device_id,
      .pinned_memory_pool = memory_pool_,
      .dvmp = dvmp_,
      .expected_artifact_size = std::nullopt};
  config.pinned_memory_timeout = hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : pinned_memory_timeout_;
  config.local_device_id = target.location.device_id;
  config.max_buffer_bytes = hints.max_buffer_bytes;
  config.p2p_comm_enabled = true;
  config.device_type = (target_location == common::memory::MemoryLocation::GPU) ? DeviceType::GPU : DeviceType::CPU;

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
  handle.replica_key = loading::ReplicaKey{.artifact_id = artifact_identifier, .device = dev_key, .replica = 0};

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

  // Ready future is already resolved (synchronous path)
  std::promise<absl::Status> promise;
  promise.set_value(absl::OkStatus());
  handle.ready_future = promise.get_future().share();

  handle.cpu_state = replica->get_memory_state(common::memory::MemoryLocation::PAGEABLE_CPU);
  handle.gpu_state = replica->get_memory_state(common::memory::MemoryLocation::GPU);

  if (target_location == common::memory::MemoryLocation::GPU) {
    const auto gpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
    handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

    auto ipc_or = replica->get_memory_manager().get_cuda_ipc_handle();
    if (ipc_or.ok()) {
      std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
    }
  }

  // Update metrics
  metrics_collector_->record_p2p_transfer(source.size_bytes, true);
  const auto duration = std::chrono::steady_clock::now() - start_time;
  const double duration_s = std::chrono::duration<double>(duration).count();
  metrics_collector_->record_operation("load_from_p2p", duration_s);
  // Unified artifact load metric with labels
  metrics_collector_->record_artifact_load(
      /*source=*/"remote",
      /*device=*/(target_location == common::memory::MemoryLocation::GPU ? std::string("gpu") : std::string("cpu")),
      /*phase=*/"finalize",
      duration_s);
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
  loading::ReplicaKey inst_key{.artifact_id = artifact_identifier, .device = dev_key, .replica = 0};

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
    auto free_status = replica->release_memory(common::memory::MemoryLocation::PAGEABLE_CPU, /*safe_release=*/true);
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
  return (tensor_size + chunk_size_ - 1) / chunk_size_;
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
  const ReplicaKey dst_key{.artifact_id = hints.artifact_id, .device = target_device, /*replica=*/.replica = 0};
  if (auto existing_or = replica_registry_->find(dst_key); existing_or.ok()) {
    const auto& replica = existing_or.value();

    MemoryLocation dst_loc =
        (target_device.type == DeviceType::GPU) ? MemoryLocation::GPU : MemoryLocation::PAGEABLE_CPU;
    std::optional<int> opt_dev;
    if (dst_loc == MemoryLocation::GPU) {
      opt_dev = target_device.ordinal;
    }

    auto fut = replica->ensure_loaded_async(dst_loc, num_thread_, opt_dev);

    ReplicaHandle handle;
    handle.replica_key = dst_key;
    handle.ready_future = fut;
    handle.cpu_state = replica->get_memory_state(MemoryLocation::PAGEABLE_CPU);
    handle.gpu_state = replica->get_memory_state(MemoryLocation::GPU);

    if (dst_loc == MemoryLocation::GPU) {
      const auto gpu_ptrs = replica->get_data_pointer(MemoryLocation::GPU);
      handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

      auto ipc_or = replica->get_memory_manager().get_cuda_ipc_handle();
      if (ipc_or.ok()) {
        std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
      }
    }
    return handle;
  }

  // Helper lambda: minimal disk-loading path.
  auto load_from_disk = [&](const DeviceKey& dev_key) -> absl::StatusOr<loading::ReplicaHandle> {
    // Guard: content-addressed IDs (mi2:...) are not paths.
    if (absl::StartsWith(hints.artifact_id, "mi2:")) {
      return absl::FailedPreconditionError(
          "LOAD_ONLY/disk fallback disabled for content-addressed artifact_id; Global Store routing required");
    }
    loading::DiskSource disk_src;
    if (!hints.disk_path.empty()) {
      disk_src.path = std::filesystem::path(hints.disk_path);
    } else {
      disk_src.path = std::filesystem::path(hints.artifact_id);
    }

    loading::ReplicaTarget target;
    target.location.type = (dev_key.type == DeviceType::GPU) ? common::memory::MemoryLocation::GPU
                                                             : common::memory::MemoryLocation::PAGEABLE_CPU;
    target.location.device_id = dev_key.ordinal;

    return ingest_from_disk_internal(
        hints.disk_path.empty() ? hints.artifact_id : hints.disk_path, disk_src, target, hints);
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
      if (hints.artifact_id.empty()) {
        return absl::InvalidArgumentError("COPY_ONLY requires hints.artifact_id to locate the source GPU instance");
      }

      const auto candidates = replica_registry_->find_by_artifact(hints.artifact_id);
      for (const auto& cand_key : candidates) {
        if (cand_key.device.type != DeviceType::GPU) {
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
            .artifact_identifier = hints.artifact_id,
            .device_type = DeviceType::GPU,
            .local_device_id = target_device.ordinal,
            .pinned_memory_pool = memory_pool_,
            .dvmp = dvmp_,
            .expected_artifact_size = expected_size};
        cfg.pinned_memory_timeout = pinned_memory_timeout_;

        auto dst_or = replica::Replica::create(cfg);
        if (!dst_or.ok()) {
          return dst_or.status();
        }
        auto dst_replica = std::shared_ptr<replica::Replica>(std::move(dst_or.value()));
        (void)replica_registry_->emplace(dst_key, gsl::not_null<std::shared_ptr<replica::Replica>>{dst_replica});

        absl::Status copy_st = dst_replica->copy_from(*src_replica);

        std::promise<absl::Status> p;
        p.set_value(copy_st);

        ReplicaHandle handle;
        handle.replica_key = dst_key;
        handle.ready_future = p.get_future().share();
        handle.cpu_state = dst_replica->get_memory_state(MemoryLocation::PAGEABLE_CPU);
        handle.gpu_state = dst_replica->get_memory_state(MemoryLocation::GPU);
        if (handle.gpu_state == MemoryState::LOADED) {
          const auto gpu_ptrs = dst_replica->get_data_pointer(MemoryLocation::GPU);
          handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

          auto ipc_or = dst_replica->get_memory_manager().get_cuda_ipc_handle();
          if (ipc_or.ok()) {
            std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
          }
        }
        return handle;
      }
      return absl::FailedPreconditionError(
          absl::StrCat("No suitable source instance for COPY_ONLY mode (artifact_id=", hints.artifact_id, ")"));
    }

    case MaterializeMode::LOAD_ONLY: {
      return load_from_disk(target_device);
    }

    case MaterializeMode::AUTO: {
      if (global_store_client_ && global_store_client_->is_connected() && !hints.artifact_id.empty()) {
        loading::MaterializeOrchestrator orchestrator(
            gsl::not_null<StoreEngine*>{this},
            gsl::not_null<components::GlobalStoreClient*>{global_store_client_.get()});
        auto orchestrated_or = orchestrator.run(hints.artifact_id, target_device, hints);
        if (orchestrated_or.ok()) {
          return *orchestrated_or;
        }
        LOG(WARNING) << "MaterializeOrchestrator failed: " << orchestrated_or.status() << "; falling back to disk load";
      }
      return absl::FailedPreconditionError(
          "AUTO materialize_replica requires a content-addressed hints.artifact_id with Global Store routing or an explicit hints.disk_path");
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
      if (is_present(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU))) {
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
      if (is_present(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU))) {
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
  MemoryLocation loc = (key.device.type == DeviceType::CPU) ? MemoryLocation::PAGEABLE_CPU : MemoryLocation::GPU;
  absl::Status st = replica->wait_until_loaded(loc, absl::InfiniteDuration());
  return st.ok() ? 0 : 1;
}

int StoreEngine::unload_replica(const ReplicaKey& key) {
  auto replica_or = replica_registry_->find(key);
  if (!replica_or.ok()) {
    return 1; // Instance not found.
  }

  const auto& replica = replica_or.value();
  MemoryLocation loc = (key.device.type == DeviceType::CPU) ? MemoryLocation::PAGEABLE_CPU : MemoryLocation::GPU;

  // Inspect current state *before* attempting the release so we can tell if
  // there was anything to unload.  This avoids treating a no-op release as a
  // success – a scenario that would allow multiple threads to report success
  // when only the first one actually freed memory.
  store::MemoryState before_state = replica->get_memory_state(loc);

  if (before_state <= MemoryState::UNALLOCATED) {
    // Nothing to release – another thread has already unloaded this instance.
    return -1;
  }

  absl::Status st = replica->release_memory(loc);
  return st.ok() ? 0 : -1;
}

MemoryState StoreEngine::get_replica_state(const ReplicaKey& key, DeviceType memory_type) const {
  auto replica_or = replica_registry_->find(key);
  if (!replica_or.ok()) {
    return MemoryState::UNINITIALIZED;
  }
  MemoryLocation loc = (memory_type == DeviceType::CPU) ? MemoryLocation::PAGEABLE_CPU : MemoryLocation::GPU;
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

absl::StatusOr<CommRegistrationInfo> StoreEngine::enable_remote_replica_access(
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
    auto cpu_status = replica->release_memory(MemoryLocation::PAGEABLE_CPU);
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
  // Determine memory location based on device type
  MemoryLocation loc = (key.device.type == DeviceType::GPU) ? MemoryLocation::GPU : MemoryLocation::PAGEABLE_CPU;

  // Fetch total size for registration
  uint64_t size = 0;
  if (auto sz_or = get_replica_size(key); sz_or.ok()) {
    size = *sz_or;
  } else {
    return sz_or.status();
  }

  const std::string artifact_id = artifact_id_override.empty() ? key.artifact_id : std::string(artifact_id_override);
  const std::string wid = worker_id_.empty() ? std::string("local") : worker_id_;
  return ReplicaRegistrationHelper::register_local_replica(
      gsl::not_null<components::GlobalStoreClient*>{global_store_client_.get()},
      wid,
      artifact_id,
      key.device,
      loc,
      size);
}

// ═══════════════════════════════════════════════════════════════════════════
// RFC-0014: Key-mapping wrappers
// ═══════════════════════════════════════════════════════════════════════════

absl::StatusOr<components::GlobalStoreClient::KeyMapping> StoreEngine::resolve_key_mapping(std::string_view key) {
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
// Distributed Memory Pool (DVMP) chunk locking API
// ═══════════════════════════════════════════════════════════════════════════

absl::Status StoreEngine::lock_chunks(const ReplicaKey& replica_key, absl::Span<const uint32_t> chunk_indices) {
  SC_TRACE_SCOPE("StoreEngine::lock_chunks");

  // Locate the exact Replica based on ReplicaKey (device-specific).
  auto replica_or2 = replica_registry_->find(replica_key);
  if (!replica_or2.ok()) {
    return absl::NotFoundError(
        absl::StrCat("Replica not found: ", replica_key.artifact_id, " @ ", replica_key.device.to_string()));
  }

  const auto& replica = replica_or2.value();

  // Get the DVMP instance via MemoryManager.
  const auto dvmp = replica->get_memory_manager().get_dvmp();
  // Forward the call to DVMP – DVMP is still keyed by artifact_id (shared CPU VA).
  return dvmp->lock_chunks(replica_key.artifact_id, chunk_indices);
}

absl::Status StoreEngine::unlock_chunks(
    const ReplicaKey& replica_key,
    absl::Span<const uint32_t> chunk_indices,
    bool copied_gpu) {
  SC_TRACE_SCOPE("StoreEngine::unlock_chunks");

  // Locate the exact Replica.
  auto replica_or3 = replica_registry_->find(replica_key);
  if (!replica_or3.ok()) {
    return absl::NotFoundError(
        absl::StrCat("Replica not found: ", replica_key.artifact_id, " @ ", replica_key.device.to_string()));
  }

  const auto& replica = replica_or3.value();

  // Get the DVMP instance via MemoryManager.
  const auto dvmp = replica->get_memory_manager().get_dvmp();

  // Forward the call to DVMP.
  return dvmp->unlock_chunks(replica_key.artifact_id, chunk_indices, copied_gpu);
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
  // Accept either a pre-existing index key or inline index data.  Only error
  // when both are absent.
  if (reg.tensor_index_key.empty() && !reg.tensor_index_data.has_value()) {
    return absl::InvalidArgumentError("tensor index key or data must be provided");
  }

  // Prepare a memory-only Replica bound to target GPU to own the allocation.
  // Use InlineBufferSource with the known total size so Replica::create() can
  // construct MemoryManager without requiring any on-disk layout.
  InlineBufferSource ib_source{.data = nullptr, .size_bytes = reg.total_size_bytes};
  replica::ReplicaConfig cfg{
      .source = ib_source,
      .artifact_identifier = reg.artifact_id,
      .device_type = DeviceType::GPU,
      .local_device_id = reg.device_id,
      .pinned_memory_pool = memory_pool_,
      .dvmp = dvmp_,
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
          return absl::ResourceExhaustedError(
              absl::StrCat(
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
  auto ipc_or = replica->get_memory_manager().get_cuda_ipc_handle();
  if (!ipc_or.ok()) {
    return ipc_or.status();
  }

  const auto gpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::GPU);
  void* base_ptr = (!gpu_ptrs.empty() ? gpu_ptrs[0] : nullptr);

  // Emplace into registry to ensure lifecycle is tracked (ReplicaKey via config).
  DeviceKey dev_key{.type = DeviceType::GPU, .ordinal = reg.device_id, .uuid = ""};
  ReplicaKey inst_key{.artifact_id = reg.artifact_id, .device = dev_key, /*replica=*/.replica = 0};
  (void)replica_registry_->emplace(inst_key, gsl::not_null<std::shared_ptr<replica::Replica>>{replica});

  // Create pending entry with cryptographically secure random ID
  // Use a combination of timestamp, process ID, and random bytes for uniqueness
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;
  auto reg_id = absl::StrCat("reg_", absl::ToUnixNanos(absl::Now()), "_", getpid(), "_", dis(gen));

  PendingRegistrationEntry entry;
  entry.registration_id = reg_id;
  entry.artifact_id = reg.artifact_id;
  entry.device_id = reg.device_id;
  entry.size_bytes = reg.total_size_bytes;
  entry.tensor_index_key = reg.tensor_index_key;
  entry.tensor_index_data = reg.tensor_index_data;
  entry.schema_version = reg.schema_version;
  entry.encoding = reg.encoding;
  entry.enable_p2p = reg.enable_p2p;
  if (reg.ttl_ms > 0) {
    entry.expiry_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(reg.ttl_ms);
  }
  entry.replica = replica;
  entry.gpu_ptr = base_ptr;
  entry.ipc_handle = *ipc_or;
  entry.plan = PendingRegistrationEntry::Plan::COALESCED;

  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_regs_.emplace(reg_id, std::move(entry));
  }

  RegistrationBeginResult out;
  out.registration_id = reg_id;
  out.device_id = reg.device_id;
  out.size_bytes = reg.total_size_bytes;
  std::memcpy(out.cuda_ipc_handle_bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
  return out;
}

absl::StatusOr<StoreEngine::RegistrationBeginResult> StoreEngine::begin_register_artifact_dvmp(
    const ArtifactRegistration& reg) {
  if (reg.total_size_bytes == 0) {
    return absl::InvalidArgumentError("total_size_bytes must be > 0");
  }
  // Accept either pre-existing index key or inline index data.
  if (reg.tensor_index_key.empty() && !reg.tensor_index_data.has_value()) {
    return absl::InvalidArgumentError("tensor index key or data must be provided");
  }

  // Build ReplicaConfig for CPU residency; use InlineBufferSource to indicate size.
  InlineBufferSource ib_source{.data = nullptr, .size_bytes = reg.total_size_bytes};
  replica::ReplicaConfig cfg{
      .source = ib_source,
      .artifact_identifier = reg.artifact_id,
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_memory_pool = memory_pool_,
      .dvmp = dvmp_,
      .expected_artifact_size = reg.total_size_bytes};
  cfg.pinned_memory_timeout = pinned_memory_timeout_;

  auto create_or = Replica::create(cfg);
  if (!create_or.ok()) {
    return create_or.status();
  }
  auto replica = std::shared_ptr<Replica>(std::move(create_or.value()));

  // Allocate DVMP-backed PAGEABLE_CPU memory.
  absl::Status st = replica->get_memory_manager().allocate_memory(MemoryLocation::PAGEABLE_CPU);
  if (!st.ok()) {
    return st;
  }

  // Register CPU-resident replica under temporary identifier for lifecycle tracking.
  DeviceKey cpu_key{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  ReplicaKey inst_key{.artifact_id = reg.artifact_id, .device = cpu_key, /*replica=*/.replica = 0};
  (void)replica_registry_->emplace(inst_key, gsl::not_null<std::shared_ptr<replica::Replica>>{replica});

  // Create registration id and track pending DVMP registration.
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;
  auto reg_id = absl::StrCat("reg_", absl::ToUnixNanos(absl::Now()), "_", getpid(), "_", dis(gen));

  PendingRegistrationEntry entry;
  entry.registration_id = reg_id;
  entry.artifact_id = reg.artifact_id;
  entry.device_id = reg.device_id;
  entry.size_bytes = reg.total_size_bytes;
  entry.tensor_index_key = reg.tensor_index_key;
  entry.tensor_index_data = reg.tensor_index_data;
  entry.schema_version = reg.schema_version;
  entry.encoding = reg.encoding;
  entry.enable_p2p = reg.enable_p2p;
  if (reg.ttl_ms > 0) {
    entry.expiry_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(reg.ttl_ms);
  }
  entry.replica = replica;
  entry.gpu_ptr = nullptr;
  std::memset(&entry.ipc_handle, 0, sizeof(entry.ipc_handle));
  entry.plan = PendingRegistrationEntry::Plan::DVMP;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_regs_.emplace(reg_id, std::move(entry));
  }

  RegistrationBeginResult out;
  out.registration_id = reg_id;
  out.device_id = reg.device_id;
  out.size_bytes = reg.total_size_bytes;
  // IPC handle remains zero for DVMP plan
  return out;
}

absl::Status StoreEngine::feed_register_dvmp_chunk(
    std::string_view registration_id,
    uint64_t offset,
    const void* data,
    size_t bytes) {
  std::shared_ptr<Replica> replica;
  std::string art_id;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_regs_.find(std::string(registration_id));
    if (it == pending_regs_.end()) {
      LOG(ERROR) << "DVMP feed: registration_id not found: " << registration_id;
      return absl::NotFoundError("registration_id not found");
    }
    if (it->second.plan != PendingRegistrationEntry::Plan::DVMP) {
      LOG(ERROR) << "DVMP feed: plan mismatch for reg_id=" << registration_id;
      return absl::FailedPreconditionError("registration plan is not DVMP");
    }
    if (offset + bytes > it->second.size_bytes) {
      LOG(ERROR) << "DVMP feed OOB: offset=" << offset << ", bytes=" << bytes
                 << ", total_size=" << it->second.size_bytes << ", reg_id=" << registration_id;
      return absl::OutOfRangeError("DVMP feed write would exceed total size");
    }
    replica = it->second.replica;
    art_id = it->second.artifact_id;
  }
  auto dvmp = replica->get_memory_manager().get_dvmp();
  return dvmp->write_at(art_id, offset, data, bytes);
}

absl::StatusOr<StoreEngine::RegistrationCommitResult> StoreEngine::commit_registered_artifact(
    std::string_view registration_id) {
  PendingRegistrationEntry entry;
  std::shared_ptr<Replica> expired_replica;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_regs_.find(std::string(registration_id));
    if (it == pending_regs_.end()) {
      return absl::NotFoundError("registration_id not found");
    }
    // TTL enforcement: if expired, cleanup and fail fast.
    // Keep the lock held to prevent race conditions during TTL check and removal
    if (it->second.expiry_time.time_since_epoch().count() > 0 &&
        std::chrono::steady_clock::now() > it->second.expiry_time) {
      expired_replica = it->second.replica;
      pending_regs_.erase(it);
      // Release outside the lock to avoid holding mutex during potentially slow operation
    } else {
      entry = it->second; // make a copy; we may erase after successful commit
    }
  }

  // Release memory outside the critical section if TTL expired
  if (expired_replica) {
    (void)expired_replica->release_memory(MemoryLocation::GPU, /*safe_release=*/true);
    return absl::DeadlineExceededError("registration expired (TTL)");
  }

  // Compute content-addressed artifact_id per RFC-0007: "mi2:<index_multihash>:<data_multihash>"
  // 1) index_multihash from canonical index bytes when provided, otherwise from key (sha256 hex)
  absl::StatusOr<std::string> index_mh_or =
      common::compute_index_multihash(entry.tensor_index_data, entry.tensor_index_key);
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }

  // 2) data_multihash via SegmentPlan linearization (PAD=0).
  absl::StatusOr<std::string> data_mh_or;
  if (entry.plan == PendingRegistrationEntry::Plan::DVMP) {
    // Hash directly from DVMP CPU memory (zero-initialized PAD regions).
    auto region_or = dvmp_->region_info(entry.artifact_id);
    if (!region_or.ok()) {
      return region_or.status();
    }
    auto mh_or = loader::compute_data_multihash_from_cpu_memory(
        gsl::not_null<const void*>{region_or->cpu_base}, entry.size_bytes);
    if (!mh_or.ok())
      return mh_or.status();
    data_mh_or = *mh_or;
  } else {
    void* gpu_ptr = nullptr;
    {
      const auto ptrs = entry.replica->get_memory_manager().get_pointer(MemoryLocation::GPU);
      gpu_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
    }
    if (entry.tensor_index_data.has_value() && !entry.tensor_index_data->empty() && entry.encoding == "json") {
      auto plan_or = loader::build_segment_plan_from_canonical_index_json(
          *entry.tensor_index_data, entry.size_bytes, /*align_bytes=*/8);
      if (plan_or.ok()) {
        if (!gpu_ptr) {
          return absl::FailedPreconditionError("GPU pointer is null; cannot hash GPU plan");
        }
        auto mh_or = loader::compute_data_multihash_from_gpu_plan(
            gsl::not_null<void*>{gpu_ptr}, entry.device_id, absl::MakeSpan(*plan_or), entry.size_bytes);
        if (!mh_or.ok())
          return mh_or.status();
        data_mh_or = *mh_or;
      } else {
        return plan_or.status();
      }
    } else {
      auto mh_or = common::compute_data_multihash_from_gpu(gpu_ptr, entry.size_bytes, entry.device_id);
      if (!mh_or.ok())
        return mh_or.status();
      data_mh_or = *mh_or;
    }
  }
  if (!data_mh_or.ok()) {
    return data_mh_or.status();
  }

  const std::string artifact_id_mi2 = absl::StrCat("mi2:", *index_mh_or, ":", *data_mh_or);

  // Replace entry.artifact_id with the content-addressed ID for registration and return
  entry.artifact_id = artifact_id_mi2;

  // Idempotent success: if the same content-addressed artifact already has a
  // replica on the target device, reclaim this allocation and return OK with
  // existed=true and the computed descriptor. No new registry mapping is
  // created and no Global Store upsert is attempted.
  {
    DeviceKey dev_key{
        .type = (entry.plan == PendingRegistrationEntry::Plan::DVMP ? DeviceType::CPU : DeviceType::GPU),
        .ordinal = (entry.plan == PendingRegistrationEntry::Plan::DVMP ? -1 : entry.device_id),
        .uuid = ""};
    loading::ReplicaKey check_key{.artifact_id = entry.artifact_id, .device = dev_key, .replica = 0};
    auto existing_or = replica_registry_->find(check_key);
    if (existing_or.ok()) {
      // Cleanup: erase pending entry and best-effort release the allocation we made
      {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_regs_.erase(std::string(registration_id));
      }
      if (entry.plan == PendingRegistrationEntry::Plan::DVMP) {
        // DVMP path allocated PAGEABLE_CPU memory
        (void)entry.replica->release_memory(MemoryLocation::PAGEABLE_CPU, /*safe_release=*/true);
      } else {
        // Coalesced/Lease-materialized path allocated GPU memory
        (void)entry.replica->release_memory(MemoryLocation::GPU, /*safe_release=*/true);
      }
      RegistrationCommitResult result;
      result.registration_id = std::string(registration_id);
      result.artifact_id = entry.artifact_id;
      result.device_id = entry.device_id;
      result.size_bytes = entry.size_bytes;
      result.existed = true;
      result.index_multihash = *index_mh_or;
      result.data_multihash = *data_mh_or;
      result.schema_version = entry.schema_version;
      result.encoding = entry.encoding;
      return result;
    }
  }

  // Also add a registry mapping for the content-addressed artifact_id so callers can
  // subsequently reference the instance by its mi2: identifier (in addition to the
  // original logical artifact_id used during Begin/Commit).
  {
    DeviceKey dev_key{
        .type = (entry.plan == PendingRegistrationEntry::Plan::DVMP ? DeviceType::CPU : DeviceType::GPU),
        .ordinal = (entry.plan == PendingRegistrationEntry::Plan::DVMP ? -1 : entry.device_id),
        .uuid = ""};
    ReplicaKey mi2_key{.artifact_id = entry.artifact_id, .device = dev_key, .replica = 0};
    (void)replica_registry_->emplace(mi2_key, gsl::not_null<std::shared_ptr<replica::Replica>>{entry.replica});
  }

  // Export remote memory keys if communication is enabled (GPU location).
  std::vector<std::string> remote_keys;
  std::vector<uint64_t> buffer_sizes;
  if (entry.plan != PendingRegistrationEntry::Plan::DVMP && entry.enable_p2p && comm_manager_->is_enabled()) {
    auto reg_info_or = entry.replica->enable_remote_memory_access(MemoryLocation::GPU, comm_manager_->get_engine());
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
        .type = (entry.plan == PendingRegistrationEntry::Plan::DVMP ? DeviceType::CPU : DeviceType::GPU),
        .ordinal = (entry.plan == PendingRegistrationEntry::Plan::DVMP ? -1 : entry.device_id),
        .uuid = ""};
    const std::string wid = worker_id_.empty() ? std::string("local") : worker_id_;
    // Generate lightweight verification info (KEY_POINTS) for receiver-side validation
    std::optional<std::string> verification_json;
    if (!remote_keys.empty()) {
      std::vector<void*> data_ptrs = entry.replica->get_memory_manager().get_pointer(MemoryLocation::GPU);
      if (!data_ptrs.empty() && data_ptrs[0] != nullptr) {
        std::vector<size_t> data_sizes{static_cast<size_t>(entry.size_bytes)};
        auto info_or = common::ArtifactVerifier::generate_verification_info(
            data_ptrs, data_sizes, entry.device_id, common::VerificationLevel::KEY_POINTS);
        if (info_or.ok()) {
          verification_json = info_or->to_json();
        }
      }
    }

    auto reg_or = global_store_client_->register_memory_replica(
        entry.artifact_id,
        /*worker_id=*/wid,
        device,
        entry.size_bytes,
        entry.tensor_index_key,
        remote_keys,
        buffer_sizes,
        entry.tensor_index_data,
        entry.encoding,
        entry.schema_version,
        /*max_concurrency=*/1,
        verification_json);
    if (!reg_or.ok()) {
      return reg_or.status();
    }
  }

  // Finalize: remove from pending list on success.
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_regs_.erase(std::string(registration_id));
  }
  RegistrationCommitResult result;
  result.registration_id = std::string(registration_id);
  result.artifact_id = entry.artifact_id;
  result.device_id = entry.device_id;
  result.size_bytes = entry.size_bytes;
  result.existed = false;
  // RFC-0007: Fill descriptor fields for upstream layers
  result.index_multihash = *index_mh_or;
  result.data_multihash = *data_mh_or;
  result.schema_version = entry.schema_version;
  result.encoding = entry.encoding;
  return result;
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
  it->second.expiry_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(ttl_ms);
  return absl::OkStatus();
}

absl::Status StoreEngine::abort_registered_artifact(std::string_view registration_id) {
  std::shared_ptr<replica::Replica> replica;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_regs_.find(std::string(registration_id));
    if (it == pending_regs_.end()) {
      return absl::NotFoundError("registration_id not found");
    }
    replica = it->second.replica;
    pending_regs_.erase(it);
  }
  if (replica) {
    // Release GPU memory; ignore failures to guarantee best-effort cleanup.
    (void)replica->release_memory(MemoryLocation::GPU, /*safe_release=*/true);
  }
  return absl::OkStatus();
}

std::vector<replica::ChunkState> StoreEngine::get_chunk_states(std::string_view artifact_id) const {
  std::vector<replica::ChunkState> out;
  auto span = dvmp_->chunk_snapshot(artifact_id);
  out.reserve(span.size());
  for (const auto& meta : span) {
    out.push_back(meta.state.load(std::memory_order_acquire));
  }
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

} // namespace tensorcast::store
