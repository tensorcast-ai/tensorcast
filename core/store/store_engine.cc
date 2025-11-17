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
// For reading/writing descriptor and canonical index files
#include <fstream>
#include <limits>
#include <system_error>

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
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/replica_config.h"
// RFC-0007 helpers for safetensors canonical index
#include <nlohmann/json.hpp>
#include "core/store/components/eviction_service.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/materialization/dataplane/metadata/safetensors_util.h"
#include "core/store/materialization/dataplane/view/view_plan_source.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/materialization/dataplane/view/view_transform_executor.h"
// Unified hashing over SeekableSource for CPU/GPU/P2P
#include "core/store/materialization/contracts/materialization_request.h"
#include "core/store/materialization/control/materialization_service.h"
#include "core/store/materialization/control/materialize_orchestrator.h"
#include "core/store/materialization/control/replica_registration_helper.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/verification/verification_utils.h"
#include "gsl/pointers"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"

namespace tensorcast::store {

using common::memory::MemoryLocation;
using components::DeviceManager;
using components::MetricsCollector;
using components::ReplicaRegistry;
using loading::MaterializeHints;
using loading::ReplicaHandle;
using loading::ReplicaKey;
using materialization::control::ReplicaRegistrationHelper;
using replica::MemoryState;
using replica::Replica;

std::optional<std::string> ComputeViewDataHash(
    replica::Replica& replica,
    MemoryLocation location,
    uint64_t view_size_bytes,
    std::optional<int> gpu_device_id) {
  if (view_size_bytes == 0) {
    return std::nullopt;
  }
  loader::verification::MemoryView mem_view;
  mem_view.location = location;
  mem_view.size_bytes = view_size_bytes;
  mem_view.gpu_device_id = gpu_device_id;
  std::shared_ptr<common::memory::GpuDeviceMemory> gpu_guard;

  if (location == MemoryLocation::GPU) {
    if (!gpu_device_id.has_value()) {
      return std::nullopt;
    }
    auto view_or = replica.get_memory_manager().get_gpu_allocation_view();
    if (!view_or.ok() || view_or->base_ptr == nullptr) {
      VLOG(1) << "ComputeViewDataHash: GPU allocation view unavailable";
      return std::nullopt;
    }
    gpu_guard = view_or->allocation;
    mem_view.base_ptr = view_or->base_ptr;
  } else {
    const auto cpu_ptrs = replica.get_memory_manager().get_pointer(MemoryLocation::CPU);
    if (cpu_ptrs.empty() || cpu_ptrs[0] == nullptr) {
      VLOG(1) << "ComputeViewDataHash: CPU memory unavailable for view hash";
      return std::nullopt;
    }
    mem_view.base_ptr = const_cast<void*>(cpu_ptrs[0]);
  }

  auto hash_or = loader::verification::compute_data_multihash(mem_view);
  if (!hash_or.ok()) {
    if (!absl::IsNotFound(hash_or.status())) {
      LOG(WARNING) << "compute_data_multihash (view) failed: " << hash_or.status();
    }
    return std::nullopt;
  }
  (void)gpu_guard;
  return *hash_or;
}

// (hashing utilities moved to core/common/artifact_hash.*)
// GPU eviction helper kept internal to this translation unit.
// ═══════════════════════════════════════════════════════════════════════════
// Construction and Destruction
// ═══════════════════════════════════════════════════════════════════════════

// New unified constructor based on StoreEngineOptions (Phase-3+)
StoreEngine::StoreEngine(const StoreEngineOptions& opts)
    : options_(opts),
      storage_path_(opts.storage_path),
      memory_pool_size_(opts.memory_pool_size),
      artifact_chunk_bytes_(
          opts.artifact_chunk_bytes == 0 ? tensorcast::common::consts::kArtifactChunkDefault
                                         : opts.artifact_chunk_bytes),
      num_thread_(opts.num_thread),
      tx_slice_bytes_(opts.tx_slice_bytes),
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
          gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>(
              std::make_shared<common::memory::PinnedBufferPool>(memory_pool_size_, tx_slice_bytes_))) {
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

  {
    const auto& device_manager_unique = device_manager_.get();
    const auto& replica_registry_unique = replica_registry_.get();
    const auto& metrics_collector_unique = metrics_collector_.get();
    auto* device_manager_ptr = device_manager_unique.get();
    auto* replica_registry_ptr = replica_registry_unique.get();
    auto* metrics_collector_ptr = metrics_collector_unique.get();
    components::RegistrationResources registration_resources{
        .device_manager = gsl::not_null<components::DeviceManager*>{device_manager_ptr},
        .replica_registry = gsl::not_null<components::ReplicaRegistry*>{replica_registry_ptr},
        .metrics_collector = gsl::not_null<components::MetricsCollector*>{metrics_collector_ptr},
        .memory_pool = memory_pool_,
        .communication_manager = comm_manager_,
        .global_store_client = global_store_client_};
    components::ReplicaFactory replica_factory =
        [](const replica::ReplicaConfig& config) -> absl::StatusOr<std::shared_ptr<replica::Replica>> {
      auto create_or = Replica::create(config);
      if (!create_or.ok()) {
        return create_or.status();
      }
      return std::shared_ptr<replica::Replica>(std::move(create_or.value()));
    };
    registration_manager_ = std::make_unique<components::ArtifactRegistrationManager>(
        std::move(registration_resources), std::move(replica_factory), artifact_chunk_bytes_, pinned_memory_timeout_);
  }

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
  if (registration_manager_) {
    registration_manager_->set_global_store_client(global_store_client_);
  }
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

  DeviceKey target_device = target.location.to_device_key();
  const bool target_is_gpu = target_device.type == DeviceType::GPU;
  const common::memory::MemoryLocation target_location =
      target_is_gpu ? common::memory::MemoryLocation::GPU : common::memory::MemoryLocation::CPU;

  int target_device_id = target_device.ordinal;
  if (target_is_gpu && !target_device.uuid.empty()) {
    auto device_result = device_manager_->find_device_by_uuid(target_device.uuid);
    if (!device_result.ok()) {
      return device_result.status();
    }
    target_device_id = device_result.value();
  }

  // Defensive check: ensure GPU ordinal is valid before proceeding.
  if (target_is_gpu) {
    const int num_gpus = device_manager_->get_num_gpus();
    if (target_device_id < 0 || target_device_id >= num_gpus) {
      return absl::InvalidArgumentError(std::string("Invalid GPU device ordinal: ") + std::to_string(target_device_id));
    }
  }

  std::filesystem::path artifact_path;
  if (source.path.is_absolute()) {
    artifact_path = source.path;
  } else if (!storage_path_.empty()) {
    artifact_path = storage_path_ / source.path;
  } else {
    artifact_path = std::filesystem::absolute(source.path);
  }
  artifact_path = artifact_path.lexically_normal();

  loading::DiskSource resolved_source = source;
  resolved_source.path = artifact_path;
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
    return absl::FailedPreconditionError(
        absl::StrCat(
            "Unsupported artifact descriptor schema_version='",
            *descriptor_schema_version,
            "'; canonical index v3 is required"));
  }

  if (!canonical_index_json.has_value()) {
    auto index_info_or = loader::read_from_artifact_dir(artifact_path, target_device_id);
    if (index_info_or.ok()) {
      const loader::IndexInfo& info = *index_info_or;
      canonical_index_json = info.canonical_index_json;
      if (!info.index_multihash.empty()) {
        computed_index_mh = info.index_multihash;
      }
      logical_total_size = std::max<uint64_t>(logical_total_size, info.total_size_bytes);
      is_safetensors = info.is_safetensors;
    } else if (!absl::IsNotFound(index_info_or.status())) {
      LOG(WARNING) << "Failed to resolve canonical index for '" << artifact_path.string()
                   << "': " << index_info_or.status();
    }
  } else {
    auto info_or = loader::canonicalize_from_raw_json(*canonical_index_json, target_device_id);
    if (info_or.ok()) {
      const loader::IndexInfo& info = *info_or;
      canonical_index_json = info.canonical_index_json;
      if (!computed_index_mh.has_value() && !info.index_multihash.empty()) {
        computed_index_mh = info.index_multihash;
      }
      logical_total_size = std::max<uint64_t>(logical_total_size, info.total_size_bytes);
    } else if (!absl::IsNotFound(info_or.status())) {
      LOG(WARNING) << "Failed to canonicalize index JSON: " << info_or.status();
    }
  }

  if (!computed_index_mh.has_value() && existing_index_mh.has_value()) {
    computed_index_mh = existing_index_mh;
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
      .artifact_chunk_bytes = artifact_chunk_bytes_,
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

    auto evict_st = components::evict_for_gpu(
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

  // Compute or obtain index multihash if still missing
  if (!computed_index_mh.has_value() && canonical_index_json.has_value()) {
    auto meta_or = loader::canonicalize_from_raw_json(*canonical_index_json, target_device_id);
    if (meta_or.ok()) {
      if (!meta_or->index_multihash.empty()) {
        computed_index_mh = meta_or->index_multihash;
      }
      logical_total_size = std::max<uint64_t>(logical_total_size, meta_or->total_size_bytes);
    } else if (!absl::IsNotFound(meta_or.status())) {
      LOG(WARNING) << "Failed to recompute canonical index metadata: " << meta_or.status();
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
    uint64_t resolved_size = 0;
    if (auto sz_or = replica->get_artifact_size(); sz_or.ok()) {
      resolved_size = *sz_or;
    } else {
      resolved_size = logical_total_size;
    }

    loader::verification::MemoryView verification_view;
    verification_view.location = target_location;
    verification_view.size_bytes = resolved_size;
    verification_view.gpu_device_id =
        (target_location == MemoryLocation::GPU) ? std::optional<int>(target_device_id) : std::nullopt;
    std::shared_ptr<common::memory::GpuDeviceMemory> metadata_gpu_guard;

    if (target_location == MemoryLocation::GPU && verification_view.size_bytes > 0) {
      auto view_or = replica->get_memory_manager().get_gpu_allocation_view();
      if (view_or.ok()) {
        metadata_gpu_guard = view_or->allocation;
        verification_view.base_ptr = view_or->base_ptr;
      }
    } else if (target_location == MemoryLocation::CPU && verification_view.size_bytes > 0) {
      const auto cpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::CPU);
      if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr) {
        verification_view.base_ptr = const_cast<void*>(cpu_ptrs[0]);
      }
    }

    auto verification_status = loader::verification::reuse_or_generate_verification_json(
        artifact_path, expected_byte_space_id, verification_view);
    if (!verification_status.ok()) {
      return verification_status;
    }
    (void)metadata_gpu_guard;
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
        loader::verification::MemoryView data_view;
        data_view.location = target_location;
        data_view.size_bytes = total;
        data_view.gpu_device_id =
            (target_location == MemoryLocation::GPU) ? std::optional<int>(target_device_id) : std::nullopt;
        std::shared_ptr<common::memory::GpuDeviceMemory> data_gpu_guard;
        if (target_location == MemoryLocation::GPU) {
          auto view_or = replica->get_memory_manager().get_gpu_allocation_view();
          if (view_or.ok()) {
            data_gpu_guard = view_or->allocation;
            data_view.base_ptr = view_or->base_ptr;
          }
        } else {
          const auto cpu_ptrs = replica->get_memory_manager().get_pointer(MemoryLocation::CPU);
          if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr) {
            data_view.base_ptr = const_cast<void*>(cpu_ptrs[0]);
          }
        }
        auto mh_or = loader::verification::compute_data_multihash(data_view);
        if (mh_or.ok()) {
          computed_data_mh = mh_or.value();
        }
        (void)data_gpu_guard;
      }
    }

    if (computed_index_mh.has_value() && computed_data_mh.has_value()) {
      uint64_t total_for_desc = logical_total_size;
      if (total_for_desc == 0) {
        if (auto sz_or = replica->get_artifact_size(); sz_or.ok()) {
          total_for_desc = *sz_or;
        }
      }
      auto desc_status = loader::verification::write_descriptor_if_absent(
          artifact_path, *computed_index_mh, *computed_data_mh, total_for_desc, "json");
      if (!desc_status.ok()) {
        return desc_status;
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
      .artifact_chunk_bytes = artifact_chunk_bytes_,
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

materialization::control::MaterializationDeps StoreEngine::make_materialization_deps() {
  auto* registry_ptr = replica_registry_.get().get();
  ABSL_CHECK_NE(registry_ptr, nullptr);
  materialization::control::MaterializationDeps deps(
      gsl::not_null<components::ReplicaRegistry*>{registry_ptr}, memory_pool_);
  deps.artifact_chunk_bytes = artifact_chunk_bytes_;
  deps.pinned_memory_timeout = pinned_memory_timeout_;
  deps.num_threads = num_thread_;
  deps.ingest_from_disk = [this](
                              const std::string& artifact_identifier,
                              const loading::DiskSource& source,
                              const loading::ReplicaTarget& target,
                              const loading::MaterializeHints& hints) {
    return this->ingest_from_disk_internal(artifact_identifier, source, target, hints);
  };
  deps.compute_view_hash = [](replica::Replica& replica,
                              MemoryLocation location,
                              uint64_t view_size_bytes,
                              std::optional<int> gpu_device_id) {
    return ComputeViewDataHash(replica, location, view_size_bytes, std::move(gpu_device_id));
  };
  if (global_store_client_ && global_store_client_->is_connected()) {
    deps.run_auto = [this](const loading::MaterializationRequest& request) -> absl::StatusOr<loading::ReplicaHandle> {
      materialization::control::MaterializeOrchestrator orchestrator(
          gsl::not_null<materialization::control::MaterializationBackend*>{this},
          gsl::not_null<components::IGlobalStoreClient*>{global_store_client_.get()});
      return orchestrator.run(request.canonical_artifact_id(), request.target_device(), request.hints());
    };
  }
  return deps;
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
  return components::evict_for_cpu(*replica_registry_, *memory_pool_, *metrics_collector_, required_size);
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
  auto request_or = loading::MaterializationRequest::Create(target_device, mode, hints, *device_manager_);
  if (!request_or.ok()) {
    return request_or.status();
  }

  materialization::control::MaterializationService service(make_materialization_deps());
  return service.Execute(request_or.value());
}

absl::StatusOr<loading::ReplicaHandle> StoreEngine::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return ingest_from_p2p_internal(artifact_identifier, source, target, hints);
}

absl::StatusOr<loading::ReplicaHandle> StoreEngine::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return ingest_from_disk_internal(artifact_identifier, source, target, hints);
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
  if (!artifact_id_override.empty() && !absl::StartsWith(artifact_id_override, "mi2:")) {
    return absl::InvalidArgumentError("artifact_id_override must be a canonical mi2: identifier");
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

absl::Status StoreEngine::unregister_replica_from_global_store(std::string_view artifact_id, int device_id) {
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  const std::string wid = worker_id_.empty() ? std::string("local") : worker_id_;
  // Only GPU replicas are relevant for LIP deregistration in this flow.
  return global_store_client_->unregister_replica_by_worker(
      artifact_id, wid, common::memory::MemoryLocation::GPU, static_cast<uint32_t>(device_id));
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
  if (!registration_manager_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_manager_->begin(reg);
}

absl::StatusOr<StoreEngine::RegistrationCommitResult> StoreEngine::commit_registered_artifact(
    std::string_view registration_id) {
  if (!registration_manager_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_manager_->commit(registration_id);
}

absl::Status StoreEngine::ingest_view_registration_chunk(
    std::string_view registration_id,
    uint64_t view_offset,
    absl::Span<const std::byte> data) {
  if (!registration_manager_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_manager_->ingest_view_chunk(registration_id, view_offset, data);
}

absl::StatusOr<uint64_t> StoreEngine::get_view_registration_ingested_bytes(std::string_view registration_id) {
  if (!registration_manager_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_manager_->get_view_ingested_bytes(registration_id);
}

absl::Status StoreEngine::keep_alive_registered_artifact(std::string_view registration_id, uint32_t ttl_ms) {
  if (!registration_manager_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_manager_->keep_alive(registration_id, ttl_ms);
}

absl::Status StoreEngine::abort_registered_artifact(std::string_view registration_id) {
  if (!registration_manager_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_manager_->abort(registration_id);
}

std::vector<replica::ChunkState> StoreEngine::get_chunk_states_telemetry(std::string_view artifact_id) const {
  return get_chunk_states_cpu_uma(artifact_id);
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

absl::StatusOr<loader::ViewPlan> StoreEngine::compute_view_plan(
    std::string_view canonical_index_json,
    const loader::ViewSpec& spec) {
  return loader::ViewPlanner::compute_view_plan(canonical_index_json, spec);
}

bool StoreEngine::view_plan_allows_alias(const loader::ViewPlan& plan) {
  if (plan.selection.total_bytes == 0) {
    return false;
  }
  if (plan.selection.requires_materialization) {
    return false;
  }
  if (plan.transform.requires_materialization || !plan.transform.tensors.empty()) {
    return false;
  }
  return plan.selection.is_contiguous && plan.selection.is_segment_aligned;
}

absl::StatusOr<std::string> StoreEngine::compute_view_data_hash_from_source(
    loader::SeekableSource& base_source,
    const loader::ViewPlan& plan,
    size_t leaf_chunk_bytes) {
  if (plan.selection.total_bytes == 0) {
    return absl::InvalidArgumentError("view plan contains no data to hash");
  }
  if (leaf_chunk_bytes == 0) {
    return absl::InvalidArgumentError("leaf_chunk_bytes must be > 0");
  }
  if (plan.selection.total_bytes > std::numeric_limits<size_t>::max()) {
    return absl::OutOfRangeError("view plan exceeds host memory limits");
  }

  loader::ViewPlanSource view_source(gsl::not_null<loader::SeekableSource*>{&base_source}, plan.selection);
  const size_t total_bytes = static_cast<size_t>(plan.selection.total_bytes);

  if (!plan.transform.requires_materialization && plan.transform.tensors.empty()) {
    return loader::compute_data_multihash_from_seekable_source(
        view_source, plan.selection.total_bytes, leaf_chunk_bytes);
  }

  std::vector<uint8_t> staging(total_bytes);
  auto read_or = view_source.read_at(0, staging.data(), staging.size());
  if (!read_or.ok()) {
    return read_or.status();
  }
  if (*read_or != staging.size()) {
    return absl::DataLossError("short read while materializing view for hashing");
  }

  absl::Status transform_status =
      loader::execute_transform(plan.transform, MemoryLocation::CPU, staging.data(), /*device_id=*/-1);
  if (!transform_status.ok()) {
    return transform_status;
  }

  return loader::compute_data_multihash_from_cpu_memory(
      gsl::not_null<const void*>{staging.data()}, plan.selection.total_bytes, leaf_chunk_bytes);
}

} // namespace tensorcast::store
