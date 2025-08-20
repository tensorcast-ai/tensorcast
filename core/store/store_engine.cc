// Copyright (c) 2025, StepCast Team. All rights reserved.

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
#include "absl/types/span.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/common/model_hash.h"
#include "core/common/trace/trace_macros.h"
#include "core/communicator/misc/common.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/loading/prepare_orchestrator.h"
#include "core/store/model/memory_state.h"
#include "core/store/model/model_config.h"
// RFC-0007 helpers for safetensors canonical index
#include <nlohmann/json.hpp>
#include "core/store/loader/canonical_index.h"
#include "core/store/loader/safetensors_util.h"
// Unified hashing over SeekableSource for CPU/GPU/P2P
#include "core/store/loader/source_hash.h"
// #include "core/store/loading/replica_registration_helper.h"

namespace stepcast::store {
namespace {} // namespace
// (hashing utilities moved to core/common/model_hash.*)
// Forward declaration for GPU eviction helper defined later in this file.
absl::Status try_evict_gpu_memory_impl(
    ModelRegistry& registry,
    DeviceManager& device_manager,
    MetricsCollector& metrics,
    int device_id,
    size_t required_bytes);

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
      device_manager_(gsl::not_null<std::unique_ptr<DeviceManager>>(std::make_unique<DeviceManager>())),
      model_registry_(gsl::not_null<std::unique_ptr<ModelRegistry>>(std::make_unique<ModelRegistry>())),
      metrics_collector_(gsl::not_null<std::unique_ptr<MetricsCollector>>(std::make_unique<MetricsCollector>())),
      memory_pool_(
          gsl::not_null<std::shared_ptr<PinnedMemoryPool>>(
              std::make_shared<PinnedMemoryPool>(memory_pool_size_, chunk_size_))),
      dvmp_(
          gsl::not_null<std::shared_ptr<memory::DistributedVirtualMemoryPool>>(
              std::make_shared<memory::DistributedVirtualMemoryPool>(opts.dvmp_chunk_size))) {
  VLOG(1) << "Initializing StoreEngine with unified Options constructor";
  VLOG(1) << "Storage path: "
          << (storage_path_.empty() ? "<empty - model_identifier will be full path>" : storage_path_.string());
  VLOG(1) << "Memory pool size: " << memory_pool_size_ / communicator::GB << "GB";
  VLOG(1) << "I/O threads: " << num_thread_ << ", chunk size: " << chunk_size_ / communicator::MB << "MB";

  initialize_components();
  initialize_global_store(opts);
  initialize_communication_manager(opts);

  metrics_collector_->update_all_metrics(*memory_pool_, *model_registry_, *device_manager_);
}

void StoreEngine::initialize_components() {
  // Initialize core components
  absl::Status status = device_manager_->initialize();
  CHECK(status.ok()) << "Failed to initialize DeviceManager: " << status.message();
}

void StoreEngine::initialize_global_store(const StoreEngineOptions& opts) {
  // Global Store client (remote coordination).  If a non-empty
  // global_store_address is provided via StoreEngineOptions, attempt to
  // connect immediately so that PrepareOrchestrator can leverage it for remote
  // replica discovery.
  if (!opts.global_store_address.empty()) {
    GlobalStoreClientConfig gs_cfg;
    gs_cfg.global_store_address = opts.global_store_address;

    global_store_client_ = std::make_unique<GlobalStoreClient>(gs_cfg);
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
    comm_manager_ = opts.comm_manager;
  } else {
    LOG(INFO) << "CommunicateEngine is not provided, will disable P2P loading/registration";
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
  metrics_collector_->update_all_metrics(*memory_pool_, *model_registry_, *device_manager_);
}

std::vector<StoreEngine::ModelInfo> StoreEngine::get_all_models_info() const {
  std::vector<ModelInfo> result;

  // Use LRU list to retrieve all known InstanceKeys. This covers every entry
  // in the registry without exposing internal storage.
  const auto instance_keys = model_registry_->get_lru_instances();

  for (const auto& key : instance_keys) {
    auto model_res = model_registry_->find(key);
    if (!model_res.ok()) {
      continue; // Instance may have been removed concurrently.
    }

    const auto& model = model_res.value();

    ModelInfo info;
    info.model_id = key.model_id;

    auto size_result = model->get_model_size();
    info.size_bytes = size_result.ok() ? size_result.value() : 0;

    auto cpu_state = model->get_memory_state(ModelLocation::PAGEABLE_CPU);
    auto gpu_state = model->get_memory_state(ModelLocation::GPU);

    auto is_present = [](MemoryState st) {
      return st == MemoryState::ALLOCATED || st == MemoryState::LOADING || st == MemoryState::LOADED;
    };

    info.cpu_state = is_present(cpu_state) ? ModelLocation::PAGEABLE_CPU : ModelLocation::NONE;
    info.gpu_state = is_present(gpu_state) ? ModelLocation::GPU : ModelLocation::NONE;

    info.gpu_device_id = -1;
    info.gpu_device_uuid.clear();

    if (key.device.type == DeviceType::GPU && is_present(gpu_state)) {
      info.gpu_device_id = key.device.ordinal;
      if (!key.device.uuid.empty()) {
        info.gpu_device_uuid = key.device.uuid;
      } else {
        // Fallback: query via CUDA API if uuid not stored.
        const auto gpu_ptrs = model->get_memory_manager().get_pointer(ModelLocation::GPU);
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

    info.is_registered_for_comm = comm_manager_ && comm_manager_->is_enabled() &&
        (cpu_state == MemoryState::LOADED || gpu_state == MemoryState::LOADED);

    // Precise access / load timestamps are not tracked at this layer after the
    // registry refactor.  We set them to the current time as a placeholder.
    auto now = std::chrono::system_clock::now();
    info.last_access_time = now;
    info.load_time = now;

    result.push_back(info);
  }

  return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Implementation - using new unified types
// ═══════════════════════════════════════════════════════════════════════════

absl::StatusOr<ModelHandle> StoreEngine::load_from_disk_internal(
    const std::string& model_identifier,
    const DiskSource& source,
    const ModelTarget& target,
    const LoadingHints& hints) {
  const auto start_time = std::chrono::steady_clock::now();
  const std::string request_id = absl::StrCat("disk_", absl::ToUnixNanos(absl::Now()));
  SC_TRACE_INIT_GUARD(request_id, model_identifier, "load_from_disk_internal");

  // Convert Location to legacy ModelLocation
  ModelLocation target_location = ModelLocation::PAGEABLE_CPU;
  if (target.location.type == ModelLocation::GPU) {
    target_location = ModelLocation::GPU;
  }

  // Resolve device ID if GPU target
  int target_device_id = target.location.device_id;
  if (target_location == ModelLocation::GPU && !target.location.device_uuid.empty()) {
    auto device_result = device_manager_->find_device_by_uuid(target.location.device_uuid);
    if (!device_result.ok()) {
      return device_result.status();
    }
    target_device_id = device_result.value();
  }

  // Defensive check: ensure GPU ordinal is valid before proceeding.
  if (target_location == ModelLocation::GPU) {
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
  DiskSource resolved_source = source;
  if (!storage_path_.empty() && !source.path.is_absolute()) {
    resolved_source.path = storage_path_ / source.path;
  }

  // Get or create model
  ModelConfig config{
      .source = resolved_source,
      .model_identifier = model_identifier,
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_memory_pool = memory_pool_,
      .dvmp = dvmp_,
      .expected_model_size = std::nullopt};
  config.pinned_memory_timeout = hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : pinned_memory_timeout_;
  config.max_buffer_bytes = hints.max_buffer_bytes;
  if (target_location == ModelLocation::GPU) {
    config.local_device_id = target_device_id;
  }
  config.device_type = (target_location == ModelLocation::GPU) ? DeviceType::GPU : DeviceType::CPU;

  auto model = get_or_create_model(model_identifier, config);
  if (!model) {
    return absl::InternalError("Failed to create model");
  }

  // Start async loading
  std::optional<int> opt_dev;
  if (target_location == ModelLocation::GPU) {
    opt_dev = target_device_id;
  }

  auto load_future = model->ensure_loaded_async(target_location, num_thread_, opt_dev);

  // Wait for allocation
  const auto allocation_timeout = hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : pinned_memory_timeout_;
  // Treat a timeout of 0 (the default used by many tests) as "wait indefinitely" rather than
  // returning immediately.  Mapping to `absl::InfiniteDuration()` avoids spurious
  // `DEADLINE_EXCEEDED` errors during unit-tests that intentionally rely on the default.
  absl::Duration wait_duration =
      (allocation_timeout.count() > 0) ? absl::Milliseconds(allocation_timeout.count()) : absl::InfiniteDuration();

  auto wait_status = model->get_memory_manager().wait_for_state(target_location, MemoryState::LOADED, wait_duration);

  // ------------------------------------------------------------------
  // NEW (Phase 3.2-3): On GPU allocation failure attempt eviction + retry
  // ------------------------------------------------------------------
  if (!wait_status.ok() && target_location == ModelLocation::GPU) {
    // Approximate bytes we need = model size (may be 0 if unknown)
    size_t required_bytes = 0;
    if (auto sz_or = model->get_model_size(); sz_or.ok()) {
      required_bytes = *sz_or;
    }

    LOG(WARNING) << "load_from_disk_internal(): initial GPU allocation failed (" << wait_status
                 << "). Attempting GPU eviction on device " << target_device_id << " for ~" << required_bytes
                 << " bytes.";

    auto evict_st = try_evict_gpu_memory_impl(
        *model_registry_, *device_manager_, *metrics_collector_, target_device_id, required_bytes);

    if (evict_st.ok()) {
      // Reset model GPU memory state then retry loading.
      (void)model->release_memory(ModelLocation::GPU, /*safe_release=*/true);

      // Trigger load again.
      load_future = model->ensure_loaded_async(target_location, num_thread_, opt_dev);

      wait_status = model->get_memory_manager().wait_for_state(target_location, MemoryState::LOADED, wait_duration);

      if (!wait_status.ok()) {
        LOG(WARNING) << "load_from_disk_internal(): Retry after eviction still failed: " << wait_status;
      }
    } else {
      LOG(WARNING) << "load_from_disk_internal(): GPU eviction did not free enough memory: " << evict_st;
    }
  }

  if (!wait_status.ok()) {
    return wait_status;
  }

  // RFC-0007: After loading, compute/verify content-addressed identity when possible.
  // Only perform strong verification when the model is resident in GPU memory (fast path).
  std::optional<std::string> computed_data_mh;
  std::optional<std::string> computed_index_mh;
  std::optional<std::string> existing_index_mh;
  std::optional<std::string> existing_data_mh;
  std::filesystem::path model_dir = resolved_source.path;
  bool is_safetensors = false;
  {
    // Probe directory for .safetensors to differentiate formats
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
      if (entry.is_regular_file()) {
        const auto name = entry.path().filename().string();
        if (name.size() >= 12 && name.rfind(".safetensors") == name.size() - 12) {
          is_safetensors = true;
          break;
        }
      }
    }
  }

  // Try to read descriptor if present
  const auto descriptor_path = model_dir / "model_descriptor.json";
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
        LOG(WARNING) << "Ignoring malformed model_descriptor.json: " << e.what();
      }
    }
  }

  // Compute data multihash from GPU memory when requested by hints
  uint64_t verify_size = 0;
  const bool force_full_digest = options_.force_full_digest_on_load;
  if (hints.verify == LoadingHints::Verify::FULL_DIGEST || force_full_digest) {
    if (auto sz_or = model->get_model_size(); sz_or.ok()) {
      verify_size = *sz_or;
    }
    if (target_location == ModelLocation::GPU) {
      const auto gpu_ptrs = model->get_memory_manager().get_pointer(ModelLocation::GPU);
      if (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr && verify_size > 0) {
        auto data_mh_or = loader::compute_data_multihash_from_gpu_memory(gpu_ptrs[0], verify_size, target_device_id);
        if (data_mh_or.ok()) {
          computed_data_mh = *data_mh_or;
        } else {
          LOG(WARNING) << "Data multihash computation (GPU) failed: " << data_mh_or.status();
        }
      }
    } else if (target_location == ModelLocation::PAGEABLE_CPU) {
      const auto cpu_ptrs = model->get_memory_manager().get_pointer(ModelLocation::PAGEABLE_CPU);
      if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr && verify_size > 0) {
        auto data_mh_or = loader::compute_data_multihash_from_cpu_memory(cpu_ptrs[0], verify_size);
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
      for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        if (entry.is_regular_file()) {
          const auto name = entry.path().filename().string();
          if (name.size() >= 12 && name.rfind(".safetensors") == name.size() - 12) {
            st_files.push_back(entry.path());
          }
        }
      }
      auto index_bytes_or = loader::BuildCanonicalIndexFromSafetensors(st_files);
      if (index_bytes_or.ok()) {
        // compute_index_multihash prefers inline data
        auto index_mh_or = model_hash::compute_index_multihash(std::optional<std::string>(index_bytes_or.value()), "");
        if (index_mh_or.ok()) {
          computed_index_mh = *index_mh_or;
        } else {
          LOG(WARNING) << "Index multihash computation failed: " << index_mh_or.status();
        }
      } else {
        LOG(WARNING) << "Failed to build canonical index from safetensors: " << index_bytes_or.status();
      }
    }
  } else {
    // Standard partition format – read tensor_index.(json|cbor) and canonicalize bytes via nlohmann::json
    const auto index_json_path = model_dir / "tensor_index.json";
    const auto index_cbor_path = model_dir / "tensor_index.cbor";
    try {
      // Read canonical index (CBOR preferred), then rebuild with stable grouping
      std::string raw_json;
      if (std::filesystem::exists(index_cbor_path)) {
        std::ifstream f(index_cbor_path, std::ios::binary);
        std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        nlohmann::json j = nlohmann::json::from_cbor(buf);
        raw_json = j.dump();
      } else if (std::filesystem::exists(index_json_path)) {
        std::ifstream f(index_json_path);
        nlohmann::json j;
        f >> j;
        raw_json = j.dump();
      }
      if (!raw_json.empty()) {
        // Apply stable canonicalization using C++ authority
        auto rebuilt_or = loader::rebuild_stable_canonical_index(raw_json, target_device_id);
        const std::string& canonical_json = rebuilt_or.ok() ? *rebuilt_or : raw_json;
        auto index_mh_or = model_hash::compute_index_multihash(std::optional<std::string>(canonical_json), "");
        if (index_mh_or.ok()) {
          computed_index_mh = *index_mh_or;
        } else {
          LOG(WARNING) << "Index multihash computation failed: " << index_mh_or.status();
        }
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to read/parse tensor_index.(json|cbor): " << e.what();
    }
  }

  // If descriptor exists, verify data_multihash matches when we computed it
  if (existing_data_mh.has_value() && computed_data_mh.has_value()) {
    if (*existing_data_mh != *computed_data_mh) {
      return absl::DataLossError("MODEL_ID_MISMATCH: data_multihash does not match loaded data");
    }
  }

  // If safetensors path lacks descriptor, write it back when we have both hashes
  if (is_safetensors && !std::filesystem::exists(descriptor_path)) {
    if (computed_index_mh.has_value() && computed_data_mh.has_value()) {
      try {
        // 1) Persist model_descriptor.json
        nlohmann::json j;
        j["model_id"] = std::string("mi2:") + *computed_index_mh + ":" + *computed_data_mh;
        j["index_multihash"] = *computed_index_mh;
        j["data_multihash"] = *computed_data_mh;
        j["schema_version"] = "v2";
        j["encoding"] = "json";
        j["total_size"] = verify_size;
        nlohmann::json hp;
        hp["chunk_size"] = 4 * 1024 * 1024;
        hp["fanout"] = 2;
        hp["algorithm"] = "sha2-256";
        j["hash_params"] = hp;
        std::ofstream of(descriptor_path);
        if (!of.is_open()) {
          return absl::PermissionDeniedError("DESCRIPTOR_NOT_WRITABLE: cannot write model_descriptor.json");
        }
        of << j.dump(2);

        // 2) Optionally persist canonical index (CBOR preferred) if not already present
        const auto index_json_path = model_dir / "tensor_index.json";
        const auto index_cbor_path = model_dir / "tensor_index.cbor";
        if (!std::filesystem::exists(index_cbor_path) && !std::filesystem::exists(index_json_path)) {
          // Rebuild canonical index bytes from safetensors headers
          std::vector<std::filesystem::path> st_files;
          for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
            if (entry.is_regular_file()) {
              const auto name = entry.path().filename().string();
              if (name.size() >= 12 && name.rfind(".safetensors") == name.size() - 12) {
                st_files.push_back(entry.path());
              }
            }
          }
          auto index_bytes_or = loader::BuildCanonicalIndexFromSafetensors(st_files);
          if (index_bytes_or.ok()) {
            // Parse to JSON then write CBOR bytes
            nlohmann::json idx_json = nlohmann::json::parse(index_bytes_or.value(), nullptr, true);
            std::vector<std::uint8_t> cbor = nlohmann::json::to_cbor(idx_json);
            std::ofstream oc(index_cbor_path, std::ios::binary);
            if (!oc.is_open()) {
              return absl::PermissionDeniedError("DESCRIPTOR_NOT_WRITABLE: cannot write tensor_index.cbor");
            }
            oc.write(reinterpret_cast<const char*>(cbor.data()), static_cast<std::streamsize>(cbor.size()));
            oc.close();
          }
        }
      } catch (const std::exception& e) {
        return absl::PermissionDeniedError(std::string("DESCRIPTOR_NOT_WRITABLE: ") + e.what());
      }
    }
  }

  // Build result using new ModelHandle structure
  ModelHandle handle;

  // Compose InstanceKey
  DeviceKey dev_key;
  if (target_location == ModelLocation::GPU) {
    dev_key = DeviceKey{.type = DeviceType::GPU, .ordinal = target_device_id, .uuid = ""};
  } else {
    dev_key = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  }
  handle.instance_key = InstanceKey{.model_id = model_identifier, .device = dev_key, /*replica=*/.replica = 0};

  // Loading future and states
  handle.ready_future = load_future;
  handle.cpu_state = model->get_memory_state(ModelLocation::PAGEABLE_CPU);
  handle.gpu_state = model->get_memory_state(ModelLocation::GPU);

  if (target_location == ModelLocation::GPU) {
    const auto gpu_ptrs = model->get_memory_manager().get_pointer(ModelLocation::GPU);
    handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

    // Attempt to obtain CUDA IPC handle bytes for the allocated GPU buffer.
    auto ipc_or = model->get_memory_manager().get_cuda_ipc_handle();
    if (ipc_or.ok()) {
      std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
    }
  }

  // Update metrics
  const auto duration = std::chrono::steady_clock::now() - start_time;
  metrics_collector_->record_operation("load_from_disk", std::chrono::duration<double>(duration).count());
  metrics_collector_->update_all_metrics(*memory_pool_, *model_registry_, *device_manager_);

  return handle;
}

absl::StatusOr<ModelHandle> StoreEngine::load_from_p2p_internal(
    const std::string& model_identifier,
    const P2PSource& source,
    const ModelTarget& target,
    const LoadingHints& hints) {
  const auto start_time = std::chrono::steady_clock::now();
  const std::string request_id = absl::StrCat("p2p_", absl::ToUnixNanos(absl::Now()));
  SC_TRACE_INIT_GUARD(request_id, model_identifier, "load_from_p2p_internal");

  if (!comm_manager_ || !comm_manager_->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }

  ModelLocation target_location = ModelLocation::PAGEABLE_CPU;
  if (target.location.type == ModelLocation::GPU) {
    target_location = ModelLocation::GPU;
  }

  // Create model with P2P source
  auto p2p_source = source;
  p2p_source.comm_engine = comm_manager_->get_shared_engine();
  ModelConfig config{
      .source = p2p_source,
      .model_identifier = model_identifier,
      .device_type = (target_location == ModelLocation::GPU ? DeviceType::GPU : DeviceType::CPU),
      .local_device_id = target.location.device_id,
      .pinned_memory_pool = memory_pool_,
      .dvmp = dvmp_,
      .expected_model_size = std::nullopt};
  config.pinned_memory_timeout = hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : pinned_memory_timeout_;
  config.local_device_id = target.location.device_id;
  config.max_buffer_bytes = hints.max_buffer_bytes;
  config.p2p_comm_enabled = true;
  config.device_type = (target_location == ModelLocation::GPU) ? DeviceType::GPU : DeviceType::CPU;

  auto model = get_or_create_model(model_identifier, config);
  if (!model) {
    return absl::InternalError("Failed to create model");
  }

  // Load synchronously for remote (maintain existing behavior)
  auto load_future =
      model->ensure_loaded_async(target_location, num_thread_, std::optional<int>(target.location.device_id));
  auto status = load_future.get();

  if (!status.ok()) {
    if (absl::IsResourceExhausted(status)) {
      // Try to evict memory
      LOG(WARNING) << "Resource exhausted, attempting memory eviction";
      auto evict_status = try_evict_memory_for_model(source.size_bytes);
      if (evict_status.ok()) {
        // Retry
        load_future =
            model->ensure_loaded_async(target_location, num_thread_, std::optional<int>(target.location.device_id));
        status = load_future.get();
      }
    }

    if (!status.ok()) {
      metrics_collector_->record_p2p_transfer(0, false);
      return status;
    }
  }

  // Build result using new ModelHandle structure
  ModelHandle handle;

  DeviceKey dev_key;
  if (target_location == ModelLocation::GPU) {
    dev_key = DeviceKey{.type = DeviceType::GPU, .ordinal = target.location.device_id, .uuid = ""};
  } else {
    dev_key = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  }
  handle.instance_key = InstanceKey{.model_id = model_identifier, .device = dev_key, /*replica=*/.replica = 0};

  // Ready future is already resolved (synchronous path)
  std::promise<absl::Status> promise;
  promise.set_value(absl::OkStatus());
  handle.ready_future = promise.get_future().share();

  handle.cpu_state = model->get_memory_state(ModelLocation::PAGEABLE_CPU);
  handle.gpu_state = model->get_memory_state(ModelLocation::GPU);

  if (target_location == ModelLocation::GPU) {
    const auto gpu_ptrs = model->get_memory_manager().get_pointer(ModelLocation::GPU);
    handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

    auto ipc_or = model->get_memory_manager().get_cuda_ipc_handle();
    if (ipc_or.ok()) {
      std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
    }
  }

  // Update metrics
  metrics_collector_->record_p2p_transfer(source.size_bytes, true);
  const auto duration = std::chrono::steady_clock::now() - start_time;
  metrics_collector_->record_operation("load_from_p2p", std::chrono::duration<double>(duration).count());
  metrics_collector_->update_all_metrics(*memory_pool_, *model_registry_, *device_manager_);

  return handle;
}

absl::StatusOr<ModelHandle> StoreEngine::load_from_buffer_internal(
    const std::string& /*model_identifier*/,
    const InlineBufferSource& /*source*/,
    const ModelTarget& /*target*/,
    const LoadingHints& /*hints*/) {
  // InlineBufferSource is a newly added type, temporarily returning unimplemented error
  // Future: implement direct model loading from memory buffer
  return absl::UnimplementedError("InlineBufferSource loading not yet implemented");
}

std::shared_ptr<Model> StoreEngine::get_or_create_model(
    const std::string& model_identifier,
    const ModelConfig& config) {
  // Build InstanceKey for the requested device (CPU when local_device_id < 0)
  DeviceKey dev_key;
  if (config.device_type == DeviceType::GPU) {
    dev_key = DeviceKey{
        .type = DeviceType::GPU, .ordinal = (config.local_device_id >= 0 ? config.local_device_id : 0), .uuid = ""};
  } else {
    dev_key = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  }
  InstanceKey inst_key{.model_id = model_identifier, .device = dev_key, /*replica=*/.replica = 0};

  // Fast-path: already present in registry
  if (auto existing_or = model_registry_->find(inst_key); existing_or.ok()) {
    return existing_or.value();
  }

  // Create new model instance
  auto model_status = Model::create(config);
  if (!model_status.ok()) {
    LOG(ERROR) << "Failed to create model: " << model_status.status().message();
    return nullptr;
  }
  auto model = std::shared_ptr<Model>(std::move(model_status.value()));

  // Register in multi-device registry (best effort)
  absl::Status emplace_status = model_registry_->emplace(inst_key, model);

  if (absl::IsAlreadyExists(emplace_status)) {
    // Another thread inserted the instance concurrently. Reuse the existing
    // entry to avoid duplicate model objects and double-loading.
    if (auto existing_or = model_registry_->find(inst_key); existing_or.ok()) {
      return existing_or.value();
    }
    // Fall through on error – treat as internal failure.
  } else if (!emplace_status.ok()) {
    // Unexpected error while registering – propagate as failure.
    LOG(ERROR) << "Failed to register model: " << emplace_status.message();
    return nullptr;
  }

  return model;
}

absl::Status StoreEngine::try_evict_memory_for_model(size_t required_size) {
  // Prefer the new multi-device LRU ordering.
  auto lru_instances = model_registry_->get_lru_instances();

  for (const auto& inst_key : lru_instances) {
    auto model_result = model_registry_->find(inst_key);
    if (!model_result.ok()) {
      continue;
    }

    const auto& model = model_result.value();

    // Only attempt to free CPU memory for now – GPU eviction will be handled
    // in a future iteration.
    auto free_status = model->release_memory(ModelLocation::PAGEABLE_CPU, /*safe_release=*/true);
    if (free_status.ok()) {
      metrics_collector_->record_memory_eviction();
      LOG(INFO) << "Evicted model " << inst_key.model_id
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

// ------------ ModelHandle helpers -------------------------

MemoryState ModelHandle::state(DeviceType type) const {
  return type == DeviceType::CPU ? cpu_state : gpu_state;
}

absl::Status ModelHandle::wait_ready(std::chrono::milliseconds timeout) {
  if (!ready_future.valid()) {
    // Nothing to wait for – treat as OK.
    return absl::OkStatus();
  }
  const auto status = ready_future.wait_for(timeout);
  if (status == std::future_status::timeout) {
    return absl::DeadlineExceededError("Timeout while waiting for model to become ready");
  }
  // Future is ready – propagate underlying status value.
  return ready_future.get();
}

// ---------------------------------------------------------------------------
// Bridge implementation of prepare() that internally maps to the legacy load()
// interface.  The new implementation first checks if a model instance already
// exists on the requested device.  If not, it tries COPY_ONLY (GPU peer copy),
// LOAD_ONLY (disk), or AUTO (orchestrator → disk) according to the requested
// mode.  All duplicate fallback branches and GPU-eviction heuristics have been
// removed to keep the codebase lean – PrepareOrchestrator now owns almost all
// decision complexity.
// ---------------------------------------------------------------------------
absl::StatusOr<ModelHandle> StoreEngine::prepare(
    const DeviceKey& target_device,
    PrepareMode mode,
    const LoadingHints& hints) {
  // ────────────────────────────────────────────────────────────────────
  // Validate target device early to avoid entering CUDA paths with
  // invalid ordinals or unsupported device types.
  // ────────────────────────────────────────────────────────────────────
  if (target_device.type == DeviceType::CPU) {
    return absl::InvalidArgumentError("CPU target device is not supported by prepare()");
  }
  if (target_device.type == DeviceType::GPU) {
    const int num_gpus = device_manager_->get_num_gpus();
    if (target_device.ordinal < 0 || target_device.ordinal >= num_gpus) {
      return absl::InvalidArgumentError(
          std::string("Invalid GPU device ordinal: ") + std::to_string(target_device.ordinal));
    }
  } else {
    // For REMOTE/NONE/DISK etc. reject in this implementation.
    return absl::InvalidArgumentError("Unsupported target device type for prepare()");
  }

  // ────────────────────────────────────────────────────────────────────
  // Fast-path: instance already present on the requested device.
  // ────────────────────────────────────────────────────────────────────
  const InstanceKey dst_key{.model_id = hints.model_id, .device = target_device, /*replica=*/.replica = 0};
  if (auto existing_or = model_registry_->find(dst_key); existing_or.ok()) {
    const auto& model = existing_or.value();

    ModelLocation dst_loc = (target_device.type == DeviceType::GPU) ? ModelLocation::GPU : ModelLocation::PAGEABLE_CPU;
    std::optional<int> opt_dev;
    if (dst_loc == ModelLocation::GPU) {
      opt_dev = target_device.ordinal;
    }

    auto fut = model->ensure_loaded_async(dst_loc, num_thread_, opt_dev);

    ModelHandle handle;
    handle.instance_key = dst_key;
    handle.ready_future = fut;
    handle.cpu_state = model->get_memory_state(ModelLocation::PAGEABLE_CPU);
    handle.gpu_state = model->get_memory_state(ModelLocation::GPU);

    if (dst_loc == ModelLocation::GPU) {
      const auto gpu_ptrs = model->get_data_pointer(ModelLocation::GPU);
      handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

      auto ipc_or = model->get_memory_manager().get_cuda_ipc_handle();
      if (ipc_or.ok()) {
        std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
      }
    }
    return handle;
  }

  // Helper lambda: minimal disk-loading path.
  auto load_from_disk = [&](const DeviceKey& dev_key) -> absl::StatusOr<ModelHandle> {
    // Guard: content-addressed IDs (mi2:...) are not paths.
    if (absl::StartsWith(hints.model_id, "mi2:")) {
      return absl::FailedPreconditionError(
          "LOAD_ONLY/disk fallback disabled for content-addressed model_id; Global Store routing required");
    }
    DiskSource disk_src;
    if (!hints.disk_path.empty()) {
      disk_src.path = std::filesystem::path(hints.disk_path);
    } else {
      disk_src.path = std::filesystem::path(hints.model_id);
    }

    ModelTarget target;
    target.location.type = (dev_key.type == DeviceType::GPU) ? ModelLocation::GPU : ModelLocation::PAGEABLE_CPU;
    target.location.device_id = dev_key.ordinal;

    return load_from_disk_internal(hints.disk_path.empty() ? hints.model_id : hints.disk_path, disk_src, target, hints);
  };

  // ────────────────────────────────────────────────────────────────────
  // Mode-specific handling
  // ────────────────────────────────────────────────────────────────────
  switch (mode) {
    case PrepareMode::COPY_ONLY: {
      // COPY_ONLY is only meaningful for GPU targets – perform a local GPU→GPU copy.
      if (target_device.type != DeviceType::GPU) {
        return absl::InvalidArgumentError("COPY_ONLY mode requires a GPU target device");
      }

      const auto candidates = model_registry_->find_by_model(hints.model_id);
      for (const auto& cand_key : candidates) {
        if (cand_key.device.type != DeviceType::GPU) {
          continue;
        }

        auto src_or = model_registry_->find(cand_key);
        if (!src_or.ok()) {
          continue;
        }
        const auto& src_model = src_or.value();
        if (src_model->get_memory_state(ModelLocation::GPU) != MemoryState::LOADED) {
          continue;
        }

        // Create destination model configuration using an inline buffer source to
        // avoid any dependency on on-disk paths when performing GPU→GPU copy.
        // The inline buffer loader requires a known total size.
        uint64_t expected_size = 0;
        if (auto sz_or = src_model->get_model_size(); sz_or.ok()) {
          expected_size = *sz_or;
        } else {
          return sz_or.status();
        }
        InlineBufferSource ib_source{.data = nullptr, .size_bytes = expected_size};
        ModelConfig cfg{
            .source = ib_source,
            .model_identifier = hints.model_id,
            .device_type = DeviceType::GPU,
            .local_device_id = target_device.ordinal,
            .pinned_memory_pool = memory_pool_,
            .dvmp = dvmp_,
            .expected_model_size = expected_size};
        cfg.pinned_memory_timeout = pinned_memory_timeout_;

        auto dst_or = Model::create(cfg);
        if (!dst_or.ok()) {
          return dst_or.status();
        }
        auto dst_model = std::shared_ptr<Model>(std::move(dst_or.value()));
        (void)model_registry_->emplace(dst_key, dst_model);

        absl::Status copy_st = dst_model->copy_from(*src_model);

        std::promise<absl::Status> p;
        p.set_value(copy_st);

        ModelHandle handle;
        handle.instance_key = dst_key;
        handle.ready_future = p.get_future().share();
        handle.cpu_state = dst_model->get_memory_state(ModelLocation::PAGEABLE_CPU);
        handle.gpu_state = dst_model->get_memory_state(ModelLocation::GPU);
        if (handle.gpu_state == MemoryState::LOADED) {
          const auto gpu_ptrs = dst_model->get_data_pointer(ModelLocation::GPU);
          handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

          auto ipc_or = dst_model->get_memory_manager().get_cuda_ipc_handle();
          if (ipc_or.ok()) {
            std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
          }
        }
        return handle;
      }
      return absl::FailedPreconditionError("No suitable source instance for COPY_ONLY mode");
    }

    case PrepareMode::LOAD_ONLY: {
      return load_from_disk(target_device);
    }

    case PrepareMode::AUTO: {
      if (global_store_client_ && global_store_client_->is_connected() && !hints.model_id.empty()) {
        PrepareOrchestrator orchestrator(this, global_store_client_.get());
        auto orchestrated_or = orchestrator.run(hints.model_id, target_device, hints);
        if (orchestrated_or.ok()) {
          return *orchestrated_or;
        }
        LOG(WARNING) << "PrepareOrchestrator failed: " << orchestrated_or.status() << "; falling back to disk load";
      }
      return absl::FailedPreconditionError(
          "AUTO prepare requires a content-addressed hints.model_id with Global Store routing or an explicit hints.disk_path");
    }
  }

  // Should be unreachable.
  return absl::InternalError("Invalid PrepareMode");
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------
std::vector<DeviceKey> StoreEngine::get_loaded_devices(std::string_view model_id) const {
  // Implementation: leverage the modern multi-device registry exclusively. Models loaded via
  // ModelRegistry::emplace are visible to this helper; no backward-compatibility fallbacks remain.
  absl::flat_hash_set<DeviceKey, DeviceKeyHash> unique_devices;
  std::vector<DeviceKey> devices;

  // ──────────────────────────────────────────────────────────────────
  // 1. Multi-device path – gather all instances whose model_id matches
  // ──────────────────────────────────────────────────────────────────
  const auto instance_keys = model_registry_->find_by_model(model_id);
  for (const auto& key : instance_keys) {
    auto model_res = model_registry_->find(key);
    if (!model_res.ok()) {
      continue;
    }
    const auto& model = model_res.value();

    auto is_present = [](MemoryState st) {
      return st == MemoryState::ALLOCATED || st == MemoryState::LOADING || st == MemoryState::LOADED;
    };

    if (key.device.type == DeviceType::CPU) {
      if (is_present(model->get_memory_state(ModelLocation::PAGEABLE_CPU))) {
        unique_devices.insert(DeviceKey{DeviceType::CPU, -1, ""});
      }
    } else if (key.device.type == DeviceType::GPU) {
      if (is_present(model->get_memory_state(ModelLocation::GPU))) {
        unique_devices.insert(DeviceKey{DeviceType::GPU, key.device.ordinal, key.device.uuid});
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

std::vector<InstanceKey> StoreEngine::list_device_models(const DeviceKey& device) const {
  // Implementation that relies solely on the new multi-index registry (InstanceKey-based). No
  // legacy fallback remains.
  std::vector<InstanceKey> list;

  // ------------------------------------------------------------------
  // 1. Multi-device registry path
  // ------------------------------------------------------------------
  const auto inst_keys = model_registry_->find_by_device(device);
  for (const auto& key : inst_keys) {
    auto model_res = model_registry_->find(key);
    if (!model_res.ok()) {
      continue;
    }
    const auto& model = model_res.value();

    auto is_present = [](MemoryState st) {
      return st == MemoryState::ALLOCATED || st == MemoryState::LOADING || st == MemoryState::LOADED;
    };

    if (device.type == DeviceType::CPU) {
      if (is_present(model->get_memory_state(ModelLocation::PAGEABLE_CPU))) {
        list.push_back(key);
      }
    } else if (device.type == DeviceType::GPU) {
      if (is_present(model->get_memory_state(ModelLocation::GPU))) {
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

// NOTE: This helper is intentionally kept internal to this translation unit;
// once the call-sites migrate we can promote it to the public header.
absl::Status try_evict_gpu_memory_impl(
    ModelRegistry& registry,
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

    auto model_res = registry.find(key);
    if (!model_res.ok()) {
      continue;
    }
    const auto& model = model_res.value();

    if (model->get_memory_state(ModelLocation::GPU) != MemoryState::LOADED) {
      continue; // Nothing to free.
    }

    // Attempt to release GPU memory (safe mode to avoid mid-transfer memory).
    auto st = model->release_memory(ModelLocation::GPU, /*safe_release=*/true);
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
// New InstanceKey-centric API wrappers
// ═══════════════════════════════════════════════════════════════════════════

int StoreEngine::wait_instance_ready(const InstanceKey& key) {
  auto model_res = model_registry_->find(key);
  if (!model_res.ok()) {
    return 1; // Not found
  }
  const auto& model = model_res.value();
  ModelLocation loc = (key.device.type == DeviceType::CPU) ? ModelLocation::PAGEABLE_CPU : ModelLocation::GPU;
  absl::Status st = model->wait_until_loaded(loc, absl::InfiniteDuration());
  return st.ok() ? 0 : 1;
}

int StoreEngine::unload_instance(const InstanceKey& key) {
  auto model_res = model_registry_->find(key);
  if (!model_res.ok()) {
    return 1; // Instance not found.
  }

  const auto& model = model_res.value();
  ModelLocation loc = (key.device.type == DeviceType::CPU) ? ModelLocation::PAGEABLE_CPU : ModelLocation::GPU;

  // Inspect current state *before* attempting the release so we can tell if
  // there was anything to unload.  This avoids treating a no-op release as a
  // success – a scenario that would allow multiple threads to report success
  // when only the first one actually freed memory.
  stepcast::store::MemoryState before_state = model->get_memory_state(loc);

  if (before_state <= MemoryState::UNALLOCATED) {
    // Nothing to release – another thread has already unloaded this instance.
    return -1;
  }

  absl::Status st = model->release_memory(loc);
  return st.ok() ? 0 : -1;
}

MemoryState StoreEngine::get_instance_state(const InstanceKey& key, DeviceType memory_type) const {
  auto model_res = model_registry_->find(key);
  if (!model_res.ok()) {
    return MemoryState::UNINITIALIZED;
  }
  ModelLocation loc = (memory_type == DeviceType::CPU) ? ModelLocation::PAGEABLE_CPU : ModelLocation::GPU;
  return model_res.value()->get_memory_state(loc);
}

absl::StatusOr<uint64_t> StoreEngine::get_instance_gpu_ptr(const InstanceKey& key) {
  if (key.device.type != DeviceType::GPU) {
    return absl::InvalidArgumentError("InstanceKey does not reference a GPU device");
  }
  auto model_res = model_registry_->find(key);
  if (!model_res.ok()) {
    return absl::NotFoundError("Model instance not found");
  }
  const auto ptrs = model_res.value()->get_memory_manager().get_pointer(ModelLocation::GPU);
  if (ptrs.empty() || ptrs[0] == nullptr) {
    return absl::FailedPreconditionError("GPU memory not available");
  }
  return reinterpret_cast<uint64_t>(ptrs[0]);
}

absl::StatusOr<uint64_t> StoreEngine::get_instance_size(const InstanceKey& key) {
  auto model_res = model_registry_->find(key);
  if (!model_res.ok()) {
    return absl::NotFoundError("Model instance not found");
  }
  auto size_or = model_res.value()->get_model_size();
  if (!size_or.ok()) {
    return size_or.status();
  }
  return *size_or;
}

absl::StatusOr<CommRegistrationInfo> StoreEngine::enable_remote_instance_access(
    const InstanceKey& key,
    ModelLocation location) {
  if (!comm_manager_ || !comm_manager_->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }
  auto model_res = model_registry_->find(key);
  if (!model_res.ok()) {
    return absl::NotFoundError("Model instance not found");
  }
  // Delegates to Model which uses chunk-scoped export APIs under the hood.
  return model_res.value()->enable_remote_memory_access(location, comm_manager_->get_engine());
}

absl::Status StoreEngine::disable_remote_instance_access(const InstanceKey& key, ModelLocation location) {
  if (!comm_manager_ || !comm_manager_->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }
  auto model_res = model_registry_->find(key);
  if (!model_res.ok()) {
    return absl::NotFoundError("Model instance not found");
  }
  // Delegates to Model which uses chunk-scoped unexport APIs under the hood.
  return model_res.value()->disable_remote_memory_access(location, comm_manager_->get_engine());
}

// ═══════════════════════════════════════════════════════════════════════════
// Memory cleanup
// ═══════════════════════════════════════════════════════════════════════════

int StoreEngine::clear_mem() {
  auto models = model_registry_->clear_all();
  std::vector<absl::Status> errors;

  for (const auto& [inst_key, model] : models) {
    // Release CPU memory with proper error tracking
    auto cpu_status = model->release_memory(ModelLocation::PAGEABLE_CPU);
    if (!cpu_status.ok()) {
      LOG(WARNING) << "Failed to release CPU memory for " << inst_key << ": " << cpu_status.message();
      errors.push_back(cpu_status);
    }

    // Release GPU memory, ignoring NotFound errors (expected when no GPU memory allocated)
    auto gpu_status = model->release_memory(ModelLocation::GPU);
    if (!gpu_status.ok() && !absl::IsNotFound(gpu_status)) {
      LOG(WARNING) << "Failed to release GPU memory for " << inst_key << ": " << gpu_status.message();
      errors.push_back(gpu_status);
    }
  }

  // Update metrics even if some releases failed
  metrics_collector_->update_all_metrics(*memory_pool_, *model_registry_, *device_manager_);

  // Log aggregated error summary if failures occurred
  if (!errors.empty()) {
    LOG(ERROR) << "Failed to release memory for " << errors.size() << " model(s) during shutdown";
    return -1;
  }

  return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// Distributed Memory Pool (DVMP) chunk locking API
// ═══════════════════════════════════════════════════════════════════════════

absl::Status StoreEngine::lock_chunks(const InstanceKey& instance_key, absl::Span<const uint32_t> chunk_indices) {
  SC_TRACE_SCOPE("StoreEngine::lock_chunks");

  // Locate the exact model instance based on InstanceKey (device-specific).
  auto model_res = model_registry_->find(instance_key);
  if (!model_res.ok()) {
    return absl::NotFoundError(
        absl::StrCat("Model instance not found: ", instance_key.model_id, " @ ", instance_key.device.to_string()));
  }

  const auto& model = model_res.value();

  // Get the DVMP instance via MemoryManager.
  const auto dvmp = model->get_memory_manager().get_dvmp();
  // Forward the call to DVMP – DVMP is still keyed by model_id (shared CPU VA).
  return dvmp->lock_chunks(instance_key.model_id, chunk_indices);
}

absl::Status StoreEngine::unlock_chunks(
    const InstanceKey& instance_key,
    absl::Span<const uint32_t> chunk_indices,
    bool copied_gpu) {
  SC_TRACE_SCOPE("StoreEngine::unlock_chunks");

  // Locate the exact model instance.
  auto model_res = model_registry_->find(instance_key);
  if (!model_res.ok()) {
    return absl::NotFoundError(
        absl::StrCat("Model instance not found: ", instance_key.model_id, " @ ", instance_key.device.to_string()));
  }

  const auto& model = model_res.value();

  // Get the DVMP instance via MemoryManager.
  const auto dvmp = model->get_memory_manager().get_dvmp();

  // Forward the call to DVMP.
  return dvmp->unlock_chunks(instance_key.model_id, chunk_indices, copied_gpu);
}

// ═══════════════════════════════════════════════════════════════════════════
// RFC-0006 – Memory TensorDict Registration (coalesced)
// ═══════════════════════════════════════════════════════════════════════════

absl::StatusOr<StoreEngine::RegistrationBeginResult> StoreEngine::begin_register_tensor_dict(
    const TensorDictRegistration& reg) {
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

  // Prepare a memory-only Model instance bound to target GPU to own the allocation.
  // Use InlineBufferSource with the known total size so Model::create() can
  // construct MemoryManager without requiring any on-disk layout.
  InlineBufferSource ib_source{.data = nullptr, .size_bytes = reg.total_size_bytes};
  ModelConfig cfg{
      .source = ib_source,
      .model_identifier = reg.model_id,
      .device_type = DeviceType::GPU,
      .local_device_id = reg.device_id,
      .pinned_memory_pool = memory_pool_,
      .dvmp = dvmp_,
      .expected_model_size = reg.total_size_bytes};
  cfg.pinned_memory_timeout = pinned_memory_timeout_;

  // Check memory pressure before allocation to avoid unexpected evictions
  // This helps prevent large registrations from causing issues with existing models
  {
    auto free_or = device_manager_->get_free_memory(reg.device_id);
    if (free_or.ok()) {
      size_t free_bytes = free_or.value();
      if (reg.total_size_bytes > free_bytes) {
        // Try to evict unused GPU memory first on the specific device
        auto evict_status = try_evict_gpu_memory_impl(
            *model_registry_, *device_manager_, *metrics_collector_, reg.device_id, reg.total_size_bytes - free_bytes);
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

  // Try to create the model with DiskSource first. If the loader fails due to
  // missing directory or missing partitions, fall back to a memory-only path
  // that uses InlineBufferSource to construct a size-known Model and allocate
  // GPU memory without touching disk.
  auto model_or = Model::create(cfg);
  if (!model_or.ok()) {
    return model_or.status();
  }
  auto model = std::shared_ptr<Model>(std::move(model_or.value()));

  // Allocate GPU memory only. No loading/copying here.
  absl::Status st = model->get_memory_manager().allocate_memory(ModelLocation::GPU);
  if (!st.ok()) {
    return st;
  }

  // Obtain CUDA IPC handle to return to caller.
  auto ipc_or = model->get_memory_manager().get_cuda_ipc_handle();
  if (!ipc_or.ok()) {
    return ipc_or.status();
  }

  const auto gpu_ptrs = model->get_memory_manager().get_pointer(ModelLocation::GPU);
  void* base_ptr = (!gpu_ptrs.empty() ? gpu_ptrs[0] : nullptr);

  // Emplace into registry to ensure lifecycle is tracked (InstanceKey via config).
  DeviceKey dev_key{.type = DeviceType::GPU, .ordinal = reg.device_id, .uuid = ""};
  InstanceKey inst_key{.model_id = reg.model_id, .device = dev_key, /*replica=*/.replica = 0};
  (void)model_registry_->emplace(inst_key, model);

  // Create pending entry with cryptographically secure random ID
  // Use a combination of timestamp, process ID, and random bytes for uniqueness
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;
  auto reg_id = absl::StrCat("reg_", absl::ToUnixNanos(absl::Now()), "_", getpid(), "_", dis(gen));

  PendingRegistrationEntry entry;
  entry.registration_id = reg_id;
  entry.model_id = reg.model_id;
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
  entry.model = model;
  entry.gpu_ptr = base_ptr;
  entry.ipc_handle = *ipc_or;

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

absl::StatusOr<StoreEngine::RegistrationCommitResult> StoreEngine::commit_registered_tensor_dict(
    std::string_view registration_id) {
  PendingRegistrationEntry entry;
  std::shared_ptr<Model> model_to_free;
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
      model_to_free = it->second.model;
      pending_regs_.erase(it);
      // Release outside the lock to avoid holding mutex during potentially slow operation
    } else {
      entry = it->second; // make a copy; we may erase after successful commit
    }
  }

  // Release memory outside the critical section if TTL expired
  if (model_to_free) {
    (void)model_to_free->release_memory(ModelLocation::GPU, /*safe_release=*/true);
    return absl::DeadlineExceededError("registration expired (TTL)");
  }

  // Compute content-addressed model_id per RFC-0007: "mi2:<index_multihash>:<data_multihash>"
  // 1) index_multihash from canonical index bytes when provided, otherwise from key (sha256 hex)
  absl::StatusOr<std::string> index_mh_or =
      model_hash::compute_index_multihash(entry.tensor_index_data, entry.tensor_index_key);
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }

  // 2) data_multihash from GPU memory tree-hash root (sha2-256 leaves, base32 multibase)
  void* gpu_ptr = nullptr;
  {
    const auto ptrs = entry.model->get_memory_manager().get_pointer(ModelLocation::GPU);
    gpu_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
  }
  absl::StatusOr<std::string> data_mh_or =
      model_hash::compute_data_multihash_from_gpu(gpu_ptr, entry.size_bytes, entry.device_id);
  if (!data_mh_or.ok()) {
    return data_mh_or.status();
  }

  const std::string model_id_mi2 = absl::StrCat("mi2:", *index_mh_or, ":", *data_mh_or);

  // Replace entry.model_id with the content-addressed ID for registration and return
  entry.model_id = model_id_mi2;

  // Also add a registry mapping for the content-addressed model_id so callers can
  // subsequently reference the instance by its mi2: identifier (in addition to the
  // original logical model_id used during Begin/Commit).
  {
    DeviceKey dev_key{.type = DeviceType::GPU, .ordinal = entry.device_id, .uuid = ""};
    InstanceKey mi2_key{.model_id = entry.model_id, .device = dev_key, /*replica=*/.replica = 0};
    (void)model_registry_->emplace(mi2_key, entry.model);
  }

  // Export remote memory keys if communication is enabled (GPU location).
  std::vector<std::string> remote_keys;
  std::vector<uint64_t> buffer_sizes;
  if (entry.enable_p2p && comm_manager_ && comm_manager_->is_enabled()) {
    auto reg_info_or = entry.model->enable_remote_memory_access(ModelLocation::GPU, comm_manager_->get_engine());
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
    DeviceKey device{.type = DeviceType::GPU, .ordinal = entry.device_id, .uuid = ""};
    auto reg_or = global_store_client_->register_memory_replica(
        entry.model_id,
        /*worker_id=*/"local",
        device,
        entry.size_bytes,
        entry.tensor_index_key,
        remote_keys,
        buffer_sizes,
        entry.tensor_index_data,
        entry.encoding,
        entry.schema_version,
        /*max_concurrency=*/1);
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
  result.model_id = entry.model_id;
  result.device_id = entry.device_id;
  result.size_bytes = entry.size_bytes;
  // RFC-0007: Fill descriptor fields for upstream layers
  result.index_multihash = *index_mh_or;
  result.data_multihash = *data_mh_or;
  result.schema_version = entry.schema_version;
  result.encoding = entry.encoding;
  return result;
}

absl::Status StoreEngine::abort_registered_tensor_dict(std::string_view registration_id) {
  std::shared_ptr<Model> model;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_regs_.find(std::string(registration_id));
    if (it == pending_regs_.end()) {
      return absl::NotFoundError("registration_id not found");
    }
    model = it->second.model;
    pending_regs_.erase(it);
  }
  if (model) {
    // Release GPU memory; ignore failures to guarantee best-effort cleanup.
    (void)model->release_memory(ModelLocation::GPU, /*safe_release=*/true);
  }
  return absl::OkStatus();
}

} // namespace stepcast::store