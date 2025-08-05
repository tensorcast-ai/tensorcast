// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/model/model.h"

#include <chrono>
#include <memory>
#include <variant>

#include "absl/functional/overload.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

#include "core/store/loader/disk_loader.h"
#include "core/store/loader/p2p_loader.h"
#include "core/store/model/memory_manager.h"

namespace stepcast::store {

//--------------------------------------------------------------------------
// Static Factory: create()
//--------------------------------------------------------------------------
absl::StatusOr<std::unique_ptr<Model>> Model::create(ModelConfig config) {
  VLOG(1) << "Model::create(" << config.model_identifier << "): Starting creation process.";

  if (config.model_identifier.empty()) {
    return absl::InvalidArgumentError("Model identifier cannot be empty.");
  }

  // --- Create MemoryManager ---
  // Pass the obtained communicator engine (can be nullptr if not needed/initialized)
  auto memory_manager = std::make_shared<MemoryManager>(
      config.model_identifier,
      config.local_device_id,
      config.pinned_memory_pool,
      config.dvmp,
      config.max_buffer_bytes,
      config.pinned_memory_timeout,
      config.require_dvmp_lock_success);

  // --- Create Loader based on Source ---
  std::unique_ptr<IModelLoader> loader;
  ModelLocation source_type = ModelLocation::NONE;

  absl::Status visitor_status = absl::OkStatus(); // Initialize status

  try {
    visitor_status = std::visit( // Capture the status from std::visit
        absl::Overload{
            [&](const DiskSource& disk_source) -> absl::Status { // Explicitly return absl::Status
              VLOG(1) << "Model(" << config.model_identifier << "): Configuring DiskLoader from path "
                      << disk_source.path;
              loader = std::make_unique<DiskLoader>(disk_source);
              source_type = ModelLocation::DISK;
              return absl::OkStatus(); // Return OK status
            },
            [&](const P2PSource& p2p_source) -> absl::Status { // Explicitly return absl::Status
              VLOG(1) << "Model(" << config.model_identifier << "): Configuring P2PLoader from "
                      << p2p_source.ip << ":" << p2p_source.port;
              loader = std::make_unique<P2PLoader>(p2p_source);
              source_type = ModelLocation::REMOTE;
              return absl::OkStatus(); // Return OK status
            },
            [&](const InlineBufferSource& buffer_source) -> absl::Status {
              VLOG(1) << "Model(" << config.model_identifier << "): Configuring InlineBufferLoader";
              // TODO: Implement InlineBufferLoader
              return absl::UnimplementedError("InlineBufferSource not yet supported");
            }},
        config.source);
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrFormat("Failed to create loader for %s: %s", config.model_identifier, e.what()));
  }

  // Check the status returned by the visitor lambda
  if (!visitor_status.ok()) {
    return visitor_status;
  }

  // Check if loader creation succeeded (it might not have if visitor lambda returned error)
  if (!loader) {
    // This condition might be reached if std::visit throws an exception other than std::exception
    // or if a new variant type is added without a corresponding visitor case.
    return absl::InternalError("Loader was not created due to an unexpected error or missing visitor.");
  }

  // --- Initialize Loader ---
  // Initialize the loader after creation and before getting its size.
  absl::Status init_status = loader->initialize();
  if (!init_status.ok()) {
    LOG(ERROR) << "Model(" << config.model_identifier << "): Failed to initialize loader: " << init_status;
    return init_status;
  }

  // --- Get Model Size and Set in Manager ---
  absl::StatusOr<uint64_t> size_status = loader->get_model_size();
  if (!size_status.ok()) {
    LOG(ERROR) << "Model(" << config.model_identifier
               << "): Failed to get model size from loader: " << size_status.status();
    return size_status.status();
  }
  uint64_t model_size = *size_status;
  if (model_size == 0) {
    return absl::FailedPreconditionError(absl::StrFormat("Model %s resolved size is 0.", config.model_identifier));
  }

  // Also check against optional expected size in config
  if (config.expected_model_size.has_value() && config.expected_model_size.value() != model_size) {
    LOG(ERROR) << "Model(" << config.model_identifier << "): Loader size " << model_size << " mismatches expected size "
               << config.expected_model_size.value();
    return absl::FailedPreconditionError("Model size mismatch between loader and config expectation.");
  }

  memory_manager->set_model_size(model_size);

  // --- Create Model Instance ---
  // Build InstanceKey for this model/device
  DeviceKey dev_key;
  if (config.device_type == ::stepcast::DeviceType::GPU) {
    // Explicit GPU target. Fall back to ordinal 0 if not provided.
    dev_key.type = ::stepcast::DeviceType::GPU;
    dev_key.ordinal = (config.local_device_id >= 0) ? config.local_device_id : 0;
  } else {
    // Default / explicit CPU target (or unsupported type which we map to CPU for now).
    dev_key.type = ::stepcast::DeviceType::CPU;
    dev_key.ordinal = -1;
  }

  InstanceKey inst_key{config.model_identifier, dev_key, /*replica=*/0};

  // Use absl::WrapUnique to manage the private constructor call
  auto model_ptr =
      absl::WrapUnique(new Model(std::move(inst_key), std::move(loader), std::move(memory_manager), source_type));
  return model_ptr;
}

//--------------------------------------------------------------------------
// Private Constructor
//--------------------------------------------------------------------------

Model::Model(
    InstanceKey key,
    std::unique_ptr<IModelLoader> loader,
    std::shared_ptr<MemoryManager> memory_manager,
    ModelLocation source_type)
    : key_(std::move(key)),
      loader_(std::move(loader)),
      memory_manager_(std::move(memory_manager)),
      original_source_type_(source_type) {}

//--------------------------------------------------------------------------
// Destructor
//--------------------------------------------------------------------------
Model::~Model() {
  // Gracefully wait for any outstanding asynchronous operations (CPU/GPU load or copy).
  // If the operations do not finish within a reasonable timeout, we abort with LOG(FATAL)
  // to avoid use‑after‑free (UB) on memory_manager_ or loader_.
  constexpr int kWaitSeconds = 30; // Tunable: max time to wait for each future.
  auto wait_for_future = [&](std::shared_future<absl::Status>& fut, const char* tag) {
    if (!fut.valid()) {
      return; // Nothing to wait for.
    }
    auto status = fut.wait_for(std::chrono::seconds(kWaitSeconds));
    if (status != std::future_status::ready) {
      // Potential UB if we proceed because background thread may dereference freed objects.
      LOG(FATAL) << "Model(" << key_.model_id << "): Timed out waiting for " << tag << " future to finish after "
                 << kWaitSeconds << " seconds. Aborting to prevent undefined behaviour.";
    }
    // Future is ready – retrieve the result (ignore, but call get() to propagate any exceptions).
    const absl::Status& st = fut.get();
    if (!st.ok()) {
      LOG(WARNING) << "Model(" << key_.model_id << "): " << tag << " operation completed with status: " << st;
    }
  };

  // We intentionally do NOT hold mutex_ while waiting to avoid deadlocks: the background
  // tasks might attempt to acquire the same mutex in their finalisation paths.
  wait_for_future(cpu_load_future_, "CPU");
  wait_for_future(gpu_load_future_, "GPU");

  // Other resources are released automatically by unique_ptr destructors.
}

//--------------------------------------------------------------------------
// Public Methods
//--------------------------------------------------------------------------

const std::string& Model::model_id() const {
  // key_.model_id is immutable after construction
  return key_.model_id;
}

absl::StatusOr<uint64_t> Model::get_model_size() const {
  // memory_manager_ is unique_ptr, assumed valid. get_model_size is thread-safe.
  if (!memory_manager_) {
    return absl::InternalError("MemoryManager is null.");
  }
  uint64_t size = memory_manager_->get_model_size();
  if (size == 0) {
    return absl::FailedPreconditionError("Model size not set or is zero.");
  }
  return size;
}

std::shared_future<absl::Status> Model::ensure_loaded_async(
    ModelLocation target_location,
    int concurrency,
    std::optional<int> device_id) {
  // Helper: create an already-ready shared_future carrying given status.
  auto make_ready_future = [](const absl::Status& st) -> std::shared_future<absl::Status> {
    return std::async(std::launch::deferred, [st] { return st; }).share();
  };

  // Dynamic device re-configuration is no longer supported. The MemoryManager
  // is now permanently bound to the device id supplied at construction time.

  // ------------------------------------------------------------------
  // Quick recovery path: if previous operation left the target location in
  // FAILED state, attempt to reset the memory so that the caller can retry.
  // This is done *before* taking the Model-level mutex to avoid deadlock
  // with MemoryManager::release_memory which takes its own mutex.
  // ------------------------------------------------------------------
  MemoryState initial_state = memory_manager_->get_state(target_location);
  if (initial_state == MemoryState::FAILED) {
    // Release any partially-allocated resources.
    absl::Status rel_status = memory_manager_->release_memory(target_location, /*safe_release=*/false);
    if (!rel_status.ok()) {
      return make_ready_future(rel_status);
    }
    // Transition the state back to UNALLOCATED so that normal loading can proceed.
    absl::Status state_status = memory_manager_->set_state(target_location, MemoryState::UNALLOCATED);
    if (!state_status.ok()) {
      return make_ready_future(state_status);
    }
    // Invalidate the cached future for this location (if any).
    {
      absl::MutexLock fut_lock(&mutex_);
      if (target_location == ModelLocation::PAGEABLE_CPU) {
        cpu_load_future_ = std::shared_future<absl::Status>();
      } else {
        gpu_load_future_ = std::shared_future<absl::Status>();
      }
    }
    VLOG(1) << "Model(" << key_.model_id << "): Reset memory state from FAILED to UNALLOCATED for "
            << location_to_string(target_location) << ".";
  }

  // ------------------------------------------------------------------
  // Phase-1: quick state inspection under mutex_
  // ------------------------------------------------------------------
  MemoryState current_state = MemoryState::UNINITIALIZED;
  std::shared_future<absl::Status>* member_future = nullptr;
  MemoryManager* mm_ptr = nullptr; // raw pointer is safe as Model owns MemoryManager lifetime
  IModelLoader* ldr_ptr = nullptr; // same rationale as above
  ModelLocation source_location = ModelLocation::NONE;
  bool need_allocation = false;
  absl::StatusOr<ModelLocation> src_status; // Declare here

  {
    absl::MutexLock lock(&mutex_);

    if (!memory_manager_ || !loader_) {
      return make_ready_future(absl::InternalError("Model loader or memory manager is invalid."));
    }

    // Get raw pointers under lock
    mm_ptr = memory_manager_.get();
    ldr_ptr = loader_.get();

    if (target_location == ModelLocation::PAGEABLE_CPU) {
      member_future = &cpu_load_future_;
    } else if (target_location == ModelLocation::GPU) {
      member_future = &gpu_load_future_;
    } else {
      return make_ready_future(absl::InvalidArgumentError("Invalid target location for ensure_loaded_async."));
    }

    current_state = memory_manager_->get_state(target_location);

    // Already LOADED
    if (current_state == MemoryState::LOADED) {
      return make_ready_future(absl::OkStatus());
    }

    // Already LOADING – return the existing shared future if valid
    if (current_state == MemoryState::LOADING) {
      // Common race condition: another thread has transitioned the state to
      // LOADING but has not yet published the shared_future.  Rather than
      // failing, create a lightweight waiting future that completes once the
      // memory manager reports LOADED (or an error state).  This behaviour
      // aligns with the expectation that *all* concurrent callers of
      // prepare() succeed when at least one thread is actively loading the
      // data.

      if (member_future->valid()) {
        return *member_future;
      }

      std::shared_future<absl::Status> fallback_wait =
          std::async(std::launch::async, [mm_ptr, target_location]() {
            // Wait indefinitely; callers can still supply a timeout when they
            // call wait_until_loaded()/wait_ready() on the resulting ModelHandle.
            return mm_ptr->wait_for_state(target_location, MemoryState::LOADED, absl::InfiniteDuration());
          }).share();

      // Publish the newly created future so that subsequent callers can reuse
      // it rather than spinning up additional threads.
      *member_future = fallback_wait;
      return fallback_wait;
    }

    // ------------------------------------------------------------------
    // NEW: Handle race window where another thread has already allocated
    // memory (state = ALLOCATED) but has not yet transitioned the state to
    // LOADING.  Launching an additional load task here would result in
    // duplicate disk reads and, more importantly, a second call to
    // allocate_buffer_pool() which fails with AlreadyExists, ultimately
    // placing the MemoryManager in FAILED state.  Instead, we treat the
    // ALLOCATED state as a signal that a load is imminent and simply wait
    // for the first thread to transition the state to LOADED or FAILED.
    // ------------------------------------------------------------------
    if (current_state == MemoryState::ALLOCATED) {
      if (member_future->valid()) {
        return *member_future; // Reuse existing future if one was published.
      }

      // Create a lightweight waiter future that blocks until the location
      // becomes LOADED (or FAILED) without spawning a second load.
      std::shared_future<absl::Status> fallback_wait =
          std::async(std::launch::async, [mm_ptr, target_location]() {
            return mm_ptr->wait_for_state(target_location, MemoryState::LOADED, absl::InfiniteDuration());
          }).share();

      *member_future = fallback_wait; // Publish for subsequent callers.
      return fallback_wait;
    }

    // FAILED previously
    if (current_state == MemoryState::FAILED) {
      return make_ready_future(absl::FailedPreconditionError("Previous operation failed."));
    }

    // For UNALLOCATED / ALLOCATED resolve best source
    need_allocation = (current_state == MemoryState::UNALLOCATED);

    // ------------------------------------------------------------------
    // Allocate memory *inside* the critical section to ensure only a single
    // thread performs the initial allocation.  This prevents a race where
    // another thread observes the still-UNALLOCATED state before the first
    // thread has finished allocation, leading to duplicate allocation and
    // duplicate load tasks (which can leave the MemoryManager in FAILED
    // state).  The allocation itself may be moderately expensive (e.g.
    // cudaMalloc) but is still far cheaper than the full disk load we perform
    // later and only happens once per instance, so holding the mutex here is
    // acceptable.
    // ------------------------------------------------------------------
    if (need_allocation) {
      absl::Status alloc_st = mm_ptr->allocate_memory(target_location);
      if (!alloc_st.ok()) {
        return make_ready_future(alloc_st);
      }

      // State is now at least ALLOCATED – subsequent concurrent callers will
      // enter the ALLOCATED/LOADING fast-path above and wait rather than
      // launching a duplicate load.
      need_allocation = false; // Prevent duplicate allocation outside lock.
    }

    src_status = find_best_source_for_target(target_location);
    if (!src_status.ok()) {
      LOG(ERROR) << "Model(" << key_.model_id << "): Failed to find best source for target "
                 << location_to_string(target_location) << ": " << src_status.status();
      return make_ready_future(src_status.status());
    }
    source_location = *src_status;
  }
  // ---------------- Mutex released ----------------

  // ------------------------------------------------------------------
  // Phase-2: potentially heavy work (allocation / async task dispatch)
  // ------------------------------------------------------------------
  // Allocation already performed under the mutex above.  At this point the
  // memory manager is guaranteed to be in at least the ALLOCATED state, so
  // we can proceed with launching the actual load/copy task without holding
  // the Model-level mutex.

  // Dispatch appropriate task
  std::shared_future<absl::Status> new_future;
  if (source_location == ModelLocation::PAGEABLE_CPU && target_location == ModelLocation::GPU) {
    new_future = mm_ptr->copy_data_async(ModelLocation::PAGEABLE_CPU, ModelLocation::GPU).share();
  } else if (source_location == ModelLocation::GPU && target_location == ModelLocation::PAGEABLE_CPU) {
    new_future = mm_ptr->copy_data_async(ModelLocation::GPU, ModelLocation::PAGEABLE_CPU).share();
  } else if (source_location == ModelLocation::DISK || source_location == ModelLocation::REMOTE) {
    new_future = ldr_ptr->load_async(memory_manager_, target_location, concurrency).share();
  } else {
    return make_ready_future(absl::InternalError("Invalid source/target combination."));
  }

  // ------------------------------------------------------------------
  // Phase‑3: store the new shared_future under lock then return a copy
  // ------------------------------------------------------------------
  {
    absl::MutexLock lock(&mutex_);
    if (member_future) {
      *member_future = new_future; // copy assignment – shared_future is cheap
    }
  }

  return new_future;
}

absl::StatusOr<ModelLocation> Model::find_best_source_for_target(ModelLocation target_location) const {
  // Assumes mutex_ is held. (Now called within lock scope in ensure_loaded_async)
  MemoryState cpu_state = memory_manager_->get_state(ModelLocation::PAGEABLE_CPU);
  MemoryState gpu_state = memory_manager_->get_state(ModelLocation::GPU);

  if (target_location == ModelLocation::PAGEABLE_CPU) {
    // If GPU is loaded, copy from GPU. Otherwise, load from original source.
    if (gpu_state == MemoryState::LOADED) {
      return ModelLocation::GPU;
    } // Check if original source is valid for loading to CPU
    if (original_source_type_ == ModelLocation::DISK) {
      return ModelLocation::DISK;
    }
    if (original_source_type_ == ModelLocation::REMOTE) {
      // Add REMOTE->CPU path if P2PLoader supports it later
      return ModelLocation::REMOTE;
    }
    return absl::InternalError("Invalid original source type.");
  }
  if (target_location == ModelLocation::GPU) {
    // If CPU is loaded, copy from CPU. Otherwise, load from original source.
    if (cpu_state == MemoryState::LOADED) {
      return ModelLocation::PAGEABLE_CPU;
    } // Can load from DISK (via CPU staging implicitly handled by DiskLoader->copy)
    // or directly from REMOTE if P2P->GPU is supported.
    if (original_source_type_ == ModelLocation::DISK) {
      // Need to ensure CPU is loaded first, then copy.
      // This logic is handled within ensure_loaded_async by calling itself recursively or chaining.
      // Let's simplify: if CPU not loaded, source is DISK (implying DISK->CPU->GPU path)
      // Actually, the loader/manager should handle the pipeline.
      // If original source is Disk, we tell the loader to load to CPU, then trigger copy.
      // Let ensure_loaded_async handle this chaining.
      // Here, just indicate the *ultimate* source.
      // If CPU not loaded, ultimate source is DISK/REMOTE.
      return original_source_type_;
    }
    if (original_source_type_ == ModelLocation::REMOTE) {
      return ModelLocation::REMOTE; // Load directly P2P->GPU
    }
    return absl::InternalError("Invalid original source type.");
  }
  return absl::InvalidArgumentError("Invalid target location.");
}

absl::Status Model::release_memory(ModelLocation location, bool safe_release) {
  absl::MutexLock lock(&mutex_);
  if (!memory_manager_) {
    return absl::InternalError("MemoryManager is null.");
  }

  VLOG(2) << "Model(" << key_.model_id << "): Releasing memory for " << location_to_string(location)
          << ", safe=" << safe_release;

  absl::Status status = memory_manager_->release_memory(location, safe_release);

  // If release was successful and state is now unallocated, invalidate the future
  if (status.ok() && memory_manager_->get_state(location) <= MemoryState::UNALLOCATED) {
    std::shared_future<absl::Status>* relevant_future = nullptr;
    if (location == ModelLocation::PAGEABLE_CPU) {
      relevant_future = &cpu_load_future_;
    } else if (location == ModelLocation::GPU) {
      relevant_future = &gpu_load_future_;
    }

    if (relevant_future && relevant_future->valid()) {
      VLOG(2) << "Model(" << key_.model_id << "): Invalidating future for released location "
              << location_to_string(location);
      // Reset the future object
      *relevant_future = std::shared_future<absl::Status>();
    }
  }
  return status;
}

MemoryState Model::get_memory_state(ModelLocation location) const {
  absl::MutexLock lock(&mutex_);
  if (!memory_manager_) {
    LOG(WARNING) << "Model(" << key_.model_id << "): MemoryManager is null, returning UNINITIALIZED state.";
    return MemoryState::UNINITIALIZED;
  }
  return memory_manager_->get_state(location);
}

std::vector<void*> Model::get_data_pointer(ModelLocation location) const {
  // memory_manager_->get_pointer is thread-safe
  if (!memory_manager_) {
    return {};
  }
  return memory_manager_->get_pointer(location);
}

absl::Status Model::wait_until_loaded(ModelLocation location, absl::Duration timeout) {
  // memory_manager_->wait_for_state is thread-safe
  if (!memory_manager_) {
    return absl::InternalError("MemoryManager is null.");
  }
  VLOG(1) << "Model(" << key_.model_id << "): Waiting until loaded for " << location_to_string(location)
          << ", timeout=" << timeout;
  return memory_manager_->wait_for_state(location, MemoryState::LOADED, timeout);
}

MemoryManager& Model::get_memory_manager() const {
  // Returning reference to unique_ptr's managed object. Okay if Model lifetime > caller usage.
  if (!memory_manager_) {
    // This should ideally not happen if create() succeeded.
    LOG(FATAL) << "Internal error: MemoryManager is null in Model.";
  }
  return *memory_manager_;
}

absl::StatusOr<CommRegistrationInfo> Model::enable_remote_memory_access(
    ModelLocation location,
    stepcast::communicator::CommunicateEngine& comm_engine) {
  // Mutex potentially not needed here IF MemoryManager::enable_remote_memory_access is fully thread-safe
  // However, accessing memory_manager_ itself might warrant the lock.
  absl::MutexLock lock(&mutex_);
  if (!memory_manager_) {
    return absl::InternalError("MemoryManager is null.");
  }

  return memory_manager_->enable_remote_memory_access(location, comm_engine);
}

absl::StatusOr<ModelVerificationInfo> Model::generate_verification_info(ModelLocation location) const {
  absl::MutexLock lock(&mutex_);
  if (!memory_manager_) {
    return absl::InternalError("MemoryManager is null.");
  }

  // Check if data is loaded at the specified location
  MemoryState state = memory_manager_->get_state(location);
  if (state != MemoryState::LOADED) {
    return absl::FailedPreconditionError(
        absl::StrFormat(
            "Model data must be loaded at %s before generating verification info. Current state: %s",
            location_to_string(location),
            state_to_string(state)));
  }

  // Get data pointers and sizes
  std::vector<void*> data_ptrs = memory_manager_->get_pointer(location);
  if (data_ptrs.empty()) {
    return absl::InternalError("No data pointers available for loaded model.");
  }

  std::vector<size_t> data_sizes;
  uint64_t model_size = memory_manager_->get_model_size();

  if (location == ModelLocation::GPU) {
    // GPU has single contiguous buffer
    data_sizes.push_back(model_size);
  } else {
    // CPU has chunked memory
    size_t chunk_size = memory_manager_->get_cpu_chunk_size();
    size_t remaining_size = model_size;
    for (size_t i = 0; i < data_ptrs.size() && remaining_size > 0; ++i) {
      size_t current_chunk_size = std::min(chunk_size, remaining_size);
      data_sizes.push_back(current_chunk_size);
      remaining_size -= current_chunk_size;
    }
  }

  // Determine device ID for verification
  int device_id = (location == ModelLocation::GPU) ? memory_manager_->get_local_device_id() : -1;

  LOG(INFO) << "Model(" << key_.model_id << "): Generating verification info for " << location_to_string(location)
            << " (" << data_ptrs.size() << " chunks, " << model_size << " bytes total)";

  return ModelVerifier::generate_verification_info(data_ptrs, data_sizes, device_id);
}

absl::Status Model::verify_model_data(
    ModelLocation location,
    const ModelVerificationInfo& expected_info,
    VerificationLevel level) const {
  absl::MutexLock lock(&mutex_);
  if (!memory_manager_) {
    return absl::InternalError("MemoryManager is null.");
  }

  // Check if data is loaded at the specified location
  MemoryState state = memory_manager_->get_state(location);
  if (state != MemoryState::LOADED) {
    return absl::FailedPreconditionError(
        absl::StrFormat(
            "Model data must be loaded at %s before verification. Current state: %s",
            location_to_string(location),
            state_to_string(state)));
  }

  // Get data pointers and sizes
  std::vector<void*> data_ptrs = memory_manager_->get_pointer(location);
  if (data_ptrs.empty()) {
    return absl::InternalError("No data pointers available for loaded model.");
  }

  std::vector<size_t> data_sizes;
  uint64_t model_size = memory_manager_->get_model_size();

  if (location == ModelLocation::GPU) {
    // GPU has single contiguous buffer
    data_sizes.push_back(model_size);
  } else {
    // CPU has chunked memory
    size_t chunk_size = memory_manager_->get_cpu_chunk_size();
    size_t remaining_size = model_size;
    for (size_t i = 0; i < data_ptrs.size() && remaining_size > 0; ++i) {
      size_t current_chunk_size = std::min(chunk_size, remaining_size);
      data_sizes.push_back(current_chunk_size);
      remaining_size -= current_chunk_size;
    }
  }

  // Determine device ID for verification
  int device_id = (location == ModelLocation::GPU) ? memory_manager_->get_local_device_id() : -1;

  LOG(INFO) << "Model(" << key_.model_id << "): Verifying " << location_to_string(location) << " data at level "
            << static_cast<int>(level);

  return ModelVerifier::verify_model_data(data_ptrs, data_sizes, expected_info, level, device_id);
}

absl::Status Model::verify_key_points(ModelLocation location, const ModelVerificationInfo& expected_info) const {
  absl::MutexLock lock(&mutex_);
  if (!memory_manager_) {
    return absl::InternalError("MemoryManager is null.");
  }

  // Check if data is loaded at the specified location
  MemoryState state = memory_manager_->get_state(location);
  if (state != MemoryState::LOADED) {
    return absl::FailedPreconditionError(
        absl::StrFormat(
            "Model data must be loaded at %s before verification. Current state: %s",
            location_to_string(location),
            state_to_string(state)));
  }

  // Get data pointers and sizes
  std::vector<void*> data_ptrs = memory_manager_->get_pointer(location);
  if (data_ptrs.empty()) {
    return absl::InternalError("No data pointers available for loaded model.");
  }

  std::vector<size_t> data_sizes;
  uint64_t model_size = memory_manager_->get_model_size();

  if (location == ModelLocation::GPU) {
    // GPU has single contiguous buffer
    data_sizes.push_back(model_size);
  } else {
    // CPU has chunked memory
    size_t chunk_size = memory_manager_->get_cpu_chunk_size();
    size_t remaining_size = model_size;
    for (size_t i = 0; i < data_ptrs.size() && remaining_size > 0; ++i) {
      size_t current_chunk_size = std::min(chunk_size, remaining_size);
      data_sizes.push_back(current_chunk_size);
      remaining_size -= current_chunk_size;
    }
  }

  // Determine device ID for verification
  int device_id = (location == ModelLocation::GPU) ? memory_manager_->get_local_device_id() : -1;

  LOG(INFO) << "Model(" << key_.model_id << "): Fast key-point verification for " << location_to_string(location);

  return ModelVerifier::verify_key_points(data_ptrs, data_sizes, expected_info, device_id);
}

absl::Status Model::disable_remote_memory_access(
    ModelLocation location,
    stepcast::communicator::CommunicateEngine& comm_engine) {
  absl::MutexLock lock(&mutex_);
  if (!memory_manager_) {
    return absl::InternalError("MemoryManager is null.");
  }

  return memory_manager_->disable_remote_memory_access(location, comm_engine);
}

absl::Status Model::copy_from(const Model& src) {
  // ────────────────────────────────────────────────────────────────────────────
  // Preconditions & quick checks (no locks held)
  // ────────────────────────────────────────────────────────────────────────────
  if (&src == this) {
    return absl::InvalidArgumentError("Cannot copy_from self");
  }

  // Currently we only support GPU → GPU direct copies.  Extend later for CPU↔GPU.
  if (src.device().type != DeviceType::GPU || device().type != DeviceType::GPU) {
    return absl::UnimplementedError("Model::copy_from supports only GPU→GPU copies at present");
  }

  // Ensure source GPU memory is LOADED.  We purposefully do not acquire src.mutex_
  // here because MemoryManager::get_state is internally thread-safe.
  if (src.get_memory_state(ModelLocation::GPU) != MemoryState::LOADED) {
    return absl::FailedPreconditionError("Source model GPU memory not in LOADED state");
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Prepare destination GPU memory: allocate if necessary.
  // ────────────────────────────────────────────────────────────────────────────
  MemoryManager& dst_mm = get_memory_manager();
  MemoryState dst_state = dst_mm.get_state(ModelLocation::GPU);
  if (dst_state == MemoryState::UNALLOCATED) {
    absl::Status alloc_st = dst_mm.allocate_memory(ModelLocation::GPU);
    if (!alloc_st.ok()) {
      return alloc_st;
    }
    dst_state = dst_mm.get_state(ModelLocation::GPU);
  }

  // Allow ALLOCATED destination state for the copy.
  if (dst_state != MemoryState::ALLOCATED) {
    return absl::FailedPreconditionError(
        absl::StrCat("Destination GPU memory must be ALLOCATED, but is ", state_to_string(dst_state)));
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Perform the actual peer-to-peer copy.  This call blocks until the copy
  // completes and the destination reaches LOADED or FAILED.
  // ────────────────────────────────────────────────────────────────────────────
  return dst_mm.copy_from_peer(src.get_memory_manager());
}

} // namespace stepcast::store