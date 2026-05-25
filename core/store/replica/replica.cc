// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/replica/replica.h"

#include <chrono>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string_view>
#include <variant>

#include "absl/functional/overload.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "core/store/materialization/contracts/representation_contract.h"
#include "core/store/materialization/dataplane/contracts/inline_buffer_loader.h"
#include "core/store/materialization/dataplane/contracts/loader.h"
#include "core/store/materialization/dataplane/loaders/disk_loader.h"
#include "core/store/materialization/dataplane/loaders/p2p_loader.h"
#include "core/store/materialization/dataplane/sources/byte_range_map_builder.h"
#include "core/store/materialization/dataplane/sources/byte_range_mapped_source.h"
#include "core/store/materialization/dataplane/sources/byte_range_program.h"
#include "core/store/materialization/dataplane/view/view_plan_source.h"
#include "core/store/materialization/dataplane/view/view_transform_executor.h"
#include "core/store/replica/replica_load_controller.h"
#include "core/store/replica/replica_placement.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::replica {

using common::memory::MemoryLocation;

namespace {

std::optional<uint64_t> compute_total_size_from_index(std::string_view index_json) {
  if (index_json.empty()) {
    return std::nullopt;
  }
  try {
    nlohmann::json parsed = nlohmann::json::parse(index_json, nullptr, true);
    if (!parsed.is_object()) {
      return std::nullopt;
    }
    uint64_t total_size = 0;
    for (auto it = parsed.begin(); it != parsed.end(); ++it) {
      const auto& arr = it.value();
      if (!arr.is_array() || arr.size() < 2) {
        continue;
      }
      uint64_t offset = arr[0].get<uint64_t>();
      uint64_t size = arr[1].get<uint64_t>();
      total_size = std::max<uint64_t>(total_size, offset + size);
    }
    if (total_size == 0) {
      return std::nullopt;
    }
    return total_size;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

StoreEngineOptions::ByteMappingConfig maybe_relax_mmap_strided_thresholds(
    StoreEngineOptions::ByteMappingConfig config,
    const loader::SeekableSource& source,
    common::memory::MemoryLocation source_type,
    const std::string& artifact_identifier) {
  if (source_type != common::memory::MemoryLocation::DISK) {
    return config;
  }
  if (source.cpu_base_ptr() == nullptr) {
    return config;
  }
  constexpr uint32_t kMmapStridedMinRanges = 8;
  constexpr uint64_t kMmapStridedMinRowLenBytes = 512;
  bool changed = false;
  if (config.strided_run_min_ranges <= kMmapStridedMinRanges) {
  } else {
    config.strided_run_min_ranges = kMmapStridedMinRanges;
    changed = true;
  }
  if (config.strided_min_row_len_bytes > kMmapStridedMinRowLenBytes) {
    config.strided_min_row_len_bytes = kMmapStridedMinRowLenBytes;
    changed = true;
  }
  if (changed) {
    VLOG(1) << "Replica(" << artifact_identifier << "): relaxing strided thresholds for mmap-capable disk source"
            << " min_ranges=" << config.strided_run_min_ranges
            << " min_row_len_bytes=" << config.strided_min_row_len_bytes;
  }
  return config;
}

loader::ByteRangeMappedSource::Options make_mmap_aware_map_options(
    std::string path,
    const loader::SeekableSource& source,
    common::memory::MemoryLocation source_type,
    const std::string& artifact_identifier,
    bool enable_direct_write_at) {
  loader::ByteRangeMappedSource::Options options{
      .path = std::move(path),
      .enable_direct_write_at = enable_direct_write_at,
  };
  if (source_type != common::memory::MemoryLocation::DISK || source.cpu_base_ptr() == nullptr) {
    return options;
  }
  constexpr uint64_t kMmapDirectGatherMinRowLenBytes = 512;
  constexpr size_t kMmapDirectGatherMinTotalBytes = 256ULL * 1024;
  constexpr uint64_t kMmapDirectGatherMaxRowsTouched = std::numeric_limits<uint64_t>::max() / 2;
  options.direct_gather_min_row_len_bytes = kMmapDirectGatherMinRowLenBytes;
  options.direct_gather_min_total_bytes = kMmapDirectGatherMinTotalBytes;
  options.direct_gather_max_rows_touched = kMmapDirectGatherMaxRowsTouched;
  VLOG(1) << "Replica(" << artifact_identifier << "): relaxing direct_gather thresholds for mmap-capable disk source"
          << " min_row_len_bytes=" << options.direct_gather_min_row_len_bytes
          << " min_total_bytes=" << options.direct_gather_min_total_bytes
          << " max_rows_touched=" << options.direct_gather_max_rows_touched;
  return options;
}

} // namespace

//--------------------------------------------------------------------------
// Static Factory: create()
//--------------------------------------------------------------------------
absl::StatusOr<std::unique_ptr<Replica>> Replica::create(ReplicaConfig config) {
  VLOG(1) << "Replica::create(" << config.artifact_identifier << "): Starting creation process.";

  if (config.artifact_identifier.empty()) {
    return absl::InvalidArgumentError("Replica identifier cannot be empty.");
  }

  // ReplicaLoadController will be constructed after resolving artifact size

  // --- Create Loader based on Source ---
  std::unique_ptr<IArtifactLoader> loader;
  common::memory::MemoryLocation source_type = common::memory::MemoryLocation::NONE;

  absl::Status visitor_status = absl::OkStatus(); // Initialize status

  try {
    visitor_status = std::visit( // Capture the status from std::visit
        absl::Overload{
            [&](const loading::DiskSource& disk_source) -> absl::Status { // Explicitly return absl::Status
              VLOG(1) << "Replica(" << config.artifact_identifier << "): Configuring DiskLoader from path "
                      << disk_source.path;
              loader = std::make_unique<DiskLoader>(disk_source);
              source_type = common::memory::MemoryLocation::DISK;
              return absl::OkStatus(); // Return OK status
            },
            [&](const tensorcast::store::P2PSource& p2p_source) -> absl::Status { // Explicitly return absl::Status
              VLOG(1) << "Replica(" << config.artifact_identifier << "): Configuring P2PLoader from " << p2p_source.ip << ":"
                      << p2p_source.port;
              loader = std::make_unique<P2PLoader>(p2p_source);
              source_type = common::memory::MemoryLocation::REMOTE;
              return absl::OkStatus(); // Return OK status
            },
            [&](const loading::InlineBufferSource& buffer_source) -> absl::Status {
              VLOG(1) << "Replica(" << config.artifact_identifier << "): Configuring InlineBufferLoader";
              loader = std::make_unique<InlineBufferLoader>(buffer_source);
              source_type = common::memory::MemoryLocation::CPU; // semantic placeholder for in-memory
              return absl::OkStatus();
            }},
        config.source);
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("Failed to create loader for ", config.artifact_identifier, ": ", e.what()));
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
    LOG(ERROR) << "Replica(" << config.artifact_identifier << "): Failed to initialize loader: " << init_status;
    return init_status;
  }

  // --- Get Replica Size and Set in Manager ---
  absl::StatusOr<uint64_t> size_status = loader->get_artifact_size();
  if (!size_status.ok()) {
    LOG(ERROR) << "Replica(" << config.artifact_identifier
               << "): Failed to get artifact size from loader: " << size_status.status();
    return size_status.status();
  }
  const uint64_t loader_size = *size_status;
  if (loader_size == 0) {
    return absl::FailedPreconditionError(absl::StrCat("Replica ", config.artifact_identifier, " resolved size is 0."));
  }

  // Optional view plan overrides the effective replica size.
  std::optional<loader::ViewPlan> view_plan = std::move(config.view_plan);
  uint64_t effective_size = loader_size;
  std::optional<uint64_t> canonical_total_size;
  if (config.canonical_index_json.has_value()) {
    canonical_total_size = compute_total_size_from_index(*config.canonical_index_json);
  }
  if (view_plan.has_value()) {
    effective_size = view_plan->view_size_bytes;
    if (effective_size == 0) {
      return absl::FailedPreconditionError(
          absl::StrCat("Replica ", config.artifact_identifier, " view plan resolved size is 0."));
    }
  } else if (canonical_total_size.has_value() && config.source_index_json.has_value()) {
    effective_size = *canonical_total_size;
  }

  // Also check against optional expected size in config (after applying view overrides)
  if (config.expected_artifact_size.has_value() && config.expected_artifact_size.value() != effective_size) {
    LOG(ERROR) << "Replica(" << config.artifact_identifier << "): Expected size "
               << config.expected_artifact_size.value() << " mismatches effective size " << effective_size;
    return absl::FailedPreconditionError("Artifact size mismatch between loader and config expectation.");
  }

  // --- Create ReplicaLoadController ---
  auto view_id = config.view_id;

  auto dev_key_or = resolve_replica_config_device_key(config);
  if (!dev_key_or.ok()) {
    return dev_key_or.status();
  }
  DeviceKey dev_key = *dev_key_or;

  auto memory_manager = std::make_shared<ReplicaLoadController>(
      config.artifact_identifier,
      dev_key,
      config.pinned_buffer_pool,
      config.async_runtime,
      config.artifact_chunk_bytes,
      config.max_buffer_bytes,
      config.pinned_memory_timeout,
      config.streaming_buffer_chunks,
      effective_size,
      view_id,
      config.memory_tier_config,
      config.cpu_shared_memory_enabled);

  // --- Create Replica Instance ---
  // Build ReplicaKey for this replica/device
  loading::ReplicaKey inst_key{
      .artifact_id = config.artifact_identifier, .view_id = std::move(view_id), .device = dev_key, .replica = 0};

  // Use absl::WrapUnique to manage the private constructor call
  auto replica_ptr = absl::WrapUnique(new Replica(
      std::move(inst_key),
      std::move(loader),
      std::move(memory_manager),
      config.async_runtime,
      source_type,
      std::move(view_plan),
      std::move(config.canonical_index_json),
      std::move(config.source_index_json),
      std::move(config.collective_load_group),
      std::move(config.variant_identity),
      config.transform_placement,
      config.byte_mapping_config,
      config.materialization_strategy,
      std::move(config.execution_strategy_plan)));
  return replica_ptr;
}

//--------------------------------------------------------------------------
// Private Constructor
//--------------------------------------------------------------------------

Replica::Replica(
    loading::ReplicaKey key,
    std::unique_ptr<IArtifactLoader> loader,
    std::shared_ptr<ReplicaLoadController> memory_manager,
    gsl::not_null<std::shared_ptr<common::AsyncRuntime>> async_runtime,
    common::memory::MemoryLocation source_type,
    std::optional<loader::ViewPlan> view_plan,
    std::optional<std::string> canonical_index_json,
    std::optional<std::string> source_index_json,
    std::optional<loading::CollectiveLoadGroupHint> collective_load_group,
    std::optional<loading::VariantIdentity> variant_identity,
    loading::TransformPlacement transform_placement,
    StoreEngineOptions::ByteMappingConfig byte_mapping_config,
    StoreEngineOptions::MaterializationStrategyConfig materialization_strategy,
    std::optional<runtime::ingestion::strategy::ExecutionStrategyPlan> execution_strategy_plan)
    : key_(std::move(key)),
      loader_(std::move(loader)),
      memory_manager_(std::move(memory_manager)),
      async_runtime_(std::move(async_runtime)),
      original_source_type_(source_type),
      view_plan_(std::move(view_plan)),
      canonical_index_json_(std::move(canonical_index_json)),
      source_index_json_(std::move(source_index_json)),
      collective_load_group_(std::move(collective_load_group)),
      variant_identity_(std::move(variant_identity)),
      transform_placement_(transform_placement),
      byte_mapping_config_(std::move(byte_mapping_config)),
      materialization_strategy_(std::move(materialization_strategy)),
      execution_strategy_plan_(std::move(execution_strategy_plan)) {}

//--------------------------------------------------------------------------
// Destructor
//--------------------------------------------------------------------------
Replica::~Replica() {
  // No blocking waits here. All background work is tracked by AsyncRuntime and
  // must be drained by the top-level owner (or the StoreEngine-owned runtime)
  // before destroying replicas.
}

//--------------------------------------------------------------------------
// Public Methods
//--------------------------------------------------------------------------

const std::string& Replica::artifact_id() const {
  // key_.artifact_id is immutable after construction
  return key_.artifact_id;
}

absl::StatusOr<uint64_t> Replica::get_artifact_size() const {
  uint64_t size = memory_manager_->get_artifact_size();
  if (size == 0) {
    return absl::FailedPreconditionError("Artifact size not set or is zero.");
  }
  return size;
}

folly::SemiFuture<absl::Status> Replica::ensure_loaded_async(
    MemoryLocation target_location,
    int concurrency,
    std::optional<int> device_id) {
  (void)device_id; // device binding is fixed at construction time

  if (target_location != MemoryLocation::CPU && target_location != MemoryLocation::GPU) {
    return folly::makeSemiFuture<absl::Status>(
        absl::InvalidArgumentError("Invalid target location for ensure_loaded_async."));
  }

  // ------------------------------------------------------------------
  // Quick recovery path: if previous operation left the target location in
  // FAILED state, attempt to reset the memory so that the caller can retry.
  // This is done *before* taking the Replica-level mutex to avoid deadlock
  // with ReplicaLoadController::release_memory which takes its own mutex.
  // ------------------------------------------------------------------
  MemoryState initial_state = memory_manager_->get_state(target_location);
  if (initial_state == MemoryState::FAILED) {
    absl::Status rel_status = memory_manager_->release_memory(target_location);
    if (!rel_status.ok()) {
      return folly::makeSemiFuture(std::move(rel_status));
    }
    absl::Status state_status = memory_manager_->set_state(target_location, MemoryState::UNALLOCATED);
    if (!state_status.ok()) {
      return folly::makeSemiFuture(std::move(state_status));
    }
    {
      absl::MutexLock lock(&mutex_);
      if (target_location == MemoryLocation::CPU) {
        cpu_ready_signal_.reset();
      } else {
        gpu_ready_signal_.reset();
      }
    }
    VLOG(1) << "Replica(" << key_.artifact_id << "): Reset memory state from FAILED to UNALLOCATED for "
            << location_to_string(target_location) << ".";
  }

  std::shared_ptr<common::ReadySignal<absl::Status>> ready_signal;
  MemoryLocation source_location = MemoryLocation::NONE;
  {
    absl::MutexLock lock(&mutex_);

    std::shared_ptr<common::ReadySignal<absl::Status>>* slot =
        (target_location == MemoryLocation::CPU) ? &cpu_ready_signal_ : &gpu_ready_signal_;

    const MemoryState current_state = memory_manager_->get_state(target_location);
    if (current_state == MemoryState::LOADED) {
      if (*slot) {
        return (*slot)->subscribe();
      }
      return folly::makeSemiFuture<absl::Status>(absl::OkStatus());
    }

    if (current_state == MemoryState::LOADING || current_state == MemoryState::ALLOCATED) {
      if (*slot) {
        return (*slot)->subscribe();
      }
      return folly::makeSemiFuture<absl::Status>(absl::InternalError("Load in progress but ready signal missing"));
    }

    if (current_state == MemoryState::FAILED) {
      return folly::makeSemiFuture<absl::Status>(absl::FailedPreconditionError("Previous operation failed."));
    }

    // Producer-side completion signal must be published before entering
    // in-flight (ALLOCATED/LOADING) to eliminate race windows and fallback
    // waiter threads.
    ready_signal = std::make_shared<common::ReadySignal<absl::Status>>();
    *slot = ready_signal;

    if (current_state <= MemoryState::UNALLOCATED) {
      absl::Status alloc_st = memory_manager_->allocate_memory(target_location);
      if (!alloc_st.ok()) {
        ready_signal->set_value(alloc_st);
        slot->reset();
        return ready_signal->subscribe();
      }
    }

    auto src_status = find_best_source_for_target(target_location);
    if (!src_status.ok()) {
      LOG(ERROR) << "Replica(" << key_.artifact_id << "): Failed to find best source for target "
                 << location_to_string(target_location) << ": " << src_status.status();
      ready_signal->set_value(src_status.status());
      slot->reset();
      return ready_signal->subscribe();
    }
    source_location = *src_status;
  }

  folly::SemiFuture<absl::Status> op = folly::makeSemiFuture(absl::OkStatus());
  if (source_location == MemoryLocation::CPU && target_location == MemoryLocation::GPU) {
    op = memory_manager_->copy_data_async(MemoryLocation::CPU, MemoryLocation::GPU);
  } else if (source_location == MemoryLocation::GPU && target_location == MemoryLocation::CPU) {
    op = memory_manager_->copy_data_async(MemoryLocation::GPU, MemoryLocation::CPU);
  } else if (source_location == MemoryLocation::DISK || source_location == MemoryLocation::REMOTE) {
    auto src_or = loader_->open_source();
    if (!src_or.ok()) {
      (void)memory_manager_->release_memory(target_location);
      ready_signal->set_value(src_or.status());
      {
        absl::MutexLock lock(&mutex_);
        if (target_location == MemoryLocation::CPU) {
          cpu_ready_signal_.reset();
        } else {
          gpu_ready_signal_.reset();
        }
      }
      return ready_signal->subscribe();
    }
    std::unique_ptr<loader::SeekableSource> source_ptr = std::move(*src_or);
    std::optional<ReplicaLoadController::CollectiveDiskLoadInput> collective_disk_load;
    std::optional<ReplicaLoadController::LocalBatchedDiskLoadInput> local_batched_disk_load;
    const std::optional<runtime::ingestion::strategy::ExecutionStrategyPlan>& execution_strategy_plan =
        execution_strategy_plan_;
    if (execution_strategy_plan.has_value() && target_location == MemoryLocation::GPU &&
        source_location == MemoryLocation::DISK) {
      const auto& plan = *execution_strategy_plan;
      if (plan.executor == runtime::ingestion::strategy::ExecutionStrategyExecutor::kOwnerFileCollective &&
          plan.collective_load_group.has_value() && plan.disk_context != nullptr &&
          plan.representation_work_plan.has_value()) {
        collective_disk_load = ReplicaLoadController::CollectiveDiskLoadInput{
            .group = *plan.collective_load_group,
            .disk_context = plan.disk_context,
            .representation_work_plan = *plan.representation_work_plan,
            .materialization_strategy = materialization_strategy_,
        };
      } else if (
          plan.executor == runtime::ingestion::strategy::ExecutionStrategyExecutor::kTensorBatchedLocal &&
          plan.disk_context != nullptr && plan.representation_work_plan.has_value()) {
        local_batched_disk_load = ReplicaLoadController::LocalBatchedDiskLoadInput{
            .disk_context = plan.disk_context,
            .representation_work_plan = *plan.representation_work_plan,
            .materialization_strategy = materialization_strategy_,
        };
      }
      LOG(INFO) << "Replica(" << key_.artifact_id << "): using execution_strategy_plan executor="
                << runtime::ingestion::strategy::execution_strategy_executor_name(plan.executor)
                << " selection_reason=" << plan.selection_reason
                << " collective_selected=" << collective_disk_load.has_value()
                << " local_batched_selected=" << local_batched_disk_load.has_value();
    }
    const StoreEngineOptions::ByteMappingConfig effective_byte_mapping_config =
        maybe_relax_mmap_strided_thresholds(byte_mapping_config_, *source_ptr, source_location, key_.artifact_id);
    if (collective_disk_load.has_value()) {
      op = memory_manager_->load_async_from_source(
          std::move(source_ptr),
          target_location,
          concurrency,
          std::nullopt,
          {},
          execution_strategy_plan,
          std::move(collective_disk_load),
          std::nullopt);
    } else {
      bool composed_view = false;
      if (!local_batched_disk_load.has_value() && canonical_index_json_.has_value() && source_index_json_.has_value()) {
        auto canonical_total_size = compute_total_size_from_index(*canonical_index_json_);
        if (!canonical_total_size.has_value()) {
          ready_signal->set_value(absl::FailedPreconditionError("canonical index total_size is unavailable"));
          return ready_signal->subscribe();
        }
        auto map_or = loader::build_byte_range_map_from_canonical_and_source_index_json(
            *canonical_index_json_, *source_index_json_, *canonical_total_size);
        if (!map_or.ok()) {
          ready_signal->set_value(map_or.status());
          return ready_signal->subscribe();
        }
        loader::ByteRangeMap effective_map = std::move(*map_or);
        if (view_plan_.has_value() && !view_plan_->is_identity) {
          auto composed_or = loader::compose_byte_range_maps(view_plan_->selection.map, effective_map);
          if (!composed_or.ok()) {
            ready_signal->set_value(composed_or.status());
            return ready_signal->subscribe();
          }
          effective_map = std::move(*composed_or);
          composed_view = true;
        }
        loader::ByteRangeCompiler compiler(effective_byte_mapping_config, "replica_disk_canonicalize");
        auto program_or = compiler.Compile(effective_map);
        if (!program_or.ok()) {
          ready_signal->set_value(program_or.status());
          return ready_signal->subscribe();
        }
        loader::ByteRangeMappedSource::Options map_opts = make_mmap_aware_map_options(
            "replica_disk_canonicalize",
            *source_ptr,
            source_location,
            key_.artifact_id,
            effective_byte_mapping_config.enable_direct_write_at);
        std::vector<std::shared_ptr<loader::SeekableSource>> sources;
        sources.emplace_back(std::move(source_ptr));
        auto mapped_or =
            loader::ByteRangeMappedSource::Create(effective_map, *program_or, std::move(sources), std::move(map_opts));
        if (!mapped_or.ok()) {
          ready_signal->set_value(mapped_or.status());
          return ready_signal->subscribe();
        }
        source_ptr = std::move(*mapped_or);
      }
      if (!local_batched_disk_load.has_value() && view_plan_.has_value() && !view_plan_->is_identity) {
        if (!composed_view) {
          source_ptr = loader::make_view_plan_source(
              std::move(source_ptr), view_plan_->selection, effective_byte_mapping_config);
        }
      }
      std::function<absl::Status()> post_load_fn;
      if (view_plan_.has_value() && !view_plan_->is_identity && view_plan_->transform.requires_materialization &&
          transform_placement_ == loading::TransformPlacement::kServer) {
        loader::TransformPlan transform_plan = view_plan_->transform;
        auto mm_shared = memory_manager_;
        post_load_fn = [mm = std::move(mm_shared), transform_plan, target_location]() -> absl::Status {
          auto ptrs = mm->get_pointer(target_location);
          if (ptrs.empty() || ptrs[0] == nullptr) {
            return absl::FailedPreconditionError("View transform requires loaded memory, but no pointer is available");
          }
          const int dev = (target_location == MemoryLocation::GPU) ? mm->get_local_device_id() : -1;
          return loader::execute_transform(transform_plan, target_location, ptrs[0], dev);
        };
      }
      op = memory_manager_->load_async_from_source(
          std::move(source_ptr),
          target_location,
          concurrency,
          std::nullopt,
          std::move(post_load_fn),
          execution_strategy_plan,
          std::nullopt,
          std::move(local_batched_disk_load));
    }
  } else {
    op = folly::makeSemiFuture<absl::Status>(absl::InternalError("Invalid source/target combination."));
  }

  auto ready_signal_copy = ready_signal;
  (void)std::move(op)
      .via(async_runtime_->serial_executor())
      .thenValue([ready_signal_copy](absl::Status status) {
        ready_signal_copy->set_value(status);
        return status;
      })
      .thenError(folly::tag_t<std::exception>{}, [ready_signal_copy](const std::exception& ex) {
        absl::Status status = absl::InternalError(ex.what());
        ready_signal_copy->set_value(status);
        return status;
      });

  return ready_signal->subscribe();
}

std::shared_ptr<common::ReadySignal<absl::Status>> Replica::ready_signal_for(MemoryLocation target_location) const {
  absl::MutexLock lock(&mutex_);
  if (target_location == MemoryLocation::CPU) {
    return cpu_ready_signal_;
  }
  if (target_location == MemoryLocation::GPU) {
    return gpu_ready_signal_;
  }
  return nullptr;
}

absl::Status Replica::mark_loaded(MemoryLocation location) {
  return memory_manager_->set_state(location, MemoryState::LOADED);
}

void Replica::set_ready_signal(MemoryLocation location, const absl::Status& status) {
  absl::MutexLock lock(&mutex_);
  std::shared_ptr<common::ReadySignal<absl::Status>>* slot = nullptr;
  if (location == MemoryLocation::CPU) {
    slot = &cpu_ready_signal_;
  } else if (location == MemoryLocation::GPU) {
    slot = &gpu_ready_signal_;
  } else {
    LOG(WARNING) << "Replica(" << key_.artifact_id << "): Invalid location for ready signal";
    return;
  }
  if (*slot == nullptr) {
    *slot = std::make_shared<common::ReadySignal<absl::Status>>();
  }
  (*slot)->set_value(status);
}

absl::StatusOr<common::memory::MemoryLocation> Replica::find_best_source_for_target(
    common::memory::MemoryLocation target_location) const {
  // Assumes mutex_ is held. (Now called within lock scope in ensure_loaded_async)
  MemoryState cpu_state = memory_manager_->get_state(common::memory::MemoryLocation::CPU);
  MemoryState gpu_state = memory_manager_->get_state(common::memory::MemoryLocation::GPU);

  if (target_location == common::memory::MemoryLocation::CPU) {
    // If GPU is loaded, copy from GPU. Otherwise, load from original source.
    if (gpu_state == MemoryState::LOADED) {
      return common::memory::MemoryLocation::GPU;
    } // Check if original source is valid for loading to CPU
    if (original_source_type_ == common::memory::MemoryLocation::DISK ||
        original_source_type_ == common::memory::MemoryLocation::CPU) {
      return common::memory::MemoryLocation::DISK;
    }
    if (original_source_type_ == common::memory::MemoryLocation::REMOTE) {
      // Add REMOTE->CPU path if P2PLoader supports it later
      return common::memory::MemoryLocation::REMOTE;
    }
    return absl::InternalError("Invalid original source type.");
  }
  if (target_location == common::memory::MemoryLocation::GPU) {
    // If CPU is loaded, copy from CPU. Otherwise, load from original source.
    if (cpu_state == MemoryState::LOADED) {
      return common::memory::MemoryLocation::CPU;
    } // Can load from DISK (via CPU staging implicitly handled by DiskLoader->copy)
    // or directly from REMOTE if P2P->GPU is supported.
    if (original_source_type_ == common::memory::MemoryLocation::DISK ||
        original_source_type_ == common::memory::MemoryLocation::CPU) {
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
    if (original_source_type_ == common::memory::MemoryLocation::REMOTE) {
      return common::memory::MemoryLocation::REMOTE; // Load directly P2P->GPU
    }
    return absl::InternalError("Invalid original source type.");
  }
  return absl::InvalidArgumentError("Invalid target location.");
}

absl::Status Replica::release_memory(MemoryLocation location) {
  absl::MutexLock lock(&mutex_);

  VLOG(2) << "Replica(" << key_.artifact_id << "): Releasing memory for " << location_to_string(location);

  absl::Status status = memory_manager_->release_memory(location);

  // If release was successful and state is now unallocated, invalidate the in-flight signal.
  if (status.ok() && memory_manager_->get_state(location) <= MemoryState::UNALLOCATED) {
    std::shared_ptr<common::ReadySignal<absl::Status>>* relevant_signal = nullptr;
    if (location == MemoryLocation::CPU) {
      relevant_signal = &cpu_ready_signal_;
    } else if (location == MemoryLocation::GPU) {
      relevant_signal = &gpu_ready_signal_;
    }

    if (relevant_signal && *relevant_signal) {
      VLOG(2) << "Replica(" << key_.artifact_id << "): Resetting ready signal for released location "
              << location_to_string(location);
      relevant_signal->reset();
    }
  }
  return status;
}

MemoryState Replica::get_memory_state(MemoryLocation location) const {
  absl::MutexLock lock(&mutex_);
  return memory_manager_->get_state(location);
}

std::vector<void*> Replica::get_data_pointer(MemoryLocation location) const {
  // memory_manager_->get_pointer is thread-safe
  return memory_manager_->get_pointer(location);
}

absl::Status Replica::wait_until_loaded(MemoryLocation location, absl::Duration timeout) {
  // memory_manager_->wait_for_state is thread-safe
  VLOG(1) << "Replica(" << key_.artifact_id << "): Waiting until loaded for " << location_to_string(location)
          << ", timeout=" << timeout;
  return memory_manager_->wait_for_state(location, MemoryState::LOADED, timeout);
}

ReplicaLoadController& Replica::get_memory_manager() const {
  // Returning reference to unique_ptr's managed object. Okay if Replica lifetime > caller usage.
  return *memory_manager_;
}

absl::StatusOr<ExportRegistration> Replica::enable_remote_memory_access(
    MemoryLocation location,
    tensorcast::communicator::engine::Communicator& comm_engine) {
  absl::MutexLock lock(&mutex_);

  // Build full chunk list using UMA snapshot size as the authoritative chunk count.
  const auto chunk_states = memory_manager_->get_chunk_states_uma(common::memory::MemoryLocation::CPU);
  std::vector<uint32_t> chunks(chunk_states.size());
  std::iota(chunks.begin(), chunks.end(), 0);

  // Delegate to chunk-scoped export (also handles GPU with coalesced ranges)
  return memory_manager_->export_chunks_for_p2p(location, chunks, comm_engine);
}

absl::StatusOr<tensorcast::common::ArtifactVerificationInfo> Replica::generate_verification_info(
    common::memory::MemoryLocation location) const {
  absl::MutexLock lock(&mutex_);
  // Check if data is loaded at the specified location
  MemoryState state = memory_manager_->get_state(location);
  if (state != MemoryState::LOADED) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            "Replica data must be loaded at ",
            location_to_string(location),
            " before generating verification info. Current state: ",
            state_to_string(state)));
  }

  // Get data pointers and sizes
  std::vector<void*> data_ptrs = memory_manager_->get_pointer(location);
  if (data_ptrs.empty()) {
    return absl::InternalError("No data pointers available for loaded replica.");
  }

  std::vector<size_t> data_sizes;
  uint64_t artifact_size = memory_manager_->get_artifact_size();

  // Single contiguous buffer for both GPU and CPU under VS
  data_sizes.push_back(artifact_size);

  // Determine device ID for verification
  int device_id = (location == common::memory::MemoryLocation::GPU) ? memory_manager_->get_local_device_id() : -1;

  LOG(INFO) << "Replica(" << key_.artifact_id << "): Generating verification info for " << location_to_string(location)
            << " (" << data_ptrs.size() << " chunks, " << artifact_size << " bytes total)";

  return tensorcast::common::ArtifactVerifier::generate_verification_info(data_ptrs, data_sizes, device_id);
}

absl::Status Replica::verify_artifact_data(
    common::memory::MemoryLocation location,
    const tensorcast::common::ArtifactVerificationInfo& expected_info,
    tensorcast::common::VerificationLevel level) const {
  absl::MutexLock lock(&mutex_);

  // Check if data is loaded at the specified location
  MemoryState state = memory_manager_->get_state(location);
  if (state != MemoryState::LOADED) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            "Replica data must be loaded at ",
            location_to_string(location),
            " before verification. Current state: ",
            state_to_string(state)));
  }

  // Get data pointers and sizes
  std::vector<void*> data_ptrs = memory_manager_->get_pointer(location);
  if (data_ptrs.empty()) {
    return absl::InternalError("No data pointers available for loaded replica.");
  }

  std::vector<size_t> data_sizes;
  uint64_t artifact_size = memory_manager_->get_artifact_size();

  // Single contiguous buffer for both GPU and CPU under VS
  data_sizes.push_back(artifact_size);

  // Determine device ID for verification
  int device_id = (location == common::memory::MemoryLocation::GPU) ? memory_manager_->get_local_device_id() : -1;

  LOG(INFO) << "Replica(" << key_.artifact_id << "): Verifying " << location_to_string(location) << " data at level "
            << static_cast<int>(level);

  return tensorcast::common::ArtifactVerifier::verify_artifact_data(
      data_ptrs, data_sizes, expected_info, level, device_id);
}

absl::Status Replica::verify_key_points(
    common::memory::MemoryLocation location,
    const tensorcast::common::ArtifactVerificationInfo& expected_info) const {
  std::vector<void*> data_ptrs;
  std::vector<size_t> data_sizes;
  int device_id = -1;
  std::shared_ptr<common::memory::GpuDeviceMemory> gpu_allocation_guard;

  {
    absl::MutexLock lock(&mutex_);
    // Check if data is loaded at the specified location
    MemoryState state = memory_manager_->get_state(location);
    if (state != MemoryState::LOADED) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "Replica data must be loaded at ",
              location_to_string(location),
              " before verification. Current state: ",
              state_to_string(state)));
    }

    // Get data pointers and sizes
    data_ptrs = memory_manager_->get_pointer(location);
    if (data_ptrs.empty()) {
      return absl::InternalError("No data pointers available for loaded replica.");
    }

    uint64_t artifact_size = memory_manager_->get_artifact_size();
    data_sizes.push_back(artifact_size);

    if (location == common::memory::MemoryLocation::GPU) {
      device_id = memory_manager_->get_local_device_id();
      gpu_allocation_guard = memory_manager_->get_gpu_allocation_shared();
      if (!gpu_allocation_guard) {
        return absl::FailedPreconditionError("GPU allocation not available for verification");
      }
    }
  }

  LOG(INFO) << "Replica(" << key_.artifact_id << "): Fast key-point verification for " << location_to_string(location);

  return tensorcast::common::ArtifactVerifier::verify_key_points(data_ptrs, data_sizes, expected_info, device_id);
}

absl::Status Replica::disable_remote_memory_access(
    MemoryLocation location,
    tensorcast::communicator::engine::Communicator& comm_engine) {
  absl::MutexLock lock(&mutex_);

  // Use chunk-scoped unexport. Chunks parameter is currently ignored internally.
  std::vector<uint32_t> empty;
  return memory_manager_->unexport_chunks_for_p2p(location, empty, comm_engine);
}

absl::Status Replica::copy_from(const Replica& src) {
  // ────────────────────────────────────────────────────────────────────────────
  // Preconditions & quick checks (no locks held)
  // ────────────────────────────────────────────────────────────────────────────
  if (&src == this) {
    return absl::InvalidArgumentError("Cannot copy_from self");
  }

  // Currently we only support GPU → GPU direct copies.  Extend later for CPU↔GPU.
  if (src.device().type != DeviceType::GPU || device().type != DeviceType::GPU) {
    return absl::UnimplementedError("Replica::copy_from supports only GPU→GPU copies at present");
  }

  // Ensure source GPU memory is LOADED.  We purposefully do not acquire src.mutex_
  // here because ReplicaLoadController::get_state is internally thread-safe.
  if (src.get_memory_state(MemoryLocation::GPU) != MemoryState::LOADED) {
    return absl::FailedPreconditionError("Source replica GPU memory not in LOADED state");
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Prepare destination GPU memory: allocate if necessary.
  // ────────────────────────────────────────────────────────────────────────────
  ReplicaLoadController& dst_mm = get_memory_manager();
  MemoryState dst_state = dst_mm.get_state(MemoryLocation::GPU);
  if (dst_state == MemoryState::UNALLOCATED) {
    absl::Status alloc_st = dst_mm.allocate_memory(MemoryLocation::GPU);
    if (!alloc_st.ok()) {
      return alloc_st;
    }
    dst_state = dst_mm.get_state(MemoryLocation::GPU);
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

} // namespace tensorcast::store::replica
