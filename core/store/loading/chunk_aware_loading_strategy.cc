// Copyright (c) 2025, TensorCast Team.

#include "core/store/loading/chunk_aware_loading_strategy.h"

#include <algorithm>
#include <filesystem>
#include <future>
#include <optional>
#include <ranges>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/types/span.h"
#include "core/common/async_copy_manager.h"
#include "core/common/cuda_api.h"
#include "core/store/components/global_store_client.h"
#include "core/store/device_registry.h"
#include "core/store/loader/disk_loader.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"
// p2p_loader.h and source.h are included indirectly via other headers. Avoid unused includes.
#include "core/store/replica/memory_manager.h"
#include "core/store/replica/replica_memory_coordinator.h"

namespace tensorcast::store::loading {

ChunkAwareLoadingStrategy::LoadPlan ChunkAwareLoadingStrategy::create_loading_plan(
    const ReplicaKey& key,
    common::memory::MemoryLocation target,
    const replica::ReplicaMemoryCoordinator& memory,
    components::GlobalStoreClient& global_store) {
  LoadPlan plan;
  plan.target = target;

  // Get chunk availability across all locations
  auto chunk_mappings = memory.get_chunk_mappings(key);
  std::vector<uint32_t> missing_chunks;

  const size_t chunk_size = memory.get_chunk_size();
  // Retrieve total artifact size (may fail if not allocated yet)
  size_t artifact_size = 0;
  if (auto size_or = memory.get_artifact_size(key); size_or.ok()) {
    artifact_size = *size_or;
  }
  plan.total_chunks = chunk_mappings.size();
  plan.total_bytes = 0;

  // Determine which chunks are missing at target
  for (size_t i = 0; i < chunk_mappings.size(); ++i) {
    const auto& mapping = chunk_mappings[i];
    bool available_at_target = false;

    if (target == common::memory::MemoryLocation::GPU) {
      // For GPU target, check if chunk is already in the target GPU
      // TODO: Get actual device ID from context
      int device_id = 0; // TODO: Obtain real device id from context
      DeviceKey dev_key = DeviceRegistry::instance().gpu_key(device_id);
      auto it = mapping.gpu_state.find(dev_key);
      available_at_target =
          (it != mapping.gpu_state.end() &&
           (it->second == replica::ChunkState::HOT || it->second == replica::ChunkState::COPIED_GPU));
    } else if (target == common::memory::MemoryLocation::PAGEABLE_CPU) {
      // For CPU target, check if chunk is resident
      available_at_target =
          (mapping.cpu_state == replica::ChunkState::HOT || mapping.cpu_state == replica::ChunkState::COLD);
    }

    if (!available_at_target) {
      missing_chunks.push_back(i);
      if (artifact_size > 0) {
        plan.total_bytes += std::min(chunk_size, artifact_size - i * chunk_size);
      }
    }
  }

  // Group chunks by best available source
  std::map<ChunkSource, std::vector<uint32_t>> source_groups;

  for (uint32_t chunk_idx : missing_chunks) {
    const auto& mapping = chunk_mappings[chunk_idx];
    bool chunk_assigned = false;

    // Priority 1: Local copy (CPU→GPU or GPU→CPU)
    if (target == common::memory::MemoryLocation::GPU &&
        (mapping.cpu_state == replica::ChunkState::HOT || mapping.cpu_state == replica::ChunkState::COLD)) {
      source_groups[ChunkSource::LOCAL_CPU].push_back(chunk_idx);
      chunk_assigned = true;
    } else if (target == common::memory::MemoryLocation::PAGEABLE_CPU) {
      // Check if chunk is available on any local GPU
      for (const auto& [device_key, state] : mapping.gpu_state) {
        if (state == replica::ChunkState::HOT || state == replica::ChunkState::COPIED_GPU) {
          source_groups[ChunkSource::LOCAL_GPU].push_back(chunk_idx);
          chunk_assigned = true;
          break;
        }
      }
    }

    if (!chunk_assigned) {
      // Priority 2: P2P from remote replica
      auto remote_status = global_store.query_chunk_locations(key.artifact_id, {chunk_idx});

      if (remote_status.ok() && !remote_status->empty()) {
        // Convert ChunkLocationInfo → P2PSource candidates (simplified)
        std::vector<P2PSource> candidates;
        for (const auto& loc : *remote_status) {
          P2PSource src;
          src.size_bytes = 0; // Unknown at this stage
          src.ip = loc.node_address;
          src.port = static_cast<uint16_t>(loc.p2p_port);
          src.location.type = common::memory::MemoryLocation::PAGEABLE_CPU; // Assume CPU for now
          candidates.push_back(std::move(src));
        }

        // Select best remote based on criteria
        auto best_remote = select_best_remote(candidates);
        source_groups[ChunkSource::REMOTE_P2P].push_back(chunk_idx);
        plan.remote_sources[chunk_idx] = best_remote;
        chunk_assigned = true;
      }
    }

    if (!chunk_assigned) {
      // Priority 3: Load from disk
      source_groups[ChunkSource::DISK].push_back(chunk_idx);
    }
  }

  // Build optimized plan with batched operations
  for (const auto& [source, chunks] : source_groups) {
    LoadOperation op;
    op.source = source;
    op.chunks = (source == ChunkSource::DISK) ? optimize_chunk_order(chunks) : chunks;
    op.target = target;

    // Set source-specific information
    if (source == ChunkSource::REMOTE_P2P && !chunks.empty()) {
      // For P2P, we'll handle per-chunk sources during execution
      // since different chunks might come from different remotes
    }

    plan.operations.push_back(std::move(op));
  }

  LOG(INFO) << "ChunkAwareLoadingStrategy: Created plan with " << plan.operations.size() << " operations for "
            << missing_chunks.size() << " missing chunks";

  return plan;
}

std::future<absl::Status> ChunkAwareLoadingStrategy::execute_plan(
    const LoadPlan& plan,
    replica::ReplicaMemoryCoordinator& memory,
    const std::shared_ptr<store::replica::MemoryManager>& mem_manager) {
  return std::async(std::launch::async, [plan, &memory, mem_manager]() {
    return execute_plan_with_progress(plan, memory, mem_manager, [](size_t, size_t) {}); // No-op progress callback
  });
}

absl::Status ChunkAwareLoadingStrategy::execute_plan_with_progress(
    const LoadPlan& plan,
    replica::ReplicaMemoryCoordinator& memory,
    const std::shared_ptr<store::replica::MemoryManager>& mem_manager,
    const ProgressCallback& progress_cb) {
  if (plan.operations.empty()) {
    return absl::OkStatus(); // Nothing to do
  }

  size_t total_chunks = 0;
  for (const auto& op : plan.operations) {
    total_chunks += op.chunks.size();
  }

  size_t completed_chunks = 0;
  absl::Status overall_status = absl::OkStatus();

  // Execute operations sequentially for now
  // TODO: Execute independent operations in parallel
  for (const auto& op : plan.operations) {
    absl::Status op_status = ChunkAwareLoadingStrategy::execute_operation(
        op, memory, mem_manager, [&completed_chunks, total_chunks, progress_cb](size_t op_completed, size_t) {
          completed_chunks += op_completed;
          progress_cb(completed_chunks, total_chunks);
        });

    if (!op_status.ok()) {
      LOG(ERROR) << "Failed to execute operation from " << static_cast<int>(op.source) << ": " << op_status;
      overall_status = op_status;
      break;
    }
  }

  // Post-loading memory management
  if (overall_status.ok()) {
    const auto& replica_key = mem_manager->replica_key();

    // Check if this was a DRAM load
    if (plan.target == common::memory::MemoryLocation::PAGEABLE_CPU) {
      // After initial DRAM load, mark configured ratio as preemptible
      float preemptible_ratio = 0.5F; // TODO: Make configurable
      auto preempt_status = mem_manager->mark_cpu_preemptible(preemptible_ratio);
      if (!preempt_status.ok()) {
        LOG(WARNING) << "Failed to mark CPU chunks as preemptible: " << preempt_status;
      } else {
        VLOG(1) << "Marked " << (preemptible_ratio * 100) << "% of CPU chunks as preemptible after DRAM load";
      }
    } else if (plan.target == common::memory::MemoryLocation::GPU) {
      // Check if GPU loading is complete
      int device_id = mem_manager->get_local_device_id();
      if (device_id >= 0 && memory.is_gpu_loading_complete(replica_key, device_id)) {
        // GPU loading complete, mark all DRAM chunks as preemptible
        auto preempt_status = mem_manager->mark_cpu_preemptible(1.0F);
        if (!preempt_status.ok()) {
          LOG(WARNING) << "Failed to mark all CPU chunks as preemptible: " << preempt_status;
        } else {
          VLOG(1) << "Marked all CPU chunks as preemptible after GPU loading completion";
        }
      }
    }
  }

  return overall_status;
}

absl::Status ChunkAwareLoadingStrategy::execute_operation(
    const LoadOperation& op,
    replica::ReplicaMemoryCoordinator& memory,
    const std::shared_ptr<store::replica::MemoryManager>& mem_manager,
    const ProgressCallback& progress_cb) {
  switch (op.source) {
    case ChunkSource::LOCAL_CPU:
      return execute_local_cpu_copy(op.chunks, op.target, memory, mem_manager, progress_cb);

    case ChunkSource::LOCAL_GPU:
      // TODO: Implement GPU->CPU copy
      return absl::UnimplementedError("GPU to CPU copy not yet implemented");

    case ChunkSource::REMOTE_P2P:
      return execute_p2p_transfer(op, memory, mem_manager, progress_cb);

    case ChunkSource::DISK:
      return execute_disk_load(op.chunks, op.target, memory, mem_manager, progress_cb);

    default:
      return absl::InternalError("Unknown chunk source");
  }
}

absl::Status ChunkAwareLoadingStrategy::execute_local_cpu_copy(
    const std::vector<uint32_t>& chunks,
    common::memory::MemoryLocation target,
    replica::ReplicaMemoryCoordinator& memory,
    const std::shared_ptr<store::replica::MemoryManager>& mem_manager,
    const ProgressCallback& progress_cb) {
  if (target != common::memory::MemoryLocation::GPU) {
    return absl::InvalidArgumentError("Local CPU copy only supports GPU target");
  }

  // Lock chunks for transfer
  absl::Status lock_status = memory.lock_chunks_for_transfer(
      mem_manager->replica_key(), common::memory::MemoryLocation::PAGEABLE_CPU, target, chunks);
  if (!lock_status.ok()) {
    return lock_status;
  }

  // Get pointers
  auto cpu_ptrs = mem_manager->get_pointer(common::memory::MemoryLocation::PAGEABLE_CPU);
  auto gpu_ptrs = mem_manager->get_pointer(common::memory::MemoryLocation::GPU);

  if (cpu_ptrs.empty() || gpu_ptrs.empty()) {
    return absl::InternalError("Memory pointers not available");
  }

  void* cpu_base = cpu_ptrs[0];
  void* gpu_base = gpu_ptrs[0];
  const size_t chunk_size = memory.get_chunk_size();

  // Get total artifact size
  size_t artifact_size = 0;
  if (auto size_or = memory.get_artifact_size(mem_manager->replica_key()); size_or.ok()) {
    artifact_size = *size_or;
  } else {
    return size_or.status();
  }

  // Copy each chunk using ACM with UMA advancement in callbacks
  size_t completed = 0;
  absl::Status copy_status = absl::OkStatus();
  absl::Mutex first_err_mu;
  absl::Status first_err = absl::OkStatus();
  common::CopyHandle last_handle;
  bool last_set = false;

  for (uint32_t chunk_idx : chunks) {
    size_t offset = chunk_idx * chunk_size;
    size_t size = std::min(chunk_size, artifact_size - offset);

    void* src = static_cast<char*>(cpu_base) + offset;
    void* dst = static_cast<char*>(gpu_base) + offset;

    // Schedule H2D via ACM; UMA state advanced in callback for this chunk
    common::HostRegion h{.base = src, .length = size, .pinned = false};
    common::DeviceRegion d{.device_id = mem_manager->get_local_device_id(), .dev_ptr = dst, .length = size};
    auto on_done = [&memory,
                    key = mem_manager->replica_key(),
                    dev = mem_manager->get_local_device_id(),
                    chunk_idx,
                    &first_err_mu,
                    &first_err]() {
      auto st = memory.update_chunk_states(
          key,
          common::memory::MemoryLocation::GPU,
          std::vector<uint32_t>{chunk_idx},
          replica::ChunkState::COPIED_GPU,
          dev);
      if (!st.ok()) {
        absl::MutexLock lk(&first_err_mu);
        if (first_err.ok())
          first_err = st;
      }
    };
    common::CopyOptions opts{.tracing_stage = "H2D/Copy", .callbacks = {.on_copy_done = on_done}};
    auto hdl_or = common::AsyncCopyManager::instance().submit_h2d(h, d, opts);
    if (!hdl_or.ok()) {
      copy_status = hdl_or.status();
      LOG(ERROR) << "Failed to schedule H2D for chunk " << chunk_idx << ": " << copy_status;
      break;
    }
    last_handle = std::move(*hdl_or);
    last_set = true;

    completed++;
    progress_cb(completed, chunks.size());
  }

  // Wait for the last submitted copy to complete and propagate any UMA callback error
  if (copy_status.ok() && last_set) {
    auto wst = last_handle.wait();
    if (!wst.ok()) {
      copy_status = wst;
    } else {
      absl::MutexLock lk(&first_err_mu);
      if (!first_err.ok()) {
        copy_status = first_err;
      }
    }
  }

  return copy_status;
}

absl::Status ChunkAwareLoadingStrategy::execute_p2p_transfer(
    const LoadOperation& op,
    replica::ReplicaMemoryCoordinator& memory,
    const std::shared_ptr<store::replica::MemoryManager>& mem_manager,
    const ProgressCallback& progress_cb) {
  // Group chunks by remote source
  std::map<std::string, std::vector<uint32_t>> chunks_by_source;

  for (uint32_t chunk_idx : op.chunks) {
    // This is a simplified version - in reality we'd look up
    // the specific P2P source for each chunk
    chunks_by_source["default"].push_back(chunk_idx);
  }

  size_t completed = 0;
  absl::Status overall_status = absl::OkStatus();

  // Execute P2P transfers
  for (const auto& [source_key, chunks] : chunks_by_source) {
    // Create P2P loader (placeholder using DiskLoader)
    auto loader =
        std::make_unique<DiskLoader>(DiskSource{.path = std::filesystem::path{}, .expected_size = std::nullopt});

    // Initialize loader
    absl::Status init_status = loader->initialize();
    if (!init_status.ok()) {
      overall_status = init_status;
      break;
    }

    // Load chunks via open_source + MemoryManager
    auto src_or = loader->open_source();
    if (!src_or.ok()) {
      overall_status = src_or.status();
      break;
    }
    auto future = mem_manager->load_async_from_source(std::move(*src_or), op.target, 4, absl::MakeSpan(chunks));

    absl::Status load_status = future.get();
    if (!load_status.ok()) {
      overall_status = load_status;
      break;
    }

    completed += chunks.size();
    progress_cb(completed, op.chunks.size());

    if (load_status.ok()) {
      // Update chunk states after successful P2P transfer
      auto state = (op.target == common::memory::MemoryLocation::GPU) ? replica::ChunkState::COPIED_GPU
                                                                      : replica::ChunkState::HOT;
      auto upd_status = memory.update_chunk_states(
          mem_manager->replica_key(),
          op.target,
          chunks,
          state,
          (op.target == common::memory::MemoryLocation::GPU) ? std::optional<int>(mem_manager->get_local_device_id())
                                                             : std::nullopt);
      (void)upd_status; // ignore result for now
    }
  }

  return overall_status;
}

absl::Status ChunkAwareLoadingStrategy::execute_disk_load(
    const std::vector<uint32_t>& chunks,
    common::memory::MemoryLocation target,
    replica::ReplicaMemoryCoordinator& memory,
    const std::shared_ptr<store::replica::MemoryManager>& mem_manager,
    const ProgressCallback& progress_cb) {
  // Create disk loader
  // Note: We'd need the actual disk source path from MemoryManager
  // For now, use the instance key to construct the path
  const std::string& artifact_id = mem_manager->replica_key().artifact_id;
  if (absl::StartsWith(artifact_id, "mi2:")) {
    return absl::FailedPreconditionError(
        "ChunkAwareLoadingStrategy: disk load path is undefined for content-addressed artifact_id; require GS routing");
  }
  std::filesystem::path disk_path = artifact_id;
  auto loader = std::make_unique<DiskLoader>(DiskSource{.path = disk_path, .expected_size = std::nullopt});

  // Initialize loader
  absl::Status init_status = loader->initialize();
  if (!init_status.ok()) {
    return init_status;
  }

  // Load chunks via open_source + MemoryManager
  auto src_or = loader->open_source();
  if (!src_or.ok()) {
    return src_or.status();
  }
  auto future = mem_manager->load_async_from_source(std::move(*src_or), target, 4, absl::MakeSpan(chunks));

  absl::Status load_status = future.get();

  if (load_status.ok()) {
    // Update chunk states
    auto state =
        (target == common::memory::MemoryLocation::GPU) ? replica::ChunkState::COPIED_GPU : replica::ChunkState::HOT;
    auto upd_status = memory.update_chunk_states(
        mem_manager->replica_key(),
        target,
        chunks,
        state,
        (target == common::memory::MemoryLocation::GPU) ? std::optional<int>(mem_manager->get_local_device_id())
                                                        : std::nullopt);
    (void)upd_status; // suppress unused warning
    progress_cb(chunks.size(), chunks.size());
  }

  return load_status;
}

P2PSource ChunkAwareLoadingStrategy::select_best_remote(const std::vector<P2PSource>& candidates) {
  if (candidates.empty()) {
    return P2PSource{};
  }

  // Use C++20 ranges algorithm for clarity and modern style
  auto best_it = std::ranges::min_element(candidates, [](const P2PSource& a, const P2PSource& b) {
    // Placeholder comparison - would use actual load metrics
    return a.size_bytes < b.size_bytes;
  });

  return (best_it != candidates.end()) ? *best_it : P2PSource{};
}

std::vector<uint32_t> ChunkAwareLoadingStrategy::optimize_chunk_order(const std::vector<uint32_t>& chunks) {
  std::vector<uint32_t> sorted = chunks;
  std::ranges::sort(sorted);
  return sorted;
}

} // namespace tensorcast::store::loading
