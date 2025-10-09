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
#include "core/store/components/global_store_client.h"
#include "core/store/device_registry.h"
#include "core/store/loader/disk_loader.h"
// p2p_loader.h and source.h are included indirectly via other headers. Avoid unused includes.
#include "core/store/replica/replica_load_controller.h"
#include "core/store/replica/unified_memory_authority.h"
#include "gsl/pointers"

namespace tensorcast::store::loading {

ChunkAwareLoadingStrategy::LoadPlan ChunkAwareLoadingStrategy::create_loading_plan(
    const ReplicaKey& key,
    common::memory::MemoryLocation target,
    const replica::UnifiedMemoryAuthority& memory,
    components::GlobalStoreClient& global_store) {
  LoadPlan plan;
  plan.target = target;

  // Determine missing chunks using UMA record-based query
  const size_t chunk_size = memory.get_artifact_chunk_bytes();
  size_t artifact_size = 0;
  if (auto size_or = memory.get_artifact_size(key); size_or.ok())
    artifact_size = *size_or;
  int device_id = 0; // TODO: Obtain actual device id from context
  std::vector<uint32_t> missing_chunks = memory.get_missing_chunks(key, target, device_id);
  // Compute total_chunks from artifact_size when available
  plan.total_chunks = (artifact_size > 0 && chunk_size > 0) ? ((artifact_size + chunk_size - 1) / chunk_size) : 0;
  plan.total_bytes = 0;
  for (uint32_t i : missing_chunks) {
    if (artifact_size > 0)
      plan.total_bytes += std::min(chunk_size, artifact_size - static_cast<size_t>(i) * chunk_size);
  }

  // Group chunks by best available source
  std::map<ChunkSource, std::vector<uint32_t>> source_groups;

  for (uint32_t chunk_idx : missing_chunks) {
    bool chunk_assigned = false;
    // Ask UMA for best local source
    auto best = memory.get_best_source_for_chunk(key, chunk_idx, target);
    if (best.type == replica::UnifiedMemoryAuthority::ChunkSource::LOCAL_CPU) {
      source_groups[ChunkSource::LOCAL_CPU].push_back(chunk_idx);
      chunk_assigned = true;
    } else if (best.type == replica::UnifiedMemoryAuthority::ChunkSource::LOCAL_GPU) {
      source_groups[ChunkSource::LOCAL_GPU].push_back(chunk_idx);
      chunk_assigned = true;
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
          src.location.type = common::memory::MemoryLocation::CPU; // Assume CPU for now
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
    replica::UnifiedMemoryAuthority& memory,
    const std::shared_ptr<store::replica::ReplicaLoadController>& mem_manager) {
  return std::async(std::launch::async, [plan, &memory, mem_manager]() {
    return execute_plan_with_progress(plan, memory, mem_manager, [](size_t, size_t) {}); // No-op progress callback
  });
}

absl::Status ChunkAwareLoadingStrategy::execute_plan_with_progress(
    const LoadPlan& plan,
    replica::UnifiedMemoryAuthority& memory,
    const std::shared_ptr<store::replica::ReplicaLoadController>& mem_manager,
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
    if (plan.target == common::memory::MemoryLocation::CPU) {
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
    replica::UnifiedMemoryAuthority& memory,
    const std::shared_ptr<store::replica::ReplicaLoadController>& mem_manager,
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
    replica::UnifiedMemoryAuthority& memory,
    const std::shared_ptr<store::replica::ReplicaLoadController>& mem_manager,
    const ProgressCallback& progress_cb) {
  if (target != common::memory::MemoryLocation::GPU) {
    return absl::InvalidArgumentError("Local CPU copy only supports GPU target");
  }

  // Plan transfer via UMA (transactional)
  const auto device_id = mem_manager->get_local_device_id();
  auto plan_or = memory.plan_load(
      mem_manager->replica_key(),
      common::memory::MemoryLocation::GPU,
      std::optional<int>(device_id),
      std::optional<absl::Span<const uint32_t>>(absl::MakeSpan(chunks)));
  if (!plan_or.ok()) {
    return plan_or.status();
  }
  const auto plan = *plan_or;
  if (plan.chunk_indices.empty()) {
    return absl::OkStatus(); // Nothing to do
  }

  // Get pointers
  auto cpu_ptrs = mem_manager->get_pointer(common::memory::MemoryLocation::CPU);
  auto gpu_ptrs = mem_manager->get_pointer(common::memory::MemoryLocation::GPU);

  if (cpu_ptrs.empty() || gpu_ptrs.empty() || cpu_ptrs[0] == nullptr || gpu_ptrs[0] == nullptr) {
    return absl::InternalError("Memory pointers not available");
  }
  gsl::not_null<void*> cpu_base{cpu_ptrs[0]};
  gsl::not_null<void*> gpu_base{gpu_ptrs[0]};

  // Copy plan ranges using ACM; UMA ledger is finalized via commit() below.
  size_t completed = 0;
  absl::Status copy_status = absl::OkStatus();
  absl::Status first_err = absl::OkStatus();
  common::CopyHandle last_handle;
  bool last_set = false;

  for (const auto& range : plan.ranges) {
    const uint64_t offset = range.first;
    const size_t size = range.second;
    void* src = static_cast<char*>(cpu_base.get()) + offset;
    void* dst = static_cast<char*>(gpu_base.get()) + offset;

    // Schedule H2D via ACM; UMA state will be recorded on commit()
    common::HostRegion h{.base = src, .length = size, .pinned = false};
    common::DeviceRegion d{.device_id = device_id, .dev_ptr = dst, .length = size};
    common::CopyOptions opts{.tracing_stage = "H2D/Copy"};
    auto hdl_or = common::AsyncCopyManager::instance().submit_h2d(h, d, opts);
    if (!hdl_or.ok()) {
      copy_status = hdl_or.status();
      LOG(ERROR) << "Failed to schedule H2D for range off=" << offset << " len=" << size << ": " << copy_status;
      break;
    }
    last_handle = std::move(*hdl_or);
    last_set = true;

    completed += 1;
    progress_cb(completed, plan.ranges.size());
  }

  // Wait for the last submitted copy to complete and propagate any UMA callback error
  if (copy_status.ok() && last_set) {
    auto wst = last_handle.wait();
    if (!wst.ok()) {
      copy_status = wst;
    }
  }

  if (!copy_status.ok()) {
    // Abort UMA session on failure
    (void)memory.abort(plan.session_id);
    return copy_status;
  }

  // Commit UMA plan for all chunks covered by this copy
  absl::Status commit_status = memory.commit(
      plan.session_id, common::memory::MemoryLocation::GPU, absl::MakeSpan(plan.chunk_indices), device_id);
  return commit_status;
}

absl::Status ChunkAwareLoadingStrategy::execute_p2p_transfer(
    const LoadOperation& op,
    replica::UnifiedMemoryAuthority& memory,
    const std::shared_ptr<store::replica::ReplicaLoadController>& mem_manager,
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

    // Plan transfer through UMA first
    auto plan_or = memory.plan_load(
        mem_manager->replica_key(),
        op.target,
        (op.target == common::memory::MemoryLocation::GPU) ? std::optional<int>(mem_manager->get_local_device_id())
                                                           : std::optional<int>(std::nullopt),
        std::optional<absl::Span<const uint32_t>>(absl::MakeSpan(chunks)));
    if (!plan_or.ok()) {
      overall_status = plan_or.status();
      break;
    }
    const auto plan = *plan_or;
    if (plan.chunk_indices.empty()) {
      continue; // Nothing to do for this source
    }

    // Load chunks via open_source + ReplicaLoadController
    auto src_or = loader->open_source();
    if (!src_or.ok()) {
      overall_status = src_or.status();
      break;
    }
    auto future =
        mem_manager->load_async_from_source(std::move(*src_or), op.target, 4, absl::MakeSpan(plan.chunk_indices));

    absl::Status load_status = future.get();
    if (!load_status.ok()) {
      overall_status = load_status;
      break;
    }

    completed += plan.chunk_indices.size();
    progress_cb(completed, op.chunks.size());
    if (load_status.ok()) {
      // Commit UMA plan
      auto cst = memory.commit(
          plan.session_id,
          op.target,
          absl::MakeSpan(plan.chunk_indices),
          (op.target == common::memory::MemoryLocation::GPU) ? std::optional<int>(mem_manager->get_local_device_id())
                                                             : std::optional<int>(std::nullopt));
      if (!cst.ok()) {
        overall_status = cst;
        break;
      }
    } else {
      (void)memory.abort(plan.session_id);
    }
  }

  return overall_status;
}

absl::Status ChunkAwareLoadingStrategy::execute_disk_load(
    const std::vector<uint32_t>& chunks,
    common::memory::MemoryLocation target,
    replica::UnifiedMemoryAuthority& memory,
    const std::shared_ptr<store::replica::ReplicaLoadController>& mem_manager,
    const ProgressCallback& progress_cb) {
  // Create disk loader
  // Note: We'd need the actual disk source path from ReplicaLoadController
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

  // Plan transfer via UMA
  auto plan_or = memory.plan_load(
      mem_manager->replica_key(),
      target,
      (target == common::memory::MemoryLocation::GPU) ? std::optional<int>(mem_manager->get_local_device_id())
                                                      : std::optional<int>(std::nullopt),
      std::optional<absl::Span<const uint32_t>>(absl::MakeSpan(chunks)));
  if (!plan_or.ok()) {
    return plan_or.status();
  }
  const auto plan = *plan_or;
  if (plan.chunk_indices.empty()) {
    return absl::OkStatus();
  }

  // Load chunks via open_source + ReplicaLoadController
  auto src_or = loader->open_source();
  if (!src_or.ok()) {
    return src_or.status();
  }
  auto future = mem_manager->load_async_from_source(std::move(*src_or), target, 4, absl::MakeSpan(plan.chunk_indices));

  absl::Status load_status = future.get();

  if (load_status.ok()) {
    // Commit UMA plan on success
    auto cst = memory.commit(
        plan.session_id,
        target,
        absl::MakeSpan(plan.chunk_indices),
        (target == common::memory::MemoryLocation::GPU) ? std::optional<int>(mem_manager->get_local_device_id())
                                                        : std::optional<int>(std::nullopt));
    if (!cst.ok()) {
      return cst;
    }
    progress_cb(plan.chunk_indices.size(), plan.chunk_indices.size());
  } else {
    (void)memory.abort(plan.session_id);
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
