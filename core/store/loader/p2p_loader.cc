// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/p2p_loader.h"

#include <future>
#include <numeric>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "core/common/memory/distributed_memory_pool.h"
#include "core/common/trace/trace_macros.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/transport/request.h"
#include "core/store/model/memory_manager.h"
#include "core/store/model/memory_state.h"
#include "core/store/model/model_location.h"

namespace stepcast::store {

P2PLoader::P2PLoader(P2PSource source)
    : source_(std::move(source)),
      initialized_(false) // Model size is now part of source_
{
  // Log details about the P2P source configuration received
  LOG(INFO) << "P2PLoader created for remote " << source_.ip << ":" << source_.port
            << ", total_size=" << source_.size_bytes << ", num_keys=" << source_.memory_keys.size();
  if (VLOG_IS_ON(1)) {
    for (size_t i = 0; i < source_.memory_keys.size(); ++i) {
      VLOG(1) << "  Remote Key[" << i << "]: " << source_.memory_keys[i] << ", Size: " << source_.buf_sizes[i];
    }
  }
}

absl::Status P2PLoader::initialize() {
  absl::MutexLock lock(&mutex_);
  if (initialized_) {
    return absl::OkStatus();
  }

  if (source_.size_bytes == 0) {
    return absl::FailedPreconditionError("P2PLoader: Model size from P2P source config is 0.");
  }
  if (source_.memory_keys.empty()) {
    return absl::FailedPreconditionError("P2PLoader: Remote memory keys list is empty.");
  }
  if (source_.memory_keys.size() != source_.buf_sizes.size()) {
    return absl::FailedPreconditionError(
        absl::StrFormat(
            "P2PLoader: Mismatch between number of remote keys (%d) and buffer sizes (%d).",
            source_.memory_keys.size(),
            source_.buf_sizes.size()));
  }

  // Validate that the sum of buffer sizes matches the total model size
  uint64_t sum_of_buffer_sizes = std::accumulate(source_.buf_sizes.begin(), source_.buf_sizes.end(), 0ULL);
  if (sum_of_buffer_sizes != source_.size_bytes) {
    return absl::FailedPreconditionError(
        absl::StrFormat(
            "P2PLoader: Sum of remote buffer sizes (%d) does not match total model size (%d).",
            sum_of_buffer_sizes,
            source_.size_bytes));
  }

  // Note: GPU sources are supported in both RDMA and TCP modes.
  // In TCP mode, the engine will transparently handle GPU->CPU staging.

  // Check if target location requires specific handling (e.g., remote GPU needs local GPU)
  // Basic check: Remote GPU source usually implies local GPU target, but MemoryManager handles allocation.
  // This loader just needs to ensure the transfer happens correctly based on target_location passed to load_async.

  // Mark as initialized
  initialized_ = true;
  LOG(INFO) << "P2PLoader initialized successfully for " << source_.ip << ":" << source_.port << ".";
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> P2PLoader::get_model_size() {
  absl::MutexLock lock(&mutex_);
  if (!initialized_) {
    // Explicit initialization required before getting size
    return absl::FailedPreconditionError("P2PLoader: Must be initialized before calling get_model_size().");
  }
  // Return the size from the source configuration
  if (source_.size_bytes == 0) {
    // This should ideally not happen if initialized_ is true and checks passed in initialize(),
    // but check defensively.
    return absl::InternalError("P2PLoader: Model size in config is zero despite being initialized.");
  }
  return source_.size_bytes;
}

std::future<absl::Status> P2PLoader::load_async(
    std::shared_ptr<MemoryManager> mem_manager,
    ModelLocation target_location,
    [[maybe_unused]] int concurrency) {
  // Only Remote GPU -> Local GPU is supported in this simplified implementation.

  // --- Validate initialization and communicator ---
  {
    absl::MutexLock lock(&mutex_);
    if (!initialized_) {
      return std::async(std::launch::deferred, [] {
        return absl::FailedPreconditionError("P2PLoader: Not initialized before load_async call.");
      });
    }
    if (!source_.comm_engine) {
      return std::async(
          std::launch::deferred, [] { return absl::InternalError("P2PLoader: CommunicateEngine is invalid."); });
    }
  }

  // Validate supported configurations
  bool is_gpu_to_gpu = (target_location == ModelLocation::GPU && source_.location.type == ModelLocation::GPU);
  bool is_to_pageable = (target_location == ModelLocation::PAGEABLE_CPU);

  if (!is_gpu_to_gpu && !is_to_pageable) {
    return std::async(std::launch::deferred, [src = source_.location.type, tgt = target_location] {
      return absl::InvalidArgumentError(
          absl::StrFormat(
              "P2PLoader: Only Remote GPU -> Local GPU or Remote -> PAGEABLE_CPU are supported. Got source=%s, target=%s.",
              location_to_string(src),
              location_to_string(tgt)));
    });
  }

  // Ensure MemoryManager model size matches the expected size.
  const uint64_t expected_size = source_.size_bytes;
  if (mem_manager->get_model_size() == 0) {
    mem_manager->set_model_size(expected_size);
  } else if (mem_manager->get_model_size() != expected_size) {
    return std::async(std::launch::deferred, [expected_size, mgr_size = mem_manager->get_model_size()] {
      return absl::FailedPreconditionError(
          absl::StrFormat(
              "P2PLoader size from config (%d) mismatch with MemoryManager size (%d).", expected_size, mgr_size));
    });
  }

  // -------------------------------------------------------------------------
  // Branch 1: Remote GPU  -> Local GPU (P2P copy directly into GPU memory)
  // -------------------------------------------------------------------------
  if (is_gpu_to_gpu) {
    P2PSource cfg_copy = source_;
    std::shared_ptr<stepcast::communicator::CommunicateEngine> engine_copy = source_.comm_engine;

    return SC_TRACE_ASYNC(
        std::launch::async,
        [mem_manager, cfg = std::move(cfg_copy), engine = std::move(engine_copy), total_size = expected_size]()
            -> absl::Status {
          // Allocate GPU memory if needed.
          MemoryState gpu_state = mem_manager->get_state(ModelLocation::GPU);
          if (gpu_state == MemoryState::UNALLOCATED) {
            absl::Status st = mem_manager->allocate_memory(ModelLocation::GPU);
            if (!st.ok()) {
              return st;
            }
          } else if (gpu_state != MemoryState::ALLOCATED) {
            return absl::FailedPreconditionError(
                absl::StrFormat(
                    "P2PLoader: GPU memory must be in ALLOCATED state, but found %s.", state_to_string(gpu_state)));
          }

          // Transition to LOADING state.
          absl::Status st = mem_manager->set_state(ModelLocation::GPU, MemoryState::LOADING);
          if (!st.ok()) {
            return st;
          }

          // Validate remote buffer configuration.
          if (cfg.memory_keys.size() != 1 || cfg.buf_sizes.size() != 1) {
            absl::Status err = absl::FailedPreconditionError(
                "P2PLoader: Remote GPU -> Local GPU requires exactly one remote key/buffer.");
            ABSL_CHECK_OK(mem_manager->finalize_load_state(ModelLocation::GPU, err));
            return err;
          }
          const std::string& remote_key = cfg.memory_keys[0];
          if (cfg.buf_sizes[0] != total_size) {
            absl::Status err =
                absl::FailedPreconditionError("P2PLoader: Remote buffer size does not match the expected model size.");
            ABSL_CHECK_OK(mem_manager->finalize_load_state(ModelLocation::GPU, err));
            return err;
          }

          // Retrieve local GPU pointer.
          auto local_ptrs = mem_manager->get_pointer(ModelLocation::GPU);
          if (local_ptrs.empty() || local_ptrs[0] == nullptr) {
            absl::Status err = absl::InternalError("P2PLoader: Local GPU pointer is null.");
            ABSL_CHECK_OK(mem_manager->finalize_load_state(ModelLocation::GPU, err));
            return err;
          }
          void* local_gpu_ptr = local_ptrs[0];

          // Perform the P2P read synchronously.
          auto fut = engine->read_tensor(
              remote_key,
              reinterpret_cast<uint64_t>(local_gpu_ptr),
              total_size,
              stepcast::communicator::COMMUNICATE_ENGINE_DEV_GPU,
              mem_manager->get_local_device_id(),
              cfg.ip,
              cfg.port,
              /*remote_offset=*/0);

          auto result = fut.get();
          st = result.status;

          // Finalize MemoryManager state.
          ABSL_CHECK_OK(mem_manager->finalize_load_state(ModelLocation::GPU, st));
          return st;
        });
  }

  // -------------------------------------------------------------------------
  // Branch 2: Remote *    -> Pageable CPU (stage data into CPU pageable memory)
  // -------------------------------------------------------------------------

  if (is_to_pageable) {
    P2PSource cfg_copy = source_;
    std::shared_ptr<stepcast::communicator::CommunicateEngine> engine_copy = source_.comm_engine;

    return SC_TRACE_ASYNC(
        std::launch::async,
        [mem_manager, cfg = std::move(cfg_copy), engine = std::move(engine_copy), total_size = expected_size
         /* concurrency hint currently unused */]() -> absl::Status {
          // Get DVMP instance from memory manager
          auto dvmp = mem_manager->get_dvmp();
          if (!dvmp) {
            return absl::InternalError("P2PLoader: DVMP not available for PAGEABLE_CPU target");
          }

          // Allocate virtual region if needed
          MemoryState cpu_state = mem_manager->get_state(ModelLocation::PAGEABLE_CPU);
          if (cpu_state == MemoryState::UNALLOCATED) {
            absl::Status st = mem_manager->allocate_memory(ModelLocation::PAGEABLE_CPU);
            if (!st.ok() && st.code() != absl::StatusCode::kAlreadyExists) {
              return st;
            }
          } else if (cpu_state != MemoryState::ALLOCATED) {
            return absl::FailedPreconditionError(
                absl::StrFormat(
                    "P2PLoader: PAGEABLE_CPU memory must be in ALLOCATED state, but found %s.",
                    state_to_string(cpu_state)));
          }

          // Transition to LOADING state
          ABSL_CHECK_OK(mem_manager->set_state(ModelLocation::PAGEABLE_CPU, MemoryState::LOADING));

          // Get base address for PAGEABLE_CPU
          auto cpu_ptrs = mem_manager->get_pointer(ModelLocation::PAGEABLE_CPU);
          CHECK(!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr) << "P2PLoader: PAGEABLE_CPU pointer is null";
          void* cpu_base = cpu_ptrs[0];

          // Calculate total chunks
          const size_t chunk_size = 256ULL * 1024 * 1024; // 256MB per chunk
          const size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;

          // Lock all chunks for transfer
          std::vector<uint32_t> chunk_indices;
          chunk_indices.reserve(num_chunks);
          for (size_t i = 0; i < num_chunks; ++i) {
            chunk_indices.push_back(static_cast<uint32_t>(i));
          }

          // Get model ID from memory manager
          const std::string& model_id = mem_manager->get_model_id();

          ABSL_CHECK_OK(dvmp->lock_chunks(model_id, chunk_indices));

          // Transfer data chunk by chunk
          absl::Status transfer_status = absl::OkStatus();

          // For now, do a single transfer of the entire model (chunk-aware transfer to be implemented later)
          if (cfg.memory_keys.size() != 1 || cfg.buf_sizes.size() != 1) {
            transfer_status = absl::FailedPreconditionError("P2PLoader: PAGEABLE_CPU requires single buffer for now");
          } else {
            const std::string& remote_key = cfg.memory_keys[0];
            if (cfg.buf_sizes[0] != total_size) {
              transfer_status = absl::FailedPreconditionError("P2PLoader: Remote buffer size mismatch");
            } else {
              // Perform the P2P read to CPU memory
              auto fut = engine->read_tensor(
                  remote_key,
                  reinterpret_cast<uint64_t>(cpu_base),
                  total_size,
                  stepcast::communicator::COMMUNICATE_ENGINE_DEV_CPU,
                  0, // CPU device ID
                  cfg.ip,
                  cfg.port,
                  /*remote_offset=*/0);

              auto result = fut.get();
              transfer_status = result.status;
            }
          }

          // Unlock chunks with appropriate state
          ABSL_CHECK_OK(dvmp->unlock_chunks(model_id, chunk_indices, false)); // false = HOT state

          // Finalize MemoryManager state
          ABSL_CHECK_OK(mem_manager->finalize_load_state(ModelLocation::PAGEABLE_CPU, transfer_status));
          return transfer_status;
        });
  }

  // Should not reach here – all valid configurations handled above.
  return std::async(std::launch::deferred, [] { return absl::InternalError("P2PLoader: Unexpected code path"); });
}

absl::Status P2PLoader::pull_chunk(std::shared_ptr<MemoryManager> mem_manager, uint32_t chunk_idx) {
  // Validate initialization
  {
    absl::MutexLock lock(&mutex_);
    if (!initialized_) {
      return absl::FailedPreconditionError("P2PLoader: Not initialized before pull_chunk call");
    }
    if (!source_.comm_engine) {
      return absl::InternalError("P2PLoader: CommunicateEngine is invalid");
    }
  }

  // Get DVMP instance
  auto* dvmp = mem_manager->get_dvmp();
  if (!dvmp) {
    return absl::InternalError("P2PLoader: DVMP not available");
  }

  // Get model ID
  const std::string& model_id = mem_manager->get_model_id();

  // Check if chunk is already resident
  absl::Status resident_status = dvmp->ensure_chunk_resident(model_id, chunk_idx);
  if (resident_status.ok()) {
    return absl::OkStatus(); // Chunk already resident
  }
  if (resident_status.code() != stepcast::memory::DistributedMemoryPool::kErrChunkRemote) {
    return resident_status; // Other error (e.g., out of range)
  }

  // Lock the chunk for transfer
  std::vector<uint32_t> chunk_indices = {chunk_idx};
  absl::Status lock_status = dvmp->lock_chunks(model_id, chunk_indices);
  if (!lock_status.ok()) {
    return lock_status;
  }

  // Get base address and calculate chunk offset
  auto cpu_ptrs = mem_manager->get_pointer(ModelLocation::PAGEABLE_CPU);
  CHECK(!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr) << "P2PLoader: PAGEABLE_CPU pointer is null";
  void* cpu_base = cpu_ptrs[0];

  const size_t chunk_size = 256ULL * 1024 * 1024; // 256MB
  size_t chunk_offset = static_cast<size_t>(chunk_idx) * chunk_size;
  void* chunk_addr = static_cast<char*>(cpu_base) + chunk_offset;

  // Calculate transfer size (handle last chunk which may be smaller)
  uint64_t model_size = mem_manager->get_model_size();
  size_t transfer_size = chunk_size;
  if (chunk_offset + chunk_size > model_size) {
    transfer_size = model_size - chunk_offset;
  }

  // Perform the P2P read for this specific chunk
  absl::Status transfer_status = absl::OkStatus();

  // For now, we assume single buffer remote source
  if (source_.memory_keys.size() != 1) {
    transfer_status = absl::UnimplementedError("P2PLoader: Multi-buffer chunk pull not yet implemented");
  } else {
    const std::string& remote_key = source_.memory_keys[0];

    // Perform the P2P read
    auto fut = source_.comm_engine->read_tensor(
        remote_key,
        reinterpret_cast<uint64_t>(chunk_addr),
        transfer_size,
        stepcast::communicator::COMMUNICATE_ENGINE_DEV_CPU,
        0, // CPU device ID
        source_.ip,
        source_.port,
        chunk_offset); // Remote offset for this chunk

    auto result = fut.get();
    transfer_status = result.status;
  }

  // Unlock chunk with appropriate state
  absl::Status unlock_status = dvmp->unlock_chunks(model_id, chunk_indices, false); // false = HOT state
  if (!unlock_status.ok()) {
    LOG(ERROR) << "Failed to unlock chunk " << chunk_idx << " after pull: " << unlock_status;
  }

  return transfer_status;
}

std::future<absl::Status> P2PLoader::load_chunks_async(
    std::shared_ptr<MemoryManager> mem_manager,
    ModelLocation target_location,
    const std::vector<uint32_t>& chunk_indices,
    int concurrency) {
  // Validate inputs
  if (chunk_indices.empty()) {
    return std::async(
        std::launch::deferred, [] { return absl::InvalidArgumentError("No chunks specified for loading"); });
  }

  // Validate initialization
  {
    absl::MutexLock lock(&mutex_);
    if (!initialized_) {
      return std::async(std::launch::deferred, [] {
        return absl::FailedPreconditionError("P2PLoader: Not initialized before load_chunks_async call");
      });
    }
    if (!source_.comm_engine) {
      return std::async(
          std::launch::deferred, [] { return absl::InternalError("P2PLoader: CommunicateEngine is invalid"); });
    }
  }

  // Get model size to validate chunk indices
  uint64_t model_size = source_.size_bytes;
  // Resolve authoritative chunk size
  size_t chunk_size = 256ULL * 1024 * 1024; // Fallback 256 MB
  if (auto um = mem_manager->get_unified_memory()) {
    chunk_size = um->get_chunk_size();
  }
  const size_t total_chunks = (model_size + chunk_size - 1) / chunk_size;

  // Validate chunk indices
  for (uint32_t idx : chunk_indices) {
    if (idx >= total_chunks) {
      return std::async(std::launch::deferred, [idx, total_chunks] {
        return absl::InvalidArgumentError(
            absl::StrFormat("Invalid chunk index %u, total chunks %lu", idx, total_chunks));
      });
    }
  }

  // Ensure model size is set in MemoryManager
  if (mem_manager->get_model_size() == 0) {
    mem_manager->set_model_size(model_size);
  } else if (mem_manager->get_model_size() != model_size) {
    return std::async(std::launch::deferred, [model_size, mgr_size = mem_manager->get_model_size()] {
      return absl::FailedPreconditionError(
          absl::StrFormat("P2PLoader size mismatch: expected %lu, got %lu", model_size, mgr_size));
    });
  }

  // Copy necessary data for async operation
  P2PSource cfg_copy = source_;
  std::shared_ptr<stepcast::communicator::CommunicateEngine> engine_copy = source_.comm_engine;
  std::vector<uint32_t> chunks_copy = chunk_indices;

  // Handle GPU target
  if (target_location == ModelLocation::GPU) {
    return SC_TRACE_ASYNC(
        std::launch::async,
        [mem_manager,
         cfg = std::move(cfg_copy),
         engine = std::move(engine_copy),
         chunks = std::move(chunks_copy),
         chunk_size = chunk_size,
         model_size = model_size]() -> absl::Status {
          // Allocate GPU memory if needed
          MemoryState gpu_state = mem_manager->get_state(ModelLocation::GPU);
          if (gpu_state == MemoryState::UNALLOCATED) {
            absl::Status st = mem_manager->allocate_memory(ModelLocation::GPU);
            if (!st.ok()) {
              return st;
            }
          }

          // Get GPU base pointer
          auto gpu_ptrs = mem_manager->get_pointer(ModelLocation::GPU);
          if (gpu_ptrs.empty() || gpu_ptrs[0] == nullptr) {
            return absl::InternalError("P2PLoader: GPU pointer is null");
          }
          void* gpu_base = gpu_ptrs[0];

          // Sort chunks for efficient transfer
          std::vector<uint32_t> sorted_chunks = chunks;
          std::sort(sorted_chunks.begin(), sorted_chunks.end());

          // For now, only support single remote buffer
          if (cfg.memory_keys.size() != 1) {
            return absl::UnimplementedError("P2PLoader: Multi-buffer chunk transfer not yet implemented");
          }
          const std::string& remote_key = cfg.memory_keys[0];

          // Transfer each chunk
          absl::Status overall_status = absl::OkStatus();
          for (uint32_t chunk_idx : sorted_chunks) {
            size_t chunk_offset = static_cast<size_t>(chunk_idx) * chunk_size;
            size_t transfer_size = std::min(chunk_size, static_cast<size_t>(model_size - chunk_offset));
            void* chunk_addr = static_cast<char*>(gpu_base) + chunk_offset;

            // Perform P2P read for this chunk
            auto fut = engine->read_tensor(
                remote_key,
                reinterpret_cast<uint64_t>(chunk_addr),
                transfer_size,
                stepcast::communicator::COMMUNICATE_ENGINE_DEV_GPU,
                mem_manager->get_local_device_id(),
                cfg.ip,
                cfg.port,
                chunk_offset); // Remote offset

            auto result = fut.get();
            if (!result.status.ok()) {
              overall_status = result.status;
              LOG(ERROR) << "Failed to transfer chunk " << chunk_idx << ": " << result.status;
              break;
            }
          }

          if (overall_status.ok()) {
            if (auto um = mem_manager->get_unified_memory()) {
              auto update_status = um->update_chunk_states(
                  mem_manager->instance_key(),
                  ModelLocation::GPU,
                  sorted_chunks,
                  ChunkState::COPIED_GPU,
                  mem_manager->get_local_device_id());
              if (!update_status.ok()) {
                LOG(WARNING) << "Failed to update chunk states: " << update_status;
              }
            }
          }

          VLOG(1) << "P2PLoader: Transferred " << sorted_chunks.size() << " chunks to GPU";
          return overall_status;
        });
  }

  // Handle PAGEABLE_CPU target
  if (target_location == ModelLocation::PAGEABLE_CPU) {
    return SC_TRACE_ASYNC(
        std::launch::async,
        [mem_manager,
         cfg = std::move(cfg_copy),
         engine = std::move(engine_copy),
         chunks = std::move(chunks_copy),
         chunk_size = chunk_size,
         model_size = model_size]() -> absl::Status {
          // Get DVMP instance
          auto dvmp = mem_manager->get_dvmp();
          if (!dvmp) {
            // Allocate region if not exists
            auto region_or = mem_manager->allocate_pageable_cpu_region();
            if (!region_or.ok() && region_or.status().code() != absl::StatusCode::kAlreadyExists) {
              return region_or.status();
            }
            dvmp = mem_manager->get_dvmp();
            if (!dvmp) {
              return absl::InternalError("Failed to get DVMP instance");
            }
          }

          // Get base address
          auto cpu_ptrs = mem_manager->get_pointer(ModelLocation::PAGEABLE_CPU);
          if (cpu_ptrs.empty() || cpu_ptrs[0] == nullptr) {
            return absl::InternalError("P2PLoader: PAGEABLE_CPU pointer is null");
          }
          void* cpu_base = cpu_ptrs[0];

          // Get model ID
          const std::string& model_id = mem_manager->get_model_id();

          // Sort chunks for efficient transfer
          std::vector<uint32_t> sorted_chunks = chunks;
          std::sort(sorted_chunks.begin(), sorted_chunks.end());

          // Lock chunks before transfer
          absl::Status lock_status = dvmp->lock_chunks(model_id, sorted_chunks);
          if (!lock_status.ok()) {
            return lock_status;
          }

          // For now, only support single remote buffer
          if (cfg.memory_keys.size() != 1) {
            auto unlock_status = dvmp->unlock_chunks(model_id, sorted_chunks, false);
            if (!unlock_status.ok()) {
              LOG(ERROR) << "Failed to unlock chunks: " << unlock_status;
            }
            return absl::UnimplementedError("P2PLoader: Multi-buffer chunk transfer not yet implemented");
          }
          const std::string& remote_key = cfg.memory_keys[0];

          // Transfer each chunk
          absl::Status overall_status = absl::OkStatus();
          for (uint32_t chunk_idx : sorted_chunks) {
            size_t chunk_offset = static_cast<size_t>(chunk_idx) * chunk_size;
            size_t transfer_size = std::min(chunk_size, static_cast<size_t>(model_size - chunk_offset));
            void* chunk_addr = static_cast<char*>(cpu_base) + chunk_offset;

            // Perform P2P read for this chunk
            auto fut = engine->read_tensor(
                remote_key,
                reinterpret_cast<uint64_t>(chunk_addr),
                transfer_size,
                stepcast::communicator::COMMUNICATE_ENGINE_DEV_CPU,
                0, // CPU device ID
                cfg.ip,
                cfg.port,
                chunk_offset); // Remote offset

            auto result = fut.get();
            if (!result.status.ok()) {
              overall_status = result.status;
              LOG(ERROR) << "Failed to transfer chunk " << chunk_idx << ": " << result.status;
              break;
            }
          }

          // Unlock chunks with appropriate state
          absl::Status unlock_status = dvmp->unlock_chunks(model_id, sorted_chunks, /*copied_gpu=*/false); // HOT
          if (!unlock_status.ok()) {
            LOG(ERROR) << "Failed to unlock chunks after transfer: " << unlock_status;
          }

          if (overall_status.ok()) {
            if (auto um = mem_manager->get_unified_memory()) {
              auto update_status = um->update_chunk_states(
                  mem_manager->instance_key(), ModelLocation::PAGEABLE_CPU, sorted_chunks, ChunkState::HOT);
              if (!update_status.ok()) {
                LOG(WARNING) << "Failed to update chunk states: " << update_status;
              }
            }
          }

          VLOG(1) << "P2PLoader: Transferred " << sorted_chunks.size() << " chunks to PAGEABLE_CPU";
          return overall_status;
        });
  }

  return std::async(std::launch::deferred, [target_location] {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "Unsupported target location for chunk loading: %s", location_to_string(target_location).c_str()));
  });
}

} // namespace stepcast::store