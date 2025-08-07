// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/p2p_loader.h"

#include <algorithm>
#include <future>
#include <memory>
#include <vector>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/store/loader/dvmp_mapped_sink.h"
#include "core/store/loader/gpu_memory_sink.h"
#include "core/store/loader/pump.h"
#include "core/store/loader/remote_key_source.h"
#include "core/store/loader/streaming_buffer_adapter.h"
#include "core/store/loader/unified_memory_sink.h"
#include "core/store/model/memory_manager.h"
#include "core/store/model/memory_state.h"

namespace stepcast::store {

P2PLoader::P2PLoader(P2PSource source) : source_(std::move(source)), initialized_(false) {}

absl::Status P2PLoader::initialize() {
  absl::MutexLock lock(&mutex_);

  if (initialized_) {
    return absl::OkStatus();
  }

  // Validate source configuration
  if (!source_.comm_engine) {
    return absl::InvalidArgumentError("CommunicateEngine is required");
  }

  if (source_.size_bytes == 0) {
    return absl::InvalidArgumentError("Model size must be greater than 0");
  }

  if (source_.memory_keys.empty()) {
    return absl::InvalidArgumentError("Memory keys are required");
  }

  LOG(INFO) << "P2PLoader initialized: size=" << source_.size_bytes << " bytes, keys=" << source_.memory_keys.size();

  initialized_ = true;
  return absl::OkStatus();
}

std::future<absl::Status> P2PLoader::load_async(
    std::shared_ptr<MemoryManager> mem_manager,
    ModelLocation target_location,
    int concurrency) {
  return std::async(std::launch::async, [this, mem_manager, target_location, concurrency]() mutable -> absl::Status {
    // Initialize if needed
    if (!initialized_) {
      auto status = initialize();
      if (!status.ok()) {
        return status;
      }
    }

    // Set default concurrency (lower for network transfers)
    if (concurrency <= 0) {
      concurrency = 2;
    }

    // Allocate target memory
    if (mem_manager->get_state(target_location) == MemoryState::UNALLOCATED) {
      auto status = mem_manager->allocate_memory(target_location);
      if (!status.ok()) {
        return status;
      }
    }

    // Create remote source
    store::loader::RemoteKeySource::Options source_opts{
        .comm_engine = source_.comm_engine,
        .memory_keys = source_.memory_keys,
        .buffer_sizes = source_.buf_sizes,
        .ip = source_.ip,
        .port = source_.port,
        .total_size = source_.size_bytes};
    store::loader::RemoteKeySource remote_source(source_opts);

    // GPU path - use streaming buffer and pump
    if (target_location == ModelLocation::GPU) {
      // Ensure streaming buffer (smaller for network)
      size_t chunk_size = mem_manager->get_pool_chunk_size();
      size_t num_chunks = std::min<size_t>(4, (source_.size_bytes + chunk_size - 1) / chunk_size);
      auto status = mem_manager->ensure_streaming_buffer(num_chunks);
      if (!status.ok()) {
        return status;
      }

      auto spb = mem_manager->get_streaming_buffer();
      if (!spb) {
        return absl::InternalError("Failed to get streaming buffer");
      }

      // Create adapter and sink pipeline
      store::loader::StreamingBufferAdapter buffer_adapter(spb);
      auto gpu_ptr = mem_manager->get_pointer(ModelLocation::GPU);
      if (gpu_ptr.empty()) {
        return absl::InternalError("GPU memory not allocated");
      }

      store::loader::GPUMemorySink::Options gpu_opts{
          .gpu_base_ptr = gpu_ptr[0],
          .total_size = source_.size_bytes,
          .chunk_size = mem_manager->get_pool_chunk_size(),
          .device_id = 0};
      store::loader::GPUMemorySink gpu_sink(gpu_opts);

      store::loader::UnifiedMemorySink::Options unified_opts{
          .inner_sink = std::make_shared<store::loader::GPUMemorySink>(std::move(gpu_sink)),
          .memory_manager = mem_manager,
          .target_location = ModelLocation::GPU,
          .device_id = 0,
          .chunk_indices = std::nullopt};
      store::loader::UnifiedMemorySink unified_sink(unified_opts);

      // Run pump
      CHECK_OK(mem_manager->set_state(ModelLocation::GPU, MemoryState::LOADING));
      status = store::loader::pump(remote_source, unified_sink, buffer_adapter, concurrency);

      if (!status.ok()) {
        CHECK_OK(mem_manager->set_state(ModelLocation::GPU, MemoryState::FAILED));
        return status;
      }

      CHECK_OK(mem_manager->set_state(ModelLocation::GPU, MemoryState::LOADED));
      LOG(INFO) << "P2P transfer to GPU complete: " << source_.size_bytes << " bytes";
      return absl::OkStatus();
    }

    // CPU path
    if (target_location == ModelLocation::PAGEABLE_CPU) {
      auto cpu_ptr = mem_manager->get_pointer(ModelLocation::PAGEABLE_CPU);
      if (cpu_ptr.empty()) {
        return absl::InternalError("CPU memory not allocated");
      }

      store::loader::DVMPMappedSink::Options dvmp_opts{
          .base_addr = cpu_ptr[0],
          .total_size = source_.size_bytes,
          .partition_paths = {},
          .partition_sizes = {},
          .populate_pages = true};
      store::loader::DVMPMappedSink dvmp_sink(dvmp_opts);

      store::loader::UnifiedMemorySink::Options unified_opts{
          .inner_sink = std::make_shared<store::loader::DVMPMappedSink>(std::move(dvmp_sink)),
          .memory_manager = mem_manager,
          .target_location = ModelLocation::PAGEABLE_CPU,
          .device_id = std::nullopt,
          .chunk_indices = std::nullopt};
      store::loader::UnifiedMemorySink unified_sink(unified_opts);

      CHECK_OK(mem_manager->set_state(ModelLocation::PAGEABLE_CPU, MemoryState::LOADING));

      // Need a small buffer for pump
      size_t num_chunks = 2;
      auto status = mem_manager->ensure_streaming_buffer(num_chunks);
      if (!status.ok()) {
        return status;
      }

      auto spb = mem_manager->get_streaming_buffer();
      if (!spb) {
        return absl::InternalError("Failed to get streaming buffer");
      }

      store::loader::StreamingBufferAdapter buffer_adapter(spb);
      status = store::loader::pump(remote_source, unified_sink, buffer_adapter, concurrency);

      if (!status.ok()) {
        CHECK_OK(mem_manager->set_state(ModelLocation::PAGEABLE_CPU, MemoryState::FAILED));
        return status;
      }

      CHECK_OK(mem_manager->set_state(ModelLocation::PAGEABLE_CPU, MemoryState::LOADED));
      LOG(INFO) << "P2P transfer to CPU complete: " << source_.size_bytes << " bytes";
      return absl::OkStatus();
    }

    return absl::UnimplementedError("Unsupported target location");
  });
}

std::future<absl::Status> P2PLoader::load_chunks_async(
    const std::shared_ptr<MemoryManager>& mem_manager,
    ModelLocation target_location,
    const std::vector<uint32_t>& chunk_indices,
    int concurrency) {
  return std::async(
      std::launch::async, [this, mem_manager, target_location, chunk_indices, concurrency]() mutable -> absl::Status {
        // Ensure initialization
        if (!initialized_) {
          auto status = initialize();
          if (!status.ok()) {
            return status;
          }
        }

        // Default single-thread for chunk transfers if not specified
        int effective_concurrency = (concurrency <= 0) ? 1 : concurrency;

        // Make sure destination memory is allocated
        if (mem_manager->get_state(target_location) == MemoryState::UNALLOCATED) {
          auto status = mem_manager->allocate_memory(target_location);
          if (!status.ok()) {
            return status;
          }
        }

        // Construct RemoteKeySource
        store::loader::RemoteKeySource::Options src_opts{
            .comm_engine = source_.comm_engine,
            .memory_keys = source_.memory_keys,
            .buffer_sizes = source_.buf_sizes,
            .ip = source_.ip,
            .port = source_.port,
            .total_size = source_.size_bytes};
        store::loader::RemoteKeySource remote_source(src_opts);

        // Build byte ranges corresponding to requested chunks
        size_t chunk_size = mem_manager->get_pool_chunk_size();
        std::vector<std::pair<uint64_t, size_t>> ranges;
        for (auto idx : chunk_indices) {
          uint64_t offset = static_cast<uint64_t>(idx) * chunk_size;
          size_t size = std::min(chunk_size, static_cast<size_t>(source_.size_bytes - offset));
          ranges.emplace_back(offset, size);
        }

        absl::Status status;
        if (target_location == ModelLocation::GPU) {
          // Prepare buffering
          size_t num_chunks = std::min<size_t>(2, chunk_indices.size());
          status = mem_manager->ensure_streaming_buffer(num_chunks);
          if (!status.ok()) {
            return status;
          }

          auto spb = mem_manager->get_streaming_buffer();
          if (!spb) {
            return absl::InternalError("Failed to get streaming buffer");
          }
          store::loader::StreamingBufferAdapter buffer_adapter(spb);

          auto gpu_ptr = mem_manager->get_pointer(ModelLocation::GPU);
          if (gpu_ptr.empty()) {
            return absl::InternalError("GPU memory not allocated");
          }

          store::loader::GPUMemorySink::Options gpu_opts{
              .gpu_base_ptr = gpu_ptr[0],
              .total_size = source_.size_bytes,
              .chunk_size = mem_manager->get_pool_chunk_size(),
              .device_id = 0};
          auto gpu_sink_ptr = std::make_shared<store::loader::GPUMemorySink>(gpu_opts);

          store::loader::UnifiedMemorySink::Options unified_opts{
              .inner_sink = gpu_sink_ptr,
              .memory_manager = mem_manager,
              .target_location = ModelLocation::GPU,
              .device_id = 0,
              .chunk_indices = chunk_indices};
          store::loader::UnifiedMemorySink unified_sink(unified_opts);

          status =
              store::loader::pump_ranges(remote_source, unified_sink, buffer_adapter, ranges, effective_concurrency);
        } else if (target_location == ModelLocation::PAGEABLE_CPU) {
          // Prepare buffering
          status = mem_manager->ensure_streaming_buffer(1);
          if (!status.ok()) {
            return status;
          }

          auto spb = mem_manager->get_streaming_buffer();
          if (!spb) {
            return absl::InternalError("Failed to get streaming buffer");
          }
          store::loader::StreamingBufferAdapter buffer_adapter(spb);

          auto cpu_ptr = mem_manager->get_pointer(ModelLocation::PAGEABLE_CPU);
          if (cpu_ptr.empty()) {
            return absl::InternalError("CPU memory not allocated");
          }

          store::loader::DVMPMappedSink::Options dvmp_opts{
              .base_addr = cpu_ptr[0],
              .total_size = source_.size_bytes,
              .partition_paths = {},
              .partition_sizes = {},
              .populate_pages = true};
          auto dvmp_sink_ptr = std::make_shared<store::loader::DVMPMappedSink>(dvmp_opts);

          store::loader::UnifiedMemorySink::Options unified_opts{
              .inner_sink = dvmp_sink_ptr,
              .memory_manager = mem_manager,
              .target_location = ModelLocation::PAGEABLE_CPU,
              .device_id = std::nullopt,
              .chunk_indices = chunk_indices};
          store::loader::UnifiedMemorySink unified_sink(unified_opts);

          status =
              store::loader::pump_ranges(remote_source, unified_sink, buffer_adapter, ranges, effective_concurrency);
        } else {
          return absl::UnimplementedError("Unsupported target location");
        }

        if (status.ok()) {
          LOG(INFO) << "P2P chunk transfer complete: " << chunk_indices.size() << " chunks";
        }

        return status;
      });
}

absl::Status P2PLoader::pull_chunk(const std::shared_ptr<MemoryManager>& mem_manager, uint32_t chunk_idx) {
  if (!initialized_) {
    auto status = initialize();
    if (!status.ok()) {
      return status;
    }
  }

  size_t chunk_size = mem_manager->get_pool_chunk_size();
  uint64_t offset = static_cast<uint64_t>(chunk_idx) * chunk_size;
  size_t bytes_to_read = std::min(chunk_size, static_cast<size_t>(source_.size_bytes - offset));

  // Destination pointer inside PAGEABLE_CPU region
  auto cpu_ptr = mem_manager->get_pointer(ModelLocation::PAGEABLE_CPU);
  if (cpu_ptr.empty()) {
    return absl::InternalError("CPU memory not allocated");
  }
  char* dest = static_cast<char*>(cpu_ptr[0]) + offset;

  // Use RemoteKeySource for direct random-access read
  store::loader::RemoteKeySource::Options src_opts{
      .comm_engine = source_.comm_engine,
      .memory_keys = source_.memory_keys,
      .buffer_sizes = source_.buf_sizes,
      .ip = source_.ip,
      .port = source_.port,
      .total_size = source_.size_bytes};
  store::loader::RemoteKeySource remote_source(src_opts);

  auto read_st = remote_source.read_at(offset, dest, bytes_to_read);
  if (!read_st.ok()) {
    return read_st.status();
  }
  if (*read_st != bytes_to_read) {
    return absl::DataLossError("Incomplete chunk read from remote");
  }
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> P2PLoader::get_model_size() {
  absl::MutexLock lock(&mutex_);
  if (!initialized_) {
    return absl::FailedPreconditionError("P2PLoader not initialized");
  }
  return source_.size_bytes;
}

} // namespace stepcast::store