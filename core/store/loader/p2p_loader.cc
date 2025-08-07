// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/p2p_loader.h"

#include <algorithm>
#include <future>
#include <memory>
#include <vector>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/common/metrics/metric_objects.h"
#include "core/store/loader/dvmp_region_sink.h"
#include "core/store/loader/file_partition_source.h"
#include "core/store/loader/gpu_memory_sink.h"
#include "core/store/loader/mux_seekable_source.h"
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

namespace {
absl::StatusOr<store::loader::FilePartitionSource::Options> build_fallback_disk_source_opts(
    const std::string& dir,
    size_t chunk_size,
    uint64_t expected_total) {
  namespace fs = std::filesystem;
  fs::path model_dir(dir);
  if (dir.empty() || !fs::exists(model_dir) || !fs::is_directory(model_dir)) {
    return absl::NotFoundError("Fallback model directory not found or not a directory");
  }
  std::vector<fs::path> paths;
  std::vector<size_t> sizes;
  uint64_t total = 0;
  for (const auto& entry : fs::directory_iterator(model_dir)) {
    if (entry.is_regular_file()) {
      const std::string filename = entry.path().filename().string();
      if (filename.rfind("tensor.data", 0) == 0) {
        paths.push_back(entry.path());
        size_t sz = fs::file_size(entry.path());
        sizes.push_back(sz);
        total += sz;
      }
    }
  }
  if (paths.empty()) {
    return absl::NotFoundError("No tensor.data partitions in fallback dir");
  }
  std::vector<std::pair<fs::path, size_t>> pair;
  for (size_t i = 0; i < paths.size(); ++i)
    pair.emplace_back(paths[i], sizes[i]);
  std::sort(
      pair.begin(), pair.end(), [](const auto& a, const auto& b) { return a.first.filename() < b.first.filename(); });
  store::loader::FilePartitionSource::Options opts;
  for (auto& p : pair) {
    opts.partition_paths.push_back(p.first);
    opts.partition_sizes.push_back(p.second);
  }
  opts.total_size = (expected_total > 0) ? expected_total : total;
  opts.chunk_size = chunk_size;
  opts.use_direct_io = (opts.total_size > 5ULL * 1024 * 1024 * 1024);
  return opts;
}
} // namespace

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
    auto remote_src_ptr = std::make_shared<store::loader::RemoteKeySource>(source_opts);
    // Optional disk fallback via env var SCSTORE_FALLBACK_MODEL_DIR
    const char* fb_dir_env = ::getenv("SCSTORE_FALLBACK_MODEL_DIR");
    std::shared_ptr<store::loader::SeekableSource> mux_source;
    if (fb_dir_env != nullptr && std::strlen(fb_dir_env) > 0) {
      auto disk_opts_or =
          build_fallback_disk_source_opts(fb_dir_env, mem_manager->get_pool_chunk_size(), source_.size_bytes);
      if (disk_opts_or.ok()) {
        auto file_src_ptr = std::make_shared<store::loader::FilePartitionSource>(*disk_opts_or);
        mux_source = std::make_shared<store::loader::MuxSeekableSource>(remote_src_ptr, file_src_ptr);
        VLOG(1) << "P2PLoader: enabled disk fallback via MuxSeekableSource using dir='" << fb_dir_env << "'";
      } else {
        LOG(WARNING) << "P2PLoader: fallback dir set but invalid: " << disk_opts_or.status();
      }
    }

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
      if (mux_source) {
        status = store::loader::pump(*mux_source, unified_sink, buffer_adapter, concurrency);
      } else {
        status = store::loader::pump(*remote_src_ptr, unified_sink, buffer_adapter, concurrency);
      }

      if (!status.ok()) {
        CHECK_OK(mem_manager->set_state(ModelLocation::GPU, MemoryState::FAILED));
        return status;
      }

      // Metrics: bytes loaded via P2P to GPU
      try {
        static const stepcast::metrics::Counter kLoaderBytes("loader_bytes_total");
        kLoaderBytes.with_labels({{"source", "p2p"}, {"location", "GPU"}}).inc(static_cast<double>(source_.size_bytes));
      } catch (...) {
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

      store::loader::DVMPRegionSink::Options dvmp_opts{.memory_manager = mem_manager, .total_size = source_.size_bytes};
      auto dvmp_ptr = std::make_shared<store::loader::DVMPRegionSink>(dvmp_opts);

      store::loader::UnifiedMemorySink::Options unified_opts{
          .inner_sink = dvmp_ptr,
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
      if (mux_source) {
        status = store::loader::pump(*mux_source, unified_sink, buffer_adapter, concurrency);
      } else {
        status = store::loader::pump(*remote_src_ptr, unified_sink, buffer_adapter, concurrency);
      }

      if (!status.ok()) {
        CHECK_OK(mem_manager->set_state(ModelLocation::PAGEABLE_CPU, MemoryState::FAILED));
        return status;
      }

      // Metrics: bytes loaded via P2P to CPU
      try {
        static const stepcast::metrics::Counter kLoaderBytes("loader_bytes_total");
        kLoaderBytes.with_labels({{"source", "p2p"}, {"location", "CPU"}}).inc(static_cast<double>(source_.size_bytes));
      } catch (...) {
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
        auto remote_src_ptr = std::make_shared<store::loader::RemoteKeySource>(src_opts);

        // Optional disk fallback via Mux
        const char* fb_dir_env2 = ::getenv("SCSTORE_FALLBACK_MODEL_DIR");
        std::shared_ptr<store::loader::SeekableSource> mux_source2;
        if (fb_dir_env2 != nullptr && std::strlen(fb_dir_env2) > 0) {
          auto disk_opts_or =
              build_fallback_disk_source_opts(fb_dir_env2, mem_manager->get_pool_chunk_size(), source_.size_bytes);
          if (disk_opts_or.ok()) {
            auto file_src_ptr = std::make_shared<store::loader::FilePartitionSource>(*disk_opts_or);
            mux_source2 = std::make_shared<store::loader::MuxSeekableSource>(remote_src_ptr, file_src_ptr);
          } else {
            LOG(WARNING) << "P2PLoader: fallback dir set but invalid: " << disk_opts_or.status();
          }
        }

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

          if (mux_source2) {
            status =
                store::loader::pump_ranges(*mux_source2, unified_sink, buffer_adapter, ranges, effective_concurrency);
          } else {
            status = store::loader::pump_ranges(
                *remote_src_ptr, unified_sink, buffer_adapter, ranges, effective_concurrency);
          }
          if (status.ok()) {
            uint64_t bytes_sum = 0;
            for (const auto& r : ranges)
              bytes_sum += r.second;
            try {
              static const stepcast::metrics::Counter kLoaderBytes("loader_bytes_total");
              kLoaderBytes.with_labels({{"source", "p2p"}, {"location", "GPU"}}).inc(static_cast<double>(bytes_sum));
            } catch (...) {
            }
          }
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

          store::loader::DVMPRegionSink::Options dvmp_opts{
              .memory_manager = mem_manager, .total_size = source_.size_bytes};
          auto dvmp_sink_ptr = std::make_shared<store::loader::DVMPRegionSink>(dvmp_opts);

          store::loader::UnifiedMemorySink::Options unified_opts{
              .inner_sink = dvmp_sink_ptr,
              .memory_manager = mem_manager,
              .target_location = ModelLocation::PAGEABLE_CPU,
              .device_id = std::nullopt,
              .chunk_indices = chunk_indices};
          store::loader::UnifiedMemorySink unified_sink(unified_opts);

          if (mux_source2) {
            status =
                store::loader::pump_ranges(*mux_source2, unified_sink, buffer_adapter, ranges, effective_concurrency);
          } else {
            status = store::loader::pump_ranges(
                *remote_src_ptr, unified_sink, buffer_adapter, ranges, effective_concurrency);
          }
          if (status.ok()) {
            uint64_t bytes_sum = 0;
            for (const auto& r : ranges)
              bytes_sum += r.second;
            try {
              static const stepcast::metrics::Counter kLoaderBytes("loader_bytes_total");
              kLoaderBytes.with_labels({{"source", "p2p"}, {"location", "CPU"}}).inc(static_cast<double>(bytes_sum));
            } catch (...) {
            }
          }
        } else {
          return absl::UnimplementedError("Unsupported target location");
        }

        if (status.ok()) {
          LOG(INFO) << "P2P chunk transfer complete: " << chunk_indices.size() << " chunks";
        }

        return status;
      });
}

absl::StatusOr<uint64_t> P2PLoader::get_model_size() {
  absl::MutexLock lock(&mutex_);
  if (!initialized_) {
    return absl::FailedPreconditionError("P2PLoader not initialized");
  }
  return source_.size_bytes;
}

} // namespace stepcast::store
