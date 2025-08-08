// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/disk_loader.h"

#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "core/common/metrics/metric_objects.h"
#include "core/common/model_verification.h"
#include "core/store/loader/dvmp_region_sink.h"
#include "core/store/loader/file_partition_source.h"
#include "core/store/loader/gpu_memory_sink.h"
#include "core/store/loader/pump.h"
#include "core/store/loader/streaming_buffer_adapter.h"
#include "core/store/loader/unified_memory_sink.h"
#include "core/store/model/memory_manager.h"
#include "core/store/model/memory_state.h"

namespace stepcast::store {

// Helper to combine paths safely
std::filesystem::path safe_path_join(const std::filesystem::path& base, const std::filesystem::path& sub) {
  if (base.empty()) {
    if (absl::StrContains(sub.string(), "..")) {
      LOG(FATAL) << "Invalid path: contains '..' - " << sub.string();
    }
    return sub;
  }

  if (sub.is_absolute() || absl::StrContains(sub.string(), "..")) {
    LOG(FATAL) << "Invalid subdirectory path: " << sub.string();
  }
  return base / sub;
}

// Helper to load verification info from disk if available
absl::StatusOr<ModelVerificationInfo> load_verification_info_from_disk(const std::filesystem::path& model_dir) {
  std::filesystem::path verification_path = model_dir / "verification.json";
  if (!std::filesystem::exists(verification_path)) {
    return absl::NotFoundError("Verification file not found");
  }

  try {
    std::ifstream file(verification_path);
    if (!file.is_open()) {
      return absl::InternalError(absl::StrFormat("Failed to open verification file: %s", verification_path.string()));
    }

    std::string json_content;
    file.seekg(0, std::ios::end);
    json_content.reserve(file.tellg());
    file.seekg(0, std::ios::beg);
    json_content.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    if (json_content.empty()) {
      return absl::InvalidArgumentError("Verification file is empty");
    }

    absl::StatusOr<ModelVerificationInfo> result = ModelVerificationInfo::from_json(json_content);
    if (!result.ok()) {
      return absl::InvalidArgumentError(
          absl::StrFormat(
              "Failed to parse verification file %s: %s", verification_path.string(), result.status().message()));
    }

    LOG(INFO) << "Successfully loaded verification info from " << verification_path.string()
              << " (model_size: " << result->model_size << ", full_hash: 0x" << std::hex << result->full_hash
              << std::dec << ")";

    return result.value();
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrFormat("Exception while loading verification file %s: %s", verification_path.string(), e.what()));
  }
}

DiskLoader::DiskLoader(DiskSource source) : source_(std::move(source)), model_size_(0), initialized_(false) {}

absl::Status DiskLoader::initialize() {
  absl::MutexLock lock(&mutex_);

  if (initialized_) {
    return absl::OkStatus();
  }

  // Use the configured path directly as the model directory
  std::filesystem::path model_dir = source_.path;

  // Check if the directory exists
  if (!std::filesystem::exists(model_dir)) {
    return absl::NotFoundError(absl::StrFormat("Model directory not found: %s", model_dir.string()));
  }

  if (!std::filesystem::is_directory(model_dir)) {
    return absl::InvalidArgumentError(absl::StrFormat("Path is not a directory: %s", model_dir.string()));
  }

  // Find all partition files
  partition_paths_.clear();
  partition_sizes_.clear();
  model_size_ = 0;

  for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
    if (entry.is_regular_file()) {
      const std::string filename = entry.path().filename().string();
      // Support only the partition naming convention defined in
      // docs/developer-guides/core/checkpoint/data-format.md:
      //   tensor.data        (single-file model)
      //   tensor.data_<n>    (multi-partition model, 0-based index)
      if (filename.starts_with("tensor.data")) {
        partition_paths_.push_back(entry.path());
        size_t file_size = std::filesystem::file_size(entry.path());
        partition_sizes_.push_back(file_size);
        model_size_ += file_size;
      }
    }
  }

  // If no partitions were detected with the supported patterns report an error.
  if (partition_paths_.empty()) {
    return absl::NotFoundError(absl::StrFormat("No model partition files found in: %s", model_dir.string()));
  }

  // Sort partitions by name to ensure consistent ordering
  std::vector<std::pair<std::filesystem::path, size_t>> path_size_pairs;
  for (size_t i = 0; i < partition_paths_.size(); ++i) {
    path_size_pairs.emplace_back(partition_paths_[i], partition_sizes_[i]);
  }

  std::sort(path_size_pairs.begin(), path_size_pairs.end(), [](const auto& a, const auto& b) {
    return a.first.filename() < b.first.filename();
  });

  partition_paths_.clear();
  partition_sizes_.clear();
  for (const auto& [path, size] : path_size_pairs) {
    partition_paths_.push_back(path);
    partition_sizes_.push_back(size);
  }

  // Try to load verification info
  auto verification_result = load_verification_info_from_disk(model_dir);
  if (verification_result.ok()) {
    // Store verification info if needed
    // verification_info_ = verification_result.value();
  }

  LOG(INFO) << "DiskLoader initialized: " << partition_paths_.size() << " partitions, total size: " << model_size_
            << " bytes";
  initialized_ = true;

  return absl::OkStatus();
}

std::future<absl::Status> DiskLoader::load_async(
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

    // Set default concurrency
    if (concurrency <= 0) {
      concurrency = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
    }

    // Allocate target memory
    if (mem_manager->get_state(target_location) == MemoryState::UNALLOCATED) {
      auto status = mem_manager->allocate_memory(target_location);
      if (!status.ok()) {
        return status;
      }
    }

    // Create source
    loader::FilePartitionSource::Options source_opts;
    {
      absl::MutexLock lock(&mutex_);
      source_opts.partition_paths = partition_paths_;
      source_opts.partition_sizes = partition_sizes_;
      source_opts.total_size = model_size_;
      source_opts.chunk_size = mem_manager->get_pool_chunk_size();
      source_opts.use_direct_io = (model_size_ > 5ULL * 1024 * 1024 * 1024); // >5GB
    }
    loader::FilePartitionSource source(source_opts);

    // GPU path - use streaming buffer and pump
    if (target_location == ModelLocation::GPU) {
      // Ensure streaming buffer
      size_t num_chunks = std::min<size_t>(8, (model_size_ + source_opts.chunk_size - 1) / source_opts.chunk_size);
      auto status = mem_manager->ensure_streaming_buffer(num_chunks);
      if (!status.ok()) {
        return status;
      }

      auto spb = mem_manager->get_streaming_buffer();
      if (!spb) {
        return absl::InternalError("Failed to get streaming buffer");
      }

      // Create adapter, sink pipeline
      loader::StreamingBufferAdapter buffer_adapter(spb);
      auto gpu_ptr = mem_manager->get_pointer(ModelLocation::GPU);
      if (gpu_ptr.empty()) {
        return absl::InternalError("GPU memory not allocated");
      }

      auto gpu_sink = std::make_shared<loader::GPUMemorySink>(loader::GPUMemorySink::Options{
          .gpu_base_ptr = gpu_ptr[0],
          .total_size = model_size_,
          .chunk_size = mem_manager->get_pool_chunk_size(),
          .device_id = mem_manager->get_local_device_id()});

      loader::UnifiedMemorySink unified_sink(
          {.inner_sink = gpu_sink,
           .memory_manager = mem_manager,
           .target_location = ModelLocation::GPU,
           .device_id = mem_manager->get_local_device_id()});

      // Run pump
      ABSL_CHECK_OK(mem_manager->set_state(ModelLocation::GPU, MemoryState::LOADING));
      status = loader::pump(source, unified_sink, buffer_adapter, concurrency);

      if (!status.ok()) {
        ABSL_CHECK_OK(mem_manager->set_state(ModelLocation::GPU, MemoryState::FAILED));
        return status;
      }

      ABSL_CHECK_OK(mem_manager->set_state(ModelLocation::GPU, MemoryState::LOADED));
      return absl::OkStatus();
    }

    // CPU path - use DVMP positioned writes into reserved region
    if (target_location == ModelLocation::PAGEABLE_CPU) {
      auto cpu_ptr = mem_manager->get_pointer(ModelLocation::PAGEABLE_CPU);
      if (cpu_ptr.empty()) {
        return absl::InternalError("CPU memory not allocated");
      }

      auto dvmp_sink = std::make_shared<loader::DVMPRegionSink>(
          loader::DVMPRegionSink::Options{.memory_manager = mem_manager, .total_size = model_size_});

      loader::UnifiedMemorySink unified_sink(
          {.inner_sink = dvmp_sink, .memory_manager = mem_manager, .target_location = ModelLocation::PAGEABLE_CPU});

      ABSL_CHECK_OK(mem_manager->set_state(ModelLocation::PAGEABLE_CPU, MemoryState::LOADING));

      // Use streaming pump for CPU path
      size_t num_chunks = 4; // Smaller buffer for CPU
      auto status = mem_manager->ensure_streaming_buffer(num_chunks);
      if (!status.ok()) {
        return status;
      }

      auto spb = mem_manager->get_streaming_buffer();
      if (!spb) {
        return absl::InternalError("Failed to get streaming buffer");
      }

      loader::StreamingBufferAdapter buffer_adapter(spb);
      status = loader::pump(source, unified_sink, buffer_adapter, concurrency);

      if (!status.ok()) {
        ABSL_CHECK_OK(mem_manager->set_state(ModelLocation::PAGEABLE_CPU, MemoryState::FAILED));
        return status;
      }

      // Metrics: count loaded bytes by source/location
      try {
        static const stepcast::metrics::Counter kLoaderBytes("loader_bytes_total");
        kLoaderBytes.with_labels({{"source", "disk"}, {"location", "CPU"}}).inc(static_cast<double>(model_size_));
      } catch (...) {
      }

      ABSL_CHECK_OK(mem_manager->set_state(ModelLocation::PAGEABLE_CPU, MemoryState::LOADED));
      return absl::OkStatus();
    }

    return absl::UnimplementedError("Unsupported target location");
  });
}

std::future<absl::Status> DiskLoader::load_chunks_async(
    std::shared_ptr<MemoryManager> mem_manager,
    ModelLocation target_location,
    const std::vector<uint32_t>& chunk_indices,
    int concurrency) {
  return std::async(
      std::launch::async, [this, mem_manager, target_location, chunk_indices, concurrency]() mutable -> absl::Status {
        // Initialize if needed
        if (!initialized_) {
          auto status = initialize();
          if (!status.ok()) {
            return status;
          }
        }

        // Set default concurrency
        if (concurrency <= 0) {
          concurrency = std::min(2, static_cast<int>(std::thread::hardware_concurrency()));
        }

        // Allocate target memory
        if (mem_manager->get_state(target_location) == MemoryState::UNALLOCATED) {
          auto status = mem_manager->allocate_memory(target_location);
          if (!status.ok()) {
            return status;
          }
        }

        // Create source
        loader::FilePartitionSource::Options source_opts;
        {
          absl::MutexLock lock(&mutex_);
          source_opts.partition_paths = partition_paths_;
          source_opts.partition_sizes = partition_sizes_;
          source_opts.total_size = model_size_;
          source_opts.chunk_size = mem_manager->get_pool_chunk_size();
        }
        loader::FilePartitionSource source(source_opts);

        // Build ranges for chunks
        std::vector<std::pair<uint64_t, size_t>> ranges;
        for (size_t idx : chunk_indices) {
          uint64_t offset = idx * source_opts.chunk_size;
          size_t size = std::min(source_opts.chunk_size, static_cast<size_t>(model_size_ - offset));
          ranges.emplace_back(offset, size);

          // Mark chunk as loading
          // mem_manager->set_chunk_state(idx, ChunkState::LOADING);
        }

        // Setup sink based on target
        absl::Status status;
        if (target_location == ModelLocation::GPU) {
          // Ensure streaming buffer
          size_t num_chunks = std::min<size_t>(4, chunk_indices.size());
          status = mem_manager->ensure_streaming_buffer(num_chunks);
          if (!status.ok()) {
            return status;
          }

          auto spb = mem_manager->get_streaming_buffer();
          if (!spb) {
            return absl::InternalError("Failed to get streaming buffer");
          }

          loader::StreamingBufferAdapter buffer_adapter(spb);
          auto gpu_ptr = mem_manager->get_pointer(ModelLocation::GPU);
          if (gpu_ptr.empty()) {
            return absl::InternalError("GPU memory not allocated");
          }

          auto gpu_sink_chunk = std::make_shared<loader::GPUMemorySink>(loader::GPUMemorySink::Options{
              .gpu_base_ptr = gpu_ptr[0],
              .total_size = model_size_,
              .chunk_size = mem_manager->get_pool_chunk_size(),
              .device_id = mem_manager->get_local_device_id()});

          loader::UnifiedMemorySink unified_sink(
              {.inner_sink = gpu_sink_chunk,
               .memory_manager = mem_manager,
               .target_location = ModelLocation::GPU,
               .device_id = mem_manager->get_local_device_id(),
               .chunk_indices = chunk_indices});

          // Run pump_ranges
          status = loader::pump_ranges(source, unified_sink, buffer_adapter, ranges, concurrency);
          if (status.ok()) {
            // Metrics: sum bytes loaded for the requested ranges
            uint64_t bytes_sum = 0;
            for (const auto& r : ranges)
              bytes_sum += r.second;
            try {
              static const stepcast::metrics::Counter kLoaderBytes("loader_bytes_total");
              kLoaderBytes.with_labels({{"source", "disk"}, {"location", "GPU"}}).inc(static_cast<double>(bytes_sum));
            } catch (...) {
            }
          }
        } else if (target_location == ModelLocation::PAGEABLE_CPU) {
          auto cpu_ptr = mem_manager->get_pointer(ModelLocation::PAGEABLE_CPU);
          if (cpu_ptr.empty()) {
            return absl::InternalError("CPU memory not allocated");
          }

          auto dvmp_sink_chunk = std::make_shared<loader::DVMPRegionSink>(
              loader::DVMPRegionSink::Options{.memory_manager = mem_manager, .total_size = model_size_});

          loader::UnifiedMemorySink unified_sink(
              {.inner_sink = dvmp_sink_chunk,
               .memory_manager = mem_manager,
               .target_location = ModelLocation::PAGEABLE_CPU,
               .chunk_indices = chunk_indices});

          // Ensure buffer for pump
          status = mem_manager->ensure_streaming_buffer(2);
          if (!status.ok()) {
            return status;
          }

          auto spb = mem_manager->get_streaming_buffer();
          if (!spb) {
            return absl::InternalError("Failed to get streaming buffer");
          }

          loader::StreamingBufferAdapter buffer_adapter(spb);
          status = loader::pump_ranges(source, unified_sink, buffer_adapter, ranges, concurrency);
          if (status.ok()) {
            uint64_t bytes_sum = 0;
            for (const auto& r : ranges)
              bytes_sum += r.second;
            try {
              static const stepcast::metrics::Counter kLoaderBytes("loader_bytes_total");
              kLoaderBytes.with_labels({{"source", "disk"}, {"location", "CPU"}}).inc(static_cast<double>(bytes_sum));
            } catch (...) {
            }
          }
        } else {
          return absl::UnimplementedError("Unsupported target location");
        }

        // Update chunk states on success
        if (status.ok()) {
          // for (size_t idx : chunk_indices) {
          //   ChunkState final_state = (target_location == ModelLocation::GPU) ?
          //       ChunkState::COPIED_GPU : ChunkState::HOT;
          //   mem_manager->set_chunk_state(idx, final_state);
          // }
        }

        return status;
      });
}

absl::StatusOr<uint64_t> DiskLoader::get_model_size() {
  absl::MutexLock lock(&mutex_);
  if (!initialized_) {
    return absl::FailedPreconditionError("DiskLoader not initialized");
  }
  return model_size_;
}

absl::StatusOr<ModelVerificationInfo> DiskLoader::get_verification_info() const {
  absl::MutexLock lock(&mutex_);
  if (!initialized_) {
    return absl::FailedPreconditionError("DiskLoader not initialized");
  }
  // Return placeholder for now
  return absl::NotFoundError("No verification info available");
}

} // namespace stepcast::store
