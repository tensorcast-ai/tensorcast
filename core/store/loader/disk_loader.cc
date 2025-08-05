// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/disk_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <future>
#include <numeric>
#include <thread>
#include <vector>
#include "core/common/cuda_api.h"
#include "core/store/model/memory_manager.h"
#include "core/store/model/unified_model_memory.h"

#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "core/common/memory/distributed_memory_pool.h"
#include "core/common/model_verification.h"
#include "core/common/trace/trace_cuda_async_fn.h"
#include "core/common/trace/trace_macros.h"
#include "core/store/loader/file_partition_reader.h"

namespace stepcast::store {

// DIRECT_IO optimization: For models > 5GB, we use O_DIRECT to bypass kernel page cache
// This reduces memory pressure and improves performance for large model loading.
// All pinned memory buffers are aligned to 512 bytes (or 4KB pages) to support DIRECT_IO.

// Helper function for multi-threaded disk reading
absl::Status disk_reader_thread(
    std::shared_ptr<StreamingPinnedBuffer> spb,
    std::shared_ptr<FilePartitionReader> reader,
    std::atomic<size_t>& next_chunk_id,
    std::atomic<int>& active_producers,
    size_t total_chunks,
    size_t chunk_size,
    size_t model_size) {
  absl::Status status = absl::OkStatus();

  while (true) {
    // Get the next chunk to read
    size_t chunk_id = next_chunk_id.fetch_add(1);
    if (chunk_id >= total_chunks)
      break;

    SC_TRACE_SCOPE("disk_read_partitions");

    // Get a free buffer slot
    auto slot_or = spb->get_free_chunk();
    if (!slot_or.ok()) {
      status = slot_or.status();
      break;
    }
    int slot = *slot_or;

    char* buffer = spb->get_chunk_ptr(slot);
    size_t global_offset = chunk_id * chunk_size;
    size_t bytes_to_read = std::min(chunk_size, model_size - global_offset);

    // Read data from disk
    absl::Status read_status = reader->read_at_offset(global_offset, buffer, bytes_to_read);
    if (!read_status.ok()) {
      LOG(ERROR) << "Failed to read chunk " << chunk_id << " from disk: " << read_status.message();
      status = read_status;
      // Return the slot back to free queue before exiting
      auto return_status = spb->return_chunk(slot);
      if (!return_status.ok()) {
        LOG(ERROR) << "Failed to return chunk after read error: " << return_status;
      }
      break;
    }

    // Mark chunk as ready for GPU transfer
    absl::Status mark_status = spb->mark_chunk_ready(slot, chunk_id, bytes_to_read);
    if (!mark_status.ok()) {
      status = mark_status;
      // Return the slot back to free queue before exiting
      auto return_status = spb->return_chunk(slot);
      if (!return_status.ok()) {
        LOG(ERROR) << "Failed to return chunk after mark_ready error: " << return_status;
      }
      break;
    }
  }

  // Decrement active producer count and signal completion if we're the last one
  int remaining = active_producers.fetch_sub(1) - 1;
  if (remaining == 0) {
    // We're the last producer thread to finish
    spb->signal_production_complete();
    VLOG(1) << "Last producer thread finished, signaled production complete";
  }

  return status;
}

// Helper to combine paths safely
std::filesystem::path safe_path_join(const std::filesystem::path& base, const std::filesystem::path& sub) {
  // If base path is empty, treat sub as the complete path
  if (base.empty()) {
    // When storage_path is empty, model_subdirectory is the full path
    // Still check for ".." to prevent directory traversal
    if (absl::StrContains(sub.string(), "..")) {
      LOG(FATAL) << "Invalid path: contains '..' - " << sub.string();
    }
    return sub;
  }

  // Basic check against path traversal, might need more robust validation
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

    // Read entire file content
    std::string json_content;
    file.seekg(0, std::ios::end);
    json_content.reserve(file.tellg());
    file.seekg(0, std::ios::beg);
    json_content.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    if (json_content.empty()) {
      return absl::InvalidArgumentError("Verification file is empty");
    }

    // Parse JSON using ModelVerificationInfo::from_json
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

  model_size_ = 0;
  partition_paths_.clear();
  partition_sizes_.clear();

  try {
    // Use the path directly from DiskSource
    std::filesystem::path model_dir = source_.path;

    VLOG(4) << "DiskLoader: Initializing from path " << model_dir.string();

    if (!std::filesystem::exists(model_dir) || !std::filesystem::is_directory(model_dir)) {
      return absl::NotFoundError(
          absl::StrFormat("Model directory not found or not a directory: %s", model_dir.string()));
    }

    for (int partition_id = 0;; ++partition_id) {
      SC_TRACE_SCOPE("disk_scan_partitions");
      auto tensor_path = model_dir / ("tensor.data_" + std::to_string(partition_id));

      if (!std::filesystem::exists(tensor_path)) {
        if (partition_id == 0) { // Must have at least one partition
          return absl::NotFoundError(absl::StrFormat("Partition 0 not found at %s", tensor_path.string()));
        }
        break; // No more partitions
      }

      if (!std::filesystem::is_regular_file(tensor_path)) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Partition path is not a regular file: %s", tensor_path.string()));
      }

      std::error_code ec;
      uint64_t file_size = std::filesystem::file_size(tensor_path, ec);
      if (ec) {
        return absl::InternalError(
            absl::StrFormat("Failed to get file size for %s: %s", tensor_path.string(), ec.message()));
      }

      if (file_size == 0 && partition_id > 0) {
        LOG(WARNING) << "DiskLoader: Partition " << partition_id << " (" << tensor_path.string() << ") is empty.";
        // Allow empty partitions, but maybe log? Decide policy.
      }

      model_size_ += file_size;
      partition_sizes_.push_back(file_size);
      partition_paths_.push_back(tensor_path);
    }

    if (model_size_ == 0 && !partition_paths_.empty()) {
      // Found partition files but total size is 0
      LOG(WARNING) << "DiskLoader: Model partitions found but total size is 0 for " << model_dir.string();
      // Decide if this is an error or allowed state
      // return absl::InvalidArgumentError(absl::StrFormat("Model size is 0 despite partitions found: %s",
      // model_dir.string()));
    } else if (partition_paths_.empty()) {
      return absl::NotFoundError(absl::StrFormat("No model partitions found in %s", model_dir.string()));
    }

    // Reject unrealistically small models (assumed to be corrupt). The exact
    // threshold is not important for production usage but helps unit-tests
    // detect corrupted inputs (e.g. a 7-byte file in E9).  We pick 1 KiB as a
    // conservative lower-bound that is well below the smallest supported
    // tensor slice yet large enough to exclude obviously invalid files.
    if (model_size_ < 1024) {
      return absl::InvalidArgumentError(
          absl::StrFormat(
              "Model size (%d bytes) is too small to be a valid checkpoint: %s", model_size_, model_dir.string()));
    }

    VLOG(4) << "DiskLoader: Initialization successful. Found " << partition_paths_.size() << " partitions, total size "
            << model_size_ << " bytes.";
    initialized_ = true;
    return absl::OkStatus();

  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrFormat("Error during DiskLoader initialization: %s", e.what()));
  }
}

absl::StatusOr<uint64_t> DiskLoader::get_model_size() {
  absl::MutexLock lock(&mutex_);
  if (!initialized_) {
    // Should not happen if initialize() succeeded or returned error
    return absl::InternalError("DiskLoader state inconsistent after initialize call.");
  }
  return model_size_;
}

std::future<absl::Status> DiskLoader::load_async(
    std::shared_ptr<MemoryManager> mem_manager,
    ModelLocation target_location,
    int concurrency) {
  // If streaming transfer is enabled and the target is GPU, we use the new
  // pipeline that directly streams data from disk -> pinned buffer -> GPU
  if (target_location == ModelLocation::GPU) {
    // Capture required data first (paths, sizes) under loader mutex
    std::vector<std::filesystem::path> paths_copy;
    std::vector<size_t> sizes_copy;
    uint64_t model_size_copy;
    {
      absl::MutexLock lock(&mutex_);
      paths_copy = partition_paths_;
      sizes_copy = partition_sizes_;
      model_size_copy = model_size_;
    }

    // Sanitize concurrency for GPU loading
    int effective_threads = concurrency;
    if (effective_threads <= 0) {
      unsigned int hw = std::min(std::thread::hardware_concurrency(), 4U);
      effective_threads = static_cast<int>(hw == 0 ? 1 : hw);
    }

    return SC_TRACE_ASYNC(
        std::launch::async,
        [mem_manager,
         paths = std::move(paths_copy),
         part_sizes = std::move(sizes_copy),
         model_size = model_size_copy,
         num_threads = effective_threads]() -> absl::Status {
          absl::Status st;

          // Allocate GPU memory if not yet done
          if (mem_manager->get_state(ModelLocation::GPU) == MemoryState::UNALLOCATED) {
            st = mem_manager->allocate_memory(ModelLocation::GPU);
            if (!st.ok()) {
              return st;
            }
          }

          // Compute streaming buffer size (number of chunks)
          const size_t chunk_size = mem_manager->get_pool_chunk_size();
          if (chunk_size == 0) {
            return absl::FailedPreconditionError("Pinned memory pool chunk size is zero");
          }

          // Note: chunk_size from PinnedMemoryPool is already aligned for DIRECT_IO

          size_t buffer_bytes = std::min<size_t>(mem_manager->get_max_buffer_bytes(), model_size);
          if (buffer_bytes == 0) {
            buffer_bytes = chunk_size;
          }

          size_t num_buffer_chunks = (buffer_bytes + chunk_size - 1) / chunk_size;
          if (num_buffer_chunks == 0) {
            num_buffer_chunks = 1;
          }

          // Allocate streaming buffer
          st = mem_manager->allocate_buffer_pool(num_buffer_chunks);
          if (!st.ok()) {
            return st;
          }

          auto spb = mem_manager->get_streaming_buffer();
          if (!spb) {
            return absl::InternalError("Streaming buffer not available after allocation");
          }

          // Mark GPU state as LOADING
          st = mem_manager->set_state(ModelLocation::GPU, MemoryState::LOADING);
          if (!st.ok()) {
            return st;
          }

          // Create file reader and open partitions
          auto file_reader = std::make_shared<FilePartitionReader>();
          st = file_reader->open_partitions(paths, part_sizes);
          if (!st.ok()) {
            return st;
          }

          // Log DIRECT_IO status
          if (file_reader->is_direct_io_enabled()) {
            LOG(INFO) << "DiskLoader: Using DIRECT_IO for efficient large model loading";
          }

          // GPU base pointer
          auto gpu_ptr_vec = mem_manager->get_pointer(ModelLocation::GPU);
          if (gpu_ptr_vec.empty()) {
            return absl::InternalError("GPU memory pointer is null");
          }
          char* gpu_base = static_cast<char*>(gpu_ptr_vec[0]);

          // Create a dedicated non-blocking CUDA stream for H2D transfers
          cudaStream_t h2d_stream;
          auto stream_status = stepcast::cuda::stream_create(&h2d_stream);
          if (!stream_status.ok()) {
            return stream_status;
          }

          size_t total_chunks = (model_size + chunk_size - 1) / chunk_size;

          // Start producer threads for disk reading
          std::vector<std::future<absl::Status>> producers;
          std::atomic<size_t> next_chunk_id{0};
          std::atomic<int> active_producers{num_threads};

          VLOG(1) << "DiskLoader: Starting GPU load with " << num_threads
                  << " disk reader threads. Total chunks: " << total_chunks;

          producers.reserve(num_threads);
          for (int i = 0; i < num_threads; ++i) {
            producers.emplace_back(std::async(std::launch::async, [&]() {
              return disk_reader_thread(
                  spb, file_reader, next_chunk_id, active_producers, total_chunks, chunk_size, model_size);
            }));
          }

          // Consumer thread (GPU copy) - runs in main async thread
          size_t chunks_consumed = 0;
          absl::Status overall_status = absl::OkStatus();
          bool consumer_should_exit = false;

          while (chunks_consumed < total_chunks && !consumer_should_exit) {
            // Get a ready chunk from the buffer
            auto ready_or = spb->get_ready_chunk();
            if (!ready_or.ok()) {
              // Check if this is because production is complete
              if (ready_or.status().code() == absl::StatusCode::kOutOfRange) {
                // Normal completion - no more chunks to consume
                break;
              }
              overall_status = ready_or.status();
              consumer_should_exit = true;
              break;
            }

            auto ready = *ready_or;

            // Perform async H2D copy
            st = trace_cuda_async(
                "h2d_copy",
                h2d_stream,
                [&]() {
                  return stepcast::cuda::memcpy_async(
                      gpu_base + ready.global_chunk_id * chunk_size,
                      ready.data_ptr,
                      ready.bytes_in_chunk,
                      cudaMemcpyHostToDevice,
                      h2d_stream);
                },
                [spb, slot_id = ready.slot_id]() {
                  auto _st = spb->return_chunk(slot_id);
                  if (!_st.ok()) {
                    LOG(ERROR) << "Failed to return chunk: " << _st;
                  }
                });

            if (!st.ok()) {
              overall_status = st;
              consumer_should_exit = true;
              break;
            }

            chunks_consumed++;
          }

          // Wait for all producer threads to complete
          // The last producer thread will signal production complete
          for (auto& f : producers) {
            absl::Status producer_status = f.get();
            if (!producer_status.ok()) {
              if (overall_status.ok()) {
                overall_status = producer_status;
              }
              LOG(ERROR) << "Producer thread failed: " << producer_status;
            }
          }

          // Synchronize CUDA stream to ensure all transfers complete
          if (overall_status.ok()) {
            auto sync_status = stepcast::cuda::stream_synchronize(h2d_stream);
            if (!sync_status.ok()) {
              return sync_status;
            }
          }

          // Cleanup
          auto destroy_status = stepcast::cuda::stream_destroy(h2d_stream);
          if (!destroy_status.ok()) {
            return destroy_status;
          }
          file_reader->close_all();

          // Release streaming buffer
          st = mem_manager->release_buffer_pool();
          if (!st.ok()) {
            LOG(ERROR) << "Failed to release buffer pool: " << st;
          }

          // Finalize GPU state
          st = mem_manager->finalize_load_state(ModelLocation::GPU, overall_status);
          if (!st.ok()) {
            LOG(ERROR) << "Failed to finalize GPU state: " << st;
            return st;
          }

          VLOG(1) << "DiskLoader: GPU load completed. Status: " << overall_status;
          return overall_status;
        });
  }

  // -------- PAGEABLE_CPU Path -------------------------------------------------
  // We attempt a zero-copy mmap into DistributedVirtualMemoryPool (DVMP) **only**
  // when every partition meets strict alignment requirements.  Otherwise we
  // either (a) fall back to buffered copy for *small* partitions (< page size),
  // or (b) abort early for large but misaligned partitions (to avoid undefined
  // MAP_FIXED behaviour).

  if (target_location == ModelLocation::PAGEABLE_CPU) {
    // ---------------------------------------------------------
    // Step-1: Examine partition sizes / alignment requirements
    // ---------------------------------------------------------
    const long page_sz_long = ::sysconf(_SC_PAGESIZE);
    const size_t page_sz = page_sz_long > 0 ? static_cast<size_t>(page_sz_long) : 4096UL;

    bool has_small_partition = false; // < page size → copy
    bool has_misaligned_partition = false; // >= page size but not page-aligned → error

    {
      absl::MutexLock lock(&mutex_);
      for (size_t sz : partition_sizes_) {
        if (sz < page_sz) {
          has_small_partition = true;
        } else if (sz % page_sz != 0) {
          has_misaligned_partition = true;
          break;
        }
      }
    }

    // Case (b): large but misaligned → immediate error (caller will surface)
    if (has_misaligned_partition && !has_small_partition) {
      return std::async(std::launch::deferred, [page_sz] {
        return absl::InvalidArgumentError(
            absl::StrFormat(
                "Partition size not page-aligned (page size = %zu). Zero-copy mmap path aborted.", page_sz));
      });
    }

    // Case (a): contains at least one small partition → fall through to
    // buffered read logic further below.
    if (has_small_partition) {
      VLOG(1)
          << "DiskLoader: Falling back to buffered read path because at least one partition is smaller than system page size ("
          << page_sz << ")";
    } else {
      // All partitions are page-aligned and large enough – safe to mmap.

      // Capture partition metadata under loader mutex for thread-safety.
      std::vector<std::filesystem::path> paths_copy;
      std::vector<size_t> sizes_copy;
      uint64_t model_size_copy;
      {
        absl::MutexLock lock(&mutex_);
        paths_copy = partition_paths_;
        sizes_copy = partition_sizes_;
        model_size_copy = model_size_;
      }

      return SC_TRACE_ASYNC(
          std::launch::async,
          [mem_manager,
           paths = std::move(paths_copy),
           part_sizes = std::move(sizes_copy),
           model_size = model_size_copy]() -> absl::Status {
            // Helper lambda for consistent finalisation.
            auto finish = [&](const absl::Status& st) -> absl::Status {
              absl::Status finalize_status = mem_manager->finalize_load_state(ModelLocation::PAGEABLE_CPU, st);
              return finalize_status.ok() ? st : finalize_status;
            };

            // ----------------------------------------------------------------------
            // DVMP region handling
            // ----------------------------------------------------------------------
            // The MemoryManager::allocate_memory() call that precedes DiskLoader
            // invocation in Model::ensure_loaded_async has already r
            // eserved the
            // DVMP virtual region.  Therefore we must reuse that reservation
            // instead of calling allocate_pageable_cpu_region() again (which would
            // race and immediately return kAlreadyExists).

            const std::string model_id = mem_manager->instance_key().model_id;

            void* base_addr = mem_manager->get_dvmp_cpu_base();

            // Fallback: In the unlikely event the region has not been allocated
            // yet (e.g., future refactor touches call-order) we attempt to
            // allocate it here.
            CHECK(base_addr != nullptr);

            // Obtain DVMP instance (guaranteed to exist after the allocation helper).
            auto dvmp = mem_manager->get_dvmp();
            if (!dvmp) {
              return finish(absl::InternalError("DVMP instance unexpectedly null after region allocation."));
            }

            // Allocate memory state in MemoryManager if not yet done.
            CHECK(mem_manager->get_state(ModelLocation::PAGEABLE_CPU) != MemoryState::UNALLOCATED);

            // Transition to LOADING.
            ABSL_CHECK_OK(mem_manager->set_state(ModelLocation::PAGEABLE_CPU, MemoryState::LOADING));

            // Map each partition file into the reserved VA region using MAP_FIXED.
            size_t offset = 0;
            for (size_t idx = 0; idx < paths.size(); ++idx) {
              const auto& path = paths[idx];
              size_t part_size = part_sizes[idx];

              int fd = ::open(path.c_str(), O_RDONLY);
              if (fd < 0) {
                return finish(
                    absl::ErrnoToStatus(errno, absl::StrFormat("Failed to open partition %s", path.string())));
              }

              int mmap_flags = MAP_PRIVATE | MAP_FIXED;
              mmap_flags |= MAP_POPULATE;

              void* target_addr = static_cast<char*>(base_addr) + offset;
              void* mapped = ::mmap(target_addr, part_size, PROT_READ, mmap_flags, fd, 0);
              int saved_errno = errno; // capture before close
              ::close(fd);

              if (mapped == MAP_FAILED) {
                return finish(absl::ErrnoToStatus(saved_errno, absl::StrFormat("mmap failed for %s", path.string())));
              }

              if (mapped != target_addr) {
                return finish(absl::InternalError("mmap did not map at the expected fixed address."));
              }

              // Advance offset
              offset += part_size;
            }

            // After successful mapping, mark all chunks as HOT via lock/unlock dance.
            size_t num_chunks = (model_size + stepcast::memory::DistributedMemoryPool::kChunk - 1) /
                stepcast::memory::DistributedMemoryPool::kChunk;
            std::vector<uint32_t> idx(num_chunks);
            std::iota(idx.begin(), idx.end(), 0U);

            ABSL_CHECK_OK(dvmp->lock_chunks(model_id, idx));
            ABSL_CHECK_OK(dvmp->unlock_chunks(model_id, idx, /*copied_gpu=*/false));
            return finish(absl::OkStatus());
          });
    }
  }

  // Non-streaming path (original): only support loading into CPU
  if (target_location != ModelLocation::PAGEABLE_CPU) {
    return std::async(std::launch::deferred, [] {
      return absl::InvalidArgumentError("DiskLoader supports GPU target only in streaming mode.");
    });
  }

  absl::StatusOr<uint64_t> size_status = get_model_size();
  if (!size_status.ok()) {
    return std::async(std::launch::deferred, [status = size_status.status()] { return status; });
  }
  uint64_t expected_size = *size_status;

  // --- Pre-checks in Memory Manager (Must be done before launching async) ---
  if (mem_manager->get_model_size() == 0) {
    mem_manager->set_model_size(expected_size);
  } else if (mem_manager->get_model_size() != expected_size) {
    return std::async(std::launch::deferred, [expected_size, mgr_size = mem_manager->get_model_size()] {
      return absl::FailedPreconditionError(
          absl::StrFormat("DiskLoader size (%d) mismatch with MemoryManager size (%d).", expected_size, mgr_size));
    });
  }

  // Ensure memory is allocated in the manager
  absl::Status alloc_status = mem_manager->allocate_memory(ModelLocation::PAGEABLE_CPU);
  if (!alloc_status.ok()) {
    return std::async(std::launch::deferred, [alloc_status] { return alloc_status; });
  }

  // --- Set Loading State ---
  {
    absl::Status state_status = mem_manager->set_state(ModelLocation::PAGEABLE_CPU, MemoryState::LOADING);
    if (!state_status.ok()) {
      return std::async(std::launch::deferred, [state_status] { return state_status; });
    }
  } // Release manager lock before potentially blocking IO

  // --- Launch Async Read Task ---
  // Capture necessary members safely
  std::vector<std::filesystem::path> paths_copy;
  std::vector<size_t> sizes_copy;
  {
    absl::MutexLock lock(&mutex_); // Lock loader mutex to copy partition info
    paths_copy = partition_paths_;
    sizes_copy = partition_sizes_;
  }

  // Sanitize concurrency: ensure at least one thread. If 0 or negative, fall back to
  // hardware_concurrency() (or 1 if unavailable).
  int effective_threads = concurrency;
  if (effective_threads <= 0) {
    unsigned int hw = std::min(std::thread::hardware_concurrency(), 4U);
    effective_threads = static_cast<int>(hw == 0 ? 1 : hw);
  }

  return SC_TRACE_ASYNC(
      std::launch::async,
      [mem_manager,
       paths = std::move(paths_copy),
       part_sizes = std::move(sizes_copy),
       num_threads = effective_threads]() -> absl::Status {
        // Helper: ensure MemoryManager state finalisation exactly once.
        auto finish = [&](const absl::Status& st) -> absl::Status {
          // Use the return value of finalize_load_state, propagating the original 'st' only if finalize succeeds.
          absl::Status finalize_status = mem_manager->finalize_load_state(ModelLocation::PAGEABLE_CPU, st);
          // If finalize_load_state failed, return its error. Otherwise, return the original status 'st'.
          return finalize_status.ok() ? st : finalize_status;
        };

        // --- Get Pinned Memory Buffers ---
        std::shared_ptr<PinnedMemory> pinned_mem = mem_manager->get_pinned_memory();
        std::shared_ptr<BatchVector> chunk_queue = mem_manager->get_host_chunk_queue();

        if (!pinned_mem || !chunk_queue) {
          return finish(absl::InternalError("MemoryManager did not provide valid PinnedMemory or BatchVector."));
        }

        const size_t num_chunks = pinned_mem->num_chunks();
        const size_t chunk_size = pinned_mem->chunk_size();
        auto& host_buffers = pinned_mem->get(); // Get vector of char*

        if (host_buffers.empty() || num_chunks == 0) {
          return finish(absl::FailedPreconditionError("No pinned memory chunks allocated."));
        }

        // --- Create FilePartitionReader ---
        auto file_reader = std::make_shared<FilePartitionReader>();
        absl::Status open_status = file_reader->open_partitions(paths, part_sizes);
        if (!open_status.ok()) {
          return finish(open_status);
        }

        // Log DIRECT_IO status
        if (file_reader->is_direct_io_enabled()) {
          LOG(INFO) << "DiskLoader: Using DIRECT_IO for efficient large model loading (CPU path)";
        }

        const size_t total_model_size = mem_manager->get_model_size();

        LOG(INFO) << "DiskLoader: Starting load from disk to CPU using " << num_threads
                  << " threads. Total Chunks: " << num_chunks;

        // --- Multi-threaded Read Logic (Simplified with FilePartitionReader) ---
        auto read_task = [&](int thread_idx) -> absl::Status {
          const size_t chunk_per_thread = (num_chunks + num_threads - 1) / num_threads;
          const size_t start_chunk = thread_idx * chunk_per_thread;
          const size_t end_chunk = std::min((thread_idx + 1) * chunk_per_thread, num_chunks);

          // Check if this thread has any work
          if (start_chunk >= num_chunks) {
            return absl::OkStatus();
          }

          for (size_t chunk_id = start_chunk; chunk_id < end_chunk; ++chunk_id) {
            SC_TRACE_SCOPE("disk_read_partitions");

            size_t global_offset = chunk_id * chunk_size;
            size_t bytes_to_read = std::min(chunk_size, total_model_size - global_offset);

            if (bytes_to_read == 0) {
              break; // Reached the end of the model data
            }

            if (chunk_id >= host_buffers.size() || !host_buffers[chunk_id]) {
              LOG(ERROR) << "DiskLoader (Thread " << thread_idx << "): Invalid buffer for chunk " << chunk_id;
              return absl::InternalError(absl::StrFormat("Invalid buffer for chunk %d", chunk_id));
            }

            char* buffer = host_buffers[chunk_id];

            // Use FilePartitionReader to read the data
            absl::Status read_status = file_reader->read_at_offset(global_offset, buffer, bytes_to_read);
            if (!read_status.ok()) {
              LOG(ERROR) << "DiskLoader (Thread " << thread_idx << "): Failed to read chunk " << chunk_id << ": "
                         << read_status;
              return read_status;
            }

            // Notify that chunk is ready (can be used for pipelining CPU->GPU)
            chunk_queue->enqueue(chunk_id, Batch{chunk_id, bytes_to_read});
          } // end for each chunk

          return absl::OkStatus();
        }; // end read_task lambda

        // --- Launch and Wait for Threads ---
        std::vector<std::future<absl::Status>> futures;
        futures.reserve(num_threads);
        for (int i = 0; i < num_threads; ++i) {
          futures.emplace_back(
              SC_TRACE_ASYNC(std::launch::async, [&, thread_idx = i]() { return read_task(thread_idx); }));
        }

        absl::Status overall_status = absl::OkStatus();
        for (auto& f : futures) {
          absl::Status thread_status = f.get();
          if (!thread_status.ok()) {
            LOG(ERROR) << "DiskLoader: Error occurred in read thread: " << thread_status;
            overall_status = thread_status; // Keep first error encountered
          }
        }

        // --- Cleanup ---
        file_reader->close_all();

        return finish(overall_status);
      }); // end SC_TRACE_ASYNC for overall load task
}

std::future<absl::Status> DiskLoader::load_chunks_async(
    std::shared_ptr<MemoryManager> mem_manager,
    ModelLocation target_location,
    const std::vector<uint32_t>& chunk_indices,
    int concurrency) {
  // Validate inputs
  if (chunk_indices.empty()) {
    return std::async(
        std::launch::deferred, [] { return absl::InvalidArgumentError("No chunks specified for loading"); });
  }

  // Get model size to validate chunk indices
  absl::StatusOr<uint64_t> size_status = get_model_size();
  if (!size_status.ok()) {
    return std::async(std::launch::deferred, [status = size_status.status()] { return status; });
  }
  uint64_t model_size = *size_status;

  // Resolve authoritative chunk size
  size_t chunk_size = 256ULL * 1024 * 1024; // Fallback default (256 MB)
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

  // Capture required data first (paths, sizes) under loader mutex
  std::vector<std::filesystem::path> paths_copy;
  std::vector<size_t> sizes_copy;
  uint64_t model_size_copy;
  {
    absl::MutexLock lock(&mutex_);
    paths_copy = partition_paths_;
    sizes_copy = partition_sizes_;
    model_size_copy = model_size_;
  }

  // Sanitize concurrency
  int effective_threads = concurrency;
  if (effective_threads <= 0) {
    unsigned int hw = std::min(std::thread::hardware_concurrency(), 4U);
    effective_threads = static_cast<int>(hw == 0 ? 1 : hw);
  }

  // GPU path with chunk support
  if (target_location == ModelLocation::GPU) {
    return SC_TRACE_ASYNC(
        std::launch::async,
        [mem_manager,
         paths = std::move(paths_copy),
         part_sizes = std::move(sizes_copy),
         model_size = model_size_copy,
         chunk_indices = chunk_indices,
         chunk_size = chunk_size,
         num_threads = effective_threads]() -> absl::Status {
          absl::Status st;

          // Allocate GPU memory if not yet done
          if (mem_manager->get_state(ModelLocation::GPU) == MemoryState::UNALLOCATED) {
            st = mem_manager->allocate_memory(ModelLocation::GPU);
            if (!st.ok()) {
              return st;
            }
          }

          // Get pool chunk size for streaming buffer
          const size_t pool_chunk_size = mem_manager->get_pool_chunk_size();
          if (pool_chunk_size == 0) {
            return absl::FailedPreconditionError("Pinned memory pool chunk size is zero");
          }

          // Calculate buffer size based on chunks to load
          size_t bytes_to_load = chunk_indices.size() * chunk_size;
          size_t buffer_bytes = std::min<size_t>(mem_manager->get_max_buffer_bytes(), bytes_to_load);
          if (buffer_bytes == 0) {
            buffer_bytes = pool_chunk_size;
          }

          size_t num_buffer_chunks = (buffer_bytes + pool_chunk_size - 1) / pool_chunk_size;
          if (num_buffer_chunks == 0) {
            num_buffer_chunks = 1;
          }

          // Allocate streaming buffer
          st = mem_manager->allocate_buffer_pool(num_buffer_chunks);
          if (!st.ok()) {
            return st;
          }

          auto spb = mem_manager->get_streaming_buffer();
          if (!spb) {
            return absl::InternalError("Streaming buffer not available after allocation");
          }

          // Mark GPU state as LOADING
          st = mem_manager->set_state(ModelLocation::GPU, MemoryState::LOADING);
          if (!st.ok()) {
            return st;
          }

          // Create file reader and open partitions
          auto file_reader = std::make_shared<FilePartitionReader>();
          st = file_reader->open_partitions(paths, part_sizes);
          if (!st.ok()) {
            return st;
          }

          // GPU base pointer
          auto gpu_ptr_vec = mem_manager->get_pointer(ModelLocation::GPU);
          if (gpu_ptr_vec.empty()) {
            return absl::InternalError("GPU memory pointer is null");
          }
          char* gpu_base = static_cast<char*>(gpu_ptr_vec[0]);

          // Create CUDA stream for H2D transfers
          cudaStream_t h2d_stream;
          auto stream_status = stepcast::cuda::stream_create(&h2d_stream);
          if (!stream_status.ok()) {
            return stream_status;
          }

          // Convert chunk indices to a sorted vector for efficient reading
          std::vector<uint32_t> sorted_chunks = chunk_indices;
          std::sort(sorted_chunks.begin(), sorted_chunks.end());

          // Producer thread for chunk-aware disk reading
          auto chunk_reader_thread = [&](std::atomic<size_t>& next_idx,
                                         std::atomic<int>& active_producers) -> absl::Status {
            absl::Status status = absl::OkStatus();

            while (true) {
              size_t idx = next_idx.fetch_add(1);
              if (idx >= sorted_chunks.size())
                break;

              uint32_t chunk_id = sorted_chunks[idx];
              SC_TRACE_SCOPE("disk_read_chunk");

              // Get a free buffer slot
              auto slot_or = spb->get_free_chunk();
              if (!slot_or.ok()) {
                status = slot_or.status();
                break;
              }
              int slot = *slot_or;

              char* buffer = spb->get_chunk_ptr(slot);
              size_t global_offset = chunk_id * chunk_size;
              size_t bytes_to_read = std::min(chunk_size, model_size - global_offset);

              // Read data from disk
              absl::Status read_status = file_reader->read_at_offset(global_offset, buffer, bytes_to_read);
              if (!read_status.ok()) {
                LOG(ERROR) << "Failed to read chunk " << chunk_id << " from disk: " << read_status.message();
                status = read_status;
                auto return_status = spb->return_chunk(slot);
                if (!return_status.ok()) {
                  LOG(ERROR) << "Failed to return chunk after read error: " << return_status;
                }
                break;
              }

              // Mark chunk as ready for GPU transfer
              absl::Status mark_status = spb->mark_chunk_ready(slot, chunk_id, bytes_to_read);
              if (!mark_status.ok()) {
                status = mark_status;
                auto return_status = spb->return_chunk(slot);
                if (!return_status.ok()) {
                  LOG(ERROR) << "Failed to return chunk after mark_ready error: " << return_status;
                }
                break;
              }
            }

            // Decrement active producer count
            int remaining = active_producers.fetch_sub(1) - 1;
            if (remaining == 0) {
              spb->signal_production_complete();
              VLOG(1) << "Last producer thread finished for chunk loading";
            }

            return status;
          };

          // Start producer threads
          std::vector<std::future<absl::Status>> producers;
          std::atomic<size_t> next_idx{0};
          std::atomic<int> active_producers{num_threads};

          VLOG(1) << "DiskLoader: Starting chunk-aware GPU load with " << num_threads << " threads. Loading "
                  << sorted_chunks.size() << " chunks";

          producers.reserve(num_threads);
          for (int i = 0; i < num_threads; ++i) {
            producers.emplace_back(
                std::async(std::launch::async, [&]() { return chunk_reader_thread(next_idx, active_producers); }));
          }

          // Consumer thread for GPU copy
          size_t chunks_consumed = 0;
          absl::Status overall_status = absl::OkStatus();

          while (chunks_consumed < sorted_chunks.size()) {
            auto ready_or = spb->get_ready_chunk();
            if (!ready_or.ok()) {
              if (ready_or.status().code() == absl::StatusCode::kOutOfRange) {
                break; // Normal completion
              }
              overall_status = ready_or.status();
              break;
            }

            auto ready = *ready_or;

            // Perform async H2D copy to the correct chunk location
            st = trace_cuda_async(
                "h2d_copy_chunk",
                h2d_stream,
                [&]() {
                  return stepcast::cuda::memcpy_async(
                      gpu_base + ready.global_chunk_id * chunk_size,
                      ready.data_ptr,
                      ready.bytes_in_chunk,
                      cudaMemcpyHostToDevice,
                      h2d_stream);
                },
                [spb, slot_id = ready.slot_id]() {
                  auto _st = spb->return_chunk(slot_id);
                  if (!_st.ok()) {
                    LOG(ERROR) << "Failed to return chunk: " << _st;
                  }
                });

            if (!st.ok()) {
              overall_status = st;
              break;
            }

            chunks_consumed++;
          }

          // Wait for all producer threads
          for (auto& f : producers) {
            absl::Status producer_status = f.get();
            if (!producer_status.ok() && overall_status.ok()) {
              overall_status = producer_status;
            }
          }

          // Synchronize CUDA stream
          if (overall_status.ok()) {
            auto sync_status = stepcast::cuda::stream_synchronize(h2d_stream);
            if (!sync_status.ok()) {
              overall_status = sync_status;
            }
          }

          // Cleanup
          auto destroy_status = stepcast::cuda::stream_destroy(h2d_stream);
          if (!destroy_status.ok() && overall_status.ok()) {
            overall_status = destroy_status;
          }
          file_reader->close_all();

          // Release streaming buffer
          st = mem_manager->release_buffer_pool();
          if (!st.ok()) {
            LOG(ERROR) << "Failed to release buffer pool: " << st;
          }

          // Update UnifiedModelMemory states on success
          if (overall_status.ok()) {
            if (auto um = mem_manager->get_unified_memory()) {
              ChunkState new_state = ChunkState::COPIED_GPU;
              int dev_id = mem_manager->get_local_device_id();
              auto update_status = um->update_chunk_states(
                  mem_manager->instance_key(),
                  ModelLocation::GPU,
                  std::vector<uint32_t>(chunk_indices.begin(), chunk_indices.end()),
                  new_state,
                  dev_id);
              if (!update_status.ok()) {
                LOG(WARNING) << "Failed to update chunk states: " << update_status;
              }
            }
          }

          // Note: We don't finalise overall GPU MemoryState here; state transitions
          // are handled by higher-level coordination logic.

          VLOG(1) << "DiskLoader: Chunk-aware GPU load completed. Status: " << overall_status;
          return overall_status;
        });
  }

  // CPU path with chunk support
  if (target_location == ModelLocation::PAGEABLE_CPU) {
    return SC_TRACE_ASYNC(
        std::launch::async,
        [mem_manager,
         paths = std::move(paths_copy),
         part_sizes = std::move(sizes_copy),
         model_size = model_size_copy,
         chunk_indices = chunk_indices,
         chunk_size = chunk_size]() -> absl::Status {
          // For CPU loading with chunks, we need to check if DVMP region exists
          const std::string model_id = mem_manager->instance_key().model_id;

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

          // Get base address for the model
          void* base_addr = nullptr;
          {
            auto cpu_ptrs = mem_manager->get_pointer(ModelLocation::PAGEABLE_CPU);
            if (!cpu_ptrs.empty()) {
              base_addr = cpu_ptrs[0];
            }
          }

          if (!base_addr) {
            return absl::InternalError("CPU memory base pointer is null");
          }

          // Create file reader
          auto file_reader = std::make_shared<FilePartitionReader>();
          absl::Status st = file_reader->open_partitions(paths, part_sizes);
          if (!st.ok()) {
            return st;
          }

          // Sort chunks for sequential disk access
          std::vector<uint32_t> sorted_chunks = chunk_indices;
          std::sort(sorted_chunks.begin(), sorted_chunks.end());

          // Lock chunks before loading
          st = dvmp->lock_chunks(model_id, sorted_chunks);
          if (!st.ok()) {
            return st;
          }

          // Load each chunk
          for (uint32_t chunk_id : sorted_chunks) {
            size_t global_offset = chunk_id * chunk_size;
            size_t bytes_to_read = std::min(chunk_size, model_size - global_offset);

            char* chunk_ptr = static_cast<char*>(base_addr) + global_offset;

            // Read directly into DVMP memory
            st = file_reader->read_at_offset(global_offset, chunk_ptr, bytes_to_read);
            if (!st.ok()) {
              // Unlock chunks on failure
              auto unlock_status = dvmp->unlock_chunks(model_id, sorted_chunks, false);
              if (!unlock_status.ok()) {
                LOG(ERROR) << "Failed to unlock chunks after read error: " << unlock_status;
              }
              return st;
            }
          }

          // Close file reader
          file_reader->close_all();

          // Unlock chunks marking them as hot
          st = dvmp->unlock_chunks(model_id, sorted_chunks, /*copied_gpu=*/false);
          if (!st.ok()) {
            return st;
          }

          if (auto um = mem_manager->get_unified_memory()) {
            auto update_status = um->update_chunk_states(
                mem_manager->instance_key(), ModelLocation::PAGEABLE_CPU, sorted_chunks, ChunkState::HOT);
            if (!update_status.ok()) {
              LOG(WARNING) << "Failed to update chunk states: " << update_status;
            }
          }

          VLOG(1) << "DiskLoader: Loaded " << sorted_chunks.size() << " chunks to CPU memory";
          return absl::OkStatus();
        });
  }

  return std::async(std::launch::deferred, [target_location] {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "Unsupported target location for chunk loading: %s", location_to_string(target_location).c_str()));
  });
}

} // namespace stepcast::store