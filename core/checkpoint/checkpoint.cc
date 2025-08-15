// Copyright (c) 2025, StepCast Team. All rights reserved.

//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
#include "checkpoint.h"

#ifndef USE_FAKE_CUDA
#include <cublas_v2.h>
#include <nvml.h>
#endif
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include "absl/log/absl_check.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmacro-redefined"
#include "absl/log/log.h"
#pragma clang diagnostic pop

#include "core/common/cuda_api.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/common/model_verification.h" // Add verification support
#include "progress_bar.h"
#include "tensor_writer.h"

#define BUFFER_SIZE 1 << 30

namespace stepcast::store {

uint64_t calculate_actual_model_size(const std::string& model_path) {
  std::filesystem::path tensor_index_path = std::filesystem::path(model_path) / "tensor_index.json";

  if (!std::filesystem::exists(tensor_index_path)) {
    LOG(FATAL) << "tensor_index.json not found in: " << model_path;
  }

  std::ifstream file(tensor_index_path);
  if (!file.is_open()) {
    LOG(FATAL) << "Failed to open tensor_index.json: " << tensor_index_path.string();
  }

  nlohmann::json tensor_index;
  try {
    file >> tensor_index;
  } catch (const nlohmann::json::exception& e) {
    LOG(FATAL) << "Failed to parse tensor_index.json: " << e.what();
  }

  uint64_t total_memory = 0;
  std::set<std::pair<uint64_t, uint64_t>> seen; // (offset, size) pairs to avoid double counting shared storages

  for (const auto& [tensor_name, tensor_info] : tensor_index.items()) {
    if (!tensor_info.is_array() || tensor_info.size() < 2) {
      LOG(FATAL) << "Invalid tensor_index.json format for tensor: " << tensor_name;
    }

    uint64_t offset = tensor_info[0].get<uint64_t>();
    uint64_t size = tensor_info[1].get<uint64_t>();

    std::pair<uint64_t, uint64_t> storage_key = {offset, size};

    if (seen.find(storage_key) == seen.end()) {
      seen.insert(storage_key);
      total_memory += size;
    }
  }

  return total_memory;
}

std::unordered_map<std::string, uint64_t> save_tensors(
    const std::vector<std::string>& tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const std::string& path) {
  const std::string tensor_filename = std::filesystem::path(path) / "tensor.data";
  // make a tensor writer
  TensorWriter writer(tensor_filename);
  // make a tensor index
  std::unordered_map<std::string, uint64_t> tensor_offsets;

  // ------------------------------------------------------------------
  // Deduplication of shared storages (pointer-based)
  // ------------------------------------------------------------------
  // A PyTorch view/slice can share the same backing storage but report a
  // *smaller* logical size.  To preserve storage-sharing semantics we write
  // the largest slice for each unique storage pointer exactly once and let
  // all aliases reuse its offset.

  // First pass: gather the maximum size for every unique storage pointer and
  // remember the tensor name that owns that max slice (used when we actually
  // issue the write).
  struct StorageMeta {
    uint64_t max_size{0};
    std::string owner_name; // name whose (ptr,size) corresponds to max_size
  };
  std::unordered_map<const char*, StorageMeta> ptr_meta;

  for (const auto& n : tensor_names) {
    const auto& [base, size] = tensor_data[n];
    const char* ptr = reinterpret_cast<const char*>(base);
    auto& meta = ptr_meta[ptr];
    if (size > meta.max_size) {
      meta.max_size = size;
      meta.owner_name = n;
    }
  }

  // Second pass: write each tensor.  When we encounter a pointer for the
  // first time we write *max_size* bytes so that subsequent aliases (possibly
  // smaller) can safely reference the same region.
  std::unordered_map<const char*, uint64_t> ptr_written_offset;

  const int total = tensor_names.size();
  int count = 0;

  for (const auto& name : tensor_names) {
    const auto& [base, size] = tensor_data[name];
    const char* data_ptr = reinterpret_cast<const char*>(base);

    auto it_ptr = ptr_written_offset.find(data_ptr);
    if (it_ptr != ptr_written_offset.end()) {
      // Already written — reuse offset.
      tensor_offsets[name] = it_ptr->second;
    } else {
      // First occurrence — write *max_size* bytes irrespective of the current
      // tensor's logical size so that larger aliases are still valid.
      const uint64_t write_size = ptr_meta[data_ptr].max_size;

      // ------------------------------------------------------------------
      // Detect whether the source pointer resides in GPU (device) memory.  A
      // direct host-side memcpy() will segfault when handed a device pointer
      // because the CPU cannot dereference it.  We therefore fall back to a
      // two-step copy when the pointer is on the GPU:
      //   1. Allocate a temporary host buffer of the exact tensor size.
      //   2. cudaMemcpy() the data into that buffer.
      //   3. Flush the host buffer to disk via TensorWriter.
      //
      // For typical model checkpoints the backing storage of a tensor is
      // written exactly once irrespective of the number of aliasing views, so
      // the additional allocation does not cause repeated copies.
      // ------------------------------------------------------------------

      bool is_device_ptr = false;

      cudaPointerAttributes attr;
      auto attr_status =
          cuda::pointer_get_attributes_full(const_cast<void*>(static_cast<const void*>(data_ptr)), &attr);
      if (attr_status.ok()) {
#if CUDART_VERSION >= 10000
        is_device_ptr = (attr.type == cudaMemoryTypeDevice);
#else
        is_device_ptr = (attr.memoryType == cudaMemoryTypeDevice);
#endif
      } else {
        // Clear sticky error to avoid poisoning later CUDA calls (and reset).
        ABSL_CHECK_OK(cuda::get_last_error());
      }

      const char* host_src_ptr = data_ptr;
      std::unique_ptr<char[]> host_buffer;

      if (is_device_ptr) {
        host_buffer = std::make_unique<char[]>(write_size);
        auto copy_status = cuda::memcpy(host_buffer.get(), data_ptr, write_size, cudaMemcpyDeviceToHost);
        if (!copy_status.ok()) {
          LOG(FATAL) << "Failed to copy tensor data from GPU to host: " << copy_status.message();
        }
        host_src_ptr = host_buffer.get();
      }

      const uint64_t offset = writer.write_record(host_src_ptr, write_size);
      ptr_written_offset[data_ptr] = offset;
      tensor_offsets[name] = offset;
    }

    // Update progress bar
    count++;
    const float progress = static_cast<float>(count) / total;
    show_progress_bar(progress, "Saving tensors: ");
  }

  return tensor_offsets;
}

/**
 * @brief Generate verification information for saved model files.
 *
 * This function reads the saved tensor partition files and generates comprehensive
 * verification information including hashes and key checkpoints.
 *
 * @param model_path Path to the directory containing saved model files
 * @return ModelVerificationInfo Generated verification information
 */
ModelVerificationInfo generate_model_verification_info_from_disk(
    const std::string& model_path,
    VerificationLevel max_level) {
  std::filesystem::path model_dir_path(model_path);

  // Try partitioned files first; if none, use .safetensors payloads
  std::vector<std::filesystem::path> file_paths;
  std::vector<size_t> data_lengths; // file size for partitions; payload size for safetensors
  uint64_t declared_total = 0;
  bool is_safetensors = false;

  for (int i = 0;; ++i) {
    std::filesystem::path p = model_dir_path / ("tensor.data_" + std::to_string(i));
    if (!std::filesystem::exists(p)) {
      if (i == 0 && file_paths.empty()) {
        p = model_dir_path / "tensor.data";
        if (!std::filesystem::exists(p)) {
          break;
        }
      } else {
        break;
      }
    }
    if (!std::filesystem::is_regular_file(p)) {
      LOG(FATAL) << "Tensor data path is not a regular file: " << p.string();
    }
    std::error_code ec;
    uint64_t sz = std::filesystem::file_size(p, ec);
    if (ec) {
      LOG(FATAL) << "Failed to get file size for " << p.string() << ": " << ec.message();
    }
    file_paths.push_back(p);
    data_lengths.push_back(sz);
    declared_total += sz;
    if (p.filename() == "tensor.data")
      break;
  }

  if (file_paths.empty()) {
    std::vector<std::filesystem::path> st_paths;
    for (const auto& entry : std::filesystem::directory_iterator(model_dir_path)) {
      if (entry.is_regular_file()) {
        const auto name = entry.path().filename().string();
        const std::string ext = ".safetensors";
        if (name.size() >= ext.size() && name.rfind(ext) == name.size() - ext.size()) {
          st_paths.push_back(entry.path());
        }
      }
    }
    if (st_paths.empty()) {
      LOG(FATAL) << "No tensor.data partitions or .safetensors files found in: " << model_path;
    }
    std::sort(
        st_paths.begin(), st_paths.end(), [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
    for (const auto& p : st_paths) {
      std::error_code ec;
      uint64_t file_size = std::filesystem::file_size(p, ec);
      if (ec) {
        LOG(FATAL) << "Failed to get file size for " << p.string() << ": " << ec.message();
      }
      uint64_t header_len_le = 0;
      int fd = ::open(p.c_str(), O_RDONLY);
      if (fd < 0) {
        LOG(FATAL) << "Failed to open safetensors file: " << p.string();
      }
      ssize_t n = ::pread(fd, &header_len_le, sizeof(header_len_le), 0);
      ::close(fd);
      if (n != static_cast<ssize_t>(sizeof(header_len_le))) {
        LOG(FATAL) << "Invalid safetensors header length: " << p.string();
      }
      uint64_t data_start = sizeof(uint64_t) + header_len_le;
      if (data_start > file_size) {
        LOG(FATAL) << "Invalid safetensors layout: data beyond EOF in " << p.string();
      }
      uint64_t payload = file_size - data_start;
      file_paths.push_back(p);
      data_lengths.push_back(static_cast<size_t>(payload));
      declared_total += payload;
    }
    is_safetensors = true;
  }

  uint64_t actual_model_size = 0;
  if (!is_safetensors) {
    actual_model_size = calculate_actual_model_size(model_path);
    if (actual_model_size > declared_total) {
      LOG(FATAL) << "Actual model size (" << actual_model_size << ") exceeds total file size (" << declared_total
                 << ")";
    }
  } else {
    actual_model_size = declared_total;
  }

  LOG(INFO) << "Generating verification info for " << file_paths.size()
            << (is_safetensors ? " safetensors payloads" : " partitions")
            << ", actual model size: " << actual_model_size << " bytes (total: " << declared_total << ")";

  // ----------------------------------------------------------------------
  // Map each partition file into memory (read-only, private) instead of
  // copying the whole content. This keeps RSS low and lets the OS handle
  // paging transparently.
  // ----------------------------------------------------------------------
  struct MappedFile {
    void* addr{nullptr};
    size_t length{0};
    int fd{-1};
  };

  std::vector<void*> data_ptrs;
  std::vector<MappedFile> mapped_files; // RAII container to unmap/close later

  // Memory-map each partition file just once. This avoids an extra copy and reduces
  // peak RSS because the kernel page cache is shared between all processes.
  // We still respect `actual_model_size` so we do not map bytes beyond the real
  // model payload (files can be padded for alignment).

  uint64_t bytes_processed = 0;

  for (size_t i = 0; i < file_paths.size() && bytes_processed < actual_model_size; ++i) {
    const auto& path = file_paths[i];
    size_t len_or_payload = data_lengths[i];

    if (len_or_payload == 0) {
      continue; // Skip empty partitions
    }

    // Only map the portion of this partition that belongs to the actual model.
    size_t bytes_to_map;
    if (!is_safetensors) {
      bytes_to_map = std::min(static_cast<uint64_t>(len_or_payload), actual_model_size - bytes_processed);
    } else {
      // For safetensors, map the whole file and shift pointer to payload
      std::error_code ec;
      uint64_t file_size = std::filesystem::file_size(path, ec);
      if (ec) {
        LOG(FATAL) << "Failed to get file size for " << path.string() << ": " << ec.message();
      }
      bytes_to_map = static_cast<size_t>(file_size);
    }

    // Open file read-only
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      LOG(FATAL) << "Failed to open file: " << path.string() << ": " << std::strerror(errno);
    }

    void* addr = ::mmap(nullptr, bytes_to_map, PROT_READ, MAP_PRIVATE, fd, 0 /*offset*/);
    if (addr == MAP_FAILED) {
      ::close(fd);
      PLOG(FATAL) << "mmap failed for file: " << path.string();
    }

    // Advise the kernel that we will access the mapping sequentially so it can
    // perform read-ahead efficiently without flooding the page cache.
#ifdef MADV_SEQUENTIAL
    ::madvise(addr, bytes_to_map, MADV_SEQUENTIAL);
#endif

    if (!is_safetensors) {
      data_ptrs.push_back(addr);
    } else {
      // Advance pointer to data payload
      uint64_t header_len_le = 0;
      ssize_t n = ::pread(fd, &header_len_le, sizeof(header_len_le), 0);
      if (n != static_cast<ssize_t>(sizeof(header_len_le))) {
        ::munmap(addr, bytes_to_map);
        ::close(fd);
        LOG(FATAL) << "Invalid safetensors header length while mapping: " << path.string();
      }
      uint64_t data_start = sizeof(uint64_t) + header_len_le;
      char* payload_ptr = static_cast<char*>(addr) + data_start;
      data_ptrs.push_back(static_cast<void*>(payload_ptr));
    }

    MappedFile mapping{addr, bytes_to_map, fd};
    mapped_files.push_back(mapping);

    // Update partition_sizes to reflect actual bytes mapped
    data_lengths[i] = (!is_safetensors) ? bytes_to_map : len_or_payload; // reflect payload length for safetensors
    bytes_processed += bytes_to_map;
  }

  // Adjust partition_sizes to only include partitions we actually read
  data_lengths.resize(data_ptrs.size());

  // Generate verification info using CPU processing (device_id = -1)
  // This will use the actual model size, not file size
  absl::StatusOr<ModelVerificationInfo> verification_result =
      ModelVerifier::generate_verification_info(data_ptrs, data_lengths, -1, max_level);

  if (!verification_result.ok()) {
    // Clean up mappings before throwing
    for (const auto& mf : mapped_files) {
      if (mf.addr != nullptr && mf.addr != MAP_FAILED) {
        ::munmap(mf.addr, mf.length);
      }
      if (mf.fd >= 0) {
        ::close(mf.fd);
      }
    }
    LOG(FATAL) << "Failed to generate verification info: " << verification_result.status().message();
  }

  LOG(INFO) << "Successfully generated verification info for model at " << model_path
            << " (actual size: " << actual_model_size << " bytes)";

  // Unmap and close files now that we are done
  for (const auto& mf : mapped_files) {
    if (mf.addr != nullptr && mf.addr != MAP_FAILED) {
      ::munmap(mf.addr, mf.length);
    }
    if (mf.fd >= 0) {
      ::close(mf.fd);
    }
  }

  return verification_result.value();
}

// Mapping from string to at::ScalarType
at::ScalarType string_to_scalar_type(const std::string& dtype_str) {
  static const std::unordered_map<std::string, at::ScalarType> dtype_map = {
      {"torch.float16", torch::kFloat16},
      {"torch.float32", torch::kFloat32},
      {"torch.float64", torch::kFloat64},
      {"torch.int16", torch::kInt16},
      {"torch.int32", torch::kInt32},
      {"torch.int64", torch::kInt64},
      {"torch.uint8", torch::kUInt8},
      {"torch.int8", torch::kInt8},
      {"torch.bfloat16", torch::kBFloat16},
      {"torch.float8_e4m3fn", torch::kFloat8_e4m3fn}};

  auto it = dtype_map.find(dtype_str);
  if (it != dtype_map.end()) {
    return it->second;
  }

  LOG(FATAL) << "Unknown dtype string: " << dtype_str;
}

std::unordered_map<std::string, torch::Tensor> restore_tensors(
    const std::unordered_map<
        std::string,
        std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::string, uint64_t>>& meta_state_dict,
    const std::unordered_map<int, std::uint64_t>& memory_base_address,
    const std::unordered_map<int, std::unordered_map<std::string, uint64_t>>& tensor_device_offsets,
    const bool from_ipc_shm) {
#ifdef USE_FAKE_CUDA
  // Stub implementation for fake CUDA
  LOG(FATAL) << "restore_tensors is not supported in fake CUDA mode";
  return {};
#else
  std::unordered_map<std::string, torch::Tensor> state_dict;
  std::unordered_set<std::uint64_t> handled_memory;
  for (const auto& [device, tensor_offset] : tensor_device_offsets) {
    for (const auto& p : tensor_offset) {
      const std::string name = p.first;
      const uint64_t offset = p.second;

      if (memory_base_address.find(device) != memory_base_address.end()) {
        std::uint64_t base_address = memory_base_address.at(device);
        auto [sizes, strides, type_str, storage_offset_elems] = meta_state_dict.at(name);
        at::ScalarType dtype = string_to_scalar_type(type_str);

        const size_t element_size = c10::elementSize(dtype);
        const std::uint64_t data_address = base_address + offset + (storage_offset_elems * element_size);

        torch::Device tensor_device(torch::kCUDA, device);
        // LOG(INFO) << "Restore tensor " << name << " to device " << device << " with offset " << offset;

        auto deleter = [from_ipc_shm](void* ptr) {
          absl::Status status;
          if (from_ipc_shm) {
            status = cuda::close_ipc_mem_handle(ptr);
          } else {
            status = cuda::free(ptr);
          }

          if (!status.ok()) {
            LOG(ERROR) << "CUDA memory cleanup failed: " << status.message();
          }
        };

        if (offset == 0 && handled_memory.find(base_address) == handled_memory.end()) {
          const torch::Tensor real_tensor = torch::from_blob(
              reinterpret_cast<void*>(data_address),
              c10::makeArrayRef(sizes),
              c10::makeArrayRef(strides),
              deleter,
              torch::TensorOptions().device(tensor_device).dtype(dtype));
          state_dict[name] = real_tensor;
          handled_memory.insert(base_address);
          // std::cerr << "Tensor " << name << " is restored to device " <<
          // device << std::endl;
        } else {
          const torch::Tensor real_tensor = torch::from_blob(
              reinterpret_cast<void*>(data_address),
              sizes,
              strides,
              [](void* ptr) {},
              torch::TensorOptions().device(tensor_device).dtype(dtype));
          state_dict[name] = real_tensor;
        }
      } else {
        std::cerr << "Cannot find device " << device << std::endl;
        exit(1);
      }
    }
  }
  return state_dict;
#endif
}

std::unordered_map<std::string, torch::Tensor> restore_tensors_from_model_path(
    const std::unordered_map<
        std::string,
        std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::string, uint64_t>>& meta_state_dict,
    const std::string& model_path,
    const std::unordered_map<std::string, uint64_t>& tensor_device_offsets,
    int device_id) {
  // USE_FAKE_CUDA: CPU restore path enforced below; GPU path is compiled out.
  std::vector<std::filesystem::path> partition_paths;
  std::vector<uint64_t> partition_sizes;
  uint64_t total_model_size = 0;

  std::filesystem::path model_dir_path(model_path);

  for (int i = 0;; ++i) {
    std::filesystem::path partition_file_path = model_dir_path / ("tensor.data_" + std::to_string(i));
    if (!std::filesystem::exists(partition_file_path)) {
      if (i == 0 &&
          partition_paths
              .empty()) { // No tensor.data_0 found, try tensor.data for backward compatibility or single file case
        partition_file_path = model_dir_path / "tensor.data";
        if (!std::filesystem::exists(partition_file_path)) {
          // If neither tensor.data_0 nor tensor.data exists, and we expected tensors, it's an error.
          if (!meta_state_dict.empty()) {
            LOG(FATAL) << "No tensor data file found (tried tensor.data_0 and tensor.data) in: " << model_path;
          }
          break; // No files, no metadata, so nothing to do.
        }
      } else {
        break; // No more tensor.data_N files
      }
    }

    if (!std::filesystem::is_regular_file(partition_file_path)) {
      LOG(FATAL) << "Tensor data path is not a regular file: " << partition_file_path.string();
    }

    std::error_code ec;
    uint64_t file_size = std::filesystem::file_size(partition_file_path, ec);
    if (ec) {
      LOG(FATAL) << "Failed to get file size for " << partition_file_path.string() << ": " << ec.message();
    }

    partition_paths.push_back(partition_file_path);
    partition_sizes.push_back(file_size);
    total_model_size += file_size;

    // If we found tensor.data (not tensor.data_0), assume it's the only file.
    if (partition_file_path.filename() == "tensor.data") {
      break;
    }
  }

  if (total_model_size == 0) {
    if (!meta_state_dict.empty()) {
      LOG(FATAL) << "Tensor data files are empty or not found, but metadata expects tensors in: " << model_path;
    }
    return {}; // Return empty map if files are empty/not found and no tensors expected.
  }

  // ------------------------------------------------------------------
  // GPU PATH (device_id >= 0)
  // ------------------------------------------------------------------
#ifndef USE_FAKE_CUDA
  if (device_id >= 0) {
    // Resolve configuration from environment variables (reuse streaming writer envs)
    const char* chunk_size_env = std::getenv("STREAMING_CHUNK_SIZE_MB");
    const char* pool_size_env = std::getenv("STREAMING_POOL_SIZE_GB");
    const char* num_buffers_env = std::getenv("STREAMING_NUM_BUFFERS");

    size_t buffer_size_mb = chunk_size_env ? std::stoull(chunk_size_env) : 256; // default 256MB
    size_t num_buffers = num_buffers_env ? std::stoull(num_buffers_env) : 4;
    size_t pool_size_gb = pool_size_env ? std::stoull(pool_size_env) : 2; // default 2GB pool

    const size_t chunk_size = buffer_size_mb << 20; // bytes
    const size_t pool_size = pool_size_gb << 30; // bytes

    LOG(INFO) << "Streaming tensor load configuration: num_buffers=" << num_buffers
              << ", buffer_size=" << buffer_size_mb << "MB, pool_size=" << pool_size_gb
              << "GB, device_id=" << device_id;

    // Create pinned memory pool and allocate host buffers
    auto pinned_pool = std::make_shared<PinnedMemoryPool>(pool_size, chunk_size);
    std::vector<char*> pinned_buffers;
    if (pinned_pool->allocate(num_buffers * chunk_size, pinned_buffers) != 0) {
      LOG(FATAL) << "Failed to allocate pinned buffers from pool";
    }

    // Allocate contiguous GPU memory to hold the entire model
    std::uint64_t gpu_base_ptr = allocate_cuda_memory(device_id, static_cast<size_t>(total_model_size));
    if (gpu_base_ptr == 0) {
      LOG(FATAL) << "Failed to allocate GPU memory";
    }

    size_t current_global_offset = 0; // Offset within GPU buffer
    size_t buf_index = 0; // Round-robin index into pinned_buffers

    for (size_t p = 0; p < partition_paths.size(); ++p) {
      const auto& partition_file_path = partition_paths[p];
      uint64_t partition_file_size = partition_sizes[p];

      if (partition_file_size == 0) {
        continue; // skip empty partition
      }

      std::ifstream file_stream(partition_file_path, std::ios::binary);
      if (!file_stream.is_open()) {
        LOG(FATAL) << "Failed to open tensor data partition file: " << partition_file_path.string();
      }

      uint64_t remaining = partition_file_size;
      while (remaining > 0) {
        size_t to_read = std::min<uint64_t>(chunk_size, remaining);

        // Get host buffer
        char* host_buf = pinned_buffers[buf_index];

        // Read from disk into pinned buffer
        if (!file_stream.read(host_buf, static_cast<std::streamsize>(to_read))) {
          file_stream.close();
          LOG(FATAL) << "Failed to read tensor data chunk from: " << partition_file_path.string();
        }

        // Copy pinned buffer to GPU memory (synchronous copy is sufficient here)
        auto copy_status = cuda::memcpy(
            reinterpret_cast<void*>(gpu_base_ptr + current_global_offset), host_buf, to_read, cudaMemcpyHostToDevice);

        if (!copy_status.ok()) {
          file_stream.close();
          LOG(FATAL) << "CUDA memcpy failed: " << copy_status.message();
        }

        // Advance offsets
        remaining -= to_read;
        current_global_offset += to_read;

        buf_index = (buf_index + 1) % pinned_buffers.size();
      }

      file_stream.close();
    }

    // Deallocate pinned buffers back to pool
    pinned_pool->deallocate(pinned_buffers);

    // Build maps for restore_tensors
    std::unordered_map<int, std::uint64_t> memory_base_address;
    memory_base_address[device_id] = gpu_base_ptr;

    std::unordered_map<int, std::unordered_map<std::string, uint64_t>> device_offsets_map;
    device_offsets_map[device_id] = tensor_device_offsets;

    // Reuse restore_tensors to build actual torch::Tensors on GPU
    return restore_tensors(meta_state_dict, memory_base_address, device_offsets_map, /*from_ipc_shm=*/false);
  }
#else
  if (device_id >= 0) {
    LOG(WARNING) << "USE_FAKE_CUDA enabled; forcing CPU restore; ignoring device_id=" << device_id;
  }
#endif

  // ------------------------------------------------------------------
  // CPU PATH (device_id < 0) - original implementation continues below
  // ------------------------------------------------------------------

  std::shared_ptr<char[]> data_buffer(new char[static_cast<size_t>(total_model_size)]);
  char* current_buffer_ptr = data_buffer.get();

  for (size_t i = 0; i < partition_paths.size(); ++i) {
    const auto& partition_file_path = partition_paths[i];
    uint64_t partition_file_size = partition_sizes[i];

    if (partition_file_size == 0) {
      continue; // Skip empty partitions
    }

    std::ifstream file_read_data(partition_file_path, std::ios::binary);
    if (!file_read_data.is_open()) {
      LOG(FATAL) << "Failed to open tensor data partition file for reading: " << partition_file_path.string();
    }

    if (!file_read_data.read(current_buffer_ptr, static_cast<std::streamsize>(partition_file_size))) {
      file_read_data.close();
      LOG(FATAL) << "Failed to read tensor data partition file: " << partition_file_path.string();
    }
    file_read_data.close();
    current_buffer_ptr += partition_file_size;
  }

  std::unordered_map<std::string, torch::Tensor> state_dict;

  for (const auto& [name, meta_tuple] : meta_state_dict) {
    const auto& sizes_vec = std::get<0>(meta_tuple);
    const auto& strides_vec = std::get<1>(meta_tuple);
    const std::string& type_str = std::get<2>(meta_tuple);
    const uint64_t storage_offset_elems = std::get<3>(meta_tuple);

    at::ScalarType dtype = string_to_scalar_type(type_str);

    uint64_t offset;
    try {
      offset = tensor_device_offsets.at(name);
    } catch (const std::out_of_range& oor) {
      LOG(FATAL) << "Offset not found for tensor: " << name;
    }

    int64_t num_elements = 1;
    if (sizes_vec.empty()) { // Scalar tensor
      num_elements = 1;
    } else {
      for (int64_t s : sizes_vec) {
        if (s == 0) { // If any dimension is 0, total elements is 0
          num_elements = 0;
          break;
        }
        num_elements *= s;
      }
    }
    if (num_elements < 0) {
      num_elements = 0; // Safety for overflow
    }

    size_t element_size_bytes = c10::elementSize(dtype);
    uint64_t tensor_size_bytes = static_cast<uint64_t>(num_elements) * element_size_bytes;

    const uint64_t storage_offset_bytes = storage_offset_elems * element_size_bytes;
    const uint64_t end_pos = offset + storage_offset_bytes + tensor_size_bytes;

    if (end_pos > total_model_size) {
      if (tensor_size_bytes > 0 || offset > total_model_size) {
        LOG(FATAL) << "Tensor " << name << " data (offset " << offset << " + storage_offset " << storage_offset_bytes
                   << ", calculated size " << tensor_size_bytes << ") exceeds total model size (" << total_model_size
                   << ").";
      }
    }

    void* tensor_ptr = data_buffer.get() + offset + storage_offset_bytes;

    c10::IntArrayRef sizes(sizes_vec);
    c10::IntArrayRef strides(strides_vec);

    auto deleter = [captured_data_buffer = data_buffer](void* /*ptr_unused*/) {
      // This lambda captures captured_data_buffer by copy (of the shared_ptr),
      // extending the lifetime of the underlying char array.
    };

    torch::Tensor tensor =
        torch::from_blob(tensor_ptr, sizes, strides, deleter, torch::TensorOptions().device(torch::kCPU).dtype(dtype));

    state_dict[name] = tensor;
  }

  return state_dict;
}

std::unordered_map<std::string, int> get_gpu_uuid() {
  int device_count = 0;
  auto device_count_status = cuda::get_device_count(&device_count);
  if (!device_count_status.ok()) {
    LOG(ERROR) << "Failed to get device count: " << device_count_status.message();
    return {};
  }

  std::unordered_map<std::string, int> uuid_to_device_id_map;

  for (int dev_id = 0; dev_id < device_count; ++dev_id) {
    cudaDeviceProp props;
    auto props_status = cuda::get_device_properties(dev_id, &props);
    if (!props_status.ok()) {
      LOG(ERROR) << "Failed to get device properties for device " << dev_id << ": " << props_status.message();
      continue;
    }

    // Convert UUID bytes to string with unsigned char casting
    char uuid_str[80];
    snprintf(
        uuid_str,
        sizeof(uuid_str),
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        (unsigned char)props.uuid.bytes[0],
        (unsigned char)props.uuid.bytes[1],
        (unsigned char)props.uuid.bytes[2],
        (unsigned char)props.uuid.bytes[3],
        (unsigned char)props.uuid.bytes[4],
        (unsigned char)props.uuid.bytes[5],
        (unsigned char)props.uuid.bytes[6],
        (unsigned char)props.uuid.bytes[7],
        (unsigned char)props.uuid.bytes[8],
        (unsigned char)props.uuid.bytes[9],
        (unsigned char)props.uuid.bytes[10],
        (unsigned char)props.uuid.bytes[11],
        (unsigned char)props.uuid.bytes[12],
        (unsigned char)props.uuid.bytes[13],
        (unsigned char)props.uuid.bytes[14],
        (unsigned char)props.uuid.bytes[15]);

    uuid_to_device_id_map[std::string(uuid_str)] = dev_id;
  }

  return uuid_to_device_id_map;
}

std::uint64_t allocate_cuda_memory(int device_id, size_t tensor_size) {
  void* ptr = nullptr;
  auto set_device_status = cuda::set_device(device_id);
  if (!set_device_status.ok()) {
    LOG(ERROR) << "Failed to set CUDA device " << device_id << ": " << set_device_status.message();
    return 0;
  }

  auto malloc_status = cuda::malloc(&ptr, tensor_size);
  if (!malloc_status.ok()) {
    LOG(ERROR) << "Failed to allocate CUDA memory: " << malloc_status.message();
    return 0;
  }

  return reinterpret_cast<std::uint64_t>(ptr);
}

std::string get_cuda_memory_handle(int device_id, std::uint64_t memory_ptr) {
  cudaIpcMemHandle_t handle;
  auto set_device_status = cuda::set_device(device_id);
  if (!set_device_status.ok()) {
    LOG(ERROR) << "Failed to set CUDA device " << device_id << ": " << set_device_status.message();
    return "";
  }

  auto ipc_status = cuda::get_ipc_mem_handle(&handle, reinterpret_cast<void*>(memory_ptr));
  if (!ipc_status.ok()) {
    LOG(ERROR) << "Failed to get IPC memory handle: " << ipc_status.message();
    return "";
  }

  return std::string(reinterpret_cast<const char*>(&handle), sizeof(cudaIpcMemHandle_t));
}

absl::StatusOr<std::uint64_t> get_cuda_memory_ptr(int device_id, const std::string& cuda_ipc_handle) {
  cudaIpcMemHandle_t ipc_handle;
  memcpy(&ipc_handle, cuda_ipc_handle.data(), sizeof(cudaIpcMemHandle_t));

  auto set_device_status = cuda::set_device(device_id);
  if (!set_device_status.ok()) {
    return set_device_status;
  }

  void* opened_ptr = nullptr;
  auto ipc_open_status = cuda::open_ipc_mem_handle(&opened_ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
  if (!ipc_open_status.ok()) {
    return ipc_open_status;
  }

  if (opened_ptr == nullptr) {
    return absl::InternalError("IPC memory handle opened but returned nullptr pointer");
  }

  return reinterpret_cast<std::uint64_t>(opened_ptr);
}

absl::Status close_cuda_memory_handle(int device_id, std::uint64_t cuda_ipc_ptr) {
  auto set_device_status = cuda::set_device(device_id);
  if (!set_device_status.ok()) {
    return set_device_status;
  }

  auto close_status = cuda::close_ipc_mem_handle(reinterpret_cast<void*>(cuda_ipc_ptr));
  if (!close_status.ok()) {
    return close_status;
  }

  return absl::OkStatus();
}

std::unordered_map<int, std::string> get_device_uuid_map() {
  const std::unordered_map<std::string, int> gpu_uuid = get_gpu_uuid();
  std::unordered_map<int, std::string> device_uuid_map;
  for (const auto& p : gpu_uuid) {
    if (device_uuid_map.find(p.second) != device_uuid_map.end()) {
      std::cerr << "Duplicate device id: " << p.second << std::endl;
      exit(1);
    }
    device_uuid_map[p.second] = p.first;
  }
  return device_uuid_map;
}

} // namespace stepcast::store
