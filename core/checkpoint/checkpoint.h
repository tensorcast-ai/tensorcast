// Copyright (c) 2025-2026, TensorCast Team.

//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//  Modified by TensorCast Team, 2025-2026.
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
#pragma once

// Always include torch headers - they're needed for torch types even in fake CUDA mode
#include <torch/torch.h>

#if 0 // Disabled - always use real torch headers
// For fake CUDA, still include torch headers when building Python extensions
// The Python extension needs real PyTorch types even in fake CUDA mode
#ifdef TORCH_EXTENSION_NAME
#include <torch/torch.h>
#else
#include <memory>
// For non-Python builds, provide minimal torch types needed
namespace torch {
class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(void* data_ptr, std::shared_ptr<void> owner = nullptr)
      : data_ptr_(data_ptr), owner_(std::move(owner)) {}

  template <typename T>
  T* data_ptr() {
    return reinterpret_cast<T*>(data_ptr_);
  }

  template <typename T>
  T* data_ptr() const {
    return reinterpret_cast<T*>(data_ptr_);
  }

 private:
  void* data_ptr_ = nullptr;
  std::shared_ptr<void> owner_;
};
} // namespace torch
#endif
#endif // Disabled section
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/common/artifact_verification.h"

namespace tensorcast::checkpoint {

/**
 * @brief Save tensor data to disk in partitioned format.
 *
 * @param tensor_names List of tensor names to save.
 * @param tensor_data Map of tensor name to (data_ptr, size) pairs.
 * @param path Directory path where tensor data will be saved.
 * @return Map of tensor name to offset within the saved data.
 */
std::unordered_map<std::string, uint64_t> save_tensors(
    const std::vector<std::string>& tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const std::string& path);

/**
 * @brief Calculate actual artifact size from tensor_index.json, accounting for shared storages.
 *
 * Get the actual artifact size by summing unique tensor storage sizes,
 * avoiding double-counting shared storages.
 *
 * @param disk_path Path to the directory containing tensor_index.json
 * @return uint64_t Actual artifact size in bytes
 */
uint64_t calculate_actual_artifact_size(const std::string& disk_path);

/**
 * @brief Generate verification information for saved replica files.
 *
 * This function reads the saved tensor partition files and generates comprehensive
 * verification information including hashes and key checkpoints for integrity checking.
 * The artifact_size in the returned verification info is calculated from tensor_index.json
 * to reflect the actual artifact size, not the aligned file sizes.
 *
 * @param disk_path Path to the directory containing saved replica files.
 * @return ArtifactVerificationInfo Generated verification information.
 */
tensorcast::common::ArtifactVerificationInfo generate_verification_info_from_disk(
    const std::string& disk_path,
    tensorcast::common::VerificationLevel max_level = tensorcast::common::VerificationLevel::FULL_HASH);

/**
 * @brief Restore tensors from GPU memory into PyTorch tensors.
 *
 * This API supports a single device per call; callers must pass exactly one
 * entry in memory_base_address and tensor_device_offsets.
 */
std::unordered_map<std::string, torch::Tensor> restore_tensors(
    const std::unordered_map<
        std::string,
        std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::string, uint64_t>>& meta_state_dict,
    const std::unordered_map<int, std::uint64_t>& memory_base_address,
    const std::unordered_map<int, std::unordered_map<std::string, uint64_t>>& tensor_device_offsets,
    bool from_ipc_shm,
    std::string lease_token = "",
    std::string local_handle_socket_path = "");

/**
 * @brief Restore CPU tensors from a local memfd mapping and bind a daemon handle lease to tensor lifetime.
 *
 * The returned tensors share a refcounted owner that:
 *  - unmaps the memfd mapping, and
 *  - releases the daemon handle lease via the local handle socket (best-effort)
 * when the last tensor is destroyed.
 */
std::unordered_map<std::string, torch::Tensor> restore_tensors_from_cpu_fd_with_lease(
    const std::unordered_map<
        std::string,
        std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::string, uint64_t>>& meta_state_dict,
    int fd,
    uint64_t size_bytes,
    uint64_t offset_bytes,
    const std::unordered_map<std::string, uint64_t>& tensor_device_offsets,
    std::string lease_token,
    std::string local_handle_socket_path);

std::unordered_map<std::string, torch::Tensor> restore_tensors_from_disk(
    const std::unordered_map<
        std::string,
        std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::string, uint64_t>>& meta_state_dict,
    const std::string& disk_path,
    const std::unordered_map<std::string, uint64_t>& tensor_device_offsets,
    int device_id = -1);

// Allocate a single contiguous CUDA buffer on the specified device and return its base address.
std::uint64_t allocate_cuda_memory(int device_id, size_t tensor_size);

// Obtain a CUDA-IPC handle for the allocation that backs memory_ptr and return
// (handle_bytes, base_offset_bytes) where base_offset_bytes is the byte offset
// of memory_ptr into the exported allocation.
absl::StatusOr<std::pair<std::string, std::uint64_t>> get_cuda_memory_handle_with_offset(
    int device_id,
    std::uint64_t memory_ptr);

absl::StatusOr<std::uint64_t> get_cuda_memory_ptr(int device_id, const std::string& cuda_ipc_handle);

absl::Status close_cuda_memory_handle(int device_id, std::uint64_t cuda_ipc_ptr);

// Get a map of device IDs to their corresponding UUIDs.
std::unordered_map<int, std::string> get_device_uuid_map();

// Get a map of GPU IDs to their corresponding UUIDs.
std::unordered_map<std::string, int> get_gpu_uuid();
} // namespace tensorcast::checkpoint
