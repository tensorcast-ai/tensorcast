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
#pragma once

#ifndef USE_FAKE_CUDA
#include <torch/torch.h>
#else
// For fake CUDA, still include torch headers when building Python extensions
// The Python extension needs real PyTorch types even in fake CUDA mode
#ifdef TORCH_EXTENSION_NAME
#include <torch/torch.h>
#else
// For non-Python builds, provide minimal torch types needed
namespace torch {
class Tensor {};
} // namespace torch
#endif
#endif
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/common/model_verification.h"

namespace stepcast::store {

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
 * @brief Calculate actual model size from tensor_index.json, accounting for shared storages.
 *
 * Get the actual model size by summing unique tensor storage sizes,
 * avoiding double-counting shared storages.
 *
 * @param model_path Path to the directory containing tensor_index.json
 * @return uint64_t Actual model size in bytes
 */
uint64_t calculate_actual_model_size(const std::string& model_path);

/**
 * @brief Generate verification information for saved model files.
 *
 * This function reads the saved tensor partition files and generates comprehensive
 * verification information including hashes and key checkpoints for integrity checking.
 * The model_size in the returned verification info is calculated from tensor_index.json
 * to reflect the actual model size, not the aligned file sizes.
 *
 * @param model_path Path to the directory containing saved model files.
 * @return ModelVerificationInfo Generated verification information.
 */
ModelVerificationInfo generate_model_verification_info_from_disk(
    const std::string& model_path,
    stepcast::store::VerificationLevel max_level = stepcast::store::VerificationLevel::FULL_HASH);

/**
 * @brief Restore tensors from GPU memory into PyTorch tensors.
 */
std::unordered_map<std::string, torch::Tensor> restore_tensors(
    const std::unordered_map<
        std::string,
        std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::string, uint64_t>>& meta_state_dict,
    const std::unordered_map<int, std::uint64_t>& memory_base_address,
    const std::unordered_map<int, std::unordered_map<std::string, uint64_t>>& tensor_device_offsets,
    const bool from_ipc_shm);

std::unordered_map<std::string, torch::Tensor> restore_tensors_from_model_path(
    const std::unordered_map<
        std::string,
        std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::string, uint64_t>>& meta_state_dict,
    const std::string& model_path,
    const std::unordered_map<std::string, uint64_t>& tensor_device_offsets,
    int device_id = -1);

// Allocate a single contiguous CUDA buffer on the specified device and return its base address.
std::uint64_t allocate_cuda_memory(int device_id, size_t tensor_size);

// Obtain a CUDA-IPC handle for a single allocation.
std::string get_cuda_memory_handle(int device_id, std::uint64_t memory_ptr);

absl::StatusOr<std::uint64_t> get_cuda_memory_ptr(int device_id, const std::string& cuda_ipc_handle);

absl::Status close_cuda_memory_handle(int device_id, std::uint64_t cuda_ipc_ptr);

// Get a map of device IDs to their corresponding UUIDs.
std::unordered_map<int, std::string> get_device_uuid_map();

// Get a map of GPU IDs to their corresponding UUIDs.
std::unordered_map<std::string, int> get_gpu_uuid();
} // namespace stepcast::store
