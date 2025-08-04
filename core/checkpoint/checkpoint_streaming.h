// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/checkpoint/streaming_tensor_writer.h"

namespace stepcast::store {

/**
 * @brief Save tensor data to disk using streaming approach with better GPU support.
 *
 * This function uses StreamingTensorWriter for efficient GPU tensor saving with:
 * - Asynchronous GPU->Host copy and disk writing
 * - Configurable buffer sizes and counts
 * - Better memory management with circular buffers
 *
 * @param tensor_names List of tensor names to save.
 * @param tensor_data Map of tensor name to (data_ptr, size) pairs.
 * @param path Directory path where tensor data will be saved.
 * @param config Optional configuration for streaming writer.
 * @return Map of tensor name to offset within the saved data.
 */
std::unordered_map<std::string, uint64_t> save_tensors_streaming(
    const std::vector<std::string>& tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const std::string& path,
    const StreamingTensorWriter::Config& config = StreamingTensorWriter::Config{});

} // namespace stepcast::store