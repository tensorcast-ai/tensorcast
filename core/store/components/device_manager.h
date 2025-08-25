// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <string>
#include <unordered_map>
#include "core/common/cuda_api.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/common/metrics/metric_objects.h"

namespace stepcast::store {

/**
 * @brief Manages GPU devices and CUDA operations for the store engine.
 *
 * This component handles:
 * - GPU device discovery and initialization
 * - CUDA stream management
 * - Device UUID mapping
 * - GPU memory queries and metrics
 */
class DeviceManager {
 public:
  struct GpuInfo {
    std::string uuid;
    size_t total_memory = 0;
    size_t free_memory = 0;
    cudaStream_t stream = nullptr;
  };

  DeviceManager();
  ~DeviceManager();

  // Disable copy and move
  DeviceManager(const DeviceManager&) = delete;
  DeviceManager& operator=(const DeviceManager&) = delete;
  DeviceManager(DeviceManager&&) = delete;
  DeviceManager& operator=(DeviceManager&&) = delete;

  /**
   * @brief Initialize GPU devices and create CUDA streams.
   * @return Status indicating success or failure
   */
  absl::Status initialize();

  /**
   * @brief Get the number of available GPUs.
   */
  int get_num_gpus() const {
    return num_gpus_;
  }

  /**
   * @brief Get GPU info by device ID.
   * @param device_id GPU device ID
   * @return GPU info or error if device not found
   */
  absl::StatusOr<const GpuInfo*> get_gpu_info(int device_id) const;

  /**
   * @brief Find device ID by UUID.
   * @param uuid GPU UUID string
   * @return Device ID or error if not found
   */
  absl::StatusOr<int> find_device_by_uuid(const std::string& uuid) const;

  /**
   * @brief Get CUDA stream for a specific device.
   * @param device_id GPU device ID
   * @return CUDA stream or error
   */
  absl::StatusOr<cudaStream_t> get_stream(int device_id) const;

  /**
   * @brief Update GPU memory metrics for all devices.
   */
  void update_gpu_metrics();

  /**
   * @brief Get current free memory for a device.
   * @param device_id GPU device ID
   * @return Free memory in bytes or error
   */
  absl::StatusOr<size_t> get_free_memory(int device_id);

 private:
  int num_gpus_ = 0;
  std::unordered_map<int, GpuInfo> gpu_info_map_;

  // Metrics
  std::unordered_map<int, stepcast::metrics::Gauge> gpu_memory_total_gauges_;
  std::unordered_map<int, stepcast::metrics::Gauge> gpu_memory_free_gauges_;
  std::unordered_map<int, stepcast::metrics::Gauge> gpu_replicas_loaded_gauges_;
};

} // namespace stepcast::store