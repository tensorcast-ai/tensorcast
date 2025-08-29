// Copyright (c) 2025, TensorCast Team.

#include "device_manager.h"

#include <cstdio>
#include "core/common/cuda_api.h"

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::store {

DeviceManager::DeviceManager() = default;

DeviceManager::~DeviceManager() {
  // Clean up CUDA streams
  for (auto& [device_id, gpu_info] : gpu_info_map_) {
    if (gpu_info.stream != nullptr) {
      auto status = cuda::set_device(device_id);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to set device " << device_id << " during cleanup: " << status.message();
      }
      status = cuda::stream_destroy(gpu_info.stream);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to destroy stream for device " << device_id << ": " << status.message();
      }
    }
  }
}

absl::Status DeviceManager::initialize() {
  auto count_status = cuda::get_device_count(&num_gpus_);
  if (!count_status.ok()) {
    return count_status;
  }

  if (num_gpus_ <= 0) {
    return absl::FailedPreconditionError("No GPUs found");
  }

  LOG(INFO) << "Initializing DeviceManager with " << num_gpus_ << " GPUs";

  for (int i = 0; i < num_gpus_; ++i) {
    auto device_status = cuda::set_device(i);
    if (!device_status.ok()) {
      return device_status;
    }

    cudaDeviceProp props;
    auto props_status = cuda::get_device_properties(i, &props);
    if (!props_status.ok()) {
      return props_status;
    }

    // Format GPU UUID
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

    GpuInfo& info = gpu_info_map_[i];
    info.uuid = std::string(uuid_str);

    LOG(INFO) << "GPU " << i << ": " << props.name << ", UUID: " << info.uuid;

    // Create CUDA stream
    auto stream_status = cuda::stream_create(&info.stream);
    if (!stream_status.ok()) {
      return stream_status;
    }

    // Initialize memory info
    size_t free_mem, total_mem;
    auto mem_info_status = cuda::get_memory_info(&free_mem, &total_mem, i);
    if (!mem_info_status.ok()) {
      return mem_info_status;
    }
    info.total_memory = total_mem;
    info.free_memory = free_mem;

    // Initialize metrics for this device
    std::string device_id_str = std::to_string(i);
    gpu_memory_total_gauges_.emplace(
        i, metrics::Gauge("store_daemon_gpu_memory_bytes", {{"device_id", device_id_str}, {"memory_type", "total"}}));
    gpu_memory_free_gauges_.emplace(
        i, metrics::Gauge("store_daemon_gpu_memory_bytes", {{"device_id", device_id_str}, {"memory_type", "free"}}));
    gpu_replicas_loaded_gauges_.emplace(
        i, tensorcast::metrics::Gauge("store_daemon_gpu_replicas_loaded", {{"device_id", device_id_str}}));

    // Set initial metric values
    gpu_memory_total_gauges_.at(i).set(static_cast<double>(total_mem));
    gpu_memory_free_gauges_.at(i).set(static_cast<double>(free_mem));
    gpu_replicas_loaded_gauges_.at(i).set(0.0);
  }

  return absl::OkStatus();
}

absl::StatusOr<const DeviceManager::GpuInfo*> DeviceManager::get_gpu_info(int device_id) const {
  auto it = gpu_info_map_.find(device_id);
  if (it == gpu_info_map_.end()) {
    return absl::NotFoundError(absl::StrCat("GPU device ", device_id, " not found"));
  }
  return &it->second;
}

absl::StatusOr<int> DeviceManager::find_device_by_uuid(const std::string& uuid) const {
  for (const auto& [device_id, info] : gpu_info_map_) {
    if (info.uuid == uuid) {
      return device_id;
    }
  }
  return absl::NotFoundError(absl::StrCat("No GPU found with UUID: ", uuid));
}

absl::StatusOr<cudaStream_t> DeviceManager::get_stream(int device_id) const {
  auto info_result = get_gpu_info(device_id);
  if (!info_result.ok()) {
    return info_result.status();
  }
  return (*info_result)->stream;
}

void DeviceManager::update_gpu_metrics() {
  for (const auto& [device_id, info] : gpu_info_map_) {
    auto status = cuda::set_device(device_id);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to set device " << device_id << " for metrics update: " << status.message();
      continue;
    }
    size_t free_mem, total_mem;
    auto mem_status = cuda::get_memory_info(&free_mem, &total_mem, device_id);
    if (mem_status.ok()) {
      gpu_memory_free_gauges_.at(device_id).set(static_cast<double>(free_mem));
      gpu_memory_total_gauges_.at(device_id).set(static_cast<double>(total_mem));
    }
  }
}

absl::StatusOr<size_t> DeviceManager::get_free_memory(int device_id) {
  auto info_result = get_gpu_info(device_id);
  if (!info_result.ok()) {
    return info_result.status();
  }

  auto status = cuda::set_device(device_id);
  if (!status.ok()) {
    return status;
  }
  size_t free_mem, total_mem;
  auto final_mem_status = cuda::get_memory_info(&free_mem, &total_mem, device_id);
  if (!final_mem_status.ok()) {
    return final_mem_status;
  }

  // Update cached value
  gpu_info_map_[device_id].free_memory = free_mem;
  return free_mem;
}

} // namespace tensorcast::store