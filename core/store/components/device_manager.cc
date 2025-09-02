// Copyright (c) 2025, TensorCast Team.

#include "device_manager.h"

#include <cstdio>
#include "core/common/cuda_api.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/metrics/provider.h"

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::store::components {

void DeviceManager::gpu_mem_bytes_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept {
  auto* self = static_cast<DeviceManager*>(state);
  if (self == nullptr) {
    return;
  }
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  for (const auto& [device_id, info] : self->gpu_info_map_) {
    obs->Observe(
        static_cast<double>(info.total_memory),
        {{"location", opentelemetry::common::AttributeValue("gpu")},
         {"device_id", opentelemetry::common::AttributeValue(std::to_string(device_id))},
         {"memory_type", opentelemetry::common::AttributeValue("total")}});
    obs->Observe(
        static_cast<double>(info.free_memory),
        {{"location", opentelemetry::common::AttributeValue("gpu")},
         {"device_id", opentelemetry::common::AttributeValue(std::to_string(device_id))},
         {"memory_type", opentelemetry::common::AttributeValue("free")}});
  }
}

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

  // Prepare OTel meter and ObservableGauge for GPU memory bytes
  meter_ = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  gpu_memory_bytes_gauge_ = meter_->CreateDoubleObservableGauge("tc_memory_pool_bytes");
  gpu_memory_bytes_gauge_->AddCallback(&DeviceManager::gpu_mem_bytes_callback, this);

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
    size_t free_mem;
    size_t total_mem;
    auto mem_info_status = cuda::get_memory_info(&free_mem, &total_mem, i);
    if (!mem_info_status.ok()) {
      return mem_info_status;
    }
    info.total_memory = total_mem;
    info.free_memory = free_mem;

    // Initial snapshot values are stored in gpu_info_map_ and emitted by the ObservableGauge callback
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
    size_t free_mem;
    size_t total_mem;
    auto mem_status = cuda::get_memory_info(&free_mem, &total_mem, device_id);
    if (mem_status.ok()) {
      gpu_info_map_[device_id].free_memory = free_mem;
      gpu_info_map_[device_id].total_memory = total_mem;
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
  size_t free_mem;
  size_t total_mem;
  auto final_mem_status = cuda::get_memory_info(&free_mem, &total_mem, device_id);
  if (!final_mem_status.ok()) {
    return final_mem_status;
  }

  // Update cached value
  gpu_info_map_[device_id].free_memory = free_mem;
  return free_mem;
}

} // namespace tensorcast::store::components
