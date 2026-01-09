// Copyright (c) 2025-2026, TensorCast Team.

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
#include "core/common/memory/cuda_memory.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::common::memory {

GpuDeviceMemory::~GpuDeviceMemory() {
  release_resources();
}

absl::Status GpuDeviceMemory::allocate(size_t size, int device_id) {
  if (allocation_type_ != AllocationType::UNINITIALIZED) {
    LOG(ERROR) << "GpuDeviceMemory::allocate called on an already initialized object (state="
               << static_cast<int>(allocation_type_) << ").";
    return absl::FailedPreconditionError("GpuDeviceMemory already initialized");
  }
  if (size == 0) {
    LOG(ERROR) << "GpuDeviceMemory::allocate called with size 0.";
    return absl::InvalidArgumentError("Size must be > 0");
  }

  auto status = cuda::set_device(device_id);
  if (!status.ok()) {
    return status;
  }

  status = cuda::malloc(&data_, size);
  if (!status.ok()) {
    return status;
  }

  // Get native IPC mem handle if possible; continue without if unsupported
  status = cuda::get_ipc_mem_handle(&handle_, data_);
  if (!status.ok()) {
    LOG(WARNING) << "GpuDeviceMemory::allocate - get_ipc_mem_handle failed (continuing without IPC handle): " << status;
  }

  size_ = size;
  device_id_ = device_id;
  allocation_type_ = AllocationType::DIRECT;

  VLOG(1) << "GpuDeviceMemory: Directly allocated " << size_ << " bytes on device " << device_id_ << " at address "
          << data_;
  return absl::OkStatus();
}

absl::Status GpuDeviceMemory::adopt_ipc_handle(const cudaIpcMemHandle_t& ipc_handle, size_t size, int device_id) {
  if (allocation_type_ != AllocationType::UNINITIALIZED) {
    LOG(ERROR) << "GpuDeviceMemory::adopt_ipc_handle called on an already initialized object.";
    return absl::FailedPreconditionError("GpuDeviceMemory already initialized");
  }

  if (size == 0) {
    LOG(ERROR) << "GpuDeviceMemory::adopt_ipc_handle called with size 0.";
    return absl::InvalidArgumentError("Size must be > 0");
  }

  auto status = cuda::set_device(device_id);
  if (!status.ok()) {
    return status;
  }

  auto mapping_or = cuda::IpcMapping::open(ipc_handle);
  if (!mapping_or.ok()) {
    return mapping_or.status();
  }

  ipc_mapping_ = std::move(*mapping_or);
  data_ = ipc_mapping_.get();
  size_ = size;
  device_id_ = device_id;
  handle_ = ipc_handle;
  allocation_type_ = AllocationType::IPC_EXTERNAL;

  LOG(INFO) << "GpuDeviceMemory: Adopted CUDA IPC memory handle on device " << device_id_ << " ptr=" << data_
            << " size " << size_;
  return absl::OkStatus();
}

void GpuDeviceMemory::release_resources() {
  if (data_ == nullptr && allocation_type_ == AllocationType::UNINITIALIZED) {
    return; // Nothing to do
  }

  switch (allocation_type_) {
    case AllocationType::DIRECT:
      if (data_) {
        VLOG(2) << "GpuDeviceMemory: Freeing directly allocated memory (" << size_ << " bytes) on device " << device_id_
                << " at address " << data_;
        // Best effort to set the correct device before freeing
        auto status = cuda::set_device(device_id_);
        if (!status.ok()) {
          LOG(ERROR) << "GpuDeviceMemory::release_resources - set_device(" << device_id_
                     << ") failed before free: " << status;
        }
        status = cuda::free(data_);
        if (!status.ok()) {
          LOG(ERROR) << "GpuDeviceMemory::release_resources - free failed for address " << data_ << ": " << status;
        }
        // Even if free fails, reset members to avoid double free attempts
      }
      break;
    case AllocationType::IPC_EXTERNAL:
      if (data_) {
        LOG(INFO) << "GpuDeviceMemory: Closing CUDA IPC memory handle for ptr " << data_ << " (" << size_ << " bytes)";
        auto status = cuda::set_device(device_id_);
        if (!status.ok()) {
          LOG(ERROR) << "GpuDeviceMemory::release_resources - set_device(" << device_id_
                     << ") failed before ipc close: " << status;
        }
        ipc_mapping_.reset();
      }
      break;
    case AllocationType::UNINITIALIZED:
      // Should have been caught earlier, but safe to do nothing.
      break;
  }

  // Reset members regardless of success/failure to prevent reuse/double-release
  data_ = nullptr;
  size_ = 0;
  device_id_ = -1;
  handle_ = {};
  allocation_type_ = AllocationType::UNINITIALIZED;
}

} // namespace tensorcast::common::memory
