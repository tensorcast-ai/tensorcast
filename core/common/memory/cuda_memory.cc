// Copyright (c) 2025, TensorCast Team.

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

#include <iomanip>
#include <sstream>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "core/common/cuda_api.h"

namespace tensorcast::common::memory {

CudaMemory::~CudaMemory() {
  release_resources();
}

absl::Status CudaMemory::allocate(size_t size, int device_id) {
  if (allocation_type_ != AllocationType::UNINITIALIZED) {
    LOG(ERROR) << "CudaMemory::allocate called on an already initialized object (state="
               << static_cast<int>(allocation_type_) << ").";
    return absl::FailedPreconditionError("CudaMemory already initialized");
  }
  if (size == 0) {
    LOG(ERROR) << "CudaMemory::allocate called with size 0.";
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

  // Get IPC handle if possible (allocation succeeded)
  // Note: IPC handle might fail if IPC is disabled or on certain platforms/configurations
  std::string ipc_handle_str;
  status = cuda::get_ipc_handle(data_, &ipc_handle_str);
  if (!status.ok()) {
    // Log but do not treat as fatal if IPC handle cannot be acquired.
    LOG(WARNING) << "CudaMemory::allocate - get_ipc_handle failed (continuing without IPC handle): " << status;
  } else {
    // Convert string handle back to cudaIpcMemHandle_t for compatibility
    // This is a temporary measure until we fully migrate to string-based handles
    if (ipc_handle_str.size() == sizeof(cudaIpcMemHandle_t) * 2) {
      for (size_t i = 0; i < sizeof(cudaIpcMemHandle_t); ++i) {
        std::string byte_str = ipc_handle_str.substr(i * 2, 2);
        reinterpret_cast<unsigned char*>(&handle_)[i] = static_cast<unsigned char>(std::stoi(byte_str, nullptr, 16));
      }
    }
  }

  size_ = size;
  device_id_ = device_id;
  allocation_type_ = AllocationType::DIRECT;

  VLOG(1) << "CudaMemory: Directly allocated " << size_ << " bytes on device " << device_id_ << " at address " << data_
          << ", handle=" << ipc_handle_str;
  return absl::OkStatus();
}

absl::Status CudaMemory::adopt_ipc_handle(const cudaIpcMemHandle_t& ipc_handle, size_t size, int device_id) {
  if (allocation_type_ != AllocationType::UNINITIALIZED) {
    LOG(ERROR) << "CudaMemory::adopt_ipc_handle called on an already initialized object.";
    return absl::FailedPreconditionError("CudaMemory already initialized");
  }

  if (size == 0) {
    LOG(ERROR) << "CudaMemory::adopt_ipc_handle called with size 0.";
    return absl::InvalidArgumentError("Size must be > 0");
  }

  auto status = cuda::set_device(device_id);
  if (!status.ok()) {
    return status;
  }

  // Convert cudaIpcMemHandle_t to string for the abstraction layer
  std::stringstream ss;
  for (int i = 0; i < sizeof(cudaIpcMemHandle_t); ++i) {
    ss << std::hex << std::setw(2) << std::setfill('0')
       << static_cast<int>(reinterpret_cast<const unsigned char*>(&ipc_handle)[i]);
  }
  std::string handle_str = ss.str();

  void* opened_ptr = nullptr;
  status = cuda::open_ipc_handle(handle_str, &opened_ptr);
  if (!status.ok()) {
    return status;
  }
  if (opened_ptr == nullptr) {
    return absl::InternalError("open_ipc_handle returned nullptr");
  }

  data_ = opened_ptr;
  size_ = size;
  device_id_ = device_id;
  handle_ = ipc_handle;
  allocation_type_ = AllocationType::IPC_EXTERNAL;

  LOG(INFO) << "CudaMemory: Adopted CUDA IPC memory handle on device " << device_id_ << " ptr=" << data_ << " size "
            << size_;
  return absl::OkStatus();
}

void CudaMemory::release_resources() {
  if (data_ == nullptr && allocation_type_ == AllocationType::UNINITIALIZED) {
    return; // Nothing to do
  }

  switch (allocation_type_) {
    case AllocationType::DIRECT:
      if (data_) {
        VLOG(2) << "CudaMemory: Freeing directly allocated memory (" << size_ << " bytes) on device " << device_id_
                << " at address " << data_;
        // Best effort to set the correct device before freeing
        auto status = cuda::set_device(device_id_);
        if (!status.ok()) {
          LOG(ERROR) << "CudaMemory::release_resources - set_device(" << device_id_
                     << ") failed before free: " << status;
        }
        status = cuda::free(data_);
        if (!status.ok()) {
          LOG(ERROR) << "CudaMemory::release_resources - free failed for address " << data_ << ": " << status;
        }
        // Even if free fails, reset members to avoid double free attempts
      }
      break;
    case AllocationType::IPC_EXTERNAL:
      if (data_) {
        LOG(INFO) << "CudaMemory: Closing CUDA IPC memory handle for ptr " << data_ << " (" << size_ << " bytes)";
        auto status = cuda::set_device(device_id_);
        if (!status.ok()) {
          LOG(ERROR) << "CudaMemory::release_resources - set_device(" << device_id_
                     << ") failed before ipc close: " << status;
        }
        status = cuda::close_ipc_handle(data_);
        if (!status.ok()) {
          LOG(ERROR) << "CudaMemory::release_resources - close_ipc_handle failed for address " << data_ << ": "
                     << status;
        }
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
