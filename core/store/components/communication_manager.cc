// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "communication_manager.h"

#include <cctype>
#include <cstdlib>
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

namespace stepcast::store {

//-------------------------------------------------------------------------
// Constructor with externally provided CommunicateEngine (Phase-3 DI)
//-------------------------------------------------------------------------
CommunicationManager::CommunicationManager(std::shared_ptr<stepcast::communicator::CommunicateEngine> external_engine)
    : enabled_(external_engine != nullptr), comm_engine_(std::move(external_engine)) {}

absl::Status CommunicationManager::initialize(const std::string& listen_addr, uint16_t listen_port, bool enable_rdma) {
  // Phase-5: RDMA enable/disable is now explicitly provided by configuration.
  //          Environment variables are no longer consulted at this layer.

  comm_engine_ = std::make_shared<stepcast::communicator::CommunicateEngine>(enable_rdma);

  auto status = comm_engine_->init(listen_addr, listen_port);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to initialize communication engine: " << status.message();
    comm_engine_.reset();
    return status;
  }

  enabled_ = true;
  VLOG(1) << "Communication engine initialized on " << listen_addr << ":" << listen_port;
  return absl::OkStatus();
}

absl::StatusOr<CommRegistrationInfo> CommunicationManager::register_memory(
    const std::vector<void*>& buffer_addresses,
    const std::vector<size_t>& buffer_sizes,
    int device_id) {
  if (!is_enabled()) {
    return absl::FailedPreconditionError("Communication engine not initialized");
  }

  if (buffer_addresses.size() != buffer_sizes.size()) {
    return absl::InvalidArgumentError("Buffer addresses and sizes must have same length");
  }

  // Register buffers with communication engine
  std::vector<std::string> remote_keys;
  remote_keys.reserve(buffer_addresses.size());

  for (size_t i = 0; i < buffer_addresses.size(); ++i) {
    // Generate a unique key for this buffer
    std::string key = absl::StrCat("buffer_", i, "_", reinterpret_cast<uintptr_t>(buffer_addresses[i]));

    // Register the tensor/buffer
    auto status = comm_engine_->register_tensor(
        key,
        reinterpret_cast<uint64_t>(buffer_addresses[i]),
        buffer_sizes[i],
        device_id >= 0 ? stepcast::communicator::COMMUNICATE_ENGINE_DEV_GPU
                       : stepcast::communicator::COMMUNICATE_ENGINE_DEV_CPU,
        device_id >= 0 ? device_id : 0);

    if (!status.ok()) {
      return absl::InternalError(absl::StrCat("Failed to register buffer ", i, ": ", status.message()));
    }
    remote_keys.push_back(key);
  }

  // Build registration info
  CommRegistrationInfo info;
  // Convert void* vector to uintptr_t vector
  for (void* addr : buffer_addresses) {
    info.buffer_addresses.push_back(reinterpret_cast<uintptr_t>(addr));
  }
  info.buffer_sizes = buffer_sizes;
  info.remote_memory_keys = remote_keys;
  info.device_id = device_id;
  info.comm_dev_type = device_id >= 0 ? stepcast::communicator::COMMUNICATE_ENGINE_DEV_GPU
                                      : stepcast::communicator::COMMUNICATE_ENGINE_DEV_CPU;

  // Calculate total model size
  info.model_size = 0;
  for (size_t size : buffer_sizes) {
    info.model_size += size;
  }

  return info;
}

} // namespace stepcast::store