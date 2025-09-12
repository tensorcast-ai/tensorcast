// Copyright (c) 2025, TensorCast Team.

#include "communication_manager.h"

#include <cctype>
#include <cstdlib>
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "core/store/components/uma_lease_provider.h"

namespace tensorcast::store::components {

//-------------------------------------------------------------------------
// Constructor with externally provided Communicator (Phase-3 DI)
//-------------------------------------------------------------------------
CommunicationManager::CommunicationManager(std::shared_ptr<communicator::engine::Communicator> external_engine)
    : enabled_(external_engine != nullptr), comm_engine_(std::move(external_engine)) {}

absl::Status CommunicationManager::initialize(const std::string& listen_addr, uint16_t listen_port, bool enable_rdma) {
  // Phase-5: RDMA enable/disable is now explicitly provided by configuration.
  //          Environment variables are no longer consulted at this layer.

  communicator::v1::CommunicatorConfig cfg;
  cfg.set_enable_rdma(enable_rdma);
  comm_engine_ = std::make_shared<communicator::engine::Communicator>(cfg);

  auto status = comm_engine_->init(listen_addr, listen_port);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to initialize communication engine: " << status.message();
    comm_engine_.reset();
    return status;
  }

  enabled_ = true;
  // Inject UMA-backed lease provider for CPU staging (DRAM)
  comm_engine_->set_dram_lease_provider(UmaLeaseProvider::instance());
  LOG(INFO) << "Communication engine initialized on " << listen_addr << ":" << listen_port;
  return absl::OkStatus();
}

absl::Status CommunicationManager::initialize_with_config(
    const std::string& listen_addr,
    uint16_t listen_port,
    const communicator::v1::CommunicatorConfig& config) {
  comm_engine_ = std::make_shared<communicator::engine::Communicator>(config);

  auto status = comm_engine_->init(listen_addr, listen_port);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to initialize communication engine (config): " << status.message();
    comm_engine_.reset();
    return status;
  }

  enabled_ = true;
  // Inject UMA-backed lease provider for CPU staging (DRAM)
  comm_engine_->set_dram_lease_provider(UmaLeaseProvider::instance());
  LOG(INFO) << "Communication engine (config) initialized on " << listen_addr << ":" << listen_port;
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
    communicator::engine::Communicator::RegisterTensorOptions opts;
    opts.register_mr = comm_engine_->is_rdma_enabled();
    opts.needs_staging = (!comm_engine_->is_rdma_enabled() && device_id >= 0);
    opts.async = false;
    auto status = comm_engine_->register_tensor_ex(
        key,
        reinterpret_cast<uint64_t>(buffer_addresses[i]),
        buffer_sizes[i],
        device_id >= 0 ? communicator::base::COMMUNICATE_ENGINE_DEV_GPU
                       : communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
        device_id >= 0 ? device_id : 0,
        opts);

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
  info.comm_dev_type =
      device_id >= 0 ? communicator::base::COMMUNICATE_ENGINE_DEV_GPU : communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

  // Calculate total artifact size
  info.artifact_size = 0;
  for (size_t size : buffer_sizes) {
    info.artifact_size += size;
  }

  return info;
}

} // namespace tensorcast::store::components
