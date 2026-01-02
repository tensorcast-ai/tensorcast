// Copyright (c) 2025-2026, TensorCast Team.

#include "communication_manager.h"

#include <cctype>
#include <cstdlib>
#include <utility>
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "core/communicator/config_io.h"

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
  auto* stager = cfg.mutable_stager();
  stager->set_buffers_per_flow(4);
  tensorcast::communicator::normalize_defaults(&cfg);
  constexpr size_t kDefaultGpuSliceBytes = 16ULL * 1024 * 1024;
  constexpr size_t kDefaultCpuSliceBytes = 4ULL * 1024 * 1024;
  const size_t num_buffers = static_cast<size_t>(stager->buffers_per_flow());
  const size_t recv_buffers = num_buffers * static_cast<size_t>(cfg.transport().tcp_conn_count());
  const size_t gpu_pool_slices = num_buffers + recv_buffers;
  const size_t gpu_pool_bytes = gpu_pool_slices * kDefaultGpuSliceBytes;
  auto gpu_pool = std::make_shared<common::memory::PinnedBufferPool>(gpu_pool_bytes, kDefaultGpuSliceBytes);
  auto cpu_pool =
      std::make_shared<common::memory::PinnedBufferPool>(num_buffers * kDefaultCpuSliceBytes, kDefaultCpuSliceBytes);

  communicator::engine::Communicator::PinnedStagingPools pools{
      .gpu_pool = std::move(gpu_pool),
      .cpu_pool = std::move(cpu_pool),
      .preregister_gpu = enable_rdma,
      .preregister_cpu = enable_rdma,
  };
  comm_engine_ = std::make_shared<communicator::engine::Communicator>(cfg, std::move(pools));

  auto status = comm_engine_->init(listen_addr, listen_port);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to initialize communication engine: " << status.message();
    comm_engine_.reset();
    return status;
  }

  enabled_ = true;
  listen_port_ = comm_engine_->listening_port();
  // DRAM stager uses default no-op lease provider; UMA export keepalive holds leases.
  LOG(INFO) << "Communication engine initialized on " << listen_addr << ":" << listen_port_;
  return absl::OkStatus();
}

absl::Status CommunicationManager::initialize_with_config(
    const std::string& listen_addr,
    uint16_t listen_port,
    const communicator::v1::CommunicatorConfig& config) {
  // Standalone initialization path (not daemon): provide conservative defaults
  // for pinned pools since the authoritative sizing is handled by the daemon's
  // pinned_memory configuration.
  communicator::v1::CommunicatorConfig normalized = config;
  tensorcast::communicator::normalize_defaults(&normalized);
  constexpr size_t kDefaultGpuSliceBytes = 16ULL * 1024 * 1024;
  constexpr size_t kDefaultCpuSliceBytes = 4ULL * 1024 * 1024;
  const size_t num_buffers = static_cast<size_t>(std::max(1, normalized.stager().buffers_per_flow()));
  const size_t tcp_conn = static_cast<size_t>(normalized.transport().tcp_conn_count());
  const size_t gpu_pool_slices = num_buffers + (num_buffers * tcp_conn);
  const size_t gpu_pool_bytes = gpu_pool_slices * kDefaultGpuSliceBytes;
  auto gpu_pool = std::make_shared<common::memory::PinnedBufferPool>(gpu_pool_bytes, kDefaultGpuSliceBytes);
  auto cpu_pool =
      std::make_shared<common::memory::PinnedBufferPool>(num_buffers * kDefaultCpuSliceBytes, kDefaultCpuSliceBytes);

  communicator::engine::Communicator::PinnedStagingPools pools{
      .gpu_pool = std::move(gpu_pool),
      .cpu_pool = std::move(cpu_pool),
      .preregister_gpu = normalized.enable_rdma(),
      .preregister_cpu = normalized.enable_rdma(),
  };
  comm_engine_ = std::make_shared<communicator::engine::Communicator>(normalized, std::move(pools));

  auto status = comm_engine_->init(listen_addr, listen_port);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to initialize communication engine (config): " << status.message();
    comm_engine_.reset();
    return status;
  }

  enabled_ = true;
  listen_port_ = comm_engine_->listening_port();
  // DRAM stager uses default no-op lease provider; UMA export keepalive holds leases.
  LOG(INFO) << "Communication engine (config) initialized on " << listen_addr << ":" << listen_port_;
  return absl::OkStatus();
}

absl::Status CommunicationManager::initialize_with_config_and_pools(
    const std::string& listen_addr,
    uint16_t listen_port,
    const communicator::v1::CommunicatorConfig& config,
    communicator::engine::Communicator::PinnedStagingPools pools) {
  comm_engine_ = std::make_shared<communicator::engine::Communicator>(config, std::move(pools));

  auto status = comm_engine_->init(listen_addr, listen_port);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to initialize communication engine (config+pools): " << status.message();
    comm_engine_.reset();
    return status;
  }

  enabled_ = true;
  listen_port_ = comm_engine_->listening_port();
  LOG(INFO) << "Communication engine (config+pools) initialized on " << listen_addr << ":" << listen_port_;
  return absl::OkStatus();
}

absl::StatusOr<ExportRegistration> CommunicationManager::register_memory(
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
    opts.direct_rdma_enabled = comm_engine_->is_rdma_enabled() && !opts.needs_staging;
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
  ExportRegistration info;
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
