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
  const uint64_t registration_id = next_registration_id_.fetch_add(1, std::memory_order_relaxed);

  for (size_t i = 0; i < buffer_addresses.size(); ++i) {
    // Export keys must be unique per registration, not just per address. Batch
    // payload pack buffers are intentionally reused, and reusing the same
    // communicator key can tear down an older live export while remote RDMA
    // reads are still in flight.
    std::string key =
        absl::StrCat("buffer_", registration_id, "_", i, "_", reinterpret_cast<uintptr_t>(buffer_addresses[i]));

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

absl::StatusOr<ExportRegistration> CommunicationManager::register_stable_local_backing_source_views(
    const std::vector<StableLocalBackingSourceView>& views) {
  if (!is_enabled()) {
    return absl::FailedPreconditionError("Communication engine not initialized");
  }
  if (views.empty()) {
    return absl::InvalidArgumentError("stable-backed source view registration requires at least one view");
  }

  const uint64_t registration_id = next_registration_id_.fetch_add(1, std::memory_order_relaxed);
  ExportRegistration info;
  info.location = common::memory::MemoryLocation::CPU;
  info.device_id = -1;
  info.comm_dev_type = communicator::base::COMMUNICATE_ENGINE_DEV_CPU;
  info.buffer_addresses.reserve(views.size());
  info.buffer_sizes.reserve(views.size());
  info.remote_memory_keys.reserve(views.size());

  std::vector<std::string> registered_keys;
  registered_keys.reserve(views.size());
  const auto cleanup_registered_keys = [&]() {
    for (const auto& key : registered_keys) {
      auto status = comm_engine_->unregister_tensor(key);
      if (!status.ok()) {
        LOG(WARNING) << "stable-backed source view cleanup failed"
                     << " key=" << key << " status=" << status;
      }
    }
  };
  for (size_t index = 0; index < views.size(); ++index) {
    const auto& view = views[index];
    if (view.address == 0 || view.size_bytes == 0) {
      cleanup_registered_keys();
      return absl::InvalidArgumentError("stable-backed source view requires non-empty address range");
    }
    if (view.keepalive == nullptr) {
      cleanup_registered_keys();
      return absl::InvalidArgumentError("stable-backed source view requires keepalive");
    }

    const std::string key = absl::StrCat("stable_backing_view_", registration_id, "_", index, "_", view.address);
    communicator::engine::Communicator::StableLocalBackingSourceView engine_view{
        .tensor_key = key,
        .addr = view.address,
        .bytes = static_cast<uint64_t>(view.size_bytes),
        .backing = view.backing,
        .keepalive = view.keepalive,
    };
    auto status = comm_engine_->register_stable_local_backing_source_view(engine_view);
    if (!status.ok()) {
      cleanup_registered_keys();
      return status;
    }
    registered_keys.push_back(key);
    info.buffer_addresses.push_back(view.address);
    info.buffer_sizes.push_back(view.size_bytes);
    info.remote_memory_keys.push_back(key);
    info.artifact_size += view.size_bytes;
  }

  return info;
}

absl::Status CommunicationManager::activate_stable_local_backing(
    const store::StableLocalBackingRef& backing,
    std::shared_ptr<void> keepalive) {
  if (!is_enabled()) {
    return absl::FailedPreconditionError("Communication engine not initialized");
  }
  return comm_engine_->activate_stable_local_backing(backing, std::move(keepalive));
}

absl::Status CommunicationManager::deactivate_stable_local_backing(std::string_view backing_id) {
  if (!is_enabled()) {
    return absl::OkStatus();
  }
  return comm_engine_->deactivate_stable_local_backing(backing_id);
}

bool CommunicationManager::stable_local_backing_supported_for_test() const {
  return comm_engine_ != nullptr && comm_engine_->stable_local_backing_supported_for_test();
}

bool CommunicationManager::stable_local_backing_active_for_test(std::string_view backing_id) const {
  return comm_engine_ != nullptr && comm_engine_->stable_local_backing_active_for_test(backing_id);
}

void CommunicationManager::set_routing_context(std::shared_ptr<communicator::routing::RoutingContext> routing_context) {
  absl::MutexLock lock(&routing_context_mu_);
  routing_context_ = std::move(routing_context);
}

std::shared_ptr<communicator::routing::RoutingContext> CommunicationManager::routing_context() const {
  absl::MutexLock lock(&routing_context_mu_);
  return routing_context_;
}

} // namespace tensorcast::store::components
