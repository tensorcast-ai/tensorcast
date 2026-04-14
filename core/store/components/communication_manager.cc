// Copyright (c) 2025-2026, TensorCast Team.

#include "communication_manager.h"

#include <algorithm>
#include <charconv>
#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "core/communicator/base/constants.h"
#include "core/communicator/config_io.h"
#include "core/communicator/topology/discovery/host_topology_builder.h"

namespace tensorcast::store::components {
namespace {

std::optional<int> parse_device_ordinal(std::string_view ordinal_text) {
  if (ordinal_text.empty()) {
    return std::nullopt;
  }
  int parsed = 0;
  const char* begin = ordinal_text.data();
  const char* end = begin + ordinal_text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, parsed);
  if (ec != std::errc() || ptr != end || parsed < 0) {
    return std::nullopt;
  }
  return parsed;
}

size_t normalized_tcp_conn_count(const communicator::v1::CommunicatorConfig& config) {
  int tcp_conn_count = config.transport().tcp_conn_count();
  if (tcp_conn_count <= 0) {
    tcp_conn_count = communicator::base::kMTcpConnCount;
  }
  return static_cast<size_t>(std::max(2, tcp_conn_count));
}

size_t required_gpu_staging_slices(const communicator::v1::CommunicatorConfig& config) {
  const size_t num_buffers = static_cast<size_t>(std::max(1, config.stager().buffers_per_flow()));
  const size_t conn_count = normalized_tcp_conn_count(config);
  return num_buffers + (num_buffers * conn_count);
}

size_t required_cpu_staging_slices(const communicator::v1::CommunicatorConfig& config) {
  return static_cast<size_t>(std::max(1, config.stager().buffers_per_flow()));
}

} // namespace

std::optional<CommunicationManager::ParsedEndpointId> CommunicationManager::parse_canonical_endpoint_id(
    std::string_view endpoint_id) {
  constexpr std::string_view kCpuMarker = "/dev/cpu/";
  constexpr std::string_view kGpuMarker = "/dev/gpu/";

  const size_t cpu_pos = endpoint_id.rfind(kCpuMarker);
  const size_t gpu_pos = endpoint_id.rfind(kGpuMarker);
  const bool is_cpu = cpu_pos != std::string_view::npos;
  const bool is_gpu = gpu_pos != std::string_view::npos;
  if (is_cpu == is_gpu) {
    return std::nullopt;
  }

  const size_t marker_pos = is_cpu ? cpu_pos : gpu_pos;
  const std::string_view marker = is_cpu ? kCpuMarker : kGpuMarker;
  const std::string_view node_id = endpoint_id.substr(0, marker_pos);
  if (node_id.empty()) {
    return std::nullopt;
  }

  const auto parsed_ordinal = parse_device_ordinal(endpoint_id.substr(marker_pos + marker.size()));
  if (!parsed_ordinal.has_value()) {
    return std::nullopt;
  }

  ParsedEndpointId result;
  result.node_id = std::string(node_id);
  result.dev_type =
      is_gpu ? communicator::base::COMMUNICATE_ENGINE_DEV_GPU : communicator::base::COMMUNICATE_ENGINE_DEV_CPU;
  result.dev_id = *parsed_ordinal;
  return result;
}

communicator::routing::EndpointBinding CommunicationManager::make_endpoint_binding(
    std::string_view endpoint_id,
    const ParsedEndpointId& parsed,
    std::string_view ip,
    uint16_t port) {
  communicator::routing::EndpointBinding binding;
  binding.endpoint_id = std::string(endpoint_id);
  binding.node_id = parsed.node_id;
  binding.ip = std::string(ip);
  binding.port = port;
  binding.dev_type = parsed.dev_type;
  binding.dev_id = parsed.dev_id;
  binding.rail_id = -1;
  return binding;
}

bool CommunicationManager::endpoint_binding_equals(
    const communicator::routing::EndpointBinding& lhs,
    const communicator::routing::EndpointBinding& rhs) {
  return lhs.endpoint_id == rhs.endpoint_id && lhs.node_id == rhs.node_id && lhs.ip == rhs.ip &&
      lhs.port == rhs.port && lhs.dev_type == rhs.dev_type && lhs.dev_id == rhs.dev_id &&
      lhs.pci_bdf == rhs.pci_bdf && lhs.rail_id == rhs.rail_id && lhs.gpu_uuid == rhs.gpu_uuid;
}

//-------------------------------------------------------------------------
// Constructor with externally provided Communicator (Phase-3 DI)
//-------------------------------------------------------------------------
CommunicationManager::CommunicationManager(std::shared_ptr<communicator::engine::Communicator> external_engine)
    : enabled_(external_engine != nullptr), comm_engine_(std::move(external_engine)) {}

absl::Status CommunicationManager::initialize(const std::string& listen_addr, uint16_t listen_port, bool enable_rdma) {
  // Phase-5: RDMA enable/disable is now explicitly provided by configuration.
  // Environment variables are no longer consulted at this layer.

  communicator::v1::CommunicatorConfig cfg;
  cfg.set_enable_rdma(enable_rdma);
  auto* stager = cfg.mutable_stager();
  stager->set_buffers_per_flow(4);
  tensorcast::communicator::normalize_defaults(&cfg);
  communicator_config_ = cfg;
  constexpr size_t kDefaultGpuSliceBytes = 16ULL * 1024 * 1024;
  constexpr size_t kDefaultCpuSliceBytes = 4ULL * 1024 * 1024;
  const size_t num_buffers = required_cpu_staging_slices(cfg);
  const size_t gpu_pool_slices = required_gpu_staging_slices(cfg);
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
  communicator_config_ = normalized;
  constexpr size_t kDefaultGpuSliceBytes = 16ULL * 1024 * 1024;
  constexpr size_t kDefaultCpuSliceBytes = 4ULL * 1024 * 1024;
  const size_t num_buffers = required_cpu_staging_slices(normalized);
  const size_t gpu_pool_slices = required_gpu_staging_slices(normalized);
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
  communicator::v1::CommunicatorConfig normalized = config;
  tensorcast::communicator::normalize_defaults(&normalized);
  communicator_config_ = normalized;
  comm_engine_ = std::make_shared<communicator::engine::Communicator>(normalized, std::move(pools));

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

  // Register buffers with communication engine.
  std::vector<std::string> remote_keys;
  remote_keys.reserve(buffer_addresses.size());

  for (size_t i = 0; i < buffer_addresses.size(); ++i) {
    std::string key = absl::StrCat("buffer_", i, "_", reinterpret_cast<uintptr_t>(buffer_addresses[i]));

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

  ExportRegistration info;
  for (void* addr : buffer_addresses) {
    info.buffer_addresses.push_back(reinterpret_cast<uintptr_t>(addr));
  }
  info.buffer_sizes = buffer_sizes;
  info.remote_memory_keys = remote_keys;
  info.device_id = device_id;
  info.comm_dev_type =
      device_id >= 0 ? communicator::base::COMMUNICATE_ENGINE_DEV_GPU : communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

  info.artifact_size = 0;
  for (size_t size : buffer_sizes) {
    info.artifact_size += size;
  }

  return info;
}

void CommunicationManager::set_routing_context(std::shared_ptr<communicator::routing::RoutingContext> routing_context) {
  absl::MutexLock lock(&routing_mu_);
  routing_context_ = std::move(routing_context);
}

absl::Status CommunicationManager::bootstrap_routing_context() {
  if (!communicator_config_.simple_numa().enable()) {
    absl::MutexLock lock(&routing_mu_);
    routing_topology_.reset();
    routing_context_.reset();
    return absl::OkStatus();
  }

  {
    absl::MutexLock lock(&routing_mu_);
    if (routing_topology_.has_value() && routing_context_ != nullptr) {
      return absl::OkStatus();
    }
  }

  auto topology_or =
      communicator::topology::discovery::build_topology_from_discovery_with_observability(communicator_config_);
  if (!topology_or.ok()) {
    return topology_or.status();
  }

  {
    absl::MutexLock lock(&routing_mu_);
    if (routing_topology_.has_value() && routing_context_ != nullptr) {
      return absl::OkStatus();
    }
    routing_topology_ = topology_or->topology;
    absl::Status rebuild_status = rebuild_routing_context_locked();
    if (!rebuild_status.ok()) {
      routing_topology_.reset();
      routing_context_.reset();
      return rebuild_status;
    }
  }
  const auto& obs = topology_or->observability;
  LOG(INFO) << "CommunicationManager: bootstrapped routing context"
            << " discovery_enabled=" << obs.discovery_enabled << " nic_endpoints=" << obs.nic_endpoint_count
            << " rail_switch_endpoints=" << obs.rail_switch_endpoint_count << " lldp_records="
            << obs.lldp_record_count << " nvlink_edges=" << obs.nvlink_edge_count;
  return absl::OkStatus();
}

void CommunicationManager::remember_p2p_source(const P2PSource& source) {
  std::vector<communicator::routing::EndpointBinding> changed_bindings;
  std::shared_ptr<communicator::routing::RoutingContext> context_to_update;
  absl::Status routing_status;
  {
    absl::MutexLock lock(&routing_mu_);

    if (const auto local = parse_canonical_endpoint_id(source.local_endpoint_id); local.has_value()) {
      const auto local_binding = make_endpoint_binding(source.local_endpoint_id, *local, /*ip=*/"", /*port=*/0);
      if (upsert_routing_binding_locked(local_binding)) {
        changed_bindings.push_back(local_binding);
      }
    } else if (!source.local_endpoint_id.empty()) {
      VLOG(1) << "Skipping non-canonical local endpoint binding: " << source.local_endpoint_id;
    }

    if (!source.remote_endpoint_id.empty() && !source.ip.empty() && source.port != 0) {
      if (const auto remote = parse_canonical_endpoint_id(source.remote_endpoint_id); remote.has_value()) {
        const auto remote_binding =
            make_endpoint_binding(source.remote_endpoint_id, *remote, source.ip, source.port);
        if (upsert_routing_binding_locked(remote_binding)) {
          changed_bindings.push_back(remote_binding);
        }
      } else {
        VLOG(1) << "Skipping non-canonical remote endpoint binding: " << source.remote_endpoint_id;
      }
    }

    if (!changed_bindings.empty() && routing_topology_.has_value()) {
      if (routing_context_) {
        context_to_update = routing_context_;
      } else {
        routing_status = rebuild_routing_context_locked();
      }
    }
  }

  if (context_to_update != nullptr && !changed_bindings.empty()) {
    routing_status = context_to_update->upsert_endpoint_bindings(std::move(changed_bindings));
    if (!routing_status.ok()) {
      LOG(WARNING) << "Failed to upsert live routing bindings; keeping existing routing context intact: "
                   << routing_status;
    }
  }

  if (!routing_status.ok()) {
    LOG(WARNING) << "Failed to apply routing bindings after remembering P2P source endpoints: " << routing_status;
  }
}

bool CommunicationManager::upsert_routing_binding_locked(const communicator::routing::EndpointBinding& binding) {
  if (binding.endpoint_id.empty()) {
    return false;
  }
  auto it = routing_bindings_.find(binding.endpoint_id);
  if (it != routing_bindings_.end() && endpoint_binding_equals(it->second, binding)) {
    return false;
  }
  routing_bindings_[binding.endpoint_id] = binding;
  return true;
}

std::vector<communicator::routing::EndpointBinding> CommunicationManager::collect_routing_bindings_locked() const {
  std::vector<communicator::routing::EndpointBinding> bindings;
  bindings.reserve(routing_bindings_.size());
  for (const auto& [endpoint_id, binding] : routing_bindings_) {
    bindings.push_back(binding);
  }
  std::sort(bindings.begin(), bindings.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.endpoint_id < rhs.endpoint_id;
  });
  return bindings;
}

absl::Status CommunicationManager::rebuild_routing_context_locked() {
  if (!routing_topology_.has_value()) {
    routing_context_.reset();
    return absl::OkStatus();
  }

  auto next_context = std::make_shared<communicator::routing::RoutingContext>(
      communicator::routing::RoutingContext::Options{}, comm_engine_);
  absl::Status status = next_context->set_topology(*routing_topology_);
  if (!status.ok()) {
    return status;
  }

  if (!routing_bindings_.empty()) {
    std::vector<communicator::routing::EndpointBinding> bindings = collect_routing_bindings_locked();
    status = next_context->set_endpoint_bindings(std::move(bindings));
    if (!status.ok()) {
      return status;
    }
  }

  routing_context_ = std::move(next_context);
  return absl::OkStatus();
}

} // namespace tensorcast::store::components
