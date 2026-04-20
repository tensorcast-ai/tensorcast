// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/routing/routing_context.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/communicator/base/constants.h"
#include "core/communicator/routing/read_helpers.h"

namespace tensorcast::communicator::routing {
namespace {

constexpr std::string_view kNicEndpointPrefix = "nic_";
constexpr std::string_view kRailSwitchPrefix = "netsw_rail_";
constexpr std::string_view kGpuEndpointMarker = "/dev/gpu/";
constexpr std::string_view kCpuEndpointMarker = "/dev/cpu/";

struct DevicePoolHint {
  std::string gpu_pool_id;
  std::string cpu_pool_id;
};

std::optional<int> parse_decimal_int(std::string_view text) {
  int value = 0;
  const auto* begin = text.data();
  const auto* end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc() || ptr != end) {
    return std::nullopt;
  }
  return value;
}

std::optional<int> parse_gpu_dev_id_from_endpoint_id(std::string_view endpoint_id) {
  const size_t marker_pos = endpoint_id.rfind(kGpuEndpointMarker);
  if (marker_pos == std::string_view::npos) {
    return std::nullopt;
  }
  const size_t value_pos = marker_pos + kGpuEndpointMarker.size();
  if (value_pos >= endpoint_id.size()) {
    return std::nullopt;
  }
  return parse_decimal_int(endpoint_id.substr(value_pos));
}

std::optional<int> parse_cpu_dev_id_from_endpoint_id(std::string_view endpoint_id) {
  const size_t marker_pos = endpoint_id.rfind(kCpuEndpointMarker);
  if (marker_pos == std::string_view::npos) {
    return std::nullopt;
  }
  const size_t value_pos = marker_pos + kCpuEndpointMarker.size();
  if (value_pos >= endpoint_id.size()) {
    return std::nullopt;
  }
  return parse_decimal_int(endpoint_id.substr(value_pos));
}

std::optional<int> parse_rail_id_from_switch_endpoint_id(std::string_view endpoint_id) {
  if (!endpoint_id.starts_with(kRailSwitchPrefix)) {
    return std::nullopt;
  }
  auto value = parse_decimal_int(endpoint_id.substr(kRailSwitchPrefix.size()));
  if (!value.has_value() || *value < 0) {
    return std::nullopt;
  }
  return value;
}

bool endpoint_has_pool(const topology::Endpoint& endpoint, std::string_view pool_id) {
  return std::find(endpoint.pool_ids.begin(), endpoint.pool_ids.end(), pool_id) != endpoint.pool_ids.end();
}

bool is_nic_binding_id(std::string_view endpoint_id) {
  return endpoint_id.starts_with(kNicEndpointPrefix);
}

std::string_view device_type_to_string(int dev_type) {
  switch (dev_type) {
    case base::COMMUNICATE_ENGINE_DEV_GPU:
      return "GPU";
    case base::COMMUNICATE_ENGINE_DEV_CPU:
      return "CPU";
    default:
      return "UNKNOWN";
  }
}

std::string_view protocol_adapter_name(ConnectionProtocol protocol) {
  switch (protocol) {
    case ConnectionProtocol::kNvlink:
      return "NVLINK";
    case ConnectionProtocol::kPcie:
      return "PCIE";
    default:
      return "ENGINE";
  }
}

std::string describe_binding(const EndpointBinding& binding) {
  return std::format(
      "endpoint={},node={},dev={}#{},rail={},addr={}",
      binding.endpoint_id,
      binding.node_id,
      device_type_to_string(binding.dev_type),
      binding.dev_id,
      binding.rail_id,
      binding.has_network_address() ? std::format("{}:{}", binding.ip, binding.port) : "none");
}

DevicePoolHint infer_device_pool_hint(std::string_view endpoint_id, const EndpointBinding& binding) {
  DevicePoolHint hint;

  if (const auto gpu_id = parse_gpu_dev_id_from_endpoint_id(endpoint_id); gpu_id.has_value() && *gpu_id >= 0) {
    hint.gpu_pool_id = std::format("gpu{}", *gpu_id);
  }
  if (const auto cpu_id = parse_cpu_dev_id_from_endpoint_id(endpoint_id); cpu_id.has_value() && *cpu_id >= 0) {
    hint.cpu_pool_id = std::format("cpu{}", *cpu_id);
  }
  if (!hint.gpu_pool_id.empty() || !hint.cpu_pool_id.empty()) {
    return hint;
  }

  if (binding.dev_type == base::COMMUNICATE_ENGINE_DEV_GPU && binding.dev_id >= 0) {
    hint.gpu_pool_id = std::format("gpu{}", binding.dev_id);
    return hint;
  }
  if (binding.dev_type == base::COMMUNICATE_ENGINE_DEV_CPU && binding.dev_id >= 0) {
    hint.cpu_pool_id = std::format("cpu{}", binding.dev_id);
    return hint;
  }
  return hint;
}

ConnectionType connection_type_from_link(const topology::Link* link) {
  if (link == nullptr) {
    return ConnectionType::kP2P;
  }
  switch (link->type) {
    case topology::LinkType::kForward:
      return ConnectionType::kForward;
    case topology::LinkType::kP2P:
      return ConnectionType::kP2P;
    case topology::LinkType::kSwitch:
      return ConnectionType::kSwitch;
    default:
      return ConnectionType::kForward;
  }
}

} // namespace

RoutingContext::Communicator::Communicator(
    std::string src_endpoint_id,
    std::string dst_endpoint_id,
    std::shared_ptr<RoutingContext> context)
    : src_endpoint_id_(std::move(src_endpoint_id)),
      dst_endpoint_id_(std::move(dst_endpoint_id)),
      context_(std::move(context)) {}

transport::future_read_result_t RoutingContext::Communicator::read_tensor(const ReadRequest& request) {
  auto channel_or = primary_channel();
  if (!channel_or.ok()) {
    return make_failed_read_future(channel_or.status(), request.tensor_key);
  }
  return channel_or.value()->read_tensor(request);
}

transport::future_read_result_t RoutingContext::Communicator::read_plan(const ReadPlan& plan) {
  const absl::Status plan_status = validate_read_plan(plan);
  if (!plan_status.ok()) {
    return make_failed_read_future(plan_status, read_plan_tensor_key(plan));
  }

  const SourceSlice& first_source = plan.source_slices.front();
  if (first_source.route.local_endpoint_id != src_endpoint_id_ ||
      first_source.route.remote_endpoint_id != dst_endpoint_id_) {
    return make_failed_read_future(
        absl::InvalidArgumentError(
            std::format(
                "read plan route context does not match communicator endpoints: plan={} -> {}, communicator={} -> {}",
                first_source.route.local_endpoint_id,
                first_source.route.remote_endpoint_id,
                src_endpoint_id_,
                dst_endpoint_id_)),
        read_plan_tensor_key(plan));
  }

  auto channel_or = primary_channel();
  if (!channel_or.ok()) {
    return make_failed_read_future(channel_or.status(), read_plan_tensor_key(plan));
  }

  const std::shared_ptr<RouteChannel>& channel = channel_or.value();
  if (channel == nullptr || channel->hops().empty() || channel->hops().front() == nullptr) {
    return make_failed_read_future(
        absl::FailedPreconditionError("read plan resolved an empty route channel"), read_plan_tensor_key(plan));
  }

  const std::shared_ptr<Connection>& hop = channel->hops().front();
  const ConnectionProtocol selected_protocol = hop->protocol();
  if (first_source.route.protocol != ConnectionProtocol::kAuto && selected_protocol != ConnectionProtocol::kAuto &&
      first_source.route.protocol != selected_protocol) {
    return make_failed_read_future(
        absl::InvalidArgumentError(
            std::format(
                "read plan protocol does not match selected channel: plan={} selected={}",
                to_string(first_source.route.protocol),
                to_string(selected_protocol))),
        read_plan_tensor_key(plan));
  }

  if (first_source.route.rail_id >= 0) {
    const int local_rail_id = hop->local_binding().rail_id;
    const int remote_rail_id = hop->remote_binding().rail_id;
    if ((local_rail_id >= 0 || remote_rail_id >= 0) && first_source.route.rail_id != local_rail_id &&
        first_source.route.rail_id != remote_rail_id) {
      return make_failed_read_future(
          absl::InvalidArgumentError(
              std::format(
                  "read plan rail_id does not match selected channel: plan={} local={} remote={}",
                  first_source.route.rail_id,
                  local_rail_id,
                  remote_rail_id)),
          read_plan_tensor_key(plan));
    }
  }

  return channel->read_plan(plan);
}

absl::StatusOr<std::shared_ptr<RouteChannel>> RoutingContext::Communicator::primary_channel() {
  absl::MutexLock lock(&mu_);
  const uint64_t generation = context_->topology_generation();
  if (primary_channel_ && primary_channel_generation_ == generation) {
    return primary_channel_;
  }
  primary_channel_.reset();
  auto channel_or = context_->build_direct_channel_with_generation(src_endpoint_id_, dst_endpoint_id_);
  if (!channel_or.ok()) {
    return channel_or.status();
  }
  primary_channel_ = channel_or->channel;
  primary_channel_generation_ = channel_or->generation;
  return primary_channel_;
}

RoutingContext::RoutingContext(
    Options options,
    std::shared_ptr<engine::Communicator> engine,
    std::shared_ptr<ConnectionAdapter> nvlink_adapter,
    std::shared_ptr<ConnectionAdapter> pcie_adapter,
    std::shared_ptr<common::AsyncRuntime> async_runtime)
    : options_(options),
      engine_(std::move(engine)),
      engine_adapter_(std::make_shared<EngineAdapter>(engine_)),
      nvlink_adapter_(std::move(nvlink_adapter)),
      pcie_adapter_(std::move(pcie_adapter)),
      async_runtime_(std::move(async_runtime)) {
  if (!async_runtime_) {
    async_runtime_ = std::make_shared<common::AsyncRuntime>();
  }
  if (!nvlink_adapter_) {
    nvlink_adapter_ = std::make_shared<NvlinkAdapter>(engine_);
  }
  if (!pcie_adapter_) {
    pcie_adapter_ = std::make_shared<PcieAdapter>(engine_);
  }
}

absl::Status RoutingContext::set_topology(topology::Topology topology) {
  absl::MutexLock lock(&mu_);
  if (topology_) {
    return absl::FailedPreconditionError("topology is immutable once set");
  }
  topology_ = std::make_shared<topology::Topology>(std::move(topology));
  connections_.clear();
  communicators_.clear();
  link_states_.clear();
  topology_generation_ += 1;
  return absl::OkStatus();
}

absl::Status RoutingContext::set_endpoint_bindings(std::vector<EndpointBinding> bindings) {
  absl::flat_hash_map<std::string, EndpointBinding> next_bindings;
  next_bindings.reserve(bindings.size());
  for (auto& binding : bindings) {
    if (binding.endpoint_id.empty()) {
      return absl::InvalidArgumentError("endpoint binding id must be non-empty");
    }
    next_bindings[binding.endpoint_id] = std::move(binding);
  }
  absl::MutexLock lock(&mu_);
  if (bindings_initialized_) {
    return absl::FailedPreconditionError("endpoint bindings are immutable once set");
  }
  bindings_ = std::move(next_bindings);
  bindings_initialized_ = true;
  return absl::OkStatus();
}

absl::Status RoutingContext::update_endpoint_binding(EndpointBinding binding) {
  if (binding.endpoint_id.empty()) {
    return absl::InvalidArgumentError("endpoint binding id must be non-empty");
  }
  return absl::FailedPreconditionError("endpoint bindings are immutable; update_endpoint_binding is not supported");
}

absl::StatusOr<std::shared_ptr<RoutingContext::Communicator>> RoutingContext::get_communicator(
    const std::string& src_endpoint_id,
    const std::string& dst_endpoint_id) {
  if (src_endpoint_id.empty() || dst_endpoint_id.empty()) {
    return absl::InvalidArgumentError("communicator endpoint ids must be non-empty");
  }
  absl::MutexLock lock(&mu_);
  const std::string key = make_peer_key(src_endpoint_id, dst_endpoint_id);
  auto it = communicators_.find(key);
  if (it != communicators_.end()) {
    auto existing = it->second.lock();
    if (existing) {
      return existing;
    }
  }
  auto communicator =
      std::shared_ptr<Communicator>(new Communicator(src_endpoint_id, dst_endpoint_id, shared_from_this()));
  communicators_[key] = communicator;
  return communicator;
}

absl::Status RoutingContext::register_tensor_ex(
    const std::string& tensor_key,
    uint64_t tensor_addr,
    uint64_t tensor_bytes,
    int dev_type,
    int dev_id,
    const engine::Communicator::RegisterTensorOptions& opts) {
  if (!engine_) {
    return absl::FailedPreconditionError("routing context missing engine");
  }
  return engine_->register_tensor_ex(tensor_key, tensor_addr, tensor_bytes, dev_type, dev_id, opts);
}

absl::Status RoutingContext::unregister_tensor(const std::string& tensor_key) {
  if (!engine_) {
    return absl::FailedPreconditionError("routing context missing engine");
  }
  return engine_->unregister_tensor(tensor_key);
}

uint64_t RoutingContext::topology_generation() const {
  absl::MutexLock lock(&mu_);
  return topology_generation_;
}

absl::StatusOr<std::shared_ptr<RouteChannel>> RoutingContext::build_direct_channel(
    const std::string& src_endpoint_id,
    const std::string& dst_endpoint_id) {
  absl::MutexLock lock(&mu_);
  return build_direct_channel_locked(src_endpoint_id, dst_endpoint_id);
}

absl::StatusOr<RoutingContext::ChannelBuildResult> RoutingContext::build_direct_channel_with_generation(
    const std::string& src_endpoint_id,
    const std::string& dst_endpoint_id) {
  absl::MutexLock lock(&mu_);
  auto channel_or = build_direct_channel_locked(src_endpoint_id, dst_endpoint_id);
  if (!channel_or.ok()) {
    return channel_or.status();
  }
  return ChannelBuildResult{std::move(channel_or).value(), topology_generation_};
}

absl::StatusOr<std::shared_ptr<RouteChannel>> RoutingContext::build_direct_channel_locked(
    const std::string& src_endpoint_id,
    const std::string& dst_endpoint_id) {
  if (!topology_) {
    return absl::FailedPreconditionError("topology is not set");
  }

  auto src_binding_it = bindings_.find(src_endpoint_id);
  if (src_binding_it == bindings_.end()) {
    return absl::NotFoundError(std::format("missing endpoint binding for source: {}", src_endpoint_id));
  }
  auto dst_binding_it = bindings_.find(dst_endpoint_id);
  if (dst_binding_it == bindings_.end()) {
    return absl::NotFoundError(std::format("missing endpoint binding for destination: {}", dst_endpoint_id));
  }

  const topology::Endpoint* src_endpoint = topology_->find_endpoint(src_endpoint_id);
  const topology::Endpoint* dst_endpoint = topology_->find_endpoint(dst_endpoint_id);
  const EndpointBinding& local_binding = src_binding_it->second;
  EndpointBinding remote_binding = dst_binding_it->second;

  const topology::Endpoint* protocol_src_endpoint = src_endpoint;
  const topology::Endpoint* protocol_dst_endpoint = dst_endpoint;
  const topology::Link* link = nullptr;
  bool used_rail_fallback = false;
  if (src_endpoint && dst_endpoint) {
    link = find_link_locked(src_endpoint_id, dst_endpoint_id);
  }

  if (!link) {
    auto rail_path_or =
        resolve_rail_matched_path_locked(src_endpoint_id, dst_endpoint_id, local_binding, remote_binding);
    if (!rail_path_or.ok()) {
      if (!src_endpoint) {
        return absl::NotFoundError(
            std::format("source endpoint not found in topology and no rail-matched fallback: {}", src_endpoint_id));
      }
      if (!dst_endpoint) {
        return absl::NotFoundError(
            std::format(
                "destination endpoint not found in topology and no rail-matched fallback: {}", dst_endpoint_id));
      }
      return absl::NotFoundError(
          std::format(
              "no direct link or rail-matched fallback between endpoints: {} -> {} ({})",
              src_endpoint_id,
              dst_endpoint_id,
              rail_path_or.status().message()));
    }

    RailMatchedPath rail_path = std::move(rail_path_or).value();
    used_rail_fallback = true;
    if (rail_path.src_topology_endpoint != nullptr) {
      protocol_src_endpoint = rail_path.src_topology_endpoint;
    }
    if (rail_path.dst_topology_endpoint != nullptr) {
      protocol_dst_endpoint = rail_path.dst_topology_endpoint;
    }
    link = rail_path.link;
    remote_binding = std::move(rail_path.remote_binding);
  }

  ConnectionProtocol protocol = ConnectionProtocol::kAuto;
  if (protocol_src_endpoint != nullptr && protocol_dst_endpoint != nullptr) {
    protocol = select_protocol_locked(*protocol_src_endpoint, *protocol_dst_endpoint, local_binding, remote_binding);
  }

  auto connection =
      get_or_create_connection_locked(src_endpoint_id, dst_endpoint_id, link, protocol, local_binding, remote_binding);
  const bool same_node = !local_binding.node_id.empty() && local_binding.node_id == remote_binding.node_id;
  const std::string_view path_kind = used_rail_fallback
      ? "cross_node_rail_fallback"
      : (same_node ? (src_endpoint_id == dst_endpoint_id ? "same_node_loopback" : "local_fanout")
                   : "cross_node_direct");
  LOG(INFO) << "[routing] route resolved: src=" << src_endpoint_id << " dst=" << dst_endpoint_id
            << " path=" << path_kind << " protocol=" << to_string(protocol)
            << " adapter=" << protocol_adapter_name(protocol) << " link=" << (link == nullptr ? "none" : link->id)
            << " link_type=" << to_string(connection->type()) << " local={" << describe_binding(local_binding) << "}"
            << " remote={" << describe_binding(remote_binding) << "}";
  std::vector<std::shared_ptr<Connection>> hops;
  hops.push_back(connection);

  std::string channel_id = std::format("{}->{}:{}", src_endpoint_id, dst_endpoint_id, to_string(protocol));
  return std::make_shared<RouteChannel>(std::move(channel_id), src_endpoint_id, dst_endpoint_id, std::move(hops));
}

std::shared_ptr<Connection> RoutingContext::get_or_create_connection_locked(
    const std::string& src_endpoint_id,
    const std::string& dst_endpoint_id,
    const topology::Link* link,
    ConnectionProtocol protocol,
    const EndpointBinding& local_binding,
    const EndpointBinding& remote_binding) {
  ConnectionKey key{src_endpoint_id, dst_endpoint_id, protocol};
  auto it = connections_.find(key);
  if (it != connections_.end()) {
    VLOG(1) << "[routing] reusing cached connection: src=" << src_endpoint_id << " dst=" << dst_endpoint_id
            << " protocol=" << to_string(protocol);
    return it->second;
  }
  auto link_state = get_or_create_link_state_locked(link);
  const ConnectionType type = connection_type_from_link(link);
  std::shared_ptr<ConnectionAdapter> adapter = engine_adapter_;
  if (protocol == ConnectionProtocol::kNvlink) {
    adapter = nvlink_adapter_;
  } else if (protocol == ConnectionProtocol::kPcie) {
    adapter = pcie_adapter_;
  }
  auto connection = std::make_shared<Connection>(
      key,
      type,
      topology_,
      link,
      local_binding,
      remote_binding,
      std::move(adapter),
      std::move(link_state),
      async_runtime_);
  connections_.emplace(std::move(key), connection);
  LOG(INFO) << "[routing] connection created: src=" << src_endpoint_id << " dst=" << dst_endpoint_id
            << " protocol=" << to_string(protocol) << " adapter=" << protocol_adapter_name(protocol)
            << " type=" << to_string(type) << " link=" << (link == nullptr ? "none" : link->id);
  return connection;
}

const topology::Link* RoutingContext::find_link_locked(
    const std::string& src_endpoint_id,
    const std::string& dst_endpoint_id) const {
  for (const auto& [link_id, link] : topology_->links()) {
    if (link.src_endpoint_id == src_endpoint_id && link.dst_endpoint_id == dst_endpoint_id) {
      return &link;
    }
  }
  return nullptr;
}

const topology::Link* RoutingContext::find_any_link_for_endpoint_locked(const std::string& endpoint_id) const {
  const topology::Link* best = nullptr;
  for (const auto& [link_id, link] : topology_->links()) {
    if (link.src_endpoint_id != endpoint_id && link.dst_endpoint_id != endpoint_id) {
      continue;
    }
    if (best == nullptr || link.id < best->id) {
      best = &link;
    }
  }
  return best;
}

int RoutingContext::infer_nic_rail_id_locked(const std::string& nic_endpoint_id) const {
  int inferred_rail_id = -1;
  for (const auto& [link_id, link] : topology_->links()) {
    if (link.type != topology::LinkType::kSwitch) {
      continue;
    }

    std::string_view switch_endpoint_id;
    if (link.src_endpoint_id == nic_endpoint_id) {
      switch_endpoint_id = link.dst_endpoint_id;
    } else if (link.dst_endpoint_id == nic_endpoint_id) {
      switch_endpoint_id = link.src_endpoint_id;
    } else {
      continue;
    }

    auto rail_id = parse_rail_id_from_switch_endpoint_id(switch_endpoint_id);
    if (!rail_id.has_value()) {
      continue;
    }
    if (inferred_rail_id < 0 || *rail_id < inferred_rail_id) {
      inferred_rail_id = *rail_id;
    }
  }
  return inferred_rail_id;
}

absl::StatusOr<RoutingContext::LocalNicSelection> RoutingContext::select_local_nic_for_source_locked(
    const std::string& src_endpoint_id,
    const EndpointBinding& src_binding) const {
  const topology::Endpoint* direct_endpoint = topology_->find_endpoint(src_endpoint_id);
  if (direct_endpoint != nullptr && direct_endpoint->kind == topology::EndpointKind::kClient &&
      direct_endpoint->type == topology::EndpointType::kNic) {
    return LocalNicSelection{
        .endpoint = direct_endpoint,
        .rail_id = infer_nic_rail_id_locked(direct_endpoint->id),
    };
  }

  std::vector<const topology::Endpoint*> nic_candidates;
  nic_candidates.reserve(topology_->endpoints().size());
  for (const auto& [endpoint_id, endpoint] : topology_->endpoints()) {
    if (endpoint.kind != topology::EndpointKind::kClient || endpoint.type != topology::EndpointType::kNic) {
      continue;
    }
    nic_candidates.push_back(&endpoint);
  }
  if (nic_candidates.empty()) {
    return absl::NotFoundError("topology has no NIC endpoints for rail matching");
  }

  std::sort(
      nic_candidates.begin(), nic_candidates.end(), [](const topology::Endpoint* lhs, const topology::Endpoint* rhs) {
        return lhs->id < rhs->id;
      });

  const DevicePoolHint source_hint = infer_device_pool_hint(src_endpoint_id, src_binding);
  const bool source_is_gpu = !source_hint.gpu_pool_id.empty();
  const bool source_is_cpu = !source_hint.cpu_pool_id.empty();
  const int preferred_rail_id = src_binding.rail_id;

  const topology::Endpoint* best_endpoint = nullptr;
  int best_endpoint_rail_id = -1;
  int best_score = std::numeric_limits<int>::min();
  for (const topology::Endpoint* candidate : nic_candidates) {
    int score = 0;
    const int candidate_rail_id = infer_nic_rail_id_locked(candidate->id);

    if (preferred_rail_id >= 0) {
      if (candidate_rail_id == preferred_rail_id) {
        score += 200;
      } else if (candidate_rail_id >= 0) {
        score -= 80;
      }
    }

    if (source_is_gpu) {
      if (endpoint_has_pool(*candidate, source_hint.gpu_pool_id)) {
        score += 120;
      } else {
        score -= 40;
      }
    }

    if (source_is_cpu) {
      if (endpoint_has_pool(*candidate, source_hint.cpu_pool_id)) {
        score += 100;
      } else {
        score -= 35;
      }
    }

    if (candidate_rail_id >= 0) {
      score += 5;
    }

    if (score > best_score) {
      best_score = score;
      best_endpoint = candidate;
      best_endpoint_rail_id = candidate_rail_id;
      continue;
    }
    if (score == best_score && best_endpoint != nullptr && candidate->id < best_endpoint->id) {
      best_endpoint = candidate;
      best_endpoint_rail_id = candidate_rail_id;
    }
  }

  if (best_endpoint == nullptr) {
    return absl::NotFoundError(std::format("failed to resolve local NIC endpoint for source: {}", src_endpoint_id));
  }
  return LocalNicSelection{
      .endpoint = best_endpoint,
      .rail_id = best_endpoint_rail_id,
  };
}

absl::StatusOr<EndpointBinding> RoutingContext::select_remote_binding_for_rail_locked(
    const EndpointBinding& dst_binding,
    int preferred_rail_id) const {
  if (dst_binding.node_id.empty()) {
    return absl::InvalidArgumentError("destination binding node_id is empty");
  }
  const DevicePoolHint dst_hint = infer_device_pool_hint(dst_binding.endpoint_id, dst_binding);

  struct RemoteCandidate {
    const EndpointBinding* binding = nullptr;
    int score = std::numeric_limits<int>::min();
  };

  std::vector<RemoteCandidate> candidates;
  candidates.reserve(bindings_.size());
  for (const auto& [endpoint_id, binding] : bindings_) {
    if (binding.node_id != dst_binding.node_id) {
      continue;
    }
    if (!binding.has_network_address()) {
      continue;
    }

    const topology::Endpoint* candidate_endpoint = topology_->find_endpoint(binding.endpoint_id);
    int candidate_rail_id = binding.rail_id;
    if (candidate_rail_id < 0 && candidate_endpoint != nullptr &&
        candidate_endpoint->kind == topology::EndpointKind::kClient &&
        candidate_endpoint->type == topology::EndpointType::kNic) {
      candidate_rail_id = infer_nic_rail_id_locked(candidate_endpoint->id);
    }

    int score = 0;
    if (preferred_rail_id >= 0) {
      if (candidate_rail_id == preferred_rail_id) {
        score += 200;
      } else if (candidate_rail_id >= 0) {
        score -= 80;
      }
    }
    if (dst_binding.rail_id >= 0 && candidate_rail_id == dst_binding.rail_id) {
      score += 80;
    }
    if (candidate_endpoint != nullptr) {
      if (candidate_endpoint->kind == topology::EndpointKind::kClient &&
          candidate_endpoint->type == topology::EndpointType::kNic) {
        score += 30;
      } else {
        score -= 30;
      }
      if (!dst_hint.gpu_pool_id.empty()) {
        if (endpoint_has_pool(*candidate_endpoint, dst_hint.gpu_pool_id)) {
          score += 120;
        } else {
          score -= 40;
        }
      }
      if (!dst_hint.cpu_pool_id.empty()) {
        if (endpoint_has_pool(*candidate_endpoint, dst_hint.cpu_pool_id)) {
          score += 100;
        } else {
          score -= 35;
        }
      }
    }
    if (binding.endpoint_id == dst_binding.endpoint_id) {
      score += 40;
    }
    if (is_nic_binding_id(binding.endpoint_id)) {
      score += 20;
    }
    if (candidate_rail_id >= 0) {
      score += 5;
    }

    candidates.push_back(
        RemoteCandidate{
            .binding = &binding,
            .score = score,
        });
  }

  if (candidates.empty()) {
    if (dst_binding.has_network_address()) {
      return dst_binding;
    }
    return absl::NotFoundError(std::format("no remote binding with network address for node: {}", dst_binding.node_id));
  }

  std::sort(candidates.begin(), candidates.end(), [](const RemoteCandidate& lhs, const RemoteCandidate& rhs) {
    if (lhs.score != rhs.score) {
      return lhs.score > rhs.score;
    }
    return lhs.binding->endpoint_id < rhs.binding->endpoint_id;
  });
  return *candidates.front().binding;
}

absl::StatusOr<RoutingContext::RailMatchedPath> RoutingContext::resolve_rail_matched_path_locked(
    const std::string& src_endpoint_id,
    const std::string& dst_endpoint_id,
    const EndpointBinding& src_binding,
    const EndpointBinding& dst_binding) const {
  if (src_binding.node_id.empty() || dst_binding.node_id.empty()) {
    return absl::FailedPreconditionError("rail-matched routing requires node_id in both endpoint bindings");
  }
  if (src_binding.node_id == dst_binding.node_id) {
    return absl::FailedPreconditionError("rail-matched routing only applies to cross-node communication");
  }

  auto local_nic_or = select_local_nic_for_source_locked(src_endpoint_id, src_binding);
  if (!local_nic_or.ok()) {
    return local_nic_or.status();
  }
  const LocalNicSelection local_nic = std::move(local_nic_or).value();
  if (local_nic.endpoint == nullptr) {
    return absl::NotFoundError(std::format("failed to resolve local NIC endpoint for source: {}", src_endpoint_id));
  }

  int preferred_rail_id = local_nic.rail_id;
  if (preferred_rail_id < 0 && src_binding.rail_id >= 0) {
    preferred_rail_id = src_binding.rail_id;
  }
  if (preferred_rail_id < 0 && dst_binding.rail_id >= 0) {
    preferred_rail_id = dst_binding.rail_id;
  }

  auto remote_binding_or = select_remote_binding_for_rail_locked(dst_binding, preferred_rail_id);
  if (!remote_binding_or.ok()) {
    return remote_binding_or.status();
  }
  EndpointBinding remote_binding = std::move(remote_binding_or).value();
  const topology::Endpoint* remote_topology_endpoint = topology_->find_endpoint(remote_binding.endpoint_id);

  const topology::Link* selected_link = nullptr;
  if (remote_topology_endpoint != nullptr) {
    selected_link = find_link_locked(local_nic.endpoint->id, remote_topology_endpoint->id);
  }
  if (selected_link == nullptr) {
    selected_link = find_any_link_for_endpoint_locked(local_nic.endpoint->id);
  }

  LOG(INFO) << "[routing] rail fallback selected: src=" << src_endpoint_id << " dst=" << dst_endpoint_id
            << " local_nic=" << local_nic.endpoint->id << " local_rail=" << local_nic.rail_id
            << " preferred_rail=" << preferred_rail_id << " remote_endpoint=" << remote_binding.endpoint_id
            << " remote_rail=" << remote_binding.rail_id
            << " link=" << (selected_link == nullptr ? "none" : selected_link->id);
  return RailMatchedPath{
      .src_topology_endpoint = local_nic.endpoint,
      .dst_topology_endpoint = remote_topology_endpoint,
      .link = selected_link,
      .remote_binding = std::move(remote_binding),
  };
}

ConnectionProtocol RoutingContext::select_protocol_locked(
    const topology::Endpoint& src_endpoint,
    const topology::Endpoint& dst_endpoint,
    const EndpointBinding& local_binding,
    const EndpointBinding& remote_binding) const {
  if (local_binding.node_id.empty() || remote_binding.node_id.empty()) {
    return ConnectionProtocol::kAuto;
  }
  if (local_binding.node_id != remote_binding.node_id) {
    return ConnectionProtocol::kAuto;
  }

  if (src_endpoint.type == topology::EndpointType::kNvlink && dst_endpoint.type == topology::EndpointType::kNvlink &&
      options_.prefer_nvlink && nvlink_adapter_ && nvlink_adapter_->is_available()) {
    return ConnectionProtocol::kNvlink;
  }

  if (src_endpoint.type == topology::EndpointType::kPcie && dst_endpoint.type == topology::EndpointType::kPcie &&
      options_.prefer_pcie && pcie_adapter_ && pcie_adapter_->is_available()) {
    return ConnectionProtocol::kPcie;
  }

  return ConnectionProtocol::kAuto;
}

std::shared_ptr<LinkState> RoutingContext::get_or_create_link_state_locked(const topology::Link* link) {
  if (!link) {
    return nullptr;
  }
  auto it = link_states_.find(link->id);
  if (it != link_states_.end()) {
    return it->second;
  }
  auto state = std::make_shared<LinkState>(link->id);
  link_states_.emplace(link->id, state);
  return state;
}

std::string RoutingContext::make_peer_key(const std::string& src_endpoint_id, const std::string& dst_endpoint_id) {
  return std::format("{}->{}", src_endpoint_id, dst_endpoint_id);
}

} // namespace tensorcast::communicator::routing
