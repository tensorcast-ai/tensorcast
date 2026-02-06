// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/routing/routing_context.h"

#include <format>
#include <utility>

#include "absl/status/status.h"
#include "core/communicator/routing/read_helpers.h"

namespace tensorcast::communicator::routing {
namespace {

ConnectionType connection_type_from_link(const topology::Link& link) {
  switch (link.type) {
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

absl::StatusOr<std::shared_ptr<RouteChannel>> RoutingContext::Communicator::primary_channel() {
  absl::MutexLock lock(&mu_);
  const uint64_t generation = context_->topology_generation();
  if (primary_channel_ && primary_channel_generation_ == generation) {
    return primary_channel_;
  }
  primary_channel_.reset();
  auto channel_or = context_->build_direct_channel_with_generation(
      src_endpoint_id_, dst_endpoint_id_);
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
    std::shared_ptr<common::AsyncRuntime> async_runtime)
    : options_(options),
      engine_(std::move(engine)),
      engine_adapter_(std::make_shared<EngineAdapter>(engine_)),
      nvlink_adapter_(std::move(nvlink_adapter)),
      async_runtime_(std::move(async_runtime)) {
  if (!async_runtime_) {
    async_runtime_ = std::make_shared<common::AsyncRuntime>();
  }
  if (!nvlink_adapter_) {
    nvlink_adapter_ = std::make_shared<NvlinkAdapter>();
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
  return absl::FailedPreconditionError(
      "endpoint bindings are immutable; update_endpoint_binding is not supported");
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
  auto communicator = std::shared_ptr<Communicator>(
      new Communicator(src_endpoint_id, dst_endpoint_id, shared_from_this()));
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

absl::StatusOr<RoutingContext::ChannelBuildResult>
RoutingContext::build_direct_channel_with_generation(
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
  const topology::Endpoint* src_endpoint = topology_->find_endpoint(src_endpoint_id);
  if (!src_endpoint) {
    return absl::NotFoundError(
        std::format("source endpoint not found: {}", src_endpoint_id));
  }
  const topology::Endpoint* dst_endpoint = topology_->find_endpoint(dst_endpoint_id);
  if (!dst_endpoint) {
    return absl::NotFoundError(
        std::format("destination endpoint not found: {}", dst_endpoint_id));
  }

  auto src_binding_it = bindings_.find(src_endpoint_id);
  if (src_binding_it == bindings_.end()) {
    return absl::NotFoundError(
        std::format("missing endpoint binding for source: {}", src_endpoint_id));
  }
  auto dst_binding_it = bindings_.find(dst_endpoint_id);
  if (dst_binding_it == bindings_.end()) {
    return absl::NotFoundError(
        std::format("missing endpoint binding for destination: {}", dst_endpoint_id));
  }

  const topology::Link* link = find_link_locked(src_endpoint_id, dst_endpoint_id);
  if (!link) {
    return absl::NotFoundError(
        std::format("no direct link between endpoints: {} -> {}", src_endpoint_id, dst_endpoint_id));
  }

  const ConnectionProtocol protocol = select_protocol_locked(
      *src_endpoint, *dst_endpoint, src_binding_it->second, dst_binding_it->second);
  auto connection = get_or_create_connection_locked(
      src_endpoint_id,
      dst_endpoint_id,
      link,
      protocol,
      src_binding_it->second,
      dst_binding_it->second);
  std::vector<std::shared_ptr<Connection>> hops;
  hops.push_back(connection);

  std::string channel_id = std::format(
      "{}->{}:{}",
      src_endpoint_id,
      dst_endpoint_id,
      to_string(protocol));
  return std::make_shared<RouteChannel>(
      std::move(channel_id),
      src_endpoint_id,
      dst_endpoint_id,
      std::move(hops));
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
    return it->second;
  }
  auto link_state = get_or_create_link_state_locked(link);
  const ConnectionType type = connection_type_from_link(*link);
  std::shared_ptr<ConnectionAdapter> adapter =
      protocol == ConnectionProtocol::kNvlink ? nvlink_adapter_ : engine_adapter_;
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

ConnectionProtocol RoutingContext::select_protocol_locked(
    const topology::Endpoint& src_endpoint,
    const topology::Endpoint& dst_endpoint,
    const EndpointBinding& local_binding,
    const EndpointBinding& remote_binding) const {
  if (!options_.prefer_nvlink) {
    return ConnectionProtocol::kAuto;
  }
  if (!nvlink_adapter_ || !nvlink_adapter_->is_available()) {
    return ConnectionProtocol::kAuto;
  }
  if (local_binding.node_id.empty() || remote_binding.node_id.empty()) {
    return ConnectionProtocol::kAuto;
  }
  if (local_binding.node_id != remote_binding.node_id) {
    return ConnectionProtocol::kAuto;
  }
  if (src_endpoint.type != topology::EndpointType::kNvlink ||
      dst_endpoint.type != topology::EndpointType::kNvlink) {
    return ConnectionProtocol::kAuto;
  }
  return ConnectionProtocol::kNvlink;
}

std::shared_ptr<LinkState> RoutingContext::get_or_create_link_state_locked(
    const topology::Link* link) {
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

std::string RoutingContext::make_peer_key(
    const std::string& src_endpoint_id,
    const std::string& dst_endpoint_id) {
  return std::format("{}->{}", src_endpoint_id, dst_endpoint_id);
}

} // namespace tensorcast::communicator::routing
