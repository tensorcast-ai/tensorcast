// Copyright (c) 2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ROUTING_ROUTING_CONTEXT_H_
#define CORE_COMMUNICATOR_ROUTING_ROUTING_CONTEXT_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/common/async_runtime.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/routing/adapter.h"
#include "core/communicator/routing/connection.h"
#include "core/communicator/routing/route_channel.h"
#include "core/communicator/routing/types.h"
#include "core/communicator/topology/topology.h"

namespace tensorcast::communicator::routing {

class RoutingContext : public std::enable_shared_from_this<RoutingContext> {
 public:
  struct Options {
    bool prefer_nvlink = true;
    bool prefer_pcie = true;
  };

  class Communicator {
   public:
    transport::future_read_result_t read_tensor(const ReadRequest& request);
    absl::StatusOr<std::shared_ptr<RouteChannel>> primary_channel();

    const std::string& src_endpoint_id() const {
      return src_endpoint_id_;
    }

    const std::string& dst_endpoint_id() const {
      return dst_endpoint_id_;
    }

   private:
    friend class RoutingContext;

    Communicator(std::string src_endpoint_id, std::string dst_endpoint_id, std::shared_ptr<RoutingContext> context);

    std::string src_endpoint_id_;
    std::string dst_endpoint_id_;
    std::shared_ptr<RoutingContext> context_;

    mutable absl::Mutex mu_;
    std::shared_ptr<RouteChannel> primary_channel_ ABSL_GUARDED_BY(mu_);
    uint64_t primary_channel_generation_ ABSL_GUARDED_BY(mu_) = 0;
  };

  RoutingContext(
      Options options,
      std::shared_ptr<engine::Communicator> engine,
      std::shared_ptr<ConnectionAdapter> nvlink_adapter = nullptr,
      std::shared_ptr<ConnectionAdapter> pcie_adapter = nullptr,
      std::shared_ptr<common::AsyncRuntime> async_runtime = nullptr);

  absl::Status set_topology(topology::Topology topology);

  absl::Status set_endpoint_bindings(std::vector<EndpointBinding> bindings);
  absl::Status update_endpoint_binding(EndpointBinding binding);

  absl::StatusOr<std::shared_ptr<Communicator>> get_communicator(
      const std::string& src_endpoint_id,
      const std::string& dst_endpoint_id);

  absl::Status register_tensor_ex(
      const std::string& tensor_key,
      uint64_t tensor_addr,
      uint64_t tensor_bytes,
      int dev_type,
      int dev_id,
      const engine::Communicator::RegisterTensorOptions& opts);

  absl::Status unregister_tensor(const std::string& tensor_key);

  const std::shared_ptr<engine::Communicator>& engine() const {
    return engine_;
  }

  uint64_t topology_generation() const;

 private:
  struct ChannelBuildResult {
    std::shared_ptr<RouteChannel> channel;
    uint64_t generation = 0;
  };

  struct RailMatchedPath {
    const topology::Endpoint* src_topology_endpoint = nullptr;
    const topology::Endpoint* dst_topology_endpoint = nullptr;
    const topology::Link* link = nullptr;
    EndpointBinding remote_binding;
  };

  struct LocalNicSelection {
    const topology::Endpoint* endpoint = nullptr;
    int rail_id = -1;
  };

  absl::StatusOr<std::shared_ptr<RouteChannel>> build_direct_channel(
      const std::string& src_endpoint_id,
      const std::string& dst_endpoint_id);
  absl::StatusOr<ChannelBuildResult> build_direct_channel_with_generation(
      const std::string& src_endpoint_id,
      const std::string& dst_endpoint_id);

  absl::StatusOr<std::shared_ptr<RouteChannel>> build_direct_channel_locked(
      const std::string& src_endpoint_id,
      const std::string& dst_endpoint_id) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::shared_ptr<Connection> get_or_create_connection_locked(
      const std::string& src_endpoint_id,
      const std::string& dst_endpoint_id,
      const topology::Link* link,
      ConnectionProtocol protocol,
      const EndpointBinding& local_binding,
      const EndpointBinding& remote_binding) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const topology::Link* find_link_locked(const std::string& src_endpoint_id, const std::string& dst_endpoint_id) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const topology::Link* find_any_link_for_endpoint_locked(const std::string& endpoint_id) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::StatusOr<LocalNicSelection> select_local_nic_for_source_locked(
      const std::string& src_endpoint_id,
      const EndpointBinding& src_binding) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  int infer_nic_rail_id_locked(const std::string& nic_endpoint_id) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::StatusOr<EndpointBinding> select_remote_binding_for_rail_locked(
      const EndpointBinding& dst_binding,
      int preferred_rail_id) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::StatusOr<RailMatchedPath> resolve_rail_matched_path_locked(
      const std::string& src_endpoint_id,
      const std::string& dst_endpoint_id,
      const EndpointBinding& src_binding,
      const EndpointBinding& dst_binding) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  ConnectionProtocol select_protocol_locked(
      const topology::Endpoint& src_endpoint,
      const topology::Endpoint& dst_endpoint,
      const EndpointBinding& local_binding,
      const EndpointBinding& remote_binding) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::shared_ptr<LinkState> get_or_create_link_state_locked(const topology::Link* link)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  static std::string make_peer_key(const std::string& src_endpoint_id, const std::string& dst_endpoint_id);

  Options options_;
  std::shared_ptr<engine::Communicator> engine_;
  std::shared_ptr<ConnectionAdapter> engine_adapter_;
  std::shared_ptr<ConnectionAdapter> nvlink_adapter_;
  std::shared_ptr<ConnectionAdapter> pcie_adapter_;
  std::shared_ptr<common::AsyncRuntime> async_runtime_;

  mutable absl::Mutex mu_;
  std::shared_ptr<topology::Topology> topology_ ABSL_GUARDED_BY(mu_);
  uint64_t topology_generation_ ABSL_GUARDED_BY(mu_) = 0;
  absl::flat_hash_map<std::string, EndpointBinding> bindings_ ABSL_GUARDED_BY(mu_);
  bool bindings_initialized_ ABSL_GUARDED_BY(mu_) = false;
  absl::flat_hash_map<ConnectionKey, std::shared_ptr<Connection>> connections_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::shared_ptr<LinkState>> link_states_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::weak_ptr<Communicator>> communicators_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::communicator::routing

#endif // CORE_COMMUNICATOR_ROUTING_ROUTING_CONTEXT_H_
