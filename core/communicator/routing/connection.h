// Copyright (c) 2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ROUTING_CONNECTION_H_
#define CORE_COMMUNICATOR_ROUTING_CONNECTION_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/communicator/routing/adapter.h"
#include "core/communicator/routing/types.h"
#include "core/communicator/topology/topology.h"
#include "core/communicator/transport/request.h"

namespace tensorcast::communicator::routing {

class LinkState {
 public:
  explicit LinkState(std::string link_id);

  void record_success(absl::Duration latency);
  void record_failure(const absl::Status& status);

  LinkStats snapshot() const;
  HealthState health() const;
  const std::string& link_id() const {
    return link_id_;
  }

 private:
  std::string link_id_;
  mutable absl::Mutex mu_;
  LinkStats stats_ ABSL_GUARDED_BY(mu_);
};

class Connection : public std::enable_shared_from_this<Connection> {
 public:
  Connection(ConnectionKey key,
             ConnectionType type,
             std::shared_ptr<const topology::Topology> topology,
             const topology::Link* link,
             EndpointBinding local_binding,
             EndpointBinding remote_binding,
             std::shared_ptr<ConnectionAdapter> adapter,
             std::shared_ptr<LinkState> link_state,
             std::shared_ptr<common::AsyncRuntime> async_runtime);

  const ConnectionKey& key() const {
    return key_;
  }

  ConnectionType type() const {
    return type_;
  }

  ConnectionProtocol protocol() const {
    return key_.protocol;
  }

  const topology::Link* link() const {
    return link_;
  }

  const EndpointBinding& local_binding() const {
    return local_binding_;
  }

  const EndpointBinding& remote_binding() const {
    return remote_binding_;
  }

  transport::future_read_result_t read_tensor(const ReadRequest& request);
  absl::Status close();

  ConnectionStats snapshot() const;
  HealthState health() const;

 private:
  void record_success(absl::Duration latency);
  void record_failure(const absl::Status& status);

  ConnectionKey key_;
  ConnectionType type_;
  // Keeps link_ valid when the routing context swaps topologies.
  std::shared_ptr<const topology::Topology> topology_ref_;
  const topology::Link* link_;
  EndpointBinding local_binding_;
  EndpointBinding remote_binding_;
  std::shared_ptr<ConnectionAdapter> adapter_;
  std::shared_ptr<LinkState> link_state_;
  std::shared_ptr<common::AsyncRuntime> async_runtime_;

  mutable absl::Mutex mu_;
  ConnectionStats stats_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::communicator::routing

#endif // CORE_COMMUNICATOR_ROUTING_CONNECTION_H_
