// Copyright (c) 2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ROUTING_ROUTE_CHANNEL_H_
#define CORE_COMMUNICATOR_ROUTING_ROUTE_CHANNEL_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "core/communicator/routing/connection.h"
#include "core/communicator/routing/types.h"
#include "core/communicator/transport/request.h"

namespace tensorcast::communicator::routing {

class RouteChannel {
 public:
  RouteChannel(
      std::string id,
      std::string src_endpoint_id,
      std::string dst_endpoint_id,
      std::vector<std::shared_ptr<Connection>> hops);

  const std::string& id() const {
    return id_;
  }

  const std::string& src_endpoint_id() const {
    return src_endpoint_id_;
  }

  const std::string& dst_endpoint_id() const {
    return dst_endpoint_id_;
  }

  const std::vector<std::shared_ptr<Connection>>& hops() const {
    return hops_;
  }

  size_t hop_count() const {
    return hops_.size();
  }

  transport::future_read_result_t read_tensor(const ReadRequest& request);
  transport::future_read_result_t read_plan(const ReadPlan& plan);
  HealthState health() const;

 private:
  std::string id_;
  std::string src_endpoint_id_;
  std::string dst_endpoint_id_;
  std::vector<std::shared_ptr<Connection>> hops_;
};

} // namespace tensorcast::communicator::routing

#endif // CORE_COMMUNICATOR_ROUTING_ROUTE_CHANNEL_H_
