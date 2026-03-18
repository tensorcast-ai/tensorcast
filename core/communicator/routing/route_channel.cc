// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/routing/route_channel.h"

#include <format>

#include "core/communicator/routing/read_helpers.h"

namespace tensorcast::communicator::routing {

RouteChannel::RouteChannel(
    std::string id,
    std::string src_endpoint_id,
    std::string dst_endpoint_id,
    std::vector<std::shared_ptr<Connection>> hops)
    : id_(std::move(id)),
      src_endpoint_id_(std::move(src_endpoint_id)),
      dst_endpoint_id_(std::move(dst_endpoint_id)),
      hops_(std::move(hops)) {}

transport::future_read_result_t RouteChannel::read_tensor(const ReadRequest& request) {
  if (hops_.empty()) {
    return make_failed_read_future(
        absl::FailedPreconditionError("route channel has no hops"),
        request.tensor_key);
  }
  if (hops_.size() != 1) {
    return make_failed_read_future(absl::UnimplementedError(
        std::format("multi-hop read not implemented (hops={})", hops_.size())),
        request.tensor_key);
  }
  return hops_.front()->read_tensor(request);
}

HealthState RouteChannel::health() const {
  if (hops_.empty()) {
    return HealthState::kUnhealthy;
  }
  bool has_unknown = false;
  for (const auto& hop : hops_) {
    if (!hop) {
      return HealthState::kUnhealthy;
    }
    const HealthState state = hop->health();
    if (state == HealthState::kUnhealthy) {
      return HealthState::kUnhealthy;
    }
    if (state == HealthState::kDegraded) {
      return HealthState::kDegraded;
    }
    if (state == HealthState::kUnknown) {
      has_unknown = true;
    }
  }
  if (has_unknown) {
    return HealthState::kUnknown;
  }
  return HealthState::kHealthy;
}

} // namespace tensorcast::communicator::routing
