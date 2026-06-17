// Copyright (c) 2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ROUTING_READ_HELPERS_H_
#define CORE_COMMUNICATOR_ROUTING_READ_HELPERS_H_

#include <future>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "core/communicator/routing/types.h"
#include "core/communicator/transport/request.h"

namespace tensorcast::communicator::routing {

inline transport::future_read_result_t make_failed_read_future(absl::Status status, std::string tensor_key = {}) {
  std::promise<transport::read_result_t> promise;
  auto future = promise.get_future();
  transport::read_result_t result;
  result.status = std::move(status);
  result.tensor_key = std::move(tensor_key);
  promise.set_value(std::move(result));
  return future;
}

inline std::string read_plan_tensor_key(const ReadPlan& plan) {
  if (plan.source_slices.empty()) {
    return {};
  }
  return plan.source_slices.front().tensor_key;
}

} // namespace tensorcast::communicator::routing

#endif // CORE_COMMUNICATOR_ROUTING_READ_HELPERS_H_
