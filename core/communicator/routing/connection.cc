// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/routing/connection.h"

#include <exception>
#include <future>
#include <utility>

#include "absl/status/status.h"
#include "core/communicator/routing/read_helpers.h"

namespace tensorcast::communicator::routing {

LinkState::LinkState(std::string link_id)
    : link_id_(std::move(link_id)) {}

void LinkState::record_success(absl::Duration latency) {
  absl::MutexLock lock(&mu_);
  stats_.success_count += 1;
  stats_.last_success = absl::Now();
  stats_.last_latency = latency;
}

void LinkState::record_failure(const absl::Status& status) {
  absl::MutexLock lock(&mu_);
  stats_.failure_count += 1;
  stats_.last_failure = absl::Now();
  stats_.last_error = status.ToString();
}

LinkStats LinkState::snapshot() const {
  absl::MutexLock lock(&mu_);
  return stats_;
}

HealthState LinkState::health() const {
  absl::MutexLock lock(&mu_);
  return derive_health(stats_);
}

Connection::Connection(ConnectionKey key,
                       ConnectionType type,
                       std::shared_ptr<const topology::Topology> topology,
                       const topology::Link* link,
                       EndpointBinding local_binding,
                       EndpointBinding remote_binding,
                       std::shared_ptr<ConnectionAdapter> adapter,
                       std::shared_ptr<LinkState> link_state,
                       std::shared_ptr<common::AsyncRuntime> async_runtime)
    : key_(std::move(key)),
      type_(type),
      topology_ref_(std::move(topology)),
      link_(link),
      local_binding_(std::move(local_binding)),
      remote_binding_(std::move(remote_binding)),
      adapter_(std::move(adapter)),
      link_state_(std::move(link_state)),
      async_runtime_(std::move(async_runtime)) {}

transport::future_read_result_t Connection::read_tensor(const ReadRequest& request) {
  if (!adapter_ || !adapter_->is_available()) {
    const absl::Status status = absl::FailedPreconditionError("connection adapter unavailable");
    record_failure(status);
    return make_failed_read_future(status, request.tensor_key);
  }
  if (!async_runtime_) {
    const absl::Status status = absl::FailedPreconditionError("connection runtime unavailable");
    record_failure(status);
    return make_failed_read_future(status, request.tensor_key);
  }

  auto start_time = absl::Now();
  auto inner_future = adapter_->read_tensor(request, local_binding_, remote_binding_);
  auto self = shared_from_this();
  auto promise = std::make_shared<std::promise<transport::read_result_t>>();
  auto future = promise->get_future();
  const std::string tensor_key = request.tensor_key;
  folly::via(async_runtime_->blocking_executor(),
             [self,
              start_time,
              tensor_key,
              promise,
              future = std::move(inner_future)]() mutable {
               transport::read_result_t result;
               try {
                 result = future.get();
               } catch (const std::exception& ex) {
                 result.status = absl::UnknownError(ex.what());
                 result.tensor_key = tensor_key;
               } catch (...) {
                 result.status = absl::UnknownError("read result future raised non-standard exception");
                 result.tensor_key = tensor_key;
               }
               const absl::Duration latency = absl::Now() - start_time;
               if (result.status.ok()) {
                 self->record_success(latency);
               } else {
                 self->record_failure(result.status);
               }
               promise->set_value(std::move(result));
             });
  return future;
}

absl::Status Connection::close() {
  if (!adapter_) {
    return absl::FailedPreconditionError("connection adapter unavailable");
  }
  return adapter_->close(remote_binding_);
}

ConnectionStats Connection::snapshot() const {
  absl::MutexLock lock(&mu_);
  return stats_;
}

HealthState Connection::health() const {
  absl::MutexLock lock(&mu_);
  return derive_health(stats_);
}

void Connection::record_success(absl::Duration latency) {
  {
    absl::MutexLock lock(&mu_);
    stats_.success_count += 1;
    stats_.last_success = absl::Now();
    stats_.last_latency = latency;
  }
  if (link_state_) {
    link_state_->record_success(latency);
  }
}

void Connection::record_failure(const absl::Status& status) {
  {
    absl::MutexLock lock(&mu_);
    stats_.failure_count += 1;
    stats_.last_failure = absl::Now();
    stats_.last_error = status.ToString();
  }
  if (link_state_) {
    link_state_->record_failure(status);
  }
}

} // namespace tensorcast::communicator::routing
