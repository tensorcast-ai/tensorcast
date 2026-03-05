// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/routing/adapter.h"

#include <format>

#include "absl/status/status.h"
#include "core/communicator/routing/read_helpers.h"

namespace tensorcast::communicator::routing {

EngineAdapter::EngineAdapter(std::shared_ptr<engine::Communicator> engine)
    : engine_(std::move(engine)) {}

transport::future_read_result_t EngineAdapter::read_tensor(
    const ReadRequest& request,
    const EndpointBinding& /*local*/,
    const EndpointBinding& remote) {
  if (engine_ == nullptr) {
    return make_failed_read_future(
        absl::FailedPreconditionError("engine adapter missing engine"),
        request.tensor_key);
  }
  if (!remote.has_network_address()) {
    return make_failed_read_future(
        absl::InvalidArgumentError(
            std::format("missing network address for endpoint: {}", remote.endpoint_id)),
        request.tensor_key);
  }

  return engine_->read_tensor(
      request.tensor_key,
      request.addr,
      request.bytes,
      request.dev_type,
      request.dev_id,
      remote.ip,
      remote.port,
      request.remote_offset);
}

absl::Status EngineAdapter::close(const EndpointBinding& remote) {
  if (engine_ == nullptr) {
    return absl::FailedPreconditionError("engine adapter missing engine");
  }
  if (!remote.has_network_address()) {
    return absl::InvalidArgumentError(
        std::format("missing network address for endpoint: {}", remote.endpoint_id));
  }
  return engine_->close_connection(remote.ip, remote.port);
}

NvlinkAdapter::NvlinkAdapter(std::shared_ptr<engine::Communicator> engine)
    : engine_(std::move(engine)) {}

transport::future_read_result_t NvlinkAdapter::read_tensor(
    const ReadRequest& request,
    const EndpointBinding& /*local*/,
    const EndpointBinding& /*remote*/) {
  if (engine_ == nullptr) {
    return make_failed_read_future(
        absl::FailedPreconditionError("nvlink adapter missing engine"),
        request.tensor_key);
  }
  return engine_->read_tensor_local(
      request.tensor_key,
      request.addr,
      request.bytes,
      request.dev_type,
      request.dev_id,
      request.remote_offset);
}

absl::Status NvlinkAdapter::close(const EndpointBinding& /*remote*/) {
  if (engine_ == nullptr) {
    return absl::FailedPreconditionError("nvlink adapter missing engine");
  }
  return absl::OkStatus();
}

PcieAdapter::PcieAdapter(std::shared_ptr<engine::Communicator> engine)
    : engine_(std::move(engine)) {}

transport::future_read_result_t PcieAdapter::read_tensor(
    const ReadRequest& request,
    const EndpointBinding& /*local*/,
    const EndpointBinding& remote) {
  if (engine_ == nullptr) {
    return make_failed_read_future(
        absl::FailedPreconditionError("pcie adapter missing engine"),
        request.tensor_key);
  }
  if (!remote.has_network_address()) {
    return make_failed_read_future(
        absl::InvalidArgumentError(
            std::format("missing network address for endpoint: {}", remote.endpoint_id)),
        request.tensor_key);
  }

  return engine_->read_tensor(
      request.tensor_key,
      request.addr,
      request.bytes,
      request.dev_type,
      request.dev_id,
      remote.ip,
      remote.port,
      request.remote_offset);
}

absl::Status PcieAdapter::close(const EndpointBinding& remote) {
  if (engine_ == nullptr) {
    return absl::FailedPreconditionError("pcie adapter missing engine");
  }
  if (!remote.has_network_address()) {
    return absl::InvalidArgumentError(
        std::format("missing network address for endpoint: {}", remote.endpoint_id));
  }
  return engine_->close_connection(remote.ip, remote.port);
}

} // namespace tensorcast::communicator::routing
