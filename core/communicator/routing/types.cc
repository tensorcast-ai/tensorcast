// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/routing/types.h"

namespace tensorcast::communicator::routing {
namespace {

HealthState derive_health_from_counts(uint64_t success_count,
                                      uint64_t failure_count,
                                      absl::Time last_success,
                                      absl::Time last_failure) {
  if (success_count == 0 && failure_count == 0) {
    return HealthState::kUnknown;
  }
  if (success_count == 0 && failure_count > 0) {
    return HealthState::kUnhealthy;
  }
  if (failure_count == 0) {
    return HealthState::kHealthy;
  }
  if (last_failure > last_success) {
    return HealthState::kDegraded;
  }
  return HealthState::kHealthy;
}

} // namespace

std::string_view to_string(ConnectionProtocol protocol) {
  switch (protocol) {
    case ConnectionProtocol::kAuto:
      return "AUTO";
    case ConnectionProtocol::kRdma:
      return "RDMA";
    case ConnectionProtocol::kTcp:
      return "TCP";
    case ConnectionProtocol::kMtcp:
      return "MTCP";
    case ConnectionProtocol::kNvlink:
      return "NVLINK";
    case ConnectionProtocol::kShm:
      return "SHM";
    default:
      return "UNKNOWN";
  }
}

std::string_view to_string(ConnectionType type) {
  switch (type) {
    case ConnectionType::kForward:
      return "FORWARD";
    case ConnectionType::kP2P:
      return "P2P";
    case ConnectionType::kSwitch:
      return "SWITCH";
    default:
      return "UNKNOWN";
  }
}

bool EndpointBinding::has_network_address() const {
  return !ip.empty() && port != 0;
}

HealthState derive_health(const ConnectionStats& stats) {
  return derive_health_from_counts(
      stats.success_count,
      stats.failure_count,
      stats.last_success,
      stats.last_failure);
}

HealthState derive_health(const LinkStats& stats) {
  return derive_health_from_counts(
      stats.success_count,
      stats.failure_count,
      stats.last_success,
      stats.last_failure);
}

} // namespace tensorcast::communicator::routing
