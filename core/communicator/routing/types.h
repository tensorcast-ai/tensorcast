// Copyright (c) 2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ROUTING_TYPES_H_
#define CORE_COMMUNICATOR_ROUTING_TYPES_H_

#include <cstdint>
#include <string>
#include <string_view>

#include "absl/hash/hash.h"
#include "absl/time/time.h"
#include "core/communicator/base/constants.h"

namespace tensorcast::communicator::routing {

enum class ConnectionProtocol {
  kAuto = 0,
  kRdma = 1,
  kTcp = 2,
  kMtcp = 3,
  kNvlink = 4,
  kShm = 5,
};

enum class ConnectionType {
  kForward = 0,
  kP2P = 1,
  kSwitch = 2,
};

std::string_view to_string(ConnectionProtocol protocol);
std::string_view to_string(ConnectionType type);

struct EndpointBinding {
  std::string endpoint_id;
  std::string node_id;
  std::string ip;
  uint16_t port = 0;
  int dev_type = base::COMMUNICATE_ENGINE_DEV_CPU;
  int dev_id = 0;

  bool has_network_address() const;
};

struct ReadRequest {
  std::string tensor_key;
  uint64_t addr = 0;
  uint64_t bytes = 0;
  int dev_type = base::COMMUNICATE_ENGINE_DEV_CPU;
  int dev_id = 0;
  uint64_t remote_offset = 0;
};

struct ConnectionKey {
  std::string src_endpoint_id;
  std::string dst_endpoint_id;
  ConnectionProtocol protocol = ConnectionProtocol::kAuto;

  bool operator==(const ConnectionKey& other) const = default;
};

template <typename H>
H AbslHashValue(H h, const ConnectionKey& key) {
  return H::combine(std::move(h), key.src_endpoint_id, key.dst_endpoint_id, key.protocol);
}

enum class HealthState {
  kUnknown = 0,
  kHealthy = 1,
  kDegraded = 2,
  kUnhealthy = 3,
};

struct ConnectionStats {
  uint64_t success_count = 0;
  uint64_t failure_count = 0;
  absl::Time last_success = absl::InfinitePast();
  absl::Time last_failure = absl::InfinitePast();
  absl::Duration last_latency = absl::ZeroDuration();
  std::string last_error;
};

struct LinkStats {
  uint64_t success_count = 0;
  uint64_t failure_count = 0;
  absl::Time last_success = absl::InfinitePast();
  absl::Time last_failure = absl::InfinitePast();
  absl::Duration last_latency = absl::ZeroDuration();
  std::string last_error;
};

HealthState derive_health(const ConnectionStats& stats);
HealthState derive_health(const LinkStats& stats);

} // namespace tensorcast::communicator::routing

#endif // CORE_COMMUNICATOR_ROUTING_TYPES_H_
