// Copyright (c) 2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_TOPOLOGY_TOPOLOGY_H_
#define CORE_COMMUNICATOR_TOPOLOGY_TOPOLOGY_H_

#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tensorcast::communicator::topology {

enum class PoolType {
  kUnspecified = 0,
  kCpu = 1,
  kGpu = 2,
};

enum class EndpointKind {
  kClient = 0,
  kSwitch = 1,
};

enum class EndpointType {
  kUnspecified = 0,
  kNic = 1,
  kPcie = 2,
  kNvlink = 3,
};

enum class LinkType {
  kForward = 0,
  kP2P = 1,
  kSwitch = 2,
};

std::string_view to_string(PoolType type);
std::string_view to_string(EndpointKind kind);
std::string_view to_string(EndpointType type);
std::string_view to_string(LinkType type);

struct Pool {
  std::string id;
  std::string name;
  PoolType type = PoolType::kUnspecified;
};

struct Endpoint {
  std::string id;
  std::string name;
  EndpointKind kind = EndpointKind::kClient;
  EndpointType type = EndpointType::kUnspecified;
  std::vector<std::string> pool_ids;
  double bandwidth_gbps = 0.0;
};

struct Link {
  std::string id;
  std::string name;
  LinkType type = LinkType::kForward;
  std::string src_endpoint_id;
  std::string dst_endpoint_id;
  double bandwidth_gbps = 0.0;
  double latency_us = 0.0;
};

struct ValidationOptions {
  bool require_endpoint_links = true;
  bool require_connected = false;
};

class Topology {
 public:
  Topology() = default;

  // Build a topology from explicit inputs only (no hardware discovery).
  static absl::StatusOr<Topology> Build(
      std::vector<Pool> pools,
      std::vector<Endpoint> endpoints,
      std::vector<Link> links,
      ValidationOptions options = {});

  const Pool* find_pool(std::string_view id) const;
  const Endpoint* find_endpoint(std::string_view id) const;
  const Link* find_link(std::string_view id) const;

  const absl::flat_hash_map<std::string, Pool>& pools() const {
    return pools_;
  }

  const absl::flat_hash_map<std::string, Endpoint>& endpoints() const {
    return endpoints_;
  }

  const absl::flat_hash_map<std::string, Link>& links() const {
    return links_;
  }

  absl::Status validate(ValidationOptions options = {}) const;

  std::string to_dot() const;

 private:
  absl::flat_hash_map<std::string, Pool> pools_;
  absl::flat_hash_map<std::string, Endpoint> endpoints_;
  absl::flat_hash_map<std::string, Link> links_;
};

} // namespace tensorcast::communicator::topology

#endif // CORE_COMMUNICATOR_TOPOLOGY_TOPOLOGY_H_
