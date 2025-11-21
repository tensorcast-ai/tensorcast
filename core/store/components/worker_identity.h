// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>

namespace tensorcast::store::components {

/**
 * @brief Canonical worker identity shared across runtime services.
 *
 * Stores the identifiers needed to publish replicas to the Global Store.
 */
struct WorkerIdentity {
  std::string worker_id;
  std::string node_id;
  std::string node_address;
  uint32_t grpc_port{0};
  uint32_t p2p_port{0};
};

} // namespace tensorcast::store::components
