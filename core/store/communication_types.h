// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "core/common/memory/memory_location.h"
#include "core/store/device_types.h"

namespace tensorcast::communicator {
class CommunicateEngine;
} // namespace tensorcast::communicator

namespace tensorcast::store {

/**
 * @brief Holds information about memory registered for communication.
 */
struct CommRegistrationInfo {
  uint64_t artifact_size = 0;
  MemoryLocation location = MemoryLocation::NONE;
  int device_id = -1; // -1 for CPU, device id for GPU
  int comm_dev_type = 0; // Device type used for communicator registration
  std::vector<uint64_t> buffer_addresses; // Addresses (cast to uint64_t) of registered regions/chunks
  std::vector<size_t> buffer_sizes; // Sizes of registered regions/chunks
  std::vector<std::string> remote_memory_keys; // Keys used for registration with CommunicateEngine
} __attribute__((aligned(128)));

/**
 * @brief P2P communication source information
 */
struct P2PSource {
  uint64_t size_bytes;
  std::string ip;
  uint16_t port;
  std::vector<std::string> memory_keys;
  std::vector<size_t> buf_sizes;
  bool enable_checksum = true;
  Location location;

  std::shared_ptr<tensorcast::communicator::CommunicateEngine> comm_engine;
};

} // namespace tensorcast::store