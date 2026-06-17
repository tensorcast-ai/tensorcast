// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include "core/common/memory/memory_location.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/routing/routing_context.h"
#include "core/store/device_types.h"

namespace tensorcast::store {

/**
 * @brief Holds information about memory registered for communication.
 */
struct ExportRegistration {
  uint64_t artifact_size = 0;
  common::memory::MemoryLocation location = common::memory::MemoryLocation::NONE;
  int device_id = -1; // -1 for CPU, device id for GPU
  int comm_dev_type = 0; // Device type used for communicator registration
  std::vector<uint64_t> buffer_addresses; // Addresses (cast to uint64_t) of registered regions/chunks
  std::vector<size_t> buffer_sizes; // Sizes of registered regions/chunks
  std::vector<std::string> remote_memory_keys; // Keys used for registration with Communicator
} __attribute__((aligned(128)));

/**
 * @brief P2P communication source information
 */
struct P2PSource {
  uint64_t size_bytes;
  std::string ip;
  uint16_t port;
  std::string local_endpoint_id;
  std::string remote_endpoint_id;
  std::vector<std::string> memory_keys;
  std::vector<size_t> buf_sizes;
  bool enable_checksum = true;
  bool source_is_view = false;
  Location location;

  // Optional verification metadata (JSON) passed from the sender side, e.g.,
  // via daemon LockTransportChunksResponse.verification_json. When present,
  // the receiver may verify the loaded replica against this info after
  // transfer. See core/common/artifact_verification.{h,cc} for the schema.
  std::string verification_json;

  // Optional on-disk fallback directory containing partition files
  // (e.g., tensor.data_0, tensor.data_1, ...). When non-empty, P2PLoader
  // will mux remote source with this disk source for resilience.
  std::string fallback_disk_dir;

  std::shared_ptr<communicator::routing::RoutingContext> routing_context;
  // Request-level budget propagated from upper layers (e.g., RPC deadline).
  // Remote source reads should fail fast when this budget is exhausted.
  std::chrono::milliseconds request_budget{0};
  // Optional artifact id used for diagnostics in remote source logs.
  std::string artifact_id;
  // Optional transport request id used only for diagnostics/correlation.
  std::string transport_request_id;
  std::shared_ptr<communicator::engine::Communicator> comm_engine;
};

} // namespace tensorcast::store
