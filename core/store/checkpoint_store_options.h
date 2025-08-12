// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace stepcast {
namespace communicator {
class CommunicateEngine;
} // namespace communicator

namespace store {

class CommunicationManager; // forward declaration

/**
 * @brief Options structure used to configure a CheckpointStore instance.
 *
 * This replaces the long positional parameter list previously used by the
 * CheckpointStore constructor, making the API safer and future-proof.  All
 * fields have sensible defaults so callers may omit values they do not need
 * to customise.
 */
struct CheckpointStoreOptions {
  // Path were model sub-directories are stored.  Empty string means fully
  // qualified paths will be provided to the loading API.
  std::string storage_path;

  // Size of the pinned host memory pool in bytes.
  size_t memory_pool_size{8ULL << 30}; // 8 GiB

  // Number of I/O worker threads.
  int num_thread{10};

  // Chunk size in bytes for streaming transfers.
  size_t chunk_size{128ULL << 20}; // 128 MiB

  // Timeout for pinned-memory allocations.
  std::chrono::milliseconds pinned_memory_timeout{std::chrono::milliseconds{30000}}; // 30 s

  // Port used by CommunicateEngine for P2P transfers.
  uint16_t p2p_port{9090};

  // Address of the Global Store gRPC service ("host:port").  Empty string means
  // the CheckpointStore will operate in standalone mode without remote
  // coordination and P2P source discovery.
  std::string global_store_address;

  // (Phase-3) Optional externally created CommunicationManager so multiple
  // CheckpointStore instances can share the same underlying CommunicateEngine
  // and listen socket.  When provided, the CheckpointStore will reuse this
  // manager instead of creating its own internal instance.
  std::shared_ptr<stepcast::store::CommunicationManager> comm_manager{nullptr};

  // Maximum number of concurrent model transfers that can use the shared
  // streaming pinned buffer pool. Each transfer receives an isolated buffer
  // instance (lease). Defaults to 1 (fully serialized).
  int streaming_buffer_max_concurrent_sessions{1};

  // Chunk size for Distributed Virtual Memory Pool (DVMP) allocations.
  // This controls the granularity of memory allocations for model chunks.
  // Default: 256 MiB for optimal GPU transfer performance.
  size_t dvmp_chunk_size{256ULL << 20}; // 256 MiB
};

} // namespace store
} // namespace stepcast