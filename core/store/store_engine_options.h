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
 * @brief Options structure used to configure a StoreEngine instance.
 *
 * This replaces the long positional parameter list previously used by the
 * StoreEngine constructor, making the API safer and future-proof.  All
 * fields have sensible defaults so callers may omit values they do not need
 * to customise.
 */
struct StoreEngineOptions {
  // Path were replica sub-directories are stored.  Empty string means fully
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
  // the StoreEngine will operate in standalone mode without remote
  // coordination and P2P source discovery.
  std::string global_store_address;

  // (Phase-3) Optional externally created CommunicationManager so multiple
  // StoreEngine instances can share the same underlying CommunicateEngine
  // and listen socket.  When provided, the StoreEngine will reuse this
  // manager instead of creating its own internal instance.
  std::shared_ptr<stepcast::store::CommunicationManager> comm_manager{nullptr};

  // Maximum number of concurrent replica transfers that can use the shared
  // streaming pinned buffer pool. Each transfer receives an isolated buffer
  // instance (lease). Defaults to 1 (fully serialized).
  int streaming_buffer_max_concurrent_sessions{1};

  // Chunk size for Distributed Virtual Memory Pool (DVMP) allocations.
  // This controls the granularity of memory allocations for replica chunks.
  // Default: 256 MiB for optimal GPU transfer performance.
  size_t dvmp_chunk_size{256ULL << 20}; // 256 MiB

  // RFC-0007: Enable strong verification (FULL_DIGEST) by default on load.
  // When true, the loader will compute the data_multihash from the loaded
  // GPU buffer to strongly validate content-addressed identity without the
  // caller needing to set MaterializeHints::verify = FULL_DIGEST.
  bool force_full_digest_on_load{false};
};

} // namespace store
} // namespace stepcast