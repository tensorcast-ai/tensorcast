// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include "core/common/const/granularity.h"
#include "core/store/components/communication_manager.h"

namespace tensorcast::store {

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

  // Transfer slice size in bytes for streaming transfers (pinned pool block size).
  // Canonical name: tx_slice_bytes
  size_t tx_slice_bytes{common::consts::kTxSliceDefault};

  // Timeout for pinned-memory allocations.
  std::chrono::milliseconds pinned_memory_timeout{std::chrono::milliseconds{30000}}; // 30 s

  // Host/interface used by the Communicator listener. Empty string falls back
  // to loopback, while "0.0.0.0" exposes the port on all interfaces.
  std::string p2p_listen_host{"127.0.0.1"};

  // Port used by Communicator for P2P transfers.
  uint16_t p2p_port{9090};

  // Toggle RDMA support for the Communicator. When true the communicator will
  // attempt to register RDMA transports; otherwise it operates in TCP-only
  // mode with staging buffers for GPU memory.
  bool enable_rdma{false};

  // Address of the Global Store gRPC service ("host:port").  Empty string means
  // the StoreEngine will operate in standalone mode without remote
  // coordination and P2P source discovery.
  std::string global_store_address;

  // (Phase-3) Optional externally created CommunicationManager so multiple
  // StoreEngine instances can share the same underlying Communicator
  // and listen socket.  When provided, the StoreEngine will reuse this
  // manager instead of creating its own internal instance.
  std::shared_ptr<components::CommunicationManager> comm_manager{nullptr};

  // Maximum number of concurrent replica transfers that can use the shared
  // streaming pinned buffer pool. Each transfer receives an isolated buffer
  // instance (lease). Defaults to 1 (fully serialized).
  int streaming_buffer_max_concurrent_sessions{1};

  // UMA/VS artifact chunk size for CPU-side allocations (bytes).
  // Canonical name: artifact_chunk_bytes
  size_t artifact_chunk_bytes{common::consts::kArtifactChunkDefault};

  // RFC-0007: Enable strong verification (FULL_DIGEST) by default on load.
  // When true, the loader will compute the data_multihash from the loaded
  // GPU buffer to strongly validate content-addressed identity without the
  // caller needing to set MaterializeHints::verify = FULL_DIGEST.
  bool force_full_digest_on_load{false};

  // Optional on-disk fallback directory for P2P loads. When set, P2PLoader
  // will mux remote source with the local disk partitions in this directory.
  std::string p2p_fallback_disk_dir;
};

} // namespace tensorcast::store
