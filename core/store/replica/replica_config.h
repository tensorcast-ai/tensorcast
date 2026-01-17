// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <optional>
#include <string>

#include "core/common/async_runtime.h"
#include "core/common/const/granularity.h"
#include "core/common/device_types.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/memory_tier_config.h"
#include "gsl/pointers"

// No forward declarations from unrelated namespaces here

namespace tensorcast::store::replica {

/**
 * @brief Runtime configuration for a Replica instance.
 *
 * This structure contains runtime-related configurations that affect
 * how the replica operates, including the source for loading.
 */
struct ReplicaConfig {
  // Source of the replica data (using new unified types)
  loading::ArtifactSource source;

  // Unique identifier for the replica (used for logging, caching keys, RDMA registration).
  std::string artifact_identifier;

  // Explicit target device type (CPU, GPU, REMOTE). Defaults to CPU. If GPU is chosen, also
  // specify `local_device_id` below. This makes the target placement unambiguous and avoids
  // the legacy convention of inferring GPU vs. CPU from the sign of `local_device_id`.
  DeviceType device_type = DeviceType::CPU;

  // Target local GPU device for operations (if applicable).
  int local_device_id = -1; // -1 means unspecified; runtime will decide

  // Memory pools for allocation (can be shared across replicas).
  gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> pinned_buffer_pool;

  // Async runtime shared across daemon + store for executor-based concurrency.
  gsl::not_null<std::shared_ptr<common::AsyncRuntime>> async_runtime;
  // UMA chunking granularity (bytes)
  size_t artifact_chunk_bytes{tensorcast::common::consts::kArtifactChunkDefault};

  // Optional: Explicitly provide artifact size if known.
  std::optional<uint64_t> expected_artifact_size;

  // Streaming transfer configuration
  // Maximum buffer size in bytes for streaming transfers. Defaults to 256 MB.
  size_t max_buffer_bytes = (256ULL << 20);

  // Timeout for pinned memory allocation operations
  std::chrono::milliseconds pinned_memory_timeout = std::chrono::milliseconds::zero();

  // Default depth for StreamingPinnedBuffer instances created by this replica.
  size_t streaming_buffer_chunks{16};

  // Whether to enable P2P communication for this replica
  bool p2p_comm_enabled = false;

  // Future runtime configurations can be added here:
  // - Variant residency metadata (view identifiers)
  std::optional<std::string> view_id = std::nullopt;
  // - View execution plan metadata for variant-aware replicas
  std::optional<loader::ViewPlan> view_plan;
  // - Transform placement preference (server/client)
  loading::TransformPlacement transform_placement = loading::TransformPlacement::kServer;
  // - Tensor compression strategies
  // - Quantization settings
  // - Memory layout preferences
  // - Performance tuning parameters
  // - Memory tier enforcement
  std::optional<MemoryTierConfig> memory_tier_config;

  // When true, UMA CPU allocations are backed by memfd + MAP_SHARED so they can
  // be exported cross-process for zero-copy CPU tensor materialization.
  bool cpu_shared_memory_enabled{false};
};

} // namespace tensorcast::store::replica
