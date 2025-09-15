// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <optional>
#include <string>

#include "core/common/device_types.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/common/memory/virtual_address_space.h"
#include "core/store/loading/loading_spec.h"
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
  // Shared Distributed Virtual Memory Pool for managing virtual address spaces
  gsl::not_null<std::shared_ptr<common::memory::VirtualAddressSpace>> virtual_addr_space;

  // Optional: Explicitly provide artifact size if known.
  std::optional<uint64_t> expected_artifact_size;

  // Streaming transfer configuration
  // Maximum buffer size in bytes for streaming transfers. Defaults to 256 MB.
  size_t max_buffer_bytes = (256ULL << 20);

  // Timeout for pinned memory allocation operations
  std::chrono::milliseconds pinned_memory_timeout = std::chrono::milliseconds::zero();

  // Whether to enable P2P communication for this replica
  bool p2p_comm_enabled = false;

  // (virtual_addr_space lock strictness is internal policy now)

  // Future runtime configurations can be added here:
  // - Tensor compression strategies
  // - Quantization settings
  // - Memory layout preferences
  // - Performance tuning parameters
};

} // namespace tensorcast::store::replica
