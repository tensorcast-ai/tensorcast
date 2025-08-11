// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <chrono>
#include <optional>
#include <string>

#include "core/common/device_types.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/store/loading/loading_spec.h"
#include "gsl/pointers"

namespace stepcast::communicator {
class CommunicateEngine;
} // namespace stepcast::communicator

namespace stepcast::store {

/**
 * @brief Runtime configuration for a Model instance.
 *
 * This structure contains runtime-related configurations that affect
 * how the model operates, including the source for loading.
 */
struct ModelConfig {
  // Source of the model data (using new unified types)
  ModelSource source;

  // Unique identifier for the model (used for logging, caching keys, RDMA registration).
  std::string model_identifier;

  // Explicit target device type (CPU, GPU, REMOTE). Defaults to CPU. If GPU is chosen, also
  // specify `local_device_id` below. This makes the target placement unambiguous and avoids
  // the legacy convention of inferring GPU vs. CPU from the sign of `local_device_id`.
  ::stepcast::DeviceType device_type = ::stepcast::DeviceType::CPU;

  // Target local GPU device for operations (if applicable).
  int local_device_id = -1; // -1 means unspecified; runtime will decide

  // Memory pools for allocation (can be shared across models).
  gsl::not_null<std::shared_ptr<PinnedMemoryPool>> pinned_memory_pool;
  // Shared Distributed Virtual Memory Pool for managing virtual address spaces
  gsl::not_null<std::shared_ptr<memory::DistributedVirtualMemoryPool>> dvmp;

  // Shared streaming buffer used by all models/memory managers (pre-initialized, not null)
  gsl::not_null<std::shared_ptr<StreamingPinnedBuffer>> streaming_buffer;

  // Optional: Explicitly provide model size if known.
  std::optional<uint64_t> expected_model_size;

  // Streaming transfer configuration
  // Maximum buffer size in bytes for streaming transfers. Defaults to 256 MB.
  size_t max_buffer_bytes = (256ULL << 20);

  // Timeout for pinned memory allocation operations
  std::chrono::milliseconds pinned_memory_timeout = std::chrono::milliseconds::zero();

  // Whether to enable P2P communication for this model
  bool p2p_comm_enabled = false;

  // (dvmp lock strictness is internal policy now)

  // Future runtime configurations can be added here:
  // - Tensor compression strategies
  // - Quantization settings
  // - Memory layout preferences
  // - Performance tuning parameters
};

} // namespace stepcast::store