// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include "absl/hash/hash.h"
#include "absl/status/status.h"

#include "core/store/communication_types.h"
#include "core/store/device_types.h"
#include "core/store/model/memory_state.h"

namespace stepcast::store {

// ══════════════════════════════════════════════════════════════════════════
// Model Sources - Describe where data comes from
// ══════════════════════════════════════════════════════════════════════════

/**
 * @brief Load model from disk
 */
struct DiskSource {
  std::filesystem::path path;
  std::optional<uint64_t> expected_size;
};

/**
 * @brief Load model from memory buffer (for testing or small models)
 */
struct InlineBufferSource {
  std::shared_ptr<const void> data;
  uint64_t size_bytes;
};

/**
 * @brief Unified source type for model loading
 */
using ModelSource = std::variant<
    DiskSource,
    P2PSource,
    InlineBufferSource
    // Future: S3Source, AzureBlobSource, OSSSource...
    >;

// ══════════════════════════════════════════════════════════════════════════
// Model Target - Describe where data goes
// ══════════════════════════════════════════════════════════════════════════

/**
 * @brief Target location for model loading
 */
struct ModelTarget {
  Location location;
};

// ══════════════════════════════════════════════════════════════════════════
// Loading Configuration
// ══════════════════════════════════════════════════════════════════════════

/**
 * @brief Collection of tuning parameters for model loading
 */
struct LoadingHints {
  size_t max_buffer_bytes = 256ULL << 20; // 256 MB default
  std::chrono::milliseconds pinned_timeout{0};
  uint32_t pipeline_concurrency = 4;

  // Hint: Prefer loading the model into the Pageable-Chunk CPU Cache (UPC-Cache)
  // instead of the traditional pinned host memory path. When set to true the
  // CheckpointStore and Loader pipeline should attempt to allocate the model
  // in PAGEABLE_CPU memory if the underlying components support it.
  bool prefer_pageable_cpu{false};

  enum class Verify : std::uint8_t { NONE, CHECKSUM, FULL_DIGEST };
  Verify verify = Verify::CHECKSUM; // Verification strategy
};

/**
 * @brief Complete loading specification for a model
 */
struct ModelLoadSpec {
  std::string identifier;
  ModelSource source;
  ModelTarget target;
  LoadingHints hints;
};

// ══════════════════════════════════════════════════════════════════════════
// Model Instance Management
// ══════════════════════════════════════════════════════════════════════════

/**
 * @brief A globally-unique key for a single model replica bound to a device.
 */
struct InstanceKey {
  std::string model_id; // e.g. "llama-7b"
  DeviceKey device; // physical placement
  uint32_t replica{0}; // multiple replicas on the same device

  bool operator==(const InstanceKey&) const = default;
};

/**
 * @brief Hash functor so we can use InstanceKey in absl::flat_hash_map.
 */
struct InstanceKeyHash {
  size_t operator()(const InstanceKey& k) const {
    return absl::HashOf(k.model_id, static_cast<int>(k.device.type), k.device.ordinal, k.replica);
  }
};

/**
 * @brief Handle returned from model loading operations
 */
struct ModelHandle {
  InstanceKey instance_key;
  std::shared_future<absl::Status> ready_future;
  MemoryState cpu_state{MemoryState::UNINITIALIZED};
  MemoryState gpu_state{MemoryState::UNINITIALIZED};
  void* gpu_base_ptr{nullptr};
  CudaIpcHandle cuda_ipc_handle;

  [[nodiscard]] const InstanceKey& key() const {
    return instance_key;
  }
  [[nodiscard]] MemoryState state(DeviceType type) const;
  absl::Status wait_ready(std::chrono::milliseconds timeout);
};

} // namespace stepcast::store