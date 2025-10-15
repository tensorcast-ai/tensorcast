// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <variant>
#include "absl/hash/hash.h"
#include "absl/status/status.h"

#include "core/store/communication_types.h"
#include "core/store/device_types.h"
#include "core/store/loader/view_planner.h"
#include "core/store/replica/memory_state.h"

namespace tensorcast::store::loading {

// ══════════════════════════════════════════════════════════════════════════
// Replica Sources - Describe where data comes from
// ══════════════════════════════════════════════════════════════════════════

/**
 * @brief Load replica from disk
 */
struct DiskSource {
  std::filesystem::path path;
  std::optional<uint64_t> expected_size;
};

/**
 * @brief Load replica from memory buffer (for testing or small artifacts)
 */
struct InlineBufferSource {
  std::shared_ptr<const void> data;
  uint64_t size_bytes;
};

/**
 * @brief Unified source type for replica loading
 */
using ArtifactSource = std::variant<
    DiskSource,
    P2PSource,
    InlineBufferSource
    // Future: S3Source, AzureBlobSource, OSSSource...
    >;

// ══════════════════════════════════════════════════════════════════════════
// Replica Target - Describe where data goes
// ══════════════════════════════════════════════════════════════════════════

/**
 * @brief Target location for replica loading
 */
struct ReplicaTarget {
  Location location;
};

// ══════════════════════════════════════════════════════════════════════════
// Variant identity – describes non-canonical byte spaces
// ══════════════════════════════════════════════════════════════════════════

enum class TransformPlacement : uint8_t { kServer = 0, kClient = 1 };

struct VariantIdentity {
  std::string canonical_artifact_id;
  std::optional<std::string> view_id;
  std::optional<loader::ViewSpec> view_spec;
  TransformPlacement placement{TransformPlacement::kServer};
  std::optional<std::string> canonical_index_json;
  std::optional<loader::ViewPlan> cached_plan;
};

// ══════════════════════════════════════════════════════════════════════════
// Loading Configuration
// ══════════════════════════════════════════════════════════════════════════

/**
 * @brief Collection of tuning parameters for replica loading
 */
struct MaterializeHints {
  size_t max_buffer_bytes = 256ULL << 20; // 256 MB default
  std::chrono::milliseconds pinned_timeout{0};
  uint32_t pipeline_concurrency = 4;
  // Content-addressed identity (mi2:...) when available.
  std::string artifact_id;
  // Optional: explicitly provide a disk path as a source hint.
  // When non-empty and content-addressed routing is unavailable, the loader
  // may use this path via DiskLoader as an explicit override.
  std::string disk_path;

  // Hint: Prefer loading the replica into the Pageable-Chunk CPU Cache (UPC-Cache)
  // instead of the traditional pinned host memory path. When set to true the
  // StoreEngine and Loader pipeline should attempt to allocate the replica
  // in CPU memory if the underlying components support it.
  bool prefer_pageable_cpu{false};

  enum class Verify : std::uint8_t { NONE, CHECKSUM, FULL_DIGEST };
  Verify verify = Verify::CHECKSUM; // Verification strategy

  // Optional variant identity describing view-specific byte spaces
  std::optional<VariantIdentity> variant;
};

/**
 * @brief Complete loading specification for a replica
 */
struct ReplicaLoadSpec {
  std::string identifier;
  ArtifactSource source;
  ReplicaTarget target;
  MaterializeHints hints;
};

// ══════════════════════════════════════════════════════════════════════════
// Replica Instance Management
// ══════════════════════════════════════════════════════════════════════════

/**
 * @brief A globally-unique key for a single replica replica bound to a device.
 */
struct ReplicaKey {
  std::string artifact_id; // e.g. "llama-7b"
  std::optional<std::string> view_id; // Optional byte-space variant identifier
  DeviceKey device; // physical placement
  uint32_t replica{0}; // multiple replicas on the same device

  bool operator==(const ReplicaKey&) const = default;
};

// Stream operator for convenient logging: LOG(INFO) << replica_key;
inline std::ostream& operator<<(std::ostream& os, const ReplicaKey& key) {
  os << "ReplicaKey{"
     << "artifact_id=" << key.artifact_id;
  if (key.view_id.has_value()) {
    os << ", view_id=" << *key.view_id;
  }
  os << ", device=";
  switch (key.device.type) {
    case DeviceType::GPU:
      os << "GPU";
      break;
    case DeviceType::CPU:
      os << "CPU";
      break;
    default:
      os << static_cast<int>(key.device.type);
      break;
  }
  os << ":" << key.device.ordinal;
  if (!key.device.uuid.empty()) {
    os << "(" << key.device.uuid << ")";
  }
  os << ", replica=" << key.replica << "}";
  return os;
}

/**
 * @brief Hash functor so we can use ReplicaKey in absl::flat_hash_map.
 */
struct ReplicaKeyHash {
  size_t operator()(const ReplicaKey& k) const {
    return absl::HashOf(
        k.artifact_id, k.view_id, static_cast<int>(k.device.type), k.device.ordinal, k.device.uuid, k.replica);
  }
};

/**
 * @brief Handle returned from replica loading operations
 */
struct ReplicaHandle {
  ReplicaKey replica_key;
  std::shared_future<absl::Status> ready_future;
  replica::MemoryState cpu_state{replica::MemoryState::UNINITIALIZED};
  replica::MemoryState gpu_state{replica::MemoryState::UNINITIALIZED};
  void* gpu_base_ptr{nullptr};
  CudaIpcHandle cuda_ipc_handle;
  std::optional<std::string> view_index_json;
  std::optional<std::string> view_data_hash;

  [[nodiscard]] const ReplicaKey& key() const {
    return replica_key;
  }

  [[nodiscard]] replica::MemoryState state(DeviceType type) const;
  absl::Status wait_ready(std::chrono::milliseconds timeout);
};

} // namespace tensorcast::store::loading
