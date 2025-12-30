// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <variant>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "folly/futures/Future.h"

#include "core/common/ready_signal.h"
#include "core/store/communication_types.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/view/view_id.h"
#include "core/store/replica/memory_state.h"

namespace tensorcast::store::loading {

using tensorcast::store::materialization::view::TransformPlacement;
using tensorcast::store::materialization::view::VariantIdentity;

// ══════════════════════════════════════════════════════════════════════════
// Replica Sources - Describe where data comes from
// ══════════════════════════════════════════════════════════════════════════

enum class SourcePreference : uint8_t { kUnspecified, kAuto, kPreferP2P, kPreferDisk };

enum class MaterializationSource : uint8_t { kUnspecified, kDisk, kP2P, kLocalReplica };

struct MaterializeIntoTargetResult {
  MaterializationSource source{MaterializationSource::kUnspecified};
};

struct DiskSource {
  std::filesystem::path path;
  std::optional<uint64_t> expected_size;
  bool require_descriptor{true};
};

struct InlineBufferSource {
  std::shared_ptr<const void> data;
  uint64_t size_bytes;
};

using ArtifactSource = std::variant<
    DiskSource,
    P2PSource,
    InlineBufferSource
    // Future: S3Source, AzureBlobSource, OSSSource...
    >;

// ══════════════════════════════════════════════════════════════════════════
// Replica Target - Describe where data goes
// ══════════════════════════════════════════════════════════════════════════

struct ReplicaTarget {
  Location location;
};

enum class MaterializeMode : uint8_t { AUTO, COPY_ONLY, LOAD_ONLY };

// ══════════════════════════════════════════════════════════════════════════
// Loading Configuration
// ══════════════════════════════════════════════════════════════════════════

struct MaterializeHints {
  size_t max_buffer_bytes = 256ULL << 20; // 256 MB default
  std::chrono::milliseconds pinned_timeout{0};
  uint32_t pipeline_concurrency = 4;
  std::string artifact_id;
  std::string disk_path;
  bool prefer_pageable_cpu{false};

  enum class Verify : std::uint8_t { NONE, CHECKSUM, FULL_DIGEST };
  Verify verify = Verify::CHECKSUM;
  SourcePreference source_preference{SourcePreference::kAuto};

  std::optional<VariantIdentity> variant;
};

struct ReplicaLoadSpec {
  std::string identifier;
  ArtifactSource source;
  ReplicaTarget target;
  MaterializeHints hints;
};

// ══════════════════════════════════════════════════════════════════════════
// Replica Instance Management
// ══════════════════════════════════════════════════════════════════════════

struct ReplicaKey {
  std::string artifact_id;
  std::optional<std::string> view_id;
  DeviceKey device;
  uint32_t replica{0};

  bool operator==(const ReplicaKey&) const = default;
};

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

struct ReplicaKeyHash {
  size_t operator()(const ReplicaKey& k) const {
    return absl::HashOf(
        k.artifact_id, k.view_id, static_cast<int>(k.device.type), k.device.ordinal, k.device.uuid, k.replica);
  }
};

struct ReplicaHandle {
  ReplicaKey replica_key;
  std::shared_ptr<common::ReadySignal<absl::Status>> ready_signal;
  replica::MemoryState cpu_state{replica::MemoryState::UNINITIALIZED};
  replica::MemoryState gpu_state{replica::MemoryState::UNINITIALIZED};
  void* gpu_base_ptr{nullptr};
  CudaIpcHandle cuda_ipc_handle;
  std::optional<std::string> view_index_json;
  std::optional<std::string> view_data_hash;
  MaterializationSource source{MaterializationSource::kUnspecified};

  [[nodiscard]] const ReplicaKey& key() const {
    return replica_key;
  }

  [[nodiscard]] replica::MemoryState state(DeviceType type) const;
  [[nodiscard]] folly::SemiFuture<absl::Status> subscribe_ready() const;
  absl::Status wait_ready(std::chrono::milliseconds timeout) const;
};

} // namespace tensorcast::store::loading

inline folly::SemiFuture<absl::Status> tensorcast::store::loading::ReplicaHandle::subscribe_ready() const {
  if (!ready_signal) {
    return folly::makeSemiFuture<absl::Status>(absl::OkStatus());
  }
  return ready_signal->subscribe();
}

inline absl::Status tensorcast::store::loading::ReplicaHandle::wait_ready(std::chrono::milliseconds timeout) const {
  if (!ready_signal) {
    return absl::OkStatus();
  }
  try {
    if (timeout.count() > 0) {
      return std::move(subscribe_ready()).get(timeout);
    }
    return std::move(subscribe_ready()).get();
  } catch (const folly::FutureTimeout&) {
    return absl::DeadlineExceededError("replica did not reach ready state before timeout");
  } catch (const std::exception& ex) {
    return absl::InternalError(ex.what());
  } catch (...) {
    return absl::InternalError("replica wait_ready failed with unknown exception");
  }
}
