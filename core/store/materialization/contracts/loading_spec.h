// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "folly/futures/Future.h"

#include "core/common/ready_signal.h"
#include "core/store/communication_types.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/view/view_id.h"
#include "core/store/replica/memory_state.h"
#include "gsl/pointers"

namespace tensorcast::store::loading {

using tensorcast::store::materialization::view::TransformPlacement;
using tensorcast::store::materialization::view::VariantIdentity;

// ══════════════════════════════════════════════════════════════════════════
// Replica Sources - Describe where data comes from
// ══════════════════════════════════════════════════════════════════════════

enum class SourcePreference : uint8_t { kUnspecified, kAuto, kPreferP2P, kPreferDisk };

enum class MaterializationSource : uint8_t { kUnspecified, kDisk, kP2P, kLocalReplica };

enum class ExportPolicy : uint8_t { kUnspecified, kNever, kAuto, kForce };
enum class SourceMutationPolicy : uint8_t { kUnspecified, kReadWrite, kReadOnly };
enum class SourceLocalityHint : uint8_t { kAuto, kHostLocal, kSharedSource };

struct MaterializeIntoTargetResult {
  MaterializationSource source{MaterializationSource::kUnspecified};
  uint64_t requested_bytes{0};
  uint64_t committed_bytes{0};
  uint64_t fallback_bytes{0};
  uint64_t residual_bytes{0};
  uint64_t actual_collective_committed_bytes{0};
  uint64_t actual_local_typed_bytes{0};
  uint64_t actual_generic_backend_bytes{0};
  bool collective_handled{false};
  bool direct_write_supported{false};
  bool source_ordered{false};
  std::string dominant_executor;
  std::string selection_reason;

  struct DebugStats {
    std::uint64_t produced_chunks{0};
    std::uint64_t produced_bytes{0};
    std::uint64_t source_read_at_us_total{0};
    std::uint64_t gpu_write_wait_us_total{0};
    std::uint64_t gpu_write_bytes_total{0};
  };

  std::optional<DebugStats> debug_stats;
};

struct IntoTargetStorage {
  gsl::not_null<void*> base_ptr;
  uint64_t length{0};
};

struct IntoTargetLayout {
  std::vector<IntoTargetStorage> storages;
  uint64_t total_size{0};
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

inline std::string build_synthetic_payload_canonical_index_json(uint64_t byte_length) {
  return std::string("{\"payload\":[0,") + std::to_string(byte_length) + ",[" + std::to_string(byte_length) +
      "],[1],\"torch.uint8\",0]}";
}

inline loader::ByteRangeMap build_identity_byte_range_map(uint64_t total_bytes) {
  loader::ByteRangeMap map;
  map.total_bytes = total_bytes;
  map.num_sources = 1;
  if (total_bytes > 0) {
    map.segments.push_back(
        loader::ByteRangeSegment{
            .kind = loader::ByteRangeSegment::Kind::kData,
            .dst_offset = 0,
            .length = total_bytes,
            .src_offset = 0,
            .source_index = 0,
        });
  }
  return map;
}

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

struct DiskMetadata {
  bool descriptor_present{false};
  std::optional<std::string> schema_version;
  std::optional<std::string> canonical_index_json;
  std::optional<std::string> source_index_json;
  std::optional<std::string> index_multihash;
  std::optional<std::string> data_multihash;
  std::optional<uint64_t> logical_total_size;
  std::optional<uint64_t> source_total_size_bytes;
  std::optional<bool> is_safetensors;
};

struct TransportSchedulingGroupHint {
  std::string group_id;
  std::string group_kind;
  uint32_t total_parts{0};
  std::string part_id;
  uint32_t priority{0};
  uint64_t epoch{0};
};

struct CollectiveLoadGroupHint {
  std::string group_id;
  uint32_t world_size{0};
  uint32_t rank{0};
};

struct RetrievalPolicy {
  SourcePreference preference{SourcePreference::kAuto};
  bool allow_p2p{true};
  bool allow_disk{true};
};

struct ExecutionTopologyContext {
  std::optional<CollectiveLoadGroupHint> collective_load_group;
  SourceLocalityHint source_locality{SourceLocalityHint::kAuto};
  std::optional<std::string> source_sharing_domain;
};

inline absl::Status validate_retrieval_policy(
    const RetrievalPolicy& policy,
    std::string_view policy_name = "source_policy") {
  if (!policy.allow_p2p && policy.preference == SourcePreference::kPreferP2P) {
    return absl::InvalidArgumentError(
        std::string(policy_name) + " disallows P2P but preference=PREFER_P2P was requested");
  }
  if (!policy.allow_disk && policy.preference == SourcePreference::kPreferDisk) {
    return absl::InvalidArgumentError(
        std::string(policy_name) + " disallows disk but preference=PREFER_DISK was requested");
  }
  return absl::OkStatus();
}

struct MaterializeHints {
  size_t max_buffer_bytes = 256ULL << 20; // 256 MB default
  std::chrono::milliseconds pinned_timeout{0};
  // Request-level budget propagated from upper layers (e.g., RPC deadline).
  // 0 means "unspecified".
  std::chrono::milliseconds request_budget{0};
  // Upper bound for each Global Store transport wait call. 0 means default.
  std::chrono::milliseconds transport_wait_timeout{0};
  // Optional requester identity used by Global Store transport scheduler.
  std::string transport_requester_worker_id;
  // Optional request idempotency key for transport scheduling.
  std::string transport_request_id;
  // Optional scheduler group hint for fairness/completion-aware dispatch.
  std::optional<TransportSchedulingGroupHint> transport_scheduling_group;
  // Optional same-host multi-rank hint for shared-window disk loading.
  std::optional<CollectiveLoadGroupHint> collective_load_group;
  // Optional topology-locality hint. This remains distinct from retrieval
  // policy so the strategy plane can reason about source sharing without
  // rewriting semantic or transport policy.
  SourceLocalityHint source_locality{SourceLocalityHint::kAuto};
  std::optional<std::string> source_sharing_domain;
  uint32_t pipeline_concurrency = 4;
  std::string artifact_id;
  bool prefer_pageable_cpu{false};
  std::optional<DiskMetadata> disk_metadata;

  enum class Verify : std::uint8_t { NONE, CHECKSUM, FULL_DIGEST };
  Verify verify = Verify::CHECKSUM;
  SourcePreference source_preference{SourcePreference::kAuto};
  bool allow_p2p{true};
  bool allow_disk{true};
  ExportPolicy export_policy{ExportPolicy::kNever};
  bool need_view_data_hash{true};
  SourceMutationPolicy source_mutation_policy{SourceMutationPolicy::kReadWrite};

  std::optional<VariantIdentity> variant;

  [[nodiscard]] RetrievalPolicy retrieval_policy() const {
    return RetrievalPolicy{
        .preference = source_preference,
        .allow_p2p = allow_p2p,
        .allow_disk = allow_disk,
    };
  }

  void set_retrieval_policy(const RetrievalPolicy& policy) {
    source_preference = policy.preference;
    allow_p2p = policy.allow_p2p;
    allow_disk = policy.allow_disk;
  }

  [[nodiscard]] ExecutionTopologyContext execution_topology() const {
    return ExecutionTopologyContext{
        .collective_load_group = collective_load_group,
        .source_locality = source_locality,
        .source_sharing_domain = source_sharing_domain,
    };
  }

  void set_execution_topology(const ExecutionTopologyContext& topology) {
    collective_load_group = topology.collective_load_group;
    source_locality = topology.source_locality;
    source_sharing_domain = topology.source_sharing_domain;
  }
};

inline int resolve_materialization_concurrency(int daemon_num_threads, const MaterializeHints& hints) {
  const int bounded_daemon_threads = std::max(1, daemon_num_threads);
  if (hints.pipeline_concurrency == 0) {
    return bounded_daemon_threads;
  }
  const uint32_t bounded_hint =
      std::min<uint32_t>(hints.pipeline_concurrency, static_cast<uint32_t>(std::numeric_limits<int>::max()));
  return std::max(1, std::min(bounded_daemon_threads, static_cast<int>(bounded_hint)));
}

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

struct CpuMemfdRegion {
  int fd{-1};
  uint64_t size_bytes{0};
  uint64_t offset_bytes{0};
};

struct ReplicaHandle {
  ReplicaKey replica_key;
  std::shared_ptr<common::ReadySignal<absl::Status>> ready_signal;
  replica::MemoryState cpu_state{replica::MemoryState::UNINITIALIZED};
  replica::MemoryState gpu_state{replica::MemoryState::UNINITIALIZED};
  void* gpu_base_ptr{nullptr};
  CudaIpcHandle cuda_ipc_handle;
  std::optional<CpuMemfdRegion> cpu_memfd_region;
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
