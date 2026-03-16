// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "absl/time/time.h"
#include "core/common/capability_token.h"

namespace tensorcast::daemon {

struct DaemonOptions {
  struct InterDaemonGrpcSecurity {
    bool tls_enabled{false};
    bool mutual_auth_enabled{false};
    std::string cert_chain_pem;
    std::string private_key_pem;
    std::string root_cert_pem;
  };

  struct PostSealPolicy {
    bool migrate_views{false};
    bool migrate_transpose_only{false};
    bool reuse_views_if_safe{false};
    bool retire_pieces{false};
  };

  // Sweep/TTL configuration
  std::chrono::seconds sessions_ttl{std::chrono::seconds(60)};
  std::chrono::seconds locks_ttl{std::chrono::seconds(120)};
  std::chrono::milliseconds sessions_sweep_interval{std::chrono::milliseconds(10000)};
  std::chrono::milliseconds locks_sweep_interval{std::chrono::milliseconds(10000)};
  std::chrono::milliseconds verification_sweep_interval{std::chrono::milliseconds(500)};
  std::chrono::milliseconds proc_check_interval{std::chrono::milliseconds(5000)};
  std::chrono::milliseconds region_sweep_interval{std::chrono::milliseconds(5000)};

  // Eviction policy
  bool enable_periodic_eviction{false};
  double gpu_memory_limit_fraction{0.90};
  std::chrono::milliseconds eviction_check_interval{std::chrono::milliseconds(1000)};

  // Region registry limits
  size_t max_vram_regions{2048};
  absl::Duration max_region_ttl{absl::Minutes(10)};

  // Observability
  bool allow_high_card_attrs{false};

  // Persistence
  std::filesystem::path persistence_log_path{"/tmp/tensorcast_persistence.log"};

  // Shared storage root for disk paths (required).
  std::filesystem::path storage_path;
  // Import metadata root under daemon runtime topology.
  std::filesystem::path import_root;

  // Stable daemon identity for control-plane actions (derived from DaemonConfig.daemon_id).
  std::string daemon_id;
  InterDaemonGrpcSecurity inter_daemon_grpc_security{};

  struct ByteArtifactRouting {
    struct PayloadTransport {
      absl::Duration ref_ttl{absl::Minutes(5)};
      uint64_t max_chunk_bytes{1ULL << 20};
      absl::Duration fetch_deadline{absl::Seconds(5)};
      absl::Duration cleanup_interval{absl::Minutes(1)};
    };

    uint64_t shard_count{4096};
    uint64_t inline_payload_threshold_bytes{1ULL << 20};
    std::chrono::milliseconds route_staleness_budget{std::chrono::milliseconds(500)};
    std::chrono::milliseconds lease_ttl{std::chrono::seconds(5)};
    std::chrono::milliseconds keepalive_interval{std::chrono::seconds(1)};
    std::chrono::milliseconds worker_directory_staleness_budget{std::chrono::seconds(2)};
    uint64_t routing_epoch{1};
    bool shard_home_eligible{true};
    PayloadTransport payload_transport{};
  };

  ByteArtifactRouting byte_artifact_routing{};
  bool gateway_ingress_enabled{false};

  // Capability token key material (v2 envelopes).
  common::CapabilityTokenConfig capability_tokens{};

  struct RetentionHandles {
    bool enabled{false};
    std::chrono::milliseconds default_ttl{std::chrono::seconds(600)};
    std::chrono::milliseconds max_ttl{std::chrono::hours(24)};
  };

  RetentionHandles retention_handles{};

  // Local handle plane (UDS) for FD handoff + lease release.
  std::string local_handle_socket_path;
  // When unset, the daemon uses a conservative default (see HandleLeaseRegistry::Options).
  // When set to 0ms, TTL is disabled and handle leases rely on explicit ReleaseHandle / PID-exit cleanup.
  std::optional<std::chrono::milliseconds> handle_lease_ttl;
  // Best-effort guardrail: limit lease-bearing handle mints per second (0 => unlimited).
  uint32_t handle_lease_max_mints_per_second{0};

  // CPU shared-memory materialization (memfd-backed UMA CPU arena).
  bool cpu_shared_memory_enabled{true};
  // Enable verification for MaterializeIntoTarget external target writes.
  bool external_target_verification_enabled{false};
  // Max concurrent transport requests per registered memory replica.
  uint32_t max_concurrency{4};

  // API behavior flags
  // If true, GetLoadedReplicasV2 uses opaque cursor tokens based on a stable
  // ordering (artifact_id, device_id). If false (default), uses numeric
  // index tokens.
  bool use_cursor_pagination{false};

  PostSealPolicy post_seal_policy{};
};

} // namespace tensorcast::daemon
