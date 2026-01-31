// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "absl/time/time.h"

namespace tensorcast::daemon {

struct DaemonOptions {
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

  // Stable daemon identity for control-plane actions (derived from DaemonConfig.daemon_id).
  std::string daemon_id;

  // Local handle plane (UDS) for FD handoff + lease release.
  std::string local_handle_socket_path;
  // When unset, the daemon uses a conservative default (see HandleLeaseRegistry::Options).
  // When set to 0ms, TTL is disabled and handle leases rely on explicit ReleaseHandle / PID-exit cleanup.
  std::optional<std::chrono::milliseconds> handle_lease_ttl;
  // Best-effort guardrail: limit lease-bearing handle mints per second (0 => unlimited).
  uint32_t handle_lease_max_mints_per_second{0};

  // CPU shared-memory materialization (memfd-backed UMA CPU arena).
  bool cpu_shared_memory_enabled{false};
  // Enable verification for MaterializeIntoTarget external target writes.
  bool external_target_verification_enabled{false};

  // API behavior flags
  // If true, GetLoadedReplicasV2 uses opaque cursor tokens based on a stable
  // ordering (artifact_id, device_id). If false (default), uses numeric
  // index tokens.
  bool use_cursor_pagination{false};

  PostSealPolicy post_seal_policy{};
};

} // namespace tensorcast::daemon
