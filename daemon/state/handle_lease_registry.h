// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <unistd.h>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "core/common/memory/memory_location.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/unified_memory_authority.h"
#include "core/store/store_engine.h"
#include "daemon/state/session_lifecycle.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/observer_result.h"

namespace tensorcast::daemon {

class HandleLeaseRegistry {
 public:
  struct CpuMemfdDescriptor {
    int fd{-1};
    uint64_t size_bytes{0};
    uint64_t offset_bytes{0};
  };

  struct Options {
    size_t capacity{4096};
    // When ttl > 0, handle leases are guarded by a deadline in SessionLifecycleManager.
    // When ttl <= 0, TTL is disabled and handle leases rely on explicit release and PID-exit cleanup.
    absl::Duration ttl{absl::ZeroDuration()};
    // Global mint rate limiter (0 => unlimited). This is a best-effort guardrail
    // against abusive/buggy clients minting lease-bearing handles too quickly.
    uint32_t max_mints_per_second{0};
  };

  HandleLeaseRegistry(Options opts, store::StoreEngine& engine, SessionLifecycleManager& lifecycle);

  [[nodiscard]] absl::StatusOr<std::string> mint_cuda_ipc_lease(const store::loading::ReplicaKey& key, pid_t pid);

  [[nodiscard]] absl::StatusOr<std::string> mint_cpu_memfd_lease(
      const store::loading::ReplicaKey& key,
      pid_t pid,
      CpuMemfdDescriptor memfd,
      absl::Span<const uint32_t> exported_chunks);

  [[nodiscard]] absl::Status release(const std::string& lease_token);

  [[nodiscard]] absl::StatusOr<CpuMemfdDescriptor> get_cpu_memfd_descriptor(const std::string& lease_token) const;

  [[nodiscard]] size_t size() const;

 private:
  enum class HandleKind : uint8_t { kCudaIpc = 0, kCpuMemfd = 1 };

  struct CpuExportState;

  struct LeaseRecord {
    HandleKind kind{HandleKind::kCudaIpc};
    store::loading::ReplicaKey key;
    SessionLifecycleManager::LeaseId lease_id{0};
    std::shared_ptr<CpuExportState> cpu_export_state;
  };

  struct CpuExportState {
    absl::CondVar cv;
    bool ready{false};
    absl::Status status;
    uint64_t refcount{0};
    CpuMemfdDescriptor memfd;
    std::vector<uint32_t> chunks;
    std::optional<store::replica::UnifiedMemoryAuthority::StableLease> stable_lease;
    std::shared_ptr<void> export_keepalive;
  };

  [[nodiscard]] std::string mint_token_locked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  [[nodiscard]] absl::Status maybe_rate_limit_mint_locked(absl::Time now) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  [[nodiscard]] absl::StatusOr<std::shared_ptr<CpuExportState>> acquire_cpu_export_state_(
      const store::loading::ReplicaKey& key,
      CpuMemfdDescriptor memfd,
      absl::Span<const uint32_t> exported_chunks);
  void release_cpu_export_state_(const store::loading::ReplicaKey& key, const std::shared_ptr<CpuExportState>& state);

  const Options opts_;
  store::StoreEngine* engine_;
  SessionLifecycleManager* lifecycle_;

  mutable absl::Mutex mu_;
  absl::BitGen bitgen_ ABSL_GUARDED_BY(mu_);
  absl::Time mint_window_start_ ABSL_GUARDED_BY(mu_){absl::UnixEpoch()};
  uint32_t mint_window_count_ ABSL_GUARDED_BY(mu_){0};
  absl::flat_hash_map<std::string, LeaseRecord> leases_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<store::loading::ReplicaKey, std::shared_ptr<CpuExportState>, store::loading::ReplicaKeyHash>
      cpu_exports_ ABSL_GUARDED_BY(mu_);

  // --- Metrics ---
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> meter_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> active_leases_gauge_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> cpu_exports_gauge_;

  static void active_leases_gauge_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept;
  static void cpu_exports_gauge_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept;
};

} // namespace tensorcast::daemon
