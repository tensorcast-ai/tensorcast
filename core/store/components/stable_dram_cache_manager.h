// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/store/components/stable_dram_cache_policy.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/unified_memory_authority.h"
#include "gsl/pointers"

namespace tensorcast::store {
class MemoryTierBudget;
} // namespace tensorcast::store

namespace tensorcast::store::replica {
class Replica;
} // namespace tensorcast::store::replica

namespace tensorcast::store::components {

class ReplicaRegistry;

class StableDramCacheManager {
 public:
  using SpillEvictableCallback = std::function<bool(const loading::ReplicaKey&, const StableDramCachePolicy&)>;

  struct Config {
    gsl::not_null<ReplicaRegistry*> registry;
    std::shared_ptr<MemoryTierBudget> memory_tier_budget;
    std::function<absl::Status(const loading::ReplicaKey&)> spill_guard;
    SpillEvictableCallback spill_evictable;
  };

  struct AdmissionRequest {
    loading::ReplicaKey key;
    std::shared_ptr<replica::Replica> replica;
    uint64_t size_bytes{0};
    StableDramCachePolicy policy;
  };

  struct AdmissionResult {
    bool admitted{false};
    bool skipped{false};
  };

  explicit StableDramCacheManager(Config config);

  absl::StatusOr<AdmissionResult> admit(const AdmissionRequest& request);
  absl::Status update_policy(
      const loading::ReplicaKey& key,
      const StableDramCachePolicy& policy,
      std::optional<absl::Time> retention_deadline = std::nullopt);

  // Export-preemption hook: reclaim stable tier bytes by evicting cache entries so
  // correctness-critical exports (e.g., CPU memfd handles) can acquire UMA stable leases.
  //
  // This is intentionally best-effort and is only expected to evict cache-managed
  // entries; it will not release unrelated stable leases (e.g., active exports).
  absl::Status preempt_for_export(uint64_t required_bytes, const loading::ReplicaKey& exclude);

  bool is_evictable(const loading::ReplicaKey& key, absl::Time now) const;
  void on_replica_evicted(const loading::ReplicaKey& key, absl::string_view reason = "");
  void on_replica_evicted(
      const loading::ReplicaKey& key,
      const std::shared_ptr<replica::Replica>& replica,
      absl::string_view reason = "");
  void set_spill_evictable_callback(SpillEvictableCallback callback);

  uint64_t bytes_used() const;

 private:
  enum class EvictionMode : uint8_t { kAdmission, kPreemptForExport };

  struct CacheEntry {
    loading::ReplicaKey key;
    uint64_t size_bytes{0};
    uint64_t stable_bytes{0};
    StableDramCachePolicy policy;
    std::optional<absl::Time> retention_deadline;
    std::optional<replica::UnifiedMemoryAuthority::StableLease> stable_lease;
  };

  absl::StatusOr<replica::UnifiedMemoryAuthority::StableLease> acquire_stable_lease(
      const loading::ReplicaKey& key,
      const std::shared_ptr<replica::Replica>& replica) const;
  std::optional<loading::ReplicaKey> find_entry_key_locked(const loading::ReplicaKey& key) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void index_entry_alias_locked(const CacheEntry& entry) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void unindex_entry_alias_locked(const CacheEntry& entry) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::Status evict_for_bytes(
      uint64_t required_bytes,
      const loading::ReplicaKey& exclude,
      absl::Time now,
      bool spill_only,
      EvictionMode mode,
      absl::string_view reason);
  bool is_entry_evictable(const CacheEntry& entry, absl::Time now, EvictionMode mode) const;
  static bool is_spill_evictable(const CacheEntry& entry, const SpillEvictableCallback& spill_evictable);

  void record_bytes_delta(int64_t delta) const;
  void record_eviction() const;
  void record_hit() const;
  void record_miss() const;
  void record_ttl_expiration() const;

  gsl::not_null<ReplicaRegistry*> registry_;
  std::shared_ptr<MemoryTierBudget> memory_tier_budget_;
  std::function<absl::Status(const loading::ReplicaKey&)> spill_guard_;
  SpillEvictableCallback spill_evictable_;

  mutable absl::Mutex mu_;
  absl::flat_hash_map<loading::ReplicaKey, CacheEntry, loading::ReplicaKeyHash> entries_ ABSL_GUARDED_BY(mu_);
  // Reverse lookup from the underlying UMA stable-lease key back to the tracked stable-cache entry.
  // This avoids repeatedly scanning the full cache on eviction/preemption paths that only know the
  // physical lease key (for example, transient forwarder cleanup retiring a physical CPU replica).
  absl::flat_hash_map<
      loading::ReplicaKey,
      absl::flat_hash_set<loading::ReplicaKey, loading::ReplicaKeyHash>,
      loading::ReplicaKeyHash>
      entry_keys_by_lease_key_ ABSL_GUARDED_BY(mu_);
  std::atomic<uint64_t> bytes_used_{0};
};

} // namespace tensorcast::store::components
