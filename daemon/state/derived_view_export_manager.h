// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <sys/types.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/store/components/global_store_client.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/store_engine.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/types.h"

namespace tensorcast::daemon {

class DerivedViewExportManager {
 public:
  struct Options {
    absl::Duration ttl{absl::Minutes(10)};
    absl::Duration retry_retire_ttl{absl::Seconds(30)};
    std::optional<uint64_t> budget_override_bytes;
    std::optional<uint64_t> headroom_override_bytes;
  };

  enum class EntryState {
    kPending = 0,
    kReady = 1,
    kDraining = 2,
  };

  struct EntrySnapshot {
    ArtifactDeviceKey key;
    store::loading::ReplicaKey replica_key;
    SessionLifecycleManager::LeaseId use_lease_id{0};
    SessionLifecycleManager::LeaseId retention_lease_id{0};
    EntryState state{EntryState::kReady};
    uint64_t generation{0};
    absl::Duration ttl{absl::ZeroDuration()};
    uint64_t resident_bytes{0};
    uint64_t active_fetches{0};
    bool accept_new_fetches{true};
    absl::Time last_access_time{absl::InfinitePast()};
    absl::Time expiry_time{absl::InfiniteFuture()};
  };

  DerivedViewExportManager(store::StoreEngine& engine, SessionLifecycleManager& lifecycle);

  DerivedViewExportManager(store::StoreEngine& engine, SessionLifecycleManager& lifecycle, Options options);

  void set_global_store_client(std::shared_ptr<store::components::IGlobalStoreClient> client);

  [[nodiscard]] absl::Status acquire_prepare_budget(uint64_t reserved_bytes);

  void release_prepare_budget(uint64_t reserved_bytes);

  [[nodiscard]] absl::Status reserve(const store::loading::ReplicaKey& key, uint64_t reserved_bytes);

  [[nodiscard]] absl::Status commit_reserved(const store::loading::ReplicaKey& key);

  [[nodiscard]] absl::Status cancel_reserved(const store::loading::ReplicaKey& key);

  [[nodiscard]] absl::Status retain_or_refresh(const store::loading::ReplicaKey& key);

  [[nodiscard]] absl::Status begin_fetch(const store::loading::ReplicaKey& key, std::string_view fetch_id);

  void end_fetch(std::string_view fetch_id, std::string_view reason);

  [[nodiscard]] std::optional<EntrySnapshot> find_entry(const store::loading::ReplicaKey& key) const;

 private:
  struct PrepareBudgetWaitContext {
    const DerivedViewExportManager* manager{nullptr};
    uint64_t reserved_bytes{0};
    uint64_t pending_budget_bytes{0};
  };

  struct LocalDrainWaitContext {
    const DerivedViewExportManager* manager{nullptr};
    ArtifactDeviceKey key;
    uint64_t generation{0};
  };

  struct Entry {
    store::loading::ReplicaKey replica_key;
    SessionLifecycleManager::LeaseId use_lease_id{0};
    SessionLifecycleManager::LeaseId retention_lease_id{0};
    EntryState state{EntryState::kReady};
    uint64_t generation{0};
    absl::Duration ttl{absl::ZeroDuration()};
    uint64_t resident_bytes{0};
    uint64_t active_fetches{0};
    bool accept_new_fetches{true};
    absl::Time last_access_time{absl::InfinitePast()};
    absl::Time expiry_time{absl::InfiniteFuture()};
  };

  struct ActiveFetch {
    ArtifactDeviceKey key;
    uint64_t generation{0};
  };

  [[nodiscard]] static std::optional<ArtifactDeviceKey> to_entry_key(const store::loading::ReplicaKey& key);

  [[nodiscard]] absl::Status install_entry(const ArtifactDeviceKey& key, const store::loading::ReplicaKey& replica_key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  [[nodiscard]] absl::Status activate_reserved_entry(const ArtifactDeviceKey& key, Entry& entry)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  [[nodiscard]] absl::Status renew_entry(const ArtifactDeviceKey& key, Entry& entry) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  [[nodiscard]] absl::Status on_retention_expired(const ArtifactDeviceKey& key, uint64_t generation);

  [[nodiscard]] absl::Status safe_retire_published_replica(
      const store::loading::ReplicaKey& key,
      std::string_view reason) const;

  [[nodiscard]] absl::Status unregister_published_replica(const store::loading::ReplicaKey& key) const;

  [[nodiscard]] absl::Status arm_retry_retention(const ArtifactDeviceKey& key, uint64_t generation);

  [[nodiscard]] absl::Status arm_retry_retire(const ArtifactDeviceKey& key, uint64_t generation);

  [[nodiscard]] absl::Status on_retry_retire(const ArtifactDeviceKey& key, uint64_t generation);

  [[nodiscard]] static bool can_acquire_prepare_budget_locked(const PrepareBudgetWaitContext* ctx)
      ABSL_NO_THREAD_SAFETY_ANALYSIS;

  [[nodiscard]] static bool can_finish_local_drain_locked(const LocalDrainWaitContext* ctx)
      ABSL_NO_THREAD_SAFETY_ANALYSIS;

  [[nodiscard]] absl::Status maybe_evict_for_budget(const std::optional<ArtifactDeviceKey>& protected_key);

  [[nodiscard]] absl::Status retire_entry(
      const ArtifactDeviceKey& key,
      const Entry& snapshot,
      std::string_view reason,
      bool retry_on_failure);

  [[nodiscard]] uint64_t current_ready_derived_bytes_locked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  [[nodiscard]] uint64_t current_derived_bytes_locked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  store::StoreEngine& engine_;
  SessionLifecycleManager& lifecycle_;
  Options options_;
  std::shared_ptr<store::components::IGlobalStoreClient> global_store_client_;
  pid_t owner_pid_{0};

  mutable absl::Mutex mu_;
  absl::flat_hash_map<ArtifactDeviceKey, Entry> entries_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, ActiveFetch> active_fetches_ ABSL_GUARDED_BY(mu_);
  uint64_t pending_prepare_bytes_ ABSL_GUARDED_BY(mu_){0};
  uint64_t next_generation_ ABSL_GUARDED_BY(mu_){1};
};

} // namespace tensorcast::daemon
