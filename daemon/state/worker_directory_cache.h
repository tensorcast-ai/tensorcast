// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/store/components/global_store_client.h"

namespace tensorcast::daemon {

class WorkerDirectoryCache {
 public:
  struct Entry {
    std::string daemon_id;
    std::string worker_id;
    std::string node_id;
    std::string node_address;
    uint32_t grpc_port{0};
    uint32_t p2p_port{0};
    std::string address;
    uint64_t capability_flags{0};
  };

  struct Snapshot {
    std::vector<Entry> entries;
    absl::Time refreshed_at{absl::UnixEpoch()};
    uint64_t cache_epoch{0};
  };

  explicit WorkerDirectoryCache(std::shared_ptr<store::components::IGlobalStoreClient> global_store_client);

  [[nodiscard]] bool is_fresh(std::string_view daemon_id, absl::Time now, absl::Duration staleness_budget) const;

  [[nodiscard]] absl::Status warm_for_daemons(
      const std::vector<std::string>& daemon_ids,
      absl::Time now,
      absl::Duration staleness_budget);

  [[nodiscard]] absl::StatusOr<Snapshot> list_workers(
      bool include_unavailable,
      uint64_t required_capability_flags,
      absl::Time now,
      absl::Duration staleness_budget);

  [[nodiscard]] absl::StatusOr<std::string> resolve_daemon_address(
      std::string_view daemon_id,
      absl::Time now,
      absl::Duration staleness_budget);

  [[nodiscard]] absl::StatusOr<Entry> resolve_daemon_entry(
      std::string_view daemon_id,
      absl::Time now,
      absl::Duration staleness_budget);

 private:
  struct CacheState {
    std::vector<Entry> entries;
    absl::flat_hash_map<std::string, size_t> index_by_daemon_id;
    absl::Time refreshed_at{absl::UnixEpoch()};
    uint64_t cache_epoch{0};
  };

  [[nodiscard]] static bool is_state_fresh(const CacheState& state, absl::Time now, absl::Duration staleness_budget);
  [[nodiscard]] static Snapshot filtered_snapshot(const CacheState& state, uint64_t required_capability_flags);

  [[nodiscard]] CacheState& state_for(bool include_unavailable) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  [[nodiscard]] const CacheState& state_for(bool include_unavailable) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void update_from_workers(
      const std::vector<store::components::ActiveWorkerInfo>& workers,
      CacheState& state,
      absl::Time now) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::shared_ptr<store::components::IGlobalStoreClient> global_store_client_;

  mutable absl::Mutex mu_;
  CacheState active_state_ ABSL_GUARDED_BY(mu_);
  CacheState include_unavailable_state_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
