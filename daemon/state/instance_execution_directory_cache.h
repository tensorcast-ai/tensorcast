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

class InstanceExecutionDirectoryCache {
 public:
  struct Entry {
    std::string instance_id;
    std::string daemon_id;
    std::string execution_host_kind;
    std::string execution_endpoint;
    std::string engine;
    uint64_t capability_flags{0};
  };

  struct Snapshot {
    std::vector<Entry> entries;
    absl::Time refreshed_at{absl::UnixEpoch()};
    uint64_t cache_epoch{0};
  };

  explicit InstanceExecutionDirectoryCache(std::shared_ptr<store::components::IGlobalStoreClient> global_store_client);

  [[nodiscard]] absl::StatusOr<Snapshot> list_instances(
      bool include_unavailable,
      uint64_t required_capability_flags,
      absl::Time now,
      absl::Duration staleness_budget);

  [[nodiscard]] absl::StatusOr<Entry> resolve_instance_execution(
      std::string_view instance_id,
      absl::Time now,
      absl::Duration staleness_budget);

 private:
  struct CacheState {
    std::vector<Entry> entries;
    absl::flat_hash_map<std::string, size_t> index_by_instance_id;
    absl::Time refreshed_at{absl::UnixEpoch()};
    uint64_t cache_epoch{0};
  };

  [[nodiscard]] static bool is_state_fresh(const CacheState& state, absl::Time now, absl::Duration staleness_budget);
  [[nodiscard]] static Snapshot filtered_snapshot(const CacheState& state, uint64_t required_capability_flags);

  [[nodiscard]] CacheState& state_for(bool include_unavailable) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  [[nodiscard]] const CacheState& state_for(bool include_unavailable) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void update_from_instances(
      const std::vector<store::components::ActiveInstanceInfo>& instances,
      CacheState& state,
      absl::Time now) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::shared_ptr<store::components::IGlobalStoreClient> global_store_client_;

  mutable absl::Mutex mu_;
  CacheState active_state_ ABSL_GUARDED_BY(mu_);
  CacheState include_unavailable_state_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
