// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/store/components/global_store_client.h"

namespace tensorcast::daemon {

class WorkerDirectoryCache {
 public:
  struct Entry {
    std::string address;
    absl::Time refreshed_at{absl::UnixEpoch()};
  };

  explicit WorkerDirectoryCache(std::shared_ptr<store::components::IGlobalStoreClient> global_store_client);

  [[nodiscard]] bool is_fresh(std::string_view daemon_id, absl::Time now, absl::Duration staleness_budget) const;

  [[nodiscard]] absl::Status warm_for_daemons(
      const std::vector<std::string>& daemon_ids,
      absl::Time now,
      absl::Duration staleness_budget);

  [[nodiscard]] absl::StatusOr<std::string> resolve_daemon_address(
      std::string_view daemon_id,
      absl::Time now,
      absl::Duration staleness_budget);

 private:
  void update_from_workers(
      const std::vector<store::components::ActiveWorkerInfo>& workers,
      const absl::flat_hash_set<std::string>& requested_daemon_ids,
      absl::Time now) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::shared_ptr<store::components::IGlobalStoreClient> global_store_client_;

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, Entry> entries_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
