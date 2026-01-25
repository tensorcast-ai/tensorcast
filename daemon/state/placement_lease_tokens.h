// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "daemon/state/session_lifecycle.h"

namespace tensorcast::daemon {

class PlacementLeaseTokens {
 public:
  struct Options {
    size_t capacity{4096};
    size_t token_bytes{32};
    // When the placement lease has no TTL (manual guard), tokens still expire
    // to keep the capability time-bounded.
    absl::Duration default_ttl{absl::Hours(24 * 30)};
  };

  explicit PlacementLeaseTokens(Options opts);

  [[nodiscard]] absl::StatusOr<std::string> mint(SessionLifecycleManager::LeaseId lease_id, absl::Duration ttl_hint);

  [[nodiscard]] absl::StatusOr<SessionLifecycleManager::LeaseId> resolve(const std::string& token) const;

  [[nodiscard]] absl::Status refresh(const std::string& token, absl::Duration ttl_hint);

  [[nodiscard]] bool erase(const std::string& token);

  void prune();

  [[nodiscard]] size_t size() const;

 private:
  struct Record {
    SessionLifecycleManager::LeaseId lease_id{0};
    absl::Time expires_at{absl::InfinitePast()};
  };

  [[nodiscard]] std::string mint_token_locked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const Options opts_;
  mutable absl::Mutex mu_;
  absl::BitGen bitgen_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, Record> tokens_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
