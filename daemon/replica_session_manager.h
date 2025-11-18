// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "core/store/materialization/contracts/loading_spec.h"

namespace tensorcast::daemon {

struct SessionEntry {
  store::loading::ReplicaKey key;
  std::shared_future<absl::Status> ready;
  std::chrono::steady_clock::time_point expiry;
};

class ReplicaSessionManager {
 public:
  explicit ReplicaSessionManager(std::chrono::seconds ttl) : ttl_(ttl) {}

  void put(
      const std::string& replica_uuid,
      const store::loading::ReplicaKey& key,
      std::shared_future<absl::Status> ready) {
    absl::MutexLock l(&mu_);
    sessions_[replica_uuid] = SessionEntry{key, std::move(ready), now() + ttl_};
  }

  std::optional<SessionEntry> get(const std::string& replica_uuid) {
    absl::MutexLock l(&mu_);
    auto it = sessions_.find(replica_uuid);
    if (it == sessions_.end()) {
      return std::nullopt;
    }
    if (expired(it->second)) {
      sessions_.erase(it);
      return std::nullopt;
    }
    return it->second;
  }

  bool erase(const std::string& replica_uuid) {
    absl::MutexLock l(&mu_);
    return sessions_.erase(replica_uuid) > 0;
  }

  // Enumerate keys for periodic sweeping
  std::vector<std::string> keys() {
    absl::MutexLock l(&mu_);
    std::vector<std::string> out;
    out.reserve(sessions_.size());
    for (const auto& kv : sessions_) {
      out.push_back(kv.first);
    }
    return out;
  }

  // Remove entry if expired; returns true if removed
  bool remove_if_expired(const std::string& key) {
    absl::MutexLock l(&mu_);
    auto it = sessions_.find(key);
    if (it == sessions_.end()) {
      return false;
    }
    if (expired(it->second)) {
      sessions_.erase(it);
      return true;
    }
    return false;
  }

 private:
  static std::chrono::steady_clock::time_point now() {
    return std::chrono::steady_clock::now();
  }

  static bool expired(const SessionEntry& e) {
    return now() >= e.expiry;
  }

  absl::Mutex mu_;
  absl::flat_hash_map<std::string, SessionEntry> sessions_ ABSL_GUARDED_BY(mu_);
  std::chrono::seconds ttl_;
};

} // namespace tensorcast::daemon
