// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/common/ready_signal.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "folly/futures/Future.h"

namespace tensorcast::daemon {

struct SessionEntry {
  store::loading::ReplicaKey key;
  std::shared_ptr<tensorcast::common::ReadySignal<absl::Status>> ready_signal;
  std::chrono::steady_clock::time_point expiry;

  [[nodiscard]] folly::SemiFuture<absl::Status> subscribe_ready() const {
    if (!ready_signal) {
      return folly::makeSemiFuture<absl::Status>(absl::OkStatus());
    }
    return ready_signal->subscribe();
  }

  [[nodiscard]] absl::Status wait_ready(std::chrono::milliseconds timeout) const {
    if (!ready_signal) {
      return absl::OkStatus();
    }
    try {
      if (timeout.count() > 0) {
        return std::move(subscribe_ready()).get(timeout);
      }
      return std::move(subscribe_ready()).get();
    } catch (const folly::FutureTimeout&) {
      return absl::DeadlineExceededError("replica did not reach ready state before timeout");
    } catch (const std::exception& ex) {
      return absl::InternalError(ex.what());
    } catch (...) {
      return absl::InternalError("replica wait_ready failed with unknown exception");
    }
  }
};

class ReplicaSessionManager {
 public:
  enum class PutResult : uint8_t { kInserted, kJoined };

  explicit ReplicaSessionManager(std::chrono::seconds ttl) : ttl_(ttl) {}

  [[nodiscard]] absl::StatusOr<PutResult> put_if_absent_or_join(
      const std::string& replica_uuid,
      const store::loading::ReplicaKey& key,
      std::shared_ptr<tensorcast::common::ReadySignal<absl::Status>> ready_signal) {
    absl::MutexLock l(&mu_);
    auto it = sessions_.find(replica_uuid);
    if (it != sessions_.end()) {
      if (expired(it->second)) {
        sessions_.erase(it);
      } else {
        if (it->second.key != key) {
          return absl::FailedPreconditionError("replica_uuid already exists with a different ReplicaKey");
        }
        it->second.expiry = now() + ttl_;
        return PutResult::kJoined;
      }
    }
    sessions_.emplace(replica_uuid, SessionEntry{key, std::move(ready_signal), now() + ttl_});
    return PutResult::kInserted;
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
