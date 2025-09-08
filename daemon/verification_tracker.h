// Copyright (c) 2025, TensorCast Team.

// VerificationTracker: manages verification status registry and completion queue
// for replica materialization readiness futures. Applies TTL and capacity
// limits to avoid unbounded growth. Extracted to slim grpc_service_impl.

#pragma once

#include <algorithm>
#include <chrono>
#include <deque>
#include <future>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grpcpp/grpcpp.h"
#include "tensorcast/daemon/v1/store_daemon.pb.h"

namespace tensorcast::daemon {

class VerificationTracker {
 public:
  struct Options {
    std::chrono::minutes ttl{std::chrono::minutes(5)};
    size_t max_entries{4096};
  };

  VerificationTracker() = default;
  explicit VerificationTracker(Options opts) : opts_(opts) {}

  // Initialize tracking for a replica_uuid: mark IN_PROGRESS and enqueue a
  // readiness future to be drained by background tasks.
  void initiate(const std::string& uuid, std::shared_future<absl::Status> ready) {
    set_status(uuid, v1::VerificationStatus::VERIFICATION_STATUS_IN_PROGRESS, "");
    absl::MutexLock l(&mu_);
    tasks_.push_back(VerifTask{.uuid = uuid, .ready = std::move(ready)});
  }

  // Return terminal status if known; otherwise std::nullopt.
  std::optional<std::pair<v1::VerificationStatus, std::string>> get_known(const std::string& uuid) const {
    absl::MutexLock l(&mu_);
    auto it = reg_.find(uuid);
    if (it == reg_.end())
      return std::nullopt;
    const auto& e = it->second;
    if (e.status == v1::VerificationStatus::VERIFICATION_STATUS_PASSED ||
        e.status == v1::VerificationStatus::VERIFICATION_STATUS_FAILED) {
      return std::make_pair(e.status, e.err);
    }
    return std::nullopt;
  }

  // Update status explicitly (e.g., synchronous wait path). Refreshes TTL.
  void update(const std::string& uuid, v1::VerificationStatus st, std::string err = "") {
    set_status(uuid, st, std::move(err));
  }

  // Non-blocking: move ready tasks and update registry to PASSED/FAILED.
  void drain_ready_and_update() {
    using namespace std::chrono_literals;
    std::vector<std::pair<std::string, absl::Status>> done;
    {
      absl::MutexLock l(&mu_);
      for (auto it = tasks_.begin(); it != tasks_.end();) {
        if (it->ready.wait_for(0ms) == std::future_status::ready) {
          done.emplace_back(it->uuid, it->ready.get());
          it = tasks_.erase(it);
        } else {
          ++it;
        }
      }
    }
    for (auto& p : done) {
      const auto& uuid = p.first;
      const auto& st = p.second;
      if (st.ok())
        set_status(uuid, v1::VerificationStatus::VERIFICATION_STATUS_PASSED, "");
      else
        set_status(uuid, v1::VerificationStatus::VERIFICATION_STATUS_FAILED, std::string(st.message()));
    }
    prune();
  }

  // Enforce TTL and capacity limits.
  void prune() {
    const auto now = std::chrono::steady_clock::now();
    {
      absl::MutexLock l(&mu_);
      for (auto it = reg_.begin(); it != reg_.end();) {
        if (it->second.expire_at < now) {
          auto to_erase = it++;
          reg_.erase(to_erase);
        } else {
          ++it;
        }
      }
      if (reg_.size() <= opts_.max_entries)
        return;
      // Drop oldest by updated_at until under cap
      std::vector<std::pair<std::string, time_point>> order;
      order.reserve(reg_.size());
      for (const auto& kv : reg_)
        order.emplace_back(kv.first, kv.second.updated_at);
      std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
      size_t to_drop = reg_.size() - opts_.max_entries;
      for (size_t i = 0; i < to_drop; ++i)
        reg_.erase(order[i].first);
    }
  }

 private:
  using time_point = std::chrono::time_point<std::chrono::steady_clock>;
  struct Entry {
    v1::VerificationStatus status{v1::VerificationStatus::VERIFICATION_STATUS_IN_PROGRESS};
    std::string err;
    time_point updated_at;
    time_point expire_at;
  };
  struct VerifTask {
    std::string uuid;
    std::shared_future<absl::Status> ready;
  };

  void set_status(const std::string& uuid, v1::VerificationStatus st, std::string err) {
    absl::MutexLock l(&mu_);
    const auto now = std::chrono::steady_clock::now();
    Entry e;
    e.status = st;
    e.err = std::move(err);
    e.updated_at = now;
    e.expire_at = now + opts_.ttl;
    reg_[uuid] = std::move(e);
  }

  mutable absl::Mutex mu_;
  Options opts_;
  absl::flat_hash_map<std::string, Entry> reg_ ABSL_GUARDED_BY(mu_);
  std::deque<VerifTask> tasks_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
