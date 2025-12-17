// Copyright (c) 2025, TensorCast Team.

// VerificationTracker: manages verification status registry and completion queue
// for replica materialization readiness futures. Applies TTL and capacity
// limits to avoid unbounded growth. Extracted to slim grpc_service_impl.

#pragma once

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "core/common/ready_signal.h"
#include "folly/Executor.h"
#include "folly/futures/Future.h"
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

  void set_serial_executor(folly::Executor::KeepAlive<> executor) {
    serial_executor_ = std::move(executor);
  }

  // Initialize tracking for a replica_uuid: mark IN_PROGRESS and subscribe to
  // readiness completion. Completion updates are executed on the serial
  // executor so registry progression is completion-driven and non-polling.
  void initiate(const std::string& uuid, std::shared_ptr<tensorcast::common::ReadySignal<absl::Status>> ready_signal) {
    set_status(uuid, v1::VerificationStatus::VERIFICATION_STATUS_IN_PROGRESS, "");
    if (!ready_signal) {
      set_status(uuid, v1::VerificationStatus::VERIFICATION_STATUS_PASSED, "");
      return;
    }
    ABSL_CHECK(serial_executor_) << "VerificationTracker serial executor must be set before initiate()";

    std::string uuid_copy = uuid;
    auto executor = serial_executor_.copy();
    (void)std::move(ready_signal->subscribe())
        .via(std::move(executor))
        .thenValue([this, uuid_copy = std::move(uuid_copy)](absl::Status st) {
          if (st.ok()) {
            set_status(uuid_copy, v1::VerificationStatus::VERIFICATION_STATUS_PASSED, "");
          } else {
            set_status(uuid_copy, v1::VerificationStatus::VERIFICATION_STATUS_FAILED, std::string(st.message()));
          }
          return st;
        })
        .thenError(folly::tag_t<std::exception>{}, [this, uuid_copy](const std::exception& ex) {
          set_status(uuid_copy, v1::VerificationStatus::VERIFICATION_STATUS_FAILED, ex.what());
          return absl::InternalError(ex.what());
        });
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
  folly::Executor::KeepAlive<> serial_executor_;
};

} // namespace tensorcast::daemon
