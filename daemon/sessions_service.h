// Copyright (c) 2025, TensorCast Team.

// SessionsService: wraps ReplicaSessionManager + VerificationTracker wiring

#pragma once

#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "daemon/background_scheduler.h"
#include "daemon/replica_session_manager.h"
#include "daemon/session_lifecycle.h"
#include "daemon/verification_tracker.h"

namespace tensorcast::daemon {

class SessionsService {
 public:
  SessionsService(
      ReplicaSessionManager& s,
      VerificationTracker& v,
      BackgroundScheduler* sched,
      SessionLifecycleManager* lifecycle = nullptr,
      absl::Duration session_ttl = absl::Seconds(60))
      : sessions_(s), verif_(v), sched_(sched), lifecycle_(lifecycle), session_ttl_(session_ttl) {}

  void put_with_verification(
      const std::string& replica_uuid,
      const store::loading::ReplicaKey& key,
      std::shared_future<absl::Status> ready) {
    sessions_.put(replica_uuid, key, ready);
    verif_.initiate(replica_uuid, ready);
    if (sched_)
      sched_->notify(TaskKind::kVerification);
    if (lifecycle_) {
      // Create or renew session principal keepalive under the unified lifecycle
      (void)lifecycle_->keepalive_session(replica_uuid, session_ttl_);
    }
  }

  [[nodiscard]] std::optional<SessionEntry> get(const std::string& replica_uuid) const {
    return sessions_.get(replica_uuid);
  }

  bool erase(const std::string& replica_uuid) {
    return sessions_.erase(replica_uuid);
  }

  void update_verification_status(const std::string& uuid, v1::VerificationStatus st, std::string err = "") {
    verif_.update(uuid, st, std::move(err));
  }

  [[nodiscard]] std::optional<std::pair<v1::VerificationStatus, std::string>> get_known(absl::string_view uuid) const {
    return verif_.get_known(std::string(uuid));
  }

 private:
  ReplicaSessionManager& sessions_;
  VerificationTracker& verif_;
  BackgroundScheduler* sched_;
  SessionLifecycleManager* lifecycle_;
  absl::Duration session_ttl_;
};

} // namespace tensorcast::daemon
