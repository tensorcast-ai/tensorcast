// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <condition_variable>
#include <mutex>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "daemon/ref_tracker.h"
#include "daemon/replica_session_manager.h"
#include "daemon/session_lifecycle.h"

using tensorcast::daemon::RefTracker;
using tensorcast::daemon::ReplicaSessionManager;
using tensorcast::daemon::SessionLifecycleManager;

TEST_CASE("Eviction notify fires when protections drop to zero", "[daemon][lifecycle][notify]") {
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  auto lip = std::make_unique<tensorcast::daemon::LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>());
  SessionLifecycleManager mgr(sessions, refs, *lip);

  // Register a notify callback
  std::mutex mu;
  std::condition_variable cv;
  bool notified = false;
  mgr.set_eviction_notify([&](const SessionLifecycleManager::ReplicaSubject& s) {
    std::lock_guard<std::mutex> g(mu);
    notified = true;
    cv.notify_one();
  });

  // Create a placement pin with short TTL to reach zero protections on expiry
  SessionLifecycleManager::ReplicaSubject subj{.artifact_id = "mi2:test:notify", .device_id = 0};
  auto pin_id_or = mgr.create_placement_lease(subj, absl::Milliseconds(20));
  REQUIRE(pin_id_or.ok());

  absl::SleepFor(absl::Milliseconds(30));
  mgr.expire_due(absl::Now());

  std::unique_lock<std::mutex> lk(mu);
  REQUIRE(cv.wait_for(lk, std::chrono::milliseconds(200), [&] { return notified; }));
}
