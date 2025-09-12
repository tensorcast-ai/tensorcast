// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <future>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/store/device_registry.h"
#include "daemon/ref_tracker.h"
#include "daemon/registration_manager.h"
#include "daemon/replica_session_manager.h"
#include "daemon/session_lifecycle.h"

using tensorcast::daemon::RefTracker;
using tensorcast::daemon::RegistrationManager;
using tensorcast::daemon::ReplicaSessionManager;
using tensorcast::daemon::SessionLifecycleManager;
using tensorcast::store::DeviceRegistry;
using tensorcast::store::loading::ReplicaKey;

TEST_CASE("Session guard expiry erases session entry", "[daemon][lifecycle][session]") {
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  auto lip = std::make_unique<tensorcast::daemon::LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>());
  SessionLifecycleManager mgr(sessions, refs, *lip);

  // Seed a session entry
  const std::string sid = "replica-uuid-123";
  ReplicaKey key{.artifact_id = "mi2:test:session", .device = DeviceRegistry::instance().gpu_key(0), .replica = 0};
  std::promise<absl::Status> p;
  p.set_value(absl::OkStatus());
  sessions.put(sid, key, p.get_future().share());
  REQUIRE(sessions.get(sid).has_value());

  // Create a short-lived session keepalive and let it expire
  REQUIRE(mgr.keepalive_session(sid, absl::Milliseconds(20)).ok());
  absl::SleepFor(absl::Milliseconds(30));
  mgr.expire_due(absl::Now());

  // Assert: session entry erased
  REQUIRE_FALSE(sessions.get(sid).has_value());
}
