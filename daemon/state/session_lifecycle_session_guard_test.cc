// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/ready_signal.h"
#include "core/store/device_registry.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/registration_manager.h"
#include "daemon/state/replica_session_manager.h"
#include "daemon/state/session_lifecycle.h"

using tensorcast::daemon::RefTracker;
using tensorcast::daemon::RegistrationManager;
using tensorcast::daemon::ReplicaSessionManager;
using tensorcast::daemon::SessionLifecycleManager;
using tensorcast::store::DeviceRegistry;
using tensorcast::store::loading::ReplicaKey;

TEST_CASE("Session guard expiry erases session entry", "[daemon][lifecycle][session]") {
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  tensorcast::daemon::IpcRegionRegistry regions(tensorcast::daemon::IpcRegionRegistry::Options{});
  auto lip =
      std::make_unique<tensorcast::daemon::LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>(), &regions);
  SessionLifecycleManager mgr(sessions, refs, *lip);

  // Seed a session entry
  const std::string sid = "replica-uuid-123";
  ReplicaKey key{.artifact_id = "mi2:test:session", .device = DeviceRegistry::instance().gpu_key(0), .replica = 0};
  auto ready = std::make_shared<tensorcast::common::ReadySignal<absl::Status>>();
  ready->set_value(absl::OkStatus());
  REQUIRE(sessions.put_if_absent_or_join(sid, key, ready).ok());
  REQUIRE(sessions.get(sid).has_value());

  // Create a short-lived session keepalive and let it expire
  REQUIRE(mgr.keepalive_session(sid, absl::Milliseconds(20)).ok());
  absl::SleepFor(absl::Milliseconds(30));
  mgr.expire_due(absl::Now());

  // Assert: session entry erased
  REQUIRE_FALSE(sessions.get(sid).has_value());
}
