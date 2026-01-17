// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include "core/store/device_registry.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/registration_manager.h"
#include "daemon/state/replica_session_manager.h"
#include "daemon/state/session_lifecycle.h"

using tensorcast::daemon::LipManager;
using tensorcast::daemon::RefTracker;
using tensorcast::daemon::RegistrationManager;
using tensorcast::daemon::ReplicaSessionManager;
using tensorcast::daemon::SessionLifecycleManager;
using tensorcast::store::DeviceRegistry;
using tensorcast::store::loading::ReplicaKey;

TEST_CASE("PID guards removed on last UseLease retire", "[daemon][lifecycle][pid]") {
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  tensorcast::daemon::IpcRegionRegistry regions(tensorcast::daemon::IpcRegionRegistry::Options{});
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>(), &regions);
  RegistrationManager reg;
  SessionLifecycleManager mgr(sessions, refs, *lip);

  const int device_id = 0;
  const int32_t pid = 222333;
  ReplicaKey subj{
      .artifact_id = "mi2:test:pidunwatch",
      .device = DeviceRegistry::instance().gpu_key(device_id),
      .replica = 0,
  };

  auto id_or = mgr.create_use_lease(subj, pid);
  REQUIRE(id_or.ok());
  REQUIRE(mgr.has_pid_guard_for_test(pid));

  // Act: release the only UseLease for this pid
  REQUIRE(mgr.release_use_lease(subj, pid).ok());

  // Assert: manager no longer tracks pid guards (PidMonitor unwatch is called internally if attached)
  REQUIRE_FALSE(mgr.has_pid_guard_for_test(pid));
}
