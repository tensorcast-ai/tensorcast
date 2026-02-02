// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include "core/store/device_registry.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/pid_monitor.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/replica_session_manager.h"
#include "daemon/state/session_lifecycle.h"

using tensorcast::daemon::PidMonitor;
using tensorcast::daemon::RefTracker;
using tensorcast::daemon::ReplicaSessionManager;
using tensorcast::daemon::SessionLifecycleManager;
using tensorcast::store::DeviceRegistry;
using tensorcast::store::loading::ReplicaKey;

TEST_CASE("PidMonitor unwatch called on last guard retire", "[daemon][lifecycle][pid]") {
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  tensorcast::daemon::IpcRegionRegistry regions(tensorcast::daemon::IpcRegionRegistry::Options{});
  auto lip =
      std::make_unique<tensorcast::daemon::LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>(), &regions);
  SessionLifecycleManager mgr(sessions, refs, *lip);

  // Attach a monitor and verify watch/unwatch transitions
  bool exit_called = false;
  PidMonitor mon([&](pid_t) { exit_called = true; });
  mgr.attach_pid_monitor(&mon);

  const int32_t pid = 556677;
  ReplicaKey subj{.artifact_id = "mi2:test:pm", .device = DeviceRegistry::instance().gpu_key(0), .replica = 0};

  auto id_or = mgr.create_use_lease(subj, pid);
  REQUIRE(id_or.ok());
  REQUIRE(mon.is_watching_for_test(pid));

  // Act: retire the only UseLease
  REQUIRE(mgr.release_use_lease(subj, pid).ok());

  REQUIRE_FALSE(mon.is_watching_for_test(pid));
}
