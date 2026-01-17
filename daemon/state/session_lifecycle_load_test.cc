// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/replica_session_manager.h"
#include "daemon/state/session_lifecycle.h"

using tensorcast::daemon::LipManager;
using tensorcast::daemon::RefTracker;
using tensorcast::daemon::ReplicaSessionManager;
using tensorcast::daemon::SessionLifecycleManager;
using tensorcast::store::DeviceRegistry;
using tensorcast::store::loading::ReplicaKey;

TEST_CASE("Lifecycle load: bulk placement TTL expiry under sweep", "[daemon][lifecycle][load]") {
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  tensorcast::daemon::IpcRegionRegistry regions(tensorcast::daemon::IpcRegionRegistry::Options{});
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>(), &regions);
  SessionLifecycleManager mgr(sessions, refs, *lip);

  const int kDevice = 0;
  const int N = 2000; // keep runtime modest while exercising data structures

  std::vector<ReplicaKey> keys;
  keys.reserve(N);
  for (int i = 0; i < N; ++i) {
    const std::string art = "mi2:load:pin:" + std::to_string(i);
    ReplicaKey key{.artifact_id = art, .device = DeviceRegistry::instance().gpu_key(kDevice), .replica = 0};
    auto id_or = mgr.create_placement_lease(key, absl::Milliseconds(50));
    REQUIRE(id_or.ok());
    keys.push_back(key);
  }

  // Let TTLs elapse and process all due expirations in one pass
  absl::SleepFor(absl::Milliseconds(80));
  mgr.expire_due(absl::Now() + absl::Seconds(1));

  for (const auto& key : keys) {
    REQUIRE(mgr.placement_pin_count_for(key) == 0);
  }
}

TEST_CASE("Lifecycle load: bulk UseLease retirements via PID exit", "[daemon][lifecycle][load]") {
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  tensorcast::daemon::IpcRegionRegistry regions(tensorcast::daemon::IpcRegionRegistry::Options{});
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>(), &regions);
  SessionLifecycleManager mgr(sessions, refs, *lip);

  const int kDevice = 0;
  const int N = 2000;

  std::vector<std::pair<ReplicaKey, int32_t>> items;
  items.reserve(N);
  for (int i = 0; i < N; ++i) {
    const std::string art = "mi2:load:use:" + std::to_string(i);
    const int32_t pid = 100000 + i;
    ReplicaKey key{.artifact_id = art, .device = DeviceRegistry::instance().gpu_key(kDevice), .replica = 0};
    auto id_or = mgr.create_use_lease(key, pid);
    REQUIRE(id_or.ok());
    refs.add_ref(key, pid);
    items.emplace_back(key, pid);
  }

  // Simulate exits; retire associated UseLeases quickly
  for (const auto& it : items) {
    mgr.handle_pid_exit(it.second);
  }

  for (const auto& it : items) {
    REQUIRE(mgr.use_count_for(it.first) == 0);
    REQUIRE(refs.ref_count(it.first) == 0);
  }
}

TEST_CASE("Lifecycle load: mass renewal prevents stale expiries", "[daemon][lifecycle][load]") {
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  tensorcast::daemon::IpcRegionRegistry regions(tensorcast::daemon::IpcRegionRegistry::Options{});
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>(), &regions);
  SessionLifecycleManager mgr(sessions, refs, *lip);

  const int kDevice = 0;
  const int N = 1000;
  std::vector<std::pair<SessionLifecycleManager::LeaseId, ReplicaKey>> leases;
  leases.reserve(N);
  for (int i = 0; i < N; ++i) {
    const std::string art = "mi2:load:renew:" + std::to_string(i);
    ReplicaKey key{.artifact_id = art, .device = DeviceRegistry::instance().gpu_key(kDevice), .replica = 0};
    auto id_or = mgr.create_placement_lease(key, absl::Milliseconds(30));
    REQUIRE(id_or.ok());
    auto id = *id_or;
    // Extend TTL significantly in a subsequent pass to create stale heap entries
    REQUIRE(mgr.renew_placement(id, absl::Milliseconds(200)).ok());
    leases.emplace_back(id, key);
  }

  // Past old deadlines, before renewed ones
  absl::SleepFor(absl::Milliseconds(60));
  mgr.expire_due(absl::Now());
  for (const auto& p : leases) {
    REQUIRE(mgr.placement_pin_count_for(p.second) == 1);
  }

  // After renewed deadlines
  absl::SleepFor(absl::Milliseconds(180));
  mgr.expire_due(absl::Now() + absl::Seconds(1));
  for (const auto& p : leases) {
    REQUIRE(mgr.placement_pin_count_for(p.second) == 0);
  }
}
