// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "daemon/lip_manager.h"
#include "daemon/ref_tracker.h"
#include "daemon/registration_manager.h"
#include "daemon/replica_session_manager.h"
#include "daemon/session_lifecycle.h"

using tensorcast::daemon::LipManager;
using tensorcast::daemon::RefTracker;
using tensorcast::daemon::RegistrationManager;
using tensorcast::daemon::ReplicaSessionManager;
using tensorcast::daemon::SessionLifecycleManager;
using tensorcast::store::DeviceRegistry;
using tensorcast::store::loading::ReplicaKey;

TEST_CASE("TTL prefetch expiry drops placement pin", "[daemon][lifecycle][ttl]") {
  // Arrange
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  // LipManager is required by API but not used by this test path
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>());
  RegistrationManager reg;

  SessionLifecycleManager mgr(sessions, refs, *lip, reg);

  const std::string artifact_id = "mi2:test:ttl";
  const int device_id = 0;
  SessionLifecycleManager::ReplicaSubject subj{.artifact_id = artifact_id, .device_id = device_id};

  // Build a ReplicaKey to query counters
  ReplicaKey key{.artifact_id = artifact_id, .device = DeviceRegistry::instance().gpu_key(device_id), .replica = 0};

  // Act: create a placement lease with a very short TTL
  auto id_or = mgr.create_placement_lease(subj, absl::Milliseconds(20));
  REQUIRE(id_or.ok());
  REQUIRE(mgr.placement_pin_count_for(key) == 1);

  // Wait for TTL to elapse and trigger expiry processing
  absl::SleepFor(absl::Milliseconds(30));
  mgr.expire_due(absl::Now());

  // Assert: placement pin dropped
  REQUIRE(mgr.placement_pin_count_for(key) == 0);
}

TEST_CASE("PID exit precedence drops UseLease and RefTracker refs", "[daemon][lifecycle][pid]") {
  // Arrange
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>());
  RegistrationManager reg;
  SessionLifecycleManager mgr(sessions, refs, *lip, reg);

  const std::string artifact_id = "mi2:test:pid";
  const int device_id = 0;
  const int32_t pid = 123456;

  SessionLifecycleManager::ReplicaSubject subj{.artifact_id = artifact_id, .device_id = device_id};
  ReplicaKey key{.artifact_id = artifact_id, .device = DeviceRegistry::instance().gpu_key(device_id), .replica = 0};

  // Seed a UseLease and a RefTracker entry for this pid
  auto use_id_or = mgr.create_use_lease(subj, pid);
  REQUIRE(use_id_or.ok());
  refs.add_ref(key, pid, /*keep_for_global=*/false);

  REQUIRE(mgr.use_count_for(key) == 1);
  REQUIRE(refs.ref_count(key) == 1);

  // Act: simulate PID exit (event-driven path)
  mgr.handle_pid_exit(pid);

  // Assert: both lifecycle use counter and ref tracker are cleared
  REQUIRE(mgr.use_count_for(key) == 0);
  REQUIRE(refs.ref_count(key) == 0);
}

TEST_CASE("Mixed guards: UseLease + Deadline retires on whichever fails first", "[daemon][lifecycle][mixed]") {
  // Arrange
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>());
  RegistrationManager reg;
  SessionLifecycleManager mgr(sessions, refs, *lip, reg);

  const std::string artifact_id = "mi2:test:mixed_use_deadline";
  const int device_id = 0;
  const int32_t pid = 333777;
  SessionLifecycleManager::ReplicaSubject subj{.artifact_id = artifact_id, .device_id = device_id};
  ReplicaKey key{.artifact_id = artifact_id, .device = DeviceRegistry::instance().gpu_key(device_id), .replica = 0};

  auto id_or = mgr.create_use_lease(subj, pid);
  REQUIRE(id_or.ok());
  auto lease_id = *id_or;
  // Track ref to observe finalizer dropping it
  refs.add_ref(key, pid, /*keep_for_global=*/false);
  REQUIRE(mgr.use_count_for(key) == 1);
  REQUIRE(refs.ref_count(key) == 1);

  // Attach a short deadline guard to the same lease
  auto gid_or = mgr.add_deadline_guard_for_test(lease_id, absl::Milliseconds(20));
  REQUIRE(gid_or.ok());

  // Sleep past deadline; pid remains alive; any guard failing retires the lease
  absl::SleepFor(absl::Milliseconds(30));
  mgr.expire_due(absl::Now());

  REQUIRE(mgr.use_count_for(key) == 0);
  REQUIRE(refs.ref_count(key) == 0);
}

TEST_CASE(
    "Mixed guards: CommitLease + Deadline affects next_deadline and retires on expiry",
    "[daemon][lifecycle][mixed]") {
  // Arrange
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>());
  RegistrationManager reg;
  SessionLifecycleManager mgr(sessions, refs, *lip, reg);

  const std::string artifact_id = "mi2:test:mixed_commit_deadline";
  const int device_id = 0;
  SessionLifecycleManager::CommitSubject csubj{.artifact_id = artifact_id, .device_id = device_id};

  auto lid_or = mgr.create_commit_lease(csubj, /*pid=*/99991);
  REQUIRE(lid_or.ok());
  auto lease_id = *lid_or;

  // Adding a deadline guard should make next_deadline finite
  auto before = mgr.next_deadline();
  (void)before;
  auto gid_or = mgr.add_deadline_guard_for_test(lease_id, absl::Milliseconds(20));
  REQUIRE(gid_or.ok());
  auto after = mgr.next_deadline();
  REQUIRE(after != absl::InfiniteFuture());

  // Let it expire and sweep
  absl::SleepFor(absl::Milliseconds(30));
  mgr.expire_due(absl::Now());

  // With only a pid guard remaining (no time-based guards), the queue should be empty
  REQUIRE(mgr.next_deadline() == absl::InfiniteFuture());
}

TEST_CASE("Deadline guard generation: renew prevents stale expiry", "[daemon][lifecycle][guards]") {
  // Arrange
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>());
  RegistrationManager reg;
  SessionLifecycleManager mgr(sessions, refs, *lip, reg);

  const std::string artifact_id = "mi2:test:renew";
  const int device_id = 0;
  SessionLifecycleManager::ReplicaSubject subj{.artifact_id = artifact_id, .device_id = device_id};
  ReplicaKey key{.artifact_id = artifact_id, .device = DeviceRegistry::instance().gpu_key(device_id), .replica = 0};

  // Create a placement lease with short TTL
  auto id_or = mgr.create_placement_lease(subj, absl::Milliseconds(50));
  REQUIRE(id_or.ok());
  auto lease_id = *id_or;
  REQUIRE(mgr.placement_pin_count_for(key) == 1);

  // Immediately renew to extend TTL significantly
  auto st = mgr.renew_placement(lease_id, absl::Milliseconds(200));
  REQUIRE(st.ok());

  // Sleep past the old deadline but before the new one
  absl::SleepFor(absl::Milliseconds(80));
  mgr.expire_due(absl::Now());
  // The stale heap entry must be ignored and pin should still be present
  REQUIRE(mgr.placement_pin_count_for(key) == 1);

  // Sleep beyond the renewed deadline and process expirations
  absl::SleepFor(absl::Milliseconds(150));
  mgr.expire_due(absl::Now());
  REQUIRE(mgr.placement_pin_count_for(key) == 0);
}

TEST_CASE("Manual guard: placement lease requires explicit release", "[daemon][lifecycle][manual]") {
  // Arrange
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>());
  RegistrationManager reg;
  SessionLifecycleManager mgr(sessions, refs, *lip, reg);

  const std::string artifact_id = "mi2:test:manual";
  const int device_id = 0;
  SessionLifecycleManager::ReplicaSubject subj{.artifact_id = artifact_id, .device_id = device_id};
  ReplicaKey key{.artifact_id = artifact_id, .device = DeviceRegistry::instance().gpu_key(device_id), .replica = 0};

  // Create a manual placement lease (no TTL)
  auto id_or = mgr.create_placement_lease(subj, absl::ZeroDuration());
  REQUIRE(id_or.ok());
  auto lease_id = *id_or;
  REQUIRE(mgr.placement_pin_count_for(key) == 1);

  // Advance time and sweep: should not expire automatically
  absl::SleepFor(absl::Milliseconds(50));
  mgr.expire_due(absl::Now());
  REQUIRE(mgr.placement_pin_count_for(key) == 1);

  // Explicitly release
  mgr.release_lease(lease_id);
  REQUIRE(mgr.placement_pin_count_for(key) == 0);
}

TEST_CASE("Manual release of UseLease drops use_count and RefTracker", "[daemon][lifecycle][manual]") {
  // Arrange
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>());
  RegistrationManager reg;
  SessionLifecycleManager mgr(sessions, refs, *lip, reg);

  const std::string artifact_id = "mi2:test:manual_use";
  const int device_id = 0;
  const int32_t pid = 424242;
  SessionLifecycleManager::ReplicaSubject subj{.artifact_id = artifact_id, .device_id = device_id};
  ReplicaKey key{.artifact_id = artifact_id, .device = DeviceRegistry::instance().gpu_key(device_id), .replica = 0};

  auto id_or = mgr.create_use_lease(subj, pid);
  REQUIRE(id_or.ok());
  auto lease_id = *id_or;
  // Seed RefTracker to observe finalizer effect
  refs.add_ref(key, pid, /*keep_for_global=*/false);

  REQUIRE(mgr.use_count_for(key) == 1);
  REQUIRE(refs.ref_count(key) == 1);

  // Act: manual release
  mgr.release_lease(lease_id);

  // Assert: counters and ref tracker updated
  REQUIRE(mgr.use_count_for(key) == 0);
  REQUIRE(refs.ref_count(key) == 0);
}

TEST_CASE("Finalizer idempotency on UseLease manual retire", "[daemon][lifecycle][idempotent]") {
  // Arrange
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>());
  RegistrationManager reg;
  SessionLifecycleManager mgr(sessions, refs, *lip, reg);

  const std::string artifact_id = "mi2:test:idemp_use";
  const int device_id = 0;
  const int32_t pid = 515151;
  SessionLifecycleManager::ReplicaSubject subj{.artifact_id = artifact_id, .device_id = device_id};
  ReplicaKey key{.artifact_id = artifact_id, .device = DeviceRegistry::instance().gpu_key(device_id), .replica = 0};

  auto id_or = mgr.create_use_lease(subj, pid);
  REQUIRE(id_or.ok());
  auto lease_id = *id_or;
  refs.add_ref(key, pid, /*keep_for_global=*/false);
  REQUIRE(mgr.use_count_for(key) == 1);
  REQUIRE(refs.ref_count(key) == 1);

  // Act: release twice; finalizer should run once and counters should not underflow
  mgr.release_lease(lease_id);
  mgr.release_lease(lease_id);

  // Assert
  REQUIRE(mgr.use_count_for(key) == 0);
  REQUIRE(refs.ref_count(key) == 0);
}

TEST_CASE("Deadline expiry idempotency on PlacementLease", "[daemon][lifecycle][idempotent]") {
  // Arrange
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>());
  RegistrationManager reg;
  SessionLifecycleManager mgr(sessions, refs, *lip, reg);

  const std::string artifact_id = "mi2:test:idemp_deadline";
  const int device_id = 0;
  SessionLifecycleManager::ReplicaSubject subj{.artifact_id = artifact_id, .device_id = device_id};
  ReplicaKey key{.artifact_id = artifact_id, .device = DeviceRegistry::instance().gpu_key(device_id), .replica = 0};

  auto id_or = mgr.create_placement_lease(subj, absl::Milliseconds(20));
  REQUIRE(id_or.ok());
  REQUIRE(mgr.placement_pin_count_for(key) == 1);

  // Wait for expiry and process twice
  absl::SleepFor(absl::Milliseconds(30));
  mgr.expire_due(absl::Now());
  mgr.expire_due(absl::Now());

  // Assert: pin count is zero and did not underflow
  REQUIRE(mgr.placement_pin_count_for(key) == 0);
}

TEST_CASE("next_deadline is InfiniteFuture for manual-only leases", "[daemon][lifecycle][scheduling]") {
  // Arrange
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  auto lip = std::make_unique<LipManager>(std::shared_ptr<tensorcast::store::StoreEngine>());
  RegistrationManager reg;
  SessionLifecycleManager mgr(sessions, refs, *lip, reg);

  const std::string artifact_id = "mi2:test:no_deadlines";
  const int device_id = 0;
  SessionLifecycleManager::ReplicaSubject subj{.artifact_id = artifact_id, .device_id = device_id};

  auto id_or = mgr.create_placement_lease(subj, absl::ZeroDuration());
  REQUIRE(id_or.ok());
  // No time-based guards exist; next_deadline should be InfiniteFuture
  REQUIRE(mgr.next_deadline() == absl::InfiniteFuture());
}
