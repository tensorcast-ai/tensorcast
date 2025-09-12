// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "daemon/lip_manager.h"
#include "daemon/ref_tracker.h"
#include "daemon/registration_manager.h"
#include "daemon/replica_session_manager.h"
#include "daemon/session_lifecycle.h"

using tensorcast::DeviceType;
using tensorcast::daemon::LipManager;
using tensorcast::daemon::RefTracker;
using tensorcast::daemon::RegistrationManager;
using tensorcast::daemon::ReplicaSessionManager;
using tensorcast::daemon::SessionLifecycleManager;
using tensorcast::store::DeviceRegistry;
using tensorcast::store::StoreEngine;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::replica::MemoryState;

TEST_CASE("Immediate unload on last UseLease with no pins", "[daemon][lifecycle][unload]") {
  // Arrange: real StoreEngine (fake CUDA backend OK), allocate a small coalesced replica
  StoreEngineOptions opts;
  opts.storage_path = std::filesystem::temp_directory_path().string();
  opts.memory_pool_size = 16ULL * 1024 * 1024; // 16MB
  opts.chunk_size = 1ULL << 20; // 1MB
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);

  // Begin a coalesced registration and commit to create GPU residency
  StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = "test:immediate_unload";
  reg.tensor_index_data = std::string("{\"tensors\":[{\"name\":\"x\",\"offset\":0,\"length\":8}]}\n");
  reg.schema_version = "v2";
  reg.encoding = "json";
  reg.device_id = 0;
  reg.total_size_bytes = 8;

  auto begin_or = engine_ptr->begin_register_artifact(reg);
  REQUIRE(begin_or.ok());
  auto commit_or = engine_ptr->commit_registered_artifact(begin_or->registration_id);
  REQUIRE(commit_or.ok());
  const auto& committed = *commit_or;

  // Verify GPU residency is present before lifecycle actions
  ReplicaKey key{.artifact_id = committed.artifact_id, .device = DeviceRegistry::instance().gpu_key(0), .replica = 0};
  auto state_before = engine_ptr->get_replica_state(key, DeviceType::GPU);
  REQUIRE(state_before > MemoryState::UNALLOCATED);

  // Build lifecycle manager wired to engine for immediate reclaim
  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  LipManager lip(engine_ptr);
  RegistrationManager regmgr;
  SessionLifecycleManager mgr(sessions, refs, lip, *engine_ptr);

  // Create a UseLease for a dummy pid; no placement pins are created
  const int32_t pid = 98765;
  SessionLifecycleManager::ReplicaSubject subj{.artifact_id = committed.artifact_id, .device_id = 0};
  auto lid_or = mgr.create_use_lease(subj, pid);
  REQUIRE(lid_or.ok());

  // Act: retire the UseLease; finalizer should attempt immediate unload
  REQUIRE(mgr.release_use_lease(subj, pid).ok());

  // Assert: engine no longer reports GPU residency (unloaded)
  auto state_after = engine_ptr->get_replica_state(key, DeviceType::GPU);
  REQUIRE(state_after <= MemoryState::UNALLOCATED);
}
