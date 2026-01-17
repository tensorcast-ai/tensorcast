// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "daemon/ref_tracker.h"
#include "daemon/registration_manager.h"
#include "daemon/replica_session_manager.h"
#include "daemon/session_lifecycle.h"
#include "daemon/sweep_tasks.h"

using tensorcast::DeviceType;
using tensorcast::daemon::EvictionTask;
using tensorcast::daemon::RefTracker;
using tensorcast::daemon::RegistrationManager;
using tensorcast::daemon::ReplicaSessionManager;
using tensorcast::daemon::SessionLifecycleManager;
using tensorcast::store::DeviceRegistry;
using tensorcast::store::StoreEngine;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::replica::MemoryState;

static StoreEngineOptions make_engine_opts() {
  StoreEngineOptions opts;
  opts.storage_path = std::filesystem::temp_directory_path().string();
  opts.memory_pool_size = 32ULL * 1024 * 1024; // 32MB
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  return opts;
}

static ReplicaKey commit_small_gpu_replica(StoreEngine& engine, const std::string& artifact_id) {
  StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = artifact_id;
  reg.tensor_index_data = std::string("{\"tensors\":[{\"name\":\"x\",\"offset\":0,\"length\":8}]}\n");
  reg.schema_version = "v3";
  reg.encoding = "json";
  reg.device_id = 0;
  reg.total_size_bytes = 8;
  auto begin_or = engine.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());
  auto commit_or = engine.commit_registered_artifact(begin_or->registration_id);
  REQUIRE(commit_or.ok());
  return {.artifact_id = commit_or->artifact_id, .device = DeviceRegistry::instance().gpu_key(0), .replica = 0};
}

TEST_CASE("EvictionTask skips when UseLease is active", "[daemon][eviction]") {
  auto engine = std::make_shared<StoreEngine>(make_engine_opts());
  ReplicaKey key = commit_small_gpu_replica(*engine, "mi2:evict:use");
  auto before = engine->get_replica_state(key, DeviceType::GPU);

  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  tensorcast::daemon::LipManager lip(engine, nullptr);
  // lifecycle without immediate reclaim
  SessionLifecycleManager mgr(sessions, refs, lip);
  // Create UseLease
  auto use_id = mgr.create_use_lease(key, /*pid=*/1111);
  REQUIRE(use_id.ok());

  EvictionTask task(*engine, refs, &mgr, /*limit=*/0.0);
  task.run_once();

  // Should not be fully removed from registry
  auto after = engine->get_replica_state(key, DeviceType::GPU);
  REQUIRE(after != MemoryState::UNINITIALIZED);
}

TEST_CASE("EvictionTask skips when Placement pin is active", "[daemon][eviction]") {
  auto engine = std::make_shared<StoreEngine>(make_engine_opts());
  ReplicaKey key = commit_small_gpu_replica(*engine, "mi2:evict:pin");
  auto before2 = engine->get_replica_state(key, DeviceType::GPU);

  ReplicaSessionManager sessions(std::chrono::seconds(60));
  RefTracker refs;
  tensorcast::daemon::LipManager lip(engine, nullptr);
  SessionLifecycleManager mgr(sessions, refs, lip);
  auto pin_id = mgr.create_placement_lease(key, absl::Minutes(10));
  REQUIRE(pin_id.ok());

  EvictionTask task(*engine, refs, &mgr, /*limit=*/0.0);
  task.run_once();

  auto after2 = engine->get_replica_state(key, DeviceType::GPU);
  REQUIRE(after2 != MemoryState::UNINITIALIZED);
}

TEST_CASE("EvictionTask evicts when no UseLease or pin", "[daemon][eviction]") {
  auto engine = std::make_shared<StoreEngine>(make_engine_opts());
  ReplicaKey key = commit_small_gpu_replica(*engine, "mi2:evict:none");
  auto before3 = engine->get_replica_state(key, DeviceType::GPU);
  if (before3 <= MemoryState::UNALLOCATED) {
    WARN("GPU residency not allocated in this environment; skipping eviction assertion.");
    return;
  }

  RefTracker refs;
  EvictionTask task(*engine, refs, /*lifecycle=*/nullptr, /*limit=*/0.0);
  task.run_once();

  REQUIRE(engine->get_replica_state(key, DeviceType::GPU) <= MemoryState::UNALLOCATED);
}
