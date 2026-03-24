// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/replica_session_manager.h"

#include <catch2/catch_test_macros.hpp>
#include <chrono>

#include "core/store/device_types.h"

namespace {

using tensorcast::DeviceType;
using tensorcast::daemon::ReplicaSessionManager;
using tensorcast::store::DeviceKey;
using tensorcast::store::loading::ReplicaKey;

ReplicaKey make_key(std::string artifact_id, std::string view_id = "", int ordinal = 0) {
  ReplicaKey k;
  k.artifact_id = std::move(artifact_id);
  if (!view_id.empty()) {
    k.view_id = std::move(view_id);
  }
  k.device = DeviceKey{.type = DeviceType::GPU, .ordinal = ordinal, .uuid = "gpu-fake-0"};
  k.replica = 0;
  return k;
}

} // namespace

TEST_CASE("ReplicaSessionManager PutIfAbsent/JoinIfMatch semantics", "[daemon][sessions]") {
  ReplicaSessionManager mgr(std::chrono::seconds(1));

  const ReplicaKey key_a = make_key("mi2:a");
  const ReplicaKey key_b = make_key("mi2:b");

  auto inserted_or = mgr.put_if_absent_or_join("op-1", key_a, nullptr);
  REQUIRE(inserted_or.ok());
  REQUIRE(*inserted_or == ReplicaSessionManager::PutResult::kInserted);

  auto joined_or = mgr.put_if_absent_or_join("op-1", key_a, nullptr);
  REQUIRE(joined_or.ok());
  REQUIRE(*joined_or == ReplicaSessionManager::PutResult::kJoined);

  auto mismatch_or = mgr.put_if_absent_or_join("op-1", key_b, nullptr);
  REQUIRE_FALSE(mismatch_or.ok());
  REQUIRE(mismatch_or.status().code() == absl::StatusCode::kFailedPrecondition);
}

TEST_CASE("ReplicaSessionManager permits reuse after expiry", "[daemon][sessions]") {
  ReplicaSessionManager mgr(std::chrono::seconds(0));

  const ReplicaKey key_a = make_key("mi2:a");
  const ReplicaKey key_b = make_key("mi2:b");

  auto inserted_or = mgr.put_if_absent_or_join("op-1", key_a, nullptr);
  REQUIRE(inserted_or.ok());
  REQUIRE(*inserted_or == ReplicaSessionManager::PutResult::kInserted);

  REQUIRE(mgr.remove_if_expired("op-1"));

  auto reused_or = mgr.put_if_absent_or_join("op-1", key_b, nullptr);
  REQUIRE(reused_or.ok());
  REQUIRE(*reused_or == ReplicaSessionManager::PutResult::kInserted);
}

TEST_CASE(
    "ReplicaSessionManager joins equivalent GPU keys when only uuid normalization differs",
    "[daemon][sessions]") {
  ReplicaSessionManager mgr(std::chrono::seconds(1));

  ReplicaKey existing = make_key("mi2:a", "view-1", 5);
  existing.device.uuid.clear();

  ReplicaKey requested = make_key("mi2:a", "view-1", 5);
  requested.device.uuid = "gpu-real-5";

  auto inserted_or = mgr.put_if_absent_or_join("op-1", existing, nullptr);
  REQUIRE(inserted_or.ok());
  REQUIRE(*inserted_or == ReplicaSessionManager::PutResult::kInserted);

  auto joined_or = mgr.put_if_absent_or_join("op-1", requested, nullptr);
  REQUIRE(joined_or.ok());
  REQUIRE(*joined_or == ReplicaSessionManager::PutResult::kJoined);

  auto stored = mgr.get("op-1");
  REQUIRE(stored.has_value());
  REQUIRE(stored->key.device.uuid == "gpu-real-5");
}
