// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/transport_lock_manager.h"

#include <catch2/catch_test_macros.hpp>

using tensorcast::DeviceType;
using tensorcast::daemon::TransportLockManager;
using tensorcast::store::DeviceKey;
using tensorcast::store::loading::ReplicaKey;

static ReplicaKey make_key(const std::string& id) {
  return ReplicaKey{
      .artifact_id = id, .device = DeviceKey{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""}, .replica = 0};
}

TEST_CASE("TransportLockManager token lifecycle", "[daemon][locks]") {
  TransportLockManager mgr(std::chrono::seconds(60));
  auto token = mgr.mint_token();
  REQUIRE(token.size() == 32);
  mgr.put(token, make_key("a1"), {1, 2, 3});

  auto entry = mgr.get(token);
  REQUIRE(entry.has_value());
  const auto& entry_ref = *entry; // NOLINT(bugprone-unchecked-optional-access)
  REQUIRE(entry_ref.chunk_indices.size() == 3);

  REQUIRE(mgr.erase(token));
  REQUIRE_FALSE(mgr.get(token).has_value());
}
