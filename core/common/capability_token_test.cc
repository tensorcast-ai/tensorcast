// Copyright (c) 2026, TensorCast Team.

#include "core/common/capability_token.h"

#include <catch2/catch_test_macros.hpp>

#include "absl/time/time.h"

namespace tensorcast::common {
namespace {

TEST_CASE("CapabilityTokenManager mints and verifies tokens", "[common][capability_token]") {
  CapabilityTokenConfig cfg{
      .active = CapabilityTokenKey{.version = 1, .secret = "secret_v1"},
      .previous = {},
  };
  CapabilityTokenManager mgr(cfg);
  REQUIRE(mgr.configured());

  tensorcast::common::v1::PlacementLeaseScope scope;
  scope.set_lease_id(42);
  auto scope_or = CapabilityTokenManager::serialize_scope_deterministic(scope);
  REQUIRE(scope_or.ok());

  const absl::Time now = absl::Now();
  const uint64_t expires_at_ms = static_cast<uint64_t>(absl::ToUnixMillis(now + absl::Seconds(10)));
  auto token_or =
      mgr.mint("daemon-1", tensorcast::common::v1::CAPABILITY_AUDIENCE_PLACEMENT_LEASE, *scope_or, expires_at_ms);
  REQUIRE(token_or.ok());

  auto env_or = mgr.verify(
      *token_or,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PLACEMENT_LEASE,
      "daemon-1",
      now,
      /*require_not_expired=*/true);
  REQUIRE(env_or.ok());
  REQUIRE(env_or->token_version() == 1);
  REQUIRE(env_or->issuer_daemon_id() == "daemon-1");
}

TEST_CASE("CapabilityTokenManager accepts previous keys", "[common][capability_token]") {
  CapabilityTokenConfig cfg_v1{
      .active = CapabilityTokenKey{.version = 1, .secret = "old_secret"},
      .previous = {},
  };
  CapabilityTokenManager old_mgr(cfg_v1);

  tensorcast::common::v1::RetentionHandleScope scope;
  scope.set_handle_id("handle-abc");
  auto scope_or = CapabilityTokenManager::serialize_scope_deterministic(scope);
  REQUIRE(scope_or.ok());

  const absl::Time now = absl::Now();
  const uint64_t expires_at_ms = static_cast<uint64_t>(absl::ToUnixMillis(now + absl::Seconds(5)));
  auto token_or =
      old_mgr.mint("daemon-2", tensorcast::common::v1::CAPABILITY_AUDIENCE_RETENTION_HANDLE, *scope_or, expires_at_ms);
  REQUIRE(token_or.ok());

  CapabilityTokenConfig cfg_v2{
      .active = CapabilityTokenKey{.version = 2, .secret = "new_secret"},
      .previous = {CapabilityTokenKey{.version = 1, .secret = "old_secret"}},
  };
  CapabilityTokenManager new_mgr(cfg_v2);
  auto env_or = new_mgr.verify(
      *token_or,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_RETENTION_HANDLE,
      "daemon-2",
      now,
      /*require_not_expired=*/true);
  REQUIRE(env_or.ok());
  REQUIRE(env_or->token_version() == 1);
}

TEST_CASE("CapabilityTokenManager enforces expiry", "[common][capability_token]") {
  CapabilityTokenConfig cfg{
      .active = CapabilityTokenKey{.version = 3, .secret = "secret_v3"},
      .previous = {},
  };
  CapabilityTokenManager mgr(cfg);

  tensorcast::common::v1::PlacementLeaseScope scope;
  scope.set_lease_id(7);
  auto scope_or = CapabilityTokenManager::serialize_scope_deterministic(scope);
  REQUIRE(scope_or.ok());

  const absl::Time now = absl::Now();
  const uint64_t expires_at_ms = static_cast<uint64_t>(absl::ToUnixMillis(now + absl::Seconds(1)));
  auto token_or =
      mgr.mint("daemon-3", tensorcast::common::v1::CAPABILITY_AUDIENCE_PLACEMENT_LEASE, *scope_or, expires_at_ms);
  REQUIRE(token_or.ok());

  const absl::Time later = now + absl::Seconds(5);
  auto env_or = mgr.verify(
      *token_or,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PLACEMENT_LEASE,
      "daemon-3",
      later,
      /*require_not_expired=*/true);
  REQUIRE_FALSE(env_or.ok());
}

} // namespace
} // namespace tensorcast::common
