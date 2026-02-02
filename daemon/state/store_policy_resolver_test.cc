// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/store_policy_resolver.h"

#include <catch2/catch_test_macros.hpp>
#include "absl/status/status.h"

using tensorcast::daemon::RequirementLevel;
using tensorcast::daemon::resolve_store_policy;
using tensorcast::store::components::StableOverflowPolicy;
using tensorcast::store::components::StableRetentionPolicy;
namespace v2 = tensorcast::daemon::v2;

TEST_CASE("StorePolicyResolver expands ha profile", "[daemon][policy]") {
  v2::StorePolicy policy;
  policy.set_profile(v2::POLICY_PROFILE_HA);
  auto resolved_or = resolve_store_policy(&policy);
  REQUIRE(resolved_or.ok());
  const auto& resolved = *resolved_or;
  CHECK(resolved.shared_disk_requirement == RequirementLevel::kMust);
  CHECK(resolved.remote_requirement == RequirementLevel::kShould);
  CHECK(resolved.local_requirement == RequirementLevel::kShould);
  CHECK(resolved.layout == v2::POLICY_LAYOUT_AUTO);
}

TEST_CASE("StorePolicyResolver expands warm profile", "[daemon][policy]") {
  v2::StorePolicy policy;
  policy.set_profile(v2::POLICY_PROFILE_WARM);
  auto resolved_or = resolve_store_policy(&policy);
  REQUIRE(resolved_or.ok());
  const auto& resolved = *resolved_or;
  CHECK(resolved.shared_disk_requirement == RequirementLevel::kNone);
  CHECK(resolved.remote_requirement == RequirementLevel::kNone);
  CHECK(resolved.local_requirement == RequirementLevel::kShould);
  CHECK(resolved.local_retention == StableRetentionPolicy::kBestEffort);
  CHECK(resolved.overflow_policy == StableOverflowPolicy::kReject);
  CHECK(resolved.layout == v2::POLICY_LAYOUT_AUTO);
}

TEST_CASE("StorePolicyResolver rejects profile with tier list", "[daemon][policy]") {
  v2::StorePolicy policy;
  policy.set_profile(v2::POLICY_PROFILE_CACHE);
  auto* tier = policy.add_must();
  tier->set_tier(v2::POLICY_TIER_SHARED_DISK);
  auto resolved_or = resolve_store_policy(&policy);
  REQUIRE_FALSE(resolved_or.ok());
  REQUIRE(resolved_or.status().code() == absl::StatusCode::kInvalidArgument);
}

TEST_CASE("StorePolicyResolver rejects spill without shared disk", "[daemon][policy]") {
  v2::StorePolicy policy;
  policy.set_overflow_policy(v2::OVERFLOW_POLICY_SPILL);
  auto resolved_or = resolve_store_policy(&policy);
  REQUIRE_FALSE(resolved_or.ok());
  REQUIRE(resolved_or.status().code() == absl::StatusCode::kInvalidArgument);
}

TEST_CASE("StorePolicyResolver rejects shared_disk retention fields", "[daemon][policy]") {
  v2::StorePolicy policy;
  auto* tier = policy.add_must();
  tier->set_tier(v2::POLICY_TIER_SHARED_DISK);
  tier->set_retention_policy(v2::RETENTION_POLICY_TTL);
  tier->set_retention_ttl_ms(1000);
  auto resolved_or = resolve_store_policy(&policy);
  REQUIRE_FALSE(resolved_or.ok());
  REQUIRE(resolved_or.status().code() == absl::StatusCode::kInvalidArgument);
}

TEST_CASE("StorePolicyResolver rejects stable_dram min_replicas != 1", "[daemon][policy]") {
  v2::StorePolicy policy;
  auto* tier = policy.add_should();
  tier->set_tier(v2::POLICY_TIER_STABLE_DRAM);
  tier->set_scope(v2::POLICY_SCOPE_LOCAL);
  tier->set_min_replicas(2);
  auto resolved_or = resolve_store_policy(&policy);
  REQUIRE_FALSE(resolved_or.ok());
  REQUIRE(resolved_or.status().code() == absl::StatusCode::kInvalidArgument);
}

TEST_CASE("StorePolicyResolver rejects retention on remote stable_dram", "[daemon][policy]") {
  v2::StorePolicy policy;
  auto* tier = policy.add_must();
  tier->set_tier(v2::POLICY_TIER_STABLE_DRAM);
  tier->set_scope(v2::POLICY_SCOPE_REMOTE);
  tier->set_retention_policy(v2::RETENTION_POLICY_PINNED);
  auto resolved_or = resolve_store_policy(&policy);
  REQUIRE_FALSE(resolved_or.ok());
  REQUIRE(resolved_or.status().code() == absl::StatusCode::kInvalidArgument);
}

TEST_CASE("StorePolicyResolver rejects must local stable_dram without pinned retention", "[daemon][policy]") {
  v2::StorePolicy policy;
  auto* tier = policy.add_must();
  tier->set_tier(v2::POLICY_TIER_STABLE_DRAM);
  tier->set_scope(v2::POLICY_SCOPE_LOCAL);
  tier->set_retention_policy(v2::RETENTION_POLICY_BEST_EFFORT);
  auto resolved_or = resolve_store_policy(&policy);
  REQUIRE_FALSE(resolved_or.ok());
  REQUIRE(resolved_or.status().code() == absl::StatusCode::kInvalidArgument);
}

TEST_CASE("StorePolicyResolver honors remote must tiers", "[daemon][policy]") {
  v2::StorePolicy policy;
  auto* tier = policy.add_must();
  tier->set_tier(v2::POLICY_TIER_STABLE_DRAM);
  tier->set_scope(v2::POLICY_SCOPE_REMOTE);
  tier->set_min_replicas(1);
  auto resolved_or = resolve_store_policy(&policy);
  REQUIRE(resolved_or.ok());
  CHECK(resolved_or->remote_requirement == RequirementLevel::kMust);
  CHECK(resolved_or->local_requirement == RequirementLevel::kNone);
}
