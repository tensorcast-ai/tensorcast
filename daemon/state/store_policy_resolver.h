// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <optional>

#include "absl/status/statusor.h"
#include "core/store/components/stable_dram_cache_policy.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

enum class RequirementLevel : uint8_t {
  kNone = 0,
  kMay = 1,
  kShould = 2,
  kMust = 3,
};

struct ResolvedStorePolicy {
  RequirementLevel local_requirement{RequirementLevel::kNone};
  RequirementLevel remote_requirement{RequirementLevel::kNone};
  RequirementLevel shared_disk_requirement{RequirementLevel::kNone};
  store::components::StableRetentionPolicy local_retention{store::components::StableRetentionPolicy::kBestEffort};
  std::optional<std::chrono::milliseconds> local_ttl;
  store::components::StableOverflowPolicy overflow_policy{store::components::StableOverflowPolicy::kEvict};
  v2::PolicyLayout layout{v2::POLICY_LAYOUT_AUTO};
};

absl::StatusOr<ResolvedStorePolicy> resolve_store_policy(const v2::StorePolicy* policy);

std::optional<store::components::StableDramCachePolicy> stable_cache_policy_from_resolved(
    const ResolvedStorePolicy& policy);

RequirementLevel max_requirement(RequirementLevel lhs, RequirementLevel rhs);

} // namespace tensorcast::daemon
