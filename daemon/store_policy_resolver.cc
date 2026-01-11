// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/store_policy_resolver.h"

#include <cstdint>

#include "absl/status/status.h"

namespace tensorcast::daemon {

namespace {

constexpr int64_t kDefaultColdTtlMs = 60'000;

int requirement_rank(RequirementLevel level) {
  return static_cast<int>(level);
}

int retention_rank(store::components::StableRetentionPolicy policy) {
  switch (policy) {
    case store::components::StableRetentionPolicy::kPinned:
      return 3;
    case store::components::StableRetentionPolicy::kTtl:
      return 2;
    case store::components::StableRetentionPolicy::kBestEffort:
    default:
      return 1;
  }
}

RequirementLevel max_level(RequirementLevel lhs, RequirementLevel rhs) {
  return requirement_rank(lhs) >= requirement_rank(rhs) ? lhs : rhs;
}

store::components::StableRetentionPolicy retention_from_proto(v2::RetentionPolicy policy) {
  switch (policy) {
    case v2::RETENTION_POLICY_TTL:
      return store::components::StableRetentionPolicy::kTtl;
    case v2::RETENTION_POLICY_PINNED:
      return store::components::StableRetentionPolicy::kPinned;
    case v2::RETENTION_POLICY_BEST_EFFORT:
    case v2::RETENTION_POLICY_UNSPECIFIED:
    default:
      return store::components::StableRetentionPolicy::kBestEffort;
  }
}

std::optional<store::components::StableOverflowPolicy> overflow_override_from_proto(v2::OverflowPolicy policy) {
  switch (policy) {
    case v2::OVERFLOW_POLICY_EVICT:
      return store::components::StableOverflowPolicy::kEvict;
    case v2::OVERFLOW_POLICY_SPILL:
      return store::components::StableOverflowPolicy::kSpill;
    case v2::OVERFLOW_POLICY_REJECT:
      return store::components::StableOverflowPolicy::kReject;
    case v2::OVERFLOW_POLICY_UNSPECIFIED:
    default:
      return std::nullopt;
  }
}

store::components::StableOverflowPolicy overflow_from_proto(v2::OverflowPolicy policy) {
  return overflow_override_from_proto(policy).value_or(store::components::StableOverflowPolicy::kEvict);
}

std::optional<v2::PolicyLayout> layout_override_from_proto(v2::PolicyLayout layout) {
  switch (layout) {
    case v2::POLICY_LAYOUT_AUTO:
    case v2::POLICY_LAYOUT_UNSHARDED:
    case v2::POLICY_LAYOUT_SHARDED:
      return layout;
    case v2::POLICY_LAYOUT_UNSPECIFIED:
    default:
      return std::nullopt;
  }
}

v2::PolicyLayout layout_from_proto(v2::PolicyLayout layout) {
  return layout_override_from_proto(layout).value_or(v2::POLICY_LAYOUT_AUTO);
}

absl::Status validate_retention_spec(const v2::TierSpec& spec) {
  if (spec.retention_policy() == v2::RETENTION_POLICY_TTL) {
    if (!spec.has_retention_ttl_ms() || spec.retention_ttl_ms() == 0) {
      return absl::InvalidArgumentError("retention_policy=ttl requires retention_ttl_ms");
    }
    return absl::OkStatus();
  }
  if (spec.has_retention_ttl_ms()) {
    return absl::InvalidArgumentError("retention_ttl_ms is only valid when retention_policy=ttl");
  }
  return absl::OkStatus();
}

void apply_local_tier(
    RequirementLevel level,
    store::components::StableRetentionPolicy retention,
    std::optional<std::chrono::milliseconds> ttl,
    ResolvedStorePolicy& resolved) {
  if (requirement_rank(level) > requirement_rank(resolved.local_requirement)) {
    resolved.local_requirement = level;
    resolved.local_retention = retention;
    resolved.local_ttl = ttl;
    return;
  }
  if (requirement_rank(level) < requirement_rank(resolved.local_requirement)) {
    return;
  }
  if (retention_rank(retention) > retention_rank(resolved.local_retention)) {
    resolved.local_retention = retention;
    resolved.local_ttl = ttl;
    return;
  }
  if (retention == resolved.local_retention && retention == store::components::StableRetentionPolicy::kTtl &&
      ttl.has_value() && resolved.local_ttl.has_value()) {
    if (*ttl > *resolved.local_ttl) {
      resolved.local_ttl = ttl;
    }
  }
}

absl::Status apply_tier(const v2::TierSpec& spec, RequirementLevel level, ResolvedStorePolicy& resolved) {
  if (spec.tier() == v2::POLICY_TIER_UNSPECIFIED) {
    return absl::InvalidArgumentError("tier must be specified");
  }

  if (spec.tier() == v2::POLICY_TIER_SHARED_DISK) {
    const v2::PolicyScope scope = spec.scope() == v2::POLICY_SCOPE_UNSPECIFIED ? v2::POLICY_SCOPE_ANY : spec.scope();
    if (scope != v2::POLICY_SCOPE_ANY) {
      return absl::InvalidArgumentError("shared_disk scope must be any");
    }
    if (spec.retention_policy() != v2::RETENTION_POLICY_UNSPECIFIED || spec.has_retention_ttl_ms()) {
      return absl::InvalidArgumentError("shared_disk does not support retention_policy or retention_ttl_ms");
    }
    const uint32_t min_replicas = spec.min_replicas() == 0 ? 1u : spec.min_replicas();
    if (min_replicas != 1u) {
      return absl::InvalidArgumentError("shared_disk min_replicas must be 1");
    }
    resolved.shared_disk_requirement = max_level(resolved.shared_disk_requirement, level);
    return absl::OkStatus();
  }

  if (spec.tier() != v2::POLICY_TIER_STABLE_DRAM) {
    return absl::InvalidArgumentError("unsupported policy tier");
  }

  const uint32_t min_replicas = spec.min_replicas() == 0 ? 1u : spec.min_replicas();
  if (min_replicas != 1u) {
    return absl::InvalidArgumentError("stable_dram min_replicas must be 1");
  }

  auto retention_status = validate_retention_spec(spec);
  if (!retention_status.ok()) {
    return retention_status;
  }
  const auto retention = retention_from_proto(spec.retention_policy());
  std::optional<std::chrono::milliseconds> ttl;
  if (retention == store::components::StableRetentionPolicy::kTtl) {
    ttl = std::chrono::milliseconds(static_cast<int64_t>(spec.retention_ttl_ms()));
  }

  const v2::PolicyScope scope = spec.scope() == v2::POLICY_SCOPE_UNSPECIFIED ? v2::POLICY_SCOPE_ANY : spec.scope();
  if (scope == v2::POLICY_SCOPE_REMOTE) {
    if (spec.retention_policy() != v2::RETENTION_POLICY_UNSPECIFIED || spec.has_retention_ttl_ms()) {
      return absl::InvalidArgumentError("retention_policy is only valid for local stable_dram");
    }
  }
  if (scope == v2::POLICY_SCOPE_LOCAL || scope == v2::POLICY_SCOPE_ANY) {
    if (level == RequirementLevel::kMust && retention != store::components::StableRetentionPolicy::kPinned) {
      return absl::InvalidArgumentError("must local stable_dram requires retention_policy=pinned");
    }
    apply_local_tier(level, retention, ttl, resolved);
  }
  if (scope == v2::POLICY_SCOPE_REMOTE || scope == v2::POLICY_SCOPE_ANY) {
    resolved.remote_requirement = max_level(resolved.remote_requirement, level);
  }
  return absl::OkStatus();
}

ResolvedStorePolicy profile_defaults(v2::PolicyProfile profile) {
  ResolvedStorePolicy resolved;
  switch (profile) {
    case v2::POLICY_PROFILE_DURABLE:
      resolved.shared_disk_requirement = RequirementLevel::kMust;
      resolved.local_requirement = RequirementLevel::kShould;
      resolved.local_retention = store::components::StableRetentionPolicy::kBestEffort;
      break;
    case v2::POLICY_PROFILE_HA:
      resolved.shared_disk_requirement = RequirementLevel::kMust;
      resolved.remote_requirement = RequirementLevel::kShould;
      resolved.local_requirement = RequirementLevel::kShould;
      resolved.local_retention = store::components::StableRetentionPolicy::kBestEffort;
      break;
    case v2::POLICY_PROFILE_COLD:
      resolved.shared_disk_requirement = RequirementLevel::kMust;
      resolved.local_requirement = RequirementLevel::kShould;
      resolved.local_retention = store::components::StableRetentionPolicy::kTtl;
      resolved.local_ttl = std::chrono::milliseconds(kDefaultColdTtlMs);
      break;
    case v2::POLICY_PROFILE_WARM:
      resolved.local_requirement = RequirementLevel::kShould;
      resolved.local_retention = store::components::StableRetentionPolicy::kBestEffort;
      resolved.overflow_policy = store::components::StableOverflowPolicy::kReject;
      break;
    case v2::POLICY_PROFILE_PINNED:
      resolved.local_requirement = RequirementLevel::kMust;
      resolved.local_retention = store::components::StableRetentionPolicy::kPinned;
      resolved.overflow_policy = store::components::StableOverflowPolicy::kReject;
      break;
    case v2::POLICY_PROFILE_CACHE:
    case v2::POLICY_PROFILE_UNSPECIFIED:
    default:
      resolved.local_requirement = RequirementLevel::kMay;
      resolved.local_retention = store::components::StableRetentionPolicy::kBestEffort;
      break;
  }
  return resolved;
}

} // namespace

RequirementLevel max_requirement(RequirementLevel lhs, RequirementLevel rhs) {
  return max_level(lhs, rhs);
}

absl::StatusOr<ResolvedStorePolicy> resolve_store_policy(const v2::StorePolicy* policy) {
  if (policy == nullptr) {
    ResolvedStorePolicy resolved = profile_defaults(v2::POLICY_PROFILE_CACHE);
    resolved.layout = v2::POLICY_LAYOUT_AUTO;
    return resolved;
  }
  const bool has_profile = policy->profile() != v2::POLICY_PROFILE_UNSPECIFIED;
  const bool has_tiers = !policy->must().empty() || !policy->should().empty() || !policy->may().empty();
  if (has_profile && has_tiers) {
    return absl::InvalidArgumentError("profile cannot be set when must/should/may are provided");
  }

  const auto overflow_override = overflow_override_from_proto(policy->overflow_policy());
  const auto layout_override = layout_override_from_proto(policy->layout());

  if (has_profile || !has_tiers) {
    const v2::PolicyProfile selected_profile = has_profile ? policy->profile() : v2::POLICY_PROFILE_CACHE;
    ResolvedStorePolicy resolved = profile_defaults(selected_profile);
    if (overflow_override.has_value()) {
      resolved.overflow_policy = *overflow_override;
    }
    if (layout_override.has_value()) {
      resolved.layout = *layout_override;
    } else {
      resolved.layout = v2::POLICY_LAYOUT_AUTO;
    }
    if (resolved.overflow_policy == store::components::StableOverflowPolicy::kSpill &&
        requirement_rank(resolved.shared_disk_requirement) < requirement_rank(RequirementLevel::kShould)) {
      return absl::InvalidArgumentError("overflow_policy=spill requires shared_disk in must or should");
    }
    return resolved;
  }

  ResolvedStorePolicy resolved;
  resolved.overflow_policy = overflow_from_proto(policy->overflow_policy());
  resolved.layout = layout_from_proto(policy->layout());

  for (const auto& spec : policy->must()) {
    auto st = apply_tier(spec, RequirementLevel::kMust, resolved);
    if (!st.ok()) {
      return st;
    }
  }
  for (const auto& spec : policy->should()) {
    auto st = apply_tier(spec, RequirementLevel::kShould, resolved);
    if (!st.ok()) {
      return st;
    }
  }
  for (const auto& spec : policy->may()) {
    auto st = apply_tier(spec, RequirementLevel::kMay, resolved);
    if (!st.ok()) {
      return st;
    }
  }

  if (resolved.overflow_policy == store::components::StableOverflowPolicy::kSpill &&
      requirement_rank(resolved.shared_disk_requirement) < requirement_rank(RequirementLevel::kShould)) {
    return absl::InvalidArgumentError("overflow_policy=spill requires shared_disk in must or should");
  }
  return resolved;
}

std::optional<store::components::StableDramCachePolicy> stable_cache_policy_from_resolved(
    const ResolvedStorePolicy& policy) {
  if (policy.local_requirement == RequirementLevel::kNone) {
    return std::nullopt;
  }
  store::components::StableDramCachePolicy out;
  out.retention_policy = policy.local_retention;
  out.retention_ttl = policy.local_ttl;
  out.overflow_policy = policy.overflow_policy;
  out.required = policy.local_requirement == RequirementLevel::kMust;
  out.require_shared_disk_for_spill = policy.shared_disk_requirement == RequirementLevel::kMust;
  out.require_remote_stable_for_spill = policy.remote_requirement == RequirementLevel::kMust;
  return out;
}

} // namespace tensorcast::daemon
