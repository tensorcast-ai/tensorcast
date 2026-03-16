// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "daemon/service/body_backing_types.h"

namespace tensorcast::daemon {

struct MintServingCapabilityRequest {
  std::string capability_id;
  absl::Time expires_at{absl::InfinitePast()};
  BodyCapabilityResolutionMode mode{BodyCapabilityResolutionMode::kLoader};
  bool local{true};
  ServingCapabilitySubjectKind subject_kind{ServingCapabilitySubjectKind::kBacking};
  LifecycleOwnerRef lifecycle_owner_ref;
  std::optional<store::runtime::ingestion::BackingIdentity> backing_identity;
  std::uint64_t backing_instance_generation{0};
  std::optional<PolicyVisibilityRef> policy_visibility_ref;
};

[[nodiscard]] absl::StatusOr<ServingCapability> mint_serving_capability(MintServingCapabilityRequest request);

[[nodiscard]] absl::Status release_serving_capability(const ServingCapability& capability);

} // namespace tensorcast::daemon
