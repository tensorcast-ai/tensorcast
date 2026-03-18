// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "core/store/store_engine.h"
#include "daemon/service/body_backing_types.h"
#include "daemon/service/byte_artifact_body_handle.h"
#include "daemon/state/store_policy_resolver.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class BodyBackingManager {
 public:
  struct StageRequest {
    std::string artifact_id;
    v2::PutIfAbsentInvariant invariant;
    std::unique_ptr<store::IArtifactLoader> loader;
    store::loading::MaterializationSource source_kind{store::loading::MaterializationSource::kUnspecified};
    std::string operation_id;
    BodyAccessClass access_class{BodyAccessClass::kHomeDefault};
    BodyRouteRole route_role{BodyRouteRole::kHomeAuthority};
    std::optional<ResolvedStorePolicy> resolved_store_policy;
  };

  struct StageResult {
    BodyDescriptor descriptor;
    BodyBackingObservation observation;
    BodyHandle body_handle;
    store::runtime::ingestion::VerifiedContentDescriptor verified_content_descriptor;
    store::runtime::ingestion::VerificationRecord verification_record;
    store::runtime::ingestion::BackingIdentity backing_identity;
  };

  struct ReuseRequest {
    std::string artifact_id;
    v2::PutIfAbsentInvariant invariant;
    BodyDescriptor descriptor;
    BodyHandle body_handle;
    std::string operation_id;
    BodyAccessClass access_class{BodyAccessClass::kHomeDefault};
    BodyRouteRole route_role{BodyRouteRole::kHomeAuthority};
    std::optional<ResolvedStorePolicy> resolved_store_policy;
  };

  explicit BodyBackingManager(store::StoreEngine& engine);

  [[nodiscard]] absl::StatusOr<StageResult> stage_body(StageRequest request) const;
  [[nodiscard]] absl::StatusOr<std::optional<StageResult>> try_reuse_body(ReuseRequest request) const;

 private:
  [[nodiscard]] absl::StatusOr<ResolvedStorePolicy> resolve_body_store_policy(
      BodyAccessClass access_class,
      BodyRouteRole route_role,
      const std::optional<ResolvedStorePolicy>& resolved_store_policy) const;
  [[nodiscard]] BodyPlacementContext normalize_placement_context(
      BodyAccessClass access_class,
      BodyRouteRole route_role,
      std::uint64_t size_bytes) const;
  [[nodiscard]] BodyBackingIntent classify_intent(
      const BodyPlacementContext& context,
      const ResolvedStorePolicy& resolved_policy) const;

  store::StoreEngine& engine_;
};

} // namespace tensorcast::daemon
