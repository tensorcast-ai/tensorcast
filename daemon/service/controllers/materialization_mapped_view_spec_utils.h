// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/store_engine.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::materialization_mapped_view_spec {

enum class ResolveViewSpecErrorReason {
  kViewMetaMissing,
  kViewParseFailed,
  kViewSpecInvalid,
  kUnknown,
};

std::string_view resolve_view_spec_error_reason(ResolveViewSpecErrorReason reason);

struct ResolvedMappedViewSpec {
  std::optional<store::loader::ViewSpec> view_spec;
  std::optional<std::string> request_view_id;
};

absl::StatusOr<ResolvedMappedViewSpec> resolve_mapped_view_spec(
    const v2::MaterializeIntoMappedTargetRequest& req,
    std::string_view resolved_artifact_id,
    store::StoreEngine& engine,
    ResolveViewSpecErrorReason* reason);

} // namespace tensorcast::daemon::materialization_mapped_view_spec
