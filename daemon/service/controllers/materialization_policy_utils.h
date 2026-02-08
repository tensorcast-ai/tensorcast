// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/store_engine.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::materialization_policy {

using store::loader::ViewSpec;

struct ResolvedSourcePolicy {
  v2::SourcePreference preference{v2::SourcePreference::SOURCE_PREFERENCE_AUTO};
  bool allow_p2p{true};
  bool allow_disk{true};
};

store::loading::SourcePreference to_hint_preference(v2::SourcePreference preference);

store::loading::ExportPolicy to_hint_export_policy(v2::ExportPolicy policy);

ResolvedSourcePolicy resolve_source_policy(const v2::SourcePolicy* policy, v2::SourcePreference legacy_preference);

absl::Status validate_source_policy(const ResolvedSourcePolicy& policy);

absl::StatusOr<ViewSpec> convert_view_spec(const tensorcast::common::v1::ViewSpec& proto);

absl::StatusOr<std::string> compute_view_id_from_spec(
    const tensorcast::common::v1::ViewSpec& view_spec,
    std::string_view canonical_index_json);

tensorcast::common::v1::ViewSpec build_view_spec_proto(const ViewSpec& spec);

bool spec_includes_transpose(const ViewSpec& spec);

store::loading::TransformPlacement resolve_transform_placement(
    v2::TransformPlacement requested,
    const std::optional<ViewSpec>& spec);

} // namespace tensorcast::daemon::materialization_policy
