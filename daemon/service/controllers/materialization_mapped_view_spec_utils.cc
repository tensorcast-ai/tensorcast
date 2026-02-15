// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_mapped_view_spec_utils.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "core/store/view_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"

namespace tensorcast::daemon::materialization_mapped_view_spec {

namespace {

using materialization_policy::convert_view_spec;
using store::loader::ViewSpec;

void set_reason(ResolveViewSpecErrorReason* reason, ResolveViewSpecErrorReason value) {
  if (reason != nullptr) {
    *reason = value;
  }
}

} // namespace

std::string_view resolve_view_spec_error_reason(ResolveViewSpecErrorReason reason) {
  switch (reason) {
    case ResolveViewSpecErrorReason::kViewMetaMissing:
      return "view_meta_missing";
    case ResolveViewSpecErrorReason::kViewParseFailed:
      return "view_parse_failed";
    case ResolveViewSpecErrorReason::kViewSpecInvalid:
      return "view_spec_invalid";
    case ResolveViewSpecErrorReason::kUnknown:
    default:
      return "transfer_error";
  }
}

absl::StatusOr<std::optional<ViewSpec>> resolve_mapped_view_spec(
    const v2::MaterializeIntoMappedTargetRequest& req,
    std::string_view resolved_artifact_id,
    store::StoreEngine& engine,
    ResolveViewSpecErrorReason* reason) {
  set_reason(reason, ResolveViewSpecErrorReason::kUnknown);

  std::optional<ViewSpec> view_spec;
  std::optional<std::string> request_view_id;

  if (!req.has_selection()) {
    return absl::InvalidArgumentError("selection is required");
  }
  const auto& selection = req.selection();
  if (selection.has_view_spec()) {
    auto spec_or = convert_view_spec(selection.view_spec());
    if (!spec_or.ok()) {
      set_reason(reason, ResolveViewSpecErrorReason::kViewSpecInvalid);
      return spec_or.status();
    }
    view_spec = std::move(*spec_or);
  }
  if (!selection.view_id().empty()) {
    request_view_id = selection.view_id();
  }

  if (!request_view_id.has_value() || view_spec.has_value()) {
    return view_spec;
  }

  auto view_meta_or = engine.get_view_metadata(std::string(resolved_artifact_id), *request_view_id);
  if (!view_meta_or.ok()) {
    set_reason(reason, ResolveViewSpecErrorReason::kViewMetaMissing);
    return view_meta_or.status();
  }
  auto spec_or = store::view::parse_view_spec_json(view_meta_or->view_spec_json);
  if (!spec_or.ok()) {
    set_reason(reason, ResolveViewSpecErrorReason::kViewParseFailed);
    return spec_or.status();
  }
  view_spec = std::move(*spec_or);

  return view_spec;
}

} // namespace tensorcast::daemon::materialization_mapped_view_spec
