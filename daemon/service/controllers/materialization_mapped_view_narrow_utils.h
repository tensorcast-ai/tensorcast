// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "daemon/service/controllers/representation_layout_types.h"

namespace tensorcast::daemon::materialization_mapped_view_narrow {

enum class ViewNarrowErrorReason {
  kTransposeUnsupported,
  kMultipleNarrowPerTensor,
  kUnknown,
};

std::string_view view_narrow_error_reason(ViewNarrowErrorReason reason);

absl::StatusOr<absl::flat_hash_map<std::string, representation_layout::ViewNarrowSpec>> build_view_narrows(
    const std::optional<store::loader::ViewSpec>& view_spec,
    ViewNarrowErrorReason* reason);

} // namespace tensorcast::daemon::materialization_mapped_view_narrow
