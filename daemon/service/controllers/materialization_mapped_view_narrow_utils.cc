// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_mapped_view_narrow_utils.h"

#include <cstdint>
#include <string>
#include <string_view>

#include "absl/status/status.h"

namespace tensorcast::daemon::materialization_mapped_view_narrow {

namespace {

using materialization_mapped_copy_plan::ViewNarrowSpec;
using store::loader::ViewOp;
using store::loader::ViewSpec;

void set_reason(ViewNarrowErrorReason* reason, ViewNarrowErrorReason value) {
  if (reason != nullptr) {
    *reason = value;
  }
}

} // namespace

std::string_view view_narrow_error_reason(ViewNarrowErrorReason reason) {
  switch (reason) {
    case ViewNarrowErrorReason::kTransposeUnsupported:
      return "view_transpose";
    case ViewNarrowErrorReason::kMultipleNarrowPerTensor:
      return "view_narrow";
    case ViewNarrowErrorReason::kUnknown:
    default:
      return "transfer_error";
  }
}

absl::StatusOr<absl::flat_hash_map<std::string, ViewNarrowSpec>> build_view_narrows(
    const std::optional<ViewSpec>& view_spec,
    ViewNarrowErrorReason* reason) {
  set_reason(reason, ViewNarrowErrorReason::kUnknown);

  absl::flat_hash_map<std::string, ViewNarrowSpec> view_narrows;
  if (!view_spec.has_value()) {
    return view_narrows;
  }

  for (const auto& [tensor_name, ops] : view_spec->tensors) {
    bool saw_narrow = false;
    for (const auto& op : ops.ops) {
      if (op.kind == ViewOp::Kind::kTranspose) {
        set_reason(reason, ViewNarrowErrorReason::kTransposeUnsupported);
        return absl::InvalidArgumentError("mapped binding does not support transpose views");
      }
      if (op.kind == ViewOp::Kind::kNarrow) {
        if (saw_narrow) {
          set_reason(reason, ViewNarrowErrorReason::kMultipleNarrowPerTensor);
          return absl::InvalidArgumentError("mapped binding supports one narrow per tensor");
        }
        saw_narrow = true;
        view_narrows.emplace(
            tensor_name,
            ViewNarrowSpec{
                .dim = static_cast<int32_t>(op.narrow.dim),
                .start = op.narrow.start,
                .end = static_cast<int64_t>(op.narrow.start + op.narrow.length),
            });
      }
    }
  }

  return view_narrows;
}

} // namespace tensorcast::daemon::materialization_mapped_view_narrow
