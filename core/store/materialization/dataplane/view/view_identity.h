// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/view/view_spec.h"

namespace tensorcast::store::loader {

using materialization::view::ViewSpec;

// Returns a deterministic semantic encoding for view identity hashing.
std::string canonicalize_view_spec_for_identity(const ViewSpec& spec);

// Computes deterministic view_id from semantic view spec + canonical index bytes.
absl::StatusOr<std::string> compute_view_id_from_spec(const ViewSpec& spec, std::string_view canonical_index_json);

} // namespace tensorcast::store::loader
