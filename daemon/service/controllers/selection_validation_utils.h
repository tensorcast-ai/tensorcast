// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "absl/types/span.h"
#include "grpcpp/support/status.h"
#include "tensorcast/common/v1/common.pb.h"

namespace tensorcast::daemon::selection_validation {

grpc::Status validate_request_tensor_names(
    const tensorcast::common::v1::ArtifactSelection& selection,
    std::vector<std::string>& ordered_tensor_names,
    std::string_view* error_reason = nullptr);

grpc::Status compute_and_validate_view_subset_hash(
    const tensorcast::common::v1::ArtifactSelection& selection,
    absl::Span<const std::string> resolved_selection_names,
    std::string& view_subset_hash,
    std::string_view* error_reason = nullptr);

grpc::Status validate_hashes_and_build_resolved_selection(
    const tensorcast::common::v1::ArtifactSelection& request_selection,
    std::string_view resolved_artifact_id,
    std::string_view resolved_view_id,
    std::string_view selected_index_json,
    bool needs_view_index,
    absl::Span<const std::string> resolved_selection_names,
    std::string_view view_subset_hash,
    const tensorcast::common::v1::ViewSpec* resolved_view_spec,
    tensorcast::common::v1::ArtifactSelection& resolved_selection,
    std::string_view* error_reason = nullptr);

} // namespace tensorcast::daemon::selection_validation
