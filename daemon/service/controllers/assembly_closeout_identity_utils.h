// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/components/global_store_client.h"
#include "daemon/state/types.h"

namespace tensorcast::daemon::assembly_closeout_identity {

absl::StatusOr<std::string> resolve_registration_canonical_index_json(
    const std::optional<CommitLeaseResult>& reused_sealed_identity,
    const CommitLeaseResult& closeout_commit_result);

absl::Status validate_registration_canonical_index_matches_commit_result(
    std::string_view canonical_index_json,
    const CommitLeaseResult& closeout_commit_result);

absl::Status validate_reused_identity_matches_closeout_result(
    const CommitLeaseResult& reused_sealed_identity,
    const CommitLeaseResult& closeout_commit_result);

absl::Status publish_workspace_seal_binding_after_success(
    std::shared_ptr<store::components::IGlobalStoreClient> client,
    std::string_view workspace_assembly_id,
    std::string_view sealed_artifact_id,
    bool binding_subject_closeout,
    bool reused_sealed_identity);

} // namespace tensorcast::daemon::assembly_closeout_identity
