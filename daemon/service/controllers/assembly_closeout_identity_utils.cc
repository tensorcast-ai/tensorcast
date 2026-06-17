// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/assembly_closeout_identity_utils.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"

namespace tensorcast::daemon::assembly_closeout_identity {
namespace {

absl::StatusOr<std::pair<std::string, std::string>> parse_mi2_multihashes(std::string_view artifact_id) {
  constexpr std::string_view kPrefix = "mi2:";
  if (!artifact_id.starts_with(kPrefix)) {
    return absl::InvalidArgumentError("artifact_id must start with \"mi2:\"");
  }
  const size_t index_begin = kPrefix.size();
  const size_t sep = artifact_id.find(':', index_begin);
  if (sep == std::string_view::npos) {
    return absl::InvalidArgumentError("artifact_id must be of form mi2:<index_multihash>:<data_multihash>");
  }
  const std::string_view index_multihash = artifact_id.substr(index_begin, sep - index_begin);
  const std::string_view data_multihash = artifact_id.substr(sep + 1);
  if (index_multihash.empty() || data_multihash.empty()) {
    return absl::InvalidArgumentError("artifact_id must include index and data multihash components");
  }
  return std::make_pair(std::string(index_multihash), std::string(data_multihash));
}

absl::Status validate_commit_result_matches_artifact_identity(const CommitLeaseResult& commit_result) {
  if (commit_result.artifact_id.empty()) {
    return absl::InvalidArgumentError("commit result artifact_id is required");
  }
  if (commit_result.index_multihash.empty()) {
    return absl::InvalidArgumentError("commit result index_multihash is required");
  }
  if (common::infer_artifact_id_kind(commit_result.artifact_id) != common::ArtifactIdKind::kMi2) {
    return absl::OkStatus();
  }
  auto parsed_or = parse_mi2_multihashes(commit_result.artifact_id);
  if (!parsed_or.ok()) {
    return parsed_or.status();
  }
  if (parsed_or->first != commit_result.index_multihash) {
    return absl::FailedPreconditionError(
        "commit result artifact_id index_multihash does not match descriptor index_multihash");
  }
  if (commit_result.data_multihash.empty()) {
    return absl::InvalidArgumentError("commit result data_multihash is required for mi2 artifact_id");
  }
  if (parsed_or->second != commit_result.data_multihash) {
    return absl::FailedPreconditionError(
        "commit result artifact_id data_multihash does not match descriptor data_multihash");
  }
  return absl::OkStatus();
}

} // namespace

absl::StatusOr<std::string> resolve_registration_canonical_index_json(
    const std::optional<CommitLeaseResult>& reused_sealed_identity,
    const CommitLeaseResult& closeout_commit_result) {
  if (reused_sealed_identity.has_value()) {
    if (reused_sealed_identity->canonical_index_json.empty()) {
      return absl::FailedPreconditionError(
          "same-binding seal reuse requires canonical_index_json on the reused seal identity");
    }
    return reused_sealed_identity->canonical_index_json;
  }
  if (closeout_commit_result.canonical_index_json.empty()) {
    return absl::FailedPreconditionError("binding closeout requires canonical_index_json for GS registration");
  }
  return closeout_commit_result.canonical_index_json;
}

absl::Status validate_registration_canonical_index_matches_commit_result(
    std::string_view canonical_index_json,
    const CommitLeaseResult& closeout_commit_result) {
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("registration canonical_index_json must not be empty");
  }
  auto identity_status = validate_commit_result_matches_artifact_identity(closeout_commit_result);
  if (!identity_status.ok()) {
    return identity_status;
  }
  auto index_multihash_or =
      common::compute_index_multihash(std::optional<std::string>(std::string(canonical_index_json)), "");
  if (!index_multihash_or.ok()) {
    return index_multihash_or.status();
  }
  if (*index_multihash_or != closeout_commit_result.index_multihash) {
    return absl::FailedPreconditionError(
        "registration canonical_index_json does not match closeout commit_result index_multihash");
  }
  if (common::infer_artifact_id_kind(closeout_commit_result.artifact_id) == common::ArtifactIdKind::kMi2) {
    auto parsed_or = parse_mi2_multihashes(closeout_commit_result.artifact_id);
    if (!parsed_or.ok()) {
      return parsed_or.status();
    }
    if (parsed_or->first != *index_multihash_or) {
      return absl::FailedPreconditionError(
          "registration canonical_index_json does not match closeout artifact_id index_multihash");
    }
  }
  return absl::OkStatus();
}

absl::Status validate_reused_identity_matches_closeout_result(
    const CommitLeaseResult& reused_sealed_identity,
    const CommitLeaseResult& closeout_commit_result) {
  auto reused_status = validate_commit_result_matches_artifact_identity(reused_sealed_identity);
  if (!reused_status.ok()) {
    return reused_status;
  }
  auto closeout_status = validate_commit_result_matches_artifact_identity(closeout_commit_result);
  if (!closeout_status.ok()) {
    return closeout_status;
  }
  if (reused_sealed_identity.artifact_id != closeout_commit_result.artifact_id) {
    return absl::FailedPreconditionError(
        "same-binding closeout mutated artifact_id instead of reusing the sealed identity");
  }
  if (reused_sealed_identity.index_multihash != closeout_commit_result.index_multihash) {
    return absl::FailedPreconditionError(
        "same-binding closeout mutated index_multihash instead of reusing the sealed identity");
  }
  if (!reused_sealed_identity.data_multihash.empty() &&
      reused_sealed_identity.data_multihash != closeout_commit_result.data_multihash) {
    return absl::FailedPreconditionError(
        "same-binding closeout mutated data_multihash instead of reusing the sealed identity");
  }
  return absl::OkStatus();
}

absl::Status publish_workspace_seal_binding_after_success(
    std::shared_ptr<store::components::IGlobalStoreClient> client,
    std::string_view workspace_assembly_id,
    std::string_view sealed_artifact_id,
    bool binding_subject_closeout,
    bool reused_sealed_identity) {
  if (!binding_subject_closeout || !reused_sealed_identity) {
    return absl::OkStatus();
  }
  if (workspace_assembly_id.empty()) {
    return absl::InvalidArgumentError("workspace_assembly_id is required");
  }
  if (sealed_artifact_id.empty()) {
    return absl::InvalidArgumentError("sealed_artifact_id is required");
  }
  if (client == nullptr || !client->is_connected()) {
    return absl::FailedPreconditionError("workspace seal binding publish requires Global Store client");
  }

  store::components::ArtifactBinding binding;
  binding.from_artifact_id = std::string(workspace_assembly_id);
  binding.to_artifact_id = std::string(sealed_artifact_id);
  binding.kind = tensorcast::global_store::v1::ARTIFACT_BINDING_KIND_SEAL;
  auto upsert_or = client->upsert_artifact_binding(binding);
  if (!upsert_or.ok()) {
    return upsert_or.status();
  }
  if (upsert_or->binding.to_artifact_id != binding.to_artifact_id) {
    return absl::FailedPreconditionError(
        "workspace seal binding artifact changed while publishing the finalized closeout result");
  }
  return absl::OkStatus();
}

} // namespace tensorcast::daemon::assembly_closeout_identity
