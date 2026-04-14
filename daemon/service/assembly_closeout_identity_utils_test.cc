// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/assembly_closeout_identity_utils.h"

#include <memory>
#include <optional>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "core/common/artifact_hash.h"
#include "core/store/testing/global_store_client_stub.h"
#include "core/store/testing/recording_global_store_client.h"

namespace {

using tensorcast::daemon::CommitLeaseResult;
namespace closeout_identity = tensorcast::daemon::assembly_closeout_identity;

std::string make_registration_index_json() {
  return R"({"alpha":[0,4,[1],[1],"torch.float32",0]})";
}

CommitLeaseResult make_reused_identity(std::string canonical_index_json) {
  auto index_multihash_or =
      tensorcast::common::compute_index_multihash(std::optional<std::string>(canonical_index_json), "");
  REQUIRE(index_multihash_or.ok());
  CommitLeaseResult result;
  result.artifact_id = "mi2:" + *index_multihash_or + ":bafkdata";
  result.index_multihash = *index_multihash_or;
  result.data_multihash = "bafkdata";
  result.canonical_index_json = std::move(canonical_index_json);
  result.schema_version = "v3";
  result.encoding = "json";
  return result;
}

class MismatchedWorkspaceSealBindingClient final : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  absl::StatusOr<tensorcast::store::components::ArtifactBindingResult> upsert_artifact_binding(
      const tensorcast::store::components::ArtifactBinding&) override {
    tensorcast::store::components::ArtifactBindingResult result;
    result.binding.from_artifact_id = "cgid:assembly-workspace";
    result.binding.to_artifact_id = "mi2:other-index:other-data";
    result.binding.kind = tensorcast::global_store::v1::ARTIFACT_BINDING_KIND_SEAL;
    result.created = false;
    return result;
  }
};

} // namespace

TEST_CASE(
    "resolve_registration_canonical_index_json prefers reused seal identity payload",
    "[daemon][assembly][closeout][identity]") {
  const CommitLeaseResult reused_identity = make_reused_identity(make_registration_index_json());
  CommitLeaseResult closeout_result = reused_identity;
  closeout_result.canonical_index_json = R"({"beta":[0,4,[1],[1],"torch.float32",0]})";

  auto resolved_or = closeout_identity::resolve_registration_canonical_index_json(reused_identity, closeout_result);

  REQUIRE(resolved_or.ok());
  CHECK(*resolved_or == reused_identity.canonical_index_json);
}

TEST_CASE(
    "resolve_registration_canonical_index_json fails closed when reused seal identity lacks canonical bytes",
    "[daemon][assembly][closeout][identity]") {
  CommitLeaseResult reused_identity = make_reused_identity(make_registration_index_json());
  reused_identity.canonical_index_json.clear();

  auto resolved_or =
      closeout_identity::resolve_registration_canonical_index_json(reused_identity, make_reused_identity("{}"));

  REQUIRE_FALSE(resolved_or.ok());
  CHECK(resolved_or.status().code() == absl::StatusCode::kFailedPrecondition);
  CHECK(resolved_or.status().message().find("canonical_index_json") != std::string::npos);
}

TEST_CASE(
    "validate_registration_canonical_index_matches_commit_result accepts converged reused identity",
    "[daemon][assembly][closeout][identity]") {
  const CommitLeaseResult commit_result = make_reused_identity(make_registration_index_json());

  auto status = closeout_identity::validate_registration_canonical_index_matches_commit_result(
      commit_result.canonical_index_json, commit_result);

  REQUIRE(status.ok());
}

TEST_CASE(
    "validate_registration_canonical_index_matches_commit_result rejects canonical index mismatch",
    "[daemon][assembly][closeout][identity]") {
  const CommitLeaseResult commit_result = make_reused_identity(make_registration_index_json());

  auto status = closeout_identity::validate_registration_canonical_index_matches_commit_result(
      R"({"beta":[0,4,[1],[1],"torch.float32",0]})", commit_result);

  REQUIRE_FALSE(status.ok());
  CHECK(status.code() == absl::StatusCode::kFailedPrecondition);
  CHECK(status.message().find("canonical_index_json") != std::string::npos);
}

TEST_CASE(
    "validate_reused_identity_matches_closeout_result rejects identity mutation",
    "[daemon][assembly][closeout][identity]") {
  const CommitLeaseResult reused_identity = make_reused_identity(make_registration_index_json());
  CommitLeaseResult closeout_result = reused_identity;
  closeout_result.artifact_id = "mi2:other-index:bafkdata";
  closeout_result.index_multihash = "other-index";

  auto status = closeout_identity::validate_reused_identity_matches_closeout_result(reused_identity, closeout_result);

  REQUIRE_FALSE(status.ok());
  CHECK(status.code() == absl::StatusCode::kFailedPrecondition);
}

TEST_CASE(
    "publish_workspace_seal_binding_after_success publishes reused binding-subject seals",
    "[daemon][assembly][closeout][identity]") {
  auto client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();

  auto status = closeout_identity::publish_workspace_seal_binding_after_success(
      client,
      "cgid:assembly-workspace",
      "mi2:bafkindex:bafkdata",
      /*binding_subject_closeout=*/true,
      /*reused_sealed_identity=*/true);

  REQUIRE(status.ok());
  REQUIRE(client->artifact_binding.has_value());
  CHECK(client->artifact_binding->from_artifact_id == "cgid:assembly-workspace");
  CHECK(client->artifact_binding->to_artifact_id == "mi2:bafkindex:bafkdata");
  CHECK(client->artifact_binding->kind == tensorcast::global_store::v1::ARTIFACT_BINDING_KIND_SEAL);
}

TEST_CASE(
    "publish_workspace_seal_binding_after_success is a no-op for non-reused or non-binding closeouts",
    "[daemon][assembly][closeout][identity]") {
  auto client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();

  auto non_binding_status = closeout_identity::publish_workspace_seal_binding_after_success(
      client,
      "cgid:assembly-workspace",
      "mi2:bafkindex:bafkdata",
      /*binding_subject_closeout=*/false,
      /*reused_sealed_identity=*/true);
  REQUIRE(non_binding_status.ok());
  CHECK_FALSE(client->artifact_binding.has_value());

  auto minted_status = closeout_identity::publish_workspace_seal_binding_after_success(
      client,
      "cgid:assembly-workspace",
      "mi2:bafkindex:bafkdata",
      /*binding_subject_closeout=*/true,
      /*reused_sealed_identity=*/false);
  REQUIRE(minted_status.ok());
  CHECK_FALSE(client->artifact_binding.has_value());
}

TEST_CASE(
    "publish_workspace_seal_binding_after_success fails closed on binding mismatch",
    "[daemon][assembly][closeout][identity]") {
  auto client = std::make_shared<MismatchedWorkspaceSealBindingClient>();

  auto status = closeout_identity::publish_workspace_seal_binding_after_success(
      client,
      "cgid:assembly-workspace",
      "mi2:bafkindex:bafkdata",
      /*binding_subject_closeout=*/true,
      /*reused_sealed_identity=*/true);

  REQUIRE_FALSE(status.ok());
  CHECK(status.code() == absl::StatusCode::kFailedPrecondition);
}
