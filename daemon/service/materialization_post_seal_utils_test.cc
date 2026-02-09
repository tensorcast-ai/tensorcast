// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_post_seal_utils.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/testing/global_store_client_stub.h"
#include "tensorcast/layout/v1/layout.pb.h"

namespace tensorcast::daemon {

namespace {

using materialization_post_seal::check_post_seal_view_reuse_safe;
using materialization_post_seal::compute_view_meta_digest;

store::components::CanonicalRange make_range(uint64_t offset, uint64_t length) {
  store::components::CanonicalRange range;
  range.offset = offset;
  range.length = length;
  return range;
}

tensorcast::layout::v1::LayoutSpecRecord make_layout(
    std::string layout_id,
    std::string proof_schema_version,
    tensorcast::layout::v1::OverlapMode overlap_mode) {
  tensorcast::layout::v1::LayoutSpecRecord record;
  record.set_layout_id(std::move(layout_id));
  auto* tensor_policy = &(*record.mutable_layout()->mutable_tensors())["tensor_a"];
  tensor_policy->set_overlap_mode(overlap_mode);
  record.mutable_layout()->set_proof_schema_version(std::move(proof_schema_version));
  return record;
}

class PostSealClientStub final : public store::testing::GlobalStoreClientStub {
 public:
  std::vector<std::string> layout_ids;
  absl::flat_hash_map<std::string, tensorcast::layout::v1::LayoutSpecRecord> layout_specs;
  absl::flat_hash_map<std::string, bool> schema_match;
  std::vector<tensorcast::global_store::v1::CheckProofCommitmentsMatchRequest> check_requests;

  absl::StatusOr<std::vector<std::string>> list_artifact_layouts(std::string_view) override {
    return layout_ids;
  }

  absl::StatusOr<tensorcast::layout::v1::LayoutSpecRecord> get_layout_spec(std::string_view layout_id) override {
    auto it = layout_specs.find(std::string(layout_id));
    if (it == layout_specs.end()) {
      return absl::NotFoundError("layout spec not found");
    }
    return it->second;
  }

  absl::StatusOr<tensorcast::global_store::v1::CheckProofCommitmentsMatchResponse> check_proof_commitments_match(
      const tensorcast::global_store::v1::CheckProofCommitmentsMatchRequest& req) override {
    check_requests.push_back(req);
    tensorcast::global_store::v1::CheckProofCommitmentsMatchResponse resp;
    auto it = schema_match.find(req.proof_schema_version());
    resp.set_match(it == schema_match.end() ? true : it->second);
    return resp;
  }
};

} // namespace

TEST_CASE("post-seal reuse returns true when layout has no replicated tensors") {
  PostSealClientStub client;
  client.layout_ids = {"layout-a"};
  client.layout_specs["layout-a"] =
      make_layout("layout-a", "", tensorcast::layout::v1::OverlapMode::OVERLAP_MODE_DISJOINT);

  auto safe_or = check_post_seal_view_reuse_safe(client, "assembly-1", "mi2-1");

  REQUIRE(safe_or.ok());
  REQUIRE(*safe_or);
  REQUIRE(client.check_requests.empty());
}

TEST_CASE("post-seal reuse fails when replicated tensor misses proof schema version") {
  PostSealClientStub client;
  client.layout_ids = {"layout-a"};
  client.layout_specs["layout-a"] =
      make_layout("layout-a", "", tensorcast::layout::v1::OverlapMode::OVERLAP_MODE_REPLICATE_EQUAL);

  auto safe_or = check_post_seal_view_reuse_safe(client, "assembly-1", "mi2-1");

  REQUIRE_FALSE(safe_or.ok());
  REQUIRE(safe_or.status().code() == absl::StatusCode::kFailedPrecondition);
}

TEST_CASE("post-seal reuse returns false when proof commitments mismatch") {
  PostSealClientStub client;
  client.layout_ids = {"layout-a"};
  client.layout_specs["layout-a"] =
      make_layout("layout-a", "proof-v1", tensorcast::layout::v1::OverlapMode::OVERLAP_MODE_REPLICATE_EQUAL);
  client.schema_match["proof-v1"] = false;

  auto safe_or = check_post_seal_view_reuse_safe(client, "assembly-1", "mi2-1");

  REQUIRE(safe_or.ok());
  REQUIRE_FALSE(*safe_or);
  REQUIRE(client.check_requests.size() == 1);
  REQUIRE(client.check_requests[0].tensor_names_size() == 1);
  REQUIRE(client.check_requests[0].tensor_names(0) == "tensor_a");
}

TEST_CASE("view meta digest is stable across canonical range ordering") {
  store::components::ViewInfo view_a;
  view_a.view_id = "view-1";
  view_a.view_size_bytes = 64;
  view_a.canonical_size_bytes = 128;
  view_a.canonical_bytes_covered = 64;
  view_a.canonical_ranges = {make_range(32, 16), make_range(0, 8), make_range(16, 8)};

  auto view_b = view_a;
  view_b.canonical_ranges = {make_range(16, 8), make_range(32, 16), make_range(0, 8)};

  const auto digest_a = compute_view_meta_digest(view_a);
  const auto digest_b = compute_view_meta_digest(view_b);

  REQUIRE(digest_a == digest_b);
}

} // namespace tensorcast::daemon
