// Copyright (c) 2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <string>

#include "absl/time/time.h"
#include "daemon/state/target_publication_registry.h"

namespace tensorcast::daemon {
namespace {

TargetPublicationRegistry::Record make_record(std::string publication_id) {
  tensorcast::common::v1::ArtifactSelection selection;
  selection.set_artifact_id("artifact-1");
  selection.set_view_id("view-1");
  selection.set_logical_layout_hash("layout-hash");
  selection.set_selection_hash("selection-hash");
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  byte_space.set_id("canonical");

  TargetPublicationRegistry::Record record;
  record.publication_id.value = std::move(publication_id);
  record.selection = selection;
  record.byte_space = byte_space;
  record.target_layout_hash = "target-layout-hash";
  record.device_uuid = "GPU-0";
  record.owner_pid = 1234;
  record.publication_subject_key =
      build_publication_subject_key(record.selection, record.byte_space, record.target_layout_hash, record.device_uuid);
  record.expires_at = absl::InfiniteFuture();
  return record;
}

TargetPublicationRegistry::StagedRecord make_staged_record(std::string publication_id) {
  TargetPublicationRegistry::StagedRecord staged;
  staged.publication = make_record(std::move(publication_id));
  staged.staging.transaction_id = "txn-1";
  staged.staging.version_set_id = "version-set-1";
  staged.staging.part_id = "part-0";
  staged.staging.staging_token = "staging-token-1";
  staged.staging.staging_epoch = 7;
  return staged;
}

} // namespace

TEST_CASE("staged target publication does not advance current subject", "[daemon][target][staged]") {
  TargetPublicationRegistry registry(TargetPublicationRegistry::Options{.capacity = 16, .ttl = absl::Minutes(5)});
  const auto current = registry.insert(make_record("publication-current"));
  auto staged = make_staged_record("publication-staged");
  staged.publication.publication_subject_key = current.publication_subject_key;
  staged.publication.selection = current.selection;
  staged.publication.byte_space = current.byte_space;

  staged = registry.insert_staged(std::move(staged));

  const auto staged_lookup = registry.lookup_staged("publication-staged", absl::UnixEpoch() + absl::Seconds(1));
  REQUIRE(staged_lookup.has_value());
  REQUIRE(staged_lookup->staging.transaction_id == "txn-1");

  const auto current_lookup = registry.lookup_current_for_subject(
      current.publication_subject_key.value,
      absl::UnixEpoch() + absl::Seconds(1),
      /*require_not_expired=*/true);
  REQUIRE(current_lookup.has_value());
  REQUIRE(current_lookup->publication_id.value == "publication-current");
  REQUIRE_FALSE(registry.is_current_for_subject(current.publication_subject_key.value, "publication-staged"));
}

TEST_CASE("staged target publication becomes current only after publish admission", "[daemon][target][staged]") {
  TargetPublicationRegistry registry(TargetPublicationRegistry::Options{.capacity = 16, .ttl = absl::Minutes(5)});
  const auto current = registry.insert(make_record("publication-current"));
  auto staged = make_staged_record("publication-staged");
  staged.publication.publication_subject_key = current.publication_subject_key;
  staged.publication.selection = current.selection;
  staged.publication.byte_space = current.byte_space;
  REQUIRE(registry.insert_staged(std::move(staged)).staging.publish_admitted == false);

  const auto published = registry.publish_staged("publication-staged");
  REQUIRE(published.has_value());
  REQUIRE(published->staging.publish_admitted);
  REQUIRE(registry.is_current_for_subject(current.publication_subject_key.value, "publication-staged"));
  REQUIRE_FALSE(registry.lookup_staged("publication-staged", absl::UnixEpoch() + absl::Seconds(1)).has_value());
}

TEST_CASE("staged target publication cleanup removes only unpublished staged records", "[daemon][target][staged]") {
  TargetPublicationRegistry registry(TargetPublicationRegistry::Options{.capacity = 16, .ttl = absl::Minutes(5)});
  auto keep_current = registry.insert(make_record("publication-current"));
  auto abort_a = make_staged_record("publication-abort-a");
  abort_a.publication.publication_subject_key = keep_current.publication_subject_key;
  auto abort_b = make_staged_record("publication-abort-b");
  abort_b.publication.publication_subject_key = keep_current.publication_subject_key;
  auto other_txn = make_staged_record("publication-other");
  other_txn.staging.transaction_id = "txn-2";
  other_txn.publication.publication_subject_key = keep_current.publication_subject_key;
  other_txn.publication.expires_at = absl::UnixEpoch() + absl::Seconds(60);
  REQUIRE(registry.insert_staged(std::move(abort_a)).staging.transaction_id == "txn-1");
  REQUIRE(registry.insert_staged(std::move(abort_b)).staging.transaction_id == "txn-1");
  REQUIRE(registry.insert_staged(std::move(other_txn)).staging.transaction_id == "txn-2");

  REQUIRE(registry.erase_staged_for_transaction("txn-1", 1) == 1);
  REQUIRE(registry.erase_staged_for_transaction("txn-1", 0) == 1);
  REQUIRE(registry.lookup_staged("publication-other", absl::UnixEpoch() + absl::Seconds(1)).has_value());
  REQUIRE(registry.is_current_for_subject(keep_current.publication_subject_key.value, "publication-current"));

  registry.prune(absl::UnixEpoch() + absl::Seconds(61));
  REQUIRE_FALSE(registry.lookup_staged("publication-other", absl::UnixEpoch() + absl::Seconds(61)).has_value());
  REQUIRE(registry.lookup("publication-current", absl::UnixEpoch() + absl::Seconds(61), true).has_value());
}

TEST_CASE("staged target publication prune works without current records", "[daemon][target][staged]") {
  TargetPublicationRegistry registry(TargetPublicationRegistry::Options{.capacity = 16, .ttl = absl::Minutes(5)});
  auto staged = make_staged_record("publication-staged-only");
  staged.publication.expires_at = absl::UnixEpoch() + absl::Seconds(10);
  REQUIRE(registry.insert_staged(std::move(staged)).staging.transaction_id == "txn-1");

  registry.prune(absl::UnixEpoch() + absl::Seconds(11));
  REQUIRE_FALSE(registry.lookup_staged("publication-staged-only", absl::UnixEpoch() + absl::Seconds(11)).has_value());
}

} // namespace tensorcast::daemon
