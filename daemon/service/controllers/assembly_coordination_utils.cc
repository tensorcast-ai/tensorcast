// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/assembly_coordination_utils.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "google/protobuf/util/time_util.h"

namespace tensorcast::daemon::assembly_coordination {
namespace {

std::vector<std::string> sorted_expected_view_ids(
    const google::protobuf::RepeatedPtrField<std::string>& expected_view_ids) {
  std::vector<std::string> sorted_ids;
  sorted_ids.reserve(static_cast<size_t>(expected_view_ids.size()));
  for (const auto& view_id : expected_view_ids) {
    if (!view_id.empty()) {
      sorted_ids.push_back(view_id);
    }
  }
  std::sort(sorted_ids.begin(), sorted_ids.end());
  sorted_ids.erase(std::unique(sorted_ids.begin(), sorted_ids.end()), sorted_ids.end());
  return sorted_ids;
}

std::vector<v2::ContributionContractEntry> sorted_contract_entries(const v2::ContributionContractSnapshot& snapshot) {
  std::vector<v2::ContributionContractEntry> entries;
  entries.reserve(static_cast<size_t>(snapshot.required_contributions_size()));
  for (const auto& entry : snapshot.required_contributions()) {
    entries.push_back(entry);
  }
  std::sort(
      entries.begin(),
      entries.end(),
      [](const v2::ContributionContractEntry& lhs, const v2::ContributionContractEntry& rhs) {
        if (lhs.view_id() != rhs.view_id()) {
          return lhs.view_id() < rhs.view_id();
        }
        if (lhs.contribution_kind() != rhs.contribution_kind()) {
          return lhs.contribution_kind() < rhs.contribution_kind();
        }
        return lhs.coverage_semantics() < rhs.coverage_semantics();
      });
  return entries;
}

v2::ContributionContractSnapshot canonicalize_snapshot(const v2::ContributionContractSnapshot& snapshot) {
  v2::ContributionContractSnapshot canonical;
  canonical.set_layout_id(snapshot.layout_id());
  canonical.set_require_live_contributions(snapshot.require_live_contributions());
  for (const auto& entry : sorted_contract_entries(snapshot)) {
    *canonical.add_required_contributions() = entry;
  }
  return canonical;
}

absl::Time timestamp_to_absl(const google::protobuf::Timestamp& ts) {
  const int64_t nanos = ts.seconds() * 1000000000LL + ts.nanos();
  return absl::UnixEpoch() + absl::Nanoseconds(nanos);
}

} // namespace

std::string contribution_slot_view_id(v2::BindingContributionKind contribution_kind, std::string_view view_id) {
  if (contribution_kind == v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL) {
    return std::string(kCanonicalFullContributionViewId);
  }
  return std::string(view_id);
}

v2::ContributionContractSnapshot build_phase1_contribution_contract(
    std::string_view layout_id,
    const google::protobuf::RepeatedPtrField<std::string>& expected_view_ids,
    bool require_live_contributions) {
  v2::ContributionContractSnapshot snapshot;
  snapshot.set_layout_id(std::string(layout_id));
  snapshot.set_require_live_contributions(require_live_contributions);
  const auto sorted_ids = sorted_expected_view_ids(expected_view_ids);
  if (sorted_ids.empty()) {
    auto* entry = snapshot.add_required_contributions();
    entry->set_view_id(std::string(kCanonicalFullContributionViewId));
    entry->set_contribution_kind(v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL);
    entry->set_coverage_semantics(std::string(kCanonicalFullCoverageSemantics));
    return snapshot;
  }
  for (const auto& view_id : sorted_ids) {
    auto* entry = snapshot.add_required_contributions();
    entry->set_view_id(view_id);
    entry->set_contribution_kind(v2::BINDING_CONTRIBUTION_KIND_PIECE_PARTIAL);
    entry->set_coverage_semantics(std::string(kPiecePartialCoverageSemantics));
  }
  return snapshot;
}

std::string compute_contribution_contract_hash(const v2::ContributionContractSnapshot& snapshot) {
  const v2::ContributionContractSnapshot canonical = canonicalize_snapshot(snapshot);
  std::string payload;
  canonical.SerializeToString(&payload);
  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  const std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(bytes);
  return tensorcast::common::multibase_multihash_sha256(digest);
}

absl::StatusOr<absl::flat_hash_set<std::string>> list_active_contributor_identities(
    const std::shared_ptr<store::components::IGlobalStoreClient>& client) {
  absl::flat_hash_set<std::string> identities;
  if (!client || !client->is_connected()) {
    return identities;
  }
  auto active_or = client->list_active_worker_identities(/*include_unavailable=*/true);
  if (!active_or.ok()) {
    return active_or.status();
  }
  for (const auto& identity : *active_or) {
    if (!identity.empty()) {
      identities.insert(identity);
    }
  }
  return identities;
}

bool assembly_contribution_is_live(
    const store::components::AssemblyContributionInfo& contribution,
    absl::Time now,
    const absl::flat_hash_set<std::string>* active_contributor_identities) {
  const bool lease_live = contribution.state == "accepted" && contribution.lease_expires_at.has_value() &&
      contribution.lease_expires_at.value() > now;
  if (!lease_live) {
    return false;
  }
  if (active_contributor_identities == nullptr) {
    return true;
  }
  if (contribution.contributor_daemon_id.empty()) {
    return false;
  }
  return active_contributor_identities->contains(contribution.contributor_daemon_id);
}

bool operation_lease_is_live(const tensorcast::operation::v1::GetOperationResponse& operation, absl::Time now) {
  if (operation.lease_generation() == 0 || operation.lease_owner().empty() || !operation.has_lease_expires_at()) {
    return false;
  }
  return timestamp_to_absl(operation.lease_expires_at()) > now;
}

bool operation_allows_contributions(const tensorcast::operation::v1::GetOperationResponse& operation) {
  return operation.status().state() == tensorcast::operation::v1::OPERATION_STATE_PENDING;
}

absl::Status validate_contribution_contract_entry(
    const v2::ContributionContractSnapshot& snapshot,
    v2::BindingContributionKind contribution_kind,
    std::string_view view_id) {
  if (!snapshot.require_live_contributions()) {
    return absl::FailedPreconditionError("assembly attempt does not accept binding-backed live contributions");
  }
  const std::string expected_view_id = contribution_slot_view_id(contribution_kind, view_id);
  for (const auto& entry : snapshot.required_contributions()) {
    if (entry.view_id() == expected_view_id && entry.contribution_kind() == contribution_kind) {
      return absl::OkStatus();
    }
  }
  return absl::FailedPreconditionError(
      absl::StrCat("binding contribution is not part of the snapped contract for view_id=", expected_view_id));
}

absl::Status validate_live_required_contributions(
    const v2::ContributionContractSnapshot& snapshot,
    absl::Span<const store::components::AssemblyContributionInfo> contributions,
    const absl::flat_hash_set<std::string>& active_contributor_identities,
    absl::Time now) {
  if (!snapshot.require_live_contributions()) {
    return absl::OkStatus();
  }

  absl::flat_hash_set<std::string> live_view_ids;
  live_view_ids.reserve(contributions.size());
  for (const auto& contribution : contributions) {
    if (!assembly_contribution_is_live(contribution, now, &active_contributor_identities)) {
      continue;
    }
    if (!contribution.view_id.empty()) {
      live_view_ids.insert(contribution.view_id);
    }
  }

  for (const auto& entry : snapshot.required_contributions()) {
    const std::string row_view_id = contribution_slot_view_id(entry.contribution_kind(), entry.view_id());
    if (live_view_ids.contains(row_view_id)) {
      continue;
    }
    if (entry.contribution_kind() == v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL) {
      return absl::FailedPreconditionError("required canonical_full contribution missing from live contributor set");
    }
    return absl::FailedPreconditionError(
        absl::StrCat("required expected_view_id missing from live contributor set: ", row_view_id));
  }
  return absl::OkStatus();
}

} // namespace tensorcast::daemon::assembly_coordination
