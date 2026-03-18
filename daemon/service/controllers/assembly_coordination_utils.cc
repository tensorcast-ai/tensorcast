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

bool require_live_contributions_until_readiness_cut(const v2::ContributionContractSnapshot& snapshot) {
  if (snapshot.required_slots_size() == 0 && snapshot.required_contributions_size() > 0) {
    return snapshot.require_live_contributions();
  }
  return snapshot.require_live_contributions_until_readiness_cut() || snapshot.require_live_contributions();
}

std::vector<v2::ContributionSlot> collect_required_slots(const v2::ContributionContractSnapshot& snapshot) {
  std::vector<v2::ContributionSlot> slots;
  if (snapshot.required_slots_size() > 0) {
    slots.reserve(static_cast<size_t>(snapshot.required_slots_size()));
    for (const auto& slot : snapshot.required_slots()) {
      slots.push_back(slot);
    }
  } else {
    slots.reserve(static_cast<size_t>(snapshot.required_contributions_size()));
    for (const auto& entry : snapshot.required_contributions()) {
      v2::ContributionSlot slot;
      slot.set_slot_key(contribution_slot_key(entry.contribution_kind(), entry.view_id()));
      slot.set_structural_view_id(contribution_structural_view_id(entry.contribution_kind(), entry.view_id()));
      slot.set_contribution_kind(entry.contribution_kind());
      slot.set_coverage_semantics(entry.coverage_semantics());
      slots.push_back(std::move(slot));
    }
  }

  std::sort(slots.begin(), slots.end(), [](const v2::ContributionSlot& lhs, const v2::ContributionSlot& rhs) {
    if (lhs.slot_key() != rhs.slot_key()) {
      return lhs.slot_key() < rhs.slot_key();
    }
    if (lhs.structural_view_id() != rhs.structural_view_id()) {
      return lhs.structural_view_id() < rhs.structural_view_id();
    }
    if (lhs.contribution_kind() != rhs.contribution_kind()) {
      return lhs.contribution_kind() < rhs.contribution_kind();
    }
    return lhs.coverage_semantics() < rhs.coverage_semantics();
  });
  slots.erase(
      std::unique(
          slots.begin(),
          slots.end(),
          [](const v2::ContributionSlot& lhs, const v2::ContributionSlot& rhs) {
            return lhs.slot_key() == rhs.slot_key() && lhs.structural_view_id() == rhs.structural_view_id() &&
                lhs.contribution_kind() == rhs.contribution_kind() &&
                lhs.coverage_semantics() == rhs.coverage_semantics();
          }),
      slots.end());
  return slots;
}

v2::CloseoutPolicySnapshot closeout_policy_for_hash(const v2::CloseoutPolicySnapshot& snapshot) {
  v2::CloseoutPolicySnapshot canonical;
  canonical.set_policy_json(snapshot.policy_json());
  canonical.set_source_policy_version(snapshot.source_policy_version());
  return canonical;
}

absl::Time timestamp_to_absl(const google::protobuf::Timestamp& ts) {
  const int64_t nanos = ts.seconds() * 1000000000LL + ts.nanos();
  return absl::UnixEpoch() + absl::Nanoseconds(nanos);
}

} // namespace

std::string contribution_slot_key(v2::BindingContributionKind contribution_kind, std::string_view structural_view_id) {
  if (contribution_kind == v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL) {
    return std::string(kCanonicalFullContributionSlotKey);
  }
  return std::string(structural_view_id);
}

std::string contribution_structural_view_id(
    v2::BindingContributionKind contribution_kind,
    std::string_view structural_view_id) {
  if (contribution_kind == v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL) {
    return std::string();
  }
  return std::string(structural_view_id);
}

v2::ContributionContractSnapshot build_phase1_contribution_contract(
    std::string_view layout_id,
    const google::protobuf::RepeatedPtrField<std::string>& expected_view_ids,
    bool require_live_contributions_until_readiness_cut) {
  v2::ContributionContractSnapshot snapshot;
  snapshot.set_layout_id(std::string(layout_id));
  snapshot.set_require_live_contributions_until_readiness_cut(require_live_contributions_until_readiness_cut);
  snapshot.set_require_live_contributions(require_live_contributions_until_readiness_cut);
  const auto sorted_ids = sorted_expected_view_ids(expected_view_ids);
  if (sorted_ids.empty()) {
    auto* slot = snapshot.add_required_slots();
    slot->set_slot_key(std::string(kCanonicalFullContributionSlotKey));
    slot->set_contribution_kind(v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL);
    slot->set_coverage_semantics(std::string(kCanonicalFullCoverageSemantics));

    auto* compat = snapshot.add_required_contributions();
    compat->set_view_id(std::string(kCanonicalFullContributionSlotKey));
    compat->set_contribution_kind(v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL);
    compat->set_coverage_semantics(std::string(kCanonicalFullCoverageSemantics));
    return snapshot;
  }

  for (const auto& view_id : sorted_ids) {
    auto* slot = snapshot.add_required_slots();
    slot->set_slot_key(view_id);
    slot->set_structural_view_id(view_id);
    slot->set_contribution_kind(v2::BINDING_CONTRIBUTION_KIND_PIECE_PARTIAL);
    slot->set_coverage_semantics(std::string(kPiecePartialCoverageSemantics));

    auto* compat = snapshot.add_required_contributions();
    compat->set_view_id(view_id);
    compat->set_contribution_kind(v2::BINDING_CONTRIBUTION_KIND_PIECE_PARTIAL);
    compat->set_coverage_semantics(std::string(kPiecePartialCoverageSemantics));
  }
  return snapshot;
}

v2::ContributionContractSnapshot canonicalize_contribution_contract(const v2::ContributionContractSnapshot& snapshot) {
  v2::ContributionContractSnapshot canonical;
  canonical.set_layout_id(snapshot.layout_id());
  canonical.set_require_live_contributions_until_readiness_cut(
      require_live_contributions_until_readiness_cut(snapshot));
  canonical.set_require_live_contributions(require_live_contributions_until_readiness_cut(snapshot));
  for (const auto& slot : collect_required_slots(snapshot)) {
    *canonical.add_required_slots() = slot;
    auto* compat = canonical.add_required_contributions();
    compat->set_view_id(
        slot.contribution_kind() == v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL ? slot.slot_key()
                                                                                 : slot.structural_view_id());
    compat->set_contribution_kind(slot.contribution_kind());
    compat->set_coverage_semantics(slot.coverage_semantics());
  }
  return canonical;
}

std::string compute_contribution_contract_hash(const v2::ContributionContractSnapshot& snapshot) {
  const v2::ContributionContractSnapshot canonical = canonicalize_contribution_contract(snapshot);
  std::string payload;
  canonical.SerializeToString(&payload);
  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  const std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(bytes);
  return tensorcast::common::multibase_multihash_sha256(digest);
}

v2::CloseoutPolicySnapshot canonicalize_closeout_policy_snapshot(const v2::CloseoutPolicySnapshot& snapshot) {
  v2::CloseoutPolicySnapshot canonical = closeout_policy_for_hash(snapshot);
  canonical.set_closeout_policy_hash(compute_closeout_policy_hash(snapshot));
  return canonical;
}

std::string compute_closeout_policy_hash(const v2::CloseoutPolicySnapshot& snapshot) {
  const v2::CloseoutPolicySnapshot canonical = closeout_policy_for_hash(snapshot);
  std::string payload;
  canonical.SerializeToString(&payload);
  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  const std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(bytes);
  return tensorcast::common::multibase_multihash_sha256(digest);
}

v2::AssemblyAttemptSpec canonicalize_attempt_spec(const v2::AssemblyAttemptSpec& spec) {
  v2::AssemblyAttemptSpec canonical;
  canonical.set_assembly_id(spec.assembly_id());
  canonical.set_layout_id(spec.layout_id());
  *canonical.mutable_contribution_contract() = canonicalize_contribution_contract(spec.contribution_contract());
  *canonical.mutable_closeout_policy() = canonicalize_closeout_policy_snapshot(spec.closeout_policy());
  canonical.set_contribution_contract_hash(compute_contribution_contract_hash(canonical.contribution_contract()));
  canonical.set_attempt_spec_hash(compute_attempt_spec_hash(spec));
  return canonical;
}

std::string compute_attempt_spec_hash(const v2::AssemblyAttemptSpec& spec) {
  v2::AssemblyAttemptSpec canonical;
  canonical.set_assembly_id(spec.assembly_id());
  canonical.set_layout_id(spec.layout_id());
  *canonical.mutable_contribution_contract() = canonicalize_contribution_contract(spec.contribution_contract());
  *canonical.mutable_closeout_policy() = canonicalize_closeout_policy_snapshot(spec.closeout_policy());
  canonical.set_contribution_contract_hash(compute_contribution_contract_hash(canonical.contribution_contract()));
  canonical.clear_attempt_spec_hash();

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
    std::string_view structural_view_id) {
  if (!require_live_contributions_until_readiness_cut(snapshot)) {
    return absl::FailedPreconditionError("assembly attempt does not accept binding-backed live contributions");
  }
  const std::string expected_slot_key = contribution_slot_key(contribution_kind, structural_view_id);
  const std::string expected_structural_view_id =
      contribution_structural_view_id(contribution_kind, structural_view_id);
  for (const auto& slot : collect_required_slots(snapshot)) {
    if (slot.slot_key() != expected_slot_key) {
      continue;
    }
    if (slot.contribution_kind() != contribution_kind) {
      continue;
    }
    if (slot.structural_view_id() != expected_structural_view_id) {
      continue;
    }
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(
      absl::StrCat("binding contribution is not part of the snapped contract for slot_key=", expected_slot_key));
}

absl::Status validate_live_required_contributions(
    const v2::ContributionContractSnapshot& snapshot,
    absl::Span<const store::components::AssemblyContributionInfo> contributions,
    const absl::flat_hash_set<std::string>& active_contributor_identities,
    absl::Time now) {
  if (!require_live_contributions_until_readiness_cut(snapshot)) {
    return absl::OkStatus();
  }

  absl::flat_hash_set<std::string> live_slot_keys;
  live_slot_keys.reserve(contributions.size());
  for (const auto& contribution : contributions) {
    if (!assembly_contribution_is_live(contribution, now, &active_contributor_identities)) {
      continue;
    }
    if (!contribution.view_id.empty()) {
      live_slot_keys.insert(contribution.view_id);
    }
  }

  for (const auto& slot : collect_required_slots(snapshot)) {
    if (live_slot_keys.contains(slot.slot_key())) {
      continue;
    }
    if (slot.contribution_kind() == v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL) {
      return absl::FailedPreconditionError("required canonical_full contribution missing from live contributor set");
    }
    return absl::FailedPreconditionError(
        absl::StrCat("required slot_key missing from live contributor set: ", slot.slot_key()));
  }
  return absl::OkStatus();
}

} // namespace tensorcast::daemon::assembly_coordination
