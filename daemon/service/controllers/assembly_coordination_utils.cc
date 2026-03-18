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

std::vector<v2::AssemblyRequirement> collect_requirements(const v2::AssemblyRequirementSetRef& requirements) {
  std::vector<v2::AssemblyRequirement> out;
  out.reserve(static_cast<size_t>(requirements.inline_requirements_size()));
  for (const auto& requirement : requirements.inline_requirements()) {
    out.push_back(requirement);
  }
  std::sort(out.begin(), out.end(), [](const v2::AssemblyRequirement& lhs, const v2::AssemblyRequirement& rhs) {
    if (lhs.slot_id() != rhs.slot_id()) {
      return lhs.slot_id() < rhs.slot_id();
    }
    if (lhs.target().kind() != rhs.target().kind()) {
      return lhs.target().kind() < rhs.target().kind();
    }
    if (lhs.target().structural_view_id() != rhs.target().structural_view_id()) {
      return lhs.target().structural_view_id() < rhs.target().structural_view_id();
    }
    return lhs.coverage_contract() < rhs.coverage_contract();
  });
  out.erase(
      std::unique(
          out.begin(),
          out.end(),
          [](const v2::AssemblyRequirement& lhs, const v2::AssemblyRequirement& rhs) {
            return lhs.slot_id() == rhs.slot_id() && lhs.target().kind() == rhs.target().kind() &&
                lhs.target().structural_view_id() == rhs.target().structural_view_id() &&
                lhs.coverage_contract() == rhs.coverage_contract();
          }),
      out.end());
  return out;
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

v2::AssemblyRequirementSetRef build_phase1_requirement_set(
    const google::protobuf::RepeatedPtrField<std::string>& expected_view_ids) {
  v2::AssemblyRequirementSetRef requirements;
  requirements.set_carrier_form("inline");
  const auto sorted_ids = sorted_expected_view_ids(expected_view_ids);
  if (sorted_ids.empty()) {
    auto* requirement = requirements.add_inline_requirements();
    requirement->set_slot_id(std::string(kCanonicalFullContributionSlotKey));
    requirement->mutable_target()->set_kind(v2::ASSEMBLY_TARGET_KIND_CANONICAL_LAYOUT);
    requirement->set_coverage_contract(std::string(kCanonicalFullCoverageSemantics));
    requirements.set_requirement_count(1);
    return requirements;
  }
  for (const auto& view_id : sorted_ids) {
    auto* requirement = requirements.add_inline_requirements();
    requirement->set_slot_id(view_id);
    requirement->mutable_target()->set_kind(v2::ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW);
    requirement->mutable_target()->set_structural_view_id(view_id);
    requirement->set_coverage_contract(std::string(kPiecePartialCoverageSemantics));
  }
  requirements.set_requirement_count(requirements.inline_requirements_size());
  return requirements;
}

v2::AssemblyRequirementSetRef canonicalize_requirement_set(const v2::AssemblyRequirementSetRef& requirements) {
  v2::AssemblyRequirementSetRef canonical;
  canonical.set_carrier_form(requirements.carrier_form().empty() ? "inline" : requirements.carrier_form());
  for (const auto& requirement : collect_requirements(requirements)) {
    *canonical.add_inline_requirements() = requirement;
  }
  canonical.set_requirement_count(canonical.inline_requirements_size());
  canonical.set_requirements_digest(compute_requirement_set_digest(canonical));
  return canonical;
}

std::string compute_requirement_set_digest(const v2::AssemblyRequirementSetRef& requirements) {
  v2::AssemblyRequirementSetRef canonical;
  canonical.set_carrier_form(requirements.carrier_form().empty() ? "inline" : requirements.carrier_form());
  for (const auto& requirement : collect_requirements(requirements)) {
    *canonical.add_inline_requirements() = requirement;
  }
  canonical.set_requirement_count(canonical.inline_requirements_size());
  canonical.clear_requirements_digest();

  std::string payload;
  canonical.SerializeToString(&payload);
  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  const std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(bytes);
  return tensorcast::common::multibase_multihash_sha256(digest);
}

v2::AssemblyReadinessPolicy canonicalize_readiness_policy(const v2::AssemblyReadinessPolicy& policy) {
  v2::AssemblyReadinessPolicy canonical;
  canonical.set_contributor_liveness_mode(
      policy.contributor_liveness_mode() == v2::ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_UNSPECIFIED
          ? v2::ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_REQUIRE_LIVE_UNTIL_CUT
          : policy.contributor_liveness_mode());
  return canonical;
}

v2::AssemblyCloseoutContract canonicalize_closeout_contract(const v2::AssemblyCloseoutContract& contract) {
  v2::AssemblyCloseoutContract canonical;
  canonical.set_kind(
      contract.kind() == v2::ASSEMBLY_CLOSEOUT_KIND_UNSPECIFIED ? v2::ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY
                                                                : contract.kind());
  canonical.set_source_version_key(contract.source_version_key());
  canonical.set_serving_version_key(contract.serving_version_key());
  canonical.set_serving_artifact_id(contract.serving_artifact_id());
  canonical.set_serving_manifest_ref(contract.serving_manifest_ref());
  canonical.set_closeout_contract_digest(compute_closeout_contract_digest(canonical));
  return canonical;
}

std::string compute_closeout_contract_digest(const v2::AssemblyCloseoutContract& contract) {
  v2::AssemblyCloseoutContract canonical;
  canonical.set_kind(
      contract.kind() == v2::ASSEMBLY_CLOSEOUT_KIND_UNSPECIFIED ? v2::ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY
                                                                : contract.kind());
  canonical.set_source_version_key(contract.source_version_key());
  canonical.set_serving_version_key(contract.serving_version_key());
  canonical.set_serving_artifact_id(contract.serving_artifact_id());
  canonical.set_serving_manifest_ref(contract.serving_manifest_ref());
  canonical.clear_closeout_contract_digest();

  std::string payload;
  canonical.SerializeToString(&payload);
  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  const std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(bytes);
  return tensorcast::common::multibase_multihash_sha256(digest);
}

absl::Status validate_dependency_ready_closeout_contract(const v2::AssemblyCloseoutContract& contract) {
  const auto canonical = canonicalize_closeout_contract(contract);
  if (canonical.kind() != v2::ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY) {
    return absl::UnimplementedError(
        "only source_publish_only closeout contracts are dependency-ready in the current execution wave");
  }
  if (!canonical.serving_version_key().empty() || !canonical.serving_artifact_id().empty() ||
      !canonical.serving_manifest_ref().empty()) {
    return absl::InvalidArgumentError(
        "source_publish_only closeout contracts may not set serving_version_key, serving_artifact_id, or "
        "serving_manifest_ref");
  }
  return absl::OkStatus();
}

v2::AssemblyAttemptIntent canonicalize_attempt_intent(const v2::AssemblyAttemptIntent& intent) {
  v2::AssemblyAttemptIntent canonical;
  canonical.set_layout_id(intent.layout_id());
  *canonical.mutable_requirements() = canonicalize_requirement_set(intent.requirements());
  *canonical.mutable_readiness_policy() = canonicalize_readiness_policy(intent.readiness_policy());
  *canonical.mutable_closeout_contract() = canonicalize_closeout_contract(intent.closeout_contract());
  canonical.set_attempt_intent_digest(compute_attempt_intent_digest(canonical));
  return canonical;
}

std::string compute_attempt_intent_digest(const v2::AssemblyAttemptIntent& intent) {
  v2::AssemblyAttemptIntent canonical;
  canonical.set_layout_id(intent.layout_id());
  *canonical.mutable_requirements() = canonicalize_requirement_set(intent.requirements());
  *canonical.mutable_readiness_policy() = canonicalize_readiness_policy(intent.readiness_policy());
  *canonical.mutable_closeout_contract() = canonicalize_closeout_contract(intent.closeout_contract());
  canonical.clear_attempt_intent_digest();

  std::string payload;
  canonical.SerializeToString(&payload);
  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  const std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(bytes);
  return tensorcast::common::multibase_multihash_sha256(digest);
}

std::string contribution_structural_view_id(
    v2::BindingContributionKind contribution_kind,
    std::string_view structural_view_id) {
  if (contribution_kind == v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL) {
    return std::string();
  }
  return std::string(structural_view_id);
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

bool operation_lease_is_live(const tensorcast::operation::v1::GetOperationResponse& operation, absl::Time now) {
  if (operation.lease_generation() == 0 || operation.lease_owner().empty() || !operation.has_lease_expires_at()) {
    return false;
  }
  return timestamp_to_absl(operation.lease_expires_at()) > now;
}

bool operation_allows_contributions(const tensorcast::operation::v1::GetOperationResponse& operation) {
  return operation.status().state() == tensorcast::operation::v1::OPERATION_STATE_PENDING;
}

bool slot_occupancy_is_live(
    const store::components::AssemblySlotOccupancyInfo& occupancy,
    absl::Time now,
    const absl::flat_hash_set<std::string>* active_contributor_identities) {
  const bool lease_live = occupancy.state == "accepted" && occupancy.lease_expires_at.has_value() &&
      occupancy.lease_expires_at.value() > now;
  if (!lease_live) {
    return false;
  }
  if (active_contributor_identities == nullptr) {
    return true;
  }
  if (occupancy.contributor_daemon_id.empty()) {
    return false;
  }
  return active_contributor_identities->contains(occupancy.contributor_daemon_id);
}

absl::Status validate_binding_requirement_entry(
    const v2::AssemblyRequirementSetRef& requirements,
    v2::BindingContributionKind contribution_kind,
    std::string_view structural_view_id) {
  const std::string expected_slot_id = contribution_slot_key(contribution_kind, structural_view_id);
  for (const auto& requirement : collect_requirements(requirements)) {
    if (requirement.slot_id() != expected_slot_id) {
      continue;
    }
    if (contribution_kind == v2::BINDING_CONTRIBUTION_KIND_PIECE_PARTIAL) {
      if (requirement.target().kind() != v2::ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW) {
        continue;
      }
      if (requirement.target().structural_view_id() != structural_view_id) {
        continue;
      }
    } else if (contribution_kind == v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL) {
      if (requirement.target().kind() != v2::ASSEMBLY_TARGET_KIND_CANONICAL_LAYOUT) {
        continue;
      }
    } else {
      continue;
    }
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(
      absl::StrCat("binding contribution is not part of the snapped requirements for slot_id=", expected_slot_id));
}

absl::Status validate_live_required_slot_occupancies(
    const v2::AssemblyRequirementSetRef& requirements,
    const v2::AssemblyReadinessPolicy& readiness_policy,
    absl::Span<const store::components::AssemblySlotOccupancyInfo> occupancies,
    const absl::flat_hash_set<std::string>& active_contributor_identities,
    absl::Time now) {
  if (canonicalize_readiness_policy(readiness_policy).contributor_liveness_mode() !=
      v2::ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_REQUIRE_LIVE_UNTIL_CUT) {
    return absl::OkStatus();
  }

  absl::flat_hash_set<std::string> live_slot_ids;
  live_slot_ids.reserve(occupancies.size());
  for (const auto& occupancy : occupancies) {
    if (!slot_occupancy_is_live(occupancy, now, &active_contributor_identities)) {
      continue;
    }
    live_slot_ids.insert(occupancy.slot_id);
  }

  for (const auto& requirement : collect_requirements(requirements)) {
    if (live_slot_ids.contains(requirement.slot_id())) {
      continue;
    }
    return absl::FailedPreconditionError(
        absl::StrCat("required slot_id missing from live contributor set: ", requirement.slot_id()));
  }
  return absl::OkStatus();
}

} // namespace tensorcast::daemon::assembly_coordination
