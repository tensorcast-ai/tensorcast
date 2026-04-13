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
#include "daemon/service/controllers/serving_artifact_manifest_utils.h"
#include "google/protobuf/util/time_util.h"

namespace tensorcast::daemon::assembly_coordination {
namespace pubv1 = tensorcast::publication::v1;

namespace {

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

absl::StatusOr<std::string_view> expected_coverage_contract_for_binding_contribution(
    v2::BindingContributionKind contribution_kind) {
  switch (contribution_kind) {
    case v2::BINDING_CONTRIBUTION_KIND_PIECE_PARTIAL:
      return kPpStructuralViewCoverageContract;
    case v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL:
      return kCanonicalFullCoverageContract;
    case v2::BINDING_CONTRIBUTION_KIND_UNSPECIFIED:
    default:
      return absl::InvalidArgumentError("unsupported binding contribution kind");
  }
}

pubv1::BindingValueRef canonicalize_binding_value_ref(const pubv1::BindingValueRef& ref) {
  pubv1::BindingValueRef canonical;
  canonical.set_binding_id(ref.binding_id());
  canonical.set_binding_layout_id(ref.binding_layout_id());
  canonical.set_binding_value_id(ref.binding_value_id());
  canonical.set_seal_generation(ref.seal_generation());
  return canonical;
}

pubv1::ServingPublicationSubject canonicalize_serving_publication_subject(
    const v2::RepresentationPublishContract& contract) {
  pubv1::ServingPublicationSubject canonical;
  if (contract.has_subject()) {
    const auto& subject = contract.subject();
    switch (subject.ref_case()) {
      case pubv1::ServingPublicationSubject::kServingArtifactId:
        canonical.set_serving_artifact_id(subject.serving_artifact_id());
        break;
      case pubv1::ServingPublicationSubject::kBindingValue:
        *canonical.mutable_binding_value() = canonicalize_binding_value_ref(subject.binding_value());
        break;
      case pubv1::ServingPublicationSubject::REF_NOT_SET:
        break;
    }
  }
  return canonical;
}

v2::RepresentationPublishContract canonicalize_representation_publish_contract(
    const v2::RepresentationPublishContract& contract) {
  v2::RepresentationPublishContract canonical;
  const auto subject = canonicalize_serving_publication_subject(contract);
  if (subject.ref_case() != pubv1::ServingPublicationSubject::REF_NOT_SET) {
    *canonical.mutable_subject() = subject;
  }
  canonical.set_serving_manifest_ref(contract.serving_manifest_ref());
  canonical.set_representation_contract_hash(contract.representation_contract_hash());
  canonical.set_serving_build_digest(contract.serving_build_digest());
  canonical.set_serving_build_digest_version(contract.serving_build_digest_version());
  return canonical;
}

} // namespace

std::string contribution_slot_key(v2::BindingContributionKind contribution_kind, std::string_view structural_view_id) {
  if (contribution_kind == v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL) {
    return std::string(kCanonicalFullContributionSlotKey);
  }
  return std::string(structural_view_id);
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

absl::StatusOr<std::string> canonicalize_coverage_contract(std::string_view coverage_contract) {
  if (coverage_contract == kPpStructuralViewCoverageContract) {
    return std::string(kPpStructuralViewCoverageContract);
  }
  if (coverage_contract == kEpStructuralViewCoverageContract) {
    return std::string(kEpStructuralViewCoverageContract);
  }
  if (coverage_contract == kCanonicalFullCoverageContract) {
    return std::string(kCanonicalFullCoverageContract);
  }
  return absl::InvalidArgumentError(absl::StrCat("unknown coverage_contract: ", coverage_contract));
}

absl::Status validate_requirement_set(const v2::AssemblyRequirementSetRef& requirements) {
  for (const auto& requirement : collect_requirements(requirements)) {
    auto coverage_or = canonicalize_coverage_contract(requirement.coverage_contract());
    if (!coverage_or.ok()) {
      return coverage_or.status();
    }
    const std::string& coverage_contract = *coverage_or;
    if (coverage_contract == kCanonicalFullCoverageContract) {
      if (requirement.slot_id() != kCanonicalFullContributionSlotKey) {
        return absl::InvalidArgumentError("canonical_full requirements must use slot_id=__canonical_full__");
      }
      if (requirement.target().kind() != v2::ASSEMBLY_TARGET_KIND_CANONICAL_LAYOUT) {
        return absl::InvalidArgumentError("canonical_full requirements must target canonical_layout");
      }
      if (!requirement.target().structural_view_id().empty()) {
        return absl::InvalidArgumentError("canonical_full requirements must not set structural_view_id");
      }
      continue;
    }
    if (requirement.target().kind() != v2::ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW) {
      return absl::InvalidArgumentError("piece requirements must target structural_view");
    }
    if (requirement.target().structural_view_id().empty()) {
      return absl::InvalidArgumentError("piece requirements must set structural_view_id");
    }
    if (requirement.slot_id() != requirement.target().structural_view_id()) {
      return absl::InvalidArgumentError("piece requirement slot_id must match structural_view_id");
    }
  }
  return absl::OkStatus();
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
  const auto kind = contract.kind() == v2::ASSEMBLY_CLOSEOUT_KIND_UNSPECIFIED
      ? v2::ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY
      : contract.kind();
  canonical.set_kind(kind);
  canonical.set_source_version_key(contract.source_version_key());
  canonical.set_serving_version_key(contract.serving_version_key());
  if (kind == v2::ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH && contract.has_representation_publish_contract()) {
    const auto representation_publish =
        canonicalize_representation_publish_contract(contract.representation_publish_contract());
    canonical.set_serving_manifest_ref(representation_publish.serving_manifest_ref());
    *canonical.mutable_representation_publish_contract() = representation_publish;
  } else {
    canonical.set_serving_artifact_id(contract.serving_artifact_id());
    canonical.set_serving_manifest_ref(contract.serving_manifest_ref());
  }
  canonical.set_closeout_contract_digest(compute_closeout_contract_digest(canonical));
  return canonical;
}

std::string compute_closeout_contract_digest(const v2::AssemblyCloseoutContract& contract) {
  v2::AssemblyCloseoutContract canonical;
  const auto kind = contract.kind() == v2::ASSEMBLY_CLOSEOUT_KIND_UNSPECIFIED
      ? v2::ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY
      : contract.kind();
  canonical.set_kind(kind);
  canonical.set_source_version_key(contract.source_version_key());
  canonical.set_serving_version_key(contract.serving_version_key());
  if (kind == v2::ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH && contract.has_representation_publish_contract()) {
    const auto representation_publish =
        canonicalize_representation_publish_contract(contract.representation_publish_contract());
    canonical.set_serving_manifest_ref(representation_publish.serving_manifest_ref());
    *canonical.mutable_representation_publish_contract() = representation_publish;
  } else {
    canonical.set_serving_artifact_id(contract.serving_artifact_id());
    canonical.set_serving_manifest_ref(contract.serving_manifest_ref());
  }
  canonical.clear_closeout_contract_digest();

  std::string payload;
  canonical.SerializeToString(&payload);
  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  const std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(bytes);
  return tensorcast::common::multibase_multihash_sha256(digest);
}

absl::Status validate_dependency_ready_closeout_contract(const v2::AssemblyCloseoutContract& contract) {
  const auto canonical = canonicalize_closeout_contract(contract);
  switch (canonical.kind()) {
    case v2::ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY:
      if (!canonical.serving_version_key().empty() || !canonical.serving_artifact_id().empty() ||
          !canonical.serving_manifest_ref().empty() || canonical.has_representation_publish_contract()) {
        return absl::InvalidArgumentError(
            "source_publish_only closeout contracts may not set serving_version_key, serving_artifact_id, "
            "serving_manifest_ref, or representation_publish_contract");
      }
      return absl::OkStatus();
    case v2::ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH: {
      if (!contract.has_representation_publish_contract()) {
        return absl::InvalidArgumentError(
            "representation_publish closeout contracts require representation_publish_contract");
      }
      if (!contract.serving_manifest_ref().empty() &&
          contract.serving_manifest_ref() != contract.representation_publish_contract().serving_manifest_ref()) {
        return absl::InvalidArgumentError(
            "serving_manifest_ref must match representation_publish_contract.serving_manifest_ref");
      }
      if (!canonical.representation_publish_contract().has_subject()) {
        return absl::InvalidArgumentError(
            "representation_publish closeout contracts require a serving publication subject");
      }
      if (!contract.serving_artifact_id().empty()) {
        return absl::InvalidArgumentError(
            "representation_publish closeout contracts must not set serving_artifact_id; "
            "use representation_publish_contract.subject");
      }
      const auto& subject = canonical.representation_publish_contract().subject();
      switch (subject.ref_case()) {
        case pubv1::ServingPublicationSubject::kServingArtifactId:
          if (subject.serving_artifact_id().empty()) {
            return absl::InvalidArgumentError(
                "artifact-backed representation_publish subjects require serving_artifact_id");
          }
          break;
        case pubv1::ServingPublicationSubject::kBindingValue:
          if (subject.binding_value().binding_id().empty() || subject.binding_value().binding_layout_id().empty() ||
              subject.binding_value().binding_value_id().empty() || subject.binding_value().seal_generation() == 0) {
            return absl::InvalidArgumentError(
                "binding-value representation_publish subjects require full binding identity");
          }
          break;
        case pubv1::ServingPublicationSubject::REF_NOT_SET:
          return absl::InvalidArgumentError(
              "representation_publish closeout contracts require a serving publication subject");
      }
      if (canonical.serving_manifest_ref().empty()) {
        return absl::InvalidArgumentError("representation_publish closeout contracts require serving_manifest_ref");
      }
      if (canonical.representation_publish_contract().representation_contract_hash().empty()) {
        return absl::InvalidArgumentError(
            "representation_publish closeout contracts require representation_contract_hash");
      }
      if (canonical.representation_publish_contract().serving_build_digest().empty()) {
        return absl::InvalidArgumentError("representation_publish closeout contracts require serving_build_digest");
      }
      if (!canonical.representation_publish_contract().serving_build_digest_version().empty() &&
          canonical.representation_publish_contract().serving_build_digest_version() !=
              serving_artifact_manifest::kPhase1ServingBuildDigestVersion) {
        return absl::InvalidArgumentError(
            "representation_publish closeout contracts require a supported serving_build_digest_version");
      }
      return absl::OkStatus();
    }
    case v2::ASSEMBLY_CLOSEOUT_KIND_ROLLOUT_GATED_PUBLISH:
      return absl::UnimplementedError(
          "rollout_gated_publish closeout contracts are not dependency-ready in the current execution wave");
    case v2::ASSEMBLY_CLOSEOUT_KIND_UNSPECIFIED:
    default:
      return absl::InternalError("unexpected assembly closeout contract kind");
  }
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
  auto expected_coverage_or = expected_coverage_contract_for_binding_contribution(contribution_kind);
  if (!expected_coverage_or.ok()) {
    return expected_coverage_or.status();
  }
  const std::string expected_slot_id = contribution_slot_key(contribution_kind, structural_view_id);
  for (const auto& requirement : collect_requirements(requirements)) {
    auto coverage_or = canonicalize_coverage_contract(requirement.coverage_contract());
    if (!coverage_or.ok()) {
      return coverage_or.status();
    }
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
    if (*coverage_or != *expected_coverage_or) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "binding contribution coverage_contract mismatch for slot_id=",
              expected_slot_id,
              " expected=",
              *expected_coverage_or,
              " actual=",
              *coverage_or));
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
