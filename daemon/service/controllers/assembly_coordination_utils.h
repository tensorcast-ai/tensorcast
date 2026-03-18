// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <string>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/store/components/global_store_client.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"
#include "tensorcast/operation/v1/operation.pb.h"

namespace tensorcast::daemon::assembly_coordination {

inline constexpr std::string_view kCanonicalFullContributionSlotKey = "__canonical_full__";
inline constexpr std::string_view kPiecePartialCoverageSemantics = "phase1_layout_expected_view";
inline constexpr std::string_view kCanonicalFullCoverageSemantics = "phase1_canonical_full";
inline constexpr absl::Duration kContributionLeaseTtl = absl::Seconds(20);
inline constexpr absl::Duration kContributionLeaseRefreshInterval = absl::Seconds(5);

std::string contribution_slot_key(v2::BindingContributionKind contribution_kind, std::string_view structural_view_id);

std::string contribution_structural_view_id(
    v2::BindingContributionKind contribution_kind,
    std::string_view structural_view_id);

v2::AssemblyRequirementSetRef build_phase1_requirement_set(
    const google::protobuf::RepeatedPtrField<std::string>& expected_view_ids);

v2::AssemblyRequirementSetRef canonicalize_requirement_set(const v2::AssemblyRequirementSetRef& requirements);

std::string compute_requirement_set_digest(const v2::AssemblyRequirementSetRef& requirements);

v2::AssemblyReadinessPolicy canonicalize_readiness_policy(const v2::AssemblyReadinessPolicy& policy);

v2::AssemblyCloseoutContract canonicalize_closeout_contract(const v2::AssemblyCloseoutContract& contract);

std::string compute_closeout_contract_digest(const v2::AssemblyCloseoutContract& contract);

absl::Status validate_dependency_ready_closeout_contract(const v2::AssemblyCloseoutContract& contract);

v2::AssemblyAttemptIntent canonicalize_attempt_intent(const v2::AssemblyAttemptIntent& intent);

std::string compute_attempt_intent_digest(const v2::AssemblyAttemptIntent& intent);

absl::StatusOr<absl::flat_hash_set<std::string>> list_active_contributor_identities(
    const std::shared_ptr<store::components::IGlobalStoreClient>& client);

bool operation_lease_is_live(
    const tensorcast::operation::v1::GetOperationResponse& operation,
    absl::Time now = absl::Now());

bool operation_allows_contributions(const tensorcast::operation::v1::GetOperationResponse& operation);

bool slot_occupancy_is_live(
    const store::components::AssemblySlotOccupancyInfo& occupancy,
    absl::Time now = absl::Now(),
    const absl::flat_hash_set<std::string>* active_contributor_identities = nullptr);

absl::Status validate_binding_requirement_entry(
    const v2::AssemblyRequirementSetRef& requirements,
    v2::BindingContributionKind contribution_kind,
    std::string_view structural_view_id);

absl::Status validate_live_required_slot_occupancies(
    const v2::AssemblyRequirementSetRef& requirements,
    const v2::AssemblyReadinessPolicy& readiness_policy,
    absl::Span<const store::components::AssemblySlotOccupancyInfo> occupancies,
    const absl::flat_hash_set<std::string>& active_contributor_identities,
    absl::Time now = absl::Now());

} // namespace tensorcast::daemon::assembly_coordination
