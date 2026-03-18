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

v2::ContributionContractSnapshot build_phase1_contribution_contract(
    std::string_view layout_id,
    const google::protobuf::RepeatedPtrField<std::string>& expected_view_ids,
    bool require_live_contributions_until_readiness_cut);

v2::ContributionContractSnapshot canonicalize_contribution_contract(const v2::ContributionContractSnapshot& snapshot);

std::string compute_contribution_contract_hash(const v2::ContributionContractSnapshot& snapshot);

v2::CloseoutPolicySnapshot canonicalize_closeout_policy_snapshot(const v2::CloseoutPolicySnapshot& snapshot);

std::string compute_closeout_policy_hash(const v2::CloseoutPolicySnapshot& snapshot);

v2::AssemblyAttemptSpec canonicalize_attempt_spec(const v2::AssemblyAttemptSpec& spec);

std::string compute_attempt_spec_hash(const v2::AssemblyAttemptSpec& spec);

absl::StatusOr<absl::flat_hash_set<std::string>> list_active_contributor_identities(
    const std::shared_ptr<store::components::IGlobalStoreClient>& client);

bool assembly_contribution_is_live(
    const store::components::AssemblyContributionInfo& contribution,
    absl::Time now = absl::Now(),
    const absl::flat_hash_set<std::string>* active_contributor_identities = nullptr);

bool operation_lease_is_live(
    const tensorcast::operation::v1::GetOperationResponse& operation,
    absl::Time now = absl::Now());

bool operation_allows_contributions(const tensorcast::operation::v1::GetOperationResponse& operation);

absl::Status validate_contribution_contract_entry(
    const v2::ContributionContractSnapshot& snapshot,
    v2::BindingContributionKind contribution_kind,
    std::string_view structural_view_id);

absl::Status validate_live_required_contributions(
    const v2::ContributionContractSnapshot& snapshot,
    absl::Span<const store::components::AssemblyContributionInfo> contributions,
    const absl::flat_hash_set<std::string>& active_contributor_identities,
    absl::Time now = absl::Now());

} // namespace tensorcast::daemon::assembly_coordination
