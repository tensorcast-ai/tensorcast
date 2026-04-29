// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/runtime/ingestion/materialization_strategy_types.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::materialization_policy {

using store::loader::ViewSpec;
using store::loading::ExecutionTopologyContext;
using store::loading::RetrievalPolicy;

struct NormalizedMaterializationRequestContext {
  RetrievalPolicy retrieval_policy;
  ExecutionTopologyContext execution_topology;
  std::optional<std::string> replica_uuid;
  bool verify_checksums{true};
  int32_t wait_for_shared_disk_ms{0};
};

struct OperationTransportContext {
  ExecutionTopologyContext execution_topology;
  std::string transport_request_id;
  std::optional<store::loading::TransportSchedulingGroupHint> transport_scheduling_group;
};

struct HashExecutionDetails {
  uint32_t hash_rounds{0};
  v2::HashLocation hash_location{v2::HashLocation::HASH_LOCATION_NONE};
  v2::HashBackend hash_backend{v2::HashBackend::HASH_BACKEND_NONE};
  uint64_t hash_bytes{0};
  uint64_t hash_wall_time_ms{0};
  bool hash_identity_forming{false};
  v2::IdentityMintStrategy identity_mint_strategy{v2::IdentityMintStrategy::IDENTITY_MINT_STRATEGY_NOT_APPLICABLE};
};

store::loading::SourcePreference to_hint_preference(v2::SourcePreference preference);

store::loading::ExportPolicy to_hint_export_policy(v2::ExportPolicy policy);

absl::StatusOr<RetrievalPolicy> resolve_retrieval_policy_compat(const v2::SourcePolicy* policy);

std::optional<store::loading::CollectiveLoadGroupHint> resolve_collective_group_hint(
    const v2::CollectiveLoadGroup* group);

std::optional<store::loading::TransportSchedulingGroupHint> resolve_transport_scheduling_group_hint(
    const v2::TransportSchedulingGroupHint* group);

absl::StatusOr<ExecutionTopologyContext> resolve_source_execution_topology(const v2::SourceExecutionTopology* topology);

absl::StatusOr<v2::CollectivePolicy> resolve_collective_policy(
    v2::CollectivePolicy requested,
    const ExecutionTopologyContext& execution_topology);

v2::CollectivePolicy default_collective_policy_for_mapped_target(const ExecutionTopologyContext& execution_topology);

bool collective_policy_requests_collective(v2::CollectivePolicy policy);

OperationTransportContext resolve_operation_transport_context(std::string_view operation_id);

absl::StatusOr<NormalizedMaterializationRequestContext> resolve_materialization_request_context(
    const v2::SourcePolicy* source_policy,
    ExecutionTopologyContext execution_topology = {},
    std::optional<std::string> replica_uuid = std::nullopt,
    bool verify_checksums = true,
    int32_t wait_for_shared_disk_ms = 0);

void apply_operation_transport_context(
    const OperationTransportContext& context,
    store::loading::MaterializeHints* hints);

void apply_request_context_to_hints(
    const NormalizedMaterializationRequestContext& context,
    store::loading::MaterializeHints* hints);

v2::ExecutionDiagnostics build_execution_diagnostics(
    const store::loading::MaterializeIntoTargetResult* result,
    v2::CollectivePolicy collective_policy,
    const ExecutionTopologyContext& execution_topology,
    const HashExecutionDetails& hash_details = {});

absl::StatusOr<ViewSpec> convert_view_spec(const tensorcast::common::v1::ViewSpec& proto);

absl::StatusOr<std::string> compute_view_id_from_spec(
    const tensorcast::common::v1::ViewSpec& view_spec,
    std::string_view canonical_index_json);

tensorcast::common::v1::ViewSpec build_view_spec_proto(const ViewSpec& spec);

bool spec_includes_transpose(const ViewSpec& spec);

store::loading::TransformPlacement resolve_transform_placement(
    v2::TransformPlacement requested,
    const std::optional<ViewSpec>& spec);

} // namespace tensorcast::daemon::materialization_policy
