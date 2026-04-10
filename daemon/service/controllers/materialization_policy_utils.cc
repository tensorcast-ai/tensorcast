// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_policy_utils.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "core/store/materialization/dataplane/view/view_identity.h"

namespace tensorcast::daemon::materialization_policy {

namespace {

using store::loader::ViewOp;

store::loading::SourceLocalityHint to_source_locality(v2::SourceLocality locality) {
  switch (locality) {
    case v2::SourceLocality::SOURCE_LOCALITY_HOST_LOCAL:
      return store::loading::SourceLocalityHint::kHostLocal;
    case v2::SourceLocality::SOURCE_LOCALITY_SHARED_SOURCE:
      return store::loading::SourceLocalityHint::kSharedSource;
    case v2::SourceLocality::SOURCE_LOCALITY_AUTO:
    case v2::SourceLocality::SOURCE_LOCALITY_UNSPECIFIED:
    default:
      return store::loading::SourceLocalityHint::kAuto;
  }
}

} // namespace

store::loading::SourcePreference to_hint_preference(v2::SourcePreference preference) {
  switch (preference) {
    case v2::SourcePreference::SOURCE_PREFERENCE_PREFER_P2P:
      return store::loading::SourcePreference::kPreferP2P;
    case v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK:
      return store::loading::SourcePreference::kPreferDisk;
    case v2::SourcePreference::SOURCE_PREFERENCE_AUTO:
    case v2::SourcePreference::SOURCE_PREFERENCE_UNSPECIFIED:
    default:
      return store::loading::SourcePreference::kAuto;
  }
}

store::loading::ExportPolicy to_hint_export_policy(v2::ExportPolicy policy) {
  switch (policy) {
    case v2::ExportPolicy::EXPORT_POLICY_FORCE:
      return store::loading::ExportPolicy::kForce;
    case v2::ExportPolicy::EXPORT_POLICY_AUTO:
      return store::loading::ExportPolicy::kAuto;
    case v2::ExportPolicy::EXPORT_POLICY_NEVER:
    case v2::ExportPolicy::EXPORT_POLICY_UNSPECIFIED:
    default:
      return store::loading::ExportPolicy::kNever;
  }
}

store::loading::SourceLocalityHint parse_source_locality_hint(std::string_view value) {
  if (value == "host_local") {
    return store::loading::SourceLocalityHint::kHostLocal;
  }
  if (value == "shared_source") {
    return store::loading::SourceLocalityHint::kSharedSource;
  }
  return store::loading::SourceLocalityHint::kAuto;
}

absl::StatusOr<RetrievalPolicy> resolve_retrieval_policy_compat(const v2::SourcePolicy* policy) {
  RetrievalPolicy resolved;
  if (policy != nullptr) {
    if (policy->preference() != v2::SourcePreference::SOURCE_PREFERENCE_UNSPECIFIED) {
      resolved.preference = to_hint_preference(policy->preference());
    }
    if (policy->has_allow_p2p()) {
      resolved.allow_p2p = policy->allow_p2p();
    }
    if (policy->has_allow_disk()) {
      resolved.allow_disk = policy->allow_disk();
    }
  }
  if (auto policy_status = store::loading::validate_retrieval_policy(resolved); !policy_status.ok()) {
    return policy_status;
  }
  return resolved;
}

std::optional<store::loading::CollectiveLoadGroupHint> resolve_collective_group_hint(
    const v2::CollectiveLoadGroup* group) {
  if (group == nullptr || group->group_id().empty()) {
    return std::nullopt;
  }
  const uint32_t world_size = group->world_size();
  const uint32_t rank = group->rank();
  if (world_size <= 1 || rank >= world_size) {
    return std::nullopt;
  }
  return store::loading::CollectiveLoadGroupHint{
      .group_id = group->group_id(),
      .world_size = world_size,
      .rank = rank,
  };
}

absl::StatusOr<ExecutionTopologyContext> resolve_source_execution_topology(
    const v2::SourceExecutionTopology* topology) {
  ExecutionTopologyContext execution_topology;
  if (topology == nullptr) {
    return execution_topology;
  }
  if (topology->has_collective_load_group()) {
    auto group_hint = resolve_collective_group_hint(&topology->collective_load_group());
    if (!group_hint.has_value()) {
      return absl::InvalidArgumentError("execution_topology.collective_load_group is invalid");
    }
    execution_topology.collective_load_group = std::move(group_hint);
  }
  execution_topology.source_locality = to_source_locality(topology->source_locality());
  if (topology->has_source_sharing_domain() && !topology->source_sharing_domain().empty()) {
    execution_topology.source_sharing_domain = topology->source_sharing_domain();
  }
  return execution_topology;
}

absl::StatusOr<v2::CollectivePolicy> resolve_collective_policy(
    v2::CollectivePolicy requested,
    const ExecutionTopologyContext& execution_topology) {
  const bool has_collective_group = execution_topology.collective_load_group.has_value();
  if (requested == v2::CollectivePolicy::COLLECTIVE_POLICY_UNSPECIFIED) {
    return has_collective_group ? v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE
                                : v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE;
  }
  if (requested == v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE && has_collective_group) {
    return absl::InvalidArgumentError("collective_policy=disable_collective conflicts with a collective_load_group");
  }
  if (requested != v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE && !has_collective_group) {
    return absl::InvalidArgumentError("collective_policy requires execution_topology.collective_load_group");
  }
  return requested;
}

v2::CollectivePolicy default_collective_policy_for_mapped_target(const ExecutionTopologyContext& execution_topology) {
  return execution_topology.collective_load_group.has_value()
      ? v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST
      : v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE;
}

bool collective_policy_requests_collective(v2::CollectivePolicy policy) {
  return policy == v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE ||
      policy == v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST;
}

OperationTransportContext resolve_operation_transport_context(std::string_view operation_id) {
  constexpr std::string_view kGroupMarker = "#tcg:";
  OperationTransportContext context;
  if (operation_id.empty()) {
    return context;
  }
  const size_t marker_pos = operation_id.find(kGroupMarker);
  if (marker_pos == std::string_view::npos) {
    context.transport_request_id = std::string(operation_id);
    return context;
  }

  context.transport_request_id = std::string(operation_id.substr(0, marker_pos));
  const std::string_view metadata = operation_id.substr(marker_pos + kGroupMarker.size());

  std::string group_kind;
  std::string group_id;
  std::string part_id;
  std::string request_id_override;
  std::string collective_group_id;
  std::string source_sharing_domain;
  int group_total_parts = 0;
  int group_priority = 0;
  int collective_world_size = 0;
  int collective_rank = -1;
  uint64_t group_epoch = 0;
  store::loading::SourceLocalityHint source_locality = store::loading::SourceLocalityHint::kAuto;

  for (const std::string_view item : absl::StrSplit(metadata, ';', absl::SkipEmpty())) {
    std::vector<std::string_view> kv = absl::StrSplit(item, absl::MaxSplits('=', 1));
    if (kv.size() != 2) {
      continue;
    }
    const std::string_view key = kv[0];
    const std::string_view value = kv[1];
    if (key == "kind") {
      group_kind = std::string(value);
      continue;
    }
    if (key == "gid") {
      group_id = std::string(value);
      continue;
    }
    if (key == "tot") {
      int parsed_total_parts = 0;
      if (absl::SimpleAtoi(value, &parsed_total_parts)) {
        group_total_parts = parsed_total_parts;
      }
      continue;
    }
    if (key == "part") {
      part_id = std::string(value);
      continue;
    }
    if (key == "pri") {
      int parsed_priority = 0;
      if (absl::SimpleAtoi(value, &parsed_priority)) {
        group_priority = parsed_priority;
      }
      continue;
    }
    if (key == "ep") {
      uint64_t parsed_epoch = 0;
      if (absl::SimpleAtoi(value, &parsed_epoch)) {
        group_epoch = parsed_epoch;
      }
      continue;
    }
    if (key == "rid") {
      request_id_override = std::string(value);
      continue;
    }
    if (key == "clid") {
      collective_group_id = std::string(value);
      continue;
    }
    if (key == "sloc") {
      source_locality = parse_source_locality_hint(value);
      continue;
    }
    if (key == "sdom") {
      source_sharing_domain = std::string(value);
      continue;
    }
    if (key == "clws") {
      int parsed_world_size = 0;
      if (absl::SimpleAtoi(value, &parsed_world_size)) {
        collective_world_size = parsed_world_size;
      }
      continue;
    }
    if (key == "clrk") {
      int parsed_rank = -1;
      if (absl::SimpleAtoi(value, &parsed_rank)) {
        collective_rank = parsed_rank;
      }
      continue;
    }
  }

  if (!request_id_override.empty()) {
    context.transport_request_id = std::move(request_id_override);
  }

  if (!group_kind.empty() && !group_id.empty() && !part_id.empty() && group_total_parts > 0) {
    store::loading::TransportSchedulingGroupHint group;
    group.group_kind = std::move(group_kind);
    group.group_id = std::move(group_id);
    group.total_parts = static_cast<uint32_t>(group_total_parts);
    group.part_id = std::move(part_id);
    group.priority = static_cast<uint32_t>(std::max(0, group_priority));
    group.epoch = group_epoch;
    context.transport_scheduling_group = std::move(group);
  }
  context.execution_topology.source_locality = source_locality;
  if (!source_sharing_domain.empty()) {
    context.execution_topology.source_sharing_domain = std::move(source_sharing_domain);
  }
  if (!collective_group_id.empty() && collective_world_size > 1 && collective_rank >= 0 &&
      collective_rank < collective_world_size) {
    context.execution_topology.collective_load_group = store::loading::CollectiveLoadGroupHint{
        .group_id = std::move(collective_group_id),
        .world_size = static_cast<uint32_t>(collective_world_size),
        .rank = static_cast<uint32_t>(collective_rank),
    };
  }
  return context;
}

absl::StatusOr<NormalizedMaterializationRequestContext> resolve_materialization_request_context(
    const v2::SourcePolicy* source_policy,
    ExecutionTopologyContext execution_topology,
    std::optional<std::string> replica_uuid,
    bool verify_checksums,
    int32_t wait_for_shared_disk_ms) {
  auto retrieval_policy_or = resolve_retrieval_policy_compat(source_policy);
  if (!retrieval_policy_or.ok()) {
    return retrieval_policy_or.status();
  }
  return NormalizedMaterializationRequestContext{
      .retrieval_policy = *retrieval_policy_or,
      .execution_topology = std::move(execution_topology),
      .replica_uuid = std::move(replica_uuid),
      .verify_checksums = verify_checksums,
      .wait_for_shared_disk_ms = wait_for_shared_disk_ms,
  };
}

void apply_operation_transport_context(
    const OperationTransportContext& context,
    store::loading::MaterializeHints* hints) {
  if (hints == nullptr) {
    return;
  }
  if (!context.transport_request_id.empty()) {
    hints->transport_request_id = context.transport_request_id;
  }
  if (context.transport_scheduling_group.has_value()) {
    hints->transport_scheduling_group = context.transport_scheduling_group;
  }
}

void apply_request_context_to_hints(
    const NormalizedMaterializationRequestContext& context,
    store::loading::MaterializeHints* hints) {
  if (hints == nullptr) {
    return;
  }
  hints->set_retrieval_policy(context.retrieval_policy);
  hints->set_execution_topology(context.execution_topology);
}

v2::ExecutionDiagnostics build_execution_diagnostics(
    const store::loading::MaterializeIntoTargetResult* result,
    v2::CollectivePolicy collective_policy,
    const ExecutionTopologyContext& execution_topology,
    const HashExecutionDetails& hash_details) {
  v2::ExecutionDiagnostics diagnostics;
  const bool collective_requested =
      collective_policy_requests_collective(collective_policy) && execution_topology.collective_load_group.has_value();
  diagnostics.set_collective_requested(collective_requested);
  diagnostics.set_collective_acknowledged(collective_requested);
  diagnostics.set_collective_policy(collective_policy);
  diagnostics.set_hash_rounds(hash_details.hash_rounds);
  diagnostics.set_hash_location(hash_details.hash_location);
  diagnostics.set_hash_backend(hash_details.hash_backend);
  diagnostics.set_hash_bytes(hash_details.hash_bytes);
  diagnostics.set_hash_wall_time_ms(hash_details.hash_wall_time_ms);
  diagnostics.set_hash_identity_forming(hash_details.hash_identity_forming);
  diagnostics.set_identity_mint_strategy(hash_details.identity_mint_strategy);
  if (result == nullptr) {
    return diagnostics;
  }
  diagnostics.set_collective_used(result->collective_handled);
  diagnostics.set_dominant_executor(result->dominant_executor);
  diagnostics.set_direct_write_supported(result->direct_write_supported);
  diagnostics.set_fallback_bytes(result->fallback_bytes);
  diagnostics.set_residual_bytes(result->residual_bytes);
  diagnostics.set_actual_collective_committed_bytes(result->actual_collective_committed_bytes);
  diagnostics.set_actual_local_typed_bytes(result->actual_local_typed_bytes);
  diagnostics.set_actual_generic_backend_bytes(result->actual_generic_backend_bytes);
  diagnostics.set_collective_skip_reason(result->collective_skip_reason);
  if (collective_requested && !result->collective_handled) {
    diagnostics.set_collective_failure_class(v2::CollectiveFailureClass::COLLECTIVE_FAILURE_CLASS_NOT_ELIGIBLE);
  }
  return diagnostics;
}

absl::StatusOr<ViewSpec> convert_view_spec(const tensorcast::common::v1::ViewSpec& proto) {
  ViewSpec spec;
  for (const auto& [tensor_name, ops_proto] : proto.tensors()) {
    store::loader::TensorViewOps ops;
    for (const auto& op_proto : ops_proto.ops()) {
      switch (op_proto.kind_case()) {
        case tensorcast::common::v1::Op::kNarrow: {
          const auto& narrow = op_proto.narrow();
          store::loader::NarrowOp op{
              .dim = static_cast<int32_t>(narrow.dim()),
              .start = narrow.start(),
              .length = narrow.length(),
          };
          ops.ops.push_back(ViewOp::Narrow(op));
          break;
        }
        case tensorcast::common::v1::Op::kTranspose: {
          const auto& transpose = op_proto.transpose();
          store::loader::TransposeOp op{
              .dim0 = static_cast<int32_t>(transpose.dim0()),
              .dim1 = static_cast<int32_t>(transpose.dim1()),
          };
          ops.ops.push_back(ViewOp::Transpose(op));
          break;
        }
        case tensorcast::common::v1::Op::KIND_NOT_SET:
          return absl::InvalidArgumentError("view op kind not set");
      }
    }
    spec.tensors.emplace(tensor_name, std::move(ops));
  }
  return spec;
}

absl::StatusOr<std::string> compute_view_id_from_spec(
    const tensorcast::common::v1::ViewSpec& view_spec,
    std::string_view canonical_index_json) {
  auto spec_or = convert_view_spec(view_spec);
  if (!spec_or.ok()) {
    return spec_or.status();
  }
  return store::loader::compute_view_id_from_spec(*spec_or, canonical_index_json);
}

tensorcast::common::v1::ViewSpec build_view_spec_proto(const ViewSpec& spec) {
  tensorcast::common::v1::ViewSpec proto;
  auto* tensors = proto.mutable_tensors();
  for (const auto& [tensor_name, ops] : spec.tensors) {
    auto& container = (*tensors)[tensor_name];
    for (const auto& op : ops.ops) {
      auto* op_proto = container.add_ops();
      switch (op.kind) {
        case ViewOp::Kind::kNarrow:
          op_proto->mutable_narrow()->set_dim(static_cast<uint32_t>(op.narrow.dim));
          op_proto->mutable_narrow()->set_start(op.narrow.start);
          op_proto->mutable_narrow()->set_length(op.narrow.length);
          break;
        case ViewOp::Kind::kTranspose:
          op_proto->mutable_transpose()->set_dim0(static_cast<uint32_t>(op.transpose.dim0));
          op_proto->mutable_transpose()->set_dim1(static_cast<uint32_t>(op.transpose.dim1));
          break;
      }
    }
  }
  return proto;
}

bool spec_includes_transpose(const ViewSpec& spec) {
  for (const auto& [_, ops] : spec.tensors) {
    for (const auto& op : ops.ops) {
      if (op.kind == ViewOp::Kind::kTranspose) {
        return true;
      }
    }
  }
  return false;
}

store::loading::TransformPlacement resolve_transform_placement(
    v2::TransformPlacement requested,
    const std::optional<ViewSpec>& spec) {
  switch (requested) {
    case v2::TransformPlacement::TRANSFORM_PLACEMENT_SERVER:
      return store::loading::TransformPlacement::kServer;
    case v2::TransformPlacement::TRANSFORM_PLACEMENT_CLIENT:
      return store::loading::TransformPlacement::kClient;
    case v2::TransformPlacement::TRANSFORM_PLACEMENT_UNSPECIFIED:
    default:
      break;
  }
  if (spec.has_value() && spec_includes_transpose(*spec)) {
    return store::loading::TransformPlacement::kClient;
  }
  return store::loading::TransformPlacement::kServer;
}

} // namespace tensorcast::daemon::materialization_policy
