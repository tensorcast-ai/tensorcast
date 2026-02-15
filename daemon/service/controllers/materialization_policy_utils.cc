// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_policy_utils.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "core/store/materialization/dataplane/view/view_identity.h"

namespace tensorcast::daemon::materialization_policy {

namespace {

using store::loader::ViewOp;

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

ResolvedSourcePolicy resolve_source_policy(const v2::SourcePolicy* policy, v2::SourcePreference legacy_preference) {
  ResolvedSourcePolicy resolved;
  resolved.preference = legacy_preference;
  if (policy != nullptr) {
    if (policy->preference() != v2::SourcePreference::SOURCE_PREFERENCE_UNSPECIFIED) {
      resolved.preference = policy->preference();
    }
    if (policy->has_allow_p2p()) {
      resolved.allow_p2p = policy->allow_p2p();
    }
    if (policy->has_allow_disk()) {
      resolved.allow_disk = policy->allow_disk();
    }
  }
  if (resolved.preference == v2::SourcePreference::SOURCE_PREFERENCE_UNSPECIFIED) {
    resolved.preference = v2::SourcePreference::SOURCE_PREFERENCE_AUTO;
  }
  return resolved;
}

absl::Status validate_source_policy(const ResolvedSourcePolicy& policy) {
  if (!policy.allow_p2p && policy.preference == v2::SourcePreference::SOURCE_PREFERENCE_PREFER_P2P) {
    return absl::InvalidArgumentError("source_policy disallows P2P but preference=PREFER_P2P was requested");
  }
  if (!policy.allow_disk && policy.preference == v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK) {
    return absl::InvalidArgumentError("source_policy disallows disk but preference=PREFER_DISK was requested");
  }
  return absl::OkStatus();
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
