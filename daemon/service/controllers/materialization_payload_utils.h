// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/store_engine.h"
#include "google/protobuf/repeated_field.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::materialization_payload {

struct DescriptorBuildResult {
  std::vector<v2::TensorPayloadDescriptor> descriptors;
  std::vector<std::string> included_names;
};

uint64_t compute_generation_from_index(std::string_view canonical_index_json);

absl::StatusOr<DescriptorBuildResult> build_descriptors_from_view_plan(
    const store::loader::ViewPlan& plan,
    const google::protobuf::RepeatedPtrField<std::string>& tensor_names,
    std::string_view device_uuid);

absl::StatusOr<DescriptorBuildResult> build_descriptors_from_index(
    std::string_view canonical_index_json,
    const google::protobuf::RepeatedPtrField<std::string>& tensor_names,
    std::string_view device_uuid);

absl::StatusOr<std::string> resolve_layout_json(
    const v2::MaterializeReplicaResponse& v1_resp,
    const v2::MaterializeReplicaRequest& v2_req,
    store::StoreEngine& engine);

absl::StatusOr<std::string> resolve_layout_json_by_key(
    const v2::MaterializeByKeyResponse& v1_resp,
    store::StoreEngine& engine);

absl::Status populate_materialize_payloads(
    v2::MaterializeReplicaResponse& resp,
    std::string_view layout_json,
    const google::protobuf::RepeatedPtrField<std::string>& tensor_names,
    std::string_view device_uuid,
    std::string_view view_subset_hash,
    bool wait_for_completion,
    std::string_view replica_uuid,
    const std::string* ticket_device_uuid,
    const std::optional<store::loader::ViewPlan>& view_plan,
    bool prefer_view_plan,
    bool fill_view_index_bytes);

absl::Status populate_materialize_payloads(
    v2::MaterializeByKeyResponse& resp,
    std::string_view layout_json,
    const google::protobuf::RepeatedPtrField<std::string>& tensor_names,
    std::string_view device_uuid,
    std::string_view view_subset_hash,
    bool wait_for_completion,
    std::string_view replica_uuid,
    const std::string* ticket_device_uuid,
    const std::optional<store::loader::ViewPlan>& view_plan,
    bool prefer_view_plan,
    bool fill_view_index_bytes);

} // namespace tensorcast::daemon::materialization_payload
