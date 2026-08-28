// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/store_engine.h"
#include "daemon/service/controllers/materialization_layout_utils.h"
#include "daemon/service/controllers/representation_transform_builder.h"
#include "daemon/state/types.h"
#include "grpcpp/support/status.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::materialization_target_plan {

using RecordMaterializeResultFn =
    void (*)(std::string_view result, std::string_view reason, v2::MaterializationSource source);
using CanonicalIndexTablePtr = std::shared_ptr<const materialization_layout::CanonicalIndexTable>;
using CanonicalIndexTableFuture = std::shared_future<absl::StatusOr<CanonicalIndexTablePtr>>;

struct TargetMaterializationPlan {
  std::vector<std::string> layout_names;
  std::optional<store::loader::ViewSpec> view_spec;
  std::optional<std::string> view_data_hash;
  std::optional<store::loader::ViewPlan> view_plan;
  std::optional<std::string> resolved_view_id;
  tensorcast::common::v1::ArtifactSelection resolved_selection;
  std::string canonical_index_json;
  std::string selected_index_json;
  std::vector<RegisterStorageMeta> publish_storages;
  std::vector<LeaseSegMeta> publish_segments;
  std::string view_subset_hash;
  bool has_subset{false};
  bool has_view_transform{false};
  uint64_t logical_total_size{0};
};

struct MappedTargetMaterializationPlan {
  std::optional<store::loader::ViewSpec> view_spec;
  std::optional<store::loader::ViewPlan> view_plan;
  tensorcast::common::v1::ArtifactSelection resolved_selection;
  representation_transform_builder::BuildRepresentationTransformResult representation;
  std::string canonical_index_json;
  std::string selected_index_json;
  std::vector<RegisterStorageMeta> publish_storages;
  std::vector<LeaseSegMeta> publish_segments;
  uint64_t logical_total_size{0};
};

struct BindingRealizationMaterializationPlanOptions {
  bool build_byte_range_maps{true};
  std::optional<std::string> canonical_index_parse_identity_key;
  CanonicalIndexTablePtr preparsed_canonical_index_table;
  CanonicalIndexTableFuture preparsed_canonical_index_table_future;
};

struct ResolvedMappedMaterializationPlanOptions {
  bool source_window_strict_coverage_proof_only{false};
};

grpc::Status build_target_materialization_plan(
    store::StoreEngine& engine,
    std::string_view resolved_artifact_id,
    const v2::MaterializeIntoTargetRequest& req,
    const v2::TargetLayout& layout,
    const std::vector<materialization_layout::TargetOffsetEntry>& offsets,
    std::string canonical_index_json,
    RecordMaterializeResultFn record_result,
    TargetMaterializationPlan& plan);

grpc::Status build_mapped_target_materialization_plan(
    store::StoreEngine& engine,
    const v2::MaterializeIntoMappedTargetRequest& req,
    std::string_view resolved_artifact_id,
    const std::vector<materialization_layout::TargetOffsetEntry>& offsets,
    std::string canonical_index_json,
    RecordMaterializeResultFn record_result,
    MappedTargetMaterializationPlan& plan);

grpc::Status build_binding_realization_materialization_plan(
    store::StoreEngine& engine,
    const tensorcast::common::v1::ArtifactSelection& selection,
    const v2::BindingRealizationPlan& realization_plan,
    std::string_view resolved_artifact_id,
    const v2::TargetLayout& target_layout,
    std::string_view target_index_json,
    const std::vector<materialization_layout::TargetOffsetEntry>& offsets,
    std::string canonical_index_json,
    RecordMaterializeResultFn record_result,
    BindingRealizationMaterializationPlanOptions options,
    MappedTargetMaterializationPlan& plan);

grpc::Status build_binding_realization_materialization_plan(
    store::StoreEngine& engine,
    const tensorcast::common::v1::ArtifactSelection& selection,
    const v2::BindingRealizationPlan& realization_plan,
    std::string_view resolved_artifact_id,
    const v2::TargetLayout& target_layout,
    std::string_view target_index_json,
    const std::vector<materialization_layout::TargetOffsetEntry>& offsets,
    std::string canonical_index_json,
    RecordMaterializeResultFn record_result,
    MappedTargetMaterializationPlan& plan);

absl::StatusOr<store::runtime::ingestion::strategy::PreparedSourceBoundExecutionPlan>
build_resolved_mapped_materialization_plan(
    std::string_view resolved_artifact_id,
    uint64_t generation,
    const store::loading::IntoTargetLayout& target_layout,
    const MappedTargetMaterializationPlan& mapped_plan,
    const std::optional<store::loading::VariantIdentity>& variant,
    std::optional<std::string_view> source_index_json = std::nullopt,
    std::shared_ptr<const materialization_layout::CanonicalIndexTable> physical_source_table = nullptr,
    ResolvedMappedMaterializationPlanOptions options = {});

} // namespace tensorcast::daemon::materialization_target_plan
