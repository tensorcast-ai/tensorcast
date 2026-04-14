// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "core/store/materialization/contracts/loading_spec.h"

namespace tensorcast::store::runtime::ingestion::strategy {

// The selected source may expose canonical bytes or an already materialized
// view/mapped byte-space. This is independent from the request's view_id:
// mapped-target requests can carry a target byte-space identity while still
// reconstructing the result from canonical/disk fallback.
enum class SourceByteSpace : std::uint8_t {
  kCanonical = 0,
  kView = 1,
};

enum class TensorJobDistribution : std::uint8_t {
  kUnknown = 0,
  kReplicated = 1,
  kDim0Partitioned = 2,
  kDim1Partitioned = 3,
};

struct TensorJobCandidate {
  std::string src_name;
  std::string dst_name;
  TensorJobDistribution distribution{TensorJobDistribution::kUnknown};
  std::vector<int64_t> src_shape;
  std::vector<int64_t> src_stride;
  std::vector<int64_t> dst_shape;
  std::vector<int64_t> dst_stride;
  std::string dtype;
  uint64_t src_logical_offset{0};
  uint64_t src_storage_offset{0};
  uint64_t src_size_bytes{0};
  uint64_t dst_base_offset{0};
  uint64_t dst_size_bytes{0};
  uint64_t element_size{0};
  int32_t dim{-1};
  int64_t src_start{0};
  int64_t src_end{0};
  int64_t dst_start{0};
  int64_t dst_end{0};
};

struct ConcatSourceFragment {
  std::string src_name;
  std::vector<int64_t> src_shape;
  std::vector<int64_t> src_stride;
  std::string dtype;
  uint64_t src_logical_offset{0};
  uint64_t src_storage_offset{0};
  uint64_t src_size_bytes{0};
  uint64_t element_size{0};
  int32_t dim{0};
  int64_t src_start{0};
  int64_t src_end{0};
  uint64_t prefix_count{0};
  uint64_t dst_block_offset_bytes{0};
  uint64_t dst_block_stride_bytes{0};
  uint64_t dst_block_bytes{0};
};

struct ConcatJobCandidate {
  std::string dst_name;
  std::vector<int64_t> dst_shape;
  std::vector<int64_t> dst_stride;
  std::string dtype;
  uint64_t dst_base_offset{0};
  uint64_t dst_size_bytes{0};
  uint64_t element_size{0};
  uint64_t prefix_count{0};
  uint64_t dst_block_stride_bytes{0};
  std::vector<ConcatSourceFragment> sources;
};

struct MappedCopyContract {
  loader::ByteRangeMap fallback_map;
  std::vector<TensorJobCandidate> tensor_job_candidates;
  std::vector<ConcatJobCandidate> concat_job_candidates;
};

struct ResolvedSourceBinding {
  loading::MaterializationSource source{loading::MaterializationSource::kDisk};
  SourceByteSpace source_byte_space{SourceByteSpace::kCanonical};
  bool source_layout_available{false};
  bool direct_write_capable{false};
  bool collective_eligible{false};
};

struct ResolvedMaterializationPlan {
  std::string artifact_id;
  uint64_t generation{0};
  std::optional<loading::VariantIdentity> variant;
  std::string canonical_index_json;
  loading::IntoTargetLayout target_layout;
  std::optional<MappedCopyContract> mapped_copy_contract;
};

enum class RouteIntentKind : std::uint8_t {
  kUnknown = 0,
  kDirectRemotePerTarget = 1,
  kRemoteBootstrapFanout = 2,
  kOwnerFileCollective = 3,
  kResidualFallback = 4,
};

enum class RouteProtocolFamily : std::uint8_t {
  kUnknown = 0,
  kAuto = 1,
  kNetwork = 2,
  kNvlink = 3,
  kPcie = 4,
};

enum class RouteFallbackPolicy : std::uint8_t {
  kUnknown = 0,
  kAllowResidualFallback = 1,
  kRequireExactCoverage = 2,
};

struct RouteHealthCostAnnotations {
  bool healthy{false};
  double health_score{0.0};
  double estimated_cost{0.0};
  bool rail_aligned{false};
};

struct RouteIntent {
  std::string src_endpoint_id;
  std::string dst_endpoint_id;
  RouteIntentKind kind{RouteIntentKind::kUnknown};
  RouteProtocolFamily preferred_protocol_family{RouteProtocolFamily::kUnknown};
  std::optional<std::string> local_ingress_anchor_id;
  RouteHealthCostAnnotations annotations;
  RouteFallbackPolicy fallback_policy{RouteFallbackPolicy::kUnknown};
};

enum class TransferLegKind : std::uint8_t {
  kUnknown = 0,
  kRemoteIngress = 1,
  kLocalFanout = 2,
  kOwnerCollective = 3,
  kResidualFallback = 4,
};

struct TopologyRuntimeSnapshot {
  bool available{false};
  uint64_t topology_generation{0};
  std::string local_node_id;
  // Planner follow-up will populate the selected local endpoint when anchor
  // selection is wired; Phase 1 keeps the field as scaffolding only.
  std::string local_endpoint_id;
};

struct TopologyStrategyInput {
  TopologyRuntimeSnapshot runtime;
  ResolvedSourceBinding source_binding;
  ResolvedMaterializationPlan resolved_plan;
  std::vector<std::string> target_endpoint_ids;
};

struct TransferLeg {
  TransferLegKind kind{TransferLegKind::kUnknown};
  std::string src_endpoint_id;
  std::string dst_endpoint_id;
  uint64_t planned_bytes{0};
  bool routed_preferred{false};
  bool exact_coverage{false};
};

struct TransferGroup {
  RouteIntent intent;
  std::string group_id;
  std::string anchor_endpoint_id;
  uint64_t planned_bytes{0};
  std::vector<TransferLeg> legs;
};

struct TopologyPlanDiagnostics {
  bool guided{false};
  std::string degrade_reason;
  std::vector<std::string> planner_notes;
};

struct TopologyGuidedPlan {
  std::vector<TransferGroup> groups;
  uint64_t routed_bytes{0};
  uint64_t local_fanout_bytes{0};
  uint64_t residual_fallback_bytes{0};
  TopologyPlanDiagnostics diagnostics;
};

struct ExecutionCommitReport {
  loading::MaterializationSource source{loading::MaterializationSource::kDisk};
  uint64_t requested_bytes{0};
  uint64_t committed_bytes{0};
  uint64_t fallback_bytes{0};
  bool collective_handled{false};
  bool direct_write_supported{false};
  bool source_ordered{false};
  std::string dominant_executor;
};

} // namespace tensorcast::store::runtime::ingestion::strategy
