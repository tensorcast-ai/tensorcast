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
