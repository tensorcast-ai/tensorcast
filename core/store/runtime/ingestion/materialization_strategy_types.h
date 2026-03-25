// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/contracts/representation_contract.h"

namespace tensorcast::store::runtime::ingestion::strategy {

// The selected source may expose canonical bytes or an already materialized
// view/mapped byte-space. This is independent from the request's view_id:
// mapped-target requests can carry a target byte-space identity while still
// reconstructing the result from canonical/disk fallback.
enum class SourceByteSpace : std::uint8_t {
  kCanonical = 0,
  kView = 1,
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
  std::optional<materialization::contracts::RepresentationTransformContract> representation_transform_contract;
  std::optional<materialization::contracts::RepresentationWorkPlan> representation_work_plan;
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
