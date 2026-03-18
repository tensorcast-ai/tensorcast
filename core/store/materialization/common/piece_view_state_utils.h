// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/view_utils.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::store::materialization::common {

namespace global_store = tensorcast::global_store::v1;

struct PieceViewStateRequest {
  std::string_view canonical_index_json;
  std::string_view view_id;
  const loader::BidirectionalViewPlan& plan;
  loader::SeekableSource& view_source;
  size_t leaf_chunk_bytes{0};
  uint64_t canonical_size_bytes{0};
};

struct PieceViewStatePayload {
  uint64_t view_size_bytes{0};
  uint64_t canonical_size_bytes{0};
  uint64_t canonical_bytes_covered{0};
  std::string view_data_hash;
  std::vector<tensorcast::store::view::CanonicalRange> canonical_ranges;
  std::vector<global_store::LeafWrite> leaf_writes;
  std::vector<global_store::PieceProofDigestWrite> proof_digests;
};

std::vector<tensorcast::store::view::CanonicalRange> canonical_ranges_from_write_plan(
    const loader::ViewWritePlan& write_plan);

absl::StatusOr<PieceViewStatePayload> build_piece_view_state_payload(const PieceViewStateRequest& request);

} // namespace tensorcast::store::materialization::common
