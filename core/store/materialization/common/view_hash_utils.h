// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "core/common/memory/memory_location.h"
#include "core/store/materialization/contracts/view/view_plan.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/replica/replica.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store {

namespace loader {
using tensorcast::store::materialization::view::SelectionPlan;
using tensorcast::store::materialization::view::ViewPlan;
} // namespace loader

struct ViewHashConfig {
  size_t default_leaf_chunk_bytes = 4ULL * 1024 * 1024;
  StoreEngineOptions::ByteMappingConfig byte_mapping{};
};

// Centralized view hashing helper that exposes both in-memory and stream-based
// entry points so higher-level orchestration code shares a single implementation.
class ViewHashComputer {
 public:
  ViewHashComputer();
  explicit ViewHashComputer(ViewHashConfig config);

  // Computes the canonical multihash of a resolved view that is already
  // materialized in the replica's memory. The helper returns std::nullopt when
  // the view is empty, when the requested memory location is not resident, or
  // when hashing fails with a recoverable error (e.g., allocation view
  // unavailable, NOT_FOUND from the verification helpers).
  [[nodiscard]] std::optional<std::string> hash_replica_view(
      replica::Replica& replica,
      common::memory::MemoryLocation location,
      uint64_t view_size_bytes,
      std::optional<int> gpu_device_id) const;

  // Streams a resolved view from the provided SeekableSource, applying the
  // associated plan transforms when required, and returns the resulting view
  // hash. When leaf_chunk_bytes is omitted the config default is used.
  [[nodiscard]] absl::StatusOr<std::string> hash_view_from_source(
      loader::SeekableSource& base_source,
      const loader::ViewPlan& plan,
      size_t leaf_chunk_bytes) const;

  [[nodiscard]] absl::StatusOr<std::string> hash_view_from_source(
      loader::SeekableSource& base_source,
      const loader::ViewPlan& plan) const;

 private:
  size_t default_leaf_chunk_bytes_;
  StoreEngineOptions::ByteMappingConfig byte_mapping_;
};

// Legacy helper preserved for callers that have not yet been migrated to the
// ViewHashComputer instance wiring. Prefer ViewHashComputer::hash_replica_view.
std::optional<std::string> compute_view_data_hash(
    replica::Replica& replica,
    common::memory::MemoryLocation location,
    uint64_t view_size_bytes,
    std::optional<int> gpu_device_id);

} // namespace tensorcast::store
