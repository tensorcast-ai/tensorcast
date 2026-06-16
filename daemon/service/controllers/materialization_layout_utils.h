// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::materialization_layout {

struct CanonicalIndexEntry {
  uint64_t logical_offset{0};
  uint64_t logical_length{0};
  uint64_t storage_offset{0};
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::string dtype;
};

struct CanonicalIndexTable {
  absl::flat_hash_map<std::string, CanonicalIndexEntry> entries;
  uint64_t logical_total_size{0};
};

struct TargetOffsetEntry {
  std::string name;
  std::string storage_id;
  uint64_t storage_offset{0};
  uint64_t logical_length{0};
};

absl::StatusOr<CanonicalIndexTable> parse_canonical_index(std::string_view index_json);

absl::StatusOr<std::shared_ptr<const CanonicalIndexTable>> parse_canonical_index_shared(std::string_view index_json);

// Uses an externally validated identity key for cache lookup instead of hashing
// the full index JSON. The key must uniquely name the exact index bytes.
absl::StatusOr<std::shared_ptr<const CanonicalIndexTable>> parse_canonical_index_shared_with_identity(
    std::string_view index_json,
    std::string_view identity_key);

absl::StatusOr<uint64_t> dtype_element_size(std::string_view dtype);

absl::StatusOr<uint64_t> product_dims(absl::Span<const int64_t> dims);

absl::StatusOr<std::vector<TargetOffsetEntry>> resolve_target_offsets(const v2::TargetLayout& layout);

} // namespace tensorcast::daemon::materialization_layout
