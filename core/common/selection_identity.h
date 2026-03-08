// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorcast/common/v1/common.pb.h"

namespace tensorcast::common {

struct SelectionIdentity {
  std::string artifact_id;
  std::string logical_layout_hash;
  std::string selection_hash;

  bool operator==(const SelectionIdentity&) const = default;
};

struct SelectionIdentityHash {
  size_t operator()(const SelectionIdentity& identity) const {
    return absl::HashOf(identity.artifact_id, identity.logical_layout_hash, identity.selection_hash);
  }
};

std::string compute_logical_layout_hash_bytes(absl::Span<const uint8_t> index_bytes, bool needs_view_index);

std::string compute_view_subset_hash_bytes(absl::Span<const std::string> names);

std::string compute_selection_hash_bytes(std::string_view view_id, std::optional<std::string_view> view_subset_hash);

std::string compute_byte_artifact_logical_layout_hash_bytes();

std::string compute_byte_artifact_selection_hash_bytes();

absl::StatusOr<SelectionIdentity> build_selection_identity(const tensorcast::common::v1::ArtifactSelection& selection);

} // namespace tensorcast::common
