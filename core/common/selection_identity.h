// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/types/span.h"

namespace tensorcast::common {

std::string compute_logical_layout_hash_bytes(absl::Span<const uint8_t> index_bytes, bool needs_view_index);

std::string compute_view_subset_hash_bytes(absl::Span<const std::string> names);

std::string compute_selection_hash_bytes(std::string_view view_id, std::optional<std::string_view> view_subset_hash);

} // namespace tensorcast::common
