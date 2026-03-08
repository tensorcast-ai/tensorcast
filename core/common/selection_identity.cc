// Copyright (c) 2026, TensorCast Team.

#include "core/common/selection_identity.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "core/common/artifact_hash.h"
#include "nlohmann/json.hpp"

namespace tensorcast::common {

namespace {

constexpr std::string_view kByteArtifactLayoutProfile = "tensorcast.byte_artifact.layout.v1\n";
constexpr std::string_view kByteArtifactSelectionProfile = "tensorcast.byte_artifact.selection.v1\n";

std::string sha256_as_string(std::string_view payload) {
  const std::vector<uint8_t> digest =
      sha256_digest_bytes(absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return std::string(reinterpret_cast<const char*>(digest.data()), digest.size());
}

} // namespace

std::string compute_logical_layout_hash_bytes(absl::Span<const uint8_t> index_bytes, bool needs_view_index) {
  std::string buffer;
  buffer.reserve(index_bytes.size() + 10);
  buffer.append(reinterpret_cast<const char*>(index_bytes.data()), index_bytes.size());
  if (needs_view_index) {
    buffer.append("|view");
  } else {
    buffer.append("|canonical");
  }
  const std::vector<uint8_t> digest =
      sha256_digest_bytes(absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size()));
  return std::string(reinterpret_cast<const char*>(digest.data()), digest.size());
}

std::string compute_view_subset_hash_bytes(absl::Span<const std::string> names) {
  if (names.empty()) {
    return std::string();
  }
  absl::flat_hash_set<std::string> seen;
  std::vector<std::string> unique;
  unique.reserve(names.size());
  for (const auto& name : names) {
    if (seen.insert(name).second) {
      unique.push_back(name);
    }
  }
  std::sort(unique.begin(), unique.end());
  nlohmann::json payload = nlohmann::json::array();
  for (const auto& name : unique) {
    payload.push_back(name);
  }
  const std::string serialized = payload.dump(-1, ' ', true);
  const std::vector<uint8_t> digest = sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()));
  return std::string(reinterpret_cast<const char*>(digest.data()), digest.size());
}

std::string compute_selection_hash_bytes(std::string_view view_id, std::optional<std::string_view> view_subset_hash) {
  std::string buffer;
  const bool has_subset = view_subset_hash.has_value() && !view_subset_hash->empty();
  buffer.reserve(view_id.size() + 12 + (has_subset ? view_subset_hash->size() : 4));
  buffer.append(view_id.data(), view_id.size());
  if (has_subset) {
    buffer.append(view_subset_hash->data(), view_subset_hash->size());
  } else {
    buffer.append("|all");
  }
  buffer.append("|v1");
  const std::vector<uint8_t> digest =
      sha256_digest_bytes(absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size()));
  return std::string(reinterpret_cast<const char*>(digest.data()), digest.size());
}

std::string compute_byte_artifact_logical_layout_hash_bytes() {
  static const std::string kHash = sha256_as_string(kByteArtifactLayoutProfile);
  return kHash;
}

std::string compute_byte_artifact_selection_hash_bytes() {
  static const std::string kHash = sha256_as_string(kByteArtifactSelectionProfile);
  return kHash;
}

} // namespace tensorcast::common
