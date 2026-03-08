// Copyright (c) 2026, TensorCast Team.

#include "core/common/selection_identity.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"

namespace tensorcast::common {
namespace {

std::string to_hex_lower(std::string_view bytes) {
  return absl::AsciiStrToLower(absl::BytesToHexString(bytes));
}

TEST_CASE("View subset hash is order-independent and duplicate-insensitive", "[selection_identity]") {
  const std::vector<std::string> names = {"b", "a", "a"};
  const std::string subset_hash = compute_view_subset_hash_bytes(names);
  REQUIRE(to_hex_lower(subset_hash) == "0473ef2dc0d324ab659d3580c1134e9d812035905c4781fdd6d529b0c6860e13");
}

TEST_CASE("Selection hash normalizes empty subset marker to none", "[selection_identity]") {
  const std::string hash_none = compute_selection_hash_bytes("", std::nullopt);
  const std::string hash_empty_subset = compute_selection_hash_bytes("", std::string_view(""));
  REQUIRE(to_hex_lower(hash_none) == "9aa60740c2337b335b9279865963ff5385937c856dca7f7ebdfe49d1a27c8795");
  REQUIRE(hash_none == hash_empty_subset);
}

TEST_CASE("Selection hash golden vectors", "[selection_identity]") {
  const std::vector<std::string> names = {"b", "a", "a"};
  const std::string subset_hash = compute_view_subset_hash_bytes(names);
  const std::string hash_empty_view_subset = compute_selection_hash_bytes("", std::string_view(subset_hash));
  const std::string hash_view_none = compute_selection_hash_bytes("view-123", std::nullopt);
  const std::string hash_view_subset = compute_selection_hash_bytes("view-123", std::string_view(subset_hash));

  REQUIRE(to_hex_lower(hash_empty_view_subset) == "5edac2095f23abba792f8e98422b0e233e49a93ad736c7d64e23f573e43f9202");
  REQUIRE(to_hex_lower(hash_view_none) == "b9a1dff059db74a724ad7df7a72d5b7f524d5cc3725f057dddac302784d9fb8c");
  REQUIRE(to_hex_lower(hash_view_subset) == "09c927caf75fcb752fa6efdf1eeaa3bb3167b6e1d5d2517923c860c2db9d98dd");
}

TEST_CASE("Byte artifact selection identity vectors", "[selection_identity][byte_artifact]") {
  const std::string layout_hash = compute_byte_artifact_logical_layout_hash_bytes();
  const std::string selection_hash = compute_byte_artifact_selection_hash_bytes();
  REQUIRE(to_hex_lower(layout_hash) == "7ee7921aeedb6147ad6860dc2e0f398d56eeae79daf2ae947eae68eb4626349c");
  REQUIRE(to_hex_lower(selection_hash) == "b231ede6078bceb7b1046c23dd93b6f7cf4e35e4042ca87e634f1e2975953fee");
}

} // namespace
} // namespace tensorcast::common
