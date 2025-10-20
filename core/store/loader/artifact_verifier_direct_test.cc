// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vector>

#include "absl/status/status.h"
#include "core/common/artifact_verification.h"

using Catch::Matchers::ContainsSubstring;
using tensorcast::common::ArtifactVerificationInfo;
using tensorcast::common::ArtifactVerifier;
using tensorcast::common::VerificationLevel;

TEST_CASE("ArtifactVerifier Direct Testing", "[verification][direct]") {
  const size_t test_size = 1024 * 4;
  std::vector<char> data(test_size);
  for (size_t i = 0; i < test_size; ++i) {
    data[i] = static_cast<char>('A' + (i % 26));
  }
  std::vector<void*> ptrs = {data.data()};
  std::vector<size_t> sizes = {test_size};

  SECTION("Generate and verify single-chunk data") {
    auto info_status = ArtifactVerifier::generate_verification_info(ptrs, sizes, -1);
    REQUIRE(info_status.ok());
    auto info = info_status.value();
    REQUIRE(info.artifact_size == test_size);
    REQUIRE(info.full_hash != 0);
    REQUIRE(ArtifactVerifier::verify_key_points(ptrs, sizes, info, -1).ok());
    REQUIRE(ArtifactVerifier::verify_artifact_data(ptrs, sizes, info, VerificationLevel::SPARSE_SAMPLING, -1).ok());
    REQUIRE(ArtifactVerifier::verify_artifact_data(ptrs, sizes, info, VerificationLevel::SEGMENT_HASHES, -1).ok());
    REQUIRE(ArtifactVerifier::verify_artifact_data(ptrs, sizes, info, VerificationLevel::FULL_HASH, -1).ok());
  }

  SECTION("Multi-chunk data produces same info as single-chunk") {
    const size_t chunk_size = 1024;
    std::vector<void*> mptrs;
    std::vector<size_t> msizes;
    for (size_t off = 0; off < test_size; off += chunk_size) {
      mptrs.push_back(data.data() + off);
      msizes.push_back(std::min(chunk_size, test_size - off));
    }
    auto multi_status = ArtifactVerifier::generate_verification_info(mptrs, msizes, -1);
    REQUIRE(multi_status.ok());
    auto multi_info = multi_status.value();
    auto single_status = ArtifactVerifier::generate_verification_info(ptrs, sizes, -1);
    REQUIRE(single_status.ok());
    auto single_info = single_status.value();
    REQUIRE(multi_info.full_hash == single_info.full_hash);
    REQUIRE(multi_info.key_values == single_info.key_values);
    REQUIRE(multi_info.segment_hashes == single_info.segment_hashes);
    REQUIRE(multi_info.sample_values == single_info.sample_values);
    REQUIRE(ArtifactVerifier::verify_artifact_data(mptrs, msizes, multi_info, VerificationLevel::FULL_HASH, -1).ok());
  }

  SECTION("Detect data corruption") {
    auto orig_status = ArtifactVerifier::generate_verification_info(ptrs, sizes, -1);
    REQUIRE(orig_status.ok());
    auto orig_info = orig_status.value();
    // Corrupt middle byte
    data[test_size / 2] = ~data[test_size / 2];
    REQUIRE(!ArtifactVerifier::verify_key_points(ptrs, sizes, orig_info, -1).ok());
    // Restore and corrupt segment
    data[test_size / 2] = ~data[test_size / 2]; // restore
    data[150] = ~data[150];
    REQUIRE(
        !ArtifactVerifier::verify_artifact_data(ptrs, sizes, orig_info, VerificationLevel::SEGMENT_HASHES, -1).ok());
    // Corrupt full hash
    auto cur_status = ArtifactVerifier::generate_verification_info(ptrs, sizes, -1);
    REQUIRE(cur_status.ok());
    auto cur_info = cur_status.value();
    cur_info.full_hash = 0xBADC0DE;
    REQUIRE(!ArtifactVerifier::verify_artifact_data(ptrs, sizes, cur_info, VerificationLevel::FULL_HASH, -1).ok());
  }

  SECTION("Invalid JSON handling") {
    auto invalid = ArtifactVerificationInfo::from_json("invalid");
    REQUIRE(!invalid.ok());
    REQUIRE_THAT(invalid.status().ToString(), ContainsSubstring("parse error"));
    auto incomplete = ArtifactVerificationInfo::from_json("{\"artifact_size\": 123}");
    REQUIRE(!incomplete.ok());
    CHECK(incomplete.status().code() == absl::StatusCode::kFailedPrecondition);
  }
}
