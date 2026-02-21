// Copyright (c) 2025-2026, TensorCast Team.

#include "catch2/catch_test_macros.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensorcast::communicator::transport::testing {
int compute_gpu_lane_for_subchunk(
    uint64_t chunk_base_offset,
    uint64_t sub_offset_in_chunk,
    uint64_t stage_unit,
    int lanes_to_use);
int compute_active_lanes(uint64_t total_bytes, size_t stage_unit, int conn_count, int buffers_per_flow_limit);
} // namespace tensorcast::communicator::transport::testing

namespace testing_ns = tensorcast::communicator::transport::testing;

namespace {

std::vector<int> CollectGpuLaneAssignments(
    uint64_t tensor_bytes,
    uint64_t stage_unit,
    uint64_t pool_chunk_size,
    int lanes_to_use) {
  std::vector<int> lanes;
  const uint64_t chunk_size = (tensor_bytes + lanes_to_use - 1) / lanes_to_use;

  for (uint64_t offset = 0; offset < tensor_bytes;) {
    uint64_t real_chunk_size = std::min<uint64_t>(chunk_size, tensor_bytes - offset);
    uint64_t remain_in_chunk = real_chunk_size;
    uint64_t sub_offset_in_chunk = 0;

    while (remain_in_chunk > 0) {
      uint64_t sub_chunk_size = std::min<uint64_t>(remain_in_chunk, pool_chunk_size);
      lanes.push_back(testing_ns::compute_gpu_lane_for_subchunk(offset, sub_offset_in_chunk, stage_unit, lanes_to_use));
      remain_in_chunk -= sub_chunk_size;
      sub_offset_in_chunk += sub_chunk_size;
    }

    offset += real_chunk_size;
  }

  return lanes;
}

} // namespace

TEST_CASE("MTcpTransport GPU lane selection follows segment order", "[mtcp_transport]") {
  constexpr uint64_t kTensorBytes = 128ULL * 1024 * 1024; // 128 MiB tensor
  constexpr uint64_t kStageUnit = 16ULL * 1024 * 1024; // 16 MiB staging chunk
  constexpr uint64_t kPoolChunkSize = 16ULL * 1024 * 1024; // matches staging buffer slice
  constexpr int kLanesToUse = 2; // two MTCP connections

  auto lanes = CollectGpuLaneAssignments(kTensorBytes, kStageUnit, kPoolChunkSize, kLanesToUse);

  REQUIRE(lanes.size() >= 2);
  std::vector<int> first_two{lanes.begin(), lanes.begin() + 2};
  REQUIRE(first_two == std::vector<int>({0, 1}));
}

TEST_CASE("MTcpTransport active lane calculation handles edge cases", "[mtcp_transport]") {
  SECTION("Clamp to available connections even for large tensors") {
    constexpr uint64_t kHugeTensor = 1ULL << 42; // 4 TiB tensor
    constexpr size_t kStageUnit = 16ULL * 1024 * 1024;
    REQUIRE(
        testing_ns::compute_active_lanes(kHugeTensor, kStageUnit, /*conn_count=*/4, /*buffers_per_flow_limit=*/0) == 4);
  }

  SECTION("Respect buffers_per_flow_limit when smaller than connections") {
    constexpr uint64_t kTensorBytes = 512ULL * 1024 * 1024;
    constexpr size_t kStageUnit = 32ULL * 1024 * 1024;
    REQUIRE(
        testing_ns::compute_active_lanes(kTensorBytes, kStageUnit, /*conn_count=*/8, /*buffers_per_flow_limit=*/3) ==
        3);
  }

  SECTION("Gracefully handle missing stage unit metadata") {
    constexpr uint64_t kTensorBytes = 64ULL * 1024 * 1024;
    REQUIRE(
        testing_ns::compute_active_lanes(
            kTensorBytes, /*stage_unit=*/0, /*conn_count=*/6, /*buffers_per_flow_limit=*/0) == 6);
  }

  SECTION("Zero-byte transfers still select a single lane") {
    REQUIRE(
        testing_ns::compute_active_lanes(/*total_bytes=*/0, /*stage_unit=*/16ULL * 1024 * 1024, /*conn_count=*/4, 0) ==
        1);
  }
}

TEST_CASE("MTcpTransport lane mapping respects request base offset", "[mtcp_transport]") {
  constexpr uint64_t kRequestBaseOffset = 1ULL << 30; // 1 GiB
  constexpr uint64_t kStageUnit = 8ULL * 1024 * 1024; // 8 MiB
  constexpr int kLanesToUse = 24;

  std::vector<int> lanes;
  lanes.reserve(6);
  for (uint64_t segment_offset = 0; segment_offset < 6 * kStageUnit; segment_offset += kStageUnit) {
    lanes.push_back(
        testing_ns::compute_gpu_lane_for_subchunk(
            kRequestBaseOffset + segment_offset, /*sub_offset_in_chunk=*/0, kStageUnit, kLanesToUse));
  }

  REQUIRE(lanes == std::vector<int>({8, 9, 10, 11, 12, 13}));
}
