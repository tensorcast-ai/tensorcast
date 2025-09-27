// Copyright (c) 2025, TensorCast Team.

#include "catch2/catch_test_macros.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace tensorcast::communicator::transport::testing {
int compute_gpu_lane_for_subchunk(
    uint64_t chunk_base_offset,
    uint64_t sub_offset_in_chunk,
    uint64_t stage_unit,
    int lanes_to_use);
}

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
