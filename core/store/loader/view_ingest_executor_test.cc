// Copyright (c) 2025, TensorCast Team.

#include "core/store/loader/view_ingest_executor.h"

#include <cstring>
#include <vector>

#include "absl/types/span.h"
#include "catch2/catch_test_macros.hpp"
#include "core/common/cuda_api.h"
#include "core/store/loader/view_planner.h"
#include "nlohmann/json.hpp"

using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::loader::BidirectionalViewPlan;
using tensorcast::store::loader::NarrowOp;
using tensorcast::store::loader::TensorViewOps;
using tensorcast::store::loader::TransposeOp;
using tensorcast::store::loader::ViewOp;
using tensorcast::store::loader::ViewPlanner;
using tensorcast::store::loader::ViewSpec;
using tensorcast::store::loader::ViewWritePlan;

namespace cuda = tensorcast::cuda;

namespace {

nlohmann::json tensor_entry(
    uint64_t offset,
    uint64_t size,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& stride,
    const std::string& dtype,
    uint64_t storage_offset) {
  nlohmann::json arr = nlohmann::json::array();
  arr.push_back(offset);
  arr.push_back(size);
  nlohmann::json j_shape = nlohmann::json::array();
  for (auto v : shape) {
    j_shape.push_back(v);
  }
  nlohmann::json j_stride = nlohmann::json::array();
  for (auto v : stride) {
    j_stride.push_back(v);
  }
  arr.push_back(j_shape);
  arr.push_back(j_stride);
  arr.push_back(dtype);
  arr.push_back(storage_offset);
  return arr;
}

absl::Span<const std::byte> as_byte_span(const std::vector<uint8_t>& data) {
  return absl::Span<const std::byte>(reinterpret_cast<const std::byte*>(data.data()), data.size());
}

std::vector<uint8_t> to_bytes(const std::vector<float>& values) {
  std::vector<uint8_t> out(values.size() * sizeof(float));
  std::memcpy(out.data(), values.data(), out.size());
  return out;
}

std::vector<uint8_t> transpose_2x3_to_3x2(const std::vector<float>& canonical) {
  std::vector<float> transposed(6);
  transposed[0] = canonical[0];
  transposed[1] = canonical[3];
  transposed[2] = canonical[1];
  transposed[3] = canonical[4];
  transposed[4] = canonical[2];
  transposed[5] = canonical[5];
  return to_bytes(transposed);
}

BidirectionalViewPlan compute_plan(const std::string& canonical_index_json, const ViewSpec& spec) {
  auto plan_or = ViewPlanner::compute_bidirectional_view_plan(canonical_index_json, spec);
  REQUIRE(plan_or.ok());
  return std::move(*plan_or);
}

} // namespace

TEST_CASE("ViewIngestExecutor ingests identity view on CPU", "[view_ingest_executor]") {
  nlohmann::json index = nlohmann::json::object();
  index["weights"] = tensor_entry(
      /*offset=*/0,
      /*size=*/32,
      /*shape=*/{8},
      /*stride=*/{1},
      /*dtype=*/"torch.float32",
      /*storage_offset=*/0);
  const std::string canonical = index.dump();

  ViewSpec spec; // identity
  BidirectionalViewPlan bidir = compute_plan(canonical, spec);

  std::vector<uint8_t> canonical_buffer(32, 0);
  std::vector<uint8_t> view_buffer(32);
  for (size_t i = 0; i < view_buffer.size(); ++i) {
    view_buffer[i] = static_cast<uint8_t>(i + 1);
  }

  tensorcast::store::loader::ViewIngestExecutor executor(std::move(bidir.write), std::move(bidir.inverse_transform));

  REQUIRE(executor.expected_view_bytes() == view_buffer.size());
  auto ingest_status = executor.ingest_chunk(
      /*view_offset=*/0,
      as_byte_span(view_buffer),
      MemoryLocation::CPU,
      canonical_buffer.data(),
      /*device_id=*/0);
  REQUIRE(ingest_status.ok());
  CHECK(executor.is_complete());

  auto finalize_status = executor.finalize(MemoryLocation::CPU, canonical_buffer.data(), 0);
  REQUIRE(finalize_status.ok());

  CHECK(canonical_buffer == view_buffer);
}

TEST_CASE("ViewIngestExecutor handles narrow view in multiple chunks", "[view_ingest_executor]") {
  nlohmann::json index = nlohmann::json::object();
  index["activation"] = tensor_entry(
      /*offset=*/0,
      /*size=*/32,
      /*shape=*/{2, 4},
      /*stride=*/{4, 1},
      /*dtype=*/"torch.float32",
      /*storage_offset=*/0);
  const std::string canonical = index.dump();

  ViewSpec spec;
  TensorViewOps ops;
  ops.ops.push_back(ViewOp::Narrow(NarrowOp{.dim = 1, .start = 1, .length = 2}));
  spec.tensors.emplace("activation", ops);

  BidirectionalViewPlan bidir = compute_plan(canonical, spec);
  REQUIRE_FALSE(bidir.write.chunks.empty());

  std::vector<uint8_t> canonical_buffer(32, 0);
  std::vector<uint8_t> view_buffer(16);
  for (size_t i = 0; i < view_buffer.size(); ++i) {
    view_buffer[i] = static_cast<uint8_t>(100 + i);
  }

  tensorcast::store::loader::ViewIngestExecutor executor(std::move(bidir.write), std::move(bidir.inverse_transform));

  // First half of the view payload.
  auto first_half =
      absl::Span<const std::byte>(reinterpret_cast<const std::byte*>(view_buffer.data()), view_buffer.size() / 2);
  auto ingest_status = executor.ingest_chunk(
      /*view_offset=*/0,
      first_half,
      MemoryLocation::CPU,
      canonical_buffer.data(),
      /*device_id=*/0);
  REQUIRE(ingest_status.ok());
  CHECK_FALSE(executor.is_complete());

  // Second half of the payload.
  auto second_half = absl::Span<const std::byte>(
      reinterpret_cast<const std::byte*>(view_buffer.data() + view_buffer.size() / 2), view_buffer.size() / 2);
  ingest_status = executor.ingest_chunk(
      /*view_offset=*/static_cast<uint64_t>(view_buffer.size() / 2),
      second_half,
      MemoryLocation::CPU,
      canonical_buffer.data(),
      /*device_id=*/0);
  REQUIRE(ingest_status.ok());
  CHECK(executor.is_complete());

  auto finalize_status = executor.finalize(MemoryLocation::CPU, canonical_buffer.data(), 0);
  REQUIRE(finalize_status.ok());

  // Expected: slices copied into canonical offsets 4..11 and 20..27.
  std::vector<uint8_t> expected(32, 0);
  std::memcpy(expected.data() + 4, view_buffer.data(), 8);
  std::memcpy(expected.data() + 20, view_buffer.data() + 8, 8);
  CHECK(canonical_buffer == expected);
}

TEST_CASE("ViewIngestExecutor applies inverse transpose", "[view_ingest_executor]") {
  nlohmann::json index = nlohmann::json::object();
  index["tensor"] = tensor_entry(
      /*offset=*/0,
      /*size=*/24,
      /*shape=*/{2, 3},
      /*stride=*/{3, 1},
      /*dtype=*/"torch.float32",
      /*storage_offset=*/0);
  const std::string canonical = index.dump();

  ViewSpec spec;
  TensorViewOps ops;
  ops.ops.push_back(ViewOp::Transpose(TransposeOp{.dim0 = 0, .dim1 = 1}));
  spec.tensors.emplace("tensor", ops);

  BidirectionalViewPlan bidir = compute_plan(canonical, spec);

  std::vector<float> canonical_values = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
  std::vector<uint8_t> canonical_buffer = to_bytes(std::vector<float>(canonical_values.size(), 0.f));
  std::vector<uint8_t> view_buffer = transpose_2x3_to_3x2(canonical_values);

  tensorcast::store::loader::ViewIngestExecutor executor(std::move(bidir.write), std::move(bidir.inverse_transform));

  auto ingest_status = executor.ingest_chunk(
      /*view_offset=*/0,
      as_byte_span(view_buffer),
      MemoryLocation::CPU,
      canonical_buffer.data(),
      /*device_id=*/0);
  REQUIRE(ingest_status.ok());
  CHECK(executor.is_complete());

  auto finalize_status = executor.finalize(MemoryLocation::CPU, canonical_buffer.data(), 0);
  REQUIRE(finalize_status.ok());

  std::vector<uint8_t> expected = to_bytes(canonical_values);
  CHECK(canonical_buffer == expected);
}

TEST_CASE("ViewIngestExecutor ingests into GPU memory (fake backend)", "[view_ingest_executor][gpu]") {
  nlohmann::json index = nlohmann::json::object();
  index["weights"] = tensor_entry(
      /*offset=*/0,
      /*size=*/16,
      /*shape=*/{4},
      /*stride=*/{1},
      /*dtype=*/"torch.float32",
      /*storage_offset=*/0);
  const std::string canonical = index.dump();

  ViewSpec spec;
  BidirectionalViewPlan bidir = compute_plan(canonical, spec);

  std::vector<uint8_t> view_buffer(16);
  for (size_t i = 0; i < view_buffer.size(); ++i) {
    view_buffer[i] = static_cast<uint8_t>(200 + i);
  }

  void* device_ptr = nullptr;
  REQUIRE(cuda::set_device(0).ok());
  auto alloc_status = cuda::malloc(&device_ptr, view_buffer.size());
  REQUIRE(alloc_status.ok());

  tensorcast::store::loader::ViewIngestExecutor executor(std::move(bidir.write), std::move(bidir.inverse_transform));

  auto ingest_status = executor.ingest_chunk(
      /*view_offset=*/0,
      as_byte_span(view_buffer),
      MemoryLocation::GPU,
      device_ptr,
      /*device_id=*/0);
  REQUIRE(ingest_status.ok());
  CHECK(executor.is_complete());

  auto finalize_status = executor.finalize(MemoryLocation::GPU, device_ptr, 0);
  REQUIRE(finalize_status.ok());

  std::vector<uint8_t> host_buffer(view_buffer.size(), 0);
  auto copy_status = cuda::memcpy(host_buffer.data(), device_ptr, host_buffer.size(), cudaMemcpyDeviceToHost);
  REQUIRE(copy_status.ok());

  CHECK(host_buffer == view_buffer);

  auto free_status = cuda::free(device_ptr);
  REQUIRE(free_status.ok());
}
