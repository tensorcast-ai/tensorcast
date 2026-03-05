// Copyright (c) 2026, TensorCast Team.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "core/communicator/base/constants.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/routing/adapter.h"
#include "core/cuda/cuda_api.h"
#include "core/testing/test_helpers.h"

namespace {

using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;
using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
using tensorcast::communicator::engine::Communicator;
using tensorcast::communicator::routing::EndpointBinding;
using tensorcast::communicator::routing::NvlinkAdapter;
using tensorcast::communicator::routing::ReadRequest;

std::shared_ptr<Communicator> make_local_test_engine() {
  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/false);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/false);
  return std::make_shared<Communicator>(cfg, std::move(pools));
}

Communicator::RegisterTensorOptions no_mr_opts() {
  Communicator::RegisterTensorOptions opts;
  opts.register_mr = false;
  opts.needs_staging = false;
  opts.async = false;
  return opts;
}

} // namespace

TEST_CASE("NvlinkAdapter requires an engine to execute datapath", "[communicator][routing][adapter]") {
  NvlinkAdapter adapter;
  CHECK_FALSE(adapter.is_available());

  ReadRequest request;
  request.tensor_key = "missing";
  request.addr = 0;
  request.bytes = 0;
  request.dev_type = COMMUNICATE_ENGINE_DEV_CPU;
  request.dev_id = -1;
  auto result = adapter.read_tensor(request, EndpointBinding{}, EndpointBinding{}).get();
  CHECK(result.status.code() == absl::StatusCode::kFailedPrecondition);
}

TEST_CASE("NvlinkAdapter copies local CPU tensor with offset bounds", "[communicator][routing][adapter]") {
  auto engine = make_local_test_engine();
  NvlinkAdapter adapter(engine);
  REQUIRE(adapter.is_available());

  std::vector<uint32_t> source = {11, 22, 33, 44, 55, 66, 77, 88};
  std::vector<uint32_t> destination(4, 0);

  auto register_status = engine->register_tensor_ex(
      "cpu_tensor",
      reinterpret_cast<uint64_t>(source.data()),
      source.size() * sizeof(uint32_t),
      COMMUNICATE_ENGINE_DEV_CPU,
      -1,
      no_mr_opts());
  REQUIRE(register_status.ok());

  ReadRequest request;
  request.tensor_key = "cpu_tensor";
  request.addr = reinterpret_cast<uint64_t>(destination.data());
  request.bytes = destination.size() * sizeof(uint32_t);
  request.dev_type = COMMUNICATE_ENGINE_DEV_CPU;
  request.dev_id = -1;
  request.remote_offset = 2 * sizeof(uint32_t);

  auto result = adapter.read_tensor(request, EndpointBinding{}, EndpointBinding{}).get();
  REQUIRE(result.status.ok());
  CHECK(destination == std::vector<uint32_t>({33, 44, 55, 66}));
  CHECK(adapter.close(EndpointBinding{}).ok());
}

TEST_CASE("NvlinkAdapter returns not found and out of range errors", "[communicator][routing][adapter]") {
  auto engine = make_local_test_engine();
  NvlinkAdapter adapter(engine);

  std::vector<uint32_t> source = {1, 2, 3, 4};
  std::vector<uint32_t> destination(2, 0);
  auto register_status = engine->register_tensor_ex(
      "bounded_tensor",
      reinterpret_cast<uint64_t>(source.data()),
      source.size() * sizeof(uint32_t),
      COMMUNICATE_ENGINE_DEV_CPU,
      -1,
      no_mr_opts());
  REQUIRE(register_status.ok());

  ReadRequest missing;
  missing.tensor_key = "unknown_key";
  missing.addr = reinterpret_cast<uint64_t>(destination.data());
  missing.bytes = destination.size() * sizeof(uint32_t);
  missing.dev_type = COMMUNICATE_ENGINE_DEV_CPU;
  missing.dev_id = -1;
  auto missing_result = adapter.read_tensor(missing, EndpointBinding{}, EndpointBinding{}).get();
  CHECK(missing_result.status.code() == absl::StatusCode::kNotFound);

  ReadRequest out_of_range;
  out_of_range.tensor_key = "bounded_tensor";
  out_of_range.addr = reinterpret_cast<uint64_t>(destination.data());
  out_of_range.bytes = destination.size() * sizeof(uint32_t);
  out_of_range.dev_type = COMMUNICATE_ENGINE_DEV_CPU;
  out_of_range.dev_id = -1;
  out_of_range.remote_offset = 3 * sizeof(uint32_t);
  auto range_result = adapter.read_tensor(out_of_range, EndpointBinding{}, EndpointBinding{}).get();
  CHECK(range_result.status.code() == absl::StatusCode::kOutOfRange);
}

TEST_CASE("NvlinkAdapter copies local GPU tensor across devices", "[communicator][routing][adapter]") {
  int device_count = 0;
  auto count_status = tensorcast::cuda::get_device_count(&device_count);
  if (!count_status.ok() || device_count < 2) {
    SKIP("requires at least two CUDA devices (or fake CUDA backend)");
  }

  auto engine = make_local_test_engine();
  NvlinkAdapter adapter(engine);
  REQUIRE(adapter.is_available());

  constexpr size_t kElems = 1024;
  const size_t bytes = kElems * sizeof(uint32_t);
  std::vector<uint32_t> host_source(kElems, 0);
  for (size_t i = 0; i < host_source.size(); ++i) {
    host_source[i] = static_cast<uint32_t>(i * 3 + 7);
  }
  std::vector<uint32_t> host_destination(kElems, 0);

  void* src_gpu = nullptr;
  void* dst_gpu = nullptr;
  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::malloc(&src_gpu, bytes).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::malloc(&dst_gpu, bytes).ok());

  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::memcpy(src_gpu, host_source.data(), bytes, cudaMemcpyHostToDevice).ok());

  auto register_status = engine->register_tensor_ex(
      "gpu_tensor",
      reinterpret_cast<uint64_t>(src_gpu),
      bytes,
      COMMUNICATE_ENGINE_DEV_GPU,
      0,
      no_mr_opts());
  REQUIRE(register_status.ok());

  ReadRequest request;
  request.tensor_key = "gpu_tensor";
  request.addr = reinterpret_cast<uint64_t>(dst_gpu);
  request.bytes = bytes;
  request.dev_type = COMMUNICATE_ENGINE_DEV_GPU;
  request.dev_id = 1;
  request.remote_offset = 0;

  auto result = adapter.read_tensor(request, EndpointBinding{}, EndpointBinding{}).get();
  REQUIRE(result.status.ok());

  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::memcpy(host_destination.data(), dst_gpu, bytes, cudaMemcpyDeviceToHost).ok());
  CHECK(host_destination == host_source);

  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::free(src_gpu).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::free(dst_gpu).ok());
}
