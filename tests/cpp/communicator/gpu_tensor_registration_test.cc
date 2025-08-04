// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <cuda_runtime.h>

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"

#include "core/communicator/engine/engine.h"
#include "tests/cpp/communicator/test_helpers.h"

using namespace stepcast::communicator;
using namespace stepcast::communicator::test;

TEST_CASE("TCP Mode GPU Tensor Registration", "[communicator][tcp][gpu]") {
  SKIP_IF_NO_CUDA();

  SECTION("Register GPU tensor in TCP mode") {
    // Create engine in TCP mode
    auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(engine->init("127.0.0.1", 0).ok());

    // Allocate GPU memory
    std::size_t tensor_size = 1024 * 1024; // 1MB
    void* gpu_ptr;
    REQUIRE(cudaMalloc(&gpu_ptr, tensor_size) == cudaSuccess);

    // Create test data
    auto test_data = create_test_pattern(tensor_size, 42);
    REQUIRE(cudaMemcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice) == cudaSuccess);

    // Register GPU tensor
    auto status = engine->register_tensor(
        "test_gpu_tensor",
        reinterpret_cast<uint64_t>(gpu_ptr),
        tensor_size,
        COMMUNICATE_ENGINE_DEV_GPU,
        0, // device_id
        false // sync
    );
    REQUIRE(status.ok());

    // Cleanup
    cudaFree(gpu_ptr);
  }
}