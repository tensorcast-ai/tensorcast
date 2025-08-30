// Copyright (c) 2025, TensorCast Team.

#include "core/common/cuda_api.h"

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"

#include "core/communicator/engine/engine.h"
#include "core/testing/test_helpers.h"

using namespace tensorcast::communicator;
using namespace tensorcast::communicator::test;

TEST_CASE("TCP Mode GPU Tensor Registration", "[communicator][tcp][gpu]") {
  SKIP_IF_NO_CUDA();

  SECTION("Register GPU tensor in TCP mode") {
    // Create engine in TCP mode
    communicator::CommunicatorConfig cfg; cfg.enable_rdma = false; /* disable RDMA */
    auto engine = std::make_shared<CommunicateEngine>(cfg);
    REQUIRE(engine->init("127.0.0.1", 0).ok());

    // Allocate GPU memory
    std::size_t tensor_size = 1024 * 1024; // 1MB
    void* gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    // Create test data
    auto test_data = create_test_pattern(tensor_size, 42);
    REQUIRE(tensorcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    // Register GPU tensor
    communicator::CommunicateEngine::RegisterTensorOptions opts; opts.register_mr=false; opts.needs_staging=true; opts.async=false;
    auto status = engine->register_tensor_ex(
        "test_gpu_tensor",
        reinterpret_cast<uint64_t>(gpu_ptr),
        tensor_size,
        COMMUNICATE_ENGINE_DEV_GPU,
        0, // device_id
        opts);
    REQUIRE(status.ok());

    // Cleanup
    REQUIRE(tensorcast::cuda::free(gpu_ptr).ok());
  }
}
