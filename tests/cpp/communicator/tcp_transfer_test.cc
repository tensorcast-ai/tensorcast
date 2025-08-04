// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <cuda_runtime.h>
#include <cstring>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "catch2/catch_test_macros.hpp"

#include "core/communicator/engine/engine.h"
#include "tests/cpp/communicator/test_helpers.h"

using namespace stepcast::communicator;
using namespace stepcast::communicator::test;

TEST_CASE("TCP Mode GPU to GPU Transfer", "[communicator][tcp][gpu][integration]") {
  SKIP_IF_NO_CUDA();

  SECTION("Basic GPU to GPU transfer via TCP") {
    // Create source and target engines in TCP mode
    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", 50055).ok());

    auto target_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(target_engine->init("127.0.0.1", 50056).ok());

    // Allocate source GPU tensor
    const std::size_t tensor_size = 4 * 1024 * 1024; // 4MB
    void* source_gpu_ptr;
    REQUIRE(cudaMalloc(&source_gpu_ptr, tensor_size) == cudaSuccess);

    // Fill with test pattern
    auto test_data = create_test_pattern(tensor_size, 42);
    REQUIRE(cudaMemcpy(source_gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice) == cudaSuccess);

    // Register source tensor
    auto status = source_engine->register_tensor(
        "test_gpu_tensor",
        reinterpret_cast<uint64_t>(source_gpu_ptr),
        tensor_size,
        COMMUNICATE_ENGINE_DEV_GPU,
        0, // device_id
        false // sync
    );
    REQUIRE(status.ok());

    // Allocate target GPU memory
    void* target_gpu_ptr;
    REQUIRE(cudaMalloc(&target_gpu_ptr, tensor_size) == cudaSuccess);

    // Perform remote read
    auto future = target_engine->read_tensor(
        "test_gpu_tensor",
        reinterpret_cast<uint64_t>(target_gpu_ptr),
        tensor_size,
        COMMUNICATE_ENGINE_DEV_GPU,
        0, // device_id
        "127.0.0.1",
        50055);

    auto result = future.get();
    CAPTURE(result.status.message());
    REQUIRE(result.status.ok());

    // Verify data
    std::vector<uint8_t> verify_data(tensor_size);
    REQUIRE(cudaMemcpy(verify_data.data(), target_gpu_ptr, tensor_size, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(verify_pattern(verify_data.data(), tensor_size, 42));

    // Cleanup
    cudaFree(source_gpu_ptr);
    cudaFree(target_gpu_ptr);
  }

  SECTION("GPU to CPU transfer via TCP") {
    // Create source engine in TCP mode
    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", 50057).ok());

    auto target_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(target_engine->init("127.0.0.1", 50058).ok());

    // Allocate source GPU tensor
    const std::size_t tensor_size = 2 * 1024 * 1024; // 2MB
    void* source_gpu_ptr;
    REQUIRE(cudaMalloc(&source_gpu_ptr, tensor_size) == cudaSuccess);

    // Fill with test pattern
    auto test_data = create_test_pattern(tensor_size, 99);
    REQUIRE(cudaMemcpy(source_gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice) == cudaSuccess);

    // Register source tensor
    REQUIRE(source_engine
                ->register_tensor(
                    "test_gpu_to_cpu",
                    reinterpret_cast<uint64_t>(source_gpu_ptr),
                    tensor_size,
                    COMMUNICATE_ENGINE_DEV_GPU,
                    0,
                    false)
                .ok());

    // Allocate target CPU memory
    void* target_cpu_ptr = std::malloc(tensor_size);
    REQUIRE(target_cpu_ptr != nullptr);

    // Perform remote read to CPU
    auto future = target_engine->read_tensor(
        "test_gpu_to_cpu",
        reinterpret_cast<uint64_t>(target_cpu_ptr),
        tensor_size,
        COMMUNICATE_ENGINE_DEV_CPU,
        0,
        "127.0.0.1",
        50057);

    auto result = future.get();
    CAPTURE(result.status.message());
    REQUIRE(result.status.ok());

    // Verify data
    REQUIRE(verify_pattern(target_cpu_ptr, tensor_size, 99));

    // Cleanup
    cudaFree(source_gpu_ptr);
    std::free(target_cpu_ptr);
  }
}

TEST_CASE("TCP Mode Large Transfer Tests", "[communicator][tcp][gpu][stress]") {
  SKIP_IF_NO_CUDA();

  SECTION("Large GPU to GPU transfer") {
    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", 50059).ok());

    auto target_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(target_engine->init("127.0.0.1", 50060).ok());

    // Test with 256MB tensor
    const std::size_t tensor_size = 256 * 1024 * 1024;
    void* source_gpu_ptr;
    void* target_gpu_ptr;

    REQUIRE(cudaMalloc(&source_gpu_ptr, tensor_size) == cudaSuccess);
    REQUIRE(cudaMalloc(&target_gpu_ptr, tensor_size) == cudaSuccess);

    // Fill with pattern (use simpler pattern for large data)
    std::vector<uint8_t> pattern(1024 * 1024); // 1MB pattern
    for (std::size_t i = 0; i < pattern.size(); ++i) {
      pattern[i] = static_cast<uint8_t>(i % 256);
    }

    // Repeat pattern to fill tensor
    for (std::size_t offset = 0; offset < tensor_size; offset += pattern.size()) {
      std::size_t copy_size = std::min(pattern.size(), tensor_size - offset);
      REQUIRE(
          cudaMemcpy(
              static_cast<uint8_t*>(source_gpu_ptr) + offset, pattern.data(), copy_size, cudaMemcpyHostToDevice) ==
          cudaSuccess);
    }

    // Register and transfer
    REQUIRE(source_engine
                ->register_tensor(
                    "large_tensor",
                    reinterpret_cast<uint64_t>(source_gpu_ptr),
                    tensor_size,
                    COMMUNICATE_ENGINE_DEV_GPU,
                    0,
                    false)
                .ok());

    auto future = target_engine->read_tensor(
        "large_tensor",
        reinterpret_cast<uint64_t>(target_gpu_ptr),
        tensor_size,
        COMMUNICATE_ENGINE_DEV_GPU,
        0,
        "127.0.0.1",
        50059);

    auto result = future.get();
    CAPTURE(result.status.message());
    REQUIRE(result.status.ok());

    // Verify first and last 1MB
    std::vector<uint8_t> verify_start(1024 * 1024);
    std::vector<uint8_t> verify_end(1024 * 1024);

    REQUIRE(
        cudaMemcpy(verify_start.data(), target_gpu_ptr, verify_start.size(), cudaMemcpyDeviceToHost) == cudaSuccess);

    REQUIRE(
        cudaMemcpy(
            verify_end.data(),
            static_cast<uint8_t*>(target_gpu_ptr) + tensor_size - verify_end.size(),
            verify_end.size(),
            cudaMemcpyDeviceToHost) == cudaSuccess);

    // Check patterns
    for (std::size_t i = 0; i < verify_start.size(); ++i) {
      REQUIRE(verify_start[i] == static_cast<uint8_t>(i % 256));
    }
    for (std::size_t i = 0; i < verify_end.size(); ++i) {
      REQUIRE(verify_end[i] == static_cast<uint8_t>(i % 256));
    }

    cudaFree(source_gpu_ptr);
    cudaFree(target_gpu_ptr);
  }
}