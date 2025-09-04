// Copyright (c) 2025, TensorCast Team.

#include <chrono>
#include <cstring>
#include <future>
#include <vector>

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"

#include "core/common/cuda_api.h"
#include "core/communicator/engine/engine.h"
#include "core/testing/test_helpers.h"

using namespace tensorcast::communicator;
using namespace tensorcast::testing;
namespace communicator = tensorcast::communicator;

TEST_CASE("TCP Mode GPU to GPU Transfer", "[communicator][tcp][gpu][integration]") {
  SKIP_IF_NO_CUDA();

  SECTION("Basic GPU to GPU transfer via TCP") {
    // Find available ports for source and target engines
    int source_port = find_available_port();
    CAPTURE(source_port);
    REQUIRE(source_port > 0);

    int target_port = find_available_port(source_port + 1);
    CAPTURE(target_port);
    REQUIRE(target_port > 0);

    // Create source and target engines in TCP mode
    communicator::v1::CommunicatorConfig cfg1;
    cfg1.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<tensorcast::communicator::engine::CommunicateEngine>(cfg1);
    auto source_init_status = source_engine->init("127.0.0.1", source_port);
    CAPTURE(source_port, source_init_status.message());
    REQUIRE(source_init_status.ok());

    communicator::v1::CommunicatorConfig cfg2;
    cfg2.set_enable_rdma(false); /* disable RDMA */
    auto target_engine = std::make_shared<tensorcast::communicator::engine::CommunicateEngine>(cfg2);
    auto target_init_status = target_engine->init("127.0.0.1", target_port);
    CAPTURE(target_port, target_init_status.message());
    REQUIRE(target_init_status.ok());

    // Allocate source GPU tensor
    const std::size_t tensor_size = 4 * 1024 * 1024; // 4MB
    void* source_gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&source_gpu_ptr, tensor_size).ok());

    // Fill with test pattern
    auto test_data = create_test_pattern(tensor_size, 42);
    REQUIRE(tensorcast::cuda::memcpy(source_gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    // Register source tensor
    tensorcast::communicator::engine::CommunicateEngine::RegisterTensorOptions opts;
    opts.register_mr = false;
    opts.needs_staging = true;
    opts.async = false;
    auto status = source_engine->register_tensor_ex(
        "test_gpu_tensor",
        reinterpret_cast<uint64_t>(source_gpu_ptr),
        tensor_size,
        tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
        0, // device_id
        opts);
    REQUIRE(status.ok());

    // Allocate target GPU memory
    void* target_gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&target_gpu_ptr, tensor_size).ok());

    // Perform remote read
    auto future = target_engine->read_tensor(
        "test_gpu_tensor",
        reinterpret_cast<uint64_t>(target_gpu_ptr),
        tensor_size,
        tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
        0, // device_id
        "127.0.0.1",
        source_port);

    REQUIRE(future.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    auto result = future.get();
    CAPTURE(result.status.message());
    REQUIRE(result.status.ok());

    // Verify data
    std::vector<uint8_t> verify_data(tensor_size);
    REQUIRE(tensorcast::cuda::memcpy(verify_data.data(), target_gpu_ptr, tensor_size, cudaMemcpyDeviceToHost).ok());
    REQUIRE(verify_pattern(verify_data.data(), tensor_size, 42));

    // Cleanup
    REQUIRE(tensorcast::cuda::free(source_gpu_ptr).ok());
    REQUIRE(tensorcast::cuda::free(target_gpu_ptr).ok());
  }

  SECTION("GPU to CPU transfer via TCP") {
    // Find available ports for source and target engines
    int source_port = find_available_port();
    CAPTURE(source_port);
    REQUIRE(source_port > 0);

    int target_port = find_available_port(source_port + 1);
    CAPTURE(target_port);
    REQUIRE(target_port > 0);

    // Create source engine in TCP mode
    communicator::v1::CommunicatorConfig cfg3;
    cfg3.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<tensorcast::communicator::engine::CommunicateEngine>(cfg3);
    auto source_init_status = source_engine->init("127.0.0.1", source_port);
    CAPTURE(source_port, source_init_status.message());
    REQUIRE(source_init_status.ok());

    communicator::v1::CommunicatorConfig cfg4;
    cfg4.set_enable_rdma(false); /* disable RDMA */
    auto target_engine = std::make_shared<tensorcast::communicator::engine::CommunicateEngine>(cfg4);
    auto target_init_status = target_engine->init("127.0.0.1", target_port);
    CAPTURE(target_port, target_init_status.message());
    REQUIRE(target_init_status.ok());

    // Allocate source GPU tensor
    const std::size_t tensor_size = 2 * 1024 * 1024; // 2MB
    void* source_gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&source_gpu_ptr, tensor_size).ok());

    // Fill with test pattern
    auto test_data = create_test_pattern(tensor_size, 99);
    REQUIRE(tensorcast::cuda::memcpy(source_gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    // Register source tensor
    tensorcast::communicator::engine::CommunicateEngine::RegisterTensorOptions opts2;
    opts2.register_mr = false;
    opts2.needs_staging = true;
    opts2.async = false;
    REQUIRE(source_engine
                ->register_tensor_ex(
                    "test_gpu_to_cpu",
                    reinterpret_cast<uint64_t>(source_gpu_ptr),
                    tensor_size,
                    tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
                    0,
                    opts2)
                .ok());

    // Allocate target CPU memory
    void* target_cpu_ptr = std::malloc(tensor_size);
    REQUIRE(target_cpu_ptr != nullptr);

    // Perform remote read to CPU
    auto future = target_engine->read_tensor(
        "test_gpu_to_cpu",
        reinterpret_cast<uint64_t>(target_cpu_ptr),
        tensor_size,
        tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
        0,
        "127.0.0.1",
        source_port);

    REQUIRE(future.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    auto result = future.get();
    CAPTURE(result.status.message());
    REQUIRE(result.status.ok());

    // Verify data
    REQUIRE(verify_pattern(target_cpu_ptr, tensor_size, 99));

    // Cleanup
    REQUIRE(tensorcast::cuda::free(source_gpu_ptr).ok());
    std::free(target_cpu_ptr);
  }
}

TEST_CASE("TCP Mode Large Transfer Tests", "[communicator][tcp][gpu][stress]") {
  SKIP_IF_NO_CUDA();

  SECTION("Large GPU to GPU transfer") {
    // Find available ports for large transfer test
    int source_port = find_available_port();
    CAPTURE(source_port);
    REQUIRE(source_port > 0);

    int target_port = find_available_port(source_port + 1);
    CAPTURE(target_port);
    REQUIRE(target_port > 0);

    communicator::v1::CommunicatorConfig cfg5;
    cfg5.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<tensorcast::communicator::engine::CommunicateEngine>(cfg5);
    auto source_init_status = source_engine->init("127.0.0.1", source_port);
    CAPTURE(source_port, source_init_status.message());
    REQUIRE(source_init_status.ok());

    communicator::v1::CommunicatorConfig cfg6;
    cfg6.set_enable_rdma(false); /* disable RDMA */
    auto target_engine = std::make_shared<tensorcast::communicator::engine::CommunicateEngine>(cfg6);
    auto target_init_status = target_engine->init("127.0.0.1", target_port);
    CAPTURE(target_port, target_init_status.message());
    REQUIRE(target_init_status.ok());

    // Test with 256MB tensor
    const std::size_t tensor_size = 256 * 1024 * 1024;
    void* source_gpu_ptr;
    void* target_gpu_ptr;

    REQUIRE(tensorcast::cuda::malloc(&source_gpu_ptr, tensor_size).ok());
    REQUIRE(tensorcast::cuda::malloc(&target_gpu_ptr, tensor_size).ok());

    // Fill with pattern (use simpler pattern for large data)
    std::vector<uint8_t> pattern(1024 * 1024); // 1MB pattern
    for (std::size_t i = 0; i < pattern.size(); ++i) {
      pattern[i] = static_cast<uint8_t>(i % 256);
    }

    // Repeat pattern to fill tensor
    for (std::size_t offset = 0; offset < tensor_size; offset += pattern.size()) {
      std::size_t copy_size = std::min(pattern.size(), tensor_size - offset);
      REQUIRE(
          tensorcast::cuda::memcpy(
              static_cast<uint8_t*>(source_gpu_ptr) + offset, pattern.data(), copy_size, cudaMemcpyHostToDevice)
              .ok());
    }

    // Register and transfer
    tensorcast::communicator::engine::CommunicateEngine::RegisterTensorOptions opts3;
    opts3.register_mr = false;
    opts3.needs_staging = true;
    opts3.async = false;
    REQUIRE(source_engine
                ->register_tensor_ex(
                    "large_tensor",
                    reinterpret_cast<uint64_t>(source_gpu_ptr),
                    tensor_size,
                    tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
                    0,
                    opts3)
                .ok());

    auto future = target_engine->read_tensor(
        "large_tensor",
        reinterpret_cast<uint64_t>(target_gpu_ptr),
        tensor_size,
        tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
        0,
        "127.0.0.1",
        source_port);

    REQUIRE(future.wait_for(std::chrono::seconds(120)) == std::future_status::ready);
    auto result = future.get();
    CAPTURE(result.status.message());
    REQUIRE(result.status.ok());

    // Verify first and last 1MB
    std::vector<uint8_t> verify_start(1024 * 1024);
    std::vector<uint8_t> verify_end(1024 * 1024);

    REQUIRE(
        tensorcast::cuda::memcpy(verify_start.data(), target_gpu_ptr, verify_start.size(), cudaMemcpyDeviceToHost)
            .ok());

    REQUIRE(
        tensorcast::cuda::memcpy(
            verify_end.data(),
            static_cast<uint8_t*>(target_gpu_ptr) + tensor_size - verify_end.size(),
            verify_end.size(),
            cudaMemcpyDeviceToHost)
            .ok());

    // Check patterns
    for (std::size_t i = 0; i < verify_start.size(); ++i) {
      REQUIRE(verify_start[i] == static_cast<uint8_t>(i % 256));
    }
    for (std::size_t i = 0; i < verify_end.size(); ++i) {
      REQUIRE(verify_end[i] == static_cast<uint8_t>(i % 256));
    }

    REQUIRE(tensorcast::cuda::free(source_gpu_ptr).ok());
    REQUIRE(tensorcast::cuda::free(target_gpu_ptr).ok());
  }
}
