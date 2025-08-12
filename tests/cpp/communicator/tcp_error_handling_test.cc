// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <atomic>
#include <chrono>
#include <thread>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "catch2/catch_test_macros.hpp"

#include "core/communicator/engine/engine.h"
#include "core/communicator/engine/gpu_tcp_stager.h"
#include "core/communicator/transport/partition_tensor.h"
#include "tests/cpp/communicator/test_helpers.h"

using namespace stepcast::communicator;
using namespace stepcast::communicator::test;

TEST_CASE("TCP Mode GPU Error Handling", "[communicator][tcp][gpu][error]") {
  SKIP_IF_NO_CUDA();

  SECTION("Invalid tensor registration") {
    auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(engine->init("127.0.0.1", 0).ok());

    // Try to register with invalid device ID
    auto status = engine->register_tensor(
        "invalid_tensor",
        0x12345678,
        1024,
        COMMUNICATE_ENGINE_DEV_GPU,
        999, // Invalid device ID
        false);
    REQUIRE_MESSAGE(!status.ok(), 
                    "Expected tensor registration to fail with invalid device ID 999, but it succeeded");
    REQUIRE_MESSAGE(status.message().find("device") != std::string::npos || 
                    status.message().find("Device") != std::string::npos,
                    "Error message should mention invalid device, but got: " << status.message());
  }

  SECTION("Staging buffer exhaustion recovery") {
    // Create a stager with very limited buffers
    GpuTcpStager stager(1024 * 1024, 1); // 1MB, only 1 buffer

    // Allocate GPU memory
    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, 1024 * 1024).ok());

    auto tensor = std::make_shared<PartitionTensor>(
        "test", reinterpret_cast<uint64_t>(gpu_ptr), 1024 * 1024, COMMUNICATE_ENGINE_DEV_GPU, nullptr);
    tensor->set_device_id(0);

    // Try to stage again without releasing - should timeout quickly
    std::atomic<bool> staging_completed(false);
    absl::StatusOr<ScopedStagedBuffer> staged2_result;
    std::thread staging_thread;

    // Stage once with RAII - should succeed
    {
      auto result1 = stager.stage_scoped(tensor, 0, 512 * 1024);
      REQUIRE(result1.ok());
      auto staged1 = std::move(*result1);
      REQUIRE(staged1.valid());
      REQUIRE(staged1.data() != nullptr);
      REQUIRE(staged1.size() == 512 * 1024);

      staging_thread = std::thread([&]() {
        staged2_result = stager.stage_scoped(tensor, 512 * 1024, 512 * 1024);
        staging_completed = true;
      });

      // Wait a bit - staging should be blocked
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      REQUIRE_MESSAGE(!staging_completed, 
                      "Second staging request should be blocked while first buffer is in use, but it completed immediately");

      // Buffer is automatically released when staged1 goes out of scope
    }

    // Wait for thread to complete
    staging_thread.join();
    REQUIRE(staging_completed);

    // The second staging should have succeeded after the first buffer was released
    REQUIRE_MESSAGE(staged2_result.ok(), 
                    "Second staging should succeed after first buffer release, but failed with: " 
                    << (staged2_result.ok() ? "OK" : staged2_result.status().message()));

    // No manual release needed - RAII handles it automatically

    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
  }

  SECTION("Zero-size transfer handling") {
    auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(engine->init("127.0.0.1", 0).ok());

    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, 1024).ok());

    // Register with zero size should fail
    auto status = engine->register_tensor(
        "zero_size",
        reinterpret_cast<uint64_t>(gpu_ptr),
        0, // Zero size
        COMMUNICATE_ENGINE_DEV_GPU,
        0,
        false);
    REQUIRE_MESSAGE(!status.ok(),
                    "Expected tensor registration to fail with zero size, but it succeeded");
    REQUIRE_MESSAGE(status.message().find("size") != std::string::npos || 
                    status.message().find("zero") != std::string::npos ||
                    status.message().find("empty") != std::string::npos,
                    "Error message should mention zero/empty size issue, but got: " << status.message());

    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
  }

  SECTION("Out of bounds staging") {
    GpuTcpStager stager(1024 * 1024, 2);

    void* gpu_ptr;
    const std::size_t tensor_size = 1024 * 1024;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    auto tensor = std::make_shared<PartitionTensor>(
        "test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, nullptr);
    tensor->set_device_id(0);

    // Try to stage beyond tensor bounds
    auto result = stager.stage(tensor, tensor_size - 100, 200); // 100 bytes past end
    REQUIRE_MESSAGE(!result.ok(),
                    "Expected staging to fail when accessing beyond tensor bounds (offset " 
                    << (tensor_size - 100) << " + size 200 > tensor_size " << tensor_size << "), but it succeeded");
    REQUIRE_MESSAGE(absl::IsInvalidArgument(result.status()),
                    "Expected InvalidArgument status for out-of-bounds staging, but got: " 
                    << result.status().ToString());

    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
  }
}