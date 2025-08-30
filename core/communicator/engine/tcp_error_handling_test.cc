// Copyright (c) 2025, TensorCast Team.

#include <atomic>
#include <chrono>
#include <thread>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "catch2/catch_test_macros.hpp"

#include "core/communicator/engine/engine.h"
#include "core/communicator/engine/gpu_net_stager.h"
#include "core/communicator/transport/partition_tensor.h"
#include "core/testing/test_helpers.h"

using namespace tensorcast;;
using namespace tensorcast::communicator;
using namespace tensorcast::communicator::test;

TEST_CASE("TCP Mode GPU Error Handling", "[communicator][tcp][gpu][error]") {
  SKIP_IF_NO_CUDA();

  SECTION("Invalid tensor registration") {
    communicator::CommunicatorConfig cfg; cfg.enable_rdma = false; /* disable RDMA */
    auto engine = std::make_shared<CommunicateEngine>(cfg);
    REQUIRE(engine->init("127.0.0.1", 0).ok());

    // Try to register with invalid device ID
    communicator::CommunicateEngine::RegisterTensorOptions opts;
    opts.register_mr = false; opts.needs_staging = true; opts.async = false;
    auto status = engine->register_tensor_ex(
        "invalid_tensor",
        0x12345678,
        1024,
        COMMUNICATE_ENGINE_DEV_GPU,
        999, // Invalid device ID
        opts);
    REQUIRE(!status.ok());
    INFO("Error message should mention invalid device, but got: " << status.message());
    // Accept various error messages about invalid GPU/device
    REQUIRE(
        (status.message().find("device") != std::string::npos || status.message().find("Device") != std::string::npos ||
         status.message().find("gpu") != std::string::npos || status.message().find("GPU") != std::string::npos));
  }

  SECTION("Staging buffer exhaustion recovery") {
    // Create a GPU stager with very limited buffers
    auto pool = std::make_shared<tensorcast::store::PinnedMemoryPool>(2 * 1024 * 1024, 1024 * 1024);
    GpuNetStager stager(1024 * 1024, 1, pool); // 1MB, only 1 buffer

    // Allocate GPU memory
    void* gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, 1024 * 1024).ok());

    auto tensor = std::make_shared<PartitionTensor>(
        "test", reinterpret_cast<uint64_t>(gpu_ptr), 1024 * 1024, COMMUNICATE_ENGINE_DEV_GPU, nullptr);
    tensor->set_device_id(0);

    // Try to stage again without releasing - should timeout quickly
    std::atomic<bool> staging_completed(false);
    absl::StatusOr<void*> staged2_ptr;
    std::thread staging_thread;

    // Stage once with RAII - should succeed
    {
      auto ptr1_or = stager.stage(tensor, 0, 512 * 1024);
      REQUIRE(ptr1_or.ok());
      void* staged1 = *ptr1_or;
      REQUIRE(staged1 != nullptr);

      staging_thread = std::thread([&]() {
        staged2_ptr = stager.stage(tensor, 512 * 1024, 512 * 1024);
        staging_completed = true;
      });

      // Wait a bit - staging should be blocked
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      INFO("Second staging request should be blocked while first buffer is in use, but it completed immediately");
      REQUIRE(!staging_completed);

      // Release the first buffer to allow the second staging to proceed
      REQUIRE(stager.release_staged_buffer(gsl::not_null<void*>{staged1}).ok());
    }

    // Wait for thread to complete
    staging_thread.join();
    REQUIRE(staging_completed);
    REQUIRE(staged2_ptr.ok());
    // Release the second buffer
    REQUIRE(stager.release_staged_buffer(gsl::not_null<void*>{*staged2_ptr}).ok());

    // No manual release needed - RAII handles it automatically

    REQUIRE(tensorcast::cuda::free(gpu_ptr).ok());
  }

  SECTION("Zero-size transfer handling") {
    communicator::CommunicatorConfig cfg; cfg.enable_rdma = false; /* disable RDMA */
    auto engine = std::make_shared<CommunicateEngine>(cfg);
    REQUIRE(engine->init("127.0.0.1", 0).ok());

    void* gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, 1024).ok());

    // Register with zero size should fail
    communicator::CommunicateEngine::RegisterTensorOptions opts;
    opts.register_mr = false; opts.needs_staging = true; opts.async = false;
    auto status = engine->register_tensor_ex(
        "zero_size",
        reinterpret_cast<uint64_t>(gpu_ptr),
        0, // Zero size
        COMMUNICATE_ENGINE_DEV_GPU,
        0,
        opts);
    INFO("Expected tensor registration to fail with zero size, but it succeeded");
    REQUIRE(!status.ok());
    INFO("Error message should mention zero/empty size issue, but got: " << status.message());
    REQUIRE(
        (status.message().find("size") != std::string::npos || status.message().find("zero") != std::string::npos ||
         status.message().find("empty") != std::string::npos));

    REQUIRE(tensorcast::cuda::free(gpu_ptr).ok());
  }

  SECTION("Out of bounds staging") {
    auto pool = std::make_shared<tensorcast::store::PinnedMemoryPool>(2 * 1024 * 1024, 1024 * 1024);
    GpuNetStager stager(1024 * 1024, 2, pool);

    void* gpu_ptr;
    const std::size_t tensor_size = 1024 * 1024;
    REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    auto tensor = std::make_shared<PartitionTensor>(
        "test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, nullptr);
    tensor->set_device_id(0);

    // Try to stage beyond tensor bounds
    auto result = stager.stage(tensor, tensor_size - 100, 200); // 100 bytes past end
    INFO(
        "Expected staging to fail when accessing beyond tensor bounds (offset "
        << (tensor_size - 100) << " + size 200 > tensor_size " << tensor_size << "), but it succeeded");
    REQUIRE(!result.ok());
    INFO("Expected InvalidArgument status for out-of-bounds staging, but got: " << result.status().ToString());
    REQUIRE(absl::IsInvalidArgument(result.status()));

    REQUIRE(tensorcast::cuda::free(gpu_ptr).ok());
  }
}
