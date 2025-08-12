// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include "core/common/cuda_api.h"
#include "core/communicator/engine/engine.h"

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"

#include "core/communicator/engine/gpu_tcp_stager.h"
#include "core/communicator/transport/partition_tensor.h"
#include "tests/cpp/communicator/test_helpers.h"

using namespace stepcast::communicator;
using namespace stepcast::communicator::test;

TEST_CASE("GpuTcpStager Basic Operations", "[communicator][tcp][gpu][stager]") {
  SKIP_IF_NO_CUDA();

  SECTION("Basic staging operations") {
    GpuTcpStager stager(1024 * 1024, 2); // 1MB chunks, 2 buffers

    // Create a GPU tensor
    std::size_t tensor_size = 512 * 1024; // 512KB
    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    auto test_data = create_test_pattern(tensor_size, 99);
    REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    // Create PartitionTensor
    auto tensor = std::make_shared<PartitionTensor>(
        "test_tensor", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, nullptr);
    tensor->set_device_id(0);

    // Stage the data
    auto staged_result = stager.stage(tensor, 0, tensor_size);
    REQUIRE(staged_result.ok());

    void* staged_ptr = *staged_result;
    REQUIRE(staged_ptr != nullptr);

    // Verify staged data
    REQUIRE(verify_pattern(staged_ptr, tensor_size, 99));

    // Release the buffer
    REQUIRE(stager.release_staged_buffer(staged_ptr).ok());

    // Check stats
    auto stats = stager.get_stats();
    REQUIRE(stats.total_staged_bytes == tensor_size);
    REQUIRE(stats.total_stage_calls == 1);

    // Cleanup
    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
  }

  SECTION("Chunk size exceeded") {
    const std::size_t chunk_size = 1024 * 1024; // 1MB
    const std::size_t tensor_size = 2 * 1024 * 1024; // 2MB

    GpuTcpStager stager(chunk_size, 2);

    // Allocate GPU memory
    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    auto tensor = std::make_shared<PartitionTensor>(
        "test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, nullptr);
    tensor->set_device_id(0);

    // Try to stage more than chunk size - should fail
    auto result = stager.stage(tensor, 0, tensor_size);
    REQUIRE_FALSE(result.ok());
    REQUIRE(absl::IsInvalidArgument(result.status()));

    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
  }

  SECTION("Buffer exhaustion and recovery") {
    const std::size_t chunk_size = 1024 * 1024; // 1MB
    const std::size_t tensor_size = 512 * 1024; // 512KB
    const std::size_t num_chunks = 2;

    GpuTcpStager stager(chunk_size, num_chunks);

    // Allocate GPU memory
    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    auto tensor = std::make_shared<PartitionTensor>(
        "test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, nullptr);
    tensor->set_device_id(0);

    // Stage multiple times without releasing
    std::vector<void*> staged_ptrs;
    for (std::size_t i = 0; i < num_chunks; ++i) {
      auto result = stager.stage(tensor, 0, tensor_size);
      REQUIRE(result.ok());
      staged_ptrs.push_back(*result);
    }

    // Release one buffer
    REQUIRE(stager.release_staged_buffer(staged_ptrs[0]).ok());

    // Now staging should succeed
    auto result = stager.stage(tensor, 0, tensor_size);
    REQUIRE(result.ok());

    // Release all buffers
    REQUIRE(stager.release_staged_buffer(*result).ok());
    for (std::size_t i = 1; i < staged_ptrs.size(); ++i) {
      REQUIRE(stager.release_staged_buffer(staged_ptrs[i]).ok());
    }

    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
  }

  SECTION("Concurrent staging stress test") {
    const std::size_t chunk_size = 1024 * 1024; // 1MB
    const std::size_t tensor_size = 256 * 1024; // 256KB
    const std::size_t num_threads = 4;
    const std::size_t iterations = 10;

    GpuTcpStager stager(chunk_size, num_threads);

    // Allocate GPU memory
    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    // Fill with test pattern
    auto test_data = create_test_pattern(tensor_size, 42);
    REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    auto tensor = std::make_shared<PartitionTensor>(
        "test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, nullptr);
    tensor->set_device_id(0);

    std::atomic<int> success_count(0);
    std::atomic<int> error_count(0);

    auto worker = [&]() {
      for (std::size_t i = 0; i < iterations; ++i) {
        auto result = stager.stage(tensor, 0, tensor_size);
        if (result.ok()) {
          void* staged_ptr = *result;

          // Verify data
          if (verify_pattern(staged_ptr, tensor_size, 42)) {
            success_count++;
          } else {
            error_count++;
          }

          // Simulate some work
          std::this_thread::sleep_for(std::chrono::milliseconds(10));

          // Release buffer
          REQUIRE(stager.release_staged_buffer(staged_ptr).ok());
        } else {
          error_count++;
        }
      }
    };

    // Launch threads
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < num_threads; ++i) {
      threads.emplace_back(worker);
    }

    // Wait for completion
    for (auto& t : threads) {
      t.join();
    }

    REQUIRE(success_count == num_threads * iterations);
    REQUIRE(error_count == 0);

    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
  }

  SECTION("Multi-GPU support") {
    int device_count;
    REQUIRE(stepcast::cuda::get_device_count(&device_count).ok());
    if (device_count < 2) {
      SKIP("Multi-GPU test requires at least 2 GPUs");
    }

    const std::size_t chunk_size = 1024 * 1024; // 1MB
    const std::size_t tensor_size = 512 * 1024; // 512KB

    GpuTcpStager stager(chunk_size, 2);

    // Test staging from different GPUs
    for (int device_id = 0; device_id < std::min(device_count, 2); ++device_id) {
      REQUIRE(stepcast::cuda::set_device(device_id).ok());

      void* gpu_ptr;
      REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

      // Fill with device-specific pattern
      auto test_data = create_test_pattern(tensor_size, device_id * 100);
      REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

      auto tensor = std::make_shared<PartitionTensor>(
          "test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, nullptr);
      tensor->set_device_id(device_id);

      // Stage from this GPU
      auto result = stager.stage(tensor, 0, tensor_size);
      REQUIRE(result.ok());

      // Verify staged data
      void* staged_ptr = *result;
      REQUIRE(verify_pattern(staged_ptr, tensor_size, device_id * 100));

      // Release and cleanup
      REQUIRE(stager.release_staged_buffer(staged_ptr).ok());
      REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
    }
  }
}

TEST_CASE("GpuTcpStager Advanced Tests", "[communicator][tcp][gpu][stager][advanced]") {
  SKIP_IF_NO_CUDA();

  SECTION("Buffer pool exhaustion with network delays") {
    const std::size_t chunk_size = 1024 * 1024; // 1MB
    const std::size_t num_buffers = 2;
    GpuTcpStager stager(chunk_size, num_buffers);

    // Allocate GPU memory
    const std::size_t tensor_size = 10 * 1024 * 1024; // 10MB
    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    auto test_data = create_test_pattern(tensor_size, 55);
    REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    auto tensor = std::make_shared<PartitionTensor>(
        "test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, nullptr);
    tensor->set_device_id(0);

    // Simulate network sends with delays
    std::vector<std::thread> workers;
    std::atomic<int> successful_stages(0);
    std::atomic<int> failed_stages(0);

    const int num_stages = 10;
    workers.reserve(num_stages);
    for (int i = 0; i < num_stages; ++i) {
      workers.emplace_back([&stager, &tensor, i, &successful_stages, &failed_stages, chunk_size]() {
        std::size_t offset = i * chunk_size;
        auto result = stager.stage(tensor, offset, chunk_size);

        if (result.ok()) {
          // Simulate network send delay
          std::this_thread::sleep_for(std::chrono::milliseconds(50));

          // Verify data before releasing
          void* staged_ptr = *result;
          if (verify_pattern(staged_ptr, chunk_size, 55 + (offset % 256))) {
            successful_stages++;
          }

          // Release buffer
          auto release_status = stager.release_staged_buffer(staged_ptr);
          if (!release_status.ok()) {
            failed_stages++;
          }
        } else {
          failed_stages++;
        }
      });
    }

    // Wait for all workers
    for (auto& w : workers) {
      w.join();
    }

    REQUIRE(successful_stages == num_stages);
    REQUIRE(failed_stages == 0);

    // Check stats
    auto stats = stager.get_stats();
    REQUIRE(stats.buffer_wait_count == 0);

    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
  }

  SECTION("Async staging with out-of-order completion") {
    const std::size_t chunk_size = 2 * 1024 * 1024; // 2MB
    GpuTcpStager stager(chunk_size, 4);

    void* gpu_ptr;
    const std::size_t tensor_size = 8 * 1024 * 1024; // 8MB
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    auto test_data = create_test_pattern(tensor_size, 66);
    REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    auto tensor = std::make_shared<PartitionTensor>(
        "test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, nullptr);
    tensor->set_device_id(0);

    // Start multiple async stages
    std::vector<int> slot_ids;
    for (std::size_t offset = 0; offset < tensor_size; offset += chunk_size) {
      auto result = stager.stage_async(tensor, offset, chunk_size);
      REQUIRE(result.ok());
      slot_ids.push_back(*result);
    }

    // Complete them out of order
    std::vector<std::size_t> completion_order = {2, 0, 3, 1};
    for (std::size_t idx : completion_order) {
      auto result = stager.wait_staging_complete(slot_ids[idx]);
      REQUIRE(result.ok());

      void* staged_ptr = *result;
      std::size_t offset = idx * chunk_size;
      REQUIRE(verify_pattern(staged_ptr, chunk_size, 66 + (offset % 256)));

      REQUIRE(stager.release_staged_buffer(staged_ptr).ok());
    }

    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
  }
}