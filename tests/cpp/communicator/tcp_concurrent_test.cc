// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <atomic>
#include <future>
#include <memory>
#include <thread>
#include <vector>
#include "core/common/cuda_api.h"

#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"

#include "core/communicator/engine/engine.h"
#include "tests/cpp/communicator/test_helpers.h"

using namespace stepcast::communicator;
using namespace stepcast::communicator::test;

TEST_CASE("TCP Mode Concurrent Operations", "[communicator][tcp][gpu][concurrent]") {
  SKIP_IF_NO_CUDA();

  absl::SetGlobalVLogLevel(2);

  SECTION("Concurrent reads from same GPU tensor") {
    // Find available port for source engine
    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    // Create a large GPU tensor
    const std::size_t tensor_size = 64 * 1024 * 1024; // 64MB
    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    // Fill with test pattern
    auto test_data = create_test_pattern(tensor_size, 77);
    REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    // Register the tensor
    REQUIRE(
        source_engine
            ->register_tensor(
                "shared_tensor", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, false)
            .ok());

    // Create multiple target engines
    const int num_readers = 4;
    std::vector<std::shared_ptr<CommunicateEngine>> target_engines;
    std::vector<void*> target_buffers;
    std::vector<int> target_ports;

    for (int i = 0; i < num_readers; ++i) {
      int target_port = find_available_port(source_port + 1 + i * 10);
      REQUIRE(target_port > 0);
      target_ports.push_back(target_port);

      auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
      REQUIRE(engine->init("127.0.0.1", target_port).ok());
      target_engines.push_back(engine);

      void* buffer = std::malloc(tensor_size);
      REQUIRE(buffer != nullptr);
      target_buffers.push_back(buffer);
    }

    // Start concurrent reads
    std::vector<std::future<stepcast::communicator::read_result_t>> futures;
    futures.reserve(num_readers);
    for (int i = 0; i < num_readers; ++i) {
      futures.push_back(
          target_engines[i]->read_tensor(
              "shared_tensor",
              reinterpret_cast<uint64_t>(target_buffers[i]),
              tensor_size,
              COMMUNICATE_ENGINE_DEV_CPU,
              0,
              "127.0.0.1",
              source_port));
    }

    // Wait for all reads to complete
    for (int i = 0; i < num_readers; ++i) {
      auto result = futures[i].get();
      REQUIRE(result.status.ok());

      // Verify data
      REQUIRE(verify_pattern(target_buffers[i], tensor_size, 77));
    }

    // Cleanup
    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
    for (auto* buffer : target_buffers) {
      std::free(buffer);
    }
  }

  SECTION("Engine shutdown during active staging") {
    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    const std::size_t tensor_size = 128 * 1024 * 1024; // 128MB - large to ensure slow transfer
    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    auto test_data = create_test_pattern(tensor_size, 88);
    REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    REQUIRE(
        source_engine
            ->register_tensor(
                "large_tensor", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, false)
            .ok());

    int target_port = find_available_port(source_port + 10);
    REQUIRE(target_port > 0);

    auto target_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(target_engine->init("127.0.0.1", target_port).ok());

    void* target_buffer = std::malloc(tensor_size);
    REQUIRE(target_buffer != nullptr);

    // Start a read
    auto future = target_engine->read_tensor(
        "large_tensor",
        reinterpret_cast<uint64_t>(target_buffer),
        tensor_size,
        COMMUNICATE_ENGINE_DEV_CPU,
        0,
        "127.0.0.1",
        source_port);

    // Immediately destroy the source engine (simulating shutdown)
    source_engine.reset();

    // The read should fail gracefully
    auto result = future.get();
    REQUIRE_FALSE(result.status.ok());

    // Cleanup
    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
    std::free(target_buffer);
  }

  SECTION("Mixed source and target combinations") {
    // Test multiple simultaneous transfers with different source/target combinations
    struct TransferSpec {
      std::string name;
      std::size_t size;
      int source_type;
      int target_type;
      uint8_t pattern_seed;
    };

    std::vector<TransferSpec> transfers = {
        {"gpu_to_gpu", 16 * 1024 * 1024, COMMUNICATE_ENGINE_DEV_GPU, COMMUNICATE_ENGINE_DEV_GPU, 10},
        {"gpu_to_cpu", 8 * 1024 * 1024, COMMUNICATE_ENGINE_DEV_GPU, COMMUNICATE_ENGINE_DEV_CPU, 20},
        {"cpu_to_cpu", 4 * 1024 * 1024, COMMUNICATE_ENGINE_DEV_CPU, COMMUNICATE_ENGINE_DEV_CPU, 30},
    };

    // Create engines
    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);
    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    int target_port = find_available_port(source_port + 10);
    REQUIRE(target_port > 0);
    auto target_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(target_engine->init("127.0.0.1", target_port).ok());

    // Prepare source tensors
    std::vector<void*> source_ptrs;
    for (const auto& spec : transfers) {
      void* ptr = nullptr;
      if (spec.source_type == COMMUNICATE_ENGINE_DEV_GPU) {
        REQUIRE(stepcast::cuda::malloc(&ptr, spec.size).ok());
        auto pattern = create_test_pattern(spec.size, spec.pattern_seed);
        REQUIRE(stepcast::cuda::memcpy(ptr, pattern.data(), spec.size, cudaMemcpyHostToDevice).ok());
      } else {
        ptr = std::malloc(spec.size);
        REQUIRE(ptr != nullptr);
        auto pattern = create_test_pattern(spec.size, spec.pattern_seed);
        std::memcpy(ptr, pattern.data(), spec.size);
      }
      source_ptrs.push_back(ptr);

      REQUIRE(source_engine
                  ->register_tensor(spec.name, reinterpret_cast<uint64_t>(ptr), spec.size, spec.source_type, 0, false)
                  .ok());
    }

    // Prepare target buffers and start transfers
    std::vector<void*> target_ptrs;
    std::vector<std::future<stepcast::communicator::read_result_t>> futures;

    for (const auto& spec : transfers) {
      void* ptr = nullptr;
      if (spec.target_type == COMMUNICATE_ENGINE_DEV_GPU) {
        REQUIRE(stepcast::cuda::malloc(&ptr, spec.size).ok());
      } else {
        ptr = std::malloc(spec.size);
        REQUIRE(ptr != nullptr);
      }
      target_ptrs.push_back(ptr);

      futures.push_back(target_engine->read_tensor(
          spec.name, reinterpret_cast<uint64_t>(ptr), spec.size, spec.target_type, 0, "127.0.0.1", source_port));
    }

    // Wait and verify
    for (std::size_t i = 0; i < transfers.size(); ++i) {
      auto result = futures[i].get();
      REQUIRE(result.status.ok());

      // Verify data
      if (transfers[i].target_type == COMMUNICATE_ENGINE_DEV_GPU) {
        std::vector<uint8_t> verify_data(transfers[i].size);
        REQUIRE(
            stepcast::cuda::memcpy(verify_data.data(), target_ptrs[i], transfers[i].size, cudaMemcpyDeviceToHost).ok());
        REQUIRE(verify_pattern(verify_data.data(), transfers[i].size, transfers[i].pattern_seed));
      } else {
        REQUIRE(verify_pattern(target_ptrs[i], transfers[i].size, transfers[i].pattern_seed));
      }
    }

    // Cleanup
    for (std::size_t i = 0; i < transfers.size(); ++i) {
      if (transfers[i].source_type == COMMUNICATE_ENGINE_DEV_GPU) {
        REQUIRE(stepcast::cuda::free(source_ptrs[i]).ok());
      } else {
        std::free(source_ptrs[i]);
      }

      if (transfers[i].target_type == COMMUNICATE_ENGINE_DEV_GPU) {
        REQUIRE(stepcast::cuda::free(target_ptrs[i]).ok());
      } else {
        std::free(target_ptrs[i]);
      }
    }
  }

  SECTION("Staging buffer exhaustion with many concurrent reads") {
    LOG(INFO) << "Test: Staging buffer exhaustion with many concurrent reads";

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    // Create a moderately sized GPU tensor
    const std::size_t tensor_size = 32 * 1024 * 1024; // 32MB
    // Note: Default staging chunk size is 64MB

    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());
    auto test_data = create_test_pattern(tensor_size, 99);
    REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    REQUIRE(source_engine
                ->register_tensor(
                    "exhaustion_test",
                    reinterpret_cast<uint64_t>(gpu_ptr),
                    tensor_size,
                    COMMUNICATE_ENGINE_DEV_GPU,
                    0,
                    false)
                .ok());

    // Create many concurrent readers (more than staging buffers)
    const int num_readers = 8; // More than the default 2 staging buffers
    std::vector<std::shared_ptr<CommunicateEngine>> target_engines;
    std::vector<void*> target_buffers;

    LOG(INFO) << "Creating " << num_readers << " concurrent readers";

    for (int i = 0; i < num_readers; ++i) {
      int target_port = find_available_port(source_port + 10 + i * 10);
      REQUIRE(target_port > 0);

      auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
      REQUIRE(engine->init("127.0.0.1", target_port).ok());
      target_engines.push_back(engine);

      void* buffer = std::malloc(tensor_size);
      REQUIRE(buffer != nullptr);
      target_buffers.push_back(buffer);
    }

    // Start all reads simultaneously
    std::atomic<int> reads_started(0);
    std::atomic<int> reads_completed(0);
    std::vector<std::future<void>> futures;

    for (int i = 0; i < num_readers; ++i) {
      futures.push_back(std::async(std::launch::async, [&, i]() {
        LOG(INFO) << "Reader " << i << " starting read";
        reads_started++;

        auto result = target_engines[i]
                          ->read_tensor(
                              "exhaustion_test",
                              reinterpret_cast<uint64_t>(target_buffers[i]),
                              tensor_size,
                              COMMUNICATE_ENGINE_DEV_CPU,
                              0,
                              "127.0.0.1",
                              source_port)
                          .get();

        if (result.status.ok()) {
          LOG(INFO) << "Reader " << i << " completed successfully";
          REQUIRE(verify_pattern(target_buffers[i], tensor_size, 99));
        } else {
          LOG(ERROR) << "Reader " << i << " failed: " << result.status;
        }

        reads_completed++;
        LOG(INFO) << "Reads completed: " << reads_completed.load() << "/" << num_readers;
      }));
    }

    // Wait for all with timeout
    auto start_time = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(30);

    for (size_t i = 0; i < futures.size(); ++i) {
      auto remaining_time = timeout - (std::chrono::steady_clock::now() - start_time);
      LOG(INFO) << "Waiting for reader " << i << " with remaining timeout: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(remaining_time).count() << "ms";

      auto status = futures[i].wait_for(remaining_time);
      if (status == std::future_status::timeout) {
        LOG(ERROR)
            << "Reader " << i << " timed out after "
            << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count()
            << " seconds";
        LOG(ERROR) << "Total reads started: " << reads_started.load();
        LOG(ERROR) << "Total reads completed: " << reads_completed.load();
      }
      REQUIRE(status != std::future_status::timeout);
    }

    LOG(INFO) << "All reads completed. Started: " << reads_started.load() << ", Completed: " << reads_completed.load();

    // Cleanup
    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
    for (auto* buffer : target_buffers) {
      std::free(buffer);
    }
  }

  SECTION("Partial read with offset and concurrent access") {
    LOG(INFO) << "Test: Partial read with offset and concurrent access";

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    const std::size_t tensor_size = 128 * 1024 * 1024; // 128MB
    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    // Fill with different patterns in different regions
    std::vector<uint8_t> test_data(tensor_size);
    for (size_t i = 0; i < tensor_size; i += 1024 * 1024) {
      std::fill(test_data.begin() + i, test_data.begin() + i + 1024 * 1024, static_cast<uint8_t>(i / (1024 * 1024)));
    }
    REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    REQUIRE(
        source_engine
            ->register_tensor(
                "offset_test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, false)
            .ok());

    // Multiple readers reading different offsets concurrently
    struct ReadSpec {
      size_t offset;
      size_t size;
      uint8_t expected_pattern;
    };

    std::vector<ReadSpec> read_specs = {
        {0, 4 * 1024 * 1024, 0},
        {10 * 1024 * 1024, 8 * 1024 * 1024, 10},
        {50 * 1024 * 1024, 16 * 1024 * 1024, 50},
        {100 * 1024 * 1024, 4 * 1024 * 1024, 100},
    };

    std::vector<std::future<void>> futures;
    std::vector<std::shared_ptr<CommunicateEngine>> engines;
    std::vector<void*> buffers;

    for (size_t i = 0; i < read_specs.size(); ++i) {
      int target_port = find_available_port(source_port + 10 + i * 10);
      REQUIRE(target_port > 0);

      auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
      REQUIRE(engine->init("127.0.0.1", target_port).ok());
      engines.push_back(engine);

      void* buffer = std::malloc(read_specs[i].size);
      REQUIRE(buffer != nullptr);
      buffers.push_back(buffer);

      futures.push_back(std::async(std::launch::async, [&, i]() {
        LOG(INFO) << "Starting partial read " << i << ": offset=" << read_specs[i].offset
                  << ", size=" << read_specs[i].size;

        auto result = engines[i]
                          ->read_tensor(
                              "offset_test",
                              reinterpret_cast<uint64_t>(buffers[i]),
                              read_specs[i].size,
                              COMMUNICATE_ENGINE_DEV_CPU,
                              0,
                              "127.0.0.1",
                              source_port,
                              read_specs[i].offset)
                          .get();

        REQUIRE(result.status.ok());

        // Verify the pattern
        uint8_t* data = static_cast<uint8_t*>(buffers[i]);
        bool pattern_correct = true;
        for (size_t j = 0; j < read_specs[i].size; j += 1024 * 1024) {
          uint8_t expected = static_cast<uint8_t>((read_specs[i].offset + j) / (1024 * 1024));
          if (data[j] != expected) {
            LOG(ERROR) << "Pattern mismatch at offset " << j << ": expected " << (int)expected << ", got "
                       << (int)data[j];
            pattern_correct = false;
            break;
          }
        }
        REQUIRE(pattern_correct);

        LOG(INFO) << "Partial read " << i << " completed successfully";
      }));
    }

    // Wait for all
    for (auto& future : futures) {
      REQUIRE(future.wait_for(std::chrono::seconds(30)) != std::future_status::timeout);
    }

    // Cleanup
    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
    for (auto* buffer : buffers) {
      std::free(buffer);
    }
  }

  SECTION("Rapid connection cycling with GPU transfers") {
    LOG(INFO) << "Test: Rapid connection cycling with GPU transfers";

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    const std::size_t tensor_size = 16 * 1024 * 1024; // 16MB
    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    auto test_data = create_test_pattern(tensor_size, 111);
    REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    REQUIRE(
        source_engine
            ->register_tensor(
                "cycling_test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, false)
            .ok());

    // Rapidly create connections, transfer, and close
    const int num_cycles = 10;
    std::atomic<int> successful_cycles(0);
    std::vector<std::future<void>> futures;

    futures.reserve(num_cycles);
    for (int i = 0; i < num_cycles; ++i) {
      futures.push_back(std::async(std::launch::async, [&, i]() {
        LOG(INFO) << "Cycle " << i << " starting";

        // Find available port for this cycle
        int target_port = find_available_port(source_port + 10 + i * 10);
        REQUIRE(target_port > 0);

        // Create new engine for each cycle
        auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
        REQUIRE(engine->init("127.0.0.1", target_port).ok());

        void* buffer = std::malloc(tensor_size);
        REQUIRE(buffer != nullptr);

        // Perform transfer
        auto result = engine
                          ->read_tensor(
                              "cycling_test",
                              reinterpret_cast<uint64_t>(buffer),
                              tensor_size,
                              COMMUNICATE_ENGINE_DEV_CPU,
                              0,
                              "127.0.0.1",
                              source_port)
                          .get();

        if (result.status.ok()) {
          REQUIRE(verify_pattern(buffer, tensor_size, 111));
          successful_cycles++;
          LOG(INFO) << "Cycle " << i << " completed successfully";
        } else {
          LOG(ERROR) << "Cycle " << i << " failed: " << result.status;
        }

        std::free(buffer);

        // Explicitly close connection
        CHECK_OK(engine->close_connection("127.0.0.1", source_port));

        // Small delay to stress connection management
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }));
    }

    // Wait for all cycles
    for (auto& future : futures) {
      REQUIRE(future.wait_for(std::chrono::seconds(60)) != std::future_status::timeout);
    }

    LOG(INFO) << "Connection cycling test completed. Successful cycles: " << successful_cycles.load() << "/"
              << num_cycles;
    REQUIRE(successful_cycles.load() == num_cycles);

    // Cleanup
    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
  }

  SECTION("Concurrent GPU-to-GPU transfers with staging buffer contention") {
    LOG(INFO) << "Test: Concurrent GPU-to-GPU transfers with staging buffer contention";

    // Check if we have at least 2 GPUs
    int device_count = 0;
    REQUIRE(stepcast::cuda::get_device_count(&device_count).ok());
    if (device_count < 2) {
      LOG(INFO) << "Skipping multi-GPU test - only " << device_count << " GPU(s) available";
      return;
    }

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    // Create tensors on different GPUs
    const std::size_t tensor_size = 32 * 1024 * 1024; // 32MB
    std::vector<void*> gpu_ptrs;

    for (int gpu_id = 0; gpu_id < std::min(2, device_count); ++gpu_id) {
      REQUIRE(stepcast::cuda::set_device(gpu_id).ok());
      void* gpu_ptr;
      REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

      auto test_data = create_test_pattern(tensor_size, 100 + gpu_id);
      REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

      gpu_ptrs.push_back(gpu_ptr);

      std::string tensor_name = "gpu_tensor_" + std::to_string(gpu_id);
      REQUIRE(source_engine
                  ->register_tensor(
                      tensor_name,
                      reinterpret_cast<uint64_t>(gpu_ptr),
                      tensor_size,
                      COMMUNICATE_ENGINE_DEV_GPU,
                      gpu_id,
                      false)
                  .ok());

      LOG(INFO) << "Registered tensor on GPU " << gpu_id;
    }

    // Multiple readers for GPU-to-GPU transfers
    std::vector<std::future<void>> futures;
    const int transfers_per_gpu = 3;

    futures.reserve(transfers_per_gpu * 2);
    for (int i = 0; i < transfers_per_gpu * 2; ++i) {
      futures.push_back(std::async(std::launch::async, [&, i]() {
        int src_gpu = i % 2;
        int dst_gpu = 1 - src_gpu;

        LOG(INFO) << "Transfer " << i << ": GPU " << src_gpu << " -> GPU " << dst_gpu;

        int target_port = find_available_port(source_port + 10 + i * 10);
        REQUIRE(target_port > 0);

        auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
        REQUIRE(engine->init("127.0.0.1", target_port).ok());

        REQUIRE(stepcast::cuda::set_device(dst_gpu).ok());
        void* dst_ptr;
        REQUIRE(stepcast::cuda::malloc(&dst_ptr, tensor_size).ok());

        std::string tensor_name = "gpu_tensor_" + std::to_string(src_gpu);
        auto result = engine
                          ->read_tensor(
                              tensor_name,
                              reinterpret_cast<uint64_t>(dst_ptr),
                              tensor_size,
                              COMMUNICATE_ENGINE_DEV_GPU,
                              dst_gpu,
                              "127.0.0.1",
                              source_port)
                          .get();

        REQUIRE(result.status.ok());

        // Verify data
        std::vector<uint8_t> verify_data(tensor_size);
        REQUIRE(stepcast::cuda::memcpy(verify_data.data(), dst_ptr, tensor_size, cudaMemcpyDeviceToHost).ok());
        REQUIRE(verify_pattern(verify_data.data(), tensor_size, 100 + src_gpu));

        LOG(INFO) << "Transfer " << i << " completed successfully";

        REQUIRE(stepcast::cuda::free(dst_ptr).ok());
      }));
    }

    // Wait for all transfers
    for (auto& future : futures) {
      REQUIRE(future.wait_for(std::chrono::seconds(60)) != std::future_status::timeout);
    }

    // Cleanup
    for (int gpu_id = 0; gpu_id < gpu_ptrs.size(); ++gpu_id) {
      REQUIRE(stepcast::cuda::set_device(gpu_id).ok());
      REQUIRE(stepcast::cuda::free(gpu_ptrs[gpu_id]).ok());
    }
  }

  SECTION("Memory pressure and error recovery") {
    LOG(INFO) << "Test: Memory pressure and error recovery";

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    // Create multiple large tensors to stress memory
    const std::size_t tensor_size = 64 * 1024 * 1024; // 64MB each
    const int num_tensors = 3;
    std::vector<void*> gpu_ptrs;

    for (int i = 0; i < num_tensors; ++i) {
      void* gpu_ptr;
      auto alloc_status = stepcast::cuda::malloc(&gpu_ptr, tensor_size);
      if (!alloc_status.ok()) {
        LOG(WARNING) << "Failed to allocate tensor " << i << " - testing with " << i << " tensors";
        break;
      }

      auto test_data = create_test_pattern(tensor_size, 200 + i);
      REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

      gpu_ptrs.push_back(gpu_ptr);

      std::string tensor_name = "pressure_tensor_" + std::to_string(i);
      REQUIRE(
          source_engine
              ->register_tensor(
                  tensor_name, reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, false)
              .ok());
    }

    LOG(INFO) << "Allocated " << gpu_ptrs.size() << " tensors";

    // Try to read all tensors concurrently
    std::vector<std::future<void>> futures;
    std::atomic<int> successful_reads(0);
    std::atomic<int> failed_reads(0);

    for (size_t i = 0; i < gpu_ptrs.size(); ++i) {
      futures.push_back(std::async(std::launch::async, [&, i]() {
        int target_port = find_available_port(source_port + 10 + i * 10);
        REQUIRE(target_port > 0);

        auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
        REQUIRE(engine->init("127.0.0.1", target_port).ok());

        void* cpu_buffer = std::malloc(tensor_size);
        if (!cpu_buffer) {
          LOG(ERROR) << "Failed to allocate CPU buffer for tensor " << i;
          failed_reads++;
          return;
        }

        std::string tensor_name = "pressure_tensor_" + std::to_string(i);
        auto result = engine
                          ->read_tensor(
                              tensor_name,
                              reinterpret_cast<uint64_t>(cpu_buffer),
                              tensor_size,
                              COMMUNICATE_ENGINE_DEV_CPU,
                              0,
                              "127.0.0.1",
                              source_port)
                          .get();

        if (result.status.ok()) {
          if (verify_pattern(cpu_buffer, tensor_size, 200 + i)) {
            successful_reads++;
            LOG(INFO) << "Read " << i << " completed successfully";
          } else {
            LOG(ERROR) << "Read " << i << " data verification failed";
            failed_reads++;
          }
        } else {
          LOG(ERROR) << "Read " << i << " failed: " << result.status;
          failed_reads++;
        }

        std::free(cpu_buffer);
      }));
    }

    // Wait for all
    for (auto& future : futures) {
      REQUIRE(future.wait_for(std::chrono::seconds(120)) != std::future_status::timeout);
    }

    LOG(INFO) << "Memory pressure test completed. Successful: " << successful_reads.load()
              << ", Failed: " << failed_reads.load();

    // At least some reads should succeed
    REQUIRE(successful_reads.load() > 0);

    // Cleanup
    for (auto* ptr : gpu_ptrs) {
      REQUIRE(stepcast::cuda::free(ptr).ok());
    }
  }

  SECTION("Deadlock prevention - all staging buffers occupied") {
    LOG(INFO) << "Test: Deadlock prevention with all staging buffers occupied";

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    // Create tensor that requires exactly one staging buffer
    const std::size_t chunk_size = 64 * 1024 * 1024; // Default staging chunk size
    const std::size_t tensor_size = chunk_size; // Exactly one chunk

    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());
    auto test_data = create_test_pattern(tensor_size, 123);
    REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    REQUIRE(
        source_engine
            ->register_tensor(
                "deadlock_test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, false)
            .ok());

    // Create readers that will hold staging buffers
    const int num_holders = 2; // Equal to default staging buffers
    std::vector<std::shared_ptr<CommunicateEngine>> holder_engines;
    std::vector<void*> holder_buffers;
    std::vector<std::future<stepcast::communicator::read_result_t>> holder_futures;

    // Start reads that will occupy all staging buffers
    for (int i = 0; i < num_holders; ++i) {
      int target_port = find_available_port(source_port + 10 + i * 10);
      REQUIRE(target_port > 0);

      auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
      REQUIRE(engine->init("127.0.0.1", target_port).ok());
      holder_engines.push_back(engine);

      void* buffer = std::malloc(tensor_size);
      REQUIRE(buffer != nullptr);
      holder_buffers.push_back(buffer);

      LOG(INFO) << "Starting holder read " << i;
      holder_futures.push_back(engine->read_tensor(
          "deadlock_test",
          reinterpret_cast<uint64_t>(buffer),
          tensor_size,
          COMMUNICATE_ENGINE_DEV_CPU,
          0,
          "127.0.0.1",
          source_port));
    }

    // Give holders time to occupy staging buffers
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Now try additional reads that will have to wait
    const int num_waiters = 3;
    std::vector<std::shared_ptr<CommunicateEngine>> waiter_engines;
    std::vector<void*> waiter_buffers;
    std::vector<std::future<stepcast::communicator::read_result_t>> waiter_futures;
    std::atomic<int> waiters_started(0);

    // Pre-allocate futures vector to avoid race conditions
    waiter_futures.resize(num_waiters);

    for (int i = 0; i < num_waiters; ++i) {
      int target_port = find_available_port(source_port + 100 + i * 10);
      REQUIRE(target_port > 0);

      auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
      REQUIRE(engine->init("127.0.0.1", target_port).ok());
      waiter_engines.push_back(engine);

      void* buffer = std::malloc(tensor_size);
      REQUIRE(buffer != nullptr);
      waiter_buffers.push_back(buffer);
    }

    // Start waiter reads in separate threads
    for (int i = 0; i < num_waiters; ++i) {
      std::thread([&, i]() {
        LOG(INFO) << "Waiter " << i << " starting read (will wait for buffer)";
        waiters_started++;
        waiter_futures[i] = waiter_engines[i]->read_tensor(
            "deadlock_test",
            reinterpret_cast<uint64_t>(waiter_buffers[i]),
            tensor_size,
            COMMUNICATE_ENGINE_DEV_CPU,
            0,
            "127.0.0.1",
            source_port);
      }).detach();
    }

    // Wait for all waiters to start
    while (waiters_started.load() < num_waiters) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    LOG(INFO) << "All waiters started, now completing holder reads";

    // Complete holder reads to free buffers
    for (int i = 0; i < num_holders; ++i) {
      auto result = holder_futures[i].get();
      REQUIRE(result.status.ok());
      REQUIRE(verify_pattern(holder_buffers[i], tensor_size, 123));
      LOG(INFO) << "Holder " << i << " completed, buffer should be freed";
    }

    // Now waiter reads should complete
    LOG(INFO) << "Waiting for waiter reads to complete";
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);

    for (int i = 0; i < num_waiters; ++i) {
      auto remaining = deadline - std::chrono::steady_clock::now();
      if (i < waiter_futures.size()) {
        REQUIRE(waiter_futures[i].wait_for(remaining) != std::future_status::timeout);
        auto result = waiter_futures[i].get();
        REQUIRE(result.status.ok());
        REQUIRE(verify_pattern(waiter_buffers[i], tensor_size, 123));
        LOG(INFO) << "Waiter " << i << " completed successfully";
      }
    }

    // Cleanup
    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
    for (auto* buffer : holder_buffers)
      std::free(buffer);
    for (auto* buffer : waiter_buffers)
      std::free(buffer);
  }

  SECTION("Chunk boundary alignment stress test") {
    LOG(INFO) << "Test: Chunk boundary alignment stress test";

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    // Test various sizes around chunk boundaries
    const std::size_t chunk_size = 64 * 1024 * 1024; // Default staging chunk size
    std::vector<std::size_t> test_sizes = {
        chunk_size - 1, // Just under one chunk
        chunk_size, // Exactly one chunk
        chunk_size + 1, // Just over one chunk
        2 * chunk_size - 1, // Just under two chunks
        2 * chunk_size, // Exactly two chunks
        2 * chunk_size + 1, // Just over two chunks
    };

    std::vector<void*> gpu_ptrs;
    std::vector<std::string> tensor_names;

    // Create tensors with different sizes
    for (size_t i = 0; i < test_sizes.size(); ++i) {
      void* gpu_ptr;
      REQUIRE(stepcast::cuda::malloc(&gpu_ptr, test_sizes[i]).ok());

      auto test_data = create_test_pattern(test_sizes[i], 150 + i);
      REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), test_sizes[i], cudaMemcpyHostToDevice).ok());

      gpu_ptrs.push_back(gpu_ptr);

      std::string tensor_name = "boundary_tensor_" + std::to_string(i);
      tensor_names.push_back(tensor_name);

      REQUIRE(
          source_engine
              ->register_tensor(
                  tensor_name, reinterpret_cast<uint64_t>(gpu_ptr), test_sizes[i], COMMUNICATE_ENGINE_DEV_GPU, 0, false)
              .ok());

      LOG(INFO) << "Registered tensor " << i << " with size " << test_sizes[i] << " ("
                << (double)test_sizes[i] / chunk_size << " chunks)";
    }

    // Read all tensors concurrently
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < test_sizes.size(); ++i) {
      futures.push_back(std::async(std::launch::async, [&, i]() {
        int target_port = find_available_port(source_port + 10 + i * 10);
        REQUIRE(target_port > 0);

        auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
        REQUIRE(engine->init("127.0.0.1", target_port).ok());

        void* cpu_buffer = std::malloc(test_sizes[i]);
        REQUIRE(cpu_buffer != nullptr);

        LOG(INFO) << "Reading tensor " << i << " (size " << test_sizes[i] << ")";

        auto result = engine
                          ->read_tensor(
                              tensor_names[i],
                              reinterpret_cast<uint64_t>(cpu_buffer),
                              test_sizes[i],
                              COMMUNICATE_ENGINE_DEV_CPU,
                              0,
                              "127.0.0.1",
                              source_port)
                          .get();

        REQUIRE(result.status.ok());
        REQUIRE(verify_pattern(cpu_buffer, test_sizes[i], 150 + i));

        LOG(INFO) << "Tensor " << i << " read completed successfully";

        std::free(cpu_buffer);
      }));
    }

    // Wait for all
    for (auto& future : futures) {
      REQUIRE(future.wait_for(std::chrono::seconds(30)) != std::future_status::timeout);
    }

    // Cleanup
    for (auto* ptr : gpu_ptrs) {
      REQUIRE(stepcast::cuda::free(ptr).ok());
    }
  }

  SECTION("Tensor registration/unregistration during active transfers") {
    LOG(INFO) << "Test: Tensor registration/unregistration during active transfers";

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    const std::size_t tensor_size = 32 * 1024 * 1024; // 32MB
    const int num_tensors = 5;

    // Create initial tensors
    std::vector<void*> gpu_ptrs;
    for (int i = 0; i < num_tensors; ++i) {
      void* gpu_ptr;
      REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

      auto test_data = create_test_pattern(tensor_size, 170 + i);
      REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

      gpu_ptrs.push_back(gpu_ptr);

      std::string tensor_name = "dynamic_tensor_" + std::to_string(i);
      REQUIRE(
          source_engine
              ->register_tensor(
                  tensor_name, reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, false)
              .ok());
    }

    std::atomic<bool> stop_flag(false);
    std::atomic<int> successful_reads(0);
    std::atomic<int> failed_reads(0);

    // Start reader threads
    std::vector<std::future<void>> reader_futures;
    for (int reader_id = 0; reader_id < 3; ++reader_id) {
      reader_futures.push_back(std::async(std::launch::async, [&, reader_id]() {
        int target_port = find_available_port(source_port + 10 + reader_id * 10);
        REQUIRE(target_port > 0);

        auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
        REQUIRE(engine->init("127.0.0.1", target_port).ok());

        while (!stop_flag.load()) {
          int tensor_id = rand() % num_tensors;
          std::string tensor_name = "dynamic_tensor_" + std::to_string(tensor_id);

          void* buffer = std::malloc(tensor_size);
          if (!buffer)
            continue;

          auto result = engine
                            ->read_tensor(
                                tensor_name,
                                reinterpret_cast<uint64_t>(buffer),
                                tensor_size,
                                COMMUNICATE_ENGINE_DEV_CPU,
                                0,
                                "127.0.0.1",
                                source_port)
                            .get();

          if (result.status.ok()) {
            if (verify_pattern(buffer, tensor_size, 170 + tensor_id)) {
              successful_reads++;
            }
          } else {
            // Expected if tensor was unregistered
            failed_reads++;
          }

          std::free(buffer);
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        LOG(INFO) << "Reader " << reader_id << " stopping. Successful: " << successful_reads.load()
                  << ", Failed: " << failed_reads.load();
      }));
    }

    // Let readers run for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Unregister and re-register tensors
    LOG(INFO) << "Starting dynamic registration/unregistration";
    for (int cycle = 0; cycle < 3; ++cycle) {
      for (int i = 0; i < num_tensors; ++i) {
        std::string tensor_name = "dynamic_tensor_" + std::to_string(i);

        // Unregister
        CHECK_OK(source_engine->unregister_tensor(tensor_name));

        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Re-register with new data
        auto test_data = create_test_pattern(tensor_size, 170 + i + cycle * 10);
        REQUIRE(stepcast::cuda::memcpy(gpu_ptrs[i], test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

        REQUIRE(source_engine
                    ->register_tensor(
                        tensor_name,
                        reinterpret_cast<uint64_t>(gpu_ptrs[i]),
                        tensor_size,
                        COMMUNICATE_ENGINE_DEV_GPU,
                        0,
                        false)
                    .ok());
      }

      LOG(INFO) << "Completed registration cycle " << cycle;
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Stop readers
    stop_flag.store(true);
    for (auto& future : reader_futures) {
      REQUIRE(future.wait_for(std::chrono::seconds(5)) != std::future_status::timeout);
    }

    LOG(INFO) << "Test completed. Total successful reads: " << successful_reads.load()
              << ", Failed reads: " << failed_reads.load();

    // Should have some successful reads
    REQUIRE(successful_reads.load() > 0);

    // Cleanup
    for (auto* ptr : gpu_ptrs) {
      REQUIRE(stepcast::cuda::free(ptr).ok());
    }
  }

  SECTION("ScopedStagedBuffer RAII and exception safety") {
    LOG(INFO) << "Test: ScopedStagedBuffer RAII and exception safety";

    // This test verifies that staged buffers are properly released even when
    // exceptions occur or when ScopedStagedBuffer goes out of scope early

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    const std::size_t tensor_size = 32 * 1024 * 1024; // 32MB
    void* gpu_ptr;
    REQUIRE(stepcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    auto test_data = create_test_pattern(tensor_size, 180);
    REQUIRE(stepcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    REQUIRE(source_engine
                ->register_tensor(
                    "raii_test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, false)
                .ok());

    // Test multiple scenarios
    const int num_iterations = 10;
    std::atomic<int> buffers_acquired(0);
    std::atomic<int> buffers_released(0);

    std::vector<std::future<void>> futures;

    for (int i = 0; i < num_iterations; ++i) {
      futures.push_back(std::async(std::launch::async, [&, i]() {
        int target_port = find_available_port(source_port + 10 + i * 10);
        REQUIRE(target_port > 0);

        auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
        REQUIRE(engine->init("127.0.0.1", target_port).ok());

        void* cpu_buffer = std::malloc(tensor_size);
        REQUIRE(cpu_buffer != nullptr);

        try {
          // Nested scope to test RAII
          {
            LOG(INFO) << "Iteration " << i << " acquiring staged buffer";
            buffers_acquired++;

            // Read tensor (which will use staging internally)
            auto result = engine
                              ->read_tensor(
                                  "raii_test",
                                  reinterpret_cast<uint64_t>(cpu_buffer),
                                  tensor_size,
                                  COMMUNICATE_ENGINE_DEV_CPU,
                                  0,
                                  "127.0.0.1",
                                  source_port)
                              .get();

            if (i % 3 == 0) {
              // Simulate error condition
              LOG(INFO) << "Iteration " << i << " simulating error";
              throw std::runtime_error("Simulated error");
            }

            REQUIRE(result.status.ok());
            REQUIRE(verify_pattern(cpu_buffer, tensor_size, 180));

            LOG(INFO) << "Iteration " << i << " completed normally";
          }
          // ScopedStagedBuffer should be released here
          buffers_released++;

        } catch (const std::exception& e) {
          LOG(INFO) << "Iteration " << i << " caught exception: " << e.what();
          // Buffer should still be released due to RAII
          buffers_released++;
        }

        std::free(cpu_buffer);
      }));
    }

    // Wait for all iterations
    for (auto& future : futures) {
      REQUIRE(future.wait_for(std::chrono::seconds(30)) != std::future_status::timeout);
    }

    LOG(INFO) << "RAII test completed. Buffers acquired: " << buffers_acquired.load()
              << ", Buffers released: " << buffers_released.load();

    // All acquired buffers should be released
    REQUIRE(buffers_acquired.load() == buffers_released.load());

    // Verify we can still use staging buffers (they weren't leaked)
    {
      int final_port = find_available_port(source_port + 200);
      REQUIRE(final_port > 0);

      auto engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
      REQUIRE(engine->init("127.0.0.1", final_port).ok());

      void* final_buffer = std::malloc(tensor_size);
      REQUIRE(final_buffer != nullptr);

      auto result = engine
                        ->read_tensor(
                            "raii_test",
                            reinterpret_cast<uint64_t>(final_buffer),
                            tensor_size,
                            COMMUNICATE_ENGINE_DEV_CPU,
                            0,
                            "127.0.0.1",
                            source_port)
                        .get();

      REQUIRE(result.status.ok());
      REQUIRE(verify_pattern(final_buffer, tensor_size, 180));

      std::free(final_buffer);
    }

    // Cleanup
    REQUIRE(stepcast::cuda::free(gpu_ptr).ok());
  }
}
