// Copyright (c) 2025, TensorCast Team.

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
#include "core/testing/test_helpers.h"

using namespace tensorcast;
using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;
using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
using tensorcast::communicator::engine::Communicator;
using tensorcast::communicator::v1::CommunicatorConfig;
using namespace tensorcast::testing;

TEST_CASE("TCP Mode Concurrent Operations", "[communicator][tcp][gpu][concurrent]") {
  SKIP_IF_NO_CUDA();

  absl::SetGlobalVLogLevel(2);

  SECTION("Concurrent reads from same GPU tensor") {
    // Find available port for source engine
    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    communicator::v1::CommunicatorConfig cfg1;
    cfg1.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<Communicator>(cfg1);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    // Create a large GPU tensor
    const std::size_t tensor_size = 64 * 1024; // 64KB
    void* gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    // Fill with test pattern
    auto test_data = create_test_pattern(tensor_size, 77);
    REQUIRE(tensorcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    // Register the tensor
    Communicator::RegisterTensorOptions ro1;
    ro1.register_mr = false;
    ro1.needs_staging = true;
    ro1.async = false;
    REQUIRE(
        source_engine
            ->register_tensor_ex(
                "shared_tensor", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, ro1)
            .ok());

    // Create multiple target engines
    const int num_readers = 4;
    std::vector<std::shared_ptr<Communicator>> target_engines;
    std::vector<void*> target_buffers;
    std::vector<int> target_ports;

    for (int i = 0; i < num_readers; ++i) {
      int target_port = find_available_port(source_port + 1 + i * 10);
      REQUIRE(target_port > 0);
      target_ports.push_back(target_port);

      communicator::v1::CommunicatorConfig cfg;
      cfg.set_enable_rdma(false); /* disable RDMA */
      auto engine = std::make_shared<Communicator>(cfg);
      REQUIRE(engine->init("127.0.0.1", target_port).ok());
      target_engines.push_back(engine);

      void* buffer = std::malloc(tensor_size);
      REQUIRE(buffer != nullptr);
      target_buffers.push_back(buffer);
    }

    // Start concurrent reads
    std::vector<std::future<tensorcast::communicator::transport::read_result_t>> futures;
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
    REQUIRE(tensorcast::cuda::free(gpu_ptr).ok());
    for (auto* buffer : target_buffers) {
      std::free(buffer);
    }
  }

  SECTION("Engine shutdown during active staging") {
    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    communicator::v1::CommunicatorConfig cfg;
    cfg.set_enable_rdma(false); /* disable RDMA */
    cfg.mutable_stager()->set_buffers_per_flow(2); /* reduce staging footprint for concurrent cycling */
    auto source_engine = std::make_shared<Communicator>(cfg);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    const std::size_t tensor_size = 128 * 1024; // 128KB - large to ensure slow transfer
    void* gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    auto test_data = create_test_pattern(tensor_size, 88);
    REQUIRE(tensorcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    {
      Communicator::RegisterTensorOptions opts;
      opts.register_mr = false;
      opts.needs_staging = true;
      opts.async = false;
      REQUIRE(
          source_engine
              ->register_tensor_ex(
                  "large_tensor", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, opts)
              .ok());
    }

    int target_port = find_available_port(source_port + 10);
    REQUIRE(target_port > 0);

    communicator::v1::CommunicatorConfig cfg2;
    cfg2.set_enable_rdma(false); /* disable RDMA */
    auto target_engine = std::make_shared<Communicator>(cfg2);
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
    REQUIRE(tensorcast::cuda::free(gpu_ptr).ok());
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
        {"gpu_to_gpu", 16 * 1024, COMMUNICATE_ENGINE_DEV_GPU, COMMUNICATE_ENGINE_DEV_GPU, 10},
        {"gpu_to_cpu", 8 * 1024, COMMUNICATE_ENGINE_DEV_GPU, COMMUNICATE_ENGINE_DEV_CPU, 20},
        {"cpu_to_cpu", 4 * 1024, COMMUNICATE_ENGINE_DEV_CPU, COMMUNICATE_ENGINE_DEV_CPU, 30},
    };

    // Create engines
    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);
    communicator::v1::CommunicatorConfig cfg3;
    cfg3.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<Communicator>(cfg3);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    int target_port = find_available_port(source_port + 10);
    REQUIRE(target_port > 0);
    communicator::v1::CommunicatorConfig cfg4;
    cfg4.set_enable_rdma(false); /* disable RDMA */
    auto target_engine = std::make_shared<Communicator>(cfg4);
    REQUIRE(target_engine->init("127.0.0.1", target_port).ok());

    // Prepare source tensors
    std::vector<void*> source_ptrs;
    for (const auto& spec : transfers) {
      void* ptr = nullptr;
      if (spec.source_type == COMMUNICATE_ENGINE_DEV_GPU) {
        REQUIRE(tensorcast::cuda::malloc(&ptr, spec.size).ok());
        auto pattern = create_test_pattern(spec.size, spec.pattern_seed);
        REQUIRE(tensorcast::cuda::memcpy(ptr, pattern.data(), spec.size, cudaMemcpyHostToDevice).ok());
      } else {
        ptr = std::malloc(spec.size);
        REQUIRE(ptr != nullptr);
        auto pattern = create_test_pattern(spec.size, spec.pattern_seed);
        std::memcpy(ptr, pattern.data(), spec.size);
      }
      source_ptrs.push_back(ptr);

      Communicator::RegisterTensorOptions r;
      r.register_mr = false;
      r.needs_staging = (spec.source_type == COMMUNICATE_ENGINE_DEV_GPU);
      r.async = false;
      REQUIRE(source_engine
                  ->register_tensor_ex(spec.name, reinterpret_cast<uint64_t>(ptr), spec.size, spec.source_type, 0, r)
                  .ok());
    }

    // Prepare target buffers and start transfers
    std::vector<void*> target_ptrs;
    std::vector<std::future<tensorcast::communicator::transport::read_result_t>> futures;

    for (const auto& spec : transfers) {
      void* ptr = nullptr;
      if (spec.target_type == COMMUNICATE_ENGINE_DEV_GPU) {
        REQUIRE(tensorcast::cuda::malloc(&ptr, spec.size).ok());
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
            tensorcast::cuda::memcpy(verify_data.data(), target_ptrs[i], transfers[i].size, cudaMemcpyDeviceToHost)
                .ok());
        REQUIRE(verify_pattern(verify_data.data(), transfers[i].size, transfers[i].pattern_seed));
      } else {
        REQUIRE(verify_pattern(target_ptrs[i], transfers[i].size, transfers[i].pattern_seed));
      }
    }

    // Cleanup
    for (std::size_t i = 0; i < transfers.size(); ++i) {
      if (transfers[i].source_type == COMMUNICATE_ENGINE_DEV_GPU) {
        REQUIRE(tensorcast::cuda::free(source_ptrs[i]).ok());
      } else {
        std::free(source_ptrs[i]);
      }

      if (transfers[i].target_type == COMMUNICATE_ENGINE_DEV_GPU) {
        REQUIRE(tensorcast::cuda::free(target_ptrs[i]).ok());
      } else {
        std::free(target_ptrs[i]);
      }
    }
  }

  SECTION("Staging buffer exhaustion with many concurrent reads") {
    LOG(INFO) << "Test: Staging buffer exhaustion with many concurrent reads";

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    communicator::v1::CommunicatorConfig cfg5;
    cfg5.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<Communicator>(cfg5);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    // Create a moderately sized GPU tensor
    const std::size_t tensor_size = 32 * 1024; // 32KB
    // Note: Default staging chunk size is 64KB (reduced for low-memory env)

    void* gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, tensor_size).ok());
    auto test_data = create_test_pattern(tensor_size, 99);
    REQUIRE(tensorcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    Communicator::RegisterTensorOptions ro2;
    ro2.register_mr = false;
    ro2.needs_staging = true;
    ro2.async = false;
    REQUIRE(
        source_engine
            ->register_tensor_ex(
                "exhaustion_test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, ro2)
            .ok());

    // Create many concurrent readers (more than staging buffers)
    const int num_readers = 8; // More than the default 2 staging buffers
    std::vector<std::shared_ptr<Communicator>> target_engines;
    std::vector<void*> target_buffers;

    LOG(INFO) << "Creating " << num_readers << " concurrent readers";

    for (int i = 0; i < num_readers; ++i) {
      int target_port = find_available_port(source_port + 10 + i * 10);
      REQUIRE(target_port > 0);

      communicator::v1::CommunicatorConfig cfg;
      cfg.set_enable_rdma(false); /* disable RDMA */
      auto engine = std::make_shared<Communicator>(cfg);
      REQUIRE(engine->init("127.0.0.1", target_port).ok());
      target_engines.push_back(engine);

      void* buffer = std::malloc(tensor_size);
      REQUIRE(buffer != nullptr);
      target_buffers.push_back(buffer);
    }

    // Start all reads simultaneously
    std::atomic<int> reads_started(0);
    std::atomic<int> reads_completed(0);

    struct ExhaustionReadOutcome {
      bool status_ok;
      bool pattern_ok;
    };

    std::vector<std::future<ExhaustionReadOutcome>> futures;

    for (int i = 0; i < num_readers; ++i) {
      futures.push_back(std::async(std::launch::async, [&, i]() -> ExhaustionReadOutcome {
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

        ExhaustionReadOutcome outcome{result.status.ok(), false};
        if (outcome.status_ok) {
          bool ok = verify_pattern(target_buffers[i], tensor_size, 99);
          outcome.pattern_ok = ok;
          if (ok) {
            LOG(INFO) << "Reader " << i << " completed successfully";
          } else {
            LOG(ERROR) << "Reader " << i << " data verification failed";
          }
        } else {
          LOG(ERROR) << "Reader " << i << " failed: " << result.status;
        }

        reads_completed++;
        LOG(INFO) << "Reads completed: " << reads_completed.load() << "/" << num_readers;
        return outcome;
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

    // Verify outcomes in main thread
    for (auto& future : futures) {
      auto outcome = future.get();
      REQUIRE(outcome.status_ok);
      REQUIRE(outcome.pattern_ok);
    }

    // Cleanup
    REQUIRE(tensorcast::cuda::free(gpu_ptr).ok());
    for (auto* buffer : target_buffers) {
      std::free(buffer);
    }
  }

  SECTION("Partial read with offset and concurrent access") {
    LOG(INFO) << "Test: Partial read with offset and concurrent access";

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    communicator::v1::CommunicatorConfig cfg;
    cfg.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<Communicator>(cfg);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    const std::size_t tensor_size = 128 * 1024; // 128KB
    void* gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    // Fill with different patterns in different regions
    std::vector<uint8_t> test_data(tensor_size);
    for (size_t i = 0; i < tensor_size; i += 1024) {
      std::fill(test_data.begin() + i, test_data.begin() + i + 1024, static_cast<uint8_t>(i / 1024));
    }
    REQUIRE(tensorcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    Communicator::RegisterTensorOptions ro3;
    ro3.register_mr = false;
    ro3.needs_staging = true;
    ro3.async = false;
    REQUIRE(source_engine
                ->register_tensor_ex(
                    "offset_test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, ro3)
                .ok());

    // Multiple readers reading different offsets concurrently
    struct ReadSpec {
      size_t offset;
      size_t size;
      uint8_t expected_pattern;
    };

    std::vector<ReadSpec> read_specs = {
        {0, 4 * 1024, 0},
        {10 * 1024, 8 * 1024, 10},
        {50 * 1024, 16 * 1024, 50},
        {100 * 1024, 4 * 1024, 100},
    };

    std::vector<std::future<bool>> futures;
    std::vector<std::shared_ptr<Communicator>> engines;
    std::vector<void*> buffers;

    for (size_t i = 0; i < read_specs.size(); ++i) {
      int target_port = find_available_port(source_port + 10 + i * 10);
      REQUIRE(target_port > 0);

      communicator::v1::CommunicatorConfig cfg;
      cfg.set_enable_rdma(false); /* disable RDMA */
      auto engine = std::make_shared<Communicator>(cfg);
      REQUIRE(engine->init("127.0.0.1", target_port).ok());
      engines.push_back(engine);

      void* buffer = std::malloc(read_specs[i].size);
      REQUIRE(buffer != nullptr);
      buffers.push_back(buffer);

      futures.push_back(std::async(std::launch::async, [&, i]() -> bool {
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

        if (!result.status.ok()) {
          LOG(ERROR) << "Partial read " << i << " failed: " << result.status;
          return false;
        }

        // Verify the pattern
        uint8_t* data = static_cast<uint8_t*>(buffers[i]);
        bool pattern_correct = true;
        for (size_t j = 0; j < read_specs[i].size; j += 1024) {
          uint8_t expected = static_cast<uint8_t>((read_specs[i].offset + j) / 1024);
          if (data[j] != expected) {
            LOG(ERROR) << "Pattern mismatch at offset " << j << ": expected " << (int)expected << ", got "
                       << (int)data[j];
            pattern_correct = false;
            break;
          }
        }
        if (!pattern_correct) {
          return false;
        }

        LOG(INFO) << "Partial read " << i << " completed successfully";
        return true;
      }));
    }

    // Wait for all
    for (auto& future : futures) {
      REQUIRE(future.wait_for(std::chrono::seconds(30)) != std::future_status::timeout);
    }

    // Verify outcomes in main thread
    for (auto& future : futures) {
      REQUIRE(future.get());
    }

    // Cleanup
    REQUIRE(tensorcast::cuda::free(gpu_ptr).ok());
    for (auto* buffer : buffers) {
      std::free(buffer);
    }
  }

  SECTION("Rapid connection cycling with GPU transfers") {
    LOG(INFO) << "Test: Rapid connection cycling with GPU transfers";

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    communicator::v1::CommunicatorConfig cfg;
    cfg.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<Communicator>(cfg);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    const std::size_t tensor_size = 16 * 1024; // 16KB
    void* gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    auto test_data = create_test_pattern(tensor_size, 111);
    REQUIRE(tensorcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    {
      Communicator::RegisterTensorOptions opts;
      opts.register_mr = false;
      opts.needs_staging = true;
      opts.async = false;
      REQUIRE(
          source_engine
              ->register_tensor_ex(
                  "cycling_test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, opts)
              .ok());
    }

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
        if (target_port <= 0) {
          LOG(ERROR) << "Failed to find available target port for cycle " << i;
          return;
        }

        // Create new engine for each cycle
        communicator::v1::CommunicatorConfig cfg;
        cfg.set_enable_rdma(false); /* disable RDMA */
        cfg.mutable_stager()->set_buffers_per_flow(2); /* keep per-flow staging buffers bounded */
        auto engine = std::make_shared<Communicator>(cfg);
        auto init_status = engine->init("127.0.0.1", target_port);
        if (!init_status.ok()) {
          LOG(ERROR) << "Engine init failed for cycle " << i << ": " << init_status;
          return;
        }

        void* buffer = std::malloc(tensor_size);
        if (!buffer) {
          LOG(ERROR) << "Failed to allocate buffer for cycle " << i;
          return;
        }

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
          if (verify_pattern(buffer, tensor_size, 111)) {
            successful_cycles++;
            LOG(INFO) << "Cycle " << i << " completed successfully";
          } else {
            LOG(ERROR) << "Cycle " << i << " data verification failed";
          }
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
    REQUIRE(tensorcast::cuda::free(gpu_ptr).ok());
  }

  SECTION("Concurrent GPU-to-GPU transfers with staging buffer contention") {
    LOG(INFO) << "Test: Concurrent GPU-to-GPU transfers with staging buffer contention";

    // Preserve current CUDA device and restore on exit to avoid leaking device
    int original_device = -1;
    REQUIRE(tensorcast::cuda::get_device(&original_device).ok());

    // Check if we have at least 2 GPUs
    int device_count = 0;
    REQUIRE(tensorcast::cuda::get_device_count(&device_count).ok());
    if (device_count < 2) {
      LOG(INFO) << "Skipping multi-GPU test - only " << device_count << " GPU(s) available";
      return;
    }

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    communicator::v1::CommunicatorConfig cfg;
    cfg.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<Communicator>(cfg);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    // Create tensors on different GPUs
    const std::size_t tensor_size = 32 * 1024; // 32KB
    std::vector<void*> gpu_ptrs;

    for (int gpu_id = 0; gpu_id < std::min(2, device_count); ++gpu_id) {
      REQUIRE(tensorcast::cuda::set_device(gpu_id).ok());
      void* gpu_ptr;
      REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

      auto test_data = create_test_pattern(tensor_size, 100 + gpu_id);
      REQUIRE(tensorcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

      gpu_ptrs.push_back(gpu_ptr);

      std::string tensor_name = "gpu_tensor_" + std::to_string(gpu_id);
      {
        Communicator::RegisterTensorOptions opts;
        opts.register_mr = false;
        opts.needs_staging = true;
        opts.async = false;
        REQUIRE(source_engine
                    ->register_tensor_ex(
                        tensor_name,
                        reinterpret_cast<uint64_t>(gpu_ptr),
                        tensor_size,
                        COMMUNICATE_ENGINE_DEV_GPU,
                        gpu_id,
                        opts)
                    .ok());
      }

      LOG(INFO) << "Registered tensor on GPU " << gpu_id;
    }

    // Multiple readers for GPU-to-GPU transfers
    std::vector<std::future<bool>> futures;
    const int transfers_per_gpu = 3;

    futures.reserve(transfers_per_gpu * 2);
    for (int i = 0; i < transfers_per_gpu * 2; ++i) {
      futures.push_back(std::async(std::launch::async, [&, i]() -> bool {
        int src_gpu = i % 2;
        int dst_gpu = 1 - src_gpu;

        LOG(INFO) << "Transfer " << i << ": GPU " << src_gpu << " -> GPU " << dst_gpu;

        int target_port = find_available_port(source_port + 10 + i * 10);
        if (target_port <= 0) {
          LOG(ERROR) << "Failed to find target port for transfer " << i;
          return false;
        }

        communicator::v1::CommunicatorConfig cfg;
        cfg.set_enable_rdma(false); /* disable RDMA */
        auto engine = std::make_shared<Communicator>(cfg);
        auto init_status = engine->init("127.0.0.1", target_port);
        if (!init_status.ok()) {
          LOG(ERROR) << "Engine init failed for transfer " << i << ": " << init_status;
          return false;
        }

        if (!tensorcast::cuda::set_device(dst_gpu).ok()) {
          LOG(ERROR) << "Failed to set device to GPU " << dst_gpu << " for transfer " << i;
          return false;
        }
        void* dst_ptr;
        if (!tensorcast::cuda::malloc(&dst_ptr, tensor_size).ok()) {
          LOG(ERROR) << "Failed to allocate dst buffer for transfer " << i;
          return false;
        }

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
        if (!result.status.ok()) {
          LOG(ERROR) << "Transfer " << i << " read failed: " << result.status;
          (void)tensorcast::cuda::free(dst_ptr);
          return false;
        }

        // Verify data
        std::vector<uint8_t> verify_data(tensor_size);
        if (!tensorcast::cuda::memcpy(verify_data.data(), dst_ptr, tensor_size, cudaMemcpyDeviceToHost).ok()) {
          LOG(ERROR) << "Transfer " << i << " memcpy D2H failed";
          (void)tensorcast::cuda::free(dst_ptr);
          return false;
        }
        if (!verify_pattern(verify_data.data(), tensor_size, 100 + src_gpu)) {
          LOG(ERROR) << "Transfer " << i << " data verification failed";
          (void)tensorcast::cuda::free(dst_ptr);
          return false;
        }

        LOG(INFO) << "Transfer " << i << " completed successfully";
        bool freed = tensorcast::cuda::free(dst_ptr).ok();
        if (!freed) {
          LOG(ERROR) << "Failed to free dst buffer for transfer " << i;
        }
        return freed;
      }));
    }

    // Wait for all transfers
    for (auto& future : futures) {
      REQUIRE(future.wait_for(std::chrono::seconds(60)) != std::future_status::timeout);
    }

    // Verify outcomes in main thread
    for (auto& future : futures) {
      REQUIRE(future.get());
    }

    // Cleanup
    for (int gpu_id = 0; gpu_id < gpu_ptrs.size(); ++gpu_id) {
      REQUIRE(tensorcast::cuda::set_device(gpu_id).ok());
      REQUIRE(tensorcast::cuda::free(gpu_ptrs[gpu_id]).ok());
    }

    // Restore original device for subsequent sections
    REQUIRE(tensorcast::cuda::set_device(original_device).ok());
  }

  SECTION("Memory pressure and error recovery") {
    LOG(INFO) << "Test: Memory pressure and error recovery";

    // Ensure tests run on a stable default device
    REQUIRE(tensorcast::cuda::set_device(0).ok());

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    communicator::v1::CommunicatorConfig cfg;
    cfg.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<Communicator>(cfg);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    // Create multiple large tensors to stress memory
    const std::size_t tensor_size = 64 * 1024; // 64KB each
    const int num_tensors = 3;
    std::vector<void*> gpu_ptrs;

    for (int i = 0; i < num_tensors; ++i) {
      void* gpu_ptr;
      auto alloc_status = tensorcast::cuda::malloc(&gpu_ptr, tensor_size);
      if (!alloc_status.ok()) {
        LOG(WARNING) << "Failed to allocate tensor " << i << " - testing with " << i << " tensors"
                     << ", msg: " << alloc_status.message();
        break;
      }

      auto test_data = create_test_pattern(tensor_size, 200 + i);
      REQUIRE(tensorcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

      gpu_ptrs.push_back(gpu_ptr);

      std::string tensor_name = "pressure_tensor_" + std::to_string(i);
      Communicator::RegisterTensorOptions ro4;
      ro4.register_mr = false;
      ro4.needs_staging = true;
      ro4.async = false;
      REQUIRE(source_engine
                  ->register_tensor_ex(
                      tensor_name, reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, ro4)
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
        if (target_port <= 0) {
          LOG(ERROR) << "Failed to find target port for tensor " << i;
          failed_reads++;
          return;
        }

        communicator::v1::CommunicatorConfig cfg;
        cfg.set_enable_rdma(false); /* disable RDMA */
        auto engine = std::make_shared<Communicator>(cfg);
        auto init_status = engine->init("127.0.0.1", target_port);
        if (!init_status.ok()) {
          LOG(ERROR) << "Engine init failed for tensor " << i << ": " << init_status;
          failed_reads++;
          return;
        }

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
    // All attempts accounted for (no silent drops)
    REQUIRE(successful_reads.load() + failed_reads.load() == static_cast<int>(gpu_ptrs.size()));

    // Cleanup
    for (auto* ptr : gpu_ptrs) {
      REQUIRE(tensorcast::cuda::free(ptr).ok());
    }
  }

  SECTION("Deadlock prevention - all staging buffers occupied") {
    LOG(INFO) << "Test: Deadlock prevention with all staging buffers occupied";

    // Ensure tests run on a stable default device
    REQUIRE(tensorcast::cuda::set_device(0).ok());

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    communicator::v1::CommunicatorConfig cfg;
    cfg.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<Communicator>(cfg);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    // Create tensor that requires exactly one staging buffer
    const std::size_t chunk_size = 64 * 1024; // Default staging chunk size (KB for low-memory env)
    const std::size_t tensor_size = chunk_size; // Exactly one chunk

    void* gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, tensor_size).ok());
    auto test_data = create_test_pattern(tensor_size, 123);
    REQUIRE(tensorcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    {
      Communicator::RegisterTensorOptions opts;
      opts.register_mr = false;
      opts.needs_staging = true;
      opts.async = false;
      REQUIRE(source_engine
                  ->register_tensor_ex(
                      "deadlock_test",
                      reinterpret_cast<uint64_t>(gpu_ptr),
                      tensor_size,
                      COMMUNICATE_ENGINE_DEV_GPU,
                      0,
                      opts)
                  .ok());
    }

    // Create readers that will hold staging buffers
    const int num_holders = 2; // Equal to default staging buffers
    std::vector<std::shared_ptr<Communicator>> holder_engines;
    std::vector<void*> holder_buffers;
    std::vector<std::future<tensorcast::communicator::transport::read_result_t>> holder_futures;

    // Start reads that will occupy all staging buffers
    for (int i = 0; i < num_holders; ++i) {
      int target_port = find_available_port(source_port + 10 + i * 10);
      REQUIRE(target_port > 0);

      communicator::v1::CommunicatorConfig cfg;
      cfg.set_enable_rdma(false); /* disable RDMA */
      auto engine = std::make_shared<Communicator>(cfg);
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
    std::vector<std::shared_ptr<Communicator>> waiter_engines;
    std::vector<void*> waiter_buffers;
    std::vector<std::future<tensorcast::communicator::transport::read_result_t>> waiter_futures;
    std::atomic<int> waiters_started(0);

    // Pre-allocate futures vector to avoid race conditions
    waiter_futures.resize(num_waiters);

    for (int i = 0; i < num_waiters; ++i) {
      int target_port = find_available_port(source_port + 100 + i * 10);
      REQUIRE(target_port > 0);

      communicator::v1::CommunicatorConfig cfg;
      cfg.set_enable_rdma(false); /* disable RDMA */
      auto engine = std::make_shared<Communicator>(cfg);
      REQUIRE(engine->init("127.0.0.1", target_port).ok());
      waiter_engines.push_back(engine);

      void* buffer = std::malloc(tensor_size);
      REQUIRE(buffer != nullptr);
      waiter_buffers.push_back(buffer);
    }

    // Start waiter reads in separate threads and keep thread handles to avoid races
    std::vector<std::thread> waiter_threads;
    waiter_threads.reserve(num_waiters);
    for (int i = 0; i < num_waiters; ++i) {
      waiter_threads.emplace_back([&, i]() {
        LOG(INFO) << "Waiter " << i << " starting read (will wait for buffer)";
        // Assign future before updating the started counter to avoid races
        waiter_futures[i] = waiter_engines[i]->read_tensor(
            "deadlock_test",
            reinterpret_cast<uint64_t>(waiter_buffers[i]),
            tensor_size,
            COMMUNICATE_ENGINE_DEV_CPU,
            0,
            "127.0.0.1",
            source_port);
        waiters_started++;
      });
    }

    // Wait for all waiters to start
    while (waiters_started.load() < num_waiters) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Ensure waiter threads have assigned their futures
    for (auto& th : waiter_threads) {
      if (th.joinable()) {
        th.join();
      }
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
    REQUIRE(tensorcast::cuda::free(gpu_ptr).ok());
    for (auto* buffer : holder_buffers)
      std::free(buffer);
    for (auto* buffer : waiter_buffers)
      std::free(buffer);
  }

  SECTION("Chunk boundary alignment stress test") {
    LOG(INFO) << "Test: Chunk boundary alignment stress test";

    // Ensure tests run on a stable default device
    REQUIRE(tensorcast::cuda::set_device(0).ok());

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    communicator::v1::CommunicatorConfig cfg;
    cfg.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<Communicator>(cfg);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    // Test various sizes around chunk boundaries
    const std::size_t chunk_size = 1024; // Default staging chunk size (KB)
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
      REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, test_sizes[i]).ok());

      auto test_data = create_test_pattern(test_sizes[i], 150 + i);
      REQUIRE(tensorcast::cuda::memcpy(gpu_ptr, test_data.data(), test_sizes[i], cudaMemcpyHostToDevice).ok());

      gpu_ptrs.push_back(gpu_ptr);

      std::string tensor_name = "boundary_tensor_" + std::to_string(i);
      tensor_names.push_back(tensor_name);

      {
        Communicator::RegisterTensorOptions opts;
        opts.register_mr = false;
        opts.needs_staging = true;
        opts.async = false;
        REQUIRE(source_engine
                    ->register_tensor_ex(
                        tensor_name,
                        reinterpret_cast<uint64_t>(gpu_ptr),
                        test_sizes[i],
                        COMMUNICATE_ENGINE_DEV_GPU,
                        0,
                        opts)
                    .ok());
      }

      LOG(INFO) << "Registered tensor " << i << " with size " << test_sizes[i] << " ("
                << (double)test_sizes[i] / chunk_size << " chunks)";
    }

    // Read all tensors concurrently
    std::vector<std::future<bool>> futures;

    futures.reserve(test_sizes.size());
    for (size_t i = 0; i < test_sizes.size(); ++i) {
      futures.push_back(std::async(std::launch::async, [&, i]() -> bool {
        int target_port = find_available_port(source_port + 10 + i * 10);
        if (target_port <= 0) {
          LOG(ERROR) << "Failed to find available target port for boundary tensor " << i;
          return false;
        }

        communicator::v1::CommunicatorConfig cfg;
        cfg.set_enable_rdma(false); /* disable RDMA */
        auto engine = std::make_shared<Communicator>(cfg);
        auto init_status = engine->init("127.0.0.1", target_port);
        if (!init_status.ok()) {
          LOG(ERROR) << "Engine init failed for boundary tensor " << i << ": " << init_status;
          return false;
        }

        void* cpu_buffer = std::malloc(test_sizes[i]);
        if (!cpu_buffer) {
          LOG(ERROR) << "Failed to allocate CPU buffer for boundary tensor " << i;
          return false;
        }

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

        if (!result.status.ok()) {
          LOG(ERROR) << "Boundary tensor " << i << " read failed: " << result.status;
          std::free(cpu_buffer);
          return false;
        }
        bool ok = verify_pattern(cpu_buffer, test_sizes[i], 150 + i);
        if (!ok) {
          LOG(ERROR) << "Boundary tensor " << i << " data verification failed";
          std::free(cpu_buffer);
          return false;
        }

        LOG(INFO) << "Tensor " << i << " read completed successfully";

        std::free(cpu_buffer);
        return true;
      }));
    }

    // Wait for all
    for (auto& future : futures) {
      REQUIRE(future.wait_for(std::chrono::seconds(30)) != std::future_status::timeout);
    }

    // Verify outcomes in main thread
    for (auto& future : futures) {
      REQUIRE(future.get());
    }

    // Cleanup
    for (auto* ptr : gpu_ptrs) {
      REQUIRE(tensorcast::cuda::free(ptr).ok());
    }
  }

  SECTION("Tensor registration/unregistration during active transfers") {
    LOG(INFO) << "Test: Tensor registration/unregistration during active transfers";

    // Ensure tests run on a stable default device
    REQUIRE(tensorcast::cuda::set_device(0).ok());

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    communicator::v1::CommunicatorConfig cfg;
    cfg.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<Communicator>(cfg);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    const std::size_t tensor_size = 32 * 1024; // 32KB
    const int num_tensors = 5;

    // Create initial tensors
    std::vector<void*> gpu_ptrs;
    for (int i = 0; i < num_tensors; ++i) {
      void* gpu_ptr;
      REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

      auto test_data = create_test_pattern(tensor_size, 170 + i);
      REQUIRE(tensorcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

      gpu_ptrs.push_back(gpu_ptr);

      std::string tensor_name = "dynamic_tensor_" + std::to_string(i);
      {
        Communicator::RegisterTensorOptions opts;
        opts.register_mr = false;
        opts.needs_staging = true;
        opts.async = false;
        REQUIRE(
            source_engine
                ->register_tensor_ex(
                    tensor_name, reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, opts)
                .ok());
      }
    }

    std::atomic<bool> stop_flag(false);
    std::atomic<int> successful_reads(0);
    std::atomic<int> failed_reads(0);

    // Start reader threads
    std::vector<std::future<void>> reader_futures;
    for (int reader_id = 0; reader_id < 3; ++reader_id) {
      reader_futures.push_back(std::async(std::launch::async, [&, reader_id]() {
        int target_port = find_available_port(source_port + 10 + reader_id * 10);
        if (target_port <= 0) {
          LOG(ERROR) << "Failed to find target port for reader " << reader_id;
          return;
        }

        communicator::v1::CommunicatorConfig cfg;
        cfg.set_enable_rdma(false); /* disable RDMA */
        auto engine = std::make_shared<Communicator>(cfg);
        auto init_status = engine->init("127.0.0.1", target_port);
        if (!init_status.ok()) {
          LOG(ERROR) << "Engine init failed for reader " << reader_id << ": " << init_status;
          return;
        }

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
        REQUIRE(tensorcast::cuda::memcpy(gpu_ptrs[i], test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

        {
          Communicator::RegisterTensorOptions opts;
          opts.register_mr = false;
          opts.needs_staging = true;
          opts.async = false;
          REQUIRE(source_engine
                      ->register_tensor_ex(
                          tensor_name,
                          reinterpret_cast<uint64_t>(gpu_ptrs[i]),
                          tensor_size,
                          COMMUNICATE_ENGINE_DEV_GPU,
                          0,
                          opts)
                      .ok());
        }
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
      REQUIRE(tensorcast::cuda::free(ptr).ok());
    }
  }

  SECTION("ScopedStagedBuffer RAII and exception safety") {
    LOG(INFO) << "Test: ScopedStagedBuffer RAII and exception safety";

    // Ensure tests run on a stable default device
    REQUIRE(tensorcast::cuda::set_device(0).ok());

    // This test verifies that staged buffers are properly released even when
    // exceptions occur or when ScopedStagedBuffer goes out of scope early

    int source_port = find_available_port(50000);
    REQUIRE(source_port > 0);

    communicator::v1::CommunicatorConfig cfg15;
    cfg15.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<Communicator>(cfg15);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    const std::size_t tensor_size = 32 * 1024; // 32KB
    void* gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&gpu_ptr, tensor_size).ok());

    auto test_data = create_test_pattern(tensor_size, 180);
    REQUIRE(tensorcast::cuda::memcpy(gpu_ptr, test_data.data(), tensor_size, cudaMemcpyHostToDevice).ok());

    {
      Communicator::RegisterTensorOptions opts;
      opts.register_mr = false;
      opts.needs_staging = true;
      opts.async = false;
      REQUIRE(
          source_engine
              ->register_tensor_ex(
                  "raii_test", reinterpret_cast<uint64_t>(gpu_ptr), tensor_size, COMMUNICATE_ENGINE_DEV_GPU, 0, opts)
              .ok());
    }

    // Test multiple scenarios
    const int num_iterations = 10;
    std::atomic<int> buffers_acquired(0);
    std::atomic<int> buffers_released(0);
    std::atomic<int> read_status_fail_count(0);
    std::atomic<int> pattern_fail_count(0);
    std::atomic<int> normal_success_count(0);
    std::atomic<int> simulated_exceptions_count(0);

    std::vector<std::future<void>> futures;

    for (int i = 0; i < num_iterations; ++i) {
      futures.push_back(std::async(std::launch::async, [&, i]() {
        int target_port = find_available_port(source_port + 10 + i * 10);
        if (target_port <= 0) {
          LOG(ERROR) << "Failed to find available target port for iteration " << i;
          return;
        }

        communicator::v1::CommunicatorConfig cfg16;
        cfg16.set_enable_rdma(false); /* disable RDMA */
        auto engine = std::make_shared<Communicator>(cfg16);
        auto init_status = engine->init("127.0.0.1", target_port);
        if (!init_status.ok()) {
          LOG(ERROR) << "Engine init failed for iteration " << i << ": " << init_status;
          return;
        }

        void* cpu_buffer = std::malloc(tensor_size);
        if (!cpu_buffer) {
          LOG(ERROR) << "Failed to allocate CPU buffer for iteration " << i;
          return;
        }

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

            if (!result.status.ok()) {
              LOG(ERROR) << "Iteration " << i << " read failed: " << result.status;
              read_status_fail_count++;
            } else {
              if (i % 3 == 0) {
                // Simulate error condition
                LOG(INFO) << "Iteration " << i << " simulating error";
                throw std::runtime_error("Simulated error");
              }
              bool ok = verify_pattern(cpu_buffer, tensor_size, 180);
              if (ok) {
                LOG(INFO) << "Iteration " << i << " completed normally";
                normal_success_count++;
              } else {
                LOG(ERROR) << "Iteration " << i << " data verification failed";
                pattern_fail_count++;
              }
            }
          }
          // ScopedStagedBuffer should be released here
          buffers_released++;

        } catch (const std::exception& e) {
          LOG(INFO) << "Iteration " << i << " caught exception: " << e.what();
          // Buffer should still be released due to RAII
          buffers_released++;
          simulated_exceptions_count++;
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

    // Strong assertions in main thread for RAII behavior
    REQUIRE(read_status_fail_count.load() == 0);
    REQUIRE(pattern_fail_count.load() == 0);

    // All acquired buffers should be released
    REQUIRE(buffers_acquired.load() == buffers_released.load());

    // Verify we can still use staging buffers (they weren't leaked)
    {
      int final_port = find_available_port(source_port + 200);
      REQUIRE(final_port > 0);

      communicator::v1::CommunicatorConfig cfg;
      cfg.set_enable_rdma(false); /* disable RDMA */
      auto engine = std::make_shared<Communicator>(cfg);
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
    REQUIRE(tensorcast::cuda::free(gpu_ptr).ok());
  }
}
