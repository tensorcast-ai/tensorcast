// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <cuda_runtime.h>
#include <cstring>
#include <vector>

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"

#include "absl/strings/str_format.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/pinned_memory_pool.h"

#include "core/communicator/engine/engine.h"
#include "core/store/loader/p2p_loader.h"
#include "core/store/loader/source.h"
#include "core/store/model/memory_manager.h"
#include "tests/cpp/communicator/test_helpers.h"

using namespace stepcast::communicator;
using namespace stepcast::store;
using namespace stepcast::communicator::test;

TEST_CASE("P2PLoader TCP Mode GPU Support", "[communicator][tcp][gpu][p2p_loader]") {
  SKIP_IF_NO_CUDA();

  SECTION("Remote GPU to Local GPU via TCP") {
    // Set up source engine with GPU tensor
    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", 50061).ok());

    const std::size_t model_size = 16 * 1024 * 1024; // 16MB
    void* source_gpu_ptr;
    REQUIRE(cudaMalloc(&source_gpu_ptr, model_size) == cudaSuccess);

    // Fill with test data
    auto test_data = create_test_pattern(model_size, 123);
    REQUIRE(cudaMemcpy(source_gpu_ptr, test_data.data(), model_size, cudaMemcpyHostToDevice) == cudaSuccess);

    // Register source tensor
    REQUIRE(source_engine
                ->register_tensor(
                    "model_weights",
                    reinterpret_cast<uint64_t>(source_gpu_ptr),
                    model_size,
                    COMMUNICATE_ENGINE_DEV_GPU,
                    0,
                    false)
                .ok());

    // Create target engine for loader
    auto target_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(target_engine->init("127.0.0.1", 50062).ok());

    // Create P2PLoader with GPU source configuration (updated API)
    P2PSource source_config;
    source_config.size_bytes = model_size;
    source_config.ip = "127.0.0.1";
    source_config.port = 50061;
    source_config.memory_keys = {"model_weights"};
    source_config.buf_sizes = {model_size};
    source_config.location.type = ModelLocation::GPU;
    source_config.location.device_id = 0;
    source_config.comm_engine = target_engine; // Local communicator used to pull remote tensor

    auto loader = std::make_shared<P2PLoader>(source_config);
    REQUIRE(loader->initialize().ok());

    // Create memory manager for target with pinned pool for streaming
    const std::size_t chunk_size = 64 * 1024 * 1024; // 64MB chunks for streaming
    const std::size_t pool_size = 128 * 1024 * 1024; // 128MB pool (2 chunks)
    auto pinned_pool = std::make_shared<PinnedMemoryPool>(pool_size, chunk_size);
    auto dvmp = std::make_shared<stepcast::memory::DistributedVirtualMemoryPool>();
    auto mem_manager = std::make_shared<MemoryManager>(
        "test_model", // model_identifier
        0, // local_device_id (GPU device 0)
        pinned_pool, // pinned_pool (needed for streaming buffer)
        dvmp, // distributed virtual memory pool
        1024 * 1024 * 1024 // max_buffer_bytes (1GB)
    );
    mem_manager->set_model_size(model_size);

    // Load asynchronously
    auto src_or = loader->open_source();
    REQUIRE(src_or.ok());
    auto load_future = mem_manager->load_async_from_source(std::move(*src_or), ModelLocation::GPU, 1);
    auto load_status = load_future.get();
    CAPTURE(load_status.message());
    REQUIRE(load_status.ok());

    // Verify data
    auto gpu_ptrs = mem_manager->get_pointer(ModelLocation::GPU);
    REQUIRE(gpu_ptrs.size() == 1);
    REQUIRE(gpu_ptrs[0] != nullptr);

    std::vector<uint8_t> verify_data(model_size);
    REQUIRE(cudaMemcpy(verify_data.data(), gpu_ptrs[0], model_size, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(verify_pattern(verify_data.data(), model_size, 123));

    // Cleanup
    REQUIRE(mem_manager->release_memory(ModelLocation::GPU).ok());
    cudaFree(source_gpu_ptr);
  }

  SECTION("Remote GPU to Local CPU via TCP") {
    // Similar test but with CPU target
    auto source_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(source_engine->init("127.0.0.1", 50063).ok());

    const std::size_t model_size = 8 * 1024 * 1024; // 8MB
    void* source_gpu_ptr;
    REQUIRE(cudaMalloc(&source_gpu_ptr, model_size) == cudaSuccess);

    auto test_data = create_test_pattern(model_size, 99);
    REQUIRE(cudaMemcpy(source_gpu_ptr, test_data.data(), model_size, cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(source_engine
                ->register_tensor(
                    "model_weights_cpu",
                    reinterpret_cast<uint64_t>(source_gpu_ptr),
                    model_size,
                    COMMUNICATE_ENGINE_DEV_GPU,
                    0,
                    false)
                .ok());

    auto target_engine = std::make_shared<CommunicateEngine>(false /* disable RDMA */);
    REQUIRE(target_engine->init("127.0.0.1", 50064).ok());

    P2PSource source_config;
    source_config.size_bytes = model_size;
    source_config.ip = "127.0.0.1";
    source_config.port = 50063;
    source_config.memory_keys = {"model_weights_cpu"};
    source_config.buf_sizes = {model_size};
    source_config.location.type = ModelLocation::GPU; // Remote data on GPU
    source_config.location.device_id = 0;
    source_config.comm_engine = target_engine;

    auto loader = std::make_shared<P2PLoader>(source_config);
    REQUIRE(loader->initialize().ok());

    // Create pinned memory pool with 4MB chunk size
    const std::size_t chunk_size = 4 * 1024 * 1024; // 4MB chunks
    const std::size_t total_pool_size = model_size; // Allocate enough for the full model
    auto pinned_pool = std::make_shared<PinnedMemoryPool>(total_pool_size, chunk_size);
    auto dvmp = std::make_shared<stepcast::memory::DistributedVirtualMemoryPool>();
    auto mem_manager = std::make_shared<MemoryManager>(
        "test_model", // model_identifier
        -1, // local_device_id (-1 for CPU)
        pinned_pool, // pinned_pool
        dvmp, // distributed virtual memory pool
        1024 * 1024 * 1024 // max_buffer_bytes (1GB)
    );
    mem_manager->set_model_size(model_size);

    // Allocate CPU memory before loading
    REQUIRE(mem_manager->allocate_memory(ModelLocation::PAGEABLE_CPU).ok());

    auto src_or = loader->open_source();
    REQUIRE(src_or.ok());
    auto load_future = mem_manager->load_async_from_source(std::move(*src_or), ModelLocation::PAGEABLE_CPU, 1);
    auto load_status = load_future.get();
    CAPTURE(load_status.message());
    // Remote GPU -> Local CPU is currently unsupported and should fail.
    REQUIRE(!load_status.ok());

    // Release any allocated CPU memory regardless of failure.
    REQUIRE(mem_manager->release_memory(ModelLocation::PAGEABLE_CPU).ok());
    cudaFree(source_gpu_ptr);
  }
}