// Copyright (c) 2025, TensorCast Team.

#include <cstring>
#include <vector>
#include "core/common/cuda_api.h"

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"

#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/pinned_memory_pool.h"

#include "core/communicator/engine/engine.h"
#include "core/store/loader/p2p_loader.h"
#include "core/store/replica/memory_manager.h"
#include "core/testing/test_helpers.h"
#include "gsl/pointers"

using tensorcast::common::memory::DistributedVirtualMemoryPool;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::common::memory::PinnedMemoryPool;
using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;
using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
using tensorcast::communicator::engine::CommunicateEngine;
using tensorcast::communicator::v1::CommunicatorConfig;
using tensorcast::store::replica::MemoryManager;
using namespace tensorcast::store;
using namespace tensorcast::testing;

TEST_CASE("P2PLoader TCP Mode GPU Support", "[communicator][tcp][gpu][p2p_loader]") {
  SKIP_IF_NO_CUDA();

  SECTION("Remote GPU to Local GPU via TCP") {
    // Find available ports for GPU to GPU P2P communication
    int source_port = find_available_port();
    REQUIRE(source_port > 0);

    int target_port = find_available_port(source_port + 1);
    REQUIRE(target_port > 0);

    // Set up source engine with GPU tensor
    CommunicatorConfig cfg1;
    cfg1.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<CommunicateEngine>(cfg1);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    const std::size_t artifact_size = 16 * 1024 * 1024; // 16MB
    void* source_gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&source_gpu_ptr, artifact_size).ok());

    // Fill with test data
    auto test_data = create_test_pattern(artifact_size, 123);
    REQUIRE(tensorcast::cuda::memcpy(source_gpu_ptr, test_data.data(), artifact_size, cudaMemcpyHostToDevice).ok());

    // Register source tensor
    CommunicateEngine::RegisterTensorOptions reg_opts;
    reg_opts.register_mr = false; // RDMA disabled
    reg_opts.needs_staging = true; // GPU over TCP requires staging
    reg_opts.async = false;
    REQUIRE(source_engine
                ->register_tensor_ex(
                    "artifact_key",
                    reinterpret_cast<uint64_t>(source_gpu_ptr),
                    artifact_size,
                    COMMUNICATE_ENGINE_DEV_GPU,
                    0,
                    reg_opts)
                .ok());

    // Create target engine for loader
    CommunicatorConfig cfg2;
    cfg2.set_enable_rdma(false); /* disable RDMA */
    auto target_engine = std::make_shared<CommunicateEngine>(cfg2);
    REQUIRE(target_engine->init("127.0.0.1", target_port).ok());

    // Create P2PLoader with GPU source configuration (updated API)
    P2PSource source_config;
    source_config.size_bytes = artifact_size;
    source_config.ip = "127.0.0.1";
    source_config.port = source_port;
    source_config.memory_keys = {"artifact_key"};
    source_config.buf_sizes = {artifact_size};
    source_config.location.type = MemoryLocation::GPU;
    source_config.location.device_id = 0;
    source_config.comm_engine = gsl::not_null<std::shared_ptr<CommunicateEngine>>{
        target_engine}; // Local communicator used to pull remote tensor

    auto loader = std::make_shared<P2PLoader>(source_config);
    REQUIRE(loader->initialize().ok());

    // Create memory manager for target with pinned pool for streaming
    const std::size_t chunk_size = 1 * 1024 * 1024; // 1MB chunks for streaming
    const std::size_t pool_size = 16 * chunk_size; // Ensure at least 16 chunks available
    auto pinned_pool = std::make_shared<PinnedMemoryPool>(pool_size, chunk_size);
    auto dvmp = std::make_shared<DistributedVirtualMemoryPool>();
    auto mem_manager = std::make_shared<MemoryManager>(
        "test_artifact", // artifact_identifier
        0, // local_device_id (GPU device 0)
        pinned_pool, // pinned_pool (needed for streaming buffer)
        dvmp, // distributed virtual memory pool
        1024 * 1024 * 1024, // max_buffer_bytes (1GB)
        std::chrono::milliseconds::zero(),
        artifact_size);

    // Load asynchronously
    auto src_or = loader->open_source();
    REQUIRE(src_or.ok());
    auto load_future = mem_manager->load_async_from_source(std::move(*src_or), MemoryLocation::GPU, 1);
    auto load_status = load_future.get();
    CAPTURE(load_status.message());
    REQUIRE(load_status.ok());

    // Verify data
    auto gpu_ptrs = mem_manager->get_pointer(MemoryLocation::GPU);
    REQUIRE(gpu_ptrs.size() == 1);
    REQUIRE(gpu_ptrs[0] != nullptr);

    std::vector<uint8_t> verify_data(artifact_size);
    REQUIRE(tensorcast::cuda::memcpy(verify_data.data(), gpu_ptrs[0], artifact_size, cudaMemcpyDeviceToHost).ok());
    REQUIRE(verify_pattern(verify_data.data(), artifact_size, 123));

    // Cleanup
    REQUIRE(mem_manager->release_memory(MemoryLocation::GPU).ok());
    REQUIRE(tensorcast::cuda::free(source_gpu_ptr).ok());
  }

  SECTION("Remote GPU to Local CPU via TCP") {
    // Find available ports for GPU to CPU P2P communication
    int source_port = find_available_port();
    REQUIRE(source_port > 0); // "Failed to find available port for GPU-to-CPU P2P source engine"

    int target_port = find_available_port(source_port + 1);
    REQUIRE(target_port > 0); // "Failed to find available port for GPU-to-CPU P2P target engine"

    // Similar test but with CPU target
    CommunicatorConfig cfg3;
    cfg3.set_enable_rdma(false); /* disable RDMA */
    auto source_engine = std::make_shared<CommunicateEngine>(cfg3);
    auto source_init_status = source_engine->init("127.0.0.1", source_port);
    CAPTURE(source_port, source_init_status.message());
    REQUIRE(source_init_status.ok());

    const std::size_t artifact_size = 8 * 1024 * 1024; // 8MB
    void* source_gpu_ptr;
    auto malloc_status = tensorcast::cuda::malloc(&source_gpu_ptr, artifact_size);
    CAPTURE(artifact_size, malloc_status.message());
    REQUIRE(malloc_status.ok());

    auto test_data = create_test_pattern(artifact_size, 99);
    auto memcpy_status =
        tensorcast::cuda::memcpy(source_gpu_ptr, test_data.data(), artifact_size, cudaMemcpyHostToDevice);
    CAPTURE(memcpy_status.message());
    REQUIRE(memcpy_status.ok());

    CommunicateEngine::RegisterTensorOptions reg_opts2;
    reg_opts2.register_mr = false;
    reg_opts2.needs_staging = true;
    reg_opts2.async = false;
    auto register_status = source_engine->register_tensor_ex(
        "artifact_key",
        reinterpret_cast<uint64_t>(source_gpu_ptr),
        artifact_size,
        COMMUNICATE_ENGINE_DEV_GPU,
        0,
        reg_opts2);
    CAPTURE(register_status.message());
    REQUIRE(register_status.ok());

    CommunicatorConfig cfg4;
    cfg4.set_enable_rdma(false); /* disable RDMA */
    auto target_engine = std::make_shared<CommunicateEngine>(cfg4);
    auto target_init_status = target_engine->init("127.0.0.1", target_port);
    CAPTURE(target_port, target_init_status.message());
    REQUIRE(target_init_status.ok());

    P2PSource source_config;
    source_config.size_bytes = artifact_size;
    source_config.ip = "127.0.0.1";
    source_config.port = source_port;
    source_config.memory_keys = {"artifact_key"};
    source_config.buf_sizes = {artifact_size};
    source_config.location.type = MemoryLocation::GPU; // Remote data on GPU
    source_config.location.device_id = 0;
    source_config.comm_engine = gsl::not_null<std::shared_ptr<CommunicateEngine>>{target_engine};

    auto loader = std::make_shared<P2PLoader>(source_config);
    auto loader_init_status = loader->initialize();
    CAPTURE(loader_init_status.message());
    REQUIRE(loader_init_status.ok());

    // Create pinned memory pool with 4MB chunk size
    const std::size_t chunk_size = 4 * 1024 * 1024; // 4MB chunks
    const std::size_t total_pool_size = 16 * chunk_size; // Ensure at least 16 chunks available
    auto pinned_pool = std::make_shared<PinnedMemoryPool>(total_pool_size, chunk_size);
    auto dvmp = std::make_shared<DistributedVirtualMemoryPool>();
    auto mem_manager = std::make_shared<MemoryManager>(
        "test_artifact", // artifact_identifier
        -1, // local_device_id (-1 for CPU)
        pinned_pool, // pinned_pool
        dvmp, // distributed virtual memory pool
        1024 * 1024 * 1024, // max_buffer_bytes (1GB)
        std::chrono::milliseconds::zero(),
        artifact_size);

    // Allocate CPU memory before loading
    REQUIRE(mem_manager->allocate_memory(MemoryLocation::PAGEABLE_CPU).ok());

    auto src_or = loader->open_source();
    REQUIRE(src_or.ok());
    auto load_future = mem_manager->load_async_from_source(std::move(*src_or), MemoryLocation::PAGEABLE_CPU, 1);
    auto load_status = load_future.get();
    CAPTURE(load_status.message());
    // Remote GPU -> Local CPU is now supported; expect success.
    REQUIRE(load_status.ok());

    // Verify data in PAGEABLE_CPU UMA region
    auto cpu_ptrs = mem_manager->get_pointer(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(cpu_ptrs.size() == 1);
    REQUIRE(cpu_ptrs[0] != nullptr);
    REQUIRE(std::memcmp(cpu_ptrs[0], test_data.data(), artifact_size) == 0);

    // Release allocated CPU memory and GPU source
    REQUIRE(mem_manager->release_memory(MemoryLocation::PAGEABLE_CPU).ok());
    REQUIRE(tensorcast::cuda::free(source_gpu_ptr).ok());
  }
}
