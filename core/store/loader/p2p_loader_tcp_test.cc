// Copyright (c) 2025, TensorCast Team.

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>
#include "core/common/cuda_api.h"

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"

#include "core/common/artifact_verification.h"
#include "core/common/const/granularity.h"
#include "core/common/memory/pinned_buffer_pool.h"

#include "core/communicator/engine/engine.h"
#include "core/store/loader/p2p_loader.h"
#include "core/store/replica/replica_load_controller.h"
#include "core/testing/test_helpers.h"
#include "gsl/pointers"

using tensorcast::common::memory::MemoryLocation;
using tensorcast::common::memory::PinnedBufferPool;
using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;
using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
using tensorcast::communicator::engine::Communicator;
using tensorcast::communicator::v1::CommunicatorConfig;
using tensorcast::store::replica::ReplicaLoadController;
using namespace tensorcast::store;
using namespace tensorcast::testing;

namespace {

constexpr size_t kArtifactChunkBytes = tensorcast::common::consts::kArtifactChunkDefault;

int randomized_port_hint(int canonical_base) {
  constexpr int kMinPort = 1024;
  constexpr int kMaxPort = 65000;
  if (canonical_base <= 0) {
    canonical_base = 50000;
  }
  thread_local std::mt19937 rng([]() {
    std::random_device rd;
    return std::mt19937(rd());
  }());
  std::uniform_int_distribution<int> dist(-4096, 4096);
  int hint = canonical_base + dist(rng);
  hint = std::clamp(hint, kMinPort, kMaxPort);
  return hint;
}

} // namespace

TEST_CASE("P2PLoader TCP Mode GPU Support", "[communicator][tcp][gpu][p2p_loader]") {
  SKIP_IF_NO_CUDA();

  SECTION("Remote GPU to Local GPU via TCP") {
    // Use dedicated tensor key and port range to avoid crosstalk with other sections.
    constexpr int kGpuGpuPortBase = 50000;
    const std::string tensor_key = "artifact_key_gpu_gpu";

    // Find available ports for GPU to GPU P2P communication
    const int gpu_gpu_base = randomized_port_hint(kGpuGpuPortBase);
    int source_port = find_available_port(gpu_gpu_base);
    REQUIRE(source_port > 0);

    int target_port = find_available_port(randomized_port_hint(gpu_gpu_base + 1));
    REQUIRE(target_port > 0);

    // Set up source engine with GPU tensor
    auto cfg1 = make_tcp_communicator_config();
    auto source_engine = std::make_shared<Communicator>(cfg1);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    const std::size_t artifact_size = 16 * 1024 * 1024; // 16MB
    void* source_gpu_ptr;
    REQUIRE(tensorcast::cuda::malloc(&source_gpu_ptr, artifact_size).ok());

    // Fill with test data
    auto test_data = create_test_pattern(artifact_size, 123);
    REQUIRE(tensorcast::cuda::memcpy(source_gpu_ptr, test_data.data(), artifact_size, cudaMemcpyHostToDevice).ok());

    // Register source tensor
    Communicator::RegisterTensorOptions reg_opts;
    reg_opts.register_mr = false; // RDMA disabled
    reg_opts.needs_staging = true; // GPU over TCP requires staging
    reg_opts.async = false;
    REQUIRE(source_engine
                ->register_tensor_ex(
                    tensor_key,
                    reinterpret_cast<uint64_t>(source_gpu_ptr),
                    artifact_size,
                    COMMUNICATE_ENGINE_DEV_GPU,
                    0,
                    reg_opts)
                .ok());

    // Create target engine for loader
    auto cfg2 = make_tcp_communicator_config();
    auto target_engine = std::make_shared<Communicator>(cfg2);
    REQUIRE(target_engine->init("127.0.0.1", target_port).ok());

    // Create P2PLoader with GPU source configuration (updated API)
    P2PSource source_config;
    source_config.size_bytes = artifact_size;
    source_config.ip = "127.0.0.1";
    source_config.port = source_port;
    source_config.memory_keys = {tensor_key};
    source_config.buf_sizes = {artifact_size};
    source_config.location.type = MemoryLocation::GPU;
    source_config.location.device_id = 0;
    source_config.comm_engine =
        gsl::not_null<std::shared_ptr<Communicator>>{target_engine}; // Local communicator used to pull remote tensor

    auto loader = std::make_shared<P2PLoader>(source_config);
    REQUIRE(loader->initialize().ok());

    // Create memory manager for target with pinned pool for streaming
    const std::size_t chunk_size = 1 * 1024 * 1024; // 1MB chunks for streaming
    const std::size_t pool_size = 16 * chunk_size; // Ensure at least 16 chunks available
    auto pinned_pool = std::make_shared<PinnedBufferPool>(pool_size, chunk_size);
    auto mem_manager = std::make_shared<ReplicaLoadController>(
        "test_artifact", // artifact_identifier
        0, // local_device_id (GPU device 0)
        pinned_pool, // pinned_pool (needed for streaming buffer)
        kArtifactChunkBytes,
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
    // Use separate tensor key and port range to avoid reuse of GPU->GPU resources.
    constexpr int kGpuCpuPortBase = 52000;
    const std::string tensor_key = "artifact_key_gpu_cpu";

    // Find available ports for GPU to CPU P2P communication
    const int gpu_cpu_base = randomized_port_hint(kGpuCpuPortBase);
    int source_port = find_available_port(gpu_cpu_base);
    REQUIRE(source_port > 0); // "Failed to find available port for GPU-to-CPU P2P source engine"

    int target_port = find_available_port(randomized_port_hint(gpu_cpu_base + 1));
    REQUIRE(target_port > 0); // "Failed to find available port for GPU-to-CPU P2P target engine"

    // Similar test but with CPU target
    auto cfg3 = make_tcp_communicator_config();
    auto source_engine = std::make_shared<Communicator>(cfg3);
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

    Communicator::RegisterTensorOptions reg_opts2;
    reg_opts2.register_mr = false;
    reg_opts2.needs_staging = true;
    reg_opts2.async = false;
    auto register_status = source_engine->register_tensor_ex(
        tensor_key,
        reinterpret_cast<uint64_t>(source_gpu_ptr),
        artifact_size,
        COMMUNICATE_ENGINE_DEV_GPU,
        0,
        reg_opts2);
    CAPTURE(register_status.message());
    REQUIRE(register_status.ok());

    auto cfg4 = make_tcp_communicator_config();
    auto target_engine = std::make_shared<Communicator>(cfg4);
    auto target_init_status = target_engine->init("127.0.0.1", target_port);
    CAPTURE(target_port, target_init_status.message());
    REQUIRE(target_init_status.ok());

    P2PSource source_config;
    source_config.size_bytes = artifact_size;
    source_config.ip = "127.0.0.1";
    source_config.port = source_port;
    source_config.memory_keys = {tensor_key};
    source_config.buf_sizes = {artifact_size};
    source_config.location.type = MemoryLocation::GPU; // Remote data on GPU
    source_config.location.device_id = 0;
    source_config.comm_engine = gsl::not_null<std::shared_ptr<Communicator>>{target_engine};

    auto loader = std::make_shared<P2PLoader>(source_config);
    auto loader_init_status = loader->initialize();
    CAPTURE(loader_init_status.message());
    REQUIRE(loader_init_status.ok());

    // Create pinned memory pool with 4MB chunk size
    const std::size_t chunk_size = 4 * 1024 * 1024; // 4MB chunks
    const std::size_t total_pool_size = 16 * chunk_size; // Ensure at least 16 chunks available
    auto pinned_pool = std::make_shared<PinnedBufferPool>(total_pool_size, chunk_size);
    auto mem_manager = std::make_shared<ReplicaLoadController>(
        "test_artifact", // artifact_identifier
        -1, // local_device_id (-1 for CPU)
        pinned_pool, // pinned_pool
        kArtifactChunkBytes,
        1024 * 1024 * 1024, // max_buffer_bytes (1GB)
        std::chrono::milliseconds::zero(),
        artifact_size);

    // Allocate CPU memory before loading
    REQUIRE(mem_manager->allocate_memory(MemoryLocation::CPU).ok());

    auto src_or = loader->open_source();
    REQUIRE(src_or.ok());
    auto load_future = mem_manager->load_async_from_source(std::move(*src_or), MemoryLocation::CPU, 1);
    auto load_status = load_future.get();
    CAPTURE(load_status.message());
    // Remote GPU -> Local CPU is now supported; expect success.
    REQUIRE(load_status.ok());

    // Verify data in CPU UMA region
    auto cpu_ptrs = mem_manager->get_pointer(MemoryLocation::CPU);
    REQUIRE(cpu_ptrs.size() == 1);
    REQUIRE(cpu_ptrs[0] != nullptr);
    REQUIRE(std::memcmp(cpu_ptrs[0], test_data.data(), artifact_size) == 0);

    // Release allocated CPU memory and GPU source
    REQUIRE(mem_manager->release_memory(MemoryLocation::CPU).ok());
    REQUIRE(tensorcast::cuda::free(source_gpu_ptr).ok());
  }

  SECTION("Remote GPU to Local GPU via TCP with sub-chunked MTCP receive") {
    // This test reproduces the regression where MTCP splits each network chunk
    // into multiple pinned-buffer sub-chunks and the final GPU bytes become zero
    // after transfer, triggering the key-point verification failure.

    const int tcp_conn_count = 8;
    const int load_concurrency = 8;
    const std::size_t pool_chunk_size = 64 * 1024 * 1024; // 64 MiB slices
    const std::size_t artifact_size = pool_chunk_size * 9 + 32; // non-aligned tail chunk

    const int gpu_gpu_subchunk_base = randomized_port_hint(50000);
    int source_port = find_available_port(gpu_gpu_subchunk_base);
    REQUIRE(source_port > 0);
    int target_port = find_available_port(randomized_port_hint(gpu_gpu_subchunk_base + 1));
    REQUIRE(target_port > 0);

    auto src_cfg = make_tcp_communicator_config(
        /*enable_rdma=*/false,
        /*gpu_chunk_mb=*/16,
        /*cpu_chunk_mb=*/4,
        /*buffers_per_flow=*/4);
    src_cfg.mutable_transport()->set_tcp_conn_count(tcp_conn_count);
    src_cfg.mutable_pool()->set_pool_size_bytes(4ull * 1024 * 1024 * 1024);
    src_cfg.mutable_pool()->set_chunk_bytes(pool_chunk_size);
    auto source_engine = std::make_shared<Communicator>(src_cfg);
    REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

    auto dst_cfg = make_tcp_communicator_config(
        /*enable_rdma=*/false,
        /*gpu_chunk_mb=*/16,
        /*cpu_chunk_mb=*/4,
        /*buffers_per_flow=*/4);
    dst_cfg.mutable_transport()->set_tcp_conn_count(tcp_conn_count);
    dst_cfg.mutable_pool()->set_pool_size_bytes(4ull * 1024 * 1024 * 1024);
    dst_cfg.mutable_pool()->set_chunk_bytes(pool_chunk_size);
    auto target_engine = std::make_shared<Communicator>(dst_cfg);
    REQUIRE(target_engine->init("127.0.0.1", target_port).ok());

    void* source_gpu_ptr = nullptr;
    auto alloc_status = tensorcast::cuda::malloc(&source_gpu_ptr, artifact_size);
    if (!alloc_status.ok()) {
      SKIP("Insufficient GPU memory to allocate test artifact: " << alloc_status.message());
    }

    auto payload = create_test_pattern(artifact_size, 7);
    REQUIRE(tensorcast::cuda::memcpy(source_gpu_ptr, payload.data(), artifact_size, cudaMemcpyHostToDevice).ok());

    Communicator::RegisterTensorOptions reg_opts;
    reg_opts.register_mr = false;
    reg_opts.needs_staging = true;
    reg_opts.async = false;

    const std::size_t remote_chunk_size = 268435456; // 256 MiB (matches UMA chunk)
    std::vector<std::string> memory_keys;
    std::vector<uint64_t> buffer_sizes;
    size_t offset = 0;
    int chunk_index = 0;
    while (offset < artifact_size) {
      const size_t chunk_bytes = std::min(remote_chunk_size, artifact_size - offset);
      std::string key = "sub_chunk_artifact_key_" + std::to_string(chunk_index++);
      auto ptr = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(source_gpu_ptr) + offset);
      REQUIRE(source_engine->register_tensor_ex(key, ptr, chunk_bytes, COMMUNICATE_ENGINE_DEV_GPU, 0, reg_opts).ok());
      memory_keys.push_back(key);
      buffer_sizes.push_back(chunk_bytes);
      offset += chunk_bytes;
    }

    std::vector<void*> data_ptrs{source_gpu_ptr};
    std::vector<size_t> data_sizes{artifact_size};
    auto ver_or = tensorcast::common::ArtifactVerifier::generate_verification_info(
        data_ptrs,
        data_sizes,
        /*device_id=*/0,
        tensorcast::common::VerificationLevel::KEY_POINTS);
    REQUIRE(ver_or.ok());

    P2PSource source_config;
    source_config.size_bytes = artifact_size;
    source_config.ip = "127.0.0.1";
    source_config.port = source_port;
    source_config.memory_keys = memory_keys;
    source_config.buf_sizes = buffer_sizes;
    source_config.location.type = MemoryLocation::GPU;
    source_config.location.device_id = 0;
    source_config.comm_engine = gsl::not_null<std::shared_ptr<Communicator>>{target_engine};
    source_config.verification_json = ver_or->to_json();

    auto loader = std::make_shared<P2PLoader>(source_config);
    REQUIRE(loader->initialize().ok());

    const std::size_t pool_buffers = 20; // ensure enough slices for MTCP receive without exhausting host memory
    const std::size_t pool_size = pool_chunk_size * pool_buffers;
    auto pinned_pool = std::make_shared<PinnedBufferPool>(pool_size, pool_chunk_size);
    auto mem_manager = std::make_shared<ReplicaLoadController>(
        "sub_chunk_artifact",
        0,
        pinned_pool,
        kArtifactChunkBytes,
        pool_size,
        std::chrono::milliseconds::zero(),
        artifact_size);

    auto src_or = loader->open_source();
    REQUIRE(src_or.ok());

    auto future = mem_manager->load_async_from_source(
        std::move(*src_or),
        MemoryLocation::GPU,
        /*concurrency=*/load_concurrency);

    auto load_status = future.get();
    CAPTURE(load_status.message());
    REQUIRE(load_status.ok());

    auto gpu_ptrs = mem_manager->get_pointer(MemoryLocation::GPU);
    REQUIRE(gpu_ptrs.size() == 1);
    REQUIRE(gpu_ptrs[0] != nullptr);

    std::vector<uint8_t> received(artifact_size);
    REQUIRE(tensorcast::cuda::memcpy(received.data(), gpu_ptrs[0], artifact_size, cudaMemcpyDeviceToHost).ok());
    REQUIRE(verify_pattern(received.data(), artifact_size, 7));

    REQUIRE(mem_manager->release_memory(MemoryLocation::GPU).ok());
    if (source_gpu_ptr != nullptr) {
      REQUIRE(tensorcast::cuda::free(source_gpu_ptr).ok());
    }

    SECTION("Remote GPU to Local GPU via TCP with limited staging credit") {
      const int tcp_conn_count = 8;
      const std::size_t stage_chunk_bytes = 16ull * 1024 * 1024; // 16 MiB
      const uint32_t stage_chunk_mb = static_cast<uint32_t>(stage_chunk_bytes / (1024 * 1024));
      const std::size_t artifact_size = stage_chunk_bytes * 6; // Requires >1 window when buffers_per_flow=2

      const int credit_base = randomized_port_hint(50500);
      int source_port = find_available_port(credit_base);
      REQUIRE(source_port > 0);
      int target_port = find_available_port(randomized_port_hint(credit_base + 1));
      REQUIRE(target_port > 0);

      auto src_cfg = make_tcp_communicator_config(
          /*enable_rdma=*/false,
          /*gpu_chunk_mb=*/stage_chunk_mb,
          /*cpu_chunk_mb=*/4,
          /*buffers_per_flow=*/2);
      src_cfg.mutable_transport()->set_tcp_conn_count(tcp_conn_count);
      src_cfg.mutable_pool()->set_pool_size_bytes(2ull * 1024 * 1024 * 1024);
      src_cfg.mutable_pool()->set_chunk_bytes(stage_chunk_bytes);
      auto source_engine = std::make_shared<Communicator>(src_cfg);
      REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

      void* source_gpu_ptr = nullptr;
      auto alloc_status = tensorcast::cuda::malloc(&source_gpu_ptr, artifact_size);
      CAPTURE(alloc_status.message());
      REQUIRE(alloc_status.ok());

      auto pattern = create_test_pattern(artifact_size, 211);
      auto h2d_status = tensorcast::cuda::memcpy(source_gpu_ptr, pattern.data(), artifact_size, cudaMemcpyHostToDevice);
      CAPTURE(h2d_status.message());
      REQUIRE(h2d_status.ok());

      Communicator::RegisterTensorOptions reg_opts;
      reg_opts.register_mr = false;
      reg_opts.needs_staging = true;
      reg_opts.async = false;
      auto reg_status = source_engine->register_tensor_ex(
          "credit_artifact",
          reinterpret_cast<uint64_t>(source_gpu_ptr),
          artifact_size,
          COMMUNICATE_ENGINE_DEV_GPU,
          0,
          reg_opts);
      CAPTURE(reg_status.message());
      REQUIRE(reg_status.ok());

      auto dst_cfg = make_tcp_communicator_config(
          /*enable_rdma=*/false,
          /*gpu_chunk_mb=*/stage_chunk_mb,
          /*cpu_chunk_mb=*/4,
          /*buffers_per_flow=*/2);
      dst_cfg.mutable_transport()->set_tcp_conn_count(tcp_conn_count);
      dst_cfg.mutable_pool()->set_pool_size_bytes(2ull * 1024 * 1024 * 1024);
      dst_cfg.mutable_pool()->set_chunk_bytes(stage_chunk_bytes);
      auto target_engine = std::make_shared<Communicator>(dst_cfg);
      auto init_status = target_engine->init("127.0.0.1", target_port);
      CAPTURE(init_status.message());
      REQUIRE(init_status.ok());

      P2PSource p2p_src;
      p2p_src.size_bytes = artifact_size;
      p2p_src.ip = "127.0.0.1";
      p2p_src.port = source_port;
      p2p_src.memory_keys = {"credit_artifact"};
      p2p_src.buf_sizes = {artifact_size};
      p2p_src.location.type = MemoryLocation::GPU;
      p2p_src.location.device_id = 0;
      p2p_src.comm_engine = gsl::not_null<std::shared_ptr<Communicator>>{target_engine};

      auto loader = std::make_shared<P2PLoader>(p2p_src);
      auto loader_init = loader->initialize();
      CAPTURE(loader_init.message());
      REQUIRE(loader_init.ok());

      // Provision the pinned pool to cover the maximum number of simultaneous staging buffers
      // (TransferService defaults to 16 chunks when credit is saturated). Keep the pool large enough
      // so we validate the intended MTCP behaviour instead of exercising OOM fallbacks.
      auto pinned_pool = std::make_shared<PinnedBufferPool>(stage_chunk_bytes * 16, stage_chunk_bytes);
      auto mem_manager = std::make_shared<ReplicaLoadController>(
          "credit_artifact",
          0,
          pinned_pool,
          kArtifactChunkBytes,
          1024 * 1024 * 1024,
          std::chrono::milliseconds::zero(),
          artifact_size);

      auto src_or = loader->open_source();
      REQUIRE(src_or.ok());
      auto load_future = mem_manager->load_async_from_source(std::move(*src_or), MemoryLocation::GPU, 1);
      auto load_status = load_future.get();
      CAPTURE(load_status.message());
      REQUIRE(load_status.ok());

      auto gpu_ptrs = mem_manager->get_pointer(MemoryLocation::GPU);
      REQUIRE(gpu_ptrs.size() == 1);
      REQUIRE(gpu_ptrs[0] != nullptr);

      std::vector<uint8_t> verify(artifact_size);
      auto d2h_status = tensorcast::cuda::memcpy(verify.data(), gpu_ptrs[0], artifact_size, cudaMemcpyDeviceToHost);
      CAPTURE(d2h_status.message());
      REQUIRE(d2h_status.ok());
      REQUIRE(verify_pattern(verify.data(), artifact_size, 211));

      REQUIRE(mem_manager->release_memory(MemoryLocation::GPU).ok());
      REQUIRE(tensorcast::cuda::free(source_gpu_ptr).ok());
    }

    SECTION("Remote GPU to Local GPU via TCP with insufficient pinned pool capacity") {
      const int tcp_conn_count = 4;
      const int buffers_per_flow = 2;
      const std::size_t stage_chunk_bytes = 16ull * 1024 * 1024; // 16 MiB
      const std::size_t artifact_size = stage_chunk_bytes * 4;
      const std::size_t communicator_pool_bytes =
          stage_chunk_bytes * buffers_per_flow * (static_cast<std::size_t>(tcp_conn_count) + 1);

      const int small_pool_base = randomized_port_hint(51000);
      int source_port = find_available_port(small_pool_base);
      REQUIRE(source_port > 0);
      int target_port = find_available_port(randomized_port_hint(small_pool_base + 1));
      REQUIRE(target_port > 0);

      auto src_cfg = make_tcp_communicator_config(
          /*enable_rdma=*/false,
          /*gpu_chunk_mb=*/16,
          /*cpu_chunk_mb=*/4,
          /*buffers_per_flow=*/buffers_per_flow);
      src_cfg.mutable_transport()->set_tcp_conn_count(tcp_conn_count);
      src_cfg.mutable_pool()->set_pool_size_bytes(communicator_pool_bytes);
      src_cfg.mutable_pool()->set_chunk_bytes(stage_chunk_bytes);
      auto source_engine = std::make_shared<Communicator>(src_cfg);
      REQUIRE(source_engine->init("127.0.0.1", source_port).ok());

      void* source_gpu_ptr = nullptr;
      auto alloc_status = tensorcast::cuda::malloc(&source_gpu_ptr, artifact_size);
      REQUIRE(alloc_status.ok());

      auto payload = create_test_pattern(artifact_size, 123);
      REQUIRE(tensorcast::cuda::memcpy(source_gpu_ptr, payload.data(), artifact_size, cudaMemcpyHostToDevice).ok());

      Communicator::RegisterTensorOptions reg_opts;
      reg_opts.register_mr = false;
      reg_opts.needs_staging = true;
      reg_opts.async = false;
      CAPTURE(source_engine
                  ->register_tensor_ex(
                      "credit_artifact_small_pool",
                      reinterpret_cast<uint64_t>(source_gpu_ptr),
                      artifact_size,
                      COMMUNICATE_ENGINE_DEV_GPU,
                      0,
                      reg_opts)
                  .message());
      REQUIRE(source_engine
                  ->register_tensor_ex(
                      "credit_artifact_small_pool",
                      reinterpret_cast<uint64_t>(source_gpu_ptr),
                      artifact_size,
                      COMMUNICATE_ENGINE_DEV_GPU,
                      0,
                      reg_opts)
                  .ok());

      auto dst_cfg = make_tcp_communicator_config(
          /*enable_rdma=*/false,
          /*gpu_chunk_mb=*/16,
          /*cpu_chunk_mb=*/4,
          /*buffers_per_flow=*/buffers_per_flow);
      dst_cfg.mutable_transport()->set_tcp_conn_count(tcp_conn_count);
      dst_cfg.mutable_pool()->set_pool_size_bytes(communicator_pool_bytes);
      dst_cfg.mutable_pool()->set_chunk_bytes(stage_chunk_bytes);
      auto target_engine = std::make_shared<Communicator>(dst_cfg);
      REQUIRE(target_engine->init("127.0.0.1", target_port).ok());

      P2PSource p2p_src;
      p2p_src.size_bytes = artifact_size;
      p2p_src.ip = "127.0.0.1";
      p2p_src.port = source_port;
      p2p_src.memory_keys = {"credit_artifact_small_pool"};
      p2p_src.buf_sizes = {artifact_size};
      p2p_src.location.type = MemoryLocation::GPU;
      p2p_src.location.device_id = 0;
      p2p_src.comm_engine = gsl::not_null<std::shared_ptr<Communicator>>{target_engine};

      auto loader = std::make_shared<P2PLoader>(p2p_src);
      REQUIRE(loader->initialize().ok());

      // ReplicaLoadController pool intentionally remains undersized to trigger pinned OOM behaviour.
      auto pinned_pool = std::make_shared<PinnedBufferPool>(stage_chunk_bytes * 2, stage_chunk_bytes);
      auto mem_manager = std::make_shared<ReplicaLoadController>(
          "credit_artifact_small_pool",
          0,
          pinned_pool,
          kArtifactChunkBytes,
          stage_chunk_bytes * 2,
          std::chrono::milliseconds::zero(),
          artifact_size);

      auto src_or = loader->open_source();
      REQUIRE(src_or.ok());
      auto load_future = mem_manager->load_async_from_source(std::move(*src_or), MemoryLocation::GPU, 1);
      auto load_status = load_future.get();
      CAPTURE(load_status.message());
      REQUIRE_FALSE(load_status.ok());
      const absl::string_view status_str = load_status.message();
      if (status_str.find("PinnedBufferPool out of memory") == absl::string_view::npos &&
          status_str.find("Failed to allocate chunks from pinned memory pool") == absl::string_view::npos) {
        FAIL_CHECK("Expected pinned buffer pool OOM message, got: " << status_str);
      }

      REQUIRE(tensorcast::cuda::free(source_gpu_ptr).ok());
    }
  }
}
