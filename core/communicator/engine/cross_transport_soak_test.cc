// Copyright (c) 2025, TensorCast Team.

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "catch2/catch_test_macros.hpp"

#include "core/common/cuda_api.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/misc/ibv_wrap.h"
#include "core/testing/test_helpers.h"

using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;
using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
using tensorcast::communicator::engine::Communicator;
using tensorcast::communicator::misc::SUCCESS;
using tensorcast::communicator::misc::wrap_ibv_free_device_list;
using tensorcast::communicator::misc::wrap_ibv_get_device_list;
using tensorcast::communicator::misc::wrap_ibv_symbols;
using tensorcast::communicator::transport::future_read_result_t;
using tensorcast::testing::create_test_pattern;
using tensorcast::testing::find_available_port;
using tensorcast::testing::verify_pattern;

namespace {

struct RdmaClient {
  std::shared_ptr<Communicator> engine;
  void* gpu_buffer = nullptr;
  std::vector<uint8_t> host_verify;
};

struct MtcpClient {
  std::shared_ptr<Communicator> engine;
  std::vector<uint8_t> buffer;
};

} // namespace

TEST_CASE("Unified staging sustains mixed RDMA and MTCP transfers", "[communicator][rdma][mtcp][soak]") {
  int cuda_device_count = 0;
  auto cuda_status = tensorcast::cuda::get_device_count(&cuda_device_count);
  if (!cuda_status.ok() || cuda_device_count == 0) {
    INFO("Skipping cross-transport soak: " << cuda_status);
    SUCCEED("CUDA path unavailable in test environment");
    return;
  }

  REQUIRE(wrap_ibv_symbols() == SUCCESS);
  struct ibv_device** rdma_devices = nullptr;
  int rdma_device_count = 0;
  auto rdma_status = wrap_ibv_get_device_list(&rdma_devices, &rdma_device_count);
  if (rdma_status != SUCCESS || rdma_device_count == 0) {
    if (rdma_devices != nullptr) {
      wrap_ibv_free_device_list(rdma_devices);
    }
    INFO("Skipping cross-transport soak: no RDMA devices");
    SUCCEED("RDMA path unavailable in test environment");
    return;
  }
  wrap_ibv_free_device_list(rdma_devices);

  const std::size_t tensor_size = 128ULL * 1024ULL * 1024ULL; // 128 MiB tensor to force windowing
  const uint8_t seed = 0x5AU;
  const int rdma_client_count = 2;
  const int mtcp_client_count = 2;
  const int iterations = 6;

  int server_port = find_available_port(56000);
  REQUIRE(server_port > 0);

  tensorcast::communicator::v1::CommunicatorConfig server_cfg;
  server_cfg.set_enable_rdma(true);
  server_cfg.mutable_stager()->set_buffers_per_flow(4);
  server_cfg.mutable_stager()->set_max_window_segments(2);
  server_cfg.mutable_stager()->set_stage_chunk_mb_gpu(16);
  server_cfg.mutable_stager()->set_stage_chunk_mb_cpu(8);
  auto server = std::make_shared<Communicator>(server_cfg);
  REQUIRE(server->init("127.0.0.1", server_port, 8).ok());

  void* server_gpu_ptr = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&server_gpu_ptr, tensor_size).ok());
  auto pattern = create_test_pattern(tensor_size, seed);
  REQUIRE(tensorcast::cuda::memcpy(server_gpu_ptr, pattern.data(), tensor_size, cudaMemcpyHostToDevice).ok());

  Communicator::RegisterTensorOptions reg_opts;
  reg_opts.register_mr = true;
  reg_opts.needs_staging = true;
  reg_opts.async = false;
  auto register_status = server->register_tensor_ex(
      "cross_transport_tensor",
      reinterpret_cast<uint64_t>(server_gpu_ptr),
      tensor_size,
      COMMUNICATE_ENGINE_DEV_GPU,
      0,
      reg_opts);
  if (!register_status.ok()) {
    INFO("Skipping cross-transport soak: " << register_status);
    SUCCEED("RDMA path unavailable in test environment");
    return;
  }

  std::vector<RdmaClient> rdma_clients;
  rdma_clients.reserve(rdma_client_count);
  for (int i = 0; i < rdma_client_count; ++i) {
    tensorcast::communicator::v1::CommunicatorConfig cfg;
    cfg.set_enable_rdma(true);
    cfg.mutable_stager()->set_buffers_per_flow(4);
    cfg.mutable_stager()->set_max_window_segments(2);
    cfg.mutable_stager()->set_stage_chunk_mb_gpu(16);
    cfg.mutable_stager()->set_stage_chunk_mb_cpu(8);
    auto engine = std::make_shared<Communicator>(cfg);
    int port = find_available_port(server_port + 10 + i * 10);
    REQUIRE(port > 0);
    REQUIRE(engine->init("127.0.0.1", port, 8).ok());

    void* gpu_dest = nullptr;
    REQUIRE(tensorcast::cuda::malloc(&gpu_dest, tensor_size).ok());
    RdmaClient client;
    client.engine = std::move(engine);
    client.gpu_buffer = gpu_dest;
    client.host_verify.resize(tensor_size);
    rdma_clients.push_back(std::move(client));
  }

  std::vector<MtcpClient> mtcp_clients;
  mtcp_clients.reserve(mtcp_client_count);
  for (int i = 0; i < mtcp_client_count; ++i) {
    tensorcast::communicator::v1::CommunicatorConfig cfg;
    cfg.set_enable_rdma(false);
    cfg.mutable_stager()->set_buffers_per_flow(4);
    cfg.mutable_stager()->set_max_window_segments(2);
    cfg.mutable_stager()->set_stage_chunk_mb_gpu(16);
    cfg.mutable_stager()->set_stage_chunk_mb_cpu(8);
    cfg.mutable_transport()->set_tcp_conn_count(4);
    auto engine = std::make_shared<Communicator>(cfg);
    int port = find_available_port(server_port + 100 + i * 10);
    REQUIRE(port > 0);
    REQUIRE(engine->init("127.0.0.1", port, 4).ok());

    MtcpClient client;
    client.engine = std::move(engine);
    client.buffer.resize(tensor_size);
    mtcp_clients.push_back(std::move(client));
  }

  for (int iter = 0; iter < iterations; ++iter) {
    std::vector<future_read_result_t> rdma_futures;
    rdma_futures.reserve(rdma_clients.size());
    for (auto& client : rdma_clients) {
      rdma_futures.emplace_back(client.engine->read_tensor(
          "cross_transport_tensor",
          reinterpret_cast<uint64_t>(client.gpu_buffer),
          tensor_size,
          COMMUNICATE_ENGINE_DEV_GPU,
          0,
          "127.0.0.1",
          server_port));
    }

    std::vector<future_read_result_t> mtcp_futures;
    mtcp_futures.reserve(mtcp_clients.size());
    for (auto& client : mtcp_clients) {
      mtcp_futures.emplace_back(client.engine->read_tensor(
          "cross_transport_tensor",
          reinterpret_cast<uint64_t>(client.buffer.data()),
          tensor_size,
          COMMUNICATE_ENGINE_DEV_CPU,
          0,
          "127.0.0.1",
          server_port));
    }

    for (auto& future : rdma_futures) {
      REQUIRE(future.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
      auto result = future.get();
      REQUIRE(result.status.ok());
    }

    for (auto& future : mtcp_futures) {
      REQUIRE(future.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
      auto result = future.get();
      REQUIRE(result.status.ok());
    }

    const bool verify_this_round = (iter == 0) || (iter == iterations - 1);
    if (verify_this_round) {
      for (auto& client : rdma_clients) {
        REQUIRE(
            tensorcast::cuda::memcpy(client.host_verify.data(), client.gpu_buffer, tensor_size, cudaMemcpyDeviceToHost)
                .ok());
        REQUIRE(verify_pattern(client.host_verify.data(), tensor_size, seed));
      }
      for (auto& client : mtcp_clients) {
        REQUIRE(verify_pattern(client.buffer.data(), tensor_size, seed));
      }
    }
  }

  REQUIRE(server->unregister_tensor("cross_transport_tensor").ok());
  REQUIRE(tensorcast::cuda::free(server_gpu_ptr).ok());
  for (auto& client : rdma_clients) {
    if (client.gpu_buffer != nullptr) {
      REQUIRE(tensorcast::cuda::free(client.gpu_buffer).ok());
      client.gpu_buffer = nullptr;
    }
    client.engine.reset();
  }
  for (auto& client : mtcp_clients) {
    client.engine.reset();
  }
}
