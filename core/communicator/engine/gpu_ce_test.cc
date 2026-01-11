// Copyright (c) 2025-2026, TensorCast Team.

#include <sstream>
#include <thread>

#include "core/communicator/engine/engine.h"
#include "core/communicator/misc/utils.h"
#include "core/cuda/cuda_api.h"
#include "core/testing/common.h"
#include "core/testing/test_helpers.h"

#include "absl/log/check.h"

using tensorcast::testing::g_actor;
using tensorcast::testing::g_chunk;
using tensorcast::testing::g_count;
using tensorcast::testing::g_gpu;
using tensorcast::testing::g_ip;
using tensorcast::testing::g_port;
using tensorcast::testing::g_rdma;
using tensorcast::testing::parse_options;

int run_server() {
  auto cfg = tensorcast::testing::make_tcp_communicator_config(g_rdma);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/g_rdma);
  tensorcast::communicator::engine::Communicator engine(cfg, std::move(pools));
  CHECK_OK(engine.init("0.0.0.0", g_port));
  uint8_t* addr[8][1024] = {{nullptr}};

  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
  for (uint32_t i = 0; i < g_gpu; i++) {
    tensorcast::cuda::set_device(static_cast<int>(i)).IgnoreError();

    for (uint32_t j = 0; j < g_chunk; j++) {
      tensorcast::cuda::malloc(reinterpret_cast<void**>(&addr[i][j]), g_count).IgnoreError();

      std::stringstream name;
      name << std::string("gpu-ce-test-tensor-");
      name << i << "-" << j;
      tensorcast::communicator::engine::Communicator::RegisterTensorOptions opts;
      opts.register_mr = (g_rdma != 0);
      opts.needs_staging = (g_rdma == 0);
      opts.async = false;
      CHECK_OK(engine.register_tensor_ex(
          name.str(),
          reinterpret_cast<uint64_t>(addr[i][j]),
          g_count,
          COMMUNICATE_ENGINE_DEV_GPU,
          static_cast<int>(i),
          opts));
    }
  }
  while (true) {
    std::this_thread::yield();
  }
  return 0;
}

int run_client() {
  auto cfg = tensorcast::testing::make_tcp_communicator_config(g_rdma);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/g_rdma);
  tensorcast::communicator::engine::Communicator engine(cfg, std::move(pools), 10);
  CHECK_OK(engine.init("0.0.0.0", g_port + 1));
  uint8_t* addr[8][1024] = {{nullptr}};

  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
  for (uint32_t i = 0; i < g_gpu; i++) {
    tensorcast::cuda::set_device(static_cast<int>(i)).IgnoreError();
    for (uint32_t j = 0; j < g_chunk; j++) {
      tensorcast::cuda::malloc(reinterpret_cast<void**>(&addr[i][j]), g_count).IgnoreError();
    }
  }

  std::vector<tensorcast::communicator::transport::future_read_result_t> futures;

  auto start = tensorcast::communicator::misc::get_us();
  for (uint32_t i = 0; i < g_gpu; i++) {
    for (uint32_t j = 0; j < g_chunk; j++) {
      std::stringstream name;
      name << std::string("gpu-ce-test-tensor-");
      name << i << "-" << j;
      futures.push_back(engine.read_tensor(
          name.str(),
          reinterpret_cast<uint64_t>(addr[i][j]),
          g_count,
          COMMUNICATE_ENGINE_DEV_GPU,
          static_cast<int>(i),
          g_ip,
          g_port));
    }
  }

  for (auto& r : futures) {
    auto result = r.get();
    printf(
        "with regmr result: key=%s, status=%d, request=%lu, rdma_connect=%lu, "
        "regmr=%lu, rdma_read=%lu\n",
        result.tensor_key.c_str(),
        static_cast<int>(result.status.code()),
        result.request_cost,
        result.rdma_queue_cost,
        result.rdma_regmr_cost,
        result.read_cost);
  }
  printf("all with result: cost=%lu\n", tensorcast::communicator::misc::get_us() - start);

  futures.clear();
  start = tensorcast::communicator::misc::get_us();
  for (uint32_t i = 0; i < g_gpu; i++) {
    for (uint32_t j = 0; j < g_chunk; j++) {
      std::stringstream name;
      name << std::string("gpu-ce-test-tensor-");
      name << i << "-" << j;
      futures.push_back(engine.read_tensor(
          name.str(),
          reinterpret_cast<uint64_t>(addr[i][j]),
          g_count,
          COMMUNICATE_ENGINE_DEV_GPU,
          static_cast<int>(i),
          g_ip,
          g_port));
    }
  }

  for (auto& r : futures) {
    auto result = r.get();
    printf(
        "no regmr result: key=%s, status=%d, request=%lu, rdma_connect=%lu, "
        "regmr=%lu, rdma_read=%lu\n",
        result.tensor_key.c_str(),
        static_cast<int>(result.status.code()),
        result.request_cost,
        result.rdma_queue_cost,
        result.rdma_regmr_cost,
        result.read_cost);
  }

  printf("all no regmr result: cost=%lu\n", tensorcast::communicator::misc::get_us() - start);
  return 0;
}

int main(int argc, char* argv[]) {
  parse_options(argc, argv);

  if (g_actor == "server") {
    run_server();
  } else {
    run_client();
  }
  return 0;
}
