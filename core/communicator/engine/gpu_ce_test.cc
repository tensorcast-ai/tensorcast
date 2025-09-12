// Copyright (c) 2025, TensorCast Team.

#include <sstream>

#include "common.h"
#include "core/common/cuda_api.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/misc/utils.h"

using tensorcast::testing::g_actor;
using tensorcast::testing::g_chunk;
using tensorcast::testing::g_count;
using tensorcast::testing::g_gpu;
using tensorcast::testing::g_ip;
using tensorcast::testing::g_port;
using tensorcast::testing::parse_options;

int run_server() {
  tensorcast::communicator::engine::Communicator engine;
  engine.init("0.0.0.0", g_port);
  uint8_t* addr[8][1024] = {{nullptr}};

  for (uint32_t i = 0; i < g_gpu; i++) {
    tensorcast::cuda::set_device(static_cast<int>(i)).IgnoreError();

    for (uint32_t j = 0; j < g_chunk; j++) {
      tensorcast::cuda::malloc(reinterpret_cast<void**>(&addr[i][j]), g_count).IgnoreError();

      std::stringstream name;
      name << std::string("gpu-ce-test-tensor-");
      name << i << "-" << j;
      engine.register_tensor(name.str(), reinterpret_cast<uint64_t>(addr[i][j]), g_count, 0, i);
    }
  }
  while (true) {
    std::this_thread::yield();
  }
  return 0;
}

int run_client() {
  tensorcast::communicator::engine::Communicator engine;
  engine.init("0.0.0.0", g_port + 1);
  uint8_t* addr[8][1024] = {{nullptr}};

  for (uint32_t i = 0; i < g_gpu; i++) {
    tensorcast::cuda::set_device(static_cast<int>(i)).IgnoreError();
    for (uint32_t j = 0; j < g_chunk; j++) {
      tensorcast::cuda::malloc(reinterpret_cast<void**>(&addr[i][j]), g_count).IgnoreError();
    }
  }

  std::vector<tensorcast::communicator::future_read_result_t> futures;

  auto start = tensorcast::communicator::get_us();
  for (uint32_t i = 0; i < g_gpu; i++) {
    for (uint32_t j = 0; j < g_chunk; j++) {
      std::stringstream name;
      name << std::string("gpu-ce-test-tensor-");
      name << i << "-" << j;
      futures.push_back(
          engine.read_tensor(name.str(), reinterpret_cast<uint64_t>(addr[i][j]), g_count, 0, i, g_ip, g_port));
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
  printf("all with result: cost=%lu\n", tensorcast::communicator::get_us() - start);

  futures.clear();
  start = tensorcast::communicator::get_us();
  for (uint32_t i = 0; i < g_gpu; i++) {
    for (uint32_t j = 0; j < g_chunk; j++) {
      std::stringstream name;
      name << std::string("gpu-ce-test-tensor-");
      name << i << "-" << j;
      futures.push_back(
          engine.read_tensor(name.str(), reinterpret_cast<uint64_t>(addr[i][j]), g_count, 0, i, g_ip, g_port));
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

  printf("all no regmr result: cost=%lu\n", tensorcast::communicator::get_us() - start);
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
