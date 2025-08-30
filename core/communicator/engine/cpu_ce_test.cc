// Copyright (c) 2025, TensorCast Team.

#include <sstream>

#include "common.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/misc/utils.h"

int run_server() {
  tensorcast::communicator::CommunicatorConfig cfg;
  cfg.enable_rdma = g_rdma;
  tensorcast::communicator::CommunicateEngine engine(cfg);
  engine.init(g_ip, g_port);
  uint8_t* addr = reinterpret_cast<uint8_t*>(malloc(g_count));
  memset(addr, 0, g_count);
  communicator::CommunicateEngine::RegisterTensorOptions opts;
  opts.register_mr = false;
  opts.needs_staging = false;
  opts.async = false;
  engine.register_tensor_ex("cpu-ce-test-tensor", reinterpret_cast<uint64_t>(addr), g_count, 0, 1, opts);
  while (true) {
    std::this_thread::yield();
  }
  return 0;
}

int run_client() {
  tensorcast::communicator::CommunicatorConfig cfg;
  cfg.enable_rdma = g_rdma;
  tensorcast::communicator::CommunicateEngine engine(cfg, 10);
  engine.init("0.0.0.0", g_port + 1);
  auto addr = reinterpret_cast<uint8_t*>(malloc(g_count));

  auto start = tensorcast::communicator::get_us();
  auto result = engine.read_tensor("cpu-ce-test-tensor", reinterpret_cast<uint64_t>(addr), g_count, 0, 1, g_ip, g_port);
  auto ret = result.get();
  printf(
      "all result: status=%d, request=%lu, rdma_connect=%lu, "
      "read=%lu, cost=%lu\n",
      ret.status,
      ret.request_cost,
      ret.rdma_queue_cost,
      ret.read_cost,
      tensorcast::communicator::get_us() - start);

  free(addr);
  while (true) {
    std::this_thread::yield();
  }
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
