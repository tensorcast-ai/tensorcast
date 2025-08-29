// Copyright (c) 2025, TensorCast Team.

#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "daemon/grpc_service_impl.h"
#include "daemon/metrics_exporter.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"

#include <pthread.h>
#include <csignal>

ABSL_FLAG(std::string, listen_addr, "0.0.0.0:50051", "gRPC listen address");
ABSL_FLAG(std::string, storage_path, "", "Optional storage path for local artifacts");
ABSL_FLAG(uint16_t, p2p_port, 9090, "P2P communication port for engine");
ABSL_FLAG(size_t, mem_pool_size, 8ULL << 30, "Pinned memory pool size (bytes)");
ABSL_FLAG(size_t, chunk_size, 128ULL << 20, "Streaming chunk size (bytes)");
ABSL_FLAG(int, io_threads, 10, "I/O worker threads for engine");
ABSL_FLAG(uint16_t, metrics_port, 9095, "Metrics HTTP port");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = absl::GetFlag(FLAGS_storage_path);
  opts.p2p_port = absl::GetFlag(FLAGS_p2p_port);
  opts.memory_pool_size = absl::GetFlag(FLAGS_mem_pool_size);
  opts.chunk_size = absl::GetFlag(FLAGS_chunk_size);
  opts.num_thread = absl::GetFlag(FLAGS_io_threads);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(opts);

  tensorcast::daemon::StoreDaemonServiceImpl service(engine);
  tensorcast::daemon::MetricsExporter metrics(engine, absl::GetFlag(FLAGS_metrics_port));
  metrics.start();

  grpc::ServerBuilder builder;
  builder.AddListeningPort(absl::GetFlag(FLAGS_listen_addr), grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "tensorcast-daemon listening on " << absl::GetFlag(FLAGS_listen_addr);
  // Install signal handling using sigwait in a dedicated thread.
  // Block SIGINT/SIGTERM in this thread (will be inherited by worker threads)
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGINT);
  sigaddset(&set, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &set, nullptr);

  std::thread sig_thread([&server, set]() mutable {
    int sig = 0;
    if (sigwait(&set, &sig) == 0) {
      LOG(INFO) << "Received signal " << sig << ", initiating gRPC shutdown...";
      server->Shutdown();
    }
  });

  server->Wait();
  if (sig_thread.joinable())
    sig_thread.join();
  metrics.stop();
  return 0;
}
