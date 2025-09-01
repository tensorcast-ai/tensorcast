// Copyright (c) 2025, TensorCast Team.

#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "core/common/otel/init.h"
#include "core/common/otel/logging_sink.h"
#include "core/store/components/communication_manager.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "daemon/grpc_service_impl.h"
#include "daemon/worker_lifecycle_manager.h"
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
ABSL_FLAG(bool, auto_register_disk_loads, false, "Automatically register disk loads with Global Store");
ABSL_FLAG(std::string, global_store_addr, "", "Global Store address host:port");
ABSL_FLAG(bool, enable_p2p_engine, false, "Enable P2P communication engine");
ABSL_FLAG(bool, enable_rdma, false, "Enable RDMA within P2P engine");
ABSL_FLAG(bool, force_full_digest_on_load, false, "Compute full digest during load for verification");
// RFC-0012 compatibility and lifecycle flags
ABSL_FLAG(int, heartbeat_interval_ms, 5000, "Worker heartbeat interval to Global Store (ms)");
ABSL_FLAG(int, chunk_sync_interval_ms, 10000, "Chunk state sync interval (ms), 0 to disable");
ABSL_FLAG(bool, confirm_requires_disk_path, false, "Require disk_path on ConfirmReplica (compat)");
ABSL_FLAG(bool, enable_p2p_access, true, "Global toggle for P2P access; overrides per-registration enable_p2p");
ABSL_FLAG(bool, evict_on_dead_pid, false, "Evict replica when all PID refs drop due to dead PIDs");
ABSL_FLAG(std::string, verification_timeout_status, "ok", "Map verification timeout to 'ok' or 'deadline'");
ABSL_FLAG(bool, enable_periodic_eviction, false, "Enable periodic eviction loop (LRU on high GPU usage)");
ABSL_FLAG(int, eviction_check_interval_ms, 30000, "Interval for eviction checks (ms)");
ABSL_FLAG(double, gpu_memory_limit_fraction, 0.75, "GPU memory usage threshold to trigger eviction (0-1)");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  // Initialize OpenTelemetry C++ SDK from environment (optional, idempotent)
  (void)tensorcast::obs::InitFromEnv("tensorcast-store-daemon", "store-daemon");
  // Optionally install log sink that enriches logs with trace_id/span_id
  tensorcast::obs::InstallOtelLogSinkFromEnv();

  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = absl::GetFlag(FLAGS_storage_path);
  opts.p2p_port = absl::GetFlag(FLAGS_p2p_port);
  opts.memory_pool_size = absl::GetFlag(FLAGS_mem_pool_size);
  opts.chunk_size = absl::GetFlag(FLAGS_chunk_size);
  opts.num_thread = absl::GetFlag(FLAGS_io_threads);
  opts.global_store_address = absl::GetFlag(FLAGS_global_store_addr);
  opts.force_full_digest_on_load = absl::GetFlag(FLAGS_force_full_digest_on_load);

  std::shared_ptr<tensorcast::store::CommunicationManager> comm_mgr;
  if (absl::GetFlag(FLAGS_enable_p2p_engine)) {
    comm_mgr = std::make_shared<tensorcast::store::CommunicationManager>();
    auto st = comm_mgr->initialize("0.0.0.0", absl::GetFlag(FLAGS_p2p_port), absl::GetFlag(FLAGS_enable_rdma));
    if (!st.ok()) {
      LOG(WARNING) << "Failed to initialize communication engine: " << st.message();
    } else {
      opts.comm_manager = comm_mgr;
    }
  }

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(opts);

  tensorcast::daemon::StoreDaemonServiceImpl service(engine);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(absl::GetFlag(FLAGS_listen_addr), grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "tensorcast-daemon listening on " << absl::GetFlag(FLAGS_listen_addr);

  // Start worker lifecycle if configured
  std::unique_ptr<tensorcast::daemon::WorkerLifecycleManager> lifecycle;
  if (!absl::GetFlag(FLAGS_global_store_addr).empty()) {
    tensorcast::daemon::WorkerLifecycleManager::Options lopts;
    lopts.global_store_addr = absl::GetFlag(FLAGS_global_store_addr);
    lopts.listen_addr = absl::GetFlag(FLAGS_listen_addr);
    lopts.p2p_port = absl::GetFlag(FLAGS_p2p_port);
    lopts.heartbeat_interval_ms = absl::GetFlag(FLAGS_heartbeat_interval_ms);
    lopts.chunk_sync_interval_ms = absl::GetFlag(FLAGS_chunk_sync_interval_ms);
    lifecycle = std::make_unique<tensorcast::daemon::WorkerLifecycleManager>(engine, &service, lopts);
    auto st = lifecycle->start();
    if (!st.ok()) {
      LOG(WARNING) << "Worker lifecycle start failed: " << st.message();
    }
    // Metrics are exported via a unified system; no HTTP server.
  }
  // Install signal handling using sigwait in a dedicated thread.
  // Block SIGINT/SIGTERM in this thread (will be inherited by worker threads)
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGINT);
  sigaddset(&set, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &set, nullptr);

  std::thread sig_thread([&server, &service, set]() mutable {
    int sig = 0;
    if (sigwait(&set, &sig) == 0) {
      LOG(INFO) << "Received signal " << sig << ", initiating gRPC shutdown...";
      service.begin_shutdown();
      server->Shutdown();
    }
  });

  server->Wait();
  if (sig_thread.joinable()) {
    sig_thread.join();
  }
  if (lifecycle) {
    lifecycle->stop();
  }

  return 0;
}
