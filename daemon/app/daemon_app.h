// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/ha/worker_lifecycle_manager.h"
#include "daemon/service/grpc_service_impl.h"
#include "daemon/state/daemon_kernel.h"
#include "daemon/state/daemon_options.h"
#include "daemon/state/local_handle_server.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"

namespace tensorcast::daemon {

class DaemonApp {
 public:
  struct GrpcOptions {
    std::string listen_addr;
    std::shared_ptr<grpc::ServerCredentials> credentials;
    int max_concurrent_streams{0};
    std::optional<int> keepalive_time_ms;
    std::optional<int> keepalive_timeout_ms;
    std::optional<int> max_connection_idle_ms;
    std::optional<int> max_connection_age_ms;
    bool tcp_nodelay{true};
    bool so_reuseport{false};
  };

  struct Options {
    std::shared_ptr<store::StoreEngine> engine;
    std::shared_ptr<common::AsyncRuntime> async_runtime;
    DaemonOptions daemon_options;
    GrpcOptions grpc;
    std::optional<WorkerLifecycleManager::Options> worker_lifecycle;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
  };

  static absl::StatusOr<std::unique_ptr<DaemonApp>> create(Options options);

  absl::Status start();
  void wait();
  absl::Status stop(absl::Time deadline);

  StoreDaemonServiceImpl& service() const {
    return *service_;
  }

  DaemonKernel& kernel() const {
    return *kernel_;
  }

 private:
  explicit DaemonApp(Options options);

  absl::Status build_grpc_server_();

  Options options_;
  std::unique_ptr<DaemonKernel> kernel_;
  std::unique_ptr<MaterializationController> materialization_controller_;
  std::unique_ptr<RegistrationController> registration_controller_;
  std::unique_ptr<TransportController> transport_controller_;
  std::unique_ptr<StatusController> status_controller_;
  std::unique_ptr<StoreDaemonServiceImpl> service_;
  std::unique_ptr<LocalHandleServer> local_handle_server_;
  std::unique_ptr<WorkerLifecycleManager> worker_lifecycle_manager_;
  std::unique_ptr<grpc::Server> grpc_server_;
  std::atomic<bool> stop_called_{false};
};

} // namespace tensorcast::daemon
