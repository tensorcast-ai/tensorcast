// Copyright (c) 2025-2026, TensorCast Team.

#include <chrono>
#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/store/store_engine.h"
#include "daemon/testing/daemon_service_harness.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"

namespace {

tensorcast::store::StoreEngineOptions make_opts_basic() {
  tensorcast::store::StoreEngineOptions opts;
  // Small pool for test environments
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  return opts;
}

} // namespace

TEST_CASE("Daemon shutdown drains AsyncRuntime with deadline", "[daemon][shutdown]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>(tensorcast::common::AsyncRuntime::Options{
      .cpu_threads = 2,
      .blocking_threads = 2,
      .thread_name_prefix = "daemon_shutdown_test",
  });
  tensorcast::daemon::DaemonOptions daemon_opts;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, runtime);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
  builder.RegisterService(&svc);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  REQUIRE(server != nullptr);
  REQUIRE(selected_port != 0);

  // Ensure drain has real in-flight work to wait for.
  absl::Notification task_started;
  absl::Notification allow_finish;
  runtime->blocking_executor()->add([&]() {
    task_started.Notify();
    allow_finish.WaitForNotification();
  });
  REQUIRE(task_started.WaitForNotificationWithTimeout(absl::Seconds(5)));

  harness->kernel().begin_shutdown();
  server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));
  server.reset();

  allow_finish.Notify();

  REQUIRE(runtime->drain(absl::Now() + absl::Seconds(5)).ok());
}
