// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/app/daemon_app.h"

#include <filesystem>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "daemon/ha/worker_lifecycle_ports.h"

namespace tensorcast::daemon {
namespace {

absl::StatusOr<std::filesystem::path> normalize_storage_root(const std::filesystem::path& storage_root) {
  if (storage_root.empty()) {
    return storage_root;
  }
  std::error_code ec;
  auto canonical = std::filesystem::weakly_canonical(storage_root, ec);
  if (ec) {
    ec.clear();
    canonical = storage_root.lexically_normal();
  }
  return canonical;
}

} // namespace

DaemonApp::DaemonApp(Options options) : options_(std::move(options)) {}

absl::StatusOr<std::unique_ptr<DaemonApp>> DaemonApp::create(Options options) {
  if (!options.engine) {
    return absl::InvalidArgumentError("DaemonApp requires StoreEngine");
  }
  if (!options.async_runtime) {
    options.async_runtime = std::make_shared<common::AsyncRuntime>();
  }
  if (options.grpc.listen_addr.empty()) {
    return absl::InvalidArgumentError("DaemonApp requires listen_addr");
  }
  if (options.daemon_options.cpu_shared_memory_enabled && options.daemon_options.local_handle_socket_path.empty()) {
    return absl::InvalidArgumentError(
        "cpu_shared_memory_enabled requires lifecycle.handle_leases.local_handle_socket_path to be set");
  }
  if (options.daemon_options.handle_lease_ttl.has_value()) {
    const auto ttl_ms = *options.daemon_options.handle_lease_ttl;
    if (ttl_ms.count() < 0) {
      return absl::InvalidArgumentError("handle_lease_ttl must be >= 0ms");
    }
  }
  auto storage_root_or = normalize_storage_root(options.daemon_options.storage_path);
  if (!storage_root_or.ok()) {
    return storage_root_or.status();
  }
  options.daemon_options.storage_path = std::move(*storage_root_or);

  auto app = std::unique_ptr<DaemonApp>(new DaemonApp(std::move(options)));
  app->kernel_ =
      std::make_unique<DaemonKernel>(app->options_.engine, app->options_.async_runtime, app->options_.daemon_options);

  if (app->options_.global_store_client && app->kernel_->persistence_manager()) {
    app->kernel_->persistence_manager()->set_global_store_client(app->options_.global_store_client.get());
  }
  if (app->options_.global_store_client) {
    app->kernel_->lip_manager().set_global_store_client(app->options_.global_store_client);
  }

  MaterializationController::Dep mdep{
      .engine = app->kernel_->engine(),
      .refs = app->kernel_->ref_tracker(),
      .sessions = app->kernel_->sessions_service(),
      .lip = app->kernel_->lip_bridge(),
      .devices = app->kernel_->device_resolver(),
      .regions = app->kernel_->region_registry(),
      .shutdown_signal = app->kernel_->shutdown_signal(),
      .async_runtime = app->kernel_->async_runtime(),
      .identity = app->kernel_->worker_identity_store(),
      .global_store_client = app->options_.global_store_client,
      .lifecycle = &app->kernel_->lifecycle_manager(),
      .handle_leases = app->kernel_->handle_leases(),
      .cpu_shared_memory_enabled = app->options_.daemon_options.cpu_shared_memory_enabled,
      .external_target_verification_enabled = app->options_.daemon_options.external_target_verification_enabled,
      .storage_path = app->options_.daemon_options.storage_path,
      .post_seal_policy = app->options_.daemon_options.post_seal_policy,
  };
  app->materialization_controller_ = std::make_unique<MaterializationController>(mdep);

  RegistrationController::Dep rdep{
      .engine = app->kernel_->engine(),
      .reg = app->kernel_->registration_manager(),
      .lip = app->kernel_->lip_manager(),
      .refs = app->kernel_->ref_tracker(),
      .identity = &app->kernel_->worker_identity_store(),
      .global_store_client = app->options_.global_store_client,
      .lifecycle = &app->kernel_->lifecycle_manager(),
      .regions = app->kernel_->region_registry(),
  };
  app->registration_controller_ = std::make_unique<RegistrationController>(rdep);

  TransportController::Dep tdep{
      .engine = app->kernel_->engine(),
      .locks = app->kernel_->transport_lock_manager(),
      .lip = app->kernel_->lip_manager()};
  app->transport_controller_ = std::make_unique<TransportController>(tdep);

  StatusController::Dep sdep{
      .engine = app->kernel_->engine(),
      .refs = app->kernel_->ref_tracker(),
      .shutdown_signal = app->kernel_->shutdown_signal(),
      .identity = app->kernel_->worker_identity_store(),
      .start_time = app->kernel_->start_time(),
      .local_handle_socket_path = app->options_.daemon_options.local_handle_socket_path,
      .cpu_shared_memory_enabled = app->options_.daemon_options.cpu_shared_memory_enabled,
  };
  app->status_controller_ = std::make_unique<StatusController>(sdep);

  StoreDaemonServiceImpl::Deps sdeps{
      .engine = app->kernel_->engine(),
      .materialization_controller = *app->materialization_controller_,
      .registration_controller = *app->registration_controller_,
      .transport_controller = *app->transport_controller_,
      .status_controller = *app->status_controller_,
      .region_registry = app->kernel_->region_registry(),
      .lip_manager = app->kernel_->lip_manager(),
      .persistence_manager = app->kernel_->persistence_manager(),
      .sessions_service = app->kernel_->sessions_service(),
      .lifecycle_manager = app->kernel_->lifecycle_manager(),
      .placement_lease_tokens = app->kernel_->placement_lease_tokens(),
      .capability_tokens = app->kernel_->capability_tokens(),
      .retention_registry = app->kernel_->retention_registry(),
      .daemon_id = app->options_.daemon_options.daemon_id,
      .shutdown_signal = app->kernel_->shutdown_signal(),
  };
  StoreDaemonServiceImpl::Options svc_opts{
      .allow_high_card_attrs = app->options_.daemon_options.allow_high_card_attrs,
      .use_cursor_pagination = app->options_.daemon_options.use_cursor_pagination,
      .storage_path = app->options_.daemon_options.storage_path,
  };
  app->service_ = std::make_unique<StoreDaemonServiceImpl>(sdeps, svc_opts);

  if (!app->options_.daemon_options.local_handle_socket_path.empty()) {
    LocalHandleServer::Options lh_opts{
        .socket_path = app->options_.daemon_options.local_handle_socket_path,
        .cpu_shared_memory_enabled = app->options_.daemon_options.cpu_shared_memory_enabled,
    };
    auto* leases = app->kernel_->handle_leases();
    if (leases == nullptr) {
      return absl::FailedPreconditionError("handle leases unavailable for local handle server");
    }
    app->local_handle_server_ = std::make_unique<LocalHandleServer>(lh_opts, *leases);
  }

  if (app->options_.worker_lifecycle.has_value()) {
    WorkerLifecyclePorts ports{
        .identity_store = app->kernel_->worker_identity_store(),
        .retire_gates = app->kernel_->retire_gates(),
        .shutdown_signal = app->kernel_->shutdown_signal(),
        .async_runtime = app->kernel_->async_runtime(),
    };
    app->worker_lifecycle_manager_ = std::make_unique<WorkerLifecycleManager>(
        std::move(app->options_.engine), ports, *app->options_.worker_lifecycle);
  }

  return app;
}

absl::Status DaemonApp::start() {
  kernel_->start();

  if (local_handle_server_) {
    const absl::Status st = local_handle_server_->start();
    if (!st.ok()) {
      return st;
    }
  }

  absl::Status grpc_st = build_grpc_server_();
  if (!grpc_st.ok()) {
    return grpc_st;
  }

  if (worker_lifecycle_manager_) {
    auto st = worker_lifecycle_manager_->start();
    if (!st.ok()) {
      LOG(WARNING) << "Worker lifecycle start failed: " << st.message();
    }
  }

  return absl::OkStatus();
}

void DaemonApp::wait() {
  if (grpc_server_) {
    grpc_server_->Wait();
  }
}

absl::Status DaemonApp::stop(absl::Time deadline) {
  bool expected = false;
  if (!stop_called_.compare_exchange_strong(expected, true)) {
    return absl::OkStatus();
  }

  kernel_->begin_shutdown();

  if (worker_lifecycle_manager_) {
    worker_lifecycle_manager_->stop();
  }

  if (grpc_server_) {
    auto remaining = deadline - absl::Now();
    if (remaining < absl::ZeroDuration()) {
      remaining = absl::ZeroDuration();
    }
    const auto shutdown_deadline =
        std::chrono::system_clock::now() + std::chrono::milliseconds(absl::ToInt64Milliseconds(remaining));
    grpc_server_->Shutdown(shutdown_deadline);
  }

  if (local_handle_server_) {
    local_handle_server_->stop();
  }

  kernel_->stop();

  return kernel_->drain_async_runtime(deadline);
}

absl::Status DaemonApp::build_grpc_server_() {
  if (grpc_server_) {
    return absl::OkStatus();
  }

  grpc::ServerBuilder builder;
  const auto creds = options_.grpc.credentials ? options_.grpc.credentials : grpc::InsecureServerCredentials();
  builder.AddListeningPort(options_.grpc.listen_addr, creds);

  if (options_.grpc.max_concurrent_streams > 0) {
    builder.AddChannelArgument("grpc.max_concurrent_streams", options_.grpc.max_concurrent_streams);
  }
  if (options_.grpc.keepalive_time_ms.has_value()) {
    builder.AddChannelArgument("grpc.keepalive_time_ms", *options_.grpc.keepalive_time_ms);
  }
  if (options_.grpc.keepalive_timeout_ms.has_value()) {
    builder.AddChannelArgument("grpc.keepalive_timeout_ms", *options_.grpc.keepalive_timeout_ms);
  }
  if (options_.grpc.max_connection_idle_ms.has_value()) {
    builder.AddChannelArgument("grpc.max_connection_idle_ms", *options_.grpc.max_connection_idle_ms);
  }
  if (options_.grpc.max_connection_age_ms.has_value()) {
    builder.AddChannelArgument("grpc.max_connection_age_ms", *options_.grpc.max_connection_age_ms);
  }
  builder.AddChannelArgument("grpc.tcp_nodelay", options_.grpc.tcp_nodelay ? 1 : 0);
  builder.AddChannelArgument("grpc.so_reuseport", options_.grpc.so_reuseport ? 1 : 0);

  builder.RegisterService(service_.get());
  grpc_server_ = builder.BuildAndStart();
  if (!grpc_server_) {
    return absl::InternalError("Failed to start gRPC server");
  }

  LOG(INFO) << "tensorcast-daemon listening on " << options_.grpc.listen_addr;
  return absl::OkStatus();
}

} // namespace tensorcast::daemon
