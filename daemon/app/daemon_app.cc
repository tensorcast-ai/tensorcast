// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/app/daemon_app.h"

#include <algorithm>
#include <filesystem>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "core/communicator/transport/request.h"
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

absl::Status fsync_directory(const std::filesystem::path& dir_path) {
  int dir_fd = ::open(dir_path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (dir_fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("open directory failed: ", dir_path.string()));
  }
  if (::fsync(dir_fd) != 0) {
    const int err = errno;
    ::close(dir_fd);
    return absl::ErrnoToStatus(err, absl::StrCat("fsync directory failed: ", dir_path.string()));
  }
  if (::close(dir_fd) != 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("close directory failed: ", dir_path.string()));
  }
  return absl::OkStatus();
}

absl::Status ensure_import_root_ready(const std::filesystem::path& import_root) {
  std::error_code ec;
  std::filesystem::create_directories(import_root, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("create import root failed: ", import_root.string()));
  }
  std::filesystem::permissions(
      import_root, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("set import root permissions failed: ", import_root.string()));
  }

  const auto probe_tmp = import_root / ".import_root_probe.tmp";
  const auto probe_path = import_root / ".import_root_probe";
  constexpr std::string_view kProbePayload = "tensorcast-import-root-probe";
  int probe_fd = ::open(probe_tmp.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0600);
  if (probe_fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("open probe file failed: ", probe_tmp.string()));
  }
  if (::write(probe_fd, kProbePayload.data(), kProbePayload.size()) < 0) {
    const int err = errno;
    ::close(probe_fd);
    return absl::ErrnoToStatus(err, absl::StrCat("write probe file failed: ", probe_tmp.string()));
  }
  if (::fsync(probe_fd) != 0) {
    const int err = errno;
    ::close(probe_fd);
    return absl::ErrnoToStatus(err, absl::StrCat("fsync probe file failed: ", probe_tmp.string()));
  }
  if (::close(probe_fd) != 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("close probe file failed: ", probe_tmp.string()));
  }
  if (::rename(probe_tmp.c_str(), probe_path.c_str()) != 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("rename probe file failed: ", probe_path.string()));
  }
  auto dir_sync_status = fsync_directory(import_root);
  if (!dir_sync_status.ok()) {
    return dir_sync_status;
  }
  std::filesystem::remove(probe_path, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("remove probe file failed: ", probe_path.string()));
  }
  dir_sync_status = fsync_directory(import_root);
  if (!dir_sync_status.ok()) {
    return dir_sync_status;
  }

  const auto db_path = import_root / "artifact_source_registry.db";
  int db_fd = ::open(db_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
  if (db_fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("open registry db file failed: ", db_path.string()));
  }
  if (::fsync(db_fd) != 0) {
    const int err = errno;
    ::close(db_fd);
    return absl::ErrnoToStatus(err, absl::StrCat("fsync registry db file failed: ", db_path.string()));
  }
  if (::close(db_fd) != 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("close registry db file failed: ", db_path.string()));
  }
  return fsync_directory(import_root);
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
  if (!options.startup_coordinator) {
    options.startup_coordinator = std::make_shared<StartupCoordinator>();
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
  if (options.daemon_options.import_root.empty()) {
    options.daemon_options.import_root = options.daemon_options.storage_path / ".tensorcast_import";
  }
  auto import_root_or = normalize_storage_root(options.daemon_options.import_root);
  if (!import_root_or.ok()) {
    return import_root_or.status();
  }
  options.daemon_options.import_root = std::move(*import_root_or);
  auto import_root_status = ensure_import_root_ready(options.daemon_options.import_root);
  if (!import_root_status.ok()) {
    return absl::FailedPreconditionError(absl::StrCat("IMPORT_ROOT_UNAVAILABLE: ", import_root_status.message()));
  }
  if (options.daemon_options.cpu_shared_memory_enabled && options.daemon_options.local_handle_socket_path.empty()) {
    options.daemon_options.local_handle_socket_path =
        (options.daemon_options.import_root / "local_handle.sock").string();
    LOG(INFO) << "Auto-selected lifecycle.handle_leases.local_handle_socket_path="
              << options.daemon_options.local_handle_socket_path;
  }
  LOG(INFO) << "Import metadata root initialized at " << options.daemon_options.import_root.string();

  auto app = std::unique_ptr<DaemonApp>(new DaemonApp(std::move(options)));
  communicator::transport::ReadRequest::set_rdma_profile_enabled_for_process(true);
  LOG(INFO) << "Enabled communicator RDMA read profiling for daemon process";
  app->kernel_ = std::make_unique<DaemonKernel>(
      app->options_.engine,
      app->options_.async_runtime,
      app->options_.daemon_options,
      app->options_.global_store_client);
  std::shared_ptr<store::components::CommunicationManager> comm_manager =
      app->kernel_->engine().get_shared_comm_manager();

  app->external_target_access_service_ = std::make_unique<ExternalTargetAccessService>(ExternalTargetAccessService::Dep{
      .devices = app->kernel_->device_resolver(),
      .regions = app->kernel_->region_registry(),
      .comm_manager = comm_manager.get(),
  });
  app->byte_artifact_controller_ = std::make_unique<ByteArtifactController>(
      ByteArtifactController::Dep{
          .body_store = app->kernel_->byte_artifact_body_store(),
          .route_resolver = app->kernel_->byte_artifact_route_resolver(),
          .payload_transport_broker = app->kernel_->payload_transport_broker(),
          .worker_directory_cache = app->kernel_->worker_directory_cache(),
          .external_target_access_service = *app->external_target_access_service_,
          .identity_store = app->kernel_->worker_identity_store(),
          .engine = app->kernel_->engine(),
          .async_runtime = app->kernel_->async_runtime(),
          .persistence_manager = app->kernel_->persistence_manager(),
          .global_store_client = app->options_.global_store_client,
          .inter_daemon_channel_credentials = app->kernel_->inter_daemon_channel_credentials(),
      },
      ByteArtifactController::Options{
          .routing =
              {
                  .shard_count = app->options_.daemon_options.byte_artifact_routing.shard_count,
                  .inline_payload_threshold_bytes =
                      app->options_.daemon_options.byte_artifact_routing.inline_payload_threshold_bytes,
                  .route_staleness_budget = app->options_.daemon_options.byte_artifact_routing.route_staleness_budget,
                  .lease_ttl = app->options_.daemon_options.byte_artifact_routing.lease_ttl,
                  .keepalive_interval = app->options_.daemon_options.byte_artifact_routing.keepalive_interval,
                  .worker_directory_staleness_budget =
                      app->options_.daemon_options.byte_artifact_routing.worker_directory_staleness_budget,
                  .routing_epoch = app->options_.daemon_options.byte_artifact_routing.routing_epoch,
                  .shard_home_eligible = app->options_.daemon_options.byte_artifact_routing.shard_home_eligible,
              },
          .publish_prereg =
              {
                  .enabled = app->options_.daemon_options.byte_artifact_routing.payload_transport.source_publish_prereg
                                 .enabled,
                  .ttl = app->options_.daemon_options.byte_artifact_routing.payload_transport.source_publish_prereg.ttl,
                  .max_live_entries = app->options_.daemon_options.byte_artifact_routing.payload_transport
                                          .source_publish_prereg.max_live_entries,
                  .max_live_bytes = app->options_.daemon_options.byte_artifact_routing.payload_transport
                                        .source_publish_prereg.max_live_bytes,
              },
          .gateway_ingress_enabled = app->options_.daemon_options.gateway_ingress_enabled,
          .batch_get_apply_threads =
              static_cast<std::uint32_t>(std::max(1, app->options_.engine->options().num_thread)),
      });

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
      .lip_manager = app->kernel_->lip_manager(),
      .registration_manager = app->kernel_->registration_manager(),
      .devices = app->kernel_->device_resolver(),
      .regions = app->kernel_->region_registry(),
      .disk_imports = app->kernel_->source_registry(),
      .binding_registry = app->kernel_->binding_registry(),
      .shutdown_signal = app->kernel_->shutdown_signal(),
      .async_runtime = app->kernel_->async_runtime(),
      .identity = app->kernel_->worker_identity_store(),
      .external_target_access_service = *app->external_target_access_service_,
      .global_store_client = app->options_.global_store_client,
      .max_concurrency = app->options_.daemon_options.max_concurrency,
      .lifecycle = &app->kernel_->lifecycle_manager(),
      .lifecycle_kernel = &app->kernel_->lifecycle_kernel(),
      .handle_leases = app->kernel_->handle_leases(),
      .capability_tokens = app->kernel_->capability_tokens(),
      .cpu_shared_memory_enabled = app->options_.daemon_options.cpu_shared_memory_enabled,
      .external_target_verification_enabled = app->options_.daemon_options.external_target_verification_enabled,
      .storage_path = app->options_.daemon_options.storage_path,
      .public_disk_source_policy = app->options_.daemon_options.public_disk_source_policy,
      .post_seal_policy = app->options_.daemon_options.post_seal_policy,
      .await_state_sync_barrier = [app_ptr = app.get()]() { return app_ptr->await_worker_state_sync_barrier(); },
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
      .handle_leases = app->kernel_->handle_leases(),
      .regions = app->kernel_->region_registry(),
      .max_concurrency = app->options_.daemon_options.max_concurrency,
      .await_state_sync_barrier = [app_ptr = app.get()]() { return app_ptr->await_worker_state_sync_barrier(); },
  };
  app->registration_controller_ = std::make_unique<RegistrationController>(rdep);

  TransportController::Dep tdep{
      .engine = app->kernel_->engine(),
      .locks = app->kernel_->transport_lock_manager(),
      .lip = app->kernel_->lip_manager(),
      .payload_transport_broker = &app->kernel_->payload_transport_broker(),
  };
  app->transport_controller_ = std::make_unique<TransportController>(tdep);

  StatusController::Dep sdep{
      .engine = app->kernel_->engine(),
      .refs = app->kernel_->ref_tracker(),
      .shutdown_signal = app->kernel_->shutdown_signal(),
      .identity = app->kernel_->worker_identity_store(),
      .start_time = app->kernel_->start_time(),
      .local_handle_socket_path = app->options_.daemon_options.local_handle_socket_path,
      .cpu_shared_memory_enabled = app->options_.daemon_options.cpu_shared_memory_enabled,
      .batch_transport_protocol_version =
          app->options_.daemon_options.byte_artifact_routing.payload_transport.batch_transport_protocol_version,
      .batch_payload_grpc_chunk_ref_enabled =
          app->options_.daemon_options.byte_artifact_routing.payload_transport.batch_transport_protocol_version > 0,
      .batch_payload_communicator_source_enabled =
          app->options_.daemon_options.byte_artifact_routing.payload_transport.batch_transport_protocol_version >= 2 &&
          app->options_.daemon_options.byte_artifact_routing.payload_transport.communicator_source_enabled,
      .batch_payload_host_memory_export_enabled =
          app->options_.daemon_options.byte_artifact_routing.payload_transport.batch_transport_protocol_version >= 2 &&
          app->options_.daemon_options.byte_artifact_routing.payload_transport.host_memory_export_enabled,
      .batch_payload_segmented_communicator_export_enabled =
          app->kernel_->payload_transport_broker().batch_transport_segmented_communicator_export_enabled(),
      .max_batch_payload_bytes =
          app->options_.daemon_options.byte_artifact_routing.payload_transport.max_batch_payload_bytes,
      .startup_coordinator = app->options_.startup_coordinator,
      .worker_directory_cache = app->kernel_->worker_directory_cache(),
      .instance_execution_directory_cache = app->kernel_->instance_execution_directory_cache(),
      .global_store_client = app->options_.global_store_client,
      .directory_staleness_budget = absl::Milliseconds(
          std::max<int64_t>(
              1,
              static_cast<int64_t>(
                  app->options_.daemon_options.byte_artifact_routing.worker_directory_staleness_budget.count()))),
  };
  app->status_controller_ = std::make_unique<StatusController>(sdep);

  KeyMappingController::Dep kmdep{
      .engine = app->kernel_->engine(),
      .shutdown_signal = app->kernel_->shutdown_signal(),
      .source_registry = &app->kernel_->source_registry(),
      .global_store_client = app->options_.global_store_client,
  };
  app->key_mapping_controller_ = std::make_unique<KeyMappingController>(kmdep);

  PersistenceRpcController::Dep prdep{
      .persistence_manager = app->kernel_->persistence_manager(),
      .shutdown_signal = app->kernel_->shutdown_signal(),
  };
  app->persistence_rpc_controller_ = std::make_unique<PersistenceRpcController>(prdep);

  ReplicaSessionController::Dep rsdep{
      .sessions = app->kernel_->sessions_service(),
      .lifecycle = app->kernel_->lifecycle_manager(),
  };
  app->replica_session_controller_ = std::make_unique<ReplicaSessionController>(rsdep);

  LeaseController::Dep ldep{
      .engine = app->kernel_->engine(),
      .lifecycle = app->kernel_->lifecycle_manager(),
      .lifecycle_kernel = app->kernel_->lifecycle_kernel(),
      .placement_lease_tokens = app->kernel_->placement_lease_tokens(),
      .capability_tokens = app->kernel_->capability_tokens(),
      .retention_registry = app->kernel_->retention_registry(),
      .daemon_id = app->options_.daemon_options.daemon_id,
      .shutdown_signal = app->kernel_->shutdown_signal(),
  };
  app->lease_controller_ = std::make_unique<LeaseController>(std::move(ldep));

  StoreDaemonServiceImpl::Deps sdeps{
      .engine = app->kernel_->engine(),
      .materialization_controller = *app->materialization_controller_,
      .byte_artifact_controller = *app->byte_artifact_controller_,
      .registration_controller = *app->registration_controller_,
      .transport_controller = *app->transport_controller_,
      .status_controller = *app->status_controller_,
      .identity_store = app->kernel_->worker_identity_store(),
      .instance_execution_directory_cache = app->kernel_->instance_execution_directory_cache(),
      .region_registry = app->kernel_->region_registry(),
      .lip_manager = app->kernel_->lip_manager(),
      .global_store_client = app->options_.global_store_client,
      .lifecycle_manager = app->kernel_->lifecycle_manager(),
      .key_mapping_controller = *app->key_mapping_controller_,
      .persistence_rpc_controller = *app->persistence_rpc_controller_,
      .replica_session_controller = *app->replica_session_controller_,
      .lease_controller = *app->lease_controller_,
      .shutdown_signal = app->kernel_->shutdown_signal(),
      .startup_coordinator = app->options_.startup_coordinator,
      .source_registry = &app->kernel_->source_registry(),
  };
  StoreDaemonServiceImpl::Options svc_opts{
      .allow_high_card_attrs = app->options_.daemon_options.allow_high_card_attrs,
      .use_cursor_pagination = app->options_.daemon_options.use_cursor_pagination,
      .gateway_ingress_enabled = app->options_.daemon_options.gateway_ingress_enabled,
      .storage_path = app->options_.daemon_options.storage_path,
      .directory_staleness_budget = absl::Milliseconds(
          std::max<int64_t>(
              1,
              static_cast<int64_t>(
                  app->options_.daemon_options.byte_artifact_routing.worker_directory_staleness_budget.count()))),
  };
  app->service_ = std::make_unique<StoreDaemonServiceImpl>(sdeps, svc_opts);

  if (!app->options_.daemon_options.local_handle_socket_path.empty()) {
    LocalHandleServer::Options lh_opts{
        .socket_path = app->options_.daemon_options.local_handle_socket_path,
        .cpu_shared_memory_enabled = app->options_.daemon_options.cpu_shared_memory_enabled,
    };
    app->local_handle_server_ =
        std::make_unique<LocalHandleServer>(lh_opts, app->kernel_->region_registry(), app->kernel_->handle_leases());
  }

  if (app->options_.worker_lifecycle.has_value()) {
    WorkerLifecyclePorts ports{
        .identity_store = app->kernel_->worker_identity_store(),
        .lip_manager = app->kernel_->lip_manager(),
        .worker_directory_cache = app->kernel_->worker_directory_cache(),
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

  if (options_.startup_coordinator) {
    if (worker_lifecycle_manager_ && options_.deferred_startup_work) {
      options_.startup_coordinator->begin_startup(
          "daemon startup still in progress: registering worker lifecycle and prewarming GPU caches");
    } else if (worker_lifecycle_manager_) {
      options_.startup_coordinator->begin_startup("daemon startup still in progress: registering worker lifecycle");
    } else if (options_.deferred_startup_work) {
      options_.startup_coordinator->begin_startup(
          "daemon startup still in progress: registering pinned pools and prewarming GPU caches");
    }
  }

  if (worker_lifecycle_manager_) {
    auto st = worker_lifecycle_manager_->start();
    if (!st.ok()) {
      if (options_.startup_coordinator) {
        options_.startup_coordinator->mark_failed(st);
      }
      return st;
    }
  }

  if (options_.deferred_startup_work) {
    auto deferred_work = options_.deferred_startup_work;
    auto startup_coordinator = options_.startup_coordinator;
    auto startup_failure_is_fatal = startup_failure_is_fatal_;
    std::thread([deferred_work = std::move(deferred_work),
                 startup_coordinator = std::move(startup_coordinator),
                 startup_failure_is_fatal]() mutable {
      const absl::Status startup_status = deferred_work();
      if (startup_status.ok()) {
        startup_coordinator->mark_ready();
        LOG(INFO) << "Deferred daemon startup tasks completed";
        return;
      }
      startup_coordinator->mark_failed(startup_status);
      if (!startup_failure_is_fatal->load(std::memory_order_acquire)) {
        LOG(WARNING) << "Deferred daemon startup failed after shutdown began: " << startup_status;
        return;
      }
      LOG(FATAL) << "Deferred daemon startup failed: " << startup_status;
    }).detach();
  } else if (options_.startup_coordinator) {
    options_.startup_coordinator->mark_ready();
  }

  return absl::OkStatus();
}

absl::Status DaemonApp::await_worker_state_sync_barrier() const {
  if (worker_lifecycle_manager_ == nullptr) {
    return absl::OkStatus();
  }
  return worker_lifecycle_manager_->wait_for_state_sync_barrier();
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
  startup_failure_is_fatal_->store(false, std::memory_order_release);

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
  if (options_.grpc.sync_server_threads > 0) {
    const int max_pollers = std::min(4, options_.grpc.sync_server_threads);
    const int min_pollers = 1;
    const int num_cqs = std::min(2, max_pollers);
    builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::NUM_CQS, num_cqs);
    builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, min_pollers);
    builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, max_pollers);
    LOG(INFO) << "Configuring sync gRPC server"
              << " num_cqs=" << num_cqs << " min_pollers=" << min_pollers << " max_pollers=" << max_pollers;
  }

  if (options_.grpc.max_concurrent_streams > 0) {
    builder.AddChannelArgument("grpc.max_concurrent_streams", options_.grpc.max_concurrent_streams);
  }
  if (options_.grpc.max_send_message_length > 0) {
    builder.AddChannelArgument("grpc.max_send_message_length", options_.grpc.max_send_message_length);
  }
  if (options_.grpc.max_receive_message_length > 0) {
    builder.AddChannelArgument("grpc.max_receive_message_length", options_.grpc.max_receive_message_length);
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
