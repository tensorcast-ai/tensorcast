// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <filesystem>
#include <utility>

#include "absl/log/log.h"

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

DaemonServiceHarness::DaemonServiceHarness(
    std::shared_ptr<common::AsyncRuntime> async_runtime,
    std::unique_ptr<DaemonKernel> kernel,
    std::unique_ptr<MaterializationController> materialization_controller,
    std::unique_ptr<RegistrationController> registration_controller,
    std::unique_ptr<TransportController> transport_controller,
    std::unique_ptr<StatusController> status_controller,
    std::unique_ptr<KeyMappingController> key_mapping_controller,
    std::unique_ptr<PersistenceRpcController> persistence_rpc_controller,
    std::unique_ptr<ReplicaSessionController> replica_session_controller,
    std::unique_ptr<LeaseController> lease_controller,
    std::unique_ptr<StoreDaemonServiceImpl> service,
    std::unique_ptr<LocalHandleServer> local_handle_server)
    : async_runtime_(std::move(async_runtime)),
      kernel_(std::move(kernel)),
      materialization_controller_(std::move(materialization_controller)),
      registration_controller_(std::move(registration_controller)),
      transport_controller_(std::move(transport_controller)),
      status_controller_(std::move(status_controller)),
      key_mapping_controller_(std::move(key_mapping_controller)),
      persistence_rpc_controller_(std::move(persistence_rpc_controller)),
      replica_session_controller_(std::move(replica_session_controller)),
      lease_controller_(std::move(lease_controller)),
      service_(std::move(service)),
      local_handle_server_(std::move(local_handle_server)) {}

DaemonServiceHarness::~DaemonServiceHarness() {
  const auto deadline = absl::Now() + absl::Seconds(5);
  const auto st = stop(deadline);
  if (!st.ok()) {
    LOG(WARNING) << "DaemonServiceHarness stop failed: " << st;
  }
}

absl::StatusOr<std::unique_ptr<DaemonServiceHarness>> DaemonServiceHarness::create(
    std::shared_ptr<store::StoreEngine> engine,
    DaemonOptions options,
    std::shared_ptr<common::AsyncRuntime> async_runtime,
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client) {
  if (!engine) {
    return absl::InvalidArgumentError("DaemonServiceHarness requires StoreEngine");
  }
  if (!async_runtime) {
    async_runtime = std::make_shared<common::AsyncRuntime>();
  }
  if (options.cpu_shared_memory_enabled && options.local_handle_socket_path.empty()) {
    return absl::InvalidArgumentError(
        "cpu_shared_memory_enabled requires lifecycle.handle_leases.local_handle_socket_path to be set");
  }
  if (options.handle_lease_ttl.has_value()) {
    const auto ttl_ms = *options.handle_lease_ttl;
    if (ttl_ms.count() < 0) {
      return absl::InvalidArgumentError("handle_lease_ttl must be >= 0ms");
    }
  }
  auto storage_root_or = normalize_storage_root(options.storage_path);
  if (!storage_root_or.ok()) {
    return storage_root_or.status();
  }
  options.storage_path = std::move(*storage_root_or);

  auto kernel = std::make_unique<DaemonKernel>(engine, async_runtime, options);
  if (global_store_client) {
    if (kernel->persistence_manager()) {
      kernel->persistence_manager()->set_global_store_client(global_store_client.get());
    }
    kernel->lip_manager().set_global_store_client(global_store_client);
  }

  MaterializationController::Dep mdep{
      .engine = kernel->engine(),
      .refs = kernel->ref_tracker(),
      .sessions = kernel->sessions_service(),
      .lip = kernel->lip_bridge(),
      .lip_manager = kernel->lip_manager(),
      .devices = kernel->device_resolver(),
      .regions = kernel->region_registry(),
      .disk_imports = kernel->source_registry(),
      .shutdown_signal = kernel->shutdown_signal(),
      .async_runtime = *async_runtime,
      .identity = kernel->worker_identity_store(),
      .global_store_client = global_store_client,
      .lifecycle = &kernel->lifecycle_manager(),
      .handle_leases = kernel->handle_leases(),
      .capability_tokens = kernel->capability_tokens(),
      .cpu_shared_memory_enabled = options.cpu_shared_memory_enabled,
      .storage_path = options.storage_path,
  };
  auto materialization_controller = std::make_unique<MaterializationController>(mdep);

  RegistrationController::Dep rdep{
      .engine = kernel->engine(),
      .reg = kernel->registration_manager(),
      .lip = kernel->lip_manager(),
      .refs = kernel->ref_tracker(),
      .identity = &kernel->worker_identity_store(),
      .global_store_client = global_store_client,
      .lifecycle = &kernel->lifecycle_manager(),
      .handle_leases = kernel->handle_leases(),
      .regions = kernel->region_registry(),
  };
  auto registration_controller = std::make_unique<RegistrationController>(rdep);

  TransportController::Dep tdep{
      .engine = kernel->engine(), .locks = kernel->transport_lock_manager(), .lip = kernel->lip_manager()};
  auto transport_controller = std::make_unique<TransportController>(tdep);

  StatusController::Dep sdep{
      .engine = kernel->engine(),
      .refs = kernel->ref_tracker(),
      .shutdown_signal = kernel->shutdown_signal(),
      .identity = kernel->worker_identity_store(),
      .start_time = kernel->start_time(),
      .local_handle_socket_path = options.local_handle_socket_path,
      .cpu_shared_memory_enabled = options.cpu_shared_memory_enabled,
  };
  auto status_controller = std::make_unique<StatusController>(sdep);

  KeyMappingController::Dep kmdep{
      .engine = kernel->engine(),
      .shutdown_signal = kernel->shutdown_signal(),
  };
  auto key_mapping_controller = std::make_unique<KeyMappingController>(kmdep);

  PersistenceRpcController::Dep prdep{
      .persistence_manager = kernel->persistence_manager(),
      .shutdown_signal = kernel->shutdown_signal(),
  };
  auto persistence_rpc_controller = std::make_unique<PersistenceRpcController>(prdep);

  ReplicaSessionController::Dep rsdep{
      .sessions = kernel->sessions_service(),
      .lifecycle = kernel->lifecycle_manager(),
  };
  auto replica_session_controller = std::make_unique<ReplicaSessionController>(rsdep);

  LeaseController::Dep ldep{
      .engine = kernel->engine(),
      .lifecycle = kernel->lifecycle_manager(),
      .placement_lease_tokens = kernel->placement_lease_tokens(),
      .capability_tokens = kernel->capability_tokens(),
      .retention_registry = kernel->retention_registry(),
      .daemon_id = options.daemon_id,
      .shutdown_signal = kernel->shutdown_signal(),
  };
  auto lease_controller = std::make_unique<LeaseController>(std::move(ldep));

  StoreDaemonServiceImpl::Deps sdeps{
      .engine = kernel->engine(),
      .materialization_controller = *materialization_controller,
      .registration_controller = *registration_controller,
      .transport_controller = *transport_controller,
      .status_controller = *status_controller,
      .region_registry = kernel->region_registry(),
      .lip_manager = kernel->lip_manager(),
      .global_store_client = global_store_client,
      .lifecycle_manager = kernel->lifecycle_manager(),
      .key_mapping_controller = *key_mapping_controller,
      .persistence_rpc_controller = *persistence_rpc_controller,
      .replica_session_controller = *replica_session_controller,
      .lease_controller = *lease_controller,
      .shutdown_signal = kernel->shutdown_signal(),
      .source_registry = &kernel->source_registry(),
  };
  StoreDaemonServiceImpl::Options svc_opts{
      .allow_high_card_attrs = options.allow_high_card_attrs,
      .use_cursor_pagination = options.use_cursor_pagination,
      .storage_path = options.storage_path,
  };
  auto service = std::make_unique<StoreDaemonServiceImpl>(sdeps, svc_opts);

  std::unique_ptr<LocalHandleServer> local_handle_server;
  if (!options.local_handle_socket_path.empty()) {
    LocalHandleServer::Options lh_opts{
        .socket_path = options.local_handle_socket_path,
        .cpu_shared_memory_enabled = options.cpu_shared_memory_enabled,
    };
    auto* leases = kernel->handle_leases();
    if (leases == nullptr) {
      return absl::FailedPreconditionError("handle leases unavailable for local handle server");
    }
    local_handle_server = std::make_unique<LocalHandleServer>(lh_opts, *leases);
  }

  return std::unique_ptr<DaemonServiceHarness>(new DaemonServiceHarness(
      std::move(async_runtime),
      std::move(kernel),
      std::move(materialization_controller),
      std::move(registration_controller),
      std::move(transport_controller),
      std::move(status_controller),
      std::move(key_mapping_controller),
      std::move(persistence_rpc_controller),
      std::move(replica_session_controller),
      std::move(lease_controller),
      std::move(service),
      std::move(local_handle_server)));
}

absl::Status DaemonServiceHarness::start() {
  kernel_->start();
  if (local_handle_server_) {
    return local_handle_server_->start();
  }
  return absl::OkStatus();
}

absl::Status DaemonServiceHarness::stop(absl::Time deadline) {
  bool expected = false;
  if (!stop_called_.compare_exchange_strong(expected, true)) {
    return absl::OkStatus();
  }
  kernel_->begin_shutdown();
  if (local_handle_server_) {
    local_handle_server_->stop();
  }
  kernel_->stop();
  return kernel_->drain_async_runtime(deadline);
}

} // namespace tensorcast::daemon
