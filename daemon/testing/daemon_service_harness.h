// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <atomic>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/app/startup_coordinator.h"
#include "daemon/service/controllers/byte_artifact_controller.h"
#include "daemon/service/controllers/external_target_access_service.h"
#include "daemon/service/controllers/key_mapping_controller.h"
#include "daemon/service/controllers/lease_controller.h"
#include "daemon/service/controllers/materialization_controller.h"
#include "daemon/service/controllers/persistence_rpc_controller.h"
#include "daemon/service/controllers/registration_controller.h"
#include "daemon/service/controllers/replica_session_controller.h"
#include "daemon/service/controllers/status_controller.h"
#include "daemon/service/controllers/transport_controller.h"
#include "daemon/service/grpc_service_impl.h"
#include "daemon/state/daemon_kernel.h"
#include "daemon/state/daemon_options.h"
#include "daemon/state/local_handle_server.h"

namespace tensorcast::daemon {

class DaemonServiceHarness {
 public:
  static absl::StatusOr<std::unique_ptr<DaemonServiceHarness>> create(
      std::shared_ptr<store::StoreEngine> engine,
      DaemonOptions options,
      std::shared_ptr<common::AsyncRuntime> async_runtime = nullptr,
      std::shared_ptr<store::components::IGlobalStoreClient> global_store_client = nullptr,
      std::shared_ptr<StartupCoordinator> startup_coordinator = nullptr);

  ~DaemonServiceHarness();

  DaemonServiceHarness(const DaemonServiceHarness&) = delete;
  DaemonServiceHarness& operator=(const DaemonServiceHarness&) = delete;

  absl::Status start();
  absl::Status stop(absl::Time deadline);

  StoreDaemonServiceImpl& service() const {
    return *service_;
  }

  DaemonKernel& kernel() const {
    return *kernel_;
  }

  MaterializationController& materialization_controller() const {
    return *materialization_controller_;
  }

  std::shared_ptr<common::AsyncRuntime> async_runtime_shared() const {
    return async_runtime_;
  }

 private:
  DaemonServiceHarness(
      std::shared_ptr<common::AsyncRuntime> async_runtime,
      std::unique_ptr<DaemonKernel> kernel,
      std::unique_ptr<ExternalTargetAccessService> external_target_access_service,
      std::unique_ptr<ByteArtifactController> byte_artifact_controller,
      std::unique_ptr<MaterializationController> materialization_controller,
      std::unique_ptr<RegistrationController> registration_controller,
      std::unique_ptr<TransportController> transport_controller,
      std::unique_ptr<StatusController> status_controller,
      std::unique_ptr<KeyMappingController> key_mapping_controller,
      std::unique_ptr<PersistenceRpcController> persistence_rpc_controller,
      std::unique_ptr<ReplicaSessionController> replica_session_controller,
      std::unique_ptr<LeaseController> lease_controller,
      std::unique_ptr<StoreDaemonServiceImpl> service,
      std::unique_ptr<LocalHandleServer> local_handle_server);

  std::shared_ptr<common::AsyncRuntime> async_runtime_;
  std::unique_ptr<DaemonKernel> kernel_;
  std::unique_ptr<ExternalTargetAccessService> external_target_access_service_;
  std::unique_ptr<ByteArtifactController> byte_artifact_controller_;
  std::unique_ptr<MaterializationController> materialization_controller_;
  std::unique_ptr<RegistrationController> registration_controller_;
  std::unique_ptr<TransportController> transport_controller_;
  std::unique_ptr<StatusController> status_controller_;
  std::unique_ptr<KeyMappingController> key_mapping_controller_;
  std::unique_ptr<PersistenceRpcController> persistence_rpc_controller_;
  std::unique_ptr<ReplicaSessionController> replica_session_controller_;
  std::unique_ptr<LeaseController> lease_controller_;
  std::unique_ptr<StoreDaemonServiceImpl> service_;
  std::unique_ptr<LocalHandleServer> local_handle_server_;
  std::atomic<bool> stop_called_{false};
};

} // namespace tensorcast::daemon
