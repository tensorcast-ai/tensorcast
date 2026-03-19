// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/common/capability_token.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "daemon/state/artifact_source_registry.h"
#include "daemon/state/background_scheduler.h"
#include "daemon/state/daemon_options.h"
#include "daemon/state/derived_view_export_manager.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/handle_lease_registry.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_bridge.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/persistence_manager.h"
#include "daemon/state/placement_lease_tokens.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/registration_manager.h"
#include "daemon/state/replica_session_manager.h"
#include "daemon/state/retention_registry.h"
#include "daemon/state/retire_gates.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/sessions_service.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/transport_lock_manager.h"
#include "daemon/state/verification_tracker.h"
#include "daemon/state/worker_identity_store.h"

namespace tensorcast::daemon {

class DaemonKernel {
 public:
  DaemonKernel(
      std::shared_ptr<store::StoreEngine> engine,
      std::shared_ptr<common::AsyncRuntime> async_runtime,
      DaemonOptions options);

  ~DaemonKernel();

  DaemonKernel(const DaemonKernel&) = delete;
  DaemonKernel& operator=(const DaemonKernel&) = delete;

  void start();
  void stop();

  void begin_shutdown();

  [[nodiscard]] absl::Status drain_async_runtime(absl::Time deadline) const;

  [[nodiscard]] store::StoreEngine& engine() const {
    return *engine_;
  }

  [[nodiscard]] common::AsyncRuntime& async_runtime() const {
    return *async_runtime_;
  }

  [[nodiscard]] std::shared_ptr<common::AsyncRuntime> async_runtime_shared() const {
    return async_runtime_;
  }

  [[nodiscard]] const DaemonOptions& options() const {
    return options_;
  }

  [[nodiscard]] std::chrono::steady_clock::time_point start_time() const {
    return start_time_;
  }

  [[nodiscard]] SessionsService& sessions_service() const {
    return *sessions_svc_;
  }

  [[nodiscard]] SessionLifecycleManager& lifecycle_manager() const {
    return *lifecycle_mgr_;
  }

  [[nodiscard]] IpcRegionRegistry& region_registry() const {
    return *region_registry_;
  }

  [[nodiscard]] LipManager& lip_manager() const {
    return *lip_mgr_;
  }

  [[nodiscard]] LipBridge& lip_bridge() const {
    return *lip_bridge_;
  }

  [[nodiscard]] DerivedViewExportManager& derived_view_export_manager() const {
    return *derived_view_export_mgr_;
  }

  [[nodiscard]] ArtifactSourceRegistry& source_registry() {
    return source_registry_;
  }

  [[nodiscard]] RegistrationManager& registration_manager() const {
    return *reg_mgr_;
  }

  [[nodiscard]] VerificationTracker& verification_tracker() const {
    return *verif_tracker_;
  }

  [[nodiscard]] HandleLeaseRegistry* handle_leases() const {
    return handle_leases_.get();
  }

  [[nodiscard]] PersistenceManager* persistence_manager() const {
    return persistence_mgr_.get();
  }

  [[nodiscard]] RefTracker& ref_tracker() {
    return refs_;
  }

  [[nodiscard]] TransportLockManager& transport_lock_manager() {
    return locks_;
  }

  [[nodiscard]] DeviceResolver& device_resolver() {
    return devices_;
  }

  [[nodiscard]] WorkerIdentityStore& worker_identity_store() const {
    return *identity_store_;
  }

  [[nodiscard]] PlacementLeaseTokens& placement_lease_tokens() const {
    return *placement_lease_tokens_;
  }

  [[nodiscard]] common::CapabilityTokenManager* capability_tokens() const {
    return capability_tokens_.get();
  }

  [[nodiscard]] RetentionRegistry* retention_registry() const {
    return retention_registry_.get();
  }

  [[nodiscard]] ShutdownSignal& shutdown_signal() {
    return shutdown_signal_;
  }

  [[nodiscard]] RetireGates& retire_gates() const {
    return *retire_gates_;
  }

  void sweep_session_lifecycle_for_test() {
    lifecycle_mgr_->sweep_once();
  }

 private:
  void configure_scheduler_tasks_();

  std::shared_ptr<store::StoreEngine> engine_;
  std::shared_ptr<common::AsyncRuntime> async_runtime_;
  DaemonOptions options_;

  std::chrono::time_point<std::chrono::steady_clock> start_time_{std::chrono::steady_clock::now()};

  ReplicaSessionManager sessions_;
  TransportLockManager locks_;
  RefTracker refs_;
  std::unique_ptr<IpcRegionRegistry> region_registry_;
  std::unique_ptr<LipManager> lip_mgr_;
  std::unique_ptr<DerivedViewExportManager> derived_view_export_mgr_;

  std::unique_ptr<BackgroundScheduler> scheduler_;
  std::shared_ptr<SessionLifecycleManager> lifecycle_mgr_;
  std::unique_ptr<PidMonitor> pid_monitor_;

  std::unique_ptr<VerificationTracker> verif_tracker_;
  std::unique_ptr<RegistrationManager> reg_mgr_;
  std::unique_ptr<HandleLeaseRegistry> handle_leases_;
  std::unique_ptr<PlacementLeaseTokens> placement_lease_tokens_;
  std::unique_ptr<common::CapabilityTokenManager> capability_tokens_;
  std::unique_ptr<RetentionRegistry> retention_registry_;

  std::unique_ptr<SessionsService> sessions_svc_;
  std::unique_ptr<LipBridge> lip_bridge_;
  std::unique_ptr<PersistenceManager> persistence_mgr_;
  ArtifactSourceRegistry source_registry_;

  DeviceResolver devices_;
  ShutdownSignal shutdown_signal_;
  std::unique_ptr<WorkerIdentityStore> identity_store_;
  std::unique_ptr<RetireGates> retire_gates_;

  std::atomic<bool> started_{false};
};

} // namespace tensorcast::daemon
