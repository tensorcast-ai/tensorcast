// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/store/store_engine.h"
#include "daemon/background_scheduler.h"
#include "daemon/device_resolver.h"
#include "daemon/lip_bridge.h"
#include "daemon/lip_manager.h"
#include "daemon/ref_tracker.h"
#include "daemon/registration_manager.h"
#include "daemon/replica_session_manager.h"
#include "daemon/service/controllers/materialization_controller.h"
#include "daemon/service/controllers/registration_controller.h"
#include "daemon/service/controllers/status_controller.h"
#include "daemon/service/controllers/transport_controller.h"
#include "daemon/session_lifecycle.h"
#include "daemon/sessions_service.h"
#include "daemon/sweep_tasks.h"
#include "daemon/transport_lock_manager.h"
#include "daemon/types.h"
#include "daemon/verification_tracker.h"
#include "grpcpp/grpcpp.h"
#include "tensorcast/daemon/v1/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class StoreDaemonServiceImpl final : public v1::StoreDaemonService::Service {
 public:
  struct Options {
    // Sweep/TTL configuration
    std::chrono::seconds sessions_ttl{std::chrono::seconds(60)};
    std::chrono::seconds locks_ttl{std::chrono::seconds(120)};
    std::chrono::milliseconds sessions_sweep_interval{std::chrono::milliseconds(10000)};
    std::chrono::milliseconds locks_sweep_interval{std::chrono::milliseconds(10000)};
    std::chrono::milliseconds verification_sweep_interval{std::chrono::milliseconds(500)};
    std::chrono::milliseconds proc_check_interval{std::chrono::milliseconds(5000)};

    // Eviction policy
    bool enable_periodic_eviction{false};
    double gpu_memory_limit_fraction{0.90};
    std::chrono::milliseconds eviction_check_interval{std::chrono::milliseconds(1000)};

    // Observability
    bool allow_high_card_attrs{false};

    // API behavior flags
    // If true, GetLoadedReplicasV2 uses opaque cursor tokens based on a stable
    // ordering (artifact_id, device_id). If false (default), uses numeric
    // index tokens.
    bool use_cursor_pagination{false};
  };

  explicit StoreDaemonServiceImpl(std::shared_ptr<store::StoreEngine> engine)
      : StoreDaemonServiceImpl(std::move(engine), Options{}) {}

  explicit StoreDaemonServiceImpl(std::shared_ptr<store::StoreEngine> engine, Options opts)
      : engine_(std::move(engine)), sessions_(opts.sessions_ttl), locks_(opts.locks_ttl), opts_(opts) {
    lip_mgr_ = std::make_unique<LipManager>(engine_);
    reg_mgr_ = std::make_unique<RegistrationManager>();
    verif_tracker_ = std::make_unique<VerificationTracker>();
    start_sweepers();
    // Wire helper services and controllers (post-scheduler construction)
    sessions_svc_ = std::make_unique<SessionsService>(
        sessions_, *verif_tracker_, scheduler_.get(), lifecycle_mgr_.get(), absl::Seconds(opts_.sessions_ttl.count()));
    lip_bridge_ = std::make_unique<LipBridge>(*lip_mgr_);
    MaterializationController::Dep dep{
        .engine = *engine_,
        .refs = refs_,
        .sessions = *sessions_svc_,
        .lip = *lip_bridge_,
        .devices = devices_,
        .is_shutting_down = is_shutting_down_,
        .lifecycle = lifecycle_mgr_.get()};
    materialization_controller_ = std::make_unique<MaterializationController>(dep);
    RegistrationController::Dep rdep{
        .engine = *engine_, .reg = *reg_mgr_, .lip = *lip_mgr_, .refs = refs_, .lifecycle = lifecycle_mgr_.get()};
    registration_controller_ = std::make_unique<RegistrationController>(rdep);
    TransportController::Dep tdep{.engine = *engine_, .locks = locks_, .lip = *lip_mgr_};
    transport_controller_ = std::make_unique<TransportController>(tdep);
    StatusController::Dep sdep{
        .engine = *engine_,
        .refs = refs_,
        .is_shutting_down = is_shutting_down_,
        .is_registered = [this]() { return this->is_registered(); },
        .worker_id = [this]() { return this->worker_id(); },
        .uptime =
            [this]() {
              return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);
            }};
    status_controller_ = std::make_unique<StatusController>(sdep);
  }

  ~StoreDaemonServiceImpl() override;

  // Initiate graceful shutdown: reject new materialization requests and allow
  // in-flight operations to complete. Called by server_main signal handler.
  void begin_shutdown() {
    is_shutting_down_.store(true);
  }

  bool is_shutting_down() const {
    return is_shutting_down_.load();
  }

  // Worker lifecycle reflection for status RPCs
  void set_worker_registered(std::string worker_id) {
    {
      absl::MutexLock l(&worker_mu_);
      worker_id_ = std::move(worker_id);
    }
    is_registered_.store(true);
  }

  bool is_registered() const {
    return is_registered_.load();
  }

  std::string worker_id() const {
    absl::MutexLock l(&worker_mu_);
    return worker_id_;
  }

  grpc::Status MaterializeReplica(
      grpc::ServerContext* ctx,
      const v1::MaterializeReplicaRequest* req,
      v1::MaterializeReplicaResponse* resp) override;

  grpc::Status ConfirmReplica(
      grpc::ServerContext* ctx,
      const v1::ConfirmReplicaRequest* req,
      v1::ConfirmReplicaResponse* resp) override;

  grpc::Status UnloadReplica(
      grpc::ServerContext* ctx,
      const v1::UnloadReplicaRequest* req,
      v1::UnloadReplicaResponse* resp) override;

  grpc::Status ClearMem(grpc::ServerContext* ctx, const v1::ClearMemRequest* req, v1::ClearMemResponse* resp) override;

  grpc::Status GetServerConfig(
      grpc::ServerContext* ctx,
      const v1::GetServerConfigRequest* req,
      v1::GetServerConfigResponse* resp) override;

  grpc::Status WaitReplicaVerification(
      grpc::ServerContext* ctx,
      const v1::WaitReplicaVerificationRequest* req,
      v1::WaitReplicaVerificationResponse* resp) override;

  grpc::Status LockTransportChunks(
      grpc::ServerContext* ctx,
      const v1::LockTransportChunksRequest* req,
      v1::LockTransportChunksResponse* resp) override;

  grpc::Status UnlockTransportChunks(
      grpc::ServerContext* ctx,
      const v1::UnlockTransportChunksRequest* req,
      v1::UnlockTransportChunksResponse* resp) override;

  grpc::Status BeginRegisterArtifact(
      grpc::ServerContext* ctx,
      const v1::BeginRegisterArtifactRequest* req,
      v1::BeginRegisterArtifactResponse* resp);

  grpc::Status CommitRegisteredArtifact(
      grpc::ServerContext* ctx,
      const v1::CommitRegisteredArtifactRequest* req,
      v1::CommitRegisteredArtifactResponse* resp) override;

  grpc::Status AbortRegisteredArtifact(
      grpc::ServerContext* ctx,
      const v1::AbortRegisteredArtifactRequest* req,
      v1::AbortRegisteredArtifactResponse* resp) override;

  grpc::Status FeedRegisterArtifactStream(
      grpc::ServerContext* ctx,
      ::grpc::ServerReader<v1::FeedRegisterArtifactStreamRequest>* reader,
      v1::FeedRegisterArtifactStreamResponse* resp) override;

  // Testing/helper overload: process a vector of streaming requests without standing up a gRPC server
  grpc::Status feed_register_artifact_stream_vector(const std::vector<v1::FeedRegisterArtifactStreamRequest>& reqs);

  grpc::Status KeepAliveRegisterArtifact(
      grpc::ServerContext* ctx,
      const v1::KeepAliveRegisterArtifactRequest* req,
      v1::KeepAliveRegisterArtifactResponse* resp) override;

  grpc::Status RevokeRegisteredArtifact(
      grpc::ServerContext* ctx,
      const v1::RevokeRegisteredArtifactRequest* req,
      v1::RevokeRegisteredArtifactResponse* resp) override;

  // RFC-0014
  grpc::Status MaterializeByKey(
      grpc::ServerContext* ctx,
      const v1::MaterializeByKeyRequest* req,
      v1::MaterializeByKeyResponse* resp) override;
  grpc::Status PublishReplicaKey(
      grpc::ServerContext* ctx,
      const v1::PublishReplicaKeyRequest* req,
      v1::PublishReplicaKeyResponse* resp) override;

  grpc::Status ResolveKeyMapping(
      grpc::ServerContext* ctx,
      const v1::ResolveKeyMappingRequest* req,
      v1::ResolveKeyMappingResponse* resp) override;

  grpc::Status GetArtifactIndexById(
      grpc::ServerContext* ctx,
      const v1::GetArtifactIndexByIdRequest* req,
      v1::GetArtifactIndexByIdResponse* resp) override;

  // Status & listing RPCs
  grpc::Status GetWorkerStatus(
      grpc::ServerContext* ctx,
      const v1::GetWorkerStatusRequest* req,
      v1::GetWorkerStatusResponse* resp) override;

  grpc::Status GetDetailedStatus(
      grpc::ServerContext* ctx,
      const v1::GetDetailedStatusRequest* req,
      v1::GetDetailedStatusResponse* resp) override;

  // Legacy GetLoadedReplicas removed; use V2

  grpc::Status GetLoadedReplicasV2(
      grpc::ServerContext* ctx,
      const v1::GetLoadedReplicasV2Request* req,
      v1::GetLoadedReplicasV2Response* resp) override;

  // Expose current ref-count for a given replica key (for HA state reporting)
  size_t ref_count_for(const store::loading::ReplicaKey& key) const {
    return refs_.ref_count(key);
  }

 private:
  std::shared_ptr<store::StoreEngine> engine_;
  ReplicaSessionManager sessions_;
  TransportLockManager locks_;
  RefTracker refs_;
  std::unique_ptr<LipManager> lip_mgr_;

  // Background sweepers (scheduler-driven)
  std::unique_ptr<BackgroundScheduler> scheduler_;
  std::chrono::time_point<std::chrono::steady_clock> start_time_{std::chrono::steady_clock::now()};
  void start_sweepers();
  void stop_sweepers();
  std::shared_ptr<SessionLifecycleManager> lifecycle_mgr_;
  std::unique_ptr<PidMonitor> pid_monitor_;

  // Helpers (moved into controllers/helpers per RFC‑0016)

  // Shutdown gating
  std::atomic<bool> is_shutting_down_{false};
  // Compatibility configuration removed

  // Worker lifecycle reflection
  std::atomic<bool> is_registered_{false};
  mutable absl::Mutex worker_mu_;
  std::string worker_id_ ABSL_GUARDED_BY(worker_mu_);

  // Verification tracker extracted to its own component
  std::unique_ptr<VerificationTracker> verif_tracker_;

  // Background task queue for auto-registration only (verification queue moved)
  absl::Mutex bg_tasks_mu_;
  std::deque<AutoRegTask> auto_reg_tasks_ ABSL_GUARDED_BY(bg_tasks_mu_);

  // Registration lifecycle state manager (Begin/Feed/KeepAlive/Commit)
  std::unique_ptr<RegistrationManager> reg_mgr_;

  // LIP registry moved into LipManager

  // LIP staged export management moved to LipManager.

  // Helper: D2D copy from LIP segments into a new coalesced destination on target device; returns CUDA IPC handle
  // bytes.
  absl::StatusOr<std::vector<uint8_t>> lip_copy_to_new_coalesced_int(
      int target_device_id,
      const std::string& canonical_index_json,
      uint64_t total_size,
      absl::Span<const LeaseSegMeta> segments,
      absl::Span<const RegisterStorageMeta> storages);

  Options opts_;

  // New helpers/controllers (RFC-0016)
  DeviceResolver devices_{store::DeviceRegistry::instance()};
  std::unique_ptr<SessionsService> sessions_svc_;
  std::unique_ptr<LipBridge> lip_bridge_;
  std::unique_ptr<MaterializationController> materialization_controller_;
  std::unique_ptr<RegistrationController> registration_controller_;
  std::unique_ptr<TransportController> transport_controller_;
  std::unique_ptr<StatusController> status_controller_;
};

} // namespace tensorcast::daemon
