// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <future>
#include <memory>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "core/store/store_engine.h"
#include "daemon/ref_tracker.h"
#include "daemon/replica_session_manager.h"
#include "daemon/transport_lock_manager.h"
#include "grpcpp/grpcpp.h"
#include "store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class StoreDaemonServiceImpl final : public daemon::StoreDaemon::Service {
 public:
  explicit StoreDaemonServiceImpl(std::shared_ptr<tensorcast::store::StoreEngine> engine)
      : engine_(std::move(engine)), sessions_(std::chrono::seconds(60)) {
    start_sweepers();
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
      const daemon::MaterializeReplicaRequest* req,
      daemon::MaterializeReplicaResponse* resp) override;

  grpc::Status ConfirmReplica(
      grpc::ServerContext* ctx,
      const daemon::ConfirmReplicaRequest* req,
      daemon::ConfirmReplicaResponse* resp) override;

  grpc::Status UnloadReplica(
      grpc::ServerContext* ctx,
      const daemon::UnloadReplicaRequest* req,
      daemon::UnloadReplicaResponse* resp) override;

  grpc::Status ClearMem(grpc::ServerContext* ctx, const daemon::ClearMemRequest* req, daemon::ClearMemResponse* resp)
      override;

  grpc::Status GetServerConfig(
      grpc::ServerContext* ctx,
      const daemon::GetServerConfigRequest* req,
      daemon::GetServerConfigResponse* resp) override;

  grpc::Status WaitReplicaVerification(
      grpc::ServerContext* ctx,
      const daemon::ReplicaVerificationRequest* req,
      daemon::ReplicaVerificationResponse* resp) override;

  grpc::Status LockTransportChunks(
      grpc::ServerContext* ctx,
      const daemon::LockChunksRequest* req,
      daemon::LockChunksResponse* resp) override;

  grpc::Status UnlockTransportChunks(
      grpc::ServerContext* ctx,
      const daemon::UnlockChunksRequest* req,
      daemon::UnlockChunksResponse* resp) override;

  grpc::Status BeginRegisterArtifact(
      grpc::ServerContext* ctx,
      const daemon::BeginRegisterArtifactRequest* req,
      daemon::BeginRegisterArtifactResponse* resp) override;

  grpc::Status CommitRegisteredArtifact(
      grpc::ServerContext* ctx,
      const daemon::CommitRegisteredArtifactRequest* req,
      daemon::CommitRegisteredArtifactResponse* resp) override;

  grpc::Status AbortRegisteredArtifact(
      grpc::ServerContext* ctx,
      const daemon::AbortRegisteredArtifactRequest* req,
      daemon::AbortRegisteredArtifactResponse* resp) override;

  // Status & listing RPCs
  grpc::Status GetWorkerStatus(
      grpc::ServerContext* ctx,
      const daemon::GetWorkerStatusRequest* req,
      daemon::GetWorkerStatusResponse* resp) override;

  grpc::Status GetDetailedStatus(
      grpc::ServerContext* ctx,
      const daemon::GetDetailedStatusRequest* req,
      daemon::GetDetailedStatusResponse* resp) override;

  // Legacy GetLoadedReplicas removed; use V2

  grpc::Status GetLoadedReplicasV2(
      grpc::ServerContext* ctx,
      const daemon::GetLoadedReplicasV2Request* req,
      daemon::GetLoadedReplicasV2Response* resp) override;

  // Expose current ref-count for a given replica key (for HA state reporting)
  size_t ref_count_for(const store::loading::ReplicaKey& key) const {
    return refs_.ref_count(key);
  }

 private:
  std::shared_ptr<tensorcast::store::StoreEngine> engine_;
  ReplicaSessionManager sessions_;
  TransportLockManager locks_{std::chrono::seconds(120)};
  RefTracker refs_;

  // Background sweepers
  std::atomic<bool> stop_{false};
  std::thread sweep_sessions_th_;
  std::thread sweep_locks_th_;
  std::thread pid_watcher_th_;
  std::thread verif_sweeper_th_;
  std::thread eviction_th_;
  std::chrono::time_point<std::chrono::steady_clock> start_time_{std::chrono::steady_clock::now()};
  void start_sweepers();
  void stop_sweepers();

  // Helpers
  static tensorcast::store::DeviceKey resolve_device(const daemon::MaterializeReplicaRequest& req);
  static tensorcast::store::DeviceKey resolve_device(const daemon::ConfirmReplicaRequest& req);
  static tensorcast::store::DeviceKey resolve_device(const daemon::UnloadReplicaRequest& req);
  static store::loading::ReplicaKey make_replica_key(const std::string& artifact_id);

  // Shutdown gating
  std::atomic<bool> is_shutting_down_{false};
  // Compatibility configuration removed

  // Worker lifecycle reflection
  std::atomic<bool> is_registered_{false};
  mutable absl::Mutex worker_mu_;
  std::string worker_id_ ABSL_GUARDED_BY(worker_mu_);

  // Lightweight verification registry keyed by replica_uuid. Tracks
  // verification progress decoupled from ready_future state for parity with
  // Python daemon semantics.
  struct VerifEntry {
    daemon::VerificationStatus status{daemon::VerificationStatus::VERIFICATION_STATUS_IN_PROGRESS};
    std::string err;
  };
  absl::Mutex verif_mu_;
  absl::flat_hash_map<std::string, VerifEntry> verif_ ABSL_GUARDED_BY(verif_mu_);
  void set_verif_status(const std::string& uuid, daemon::VerificationStatus st, std::string err = "");

  // Background task queue for verification completion and auto-registration
  struct VerifTask {
    std::string uuid;
    std::shared_future<absl::Status> ready;
  };
  struct AutoRegTask {
    store::loading::ReplicaKey key;
    std::string disk_path;
    std::shared_future<absl::Status> ready;
  };
  absl::Mutex bg_tasks_mu_;
  std::deque<VerifTask> verif_tasks_ ABSL_GUARDED_BY(bg_tasks_mu_);
  std::deque<AutoRegTask> auto_reg_tasks_ ABSL_GUARDED_BY(bg_tasks_mu_);
};

} // namespace tensorcast::daemon
