// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "absl/log/log.h"
#include "core/store/store_engine.h"
#include "daemon/ref_tracker.h"
#include "daemon/replica_session_manager.h"
#include "daemon/transport_lock_manager.h"
#include "grpcpp/grpcpp.h"
#include "proto/store_daemon.grpc.pb.h"

namespace stepcast::daemon {

class StoreDaemonServiceImpl final : public ::store_daemon::StoreDaemon::Service {
 public:
  explicit StoreDaemonServiceImpl(std::shared_ptr<stepcast::store::StoreEngine> engine)
      : engine_(std::move(engine)), sessions_(std::chrono::seconds(60)) {
    start_sweepers();
  }

  ~StoreDaemonServiceImpl() override;

  grpc::Status MaterializeReplica(
      grpc::ServerContext* ctx,
      const ::store_daemon::MaterializeReplicaRequest* req,
      ::store_daemon::MaterializeReplicaResponse* resp) override;

  grpc::Status ConfirmReplica(
      grpc::ServerContext* ctx,
      const ::store_daemon::ConfirmReplicaRequest* req,
      ::store_daemon::ConfirmReplicaResponse* resp) override;

  grpc::Status UnloadReplica(
      grpc::ServerContext* ctx,
      const ::store_daemon::UnloadReplicaRequest* req,
      ::store_daemon::UnloadReplicaResponse* resp) override;

  grpc::Status ClearMem(
      grpc::ServerContext* ctx,
      const ::store_daemon::ClearMemRequest* req,
      ::store_daemon::ClearMemResponse* resp) override;

  grpc::Status GetServerConfig(
      grpc::ServerContext* ctx,
      const ::store_daemon::GetServerConfigRequest* req,
      ::store_daemon::GetServerConfigResponse* resp) override;

  grpc::Status WaitReplicaVerification(
      grpc::ServerContext* ctx,
      const ::store_daemon::ReplicaVerificationRequest* req,
      ::store_daemon::ReplicaVerificationResponse* resp) override;

  grpc::Status LockTransportChunks(
      grpc::ServerContext* ctx,
      const ::store_daemon::LockChunksRequest* req,
      ::store_daemon::LockChunksResponse* resp) override;

  grpc::Status UnlockTransportChunks(
      grpc::ServerContext* ctx,
      const ::store_daemon::UnlockChunksRequest* req,
      ::store_daemon::UnlockChunksResponse* resp) override;

  grpc::Status BeginRegisterArtifact(
      grpc::ServerContext* ctx,
      const ::store_daemon::BeginRegisterArtifactRequest* req,
      ::store_daemon::BeginRegisterArtifactResponse* resp) override;

  grpc::Status CommitRegisteredArtifact(
      grpc::ServerContext* ctx,
      const ::store_daemon::CommitRegisteredArtifactRequest* req,
      ::store_daemon::CommitRegisteredArtifactResponse* resp) override;

  grpc::Status AbortRegisteredArtifact(
      grpc::ServerContext* ctx,
      const ::store_daemon::AbortRegisteredArtifactRequest* req,
      ::store_daemon::AbortRegisteredArtifactResponse* resp) override;

  // Status & listing RPCs
  grpc::Status GetWorkerStatus(
      grpc::ServerContext* ctx,
      const ::store_daemon::GetWorkerStatusRequest* req,
      ::store_daemon::GetWorkerStatusResponse* resp) override;

  grpc::Status GetDetailedStatus(
      grpc::ServerContext* ctx,
      const ::store_daemon::GetDetailedStatusRequest* req,
      ::store_daemon::GetDetailedStatusResponse* resp) override;

  grpc::Status GetLoadedReplicas(
      grpc::ServerContext* ctx,
      const ::store_daemon::GetLoadedReplicasRequest* req,
      ::store_daemon::GetLoadedReplicasResponse* resp) override;

 private:
  std::shared_ptr<stepcast::store::StoreEngine> engine_;
  ReplicaSessionManager sessions_;
  TransportLockManager locks_{std::chrono::seconds(120)};
  RefTracker refs_;

  // Background sweepers
  std::atomic<bool> stop_{false};
  std::thread sweep_sessions_th_;
  std::thread sweep_locks_th_;
  std::chrono::time_point<std::chrono::steady_clock> start_time_{std::chrono::steady_clock::now()};
  void start_sweepers();
  void stop_sweepers();

  // Helpers
  static stepcast::store::DeviceKey resolve_device(const ::store_daemon::MaterializeReplicaRequest& req);
  static stepcast::store::DeviceKey resolve_device(const ::store_daemon::ConfirmReplicaRequest& req);
  static stepcast::store::DeviceKey resolve_device(const ::store_daemon::UnloadReplicaRequest& req);
  static stepcast::store::ReplicaKey make_replica_key(const std::string& artifact_id);
};

} // namespace stepcast::daemon
