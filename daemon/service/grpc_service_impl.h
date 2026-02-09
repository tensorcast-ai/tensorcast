// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "core/store/store_engine.h"
#include "daemon/service/controllers/key_mapping_controller.h"
#include "daemon/service/controllers/lease_controller.h"
#include "daemon/service/controllers/materialization_controller.h"
#include "daemon/service/controllers/persistence_rpc_controller.h"
#include "daemon/service/controllers/registration_controller.h"
#include "daemon/service/controllers/replica_session_controller.h"
#include "daemon/service/controllers/status_controller.h"
#include "daemon/service/controllers/transport_controller.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/shutdown_signal.h"
#include "grpcpp/grpcpp.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class StoreDaemonServiceImpl final : public v2::StoreDaemonService::Service {
 public:
  struct Options {
    bool allow_high_card_attrs{false};
    bool use_cursor_pagination{false};
    std::filesystem::path storage_path;
  };

  struct Deps {
    store::StoreEngine& engine;
    MaterializationController& materialization_controller;
    RegistrationController& registration_controller;
    TransportController& transport_controller;
    StatusController& status_controller;
    IpcRegionRegistry& region_registry;
    LipManager& lip_manager;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    SessionLifecycleManager& lifecycle_manager;
    KeyMappingController& key_mapping_controller;
    PersistenceRpcController& persistence_rpc_controller;
    ReplicaSessionController& replica_session_controller;
    LeaseController& lease_controller;
    ShutdownSignal& shutdown_signal;
  };

  StoreDaemonServiceImpl(Deps deps, Options opts);

  grpc::Status MaterializeReplica(
      grpc::ServerContext* ctx,
      const v2::MaterializeReplicaRequest* req,
      v2::MaterializeReplicaResponse* resp) override;

  grpc::Status MaterializeIntoTarget(
      grpc::ServerContext* ctx,
      const v2::MaterializeIntoTargetRequest* req,
      v2::MaterializeIntoTargetResponse* resp) override;

  grpc::Status MaterializeIntoMappedTarget(
      grpc::ServerContext* ctx,
      const v2::MaterializeIntoMappedTargetRequest* req,
      v2::MaterializeIntoTargetResponse* resp) override;

  grpc::Status ConfirmReplica(
      grpc::ServerContext* ctx,
      const v2::ConfirmReplicaRequest* req,
      v2::ConfirmReplicaResponse* resp) override;

  grpc::Status UnloadReplica(
      grpc::ServerContext* ctx,
      const v2::UnloadReplicaRequest* req,
      v2::UnloadReplicaResponse* resp) override;

  grpc::Status ClearMem(grpc::ServerContext* ctx, const v2::ClearMemRequest* req, v2::ClearMemResponse* resp) override;

  grpc::Status GetServerConfig(
      grpc::ServerContext* ctx,
      const v2::GetServerConfigRequest* req,
      v2::GetServerConfigResponse* resp) override;

  grpc::Status WaitReplicaVerification(
      grpc::ServerContext* ctx,
      const v2::WaitReplicaVerificationRequest* req,
      v2::WaitReplicaVerificationResponse* resp) override;

  grpc::Status LockTransportChunks(
      grpc::ServerContext* ctx,
      const v2::LockTransportChunksRequest* req,
      v2::LockTransportChunksResponse* resp) override;

  grpc::Status RegisterVramRegion(
      grpc::ServerContext* ctx,
      const v2::RegisterVramRegionRequest* req,
      v2::RegisterVramRegionResponse* resp) override;

  grpc::Status UnregisterVramRegion(
      grpc::ServerContext* ctx,
      const v2::UnregisterVramRegionRequest* req,
      v2::UnregisterVramRegionResponse* resp) override;

  grpc::Status DeregisterArtifact(
      grpc::ServerContext* ctx,
      const v2::DeregisterArtifactRequest* req,
      v2::DeregisterArtifactResponse* resp) override;

  grpc::Status PublishTargetReplica(
      grpc::ServerContext* ctx,
      const v2::PublishTargetReplicaRequest* req,
      v2::PublishTargetReplicaResponse* resp) override;

  grpc::Status RetirePublishedReplica(
      grpc::ServerContext* ctx,
      const v2::RetirePublishedReplicaRequest* req,
      v2::RetirePublishedReplicaResponse* resp) override;

  grpc::Status UnlockTransportChunks(
      grpc::ServerContext* ctx,
      const v2::UnlockTransportChunksRequest* req,
      v2::UnlockTransportChunksResponse* resp) override;

  grpc::Status BeginRegisterArtifact(
      grpc::ServerContext* ctx,
      const v2::BeginRegisterArtifactRequest* req,
      v2::BeginRegisterArtifactResponse* resp) override;

  grpc::Status CommitRegisteredArtifact(
      grpc::ServerContext* ctx,
      const v2::CommitRegisteredArtifactRequest* req,
      v2::CommitRegisteredArtifactResponse* resp) override;

  grpc::Status AbortRegisteredArtifact(
      grpc::ServerContext* ctx,
      const v2::AbortRegisteredArtifactRequest* req,
      v2::AbortRegisteredArtifactResponse* resp) override;

  grpc::Status FeedRegisterArtifactStream(
      grpc::ServerContext* ctx,
      ::grpc::ServerReader<v2::FeedRegisterArtifactStreamRequest>* reader,
      v2::FeedRegisterArtifactStreamResponse* resp) override;

  // Testing/helper overload: process a vector of streaming requests without standing up a gRPC server
  grpc::Status feed_register_artifact_stream_vector(const std::vector<v2::FeedRegisterArtifactStreamRequest>& reqs);

  grpc::Status KeepAliveRegisterArtifact(
      grpc::ServerContext* ctx,
      const v2::KeepAliveRegisterArtifactRequest* req,
      v2::KeepAliveRegisterArtifactResponse* resp) override;

  grpc::Status RevokeRegisteredArtifact(
      grpc::ServerContext* ctx,
      const v2::RevokeRegisteredArtifactRequest* req,
      v2::RevokeRegisteredArtifactResponse* resp) override;

  grpc::Status MaterializeByKey(
      grpc::ServerContext* ctx,
      const v2::MaterializeByKeyRequest* req,
      v2::MaterializeByKeyResponse* resp) override;

  grpc::Status ResolveArtifactFromDisk(
      grpc::ServerContext* ctx,
      const v2::ResolveArtifactFromDiskRequest* req,
      v2::ResolveArtifactFromDiskResponse* resp) override;

  grpc::Status QueryReplicaStatus(
      grpc::ServerContext* ctx,
      const v2::QueryReplicaStatusRequest* req,
      v2::QueryReplicaStatusResponse* resp) override;

  grpc::Status WaitReplicaStatus(
      grpc::ServerContext* ctx,
      const v2::WaitReplicaStatusRequest* req,
      v2::WaitReplicaStatusResponse* resp) override;

  grpc::Status ReleaseReplica(
      grpc::ServerContext* ctx,
      const v2::ReleaseReplicaRequest* req,
      v2::ReleaseReplicaResponse* resp) override;

  grpc::Status CreatePlacementLease(
      grpc::ServerContext* ctx,
      const v2::CreatePlacementLeaseRequest* req,
      v2::CreatePlacementLeaseResponse* resp) override;

  grpc::Status RenewPlacementLease(
      grpc::ServerContext* ctx,
      const v2::RenewPlacementLeaseRequest* req,
      v2::RenewPlacementLeaseResponse* resp) override;

  grpc::Status ReleasePlacementLease(
      grpc::ServerContext* ctx,
      const v2::ReleasePlacementLeaseRequest* req,
      v2::ReleasePlacementLeaseResponse* resp) override;

  grpc::Status AcquireRetentionHandle(
      grpc::ServerContext* ctx,
      const v2::AcquireRetentionHandleRequest* req,
      v2::AcquireRetentionHandleResponse* resp) override;

  grpc::Status RenewRetentionHandle(
      grpc::ServerContext* ctx,
      const v2::RenewRetentionHandleRequest* req,
      v2::RenewRetentionHandleResponse* resp) override;

  grpc::Status ReleaseRetentionHandle(
      grpc::ServerContext* ctx,
      const v2::ReleaseRetentionHandleRequest* req,
      v2::ReleaseRetentionHandleResponse* resp) override;

  grpc::Status PublishReplicaKey(
      grpc::ServerContext* ctx,
      const v2::PublishReplicaKeyRequest* req,
      v2::PublishReplicaKeyResponse* resp) override;

  grpc::Status ResolveKeyMapping(
      grpc::ServerContext* ctx,
      const v2::ResolveKeyMappingRequest* req,
      v2::ResolveKeyMappingResponse* resp) override;
  grpc::Status SwapKeyMapping(
      grpc::ServerContext* ctx,
      const v2::SwapKeyMappingRequest* req,
      v2::SwapKeyMappingResponse* resp) override;

  grpc::Status GetArtifactIndexById(
      grpc::ServerContext* ctx,
      const v2::GetArtifactIndexByIdRequest* req,
      v2::GetArtifactIndexByIdResponse* resp) override;

  grpc::Status SealAssembly(
      grpc::ServerContext* ctx,
      const v2::SealAssemblyRequest* req,
      v2::SealAssemblyResponse* resp) override;

  grpc::Status StartSealAssembly(
      grpc::ServerContext* ctx,
      const v2::StartSealAssemblyRequest* req,
      v2::StartSealAssemblyResponse* resp) override;

  grpc::Status GetOperation(
      grpc::ServerContext* ctx,
      const tensorcast::operation::v1::GetOperationRequest* req,
      tensorcast::operation::v1::GetOperationResponse* resp) override;

  grpc::Status WaitOperation(
      grpc::ServerContext* ctx,
      const v2::WaitOperationRequest* req,
      v2::WaitOperationResponse* resp) override;

  grpc::Status StartPersistence(
      grpc::ServerContext* ctx,
      const v2::StartPersistenceRequest* req,
      v2::StartPersistenceResponse* resp) override;

  grpc::Status QueryPersistenceStatus(
      grpc::ServerContext* ctx,
      const v2::QueryPersistenceStatusRequest* req,
      v2::QueryPersistenceStatusResponse* resp) override;

  grpc::Status GetWorkerStatus(
      grpc::ServerContext* ctx,
      const v2::GetWorkerStatusRequest* req,
      v2::GetWorkerStatusResponse* resp) override;

  grpc::Status GetDetailedStatus(
      grpc::ServerContext* ctx,
      const v2::GetDetailedStatusRequest* req,
      v2::GetDetailedStatusResponse* resp) override;

  grpc::Status GetLoadedReplicasV2(
      grpc::ServerContext* ctx,
      const v2::GetLoadedReplicasV2Request* req,
      v2::GetLoadedReplicasV2Response* resp) override;

 private:
  store::StoreEngine* engine_;
  MaterializationController* materialization_controller_;
  RegistrationController* registration_controller_;
  TransportController* transport_controller_;
  StatusController* status_controller_;
  IpcRegionRegistry* region_registry_;
  LipManager* lip_manager_;
  std::shared_ptr<store::components::IGlobalStoreClient> global_store_client_;
  SessionLifecycleManager* lifecycle_manager_;
  KeyMappingController* key_mapping_controller_;
  PersistenceRpcController* persistence_rpc_controller_;
  ReplicaSessionController* replica_session_controller_;
  LeaseController* lease_controller_;
  ShutdownSignal* shutdown_signal_;
  Options opts_;
};

} // namespace tensorcast::daemon
