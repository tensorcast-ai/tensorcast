// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "absl/time/time.h"
#include "core/store/store_engine.h"
#include "daemon/app/startup_coordinator.h"
#include "daemon/service/controllers/byte_artifact_controller.h"
#include "daemon/service/controllers/key_mapping_controller.h"
#include "daemon/service/controllers/lease_controller.h"
#include "daemon/service/controllers/materialization_controller.h"
#include "daemon/service/controllers/persistence_rpc_controller.h"
#include "daemon/service/controllers/registration_controller.h"
#include "daemon/service/controllers/replica_session_controller.h"
#include "daemon/service/controllers/status_controller.h"
#include "daemon/service/controllers/transport_controller.h"
#include "daemon/state/artifact_source_registry.h"
#include "daemon/state/instance_execution_directory_cache.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/worker_identity_store.h"
#include "grpcpp/grpcpp.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class StoreDaemonServiceImpl final : public v2::StoreDaemonService::Service {
 public:
  struct Options {
    bool allow_high_card_attrs{false};
    bool use_cursor_pagination{false};
    bool gateway_ingress_enabled{false};
    std::filesystem::path storage_path;
    absl::Duration directory_staleness_budget{absl::Seconds(2)};
  };

  struct Deps {
    store::StoreEngine& engine;
    MaterializationController& materialization_controller;
    ByteArtifactController& byte_artifact_controller;
    RegistrationController& registration_controller;
    TransportController& transport_controller;
    StatusController& status_controller;
    WorkerIdentityStore& identity_store;
    InstanceExecutionDirectoryCache& instance_execution_directory_cache;
    IpcRegionRegistry& region_registry;
    LipManager& lip_manager;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    SessionLifecycleManager& lifecycle_manager;
    KeyMappingController& key_mapping_controller;
    PersistenceRpcController& persistence_rpc_controller;
    ReplicaSessionController& replica_session_controller;
    LeaseController& lease_controller;
    ShutdownSignal& shutdown_signal;
    std::shared_ptr<StartupCoordinator> startup_coordinator;
    ArtifactSourceRegistry* source_registry{nullptr};
  };

  StoreDaemonServiceImpl(Deps deps, Options opts);

  grpc::Status ExecutePlan(grpc::ServerContext* ctx, const v2::ExecutePlanRequest* req, v2::ExecutePlanResponse* resp)
      override;

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

  grpc::Status CreateOwnedBinding(
      grpc::ServerContext* ctx,
      const v2::CreateOwnedBindingRequest* req,
      v2::CreateOwnedBindingResponse* resp) override;

  grpc::Status CreateBinding(
      grpc::ServerContext* ctx,
      const v2::CreateBindingRequest* req,
      v2::CreateBindingResponse* resp) override;

  grpc::Status CommitBindingArtifact(
      grpc::ServerContext* ctx,
      const v2::CommitBindingArtifactRequest* req,
      v2::CommitBindingArtifactResponse* resp) override;

  grpc::Status BeginBindingUpdate(
      grpc::ServerContext* ctx,
      const v2::BeginBindingUpdateRequest* req,
      v2::BeginBindingUpdateResponse* resp) override;

  grpc::Status SubmitBindingContribution(
      grpc::ServerContext* ctx,
      const v2::SubmitBindingContributionRequest* req,
      v2::SubmitBindingContributionResponse* resp) override;

  grpc::Status SealBinding(grpc::ServerContext* ctx, const v2::SealBindingRequest* req, v2::SealBindingResponse* resp)
      override;

  grpc::Status PromoteBindingCurrentValue(
      grpc::ServerContext* ctx,
      const v2::PromoteBindingCurrentValueRequest* req,
      v2::PromoteBindingCurrentValueResponse* resp) override;

  grpc::Status StartAssemblyAttempt(
      grpc::ServerContext* ctx,
      const v2::StartAssemblyAttemptRequest* req,
      v2::StartAssemblyAttemptResponse* resp) override;
  grpc::Status SealAssemblyAttempt(
      grpc::ServerContext* ctx,
      const v2::SealAssemblyAttemptRequest* req,
      v2::SealAssemblyAttemptResponse* resp) override;

  grpc::Status RefillOwnedBinding(
      grpc::ServerContext* ctx,
      const v2::RefillOwnedBindingRequest* req,
      v2::RefillOwnedBindingResponse* resp) override;

  grpc::Status CloseOwnedBinding(
      grpc::ServerContext* ctx,
      const v2::CloseOwnedBindingRequest* req,
      v2::CloseOwnedBindingResponse* resp) override;

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

  grpc::Status ListDirectoryWorkers(
      grpc::ServerContext* ctx,
      const v2::ListDirectoryWorkersRequest* req,
      v2::ListDirectoryWorkersResponse* resp) override;

  grpc::Status ListDirectoryInstances(
      grpc::ServerContext* ctx,
      const v2::ListDirectoryInstancesRequest* req,
      v2::ListDirectoryInstancesResponse* resp) override;

  grpc::Status ResolveInstanceExecution(
      grpc::ServerContext* ctx,
      const v2::ResolveInstanceExecutionRequest* req,
      v2::ResolveInstanceExecutionResponse* resp) override;

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

  grpc::Status StartPublishTargetReplica(
      grpc::ServerContext* ctx,
      const v2::PublishTargetReplicaRequest* req,
      v2::StartPublishTargetReplicaResponse* resp) override;

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

  grpc::Status ImportArtifactFromPath(
      grpc::ServerContext* ctx,
      const v2::ImportArtifactFromPathRequest* req,
      v2::ImportArtifactFromPathResponse* resp) override;

  grpc::Status ResolvePublicDiskSource(
      grpc::ServerContext* ctx,
      const v2::ResolvePublicDiskSourceRequest* req,
      v2::ResolvePublicDiskSourceResponse* resp) override;

  grpc::Status ImportArtifactFromPathStream(
      grpc::ServerContext* ctx,
      const v2::ImportArtifactFromPathRequest* req,
      grpc::ServerWriter<v2::ImportArtifactFromPathStreamEvent>* writer) override;

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

  grpc::Status ListArtifactLayouts(
      grpc::ServerContext* ctx,
      const v2::ListArtifactLayoutsRequest* req,
      v2::ListArtifactLayoutsResponse* resp) override;

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

  grpc::Status BatchExists(grpc::ServerContext* ctx, const v2::BatchExistsRequest* req, v2::BatchExistsResponse* resp)
      override;

  grpc::Status BatchGetIntoRegion(
      grpc::ServerContext* ctx,
      const v2::BatchGetIntoRegionRequest* req,
      v2::BatchGetIntoRegionResponse* resp) override;

  grpc::Status BatchPutIfAbsentFromRegion(
      grpc::ServerContext* ctx,
      const v2::BatchPutIfAbsentFromRegionRequest* req,
      v2::BatchPutIfAbsentFromRegionResponse* resp) override;

  grpc::Status BatchTouchTtl(
      grpc::ServerContext* ctx,
      const v2::BatchTouchTtlRequest* req,
      v2::BatchTouchTtlResponse* resp) override;

  grpc::Status HomeBatchExists(
      grpc::ServerContext* ctx,
      const v2::HomeBatchExistsRequest* req,
      v2::HomeBatchExistsResponse* resp) override;

  grpc::Status HomeBatchGet(
      grpc::ServerContext* ctx,
      const v2::HomeBatchGetRequest* req,
      v2::HomeBatchGetResponse* resp) override;

  grpc::Status HomeBatchPutIfAbsent(
      grpc::ServerContext* ctx,
      const v2::HomeBatchPutIfAbsentRequest* req,
      v2::HomeBatchPutIfAbsentResponse* resp) override;

  grpc::Status HomeBatchTouchTtl(
      grpc::ServerContext* ctx,
      const v2::HomeBatchTouchTtlRequest* req,
      v2::HomeBatchTouchTtlResponse* resp) override;

  grpc::Status FetchPayloadRefChunk(
      grpc::ServerContext* ctx,
      const v2::FetchPayloadRefChunkRequest* req,
      v2::FetchPayloadRefChunkResponse* resp) override;

  grpc::Status RouteAuthorityStage(
      grpc::ServerContext* ctx,
      const v2::RouteAuthorityStageRequest* req,
      v2::RouteAuthorityStageResponse* resp) override;

 private:
  grpc::Status block_if_startup_pending() const;

  store::StoreEngine* engine_;
  MaterializationController* materialization_controller_;
  ByteArtifactController* byte_artifact_controller_;
  RegistrationController* registration_controller_;
  TransportController* transport_controller_;
  StatusController* status_controller_;
  WorkerIdentityStore* identity_store_;
  InstanceExecutionDirectoryCache* instance_execution_directory_cache_;
  IpcRegionRegistry* region_registry_;
  LipManager* lip_manager_;
  std::shared_ptr<store::components::IGlobalStoreClient> global_store_client_;
  SessionLifecycleManager* lifecycle_manager_;
  KeyMappingController* key_mapping_controller_;
  PersistenceRpcController* persistence_rpc_controller_;
  ReplicaSessionController* replica_session_controller_;
  LeaseController* lease_controller_;
  ShutdownSignal* shutdown_signal_;
  std::shared_ptr<StartupCoordinator> startup_coordinator_;
  ArtifactSourceRegistry* source_registry_{nullptr};
  Options opts_;
};

} // namespace tensorcast::daemon
