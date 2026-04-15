// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/grpc_service_impl.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;

Status StoreDaemonServiceImpl::MaterializeReplica(
    grpc::ServerContext* ctx,
    const v2::MaterializeReplicaRequest* req,
    v2::MaterializeReplicaResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"MaterializeReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->materialize_replica(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::MaterializeIntoTarget(
    grpc::ServerContext* ctx,
    const v2::MaterializeIntoTargetRequest* req,
    v2::MaterializeIntoTargetResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"MaterializeIntoTarget", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->materialize_into_target(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::MaterializeIntoMappedTarget(
    grpc::ServerContext* ctx,
    const v2::MaterializeIntoMappedTargetRequest* req,
    v2::MaterializeIntoTargetResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"MaterializeIntoMappedTarget", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->materialize_into_mapped_target(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::CreateOwnedBinding(
    grpc::ServerContext* ctx,
    const v2::CreateOwnedBindingRequest* req,
    v2::CreateOwnedBindingResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"CreateOwnedBinding", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->create_owned_binding(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::CreateBinding(
    grpc::ServerContext* ctx,
    const v2::CreateBindingRequest* req,
    v2::CreateBindingResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"CreateBinding", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->create_binding(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::CommitBindingArtifact(
    grpc::ServerContext* ctx,
    const v2::CommitBindingArtifactRequest* req,
    v2::CommitBindingArtifactResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"CommitBindingArtifact", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->commit_binding_artifact(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::BeginBindingUpdate(
    grpc::ServerContext* ctx,
    const v2::BeginBindingUpdateRequest* req,
    v2::BeginBindingUpdateResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"BeginBindingUpdate", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->begin_binding_update(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::SealBinding(
    grpc::ServerContext* ctx,
    const v2::SealBindingRequest* req,
    v2::SealBindingResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"SealBinding", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->seal_binding(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::PromoteBindingCurrentValue(
    grpc::ServerContext* ctx,
    const v2::PromoteBindingCurrentValueRequest* req,
    v2::PromoteBindingCurrentValueResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"PromoteBindingCurrentValue", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->promote_binding_current_value(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::SubmitBindingContribution(
    grpc::ServerContext* ctx,
    const v2::SubmitBindingContributionRequest* req,
    v2::SubmitBindingContributionResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"SubmitBindingContribution", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->submit_binding_contribution(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::StartAssemblyAttempt(
    grpc::ServerContext* ctx,
    const v2::StartAssemblyAttemptRequest* req,
    v2::StartAssemblyAttemptResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"StartAssemblyAttempt", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->start_assembly_attempt(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::SealAssemblyAttempt(
    grpc::ServerContext* ctx,
    const v2::SealAssemblyAttemptRequest* req,
    v2::SealAssemblyAttemptResponse* resp) {
  if (materialization_controller_ == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "materialization controller unavailable"};
  }
  if (ctx == nullptr || req == nullptr || resp == nullptr) {
    return {StatusCode::INVALID_ARGUMENT, "invalid RPC arguments"};
  }
  RpcContext rctx{"SealAssemblyAttempt", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->seal_assembly_attempt(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RefillOwnedBinding(
    grpc::ServerContext* ctx,
    const v2::RefillOwnedBindingRequest* req,
    v2::RefillOwnedBindingResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"RefillOwnedBinding", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->refill_owned_binding(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::CloseOwnedBinding(
    grpc::ServerContext* ctx,
    const v2::CloseOwnedBindingRequest* req,
    v2::CloseOwnedBindingResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"CloseOwnedBinding", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->close_owned_binding(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ConfirmReplica(
    grpc::ServerContext* ctx,
    const v2::ConfirmReplicaRequest* req,
    v2::ConfirmReplicaResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"ConfirmReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->confirm(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ImportArtifactFromPath(
    grpc::ServerContext* ctx,
    const v2::ImportArtifactFromPathRequest* req,
    v2::ImportArtifactFromPathResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"ImportArtifactFromPath", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->import_artifact_from_path(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ResolvePublicDiskSource(
    grpc::ServerContext* ctx,
    const v2::ResolvePublicDiskSourceRequest* req,
    v2::ResolvePublicDiskSourceResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"ResolvePublicDiskSource", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->resolve_public_disk_source(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::PromoteMountedSourceArtifact(
    grpc::ServerContext* ctx,
    const v2::PromoteMountedSourceArtifactRequest* req,
    v2::PromoteMountedSourceArtifactResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"PromoteMountedSourceArtifact", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->promote_mounted_source_artifact(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ImportArtifactFromPathStream(
    grpc::ServerContext* ctx,
    const v2::ImportArtifactFromPathRequest* req,
    grpc::ServerWriter<v2::ImportArtifactFromPathStreamEvent>* writer) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"ImportArtifactFromPathStream", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->import_artifact_from_path_stream(rctx, *req, *writer);
}

Status StoreDaemonServiceImpl::QueryReplicaStatus(
    grpc::ServerContext* ctx,
    const v2::QueryReplicaStatusRequest* req,
    v2::QueryReplicaStatusResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"QueryReplicaStatus", *ctx, opts_.allow_high_card_attrs};
  return replica_session_controller_->query_replica_status(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::WaitReplicaStatus(
    grpc::ServerContext* ctx,
    const v2::WaitReplicaStatusRequest* req,
    v2::WaitReplicaStatusResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"WaitReplicaStatus", *ctx, opts_.allow_high_card_attrs};
  return replica_session_controller_->wait_replica_status(rctx, *ctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ReleaseReplica(
    grpc::ServerContext* ctx,
    const v2::ReleaseReplicaRequest* req,
    v2::ReleaseReplicaResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"ReleaseReplica", *ctx, opts_.allow_high_card_attrs};
  return replica_session_controller_->release_replica(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::CreatePlacementLease(
    grpc::ServerContext* ctx,
    const v2::CreatePlacementLeaseRequest* req,
    v2::CreatePlacementLeaseResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"CreatePlacementLease", *ctx, opts_.allow_high_card_attrs};
  return lease_controller_->create_placement_lease(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RenewPlacementLease(
    grpc::ServerContext* ctx,
    const v2::RenewPlacementLeaseRequest* req,
    v2::RenewPlacementLeaseResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"RenewPlacementLease", *ctx, opts_.allow_high_card_attrs};
  return lease_controller_->renew_placement_lease(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ReleasePlacementLease(
    grpc::ServerContext* ctx,
    const v2::ReleasePlacementLeaseRequest* req,
    v2::ReleasePlacementLeaseResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"ReleasePlacementLease", *ctx, opts_.allow_high_card_attrs};
  return lease_controller_->release_placement_lease(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::AcquireRetentionHandle(
    grpc::ServerContext* ctx,
    const v2::AcquireRetentionHandleRequest* req,
    v2::AcquireRetentionHandleResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"AcquireRetentionHandle", *ctx, opts_.allow_high_card_attrs};
  return lease_controller_->acquire_retention_handle(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RenewRetentionHandle(
    grpc::ServerContext* ctx,
    const v2::RenewRetentionHandleRequest* req,
    v2::RenewRetentionHandleResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"RenewRetentionHandle", *ctx, opts_.allow_high_card_attrs};
  return lease_controller_->renew_retention_handle(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ReleaseRetentionHandle(
    grpc::ServerContext* ctx,
    const v2::ReleaseRetentionHandleRequest* req,
    v2::ReleaseRetentionHandleResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"ReleaseRetentionHandle", *ctx, opts_.allow_high_card_attrs};
  return lease_controller_->release_retention_handle(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::PublishReplicaKey(
    grpc::ServerContext* ctx,
    const v2::PublishReplicaKeyRequest* req,
    v2::PublishReplicaKeyResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"PublishReplicaKey", *ctx, opts_.allow_high_card_attrs};
  return key_mapping_controller_->publish_replica_key(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ResolveKeyMapping(
    grpc::ServerContext* ctx,
    const v2::ResolveKeyMappingRequest* req,
    v2::ResolveKeyMappingResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"ResolveKeyMapping", *ctx, opts_.allow_high_card_attrs};
  return key_mapping_controller_->resolve_key_mapping(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::SwapKeyMapping(
    grpc::ServerContext* ctx,
    const v2::SwapKeyMappingRequest* req,
    v2::SwapKeyMappingResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"SwapKeyMapping", *ctx, opts_.allow_high_card_attrs};
  return key_mapping_controller_->swap_key_mapping(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::GetArtifactIndexById(
    grpc::ServerContext* ctx,
    const v2::GetArtifactIndexByIdRequest* req,
    v2::GetArtifactIndexByIdResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"GetArtifactIndexById", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->get_artifact_index_by_id(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ListArtifactLayouts(
    grpc::ServerContext* ctx,
    const v2::ListArtifactLayoutsRequest* req,
    v2::ListArtifactLayoutsResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"ListArtifactLayouts", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->list_artifact_layouts(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::BatchExists(
    grpc::ServerContext* ctx,
    const v2::BatchExistsRequest* req,
    v2::BatchExistsResponse* resp) {
  RpcContext rctx{"BatchExists", *ctx, opts_.allow_high_card_attrs};
  return byte_artifact_controller_->batch_exists(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::BatchGetIntoRegion(
    grpc::ServerContext* ctx,
    const v2::BatchGetIntoRegionRequest* req,
    v2::BatchGetIntoRegionResponse* resp) {
  RpcContext rctx{"BatchGetIntoRegion", *ctx, opts_.allow_high_card_attrs};
  return byte_artifact_controller_->batch_get_into_region(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::BatchPutIfAbsentFromRegion(
    grpc::ServerContext* ctx,
    const v2::BatchPutIfAbsentFromRegionRequest* req,
    v2::BatchPutIfAbsentFromRegionResponse* resp) {
  RpcContext rctx{"BatchPutIfAbsentFromRegion", *ctx, opts_.allow_high_card_attrs};
  return byte_artifact_controller_->batch_put_if_absent_from_region(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::BatchTouchTtl(
    grpc::ServerContext* ctx,
    const v2::BatchTouchTtlRequest* req,
    v2::BatchTouchTtlResponse* resp) {
  RpcContext rctx{"BatchTouchTtl", *ctx, opts_.allow_high_card_attrs};
  return byte_artifact_controller_->batch_touch_ttl(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::HomeBatchExists(
    grpc::ServerContext* ctx,
    const v2::HomeBatchExistsRequest* req,
    v2::HomeBatchExistsResponse* resp) {
  RpcContext rctx{"HomeBatchExists", *ctx, opts_.allow_high_card_attrs};
  return byte_artifact_controller_->home_batch_exists(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::HomeBatchGet(
    grpc::ServerContext* ctx,
    const v2::HomeBatchGetRequest* req,
    v2::HomeBatchGetResponse* resp) {
  RpcContext rctx{"HomeBatchGet", *ctx, opts_.allow_high_card_attrs};
  return byte_artifact_controller_->home_batch_get(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::HomeBatchPutIfAbsent(
    grpc::ServerContext* ctx,
    const v2::HomeBatchPutIfAbsentRequest* req,
    v2::HomeBatchPutIfAbsentResponse* resp) {
  RpcContext rctx{"HomeBatchPutIfAbsent", *ctx, opts_.allow_high_card_attrs};
  return byte_artifact_controller_->home_batch_put_if_absent(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::FetchPayloadRefChunk(
    grpc::ServerContext* ctx,
    const v2::FetchPayloadRefChunkRequest* req,
    v2::FetchPayloadRefChunkResponse* resp) {
  RpcContext rctx{"FetchPayloadRefChunk", *ctx, opts_.allow_high_card_attrs};
  return transport_controller_->fetch_payload_ref_chunk(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::FetchBatchPayloadRefChunk(
    grpc::ServerContext* ctx,
    const v2::FetchBatchPayloadRefChunkRequest* req,
    v2::FetchBatchPayloadRefChunkResponse* resp) {
  RpcContext rctx{"FetchBatchPayloadRefChunk", *ctx, opts_.allow_high_card_attrs};
  return transport_controller_->fetch_batch_payload_ref_chunk(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RouteAuthorityStage(
    grpc::ServerContext* ctx,
    const v2::RouteAuthorityStageRequest* req,
    v2::RouteAuthorityStageResponse* resp) {
  RpcContext rctx{"RouteAuthorityStage", *ctx, opts_.allow_high_card_attrs};
  return transport_controller_->route_authority_stage(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::HomeBatchTouchTtl(
    grpc::ServerContext* ctx,
    const v2::HomeBatchTouchTtlRequest* req,
    v2::HomeBatchTouchTtlResponse* resp) {
  RpcContext rctx{"HomeBatchTouchTtl", *ctx, opts_.allow_high_card_attrs};
  return byte_artifact_controller_->home_batch_touch_ttl(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::SealAssembly(
    grpc::ServerContext* ctx,
    const v2::SealAssemblyRequest* req,
    v2::SealAssemblyResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"SealAssembly", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->seal_assembly(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::StartSealAssembly(
    grpc::ServerContext* ctx,
    const v2::StartSealAssemblyRequest* req,
    v2::StartSealAssemblyResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"StartSealAssembly", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->start_seal_assembly(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::GetOperation(
    grpc::ServerContext* ctx,
    const tensorcast::operation::v1::GetOperationRequest* req,
    tensorcast::operation::v1::GetOperationResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"GetOperation", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->get_operation(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::WaitOperation(
    grpc::ServerContext* ctx,
    const v2::WaitOperationRequest* req,
    v2::WaitOperationResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"WaitOperation", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->wait_operation(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::StartPersistence(
    grpc::ServerContext* ctx,
    const v2::StartPersistenceRequest* req,
    v2::StartPersistenceResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"StartPersistence", *ctx, opts_.allow_high_card_attrs};
  return persistence_rpc_controller_->start_persistence(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::QueryPersistenceStatus(
    grpc::ServerContext* ctx,
    const v2::QueryPersistenceStatusRequest* req,
    v2::QueryPersistenceStatusResponse* resp) {
  RpcContext rctx{"QueryPersistenceStatus", *ctx, opts_.allow_high_card_attrs};
  return persistence_rpc_controller_->query_persistence_status(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::UnloadReplica(
    grpc::ServerContext* ctx,
    const v2::UnloadReplicaRequest* req,
    v2::UnloadReplicaResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"UnloadReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->unload(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::GetServerConfig(
    grpc::ServerContext* ctx,
    const v2::GetServerConfigRequest* /*req*/,
    v2::GetServerConfigResponse* resp) {
  RpcContext rctx{"GetServerConfig", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_server_config(rctx, *resp);
}

Status StoreDaemonServiceImpl::WaitReplicaVerification(
    grpc::ServerContext* ctx,
    const v2::WaitReplicaVerificationRequest* req,
    v2::WaitReplicaVerificationResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"WaitReplicaVerification", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->wait_verification(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::LockTransportChunks(
    grpc::ServerContext* ctx,
    const v2::LockTransportChunksRequest* req,
    v2::LockTransportChunksResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"LockTransportChunks", *ctx, opts_.allow_high_card_attrs};
  return transport_controller_->lock(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::PublishTargetReplica(
    grpc::ServerContext* ctx,
    const v2::PublishTargetReplicaRequest* req,
    v2::PublishTargetReplicaResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"PublishTargetReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->publish_target_replica(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::StartPublishTargetReplica(
    grpc::ServerContext* ctx,
    const v2::PublishTargetReplicaRequest* req,
    v2::StartPublishTargetReplicaResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"StartPublishTargetReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->start_publish_target_replica(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::UnlockTransportChunks(
    grpc::ServerContext* ctx,
    const v2::UnlockTransportChunksRequest* req,
    v2::UnlockTransportChunksResponse* /*resp*/) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"UnlockTransportChunks", *ctx, opts_.allow_high_card_attrs};
  v2::UnlockTransportChunksResponse dummy;
  return transport_controller_->unlock(rctx, *req, dummy);
}

Status StoreDaemonServiceImpl::BeginRegisterArtifact(
    grpc::ServerContext* ctx,
    const v2::BeginRegisterArtifactRequest* req,
    v2::BeginRegisterArtifactResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"BeginRegisterArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->begin(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::CommitRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v2::CommitRegisteredArtifactRequest* req,
    v2::CommitRegisteredArtifactResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"CommitRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->commit(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::AbortRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v2::AbortRegisteredArtifactRequest* req,
    v2::AbortRegisteredArtifactResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"AbortRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->abort(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::FeedRegisterArtifactStream(
    grpc::ServerContext* ctx,
    ::grpc::ServerReader<v2::FeedRegisterArtifactStreamRequest>* reader,
    v2::FeedRegisterArtifactStreamResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"FeedRegisterArtifactStream", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->feed_stream(rctx, *reader, *resp);
}

grpc::Status StoreDaemonServiceImpl::feed_register_artifact_stream_vector(
    const std::vector<v2::FeedRegisterArtifactStreamRequest>& reqs) {
  return registration_controller_->feed_vector(reqs);
}

Status StoreDaemonServiceImpl::KeepAliveRegisterArtifact(
    grpc::ServerContext* ctx,
    const v2::KeepAliveRegisterArtifactRequest* req,
    v2::KeepAliveRegisterArtifactResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"KeepAliveRegisterArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->keep_alive(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RevokeRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v2::RevokeRegisteredArtifactRequest* req,
    v2::RevokeRegisteredArtifactResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  RpcContext rctx{"RevokeRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->revoke(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::GetWorkerStatus(
    grpc::ServerContext* ctx,
    const v2::GetWorkerStatusRequest* /*req*/,
    v2::GetWorkerStatusResponse* resp) {
  RpcContext rctx{"GetWorkerStatus", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_worker_status(rctx, *resp);
}

Status StoreDaemonServiceImpl::ListDirectoryWorkers(
    grpc::ServerContext* ctx,
    const v2::ListDirectoryWorkersRequest* req,
    v2::ListDirectoryWorkersResponse* resp) {
  RpcContext rctx{"ListDirectoryWorkers", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->list_directory_workers(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ListDirectoryInstances(
    grpc::ServerContext* ctx,
    const v2::ListDirectoryInstancesRequest* req,
    v2::ListDirectoryInstancesResponse* resp) {
  RpcContext rctx{"ListDirectoryInstances", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->list_directory_instances(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ResolveInstanceExecution(
    grpc::ServerContext* ctx,
    const v2::ResolveInstanceExecutionRequest* req,
    v2::ResolveInstanceExecutionResponse* resp) {
  RpcContext rctx{"ResolveInstanceExecution", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->resolve_instance_execution(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::GetDetailedStatus(
    grpc::ServerContext* ctx,
    const v2::GetDetailedStatusRequest* /*req*/,
    v2::GetDetailedStatusResponse* resp) {
  RpcContext rctx{"GetDetailedStatus", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_detailed_status(rctx, *resp);
}

Status StoreDaemonServiceImpl::GetLoadedReplicasV2(
    grpc::ServerContext* ctx,
    const v2::GetLoadedReplicasV2Request* req,
    v2::GetLoadedReplicasV2Response* resp) {
  RpcContext rctx{"GetLoadedReplicasV2", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_loaded_replicas_v2(rctx, *req, *resp, opts_.use_cursor_pagination);
}

} // namespace tensorcast::daemon
