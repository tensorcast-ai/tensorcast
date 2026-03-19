// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_controller.h"

#include <utility>

namespace tensorcast::daemon {

MaterializationController::MaterializationController(Dep d)
    : assembly_operation_service_(
          AssemblyOperationService::Dep{
              .engine = d.engine,
              .devices = d.devices,
              .shutdown_signal = d.shutdown_signal,
              .async_runtime = d.async_runtime,
              .identity = d.identity,
              .global_store_client = d.global_store_client,
              .post_seal_policy = d.post_seal_policy,
          }),
      disk_artifact_service_(
          DiskArtifactService::Dep{
              .engine = d.engine,
              .source_registry = d.disk_imports,
              .shutdown_signal = d.shutdown_signal,
              .storage_path = d.storage_path,
          }),
      replica_materialization_service_(
          ReplicaMaterializationService::Dep{
              .engine = d.engine,
              .refs = d.refs,
              .sessions = d.sessions,
              .lip = d.lip,
              .devices = d.devices,
              .disk_imports = d.disk_imports,
              .shutdown_signal = d.shutdown_signal,
              .async_runtime = &d.async_runtime,
              .global_store_client = d.global_store_client,
              .lifecycle = d.lifecycle,
              .derived_view_exports = d.derived_view_exports,
              .handle_leases = d.handle_leases,
              .cpu_shared_memory_enabled = d.cpu_shared_memory_enabled,
              .post_seal_policy = d.post_seal_policy,
              .storage_path = d.storage_path,
          }),
      replica_lifecycle_service_(
          ReplicaLifecycleService::Dep{
              .engine = d.engine,
              .refs = d.refs,
              .sessions = d.sessions,
              .lifecycle = d.lifecycle,
              .devices = d.devices,
          }),
      target_materialization_service_(
          TargetMaterializationService::Dep{
              .engine = d.engine,
              .lip_manager = d.lip_manager,
              .devices = d.devices,
              .regions = d.regions,
              .disk_imports = d.disk_imports,
              .shutdown_signal = d.shutdown_signal,
              .identity = d.identity,
              .global_store_client = d.global_store_client,
              .max_concurrency = d.max_concurrency,
              .capability_tokens = d.capability_tokens,
              .external_target_verification_enabled = d.external_target_verification_enabled,
              .storage_path = d.storage_path,
          }) {}

TargetWriteRegistry::Record MaterializationController::insert_target_write_for_testing(
    TargetWriteRegistry::Record record) {
  return target_materialization_service_.insert_target_write_for_testing(std::move(record));
}

grpc::Status MaterializationController::materialize_replica(
    RpcContext& rctx,
    const v2::MaterializeReplicaRequest& req,
    v2::MaterializeReplicaResponse& resp) {
  return replica_materialization_service_.materialize_replica(rctx, req, resp);
}

grpc::Status MaterializationController::materialize_into_target(
    RpcContext& rctx,
    const v2::MaterializeIntoTargetRequest& req,
    v2::MaterializeIntoTargetResponse& resp) {
  return target_materialization_service_.materialize_into_target(rctx, req, resp);
}

grpc::Status MaterializationController::materialize_into_mapped_target(
    RpcContext& rctx,
    const v2::MaterializeIntoMappedTargetRequest& req,
    v2::MaterializeIntoTargetResponse& resp) {
  return target_materialization_service_.materialize_into_mapped_target(rctx, req, resp);
}

grpc::Status MaterializationController::publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::PublishTargetReplicaResponse& resp) {
  return target_materialization_service_.publish_target_replica(rctx, req, resp);
}

grpc::Status MaterializationController::import_artifact_from_path(
    RpcContext& rctx,
    const v2::ImportArtifactFromPathRequest& req,
    v2::ImportArtifactFromPathResponse& resp) {
  return disk_artifact_service_.import_artifact_from_path(rctx, req, resp);
}

grpc::Status MaterializationController::import_artifact_from_path_stream(
    RpcContext& rctx,
    const v2::ImportArtifactFromPathRequest& req,
    grpc::ServerWriter<v2::ImportArtifactFromPathStreamEvent>& writer) {
  return disk_artifact_service_.import_artifact_from_path_stream(rctx, req, writer);
}

grpc::Status MaterializationController::get_artifact_index_by_id(
    RpcContext& rctx,
    const v2::GetArtifactIndexByIdRequest& req,
    v2::GetArtifactIndexByIdResponse& resp) {
  return disk_artifact_service_.get_artifact_index_by_id(rctx, req, resp);
}

grpc::Status MaterializationController::seal_assembly(
    RpcContext& rctx,
    const v2::SealAssemblyRequest& req,
    v2::SealAssemblyResponse& resp) {
  return assembly_operation_service_.seal_assembly(rctx, req, resp);
}

grpc::Status MaterializationController::start_seal_assembly(
    RpcContext& rctx,
    const v2::StartSealAssemblyRequest& req,
    v2::StartSealAssemblyResponse& resp) {
  return assembly_operation_service_.start_seal_assembly(rctx, req, resp);
}

grpc::Status MaterializationController::get_operation(
    RpcContext& rctx,
    const tensorcast::operation::v1::GetOperationRequest& req,
    tensorcast::operation::v1::GetOperationResponse& resp) {
  return assembly_operation_service_.get_operation(rctx, req, resp);
}

grpc::Status MaterializationController::wait_operation(
    RpcContext& rctx,
    const v2::WaitOperationRequest& req,
    v2::WaitOperationResponse& resp) {
  return assembly_operation_service_.wait_operation(rctx, req, resp);
}

grpc::Status MaterializationController::confirm(
    RpcContext& rctx,
    const v2::ConfirmReplicaRequest& req,
    v2::ConfirmReplicaResponse& resp) const {
  return replica_lifecycle_service_.confirm(rctx, req, resp);
}

grpc::Status MaterializationController::unload(
    RpcContext& rctx,
    const v2::UnloadReplicaRequest& req,
    v2::UnloadReplicaResponse& resp) {
  return replica_lifecycle_service_.unload(rctx, req, resp);
}

grpc::Status MaterializationController::wait_verification(
    RpcContext& rctx,
    const v2::WaitReplicaVerificationRequest& req,
    v2::WaitReplicaVerificationResponse& resp) {
  return replica_lifecycle_service_.wait_verification(rctx, req, resp);
}

} // namespace tensorcast::daemon
