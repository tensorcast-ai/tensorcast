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
              .global_store_client = d.global_store_client,
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
              .global_store_client = d.global_store_client,
              .lifecycle = d.lifecycle,
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
              .lifecycle = *d.lifecycle,
              .lifecycle_kernel = *d.lifecycle_kernel,
              .shutdown_signal = d.shutdown_signal,
              .identity = d.identity,
              .external_target_access_service = d.external_target_access_service,
              .global_store_client = d.global_store_client,
              .max_concurrency = d.max_concurrency,
              .capability_tokens = d.capability_tokens,
              .external_target_verification_enabled = d.external_target_verification_enabled,
              .storage_path = d.storage_path,
          }),
      owner_binding_service_(
          OwnedBindingService::Dep{
              .engine = d.engine,
              .devices = d.devices,
              .disk_imports = d.disk_imports,
              .bindings = d.binding_registry,
              .shutdown_signal = d.shutdown_signal,
              .identity = d.identity,
              .global_store_client = d.global_store_client,
              .handle_leases = d.handle_leases,
              .capability_tokens = d.capability_tokens,
              .target_materialization_service = &target_materialization_service_,
              .storage_path = d.storage_path,
          }) {}

absl::StatusOr<TargetPublicationRegistry::Record> MaterializationController::insert_target_publication_for_testing(
    TargetPublicationRegistry::Record record) {
  return target_materialization_service_.insert_target_publication_for_testing(std::move(record));
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

grpc::Status MaterializationController::create_owned_binding(
    RpcContext& rctx,
    const v2::CreateOwnedBindingRequest& req,
    v2::CreateOwnedBindingResponse& resp) {
  return owner_binding_service_.create_owned_binding(rctx, req, resp);
}

grpc::Status MaterializationController::refill_owned_binding(
    RpcContext& rctx,
    const v2::RefillOwnedBindingRequest& req,
    v2::RefillOwnedBindingResponse& resp) {
  return owner_binding_service_.refill_owned_binding(rctx, req, resp);
}

grpc::Status MaterializationController::close_owned_binding(
    RpcContext& rctx,
    const v2::CloseOwnedBindingRequest& req,
    v2::CloseOwnedBindingResponse& resp) {
  return owner_binding_service_.close_owned_binding(rctx, req, resp);
}

grpc::Status MaterializationController::publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::PublishTargetReplicaResponse& resp) {
  return target_materialization_service_.publish_target_replica(rctx, req, resp);
}

absl::StatusOr<TargetPublishService::TargetPublicationFrontDoorContext> MaterializationController::
    inspect_target_publication_context_for_testing(const v2::PublishTargetReplicaRequest& req, absl::Time now) {
  return target_materialization_service_.inspect_target_publication_context_for_testing(req, now);
}

absl::StatusOr<RoutedAuthorityRequest> MaterializationController::
    build_target_publication_workflow_routed_request_for_testing(
        const v2::PublishTargetReplicaRequest& req,
        absl::Time now) const {
  return target_materialization_service_.build_target_publication_workflow_routed_request_for_testing(req, now);
}

absl::StatusOr<RoutedAuthorityRequest> MaterializationController::
    build_target_publication_workflow_continuation_request_for_testing(
        const RoutedAuthorityRequest& routed_request,
        const OwnerStageReply& workflow_gate_reply) const {
  return target_materialization_service_.build_target_publication_workflow_continuation_request_for_testing(
      routed_request, workflow_gate_reply);
}

absl::StatusOr<std::optional<OwnerStageReply>> MaterializationController::maybe_route_authority_stage(
    const RoutedAuthorityRequest& routed_request,
    absl::Time now) {
  return target_materialization_service_.maybe_route_authority_stage(routed_request, now);
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
