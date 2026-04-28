// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_controller.h"

#include <algorithm>
#include <utility>

#include "absl/time/time.h"
#include "daemon/util/status_utils.h"

namespace tensorcast::daemon {

using status_utils::to_grpc_status;

namespace {

void merge_operation_ref_metadata(
    const tensorcast::operation::v1::OperationRef& source,
    tensorcast::operation::v1::OperationRef* target) {
  if (target == nullptr) {
    return;
  }
  if (target->operation_id().empty() && !source.operation_id().empty()) {
    target->set_operation_id(source.operation_id());
  }
  if (target->kind().empty() && !source.kind().empty()) {
    target->set_kind(source.kind());
  }
  if (target->target_artifact_id().empty() && !source.target_artifact_id().empty()) {
    target->set_target_artifact_id(source.target_artifact_id());
  }
  if (target->authority_scope_kind().empty() && !source.authority_scope_kind().empty()) {
    target->set_authority_scope_kind(source.authority_scope_kind());
  }
  if (target->authority_scope_id().empty() && !source.authority_scope_id().empty()) {
    target->set_authority_scope_id(source.authority_scope_id());
  }
  if (target->attachment_kind().empty() && !source.attachment_kind().empty()) {
    target->set_attachment_kind(source.attachment_kind());
  }
  if (target->recovery_class().empty() && !source.recovery_class().empty()) {
    target->set_recovery_class(source.recovery_class());
  }
  if (target->fencing_digest().empty() && !source.fencing_digest().empty()) {
    target->set_fencing_digest(source.fencing_digest());
  }
}

bool operation_state_is_terminal(tensorcast::operation::v1::OperationState state) {
  using State = tensorcast::operation::v1::OperationState;
  return state == State::OPERATION_STATE_SUCCESS || state == State::OPERATION_STATE_FAILED ||
      state == State::OPERATION_STATE_CANCELLED;
}

} // namespace

MaterializationController::MaterializationController(Dep d)
    : global_store_client_(d.global_store_client),
      shutdown_signal_(&d.shutdown_signal),
      assembly_operation_service_(
          AssemblyOperationService::Dep{
              .engine = d.engine,
              .devices = d.devices,
              .shutdown_signal = d.shutdown_signal,
              .async_runtime = d.async_runtime,
              .identity = d.identity,
              .bindings = d.binding_registry,
              .global_store_client = d.global_store_client,
              .lip_manager = &d.lip_manager,
              .lifecycle = d.lifecycle,
              .post_seal_policy = d.post_seal_policy,
              .await_state_sync_barrier = d.await_state_sync_barrier,
          }),
      disk_artifact_service_(
          DiskArtifactService::Dep{
              .engine = d.engine,
              .source_registry = d.disk_imports,
              .shutdown_signal = d.shutdown_signal,
              .global_store_client = d.global_store_client,
              .storage_path = d.storage_path,
              .public_disk_source_policy = d.public_disk_source_policy,
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
              .async_runtime = d.async_runtime,
              .identity = d.identity,
              .external_target_access_service = d.external_target_access_service,
              .global_store_client = d.global_store_client,
              .max_concurrency = d.max_concurrency,
              .capability_tokens = d.capability_tokens,
              .external_target_verification_enabled = d.external_target_verification_enabled,
              .storage_path = d.storage_path,
              .await_state_sync_barrier = d.await_state_sync_barrier,
          }),
      owner_binding_service_(
          OwnedBindingService::Dep{
              .engine = d.engine,
              .devices = d.devices,
              .disk_imports = d.disk_imports,
              .bindings = d.binding_registry,
              .shutdown_signal = d.shutdown_signal,
              .async_runtime = d.async_runtime,
              .identity = d.identity,
              .global_store_client = d.global_store_client,
              .lifecycle = d.lifecycle,
              .handle_leases = d.handle_leases,
              .registration_manager = &d.registration_manager,
              .lip_manager = &d.lip_manager,
              .refs = &d.refs,
              .regions = &d.regions,
              .max_concurrency = d.max_concurrency,
              .capability_tokens = d.capability_tokens,
              .target_materialization_service = &target_materialization_service_,
              .storage_path = d.storage_path,
              .await_state_sync_barrier = d.await_state_sync_barrier,
          }) {
  // Register publish replay admission as one child-owner policy behind the
  // shared observation-path dispatcher. The dispatcher itself does not own
  // publish semantics.
  public_operation_admission_service_.register_handler(
      std::string(TargetPublishService::public_operation_kind()),
      [this](const tensorcast::operation::v1::OperationRef& operation_ref, absl::Time now) {
        return target_materialization_service_.admit_public_operation(operation_ref, now);
      });
}

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

grpc::Status MaterializationController::create_binding(
    RpcContext& rctx,
    const v2::CreateBindingRequest& req,
    v2::CreateBindingResponse& resp) {
  return owner_binding_service_.create_binding(rctx, req, resp);
}

grpc::Status MaterializationController::commit_binding_artifact(
    RpcContext& rctx,
    const v2::CommitBindingArtifactRequest& req,
    v2::CommitBindingArtifactResponse& resp) {
  return owner_binding_service_.commit_binding_artifact(rctx, req, resp);
}

grpc::Status MaterializationController::begin_binding_update(
    RpcContext& rctx,
    const v2::BeginBindingUpdateRequest& req,
    v2::BeginBindingUpdateResponse& resp) {
  return owner_binding_service_.begin_binding_update(rctx, req, resp);
}

grpc::Status MaterializationController::submit_binding_contribution(
    RpcContext& rctx,
    const v2::SubmitBindingContributionRequest& req,
    v2::SubmitBindingContributionResponse& resp) {
  return owner_binding_service_.submit_binding_contribution(rctx, req, resp);
}

grpc::Status MaterializationController::seal_binding(
    RpcContext& rctx,
    const v2::SealBindingRequest& req,
    v2::SealBindingResponse& resp) {
  return owner_binding_service_.seal_binding(rctx, req, resp);
}

grpc::Status MaterializationController::promote_binding_current_value(
    RpcContext& rctx,
    const v2::PromoteBindingCurrentValueRequest& req,
    v2::PromoteBindingCurrentValueResponse& resp) {
  return owner_binding_service_.promote_binding_current_value(rctx, req, resp);
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

grpc::Status MaterializationController::start_publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::StartPublishTargetReplicaResponse& resp) {
  return target_materialization_service_.start_publish_target_replica(rctx, req, resp);
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

grpc::Status MaterializationController::resolve_public_disk_source(
    RpcContext& rctx,
    const v2::ResolvePublicDiskSourceRequest& req,
    v2::ResolvePublicDiskSourceResponse& resp) {
  return disk_artifact_service_.resolve_public_disk_source(rctx, req, resp);
}

grpc::Status MaterializationController::promote_mounted_source_artifact(
    RpcContext& rctx,
    const v2::PromoteMountedSourceArtifactRequest& req,
    v2::PromoteMountedSourceArtifactResponse& resp) {
  return disk_artifact_service_.promote_mounted_source_artifact(rctx, req, resp);
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

grpc::Status MaterializationController::list_artifact_layouts(
    RpcContext& rctx,
    const v2::ListArtifactLayoutsRequest& req,
    v2::ListArtifactLayoutsResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.artifact_id());
  if (req.artifact_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }
  auto layouts_or = global_store_client_->list_artifact_layouts(req.artifact_id());
  if (!layouts_or.ok()) {
    return to_grpc_status(layouts_or.status());
  }
  for (const auto& layout_id : *layouts_or) {
    resp.add_layout_ids(layout_id);
  }
  return grpc::Status::OK;
}

grpc::Status MaterializationController::seal_assembly(
    RpcContext& rctx,
    const v2::SealAssemblyRequest& req,
    v2::SealAssemblyResponse& resp) {
  return assembly_operation_service_.seal_assembly(rctx, req, resp);
}

grpc::Status MaterializationController::start_assembly_attempt(
    RpcContext& rctx,
    const v2::StartAssemblyAttemptRequest& req,
    v2::StartAssemblyAttemptResponse& resp) {
  return assembly_operation_service_.start_assembly_attempt(rctx, req, resp);
}

grpc::Status MaterializationController::seal_assembly_attempt(
    RpcContext& rctx,
    const v2::SealAssemblyAttemptRequest& req,
    v2::SealAssemblyAttemptResponse& resp) {
  return assembly_operation_service_.seal_assembly_attempt(rctx, req, resp);
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
  auto& span = rctx.span();
  span->SetAttribute("tc.operation.id", req.operation_id());

  if (req.operation_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "operation_id is required"};
  }
  if (shutdown_signal_ != nullptr && shutdown_signal_->is_shutting_down()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  auto op_or = global_store_client_->get_operation(req);
  if (!op_or.ok()) {
    return to_grpc_status(op_or.status());
  }
  if (req.has_ref()) {
    merge_operation_ref_metadata(req.ref(), op_or->mutable_ref());
  }
  auto admit_status = public_operation_admission_service_.admit(op_or->ref(), absl::Now());
  if (!admit_status.ok()) {
    return to_grpc_status(admit_status);
  }
  resp = std::move(*op_or);
  rctx.mark_success();
  return grpc::Status::OK;
}

grpc::Status MaterializationController::wait_operation(
    RpcContext& rctx,
    const v2::WaitOperationRequest& req,
    v2::WaitOperationResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.operation.id", req.operation_id());

  if (req.operation_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "operation_id is required"};
  }
  if (shutdown_signal_ != nullptr && shutdown_signal_->is_shutting_down()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  const uint64_t timeout_ms = req.timeout_ms();
  const absl::Time start = absl::Now();
  const absl::Time deadline = timeout_ms > 0 ? start + absl::Milliseconds(timeout_ms) : absl::InfiniteFuture();
  absl::Duration sleep = absl::Milliseconds(50);

  tensorcast::operation::v1::GetOperationRequest get_req;
  get_req.set_operation_id(req.operation_id());
  if (req.has_ref()) {
    get_req.mutable_ref()->CopyFrom(req.ref());
  }

  while (absl::Now() < deadline) {
    auto op_or = global_store_client_->get_operation(get_req);
    if (!op_or.ok()) {
      return to_grpc_status(op_or.status());
    }
    if (req.has_ref()) {
      merge_operation_ref_metadata(req.ref(), op_or->mutable_ref());
    }
    auto admit_status = public_operation_admission_service_.admit(op_or->ref(), absl::Now());
    if (!admit_status.ok()) {
      return to_grpc_status(admit_status);
    }
    const auto state = op_or->status().state();
    resp.mutable_operation()->Swap(&(*op_or));
    if (operation_state_is_terminal(state)) {
      rctx.mark_success();
      return grpc::Status::OK;
    }
    absl::SleepFor(sleep);
    sleep = std::min(sleep * 12 / 10, absl::Milliseconds(500));
  }

  auto op_or = global_store_client_->get_operation(get_req);
  if (!op_or.ok()) {
    return to_grpc_status(op_or.status());
  }
  if (req.has_ref()) {
    merge_operation_ref_metadata(req.ref(), op_or->mutable_ref());
  }
  auto admit_status = target_materialization_service_.admit_public_operation(op_or->ref(), absl::Now());
  if (!admit_status.ok()) {
    return to_grpc_status(admit_status);
  }
  resp.mutable_operation()->Swap(&(*op_or));
  rctx.mark_success();
  return grpc::Status::OK;
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
