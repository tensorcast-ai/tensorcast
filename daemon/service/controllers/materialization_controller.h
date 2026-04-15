// Copyright (c) 2025-2026, TensorCast Team.

// MaterializationController: handles materialize/by_key/get_artifact_index RPCs

#pragma once

#include <filesystem>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/common/capability_token.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/service/controllers/assembly_operation_service.h"
#include "daemon/service/controllers/disk_artifact_service.h"
#include "daemon/service/controllers/external_target_access_service.h"
#include "daemon/service/controllers/owned_binding_service.h"
#include "daemon/service/controllers/public_operation_admission_service.h"
#include "daemon/service/controllers/replica_lifecycle_service.h"
#include "daemon/service/controllers/replica_materialization_service.h"
#include "daemon/service/controllers/target_materialization_service.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/artifact_source_registry.h"
#include "daemon/state/binding_registry.h"
#include "daemon/state/daemon_options.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/handle_lease_registry.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_bridge.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/registration_manager.h"
#include "daemon/state/routed_authority_protocol.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/sessions_service.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/target_publication_registry.h"
#include "daemon/state/worker_identity_store.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class MaterializationController {
 public:
  struct Dep {
    store::StoreEngine& engine;
    RefTracker& refs;
    SessionsService& sessions;
    LipBridge& lip;
    LipManager& lip_manager;
    RegistrationManager& registration_manager;
    DeviceResolver& devices;
    IpcRegionRegistry& regions;
    ArtifactSourceRegistry& disk_imports;
    BindingRegistry& binding_registry;
    ShutdownSignal& shutdown_signal;
    common::AsyncRuntime& async_runtime;
    WorkerIdentityStore& identity;
    ExternalTargetAccessService& external_target_access_service;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    uint32_t max_concurrency{4};
    SessionLifecycleManager* lifecycle{nullptr};
    LifecycleKernel* lifecycle_kernel{nullptr};
    HandleLeaseRegistry* handle_leases{nullptr};
    common::CapabilityTokenManager* capability_tokens{nullptr};
    bool cpu_shared_memory_enabled{true};
    bool external_target_verification_enabled{false};
    std::filesystem::path storage_path;
    DaemonOptions::PublicDiskSourcePolicy public_disk_source_policy{};
    DaemonOptions::PostSealPolicy post_seal_policy{};
  };

  explicit MaterializationController(Dep d);

  grpc::Status materialize_replica(
      RpcContext& rctx,
      const v2::MaterializeReplicaRequest& req,
      v2::MaterializeReplicaResponse& resp);

  grpc::Status materialize_into_target(
      RpcContext& rctx,
      const v2::MaterializeIntoTargetRequest& req,
      v2::MaterializeIntoTargetResponse& resp);

  grpc::Status materialize_into_mapped_target(
      RpcContext& rctx,
      const v2::MaterializeIntoMappedTargetRequest& req,
      v2::MaterializeIntoTargetResponse& resp);

  grpc::Status create_owned_binding(
      RpcContext& rctx,
      const v2::CreateOwnedBindingRequest& req,
      v2::CreateOwnedBindingResponse& resp);

  grpc::Status create_binding(RpcContext& rctx, const v2::CreateBindingRequest& req, v2::CreateBindingResponse& resp);

  grpc::Status commit_binding_artifact(
      RpcContext& rctx,
      const v2::CommitBindingArtifactRequest& req,
      v2::CommitBindingArtifactResponse& resp);

  grpc::Status begin_binding_update(
      RpcContext& rctx,
      const v2::BeginBindingUpdateRequest& req,
      v2::BeginBindingUpdateResponse& resp);

  grpc::Status submit_binding_contribution(
      RpcContext& rctx,
      const v2::SubmitBindingContributionRequest& req,
      v2::SubmitBindingContributionResponse& resp);

  grpc::Status seal_binding(RpcContext& rctx, const v2::SealBindingRequest& req, v2::SealBindingResponse& resp);

  grpc::Status promote_binding_current_value(
      RpcContext& rctx,
      const v2::PromoteBindingCurrentValueRequest& req,
      v2::PromoteBindingCurrentValueResponse& resp);

  grpc::Status refill_owned_binding(
      RpcContext& rctx,
      const v2::RefillOwnedBindingRequest& req,
      v2::RefillOwnedBindingResponse& resp);

  grpc::Status close_owned_binding(
      RpcContext& rctx,
      const v2::CloseOwnedBindingRequest& req,
      v2::CloseOwnedBindingResponse& resp);

  grpc::Status publish_target_replica(
      RpcContext& rctx,
      const v2::PublishTargetReplicaRequest& req,
      v2::PublishTargetReplicaResponse& resp);

  grpc::Status start_publish_target_replica(
      RpcContext& rctx,
      const v2::PublishTargetReplicaRequest& req,
      v2::StartPublishTargetReplicaResponse& resp);

  [[nodiscard]] absl::StatusOr<TargetPublishService::TargetPublicationFrontDoorContext>
  inspect_target_publication_context_for_testing(const v2::PublishTargetReplicaRequest& req, absl::Time now);

  [[nodiscard]] absl::StatusOr<RoutedAuthorityRequest> build_target_publication_workflow_routed_request_for_testing(
      const v2::PublishTargetReplicaRequest& req,
      absl::Time now) const;

  [[nodiscard]] absl::StatusOr<RoutedAuthorityRequest>
  build_target_publication_workflow_continuation_request_for_testing(
      const RoutedAuthorityRequest& routed_request,
      const OwnerStageReply& workflow_gate_reply) const;

  [[nodiscard]] absl::StatusOr<std::optional<OwnerStageReply>> maybe_route_authority_stage(
      const RoutedAuthorityRequest& routed_request,
      absl::Time now);

  grpc::Status import_artifact_from_path(
      RpcContext& rctx,
      const v2::ImportArtifactFromPathRequest& req,
      v2::ImportArtifactFromPathResponse& resp);

  grpc::Status resolve_public_disk_source(
      RpcContext& rctx,
      const v2::ResolvePublicDiskSourceRequest& req,
      v2::ResolvePublicDiskSourceResponse& resp);

  grpc::Status promote_mounted_source_artifact(
      RpcContext& rctx,
      const v2::PromoteMountedSourceArtifactRequest& req,
      v2::PromoteMountedSourceArtifactResponse& resp);

  grpc::Status import_artifact_from_path_stream(
      RpcContext& rctx,
      const v2::ImportArtifactFromPathRequest& req,
      grpc::ServerWriter<v2::ImportArtifactFromPathStreamEvent>& writer);

  grpc::Status get_artifact_index_by_id(
      RpcContext& rctx,
      const v2::GetArtifactIndexByIdRequest& req,
      v2::GetArtifactIndexByIdResponse& resp);

  grpc::Status list_artifact_layouts(
      RpcContext& rctx,
      const v2::ListArtifactLayoutsRequest& req,
      v2::ListArtifactLayoutsResponse& resp);

  grpc::Status seal_assembly(RpcContext& rctx, const v2::SealAssemblyRequest& req, v2::SealAssemblyResponse& resp);

  grpc::Status start_assembly_attempt(
      RpcContext& rctx,
      const v2::StartAssemblyAttemptRequest& req,
      v2::StartAssemblyAttemptResponse& resp);

  grpc::Status seal_assembly_attempt(
      RpcContext& rctx,
      const v2::SealAssemblyAttemptRequest& req,
      v2::SealAssemblyAttemptResponse& resp);

  grpc::Status start_seal_assembly(
      RpcContext& rctx,
      const v2::StartSealAssemblyRequest& req,
      v2::StartSealAssemblyResponse& resp);

  grpc::Status get_operation(
      RpcContext& rctx,
      const tensorcast::operation::v1::GetOperationRequest& req,
      tensorcast::operation::v1::GetOperationResponse& resp);

  grpc::Status wait_operation(RpcContext& rctx, const v2::WaitOperationRequest& req, v2::WaitOperationResponse& resp);

  grpc::Status confirm(RpcContext& rctx, const v2::ConfirmReplicaRequest& req, v2::ConfirmReplicaResponse& resp) const;

  grpc::Status unload(RpcContext& rctx, const v2::UnloadReplicaRequest& req, v2::UnloadReplicaResponse& resp);

  grpc::Status wait_verification(
      RpcContext& rctx,
      const v2::WaitReplicaVerificationRequest& req,
      v2::WaitReplicaVerificationResponse& resp);

  // Test helper: inject a target publication record without materialization.
  [[nodiscard]] absl::StatusOr<TargetPublicationRegistry::Record> insert_target_publication_for_testing(
      TargetPublicationRegistry::Record record);

 private:
  std::shared_ptr<store::components::IGlobalStoreClient> global_store_client_;
  ShutdownSignal* shutdown_signal_{nullptr};
  AssemblyOperationService assembly_operation_service_;
  DiskArtifactService disk_artifact_service_;
  ReplicaMaterializationService replica_materialization_service_;
  ReplicaLifecycleService replica_lifecycle_service_;
  TargetMaterializationService target_materialization_service_;
  OwnedBindingService owner_binding_service_;
  PublicOperationAdmissionService public_operation_admission_service_;
};

} // namespace tensorcast::daemon
