// Copyright (c) 2025-2026, TensorCast Team.

// MaterializationController: handles materialize/by_key/get_artifact_index RPCs

#pragma once

#include <filesystem>
#include <memory>

#include "core/common/async_runtime.h"
#include "core/common/capability_token.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/service/controllers/assembly_operation_service.h"
#include "daemon/service/controllers/disk_artifact_service.h"
#include "daemon/service/controllers/replica_lifecycle_service.h"
#include "daemon/service/controllers/replica_materialization_service.h"
#include "daemon/service/controllers/target_materialization_service.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/artifact_source_registry.h"
#include "daemon/state/daemon_options.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/handle_lease_registry.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_bridge.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/sessions_service.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/target_write_registry.h"
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
    DeviceResolver& devices;
    IpcRegionRegistry& regions;
    ArtifactSourceRegistry& disk_imports;
    ShutdownSignal& shutdown_signal;
    common::AsyncRuntime& async_runtime;
    WorkerIdentityStore& identity;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    SessionLifecycleManager* lifecycle{nullptr};
    HandleLeaseRegistry* handle_leases{nullptr};
    common::CapabilityTokenManager* capability_tokens{nullptr};
    bool cpu_shared_memory_enabled{false};
    bool external_target_verification_enabled{false};
    std::filesystem::path storage_path;
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

  grpc::Status publish_target_replica(
      RpcContext& rctx,
      const v2::PublishTargetReplicaRequest& req,
      v2::PublishTargetReplicaResponse& resp);

  grpc::Status import_artifact_from_path(
      RpcContext& rctx,
      const v2::ImportArtifactFromPathRequest& req,
      v2::ImportArtifactFromPathResponse& resp);

  grpc::Status import_artifact_from_path_stream(
      RpcContext& rctx,
      const v2::ImportArtifactFromPathRequest& req,
      grpc::ServerWriter<v2::ImportArtifactFromPathStreamEvent>& writer);

  grpc::Status get_artifact_index_by_id(
      RpcContext& rctx,
      const v2::GetArtifactIndexByIdRequest& req,
      v2::GetArtifactIndexByIdResponse& resp);

  grpc::Status seal_assembly(RpcContext& rctx, const v2::SealAssemblyRequest& req, v2::SealAssemblyResponse& resp);

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

  // Test helper: inject a target write record without materialization.
  TargetWriteRegistry::Record insert_target_write_for_testing(TargetWriteRegistry::Record record);

 private:
  AssemblyOperationService assembly_operation_service_;
  DiskArtifactService disk_artifact_service_;
  ReplicaMaterializationService replica_materialization_service_;
  ReplicaLifecycleService replica_lifecycle_service_;
  TargetMaterializationService target_materialization_service_;
};

} // namespace tensorcast::daemon
