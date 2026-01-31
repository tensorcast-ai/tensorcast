// Copyright (c) 2025-2026, TensorCast Team.

// MaterializationController: handles materialize/by_key/get_artifact_index RPCs

#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "core/common/async_runtime.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/daemon_options.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/handle_lease_registry.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_bridge.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/sessions_service.h"
#include "daemon/state/shutdown_signal.h"
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
    DeviceResolver& devices;
    IpcRegionRegistry& regions;
    ShutdownSignal& shutdown_signal;
    common::AsyncRuntime& async_runtime;
    WorkerIdentityStore& identity;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    SessionLifecycleManager* lifecycle{nullptr};
    HandleLeaseRegistry* handle_leases{nullptr};
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

  grpc::Status materialize_by_key(
      RpcContext& rctx,
      const v2::MaterializeByKeyRequest& req,
      v2::MaterializeByKeyResponse& resp);

  grpc::Status materialize_into_target(
      RpcContext& rctx,
      const v2::MaterializeIntoTargetRequest& req,
      v2::MaterializeIntoTargetResponse& resp);

  grpc::Status resolve_artifact_from_disk(
      RpcContext& rctx,
      const v2::ResolveArtifactFromDiskRequest& req,
      v2::ResolveArtifactFromDiskResponse& resp);

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

 private:
  Dep d_;
  std::filesystem::path storage_path_;

  absl::Mutex seal_mu_;
  absl::flat_hash_set<std::string> active_seal_operations_ ABSL_GUARDED_BY(seal_mu_);
};

} // namespace tensorcast::daemon
