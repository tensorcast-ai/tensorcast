// Copyright (c) 2025, TensorCast Team.

// MaterializationController: handles materialize/by_key/get_artifact_index RPCs

#pragma once

#include <atomic>
#include <filesystem>
#include <vector>

#include "core/store/store_engine.h"
#include "daemon/device_resolver.h"
#include "daemon/lip_bridge.h"
#include "daemon/ref_tracker.h"
#include "daemon/rpc_context.h"
#include "daemon/session_lifecycle.h"
#include "daemon/sessions_service.h"
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
    std::atomic<bool>& is_shutting_down;
    SessionLifecycleManager* lifecycle{nullptr};
    std::vector<std::filesystem::path> disk_path_whitelist;
  };

  explicit MaterializationController(Dep d);

  grpc::Status materialize_replica(
      RpcContext& rctx,
      const v1::MaterializeReplicaRequest& req,
      v1::MaterializeReplicaResponse& resp);

  grpc::Status materialize_by_key(
      RpcContext& rctx,
      const v1::MaterializeByKeyRequest& req,
      v1::MaterializeByKeyResponse& resp);

  grpc::Status materialize_replica_v2(
      RpcContext& rctx,
      const v2::MaterializeReplicaRequest& req,
      v2::MaterializeReplicaResponse& resp);

  grpc::Status materialize_by_key_v2(
      RpcContext& rctx,
      const v2::MaterializeByKeyRequest& req,
      v2::MaterializeByKeyResponse& resp);

  grpc::Status resolve_artifact_from_disk(
      RpcContext& rctx,
      const v2::ResolveArtifactFromDiskRequest& req,
      v2::ResolveArtifactFromDiskResponse& resp);

  grpc::Status query_replica_status(
      RpcContext& rctx,
      const v2::QueryReplicaStatusRequest& req,
      v2::QueryReplicaStatusResponse& resp);

  grpc::Status release_replica(
      RpcContext& rctx,
      const v2::ReleaseReplicaRequest& req,
      v2::ReleaseReplicaResponse& resp);

  grpc::Status get_artifact_index_by_id(
      RpcContext& rctx,
      const v1::GetArtifactIndexByIdRequest& req,
      v1::GetArtifactIndexByIdResponse& resp);

  grpc::Status confirm(RpcContext& rctx, const v1::ConfirmReplicaRequest& req, v1::ConfirmReplicaResponse& resp) const;

  grpc::Status unload(RpcContext& rctx, const v1::UnloadReplicaRequest& req, v1::UnloadReplicaResponse& resp);

  grpc::Status wait_verification(
      RpcContext& rctx,
      const v1::WaitReplicaVerificationRequest& req,
      v1::WaitReplicaVerificationResponse& resp);

 private:
  Dep d_;
  std::vector<std::filesystem::path> disk_path_whitelist_;
  bool whitelist_enforced_{false};
};

} // namespace tensorcast::daemon
