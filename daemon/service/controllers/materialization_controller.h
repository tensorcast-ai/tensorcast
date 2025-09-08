// Copyright (c) 2025, TensorCast Team.

// MaterializationController: handles Materialize/ByKey/GetArtifactIndex RPCs

#pragma once

#include <atomic>
#include <memory>

#include "core/store/store_engine.h"
#include "daemon/device_resolver.h"
#include "daemon/lip_bridge.h"
#include "daemon/ref_tracker.h"
#include "daemon/rpc_context.h"
#include "daemon/sessions_service.h"
#include "daemon/status_utils.h"
#include "grpcpp/grpcpp.h"
#include "tensorcast/daemon/v1/store_daemon.grpc.pb.h"

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
  };

  explicit MaterializationController(Dep d) : d_(d) {}

  grpc::Status MaterializeReplica(
      RpcContext& rctx,
      const v1::MaterializeReplicaRequest& req,
      v1::MaterializeReplicaResponse& resp);

  grpc::Status MaterializeByKey(
      RpcContext& rctx,
      const v1::MaterializeByKeyRequest& req,
      v1::MaterializeByKeyResponse& resp);

  grpc::Status GetArtifactIndexById(
      RpcContext& rctx,
      const v1::GetArtifactIndexByIdRequest& req,
      v1::GetArtifactIndexByIdResponse& resp);

  grpc::Status Confirm(RpcContext& rctx, const v1::ConfirmReplicaRequest& req, v1::ConfirmReplicaResponse& resp);

  grpc::Status Unload(RpcContext& rctx, const v1::UnloadReplicaRequest& req, v1::UnloadReplicaResponse& resp);

  grpc::Status WaitVerification(
      RpcContext& rctx,
      const v1::WaitReplicaVerificationRequest& req,
      v1::WaitReplicaVerificationResponse& resp);

 private:
  Dep d_;
};

} // namespace tensorcast::daemon
