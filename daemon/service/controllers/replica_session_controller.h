// Copyright (c) 2025-2026, TensorCast Team.

// ReplicaSessionController: handles replica session status and release RPCs.

#pragma once

#include "daemon/service/rpc_context.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/sessions_service.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class ReplicaSessionController {
 public:
  struct Dep {
    SessionsService& sessions;
    SessionLifecycleManager& lifecycle;
  };

  explicit ReplicaSessionController(Dep d) : d_(d) {}

  grpc::Status query_replica_status(
      RpcContext& rctx,
      const v2::QueryReplicaStatusRequest& req,
      v2::QueryReplicaStatusResponse& resp);

  grpc::Status wait_replica_status(
      RpcContext& rctx,
      grpc::ServerContext& ctx,
      const v2::WaitReplicaStatusRequest& req,
      v2::WaitReplicaStatusResponse& resp);

  grpc::Status release_replica(
      RpcContext& rctx,
      const v2::ReleaseReplicaRequest& req,
      v2::ReleaseReplicaResponse& resp);

 private:
  Dep d_;
};

} // namespace tensorcast::daemon
