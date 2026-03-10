// Copyright (c) 2025-2026, TensorCast Team.

// PersistenceRpcController: handles persistence control/status RPCs.

#pragma once

#include "daemon/service/rpc_context.h"
#include "daemon/state/persistence_manager.h"
#include "daemon/state/shutdown_signal.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class PersistenceRpcController {
 public:
  struct Dep {
    PersistenceManager* persistence_manager{nullptr};
    ShutdownSignal& shutdown_signal;
  };

  explicit PersistenceRpcController(Dep d) : d_(d) {}

  grpc::Status start_persistence(
      RpcContext& rctx,
      const v2::StartPersistenceRequest& req,
      v2::StartPersistenceResponse& resp);

  grpc::Status query_persistence_status(
      RpcContext& rctx,
      const v2::QueryPersistenceStatusRequest& req,
      v2::QueryPersistenceStatusResponse& resp);

 private:
  Dep d_;
};

} // namespace tensorcast::daemon
